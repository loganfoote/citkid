import numpy as np
import pytest
import zarr
from citkid.vna import full_sweep as fs

def test_coupling_error_pair():
    k = fs.coupling_error([1e9, 1e9 + 1e6, np.nan], [2e4, 1e4, 1e4],
                          [0.8, 0.5, 0.5])
    np.testing.assert_allclose(k[0], 0.5 * (1e9 + 1e6) / 1e4 / 2e6)
    np.testing.assert_allclose(k[1], 0.8 * 1e9 / 2e4 / 2e6)
    assert np.isnan(k[2])

def test_reliability_flags():
    flags, misfit = fs.reliability_flags(
        coupling = [0.5, 0., 0., 0.], excess = [0., 100., 0., 0.],
        snr = [100., 100., 100., 100.],
        narrow = [False, False, True, False], a = [0., 0., 0., 0.9])
    assert list(flags) == ['overlap', 'poor fit', 'distorted', 'bifurcated']
    np.testing.assert_allclose(misfit[1], np.sqrt(200) / 100)

def test_fit_full_sweep(full_sweep):
    f, z, gain, t = full_sweep
    rng = np.random.default_rng(3)
    lw = t['fr'] / t['Qr']
    # Listed frequencies off by up to 0.3 linewidths, in reversed order,
    # plus one spot with no resonance
    fres = (t['fr'] + rng.uniform(-0.3, 0.3, len(lw)) * lw)[::-1]
    fres = np.r_[fres, 1.0195e9]
    res, g, info = fs.fit_full_sweep(f, z, fres)
    assert list(res.columns) == fs.FIT_COLUMNS
    np.testing.assert_array_equal(res.fres_input, fres)
    ok = res.fit_status.values[:-1] == 'ok'
    assert ok.mean() > 0.9
    fr_err = np.abs(res.fr.values[:-1] - t['fr'][::-1]) / lw[::-1]
    assert np.all(fr_err[ok] < 0.05)
    Qr_err = np.abs(res.Qr.values[:-1] / t['Qr'][::-1] - 1)
    assert np.median(Qr_err[ok]) < 0.01
    # The empty spot fails and keeps its listed frequency
    assert res.fit_status.iloc[-1].startswith('failed')
    assert res.fr.iloc[-1] == fres[-1] and np.isnan(res.Qr.iloc[-1])
    assert np.median(np.abs(g / gain - 1)) < 2e-3
    assert np.all(res.a.values[:-1][ok] == 0)

def test_fit_full_sweep_workers_identical(full_sweep):
    f, z, _, t = full_sweep
    a = fs.fit_full_sweep(f, z, t['fr'], n_workers = 1)[0]
    b = fs.fit_full_sweep(f, z, t['fr'], n_workers = 2)[0]
    for c in a.columns:
        if a[c].dtype.kind == 'f':
            np.testing.assert_array_equal(a[c].values, b[c].values)
        else:
            assert list(a[c]) == list(b[c])

def test_run_full_sweep_fit(full_sweep, tmp_path, capsys):
    f, z, gain, t = full_sweep
    path = str(tmp_path / 'sweep.zarr')
    grp = zarr.open_group(path, mode = 'a')
    with pytest.raises(KeyError):
        fs.run_full_sweep_fit(f, z, grp, progress = False)
    grp.create_array('fres_manual', data = t['fr'])
    res, report = fs.run_full_sweep_fit(f, z, grp, progress = False)
    assert report in capsys.readouterr().out
    out = grp['full_sweep_fit']
    for c in fs.FIT_COLUMNS:
        assert out[c].shape == (len(t['fr']),)
    assert out['fit_status'][0] == 'ok'
    assert out['gain/gain'].dtype == np.complex128
    assert out.attrs['report'] == report
    assert out.attrs['fit_nonlinear'] is False
    # The description can be reprinted from the saved results
    from citkid.vna.sweep_info import run_sweep_info
    assert run_sweep_info(grp) == report
    with pytest.raises(FileExistsError):
        fs.run_full_sweep_fit(f, z, grp, progress = False)
    fs.run_full_sweep_fit(f, z, path, overwrite = True, progress = False)
