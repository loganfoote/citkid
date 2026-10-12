"""Tests for citkid.pipeline.sweep_series."""

import os
import shutil

import numpy as np
import pytest
import zarr

from citkid.pipeline import DataSet
from citkid.pipeline.sweep_series import (
    a_vs_power_fit,
    ares_for_a,
    open_series_datasets,
    pick_optimal_ares,
    responsivity_fit,
    run_series_analysis,
    save_popt,
    target_sweep_source,
    target_sweep_steps,
)
from citkid.res.funcs import nonlinear_iq

FRES = np.array([4.0e8, 4.5e8])
QR = 2e4
NPTS = 200
POWERS = np.array([-50., -46., -42., -38.])
NT = 50


def a_true(power):
    """Synthetic nonlinearity parameter, linear in tone amplitude."""
    return 0.1 * 10 ** ((power + 50) / 20)


def write_power_sweep(path, prefix = 'set', reverse_f = False):
    """
    Write a synthetic power sweep in the target_sweep format, with a
    timestream group in each set, plus a fres_all array in the store root.

    Parameters:
    path (str): zarr store path.
    prefix (str): set group prefix.
    reverse_f (bool): if True, store the sweeps in decreasing frequency.
    """
    rng = np.random.default_rng(0)
    root = zarr.open_group(path, mode = 'w')
    root.create_array('fres_all', data = FRES * 1.0)
    for i, power in enumerate(POWERS):
        grp = root.create_group(f'{prefix}{i}')
        grp.create_array('fres', data = FRES)
        grp.create_array('qres', data = np.full(2, QR / 12))
        grp.create_array('ares', data = np.full(2, power))
        grp.create_array('res_idxs', data = np.arange(2, dtype = np.int32))
        ff = np.array([np.linspace(fr - 6 * fr / QR, fr + 6 * fr / QR, NPTS)
                       for fr in FRES])
        fg = np.array([np.linspace(fr - 60 * fr / QR, fr + 60 * fr / QR, NPTS)
                       for fr in FRES])
        def s21(f, fr):
            return nonlinear_iq(f, fr, QR, 0.6, 0.05, a_true(power), 1.0,
                                0.1, 1e-9, True)
        zf = np.array([s21(f, fr) for f, fr in zip(ff, FRES)])
        zf = zf + 1e-4 * rng.standard_normal(zf.shape)
        zg = np.array([s21(f, fr) for f, fr in zip(fg, FRES)])
        if reverse_f:
            ff, zf, fg, zg = ff[:, ::-1], zf[:, ::-1], fg[:, ::-1], zg[:, ::-1]
        for name, f, z in [('fine_sweep', ff, zf), ('gain_sweep', fg, zg)]:
            sub = grp.create_group(name)
            sub.create_array('f', data = f)
            sub.create_array('z', data = z)
        ts = grp.create_group('ts_00')
        ts.create_array('z', data = rng.integers(-100, 100, (2, 2, NT),
                                                 dtype = np.int32))
        ts.create_array('counts_to_s21', data = np.array([1e-3, 2e-3]))
        ts.create_array('dt', data = np.array(1e-3))
        ts.create_array('fres', data = FRES + 1.0)


@pytest.fixture
def data_dir(tmp_path):
    """Directory holding a synthetic power sweep, 'sweep.zarr'."""
    main_dir = tmp_path / 'data'
    main_dir.mkdir()
    write_power_sweep(str(main_dir / 'sweep.zarr'))
    return str(main_dir)


def run_steps(steps, data_idx = 0):
    """Run loading steps directly; return {name: output}."""
    out = {}
    for step in steps:
        if step.func_type in ('global', 'global-res'):
            res = step.func()
        else:
            arg = np.array([data_idx]) if step.func_type == 'vectorized' \
                else data_idx
            res = step.func(arg)
        res = res if isinstance(res, tuple) else (res,)
        out.update(zip(step.return_names, res))
    return out


################################################################################
############################# target_sweep_steps ###############################
################################################################################

def test_steps_load_set_and_sort(tmp_path):
    write_power_sweep(str(tmp_path / 's.zarr'), reverse_f = True)
    out = run_steps(target_sweep_steps('s.zarr', 'set2', main_dir = str(tmp_path)))
    assert out['nrows'] == 2
    np.testing.assert_array_equal(out['fres_all'], FRES)
    np.testing.assert_array_equal(out['ares'], [-42., -42.])
    assert np.all(np.diff(out['ff']) > 0) and np.all(np.diff(out['fg']) > 0)
    assert out['ff'].shape == out['zf'].shape == (NPTS,)


def test_steps_f_avg_averages_fine_sweep(data_dir):
    raw = zarr.open_group(os.path.join(data_dir, 'sweep.zarr'), mode = 'r')
    out = run_steps(target_sweep_steps('sweep.zarr', 'set0', main_dir = data_dir,
                                       f_avg = 3))
    n = NPTS // 3 * 3
    np.testing.assert_allclose(
        out['ff'], raw['set0/fine_sweep/f'][0, :n].reshape(-1, 3).mean(1))
    np.testing.assert_allclose(
        out['zf'], raw['set0/fine_sweep/z'][0, :n].reshape(-1, 3).mean(1))


def test_steps_fres_all_from_other_store_and_constant_qres(data_dir):
    other = zarr.open_group(os.path.join(data_dir, 'prep.zarr'), mode = 'w')
    other.create_array('fres_all', data = np.array([1e8, 2e8, 3e8]))
    out = run_steps(target_sweep_steps(
        'sweep.zarr', 'set0', main_dir = data_dir,
        fres_all = 'prep.zarr/fres_all', qres_all = 6000))
    np.testing.assert_array_equal(out['fres_all'], [1e8, 2e8, 3e8])
    np.testing.assert_array_equal(out['qres_all'], [6000.] * 3)


def test_steps_default_fres_all_uses_set_tones(data_dir):
    out = run_steps(target_sweep_steps('sweep.zarr', 'set0', main_dir = data_dir))
    np.testing.assert_array_equal(out['fres_all'], FRES)
    np.testing.assert_array_equal(out['qres_all'], [QR / 12] * 2)


def test_steps_timestream(data_dir):
    raw = zarr.open_group(os.path.join(data_dir, 'sweep.zarr'), mode = 'r')
    steps = target_sweep_steps('sweep.zarr', 'set1', main_dir = data_dir,
                               ts = 'ts_00', ts_nsamps = 20)
    assert [s.name for s in steps][-2:] == ['load_ft', 'load_zt']
    assert steps[0].return_names == ['fres_all', 'qres_all', 'dt', 'nrows']
    out = run_steps(steps, data_idx = 1)
    z = raw['set1/ts_00/z'][:, 1, :20]
    np.testing.assert_allclose(out['zt'], [(z[0] + 1j * z[1]) * 2e-3])
    np.testing.assert_array_equal(out['ft'], [FRES[1] + 1.0])
    assert float(out['dt']) == 1e-3


@pytest.mark.parametrize('kwargs, match', [
    (dict(fres_all = 'x.zarr/fres_all'), 'qres_all is required'),
    (dict(f_avg = 0), 'f_avg'),
    (dict(ts_nsamps = 0), 'ts_nsamps'),
])
def test_steps_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match = match):
        target_sweep_steps('x.zarr', **kwargs)


################################################################################
########################## source and DataSet opening ##########################
################################################################################

def test_source_defines_steps_and_rejects_non_literals(data_dir):
    source = target_sweep_source(data_dir, 'sweep.zarr', 'set0', f_avg = np.int64(2))
    namespace = {}
    exec(source, namespace)
    names = [s.name for s in namespace['custom_cal_steps']]
    assert names == ['load_global_data', 'load_global_res_data', 'load_data_f',
                     'load_data_g']
    assert 'f_avg = 2,' in source
    with pytest.raises(TypeError, match = 'fres_all'):
        target_sweep_source(data_dir, 'sweep.zarr', fres_all = np.ones(2))


def test_dataset_custom_source_embeds_and_reopens(data_dir, tmp_path):
    out = zarr.open_group(str(tmp_path / 'out.zarr'), mode = 'a')
    source = target_sweep_source(data_dir, 'sweep.zarr', 'set0')
    DS = DataSet(out.require_group('set0'), cal_yaml_path = 'iq',
                 custom_source = source)
    ff = DS.ff[0]
    # Reopen with only the zarr group: the embedded source is used
    DS2 = DataSet(out['set0'])
    np.testing.assert_array_equal(DS2.ff[0], ff)
    # A different definition is refused
    with pytest.raises(ValueError, match = 'do not match'):
        DataSet(out['set0'], cal_yaml_path = 'iq',
                custom_source = target_sweep_source(data_dir, 'sweep.zarr',
                                                    'set0', f_avg = 2))


def test_dataset_custom_source_exclusive_with_custom_path(data_dir):
    with pytest.raises(ValueError, match = 'custom_source'):
        DataSet(zarr.group(), cal_yaml_path = 'iq', custom_path = 'x.py',
                custom_source = 'custom_cal_steps = []')


def test_open_series_datasets_finds_sets_and_survives_move(data_dir, tmp_path):
    out_path = str(tmp_path / 'out.zarr')
    datasets = open_series_datasets(data_dir, 'sweep.zarr', out_path)
    assert len(datasets) == len(POWERS)
    assert sorted(zarr.open_group(out_path).group_keys()) == \
        [f'set{i}' for i in range(len(POWERS))]
    np.testing.assert_array_equal(datasets[3].ares[:], [-38., -38.])
    ff = datasets[1].ff[0]

    moved = str(tmp_path / 'moved')
    shutil.move(data_dir, moved)
    reopened = open_series_datasets(moved, 'sweep.zarr', out_path)
    np.testing.assert_array_equal(reopened[1].ff[0], ff)
    # and directly, with custom_main_dir_overwrite
    DS = DataSet(zarr.open_group(out_path)['set2'],
                 custom_main_dir_overwrite = moved)
    assert DS.ff[1].shape == (NPTS,)


def test_open_series_datasets_prefix_and_subset(tmp_path):
    write_power_sweep(str(tmp_path / 'old.zarr'), prefix = 'amp')
    datasets = open_series_datasets(str(tmp_path), 'old.zarr', zarr.group(),
                                    set_idxs = [2, 0], prefix = 'amp')
    assert [float(DS.ares[0]) for DS in datasets] == [-42., -50.]
    with pytest.raises(ValueError, match = "no 'set' groups"):
        open_series_datasets(str(tmp_path), 'old.zarr', zarr.group())


def test_open_series_datasets_rejects_different_res_idxs(data_dir):
    raw = zarr.open_group(os.path.join(data_dir, 'sweep.zarr'), mode = 'a')
    raw['set2/res_idxs'][...] = np.array([1, 0], dtype = np.int32)
    with pytest.raises(ValueError, match = 'res_idxs of set2 differ from set0'):
        open_series_datasets(data_dir, 'sweep.zarr', zarr.group())


################################################################################
################################## Fits ########################################
################################################################################

def test_a_vs_power_fit_recovers_line_and_cuts_bifurcated():
    xy_fit = a_vs_power_fit(a_max = 0.8)
    x = np.array([-50., -46., -42., -38., -34.])
    y = a_true(x)
    y[-1] = 2.0  # bifurcated point, left out
    popt, _ = xy_fit.run(x, y)
    np.testing.assert_allclose(popt, [0.1 * 10 ** 2.5, 0.0], atol = 1e-9)
    np.testing.assert_allclose(xy_fit.model(x[:-1], *popt), y[:-1])
    popt, _ = xy_fit.run(x[:2], np.array([0.9, 1.0]))
    assert np.all(np.isnan(popt))


def test_ares_for_a_inverts_fit():
    popt = np.array([[0.1 * 10 ** 2.5, 0.0], [np.nan, np.nan], [1.0, 1.0]])
    ares = ares_for_a(popt, 0.4)
    np.testing.assert_allclose(ares[0], -50 + 20 * np.log10(4))
    assert np.isnan(ares[1]) and np.isnan(ares[2])  # a never reaches 0.4
    assert np.isclose(ares_for_a(popt[0], 0.1), -50.)


def test_responsivity_fit_recovers_parameters():
    from citkid.responsivity.funcs import responsivity_int_x0
    P = np.geomspace(1e-14, 1e-10, 12)
    f1 = 4e8
    y = f1 * (1 + responsivity_int_x0(P, -2e8, 1e-12, 0.0))
    xy_fit = responsivity_fit()
    popt, p0 = xy_fit.run(P, y)
    assert len(popt) == len(p0) == 4
    np.testing.assert_allclose(xy_fit.model(P, *popt), y, rtol = 1e-7)


################################################################################
################################ Power tuning ##################################
################################################################################

class StubDS:
    """Minimal stand-in for a DataSet with saved iq_popt rows."""

    def __init__(self, ares, a, ff = None):
        self.nrows = len(ares)
        self.ares = np.asarray(ares, dtype = float)
        self.res_idxs = np.arange(self.nrows)
        popt = np.zeros((self.nrows, 8))
        popt[:, 4] = a
        self._saved = ~np.isnan(np.asarray(a, dtype = float))
        self.iq_popt = popt
        self.ff = np.zeros((self.nrows, 3)) if ff is None else ff
        self.zf = self.ff * 1j

    def saved_row_mask(self, name):
        assert name == 'iq_popt'
        return self._saved


def test_pick_optimal_ares_cases():
    # columns: normal, bifurcates at 2nd point, all bad, NaN gap, shuffled
    powers = [-50., -45., -40., -35.]
    a = np.array([
        [0.1, 0.6, np.nan, 0.1, 0.45],
        [0.2, 0.7, np.nan, np.nan, 0.1],
        [0.4, 0.2, np.nan, 0.3, 0.2],
        [0.55, 0.9, np.nan, 0.6, 0.6],
    ])
    aress = np.repeat(np.array(powers)[:, None], 5, axis = 1)
    aress[:, 4] = [-40., -50., -45., -35.]  # shuffled powers
    datasets = [StubDS(aress[i], a[i]) for i in range(4)]
    ares_opt, opt_idxs = pick_optimal_ares(datasets, a_max = 0.5)
    np.testing.assert_array_equal(opt_idxs, [2, -1, -1, 2, 0])
    np.testing.assert_array_equal(ares_opt[[0, 3, 4]], [-40., -40., -40.])
    # column 1: first point (-50, a = 0.6) is already past a_max
    assert np.isnan(ares_opt[1]) and np.isnan(ares_opt[2])


def test_save_popt_writes_optimal_sweeps():
    ff = [np.full((2, 3), i + 1.) for i in range(2)]
    datasets = [StubDS([-50., -50.], [0.1, 0.1], ff[0]),
                StubDS([-40., -40.], [0.2, 0.9], ff[1])]
    ares_opt, opt_idxs = pick_optimal_ares(datasets, a_max = 0.5)
    ares_opt[np.isnan(ares_opt)] = -45.  # user's own fallback
    grp = zarr.group()
    save_popt(grp, datasets, ares_opt, opt_idxs)
    np.testing.assert_array_equal(grp['ares_opt'][:], [-40., -50.])
    np.testing.assert_array_equal(grp['opt_idxs'][:], [1, 0])
    np.testing.assert_array_equal(grp['ff_opt'][:], [[2.] * 3, [1.] * 3])
    np.testing.assert_array_equal(grp['zf_opt'][:], [[2j] * 3, [1j] * 3])
    np.testing.assert_array_equal(grp['res_idxs'][:], [0, 1])


def test_power_tuning_end_to_end(data_dir, tmp_path):
    datasets = open_series_datasets(data_dir, 'sweep.zarr',
                                    str(tmp_path / 'out.zarr'))
    run_series_analysis(datasets, 'iq', verbose = False)
    for DS, power in zip(datasets, POWERS):
        np.testing.assert_allclose(DS.iq_popt[:][:, 4], a_true(power),
                                   rtol = 0.05, atol = 0.01)
    # a = 0.1, 0.16, 0.25, 0.40
    ares_opt, opt_idxs = pick_optimal_ares(datasets, a_max = 0.3)
    np.testing.assert_array_equal(ares_opt, [-42., -42.])
    np.testing.assert_array_equal(opt_idxs, [2, 2])
