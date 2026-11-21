"""
Tests of the power-sweep NEP analysis (``citkid.nep.sweep``).
"""
import asyncio
import numpy as np
import pytest
from scipy.constants import h
from citkid import daq_layout as dl
from citkid.nep import sweep as sw
from citkid.responsivity.funcs import responsivity_int_x0


def synthetic_sweep(n_tones = 3, m = 6, R0 = -2e12, P0 = 2e-15,
                    eta = 0.6, nu = 150e9, sxx0 = 1e-18):
    """
    Make a PowerSweep without a store, with fr and Sxx following the
    responsivity and photon-noise models.

    Parameters:
    n_tones (int): tones. Default 3.
    m (int): points. Default 6.
    R0, P0 (float): responsivity parameters. Default -2e12 1/W, 2e-15 W.
    eta (float): optical efficiency. Default 0.6.
    nu (float): photon frequency, Hz. Default 150 GHz.
    sxx0 (float): Sxx floor at zero power (1/Hz). Default 1e-18.

    Returns:
    sweep (PowerSweep): the sweep, values filled, tones not yet fit.
    """
    sweep = sw.PowerSweep.__new__(sw.PowerSweep)
    sweep.root = sweep.grp = None
    sweep.power = np.linspace(2e-16, 2e-14, m)
    sweep.points = [dict(name = f'p{j}', power_w = float(p))
                    for j, p in enumerate(sweep.power)]
    sweep.timestreams, sweep.datasets = [], {}
    sweep.res_idxs = np.arange(n_tones)
    sweep.rows = np.tile(np.arange(n_tones)[:, None], (1, m))
    sweep.nu = np.full(n_tones, nu)
    sweep._nu_setting = nu
    sweep.freq_hz = 10.
    sweep.settings = {}
    f1 = 1e9
    x = responsivity_int_x0(sweep.power, R0, P0, 0.)
    sweep.fr = np.tile(f1 * (1 + x), (n_tones, 1))
    resp = np.abs(R0) * (1 + sweep.power / P0) ** -0.5
    nep_photon = np.sqrt(2 * h * nu * sweep.power / eta)
    sweep.sxx = np.tile((nep_photon * resp) ** 2 + sxx0, (n_tones, 1))
    sweep.x = np.full((n_tones, m), np.nan)
    sweep.nep = np.full((n_tones, m), np.nan)
    sweep.fields = {k: np.full(n_tones, np.nan) for k in sw.TONE_FIELDS}
    sweep.fields['n_fit'] = np.zeros(n_tones, dtype = int)
    sweep.fields['p_min'] = np.zeros(n_tones)
    sweep.status = [''] * n_tones
    sweep.bad, sweep.limits, sweep.changed, sweep.review = {}, {}, set(), {}
    sweep.psd_path = None
    return sweep


def test_compute_tone_fits_responsivity_and_eta():
    """
    From fr and Sxx following the models, the responsivity fit recovers
    R0 and P0, NEP(P) = sqrt(Sxx) / |R(P)|, the photon-noise fit recovers
    eta above p_min, NEPinc is the lowest power's NEP and NEPabs = eta
    NEPinc; the overview lists the quantities.
    """
    sweep = synthetic_sweep()
    sweep.fields['p_min'][1] = 5e-15
    sweep.compute_all()
    f = sweep.fields
    assert sweep.status == ['', '', '']
    np.testing.assert_allclose(f['R0'], -2e12, rtol = 0.05)
    np.testing.assert_allclose(f['P0'], 2e-15, rtol = 0.1)
    np.testing.assert_allclose(f['f1'], sweep.fr[:, 0])
    np.testing.assert_allclose(sweep.x[0], (sweep.fr[0] - sweep.fr[0, 0])
                               / sweep.fr[0, 0])
    resp = np.abs(f['R0'][0]) * (1 + sweep.power / f['P0'][0]) ** -0.5
    np.testing.assert_allclose(sweep.nep[0], np.sqrt(sweep.sxx[0]) / resp)
    # Above p_min the floor is negligible: eta within a few percent
    assert f['eta'][1] == pytest.approx(0.6, rel = 0.05)
    assert f['n_fit'][1] == 4 and f['n_fit'][0] == 6
    np.testing.assert_allclose(f['nep_inc'], sweep.nep[:, 0])
    np.testing.assert_allclose(f['nep_abs'], f['eta'] * f['nep_inc'])
    ov = sweep.overview_values()
    assert set(ov) == {'R0', 'P0', 'x0', 'eta', 'nep_inc', 'nep_abs'}
    assert sweep.outside_limits() == []
    sweep.limits['eta'] = [0.7, 1.]
    assert sweep.outside_limits() == [0, 1, 2]


def test_compute_tone_statuses():
    """
    Too few IQ fits, x rising with power (R0 > 0) and a missing photon
    frequency each give a status and NaN results, never a NEP from |R0|.
    """
    sweep = synthetic_sweep()
    sweep.fr[0, 2:] = np.nan
    sweep.fr[1] = 2e9 - (sweep.fr[1] - 1e9)      # x rises with power
    sweep.nu[2] = np.nan
    sweep.compute_all()
    assert sweep.status[0].startswith('2 points with an IQ fit')
    assert np.isnan(sweep.fields['R0'][0]) and np.isnan(sweep.nep[0]).all()
    assert 'x rises with power' in sweep.status[1]
    assert sweep.fields['R0'][1] > 0 and np.isnan(sweep.nep[1]).all()
    assert np.isnan(sweep.fields['nep_inc'][1])
    assert 'no photon frequency' in sweep.status[2]
    assert np.isfinite(sweep.fields['nep_inc'][2])
    assert np.isnan(sweep.fields['eta'][2])


def power_store(tmp_path, n_points = 4):
    """
    Take a simulated power sweep with 04c (the device does not respond
    to power, so the responsivity fits fail: the plumbing is what is
    tested).

    Parameters:
    tmp_path (pathlib.Path): where the store goes.
    n_points (int): points. Default 4.

    Returns:
    path (pathlib.Path): the store.
    """
    from ..procedures.test_scripts import load_script
    from citkid.crs.sim import simulate_vna_store, simulated_crs
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 5,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, verbose = False))
    fr, qr = truth.fr.values, truth.Qr.values
    n = len(fr)
    dl.save_resonators(root, 0, fr, qr / 5, 'test', Qr = qr)
    tune, _ = dl.new_tuning(root, 0)
    crs = simulated_crs(path, [1], seed = 0, global_control = True)
    asyncio.run(crs.attach())
    dl.write_tune_config(tune, dict(dl.tune_config(crs), tone_sets = [
        {'1': float(np.median(fr))}]))
    dl.save_tones(dl.create_group(tune, 'tones', 'decision'),
                  fr, qr / 5, -55., np.arange(n),
                  tone_set = np.zeros(n, int), tone_module = np.ones(n, int))
    m = load_script('04c_power_sweep.py')
    m.POINTS = [dict(name = f'P{j}') for j in range(n_points)]
    m.RUNS = [dict(dec_stage = 6, duration_s = 0.5)]
    m.NPOINTS_FINE, m.NPOINTS_GAIN, m.NSAMPS = 60, 10, 10
    m.FRES_UPDATE = None
    m.set_optical_power = lambda name: None

    async def same_board(args, root, path):
        """
        Stand in for connect.

        Returns:
        crs (SimulatedCRS): the board.
        """
        return crs

    m.connect = same_board
    m.main(m.parse_args(['--store', str(path), '--simulate']))
    return path


def test_sweep_batch_recompute_and_reload(tmp_path, capsys):
    """
    09 in batch: the points come from POINTS (point names to powers), the
    calibration,
    IQ fits and spectra are made, a sweep is saved with the review data;
    recomputing a tone at a point without searching again gives the same
    Sxx as the batch within the decimation's tolerance and notes the
    point; the final pass computes it again; a saved sweep reloads with
    its values, and 09 resumes it.
    """
    from ..procedures.test_scripts import load_script
    path = power_store(tmp_path)
    m = load_script('09_nep.py')
    m.NAME = 'bb'
    m.NU = 150e9
    m.CN_N_COMP = 1
    m.POINTS = {f'P{j}': 1e-15 * (j + 1) for j in range(4)}
    # Short, slow simulated streams: a 30 Hz NEP on a grid up to the
    # Nyquist frequency, and modes kept at about 100 Hz
    m.FREQ_HZ, m.FMAX_DIV, m.MODES_RATE_HZ = 30., 1., 100.
    args = m.parse_args(['--store', str(path), '--no-gui'])
    sweep = m.main(args)
    out = capsys.readouterr().out
    assert 'Points (power' in out and 'Calibrating into ts_cal0' in out
    root = dl.open_store(path)
    assert dl.calibration_name(root['ts_cal0']) == 'bb'
    np.testing.assert_allclose([p['power_w'] for p in sweep.points],
                               [1e-15, 2e-15, 3e-15, 4e-15])
    assert sweep.psd_path == 'ts_psd0' and 'nep/bb' in root
    assert root['ts_psd0'].attrs['name'] == 'bb'
    assert len(sweep.res_idxs) == 5 and sweep.rows.shape == (5, 4)
    assert (sweep.rows >= 0).all()
    assert np.isfinite(sweep.fr).all() and np.isfinite(sweep.sxx).all()
    # The device does not respond to power: a status, or a fit that
    # means nothing; either way every tone has arrays
    assert len(sweep.status) == 5 and sweep.nep.shape == (5, 4)
    for i, st in enumerate(sweep.status):
        assert st or np.isfinite(sweep.fields['nep_inc'][i])
    assert set(sweep.review) == {0, 1, 2, 3}
    rev = sweep.review[0]
    assert set(rev['writes']) == {0}
    assert rev['writes'][0]['modes_dec'].shape[0] == 1
    assert rev['writes'][0]['dt_dec'] >= 4 * sweep.timestreams[0]['dt']
    # Recompute one tone at one point: the same Sxx without a search
    sxx0 = sweep.sxx[1, 2]
    sweep.recompute(1, 2)
    assert sweep.changed == {2}
    assert sweep.sxx[1, 2] == pytest.approx(sxx0, rel = 0.5)
    assert np.isfinite(sweep.sxx[1, 2])
    sweep.save()
    # Reload: the values and the review data come back
    again = sw.PowerSweep.load(dl.open_store(path, mode = 'a'),
                               dl.open_store(path, mode = 'a')['nep/bb'])
    np.testing.assert_allclose(again.sxx, sweep.sxx)
    np.testing.assert_allclose(again.fr, sweep.fr)
    assert again.changed == {2} and again.status == sweep.status
    assert set(again.review[2]['glitch_idxs']) == set(rev['glitch_idxs'])
    np.testing.assert_allclose(again.review[0]['writes'][0]['modes_dec'],
                               rev['writes'][0]['modes_dec'])
    # The final pass computes point 2 again and clears the note
    done = again.final_pass(verbose = False)
    assert done == [2] and again.changed == set()
    assert again.sxx[1, 2] == pytest.approx(sxx0, rel = 1e-6)
    # 09 resumes the saved result unless --overwrite
    resumed = m.main(m.parse_args(['--store', str(path), '--no-gui']))
    assert 'Resuming nep/bb' in capsys.readouterr().out
    assert resumed.psd_path == 'ts_psd0'
    assert sw.list_results(root) == ['bb'] and 'bb: 4 points' in \
        sw.describe_results(root)
    # POINTS by point name or run name, as a dict or a list of dicts
    points = sw.select_points(root, {'P0_dec6': 9e-15, 'P1': 2e-15,
                                     'P2': 3e-15, 'P3': 4e-15})
    np.testing.assert_allclose([p['power_w'] for p in points],
                               [2e-15, 3e-15, 4e-15, 9e-15])
    assert [p['point'] for p in points] == ['P1', 'P2', 'P3', 'P0']
    points = sw.select_points(root, [dict(run = 0, power_w = 1e-15),
                                     dict(name = 'P1', power_w = 2e-15)])
    assert [p['run'] for p in points] == [0, 1]
    with pytest.raises(KeyError, match = 'no run'):
        sw.select_points(root, {'nope': 1e-15, 'P1': 2e-15})
    with pytest.raises(ValueError, match = 'at least two'):
        sw.select_points(root, {'P0': 1e-15})
    with pytest.raises(ValueError, match = 'no run has a recorded'):
        sw.select_points(root, {})
    with pytest.raises(ValueError, match = 'has no power'):
        sw.select_points(root, [dict(name = 'P0'), dict(name = 'P1')])
    assert [p['point'] for p in sw.list_points(root)] == ['P0', 'P1',
                                                           'P2', 'P3']
