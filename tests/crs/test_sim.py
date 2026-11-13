"""
Tests for the simulated CRS (``citkid.crs.sim``).
"""
import numpy as np
import pytest
from citkid import daq_layout as dl
from citkid.crs import sim as csim
from citkid.crs.procedures import SweepCancelled, vna_sweep
from citkid.vna import simulate as sim

SMALL = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 15)


def make_crs(**kwargs):
    """
    Make a simulated CRS with a small device.

    Parameters:
    **kwargs: ``SimulatedCRS`` arguments.

    Returns:
    crs (SimulatedCRS): the board.
    """
    rng = np.random.default_rng(0)
    truth = sim.draw_device(rng, **SMALL)
    gain = sim.draw_gain(rng, f_lo = 1e8, f_hi = 5e9)
    return csim.SimulatedCRS(truth, gain, **kwargs)


@pytest.mark.asyncio
async def test_noise_scales_with_samples_and_power():
    crs = make_crs()
    await crs.set_nco({1: 1.05e9})
    freqs = np.linspace(1.0e9, 1.1e9, 1024 * 20).reshape(1024, 20)
    sig = {}
    for nsamps, power in ((100, -55.), (25, -55.), (100, -45.)):
        z = [(await crs.sweep(freqs, np.full(1024, power), nsamps,
                              ch_map = {1: range(1024)}))[1]
             for _ in range(2)]
        g = np.abs(sim.gain_s21(freqs, crs.gain))
        sig[nsamps, power] = np.std((z[0] - z[1]) / g) / 2
    ref = 10 ** (-crs.snr_db / 20)
    assert sig[100, -55.] == pytest.approx(ref, rel = 0.05)
    assert sig[25, -55.] == pytest.approx(2 * ref, rel = 0.05)
    assert sig[100, -45.] == pytest.approx(ref / 10 ** 0.5, rel = 0.05)


@pytest.mark.asyncio
async def test_sweep_checks_and_dithering():
    crs = make_crs(module_idxs = (1, 2))
    await crs.set_nco({1: 1.05e9})
    freqs = np.linspace(1.0e9, 1.1e9, 400).reshape(20, 20)
    with pytest.raises(ValueError, match = 'ch_map'):
        await crs.sweep(freqs, np.full(20, -55.), 10)
    seen = []
    f, _ = await crs.sweep(freqs, np.full(20, -55.), 10,
                           ch_map = {1: range(20)},
                           progress = lambda d, t: seen.append((d, t)))
    assert seen == [(0, 400), (400, 400)]
    assert 0 < np.abs(f - freqs).max() <= 50
    await crs.set_nco({1: 2.0e9})
    with pytest.raises(ValueError, match = 'bandwidth'):
        await crs.sweep(freqs, np.full(20, -55.), 10, ch_map = {1: range(20)})
    await crs.set_nco({2: 2.75e9})
    with pytest.raises(ValueError, match = 'Nyquist'):
        await crs.sweep(freqs + 1.7e9, np.full(20, -55.), 10,
                        ch_map = {2: range(20)})
    await crs.configure_modules(nyquist_zone = 2, module_idxs = [2])
    await crs.sweep(freqs + 1.7e9, np.full(20, -55.), 10,
                    ch_map = {2: range(20)})


@pytest.mark.asyncio
async def test_vna_sweep_finds_the_device():
    crs = make_crs(snr_db = 60.)
    await crs.attach()
    f, z, _ = await vna_sweep(crs, 1.0e9, 1.1e9, -55., 100, verbose = False)
    zc = z / sim.gain_s21(f, crs.gain)
    for fr, Qr in zip(crs.truth.fr, crs.truth.Qr):
        near = np.abs(f - fr) < 3 * fr / Qr
        f_min = f[near][np.argmin(np.abs(zc[near]))]
        assert abs(f_min - fr) < 0.5 * fr / Qr


@pytest.mark.asyncio
async def test_simulate_vna_store(tmp_path):
    root, truth = await csim.simulate_vna_store(
        tmp_path / 's.zarr', seed = 3, device_kwargs = SMALL, step = 1e-5,
        verbose = False)
    f, z = dl.load_vna_sweep(tmp_path / 's.zarr')
    assert 1e8 - 50 <= f[0] and f[-1] < 2.5e9 and np.all(np.isfinite(z))
    assert root.attrs['setup']['simulated'] is True
    assert root['vna0'].attrs['step'] == 1e-5
    loaded = csim.load_truth(tmp_path / 's.zarr')
    assert list(loaded.columns) == list(truth.columns)
    np.testing.assert_array_equal(loaded.fr_d, truth.fr_d)
    assert 'Qi_d' in loaded and root['simulation'].attrs['seed'] == 3


################################################################################
############################ Stitched target sweeps ############################
################################################################################
async def _stitched(tmp_path, n_sets = 2):
    """
    Take a stitched target sweep of a small simulated device split into
    tone sets on one module.

    Parameters:
    tmp_path (pathlib.Path): directory.
    n_sets (int): number of tone sets (tones alternate between them).

    Returns:
    crs (SimulatedCRS): the board.
    grp (zarr.Group): the stitched sweep.
    tone_set (np.array): each tone's set.
    """
    from citkid.crs.procedures import stitched_target_sweep
    crs = make_crs()
    await crs.attach()
    fres = crs.truth.fr.values
    n = len(fres)
    tone_set = np.arange(n) % n_sets
    ncos = [{'1': 1.05e9 + 1e3 * s} for s in range(n_sets)]
    root = dl.create_store(tmp_path / 's.zarr', 'd', 'c')
    grp = root.create_group('sweep')
    await stitched_target_sweep(
        crs, fres, np.linspace(-60, -50, n), 2e3, np.arange(n), tone_set,
        np.ones(n, int), ncos, grp, verbose = False, npoints_fine = 50,
        npoints_gain = 10)
    return crs, grp, tone_set


@pytest.mark.asyncio
async def test_stitched_target_sweep(tmp_path):
    crs, grp, tone_set = await _stitched(tmp_path)
    n = len(tone_set)
    assert grp.attrs['complete'] is True
    assert grp['fine_sweep/f'].shape == (n, 50)
    assert grp['gain_sweep/z'].shape == (n, 10)
    np.testing.assert_array_equal(grp['res_idxs'][:], np.arange(n))
    np.testing.assert_allclose(grp['ares'][:], np.linspace(-60, -50, n))
    # Each tone's rows are its own sweep, around its own frequency
    f = grp['fine_sweep/f'][:]
    assert np.all(np.abs(f.mean(axis = 1) - crs.truth.fr.values) < 1e4)
    np.testing.assert_array_equal(grp['tone_set'][:], tone_set)
    np.testing.assert_array_equal(
        grp['tone_channel'][:], np.arange(n) // 2)
    assert [d['1'] for d in grp.attrs['pass_ncos']] == [1.05e9, 1.05e9 + 1e3]
    assert len(grp.attrs['pass_times']) == 2
    assert not [k for k in grp.attrs if k.startswith('nco_module')]
    assert not [k for k in grp['fine_sweep'].array_keys()
                if k.startswith('chs_module')]
    assert len(grp['fine_sweep'].attrs['pass_timestamps']) == 2


@pytest.mark.asyncio
async def test_stitched_target_sweep_errors(tmp_path):
    from citkid.crs.procedures import stitched_target_sweep
    crs = make_crs()
    await crs.attach()
    root = dl.create_store(tmp_path / 's.zarr', 'd', 'c')
    grp = root.create_group('sweep')
    args = ([1.0e9, 1.01e9], -55., 2e3, [0, 1])
    with pytest.raises(ValueError, match = 'index ncos'):
        await stitched_target_sweep(crs, *args, [0, 1], [1, 1],
                                    [{'1': 1.05e9}], grp)
    with pytest.raises(ValueError, match = 'no NCO'):
        await stitched_target_sweep(crs, *args, [0, 0], [1, 2],
                                    [{'1': 1.05e9}], grp)
    grp.create_group('x')
    with pytest.raises(ValueError, match = 'empty'):
        await stitched_target_sweep(crs, *args, [0, 0], [1, 1],
                                    [{'1': 1.05e9}], grp)


@pytest.mark.asyncio
async def test_stitched_power_sweep_resumes(tmp_path):
    from citkid.crs.procedures import stitched_power_sweep
    crs = make_crs()
    await crs.attach()
    fres = crs.truth.fr.values[:4]
    root = dl.create_store(tmp_path / 's.zarr', 'd', 'c')
    ps = root.create_group('power_sweep')
    kw = dict(verbose = False, npoints_fine = 20, npoints_gain = None)
    args = (crs, fres, [-60., -55., -50.], 2e3, np.arange(4), [0] * 4,
            [1] * 4, [{'1': 1.05e9}], ps)
    # Pretend set1 was interrupted (empty) and set0 is done
    await stitched_power_sweep(*args, **kw)
    t0 = ps['set0'].attrs['pass_times']
    del ps['set1']
    ps.create_group('set1')
    await stitched_power_sweep(*args, **kw)
    assert ps['set0'].attrs['pass_times'] == t0
    assert ps['set1'].attrs['complete'] is True
    np.testing.assert_allclose(ps['set1/ares'][:], -55.)
    ps['set2'].create_group('junk')
    ps['set2'].attrs['complete'] = False
    with pytest.raises(ValueError, match = 'neither empty'):
        await stitched_power_sweep(*args, **kw)


@pytest.mark.asyncio
async def test_tuning_config_with_tone_sets(tmp_path):
    crs = make_crs()
    await crs.attach()
    plan = dict(ncos = [{1: 1.0e9}, {1: 1.5e9}], extended_bw = False)
    config = dl.tuning_config(crs, plan)
    assert 'ncos' not in config
    assert config['tone_sets'] == [{'1': 1.0e9}, {'1': 1.5e9}]
    root = dl.create_store(tmp_path / 's.zarr', 'd', 'c')
    tune = root.create_group('tune0')
    dl.write_tune_config(tune, config)
    # Any NCO is fine (each pass sets its own); other settings must match
    await crs.set_nco({1: 2.0e9})
    dl.check_tune_config(tune, crs)
    await crs.configure_modules(full_scale_dbm = 5)
    with pytest.raises(RuntimeError, match = 'full_scale_dbm'):
        dl.check_tune_config(tune, crs)
    with pytest.raises(ValueError, match = 'extended_bw'):
        dl.tuning_config(crs, dict(plan, extended_bw = True))



@pytest.mark.asyncio
async def test_vna_sweep_cancelled_on_simulated_board():
    """
    A cancelled sweep of the simulated board leaves no tones written and
    the Nyquist zones as they were.
    """
    crs = make_crs(module_idxs = (1,))
    reports = []
    with pytest.raises(SweepCancelled):
        await vna_sweep(crs, 2.6e9, 3.6e9, -55., 10, step = 1e-4,
                        verbose = False, progress = lambda d, t:
                        reports.append(d), cancelled = lambda: len(reports) > 2)
    assert crs.ch_map == {} and crs.ntones == 0
    assert all(cfg['nyquist_zone'] == 1 for cfg in crs.module_cfg.values())
