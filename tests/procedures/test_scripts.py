"""
Tests for the procedure scripts in ``procedures/``, run against a fake CRS.
"""
import asyncio
import importlib.util
import os
import numpy as np
import pytest
from citkid import daq_layout as dl
from ..crs.fake_crs import FakeVnaCRS

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', '..', 'procedures')


def load_script(name):
    """
    Import a procedure script as a module.

    Parameters:
    name (str): file name, e.g. '01_vna_sweep.py'.

    Returns:
    module (module): the script's module (its ``__main__`` block not run).
    """
    path = os.path.join(SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_factory(monkeypatch, module, fake):
    """
    Make the script build ``fake`` instead of a real CRS, recording the
    constructor arguments.

    Parameters:
    monkeypatch (pytest.MonkeyPatch): fixture.
    module (module): the script's module.
    fake (FakeVnaCRS): the fake to return.

    Returns:
    made (list of dict): filled with each constructor call's arguments.
    """
    made = []

    def factory(**kwargs):
        """
        Record the arguments and return the fake.

        Parameters:
        **kwargs: CRS arguments.

        Returns:
        fake (FakeVnaCRS): the fake.
        """
        made.append(kwargs)
        return fake

    monkeypatch.setattr(module, 'CRS', factory)
    return made


@pytest.mark.asyncio
@pytest.mark.parametrize('argv, extended', [([], False),
                                            (['--extended-bw'], True)])
async def test_crs_init(monkeypatch, argv, extended):
    m = load_script('crs_init.py')
    fake = FakeVnaCRS(modules = (1, 2, 3, 4))
    made = fake_factory(monkeypatch, m, fake)
    await m.main(m.parse_args(['--clock', 'SMA'] + argv))
    assert made == [dict(serial_number = m.SERIAL_NUMBER,
                         interface = m.INTERFACE)]
    name, kw = fake.calls[0]
    assert name == 'configure_system' and kw['clock_source'] == 'SMA'
    assert kw['analog_bank_high'] is m.ANALOG_BANK_HIGH
    assert fake.extended_bw is extended
    assert fake.bw == (600e6 if extended else 500e6)


def test_vna_quick_check_no_gui(monkeypatch, tmp_path):
    m = load_script('vna_quick_check.py')
    fake = FakeVnaCRS(modules = (1, 2), fres = (1.0e9, 1.6e9, 2.1e9))
    made = fake_factory(monkeypatch, m, fake)
    png = tmp_path / 'q.png'
    f, z, n = m.main(m.parse_args(
        ['--modules', '1', '2', '--png', str(png), '--no-gui']))
    assert made[0]['module_idxs'] == [1, 2]
    assert made[0]['global_control'] is False
    assert fake.calls[0][0] == 'attach'
    # Zone 1 by default, geometric at the settings' step up to the
    # dithering
    step = 1 / (m.QR_MEDIAN * m.PTS_PER_LW)
    assert 100e6 - 50 <= f[0] < 100e6 + 50 and f[-1] < 2.5e9
    d = np.diff(np.log(f)) / np.log1p(step)
    assert 0.9 < d.min() and d.max() < 1.1
    assert png.exists()
    assert n == 3
    # Nothing is saved to zarr
    assert not any(p.suffix == '.zarr' for p in tmp_path.iterdir())


def test_vna_quick_check_no_count(monkeypatch, tmp_path):
    m = load_script('vna_quick_check.py')
    fake_factory(monkeypatch, m, FakeVnaCRS())
    _, _, n = m.main(m.parse_args(['--no-count', '--no-gui']))
    assert n is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('nyquist, lo, hi', [('2', 2.5e9, 5e9),
                                             ('both', 100e6, 5e9)])
def test_vna_quick_check_nyquist(monkeypatch, nyquist, lo, hi):
    m = load_script('vna_quick_check.py')
    fake = FakeVnaCRS(modules = (1, 2))
    fake_factory(monkeypatch, m, fake)
    f, _, _ = m.main(m.parse_args(
        ['--modules', '1', '2', '--nyquist', nyquist, '--qr-median', '1e4',
         '--pts-per-lw', '1', '--no-gui', '--no-count']))
    # Nothing within the 10 MHz guard band around 2.5 GHz is swept
    start = lo + 10e6 if lo == 2.5e9 else lo
    assert start - 50 <= f[0] < start * (1 + 1e-4)
    assert hi * (1 - 2e-4) < f[-1] < hi
    assert not np.any(np.abs(f - 2.5e9) < 10e6 - 50)
    zones = {zn for sz in fake.sweep_zones for zn in sz.values()}
    assert zones == ({1, 2} if nyquist == 'both' else {2})
    # Back to zone 1 afterwards
    assert all(c['nyquist_zone'] == 1 for c in fake.module_cfg.values())


def test_vna_quick_check_gui(monkeypatch):
    """
    Check that the window gets the settings and a connect function that
    attaches to this session's modules.
    """
    m = load_script('vna_quick_check.py')
    fake = FakeVnaCRS(modules = (2,))
    made = fake_factory(monkeypatch, m, fake)
    seen = {}

    def fake_run(connect, module_idxs, **kwargs):
        """
        Record the window arguments and connect.

        Parameters:
        connect (callable): coroutine function.
        module_idxs (list of int): modules.
        **kwargs: settings.

        Returns:
        result (str): a marker.
        """
        seen.update(kwargs, module_idxs = module_idxs,
                    crs = asyncio.run(connect()))
        return 'closed'

    monkeypatch.setattr(m, 'run_quick_check', fake_run)
    out = m.main(m.parse_args(['--modules', '2', '--nyquist', 'both',
                               '--amplitude', '-45', '--no-count']))
    assert out == 'closed' and seen['crs'] is fake
    assert seen['module_idxs'] == [2] and seen['nyquist'] == 'both'
    assert seen['amplitude'] == -45. and seen['count'] is False
    assert seen['qr_median'] == m.QR_MEDIAN
    assert seen['pts_per_lw'] == m.PTS_PER_LW
    assert made[0]['global_control'] is False
    assert fake.calls == [('attach', dict(full_scale_dbm = 7.))]


@pytest.mark.asyncio
async def test_01_vna_sweep(monkeypatch, tmp_path):
    m = load_script('01_vna_sweep.py')
    fake = FakeVnaCRS(modules = (3,), fres = (1.2e9,))
    made = fake_factory(monkeypatch, m, fake)
    argv = ['--data-dir', str(tmp_path), '--cooldown', 'cd1', '--device',
            'devA', '--modules', '3', '--nsamps', '7', '--qr-median', '1e4',
            '--pts-per-lw', '5', '--full-scale', '5']
    k = await m.main(m.parse_args(argv))
    assert k == 0
    assert made[0]['module_idxs'] == [3] and not made[0]['global_control']
    assert fake.calls == [('attach', dict(full_scale_dbm = 5.))]
    path = tmp_path / 'cd1' / 'devA.zarr'
    root = dl.open_store(path)
    assert root.attrs['device'] == 'devA' and root.attrs['setup'] == \
        m.SETUP
    grp = root['vna0']
    assert grp.attrs['nsamps'] == 7 and grp.attrs['amplitude'] == \
        m.AMPLITUDE
    assert grp.attrs['step'] == pytest.approx(2e-5)
    f, z = dl.load_vna_sweep(path)
    d = np.diff(np.log(f)) / np.log1p(2e-5)
    assert 0.9 < d.min() and d.max() < 1.1
    # A second run adds sweep1 to the same store
    assert await m.main(m.parse_args(argv)) == 1


def test_scripts_line_length_and_settings():
    """
    Check the scripts keep to 80 columns and have a settings block and a
    main guard.
    """
    for name in sorted(os.listdir(SCRIPTS)):
        if not name.endswith('.py'):
            continue
        with open(os.path.join(SCRIPTS, name), encoding = 'utf-8') as fh:
            text = fh.read()
        assert all(len(line) <= 80 for line in text.splitlines()), name
        assert 'Settings' in text and "if __name__ == '__main__':" in text


def test_02_find_resonances(tmp_path, capsys):
    """
    Run step 02 on a small simulated store (finder list given, so no CNN;
    no manual check), then check resuming, redoing, and that a list a
    tuning uses is kept.
    """
    from citkid.crs.sim import simulate_vna_store
    m = load_script('02_find_resonances.py')
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 12,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, nyquist = '1', verbose = False))
    dl.write_array(root['vna0'], 'fres_auto', truth.fr.values)
    argv = ['--store', str(path), '--no-manual']
    assert m.main(m.parse_args(argv)) == 0
    out = capsys.readouterr().out
    assert 'find: done already' in out and 'resonators: saved' in out
    root = dl.open_store(path)
    grp = root['vna0']
    assert grp['full_sweep_fit'].attrs['kind'] == 'analysis'
    assert grp.attrs['fres_manual_checked'] is False
    res = dl.load_resonators(path, 0)
    assert len(res['fres_all']) == len(truth)
    np.testing.assert_allclose(res['fres_all'], truth.fr_d,
                               atol = 0.2 * (truth.fr / truth.Qr).max())
    assert np.isfinite(res['Qr_all']).all()
    np.testing.assert_allclose(res['qres_all'], res['Qr_all'] / m.SPAN_LW)
    assert root['resonators0'].attrs['source'] == 'vna0/full_sweep_fit'
    # Everything done: a rerun skips all steps
    assert m.main(m.parse_args(argv)) == 0
    assert capsys.readouterr().out.count('done already') == 4
    # Redo the list with other spans
    m.main(m.parse_args(argv + ['--redo', 'resonators', '--span-lw', '10']))
    np.testing.assert_allclose(dl.load_resonators(path, 0)['qres_all'],
                               res['Qr_all'] / 10)
    # Once a tuning uses the list, it is kept
    dl.new_tuning(dl.open_store(path, mode = 'a'), 0)
    assert m.main(m.parse_args(argv + ['--redo', 'resonators'])) is None
    assert 'is used by' in capsys.readouterr().out


def test_simulate_store(tmp_path):
    m = load_script('simulate_store.py')
    path = tmp_path / 'sim.zarr'
    truth = asyncio.run(m.main(m.parse_args(
        ['--store', str(path), '--seed', '2', '--pts-per-lw', '2.5'])))
    root = dl.open_store(path)
    assert sorted(root.group_keys()) == ['simulation', 'vna0']
    assert root['vna0'].attrs['step'] == pytest.approx(2e-5)
    assert len(truth) > 500 and root['simulation'].attrs['seed'] == 2


@pytest.mark.parametrize('n_cal', [0, 1])
def test_03_tune(tmp_path, monkeypatch, capsys, n_cal):
    """
    Run step 03 on a small simulated store with stand-ins for the two
    windows, then check resuming, the redo rules, and that the final tones
    port to a new store.
    """
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 10,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, verbose = False))
    dl.save_resonators(root, 0, truth.fr.values, truth.Qr.values / 5,
                       'test', Qr = truth.Qr.values)
    m = load_script('03_tune.py')
    monkeypatch.setattr(m, 'POWERS', np.array([-60., -55., -50.]))
    monkeypatch.setattr(m, 'NPOINTS_FINE', 40)
    monkeypatch.setattr(m, 'NPOINTS_GAIN', 10)
    monkeypatch.setattr(m, 'N_CAL_TONES', n_cal)
    calls = []

    def fake_iq_series(datasets, root, xy_fit, y_target, **kwargs):
        """
        Stand in for the IQ series window: check its inputs and choose a
        power for all but the first tone.

        Parameters:
        datasets (list of DataSet): one per power.
        root (zarr.Group): the power_fits group.
        xy_fit (SeriesXYFit): the a vs power fit.
        y_target (float): target a.
        **kwargs: other window arguments.
        """
        calls.append('iq')
        # Calibration tones are not reviewed
        assert kwargs['data_idxs'] == list(range(len(truth)))
        assert len(datasets) == 3 and y_target == m.ANL
        n = len(np.asarray(datasets[0].res_idxs))
        chosen = np.full(n, -47.)
        chosen[1] = -60.     # clipped
        chosen[0] = np.nan   # fallback: the median of the others
        root.require_group('xy_fit').create_array('x_chosen', data = chosen)

    def fake_fqfinder(f, z, fres, qres, res_idxs, zarr_grp, **kwargs):
        """
        Stand in for fq_finder: keep every tone but the last.

        Parameters:
        f, z (np.array): fine sweeps.
        fres, qres, res_idxs (np.array): tones.
        zarr_grp (zarr.Group): the fq_sweep group.
        **kwargs: other arguments.
        """
        calls.append('fq')
        zarr_grp.create_array('fres_opt', data = fres + 1e3)
        zarr_grp.create_array('qres_opt', data = qres)
        rr = np.array([''] * len(fres), dtype = object)
        rr[-1] = 'bad'
        arr = zarr_grp.create_array('reject_reason', shape = (len(fres),),
                                    dtype = str, fill_value = '')
        arr[:] = rr.astype(str)

    monkeypatch.setattr(m, 'run_iq_series', fake_iq_series)
    monkeypatch.setattr(m, 'run_fqfinder', fake_fqfinder)
    argv = ['--store', str(path), '--simulate']
    assert m.main(m.parse_args(argv)) == 0
    assert calls == ['iq', 'fq']
    root = dl.open_store(path)
    tune = root['tune0']
    assert tune.attrs['resonators'] == 'resonators0'
    assert tune['power_sweep'].attrs['complete'] is True
    assert sorted(dl.list_indices(tune['power_sweep'], 'set')) == [0, 1, 2]
    pc = tune['power_choice']
    assert pc['ares'][0] == -47. and pc['fallback'][0]
    assert pc['ares'][1] == m.ARES_LIMITS[0]
    np.testing.assert_allclose(pc['ares'][2:], -47.)
    np.testing.assert_allclose(tune['fq_sweep/ares'][:], pc['ares'][:])
    t = dl.load_tones(tune['tones'])
    n = tune['tones_init/fres'].shape[0]
    assert len(t['fres']) == n - 1
    np.testing.assert_allclose(t['fres'],
                               tune['fq_sweep/fres_opt'][:][:-1])
    assert 'tone_module' in t and t['tone_module'][0] == 1
    # Calibration tones: negative res_idxs, at the median power, away
    # from every resonance
    init = dl.load_tones(tune['tones_init'])
    cal = init['res_idxs'] < 0
    assert cal.sum() == n_cal
    if n_cal:
        np.testing.assert_allclose(pc['ares'][:][cal], -47.)
        lw = truth.fr.values / truth.Qr.values
        assert np.all(np.abs(init['fres'][cal] - truth.fr.values) >=
                      10 * lw)
    # A rerun skips everything; power choices can't be redone after the
    # fq sweep; tones can
    assert m.main(m.parse_args(argv)) == 0
    assert calls == ['iq', 'fq']
    with pytest.raises(ValueError, match = 'new tuning'):
        m.main(m.parse_args(argv + ['--redo', 'power_fits']))
    m.main(m.parse_args(argv + ['--redo', 'tones']))
    # The tuning ports without its sweeps
    dl.port_tuning(path, 0, tmp_path / 'b.zarr')
    b = dl.open_store(tmp_path / 'b.zarr')
    assert sorted(b['tune0'].group_keys()) == ['crs_config', 'tones']
    np.testing.assert_array_equal(dl.load_tones(b['tune0/tones'])['ares'],
                                  t['ares'])


def test_04_stream(tmp_path, monkeypatch):
    """
    Check that step 04 opens the window on the latest tuning with tones,
    with the settings, and a connect function that attaches with global
    control and checks the CRS against the store.
    """
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 6,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, verbose = False))
    m = load_script('04_stream.py')
    with pytest.raises(KeyError, match = '03_tune'):
        m.main(m.parse_args(['--store', str(path), '--simulate']))
    dl.save_resonators(root, 0, truth.fr.values, 4e3, 'test')
    for _ in range(2):
        tune, k = dl.new_tuning(root, 0)
    dl.save_tones(dl.create_group(root['tune0'], 'tones', 'decision'),
                  truth.fr.values, 4e3, -55., np.arange(len(truth)),
                  tone_module = np.ones(len(truth), int))
    seen = {}

    def fake_gui(connect, tune, **kwargs):
        """
        Record the window arguments and connect.

        Parameters:
        connect (callable): coroutine function.
        tune (zarr.Group): the tuning.
        **kwargs: settings.

        Returns:
        runs (list): none.
        """
        seen.update(kwargs, tune = tune.basename,
                    crs = asyncio.run(connect()))
        return []

    monkeypatch.setattr(m, 'run_stream_gui', fake_gui)
    assert m.main(m.parse_args(['--store', str(path), '--simulate'])) == []
    # tune1 has no tones, so the latest with tones is tune0
    assert seen['tune'] == 'tune0'
    assert seen['crs'].global_control is m.GLOBAL_CONTROL
    assert seen['presets'] == m.PRESETS
    assert seen['sweep_kwargs']['npoints_fine'] == m.NPOINTS_FINE
    with pytest.raises(KeyError, match = 'no tones'):
        m.main(m.parse_args(['--store', str(path), '--simulate', '--tune',
                             '1']))


def test_archive_store_script(tmp_path, capsys):
    m = load_script('archive_store.py')
    root = dl.create_store(tmp_path / 'cd1' / 'devA.zarr', 'devA', 'cd1')
    dl.save_resonators(root, 0, [1e9, 2e9], 4e3, 'test')
    argv = ['--data-dir', str(tmp_path), '--cooldown', 'cd1', '--device',
            'devA']
    zp = m.main(m.parse_args(argv))
    assert zp.endswith('devA.zarr.zip') and 'resonators0' in \
        capsys.readouterr().out
    np.testing.assert_array_equal(dl.load_resonators(zp, 0)['fres_all'],
                                  [1e9, 2e9])
    # The store stays; an existing zip is not overwritten
    assert (tmp_path / 'cd1' / 'devA.zarr').exists()
    with pytest.raises(FileExistsError):
        m.main(m.parse_args(argv))
    other = tmp_path / 'b.zip'
    assert m.main(m.parse_args(argv + ['--zip', str(other)])) == str(other)


def test_05_calibrate_streams(tmp_path, monkeypatch, capsys):
    """
    List the runs, calibrate a choice of streams in batch (two decimation
    stages: two DataSets), continue in the same cal group, start a new one
    for another choice, and open the window when asked.
    """
    from citkid.crs.procedures import stream_tone_sets
    from ..crs.test_streaming import make_crs, make_tuning
    crs = make_crs()

    async def take():
        """
        Take a stage-6 and a chunked stage-0 run.

        Returns:
        tune (zarr.Group): the tuning.
        """
        await crs.attach()
        _, tune = make_tuning(tmp_path, crs)
        kw = dict(npoints_fine = 100, npoints_gain = 20, verbose = False)
        await stream_tone_sets(crs, tune, 'dark6', 1., 6, **kw)
        await stream_tone_sets(crs, tune, 'dark0', 0.02, 0, **kw)
        return tune

    asyncio.run(take())
    m = load_script('05_calibrate_streams.py')
    argv = ['--store', str(tmp_path / 's.zarr'), '--no-gui']
    monkeypatch.setattr(m, 'CALIBRATE', [])
    assert m.main(m.parse_args(argv)) is None
    assert 'dark6' in capsys.readouterr().out
    sel = [dict(name = 'dark6', chunks = 1), dict(run = 1, tone_sets = [1])]
    monkeypatch.setattr(m, 'CALIBRATE', sel)
    ds = m.main(m.parse_args(argv))
    assert len(ds) == 2
    root = dl.open_store(tmp_path / 's.zarr')
    cal = root['tune0/cal0']
    assert cal.attrs['kind'] == 'analysis' and cal.attrs['selection'] == sel
    n_set1 = int((root['tune0/tones/tone_set'][:] == 1).sum())
    sizes = sorted(len(d.res_idxs[:]) for d in ds.values())
    assert sizes == sorted([root['tune0/tones/fres'].shape[0], n_set1])
    for d in ds.values():
        x = d.xt[0]
        assert np.isfinite(np.asarray(x)).all()
    # Same choice: same cal group; a new one otherwise
    m.main(m.parse_args(argv))
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr')['tune0'],
                           'cal') == [0]
    monkeypatch.setattr(m, 'CALIBRATE', sel[:1])
    m.main(m.parse_args(argv))
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr')['tune0'],
                           'cal') == [0, 1]
    # The window is opened unless --no-gui
    opened = []
    monkeypatch.setattr(m, 'run_ts_analysis',
                        lambda AR, title = '': opened.append(title))
    m.main(m.parse_args(argv[:2]))
    assert len(opened) == 1 and 'cal1' in opened[0]


def test_03_tune_windows(tmp_path, monkeypatch, capsys):
    """
    With STREAM_DEC = 0, NCO windows hold at most 128 tones; --list-windows
    prints them and makes no tuning; --windows keeps only some.
    """
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    dense = dict(f_lo = 1.0e9, f_hi = 1.3e9, n_banks = 1, n_per_bank = 150,
                 bank_fill = 0.9, pair_frac = 0., yield_frac = 1.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = dense, step = 1e-4, verbose = False))
    dl.save_resonators(root, 0, truth.fr.values, truth.Qr.values / 5,
                       'test', Qr = truth.Qr.values)
    m = load_script('03_tune.py')
    monkeypatch.setattr(m, 'STREAM_DEC', 0)
    monkeypatch.setattr(m, 'MIN_SEP_LW', 0.)
    argv = ['--store', str(path), '--simulate']
    assert m.main(m.parse_args(argv + ['--list-windows'])) is None
    out = capsys.readouterr().out
    assert 'at most 128 tones' in out and '  1: NCO' in out
    assert dl.list_indices(dl.open_store(path), 'tune') == []
    root = dl.open_store(path, mode = 'a')
    tune = m.plan(root, m.parse_args(argv + ['--windows', '1']), str(path))
    t = dl.load_tones(tune['tones_init'])
    assert 0 < len(t['fres']) <= 128 and set(t['tone_set'].tolist()) == {0}
    assert tune['tones_init'].attrs['windows'] == [1]
    assert len(tune['crs_config'].attrs['config']['tone_sets']) == 1
    # Window 1 is the higher one: its tones are above window 0's
    assert t['fres'].min() > np.sort(truth.fr.values)[100]
    with pytest.raises(ValueError, match = 'no windows'):
        m.plan(root, m.parse_args(argv + ['--windows', '7']), str(path))
