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
    csv = tmp_path / 'q.csv'
    f, z, n = m.main(m.parse_args(
        ['--modules', '1', '2', '--csv', str(csv), '--no-gui']))
    assert made[0]['module_idxs'] == [1, 2]
    assert made[0]['global_control'] is False
    assert fake.calls[0][0] == 'attach'
    # Zone 1 by default, geometric at the settings' step up to the
    # dithering
    step = 1 / (m.QR_MEDIAN * m.PTS_PER_LW)
    assert 100e6 - 50 <= f[0] < 100e6 + 50 and f[-1] < 2.5e9
    d = np.diff(np.log(f)) / np.log1p(step)
    assert 0.9 < d.min() and d.max() < 1.1
    saved = np.genfromtxt(csv, delimiter = ',', names = True)
    assert saved.dtype.names == ('f', 'i', 'q')
    assert np.allclose(saved['f'], f, rtol = 0, atol = 1e-6)
    assert np.allclose(saved['i'] + 1j * saved['q'], z, rtol = 1e-8)
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
                    crs = asyncio.run(connect(module_idxs)))
        asyncio.run(connect([1, 3]))
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
    # The window can reconnect to other modules
    assert [mk['module_idxs'] for mk in made] == [[2], [1, 3]]
    assert fake.calls[0] == ('attach', dict(full_scale_dbm = 7.))


@pytest.mark.asyncio
async def test_01_vna_sweep(monkeypatch, tmp_path):
    m = load_script('01_vna_sweep.py')
    fake = FakeVnaCRS(modules = (3,), fres = (1.2e9,))
    made = fake_factory(monkeypatch, m, fake)
    argv = ['--data-dir', str(tmp_path), '--cooldown', 'cd1', '--device',
            'devA', '--modules', '3', '--nsamps', '7', '--qr-median', '1e4',
            '--pts-per-lw', '5', '--notes',
            '\n20 dB on the DAC line\nLPF on the ADC\n']
    k = await m.main(m.parse_args(argv))
    assert k == 0
    assert made[0]['module_idxs'] == [3] and not made[0]['global_control']
    assert fake.calls == [('attach', dict(full_scale_dbm = 7.))]
    path = tmp_path / 'cd1' / 'devA.zarr'
    root = dl.open_store(path)
    assert root.attrs['device'] == 'devA' and root.attrs['setup'] == \
        m.SETUP
    grp = root['vna0']
    assert grp.attrs['nsamps'] == 7 and grp.attrs['amplitude'] == \
        m.AMPLITUDE
    assert grp.attrs['step'] == pytest.approx(2e-5)
    assert grp.attrs['notes'] == '20 dB on the DAC line\nLPF on the ADC'
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
    # Dip depths measured on the sweep: close to the fitted Qr/Qc for
    # these isolated resonators
    fit = grp['full_sweep_fit']
    order = np.argsort(fit['fr'][:])
    np.testing.assert_allclose(res['depth_all'], fit['amp'][:][order],
                               atol = 0.05)
    assert ((res['depth_all'] > 0) & (res['depth_all'] < 1)).all()
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
        xy_fit (SeriesXYFit): the a_nl vs power fit.
        y_target (float): target a_nl.
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
        from citkid.crs import board_lock
        seen.update(kwargs, tune = tune.basename,
                    crs = asyncio.run(connect()),
                    holder = board_lock.holder(0))
        return []

    monkeypatch.setattr(m, 'run_stream_gui', fake_gui)
    assert m.main(m.parse_args(['--store', str(path), '--simulate'])) == []
    # tune1 has no tones, so the latest with tones is tune0
    assert seen['tune'] == 'tune0'
    assert seen['crs'].global_control is True
    # The board was this script's while the window was open, and is free
    # after
    from citkid.crs import board_lock
    assert seen['holder']['pid'] == os.getpid()
    assert '04_stream.py' in seen['holder']['owner']
    assert board_lock.holder(0) is None
    assert seen['presets'] == m.PRESETS
    assert seen['sweep_kwargs']['npoints_fine'] == m.NPOINTS_FINE
    with pytest.raises(KeyError, match = 'no tones'):
        m.main(m.parse_args(['--store', str(path), '--simulate', '--tune',
                             '1']))
    # Simulated photons and cosmic rays, on a quieter board
    assert seen['crs'].pulses is None
    snr = seen['crs'].snr_db
    m.main(m.parse_args(['--store', str(path), '--simulate',
                         '--sim-pulses']))
    assert seen['crs'].pulses == m.SIM_PULSE_SETTINGS
    assert seen['crs'].snr_db == m.SIM_SNR_DB != snr


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
    # One DataSet per tone set: set 0 from the first entry, set 1 from
    # both (its windows one after another)
    assert sorted(ds) == ['set0', 'set1']
    root = dl.open_store(tmp_path / 's.zarr')
    cal = root['ts_cal0']
    assert cal.attrs['kind'] == 'analysis' and cal.attrs['selection'] == sel
    tone_set = root['tune0/tones/tone_set'][:]
    for s in (0, 1):
        d = ds[f'set{s}']
        assert len(d.res_idxs[:]) == int((tone_set == s).sum())
        assert np.isfinite(np.asarray(d.xt[0])).all()
    assert len(np.asarray(ds['set1'].source_bounds)) == 2
    assert len(np.asarray(ds['set0'].source_bounds)) == 1
    # Same choice: same cal group; a new one otherwise
    m.main(m.parse_args(argv))
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr'),
                           'ts_cal') == [0]
    monkeypatch.setattr(m, 'CALIBRATE', sel[:1])
    m.main(m.parse_args(argv))
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr'),
                           'ts_cal') == [0, 1]
    # The window is opened unless --no-gui
    opened = []
    monkeypatch.setattr(m, 'run_ts_analysis',
                        lambda AR, title = '': opened.append(title))
    m.main(m.parse_args(argv[:2]))
    # One window per tone set
    assert len(opened) == 2 and all('ts_cal1' in t for t in opened)


def test_03_tune_keep_stronger(tmp_path, monkeypatch):
    """
    With PAIRS = 'keep_stronger', of two close resonators the deeper one
    keeps its tone; a list without dip depths can't be used.
    """
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 6,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, verbose = False))
    fres = np.array([1.00e9, 1.02e9, 1.02e9 + 1e3, 1.05e9])
    Qr = np.full(4, 2e4)
    m = load_script('03_tune.py')
    monkeypatch.setattr(m, 'PAIRS', 'keep_stronger')
    argv = m.parse_args(['--store', str(path), '--simulate'])
    # The second of the close pair (index 2) is deeper
    dl.save_resonators(root, 0, fres, Qr / 5, 'test', Qr = Qr,
                       depth = [0.5, 0.2, 0.6, 0.4])
    tune = m.plan(dl.open_store(path, mode = 'a'), argv, str(path))
    t = dl.load_tones(tune['tones_init'])
    assert sorted(t['res_idxs'].tolist()) == [0, 2, 3]
    assert tune['tones_init'].attrs['n_dropped'] == 1
    # No depths: an error that says what to do
    dl.save_resonators(dl.open_store(path, mode = 'a'), 1, fres, Qr / 5,
                       'test', Qr = Qr)
    with pytest.raises(ValueError, match = 'redo resonators'):
        m.plan(dl.open_store(path, mode = 'a'),
               m.parse_args(['--store', str(path), '--simulate',
                             '--resonators', '1']), str(path))


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


def test_vna_report(tmp_path):
    """
    Export a screened simulated sweep: plot, info text, and resonator CSV,
    with res_idx matching the store's resonator list.
    """
    import pandas as pd
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 10,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, verbose = False))
    m = load_script('vna_report.py')
    out = tmp_path / 'reports'
    with pytest.raises(KeyError, match = 'full sweep fit'):
        m.main(m.parse_args(['--store', str(path), '--out', str(out)]))
    dl.write_array(root['vna0'], 'fres_auto', truth.fr.values[::-1])
    m2 = load_script('02_find_resonances.py')
    m2.main(m2.parse_args(['--store', str(path), '--no-manual', '--notes',
                           'LPF on the ADC']))
    paths = m.main(m.parse_args(['--store', str(path), '--out', str(out)]))
    assert paths['plot'].endswith('simdevice_vna0.png')
    assert os.path.getsize(paths['plot']) > 10000
    info = open(paths['info'], encoding = 'utf-8').read()
    assert 'Device: simdevice' in info and 'Tone power: -55.0 dBm' in info
    assert f'Resonances: {len(truth)}' in info and 'Modules: 1' in info
    assert 'checked by hand: False' in info
    assert 'Notes (fit)' in info and 'LPF on the ADC' in info
    assert isinstance(dl.open_store(path)['vna0/full_sweep_fit'].attrs[
        'notes'], list)
    table = pd.read_csv(paths['csv'])
    res = dl.load_resonators(path, 0)
    assert list(table.res_idx) == list(range(len(truth)))
    np.testing.assert_allclose(table.fr, res['fres_all'])
    np.testing.assert_allclose(table.fr, np.sort(truth.fr_d.values),
                               atol = 0.2 * (truth.fr / truth.Qr).max())
    assert {'Qi', 'Qc', 'phi', 'flags', 'valid_Qc'} <= set(table.columns)


@pytest.fixture(scope = 'module')
def photon_store(tmp_path_factory):
    """
    Make a store with two calibrated stage-0 runs of a simulated device with
    photon pulses and cosmic rays.

    Returns:
    path (pathlib.Path): the store.
    log (list of dict): the simulated pulses of each stream, in order:
        run 0's tone set 0 (one write), its tone set 1 (two writes), then
        run 1's the same way.
    """
    from citkid.crs.procedures import stream_tone_sets
    from citkid.crs.sim import DEFAULT_PULSES
    from ..crs.test_streaming import make_crs, make_tuning
    tmp = tmp_path_factory.mktemp('photons')
    crs = make_crs()
    crs.snr_db = 80.
    crs.pulses = dict(DEFAULT_PULSES, rate = 30., cosmic_rate = 5.)

    async def take():
        """
        Take two runs.
        """
        await crs.attach()
        _, tune = make_tuning(tmp, crs)
        kw = dict(npoints_fine = 100, npoints_gain = 20, verbose = False)
        await stream_tone_sets(crs, tune, 'dark0', 1.5, 0, **kw)
        await stream_tone_sets(crs, tune, 'dark0b', 1.5, 0, **kw)

    asyncio.run(take())
    m5 = load_script('05_calibrate_streams.py')
    m5.CALIBRATE = [dict(name = 'dark0'), dict(name = 'dark0b')]
    m5.main(m5.parse_args(['--store', str(tmp / 's.zarr'), '--no-gui']))
    return tmp / 's.zarr', crs.pulse_log


def check_photons(grp, log, dt):
    """
    Check one saved timestream's photons against the simulation.

    Parameters:
    grp (zarr.Group): the ``timestream{j}`` group.
    log (list of dict): the simulated pulses of its tone writes.
    dt (float): sample period, s.
    """
    status = grp['status'][:]
    assert all(s == '' for s in status)
    assert grp['template'].shape == (len(status), grp.attrs['pre'] +
                                     grp.attrs['post'])
    np.testing.assert_allclose(grp['amp_ph'][:], 2e-6, rtol = 0.1)
    np.testing.assert_allclose(grp['tau'][:], 150e-6, rtol = 0.15)
    # About 30 photons/s per tone are averaged (minus coincidences and
    # pile-up; each tone's count is Poisson)
    n_used = grp['n_used'][:]
    expected = 30 * 1.5 * len(n_used)
    assert 0.8 * expected < n_used.sum() < 1.1 * expected
    assert np.all(n_used > 0.4 * 30 * 1.5)
    p = grp['pulses']
    used, win = p['used'][:], p['in_window'][:]
    assert np.all(win[used]) and np.all(p['mult'][:][used] == 1)
    assert np.all(p['candidate'][:][win])
    # Cosmic rays: the simulated ones, each on every tone of its write
    ev, hits = grp['cosmic/events'], grp['cosmic/hits']
    n_true = sum(len(entry['cosmic']) for entry in log)
    assert 0.85 * n_true <= len(ev['t0'][:]) <= n_true
    np.testing.assert_allclose(ev['time_s'][:], ev['t0'][:] * dt)
    assert len(hits['event'][:]) == ev['mult'][:].sum()
    # Their heights: mostly above the photon height window
    assert np.median(hits['h'][:]) > 2


def test_06_find_photons(photon_store, monkeypatch, capsys):
    """
    List the timestreams (one per run and tone set), then find photons in
    run 0's tone set 0 and both tone sets of run 1, with each way of
    reusing run 0: its thresholds (after the window changes them), its
    filters, or nothing (the window opens for each).
    """
    import citkid.pipeline.interactive.photon_thresholds as pt
    path, log = photon_store
    m = load_script('06_find_photons.py')
    argv = ['--store', str(path)]
    monkeypatch.setattr(m, 'TIMESTREAMS', [])
    assert m.main(m.parse_args(argv)) is None
    out = capsys.readouterr().out
    assert "0: ts0, 'dark0', set 0" in out and \
        "3: ts1, 'dark0b', set 1" in out
    monkeypatch.setattr(m, 'TIMESTREAMS', [0, 'dark0b'])
    windows = []

    def fake_window(ts, results, thresholds, rerun, title = ''):
        """
        Stand in for the threshold window: widen hmax for all tones, give
        one tone its own hmin, and rerun them all (as the window does).

        Parameters:
        ts (dict): the timestream.
        results (dict): its results.
        thresholds (Thresholds): its thresholds.
        rerun (callable): reruns tones.
        title (str): window title.

        Returns:
        thresholds, results: as the window.
        """
        windows.append(ts['run'])
        thresholds.hmax = 2.
        r0 = int(ts['rows'][0])
        thresholds.overrides[int(ts['res_idxs'][0])] = dict(hmin = 0.6)
        results = {**results, **rerun(ts['rows'], thresholds)}
        assert results[r0]['hmin'] == 0.6 and results[r0]['hmax'] == 2.
        return thresholds, results

    monkeypatch.setattr(pt, 'run_photon_thresholds', fake_window)
    out = m.main(m.parse_args(argv + ['--reuse', 'settings']))
    assert windows == ['ts0']
    root = dl.open_store(path)
    grp = root['ts_photons0']
    assert grp.attrs['source'] == 'ts_cal0' and grp.attrs['kind'] == \
        'analysis'
    dt = grp['timestream0'].attrs['dt']
    # Simulated streams of each timestream (see photon_store)
    logs = (log[:1], log[3:4], log[4:6])
    for j, entries in enumerate(logs):
        g = grp[f'timestream{j}']
        check_photons(g, entries, dt)
        th = g.attrs['thresholds']
        assert th['hmax'] == 2. and list(th['overrides'].values()) == \
            [dict(hmin = 0.6)]
        # The override is for one resonator of tone set 0
        r0 = int(next(iter(th['overrides'])))
        hmin = np.where(g['res_idxs'][:] == r0, 0.6, 0.5)
        np.testing.assert_allclose(g['hmin'][:], hmin)
        np.testing.assert_allclose(g['hmax'][:], 2.)
    assert grp['timestream1'].attrs['mode'] == 'settings from timestream0'
    assert [grp[f'timestream{j}'].attrs['tone_set'] for j in range(3)] == \
        [0, 0, 1]
    # The calibrated x of whole runs is kept with the calibration
    assert 'ts1_set1' in root['ts_cal0/applied']
    # Filters of the first timestream, no window
    m.main(m.parse_args(argv + ['--reuse', 'filter', '--no-gui']))
    g = dl.open_store(path)['ts_photons1']
    assert g['timestream1'].attrs['mode'] == 'filter from timestream0'
    check_photons(g['timestream1'], logs[1], dt)
    np.testing.assert_allclose(g['timestream1/filter'][:],
                               g['timestream0/filter'][:])
    # From scratch: the window opens for each timestream
    windows.clear()
    m.main(m.parse_args(argv + ['--reuse', 'none']))
    assert windows == ['ts0', 'ts1', 'ts1']
    g = dl.open_store(path)['ts_photons2']
    assert g['timestream1'].attrs['mode'] == 'full'
