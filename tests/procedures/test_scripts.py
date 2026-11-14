"""
Tests for the procedure scripts in ``procedures/``, run against a fake CRS.
"""
import asyncio
import importlib.util
import os
import numpy as np
import pytest
import zarr
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
    f, z = m.main(m.parse_args(
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
    # Nothing is saved to zarr
    assert not any(p.suffix == '.zarr' for p in tmp_path.iterdir())


def test_vna_quick_check_no_csv(monkeypatch, tmp_path):
    m = load_script('vna_quick_check.py')
    fake_factory(monkeypatch, m, FakeVnaCRS())
    f, z = m.main(m.parse_args(['--no-gui']))
    assert len(f) == len(z) > 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('nyquist, lo, hi', [('2', 2.5e9, 5e9),
                                             ('both', 100e6, 5e9)])
def test_vna_quick_check_nyquist(monkeypatch, nyquist, lo, hi):
    m = load_script('vna_quick_check.py')
    fake = FakeVnaCRS(modules = (1, 2))
    fake_factory(monkeypatch, m, fake)
    f, _ = m.main(m.parse_args(
        ['--modules', '1', '2', '--nyquist', nyquist, '--qr-median', '1e4',
         '--pts-per-lw', '1', '--no-gui']))
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
                               '--amplitude', '-45']))
    assert out == 'closed' and seen['crs'] is fake
    assert seen['module_idxs'] == [2] and seen['nyquist'] == 'both'
    assert seen['amplitude'] == -45. and 'count' not in seen
    assert seen['qr_median'] == m.QR_MEDIAN
    assert seen['pts_per_lw'] == m.PTS_PER_LW
    assert made[0]['global_control'] is False
    # The window can reconnect to other modules
    assert [mk['module_idxs'] for mk in made] == [[2], [1, 3]]
    assert fake.calls[0] == ('attach', dict(full_scale_dbm = 7.))


def test_vna_view_script(monkeypatch, tmp_path):
    """
    The script opens the window on the store and sweep it is given.
    """
    m = load_script('vna_view.py')
    seen = []

    def fake_view(path, k):
        """
        Record the arguments.

        Parameters:
        path (str): store path.
        k (int or None): sweep index.

        Returns:
        k (int): a sweep index.
        """
        seen.append((path, k))
        return 2 if k is None else k

    monkeypatch.setattr(m, 'run_vna_view', fake_view)
    assert m.main(m.parse_args(['--data-dir', str(tmp_path), '--cooldown',
                                'cd1', '--device', 'devA'])) == 2
    assert seen[-1] == (str(tmp_path / 'cd1' / 'devA.zarr'), None)
    assert m.main(m.parse_args(['--store', '~/x.zarr', '--vna', '1'])) == 1
    assert seen[-1] == (os.path.expanduser('~/x.zarr'), 1)


@pytest.mark.parametrize('script', [
    '01_vna_sweep.py', '02_find_resonances.py', '03_tune.py', '04_stream.py',
    '05_calibrate_streams.py', '06_find_photons.py', '07_psds.py',
    'archive_store.py',
    'vna_view.py', 'vna_report.py', 'store_browser.py'])
def test_store_path_suffix(script, tmp_path):
    """
    Every procedure names the store <device>_<suffix>.zarr, or
    <device>.zarr (no underscore) with the default blank suffix.
    """
    m = load_script(script)
    assert m.SUFFIX == ''
    base = ['--data-dir', str(tmp_path), '--cooldown', 'cd1', '--device',
            'devA']
    assert m.store_path(m.parse_args(base)) == \
        str(tmp_path / 'cd1' / 'devA.zarr')
    assert m.store_path(m.parse_args(base + ['--suffix', 'warm'])) == \
        str(tmp_path / 'cd1' / 'devA_warm.zarr')
    assert m.store_path(m.parse_args(base + ['--suffix', ''])) == \
        str(tmp_path / 'cd1' / 'devA.zarr')


def test_store_browser_script(monkeypatch, tmp_path, capsys):
    """
    The script opens the browser on the store it is given, or prints the
    runs with --runs.
    """
    m = load_script('store_browser.py')
    seen = []

    def fake_browser(path):
        """
        Record the path and pretend the user picked a store.

        Parameters:
        path (str or None): the path given.

        Returns:
        path (str): the store shown.
        """
        seen.append(path)
        return path or '/picked.zarr'

    monkeypatch.setattr(m, 'run_store_browser', fake_browser)
    store = tmp_path / 'cd1' / 'devA_warm.zarr'
    dl.create_store(store, 'devA', 'cd1')
    base = ['--data-dir', str(tmp_path), '--cooldown', 'cd1', '--device',
            'devA', '--suffix', 'warm']
    assert m.main(m.parse_args(base)) == str(store)
    # The dialog chooses the store with --pick, or when it doesn't exist
    assert m.main(m.parse_args(base + ['--pick'])) == '/picked.zarr'
    assert m.main(m.parse_args(base[:-2])) == '/picked.zarr'
    assert seen == [str(store), None, None]
    assert 'does not exist' in capsys.readouterr().out
    path = tmp_path / 'r.zarr'
    root = dl.create_store(path, 'devA', 'cd1')
    dl.save_resonators(root, 0, [1e9], 2e4, 'x')
    tune, _ = dl.new_tuning(root, 0)
    dl.register_stream_run(tune, dict(
        name = 'dark', notes = 'lights off', dec_stage = 6, duration_s = 2.,
        tuning = 'tune0', tone_set = 0, calibration = 'tune0/cal0',
        paths = []))
    assert m.main(m.parse_args(['--store', str(path), '--runs'])) == \
        str(path)
    out = capsys.readouterr().out
    assert len(seen) == 3
    assert '2 top-level groups' in out and 'ts0: dark, tune0' in out
    assert 'lights off' in out


def test_vna_view_script_window(tmp_path):
    """
    The script really opens the view window on a simulated store.
    """
    from pyqtgraph.Qt import QtCore, QtWidgets
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 6,
                 pair_frac = 0.)
    asyncio.run(simulate_vna_store(path, device_kwargs = small, step = 5e-5,
                                   verbose = False))
    m = load_script('vna_view.py')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QtCore.QTimer.singleShot(0, app.quit)
    assert m.main(m.parse_args(['--store', str(path)])) == 0


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


@pytest.mark.asyncio
async def test_01_vna_sweep_view(monkeypatch, tmp_path):
    """
    With --view (or the VIEW setting), the sweep opens in the view window
    once saved; --no-view overrides the setting.
    """
    m = load_script('01_vna_sweep.py')
    fake_factory(monkeypatch, m, FakeVnaCRS(modules = (1,)))
    seen = []

    def fake_view(root, k):
        """
        Record the store and sweep shown.

        Parameters:
        root (zarr.Group): the store root.
        k (int): sweep index.
        """
        seen.append((root, k))

    monkeypatch.setattr(m, 'run_vna_view', fake_view)
    argv = ['--data-dir', str(tmp_path), '--cooldown', 'cd1', '--device',
            'devA', '--qr-median', '1e4', '--pts-per-lw', '1']
    assert await m.main(m.parse_args(argv)) == 0 and seen == []
    assert await m.main(m.parse_args(argv + ['--view'])) == 1
    root, k = seen[-1]
    assert k == 1 and root.attrs['device'] == 'devA'
    # The sweep was saved before the window opened
    assert len(dl.load_vna_sweep(root, 1)[0]) > 0
    monkeypatch.setattr(m, 'VIEW', True)
    assert await m.main(m.parse_args(argv)) == 2 and seen[-1][1] == 2
    assert await m.main(m.parse_args(argv + ['--no-view'])) == 3
    assert len(seen) == 2


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


def test_03_tune_frequency_range(tmp_path, monkeypatch, capsys):
    """
    FMIN/FMAX (--fmin/--fmax) keep only the resonators in a frequency
    range as tones; the ones outside still count as close neighbours.
    """
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.3e9, n_banks = 1, n_per_bank = 20,
                 pair_frac = 0., yield_frac = 1.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, step = 1e-4, verbose = False))
    fr = np.sort(truth.fr.values)
    dl.save_resonators(root, 0, fr, np.full(len(fr), 2e4), 'test',
                       Qr = np.full(len(fr), 1e5))
    m = load_script('03_tune.py')
    monkeypatch.setattr(m, 'MIN_SEP_LW', 0.)
    argv = ['--store', str(path), '--simulate']
    root = dl.open_store(path, mode = 'a')
    # Default: no limit
    assert m.frequency_range(fr).all()
    tune = m.plan(root, m.parse_args(argv), str(path))
    assert len(dl.load_tones(tune['tones_init'])['fres']) == len(fr)
    assert tune['tones_init'].attrs['fmin'] is None
    assert tune['tones_init'].attrs['n_out_of_range'] == 0
    assert 'outside' not in capsys.readouterr().out
    # The range is inclusive at both ends
    lo, hi = fr[5], fr[12]
    tune = m.plan(root, m.parse_args(
        argv + ['--fmin', repr(lo), '--fmax', repr(hi)]), str(path))
    t = dl.load_tones(tune['tones_init'])
    np.testing.assert_allclose(np.sort(t['fres']), fr[5:13])
    np.testing.assert_array_equal(np.sort(t['res_idxs']), np.arange(5, 13))
    a = tune['tones_init'].attrs
    assert a['fmin'] == lo and a['fmax'] == hi and a['n_out_of_range'] == 12
    assert f'12 outside {lo / 1e9:.6g}-{hi / 1e9:.6g} GHz' in \
        capsys.readouterr().out
    # Only one end
    tune = m.plan(root, m.parse_args(argv + ['--fmax', repr(fr[3])]),
                  str(path))
    assert len(dl.load_tones(tune['tones_init'])['fres']) == 4
    assert 'outside below' in capsys.readouterr().out
    # A resonance just outside the range still makes its neighbour too
    # close: a second list with an extra resonance 3 linewidths below fr[6]
    extra = fr[6] - 3 * fr[6] / 1e5
    fr2 = np.sort(np.append(fr, extra))
    dl.save_resonators(root, 1, fr2, np.full(len(fr2), 2e4), 'test',
                       Qr = np.full(len(fr2), 1e5))
    monkeypatch.setattr(m, 'MIN_SEP_LW', 5.)
    tune = m.plan(root, m.parse_args(
        argv + ['--resonators', '1', '--fmin', repr(fr[6])]), str(path))
    t = dl.load_tones(tune['tones_init'])
    assert fr[6] not in t['fres'] and extra not in t['fres']
    np.testing.assert_allclose(np.sort(t['fres']), fr[7:])
    a = tune['tones_init'].attrs
    assert a['n_dropped'] == 2 and a['n_out_of_range'] == 6
    monkeypatch.setattr(m, 'MIN_SEP_LW', 0.)
    # Errors
    with pytest.raises(ValueError, match = 'above FMAX'):
        m.plan(root, m.parse_args(argv + ['--fmin', '2e9', '--fmax', '1e9']),
               str(path))
    with pytest.raises(ValueError, match = 'no tones'):
        m.plan(root, m.parse_args(argv + ['--fmin', '2e9']), str(path))


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


def check_override(g):
    """
    Check the thresholds made in the stand-in window (kHz/GHz): hmax 3.5 for
    every tone, hmin 1.2 for one resonator of tone set 0.

    Parameters:
    g (zarr.Group): a ``timestream{j}`` group.
    """
    th = g.attrs['thresholds']
    assert th['hmax'] == 3.5 and list(th['overrides'].values()) == \
        [dict(hmin = 1.2)]
    r0 = int(next(iter(th['overrides'])))
    np.testing.assert_allclose(g['hmin'][:], np.where(
        g['res_idxs'][:] == r0, 1.2, 1.))
    np.testing.assert_allclose(g['hmax'][:], 3.5)


def test_06_find_photons(photon_store, monkeypatch, capsys):
    """
    List the timestreams (one per run and tone set); search run 0's tone
    set 0 from scratch (the window changes its thresholds); then search
    other timestreams with that result as the reference: its thresholds,
    or its filters; and another from scratch. No timestream is saved.
    """
    import citkid.pipeline.interactive.photon_thresholds as pt
    path, log = photon_store
    m = load_script('06_find_photons.py')
    argv = ['--store', str(path)]
    monkeypatch.setattr(m, 'TIMESTREAMS', [])
    assert m.main(m.parse_args(argv)) is None
    out = capsys.readouterr().out
    assert "0: ts0, 'dark0', set 0" in out and \
        "3: ts1, 'dark0b', set 1" in out and 'Previous results' in out
    windows = []

    def fake_window(ts, results, thresholds, rerun, data = None,
                    settings = None, title = ''):
        """
        Stand in for the threshold window: widen hmax for all tones, give
        one tone its own hmin, and rerun them all (as the window does).

        Parameters:
        ts (dict): the timestream.
        results (dict): its results.
        thresholds (Thresholds): its thresholds.
        rerun (callable): reruns tones.
        data (callable): a tone's detection timestream, for the snippet.
        settings (DetectionSettings): the detection settings.
        title (str): window title.

        Returns:
        thresholds, results: as the window.
        """
        windows.append(ts['run'])
        # The window gets the data for its snippet and the tau settings
        d = data(int(ts['rows'][0]))
        assert isinstance(d, np.ndarray) and len(d) == ts['stop'] - ts['start']
        assert settings.tau_skip == 2 and settings.tau_lifetimes == 5.
        # Thresholds are in kHz/GHz; the simulated photons are 2 kHz/GHz
        thresholds.hmax = 3.5
        r0 = int(ts['rows'][0])
        thresholds.overrides[int(ts['res_idxs'][0])] = dict(hmin = 1.2)
        results = {**results, **rerun(ts['rows'], thresholds)}
        assert results[r0]['hmin'] == 1.2 and results[r0]['hmax'] == 3.5
        return thresholds, results

    monkeypatch.setattr(pt, 'run_photon_thresholds', fake_window)
    # Simulated streams of each timestream (see photon_store): ts0 set 0,
    # ts0 set 1, ts1 set 0, ts1 set 1
    logs = (log[:1], log[1:3], log[3:4], log[4:6])
    # From scratch, with the window
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv))
    assert windows == ['ts0']
    root = dl.open_store(path)
    g0 = root['ts_photons0']
    assert g0.attrs['source'] == 'ts_cal0' and g0.attrs['kind'] == \
        'analysis' and g0.attrs['reference'] is None
    dt = g0['timestream0'].attrs['dt']
    check_photons(g0['timestream0'], logs[0], dt)
    check_override(g0['timestream0'])
    assert g0['timestream0'].attrs['mode'] == 'full'
    # Its thresholds for both tone sets of run 1, no window
    monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0b'])
    m.main(m.parse_args(argv + ['--reference', '0']))
    assert windows == ['ts0']
    g1 = dl.open_store(path)['ts_photons1']
    assert g1.attrs['reference'] == 'ts_photons0' and \
        g1.attrs['reuse'] == 'settings'
    for j, entries in ((0, logs[2]), (1, logs[3])):
        g = g1[f'timestream{j}']
        assert g.attrs['mode'] == 'settings from ts_photons0/timestream0'
        assert g.attrs['tone_set'] == j
        check_photons(g, entries, dt)
        check_override(g)
    # Its filters for run 1's tone set 0: heights in its photon units
    monkeypatch.setattr(m, 'TIMESTREAMS', [2])
    m.main(m.parse_args(argv + ['--reference', '0', '--reuse', 'filter']))
    g2 = dl.open_store(path)['ts_photons2/timestream0']
    assert g2.attrs['mode'] == 'filter from ts_photons0/timestream0'
    check_photons(g2, logs[2], dt)
    np.testing.assert_allclose(g2['amp_ph'][:],
                               g0['timestream0/amp_ph_own'][:])
    # Another timestream from scratch: the window opens again
    monkeypatch.setattr(m, 'TIMESTREAMS', [1])
    m.main(m.parse_args(argv))
    assert windows == ['ts0', 'ts0']
    check_photons(dl.open_store(path)['ts_photons3/timestream0'], logs[1],
                  dt)
    # No timestream is saved: every saved array is far shorter than one
    root = dl.open_store(path)
    nsamples = g0['timestream0'].attrs['stop']
    sizes = [max(v.shape)
             for grp in ['ts_cal0'] + [f'ts_photons{q}' for q in range(4)]
             for _, v in root[grp].members(max_depth = None)
             if isinstance(v, zarr.Array) and v.shape]
    assert max(sizes) < nsamples / 10


class FakeNcoCRS:
    """
    Fake board for nco_phase_check.py: tones read a fixed phase per
    frequency, rotated by a frame angle that every forced NCO setting
    changes by ``step_deg``.
    """

    def __init__(self, step_deg = 30.):
        self.step_deg = step_deg
        self.frame = 0.
        self.calls = []
        self.nco_freqs = {}

    async def attach(self, **kw):
        self.calls.append(('attach', kw))

    async def set_nco(self, nco_freqs, verbose = True, force = False):
        self.calls.append(('set_nco', dict(nco_freqs), force))
        self.nco_freqs.update(nco_freqs)
        if force:
            self.frame += self.step_deg
        return sorted(nco_freqs)

    async def sweep(self, frequencies, ares, nsamps, verbose = True):
        f = np.asarray(frequencies, float)
        self.calls.append(('sweep', f.shape, int(nsamps)))
        phase = np.radians(self.frame) + 1e-9 * f
        return f, 0.5 * np.exp(1j * phase)

    async def clear_tones(self):
        self.calls.append(('clear_tones',))


def test_nco_phase_check(monkeypatch, capsys):
    m = load_script('nco_phase_check.py')
    fake = FakeNcoCRS(step_deg = 30.)
    made = fake_factory(monkeypatch, m, fake)
    out = m.main(m.parse_args(['--module', '2', '--nco', '6.5e8',
                               '--repeats', '2', '--nsamps', '50']))
    assert made == [dict(serial_number = m.SERIAL_NUMBER,
                         interface = m.INTERFACE, module_idxs = [2],
                         global_control = False)]
    assert fake.calls[0][0] == 'attach' and fake.calls[-1] == \
        ('clear_tones',)
    # Every NCO setting is forced, at the NCO or the away frequency
    sets = [c for c in fake.calls if c[0] == 'set_nco']
    assert all(c[2] for c in sets)
    assert {c[1][2] for c in sets} == {6.5e8, 6.5e8 + m.AWAY_HZ}
    assert len(out) == 2 and set(out[0]) == {'a', 'b', 'c', 'd'}
    # The control shows no change; the same setting shows the frame step,
    # away and back twice that, common to all tones
    dphi, ratio = m.phase_changes(out[0]['a'], out[0]['b'])
    assert np.allclose(dphi, 0.) and np.allclose(ratio, 1.)
    assert np.allclose(m.phase_changes(out[0]['b'], out[0]['c'])[0], 30.)
    assert np.allclose(m.phase_changes(out[0]['c'], out[0]['d'])[0], 60.)
    text = capsys.readouterr().out
    assert 'Control (no NCO setting)' in text
    assert 'common change +30.00 deg, spread across tones 0.00 deg' in text
    assert 'common change +60.00 deg' in text
    # The analysis wraps the phase and describes a tone-dependent change
    dphi, _ = m.phase_changes([1., 1j], [-1j, -1.])
    assert np.allclose(dphi, [-90., 90.])
    desc = m.describe('x', [1e6, 2e6], [1., 1.], [np.exp(0.5j), np.exp(1j)])
    assert 'spread across tones 14.32 deg' in desc


def test_07_psds(photon_store, monkeypatch, capsys):
    """
    List the timestreams of every calibration; analyze a set of them into
    one group (reduced frequencies shared, common modes removed per tone
    write); then without common-mode removal, with peak deglitching, and
    with the pulses found by a photon result's filters. No timestream is
    saved.
    """
    from citkid.pipeline import psds
    path, _ = photon_store
    m = load_script('07_psds.py')
    argv = ['--store', str(path)]
    monkeypatch.setattr(m, 'TIMESTREAMS', [])
    assert m.main(m.parse_args(argv)) is None
    out = capsys.readouterr().out
    assert "0: ts0, 'dark0', set 0" in out and 'ts_cal0/set1' in out
    assert 'Previous results' in out
    # Run 0's tone set 0 and both tone sets of run 1
    monkeypatch.setattr(m, 'TIMESTREAMS', [0, 'dark0b'])
    res = m.main(m.parse_args(argv))
    assert sorted(res) == ['timestream0', 'timestream1', 'timestream2']
    root = dl.open_store(path)
    g = root['ts_psd0']
    assert g.attrs['kind'] == 'analysis' and g.attrs['reference'] is None
    assert g.attrs['timestreams'] == [0, 'dark0b']
    assert g.attrs['settings']['cn_remove'] is True
    # 1.5 s at stage 0: 6.7 Hz to 954 Hz
    np.testing.assert_array_equal(g['f_reduced'][:], [10., 30., 100., 300.])
    dt = g['timestream0'].attrs['dt']
    nsamples = g['timestream0'].attrs['stop']
    levels = {}
    for j, (run, tone_set) in enumerate([('ts0', 0), ('ts1', 0),
                                         ('ts1', 1)]):
        t = g[f'timestream{j}']
        a = t.attrs
        assert a['run'] == run and a['tone_set'] == tone_set
        assert a['calibration'] == 'ts_cal0' and a['cn_removed'] is True
        assert a['fmin'] == pytest.approx(10 / (nsamples * dt))
        assert a['cn_lowpass_hz'] == pytest.approx(0.05 / dt)
        assert a['cn_highpass_hz'] == pytest.approx(a['fmin'])
        n = len(t['res_idxs'][:])
        assert all(s == '' for s in t['status'][:])
        assert t['valid_reduced'][:].all()
        nb = len(t['f_binned'][:])
        assert 50 < nb < 100 + 20
        for key in ('sxx_binned', 'sxx_binned_cn'):
            assert t[key].shape == (n, nb)
            assert np.isfinite(t[key][:]).all()
        for key in ('sxx_reduced', 'sxx_reduced_cn'):
            assert t[key].shape == (n, 4) and np.isfinite(t[key][:]).all()
        assert t['cn_weights'].shape[0] == n
        np.testing.assert_array_equal(t['n_glitches'][:], 0)
        assert np.all(t['write_idx'][:] >= 0)
        # The simulated noise is white (plus photon pulses): the reduced
        # values are flat and alike across tones
        sxx = t['sxx_reduced'][:]
        assert np.all(sxx < 3 * np.median(sxx))
        assert np.all(sxx > np.median(sxx) / 3)
        # The same tones in the two runs: the same level
        if tone_set in levels:
            np.testing.assert_allclose(np.median(sxx), levels[tone_set],
                                       rtol = 0.5)
        levels[tone_set] = np.median(sxx)
        # Uncorrelated tones: removing modes lowers the spectra a little
        # and never raises them much
        ratio = t['sxx_reduced_cn'][:] / sxx
        assert np.all(ratio < 1.3) and np.median(ratio) < 1.
        cm = t['common_modes']
        writes = sorted(cm.group_keys())
        assert len(writes) == (2 if tone_set == 1 else 1)
        for w in writes:
            wg = cm[w]
            k = wg.attrs['n_comp']
            assert 0 < k <= 3 and k < len(wg['rows'][:])
            assert wg['sxx_binned'].shape == (k, nb)
            assert wg['sigma_iter'].shape == (5, len(wg['rows'][:]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([cm[w]['rows'][:] for w in writes])),
            np.arange(n))
    loaded = psds.load_group(root, 0)
    assert len(loaded['timestreams']) == 3
    assert loaded['timestreams'][2]['common_modes'][1]['n_comp'] > 0
    assert 'common modes 3 modes' in psds.describe_results(root)
    # Without common-mode removal: no cn arrays
    monkeypatch.setattr(m, 'TIMESTREAMS', [dict(name = 'dark0b',
                                                tone_sets = [0], cal = 0)])
    m.main(m.parse_args(argv + ['--no-cn']))
    t = dl.open_store(path)['ts_psd1/timestream0']
    assert t.attrs['cn_removed'] is False and 'sxx_binned_cn' not in t
    assert 'common_modes' not in t and t.attrs['run'] == 'ts1'
    assert dl.open_store(path)['ts_psd1'].attrs['timestreams'] == \
        [dict(name = 'dark0b', tone_sets = [0], cal = 0)]
    # Peak deglitching finds the simulated pulses (about 30 photons/s per
    # tone plus cosmic rays)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv + ['--deglitch', 'peaks']))
    t = dl.open_store(path)['ts_psd2/timestream0']
    assert t.attrs['deglitch'] == 'peaks'
    ng = t['n_glitches'][:]
    # About 45 pulses per tone, found in proportion to each tone's
    # pulse signal-to-noise
    assert np.all(ng > 0) and 60 < ng.sum() < 600
    assert np.all(t['glitch_frac'][:] > 0)
    assert np.all(t['glitch_frac'][:] < 0.3)
    # With the pulses deglitched, the low-frequency spectrum is lower
    raw = dl.open_store(path)['ts_psd0/timestream0/sxx_reduced'][:, 0]
    assert np.median(t['sxx_reduced'][:, 0] / raw) < 0.7
    # A photon result's filters: cosmic rays only (a few per second), or
    # every pulse
    with pytest.raises(ValueError, match = 'PHOTONS'):
        m.main(m.parse_args(argv + ['--deglitch', 'photons']))
    root = dl.open_store(path, mode = 'a')
    if not dl.list_indices(root, 'ts_photons'):
        m6 = load_script('06_find_photons.py')
        monkeypatch.setattr(m6, 'TIMESTREAMS', [0])
        m6.main(m6.parse_args(argv + ['--no-gui']))
    p = dl.list_indices(dl.open_store(path), 'ts_photons')[0]
    m.main(m.parse_args(argv + ['--deglitch', 'photons', '--photons',
                                str(p)]))
    root = dl.open_store(path)
    g = root['ts_psd3']
    assert g.attrs['reference'] == f'ts_photons{p}'
    t = g['timestream0']
    n_cosmic = t['n_glitches'][:]
    assert np.all(n_cosmic >= 0) and np.all(n_cosmic < 30)
    assert n_cosmic.sum() > 0
    monkeypatch.setattr(m, 'GLITCH_PULSES', 'all')
    m.main(m.parse_args(argv + ['--deglitch', 'photons', '--photons',
                                str(p)]))
    t = dl.open_store(path)['ts_psd4/timestream0']
    assert np.all(t['n_glitches'][:] > n_cosmic)
    assert np.all(t['n_glitches'][:] > 20)
    # No timestream is saved
    nsamples = g['timestream0'].attrs['stop']
    sizes = [max(v.shape)
             for q in dl.list_indices(root, 'ts_psd')
             for _, v in root[f'ts_psd{q}'].members(max_depth = None)
             if isinstance(v, zarr.Array) and v.shape]
    assert max(sizes) < nsamples / 10
