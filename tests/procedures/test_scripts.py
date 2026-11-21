"""
Tests for the procedure scripts in ``procedures/``, run against a fake CRS.
"""
import asyncio
import importlib.util
import os
import sys
import numpy as np
import pytest
import zarr
from citkid import daq_layout as dl
from ..crs.fake_crs import FakeVnaCRS

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', '..', 'procedures')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)  # the scripts' settings.py


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
    '01_vna_sweep.py', '02_find_resonances.py', '03_tune.py',
    '04a_stream.py', '04b_stream_batch.py',
    '05_calibrate_streams.py', '06a_find_photons.py', '06b_hee.py',
    '07_psds.py',
    'archive_store.py',
    'vna_view.py', 'store_browser.py'])
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
    # A bare Windows drive is its root, not the current folder on it
    assert dl.store_path('Z:', 'cd1', 'devA') == \
        os.path.join('Z:' + os.sep, 'cd1', 'devA.zarr')
    # A wrong path: the error names it in full
    with pytest.raises(ValueError, match = 'nowhere.zarr.*not a DAQ'):
        dl.open_store(str(tmp_path / 'nowhere.zarr'), mode = 'a')


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
    names = sorted(os.listdir(SCRIPTS)) + [
        os.path.join('simulation', n)
        for n in sorted(os.listdir(os.path.join(SCRIPTS, 'simulation')))]
    for name in names:
        if not name.endswith('.py'):
            continue
        with open(os.path.join(SCRIPTS, name), encoding = 'utf-8') as fh:
            text = fh.read()
        assert all(len(line) <= 80 for line in text.splitlines()), name
        assert 'Settings' in text, name
        if name not in ('settings.py', 'analysis_template.py'):
            assert "if __name__ == '__main__':" in text, name


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


def test_simulate_store(tmp_path, monkeypatch):
    m = load_script('simulation/simulate_store.py')
    monkeypatch.setattr(m, 'PULSES', dict(rate = 5.))
    path = tmp_path / 'sim.zarr'
    truth = asyncio.run(m.main(m.parse_args(
        ['--store', str(path), '--seed', '2', '--pts-per-lw', '2.5'])))
    root = dl.open_store(path)
    assert sorted(root.group_keys()) == ['simulation', 'vna0']
    assert root['vna0'].attrs['step'] == pytest.approx(2e-5)
    assert len(truth) > 500 and root['simulation'].attrs['seed'] == 2
    assert root['simulation'].attrs['pulses'] == dict(rate = 5.)


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
    answers = []
    monkeypatch.setattr(m, 'ask_make_tones', lambda t: answers.pop(0))
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
    # A tone marked bad in the store's tone marks (any stage) leaves the
    # tuned set on the next tones step
    root = dl.open_store(path, mode = 'a')
    first = int(tune['fq_sweep/res_idxs'][0])
    dl.add_tone_mark(root, first, 'bad', 'double resonance', 'test')
    m.main(m.parse_args(argv + ['--redo', 'tones']))
    assert 'marked bad' in capsys.readouterr().out
    t2 = dl.load_tones(dl.open_store(path)['tune0/tones'])
    assert len(t2['fres']) == n - 2 and first not in t2['res_idxs']
    dl.clear_tone_mark(root, first, 'test', 'test')
    m.main(m.parse_args(argv + ['--redo', 'tones']))
    t = dl.load_tones(dl.open_store(path)['tune0/tones'])
    assert len(t['fres']) == n - 1
    # A new tuning from tune0's tones: its powers are kept, the plan,
    # power sweep and power fits are skipped, and only the frequencies
    # and spans are optimized again, from PREDICT's guess
    monkeypatch.setattr(m, 'PREDICT',
                        lambda f, q, a: (f + 5e3, q * 2., a - 1.))
    # Porting needs a new name: an existing tuning is not continued
    with pytest.raises(ValueError, match = 'new name'):
        m.main(m.parse_args(argv + ['--port-tune', '0']))
    with pytest.raises(KeyError, match = 'no tuning'):
        m.main(m.parse_args(argv + ['--tuning', '5']))
    assert m.main(m.parse_args(argv + ['--tuning', 'ported',
                                       '--port-tune', '0'])) == 1
    assert calls == ['iq', 'fq', 'fq']
    root = dl.open_store(path)
    t1 = root['tune1']
    assert dl.tuning_name(t1) == 'ported' and \
        dl.tuning_name(root['tune0']) == 'tune0'
    assert dl.find_tuning(root, 'ported').basename == 'tune1'
    assert t1.attrs['ported_powers'] and 'power_sweep' not in t1
    assert t1.attrs['ported_from'] == dict(store = None, group = 'tune0')
    t0 = dl.load_tones(root['tune0/tones'])
    init = dl.load_tones(t1['tones_init'])
    np.testing.assert_allclose(init['fres'], t0['fres'] + 5e3)
    np.testing.assert_allclose(init['qres'], t0['qres'] * 2.)
    np.testing.assert_allclose(t1['power_choice/ares'][:], t0['ares'] - 1.)
    assert not t1['power_choice/fallback'][:].any()
    np.testing.assert_allclose(t1['fq_sweep/ares'][:], t0['ares'] - 1.)
    assert len(dl.load_tones(t1['tones'])['fres']) == len(t0['fres']) - 1
    assert 'at its powers, frequencies, spans and powers predicted' in \
        capsys.readouterr().out
    # Run again with the same settings: the ported tuning is continued;
    # with another source it is refused
    assert m.main(m.parse_args(argv + ['--tuning', 'ported',
                                       '--port-tune', '0'])) == 1
    assert 'continuing tune1' in capsys.readouterr().out
    with pytest.raises(ValueError, match = 'not ported from'):
        m.main(m.parse_args(argv + ['--tuning', 'ported',
                                    '--port-tune', 'ported']))
    # fq_finder closed early: stop, or make the tones from the set ones
    def unfinished(f, z, fres, qres, res_idxs, zarr_grp, **kwargs):
        """
        Stand in for fq_finder, leaving the first resonator unset.

        Parameters:
        f, z, fres, qres, res_idxs, zarr_grp, **kwargs: see
            ``fake_fqfinder``.
        """
        if 'fres_opt' not in zarr_grp:
            fake_fqfinder(f, z, fres, qres, res_idxs, zarr_grp, **kwargs)
        zarr_grp['fres_opt'][0] = np.nan

    monkeypatch.setattr(m, 'run_fqfinder', unfinished)
    answers[:] = [False]
    assert m.main(m.parse_args(argv + ['--tuning', 'part',
                                       '--port-tune', '0'])) == 2
    out = capsys.readouterr().out
    assert 'not finished' in out and 'tones' not in root['tune2']
    answers[:] = [True]
    monkeypatch.setattr(m, 'run_fqfinder', unfinished)
    assert m.main(m.parse_args(argv + ['--tuning', 'part',
                                       '--port-tune', '0'])) == 2
    assert len(dl.load_tones(root['tune2/tones'])['fres']) == \
        len(t0['fres']) - 2
    monkeypatch.setattr(m, 'PREDICT', lambda f, q, a: (f[:1], q, a))
    with pytest.raises(ValueError, match = 'same shape'):
        m.main(m.parse_args(argv + ['--tuning', 'p2', '--port-tune',
                                    'ported']))
    monkeypatch.setattr(m, 'PREDICT', None)
    # The tuning ports without its sweeps
    dl.port_tuning(path, 0, tmp_path / 'b.zarr')
    b = dl.open_store(tmp_path / 'b.zarr')
    assert sorted(b['tune0'].group_keys()) == ['crs_config', 'tones']
    np.testing.assert_array_equal(dl.load_tones(b['tune0/tones'])['ares'],
                                  t['ares'])


def test_04a_stream(tmp_path, monkeypatch):
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
    m = load_script('04a_stream.py')
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
    assert '04a_stream.py' in seen['holder']['owner']
    assert board_lock.holder(0) is None
    assert seen['presets'] == m.PRESETS
    assert seen['sweep_kwargs']['npoints_fine'] == m.NPOINTS_FINE
    assert seen['fres_update'] is None
    with pytest.raises(KeyError, match = 'no tones'):
        m.main(m.parse_args(['--store', str(path), '--simulate',
                             '--tuning', '1']))
    # Simulated photons and HEEs come from the store (simulate_store.py)
    assert seen['crs'].pulses is None
    root['simulation'].attrs['pulses'] = dict(rate = 5.)
    m.main(m.parse_args(['--store', str(path), '--simulate']))
    assert seen['crs'].pulses['rate'] == 5.


def test_04b_stream_batch(tmp_path, monkeypatch, capsys):
    """
    Step 04b calibrates the tone set (moving the tones to the sweep's
    resonances when asked), writes its tones, takes RUNS and frees the
    board; CALIBRATION reuses a calibration instead of sweeping again.
    """
    from citkid.crs import board_lock
    from citkid.crs.sim import simulate_vna_store, simulated_crs
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 6,
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
    # The tones a little off the resonances: the update moves them back
    dl.save_tones(dl.create_group(tune, 'tones', 'decision'),
                  fr + 0.3 * fr / qr, qr / 5, -55., np.arange(n),
                  tone_set = np.zeros(n, int), tone_module = np.ones(n, int))
    m = load_script('04b_stream_batch.py')
    monkeypatch.setattr(m, 'RUNS', [
        dict(name = 'a', dec_stage = 6, duration_s = 0.2),
        dict(name = 'b', dec_stage = 6, duration_s = 0.1, notes = 'n')])
    monkeypatch.setattr(m, 'NPOINTS_FINE', 60)
    monkeypatch.setattr(m, 'NPOINTS_GAIN', 10)
    monkeypatch.setattr(m, 'NSAMPS', 10)
    monkeypatch.setattr(m, 'FRES_UPDATE', dict(method = 'spacing',
                                                max_shift = 0.1))
    # The same board for both runs, as on hardware (a new simulated one
    # would not know the NCO settings the calibration was taken at)

    async def same_board(args, root, path):
        """
        Stand in for connect: the board made above.

        Parameters:
        args, root, path: ignored.

        Returns:
        crs (SimulatedCRS): the board.
        """
        return crs

    monkeypatch.setattr(m, 'connect', same_board)
    argv = ['--store', str(path), '--simulate', '--notes', 'cold']
    result = m.main(m.parse_args(argv))
    assert result['calibration'] == 'tune0/cal0'
    assert result['runs'] == [['tune0/cal0/write0/ts0'],
                              ['tune0/cal0/write0/ts1']]
    assert board_lock.holder(0) is None
    root = dl.open_store(path)
    runs = dl.stream_runs(root)
    assert [r['name'] for r in runs] == ['a', 'b'] and runs[1]['notes'] == 'n'
    cal = root['tune0/cal0']
    assert cal.attrs['notes'] == 'cold'
    up = cal.attrs['fres_update']
    assert up['method'] == 'spacing' and up['n_updated'] >= n - 1
    assert up['n_outside'] <= 1 and 'fres_sweep' in cal
    step = (fr / qr * 5) / 60
    moved = cal['fres'][:] != cal['fres_sweep'][:]
    assert np.all(np.abs(cal['fres'][:][moved] - fr[moved]) < 6 * step[moved])
    assert np.all(np.abs(cal['fres_sweep'][:] - fr - 0.3 * fr / qr) < 1.)
    out = capsys.readouterr().out
    assert 'moved to the sweep' in out and "a: ['tune0/cal0/write0/ts0']" in out
    # Reusing the calibration: no new sweep, the tones written again
    monkeypatch.setattr(m, 'CALIBRATION', 'tune0/cal0')
    monkeypatch.setattr(m, 'RUNS', [dict(name = 'c', dec_stage = 6,
                                         duration_s = 0.1)])
    result = m.main(m.parse_args(argv))
    assert result['calibration'] == 'tune0/cal0'
    assert result['runs'] == [['tune0/cal0/write1/ts2']]
    assert [c['name'] for c in dl.calibrations(root['tune0'])] == ['cal0']


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
    # One DataSet per calibration sweep: set 0 from the first entry,
    # and set 1 of each entry on its own sweep (the second run
    # calibrated set 1 anew), so never joined
    names = sorted(n for n in ds if not n.endswith('_caltones'))
    assert names[0] == 'set0' and len(names) == 3 and \
        all(n.startswith('set1_cal') for n in names[1:])
    root = dl.open_store(tmp_path / 's.zarr')
    cal = root['ts_cal0']
    assert cal.attrs['kind'] == 'analysis' and cal.attrs['selection'] == sel
    tone_set = root['tune0/tones/tone_set'][:]
    for name in names:
        d, s = ds[name], (0 if name == 'set0' else 1)
        assert len(d.res_idxs[:]) == int((tone_set == s).sum())
        assert np.isfinite(np.asarray(d.xt[0])).all()
        assert len(np.asarray(d.source_bounds)) == 1
    # Same choice: same cal group; a new one otherwise
    m.main(m.parse_args(argv))
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr'),
                           'ts_cal') == [0]
    monkeypatch.setattr(m, 'CALIBRATE', sel[:1])
    m.main(m.parse_args(argv))
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr'),
                           'ts_cal') == [0, 1]
    # CALIBRATION (--cal) continues a group that is not the latest,
    # by index in any form, if it was made with the same choice
    monkeypatch.setattr(m, 'CALIBRATE', sel)
    for which in ('0', 'ts_cal0'):
        m.main(m.parse_args(argv + ['--cal', which]))
        assert "Continuing ts_cal0 ('ts_cal0')" in \
            capsys.readouterr().out
    assert dl.list_indices(dl.open_store(tmp_path / 's.zarr'),
                           'ts_cal') == [0, 1]
    with pytest.raises(ValueError, match = 'another choice'):
        m.main(m.parse_args(argv + ['--cal', '1']))
    with pytest.raises(ValueError, match = 'no ts_cal group'):
        m.main(m.parse_args(argv + ['--cal', '7']))
    with pytest.raises(ValueError, match = 'not an index'):
        m.main(m.parse_args(argv + ['--cal', '0', '--new']))
    # A name no calibration has starts a new one with it, which the
    # name then continues; --new with the name starts another
    m.main(m.parse_args(argv + ['--cal', 'bbt']))
    assert "Calibrating into ts_cal2 ('bbt')" in capsys.readouterr().out
    m.main(m.parse_args(argv + ['--cal', 'bbt']))
    assert "Continuing ts_cal2 ('bbt')" in capsys.readouterr().out
    m.main(m.parse_args(argv + ['--cal', 'bbt', '--new']))
    assert "Calibrating into ts_cal3 ('bbt')" in capsys.readouterr().out
    root = dl.open_store(tmp_path / 's.zarr')
    assert dl.find_calibration(root, 'bbt').path == 'ts_cal3'
    assert dl.calibration_name(root['ts_cal3']) == 'bbt'
    assert dl.item_name(root['ts_cal3']) == 'bbt'
    assert dl.calibration_name(root['ts_cal0']) == 'ts_cal0'
    assert dl.find_calibration(root, 2).path == 'ts_cal2'
    assert dl.find_calibration(root, 'nope') is None
    monkeypatch.setattr(m, 'CALIBRATE', sel[:1])
    # The window is opened unless --no-gui
    opened = []
    monkeypatch.setattr(m, 'run_ts_analysis',
                        lambda AR, title = '': opened.append(title))
    m.main(m.parse_args(argv[:2]))
    # One window per tone set
    # A new group: the latest (ts_cal3, 'bbt') has another choice
    assert len(opened) == 2 and all('ts_cal4' in t for t in opened)
    # FIT_IQ: the window with the IQ fit panel, then every row fit in
    # batch; the DataSet records it
    iq_opened = []
    monkeypatch.setattr(m, 'run_iq_ts_analysis',
                        lambda AR, title = '': iq_opened.append(title))
    monkeypatch.setattr(m, 'FIT_IQ', True)
    ds = m.main(m.parse_args(argv[:2] + ['--new']))
    assert len(iq_opened) == 2 and not opened[2:]
    root = dl.open_store(tmp_path / 's.zarr')
    c = dl.list_indices(root, 'ts_cal')[-1]
    for name in (n for n in ds if not n.endswith('_caltones')):
        d = ds[name]
        assert d.saved_row_mask('iq_popt').all()
        popt = np.asarray(d.iq_popt[:], float)
        assert popt.shape[1] == 8 and np.all(popt[:, 1] > 0)
        assert root[f'ts_cal{c}/{name}'].attrs['iq_fit'] is True
    # Fitting again does nothing to the rows already fit
    from citkid.pipeline.analysis import AnalysisRunner
    assert len(m.fit_resonances(
        AnalysisRunner(ds['set0'], analysis_yaml_path = 'ts'),
        dl.open_store(tmp_path / 's.zarr', mode = 'a')[f'ts_cal{c}/set0'])
           ) == 0


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


@pytest.fixture(scope = 'module')
def photon_store(tmp_path_factory):
    """
    Make a store with two calibrated stage-0 runs of a simulated device with
    photon pulses and HEEs.

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
    crs.pulses = dict(DEFAULT_PULSES, rate = 30., hee_rate = 5.)

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
    # High-energy events: the simulated ones, each on every tone of its write
    ev, hits = grp['hee/events'], grp['hee/hits']
    n_true = sum(len(entry['hee']) for entry in log)
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
    or its filters; another from scratch under a name; continue from a
    saved result (bad marks with reasons, per-tone decay fits); refit
    the decay times; migrate results of the earlier layout; leave
    uncalibrated tones out. One result per timestream, replaced on a
    rerun. No timestream is saved.
    """
    import citkid.pipeline.interactive.photon_thresholds as pt
    from citkid.pipeline import photon_finder as pf
    from citkid.zarr_util import copy_group
    path, log = photon_store
    m = load_script('06a_find_photons.py')
    argv = ['--store', str(path)]
    monkeypatch.setattr(m, 'TIMESTREAMS', [])
    assert m.main(m.parse_args(argv)) is None
    out = capsys.readouterr().out
    assert "0: ts0, 'dark0', set 0" in out and \
        "3: ts1, 'dark0b', set 1" in out and 'Previous results' in out
    windows = []

    def fake_window(ts, results, thresholds, rerun, data = None,
                    settings = None, title = '', **kwargs):
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
        # The window gets the data for its snippet and the fit settings
        d = data(int(ts['rows'][0]))
        assert isinstance(d, np.ndarray) and len(d) == ts['stop'] - ts['start']
        assert settings.tau_skip == 2 and settings.tau_fit_s == 1e-3
        assert thresholds.tau_skip == 2 and thresholds.tau_fit_s == 1e-3
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
    # From scratch, with the window: the result is named after the run
    # and tone set (the run has two tone sets)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv))
    assert windows == ['ts0']
    root = dl.open_store(path)
    assert pf.list_results(root) == ['dark0_set0']
    g0 = root['photons/dark0_set0']
    assert g0.attrs['calibration'] == 'ts_cal0' and g0.attrs['kind'] == \
        'analysis' and g0.attrs['reference'] is None
    assert g0.attrs['continued_from'] is None
    dt = g0.attrs['dt']
    check_photons(g0, logs[0], dt)
    check_override(g0)
    assert g0.attrs['mode'] == 'full'
    np.testing.assert_array_equal(g0['tau_skip'][:], 2)
    np.testing.assert_allclose(g0['tau_fit_s'][:], 1e-3)
    assert not g0['bad'][:].any() and all(r == '' for r in g0['bad_reason'])
    assert 'dark0_set0: ts0 set 0, ts_cal0, full' in pf.describe_results(root)
    # Its thresholds for both tone sets of run 1, no window
    monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0b'])
    m.main(m.parse_args(argv + ['--reference', 'dark0_set0']))
    assert windows == ['ts0']
    root = dl.open_store(path)
    for j, entries in ((0, logs[2]), (1, logs[3])):
        g = root[f'photons/dark0b_set{j}']
        assert g.attrs['mode'] == 'settings from photons/dark0_set0'
        assert g.attrs['reference'] == 'photons/dark0_set0'
        assert g.attrs['reuse'] == 'settings' and g.attrs['tone_set'] == j
        check_photons(g, entries, dt)
        check_override(g)
    # Its filters for run 1's tone set 0: heights in its photon units,
    # replacing that timestream's result. Without the window, a saved
    # result is only replaced with --overwrite: otherwise it is skipped
    monkeypatch.setattr(m, 'TIMESTREAMS', [2])
    before = dl.open_store(path)['photons/dark0b_set0'].attrs['created']
    assert m.main(m.parse_args(argv + ['--reference', 'dark0_set0',
                                       '--reuse', 'filter'])) == {}
    out = capsys.readouterr().out
    assert 'photons/dark0b_set0 exists: skipped' in out and \
        'Skipped photons/dark0b_set0' in out
    assert dl.open_store(path)['photons/dark0b_set0'].attrs['created'] == \
        before
    m.main(m.parse_args(argv + ['--reference', 'dark0_set0', '--reuse',
                                'filter', '--overwrite']))
    assert 'Replacing photons/dark0b_set0' in capsys.readouterr().out
    root = dl.open_store(path)
    assert pf.list_results(root) == ['dark0_set0', 'dark0b_set0',
                                     'dark0b_set1']
    g2 = root['photons/dark0b_set0']
    assert g2.attrs['mode'] == 'filter from photons/dark0_set0'
    check_photons(g2, logs[2], dt)
    np.testing.assert_allclose(g2['amp_ph'][:], g0['amp_ph_own'][:])
    # Another timestream from scratch, under a name: the window opens
    # again; a name needs a single timestream
    monkeypatch.setattr(m, 'TIMESTREAMS', [1])
    m.main(m.parse_args(argv + ['--name', 'first_set1']))
    assert windows == ['ts0', 'ts0']
    check_photons(dl.open_store(path)['photons/first_set1'], logs[1], dt)
    monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0b'])
    with pytest.raises(ValueError, match = 'single'):
        m.main(m.parse_args(argv + ['--name', 'x']))
    # With the window, a run on a timestream whose result is saved asks:
    # Cancel skips it, Overwrite detects again from scratch
    asked = []

    def answer(name, grp):
        """
        Stand in for the popup: record the result's description and
        answer with the next choice.

        Parameters:
        name (str): the result's name.
        grp (zarr.Group): the result.

        Returns:
        choice (str or None): the next answer.
        """
        asked.append(pf.describe_result(grp))
        return answers.pop(0)

    monkeypatch.setattr(m, 'ask_existing', answer)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    answers = [None]
    before = dl.open_store(path)['photons/dark0_set0'].attrs['created']
    assert m.main(m.parse_args(argv)) == {}
    assert 'Skipped photons/dark0_set0' in capsys.readouterr().out
    assert dl.open_store(path)['photons/dark0_set0'].attrs['created'] == \
        before and windows == ['ts0', 'ts0']
    assert asked == ['ts0 tone set 0, 6 tones\ncreated ' + before
                     + ', full\n1 tones with their own thresholds, 0 with '
                     'their own decay fit, 0 marked bad']
    answers = ['overwrite']
    m.main(m.parse_args(argv))
    assert 'Replacing photons/dark0_set0' in capsys.readouterr().out
    assert windows == ['ts0', 'ts0', 'ts0'] and answers == []
    g0 = dl.open_store(path)['photons/dark0_set0']
    assert g0.attrs['mode'] == 'full' and g0.attrs['created'] != before
    check_photons(g0, logs[0], dt)
    check_override(g0)
    # A saved result loads back as detection results
    loaded, th = pf.load_results(g0)
    r0 = int(g0['rows'][0])
    res = loaded[r0]
    for key in ('cand_idx', 'events', 'template', 'history', 'neg_h',
                'init_h', 'coinc_idx', 'coinc_raw', 'log', 'filter', 'psd'):
        assert key in res, key
    assert len(res['log']) > 1 and len(res['history']) > 0
    pulses = g0['pulses']
    mine = pulses['row'][:] == r0
    np.testing.assert_array_equal(res['coinc_idx'], pulses['idx'][:][mine])
    np.testing.assert_array_equal(
        res['cand_idx'], pulses['idx'][:][mine & pulses['candidate'][:]])
    assert res['hmin'] == 1.2 and th.overrides == {int(g0['res_idxs'][0]):
                                                    dict(hmin = 1.2)}
    assert np.sum(np.isin(res['events'], res['cand_idx'])) == \
        len(res['events']) and th.bad == {}
    # Continue from it: the window opens on the saved results (no
    # detection first); one tone is marked bad with a reason, one gets
    # its own thresholds and is rerun, one gets its own decay fit
    reruns = []

    def continue_window(ts, results, thresholds, rerun, data = None,
                        settings = None, title = '', **kwargs):
        """
        Stand in for the window when continuing: check the results came
        from the store, mark the second tone bad, rerun the third, give
        the fourth its own decay fit.

        Parameters:
        ts (dict): the timestream.
        results (dict): the loaded results.
        thresholds (Thresholds): the loaded thresholds.
        rerun (callable): reruns tones.
        data (callable): a tone's detection timestream.
        settings (DetectionSettings): the settings.
        title (str): window title.

        Returns:
        thresholds, results: as the window.
        """
        windows.append('continue')
        assert thresholds.hmax == 3.5
        assert set(results) == set(int(r) for r in ts['rows'])
        assert all(isinstance(r, dict) for r in results.values())
        thresholds.set_bad(int(ts['res_idxs'][1]), 'double resonance')
        row = int(ts['rows'][2])
        thresholds.overrides[int(ts['res_idxs'][2])] = dict(hmax = 3.)
        new = rerun([row], thresholds)
        reruns.append(row)
        assert new[row]['hmax'] == 3.
        thresholds.set_fit(int(ts['res_idxs'][3]), 3, 0.5e-3)
        thresholds.limits['tau'] = [100e-6, 200e-6]
        return thresholds, {**results, **new}

    monkeypatch.setattr(pt, 'run_photon_thresholds', continue_window)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    answers = ['continue']
    m.main(m.parse_args(argv))
    assert windows[-1] == 'continue' and len(reruns) == 1 and answers == []
    assert 'Replacing photons/dark0_set0' in capsys.readouterr().out
    root = dl.open_store(path)
    t4 = root['photons/dark0_set0']
    assert t4.attrs['continued_from'] == 'photons/dark0_set0'
    assert t4.attrs['mode'] == 'continued from photons/dark0_set0'
    th = t4.attrs['thresholds']
    bad_res = int(t4['res_idxs'][1])
    assert th['bad'] == {str(bad_res): 'double resonance'}
    assert th['hmax'] == 3.5 and th['limits']['tau'] == [100e-6, 200e-6]
    assert th['overrides'][str(int(t4['res_idxs'][0]))] == dict(hmin = 1.2)
    assert th['overrides'][str(int(t4['res_idxs'][2]))] == dict(hmax = 3.)
    assert th['overrides'][str(int(t4['res_idxs'][3]))] == \
        dict(tau_skip = 3, tau_fit_s = 0.5e-3)
    np.testing.assert_array_equal(t4['bad'][:],
                                  t4['res_idxs'][:] == bad_res)
    assert list(t4['bad_reason'][:]) == ['', 'double resonance', '', '', '',
                                         '']
    np.testing.assert_array_equal(t4['tau_skip'][:], [2, 2, 2, 3, 2, 2])
    np.testing.assert_allclose(t4['tau_fit_s'][:],
                               [1e-3, 1e-3, 1e-3, 0.5e-3, 1e-3, 1e-3])
    # The bad tone is detected but left out of the HEEs and has no
    # mean pulse; the others are as before; the per-tone fit is used
    assert all(s == '' for s in t4['status'][:])
    assert t4['n_used'][:][1] == 0 and np.isnan(t4['tau'][:][1])
    assert bad_res not in t4['hee/hits/res_idx'][:]
    assert t4['n_used'][:][0] == g0['n_used'][:][0]
    np.testing.assert_allclose(t4['hmax'][:][2], 3.)
    assert np.isfinite(t4['tau'][:][3])
    assert 'dark0_set0: ts0 set 0, ts_cal0, continued from ' \
        'photons/dark0_set0' in pf.describe_results(root)
    assert '1 bad' in pf.describe_results(root)
    # With --no-gui the saved results go straight to the summary; the
    # --continue flag does not ask
    m.main(m.parse_args(argv + ['--continue', '--no-gui']))
    assert windows[-1] == 'continue' and len(asked) == 3
    assert ('2 tones with their own thresholds, 1 with their own decay '
            'fit, 1 marked bad') in pf.describe_result(
        dl.open_store(path)['photons/dark0_set0'])
    # The bad mark carries into a reference: no filter from that tone
    t4 = dl.open_store(path)['photons/dark0_set0']
    thr, filters = pf.load_reference(t4)
    assert bad_res not in filters and len(filters) == 5
    assert thr.bad == {bad_res: 'double resonance'}
    # Refit the decay times with other values, in place, for every tone
    monkeypatch.setattr(m, 'TAU_FIT_S', 0.3e-3)
    monkeypatch.setattr(m, 'TAU_SKIP', 3)
    tau_before = t4['tau'][:]
    taus = m.main(m.parse_args(argv + ['--refit-tau']))
    t4 = dl.open_store(path)['photons/dark0_set0']
    assert list(taus) == ['dark0_set0']
    np.testing.assert_array_equal(taus['dark0_set0'], t4['tau'][:])
    ok = np.isfinite(t4['tau'][:])
    assert ok.sum() == 5 and not ok[1]
    np.testing.assert_allclose(t4['tau'][:][ok], 150e-6, rtol = 0.4)
    assert not np.allclose(t4['tau'][:][ok], tau_before[ok])
    assert t4.attrs['settings']['tau_fit_s'] == 0.3e-3 and \
        t4.attrs['settings']['tau_skip'] == 3 and 'tau_refit' in t4.attrs
    np.testing.assert_array_equal(t4['tau_skip'][:], 3)
    assert t4.attrs['thresholds']['tau_fit_s'] == 0.3e-3
    assert 'tau_fit_s' not in t4.attrs['thresholds']['overrides'].get(
        str(int(t4['res_idxs'][3])), {})
    assert t4.attrs['settings']['hmin'] == m.HMIN
    monkeypatch.setattr(m, 'TAU_FIT_S', 1e-3)
    monkeypatch.setattr(m, 'TAU_SKIP', 2)
    # A result without detection detail (an earlier version) can still be
    # continued: the detection runs again with its thresholds first
    root = dl.open_store(path, mode = 'a')
    copy_group(root['photons/dark0_set0'], root['photons'].create_group(
        'old'), exclude = ('detail',))
    m.main(m.parse_args(argv + ['--continue', '--name', 'old']))
    out = capsys.readouterr().out
    assert 'holds no detection detail' in out and windows[-1] == 'continue'
    g_old = dl.open_store(path)['photons/old']
    assert g_old.attrs['mode'] == 'continued from photons/old'
    assert 'detail' in g_old
    # Results of the earlier layout (ts_photons{p}/timestream{j}) are
    # moved into photons/<run name>; a taken name is left where it is
    root = dl.open_store(path, mode = 'a')
    for p in (0, 1):
        old = dl.create_group(root, f'ts_photons{p}', 'analysis', attrs = dict(
            source = 'ts_cal0', reference = None, reuse = None,
            timestreams = [0]))
        copy_group(root['photons/first_set1'], old.create_group('timestream0'))
    assert 'ts_photons0, ts_photons1' in pf.describe_results(root)
    moved = pf.migrate_results(root, which = [0])
    assert moved == {'dark0': 'ts_photons0/timestream0'}
    root = dl.open_store(path, mode = 'a')
    assert 'ts_photons0' not in root and 'ts_photons1' in root
    g = root['photons/dark0']
    assert g.attrs['migrated_from'] == 'ts_photons0/timestream0'
    assert g.attrs['calibration'] == 'ts_cal0' and g.attrs['run'] == 'ts0'
    np.testing.assert_array_equal(g['res_idxs'][:],
                                  root['photons/first_set1/res_idxs'][:])
    assert pf.migrate_results(root) == {}
    assert 'ts_photons1' in root and 'not moved' in capsys.readouterr().out
    # Tones whose x calibration failed are left out of the listing:
    # NaN poly_x, or a row that was never saved
    arr = root['ts_cal0/set0_cal0/poly_x/data']
    saved = arr[0]
    arr[0] = np.nan
    mask = root['ts_cal0/set0_cal0/poly_x/row_exists']
    mask_saved = mask[1]
    mask[1] = False
    try:
        monkeypatch.setattr(m, 'TIMESTREAMS', [])
        m.main(m.parse_args(argv))
        out = capsys.readouterr().out
        assert "0: ts0, 'dark0', set 0, 1.5 s, 4 tones (2 uncalibrated " \
            "left out)" in out
        assert "1: ts0, 'dark0', set 1, 1.5 s, 6 tones, set1" in out
    finally:
        arr[0] = saved
        mask[1] = mask_saved
    # No timestream is saved: every saved array is far shorter than one
    root = dl.open_store(path)
    nsamples = g0.attrs['stop']
    sizes = [max(v.shape)
             for grp in ['ts_cal0', 'photons']
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


def test_06b_hee(photon_store, monkeypatch, capsys):
    """
    Fit HEE tails: from triggers where no photon result exists
    (the window changes the thresholds and marks a tone bad), from the
    photon result's HEEs where one does; a saved result is
    skipped, replaced or continued; one result per timestream.
    """
    import citkid.pipeline.interactive.hee_tails as ct
    from citkid.pipeline import hee_finder as cf
    from citkid.pipeline import photon_finder as pf
    path, log = photon_store
    m = load_script('06b_hee.py')
    argv = ['--store', str(path)]
    if 'dark0_set0' not in pf.list_results(dl.open_store(path)):
        # The photon result of run 0's tone set 0 (made by the 06a test
        # when the whole file runs)
        m6 = load_script('06a_find_photons.py')
        monkeypatch.setattr(m6, 'TIMESTREAMS', [0])
        m6.main(m6.parse_args(argv + ['--no-gui']))
    monkeypatch.setattr(m, 'LEVEL', 3.)
    monkeypatch.setattr(m, 'FIT_S', 1e-3)
    monkeypatch.setattr(m, 'TIMESTREAMS', [])
    assert m.main(m.parse_args(argv)) is None
    out = capsys.readouterr().out
    assert "1: ts0, 'dark0', set 1" in out and 'Previous results' in out
    windows = []

    def fake_window(ts, results, thresholds, refit, data = None,
                    title = '', ui_scale = 1., **kwargs):
        """
        Stand in for the window: narrow the height window for all tones
        and refit them, mark one tone bad.

        Parameters:
        ts (dict): the timestream.
        results (dict): its results.
        thresholds (TailThresholds): its thresholds.
        refit (callable): refits tones.
        title (str): window title.
        ui_scale (float): font scale.

        Returns:
        thresholds, results: as the window.
        """
        windows.append(ts['run'])
        assert all(isinstance(r, dict) for r in results.values())
        assert thresholds.level == 3. and thresholds.fit_s == 1e-3
        # The window gets the timestreams for its plot
        d = data(int(ts['rows'][0]))
        assert isinstance(d, np.ndarray) and len(d) == ts['stop'] - ts['start']
        thresholds.set_all(hmax = 50.)
        results = {**results, **refit(list(ts['rows']), thresholds)}
        assert all(r['hmax'] == 50. for r in results.values())
        thresholds.set_bad(int(ts['res_idxs'][1]), 'weird pulse shape')
        return thresholds, results

    monkeypatch.setattr(ct, 'run_hee_tails', fake_window)
    # Run 0's tone set 1 has no photon result: events from triggers
    monkeypatch.setattr(m, 'TIMESTREAMS', [1])
    res = m.main(m.parse_args(argv))
    assert windows == ['ts0'] and list(res) == ['dark0_set1']
    assert 'events from triggers' in capsys.readouterr().out
    root = dl.open_store(path)
    assert cf.list_results(root) == ['dark0_set1']
    g = root['hee/dark0_set1']
    assert g.attrs['events'] == 'triggers' and g.attrs['kind'] == 'analysis'
    assert g.attrs['calibration'] == 'ts_cal0' and \
        g.attrs['continued_from'] \
        is None
    # The simulated rays (every tone of each of the two writes), within
    # the dead time
    n_true = sum(len(entry['hee']) for entry in log[1:3])
    assert 0.8 * n_true <= g.attrs['n_events'] <= n_true
    assert g['events/t0'].shape == (g.attrs['n_events'],)
    assert set(g['events/write_idx'][:].tolist()) <= {0, 1}
    assert np.all(g['events/mult'][:] >= 3)
    assert all(s == '' for s in g['status'][:])
    np.testing.assert_allclose(g['hmax'][:], 50.)
    np.testing.assert_allclose(g['level'][:], 3.)
    bad = g['bad'][:]
    assert bad.tolist() == [False, True, False, False, False, False]
    assert g['bad_reason'][:][1] == 'weird pulse shape'
    assert g.attrs['thresholds']['bad'] == {str(int(g['res_idxs'][1])):
                                            'weird pulse shape'}
    # Decay times of the aligned means: the simulation's 150 us, from
    # pulses of 10 kHz/GHz times a log-normal factor, mostly in the window
    tau = g['tau'][:]
    assert np.isfinite(tau).all()
    # A handful of simulated pulses per tone: a loose tolerance
    np.testing.assert_allclose(tau, 150e-6, rtol = 0.3)
    assert g.attrs['anchor'] == g.attrs['pre']
    # Each tone sees the events of its own write
    n_write = {int((g['events/write_idx'][:] == w).sum()) for w in (0, 1)}
    assert set(g['n_pulses'][:].tolist()) <= n_write
    assert np.all(g['n_used'][:] >= 0.5 * g['n_pulses'][:])
    p = g['pulses']
    assert p['row'].shape[0] == g['n_pulses'][:].sum()
    assert p['used'][:].sum() == g['n_used'][:].sum()
    assert np.all(p['start'][:][~p['used'][:]] == -1)
    assert g['avg_pulse'].shape[1] == g.attrs['pre'] + g.attrs['post']
    assert 'dark0_set1: ts0 set 1' in cf.describe_results(root) and \
        '1 bad' in cf.describe_results(root)
    # Run 0's tone set 0 has a photon result: its HEEs are the
    # events (no window)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv + ['--no-gui']))
    assert windows == ['ts0']
    root = dl.open_store(path)
    g0 = root['hee/dark0_set0']
    assert g0.attrs['events'] == 'photons/dark0_set0'
    ph = root['photons/dark0_set0/hee/events']
    np.testing.assert_array_equal(g0['events/t0'][:], ph['t0'][:])
    assert g0.attrs['n_events'] == len(ph['t0'][:])
    np.testing.assert_allclose(g0['tau'][:], 150e-6, rtol = 0.3)
    # Asking for photons where there is no result fails; triggers can be
    # forced where there is one; a photon result saved without its HEEs
    # counts as none
    monkeypatch.setattr(m, 'TIMESTREAMS', [1])
    with pytest.raises(KeyError, match = 'no photon result'):
        m.main(m.parse_args(argv + ['--events', 'photons', '--overwrite',
                                    '--no-gui']))
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    dl.open_store(path, mode = 'a')['photons'].create_group('nohee')
    with pytest.raises(KeyError, match = 'no photon result'):
        m.main(m.parse_args(argv + ['--events', 'photons', '--photons',
                                    'nohee', '--overwrite', '--no-gui']))
    m.main(m.parse_args(argv + ['--photons', 'nohee', '--overwrite',
                                '--no-gui']))
    out = capsys.readouterr().out
    assert 'holds no HEEs' in out and 'events from triggers' in out
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv + ['--events', 'triggers', '--overwrite',
                                '--no-gui']))
    assert 'Replacing hee/dark0_set0' in capsys.readouterr().out
    g0 = dl.open_store(path)['hee/dark0_set0']
    assert g0.attrs['events'] == 'triggers'
    # A saved result: skipped without the window, unless --overwrite or
    # --continue; with the window, the popup decides
    before = g0.attrs['created']
    assert m.main(m.parse_args(argv + ['--no-gui'])) == {}
    assert 'hee/dark0_set0 exists: skipped' in capsys.readouterr().out
    assert dl.open_store(path)['hee/dark0_set0'].attrs['created'] == \
        before
    answers = ['overwrite']
    monkeypatch.setattr(m, 'ask_existing', lambda name, grp: answers.pop(0))
    m.main(m.parse_args(argv))
    assert windows == ['ts0', 'ts0'] and answers == []
    assert dl.open_store(path)['hee/dark0_set0'].attrs['created'] != \
        before
    answers = [None]
    assert m.main(m.parse_args(argv)) == {}
    assert 'Skipped hee/dark0_set0' in capsys.readouterr().out
    # Continue from a saved result: its thresholds and bad marks open in
    # the window; the result is replaced
    seen = []

    def continue_window(ts, results, thresholds, refit, data = None,
                        title = '', ui_scale = 1., **kwargs):
        """
        Stand in for the window when continuing: record the thresholds.

        Parameters:
        ts (dict): the timestream.
        results (dict): the results.
        thresholds (TailThresholds): the loaded thresholds.
        refit (callable): refits tones.
        title (str): window title.
        ui_scale (float): font scale.

        Returns:
        thresholds, results: unchanged.
        """
        seen.append((thresholds.hmax, dict(thresholds.bad)))
        return thresholds, results

    monkeypatch.setattr(ct, 'run_hee_tails', continue_window)
    monkeypatch.setattr(m, 'TIMESTREAMS', [1])
    m.main(m.parse_args(argv + ['--continue']))
    bad_res = int(dl.open_store(path)['hee/dark0_set1']['res_idxs'][1])
    assert seen == [(50., {bad_res: 'weird pulse shape'})]
    g1 = dl.open_store(path)['hee/dark0_set1']
    assert g1.attrs['continued_from'] == 'hee/dark0_set1'
    assert g1['bad'][:][1]
    with pytest.raises(KeyError, match = 'no saved result'):
        monkeypatch.setattr(m, 'TIMESTREAMS', [3])
        m.main(m.parse_args(argv + ['--continue']))
    # A name needs a single timestream
    monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0b'])
    with pytest.raises(ValueError, match = 'single'):
        m.main(m.parse_args(argv + ['--name', 'x']))
    assert cf.list_results(dl.open_store(path)) == ['dark0_set0',
                                                    'dark0_set1']


def test_07_psds(photon_store, monkeypatch, capsys):
    """
    List the timestreams of every calibration; analyze a set of them into
    one group (reduced frequencies shared, common modes removed per tone
    write); then without common-mode removal, with peak deglitching, and
    with the pulses found by a photon result's filters. No timestream is
    saved.
    """
    import citkid.pipeline.interactive.psd_overview as po
    from citkid.pipeline import psds
    path, _ = photon_store
    m = load_script('07_psds.py')
    windows = []
    marking = []

    def fake_overview(ts, result, f_reduced, bad = None, limits = None,
                      root = None, stage = '', title = '', ui_scale = 1.):
        """
        Stand in for the overview window: mark the first tone bad, with
        a reason, in the result and the store's tone marks.

        Parameters:
        ts (dict): the timestream.
        result (dict): its spectra.
        f_reduced (np.array): the group's reduced frequencies.
        bad, limits (dict or None): saved marks and limits.
        root (zarr.Group): the store root.
        stage (str): the result's path.
        title (str): window title.
        ui_scale (float): font scale.

        Returns:
        bad, limits (dict): as the window.
        """
        windows.append(stage)
        assert len(result['sxx_reduced']) == len(ts['rows'])
        # The window sees the raw spectra, before the common modes
        assert 'sxx_reduced_cn' not in result
        if not marking:
            return {}, {}
        r = int(ts['res_idxs'][0])
        return {r: 'high noise'}, {'10 Hz': [1e-17, 1e-15]}

    monkeypatch.setattr(po, 'run_psd_overview', fake_overview)
    argv = ['--store', str(path)]
    monkeypatch.setattr(m, 'TIMESTREAMS', [])
    assert m.main(m.parse_args(argv)) is None
    out = capsys.readouterr().out
    assert "0: ts0, 'dark0', set 0" in out and 'ts_cal0/set1' in out
    assert 'Previous results' in out
    # Run 0's tone set 0 and both tone sets of run 1
    monkeypatch.setattr(m, 'TIMESTREAMS', [0, 'dark0b'])
    res = m.main(m.parse_args(argv + ['--name', 'dark']))
    assert sorted(res) == ['timestream0', 'timestream1', 'timestream2']
    root = dl.open_store(path)
    g = root['ts_psd0']
    assert g.attrs['kind'] == 'analysis' and g.attrs['reference'] is None
    assert g.attrs['name'] == 'dark'
    assert g.attrs['timestreams'] == [0, 'dark0b']
    assert g.attrs['settings']['cn_remove'] is True
    # The window ran per timestream; its marks and limits are saved
    assert windows == ['ts_psd0/timestream0', 'ts_psd0/timestream1',
                       'ts_psd0/timestream2']
    # A run whose window marks the first tone bad: the mark, reason and
    # limits are saved, and the tone stayed out of the common modes
    marking.append(True)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv + ['--name', 'marks']))
    marking.clear()
    monkeypatch.setattr(m, 'TIMESTREAMS', [0, 'dark0b'])
    root = dl.open_store(path)
    t0 = root[f"ts_psd{dl.list_indices(root, 'ts_psd')[-1]}/timestream0"]
    assert t0['bad'][:].tolist() == [True] + [False] * 5
    assert list(t0['bad_reason'][:]) == ['high noise'] + [''] * 5
    assert t0.attrs['limits'] == {'10 Hz': [1e-17, 1e-15]}
    assert t0['status'][:][0] == 'marked bad: high noise'
    assert np.isnan(t0['sxx_reduced_cn'][:][0]).all()
    assert np.isnan(t0['cn_weights'][:][0]).all()
    assert np.isfinite(t0['sxx_reduced_cn'][:][1:]).all()
    monkeypatch.setattr(po, 'run_psd_overview',
                        lambda *a, **k: ({}, {}))
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
        # The spectra end at 0.8 of the Nyquist frequency
        fb = t['f_binned'][:]
        kept = fb <= 0.8 * 0.5 / dt
        assert kept.sum() > 50 and not kept.all()
        for key in ('sxx_binned', 'sxx_binned_cn'):
            assert t[key].shape == (n, nb)
            assert np.isfinite(t[key][:][:, kept]).all()
            assert np.isnan(t[key][:][:, ~kept]).all()
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
    assert psds.load_group(root, 'dark')['name'] == 'dark'
    with pytest.raises(KeyError, match = 'named'):
        psds.load_group(root, 'light')
    assert "ts_psd0: 'dark', ts0 set 0" in psds.describe_results(root)
    assert 'common modes 3 modes' in psds.describe_results(root)
    # Without common-mode removal: no cn arrays
    monkeypatch.setattr(m, 'TIMESTREAMS', [dict(name = 'dark0b',
                                                tone_sets = [0], cal = 0)])
    m.main(m.parse_args(argv + ['--no-cn']))
    t = dl.open_store(path)['ts_psd2/timestream0']
    assert dl.open_store(path)['ts_psd2'].attrs['name'] is None
    assert t.attrs['cn_removed'] is False and 'sxx_binned_cn' not in t
    assert 'common_modes' not in t and t.attrs['run'] == 'ts1'
    assert dl.open_store(path)['ts_psd2'].attrs['timestreams'] == \
        [dict(name = 'dark0b', tone_sets = [0], cal = 0)]
    # Peak deglitching finds the simulated pulses (about 30 photons/s per
    # tone plus HEEs)
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    m.main(m.parse_args(argv + ['--deglitch', 'peaks']))
    t = dl.open_store(path)['ts_psd3/timestream0']
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
    # A photon result's filters: HEEs only (a few per second), or
    # every pulse
    with pytest.raises(ValueError, match = 'PHOTONS'):
        m.main(m.parse_args(argv + ['--deglitch', 'photons', '--new']))
    root = dl.open_store(path, mode = 'a')
    if 'photons' not in root or 'dark0_set0' not in root['photons']:
        m6 = load_script('06a_find_photons.py')
        monkeypatch.setattr(m6, 'TIMESTREAMS', [0])
        m6.main(m6.parse_args(argv + ['--no-gui']))
    m.main(m.parse_args(argv + ['--deglitch', 'photons', '--photons',
                                'dark0_set0', '--new']))
    root = dl.open_store(path)
    g = root['ts_psd4']
    assert g.attrs['reference'] == 'photons/dark0_set0'
    t = g['timestream0']
    n_hee = t['n_glitches'][:]
    # The tone marked bad in the photon result has no filter: -1
    has = n_hee >= 0
    assert has.sum() >= 5 and np.all(n_hee[has] < 30)
    assert n_hee[has].sum() > 0
    monkeypatch.setattr(m, 'GLITCH_PULSES', 'all')
    m.main(m.parse_args(argv + ['--deglitch', 'photons', '--photons',
                                'dark0_set0', '--new']))
    t = dl.open_store(path)['ts_psd5/timestream0']
    assert np.all(t['n_glitches'][:][has] > n_hee[has])
    assert np.all(t['n_glitches'][:][has] > 20)
    # No timestream is saved
    nsamples = g['timestream0'].attrs['stop']
    sizes = [max(v.shape)
             for q in dl.list_indices(root, 'ts_psd')
             for _, v in root[f'ts_psd{q}'].members(max_depth = None)
             if isinstance(v, zarr.Array) and v.shape]
    assert max(sizes) < nsamples / 10
    # A set that exists: without the window it is skipped unless --new,
    # --load or --overwrite; with the window the popup decides
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    assert m.main(m.parse_args(argv + ['--name', 'marks', '--no-gui'])) \
        is None
    assert 'ts_psd1 exists: skipped' in capsys.readouterr().out
    assert dl.list_indices(dl.open_store(path), 'ts_psd') == list(range(6))
    # --load: computed again in place with the saved review, no window
    calls = []
    monkeypatch.setattr(po, 'run_psd_overview',
                        lambda *a, **k: calls.append(k) or ({}, {}))
    created = dl.open_store(path)['ts_psd1'].attrs['created']
    monkeypatch.setattr(m, 'CN_N_COMP', 2)
    res = m.main(m.parse_args(argv + ['--name', 'marks', '--load']))
    assert calls == [] and sorted(res) == ['timestream0']
    assert 'Computing ts_psd1 again' in capsys.readouterr().out
    root = dl.open_store(path)
    assert dl.list_indices(root, 'ts_psd') == list(range(6))
    g1 = root['ts_psd1']
    assert g1.attrs['created'] == created
    # The saved settings, not the script's
    assert g1.attrs['settings']['cn_n_comp'] == 3
    t0 = g1['timestream0']
    assert t0['bad'][:].tolist() == [True] + [False] * 5
    assert list(t0['bad_reason'][:]) == ['high noise'] + [''] * 5
    assert t0.attrs['limits'] == {'10 Hz': [1e-17, 1e-15]}
    assert t0['status'][:][0] == 'marked bad: high noise'
    assert np.isnan(t0['sxx_reduced_cn'][:][0]).all()
    assert 'New' not in psds.describe_group(g1)
    assert '1 marked bad, limits at 1 frequencies' in \
        psds.describe_group(g1)
    # The popup: Cancel stops; New makes another set, with the window
    # opening on the saved limits
    answers = []
    monkeypatch.setattr(po, 'ask_existing_psd',
                        lambda path_, desc: answers.pop(0))
    answers.append(None)
    assert m.main(m.parse_args(argv + ['--name', 'marks'])) is None
    assert 'Skipped ts_psd1' in capsys.readouterr().out
    answers.append('new')
    m.main(m.parse_args(argv + ['--name', 'marks']))
    assert [c['limits'] for c in calls] == [{'10 Hz': [1e-17, 1e-15]}]
    root = dl.open_store(path)
    assert dl.list_indices(root, 'ts_psd') == list(range(7))
    assert root['ts_psd6'].attrs['name'] == 'marks'
    assert root['ts_psd6'].attrs['settings']['cn_n_comp'] == 2
    assert not root['ts_psd6/timestream0/bad'][:].any()
    # --overwrite replaces the latest set with that name in place
    answers.append('overwrite')
    m.main(m.parse_args(argv + ['--name', 'marks', '--notes', 'again']))
    root = dl.open_store(path)
    assert dl.list_indices(root, 'ts_psd') == list(range(7))
    assert root['ts_psd6'].attrs['notes'] == 'again'
    assert root['ts_psd1'].attrs['settings']['cn_n_comp'] == 3
    assert psds.matching_group(root, 'marks', [0]).path == 'ts_psd6'
    # Without a name, a set matches on its timestreams
    assert psds.matching_group(root, None, [0]).path == 'ts_psd5'
    assert psds.matching_group(root, None, [1]) is None
    assert psds.matching_group(root, 'none', [0]) is None
    # A run that fails leaves no empty set behind; a failed --load keeps
    # the saved spectra
    monkeypatch.setattr(m, 'MEMORY_GB', 1e-9)
    answers.append('new')
    with pytest.raises(MemoryError):
        m.main(m.parse_args(argv + ['--name', 'marks']))
    root = dl.open_store(path)
    assert dl.list_indices(root, 'ts_psd') == list(range(7))
    assert 'ts_psd7 removed' in capsys.readouterr().out
    with pytest.raises(MemoryError):
        m.main(m.parse_args(argv + ['--name', 'marks', '--load']))
    root = dl.open_store(path)
    assert 'f_reduced' in root['ts_psd6']
    assert 'sxx_reduced' in root['ts_psd6/timestream0']


def test_07_psds_uncalibrated(photon_store, monkeypatch, capsys):
    """
    A tone whose x calibration failed is left out of the timestream, and
    the per-tone arrays follow the tones kept, not the DataSet's rows.
    """
    import citkid.pipeline.interactive.psd_overview as po
    path, _ = photon_store
    m = load_script('07_psds.py')
    monkeypatch.setattr(po, 'run_psd_overview', lambda *a, **k: ({}, {}))
    root = dl.open_store(path, mode = 'a')
    arr = root['ts_cal0/set0_cal0/poly_x/data']
    saved = arr[0]
    arr[0] = np.nan
    try:
        monkeypatch.setattr(m, 'TIMESTREAMS', [0])
        res = m.main(m.parse_args(['--store', str(path), '--name',
                                   'uncal']))
    finally:
        arr[0] = saved
    assert '5 tones' in capsys.readouterr().out
    g = res['timestream0']
    assert g['sxx_reduced'].shape[0] == 5 and g['write_idx'].tolist() == \
        [0] * 5
    assert all(s == '' for s in g['status'])
    cols = np.isfinite(g['sxx_binned']).any(axis = 0)
    assert cols.sum() > 50 and np.isfinite(g['sxx_binned'][:, cols]).all()
    assert np.isfinite(g['sxx_reduced'][:, g['valid_reduced']]).all()


def test_export_s21_h5(tmp_path, monkeypatch, capsys):
    """
    Write a screened sweep as f, i, q and fres in an h5 file, choosing
    the sweep when several have fitted resonators; the dialogs are
    skipped with --no-gui.
    """
    h5py = pytest.importorskip('h5py')
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 10,
                 pair_frac = 0.)
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = small, verbose = False))
    m = load_script('export_s21_h5.py')
    with pytest.raises(KeyError, match = 'no VNA sweep with fitted'):
        m.main(m.parse_args(['--store', str(path), '--no-gui',
                             '--out', str(tmp_path)]))
    dl.write_array(root['vna0'], 'fres_auto', truth.fr.values[::-1])
    m2 = load_script('02_find_resonances.py')
    m2.main(m2.parse_args(['--store', str(path), '--no-manual']))
    out = m.main(m.parse_args(['--store', str(path), '--no-gui',
                               '--out', str(tmp_path)]))
    assert out == os.path.join(str(tmp_path), 's21_simdevice.h5')
    f, z = dl.load_vna_sweep(dl.open_store(path), 0)
    with h5py.File(out, 'r') as fh:
        np.testing.assert_array_equal(fh['f'][:], f)
        np.testing.assert_array_equal(fh['i'][:], z.real)
        np.testing.assert_array_equal(fh['q'][:], z.imag)
        fres = fh['fres'][:]
        assert fh.attrs['vna'] == 0 and fh.attrs['device'] == 'simdevice'
    assert len(fres) == len(truth) and np.all(np.diff(fres) > 0)
    # A second screened sweep: the sweep is asked for (in the terminal
    # without the window); a sweep without resonators is refused
    root = dl.open_store(path, mode = 'a')
    from citkid.zarr_util import copy_group
    copy_group(root['vna0'], dl.create_group(root, 'vna1', 'raw'))
    copy_group(root['resonators0'], dl.create_group(root, 'resonators1',
                                                    'decision'))
    copy_group(root['vna0'], dl.create_group(root, 'vna2', 'raw'))
    assert m.sweeps_with_resonators(root) == [0, 1]
    monkeypatch.setattr('builtins.input', lambda prompt = '': '1')
    out2 = tmp_path / 'chosen.h5'
    assert m.main(m.parse_args(['--store', str(path), '--no-gui', '--out',
                                str(out2)])) == str(out2)
    assert 'vna0:' in capsys.readouterr().out
    with h5py.File(out2, 'r') as fh:
        assert fh.attrs['vna'] == 1
    with pytest.raises(KeyError, match = 'no fitted resonators'):
        m.main(m.parse_args(['--store', str(path), '--no-gui', '--vna', '2',
                             '--out', str(out2)]))


def test_add_loop_psds(photon_store, monkeypatch, capsys):
    """
    The temporary script adds the loop-component spectra to a PSD group
    saved without them, in place, giving what 07 computes itself.
    """
    import citkid.pipeline.interactive.psd_overview as po
    path, _ = photon_store
    argv = ['--store', str(path)]
    m7 = load_script('07_psds.py')
    monkeypatch.setattr(po, 'run_psd_overview', lambda *a, **k: ({}, {}))
    monkeypatch.setattr(m7, 'TIMESTREAMS', [0])
    m7.main(m7.parse_args(argv + ['--name', 'loop']))
    root = dl.open_store(path, mode = 'a')
    p = dl.list_indices(root, 'ts_psd')[-1]
    t = root[f'ts_psd{p}/timestream0']
    saved = {k: t[k][:] for k in ('spar_binned', 'sper_binned',
                                  'sfactor_reduced')}
    for k in saved:
        del t[k]
    m = load_script('add_loop_psds.py')
    done = m.main(m.parse_args(argv + ['--psd', str(p)]))
    assert done == [f'ts_psd{p}/timestream0']
    t = dl.open_store(path)[f'ts_psd{p}/timestream0']
    for k, v in saved.items():
        np.testing.assert_allclose(t[k][:], v, rtol = 1e-6, equal_nan = True)
    # A group that has them is skipped unless --redo
    assert m.main(m.parse_args(argv + ['--psd', str(p)])) == []
    assert 'already' in capsys.readouterr().out
    assert m.main(m.parse_args(argv + ['--psd', str(p), '--redo'])) == \
        [f'ts_psd{p}/timestream0']


def test_device_parts():
    """
    A device name splits into the wafer (a 'p' between digits read as a
    period) and the die.
    """
    from citkid.pipeline.site import device_parts
    assert device_parts('L261022p1-die7') == ('L261022.1', '7')
    assert device_parts('Be260916-die2') == ('Be260916', '2')
    assert device_parts('devA') == ('devA', '')
    assert device_parts('p1p2-die12') == ('p1.2', '12')


def test_08_report_site(photon_store, monkeypatch, capsys, tmp_path):
    """
    The site: a device page with the figures and tables, one page per
    resonator, a manifest and the top index; a store that has not
    changed is left alone unless --force; notes.md is shown; set_notes.py
    changes a group's notes, which the next build picks up.
    """
    import json
    import citkid.pipeline.interactive.psd_overview as po
    from citkid.pipeline import hee_finder as cf
    from citkid.pipeline import photon_finder as pf
    path, _ = photon_store
    root = dl.open_store(path)
    store_argv = ['--store', str(path)]
    if 'dark0_set0' not in pf.list_results(root):
        m6 = load_script('06a_find_photons.py')
        monkeypatch.setattr(m6, 'TIMESTREAMS', [0])
        m6.main(m6.parse_args(store_argv + ['--no-gui']))
    if 'dark0_set0' not in cf.list_results(root):
        m6b = load_script('06b_hee.py')
        monkeypatch.setattr(m6b, 'TIMESTREAMS', [0])
        m6b.main(m6b.parse_args(store_argv + ['--no-gui']))
    if not dl.list_indices(root, 'ts_psd'):
        m7 = load_script('07_psds.py')
        monkeypatch.setattr(po, 'run_psd_overview',
                            lambda *a, **k: ({}, {}))
        monkeypatch.setattr(m7, 'TIMESTREAMS', [0])
        m7.main(m7.parse_args(store_argv + ['--name', 'site']))
    root = dl.open_store(path)
    device, cooldown = root.attrs['device'], root.attrs['cooldown']
    m = load_script('08_report_site.py')
    out = tmp_path / 'reports'
    argv = ['--store', str(path), '--out', str(out), '--no-gui',
            '--update']
    built = m.main(m.parse_args(argv))
    dev = out / cooldown / device
    assert built == [str(dev)]
    # The top page only with INDEX (off in the template)
    assert not (out / 'index.html').exists()
    assert (dev / 'index.html').exists()
    monkeypatch.setattr(m, 'INDEX', True)
    assert m.main(m.parse_args(argv)) == [str(dev)]
    assert (out / 'index.html').exists()
    manifest = json.loads((dev / 'manifest.json').read_text('utf-8'))
    assert manifest['device'] == device and manifest['res_pages'] is True
    assert len(manifest['runs']) >= 2 and manifest['tunings']
    assert manifest['photons'] and manifest['hee'] and manifest['psds']
    ov = manifest['overview']
    assert ov['n_tones'] and ov['fres_mhz'][0] <= ov['fres_mhz'][1]
    assert ov['photon_tau_us'] and ov['photon_tau_us'][2] >= 1
    assert ov['sxx_cn_10hz'] and ov['sxx_cn_10hz'][0] > 0
    top = (out / 'index.html').read_text('utf-8')
    assert 'photon_tau_us' in top and ' to ' in top
    # The template's helper: the calibration applied to one run
    from citkid.pipeline.stream_sources import calibrated_timestream
    name = manifest['runs'][0]['name']
    DS, ts, got = calibrated_timestream(root, name,
                                        outputs = ('xt', 'thetat', 'At'))
    assert ts['name'] == name and got['xt'].shape[0] == len(ts['rows'])
    assert got['At'].shape == got['xt'].shape == got['thetat'].shape
    assert len(DS.fres[list(ts['rows'])]) == len(ts['rows'])
    assert np.isfinite(got['xt']).any()
    n_tones = len(root['tune0/tones/res_idxs'][:])
    pages = sorted((dev / 'res').glob('res*.html'))
    assert len(pages) == n_tones
    assert len(list((dev / 'res').glob('res*.png'))) == n_tones
    figs = os.listdir(dev / 'figures')
    assert any(f.endswith('_tuning.png') for f in figs)
    assert 'Stream calibrations' in (dev / 'index.html').read_text('utf-8')
    assert any('lifetime_photons' in f for f in figs)
    assert any(f.startswith(f'{device}_psd') for f in figs)
    data = os.listdir(dev / 'data')
    assert f'{device}_resonators.csv' in data and \
        f'{device}_tune0_tuning.csv' in data
    page = (dev / 'index.html').read_text('utf-8')
    assert 'What was taken' in page and 'select class="versions"' in page
    assert 'res/res' in page and 'Tone marks' in page
    text = pages[0].read_text('utf-8')
    assert 'Tuning' in text and 'Device page' in text
    top = (out / 'index.html').read_text('utf-8')
    assert f'{cooldown}/{device}/index.html' in top
    assert '<th>die</th>' in top and '<th>suffix</th>' in top
    assert 'input class="colfilter"' in top and str(path) not in top
    assert '<details' in page and 'input class="rowfilter"' in page
    # The section markers are CSS escapes, not Python octal escapes
    assert 'content: "\\25B8 "' in page and chr(0x15) not in page
    assert str(path) not in page
    # --index-only rebuilds the top page alone
    (out / 'index.html').unlink()
    assert m.main(m.parse_args(['--out', str(out), '--index-only'])) == []
    assert (out / 'index.html').exists()
    # An unchanged store is up to date for build_device, but --update
    # (the user asked for the report again) rebuilds it anyway
    assert m.site.build_device(str(path), str(out),
                               progress = print) is None
    assert 'up to date' in capsys.readouterr().out
    assert m.main(m.parse_args(argv[:-1])) is None     # no --update
    assert 'exists' in capsys.readouterr().out
    assert m.main(m.parse_args(argv + ['--no-res-pages'])) == [str(dev)]
    assert 'up to date' not in capsys.readouterr().out
    # notes.md is shown
    (dev / 'notes.md').write_text('Hand-written notes here', 'utf-8')
    assert m.main(m.parse_args(argv + ['--no-res-pages'])) == [str(dev)]
    page = (dev / 'index.html').read_text('utf-8')
    assert 'Hand-written notes here' in page and 'res/res' not in page
    assert (dev / 'notes.md').exists()
    # Notes on a group change the store, so the site rebuilds
    sn = load_script('set_notes.py')
    notes = sn.main(sn.parse_args(['--store', str(path), '--path',
                                   'tune0', 'retuned after warm-up']))
    assert ('tune0', 'retuned after warm-up') in notes
    assert 'tune0: retuned' in capsys.readouterr().out
    assert sn.main(sn.parse_args(['--store', str(path), '--path',
                                  'tune0', ''])) == [
        n for n in notes if n[0] != 'tune0']
    sn.main(sn.parse_args(['--store', str(path), '--path', 'tune0',
                           'retuned']))
    assert m.main(m.parse_args(argv + ['--no-res-pages'])) == [str(dev)]
    assert 'retuned' in (dev / 'index.html').read_text('utf-8')
    # With no store, a dialog chooses one; cancelled, nothing is built
    import citkid.daq_browser as db
    monkeypatch.setattr(db, 'choose_store', lambda start = None: None)
    assert m.main(m.parse_args(['--out', str(out), '--no-gui'])) is None
    monkeypatch.setattr(db, 'choose_store', lambda start = None: str(path))
    assert m.main(m.parse_args(['--out', str(out), '--force', '--update',
                                '--no-res-pages', '--no-gui'])) == [str(dev)]
    # A report that exists: without the window, --update or --overwrite
    # is needed; with the window, a popup decides. Overwrite drops the
    # attachments, update keeps them
    (dev / 'attachments').mkdir(exist_ok = True)
    (dev / 'attachments' / 'keep.txt').write_text('x')
    assert m.main(m.parse_args(['--store', str(path), '--out', str(out),
                                '--no-gui'])) is None
    assert 'exists' in capsys.readouterr().out
    # Update rebuilds the pages even though the store has not changed
    capsys.readouterr()
    assert m.main(m.parse_args(['--store', str(path), '--out', str(out),
                                '--no-gui', '--update',
                                '--no-res-pages'])) == [str(dev)]
    assert 'up to date' not in capsys.readouterr().out
    assert (dev / 'attachments' / 'keep.txt').exists()
    answers = []
    monkeypatch.setattr(db, 'ask_existing_report',
                        lambda out_dir: answers.pop(0))
    monkeypatch.setattr(db, 'run_store_browser',
                        lambda path_, **k: path_)
    answers.append(None)
    assert m.main(m.parse_args(['--store', str(path), '--out', str(out)])) \
        is None
    assert (dev / 'attachments' / 'keep.txt').exists()
    answers.append('overwrite')
    assert m.main(m.parse_args(['--store', str(path), '--out', str(out),
                                '--no-res-pages'])) == [str(dev)]
    assert not (dev / 'attachments').exists()
    assert (dev / 'index.html').exists()
    # With the window, the store browser opens on its Notes tab first,
    # with the site folder for attachments; the store shown last is
    # built. An attached file is linked from the page and makes the
    # device out of date
    from citkid.pipeline import site
    opened = []

    def fake_browser(path_, title = '', ui_scale = 1., site_dir = None,
                     tab = None):
        """
        Stand in for the browser: attach a file to the tuning.

        Parameters:
        path_ (str): the store.
        title (str): window title.
        ui_scale (float): font scale.
        site_dir (str): the site folder.
        tab (str): the tab opened.

        Returns:
        path_ (str): the store shown.
        """
        opened.append((path_, site_dir, tab))
        f = tmp_path / 'wiring.txt'
        f.write_text('wiring')
        site.add_attachment(site.device_dir(site_dir, dl.open_store(path_)),
                            'tune0', str(f))
        return path_

    monkeypatch.setattr(db, 'run_store_browser', fake_browser)
    assert m.main(m.parse_args(['--out', str(out), '--no-res-pages',
                                '--update'])) == [str(dev)]
    assert opened == [(str(path), str(out), 'Notes')]
    page = (dev / 'index.html').read_text('utf-8')
    assert 'attachments/tune0/wiring.txt' in page and 'Attached files' in page
    assert page.count('attachments/tune0/wiring.txt') >= 3
    assert (dev / 'attachments' / 'tune0' / 'wiring.txt').exists()
    # Rebuilt again on request, the attachment stays
    assert m.main(m.parse_args(argv + ['--no-res-pages'])) == [str(dev)]
    assert (dev / 'attachments' / 'tune0' / 'wiring.txt').exists()


def test_06_tones_before_window(photon_store, monkeypatch, capsys):
    """
    With TONES_BEFORE_WINDOW only the first tones are detected before
    the window; the window runs others on request, and those still not
    run when it closes take its thresholds. With WINDOW_FIRST_ONLY the
    window opens on the first timestream only and the others take its
    result as the reference.
    """
    import citkid.pipeline.interactive.photon_thresholds as pt
    path, _ = photon_store
    m = load_script('06a_find_photons.py')
    argv = ['--store', str(path), '--overwrite']
    windows = []

    def fake_window(ts, results, thresholds, rerun, **kwargs):
        """
        Stand in for the window: check only the first two tones ran,
        widen hmax and run a third tone with it.

        Parameters:
        ts (dict): the timestream.
        results (dict): its results.
        thresholds (Thresholds): its thresholds.
        rerun (callable): reruns tones.

        Returns:
        thresholds, results: as the window.
        """
        windows.append(ts['run'])
        assert sorted(results) == sorted(int(r) for r in ts['rows'][:2])
        assert len(ts['rows']) > 3
        thresholds.hmax = 3.4
        r = int(ts['rows'][2])
        results = {**results, **rerun([r], thresholds)}
        assert results[r]['hmax'] == 3.4
        return thresholds, results

    monkeypatch.setattr(pt, 'run_photon_thresholds', fake_window)
    monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0'])
    monkeypatch.setattr(m, 'TONES_BEFORE_WINDOW', 2)
    monkeypatch.setattr(m, 'WINDOW_FIRST_ONLY', True)
    out = m.main(m.parse_args(argv))
    assert windows == ['ts0'] and list(out) == ['dark0_set0', 'dark0_set1']
    text = capsys.readouterr().out
    assert 'tones not run in the window' in text
    assert 'window opened on photons/dark0_set0 only' in text
    root = dl.open_store(path)
    g0, g1 = root['photons/dark0_set0'], root['photons/dark0_set1']
    assert all(s == '' for s in g0['status'][:])
    # The two tones run before the window keep the script's hmax
    np.testing.assert_allclose(g0['hmax'][:2], 3.2)
    np.testing.assert_allclose(g0['hmax'][2:], 3.4)
    assert g0.attrs['reference'] is None and g0.attrs['mode'] == 'full'
    assert g1.attrs['reference'] == 'photons/dark0_set0'
    assert g1.attrs['reuse'] == 'settings'
    assert all(s == '' for s in g1['status'][:])
    np.testing.assert_allclose(g1['hmax'][:], 3.4)
    # Without the window, every tone is detected before anything else
    monkeypatch.setattr(m, 'TIMESTREAMS', [0])
    out = m.main(m.parse_args(argv + ['--no-gui']))
    assert windows == ['ts0'] and 'not run' not in capsys.readouterr().out
    assert all(isinstance(r, dict)
               for r in out['dark0_set0']['results'].values())


def test_06b_tones_before_window(photon_store, monkeypatch, capsys):
    """
    As ``test_06_tones_before_window`` for the HEE tails: the window
    fits a tone not run from scratch, the rest are fitted when it closes,
    and later timestreams take the first window's values.
    """
    import citkid.pipeline.interactive.hee_tails as ct
    path, _ = photon_store
    m = load_script('06b_hee.py')
    argv = ['--store', str(path), '--overwrite']
    windows = []

    def fake_window(ts, results, thresholds, refit, **kwargs):
        """
        Stand in for the window: check only the first two tones ran,
        narrow hmax and refit one tone that ran and one that did not.

        Parameters:
        ts (dict): the timestream.
        results (dict): its results.
        thresholds (TailThresholds): its thresholds.
        refit (callable): refits tones.

        Returns:
        thresholds, results: as the window.
        """
        windows.append(ts['run'])
        assert sorted(results) == sorted(int(r) for r in ts['rows'][:2])
        thresholds.set_all(hmax = 60.)
        r0, r2 = int(ts['rows'][0]), int(ts['rows'][2])
        new = refit([r0, r2], thresholds)
        assert new[r0]['hmax'] == 60. and new[r2]['hmax'] == 60.
        return thresholds, {**results, **new}

    monkeypatch.setattr(ct, 'run_hee_tails', fake_window)
    monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0'])
    monkeypatch.setattr(m, 'TONES_BEFORE_WINDOW', 2)
    monkeypatch.setattr(m, 'WINDOW_FIRST_ONLY', True)
    out = m.main(m.parse_args(argv))
    assert windows == ['ts0'] and list(out) == ['dark0_set0', 'dark0_set1']
    text = capsys.readouterr().out
    assert 'tones not run in the window' in text
    assert 'Thresholds of the window on hee/dark0_set0, no window' in text
    root = dl.open_store(path)
    g0, g1 = root['hee/dark0_set0'], root['hee/dark0_set1']
    assert g0.attrs['thresholds_from'] is None
    assert g1.attrs['thresholds_from'] == 'hee/dark0_set0'
    # The second tone ran before the window and was not refitted
    np.testing.assert_allclose(g0['hmax'][1], 1000.)
    np.testing.assert_allclose(np.delete(g0['hmax'][:], 1), 60.)
    np.testing.assert_allclose(g1['hmax'][:], 60.)
    for g in (g0, g1):
        assert all(s == '' for s in g['status'][:])


def test_05_window_saves_global_prerequisites(tmp_path, monkeypatch, capsys):
    """
    The window computes ``fr_spans`` (a global step with no panel) and it
    is saved, so a later run continuing the calibration (e.g. to add the
    IQ fits) can open the window on it; a calibration whose ``fr_spans``
    is missing while its rows are saved is repaired, keeping the rows.
    """
    from unittest.mock import patch
    from pyqtgraph.Qt import QtWidgets
    from citkid.crs.procedures import stream_tone_sets
    from citkid.pipeline.interactive.core import InteractiveAnalysisWindow
    from citkid.pipeline.interactive.ts_analysis import ts_panels
    from citkid.qt_compat import get_qapp
    from ..crs.test_streaming import make_crs, make_tuning
    crs = make_crs()

    async def take():
        """
        Take a stage-6 run.
        """
        await crs.attach()
        _, tune = make_tuning(tmp_path, crs)
        await stream_tone_sets(crs, tune, 'dark6', 1., 6, npoints_fine = 100,
                               npoints_gain = 20, verbose = False)

    asyncio.run(take())
    m = load_script('05_calibrate_streams.py')
    argv = ['--store', str(tmp_path / 's.zarr')]
    monkeypatch.setattr(m, 'CALIBRATE', [dict(name = 'dark6')])
    app = get_qapp('t')
    errors = []

    def run_window(AR, title = ''):
        """
        Open the real window on the first row, then close it (as the
        user would), recording panel errors.

        Parameters:
        AR (AnalysisRunner): the DataSet's runner.
        title (str): window title.
        """
        win = InteractiveAnalysisWindow(AR, ts_panels(AR), title = title)
        win.show()
        for _ in range(30):
            app.processEvents()
        errors.extend((p.step_names, str(p._last_error))
                      for p in win.panels if p._last_error is not None)
        with patch.object(QtWidgets.QMessageBox, 'question',
                          return_value = QtWidgets.QMessageBox.Yes):
            win.close()
        for _ in range(10):
            app.processEvents()

    monkeypatch.setattr(m, 'run_ts_analysis', run_window)
    m.main(m.parse_args(argv))
    assert errors == []
    root = dl.open_store(tmp_path / 's.zarr')
    grp = root['ts_cal0/set0']
    assert 'fr_spans' in grp and 'data' in grp['fr_spans']
    assert root['ts_cal0/set0/p_amp/row_exists'][:].all()
    # Continue the calibration with the window: nothing is rerun or
    # refused
    m.main(m.parse_args(argv))
    out = capsys.readouterr().out
    assert errors == [] and 'auto-prerequisite' not in out
    assert 'Continuing ts_cal0' in out
    # A calibration from before this fix: fr_spans missing, rows saved
    root = dl.open_store(tmp_path / 's.zarr', mode = 'a')
    del root['ts_cal0/set0/fr_spans']
    m.main(m.parse_args(argv))
    assert errors == [] and 'auto-prerequisite' not in \
        capsys.readouterr().out
    root = dl.open_store(tmp_path / 's.zarr')
    assert 'data' in root['ts_cal0/set0/fr_spans']
    assert root['ts_cal0/set0/p_amp/row_exists'][:].all()


def test_06_names_the_calibration_with_the_timestream(
    photon_store, monkeypatch
):
    """
    A run name not in the chosen calibration: the error says which
    calibrations have it (to set CALIBRATION), or that none does.
    """
    from citkid.pipeline import photon_finder as pf
    path, _ = photon_store
    root = dl.open_store(path, mode = 'a')
    # An empty calibration, the latest
    dl.create_group(root, 'ts_cal8', 'analysis', attrs = dict(
        selection = [], name = 'empty'))
    try:
        m = load_script('06a_find_photons.py')
        monkeypatch.setattr(m, 'TIMESTREAMS', ['dark0'])
        with pytest.raises(KeyError, match = r"no timestream 'dark0' in "
                           r"ts_cal8 \('empty'\); it is in ts_cal0 "
                           r"\('ts_cal0'\): set CALIBRATION"):
            m.main(m.parse_args(['--store', str(path)]))
        monkeypatch.setattr(m, 'TIMESTREAMS', ['nope'])
        with pytest.raises(KeyError, match = r"no timestream 'nope' in "
                           r"ts_cal8 .* \(see the list\)"):
            m.main(m.parse_args(['--store', str(path)]))
        assert pf.calibrations_with(root, 'dark0') == ["ts_cal0 ('ts_cal0')"]
        assert pf.calibrations_with(root, 'nope') == []
    finally:
        del root['ts_cal8']


def test_04c_power_sweep(tmp_path, monkeypatch, capsys):
    """
    Step 04c sets the power at each point (the hook), calibrates the tone
    set (or reuses an earlier point's calibration) and takes RUNS named
    after the point, and saves the point's name with the sweep and the
    runs; --from starts at a later point.
    """
    from citkid.crs.sim import simulate_vna_store, simulated_crs
    path = tmp_path / 'sim.zarr'
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 6,
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
    monkeypatch.setattr(m, 'POINTS', [dict(name = 'BBT40K'),
                                      dict(name = 'BBT80K', cal = 'BBT40K')])
    monkeypatch.setattr(m, 'RUNS', [dict(dec_stage = 6, duration_s = 0.2),
                                    dict(dec_stage = 6, duration_s = 0.1)])
    monkeypatch.setattr(m, 'NPOINTS_FINE', 60)
    monkeypatch.setattr(m, 'NPOINTS_GAIN', 10)
    monkeypatch.setattr(m, 'NSAMPS', 10)
    monkeypatch.setattr(m, 'FRES_UPDATE', None)
    set_calls = []
    monkeypatch.setattr(m, 'set_optical_power',
                        lambda name: set_calls.append(name))

    async def same_board(args, root, path):
        """
        Stand in for connect: the board made above.

        Parameters:
        args, root, path: ignored.

        Returns:
        crs (SimulatedCRS): the board.
        """
        return crs

    monkeypatch.setattr(m, 'connect', same_board)
    argv = ['--store', str(path), '--simulate']
    results = m.main(m.parse_args(argv))
    assert set_calls == ['BBT40K', 'BBT80K']
    assert list(results) == ['BBT40K', 'BBT80K']
    root = dl.open_store(path)
    # The second point reused the first's calibration: one sweep,
    # named after the point it was taken for
    assert results['BBT40K']['calibration'] == 'tune0/cal0'
    assert results['BBT80K']['calibration'] == 'tune0/cal0'
    assert root['tune0/cal0'].attrs['point'] == 'BBT40K'
    assert 'BBT40K' in root['tune0/cal0'].attrs['notes']
    assert [c['name'] for c in dl.calibrations(root['tune0'])] == ['cal0']
    runs = dl.stream_runs(root)
    assert [r['name'] for r in runs] == [
        'BBT40K_dec6', 'BBT40K_dec6_2', 'BBT80K_dec6', 'BBT80K_dec6_2']
    assert all(r['point'] == r['name'].split('_')[0] for r in runs)
    assert 'power_w' not in runs[0] and runs[0]['notes'] == 'BBT40K'
    # --from starts at a later point; a point reusing a sweep by path
    monkeypatch.setattr(m, 'POINTS', [dict(name = 'BBT40K'),
                                      dict(name = 'BBT80K',
                                           cal = 'tune0/cal0')])
    results = m.main(m.parse_args(argv + ['--from', 'BBT80K']))
    assert list(results) == ['BBT80K'] and len(set_calls) == 3
    assert results['BBT80K']['calibration'] == 'tune0/cal0'
    runs = dl.stream_runs(dl.open_store(path))
    assert [r['name'] for r in runs[4:]] == ['BBT80K_dec6_3',
                                            'BBT80K_dec6_4']
    with pytest.raises(KeyError, match = 'no point'):
        m.main(m.parse_args(argv + ['--from', 'nope']))
    # Reusing a point not taken before it is refused
    monkeypatch.setattr(m, 'POINTS', [dict(name = 'X', cal = 'Y')])
    with pytest.raises(KeyError, match = 'not an earlier point'):
        m.main(m.parse_args(argv))
