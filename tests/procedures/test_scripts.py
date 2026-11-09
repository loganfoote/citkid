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
    name (str): file name, e.g. '02_vna_sweep.py'.

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
async def test_00_crs_init(monkeypatch, argv, extended):
    m = load_script('00_crs_init.py')
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


def test_01_vna_quick_check_no_gui(monkeypatch, tmp_path):
    m = load_script('01_vna_quick_check.py')
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


def test_01_vna_quick_check_no_count(monkeypatch, tmp_path):
    m = load_script('01_vna_quick_check.py')
    fake_factory(monkeypatch, m, FakeVnaCRS())
    _, _, n = m.main(m.parse_args(['--no-count', '--no-gui']))
    assert n is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('nyquist, lo, hi', [('2', 2.5e9, 5e9),
                                             ('both', 100e6, 5e9)])
def test_01_vna_quick_check_nyquist(monkeypatch, nyquist, lo, hi):
    m = load_script('01_vna_quick_check.py')
    fake = FakeVnaCRS(modules = (1, 2))
    fake_factory(monkeypatch, m, fake)
    f, _, _ = m.main(m.parse_args(
        ['--modules', '1', '2', '--nyquist', nyquist, '--qr-median', '1e4',
         '--pts-per-lw', '1', '--no-gui', '--no-count']))
    # Up to 10 MHz next to 2.5 GHz may be left out to save a pass
    assert lo - 50 <= f[0] < lo + 10e6 and hi * (1 - 2e-4) < f[-1] < hi
    zones = {zn for sz in fake.sweep_zones for zn in sz.values()}
    assert zones == ({1, 2} if nyquist == 'both' else {2})
    # Back to zone 1 afterwards
    assert all(c['nyquist_zone'] == 1 for c in fake.module_cfg.values())


def test_01_vna_quick_check_gui(monkeypatch):
    """
    Check that the window gets the settings and a connect function that
    attaches to this session's modules.
    """
    m = load_script('01_vna_quick_check.py')
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
async def test_02_vna_sweep(monkeypatch, tmp_path):
    m = load_script('02_vna_sweep.py')
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
    grp = root['vna/sweep0']
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
