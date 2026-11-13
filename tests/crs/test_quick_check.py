"""
Tests for the interactive VNA quick check, run against a fake CRS.
"""
import asyncio
import time
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from citkid.crs.quick_check import (
    AsyncWorker, QuickCheckWindow, sweep_plan_text
)
from citkid.crs.sim import SimulatedCRS
from citkid.vna import simulate as sim
from .fake_crs import FakeVnaCRS


class SlowSimulatedCRS(SimulatedCRS):
    """
    Simulated board whose passes take a while, so a sweep can be cancelled
    in the middle from the window.
    """

    async def sweep(self, *args, **kwargs):
        """
        Sweep after a pause.

        Parameters:
        *args, **kwargs: ``SimulatedCRS.sweep`` arguments.

        Returns:
        f, z (np.array): the pass.
        """
        await asyncio.sleep(0.15)
        return await super().sweep(*args, **kwargs)


def make_simulated_crs(slow = False):
    """
    Make a simulated board with a small device on module 1.

    Parameters:
    slow (bool): If True, a ``SlowSimulatedCRS``. Default False.

    Returns:
    crs (SimulatedCRS): the board.
    """
    rng = np.random.default_rng(0)
    small = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 8,
                 pair_frac = 0.)
    cls = SlowSimulatedCRS if slow else SimulatedCRS
    return cls(sim.draw_device(rng, **small),
               sim.draw_gain(rng, f_lo = 1e8, f_hi = 5e9), module_idxs = (1,))


@pytest.fixture(scope = 'module')
def qapp():
    """
    Return the QApplication (offscreen).

    Returns:
    app (QApplication): the application.
    """
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(cond, timeout = 30.):
    """
    Process Qt events until a condition holds.

    Parameters:
    cond (callable): the condition.
    timeout (float): seconds before failing. Default 30.
    """
    t0 = time.time()
    while not cond():
        QtWidgets.QApplication.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError('condition not met')
        time.sleep(0.005)


def close(win):
    """
    Close a window and delete it now (pyqtgraph items left for garbage
    collection at exit can raise errors).

    Parameters:
    win (QuickCheckWindow): the window.

    Returns:
    worker (AsyncWorker): its background worker (Python state only).
    """
    worker = win.worker
    win.close()
    win.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None,
                                             QtCore.QEvent.DeferredDelete)
    return worker


def connector(crs, error = None, calls = None):
    """
    Return a connect coroutine function for the window.

    Parameters:
    crs (FakeVnaCRS): the fake to return.
    error (Exception or None): raise this instead. Default None.
    calls (list or None): filled with the modules of each call.

    Returns:
    connect (callable): coroutine function taking the modules.
    """
    async def connect(modules):
        """
        Return the fake, or raise.

        Parameters:
        modules (list of int): the modules asked for.

        Returns:
        crs (FakeVnaCRS): the fake.
        """
        await asyncio.sleep(0)
        if calls is not None:
            calls.append(list(modules))
        if error is not None:
            raise error
        return crs
    return connect


def make_window(crs, **kwargs):
    """
    Open a window on a fake CRS and wait for it to connect.

    Parameters:
    crs (FakeVnaCRS): the fake.
    **kwargs: ``QuickCheckWindow`` arguments.

    Returns:
    win (QuickCheckWindow): the window.
    """
    kwargs.setdefault('pts_per_lw', 1.)
    win = QuickCheckWindow(connector(crs), crs.module_idxs, **kwargs)
    wait_for(lambda: win.crs is not None or 'Could not' in
             win.status.text())
    return win


def test_sweep_plan_text():
    text = sweep_plan_text(100e6, 2.5e9, 500e6, [1], 2e-5, 10)
    assert 'points' in text and 'passes' in text and ' s,' in text
    with pytest.raises(ValueError):
        sweep_plan_text(2e9, 1e9, 500e6, [1], 2e-5, 10)


def test_async_worker(qapp):
    worker = AsyncWorker()
    out = []

    async def job(x):
        """
        Return twice x, or fail.

        Parameters:
        x (int): input.

        Returns:
        y (int): 2 x.
        """
        await asyncio.sleep(0.01)
        if x < 0:
            raise ValueError('negative')
        return 2 * x

    worker.submit(job(3), lambda r, e: out.append((r, e)))
    wait_for(lambda: len(out) == 1)
    assert out[0] == (6, None) and not worker.busy
    worker.submit(job(-1), lambda r, e: out.append((r, e)))
    wait_for(lambda: len(out) == 2)
    assert out[1][0] is None and isinstance(out[1][1], ValueError)
    # Stopping cancels a running job and ends the thread
    worker.submit(asyncio.sleep(60), lambda r, e: out.append((r, e)))
    assert worker.busy
    worker.stop()
    assert not worker.running


def test_window_sweeps_on_connect(qapp, tmp_path):
    crs = FakeVnaCRS(modules = (1, 2), fres = (1.0e9, 1.6e9, 2.1e9))
    win = make_window(crs, amplitude = -40., nsamps = 7,
                      sweep_on_start = True)
    wait_for(lambda: win.result is not None)
    f, z, bands = win.result
    assert 100e6 - 50 <= f[0] and f[-1] < 2.5e9
    assert f'{len(f):,} points.' == win.status.text()
    call = crs.sweep_calls[0]
    assert np.all(call['ares'] == -40.) and call['nsamps'] == 7
    # Band edges are marked on both plots
    assert len(win._band_lines) == 2 * (len(bands['chunk_ncos']) - 1)
    assert win.sweep_btn.isEnabled() and win.save_btn.isEnabled()
    assert win.save_csv(tmp_path / 'p.csv')
    saved = np.genfromtxt(tmp_path / 'p.csv', delimiter = ',', names = True)
    assert np.allclose(saved['f'], f, rtol = 0, atol = 1e-6)
    assert np.allclose(saved['i'] + 1j * saved['q'], z, rtol = 1e-8)
    assert not close(win).running


def test_window_progress_bar(qapp):
    crs = FakeVnaCRS(modules = (1, 2))
    win = make_window(crs, sweep_on_start = False)
    # Hidden when idle (never shown while connecting)
    assert win.progress.isHidden()
    reports = []
    win._progress.connect(lambda d, t: reports.append((d, t)))
    win.sweep()
    # Shown as soon as the sweep starts, indeterminate until a report
    assert not win.progress.isHidden()
    assert win.progress.maximum() == 0 and win.progress.format() == ''
    wait_for(lambda: win.result is not None)
    f = win.result[0]
    # Reports came through the GUI thread, from 0 to every point
    assert reports[0] == (0, len(f)) and reports[-1] == (len(f), len(f))
    # Hidden again when the sweep is done
    assert win.progress.isHidden()
    # The bar shows the percentage and the time left while a sweep runs
    now = [0.]
    win._clock.clock = lambda: now[0]
    win._show_progress(True)
    win._on_progress(0, 10000)
    now[0] = 30.
    win._on_progress(2500, 10000)
    assert win.progress.maximum() == 10000 and win.progress.value() == 2500
    assert win.progress.format() == '25%, about 1 min 30 s left'
    win._on_progress(0, 0)
    assert win.progress.maximum() == 0
    win._show_progress(False)
    # Late reports after the sweep ends don't bring the bar back
    win._on_progress(5, 10)
    assert win.progress.isHidden() and win.progress.maximum() == 0
    close(win)


def test_window_settings_and_plan(qapp):
    crs = FakeVnaCRS(modules = (1,))
    # By default the window waits for the Sweep button once connected
    win = make_window(crs)
    assert crs.sweep_calls == [] and win.sweep_btn.isEnabled()
    assert 'Press Sweep (Ctrl+R)' in win.status.text()
    # Choosing a zone sets the range
    win.zone.setCurrentIndex(2)
    s = win.settings()
    assert (s['fmin'], s['fmax']) == (100e6, 5e9)
    win.zone.setCurrentIndex(1)
    assert win.settings()['fmin'] == 2.5e9
    # The resolution comes from the median Qr and points per linewidth
    win.qr_median.setValue(1e4)
    win.pts_per_lw.setValue(4.)
    assert win.settings()['step'] == pytest.approx(2.5e-5)
    assert 'passes' in win.plan_label.text()
    # An empty range can't be swept
    win.fmin.setValue(4000.)
    win.fmax.setValue(3000.)
    assert 'Invalid' in win.plan_label.text()
    assert not win.sweep_btn.isEnabled()
    win.fmax.setValue(4500.)
    assert win.sweep_btn.isEnabled()
    win.sweep()
    wait_for(lambda: win.result is not None)
    f, _, bands = win.result
    assert f[0] >= 4e9 - 50 and f[-1] < 4.5e9
    assert set(bands['chunk_zones']) == {2}
    close(win)


def test_window_cancel_sweep(qapp):
    """
    Cancel stops a sweep of the simulated board between its points: the
    tones are cleared, the window reports it and is idle again, and a new
    sweep runs with changed settings.
    """
    crs = make_simulated_crs(slow = True)
    win = make_window(crs, nsamps = 1)
    # Nothing to cancel while idle or connecting
    assert not win.cancel_btn.isEnabled() and not win.sweeping
    win.cancel()
    assert not win._cancel.is_set()
    reports = []
    win._progress.connect(lambda d, t: reports.append(d))
    win.sweep()
    assert win.sweeping and win.cancel_btn.isEnabled()
    assert not win.sweep_btn.isEnabled() and not win.fmin.isEnabled()
    # Let the first pass finish, then cancel during the second
    wait_for(lambda: reports and reports[-1] > 0)
    total = reports[-1]
    win.cancel()
    assert 'Cancelling' in win.status.text()
    assert not win.cancel_btn.isEnabled()
    wait_for(lambda: not win.worker.busy)
    wait_for(lambda: 'Cancelled' in win.status.text())
    # Idle again: nothing kept, the settings enabled, no tones written,
    # the zones restored, and the progress bar hidden
    assert win.result is None and not win.sweeping
    assert win.sweep_btn.isEnabled() and win.fmin.isEnabled()
    assert not win.cancel_btn.isEnabled() and not win.save_btn.isEnabled()
    assert crs.ch_map == {} and crs.ntones == 0
    assert crs.module_cfg[1]['nyquist_zone'] == 1
    assert win.progress.isHidden()
    assert 0 < reports[-1] < total * 10
    # A new sweep with other settings runs to the end
    win.fmin.setValue(1000.)
    win.fmax.setValue(1200.)
    win.sweep()
    assert not win._cancel.is_set()
    wait_for(lambda: win.result is not None)
    f = win.result[0]
    assert 1e9 - 50 <= f[0] and f[-1] < 1.2e9
    assert 'points' in win.status.text() and win.save_btn.isEnabled()
    assert not close(win).running


def test_window_connect_and_sweep_errors(qapp):
    crs = FakeVnaCRS()
    win = QuickCheckWindow(connector(crs, RuntimeError('no board')), [1])
    wait_for(lambda: 'Could not connect' in win.status.text())
    assert win.crs is None and not win.sweep_btn.isEnabled()
    close(win)
    crs = FakeVnaCRS(fail_pass = 0)
    win = make_window(crs, sweep_on_start = True)
    wait_for(lambda: 'Sweep failed' in win.status.text())
    assert win.result is None and win.sweep_btn.isEnabled()
    assert win.progress.isHidden()
    # Zones are restored after the failure
    assert crs.module_cfg[1]['nyquist_zone'] == 1
    close(win)


def test_window_modules(qapp):
    calls = []
    crs = FakeVnaCRS(modules = (1, 2), fres = (1.0e9, 1.6e9))
    win = QuickCheckWindow(connector(crs, calls = calls), [1],
                           pts_per_lw = 1.)
    wait_for(lambda: win.crs is not None)
    assert calls == [[1]] and win.modules.text() == '1'
    # Other modules: reconnect
    win.modules.setText('1, 2')
    win._on_connect_clicked()
    wait_for(lambda: len(calls) == 2 and win.sweep_btn.isEnabled())
    assert calls[1] == [1, 2]
    for bad in ('', '9', '1 1', 'a'):
        win.modules.setText(bad)
        win._on_connect_clicked()
        assert 'Modules must be' in win.status.text() and len(calls) == 2
    win.modules.setText('1 2')
    win.sweep()
    wait_for(lambda: win.result is not None and not win.worker.busy)
    assert len(win.result) == 3 and win.save_btn.isEnabled()
    close(win)


def test_window_sweep_connects_typed_modules(qapp):
    """
    Sweep with other modules typed than the connected ones connects to
    them first (so a sweep is never of modules the window doesn't show);
    invalid text stops it.
    """
    calls = []
    crs = FakeVnaCRS(modules = (1, 2), fres = (1.0e9, 1.6e9))
    win = QuickCheckWindow(connector(crs, calls = calls), [1],
                           pts_per_lw = 1.)
    wait_for(lambda: win.crs is not None)
    assert calls == [[1]]
    assert not crs.sweep_calls
    win.modules.setText('2')
    win.sweep()
    wait_for(lambda: win.result is not None and not win.worker.busy)
    assert calls == [[1], [2]] and win._module_idxs == [2]
    n1 = len(crs.sweep_calls)     # passes of the first sweep
    assert n1 and not win._sweep_after_connect
    # Same modules: no reconnection
    result = win.result
    win.sweep()
    wait_for(lambda: win.result is not result and not win.worker.busy)
    assert calls == [[1], [2]] and len(crs.sweep_calls) == 2 * n1
    # Invalid text: no sweep
    win.modules.setText('9')
    win.sweep()
    assert 'Modules must be' in win.status.text()
    assert len(crs.sweep_calls) == 2 * n1 and not win.worker.busy
    close(win)


def test_window_modules_outside_bank(qapp):
    """
    Modules outside the board's active bank are named in the status (the
    session keeps only the bank's modules).
    """
    crs = FakeVnaCRS(modules = (5, 6))
    crs.analog_bank_high = True
    win = QuickCheckWindow(connector(crs), [4, 5])
    wait_for(lambda: 'Connected' in win.status.text())
    assert 'Modules [4] are not in the active analog bank (modules 5-8)' \
        in win.status.text()
    close(win)


def test_progress_clock():
    """
    The time left follows the rate since the step's first report; a new
    total or less done starts a new step.
    """
    from citkid.crs.quick_check import ProgressClock, format_duration
    now = [0.]
    clock = ProgressClock(clock = lambda: now[0])
    assert clock.text(0, 100) == '0%, estimating time left'
    now[0] = 10.
    assert clock.text(25, 100) == '25%, about 30 s left'
    now[0] = 20.
    assert clock.text(50, 100) == '50%, about 20 s left'
    assert clock.text(100, 100) == '100%, finishing'
    # Next step (another stream): measured from its own start
    now[0] = 100.
    assert clock.text(0, 4000) == '0%, estimating time left'
    now[0] = 160.
    assert clock.text(1000, 4000) == '25%, about 3 min 00 s left'
    clock.reset()
    assert clock.text(1000, 4000).endswith('estimating time left')
    assert format_duration(59.6) == '1 min 00 s'
    assert format_duration(3725) == '1 h 02 min'
    assert format_duration(-1) == '0 s'
