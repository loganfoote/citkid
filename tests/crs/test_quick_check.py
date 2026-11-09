"""
Tests for the interactive VNA quick check, run against a fake CRS.
"""
import asyncio
import time
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from citkid.crs.quick_check import (
    AsyncWorker, QuickCheckWindow, rough_resonances, sweep_plan_text
)
from .fake_crs import FakeVnaCRS


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


def connector(crs, error = None):
    """
    Return a connect coroutine function for the window.

    Parameters:
    crs (FakeVnaCRS): the fake to return.
    error (Exception or None): raise this instead. Default None.

    Returns:
    connect (callable): coroutine function.
    """
    async def connect():
        """
        Return the fake, or raise.

        Returns:
        crs (FakeVnaCRS): the fake.
        """
        await asyncio.sleep(0)
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


def test_rough_resonances_and_plan_text():
    f = 1e9 * np.exp(np.arange(200000) * 2e-6)
    z = np.ones_like(f, complex)
    for fr in (1.05e9, 1.2e9):
        z *= 1 - 0.5 / (1 + 2j * 2e4 * (f - fr) / fr)
    z += 1e-3 * np.random.default_rng(0).standard_normal(len(f))
    fr = rough_resonances(f, z)
    assert len(fr) == 2 and np.allclose(fr, [1.05e9, 1.2e9], rtol = 1e-4)
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
    f, z, bands, fr = win.result
    assert 100e6 - 50 <= f[0] and f[-1] < 2.5e9
    assert len(fr) == 3 and 'About 3 resonances' in win.status.text()
    call = crs.sweep_calls[0]
    assert np.all(call['ares'] == -40.) and call['nsamps'] == 7
    # Band edges are marked on both plots, resonances on |S21|
    assert len(win._band_lines) == 2 * (len(bands['chunk_ncos']) - 1)
    assert len(win.markers.data) == 3
    assert win.sweep_btn.isEnabled() and win.save_btn.isEnabled()
    assert win.save_plot(tmp_path / 'p.png')
    assert (tmp_path / 'p.png').stat().st_size > 0
    assert not close(win).running


def test_window_progress_bar(qapp):
    crs = FakeVnaCRS(modules = (1, 2))
    win = make_window(crs, sweep_on_start = False, count = False)
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
    # The bar shows points and a percentage while a sweep runs
    win._show_progress(True)
    win._on_progress(2500, 10000)
    assert win.progress.maximum() == 10000 and win.progress.value() == 2500
    assert win.progress.format() == '2,500 / 10,000 points (%p%)'
    assert win.progress.text() == '2,500 / 10,000 points (25%)'
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
    win = make_window(crs, count = False)
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
    f, _, bands, fr = win.result
    assert fr is None and f[0] >= 4e9 - 50 and f[-1] < 4.5e9
    assert set(bands['chunk_zones']) == {2}
    close(win)


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
