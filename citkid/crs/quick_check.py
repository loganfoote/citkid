"""
Interactive quick check of a device with VNA sweeps.

``run_quick_check`` opens a window where the sweep settings (Nyquist zone,
range, power, resolution, averaging) can be changed and the sweep run
again, showing |S21| and phase with the band edges and a rough count of
the resonances (matched filter). Nothing is saved to zarr; the plot can be
saved as a PNG. ``procedures/01_vna_quick_check.py`` is the script.

The CRS lives on a background thread with its own asyncio event loop
(``AsyncWorker``), so the window stays responsive while sweeping. The
sweep itself is ``crs.procedures.vna_sweep``.
"""
import asyncio
import threading
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from ..qt_compat import (
    Qt, delete_on_close, fit_window_to_screen, get_qapp,
    widget_font_stylesheet
)
from ..vna.matched_filter import find_candidates_mf
from ..vna.sweep_info import prepare_full_sweep
from .procedures import (
    VNA_RANGES, plan_vna_sweep, vna_range, vna_step, vna_sweep
)
from .util import get_sample_freq

# Zone choices: (value, label)
ZONES = (('1', '1: 100-2500 MHz'), ('2', '2: 2500-5000 MHz'),
         ('both', 'both: 100-5000 MHz'))

################################################################################
################################### Helpers ####################################
################################################################################
def rough_resonances(f, z, threshold = 12.):
    """
    Find resonances roughly with the matched filter, for a quick count.

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21.
    threshold (float): detection threshold (noise units). Default 12 (the
        CNN finder's matched-filter setting).

    Returns:
    fr (np.array): resonance frequencies in Hz, ascending.
    """
    fp, zp, _ = prepare_full_sweep(f, z)
    return find_candidates_mf(fp, zp, threshold = threshold,
                              suppress_sidelobes = True)[0]

def sweep_plan_text(fmin, fmax, bw, module_idxs, step, nsamps):
    """
    Describe the VNA sweep plan and its shortest possible duration.

    Parameters:
    fmin, fmax (float): range in Hz.
    bw (float): module bandwidth in Hz.
    module_idxs (list of int): modules to sweep with.
    step (float): df/f.
    nsamps (int): samples per point.

    Returns:
    text (str): points, bands, passes, points per channel, and the
        sampling time (sweeps run at decimation stage 6; overheads add to
        it).

    Raises:
    ValueError: if the plan is invalid (``plan_vna_sweep``).
    """
    bands, freqs, npc = plan_vna_sweep(fmin, fmax, bw, module_idxs, step)
    npass = int(bands['chunk_passes'].max()) + 1
    t = npass * npc * nsamps / get_sample_freq(6)
    t = f'{t:.0f} s' if t < 120 else f'{t / 60:.1f} min'
    return (f'{sum(fb.size for fb in freqs):,} points, {len(freqs)} bands '
            f'in {npass} passes of {npc} points per channel. Sampling '
            f'takes {t}, plus overheads.')

class AsyncWorker(QtCore.QObject):
    """
    Run coroutines on a background thread's asyncio event loop and report
    their results on the GUI thread.

    Attributes:
    loop (asyncio.AbstractEventLoop): the background loop.
    future (concurrent.futures.Future or None): the last submitted job.
    """
    _done = QtCore.Signal(object, object, object)

    def __init__(self, parent = None):
        """
        Start the background loop.

        Parameters:
        parent (QObject or None): Qt parent. Default None.
        """
        super().__init__(parent)
        self.loop = asyncio.new_event_loop()
        self.future = None
        self._thread = threading.Thread(target = self._run, daemon = True)
        self._thread.start()
        self._done.connect(self._dispatch)

    def _run(self):
        """
        Run the loop until ``stop``.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @property
    def busy(self):
        """
        Return whether a job is running.

        Returns:
        busy (bool): True while the last job hasn't finished.
        """
        return self.future is not None and not self.future.done()

    @property
    def running(self):
        """
        Return whether the background thread is running.

        Returns:
        running (bool): True until ``stop``.
        """
        return self._thread.is_alive()

    def submit(self, coro, callback):
        """
        Run a coroutine on the loop.

        Parameters:
        coro (coroutine): the job.
        callback (callable): called on the GUI thread as ``callback(result,
            error)`` when the job ends: ``error`` is None or the exception
            it raised.

        Returns:
        future (concurrent.futures.Future): the job.
        """
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        self.future = fut

        def done(fut):
            """
            Pass the outcome to the GUI thread.

            Parameters:
            fut (concurrent.futures.Future): the finished job.
            """
            try:
                result, error = fut.result(), None
            except BaseException as e:
                result, error = None, e
            self._done.emit(callback, result, error)

        fut.add_done_callback(done)
        return fut

    def _dispatch(self, callback, result, error):
        """
        Call a job's callback (GUI thread).

        Parameters:
        callback (callable): the callback.
        result: the job's result.
        error (BaseException or None): the job's exception.
        """
        callback(result, error)

    def stop(self, timeout = 30.):
        """
        Cancel the running job, let it clean up (e.g. restore the Nyquist
        zones), and stop the loop and its thread.

        Parameters:
        timeout (float): seconds to wait for the cleanup. Default 30.
        """
        if not self.running:
            return
        if self.busy:
            self.future.cancel()

        async def drain():
            """
            Wait for the loop's other tasks to finish.
            """
            tasks = [t for t in asyncio.all_tasks()
                     if t is not asyncio.current_task()]
            if tasks:
                await asyncio.wait(tasks, timeout = timeout)

        try:
            asyncio.run_coroutine_threadsafe(drain(), self.loop).result(
                timeout + 1)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout)
        if not self._thread.is_alive():
            self.loop.close()

################################################################################
#################################### Window ####################################
################################################################################
class QuickCheckWindow(QtWidgets.QMainWindow):
    """
    Window for quick VNA checks: settings on the left, |S21| and phase on
    the right.

    Attributes:
    crs (CRS or None): the connected instrument (None until connected).
    result (tuple or None): the last sweep: ``f``, ``z``, ``bands``
        (``vna_sweep`` outputs), and ``fr`` (rough resonance frequencies,
        or None if not counted).
    """

    def __init__(
        self, connect, module_idxs, nyquist = '1', fmin = None, fmax = None,
        amplitude = -50., qr_median = 2e4, pts_per_lw = 5., nsamps = 10,
        count = True, sweep_on_start = True, title = 'VNA quick check',
        ui_scale = 1.0
    ):
        """
        Build the window and start connecting.

        Parameters:
        connect (callable): coroutine function with no arguments that
            returns an attached CRS (run on the background loop).
        module_idxs (list of int): the session's modules (for the plan
            shown before connecting).
        nyquist (str): starting zone: '1' (default), '2', or 'both'.
        fmin, fmax (float or None): starting range in Hz. None (default):
            the zone's.
        amplitude (float): tone power in dBm. Default -50.
        qr_median (float): expected median Qr. Default 2e4.
        pts_per_lw (float): points per linewidth at ``qr_median``. Default
            5.
        nsamps (int): samples per point. Default 10.
        count (bool): If True (default), count resonances after sweeping.
        sweep_on_start (bool): If True (default), sweep once connected.
        title (str): window title. Default 'VNA quick check'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(title)
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.crs = None
        self.result = None
        self._plan_ok = False
        self.worker = AsyncWorker(self)
        self._module_idxs = [int(m) for m in module_idxs]
        self._sweep_on_start = sweep_on_start
        self._band_lines = []
        self._build_controls(nyquist, fmin, fmax, amplitude, qr_median,
                             pts_per_lw, nsamps, count)
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        self._set_busy(True)
        self._set_status('Connecting to the CRS...')
        self._update_plan()
        self.worker.submit(connect(), self._on_connected)

    ############################################################################
    def _build_controls(
        self, nyquist, fmin, fmax, amplitude, qr_median, pts_per_lw, nsamps,
        count
    ):
        """
        Build the settings panel.

        Parameters:
        nyquist, fmin, fmax, amplitude, qr_median, pts_per_lw, nsamps,
            count: starting values (see ``__init__``).
        """
        def spin(lo, hi, value, decimals = 1, step = 1., suffix = ''):
            """
            Make a number box.

            Parameters:
            lo, hi (float): range.
            value (float): starting value.
            decimals (int): decimals shown. Default 1.
            step (float): arrow step. Default 1.
            suffix (str): unit shown. Default ''.

            Returns:
            box (QDoubleSpinBox): the box.
            """
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setSuffix(suffix)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self._update_plan)
            return box

        self._panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self._panel)
        self.zone = QtWidgets.QComboBox()
        for value, label in ZONES:
            self.zone.addItem(label, value)
        self.zone.setCurrentIndex([z for z, _ in ZONES].index(nyquist))
        lo, hi = vna_range(nyquist, fmin, fmax)
        self.fmin = spin(1., 5000., lo / 1e6, suffix = ' MHz', step = 10.)
        self.fmax = spin(1., 5000., hi / 1e6, suffix = ' MHz', step = 10.)
        self.zone.currentIndexChanged.connect(self._on_zone)
        self.amplitude = spin(-120., 7., amplitude, suffix = ' dBm')
        self.qr_median = spin(100., 1e7, qr_median, decimals = 0,
                              step = 1000.)
        self.pts_per_lw = spin(0.5, 200., pts_per_lw, step = 1.)
        self.nsamps = QtWidgets.QSpinBox()
        self.nsamps.setRange(1, 100000)
        self.nsamps.setValue(int(nsamps))
        self.nsamps.setKeyboardTracking(False)
        self.nsamps.valueChanged.connect(self._update_plan)
        self.count = QtWidgets.QCheckBox('Count resonances')
        self.count.setChecked(bool(count))
        form.addRow('Nyquist zone', self.zone)
        form.addRow('From', self.fmin)
        form.addRow('To', self.fmax)
        form.addRow('Power', self.amplitude)
        form.addRow('Median Qr', self.qr_median)
        form.addRow('Points per linewidth', self.pts_per_lw)
        form.addRow('Samples per point', self.nsamps)
        form.addRow(self.count)
        # Width of the settings; the wrapped labels below follow it
        width = self._panel.sizeHint().width() + 40
        self.plan_label = QtWidgets.QLabel()
        self.plan_label.setWordWrap(True)
        form.addRow(self.plan_label)
        self.sweep_btn = QtWidgets.QPushButton('Sweep (Ctrl+R)')
        self.sweep_btn.clicked.connect(self.sweep)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, self.sweep)
        self.save_btn = QtWidgets.QPushButton('Save plot...')
        self.save_btn.clicked.connect(self._ask_save_plot)
        form.addRow(self.sweep_btn)
        form.addRow(self.save_btn)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self._inputs = (self.zone, self.fmin, self.fmax, self.amplitude,
                        self.qr_median, self.pts_per_lw, self.nsamps,
                        self.count)
        self._panel.setFixedWidth(width)

    def _build_plots(self):
        """
        Build the |S21| and phase plots.
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.mag_plot = self.plots.addPlot(row = 0, col = 0)
        self.phase_plot = self.plots.addPlot(row = 1, col = 0)
        self.phase_plot.setXLink(self.mag_plot)
        self.mag_plot.setLabel('left', '|S21| (dB)')
        self.phase_plot.setLabel('left', 'Phase (rad)')
        self.phase_plot.setLabel('bottom', 'Frequency (MHz)')
        self.mag_curve = self.mag_plot.plot(pen = pg.mkPen(width = 1))
        self.phase_curve = self.phase_plot.plot(pen = pg.mkPen(width = 1))
        for item in (self.mag_curve, self.phase_curve):
            item.setDownsampling(auto = True, method = 'peak')
            item.setClipToView(True)
        self.markers = pg.ScatterPlotItem(
            size = 7, symbol = 't', brush = pg.mkBrush(255, 80, 80))
        self.mag_plot.addItem(self.markers)

    ############################################################################
    def _set_status(self, text):
        """
        Show a status message.

        Parameters:
        text (str): the message.
        """
        self.status.setText(text)

    def _set_busy(self, busy):
        """
        Enable or disable the inputs while connecting or sweeping.

        Parameters:
        busy (bool): True while a job runs.
        """
        for w in self._inputs:
            w.setEnabled(not busy)
        self.save_btn.setEnabled(not busy and self.result is not None)
        self._update_sweep_button()

    def _update_sweep_button(self):
        """
        Enable the sweep button when connected, idle, and the plan is
        valid.
        """
        self.sweep_btn.setEnabled(self.crs is not None and self._plan_ok
                                  and not self.worker.busy)

    def _on_zone(self):
        """
        Set the range to the chosen zone's.
        """
        lo, hi = VNA_RANGES[self.zone.currentData()]
        for box, value in ((self.fmin, lo), (self.fmax, hi)):
            box.blockSignals(True)
            box.setValue(value / 1e6)
            box.blockSignals(False)
        self._update_plan()

    def settings(self):
        """
        Return the current sweep settings.

        Returns:
        settings (dict): ``fmin``, ``fmax`` (Hz), ``amplitude`` (dBm),
            ``nsamps``, ``step`` (df/f), ``count`` (bool).
        """
        return dict(
            fmin = self.fmin.value() * 1e6, fmax = self.fmax.value() * 1e6,
            amplitude = self.amplitude.value(), nsamps = self.nsamps.value(),
            step = vna_step(self.qr_median.value(), self.pts_per_lw.value()),
            count = self.count.isChecked())

    def _update_plan(self):
        """
        Show the plan for the current settings, or why they are invalid.
        """
        s = self.settings()
        bw = self.crs.bw if self.crs is not None else 500e6
        modules = self.crs.active_module_idxs if self.crs is not None \
            else self._module_idxs
        try:
            text = sweep_plan_text(s['fmin'], s['fmax'], bw, modules,
                                   s['step'], s['nsamps'])
            self._plan_ok = True
        except ValueError as e:
            text = f'Invalid settings: {e}'
            self._plan_ok = False
        self.plan_label.setText(text)
        self._update_sweep_button()

    ############################################################################
    def _on_connected(self, crs, error):
        """
        Keep the CRS and sweep once (GUI thread).

        Parameters:
        crs (CRS or None): the attached instrument.
        error (BaseException or None): the connection error.
        """
        if error is not None:
            self._set_status(f'Could not connect: {error!r}')
            self._set_busy(False)
            self.sweep_btn.setEnabled(False)
            return
        self.crs = crs
        self._set_busy(False)
        self._update_plan()
        self._set_status(
            f'Connected: modules {list(crs.active_module_idxs)}, '
            f'{crs.bw / 1e6:.0f} MHz per module.')
        if self._sweep_on_start:
            self.sweep()

    def sweep(self):
        """
        Sweep with the current settings (on the background loop).
        """
        if not self.sweep_btn.isEnabled():
            return
        s = self.settings()
        crs = self.crs

        async def job():
            """
            Sweep and count.

            Returns:
            result (tuple): ``f``, ``z``, ``bands``, ``fr``.
            """
            f, z, bands = await vna_sweep(
                crs, s['fmin'], s['fmax'], s['amplitude'], s['nsamps'],
                step = s['step'], verbose = False)
            fr = rough_resonances(f, z) if s['count'] else None
            return f, z, bands, fr

        self.worker.submit(job(), self._on_swept)
        self._set_busy(True)
        self._set_status('Sweeping...')

    def _on_swept(self, result, error):
        """
        Show a finished sweep (GUI thread).

        Parameters:
        result (tuple or None): ``f``, ``z``, ``bands``, ``fr``.
        error (BaseException or None): the sweep error.
        """
        if error is not None:
            self._set_status(f'Sweep failed: {error!r}')
            self._set_busy(False)
            return
        self.result = result
        f, z, bands, fr = result
        self._plot(f, z, bands, fr)
        text = f'Swept {len(f):,} points.'
        if fr is not None:
            text += f' About {len(fr)} resonances (rough count).'
        self._set_status(text)
        self._set_busy(False)

    def _plot(self, f, z, bands, fr):
        """
        Plot a sweep.

        Parameters:
        f, z, bands, fr: see ``_on_swept``.
        """
        fm = f / 1e6
        db = 20 * np.log10(np.abs(z))
        self.mag_curve.setData(fm, db)
        self.phase_curve.setData(fm, np.unwrap(np.angle(z)))
        for line in self._band_lines:
            line[0].removeItem(line[1])
        self._band_lines = []
        pen = pg.mkPen((150, 150, 150), style = Qt.DotLine)
        for lo in bands['chunk_edges'][1:, 0] / 1e6:
            for plot in (self.mag_plot, self.phase_plot):
                line = pg.InfiniteLine(lo, pen = pen)
                plot.addItem(line)
                self._band_lines.append((plot, line))
        if fr is None or not len(fr):
            self.markers.setData([], [])
        else:
            self.markers.setData(fr / 1e6, np.interp(fr, f, db) + 1)
        self.mag_plot.enableAutoRange()
        self.phase_plot.enableAutoRange()

    ############################################################################
    def save_plot(self, path):
        """
        Save the plots as an image.

        Parameters:
        path (str): file path (.png, .jpg, ...).

        Returns:
        ok (bool): True if saved.
        """
        return bool(self.plots.grab().save(str(path)))

    def _ask_save_plot(self):
        """
        Ask for a file name and save the plots.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save plot', 'quick_check.png', 'PNG (*.png)')
        if path:
            ok = self.save_plot(path)
            self._set_status(f'Saved {path}' if ok else
                             f'Could not save {path}')

    def closeEvent(self, event):
        """
        Ask before closing during a sweep; then stop the background loop.

        Parameters:
        event (QCloseEvent): the close event.
        """
        if self.worker.busy:
            answer = QtWidgets.QMessageBox.question(
                self, 'Sweep running',
                'A sweep is running. Stop it and close?')
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self.worker.stop()
        event.accept()

################################################################################
################################### Launcher ###################################
################################################################################
def run_quick_check(
    connect, module_idxs, nyquist = '1', fmin = None, fmax = None,
    amplitude = -50., qr_median = 2e4, pts_per_lw = 5., nsamps = 10,
    count = True, title = 'VNA quick check', ui_scale = 1.0
):
    """
    Open the quick-check window and wait until it is closed.

    Parameters:
    connect (callable): coroutine function with no arguments that returns
        an attached CRS, e.g. creating ``CRS(..., global_control = False)``
        and awaiting ``attach``. It runs on the window's background event
        loop, which every sweep then uses.
    module_idxs (list of int): the session's modules.
    nyquist, fmin, fmax, amplitude, qr_median, pts_per_lw, nsamps, count,
        title, ui_scale: starting settings (see ``QuickCheckWindow``).

    Returns:
    result (tuple or None): the last sweep (``f``, ``z``, ``bands``,
        ``fr``), or None if there was none.
    """
    app = get_qapp(title)
    win = QuickCheckWindow(
        connect, module_idxs, nyquist = nyquist, fmin = fmin, fmax = fmax,
        amplitude = amplitude, qr_median = qr_median,
        pts_per_lw = pts_per_lw, nsamps = nsamps, count = count,
        title = title, ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    # exec() is the modern name (PyQt6+); exec_() is kept for PyQt5.
    (getattr(app, 'exec', None) or app.exec_)()
    return win.result
