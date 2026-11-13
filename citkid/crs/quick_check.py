"""
Interactive quick check of a device with VNA sweeps.

``run_quick_check`` opens a window with the given sweep settings (Nyquist
zone, range, power, resolution, averaging), connects to the CRS, and waits
for the Sweep button (Ctrl+R). The settings can be changed and the sweep
run again; it shows |S21| and phase with the band edges and a rough count
of the resonances (matched filter). A running sweep can be cancelled
(Cancel, Esc): it stops at the next point, clears the board's tones, and
the settings can be changed and the sweep run again. Nothing is saved to
zarr; the sweep can be saved as a CSV of frequency, I, and Q
(``save_sweep_csv``). ``procedures/vna_quick_check.py`` is the script.

The CRS lives on a background thread with its own asyncio event loop
(``AsyncWorker``), so the window stays responsive while sweeping. The
sweep itself is ``crs.procedures.vna_sweep``; its ``progress`` callback
drives the window's progress bar, and its ``cancelled`` callable reads the
window's cancel flag (a ``threading.Event``), so cancelling is
cooperative: the sweep ends between points, not in the middle of a
board command.
"""
import asyncio
import threading
import time
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
    SweepCancelled, VNA_RANGES, plan_vna_sweep, vna_range, vna_step,
    vna_sweep
)
from .util import get_sample_freq

# Zone choices: (value, label)
ZONES = (('1', '1: 100-2500 MHz'), ('2', '2: 2500-5000 MHz'),
         ('both', 'both: 100-5000 MHz'))

################################################################################
################################### Helpers ####################################
################################################################################
def save_sweep_csv(path, f, z):
    """
    Save a sweep as a CSV file with columns ``f`` (Hz), ``i``, and ``q``.

    Parameters:
    path (str): file path.
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 (i + 1j * q).
    """
    data = np.column_stack([f, np.real(z), np.imag(z)])
    np.savetxt(str(path), data, delimiter = ',', header = 'f,i,q',
               comments = '', fmt = ['%.6f', '%.9e', '%.9e'])

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

def format_duration(seconds):
    """
    Format a duration for a progress bar.

    Parameters:
    seconds (float): the duration, s.

    Returns:
    text (str): e.g. '45 s', '3 min 20 s', '1 h 05 min'.
    """
    s = int(round(max(seconds, 0)))
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60} min {s % 60:02d} s'
    return f'{s // 3600} h {(s % 3600) // 60:02d} min'

class ProgressClock:
    """
    Estimate the time left of a job from its progress reports.

    The rate is measured from the first report of the current step: a
    report with another total, or with less done than the last one, starts
    a new step (e.g. the next stream of a run).

    Attributes:
    clock (callable): time source, s. Default ``time.monotonic``.
    """

    def __init__(self, clock = time.monotonic):
        """
        Start with no step.

        Parameters:
        clock (callable): time source, s. Default ``time.monotonic``.
        """
        self.clock = clock
        self.reset()

    def reset(self):
        """
        Forget the current step.
        """
        self._start = None
        self._last = (None, None)

    def text(self, done, total):
        """
        Record a report and describe the progress.

        Parameters:
        done (int): units done in this step.
        total (int): units in this step.

        Returns:
        text (str): percentage and estimated time left, e.g. '45%, about
            2 min 10 s left' ('estimating time left' until it can tell).
        """
        now = self.clock()
        last_done, last_total = self._last
        if self._start is None or total != last_total or \
                (last_done is not None and done < last_done):
            self._start = (now, done)
        self._last = (done, total)
        pct = 100 * done / total if total > 0 else 0
        t0, d0 = self._start
        if done >= total:
            return f'{pct:.0f}%, finishing'
        if done <= d0 or now <= t0:
            return f'{pct:.0f}%, estimating time left'
        left = (total - done) * (now - t0) / (done - d0)
        return f'{pct:.0f}%, about {format_duration(left)} left'

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
        # Queued: a job that ends before ``submit`` returns (its done
        # callback then runs on the GUI thread) still reports from the
        # event loop, after the caller has set up its status
        self._done.connect(self._dispatch, Qt.QueuedConnection)

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
    progress (QProgressBar): points taken in the running sweep; hidden
        when idle.
    sweeping (bool): True while a sweep runs (not while connecting or
        counting).
    """
    # Sweep progress (points done, total), emitted from the background
    # thread and delivered on the GUI thread
    _progress = QtCore.Signal(int, int)

    def __init__(
        self, connect, module_idxs, nyquist = '1', fmin = None, fmax = None,
        amplitude = -50., qr_median = 2e4, pts_per_lw = 5., nsamps = 10,
        count = True, sweep_on_start = False, title = 'VNA quick check',
        ui_scale = 1.0
    ):
        """
        Build the window and start connecting.

        Parameters:
        connect (callable): coroutine function taking a list of modules
            and returning a CRS attached to them (run on the background
            loop). It is called again when the modules are changed in the
            window.
        module_idxs (list of int): the starting modules (the ones connected
            to the device; several if it is split between modules).
        nyquist (str): starting zone: '1' (default), '2', or 'both'.
        fmin, fmax (float or None): starting range in Hz. None (default):
            the zone's.
        amplitude (float): tone power in dBm. Default -50.
        qr_median (float): expected median Qr. Default 2e4.
        pts_per_lw (float): points per linewidth at ``qr_median``. Default
            5.
        nsamps (int): samples per point. Default 10.
        count (bool): If True (default), count resonances after sweeping.
        sweep_on_start (bool): If True, sweep as soon as connected. Default
            False: wait for the Sweep button (Ctrl+R).
        title (str): window title. Default 'VNA quick check'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(title)
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.crs = None
        self.result = None
        self.sweeping = False
        self._plan_ok = False
        # Set to ask the running sweep to stop; read on the background loop
        self._cancel = threading.Event()
        self.worker = AsyncWorker(self)
        self._module_idxs = [int(m) for m in module_idxs]
        self._connect = connect
        self._sweep_on_start = sweep_on_start
        self._sweep_after_connect = False   # Sweep pressed with new modules
        self._band_lines = []
        self._clock = ProgressClock()
        self._build_controls(nyquist, fmin, fmax, amplitude, qr_median,
                             pts_per_lw, nsamps, count)
        self._build_plots()
        self._progress.connect(self._on_progress)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        self._update_plan()
        self.reconnect(self._module_idxs)

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
        self.modules = QtWidgets.QLineEdit(
            ' '.join(str(m) for m in self._module_idxs))
        self.modules.setToolTip('Modules connected to the device, e.g. 1 or '
                                '1 2 3 4; Sweep connects to them')
        self.connect_btn = QtWidgets.QPushButton('Connect')
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.modules)
        row.addWidget(self.connect_btn)
        form.addRow('Modules', row)
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
        self.count = QtWidgets.QCheckBox('Count resonances after sweeping')
        self.count.setChecked(bool(count))
        self.count_btn = QtWidgets.QPushButton('Count now')
        self.count_btn.setToolTip('Count the resonances of the last sweep '
                                  '(no new data)')
        self.count_btn.clicked.connect(self.count_resonances)
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
        self.cancel_btn = QtWidgets.QPushButton('Cancel (Esc)')
        self.cancel_btn.setToolTip('Stop the running sweep at the next '
                                   'point; nothing is kept')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        QtWidgets.QShortcut(QtGui.QKeySequence('Esc'), self, self.cancel)
        self.save_btn = QtWidgets.QPushButton('Save data (CSV)...')
        self.save_btn.clicked.connect(self._ask_save_csv)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.sweep_btn, stretch = 1)
        row.addWidget(self.cancel_btn)
        form.addRow(row)
        form.addRow(self.count_btn)
        form.addRow(self.save_btn)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        form.addRow(self.progress)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self._inputs = (self.modules, self.connect_btn, self.zone,
                        self.fmin, self.fmax, self.amplitude,
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
        self.count_btn.setEnabled(not busy and self.result is not None)
        # Only a sweep can be cancelled (not connecting or counting), and
        # only once
        self.cancel_btn.setEnabled(busy and self.sweeping and
                                   not self._cancel.is_set())
        self._update_sweep_button()

    def _show_progress(self, show):
        """
        Show the progress bar, reset and indeterminate until the first
        report, or hide it.

        Parameters:
        show (bool): True to show it.
        """
        self.progress.setRange(0, 0)
        self.progress.setFormat('')
        self.progress.setVisible(show)
        self._clock.reset()

    def _on_progress(self, done, total):
        """
        Update the progress bar (GUI thread).

        Parameters:
        done (int): points taken so far.
        total (int): points in the sweep.
        """
        if self.progress.isHidden():
            return
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(min(done, total))
        self.progress.setFormat(self._clock.text(done, total))

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
        if not len(modules):
            modules = [1]
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
    def parse_modules(self):
        """
        Read the modules typed in the window.

        Returns:
        modules (list of int or None): the modules (1-8, no repeats), or
            None if the text isn't valid.
        """
        try:
            mods = [int(m) for m in self.modules.text().replace(',', ' ')
                    .split()]
        except ValueError:
            return None
        if not mods or len(set(mods)) != len(mods) or \
                not all(1 <= m <= 8 for m in mods):
            return None
        return mods

    def _on_connect_clicked(self):
        """
        Connect to the modules typed in the window.
        """
        mods = self.parse_modules()
        if mods is None:
            self._set_status('Modules must be numbers 1-8, e.g. "1" or '
                             '"1 2 3 4".')
            return
        self.reconnect(mods)

    def reconnect(self, module_idxs):
        """
        Connect to a list of modules (on the background loop), replacing
        the current connection when it succeeds.

        Parameters:
        module_idxs (list of int): the modules.
        """
        self._module_idxs = [int(m) for m in module_idxs]
        self.worker.submit(self._connect(list(self._module_idxs)),
                           self._on_connected)
        self._set_busy(True)
        self._set_status(f'Connecting to modules {self._module_idxs}...')

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
            return
        self.crs = crs
        self._set_busy(False)
        self._update_plan()
        text = (f'Connected: modules {list(crs.active_module_idxs)}, '
                f'{crs.bw / 1e6:.0f} MHz per module.')
        # attach keeps only the modules of the board's active analog bank
        dropped = [m for m in self._module_idxs
                   if m not in crs.active_module_idxs]
        if dropped:
            bank = '5-8' if getattr(crs, 'analog_bank_high', False) \
                else '1-4'
            text += (f' Modules {dropped} are not in the active analog bank '
                     f'(modules {bank}); crs_init.py sets the bank.')
        if self._sweep_on_start or self._sweep_after_connect:
            self._sweep_after_connect = False
            self._set_status(text)
            self.sweep()
        else:
            self._set_status(text + ' Press Sweep (Ctrl+R) to start.')

    def sweep(self):
        """
        Sweep with the current settings (on the background loop). If the
        modules typed in the window aren't the connected ones, connect to
        them first, so the sweep is never of other modules than the
        window shows.
        """
        if not self.sweep_btn.isEnabled():
            return
        mods = self.parse_modules()
        if mods is None:
            self._set_status('Modules must be numbers 1-8, e.g. "1" or '
                             '"1 2 3 4".')
            return
        if mods != self._module_idxs:
            self._sweep_after_connect = True
            self.reconnect(mods)
            return
        s = self.settings()
        crs = self.crs
        report = self._progress.emit
        self._cancel.clear()
        cancelled = self._cancel.is_set

        async def job():
            """
            Sweep and count.

            Returns:
            result (tuple): ``f``, ``z``, ``bands``, ``fr``.
            """
            f, z, bands = await vna_sweep(
                crs, s['fmin'], s['fmax'], s['amplitude'], s['nsamps'],
                step = s['step'], verbose = False, progress = report,
                cancelled = cancelled)
            fr = rough_resonances(f, z) if s['count'] else None
            return f, z, bands, fr

        self.sweeping = True
        self.worker.submit(job(), self._on_swept)
        self._set_busy(True)
        self._show_progress(True)
        self._set_status(f'Sweeping modules {list(crs.active_module_idxs)}'
                         '... (Cancel or Esc to stop)')

    def cancel(self):
        """
        Ask the running sweep to stop (the Cancel button, Esc). The sweep
        ends at its next point, on the background loop, and
        ``_on_swept`` then reports it as cancelled. Does nothing when no
        sweep runs.
        """
        if not self.sweeping or self._cancel.is_set():
            return
        self._cancel.set()
        self.cancel_btn.setEnabled(False)
        self._set_status('Cancelling...')

    def _on_swept(self, result, error):
        """
        Show a finished sweep (GUI thread).

        Parameters:
        result (tuple or None): ``f``, ``z``, ``bands``, ``fr``.
        error (BaseException or None): the sweep error.
        """
        self._show_progress(False)
        self.sweeping = False
        if isinstance(error, SweepCancelled):
            self._set_status('Cancelled. Change the settings and press '
                             'Sweep (Ctrl+R) to run again.')
            self._set_busy(False)
            return
        if error is not None:
            self._set_status(f'Sweep failed: {error!r}')
            self._set_busy(False)
            return
        self.result = result
        f, z, bands, fr = result
        self._plot(f, z, bands, fr)
        text = f'{len(f):,} points.'
        if fr is not None:
            text += f' About {len(fr)} resonances (rough count).'
        self._set_status(text)
        self._set_busy(False)

    def count_resonances(self):
        """
        Count the resonances of the last sweep (no new data), on the
        background loop.
        """
        if self.result is None or self.worker.busy:
            return
        f, z, bands, _ = self.result

        async def job():
            """
            Count.

            Returns:
            result (tuple): ``f``, ``z``, ``bands``, ``fr``.
            """
            return f, z, bands, rough_resonances(f, z)

        self.worker.submit(job(), self._on_swept)
        self._set_busy(True)
        self._set_status('Counting resonances...')

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
    def save_csv(self, path):
        """
        Save the last sweep as a CSV of frequency (Hz), I, and Q.

        Parameters:
        path (str): file path.

        Returns:
        ok (bool): True if saved, False if there is no sweep or writing
            failed.
        """
        if self.result is None:
            return False
        f, z, _, _ = self.result
        try:
            save_sweep_csv(path, f, z)
        except OSError:
            return False
        return True

    def _ask_save_csv(self):
        """
        Ask for a file name and save the last sweep as a CSV.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save data', 'quick_check.csv', 'CSV (*.csv)')
        if path:
            ok = self.save_csv(path)
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
        # Let a running sweep stop at its next point (clearing the tones)
        # while the worker drains
        self._cancel.set()
        self.worker.stop()
        event.accept()

################################################################################
################################### Launcher ###################################
################################################################################
def run_quick_check(
    connect, module_idxs, nyquist = '1', fmin = None, fmax = None,
    amplitude = -50., qr_median = 2e4, pts_per_lw = 5., nsamps = 10,
    count = True, sweep_on_start = False, title = 'VNA quick check',
    ui_scale = 1.0
):
    """
    Open the quick-check window and wait until it is closed.

    Parameters:
    connect (callable): coroutine function taking a list of modules and
        returning a CRS attached to them, e.g. creating ``CRS(...,
        module_idxs = modules, global_control = False)`` and awaiting
        ``attach``. It runs on the window's background event loop, which
        every sweep then uses, and again when the modules are changed in
        the window.
    module_idxs (list of int): the starting modules.
    nyquist, fmin, fmax, amplitude, qr_median, pts_per_lw, nsamps, count,
        sweep_on_start, title, ui_scale: starting settings (see
        ``QuickCheckWindow``). By default the window waits for the Sweep
        button once connected.

    Returns:
    result (tuple or None): the last sweep (``f``, ``z``, ``bands``,
        ``fr``), or None if there was none.
    """
    app = get_qapp(title)
    win = QuickCheckWindow(
        connect, module_idxs, nyquist = nyquist, fmin = fmin, fmax = fmax,
        amplitude = amplitude, qr_median = qr_median,
        pts_per_lw = pts_per_lw, nsamps = nsamps, count = count,
        sweep_on_start = sweep_on_start, title = title, ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    # exec() is the modern name (PyQt6+); exec_() is kept for PyQt5.
    (getattr(app, 'exec', None) or app.exec_)()
    return win.result
