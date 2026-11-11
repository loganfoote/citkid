"""
Interactive timestreams of a tuning: preview one tone set, then take named
timestreams of every tone set.

``run_stream_gui`` opens a window on a tuning (``tune{k}`` with ``tones``,
from ``procedures/03_tune.py``):

- Preview: pick a tone set and press Calibrate (a target sweep of the
  set, kept in memory), then Capture (write the tones and stream a few
  seconds into memory). Pick a tone to see its timestream as the
  resonance frequency shift x (``xcal.tangent.loop_tangent_shift``), as I
  and Q, or along and across the IQ loop, with its calibration loop and
  the stream in the IQ plane. Nothing is saved.
- Run: enter a name (and notes), a decimation stage, and a length, and
  run: every tone set gets a new calibration sweep and its stream, saved
  as the next run ``ts{n}`` at the store root
  (``crs.procedures.stream_tone_sets``). Presets fill in the
  usual runs (e.g. a long dark stream at stage 6 and a short one at stage
  0). Repeat for the next timestream.

The CRS lives on a background event loop (``crs.quick_check.AsyncWorker``)
so the window stays responsive. Streaming at another decimation stage than
the board's needs a session with global control, and other sessions can't
sweep until the stream ends and the board is back at stage 6.
"""
import numpy as np
import pyqtgraph as pg
import zarr
from pyqtgraph.Qt import QtCore, QtWidgets
from .. import daq_layout as dl
from ..qt_compat import (
    delete_on_close, fit_window_to_screen, get_qapp, widget_font_stylesheet
)
from ..xcal.tangent import loop_tangent_shift
from .procedures import (
    plan_stream_chunks, stitched_target_sweep, stream_tone_sets,
    write_stream_chunk
)
from .quick_check import AsyncWorker

# Preview display modes
MODES = ('Frequency shift x', 'Along / across loop (Hz)', 'I and Q')
# Most stream samples drawn in the IQ plane
MAX_IQ_POINTS = 5000

class StreamWindow(QtWidgets.QMainWindow):
    """
    Window for previewing and taking a tuning's timestreams.

    Attributes:
    crs (CRS or None): the connected instrument (None until connected).
    tune (zarr.Group): the tuning.
    tones (dict): the tuning's tones (``daq_layout.load_tones``).
    cal (dict or None): the preview calibration of one tone set: ``set``,
        ``rows`` (tone indices), ``f``, ``z`` (fine sweeps), ``fres``.
    preview (dict or None): the last capture: ``rows`` (tone indices),
        ``fres`` (written frequencies), ``zt``, ``dt``.
    runs (list of list of str): stream paths of each run of this session.
    """
    _message = QtCore.Signal(str)

    def __init__(
        self, connect, tune, preview_dec = 6, preview_s = 5., presets = (),
        sweep_kwargs = None, title = 'Timestreams', ui_scale = 1.0
    ):
        """
        Build the window and start connecting.

        Parameters:
        connect (callable): coroutine function with no arguments returning
            an attached CRS (run on the background loop).
        tune (zarr.Group): the tuning, writable, with ``tones`` and
            ``crs_config``.
        preview_dec (int): preview decimation stage. Default 6.
        preview_s (float): preview length in seconds. Default 5.
        presets (sequence of dict): runs offered in the preset list, each
            with ``name``, ``dec_stage``, ``duration_s`` (and optionally
            ``notes`` and ``tone_sets``, the tone sets to run; all if not
            given). Default none.
        sweep_kwargs (dict or None): calibration sweep settings
            (``target_sweep``). Default None (its defaults).
        title (str): window title. Default 'Timestreams'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(f'{title}: {tune.basename}')
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.tune = tune
        self.tones = dl.load_tones(tune['tones'])
        if 'tone_module' not in self.tones:
            raise ValueError(f'{tune.basename}/tones has no tone_module; '
                             'make the tuning with 03_tune.py')
        self.ncos = tune['crs_config'].attrs['config']['tone_sets']
        self.sweep_kwargs = dict(sweep_kwargs or {})
        self.presets = [dict(p) for p in presets]
        self.crs, self.cal, self.preview, self.runs = None, None, None, []
        self.worker = AsyncWorker(self)
        self._message.connect(self._set_status)
        self._build_panel(preview_dec, preview_s)
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        self._update_buttons()
        self._set_status('Connecting to the CRS...')
        self.worker.submit(connect(), self._on_connected)

    ############################################################################
    def _build_panel(self, preview_dec, preview_s):
        """
        Build the settings panel.

        Parameters:
        preview_dec (int): starting preview decimation stage.
        preview_s (float): starting preview length in seconds.
        """
        def dec_box(value):
            """
            Make a decimation stage box.

            Parameters:
            value (int): starting stage.

            Returns:
            box (QSpinBox): the box.
            """
            box = QtWidgets.QSpinBox()
            box.setRange(0, 6)
            box.setValue(int(value))
            return box

        def secs_box(value):
            """
            Make a length box in seconds.

            Parameters:
            value (float): starting length.

            Returns:
            box (QDoubleSpinBox): the box.
            """
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(0.1, 1e6)
            box.setDecimals(1)
            box.setSuffix(' s')
            box.setValue(float(value))
            return box

        self._panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self._panel)
        n_sets = len(set(self.tones['tone_set'].tolist()))
        outer.addWidget(QtWidgets.QLabel(
            f"{self.tune.basename}: {len(self.tones['fres'])} tones in "
            f'{n_sets} tone set(s)'))

        box = QtWidgets.QGroupBox('Preview (not saved)')
        form = QtWidgets.QFormLayout(box)
        self.set_box = QtWidgets.QComboBox()
        for s in sorted(set(self.tones['tone_set'].tolist())):
            n = int(np.sum(self.tones['tone_set'] == s))
            self.set_box.addItem(f'Tone set {s} ({n} tones)', s)
        self.set_box.currentIndexChanged.connect(self._on_set)
        self.preview_dec = dec_box(preview_dec)
        self.preview_s = secs_box(preview_s)
        self.cal_btn = QtWidgets.QPushButton('Calibrate')
        self.cal_btn.clicked.connect(self.calibrate)
        self.capture_btn = QtWidgets.QPushButton('Capture')
        self.capture_btn.clicked.connect(self.capture)
        self.tone_box = QtWidgets.QComboBox()
        self.tone_box.currentIndexChanged.connect(self._on_tone)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.currentIndexChanged.connect(self._plot)
        form.addRow('Tone set', self.set_box)
        form.addRow(self.cal_btn)
        form.addRow('Stage', self.preview_dec)
        form.addRow('Length', self.preview_s)
        form.addRow(self.capture_btn)
        form.addRow('Tone', self.tone_box)
        form.addRow('Show', self.mode_box)
        outer.addWidget(box)

        box = QtWidgets.QGroupBox('Run on all tone sets (saved)')
        form = QtWidgets.QFormLayout(box)
        self.preset_box = QtWidgets.QComboBox()
        self.preset_box.addItem('(custom)')
        for p in self.presets:
            self.preset_box.addItem(
                f"{p['name']}: stage {p['dec_stage']}, {p['duration_s']:g} s")
        self.preset_box.currentIndexChanged.connect(self._on_preset)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.textChanged.connect(self._update_buttons)
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setFixedHeight(60)
        self.run_dec = dec_box(6)
        self.run_s = secs_box(60.)
        # Tone sets (NCO settings) of the run: all by default; fewer saves
        # time, e.g. a long run on one tone set
        self.run_sets = QtWidgets.QListWidget()
        for s in sorted(set(self.tones['tone_set'].tolist())):
            item = QtWidgets.QListWidgetItem(f'Tone set {s}')
            item.setData(QtCore.Qt.UserRole, int(s))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.run_sets.addItem(item)
        self.run_sets.setFixedHeight(min(
            100, 22 * self.run_sets.count() + 6))
        self.run_sets.itemChanged.connect(self._update_buttons)
        self.run_btn = QtWidgets.QPushButton('Run')
        self.run_btn.clicked.connect(self.run)
        form.addRow('Preset', self.preset_box)
        form.addRow('Name', self.name_edit)
        form.addRow('Notes', self.notes_edit)
        form.addRow('Stage', self.run_dec)
        form.addRow('Length', self.run_s)
        form.addRow('Tone sets', self.run_sets)
        form.addRow(self.run_btn)
        outer.addWidget(box)

        self.runs_list = QtWidgets.QListWidget()
        for r in self.tuning_runs():
            self.runs_list.addItem(self._run_label(r))
        outer.addWidget(QtWidgets.QLabel('Runs of this tuning'))
        outer.addWidget(self.runs_list)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self._inputs = (self.set_box, self.preview_dec, self.preview_s,
                        self.preset_box, self.name_edit, self.notes_edit,
                        self.run_dec, self.run_s, self.run_sets)
        self._panel.setFixedWidth(self._panel.sizeHint().width() + 40)
        if self.presets:
            self.preset_box.setCurrentIndex(1)

    def _build_plots(self):
        """
        Build the timestream and IQ plots.
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.ts_plot = self.plots.addPlot(row = 0, col = 0)
        self.ts_plot.setLabel('bottom', 'Time (s)')
        self.ts_plot.addLegend()
        self.iq_plot = self.plots.addPlot(row = 1, col = 0)
        self.iq_plot.setAspectLocked(True)
        self.iq_plot.setLabel('bottom', 'I')
        self.iq_plot.setLabel('left', 'Q')

    ############################################################################
    @staticmethod
    def _run_label(run):
        """
        Describe a run for the runs list.

        Parameters:
        run (dict): a run (``daq_layout.stream_runs``).

        Returns:
        label (str): one line.
        """
        return (f"{run['group']}: {run.get('name')}, stage "
                f"{run.get('dec_stage')}, {run.get('duration_s', 0):g} s, "
                f"sets {run.get('tone_sets')}, {len(run['paths'])} "
                f"stream(s), {run.get('created', '')}")

    def tuning_runs(self):
        """
        Return this tuning's timestream runs (``ts{n}`` at the store root).

        Returns:
        runs (list of dict): ``daq_layout.stream_runs`` of the tuning.
        """
        return dl.stream_runs(dl.store_root(self.tune), self.tune.basename)

    def _set_status(self, text):
        """
        Show a status message.

        Parameters:
        text (str): the message.
        """
        self.status.setText(text)

    def _update_buttons(self):
        """
        Enable the buttons that can be used now.
        """
        idle = self.crs is not None and not self.worker.busy
        for w in self._inputs:
            w.setEnabled(not self.worker.busy)
        self.cal_btn.setEnabled(idle)
        self.capture_btn.setEnabled(
            idle and self.cal is not None and
            self.cal['set'] == self.set_box.currentData())
        self.run_btn.setEnabled(idle and bool(self.name_edit.text().strip())
                                and bool(self.run_set_idxs()))

    def _set_rows(self):
        """
        Return the tone indices of the selected tone set.

        Returns:
        rows (np.array): indices into the tuning's tones.
        """
        return np.flatnonzero(self.tones['tone_set'] ==
                              self.set_box.currentData())

    def _on_set(self):
        """
        List the selected tone set's tones.
        """
        self.tone_box.blockSignals(True)
        self.tone_box.clear()
        for r in self._set_rows():
            self.tone_box.addItem(
                f"res {self.tones['res_idxs'][r]}: "
                f"{self.tones['fres'][r] / 1e6:.4f} MHz", int(r))
        self.tone_box.blockSignals(False)
        self._update_buttons()
        self._plot()

    def _on_tone(self):
        """
        Show the selected tone, or say it needs a capture.
        """
        r = self.tone_box.currentData()
        if self.preview is not None and r is not None and \
                r not in self.preview['rows']:
            self._set_status('This tone was not in the last capture (at '
                             'fast stages, a set streams in chunks): press '
                             'Capture.')
        self._plot()

    def _on_preset(self):
        """
        Fill in the run settings from the chosen preset.
        """
        i = self.preset_box.currentIndex() - 1
        if i < 0:
            return
        p = self.presets[i]
        self.name_edit.setText(p['name'])
        self.notes_edit.setPlainText(p.get('notes', ''))
        self.run_dec.setValue(int(p['dec_stage']))
        self.run_s.setValue(float(p['duration_s']))
        self.set_run_sets(p.get('tone_sets'))

    def run_set_idxs(self):
        """
        Return the tone sets checked for the run.

        Returns:
        sets (list of int): the checked tone sets.
        """
        return [self.run_sets.item(i).data(QtCore.Qt.UserRole)
                for i in range(self.run_sets.count())
                if self.run_sets.item(i).checkState() == QtCore.Qt.Checked]

    def set_run_sets(self, sets = None):
        """
        Check the run's tone sets.

        Parameters:
        sets (list of int or None): tone sets to check. None (default): all.
        """
        for i in range(self.run_sets.count()):
            item = self.run_sets.item(i)
            on = sets is None or item.data(QtCore.Qt.UserRole) in sets
            item.setCheckState(QtCore.Qt.Checked if on
                               else QtCore.Qt.Unchecked)

    def _submit(self, coro, callback, message):
        """
        Run a job on the background loop.

        Parameters:
        coro (coroutine): the job.
        callback (callable): ``callback(result, error)`` on the GUI thread.
        message (str): status message while it runs.
        """
        self.worker.submit(coro, callback)
        self._set_status(message)
        self._update_buttons()

    ############################################################################
    def _on_connected(self, crs, error):
        """
        Keep the CRS (GUI thread).

        Parameters:
        crs (CRS or None): the attached instrument.
        error (BaseException or None): the connection error.
        """
        if error is not None:
            self._set_status(f'Could not connect: {error!r}')
            self._update_buttons()
            return
        try:
            dl.check_tune_config(self.tune, crs)
        except RuntimeError as e:
            self._set_status(f'The board does not match the tuning: {e}')
            self._update_buttons()
            return
        self.crs = crs
        self._on_set()
        self._set_status('Connected. Calibrate a tone set to preview it.')

    def calibrate(self):
        """
        Take a calibration sweep of the selected tone set, in memory.
        """
        s, rows, t = self.set_box.currentData(), self._set_rows(), self.tones
        crs, ncos, kwargs = self.crs, self.ncos, self.sweep_kwargs

        async def job():
            """
            Sweep the set into a memory group.

            Returns:
            cal (dict): the calibration.
            """
            grp = zarr.open_group(zarr.storage.MemoryStore(), mode = 'w')
            await stitched_target_sweep(
                crs, t['fres'][rows], t['ares'][rows], t['qres'][rows],
                t['res_idxs'][rows], t['tone_set'][rows],
                t['tone_module'][rows], ncos, grp, verbose = False, **kwargs)
            return dict(set = s, rows = rows, f = grp['fine_sweep/f'][:],
                        z = grp['fine_sweep/z'][:], fres = grp['fres'][:])

        self.preview = None
        self._submit(job(), self._on_calibrated,
                     f'Calibrating tone set {s} ({len(rows)} tones)...')

    def _on_calibrated(self, cal, error):
        """
        Keep the calibration (GUI thread).

        Parameters:
        cal (dict or None): the calibration.
        error (BaseException or None): the sweep error.
        """
        if error is None:
            self.cal = cal
            self._set_status(f"Tone set {cal['set']} calibrated. Capture "
                             'to preview it.')
        else:
            self._set_status(f'Calibration failed: {error!r}')
        self._update_buttons()
        self._plot()

    def capture(self):
        """
        Write the tones (the chunk holding the selected tone) and stream a
        preview into memory.
        """
        cal, t, crs = self.cal, self.tones, self.crs
        dec, secs = self.preview_dec.value(), self.preview_s.value()
        modules = t['tone_module'][cal['rows']]
        try:
            chunks = plan_stream_chunks(modules, dec)
        except ValueError as e:
            self._set_status(str(e))
            return
        r = self.tone_box.currentData()
        pos = int(np.flatnonzero(cal['rows'] == r)[0]) if r is not None \
            else 0
        chunk = next(c for c in chunks if pos in c)
        ncos = self.ncos[cal['set']]

        async def job():
            """
            Write the chunk and stream it into memory.

            Returns:
            preview (dict): the capture.
            """
            fres = await write_stream_chunk(
                crs, cal['fres'][chunk], t['ares'][cal['rows'][chunk]],
                modules[chunk], ncos)
            try:
                zt = await crs.stream_to_memory(secs, dec, verbose = False)
            finally:
                await crs.clear_tones()
            return dict(rows = cal['rows'][chunk], pos = chunk, fres = fres,
                        zt = zt, dt = 1 / crs.sample_freq)

        self._submit(job(), self._on_captured,
                     f'Streaming {len(chunk)} tones for {secs:g} s at stage '
                     f'{dec}...')

    def _on_captured(self, preview, error):
        """
        Show a capture (GUI thread).

        Parameters:
        preview (dict or None): the capture.
        error (BaseException or None): the stream error.
        """
        if error is None:
            self.preview = preview
            self._set_status(f"Captured {preview['zt'].shape[1]:,} samples "
                             f"of {len(preview['rows'])} tones.")
        else:
            self._set_status(f'Capture failed: {error!r}')
        self._update_buttons()
        self._plot()

    def run(self):
        """
        Take the named timestream of every tone set (saved).
        """
        name = self.name_edit.text().strip()
        notes = self.notes_edit.toPlainText()
        dec, secs = self.run_dec.value(), self.run_s.value()
        coro = stream_tone_sets(
            self.crs, self.tune, name, secs, dec, notes = notes,
            set_idxs = self.run_set_idxs(),
            status = self._message.emit, verbose = False,
            **self.sweep_kwargs)
        self._submit(coro, self._on_run, f'Running {name}...')

    def _on_run(self, paths, error):
        """
        List a finished run (GUI thread).

        Parameters:
        paths (list of str or None): the streams.
        error (BaseException or None): the run error.
        """
        if error is None:
            self.runs.append(paths)
            run = self.tuning_runs()[-1]
            self.runs_list.addItem(self._run_label(run))
            self._set_status(f"{run['name']}: saved {len(paths)} "
                             f"stream(s) as {run['group']}. Ready for the "
                             'next one.')
        else:
            self._set_status(f'Run failed: {error!r}')
        self._update_buttons()

    ############################################################################
    def _plot(self):
        """
        Plot the selected tone's preview and calibration loop.
        """
        self.ts_plot.clear()
        self.iq_plot.clear()
        if self.ts_plot.legend is not None:
            self.ts_plot.legend.clear()
        r = self.tone_box.currentData()
        if self.cal is None or r is None or r not in self.cal['rows']:
            return
        c = int(np.flatnonzero(self.cal['rows'] == r)[0])
        f, z = self.cal['f'][c], self.cal['z'][c]
        self.iq_plot.plot(z.real, z.imag, pen = pg.mkPen(width = 1))
        p = self.preview
        if p is None or r not in p['rows']:
            return
        k = int(np.flatnonzero(p['rows'] == r)[0])
        zt, f_tone = p['zt'][k], float(p['fres'][k])
        step = max(1, len(zt) // MAX_IQ_POINTS)
        self.iq_plot.plot(zt.real[::step], zt.imag[::step], pen = None,
                          symbol = 'o', symbolSize = 3,
                          symbolBrush = pg.mkBrush(255, 120, 60, 120))
        t = np.arange(len(zt)) * p['dt']
        mode = self.mode_box.currentText()
        if mode == MODES[2]:
            self.ts_plot.plot(t, zt.real, pen = 'c', name = 'I')
            self.ts_plot.plot(t, zt.imag, pen = 'm', name = 'Q')
            self.ts_plot.setLabel('left', 'S21')
            return
        dfr, dperp, z0, _ = loop_tangent_shift(f, z, f_tone, zt)
        self.iq_plot.plot([z0.real], [z0.imag], pen = None, symbol = '+',
                          symbolSize = 14, symbolPen = 'w')
        if mode == MODES[0]:
            self.ts_plot.plot(t, dfr / f_tone, pen = 'c', name = 'x')
            self.ts_plot.setLabel('left', 'x = dfr / fr')
        else:
            self.ts_plot.plot(t, dfr, pen = 'c', name = 'along (dfr)')
            self.ts_plot.plot(t, dperp, pen = 'm', name = 'across')
            self.ts_plot.setLabel('left', 'Hz')

    def closeEvent(self, event):
        """
        Ask before closing during a job; then stop the background loop.

        Parameters:
        event (QCloseEvent): the close event.
        """
        if self.worker.busy:
            answer = QtWidgets.QMessageBox.question(
                self, 'Busy', 'A sweep or stream is running. Stop it and '
                'close? (A stopped run keeps the streams already saved.)')
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self.worker.stop()
        event.accept()

def run_stream_gui(
    connect, tune, preview_dec = 6, preview_s = 5., presets = (),
    sweep_kwargs = None, title = 'Timestreams', ui_scale = 1.0
):
    """
    Open the timestream window on a tuning and wait until it is closed.

    Parameters:
    connect (callable): coroutine function with no arguments returning an
        attached CRS (with global control to stream at another decimation
        stage than the board's). It runs on the window's background event
        loop, which every sweep and stream then uses.
    tune (zarr.Group): the tuning, writable.
    preview_dec, preview_s, presets, sweep_kwargs, title, ui_scale: see
        ``StreamWindow``.

    Returns:
    runs (list of list of str): stream paths of each run of the session.
    """
    app = get_qapp(title)
    win = StreamWindow(connect, tune, preview_dec = preview_dec,
                       preview_s = preview_s, presets = presets,
                       sweep_kwargs = sweep_kwargs, title = title,
                       ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.runs
