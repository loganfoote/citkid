"""
Interactive timestreams of a tuning: calibrate its tone sets once, preview
them, then take named timestreams without sweeping or writing tones again.

``run_stream_gui`` opens a window on a tuning (``tune{k}`` with ``tones``,
from ``procedures/03_tune.py``):

- Calibration: pick a tone set and the sweep settings (fine and gain
  points, samples per point, gain span), add notes if you like (saved
  with the sweep; the box is emptied once it is taken), and press
  Calibrate: a target sweep of the set, saved as the tuning's next
  ``cal{j}`` (``crs.procedures.calibrate_tone_set``). A progress bar
  follows every
  sweep and stream (streams by their elapsed time), with the estimated
  time left. The
  calibration list shows the set's saved sweeps, the latest usable one
  chosen; pick another to use it instead. A calibration takes long, so
  one is taken per set and every preview and run after that uses it,
  for as long as it is in frame: setting an NCO (another tone set)
  starts a new phase frame, after which the set has to be calibrated
  again (``crs.procedures.calibration_in_frame``; stale ones are marked
  in the list and not used). A restarted session takes over the NCO
  settings the board still has (``CRS.recover_nco_state``), so its
  calibrations stay usable. So work one tone set at a time: calibrate
  it, then write tones and stream as often as needed.
- Preview: Capture writes the chosen calibration's tones (the chunk
  holding the selected tone) and streams a few seconds into memory. Pick
  a tone to see its timestream as the resonance frequency shift x
  (``xcal.tangent.loop_tangent_shift``), as I and Q, or along and across
  the IQ loop, with its calibration loop and the stream in the IQ plane.
  R fits the plots to their data again. Nothing is saved.
- Run: enter a name (and notes), a decimation stage, and a length, and
  run: the selected tone set gets its tones written (unless they are on
  the board) and its stream, saved under its chosen calibration and the
  tone write (``tune{k}/cal{j}/write{w}/ts{n}``), and the run is added
  to the tuning's run list (``crs.procedures.stream_tone_sets``).
  Presets fill in the usual runs.
  Repeat for the next timestream. For another tone set, select it and
  calibrate it (its NCO is set, so the sets calibrated before go stale).
- Runs of this tuning: the run list. View run (or a double-click) reads
  the selected run's first zarr chunks (as many as the box says) from
  the store and shows them in the plots like a capture, with the
  calibration loop the run was taken on (``load_run_view``), until the
  next capture or view. A progress bar below the list follows the
  sweeps and streams.

Tones stay on the board: whatever wrote them last (Capture, Write tones,
or a run), the next capture or run of the same tones from the same
calibration streams without writing them again, which the window reports
("Tones on the board"). Write tones writes the selected set for the run's
stage on purpose (when it streams in one chunk). A different set, chunk
or calibration, or a new calibration sweep (which clears the board),
means a write.

The CRS lives on a background event loop (``crs.quick_check.AsyncWorker``)
so the window stays responsive. Streaming at another decimation stage than
the board's needs a session with global control, and other sessions can't
sweep until the stream ends and the board is back at stage 6.
"""
import inspect
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from .. import daq_layout as dl
from ..qt_compat import (
    delete_on_close, fit_window_to_screen, get_qapp, widget_font_stylesheet
)
from ..xcal.tangent import loop_tangent_shift
from .procedures import (
    calibrate_tone_set, calibration_in_frame, plan_stream_chunks,
    stream_tone_sets, target_sweep, tones_written, write_set_tones
)
from .quick_check import AsyncWorker, ProgressClock
from .util import get_sample_freq

# Preview display modes
MODES = ('Frequency shift x', 'Along / across loop (Hz)', 'I and Q')
# Most stream samples drawn in the IQ plane
MAX_IQ_POINTS = 5000
# Calibration sweep settings editable in the window: target_sweep name,
# label, (min, max), and whether it is a whole number
CAL_SETTINGS = (
    ('npoints_fine', 'Fine points', (10, 100000), True),
    ('npoints_gain', 'Gain points', (2, 10000), True),
    ('nsamps', 'Samples / point', (1, 1000000), True),
    ('gain_span_factor', 'Gain span factor', (1.1, 1000.), False),
)

def load_run_view(root, run, set_rows, chunks = 1, tone = None):
    """
    Read the first chunks of a saved run's stream, with the calibration
    loop of its tones, in the shape of a capture (``StreamWindow``'s
    ``preview``) plus the loop.

    Parameters:
    root (zarr.Group): the store root.
    run (dict): the run (``daq_layout.stream_runs``: ``paths``,
        ``calibration``, ``chunk_samples``, ``tone_set``, ``group``,
        ``name``).
    set_rows (np.array): the tuning's tone indices of the run's tone set,
        in the calibration's order.
    chunks (int): zarr chunks along time to read. Default 1.
    tone (int or None): a tone index of the tuning: the stream holding it
        is read (a set streams in several writes at fast stages). None
        (default), or a tone in none of them: the first stream.

    Returns:
    view (dict): ``rows`` (tone indices of the tuning), ``fres`` (the
        tones' written frequencies), ``zt`` (complex64 S21 timestreams,
        (ntones, nsamples)), ``dt``, ``f``, ``z`` (the calibration sweep
        of those tones), ``tone_set``, ``calibration``, ``source`` (the
        stream's path), ``label``, ``chunks`` (read, of ``n_chunks``).

    Raises:
    KeyError: if the run has no streams on a calibration of the store.
    """
    paths = list(run.get('paths') or [])
    if not paths or not run.get('calibration'):
        raise KeyError(f"{run.get('group')} has no streams on a calibration")
    cal = root[run['calibration']]
    chosen = None
    for path in paths:
        write = root[path.rsplit('/', 1)[0]]
        rows = set_rows[write['sweep_rows'][:]]
        if chosen is None or (tone is not None and tone in rows):
            chosen = (path, write, rows)
    path, write, rows = chosen
    stream = root[path]
    n = int(run['chunk_samples']) * max(int(chunks), 1)
    z = stream['z'][:, :, :n]
    scale = np.asarray(stream['counts_to_s21'][...])
    zt = ((z[0] + 1j * z[1]) * scale[:, None]).astype(np.complex64)
    sweep_rows = write['sweep_rows'][:]
    return dict(
        rows = rows, fres = write['fres'][:], zt = zt,
        dt = float(np.asarray(stream['dt'][...])),
        f = cal['fine_sweep/f'][:][sweep_rows],
        z = cal['fine_sweep/z'][:][sweep_rows],
        tone_set = int(run.get('tone_set')), calibration = cal.path,
        source = path, label = f"{run['group']} ({run.get('name')})",
        chunks = (min(int(chunks), int(run.get('n_chunks', 1))),
                  int(run.get('n_chunks', 1))))

class StreamWindow(QtWidgets.QMainWindow):
    """
    Window for calibrating, previewing, and taking a tuning's timestreams.

    Attributes:
    crs (CRS or None): the connected instrument (None until connected).
    tune (zarr.Group): the tuning.
    tones (dict): the tuning's tones (``daq_layout.load_tones``).
    cals (list of dict): the tuning's saved calibrations
        (``daq_layout.calibrations``).
    cal (dict or None): the chosen calibration of the selected tone set:
        ``name``, ``path``, ``set``, ``rows`` (tone indices), ``f``, ``z``
        (fine sweeps), ``fres``.
    chosen (dict): per tone set, the calibration name chosen for it.
    written (dict): the board's tones (``crs.procedures.write_set_tones``).
    preview (dict or None): the last capture: ``rows`` (tone indices),
        ``fres`` (written frequencies), ``zt``, ``dt``.
    runs (list of list of str): stream paths of each run of this session.
    """
    _message = QtCore.Signal(str)
    # Sweep progress (points done, total), emitted from the background
    # thread and delivered on the GUI thread
    _progress = QtCore.Signal(int, int)

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
            with ``name``, ``dec_stage``, ``duration_s``, and optionally
            ``notes``. Default none.
        sweep_kwargs (dict or None): starting calibration sweep settings
            (``target_sweep``); those in ``CAL_SETTINGS`` can be changed
            in the window. Default None (its defaults).
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
        self.sweep_kwargs = dict(sweep_kwargs or {})
        self.presets = [dict(p) for p in presets]
        self.cals = dl.calibrations(tune)
        self.crs, self.cal, self.preview, self.runs = None, None, None, []
        self.view = None            # a saved run shown instead of a capture
        self.chosen, self.written = {}, {}
        self.worker = AsyncWorker(self)
        self._clock = ProgressClock()
        self._progress_on = False
        self._message.connect(self._set_status)
        self._progress.connect(self._on_progress)
        self._build_panel(preview_dec, preview_s)
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        self._on_set()
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
        sets = sorted(set(self.tones['tone_set'].tolist()))
        outer.addWidget(QtWidgets.QLabel(
            f"{self.tune.basename}: {len(self.tones['fres'])} tones in "
            f'{len(sets)} tone set(s)'))

        # Calibration
        box = QtWidgets.QGroupBox('Calibration (saved as cal{j})')
        form = QtWidgets.QFormLayout(box)
        self.set_box = QtWidgets.QComboBox()
        for s in sets:
            n = int(np.sum(self.tones['tone_set'] == s))
            self.set_box.addItem(f'Tone set {s} ({n} tones)', s)
        self.set_box.currentIndexChanged.connect(self._on_set)
        form.addRow('Tone set', self.set_box)
        # Settings not given start from target_sweep's defaults
        defaults = inspect.signature(target_sweep).parameters
        self.cal_params = {}
        for key, label, (lo, hi), whole in CAL_SETTINGS:
            if whole:
                spin = QtWidgets.QSpinBox()
                spin.setRange(int(lo), int(hi))
            else:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(float(lo), float(hi))
                spin.setDecimals(1)
            value = self.sweep_kwargs.get(key, defaults[key].default)
            spin.setValue(value)
            self.cal_params[key] = spin
            form.addRow(label, spin)
        self.cal_notes_edit = QtWidgets.QPlainTextEdit()
        self.cal_notes_edit.setFixedHeight(60)
        self.cal_notes_edit.setPlaceholderText(
            'Notes saved with the next calibration (optional)')
        form.addRow('Notes', self.cal_notes_edit)
        self.cal_btn = QtWidgets.QPushButton('Calibrate')
        self.cal_btn.clicked.connect(self.calibrate)
        form.addRow(self.cal_btn)
        self.cal_box = QtWidgets.QComboBox()
        self.cal_box.currentIndexChanged.connect(self._on_cal)
        form.addRow('Use', self.cal_box)
        outer.addWidget(box)

        # Preview
        box = QtWidgets.QGroupBox('Preview (not saved)')
        form = QtWidgets.QFormLayout(box)
        self.preview_dec = dec_box(preview_dec)
        self.preview_s = secs_box(preview_s)
        self.capture_btn = QtWidgets.QPushButton('Capture')
        self.capture_btn.clicked.connect(self.capture)
        self.tone_box = QtWidgets.QComboBox()
        self.tone_box.currentIndexChanged.connect(self._on_tone)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.currentIndexChanged.connect(self._plot)
        form.addRow('Stage', self.preview_dec)
        form.addRow('Length', self.preview_s)
        form.addRow(self.capture_btn)
        form.addRow('Tone', self.tone_box)
        form.addRow('Show', self.mode_box)
        outer.addWidget(box)

        # Run
        box = QtWidgets.QGroupBox('Run on the selected tone set (saved as '
                                  'ts{n})')
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
        self.run_dec.valueChanged.connect(self._update_buttons)
        self.run_s = secs_box(60.)
        self.write_btn = QtWidgets.QPushButton('Write tones')
        self.write_btn.clicked.connect(self.write_tones)
        self.run_btn = QtWidgets.QPushButton('Run')
        self.run_btn.clicked.connect(self.run)
        self.written_label = QtWidgets.QLabel()
        self.written_label.setWordWrap(True)
        form.addRow('Preset', self.preset_box)
        form.addRow('Name', self.name_edit)
        form.addRow('Notes', self.notes_edit)
        form.addRow('Stage', self.run_dec)
        form.addRow('Length', self.run_s)
        form.addRow(self.write_btn)
        form.addRow(self.run_btn)
        form.addRow(self.written_label)
        outer.addWidget(box)

        self.runs_list = QtWidgets.QListWidget()
        for r in self.tuning_runs():
            self.runs_list.addItem(self._run_label(r))
        self.runs_list.currentRowChanged.connect(self._on_run_selected)
        self.runs_list.itemDoubleClicked.connect(lambda _: self.view_run())
        outer.addWidget(QtWidgets.QLabel('Runs of this tuning'))
        outer.addWidget(self.runs_list)
        # View a saved run in the plots: its first chunks
        row = QtWidgets.QHBoxLayout()
        self.view_btn = QtWidgets.QPushButton('View run')
        self.view_btn.setToolTip('Show the selected run\'s saved data in '
                                 'the plots (its first chunks), like a '
                                 'capture')
        self.view_btn.clicked.connect(self.view_run)
        self.view_chunks = QtWidgets.QSpinBox()
        self.view_chunks.setRange(1, 1)
        self.view_chunks.setToolTip('zarr chunks along time to load')
        row.addWidget(self.view_btn)
        row.addWidget(QtWidgets.QLabel('chunks'))
        row.addWidget(self.view_chunks)
        self.view_label = QtWidgets.QLabel()
        row.addWidget(self.view_label, stretch = 1)
        outer.addLayout(row)
        # The progress bar has a place of its own, kept while idle so that
        # nothing moves when a job starts
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self._show_progress(False)
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self._inputs = (self.set_box, self.cal_box, self.cal_notes_edit,
                        self.preview_dec, self.preview_s, self.preset_box,
                        self.name_edit, self.notes_edit, self.run_dec,
                        self.run_s, self.view_chunks) + \
            tuple(self.cal_params.values())
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
                f"set {run.get('tone_set', run.get('tone_sets'))}, "
                f"{run.get('calibration', '')}, {len(run['paths'])} "
                f"stream(s), {run.get('created', '')}")

    @staticmethod
    def _cal_label(cal):
        """
        Describe a calibration for the calibration list.

        Parameters:
        cal (dict): a calibration (``daq_layout.calibrations``).

        Returns:
        label (str): one line.
        """
        notes = ' '.join(cal.get('notes', '').split())
        return (f"{cal['name']}: {cal['ntones']} tones, "
                f"{cal['npoints_fine']} fine points x {cal['nsamps']} "
                f"samples, {cal.get('created', '')}"
                + ('' if cal['complete'] else ', incomplete')
                + ('' if cal.get('in_frame') else ', stale (NCO set since)')
                + (f' - {notes}' if notes else ''))

    def tuning_runs(self):
        """
        Return this tuning's timestream runs (its run list's ``ts{n}``).

        Returns:
        runs (list of dict): ``daq_layout.stream_runs`` of the tuning.
        """
        return dl.stream_runs(dl.store_root(self.tune), self.tune.basename)

    def cal_kwargs(self):
        """
        Return the calibration sweep settings: the starting ones with the
        window's values.

        Returns:
        kwargs (dict): ``target_sweep`` keyword arguments.
        """
        kw = dict(self.sweep_kwargs)
        for key, spin in self.cal_params.items():
            kw[key] = spin.value()
        return kw

    def _set_status(self, text):
        """
        Show a status message.

        Parameters:
        text (str): the message.
        """
        self.status.setText(text)

    def _show_progress(self, show):
        """
        Start the progress bar (indeterminate, until the first report) or
        put it to rest (empty; it keeps its place).

        Parameters:
        show (bool): whether a job with progress reports is running.
        """
        self._progress_on = bool(show)
        self.progress.setRange(0, 0 if show else 1)
        if not show:
            self.progress.setValue(0)
        self.progress.setFormat('')
        self._clock.reset()

    def _on_progress(self, done, total):
        """
        Update the progress bar (GUI thread).

        Parameters:
        done (int): points taken so far.
        total (int): points in the sweep.
        """
        if not self._progress_on:
            return
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(min(done, total))
        self.progress.setFormat(self._clock.text(done, total))

    def _run_chunks(self):
        """
        Return the selected tone set's chunks at the run's stage.

        Returns:
        chunks (list of np.array or None): ``plan_stream_chunks``, or None
            if the set can't stream at that stage.
        """
        try:
            return plan_stream_chunks(
                self.tones['tone_module'][self._set_rows()],
                self.run_dec.value())
        except ValueError:
            return None

    def _update_buttons(self):
        """
        Enable the buttons that can be used now, and say what is on the
        board.
        """
        idle = self.crs is not None and not self.worker.busy
        for w in self._inputs:
            w.setEnabled(not self.worker.busy)
        self.cal_btn.setEnabled(idle)
        has_cal = self.cal is not None and self.cal['in_frame'] and \
            self.cal['set'] == self.set_box.currentData()
        self.capture_btn.setEnabled(idle and has_cal)
        chunks = self._run_chunks()
        self.write_btn.setEnabled(idle and has_cal and chunks is not None
                                  and len(chunks) == 1)
        self.run_btn.setEnabled(idle and has_cal and
                                bool(self.name_edit.text().strip()))
        self.view_btn.setEnabled(not self.worker.busy and
                                 self.runs_list.currentRow() >= 0)
        w = self.written
        self.written_label.setText(
            'Tones on the board: none (or unknown)' if not w else
            f"Tones on the board: {len(w['rows'])} of tone set "
            f"{w['tone_set']} from {w['calibration']}")

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
        List the selected tone set's tones and calibrations.
        """
        self.tone_box.blockSignals(True)
        self.tone_box.clear()
        for r in self._set_rows():
            self.tone_box.addItem(
                f"res {self.tones['res_idxs'][r]}: "
                f"{self.tones['fres'][r] / 1e6:.4f} MHz", int(r))
        self.tone_box.blockSignals(False)
        self._refresh_cals()

    def _refresh_cals(self, select = None):
        """
        List the selected tone set's saved calibrations and choose one.

        Parameters:
        select (str or None): calibration name to choose. None (default):
            the one chosen before for this set, else the latest.
        """
        s = self.set_box.currentData()
        self.cals = dl.calibrations(self.tune)
        for c in self.cals:
            c['in_frame'] = self.crs is not None and \
                calibration_in_frame(self.crs, self.tune[c['name']])
        of_set = [c for c in self.cals if c['tone_set'] == s]
        self.cal_box.blockSignals(True)
        self.cal_box.clear()
        for c in of_set:
            self.cal_box.addItem(self._cal_label(c), c['name'])
        names = [c['name'] for c in of_set]
        usable = [c['name'] for c in of_set if c['in_frame']]
        want = select if select in names else self.chosen.get(s)
        if want not in usable:
            want = usable[-1] if usable else (names[-1] if names else None)
        self.cal_box.setCurrentIndex(names.index(want) if want in names
                                     else -1)
        self.cal_box.blockSignals(False)
        self._on_cal()

    def _on_cal(self):
        """
        Load the chosen calibration of the selected tone set.
        """
        s, name = self.set_box.currentData(), self.cal_box.currentData()
        self.cal, self.preview = None, None
        if name is not None:
            grp = self.tune[name]
            rows = self._set_rows()
            if grp['fres'].shape[0] == len(rows) and 'fine_sweep' in grp:
                self.chosen[s] = name
                self.cal = dict(
                    name = name, path = grp.path, set = s, rows = rows,
                    f = grp['fine_sweep/f'][:], z = grp['fine_sweep/z'][:],
                    fres = grp['fres'][:],
                    in_frame = self.crs is not None and
                    calibration_in_frame(self.crs, grp))
                if not self.cal['in_frame']:
                    self._set_status(f'{name} is stale: an NCO was set since '
                                     'it was taken. Calibrate tone set '
                                     f'{s} again to capture or write it.')
            else:
                self._set_status(f'{name} does not match tone set {s}: '
                                 'take a new calibration.')
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

    def _submit(self, coro, callback, message, progress = False):
        """
        Run a job on the background loop.

        Parameters:
        coro (coroutine): the job.
        callback (callable): ``callback(result, error)`` on the GUI thread.
        message (str): status message while it runs.
        progress (bool): If True, show the progress bar until it is done.
            Default False.
        """
        def done(result, error):
            """
            Hide the progress bar, then call back.

            Parameters:
            result: the job's result.
            error (BaseException or None): its error.
            """
            self._show_progress(False)
            callback(result, error)

        self._show_progress(progress)
        self.worker.submit(coro, done)
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
        self.written.clear()
        # Whether saved calibrations are in frame depends on the board
        self._refresh_cals()
        self._set_status('Connected. Calibrate a tone set.')

    def calibrate(self):
        """
        Take a calibration sweep of the selected tone set, saved as the
        tuning's next ``cal{j}`` with the notes in the box.
        """
        s, kw = self.set_box.currentData(), self.cal_kwargs()
        crs, tune, report = self.crs, self.tune, self._progress.emit
        notes = self.cal_notes_edit.toPlainText()

        async def job():
            """
            Sweep the set.

            Returns:
            name (str): the calibration's name.
            """
            grp = await calibrate_tone_set(crs, tune, s, verbose = False,
                                           progress = report, notes = notes,
                                           **kw)
            return grp.basename

        self._submit(job(), self._on_calibrated,
                     f'Calibrating tone set {s} ({len(self._set_rows())} '
                     'tones)...', progress = True)

    def _on_calibrated(self, name, error):
        """
        List and choose the new calibration (GUI thread).

        Parameters:
        name (str or None): the calibration's name.
        error (BaseException or None): the sweep error.
        """
        # The sweep cleared the board's tones
        self.written.clear()
        if error is None:
            # The notes were for that calibration; the next starts empty
            self.cal_notes_edit.clear()
            self._refresh_cals(select = name)
            self._set_status(f"Tone set {self.cal['set']} calibrated as "
                             f'{name}. Capture to preview it.')
        else:
            self._set_status(f'Calibration failed: {error!r}')
        self._update_buttons()

    def capture(self):
        """
        Write the chosen calibration's tones (the chunk holding the
        selected tone) unless they are on the board, and stream a preview
        into memory.
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
        grp, written = self.tune[cal['name']], self.written
        reuse = tones_written(written, cal['set'], chunk, cal['path'])
        report = self._progress.emit

        async def job():
            """
            Write the chunk if needed and stream it into memory.

            Returns:
            preview (dict): the capture.
            """
            fres = await write_set_tones(crs, grp, chunk, written)
            zt = await crs.stream_to_memory(secs, dec, verbose = False,
                                            progress = report)
            # The stream's stage: the CRS may be back at stage 6 by now
            return dict(rows = cal['rows'][chunk], pos = chunk, fres = fres,
                        zt = zt, dt = 1 / get_sample_freq(dec))

        self._submit(job(), self._on_captured,
                     ('Streaming' if reuse else 'Writing and streaming')
                     + f' {len(chunk)} tones for {secs:g} s at stage '
                     f'{dec}...', progress = True)

    def _on_captured(self, preview, error):
        """
        Show a capture (GUI thread).

        Parameters:
        preview (dict or None): the capture.
        error (BaseException or None): the stream error.
        """
        if error is None:
            self.preview, self.view = preview, None
            self._set_status(f"Captured {preview['zt'].shape[1]:,} samples "
                             f"of {len(preview['rows'])} tones.")
        else:
            self._set_status(f'Capture failed: {error!r}')
        self._update_buttons()
        self._plot()

    def write_tones(self):
        """
        Write the selected tone set's tones for the run's stage (when it
        streams in one chunk), even if they are on the board already.
        """
        cal, chunks = self.cal, self._run_chunks()
        if cal is None or chunks is None or len(chunks) != 1:
            self._set_status('Write tones needs a calibration and a tone '
                             "set that streams in one chunk at the run's "
                             'stage.')
            return
        grp, written, rows = self.tune[cal['name']], self.written, chunks[0]
        crs = self.crs

        async def job():
            """
            Write the set.

            Returns:
            n (int): tones written.
            """
            written.clear()
            await write_set_tones(crs, grp, rows, written)
            return len(rows)

        self._submit(job(), self._on_written,
                     f"Writing {len(rows)} tones of tone set {cal['set']}...")

    def _on_written(self, n, error):
        """
        Report the write (GUI thread).

        Parameters:
        n (int or None): tones written.
        error (BaseException or None): the write error.
        """
        if error is None:
            self._set_status(f'{n} tones written; they stay on the board '
                             'for the next captures and runs.')
        else:
            self._set_status(f'Write failed: {error!r}')
        self._update_buttons()

    def run(self):
        """
        Take the named timestream of the selected tone set (saved), with
        its chosen calibration.
        """
        name = self.name_edit.text().strip()
        notes = self.notes_edit.toPlainText()
        dec, secs = self.run_dec.value(), self.run_s.value()
        s, cal = self.cal['set'], self.cal['path']
        coro = stream_tone_sets(
            self.crs, self.tune, name, secs, dec, notes = notes,
            set_idxs = [s], calibrations = {s: cal}, written = self.written,
            keep_tones = True, status = self._message.emit, verbose = False,
            progress = self._progress.emit, **self.cal_kwargs())
        self._submit(coro, self._on_run,
                     f'Running {name} on tone set {s} with {cal}...',
                     progress = True)

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
        self._refresh_cals(select = self.chosen.get(self.set_box.currentData()))

    def _on_run_selected(self, row):
        """
        Offer the selected run's chunks to view.

        Parameters:
        row (int): the row in the runs list.
        """
        runs = self.tuning_runs()
        run = runs[row] if 0 <= row < len(runs) else None
        n = int(run.get('n_chunks', 0)) if run else 0
        self.view_chunks.setRange(1, max(n, 1))
        self.view_chunks.setValue(1)
        self.view_label.setText(
            f"of {n} ({run['chunk_s']:.3g} s each)" if n else
            ('' if run is None else 'no streams'))
        self._update_buttons()

    def view_run(self):
        """
        Show the selected run's saved data (its first chunks) in the
        plots, with the calibration it was taken on, as a capture.
        """
        row = self.runs_list.currentRow()
        runs = self.tuning_runs()
        if not 0 <= row < len(runs) or self.worker.busy:
            return
        run = runs[row]
        if not run.get('n_chunks'):
            self._set_status(f"{run['group']} has no streams to show.")
            return
        root = dl.store_root(self.tune)
        k = int(self.view_chunks.value())
        tone = self.tone_box.currentData()
        set_rows = np.flatnonzero(self.tones['tone_set'] ==
                                  run.get('tone_set'))

        async def job():
            """
            Read the run's data.

            Returns:
            view (dict): ``load_run_view``.
            """
            return load_run_view(root, run, set_rows, k, tone)

        self._submit(job(), self._on_viewed,
                     f"Reading {k} chunk(s) of {run['group']}...")

    def _on_viewed(self, view, error):
        """
        Show a saved run (GUI thread).

        Parameters:
        view (dict or None): ``load_run_view``.
        error (BaseException or None): the read error.
        """
        if error is None:
            self.view = view
            if self.set_box.currentData() != view['tone_set']:
                self.set_box.setCurrentIndex(
                    self.set_box.findData(view['tone_set']))
            if self.tone_box.currentData() not in view['rows']:
                self.tone_box.setCurrentIndex(
                    self.tone_box.findData(int(view['rows'][0])))
            self._set_status(
                f"Showing {view['label']}: {view['zt'].shape[1]:,} samples "
                f"({view['zt'].shape[1] * view['dt']:.3g} s) of "
                f"{len(view['rows'])} tones from {view['source']}, on "
                f"{view['calibration']}. Capture or View run to replace it.")
        else:
            self._set_status(f'Could not read the run: {error!r}')
        self._update_buttons()
        self._plot()

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
        # A saved run brings its own calibration loop; a capture uses the
        # chosen calibration's
        cal = self.view if self.view is not None else self.cal
        if cal is None or r is None or r not in cal['rows']:
            return
        c = int(np.flatnonzero(cal['rows'] == r)[0])
        f, z = cal['f'][c], cal['z'][c]
        self.iq_plot.plot(z.real, z.imag, pen = pg.mkPen(width = 1))
        p = self.view if self.view is not None else self.preview
        if p is None or r not in p['rows']:
            return
        k = int(np.flatnonzero(p['rows'] == r)[0])
        zt, f_tone = p['zt'][k], float(p['fres'][k])
        dfr, dperp, z0, _ = loop_tangent_shift(f, z, f_tone, zt)
        step = max(1, len(zt) // MAX_IQ_POINTS)
        self.iq_plot.plot(zt.real[::step], zt.imag[::step], pen = None,
                          symbol = 'o', symbolSize = 3,
                          symbolBrush = pg.mkBrush(255, 120, 60, 120))
        self.iq_plot.plot([z0.real], [z0.imag], pen = None, symbol = '+',
                          symbolSize = 14, symbolPen = 'w')
        t = np.arange(len(zt)) * p['dt']
        mode = self.mode_box.currentText()
        if mode == MODES[2]:
            self.ts_plot.plot(t, zt.real, pen = 'c', name = 'I')
            self.ts_plot.plot(t, zt.imag, pen = 'm', name = 'Q')
            self.ts_plot.setLabel('left', 'S21')
        elif mode == MODES[0]:
            self.ts_plot.plot(t, dfr / f_tone, pen = 'c', name = 'x')
            self.ts_plot.setLabel('left', 'x = dfr / fr')
        else:
            self.ts_plot.plot(t, dfr, pen = 'c', name = 'along (dfr)')
            self.ts_plot.plot(t, dperp, pen = 'm', name = 'across')
            self.ts_plot.setLabel('left', 'Hz')
        self.rescale()

    def rescale(self):
        """
        Fit both plots to their data again (the R key), e.g. after zooming
        or changing what is shown.
        """
        # Enabling auto-range fits the view at once (autoRange() would
        # set a fixed range and turn auto-range off again)
        for plot in (self.ts_plot, self.iq_plot):
            plot.enableAutoRange()

    def keyPressEvent(self, event):
        """
        R: rescale the plots.

        Parameters:
        event (QKeyEvent): the key event.
        """
        if event.key() == QtCore.Qt.Key_R and \
                not isinstance(self.focusWidget(), (QtWidgets.QLineEdit,
                                                    QtWidgets.QPlainTextEdit)):
            self.rescale()
            event.accept()
            return
        super().keyPressEvent(event)

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
