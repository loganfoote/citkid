"""
Interactive timestreams of a tuning: calibrate its tone sets once, write
their tones, preview them, then take named timestreams without sweeping
or writing tones again.

``run_stream_gui`` opens a window on a tuning (``tune{k}`` with ``tones``,
from ``procedures/3_tune.py``):

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
  it, write its tones, then preview and stream as often as needed.
- Tones on the board: Write tones writes the chosen chunk of the set
  (planned for the chosen stage: a fast stage takes fewer tones, so a
  big set is written and streamed one chunk at a time) from its
  calibration; Clear tones clears the board. Nothing else writes
  tones: a preview or run streams exactly what is written and refuses
  otherwise.
- Preview: Capture streams the board's tones a few seconds into
  memory. Pick
  a tone to see its timestream as the resonance frequency shift x
  (``xcal.tangent.loop_tangent_shift``), as I and Q, or parallel and
perpendicular to
  the IQ loop, with its calibration loop and the stream in the IQ plane.
  R fits the plots to their data again. Nothing is saved.
- Run: enter a name (and notes), a decimation stage, and a length, and
  run: the board's tones are streamed and saved under their
  calibration and tone write (``tune{k}/cal{j}/write{w}/ts{n}``), and
  the run is added to the tuning's run list
  (``crs.procedures.stream_tone_sets`` with ``only_written``).
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
import threading
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from .. import daq_layout as dl
from ..qt_compat import fg
from ..qt_compat import add_help
from ..qt_compat import (
    delete_on_close, fit_window_to_screen, get_qapp, widget_font_stylesheet
)
from ..xcal.tangent import loop_tangent_shift
from .procedures import (
    SweepCancelled, calibrate_tone_set, calibration_in_frame,
    plan_stream_chunks, stream_tone_sets, target_sweep, tones_written,
    write_set_tones
)
from .quick_check import AsyncWorker, ProgressClock
from .util import StreamCancelled, get_sample_freq
from ..signal.psd import get_psd

# Preview display modes
MODES = ('Frequency shift x', 'Parallel / perpendicular to loop (Hz)',
         'I and Q')
# PSD axis label of each mode
PSD_LABELS = ('Sxx (1/Hz)', 'Spar, Sper (Hz^2/Hz)', 'SII, SQQ (1/Hz)')
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

def load_run_view(root, run, set_rows, chunks = 1, tone = None, start = 0):
    """
    Read some chunks of a saved run's stream, with the calibration
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
    start (int): the first chunk read. Default 0.

    Returns:
    view (dict): ``rows`` (tone indices of the tuning), ``fres`` (the
        tones' written frequencies), ``zt`` (complex64 S21 timestreams,
        (ntones, nsamples)), ``dt``, ``f``, ``z`` (the calibration sweep
        of those tones), ``tone_set``, ``calibration``, ``source`` (the
        stream's path), ``label``, ``chunks`` (read, of ``n_chunks``),
        ``start`` (the first chunk), ``t0`` (its time from the run's
        start, s).

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
    n_chunks = int(run.get('n_chunks', 1))
    start = min(max(int(start), 0), max(n_chunks - 1, 0))
    per = int(run['chunk_samples'])
    a = start * per
    n = per * max(int(chunks), 1)
    z = stream['z'][:, :, a:a + n]
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
        chunks = (min(int(chunks), n_chunks - start), n_chunks),
        start = start,
        t0 = a * float(np.asarray(stream['dt'][...])))

def plot_view(ts_plot, iq_plot, psd_plot, cal, p, r, mode = 0,
              offset = False, show_psd = False, label = ''):
    """
    Draw one tone of a capture or a saved run: the calibration loop with
    the timestream's points on it, the timestream in the chosen form,
    and its PSD.

    Parameters:
    ts_plot, iq_plot (PlotItem): the timestream and IQ plots.
    psd_plot (PlotWidget): the PSD plot.
    cal (dict or None): ``rows``, ``f``, ``z`` of the calibration
        sweep (``load_run_view``, or a calibration's loop). None:
        nothing is drawn.
    p (dict or None): the timestreams: ``rows``, ``zt``, ``fres``,
        ``dt`` and optionally ``t0`` (``load_run_view``, or a capture).
        None: only the loop.
    r (int or None): the tone (an index of the tuning's tones).
    mode (int): ``MODES`` index: 0 x, 1 parallel / perpendicular, 2 I
        and Q. Default 0.
    offset (bool): If True, subtract each series' mean. Default False.
    show_psd (bool): If True, draw the PSD too. Default False.
    label (str): what is shown, for the titles. Default ''.
    """
    ts_plot.clear()
    iq_plot.clear()
    psd_plot.clear()
    ts_plot.setTitle(None)
    psd_plot.setTitle(None)
    for plot in (ts_plot, psd_plot.getPlotItem()):
        if plot.legend is not None:
            plot.legend.clear()
    if cal is None or r is None or r not in cal['rows']:
        return
    c = int(np.flatnonzero(cal['rows'] == r)[0])
    f, z = cal['f'][c], cal['z'][c]
    # The sweep as points, bigger than the timestream's
    iq_plot.plot(z.real, z.imag, pen = None, symbol = 'o',
                 symbolSize = 7, symbolPen = None,
                 symbolBrush = pg.mkBrush(120, 180, 255, 200))
    if p is None or r not in p['rows']:
        return
    k = int(np.flatnonzero(p['rows'] == r)[0])
    zt, f_tone = p['zt'][k], float(p['fres'][k])
    ts_plot.setTitle(f'{label}: tone {r}')
    psd_plot.setTitle(f'PSD of {label}: tone {r}')
    dfr, dperp, z0, _ = loop_tangent_shift(f, z, f_tone, zt)
    step = max(1, len(zt) // MAX_IQ_POINTS)
    iq_plot.plot(zt.real[::step], zt.imag[::step], pen = None,
                 symbol = 'o', symbolSize = 3,
                 symbolBrush = pg.mkBrush(255, 120, 60, 120))
    iq_plot.plot([z0.real], [z0.imag], pen = None, symbol = '+',
                 symbolSize = 14, symbolPen = fg())
    t = np.arange(len(zt)) * p['dt'] + float(p.get('t0', 0.))
    if mode == 2:
        series = [('I', zt.real), ('Q', zt.imag)]
        ts_plot.setLabel('left', 'S21')
    elif mode == 0:
        series = [('x', dfr / f_tone)]
        ts_plot.setLabel('left', 'x = dfr / fr')
    else:
        series = [('parallel (dfr)', dfr), ('perpendicular', dperp)]
        ts_plot.setLabel('left', 'Hz')
    if offset:
        series = [(name, y - np.nanmean(y)) for name, y in series]
    for (name, y), pen in zip(series, ('c', 'm')):
        ts_plot.plot(t, y, pen = pen, name = name)
    if show_psd:
        psd_plot.setLabel('left', PSD_LABELS[mode])
        for (name, y), pen in zip(series, ('c', 'm')):
            f, psd = get_psd(y - np.mean(y), p['dt'], return_freqs = True)
            ok = (f > 0) & (psd > 0)
            psd_plot.plot(f[ok], psd[ok], pen = pen, name = name)
    for plot in (ts_plot, iq_plot, psd_plot):
        plot.enableAutoRange()

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
        sweep_kwargs = None, title = 'Timestreams', ui_scale = 1.0,
        fres_update = None, setup = None, extra_features = False
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
        fres_update (dict or None): move the tone frequencies to the
            resonances each calibration sweep shows, within a window
            (``procedures.apply_fres_update``: ``method``, ``max_shift``,
            ``npoints_update``).
            Default None: no.
        setup (dict or None): the readout chain (the scripts' ``SETUP``),
            saved in each run's record. Default None.
        extra_features (bool): If True, show the Adjust tones box (a new
            tuning with every power or frequency moved). Default False.
        title (str): window title. Default 'Timestreams'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self._title = title
        self.setWindowTitle(f'{title}: {dl.tuning_name(tune)} '
                            f'({tune.basename})')
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.tune = tune
        self.tones = dl.load_tones(tune['tones'])
        if 'tone_module' not in self.tones:
            raise ValueError(f'{tune.basename}/tones has no tone_module; '
                             'make the tuning with 3_tune.py')
        self.sweep_kwargs = dict(sweep_kwargs or {})
        self.fres_update = dict(fres_update) if fres_update else None
        self.setup = dict(setup) if setup else None
        self.extra_features = bool(extra_features)
        self.presets = [dict(p) for p in presets]
        self.cals = dl.calibrations(tune)
        self.crs, self.cal, self.preview, self.runs = None, None, None, []
        self.view = None            # a saved run shown instead of a capture
        self.chosen, self.written = {}, {}
        self.worker = AsyncWorker(self)
        # Set to ask the running calibration, capture or run to stop;
        # read on the background loop
        self._cancel = threading.Event()
        self._cancellable = False
        self._clock = ProgressClock()
        self._progress_on = False
        self._message.connect(self._set_status)
        self._progress.connect(self._on_progress)
        self._build_panel(preview_dec, preview_s)
        self._fill_tune_box()
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._scroll)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.plots, stretch = 2)
        right.addWidget(self.psd_plot, stretch = 1)
        lay.addLayout(right, stretch = 1)
        self.setCentralWidget(central)
        add_help(self, 'Timestreams', [
            ('Steps', [
                ('Tuning', 'The tuning the window works on (every '
                 'tuning of the store with tones); changing it drops '
                 'the capture and clears the board'),
                ('Calibration', 'Calibrate the selected tone set (saved as '
                 'cal{j}); Use picks a saved calibration'),
                ('Tones on the board', 'Write tones writes the chosen chunk of '
                 'the set for the stage (fast stages take fewer tones); Clear '
                 'tones clears the board; nothing else writes tones'),
                ('Preview', 'Capture a few seconds of the tones on the board '
                 '(not saved)'),
                ('Plots', 'The capture or the viewed run: one tone as x, '
                 'parallel / perpendicular to the loop, or I and Q; Show PSD '
                 'adds its spectrum; Subtract offset removes the mean of '
                 'each series'),
                ('Run', 'Stream the tones on the board and save them as '
                 'cal{j}/write{w}/ts{n}; a name is used once per tuning'),
                ('View run', 'Show a saved run like a capture (its first '
                 'chunks)'),
                ('Cancel', 'Stop the running calibration, capture or run after '
                 'asking; what it saved is removed'),
                ('', 'A grey button says beside it what it needs')]),
            ('Keyboard', [('R', 'Rescale the plots'),
                          ('H', 'Show / hide this help')]),
            ('Mouse', [('Scroll / drag', 'Zoom / pan a plot'),
                       ('Right-click', 'Plot menu (axes, log scale, export)')]),
        ])
        # Fit the screen: the panel scrolls when the screen is shorter
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        h = self._panel.sizeHint().height() + 80
        if avail is not None:
            h = min(h, avail.height() - 80)
        self.resize(self._scroll.width() + 900, h)
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

        self._reasons = {}

        def reason_label(btn):
            """
            Make the label beside a button that says why it is
            disabled (``_enable``).

            Parameters:
            btn (QPushButton): the button.

            Returns:
            label (QLabel): the label, blank while the button is
                enabled.
            """
            label = QtWidgets.QLabel()
            label.setStyleSheet('color: gray')
            label.setFixedHeight(btn.sizeHint().height())
            self._reasons[btn] = (label, btn.toolTip())
            return label

        def action_row(btn):
            """
            Put a button in a row with its reason label.

            Parameters:
            btn (QPushButton): the button.

            Returns:
            row (QHBoxLayout): the row.
            """
            row = QtWidgets.QHBoxLayout()
            row.addWidget(btn)
            row.addWidget(reason_label(btn), stretch = 1)
            return row

        self._panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self._panel)
        # Two columns, in the order of the steps: calibrate and write
        # tones on the left, preview, plots and run on the right
        left, right = QtWidgets.QVBoxLayout(), QtWidgets.QVBoxLayout()
        sets = sorted(set(self.tones['tone_set'].tolist()))
        # The tuning: every tuning of the store with tones
        self.tune_box = QtWidgets.QComboBox()
        self.tune_box.setToolTip('The tuning whose tone sets, '
                                 'calibrations and runs the window '
                                 'works on')
        self.tune_box.currentIndexChanged.connect(self._on_tune_box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Tuning'))
        row.addWidget(self.tune_box, stretch = 1)
        outer.addLayout(row)
        self.header = QtWidgets.QLabel(self._header_text())
        outer.addWidget(self.header)

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
        form.addRow(action_row(self.cal_btn))
        self.cal_box = QtWidgets.QComboBox()
        self.cal_box.currentIndexChanged.connect(self._on_cal)
        form.addRow('Use', self.cal_box)
        left.addWidget(box)

        # Tones on the board: written here, never by a preview or run
        box = QtWidgets.QGroupBox('Tones on the board')
        form = QtWidgets.QFormLayout(box)
        self.write_dec = dec_box(6)
        self.write_dec.setToolTip('The stage the chunks are planned '
                                  'for: fast stages take fewer tones')
        self.write_dec.valueChanged.connect(self._update_chunks)
        self.chunk_box = QtWidgets.QComboBox()
        self.chunk_box.setToolTip('The part of the tone set to write '
                                  '(all of it when it fits)')
        self.write_btn = QtWidgets.QPushButton('Write tones')
        self.write_btn.clicked.connect(self.write_tones)
        self.clear_btn = QtWidgets.QPushButton('Clear tones')
        self.clear_btn.clicked.connect(self.clear_tones)
        self.written_label = QtWidgets.QLabel()
        self.written_label.setWordWrap(True)
        form.addRow('Stage', self.write_dec)
        form.addRow('Chunk', self.chunk_box)
        form.addRow(action_row(self.write_btn))
        form.addRow(action_row(self.clear_btn))
        form.addRow(self.written_label)
        left.addWidget(box)
        if self.extra_features:
            left.addWidget(self._build_adjust_box())

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
        self.psd_box = QtWidgets.QCheckBox('Show PSD')
        self.psd_box.toggled.connect(self._on_psd_box)
        self.offset_box = QtWidgets.QCheckBox('Subtract offset')
        self.offset_box.setToolTip('Subtract the mean of each series '
                                   'shown (I and Q separately)')
        self.offset_box.toggled.connect(self._plot)
        form.addRow('Stage', self.preview_dec)
        form.addRow('Length', self.preview_s)
        form.addRow(action_row(self.capture_btn))
        right.addWidget(box)

        # Plots: of the capture or the viewed run, whichever is shown
        box = QtWidgets.QGroupBox('Plots (the capture or the viewed run)')
        form = QtWidgets.QFormLayout(box)
        form.addRow('Tone', self.tone_box)
        form.addRow('Show', self.mode_box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.psd_box)
        row.addWidget(self.offset_box)
        form.addRow('', row)
        right.addWidget(box)

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
        self.run_btn = QtWidgets.QPushButton('Run')
        self.run_btn.clicked.connect(self.run)
        form.addRow('Preset', self.preset_box)
        form.addRow('Name', self.name_edit)
        form.addRow('Notes', self.notes_edit)
        form.addRow('Stage', self.run_dec)
        form.addRow('Length', self.run_s)
        form.addRow(action_row(self.run_btn))
        right.addWidget(box)
        left.addStretch(1)
        right.addStretch(1)
        cols = QtWidgets.QHBoxLayout()
        cols.addLayout(left)
        cols.addLayout(right)
        outer.addLayout(cols)

        self.runs_list = QtWidgets.QListWidget()
        for r in self.tuning_runs():
            self.runs_list.addItem(self._run_label(r))
        self.runs_list.currentRowChanged.connect(self._on_run_selected)
        self.runs_list.itemDoubleClicked.connect(lambda _: self.view_run())
        # Short by default; takes the spare height of a taller window
        self.runs_list.setMinimumHeight(80)
        self.runs_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Ignored)
        outer.addWidget(QtWidgets.QLabel('Runs of this tuning'))
        outer.addWidget(self.runs_list, stretch = 1)
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
        row.addWidget(reason_label(self.view_btn))
        self.view_label = QtWidgets.QLabel()
        row.addWidget(self.view_label, stretch = 1)
        outer.addLayout(row)
        # The progress bar has a place of its own, kept while idle so that
        # nothing moves when a job starts
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self._show_progress(False)
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        self.cancel_btn.setToolTip('Stop the running calibration, capture '
                                   'or run; what it saved is removed. '
                                   'No shortcut, on purpose.')
        self.cancel_btn.clicked.connect(lambda: self.cancel())
        self.cancel_btn.setEnabled(False)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.progress, stretch = 1)
        row.addWidget(self.cancel_btn)
        outer.addLayout(row)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self._inputs = (self.tune_box, self.set_box, self.cal_box,
                        self.cal_notes_edit,
                        self.write_dec, self.chunk_box, self.preview_dec,
                        self.preview_s, self.preset_box,
                        self.name_edit, self.notes_edit, self.run_dec,
                        self.run_s, self.view_chunks) + \
            tuple(self.cal_params.values())
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidget(self._panel)
        self._scroll.setWidgetResizable(True)
        # As wide as the panel wants, up to half the screen
        width = self._panel.sizeHint().width() + 40
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            width = min(width, screen.availableGeometry().width() // 2)
        self._scroll.setFixedWidth(width)
        self._scroll.setMinimumHeight(300)
        if self.presets:
            self.preset_box.setCurrentIndex(1)

    def _build_adjust_box(self):
        """
        Build the Adjust tones box (``extra_features``): every tone's
        power offset or set, every frequency offset, saved as a new
        tuning the window then works on (``daq_layout.derive_tuning``).

        Returns:
        box (QGroupBox): the box.
        """
        box = QtWidgets.QGroupBox('Adjust tones (saved as a new tuning)')
        form = QtWidgets.QFormLayout(box)
        self.adjust_power = QtWidgets.QDoubleSpinBox()
        self.adjust_power.setRange(-40., 40.)
        self.adjust_power.setDecimals(1)
        self.adjust_power.setSuffix(' dB')
        self.adjust_power.setToolTip('Added to every tone power '
                                     '(negative lowers them)')
        form.addRow('Power offset', self.adjust_power)
        self.adjust_freq = QtWidgets.QDoubleSpinBox()
        self.adjust_freq.setRange(-1., 1.)
        self.adjust_freq.setDecimals(6)
        self.adjust_freq.setSingleStep(1e-4)
        self.adjust_freq.setToolTip('Added to every tone frequency, as '
                                    'a fraction of the total span '
                                    '(highest minus lowest tone)')
        form.addRow('Frequency offset', self.adjust_freq)
        row = QtWidgets.QHBoxLayout()
        self.adjust_const = QtWidgets.QCheckBox('Set every power to')
        self.adjust_const.setToolTip('Every tone gets this power; the '
                                     'power offset is then not used')
        self.adjust_dbm = QtWidgets.QDoubleSpinBox()
        self.adjust_dbm.setRange(-120., 10.)
        self.adjust_dbm.setDecimals(1)
        self.adjust_dbm.setSuffix(' dBm')
        self.adjust_dbm.setValue(-55.)
        self.adjust_dbm.setEnabled(False)
        self.adjust_const.toggled.connect(self.adjust_dbm.setEnabled)
        row.addWidget(self.adjust_const)
        row.addWidget(self.adjust_dbm, stretch = 1)
        form.addRow(row)
        self.adjust_name = QtWidgets.QLineEdit()
        self.adjust_name.setPlaceholderText('name of the new tuning')
        form.addRow('Name', self.adjust_name)
        self.adjust_btn = QtWidgets.QPushButton('Make tuning')
        self.adjust_btn.setToolTip('Save the adjusted tones as a new '
                                   'tuning and switch to it (it needs '
                                   'its own calibrations)')
        self.adjust_btn.clicked.connect(self.adjust_tones)
        form.addRow(self.adjust_btn)
        return box

    def adjust_tones(self):
        """
        Save the adjusted tones as a new tuning (the Adjust tones box)
        and switch the window to it.

        Returns:
        tune (zarr.Group or None): the new tuning, or None if nothing
            was made (no name, a used name, nothing to change).
        """
        name = self.adjust_name.text().strip()
        if not name:
            self._set_status('Adjust tones needs a name for the new '
                             'tuning.')
            return None
        const = self.adjust_const.isChecked()
        root = dl.store_root(self.tune, mode = 'r+')
        try:
            tune, k = dl.derive_tuning(
                root, root[self.tune.basename], name,
                power_db = self.adjust_power.value(),
                freq_frac = self.adjust_freq.value(),
                power_dbm = self.adjust_dbm.value() if const else None)
        except (ValueError, KeyError) as exc:
            self._set_status(f'Adjust tones: {exc}')
            return None
        self._fill_tune_box()
        self.set_tuning(tune)
        self._set_status(
            f'Tuning {name} ({tune.basename}) made from '
            f"{tune.attrs['derived_from']['name']}: calibrate it before "
            'writing tones.')
        return tune

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
        # The PSD of what the timestream plot shows (the box)
        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setLogMode(x = True, y = True)
        self.psd_plot.setLabel('bottom', 'Frequency (Hz)')
        self.psd_plot.addLegend()
        self.psd_plot.setVisible(False)

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
        label = f" {cal['label']}" if cal.get('label') else ''
        return (f"{cal['name']}{label}: {cal['ntones']} tones, "
                f"{cal['npoints_fine']} fine points x {cal['nsamps']} "
                f"samples, {cal.get('created', '')}"
                + ('' if cal['complete'] else ', incomplete')
                + ('' if cal.get('in_frame') else ', stale (NCO set since)')
                + (f' - {notes}' if notes else ''))

    def run_names(self):
        """
        Return the names of the tuning's runs (a name is used once).

        Returns:
        names (set of str): the names.
        """
        return {str(r.get('name')) for r in self.tuning_runs()}

    def _header_text(self):
        """
        Describe the tuning for the header line.

        Returns:
        text (str): name, group, tones and tone sets.
        """
        sets = sorted(set(self.tones['tone_set'].tolist()))
        return (f'{dl.tuning_name(self.tune)} ({self.tune.basename}): '
                f"{len(self.tones['fres'])} tones in {len(sets)} tone set(s)")

    def _fill_tune_box(self):
        """
        List the store's tunings with tones in the Tuning box and select
        the window's.
        """
        root = dl.store_root(self.tune)
        self.tune_box.blockSignals(True)
        self.tune_box.clear()
        for k in dl.list_indices(root, 'tune'):
            t = root[dl.set_name(k, 'tune')]
            if 'tones' in t:
                self.tune_box.addItem(
                    f'{dl.tuning_name(t)} ({t.basename})', t.basename)
        self.tune_box.setCurrentIndex(
            self.tune_box.findData(self.tune.basename))
        self.tune_box.blockSignals(False)

    def _on_tune_box(self):
        """
        Switch to the tuning chosen in the Tuning box.
        """
        name = self.tune_box.currentData()
        if name and name != self.tune.basename:
            self.set_tuning(dl.store_root(self.tune, mode = 'r+')[name])

    def set_tuning(self, tune):
        """
        Work on another tuning of the store: its tone sets, calibrations
        and runs replace the window's; the capture, the viewed run and
        the chosen calibration are dropped, and tones written from the
        tuning before are cleared from the board.

        Parameters:
        tune (zarr.Group): the tuning, writable, with ``tones``.

        Raises:
        ValueError: if its tones have no ``tone_module``.
        """
        tones = dl.load_tones(tune['tones'])
        if 'tone_module' not in tones:
            raise ValueError(f'{tune.basename}/tones has no tone_module; '
                             'make the tuning with 3_tune.py')
        if self.written and self.crs is not None and not self.worker.busy:
            self.clear_tones()
        self.tune, self.tones = tune, tones
        self.setWindowTitle(f'{self._title}: {dl.tuning_name(tune)} '
                            f'({tune.basename})')
        self.header.setText(self._header_text())
        self.cal, self.preview, self.view, self.runs = None, None, None, []
        self.chosen = {}
        self.set_box.blockSignals(True)
        self.set_box.clear()
        for s in sorted(set(tones['tone_set'].tolist())):
            n = int(np.sum(tones['tone_set'] == s))
            self.set_box.addItem(f'Tone set {s} ({n} tones)', s)
        self.set_box.blockSignals(False)
        self.runs_list.blockSignals(True)
        self.runs_list.clear()
        for r in self.tuning_runs():
            self.runs_list.addItem(self._run_label(r))
        self.runs_list.blockSignals(False)
        if self.tune_box.currentData() != tune.basename:
            self.tune_box.blockSignals(True)
            self.tune_box.setCurrentIndex(
                self.tune_box.findData(tune.basename))
            self.tune_box.blockSignals(False)
        self._on_set()
        self._on_run_selected(-1)
        self._set_status(f'Tuning {dl.tuning_name(tune)} ({tune.basename}).')

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

    def _chunks(self, dec):
        """
        Return the selected tone set's chunks at a stage.

        Parameters:
        dec (int): the decimation stage.

        Returns:
        chunks (list of np.array or None): ``plan_stream_chunks``, or None
            if the set can't stream at that stage.
        """
        try:
            return plan_stream_chunks(
                self.tones['tone_module'][self._set_rows()], int(dec))
        except ValueError:
            return None

    def _update_chunks(self):
        """
        List the selected tone set's chunks at the write stage.
        """
        chunks = self._chunks(self.write_dec.value()) or []
        self.chunk_box.blockSignals(True)
        self.chunk_box.clear()
        for i, rows in enumerate(chunks):
            text = f'{len(rows)} tones' if len(chunks) == 1 else \
                f'{i + 1} of {len(chunks)}: {len(rows)} tones'
            self.chunk_box.addItem(text, i)
        if not chunks:
            self.chunk_box.addItem('(none: the stage takes no tones)', None)
        self.chunk_box.blockSignals(False)
        self._update_buttons()

    def board_ready(self):
        """
        Say whether the board holds tones of the chosen calibration
        (written here with Write tones), which a preview or run needs.

        Returns:
        ready (bool): True if so.
        """
        w, cal = self.written, self.cal
        return bool(w) and cal is not None and \
            w.get('calibration') == cal['path'] and \
            w.get('tone_set') == cal['set']

    def _fits(self, dec):
        """
        Say whether the board's tones can stream at a stage, and
        explain if not.

        Parameters:
        dec (int): the decimation stage.

        Returns:
        ok (bool): True if they can.
        """
        rows = self.cal['rows'][np.asarray(self.written['rows'])]
        try:
            n = len(plan_stream_chunks(self.tones['tone_module'][rows],
                                       int(dec)))
        except ValueError as e:
            self._set_status(str(e))
            return False
        if n != 1:
            self._set_status(f'The {len(rows)} tones on the board are too '
                             f'many for stage {dec}: write a smaller '
                             'chunk (set the stage above Chunk).')
            return False
        return True

    def _update_buttons(self):
        """
        Enable the buttons that can be used now, and say what is on the
        board.
        """
        busy = self.worker.busy
        for w in self._inputs:
            w.setEnabled(not busy)
        wait = 'busy' if busy else \
            'connecting to the CRS' if self.crs is None else None
        self._enable(self.cal_btn, wait)
        cal = wait or self._cal_reason()
        self._enable(self.write_btn, cal or (
            None if self.chunk_box.currentData() is not None
            else 'the stage takes no tones'))
        self._enable(self.clear_btn, wait or (
            None if self.written else 'no tones written'))
        ready = cal or (None if self.board_ready()
                        else 'write the tones first')
        self._enable(self.capture_btn, ready)
        name = self.name_edit.text().strip()
        self._enable(self.run_btn, ready or (
            'enter a name' if not name else
            f'A run named {name!r} exists: choose another name'
            if name in self.run_names() else None))
        self._enable(self.view_btn, 'busy' if busy else (
            None if self.runs_list.currentRow() >= 0 else 'select a run'))
        self.cancel_btn.setEnabled(self.worker.busy and self._cancellable
                                   and not self._cancel.is_set())
        w = self.written
        self.written_label.setText(
            'Tones on the board: none (or unknown)' if not w else
            f"Tones on the board: {len(w['rows'])} of tone set "
            f"{w['tone_set']} from {w['calibration']}")

    def _cal_reason(self):
        """
        Say why the chosen calibration cannot be used for the selected
        tone set.

        Returns:
        reason (str or None): the reason; None if it can.
        """
        if self.cal is None:
            return 'calibrate first'
        if self.cal['set'] != self.set_box.currentData():
            return 'the calibration is of another tone set'
        if not self.cal['in_frame']:
            return 'the calibration is out of frame: calibrate again'
        return None

    def _enable(self, btn, reason):
        """
        Enable a button, or disable it and say why beside it (and in
        its tooltip).

        Parameters:
        btn (QPushButton): the button.
        reason (str or None): why it is disabled; None: enable it.
        """
        btn.setEnabled(reason is None)
        label, tip = self._reasons[btn]
        label.setText('' if reason is None else reason)
        btn.setToolTip(tip if reason is None else reason)

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
        self._fill_tones(self._set_rows())
        self._update_chunks()
        self._refresh_cals()

    def _fill_tones(self, rows):
        """
        List tones in the Tone box.

        Parameters:
        rows (array-like int): indices into the tuning's tones.
        """
        self.tone_box.blockSignals(True)
        self.tone_box.clear()
        for r in rows:
            self.tone_box.addItem(
                f"res {self.tones['res_idxs'][r]}: "
                f"{self.tones['fres'][r] / 1e6:.4f} MHz", int(r))
        self.tone_box.blockSignals(False)

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

    def _submit(self, coro, callback, message, progress = False,
                cancellable = False):
        """
        Run a job on the background loop.

        Parameters:
        coro (coroutine): the job.
        callback (callable): ``callback(result, error)`` on the GUI thread.
        message (str): status message while it runs.
        progress (bool): If True, show the progress bar until it is done.
            Default False.
        cancellable (bool): If True, the Cancel button can stop it (the
            job reads ``self._cancel``). Default False.
        """
        def done(result, error):
            """
            Hide the progress bar, then call back.

            Parameters:
            result: the job's result.
            error (BaseException or None): its error.
            """
            self._cancellable = False
            self._show_progress(False)
            callback(result, error)

        self._cancel.clear()
        self._cancellable = bool(cancellable)
        self._show_progress(progress)
        self.worker.submit(coro, done)
        self._set_status(message)
        self._update_buttons()

    def cancel(self, ask = True):
        """
        Ask the running calibration, capture or run to stop (the Cancel
        button), after asking whether to. It stops at its next sweep
        point or packet batch, on the background loop, and its
        callback reports what was removed. Does nothing when no such
        job runs.

        Parameters:
        ask (bool): If True (default), ask in a box first.
        """
        if not (self.worker.busy and self._cancellable) or \
                self._cancel.is_set():
            return
        if ask:
            box = QtWidgets.QMessageBox
            answer = box.question(
                self, 'Cancel', 'Stop the running calibration, capture '
                'or run? What it saved so far is removed.',
                box.Yes | box.No, box.No)
            if answer != box.Yes:
                return
        self._cancel.set()
        self.cancel_btn.setEnabled(False)
        self._set_status('Cancelling...')

    @staticmethod
    def _was_cancelled(error):
        """
        Say whether a job's error is a cancellation.

        Parameters:
        error (BaseException or None): the job's error.

        Returns:
        cancelled (bool): True if so.
        """
        return isinstance(error, (SweepCancelled, StreamCancelled))

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
        cancelled, fres_update = self._cancel.is_set, self.fres_update

        async def job():
            """
            Sweep the set.

            Returns:
            name (str): the calibration's name.
            """
            grp = await calibrate_tone_set(crs, tune, s, verbose = False,
                                           progress = report, notes = notes,
                                           cancelled = cancelled,
                                           fres_update = fres_update, **kw)
            return grp.basename

        self._submit(job(), self._on_calibrated,
                     f'Calibrating tone set {s} ({len(self._set_rows())} '
                     'tones)...', progress = True, cancellable = True)

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
            up = self.tune[name].attrs.get('fres_update')
            moved = '' if not up else (
                f" {up['n_updated']} tone frequencies moved to the sweep's "
                f"resonances, {up['n_outside']} kept.")
            self._set_status(f"Tone set {self.cal['set']} calibrated as "
                             f'{name}.{moved} Write the tones, then capture '
                             'or run.')
        elif self._was_cancelled(error):
            self._refresh_cals(select = self.chosen.get(
                self.set_box.currentData()))
            self._set_status('Calibration cancelled: nothing saved, the '
                             'tones cleared.')
        else:
            self._set_status(f'Calibration failed: {error!r}')
        self._update_buttons()

    def capture(self):
        """
        Stream the tones on the board (Write tones first) into memory
        for a preview.
        """
        if not self.board_ready():
            self._set_status('Write the tones first.')
            return
        cal, crs = self.cal, self.crs
        dec, secs = self.preview_dec.value(), self.preview_s.value()
        if not self._fits(dec):
            return
        chunk = np.asarray(self.written['rows'], dtype = np.int64)
        fres = np.asarray(self.written['fres'], dtype = np.float64)
        report, cancelled = self._progress.emit, self._cancel.is_set

        async def job():
            """
            Stream the board's tones into memory.

            Returns:
            preview (dict): the capture.
            """
            zt = await crs.stream_to_memory(secs, dec, verbose = False,
                                            progress = report,
                                            cancelled = cancelled)
            # The stream's stage: the CRS may be back at stage 6 by now
            return dict(rows = cal['rows'][chunk], pos = chunk, fres = fres,
                        zt = zt, dt = 1 / get_sample_freq(dec))

        self._submit(job(), self._on_captured,
                     f'Streaming {len(chunk)} tones for {secs:g} s at '
                     f'stage {dec}...', progress = True, cancellable = True)

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
        elif self._was_cancelled(error):
            self._set_status('Capture cancelled.')
        else:
            self._set_status(f'Capture failed: {error!r}')
        self._update_buttons()
        self._plot()

    def write_tones(self):
        """
        Write the chosen chunk of the selected tone set (planned for
        the write stage) from its calibration, even if it is on the
        board already. Previews and runs stream what is written.
        """
        cal, chunks = self.cal, self._chunks(self.write_dec.value())
        i = self.chunk_box.currentData()
        if cal is None or not chunks or i is None or i >= len(chunks):
            self._set_status('Write tones needs a calibration and a chunk '
                             'the stage can take.')
            return
        grp, written, rows = self.tune[cal['name']], self.written, chunks[i]
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

    def clear_tones(self):
        """
        Clear the board's tones (the button).
        """
        crs, written = self.crs, self.written

        async def job():
            """
            Clear the tones.
            """
            await crs.clear_tones()
            written.clear()

        self._submit(job(), self._on_cleared, 'Clearing the tones...')

    def _on_cleared(self, _result, error):
        """
        Report the clear (GUI thread).

        Parameters:
        _result: unused.
        error (BaseException or None): the error.
        """
        self._set_status('Tones cleared.' if error is None else
                         f'Clear failed: {error!r}')
        self._update_buttons()

    def _on_written(self, n, error):
        """
        Report the write (GUI thread).

        Parameters:
        n (int or None): tones written.
        error (BaseException or None): the write error.
        """
        if error is None:
            self._set_status(f'{n} tones written; captures and runs stream '
                             'them until they are cleared or written '
                             'again.')
        else:
            self._set_status(f'Write failed: {error!r}')
        self._update_buttons()

    def run(self):
        """
        Take the named timestream of the tones on the board (Write tones
        first), saved under their calibration.
        """
        name = self.name_edit.text().strip()
        if name in self.run_names():
            self._set_status(f'A run named {name!r} exists: choose '
                             'another name.')
            return
        if not self.board_ready():
            self._set_status('Write the tones first.')
            return
        notes = self.notes_edit.toPlainText()
        dec, secs = self.run_dec.value(), self.run_s.value()
        if not self._fits(dec):
            return
        s, cal = self.cal['set'], self.cal['path']
        coro = stream_tone_sets(
            self.crs, self.tune, name, secs, dec, notes = notes,
            set_idxs = [s], calibrations = {s: cal}, written = self.written,
            keep_tones = True, status = self._message.emit, verbose = False,
            progress = self._progress.emit, only_written = True,
            cancelled = self._cancel.is_set, setup = self.setup)
        self._submit(coro, self._on_run,
                     f'Running {name} on tone set {s} with {cal}...',
                     progress = True, cancellable = True)

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
        elif self._was_cancelled(error):
            self._set_status('Run cancelled: its streams and record '
                             'removed; the tones stay on the board.')
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
            # A run of another set: list its tones, changing no
            # setting (the chosen set, calibration, stages stay)
            if self.set_box.currentData() != view['tone_set']:
                self._fill_tones(view['rows'])
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
        Plot the selected tone's preview and calibration loop
        (``plot_view``): a saved run brings its own calibration loop, a
        capture uses the chosen calibration's.
        """
        r = self.tone_box.currentData()
        cal = self.view if self.view is not None else self.cal
        p = self.view if self.view is not None else self.preview
        label = p['label'] if self.view is not None else \
            'Preview (not saved)'
        plot_view(self.ts_plot, self.iq_plot, self.psd_plot, cal, p, r,
                  mode = self.mode_box.currentIndex(),
                  offset = self.offset_box.isChecked(),
                  show_psd = self.psd_box.isChecked(), label = label)

    def _on_psd_box(self, show):
        """
        Show or hide the PSD plot and draw it (the box).

        Parameters:
        show (bool): the box.
        """
        self.psd_plot.setVisible(bool(show))
        self._plot()

    def rescale(self):
        """
        Fit both plots to their data again (the R key), e.g. after zooming
        or changing what is shown.
        """
        # Enabling auto-range fits the view at once (autoRange() would
        # set a fixed range and turn auto-range off again)
        for plot in (self.ts_plot, self.iq_plot, self.psd_plot):
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
    sweep_kwargs = None, title = 'Timestreams', ui_scale = 1.0,
    fres_update = None, setup = None, extra_features = False
):
    """
    Open the timestream window on a tuning and wait until it is closed.

    Parameters:
    connect (callable): coroutine function with no arguments returning an
        attached CRS (with global control to stream at another decimation
        stage than the board's). It runs on the window's background event
        loop, which every sweep and stream then uses.
    tune (zarr.Group): the tuning, writable.
    preview_dec, preview_s, presets, sweep_kwargs, title, ui_scale,
        fres_update, setup, extra_features: see ``StreamWindow``.

    Returns:
    runs (list of list of str): stream paths of each run of the session.
    """
    app = get_qapp(title)
    win = StreamWindow(connect, tune, preview_dec = preview_dec,
                       preview_s = preview_s, presets = presets,
                       sweep_kwargs = sweep_kwargs, title = title,
                       ui_scale = ui_scale, fres_update = fres_update,
                       setup = setup, extra_features = extra_features)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.runs
