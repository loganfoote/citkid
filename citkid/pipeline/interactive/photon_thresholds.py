"""
Window for choosing the photon detection thresholds of one timestream.

The left panel lists the tones (resonator index, pulses in the height
window, rate) and holds the thresholds. The plots show, for the selected
tone or for all tones together:

- the first-pass peak heights, in units of their median, with the upper
  cut ``init_hmax``;
- the heights of the optimal filter peaks (the coincidence candidates, so
  heights below hmin are shown too), in kHz/GHz (x times 1e6), with the
  expected noise peaks (the inverted filter output's peaks, scaled to the
  whole record) and the height window [hmin, hmax];
- the tone's mean pulse in the height window, with the fit of its decay
  (``signal.photons.fit_tau``: from ``tau_skip`` samples after the
  maximum, over ``tau_lifetimes`` decay times) and the decay time;
- a snippet of the tone's data (``data`` given): the detection timestream
  and the optimal filter's output, both in units of the mean photon, with
  the height window as horizontal lines and the filter peaks marked
  (events in the window, peaks below hmin, peaks above hmax), so the
  histogram can be read against the pulses themselves. The snippet's
  length and position can be changed, and Prev/Next jump between the
  filter peaks.

Heights and the thresholds are in kHz/GHz (the fractional frequency
shift x times 1e6). With "Heights in" set to photons, the plots are drawn in
units of each tone's mean photon pulse (``amp_ph``) instead; the
threshold boxes stay in kHz/GHz and the lines show where they fall for the
selected tone. R (or Rescale) fits every plot to its data again.

The plots don't pan or zoom with the mouse (R rescales them); the mouse
sets the thresholds instead. The lines can be dragged; a click on a
histogram moves its nearer line to the click (the init_hmax line on the
first-pass histogram) and a drag there carries it along; a click on the
snippet moves the nearer of hmin and hmax to the click's height, and a
drag on the snippet sets hmin and hmax from the drag's bottom and top.
The mean pulse plot marks where the optimal filter's template ends
(``filter_post``). "Rerun" applies the
thresholds either
to the selected tone (the default: an override for that resonator) or to
all tones (the global values, used by every tone without an override),
and runs the detection again for the tones they change, through the
``rerun`` callable (``pipeline.photon_finder.detect_timestream``). A and
D (or the buttons) move to the previous and next tone; thresholds changed
but not rerun are applied before moving on. "Revert tone to defaults"
puts the thresholds the window opened with (the script's) back on the
selected tone. Close the window (or press Done) to keep the thresholds
and results.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
from ...qt_compat import (
    PlainAxisItem, Qt, delete_on_close, fit_window_to_screen, get_qapp,
    hint_label, run_responsive, widget_font_stylesheet
)
from ...signal.photons import (
    PPM, DetectionSettings, OptimalFilter, fit_tau
)

# Default snippet: long enough for this many pulses at the median rate of
# the tones, at least this long, or this many samples if that is longer
SNIPPET_PULSES = 10
SNIPPET_S = 0.2
SNIPPET_SAMPLES = 400
# Slider steps over the record
SLIDER_STEPS = 1000
# Height units offered: label, and whether heights are in kHz/GHz (x
# times 1e6, the thresholds' units) rather than in each tone's mean photon
UNITS = (('kHz/GHz', True), ('photons', False))

def filter_snippet(d, res, start, stop):
    """
    Compute the detection's optimal filter output on part of a detection
    timestream, in kHz/GHz: the same filter and unit as ``run_detection``'s
    heights, so its peaks are the histogram's.

    Parameters:
    d (np.array): the detection timestream.
    res (dict): the tone's detection result (``filter``, ``p`` or
        ``filter_p``, ``nfft``, ``amp_ph``, ``sign``).
    start, stop (int): sample range.

    Returns:
    y (np.array): the filter output on ``d[start:stop]``.
    """
    n = len(d)
    start, stop = max(int(start), 0), min(int(stop), n)
    if stop <= start:
        return np.zeros(0, dtype = np.float32)
    nfft = int(res['nfft'])
    of = OptimalFilter.from_kernel(res['filter'],
                                   res.get('filter_p', res['p']), nfft)
    lo, hi = max(start - nfft, 0), min(stop + nfft, n)
    y = of.apply(np.asarray(d[lo:hi], dtype = np.float64),
                 int(res['sign'])) * PPM
    return y[start - lo:stop - lo]

class ThresholdViewBox(pg.ViewBox):
    """
    ViewBox that hands clicks and drags to callbacks (to set thresholds).

    The view box has no pan, zoom, or menu: a left click and a left drag
    go to the callbacks (with or without Shift), and nothing else
    happens with the mouse. (pyqtgraph turns a slow or slightly moving
    click into a drag, which used to pan the plot.)

    Attributes:
    on_click (callable or None): ``on_click(x, y)`` with the click in
        data coordinates.
    on_drag (callable or None): ``on_drag(p0, p1, finished)`` with the
        drag's start and current points as (x, y) in data coordinates,
        while dragging and at the end.
    """

    def __init__(self, on_click = None, on_drag = None, **kwargs):
        """
        Make the view box.

        Parameters:
        on_click, on_drag (callable or None): see the attributes.
        **kwargs: ``pg.ViewBox`` arguments.
        """
        super().__init__(**kwargs)
        self.on_click = on_click
        self.on_drag = on_drag
        self.setMouseEnabled(x = False, y = False)
        self.setMenuEnabled(False)

    def mouseClickEvent(self, ev):
        """
        Left click: the callback; other buttons do nothing.

        Parameters:
        ev: the mouse click event.
        """
        ev.accept()
        if self.on_click is not None and ev.button() == Qt.LeftButton:
            p = self.mapSceneToView(ev.scenePos())
            self.on_click(p.x(), p.y())

    def mouseDragEvent(self, ev, axis = None):
        """
        Left drag: the callback; other buttons do nothing.

        Parameters:
        ev: the mouse drag event.
        axis (int or None): the axis dragged, for axis drags (unused).
        """
        ev.accept()
        if self.on_drag is not None and ev.button() == Qt.LeftButton:
            p0 = self.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.mapSceneToView(ev.scenePos())
            self.on_drag((p0.x(), p0.y()), (p1.x(), p1.y()), ev.isFinish())

    def wheelEvent(self, ev, axis = None):
        """
        The wheel does nothing (no zoom).

        Parameters:
        ev: the wheel event.
        axis (int or None): unused.
        """
        ev.accept()

class PhotonThresholdWindow(QtWidgets.QMainWindow):
    """
    Photon threshold window (see the module description).

    Attributes:
    results (dict): row: detection result (dict) or error message (str).
    thresholds (Thresholds): the current thresholds.
    rows (list of int): the timestream's DataSet rows.
    res_idxs (list of int): their resonator indices.
    snippet (dict or None): the data snippet shown: ``row``, ``start``,
        ``stop`` (samples), ``y`` (filter output, kHz/GHz), ``raw``
        (detection timestream, kHz/GHz).
    """

    def __init__(
        self, ts, results, thresholds, rerun, data = None, settings = None,
        title = 'Photon thresholds', ui_scale = 1.0
    ):
        """
        Build the window.

        Parameters:
        ts (dict): the timestream (``photon_finder.list_timestreams``).
        results (dict): its ``detect_timestream`` results.
        thresholds (Thresholds): starting thresholds (changed in place).
        rerun (callable): function of (rows, thresholds) returning the new
            results of those rows (dict row: result), e.g. a wrapper of
            ``detect_timestream``.
        data (callable or None): function of a row returning its detection
            timestream (np.array), or None or an error message (str) if
            there is none, for the data snippet. Default None: no
            snippet.
        settings (DetectionSettings or None): for the decay-time fit of
            the mean pulse (``tau_skip``, ``tau_lifetimes``). Default
            None: the defaults.
        title (str): window title. Default 'Photon thresholds'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(title)
        self._ui_scale = float(ui_scale) if ui_scale else 1.0
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.results = dict(results)
        self.thresholds = thresholds
        self.rows = [int(r) for r in ts['rows']]
        self.res_idxs = [int(r) for r in ts['res_idxs']]
        self.dt = float(ts['dt'])
        self._record_s = (float(ts['stop'] - ts['start']) * self.dt
                          if 'start' in ts and 'stop' in ts else None)
        # The thresholds the window opened with (the script's)
        self._defaults = dict(hmin = thresholds.hmin, hmax = thresholds.hmax,
                              init_hmax = thresholds.init_hmax)
        self._row = None
        self._rerun = rerun
        self._data = data
        self._settings = DetectionSettings() if settings is None else \
            settings
        self._d = (None, None)      # (row, detection timestream) cached
        self.snippet = None
        self._updating = False
        self._drag_box = None       # the box a histogram drag moves
        self._build_controls()
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots_widget, stretch = 1)
        self.setCentralWidget(central)
        self._fill_list()
        self.tone_list.setCurrentRow(0)

    ############################################################################
    def _build_controls(self):
        """
        Build the left panel: tone list, thresholds, and buttons.
        """
        self._panel = QtWidgets.QWidget()
        # Fixed: the panel's preferred width would follow the status text
        # and the tone list, and the layout would shift while rerunning
        self._panel.setFixedWidth(int(330 * self._ui_scale))
        lay = QtWidgets.QVBoxLayout(self._panel)
        self.tone_list = QtWidgets.QListWidget()
        self.tone_list.currentRowChanged.connect(self._on_row_changed)
        lay.addWidget(QtWidgets.QLabel('Tones (* : own thresholds)'))
        lay.addWidget(self.tone_list, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.prev_tone_btn = QtWidgets.QPushButton('< Prev (A)')
        self.prev_tone_btn.clicked.connect(lambda: self.step_tone(-1))
        self.next_tone_btn = QtWidgets.QPushButton('Next (D) >')
        self.next_tone_btn.clicked.connect(lambda: self.step_tone(1))
        row.addWidget(self.prev_tone_btn)
        row.addWidget(self.next_tone_btn)
        lay.addLayout(row)
        self.all_tones = QtWidgets.QCheckBox('Plot all tones together')
        self.all_tones.toggled.connect(self.update_plots)
        lay.addWidget(self.all_tones)
        form = QtWidgets.QFormLayout()
        self.units_box = QtWidgets.QComboBox()
        for label, _ in UNITS:
            self.units_box.addItem(label)
        self.units_box.setToolTip('kHz/GHz: x times 1e6 (the '
                                  'thresholds\' units); photons: in units '
                                  'of each tone\'s mean photon pulse')
        self.units_box.currentIndexChanged.connect(self._on_units)
        form.addRow('Heights in', self.units_box)

        def spin(lo, hi, step):
            """
            Make a threshold spin box.

            Parameters:
            lo, hi (float): range.
            step (float): step.

            Returns:
            box (QDoubleSpinBox): the spin box.
            """
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(2)
            box.setSingleStep(step)
            box.valueChanged.connect(self._on_spin)
            return box

        self.hmin_box = spin(0.01, 100., 0.05)
        self.hmax_box = spin(0.01, 1000., 0.05)
        self.init_hmax_box = spin(0.1, 1000., 0.5)
        form.addRow('hmin (kHz/GHz)', self.hmin_box)
        form.addRow('hmax (kHz/GHz)', self.hmax_box)
        form.addRow('init_hmax (x median)', self.init_hmax_box)
        lay.addLayout(form)
        self.scope_tone = QtWidgets.QRadioButton('Apply to this tone')
        self.scope_all = QtWidgets.QRadioButton('Apply to all tones')
        self.scope_tone.setChecked(True)
        lay.addWidget(self.scope_tone)
        lay.addWidget(self.scope_all)
        self.rerun_btn = QtWidgets.QPushButton('Rerun (Ctrl+R)')
        self.rerun_btn.clicked.connect(self.rerun)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, self.rerun)
        self.reset_btn = QtWidgets.QPushButton(
            'Revert tone to defaults (Ctrl+D)')
        self.reset_btn.setToolTip("Put the thresholds the window opened "
                                  "with (the script's) back on this tone "
                                  'and rerun it')
        self.reset_btn.clicked.connect(self.reset_tone)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+D'), self,
                            self.reset_tone)
        self.rescale_btn = QtWidgets.QPushButton('Rescale plots (R)')
        self.rescale_btn.clicked.connect(self.rescale)
        self.done_btn = QtWidgets.QPushButton('Done')
        self.done_btn.clicked.connect(self.close)
        for b in (self.rerun_btn, self.reset_btn, self.rescale_btn,
                  self.done_btn):
            lay.addWidget(b)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def _build_plots(self):
        """
        Build the plots (the histograms, the mean pulse, the data snippet
        with its controls) and the draggable threshold lines.
        """
        self.plots = pg.GraphicsLayoutWidget()
        # A click on a histogram moves its nearer threshold line, a drag
        # carries it along
        self.init_plot = self.plots.addPlot(
            row = 0, col = 0,
            viewBox = ThresholdViewBox(on_click = self._click_init,
                                       on_drag = self._drag_init))
        self.init_plot.setLabel('bottom', 'First-pass height / median')
        self.init_plot.setLabel('left', 'Peaks')
        self.init_plot.setLogMode(y = True)
        self.h_plot = self.plots.addPlot(
            row = 1, col = 0,
            viewBox = ThresholdViewBox(on_click = self._click_hist,
                                       on_drag = self._drag_hist))
        self.h_plot.setLabel('bottom', 'Filter height (kHz/GHz)')
        self.h_plot.setLabel('left', 'Peaks')
        self.h_plot.setLogMode(y = True)
        self.h_plot.addLegend()
        self.pulse_plot = self.plots.addPlot(row = 2, col = 0,
                                             viewBox = ThresholdViewBox())
        self.pulse_plot.setLabel('bottom', 'Time (ms)')
        self.pulse_plot.setLabel('left', 'Mean pulse (kHz/GHz)')
        self.init_curve = self.init_plot.plot(stepMode = 'center')
        self.h_curve = self.h_plot.plot(stepMode = 'center', name = 'peaks')
        self.noise_curve = self.h_plot.plot(
            stepMode = 'center', pen = pg.mkPen('r'), name = 'noise')
        self.pulse_curve = self.pulse_plot.plot(name = 'mean pulse')
        self.fit_curve = self.pulse_plot.plot(
            pen = pg.mkPen('y', width = 2, style = Qt.DashLine),
            name = 'decay fit')
        self.pulse_plot.addLegend(offset = (-10, 10))
        # Where the optimal filter's template ends
        self.filter_end_line = pg.InfiniteLine(
            0, angle = 90, pen = pg.mkPen((180, 180, 180), style = Qt.DashLine),
            label = 'filter end', labelOpts = dict(position = 0.9,
                                                   color = (180, 180, 180)))
        self.pulse_plot.addItem(self.filter_end_line, ignoreBounds = True)
        self.filter_end_line.hide()

        def line(plot, color):
            """
            Add a draggable vertical line.

            Parameters:
            plot (PlotItem): the plot.
            color (str): pen color.

            Returns:
            line (InfiniteLine): the line.
            """
            ln = pg.InfiniteLine(0, movable = True,
                                 pen = pg.mkPen(color, width = 2))
            plot.addItem(ln)
            ln.sigPositionChanged.connect(self._on_line)
            return ln

        self.init_line = line(self.init_plot, 'y')
        self.hmin_line = line(self.h_plot, 'g')
        self.hmax_line = line(self.h_plot, 'g')
        self._build_snippet()

    def _build_snippet(self):
        """
        Build the data snippet plot (the raw detection timestream and the
        filter output in photons, the height window as horizontal lines,
        the filter peaks marked) and its controls below the plots.
        """
        # Plain tick numbers: no "(x0.001)" on the time axis. A click
        # moves the nearer threshold to the click's height, a drag sets
        # both from the drag's bottom and top
        self.snippet_plot = self.plots.addPlot(
            row = 3, col = 0, axisItems = {'bottom': PlainAxisItem('bottom'),
                                           'left': PlainAxisItem('left')},
            viewBox = ThresholdViewBox(on_click = self._click_snippet,
                                       on_drag = self._drag_snippet))
        self.snippet_plot.setLabel('bottom', 'Time (s)')
        self.snippet_plot.setLabel('left', 'Height (kHz/GHz)')
        self.snippet_plot.addLegend(offset = (-10, 10))
        self.raw_curve = self.snippet_plot.plot(
            pen = pg.mkPen((150, 150, 150)), name = 'data')
        self.y_curve = self.snippet_plot.plot(
            pen = pg.mkPen((80, 160, 255), width = 1.5), name = 'filter')
        self.event_marks = pg.ScatterPlotItem(
            size = 9, symbol = 'o', pen = pg.mkPen('g', width = 2),
            brush = pg.mkBrush(0, 0, 0, 0), name = 'in window')
        self.low_marks = pg.ScatterPlotItem(
            size = 9, symbol = 'x', pen = pg.mkPen('r', width = 2),
            brush = pg.mkBrush('r'), name = 'below hmin')
        self.high_marks = pg.ScatterPlotItem(
            size = 10, symbol = 't', pen = pg.mkPen((255, 170, 0)),
            brush = pg.mkBrush(255, 170, 0), name = 'above hmax')
        for item in (self.event_marks, self.low_marks, self.high_marks):
            self.snippet_plot.addItem(item)
        self.hmin_hline = pg.InfiniteLine(
            0, angle = 0, movable = True, pen = pg.mkPen('g', width = 2))
        self.hmax_hline = pg.InfiniteLine(
            0, angle = 0, movable = True, pen = pg.mkPen('g', width = 2))
        for ln in (self.hmin_hline, self.hmax_hline):
            # The heights shown are set from hmax (``_range_snippet``),
            # not from the data or the lines
            self.snippet_plot.addItem(ln, ignoreBounds = True)
            ln.sigPositionChanged.connect(self._on_line)
        # Controls under the plots
        self.plots_widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(self.plots_widget)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.plots, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Snippet'))
        self.snip_len = QtWidgets.QDoubleSpinBox()
        self.snip_len.setRange(self.dt * 10, 1e6)
        self.snip_len.setDecimals(3)
        self.snip_len.setSuffix(' s')
        self.snip_len.setValue(self.default_snippet_s())
        self.snip_len.setKeyboardTracking(False)
        self.snip_len.valueChanged.connect(self.update_snippet)
        row.addWidget(self.snip_len)
        self.prev_btn = QtWidgets.QPushButton('< Prev peak')
        self.prev_btn.clicked.connect(lambda: self.jump(-1))
        self.next_btn = QtWidgets.QPushButton('Next peak >')
        self.next_btn.clicked.connect(lambda: self.jump(1))
        row.addWidget(self.prev_btn)
        self.snip_pos = QtWidgets.QSlider(Qt.Horizontal)
        self.snip_pos.setRange(0, SLIDER_STEPS)
        self.snip_pos.valueChanged.connect(self.update_snippet)
        row.addWidget(self.snip_pos, stretch = 1)
        row.addWidget(self.next_btn)
        vbox.addLayout(row)
        self.snip_label = QtWidgets.QLabel()
        self.snip_label.setWordWrap(True)
        vbox.addWidget(self.snip_label)
        vbox.addWidget(hint_label(
            'Snippet: the detection timestream (gray) and the filter '
            'output (blue), in kHz/GHz unless Heights in says photons. '
            'Drag the green lines, click to move the nearer one, or drag '
            'on the plot to set hmin and hmax from the bottom and top of '
            'the drag (a click on a histogram moves its nearer line too); '
            'the slider moves along the record, Prev/Next to the '
            'neighbouring filter peaks. The plots do not pan or zoom; R '
            'rescales them.'))

    ############################################################################
    def _fill_list(self):
        """
        Fill the tone list from the results.
        """
        current = self.tone_list.currentRow()
        self.tone_list.blockSignals(True)
        self.tone_list.clear()
        for row, r_idx in zip(self.rows, self.res_idxs):
            res = self.results.get(row)
            mark = '*' if r_idx in self.thresholds.overrides else ' '
            if isinstance(res, dict):
                text = (f"{mark} res {r_idx}: {len(res['events'])} pulses, "
                        f"{res['rate']:.3g}/s")
            else:
                text = f'{mark} res {r_idx}: failed'
            self.tone_list.addItem(text)
        self.tone_list.blockSignals(False)
        if current >= 0:
            self.tone_list.setCurrentRow(current)

    def current(self):
        """
        Return the selected tone.

        Returns:
        i (int): its index in ``rows``.
        """
        return max(self.tone_list.currentRow(), 0)

    def _on_row_changed(self, i):
        """
        Show the tone chosen in the list, first applying thresholds that
        were changed for the tone shown before but not rerun.

        Parameters:
        i (int): index in ``rows`` (negative: nothing chosen).
        """
        if i < 0:
            return
        prev = self._row
        if prev is not None and prev != i and self.pending(prev):
            self.apply(prev)
        self._row = i
        self.select(i)

    def step_tone(self, step):
        """
        Move to another tone (the A and D keys, the Prev and Next
        buttons), rerunning the current one first if its thresholds were
        changed.

        Parameters:
        step (int): -1 for the previous tone, +1 for the next.
        """
        i = self.current() + step
        if 0 <= i < len(self.rows):
            self.tone_list.setCurrentRow(i)

    def pending(self, i):
        """
        Say whether the boxes hold other thresholds than tone i runs with.

        Parameters:
        i (int): index in ``rows``.

        Returns:
        pending (bool): True if they differ.
        """
        th = self.thresholds.for_tone(self.res_idxs[i])
        vals = self.values()
        init = 1000. if th['init_hmax'] is None else th['init_hmax']
        return not (np.isclose(vals['hmin'], th['hmin']) and
                    np.isclose(vals['hmax'], th['hmax']) and
                    np.isclose(vals['init_hmax'], init))

    def default_snippet_s(self):
        """
        Return the snippet length showing about ``SNIPPET_PULSES`` pulses
        at the tones' median pulse rate.

        Returns:
        length (float): seconds, at least ``SNIPPET_S`` and
            ``SNIPPET_SAMPLES`` samples, at most the record.
        """
        rates = [r['rate'] for r in self.results.values()
                 if isinstance(r, dict) and r['rate'] > 0]
        length = max(SNIPPET_S, SNIPPET_SAMPLES * self.dt)
        if rates:
            length = max(length, SNIPPET_PULSES / float(np.median(rates)))
        if self._record_s:
            length = min(length, self._record_s)
        return length

    def select(self, i):
        """
        Show tone i: its thresholds and plots.

        Parameters:
        i (int): index in ``rows``.
        """
        if i < 0:
            return
        th = self.thresholds.for_tone(self.res_idxs[i])
        self._set_values(th['hmin'], th['hmax'], th['init_hmax'])
        res = self.results.get(self.rows[i])
        if isinstance(res, dict):
            self.status.setText(
                f"mean photon: {res['amp_ph'] * 1e6:.3g} kHz/GHz (1 photon = "
                f"{res['amp_ph']:.3g} in x)\n" + '\n'.join(res['log']))
        else:
            self.status.setText(f'Detection failed: {res}')
        self.update_plots()
        self._show_first_event()

    def in_khz(self):
        """
        Return whether heights are shown in kHz/GHz rather than photons.

        Returns:
        kHz/GHz (bool): True for kHz/GHz.
        """
        return UNITS[self.units_box.currentIndex()][1]

    def _scale(self, res = None):
        """
        Return the factor from kHz/GHz to the shown units.

        Parameters:
        res (dict or None): a tone's result. None (default): the selected
            tone's.

        Returns:
        k (float): 1 for kHz/GHz; one over the tone's mean photon in kHz/GHz for
            photons (1 without a result).
        """
        if self.in_khz():
            return 1.
        if res is None:
            res = self.results.get(self.rows[self.current()])
        return 1. / (float(res['amp_ph']) * PPM) if isinstance(res, dict) \
            else 1.

    def _on_units(self):
        """
        Redraw everything in the chosen units.
        """
        unit = 'kHz/GHz' if self.in_khz() else 'photons'
        self.h_plot.setLabel('bottom', f'Filter height ({unit})')
        self.pulse_plot.setLabel('left', f'Mean pulse ({unit})')
        self.snippet_plot.setLabel('left', f'Height ({unit})')
        self.update_plots()
        self._updating = True
        self._sync_lines()
        self._updating = False
        self.update_snippet()
        self.rescale()

    def rescale(self):
        """
        Fit every plot to its data again (the R key, the Rescale button).
        """
        # Enabling auto-range fits the view at once (autoRange() would
        # set a fixed range and turn auto-range off again)
        for plot in (self.init_plot, self.h_plot, self.pulse_plot):
            plot.enableAutoRange()
        self._range_snippet()

    def keyPressEvent(self, event):
        """
        R: rescale the plots; A and D: the previous and next tone (outside
        the text boxes).

        Parameters:
        event (QKeyEvent): the key event.
        """
        typing = isinstance(self.focusWidget(), (QtWidgets.QLineEdit,
                                                 QtWidgets.QAbstractSpinBox))
        keys = {Qt.Key_R: self.rescale,
                Qt.Key_A: lambda: self.step_tone(-1),
                Qt.Key_D: lambda: self.step_tone(1)}
        if event.key() in keys and not typing:
            keys[event.key()]()
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_values(self, hmin, hmax, init_hmax):
        """
        Set the spin boxes and lines without triggering updates.

        Parameters:
        hmin, hmax (float): height window.
        init_hmax (float or None): first-pass cut.
        """
        self._updating = True
        self.hmin_box.setValue(hmin)
        self.hmax_box.setValue(hmax)
        self.init_hmax_box.setValue(1000. if init_hmax is None else
                                    init_hmax)
        self._sync_lines()
        self._updating = False

    def _sync_lines(self):
        """
        Move every threshold line to its spin box's value, and mark the
        snippet's peaks again.
        """
        k = self._scale()
        for ln in (self.hmin_line, self.hmin_hline):
            ln.setValue(self.hmin_box.value() * k)
        for ln in (self.hmax_line, self.hmax_hline):
            ln.setValue(self.hmax_box.value() * k)
        self.init_line.setValue(self.init_hmax_box.value())
        self._mark_snippet()

    def values(self):
        """
        Return the thresholds in the spin boxes.

        Returns:
        values (dict): 'hmin', 'hmax', 'init_hmax'.
        """
        return dict(hmin = self.hmin_box.value(),
                    hmax = self.hmax_box.value(),
                    init_hmax = self.init_hmax_box.value())

    def _on_spin(self):
        """
        Move the lines to the spin boxes' values.
        """
        if self._updating:
            return
        self._updating = True
        self._sync_lines()
        self._updating = False

    def _click_init(self, x, _y):
        """
        Move the init_hmax line to a click on the first-pass histogram.

        Parameters:
        x (float): the click's height (units of the median).
        _y (float): unused.
        """
        self.init_hmax_box.setValue(max(float(x), 0.1))

    def _drag_init(self, _p0, p1, _finished):
        """
        Carry the init_hmax line along a drag on the first-pass histogram.

        Parameters:
        _p0 (tuple): unused.
        p1 (tuple): the drag's current (x, y).
        _finished (bool): unused.
        """
        self._click_init(*p1)

    def _click_hist(self, x, _y):
        """
        Move the nearer of hmin and hmax to a click on the height
        histogram.

        Parameters:
        x (float): the click's height (shown units).
        _y (float): unused.
        """
        self._set_nearer(x)

    def _drag_hist(self, p0, p1, finished):
        """
        Carry the line nearer to a drag's start along the drag on the
        height histogram.

        Parameters:
        p0, p1 (tuple): the drag's start and current (x, y).
        finished (bool): True at the end of the drag.
        """
        self._set_nearer(p1[0], start = p0[0], finished = finished)

    def _click_snippet(self, _x, y):
        """
        Move the nearer of hmin and hmax to a click's height on the
        snippet.

        Parameters:
        _x (float): unused.
        y (float): the click's height (shown units).
        """
        self._set_nearer(y)

    def _set_nearer(self, value, start = None, finished = True):
        """
        Set whichever of hmin and hmax is nearer to a height. During a
        drag (``start`` given), the box chosen at the drag's start is
        kept until the drag ends.

        Parameters:
        value (float): the height, in the shown units.
        start (float or None): a drag's starting height (shown units).
            None (default): a click, the box nearer to ``value``.
        finished (bool): False while a drag goes on. Default True.
        """
        k = self._scale()
        v = float(value) / k
        box = self._drag_box
        if box is None:
            ref = v if start is None else float(start) / k
            box = self.hmin_box if abs(ref - self.hmin_box.value()) <= \
                abs(ref - self.hmax_box.value()) else self.hmax_box
        self._drag_box = None if finished else box
        box.setValue(v)

    def _drag_snippet(self, p0, p1, _finished):
        """
        Set hmin and hmax from the bottom and top of a drag on the
        snippet (as it goes, so the lines follow the mouse).

        Parameters:
        p0, p1 (tuple): the drag's start and current (x, y) (shown
            units).
        _finished (bool): unused.
        """
        k = self._scale()
        lo, hi = sorted((float(p0[1]) / k, float(p1[1]) / k))
        if hi - lo < 1e-6:
            return
        # The lines are moved under the guard: a moved line reports back
        # to the boxes, and would copy its twin's old value otherwise
        self._updating = True
        self.hmin_box.setValue(lo)
        self.hmax_box.setValue(hi)
        self._sync_lines()
        self._updating = False

    def _on_line(self):
        """
        Copy a dragged line to its spin box (and to the line's twin on the
        other plot).
        """
        if self._updating:
            return
        self._updating = True
        moved = self.sender()
        hmin = self.hmin_hline if moved is self.hmin_hline else \
            self.hmin_line
        hmax = self.hmax_hline if moved is self.hmax_hline else \
            self.hmax_line
        k = self._scale()
        self.hmin_box.setValue(hmin.value() / k)
        self.hmax_box.setValue(hmax.value() / k)
        self.init_hmax_box.setValue(self.init_line.value())
        self._sync_lines()
        self._updating = False

    ############################################################################
    def update_plots(self):
        """
        Draw the histograms and the mean pulse of the selected tone (or the
        histograms of all tones).
        """
        if self.all_tones.isChecked():
            chosen = [self.results.get(r) for r in self.rows]
        else:
            chosen = [self.results.get(self.rows[self.current()])]
        chosen = [r for r in chosen if isinstance(r, dict)]
        init, h, noise_h, noise_w = [], [], [], []
        for res in chosen:
            k = self._scale(res)
            if len(res['init_h']):
                init.append(res['init_h'] / np.median(res['init_h']))
            h.append(res['coinc_h'] * k)
            noise_h.append(res['neg_h'] * k)
            noise_w.append(np.full(len(res['neg_h']),
                                   1 / max(res['clean_frac'], 1e-6)))
        self._hist(self.init_curve, init, np.linspace(0, 10, 201))
        top = max([self.hmax_box.value() * self._scale() * 1.5] +
                  [float(np.percentile(x, 99)) for x in h if len(x)])
        bins = np.linspace(0, top, 151)
        self._hist(self.h_curve, h, bins)
        self._hist(self.noise_curve, noise_h, bins, noise_w)
        if len(chosen) == 1 and len(chosen[0]['template']):
            res = chosen[0]
            t = (np.arange(len(res['template'])) - res['p']) * self.dt
            self.pulse_curve.setData(
                t * 1e3, res['template'] * PPM * self._scale(res))
            self.filter_end_line.setValue(
                res.get('filter_post', res['post']) * self.dt * 1e3)
            self.filter_end_line.show()
            self._fit_pulse(res)
        else:
            self.pulse_curve.setData([], [])
            self.fit_curve.setData([], [])
            self.filter_end_line.hide()
            self.pulse_plot.setTitle('')

    def _fit_pulse(self, res):
        """
        Fit the mean pulse's decay and draw the fit over its range.

        Parameters:
        res (dict): the tone's result (``template``, ``p``, ``amp_ph``).
        """
        s = self._settings
        tau, popt, n = fit_tau(res['template'], res['p'], self.dt,
                               s.tau_skip, s.tau_lifetimes)
        if not np.isfinite(tau):
            self.fit_curve.setData([], [])
            self.pulse_plot.setTitle('Mean pulse: decay fit failed')
            return
        t = np.arange(n) * self.dt
        y = (popt[0] * np.exp(-t / popt[1]) + popt[2]) * PPM \
            * self._scale(res)
        self.fit_curve.setData((t + int(s.tau_skip) * self.dt) * 1e3, y)
        self.pulse_plot.setTitle(
            f'Mean pulse: tau = {tau * 1e6:.3g} us (fit from '
            f'{s.tau_skip} samples after the peak, {n} samples)')

    ############################################################################
    def _tone_data(self):
        """
        Return the selected tone's detection timestream and result.

        Returns:
        d (np.array or None): the timestream (None without data).
        res (dict or None): its result (None if the detection failed).
        """
        row = self.rows[self.current()]
        res = self.results.get(row)
        if self._data is None or not isinstance(res, dict):
            return None, None
        if self._d[0] != row:
            d = self._data(row)
            self._d = (row, d if isinstance(d, np.ndarray) else None)
        return self._d[1], res

    def _snippet_range(self, d):
        """
        Return the snippet's sample range from the controls.

        Parameters:
        d (np.array): the timestream.

        Returns:
        start, stop (int): the range.
        """
        n = len(d)
        length = min(max(int(round(self.snip_len.value() / self.dt)), 10), n)
        start = int(round(self.snip_pos.value() / SLIDER_STEPS *
                          (n - length)))
        return start, start + length

    def _show_first_event(self):
        """
        Put the snippet on the selected tone's first event (or at the
        start without events) and draw it.
        """
        d, res = self._tone_data()
        self.snip_pos.blockSignals(True)
        self.snip_pos.setValue(0)
        if d is not None and len(res['events']):
            self._center_on(int(res['events'][0]), d)
        self.snip_pos.blockSignals(False)
        self.update_snippet()

    def _center_on(self, idx, d):
        """
        Set the slider so the snippet is centered on a sample (without
        redrawing).

        Parameters:
        idx (int): the sample.
        d (np.array): the timestream.
        """
        n = len(d)
        length = min(max(int(round(self.snip_len.value() / self.dt)), 10), n)
        if n <= length:
            self.snip_pos.setValue(0)
            return
        start = min(max(idx - length // 2, 0), n - length)
        self.snip_pos.setValue(int(round(start / (n - length) *
                                         SLIDER_STEPS)))

    def jump(self, direction):
        """
        Move the snippet to the next (or previous) filter peak of the
        selected tone, from the snippet's center (Prev/Next buttons).

        Parameters:
        direction (int): +1 for the next peak, -1 for the previous.
        """
        d, res = self._tone_data()
        if d is None or not len(res['coinc_idx']):
            return
        start, stop = self._snippet_range(d)
        center = (start + stop) // 2
        peaks = np.sort(res['coinc_idx'])
        margin = max((stop - start) // 10, 1)
        if direction > 0:
            later = peaks[peaks > center + margin]
            if not len(later):
                return
            target = later[0]
        else:
            earlier = peaks[peaks < center - margin]
            if not len(earlier):
                return
            target = earlier[-1]
        self.snip_pos.blockSignals(True)
        self._center_on(int(target), d)
        self.snip_pos.blockSignals(False)
        self.update_snippet()

    def update_snippet(self):
        """
        Draw the data snippet of the selected tone: the detection
        timestream and the filter output in photons over the chosen
        range, with the filter peaks marked.
        """
        d, res = self._tone_data()
        if d is None:
            self.snippet = None
            for curve in (self.raw_curve, self.y_curve):
                curve.setData([], [])
            self._mark_snippet()
            self.snip_label.setText(
                'No data for the snippet' if self._data is None or
                not isinstance(res, dict) else 'The tone has no detection '
                'timestream')
            return
        start, stop = self._snippet_range(d)
        raw = int(res['sign']) * np.asarray(d[start:stop], dtype = float) \
            * PPM
        y = filter_snippet(d, res, start, stop)
        t = np.arange(start, stop) * self.dt
        # Kept in kHz/GHz; drawn in the chosen units
        self.snippet = dict(row = self.rows[self.current()], start = start,
                            stop = stop, y = y, raw = raw)
        k = self._scale(res)
        self.raw_curve.setData(t, raw * k)
        self.y_curve.setData(t, y * k)
        self._mark_snippet()
        self._range_snippet()

    def _range_snippet(self):
        """
        Fit the snippet to its time range, with the heights shown up to a
        little above hmax (a cosmic ray would flatten the photons
        otherwise; peaks above the top are drawn at its edge).
        """
        self.snippet_plot.enableAutoRange(x = True)
        k = self._scale()
        hmax = self.hmax_box.value() * k
        self.snippet_plot.setYRange(-0.3 * hmax, 1.3 * hmax, padding = 0)

    def _mark_snippet(self):
        """
        Mark the filter peaks in the snippet by the current thresholds:
        events in the height window, peaks below hmin, peaks above hmax.
        """
        snip = self.snippet
        if snip is None:
            for marks in (self.event_marks, self.low_marks, self.high_marks):
                marks.setData([], [])
            return
        res = self.results.get(snip['row'])
        idx = np.asarray(res['coinc_idx'])
        h = np.asarray(res['coinc_h'])
        inside = (idx >= snip['start']) & (idx < snip['stop'])
        idx, h = idx[inside], h[inside]
        hmin, hmax = self.hmin_box.value(), self.hmax_box.value()
        t = idx * self.dt
        k = self._scale(res)
        groups = ((self.low_marks, h < hmin),
                  (self.event_marks, (h >= hmin) & (h <= hmax)),
                  (self.high_marks, h > hmax))
        counts = []
        top = 1.3 * hmax * k * 0.97
        for marks, pick in groups:
            marks.setData(t[pick], np.minimum(h[pick] * k, top))
            counts.append(int(pick.sum()))
        self.snip_label.setText(
            f"{snip['start'] * self.dt:.4g}-{snip['stop'] * self.dt:.4g} s "
            f"of {len(self._d[1]) * self.dt:.4g} s: {counts[1]} pulses in "
            f'[{hmin:g}, {hmax:g}], {counts[0]} peaks below hmin, '
            f'{counts[2]} above hmax')

    @staticmethod
    def _hist(curve, parts, bins, weights = None):
        """
        Draw a step histogram (counts below 1 are drawn at 0.5 for the log
        axis).

        Parameters:
        curve (PlotDataItem): the curve.
        parts (list of np.array): values to histogram together.
        bins (np.array): bin edges.
        weights (list of np.array or None): weights. Default None.
        """
        if not parts:
            curve.setData([], [])
            return
        x = np.concatenate(parts)
        w = None if weights is None else np.concatenate(weights)
        counts, _ = np.histogram(x, bins = bins, weights = w)
        curve.setData(bins, np.maximum(counts, 0.5))

    ############################################################################
    def apply(self, i = None):
        """
        Apply the boxes' thresholds (to tone i as its own, or to all
        tones as the global values, by the scope chosen) and run the
        detection again where they changed.

        Parameters:
        i (int or None): index in ``rows``. None (default): the selected
            tone.

        Returns:
        rows (list of int): the DataSet rows rerun.
        """
        i = self.current() if i is None else int(i)
        vals = self.values()
        th = self.thresholds
        if self.scope_tone.isChecked():
            th.overrides[self.res_idxs[i]] = vals
            rows = [self.rows[i]]
        else:
            th.hmin, th.hmax = vals['hmin'], vals['hmax']
            th.init_hmax = vals['init_hmax']
            rows = [r for r, ri in zip(self.rows, self.res_idxs)
                    if ri not in th.overrides]
        self.status.setText(f'Running {len(rows)} tones...')
        new = run_responsive(self._rerun, rows, th)
        self.results.update(new)
        self._fill_list()
        return rows

    def rerun(self):
        """
        Apply the thresholds (to the selected tone or to all tones) and run
        the detection again where they changed (the Rerun button).
        """
        self.apply()
        self.select(self.current())

    def reset_tone(self):
        """
        Put the default thresholds (the ones the window opened with) back
        on the selected tone and run it again: its own thresholds are
        dropped if the global ones are the defaults, else set to them.
        """
        i = self.current()
        r_idx, row = self.res_idxs[i], self.rows[i]
        th = self.thresholds
        glob = dict(hmin = th.hmin, hmax = th.hmax, init_hmax = th.init_hmax)
        if glob == self._defaults:
            if th.overrides.pop(r_idx, None) is None and not self.pending(i):
                return
        else:
            th.overrides[r_idx] = dict(self._defaults)
        self.status.setText('Running 1 tone...')
        self.results.update(run_responsive(self._rerun, [row], th))
        self._fill_list()
        self.select(i)

def run_photon_thresholds(
    ts, results, thresholds, rerun, data = None, settings = None,
    title = 'Photon thresholds', ui_scale = 1.0
):
    """
    Open the photon threshold window and wait until it is closed.

    Parameters:
    ts, results, thresholds, rerun, data, settings, title, ui_scale: see
        ``PhotonThresholdWindow``.

    Returns:
    thresholds (Thresholds): the chosen thresholds.
    results (dict): the detection results with them.
    """
    app = get_qapp(title)
    win = PhotonThresholdWindow(ts, results, thresholds, rerun, data = data,
                                settings = settings, title = title,
                                ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.thresholds, win.results
