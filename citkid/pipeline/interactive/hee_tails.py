"""
High-energy event (HEE) tail window: review the tail fits of a
timestream's HEEs (``pipeline.hee_finder``) tone by tone, and change
the threshold, the tail level and the fit length.

Left: the tone list (``*``: reviewed, ``x``: bad), the threshold (the
lowest peak height of a pulse that is averaged), the tail level (where
the pulses are lined up and the fit starts) and the fit length, their
scope (this tone, or all tones), Apply, mark bad (with a reason), add a
note, the tone overview (``photon_thresholds.ToneOverviewWindow``:
histograms over all tones of the median height, lifetime and pulse
rate, with review limits), rescale and Done. Right: the tone's whole
timestream with the pulses found (green: averaged; red: left out) and
the threshold line; the aligned pulses (blue) with their mean (white),
the fit (yellow) and the level line. Dragging a line changes its box;
Apply refits the tone(s) from the pulses already cut
(``hee_finder.refit_timestream``), so it is immediate. The plots pan
and zoom with the mouse; R puts them back (the timestream's heights up
to a little above the 99th percentile of the pulses; the mean pulse
over the fit range, up to a little above its own peak).

Keys: A and D move through the tones, B marks the selected tone bad,
T adds a note, O opens the overview, R rescales the plots, Ctrl+R
applies.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
from ...qt_compat import (
    PlainAxisItem, Qt, delete_on_close, fit_window_to_screen, forward_keys,
    freeze_heights, get_qapp, widget_font_stylesheet
)
from ...signal.photons import PPM
from ... import daq_layout as dl
from .marks import ask_note, ask_reason, mark_label
from .photon_thresholds import ToneOverviewWindow, _sync_box

# The overview's quantities: key, axis label
OVERVIEW = (('height', 'Median pulse height (kHz/GHz)'),
            ('tau', 'Lifetime of the mean tail (us)'),
            ('rate', 'Pulse rate (1/s)'))
# Most pulses drawn in the pulse plot (the mean uses all of them)
MAX_DRAWN = 200
# Bins of the timestream envelope
ENVELOPE_BINS = 4000


def x_label(sign, unit = 'kHz/GHz'):
    """
    Return a y-axis label for the fractional frequency shift x as drawn:
    x itself, or -x when the sign is flipped so pulses go up.

    Parameters:
    sign (int): 1 for x, -1 for -x.
    unit (str): the unit shown. Default 'kHz/GHz'.

    Returns:
    label (str): e.g. '-x (kHz/GHz)'.
    """
    return f"{'-x' if int(sign) < 0 else 'x'} ({unit})"


def envelope(d, nbins = ENVELOPE_BINS):
    """
    Reduce a long timestream to a min/max envelope for drawing.

    Parameters:
    d (np.array): the timestream.
    nbins (int): bins. Default ``ENVELOPE_BINS``. A timestream shorter
        than 2 * nbins is returned as it is.

    Returns:
    idx (np.array): sample index of each point.
    y (np.array): the points (each bin's minimum then maximum).
    """
    d = np.asarray(d, dtype = float)
    n = len(d)
    if n <= 2 * nbins:
        return np.arange(n), d
    width = n // nbins
    m = width * nbins
    blocks = d[:m].reshape(nbins, width)
    idx = np.repeat(np.arange(nbins) * width + width / 2, 2)
    y = np.column_stack((blocks.min(axis = 1), blocks.max(axis = 1))).ravel()
    return idx, y


class HeeTailWindow(QtWidgets.QMainWindow):
    """
    Window for reviewing the HEE tail fits of a timestream.

    Attributes:
    results (dict): row: ``hee_finder.analyze_tone`` result or error.
    thresholds (TailThresholds): the thresholds (changed in place).
    rows, res_idxs (list of int): the tones.
    dt (float): sample period, s.
    overview (ToneOverviewWindow or None): the overview, once opened.
    """

    def __init__(
        self, ts, results, thresholds, refit, data = None,
        title = 'HEE tails', ui_scale = 1.0, root = None, stage = ''
    ):
        """
        Build the window.

        Parameters:
        ts (dict): the timestream (``photon_finder.list_timestreams``).
        results (dict): its ``hee_finder.analyze_timestream`` results.
        thresholds (TailThresholds): starting thresholds (changed in
            place).
        refit (callable): function of (rows, thresholds) returning the
            new results of those rows (dict row: result), e.g. a wrapper
            of ``hee_finder.refit_timestream``.
        data (callable or None): function of a row returning its
            detection timestream (np.array), or an error message (str),
            for the timestream plot. Default None: no timestream plot.
        title (str): window title. Default 'HEE tails'.
        ui_scale (float): font size multiplier. Default 1.0.
        root (zarr.Group or None): the store root, writable, for the
            tone-marks registry. Default None: no registry.
        stage (str): the result's path, recorded with the marks. Default
            ''.
        """
        super().__init__()
        self.setWindowTitle(title)
        self._ui_scale = float(ui_scale) if ui_scale else 1.0
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.root, self.stage = root, stage
        self.results = dict(results)
        self.thresholds = thresholds
        self.rows = [int(r) for r in ts['rows']]
        self.res_idxs = [int(r) for r in ts['res_idxs']]
        self.dt = float(ts['dt'])
        self._record_s = float(ts['stop'] - ts['start']) * self.dt \
            if 'start' in ts and 'stop' in ts else None
        self._refit = refit
        self._data = data
        self._d = (None, None)      # (row, timestream) cached
        self._row = None
        self._updating = False
        self._list_rows = []
        self._only_outside = False
        self._include_bad = True
        self.overview = None
        self._ask_reason = self.ask_bad_reason
        self._ask_note = lambda: ask_note(
            self, f'Note on res {self.res_idxs[self.current()]}')
        self._build_controls()
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        self._fill_list()
        self.tone_list.setCurrentRow(0)

    ############################################################################
    def _build_controls(self):
        """
        Build the left panel: tone list, the boxes, and the buttons.
        """
        self._panel = QtWidgets.QWidget()
        self._panel.setFixedWidth(int(330 * self._ui_scale))
        lay = QtWidgets.QVBoxLayout(self._panel)
        self.list_label = QtWidgets.QLabel('Tones (* : reviewed, x : bad)')
        lay.addWidget(self.list_label)
        self.only_outside = QtWidgets.QCheckBox(
            'Show only flagged tones (outside the overview limits)')
        self.only_outside.toggled.connect(self.set_only_outside)
        lay.addWidget(self.only_outside)
        self.include_bad = QtWidgets.QCheckBox('Include tones marked bad')
        self.include_bad.setChecked(True)
        self.include_bad.toggled.connect(self.set_include_bad)
        lay.addWidget(self.include_bad)
        self.tone_list = QtWidgets.QListWidget()
        self.tone_list.currentRowChanged.connect(self._on_row_changed)
        # The shortcuts keep working while the list has the focus
        forward_keys(self.tone_list, self)
        lay.addWidget(self.tone_list, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.prev_tone_btn = QtWidgets.QPushButton('< Prev (A)')
        self.prev_tone_btn.clicked.connect(lambda: self.step_tone(-1))
        self.next_tone_btn = QtWidgets.QPushButton('Next (D) >')
        self.next_tone_btn.clicked.connect(lambda: self.step_tone(1))
        row.addWidget(self.prev_tone_btn)
        row.addWidget(self.next_tone_btn)
        lay.addLayout(row)
        form = QtWidgets.QFormLayout()

        def spin(lo, hi, step):
            """
            Make a spin box.

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

        self.hmin_box = spin(0., 1e4, 0.5)
        self.level_box = spin(0., 1e4, 0.5)
        self.fit_box = spin(0.01, 1000., 0.1)
        form.addRow('threshold (kHz/GHz)', self.hmin_box)
        form.addRow('tail level (kHz/GHz)', self.level_box)
        form.addRow('fit length (ms)', self.fit_box)
        lay.addLayout(form)
        self.scope_tone = QtWidgets.QRadioButton('Apply to this tone')
        self.scope_all = QtWidgets.QRadioButton('Apply to all tones')
        self.scope_tone.setChecked(True)
        lay.addWidget(self.scope_tone)
        lay.addWidget(self.scope_all)
        self.apply_btn = QtWidgets.QPushButton('Apply (Ctrl+R)')
        self.apply_btn.clicked.connect(self.rerun)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, self.rerun)
        lay.addWidget(self.apply_btn)
        self.bad_btn = QtWidgets.QPushButton('Mark tone bad (B)')
        self.bad_btn.clicked.connect(self.toggle_bad)
        self.note_btn = QtWidgets.QPushButton('Add note (T)')
        self.note_btn.clicked.connect(self.add_note)
        self.overview_btn = QtWidgets.QPushButton('Tone overview (O)')
        self.overview_btn.clicked.connect(self.open_overview)
        self.rescale_btn = QtWidgets.QPushButton('Rescale plots (R)')
        self.rescale_btn.clicked.connect(self.rescale)
        self.done_btn = QtWidgets.QPushButton('Done')
        self.done_btn.clicked.connect(self.close)
        for b in (self.bad_btn, self.note_btn, self.overview_btn,
                  self.rescale_btn, self.done_btn):
            lay.addWidget(b)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setFixedHeight(int(80 * self._ui_scale))
        self.status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.status)
        freeze_heights(lay, skip = (self.tone_list,))

    def _build_plots(self):
        """
        Build the plots: the whole timestream with the pulses found and
        the threshold line, and the aligned pulses with their mean, the
        fit and the level line.
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.ts_plot = self.plots.addPlot(
            row = 0, col = 0, axisItems = {'bottom': PlainAxisItem('bottom'),
                                           'left': PlainAxisItem('left')})
        self.ts_plot.setLabel('bottom', 'Time (s)')
        self.ts_plot.setLabel('left', x_label(-1))
        self.ts_plot.addLegend(offset = (-10, 10))
        self.ts_curve = self.ts_plot.plot(pen = pg.mkPen((150, 150, 150)),
                                          name = 'data')
        self.used_marks = pg.ScatterPlotItem(
            size = 9, symbol = 'o', pen = pg.mkPen('g', width = 2),
            brush = pg.mkBrush(0, 0, 0, 0), name = 'averaged')
        self.other_marks = pg.ScatterPlotItem(
            size = 9, symbol = 'x', pen = pg.mkPen('r', width = 2),
            brush = pg.mkBrush('r'), name = 'left out')
        for item in (self.used_marks, self.other_marks):
            self.ts_plot.addItem(item)
        self.pulse_plot = self.plots.addPlot(row = 1, col = 0)
        self.pulse_plot.setLabel('bottom', 'Time from the tail start (ms)')
        self.pulse_plot.setLabel('left', x_label(-1))
        self.pulse_plot.addLegend(offset = (-10, 10))
        self.pulses_curve = self.pulse_plot.plot(
            pen = pg.mkPen((80, 160, 255, 110)), name = 'pulses',
            connect = 'finite')
        self.avg_curve = self.pulse_plot.plot(
            pen = pg.mkPen('w', width = 2), name = 'mean',
            connect = 'finite')
        self.fit_curve = self.pulse_plot.plot(
            pen = pg.mkPen('y', width = 2, style = Qt.DashLine),
            name = 'fit')
        self.anchor_line = pg.InfiniteLine(
            0, angle = 90, pen = pg.mkPen((180, 180, 180),
                                          style = Qt.DashLine))
        self.pulse_plot.addItem(self.anchor_line, ignoreBounds = True)

        def line(plot, color):
            """
            Add a draggable horizontal line.

            Parameters:
            plot (PlotItem): the plot.
            color (str): pen color.

            Returns:
            line (InfiniteLine): the line.
            """
            ln = pg.InfiniteLine(0, angle = 0, movable = True,
                                 pen = pg.mkPen(color, width = 2))
            plot.addItem(ln)
            ln.sigPositionChanged.connect(self._on_line)
            return ln

        self.hmin_line = line(self.ts_plot, 'g')
        self.level_line = line(self.pulse_plot, 'm')

    ############################################################################
    def _list_text(self, i):
        """
        Return a tone's line in the list.

        Parameters:
        i (int): index in ``rows``.

        Returns:
        text (str): the line.
        """
        r_idx = self.res_idxs[i]
        res = self.results.get(self.rows[i])
        bad = self.thresholds.is_bad(r_idx)
        mark = 'x' if bad else '*' if r_idx in self.thresholds.reviewed \
            else ' '
        if bad:
            reason = self.thresholds.bad[int(r_idx)]
            return f'{mark} res {r_idx}: bad' + (f' ({reason})' if reason
                                                 else '')
        if isinstance(res, dict):
            tau = f"{res['tau'] * 1e6:.3g} us" if np.isfinite(res['tau']) \
                else 'no fit'
            return (f"{mark} res {r_idx}: {res['n_used']} of "
                    f"{len(res['h'])} pulses, {tau}")
        return f'{mark} res {r_idx}: failed'

    def _fill_list(self):
        """
        Fill the tone list from the results: every tone, or only the
        tones outside the overview's limits, keeping the selection (the
        selected tone stays listed).
        """
        selected = self._list_rows[self.tone_list.currentRow()] \
            if self._list_rows and self.tone_list.currentRow() >= 0 else None
        outside = self.outside_limits() if self._only_outside else None
        if outside is not None and selected is not None:
            outside = set(outside) | {selected}
        self.tone_list.blockSignals(True)
        self.tone_list.clear()
        self._list_rows = []
        for i in range(len(self.rows)):
            if outside is not None and i not in outside:
                continue
            if not self._include_bad and i != selected and \
                    self.thresholds.is_bad(self.res_idxs[i]):
                continue
            self.tone_list.addItem(self._list_text(i))
            self._list_rows.append(i)
        self.tone_list.blockSignals(False)
        n = len(self._list_rows)
        self.list_label.setText(
            f'Tones (* : reviewed, x : bad): {n} of {len(self.rows)}'
            if outside is None else
            f'Tones outside the limits: {n} of {len(self.rows)}')
        if selected in self._list_rows:
            self.tone_list.setCurrentRow(self._list_rows.index(selected))
        elif n:
            self.tone_list.setCurrentRow(0)

    def current(self):
        """
        Return the selected tone.

        Returns:
        i (int): index in ``rows`` (-1 if none).
        """
        li = self.tone_list.currentRow()
        return self._list_rows[li] if 0 <= li < len(self._list_rows) \
            else -1

    def _on_row_changed(self, li):
        """
        Show the tone chosen in the list, marking the one left as
        reviewed.

        Parameters:
        li (int): list row (negative: nothing chosen).
        """
        if li < 0 or li >= len(self._list_rows):
            return
        prev = self._row
        i = self._list_rows[li]
        if prev is not None and prev != i:
            self._set_reviewed(prev)
        self._row = i
        self.select(i)

    def _set_reviewed(self, i):
        """
        Mark a tone reviewed (looked at and left, or given its own
        values) and show the ``*`` in the list.

        Parameters:
        i (int): index in ``rows``.
        """
        self.thresholds.mark_reviewed(self.res_idxs[i])
        if i in self._list_rows:
            item = self.tone_list.item(self._list_rows.index(i))
            if item is not None:
                item.setText(self._list_text(i))

    def closeEvent(self, event):
        """
        Count the tone shown as reviewed, then close.

        Parameters:
        event (QCloseEvent): the close event.
        """
        if self._row is not None:
            self._set_reviewed(self._row)
        super().closeEvent(event)

    def step_tone(self, step):
        """
        Move to another tone of the list (A and D, Prev and Next).

        Parameters:
        step (int): -1 for the previous tone, +1 for the next.
        """
        li = self.tone_list.currentRow() + step
        if 0 <= li < len(self._list_rows):
            self.tone_list.setCurrentRow(li)

    def select_tone(self, i):
        """
        Select a tone by its index (e.g. from the overview), listing
        every tone again if it is filtered out.

        Parameters:
        i (int): index in ``rows``.
        """
        if i not in self._list_rows:
            self.set_only_outside(False)
        if i not in self._list_rows:
            self.set_include_bad(True)
        if i in self._list_rows:
            self.tone_list.setCurrentRow(self._list_rows.index(i))

    def set_only_outside(self, only):
        """
        List every tone, or only the tones outside the review limits
        (the box above the tone list).

        Parameters:
        only (bool): True for only the tones outside.
        """
        self._only_outside = bool(only)
        _sync_box(self.only_outside, self._only_outside)
        self._fill_list()

    def set_include_bad(self, include):
        """
        List the tones marked bad, or leave them out (the box above the
        tone list).

        Parameters:
        include (bool): True to list them.
        """
        self._include_bad = bool(include)
        _sync_box(self.include_bad, self._include_bad)
        self._fill_list()

    def select(self, i):
        """
        Show tone i: its values, plots and summary.

        Parameters:
        i (int): index in ``rows``.
        """
        if i < 0:
            return
        th = self.thresholds.for_tone(self.res_idxs[i])
        self._set_values(th)
        res = self.results.get(self.rows[i])
        bad = self.thresholds.is_bad(self.res_idxs[i])
        self.bad_btn.setText('Clear bad mark (B)' if bad
                             else 'Mark tone bad (B)')
        head = ('MARKED BAD: ' + (self.thresholds.bad[int(self.res_idxs[i])]
                                  or 'no reason given') + '\n') if bad \
            else ''
        marks = mark_label(self.root, self.res_idxs[i])
        if marks and not bad:
            head += marks + '\n'
        elif marks and 'note' in marks:
            head += marks.split(', ')[-1] + '\n'
        if isinstance(res, dict):
            tau = f"{res['tau'] * 1e6:.3g} us over {res['n_fit']} samples" \
                if np.isfinite(res['tau']) else 'failed'
            self.status.setText(
                head + f"{len(res['h'])} pulses, {res['n_used']} averaged "
                f"(peak above the threshold and reaching the level)\n"
                f'lifetime of the mean tail: {tau}')
        else:
            self.status.setText(head + f'Failed: {res}')
        self.update_plots()

    ############################################################################
    def _set_values(self, th):
        """
        Put values in the boxes and lines.

        Parameters:
        th (dict): 'hmin', 'level', 'fit_s'.
        """
        self._updating = True
        self.hmin_box.setValue(th['hmin'])
        self.level_box.setValue(th['level'])
        self.fit_box.setValue(th['fit_s'] * 1e3)
        self._sync_lines()
        self._updating = False

    def _sync_lines(self):
        """
        Move the lines to the boxes' values.
        """
        self.hmin_line.setValue(self.hmin_box.value())
        self.level_line.setValue(self.level_box.value())

    def values(self):
        """
        Return the values in the boxes.

        Returns:
        values (dict): 'hmin', 'level' (kHz/GHz), 'fit_s' (s).
        """
        return dict(hmin = float(self.hmin_box.value()),
                    level = float(self.level_box.value()),
                    fit_s = float(self.fit_box.value()) * 1e-3)

    def _on_spin(self):
        """
        Move the lines to the spin boxes' values.
        """
        if self._updating:
            return
        self._updating = True
        self._sync_lines()
        self._updating = False

    def _on_line(self):
        """
        Put a dragged line's value in its box.
        """
        if self._updating:
            return
        self._updating = True
        for ln, box in ((self.hmin_line, self.hmin_box),
                        (self.level_line, self.level_box)):
            if ln is self.sender():
                box.setValue(float(ln.value()))
        self._sync_lines()
        self._updating = False

    def keyPressEvent(self, event):
        """
        R: rescale; B: mark bad; T: note; O: overview; A and D: the
        previous and next tone (outside the text boxes).

        Parameters:
        event (QKeyEvent): the key event.
        """
        typing = isinstance(self.focusWidget(), (QtWidgets.QLineEdit,
                                                 QtWidgets.QAbstractSpinBox))
        keys = {Qt.Key_R: self.rescale,
                Qt.Key_B: self.toggle_bad,
                Qt.Key_T: self.add_note,
                Qt.Key_O: self.open_overview,
                Qt.Key_A: lambda: self.step_tone(-1),
                Qt.Key_D: lambda: self.step_tone(1)}
        if event.key() in keys and not typing:
            keys[event.key()]()
            event.accept()
            return
        super().keyPressEvent(event)

    def rescale(self):
        """
        Put the plots back on their default ranges (R): the whole record,
        with the heights up to a little above the 99th percentile of the
        pulses (a few huge pulses don't set the scale; scroll to zoom,
        drag to pan, right-drag to zoom one axis), and the mean pulse
        over the fit range, up to a little above the mean's own peak
        (not the single pulses' peaks).
        """
        res = self.results.get(self.rows[self.current()])
        self.ts_plot.enableAutoRange(x = True)
        if isinstance(res, dict) and len(res['h']):
            top = max(float(np.percentile(res['h'], 99)),
                      2 * self.hmin_box.value(), 1.)
            self.ts_plot.setYRange(-0.2 * top, 1.2 * top, padding = 0)
        else:
            self.ts_plot.enableAutoRange(y = True)
        avg = np.asarray(res['avg'], dtype = float) * PPM \
            if isinstance(res, dict) else np.zeros(0)
        if np.isfinite(avg).any():
            fit_ms = float(res['fit_s']) * 1e3
            peak = float(np.nanmax(avg))
            self.pulse_plot.setXRange(-0.1 * fit_ms, 1.1 * fit_ms,
                                      padding = 0)
            self.pulse_plot.setYRange(-0.2 * peak, 1.2 * peak, padding = 0)
        else:
            self.pulse_plot.enableAutoRange()

    ############################################################################
    def _tone_data(self, row):
        """
        Return a tone's detection timestream, cached.

        Parameters:
        row (int): DataSet row.

        Returns:
        d (np.array or None): the timestream (None without data).
        """
        if self._data is None:
            return None
        if self._d[0] != row:
            d = self._data(row)
            self._d = (row, d if isinstance(d, np.ndarray) else None)
        return self._d[1]

    def update_plots(self):
        """
        Draw the selected tone: its timestream with the pulses found,
        and its aligned pulses with their mean and the fit.
        """
        row = self.rows[self.current()]
        res = self.results.get(row)
        d = self._tone_data(row)
        # Pulses are drawn positive: the axes say whether that is x or -x
        sign = int(res['sign']) if isinstance(res, dict) else -1
        self.ts_plot.setLabel('left', x_label(sign))
        self.pulse_plot.setLabel('left', x_label(sign))
        if d is not None:
            idx, y = envelope(d)
            self.ts_curve.setData(idx * self.dt, sign * y * PPM)
        else:
            self.ts_curve.setData([], [])
        if not isinstance(res, dict) or not len(res['h']):
            for item in (self.used_marks, self.other_marks):
                item.setData([], [])
            for curve in (self.pulses_curve, self.avg_curve, self.fit_curve):
                curve.setData([], [])
            self.pulse_plot.setTitle('')
            return
        t = (np.asarray(res['t0']) + np.asarray(res['p']) - res['pre']) \
            * self.dt
        h = np.asarray(res['h'], dtype = float)
        used = np.asarray(res['used'], dtype = bool)
        self.used_marks.setData(t[used], h[used])
        self.other_marks.setData(t[~used], h[~used])
        self.rescale()
        aligned = np.asarray(res['aligned'], dtype = float) * PPM
        tt = (np.arange(aligned.shape[1]) - res['anchor']) * self.dt * 1e3
        if len(aligned):
            chosen = aligned[:MAX_DRAWN]
            block = np.full((len(chosen), aligned.shape[1] + 1), np.nan)
            block[:, :-1] = chosen
            self.pulses_curve.setData(np.tile(np.r_[tt, np.nan],
                                              len(chosen)), block.ravel())
        else:
            self.pulses_curve.setData([], [])
        avg = np.asarray(res['avg'], dtype = float) * PPM
        self.avg_curve.setData(tt, avg if np.isfinite(avg).any() else [])
        if np.isfinite(res['tau']) and res['n_fit'] > 0:
            A, tau, B = res['popt']
            tf = np.arange(res['n_fit']) * self.dt
            self.fit_curve.setData(tf * 1e3, (A * np.exp(-tf / tau) + B)
                                   * PPM)
            self.pulse_plot.setTitle(f'tau = {tau * 1e6:.3g} us '
                                     f"({res['n_used']} pulses averaged)")
        else:
            self.fit_curve.setData([], [])
            self.pulse_plot.setTitle('decay fit failed' if res['n_used']
                                     else 'no pulses to average')

    ############################################################################
    def apply(self, i = None):
        """
        Apply the boxes' values (to tone i as its own, or to all tones
        as the global values, by the scope chosen) and refit.

        Parameters:
        i (int or None): index in ``rows``. None (default): the selected
            tone.

        Returns:
        rows (list of int): the DataSet rows refitted.
        """
        i = self.current() if i is None else int(i)
        vals = self.values()
        th = self.thresholds
        if self.scope_tone.isChecked():
            th.overrides.setdefault(self.res_idxs[i], {}).update(vals)
            th.mark_reviewed(self.res_idxs[i])
            rows = [self.rows[i]]
        else:
            th.set_all(**vals)
            rows = list(self.rows)
            self.scope_tone.setChecked(True)
        self.results.update(self._refit(rows, th))
        self._fill_list()
        self._refresh_overview()
        return rows

    def rerun(self):
        """
        Apply the values and refit (the Apply button, Ctrl+R).
        """
        self.apply()
        self.select(self.current())

    def toggle_bad(self):
        """
        Mark the selected tone bad, asking for a reason, or clear its
        mark (the button, or B).
        """
        i = self.current()
        if i < 0:
            return
        r_idx = self.res_idxs[i]
        if self.thresholds.is_bad(r_idx):
            self.thresholds.clear_bad(r_idx)
            self._record('clear', '')
        else:
            reason = self._ask_reason()
            if reason is None:
                return
            self.thresholds.set_bad(r_idx, reason)
            self._record('bad', reason)
        self._fill_list()
        self.select(i)
        self._refresh_overview()

    def add_note(self):
        """
        Record a note on the selected tone in the tone-marks registry
        (the button, or T). The tone is kept.
        """
        i = self.current()
        if i < 0:
            return
        note = self._ask_note()
        if note:
            self._record('note', note)
            self.select(i)

    def _record(self, kind, text):
        """
        Write a mark on the selected tone to the registry, if there is
        one.

        Parameters:
        kind (str): 'bad', 'note' or 'clear'.
        text (str): the reason or note.
        """
        if self.root is not None:
            dl.add_tone_mark(self.root, self.res_idxs[self.current()], kind,
                             text, self.stage)

    def ask_bad_reason(self):
        """
        Ask why the selected tone is bad: one of the registry's reasons,
        or typed.

        Returns:
        reason (str or None): the reason, or None if cancelled.
        """
        return ask_reason(self, f'Mark res {self.res_idxs[self.current()]} '
                          'bad')

    ############################################################################
    def overview_values(self):
        """
        Return every tone's values for the overview: the median peak
        height of the averaged pulses, the mean tail's lifetime, and
        the rate of averaged pulses (per second of the record, so tones
        and devices compare).

        Returns:
        values (dict): 'height' (kHz/GHz), 'tau' (s), 'rate' (1/s):
            np.array per tone, NaN for tones without a result.
        """
        n = len(self.rows)
        out = {k: np.full(n, np.nan) for k, _ in OVERVIEW}
        for i, row in enumerate(self.rows):
            res = self.results.get(row)
            if not isinstance(res, dict) or not res['n_used']:
                continue
            used = np.asarray(res['used'], dtype = bool)
            out['height'][i] = np.median(np.asarray(res['h'])[used])
            out['tau'][i] = res['tau']
            out['rate'][i] = res['n_used'] / self._record_s \
                if self._record_s else np.nan
        return out

    def outside_limits(self, values = None):
        """
        Find the tones outside the overview's review limits (or without
        a value), and the tones marked bad.

        Parameters:
        values (dict or None): ``overview_values`` output. Default None:
            computed.

        Returns:
        outside (set of int): indices in ``rows``.
        """
        values = self.overview_values() if values is None else values
        outside = {i for i, r in enumerate(self.res_idxs)
                   if self.thresholds.is_bad(r)}
        for key, _ in OVERVIEW:
            v = values[key]
            lim = self.thresholds.limits.get(key)
            bad = ~np.isfinite(v)
            if lim is not None:
                tol = [1e-9 * max(abs(x), 1e-300) for x in lim]
                bad |= (v < lim[0] - tol[0]) | (v > lim[1] + tol[1])
            outside |= set(np.flatnonzero(bad).tolist())
        return outside

    def open_overview(self):
        """
        Open the tone overview window, or bring it to the front (the
        button, or O).
        """
        if self.overview is None:
            self.overview = ToneOverviewWindow(self, items = OVERVIEW)
        self.overview.refresh()
        self.overview.show()
        self.overview.raise_()
        self.overview.activateWindow()

    def _refresh_overview(self):
        """
        Refresh the overview window if it is open.
        """
        if self.overview is not None and self.overview.isVisible():
            self.overview.refresh()


def run_hee_tails(
    ts, results, thresholds, refit, data = None, title = 'HEE tails',
    ui_scale = 1.0, root = None, stage = ''
):
    """
    Open the HEE tail window and wait until it is closed.

    Parameters:
    ts, results, thresholds, refit, data, title, ui_scale, root, stage:
        see ``HeeTailWindow``.

    Returns:
    thresholds (TailThresholds): the chosen thresholds.
    results (dict): the results with them.
    """
    app = get_qapp(title)
    win = HeeTailWindow(ts, results, thresholds, refit, data = data,
                           title = title, ui_scale = ui_scale, root = root,
                           stage = stage)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.thresholds, win.results
