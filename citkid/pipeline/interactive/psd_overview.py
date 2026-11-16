"""
PSD overview window: histograms over all tones of a timestream's noise
spectra at chosen frequencies (``pipeline.psds``), with review limits,
the selected tone's spectrum, and marking tones bad with a reason.

Left: three histograms of Sxx (1/Hz, on a log axis labelled 1e-16,
1e-15, ...) at frequencies picked from the reduced grid (before or after
common-mode removal), each with a low and a high review limit (red,
draggable); the histograms do not pan or zoom.
Right: the tones outside the limits (or every tone), the selected tone's
binned spectrum (before and after common-mode removal; this plot pans
and zooms), Mark bad (B, with a reason), Add note (T) and Done. Bad
marks go to the result and to the store's tone marks.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from ...qt_compat import (
    Qt, delete_on_close, fit_window_to_screen, forward_keys, freeze_heights,
    get_qapp, widget_font_stylesheet
)
from .marks import ask_note, ask_reason, mark_label, record, registry

# Histograms shown
N_HIST = 3


class LogAxisItem(pg.AxisItem):
    """
    Axis of a log-scaled plot with plain tick labels: 1e-16, 1e-15, ...
    (or 1, 10, 100 for frequencies), not powers with superscripts.

    Attributes:
    fmt (str): 'e' for 1e-16 style labels, 'g' for 1, 10, 100.
    """

    def __init__(self, orientation, fmt = 'e', **kwargs):
        """
        Make the axis.

        Parameters:
        orientation (str): 'bottom' or 'left'.
        fmt (str): 'e' (default) or 'g', see the class.
        **kwargs: ``AxisItem`` arguments.
        """
        super().__init__(orientation, **kwargs)
        self.fmt = fmt

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        """
        Return the ticks of a log axis: the decades in view, no ticks in
        between (the labels would overlap).

        Parameters:
        minVal, maxVal (float): the view range, log10 of the values.
        size (float): axis length in pixels (unused).
        stdTicks (list): the linear ticks (unused).

        Returns:
        ticks (list): one level, ``(spacing, [positions])``.
        """
        lo, hi = int(np.ceil(minVal)), int(np.floor(maxVal))
        if hi < lo:
            lo, hi = int(np.floor(minVal)), int(np.ceil(maxVal))
        return [(1.0, [float(v) for v in range(lo, hi + 1)])]

    def logTickStrings(self, values, scale, spacing):
        """
        Return the tick labels of a log axis.

        Parameters:
        values (list of float): tick positions, log10 of the values.
        scale (float): unused.
        spacing (float): unused.

        Returns:
        strings (list of str): the labels.
        """
        return [f'{10. ** v:.0e}' if self.fmt == 'e' else f'{10. ** v:g}'
                for v in values]


def valid_frequencies(result, f_reduced):
    """
    Return the reduced frequencies a timestream has spectra at.

    Parameters:
    result (dict): ``psds.analyze_timestream`` output (or a loaded
        timestream).
    f_reduced (np.array): the group's reduced frequencies, Hz.

    Returns:
    idx (np.array, int): indices into ``f_reduced``.
    """
    valid = np.asarray(result['valid_reduced'], dtype = bool)
    sxx = np.asarray(result['sxx_reduced'], dtype = float)
    has = np.isfinite(sxx).any(axis = 0) if len(sxx) else valid
    return np.flatnonzero(valid & has)


class PsdOverviewWindow(QtWidgets.QMainWindow):
    """
    Window for reviewing one timestream's spectra tone by tone.

    Attributes:
    bad (dict): res_idx: reason, the tones marked bad (changed in place).
    limits (dict): review limits, '<frequency> Hz' (and ' cn' after
        common-mode removal): [low, high] of Sxx, 1/Hz.
    """

    def __init__(
        self, ts, result, f_reduced, bad = None, limits = None, root = None,
        stage = '', title = 'Noise spectra', ui_scale = 1.0
    ):
        """
        Build the window.

        Parameters:
        ts (dict): the timestream (``psds.list_timestreams``).
        result (dict): its ``psds.analyze_timestream`` output.
        f_reduced (np.array): the group's reduced frequencies, Hz.
        bad (dict or None): res_idx: reason, tones already bad (changed
            in place). Default None: none.
        limits (dict or None): saved review limits. Default None.
        root (zarr.Group or None): the store root, writable, for the
            tone-marks registry. Default None.
        stage (str): the result's path, recorded with the marks.
        title (str): window title. Default 'Noise spectra'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(title)
        self._ui_scale = float(ui_scale) if ui_scale else 1.0
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.root, self.stage = root, stage
        self.rows = [int(r) for r in ts['rows']]
        self.res_idxs = [int(r) for r in ts['res_idxs']]
        self.result = result
        self.f_reduced = np.asarray(f_reduced, dtype = float)
        self.f_idx = valid_frequencies(result, self.f_reduced)
        self.bad = {} if bad is None else bad
        self.limits = {} if limits is None else limits
        self.has_cn = 'sxx_reduced_cn' in result and \
            result['sxx_reduced_cn'] is not None
        self._updating = False
        self._listed = []
        self._only_outside = True
        self._ask_reason = lambda: ask_reason(
            self, f'Mark res {self.res_idxs[self.current()]} bad')
        self._ask_note = lambda: ask_note(
            self, f'Note on res {self.res_idxs[self.current()]}')
        # Tones the analysis left out as bad keep their reason
        for i, s in enumerate(result['status']):
            if str(s).startswith('marked bad') and \
                    self.res_idxs[i] not in self.bad:
                self.bad[self.res_idxs[i]] = str(s)[len('marked bad: '):]
        self._build()
        self.refresh()
        if self.tone_list.count():
            self.tone_list.setCurrentRow(0)

    ############################################################################
    def _build(self):
        """
        Build the histograms, the list, the spectrum plot and the buttons.
        """
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        left = QtWidgets.QVBoxLayout()
        self.cn_box = QtWidgets.QCheckBox('After common-mode removal')
        self.cn_box.setChecked(self.has_cn)
        self.cn_box.setEnabled(self.has_cn)
        self.cn_box.toggled.connect(self.refresh)
        left.addWidget(self.cn_box)
        self.hist_widget = pg.GraphicsLayoutWidget()
        self.plots, self.curves, self.lines, self.combos = [], [], [], []
        n = len(self.f_idx)
        picks = [self.f_idx[int(round(k * (n - 1) / max(N_HIST - 1, 1)))]
                 for k in range(N_HIST)] if n else []
        for k in range(N_HIST):
            combo = QtWidgets.QComboBox()
            for j in self.f_idx:
                combo.addItem(f'{self.f_reduced[j]:g} Hz', int(j))
            if picks:
                combo.setCurrentIndex(list(self.f_idx).index(picks[k]))
            combo.currentIndexChanged.connect(self.refresh)
            self.combos.append(combo)
            plot = self.hist_widget.addPlot(
                row = k, col = 0,
                axisItems = {'bottom': LogAxisItem('bottom')})
            plot.setLogMode(x = True)
            plot.setLabel('bottom', 'Sxx (1/Hz)')
            plot.setLabel('left', 'tones')
            # The histograms are a metric, not a plot to roam in
            plot.setMouseEnabled(False, False)
            plot.hideButtons()
            curve = pg.PlotCurveItem(np.array([0., 1.]), np.array([0.]),
                                     stepMode = 'center', fillLevel = 0,
                                     brush = (100, 100, 250, 120),
                                     pen = pg.mkPen((100, 100, 250)))
            plot.addItem(curve)
            pair = []
            for which in ('low', 'high'):
                line = pg.InfiniteLine(angle = 90, movable = True,
                                       pen = pg.mkPen('r', width = 2),
                                       label = which,
                                       labelOpts = dict(position = 0.9))
                line.sigPositionChangeFinished.connect(self._on_line)
                plot.addItem(line)
                pair.append(line)
            plot.scene().sigMouseClicked.connect(
                lambda ev, k = k: self._on_click(ev, k))
            self.plots.append(plot)
            self.curves.append(curve)
            self.lines.append(tuple(pair))
        combos = QtWidgets.QHBoxLayout()
        for combo in self.combos:
            combos.addWidget(combo)
        left.addLayout(combos)
        left.addWidget(self.hist_widget, stretch = 1)
        lay.addLayout(left, stretch = 1)

        right = QtWidgets.QVBoxLayout()
        self.only_outside = QtWidgets.QCheckBox(
            'List only the tones outside the limits')
        self.only_outside.setChecked(True)
        self.only_outside.toggled.connect(self._on_only_outside)
        right.addWidget(self.only_outside)
        self.count_label = QtWidgets.QLabel()
        right.addWidget(self.count_label)
        self.tone_list = QtWidgets.QListWidget()
        self.tone_list.currentRowChanged.connect(self._on_row_changed)
        # The shortcuts keep working while the list has the focus
        forward_keys(self.tone_list, self)
        right.addWidget(self.tone_list, stretch = 1)
        self.spec_plot = pg.PlotWidget(
            axisItems = {'bottom': LogAxisItem('bottom', fmt = 'g'),
                         'left': LogAxisItem('left')})
        self.spec_plot.setLogMode(x = True, y = True)
        self.spec_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spec_plot.setLabel('left', 'Sxx (1/Hz)')
        self.spec_plot.addLegend(offset = (-10, 10))
        self.spec_curve = self.spec_plot.plot(
            pen = pg.mkPen((150, 150, 150)), name = 'raw')
        self.spec_cn_curve = self.spec_plot.plot(
            pen = pg.mkPen((80, 160, 255), width = 1.5),
            name = 'common modes removed')
        self.spec_plot.setMinimumHeight(int(220 * self._ui_scale))
        right.addWidget(self.spec_plot, stretch = 1)
        self.bad_btn = QtWidgets.QPushButton('Mark tone bad (B)')
        self.bad_btn.clicked.connect(self.toggle_bad)
        self.note_btn = QtWidgets.QPushButton('Add note (T)')
        self.note_btn.clicked.connect(self.add_note)
        self.done_btn = QtWidgets.QPushButton('Done')
        self.done_btn.clicked.connect(self.close)
        for b in (self.bad_btn, self.note_btn, self.done_btn):
            right.addWidget(b)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setFixedHeight(int(60 * self._ui_scale))
        self.status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right.addWidget(self.status)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(int(380 * self._ui_scale))
        freeze_heights(right, skip = (self.tone_list, self.spec_plot))
        lay.addWidget(panel)
        self.setCentralWidget(central)
        self.resize(int(1100 * self._ui_scale), int(750 * self._ui_scale))

    ############################################################################
    def _key(self, k):
        """
        Return the limits key of histogram k.

        Parameters:
        k (int): histogram.

        Returns:
        key (str): e.g. '10 Hz cn'.
        """
        j = self.combos[k].currentData()
        return f'{self.f_reduced[j]:g} Hz' + (' cn' if self.cn_box.isChecked()
                                              else '')

    def values(self, k):
        """
        Return every tone's Sxx at histogram k's frequency.

        Parameters:
        k (int): histogram.

        Returns:
        v (np.array): per tone, 1/Hz, NaN without a value.
        """
        j = self.combos[k].currentData()
        if j is None:
            return np.full(len(self.rows), np.nan)
        key = 'sxx_reduced_cn' if self.cn_box.isChecked() else 'sxx_reduced'
        sxx = np.asarray(self.result[key], dtype = float)[:, j]
        return np.where(sxx > 0, sxx, np.nan)

    def outside_limits(self):
        """
        Find the tones outside the review limits or without a value.

        Returns:
        outside (set of int): indices in ``rows``.
        """
        outside = set()
        for k in range(N_HIST):
            v = self.values(k)
            lim = self.limits.get(self._key(k))
            bad = ~np.isfinite(v)
            if lim is not None:
                with np.errstate(invalid = 'ignore'):
                    bad |= (v < lim[0] * (1 - 1e-9)) | \
                        (v > lim[1] * (1 + 1e-9))
            outside |= set(np.flatnonzero(bad).tolist())
        return outside

    def refresh(self):
        """
        Draw the histograms (bins even in log Sxx) with their limits
        (unset limits at the edges of the data), and list the tones.
        """
        self._updating = True
        for k in range(N_HIST):
            v = self.values(k)
            v = v[np.isfinite(v)]
            key = self._key(k)
            lim = self.limits.get(key)
            if lim is None:
                lim = [float(v.min()), float(v.max())] if len(v) else \
                    [1e-18, 1e-14]
                self.limits[key] = lim
            # The lines live in the plot's log coordinates
            self.lines[k][0].setValue(np.log10(lim[0]))
            self.lines[k][1].setValue(np.log10(lim[1]))
            self.plots[k].setTitle(key)
            if len(v):
                lv = np.log10(v)
                lo = min(lv.min(), np.log10(lim[0]))
                hi = max(lv.max(), np.log10(lim[1]))
                span = (hi - lo) or 1.
                edges = np.linspace(lo - 0.05 * span, hi + 0.05 * span,
                                    max(10, min(60, len(v) // 2 + 10)))
                counts, _ = np.histogram(lv, edges)
                # The curve is drawn in the plot's log coordinates
                self.curves[k].setData(edges, counts)
                self.plots[k].setXRange(edges[0], edges[-1], padding = 0)
                self.plots[k].setYRange(0, max(counts.max(), 1) * 1.1,
                                        padding = 0)
            else:
                self.curves[k].setData(np.array([0., 1.]), np.array([0.]))
        self._updating = False
        self._fill_list()

    def _fill_list(self):
        """
        List the tones outside the limits (or all), with their values,
        keeping the selection.
        """
        selected = self.current()
        outside = self.outside_limits()
        listed = sorted(outside) if self._only_outside else \
            list(range(len(self.rows)))
        if selected >= 0 and selected not in listed:
            listed = sorted(set(listed) | {selected})
        vals = [self.values(k) for k in range(N_HIST)]
        self.tone_list.blockSignals(True)
        self.tone_list.clear()
        self._listed = listed
        for i in listed:
            r = self.res_idxs[i]
            parts = [f'{v[i]:.3g}' if np.isfinite(v[i]) else '-'
                     for v in vals]
            mark = 'x' if r in self.bad else '*' if i in outside else ' '
            self.tone_list.addItem(f"{mark} res {r}: {', '.join(parts)}")
        self.tone_list.blockSignals(False)
        self.count_label.setText(f'{len(outside)} of {len(self.rows)} tones '
                                 f'outside the limits, {len(self.bad)} bad')
        if selected in listed:
            self.tone_list.setCurrentRow(listed.index(selected))
        elif listed:
            self.tone_list.setCurrentRow(0)

    def current(self):
        """
        Return the selected tone.

        Returns:
        i (int): index in ``rows`` (-1 if none).
        """
        li = self.tone_list.currentRow()
        return self._listed[li] if 0 <= li < len(self._listed) else -1

    def step_tone(self, step):
        """
        Move to another tone of the list (A and D).

        Parameters:
        step (int): -1 for the previous tone, +1 for the next.
        """
        li = self.tone_list.currentRow() + step
        if 0 <= li < len(self._listed):
            self.tone_list.setCurrentRow(li)

    def _on_only_outside(self, only):
        """
        List only the tones outside the limits, or every tone.

        Parameters:
        only (bool): the box.
        """
        self._only_outside = bool(only)
        self._fill_list()

    def _on_row_changed(self, li):
        """
        Show the tone chosen in the list.

        Parameters:
        li (int): list row.
        """
        if 0 <= li < len(self._listed):
            self.select(self._listed[li])

    def select(self, i):
        """
        Show tone i: its spectrum and status.

        Parameters:
        i (int): index in ``rows``.
        """
        r = self.res_idxs[i]
        f = np.asarray(self.result['f_binned'], dtype = float)
        sxx = np.asarray(self.result['sxx_binned'], dtype = float)[i]
        ok = (f > 0) & (sxx > 0)
        self.spec_curve.setData(f[ok], sxx[ok])
        if self.has_cn:
            cn = np.asarray(self.result['sxx_binned_cn'], dtype = float)[i]
            ok = (f > 0) & (cn > 0)
            self.spec_cn_curve.setData(f[ok], cn[ok])
        else:
            self.spec_cn_curve.setData([], [])
        self.spec_plot.enableAutoRange()
        self.bad_btn.setText('Clear bad mark (B)' if r in self.bad
                             else 'Mark tone bad (B)')
        head = f'BAD: {self.bad[r] or "no reason given"}\n' if r in self.bad \
            else ''
        marks = mark_label(self.root, r)
        if marks and r not in self.bad:
            head += marks + '\n'
        status = str(self.result['status'][i])
        self.status.setText(head + f'res {r}: ' + (status or 'analyzed'))

    def _on_line(self):
        """
        Keep the limits where the lines were dragged and list again.
        """
        if self._updating:
            return
        for k in range(N_HIST):
            lo, hi = sorted(10. ** float(ln.value()) for ln in self.lines[k])
            self.limits[self._key(k)] = [lo, hi]
        self._fill_list()

    def _on_click(self, ev, k):
        """
        Select the tone nearest to a clicked value in histogram k.

        Parameters:
        ev (MouseClickEvent): the click.
        k (int): histogram.
        """
        plot = self.plots[k]
        if not plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = plot.vb.mapSceneToView(ev.scenePos()).x()
        v = self.values(k)
        if not np.isfinite(v).any():
            return
        with np.errstate(invalid = 'ignore'):
            i = int(np.nanargmin(np.abs(np.log10(v) - x)))
        if i not in self._listed:
            self.only_outside.setChecked(False)
        if i in self._listed:
            self.tone_list.setCurrentRow(self._listed.index(i))

    def keyPressEvent(self, event):
        """
        B marks the selected tone bad, T adds a note, A and D move
        through the listed tones.

        Parameters:
        event (QKeyEvent): the key event.
        """
        keys = {Qt.Key_B: self.toggle_bad, Qt.Key_T: self.add_note,
                Qt.Key_A: lambda: self.step_tone(-1),
                Qt.Key_D: lambda: self.step_tone(1)}
        if event.key() in keys and not isinstance(
                self.focusWidget(), (QtWidgets.QLineEdit,
                                     QtWidgets.QAbstractSpinBox)):
            keys[event.key()]()
            event.accept()
            return
        super().keyPressEvent(event)

    ############################################################################
    def toggle_bad(self):
        """
        Mark the selected tone bad with a reason, or clear its mark (the
        button, or B); the store's tone marks follow.
        """
        i = self.current()
        if i < 0:
            return
        r = self.res_idxs[i]
        if r in self.bad:
            del self.bad[r]
            record(self.root, r, 'clear', '', self.stage)
        else:
            reason = self._ask_reason()
            if reason is None:
                return
            self.bad[r] = reason
            record(self.root, r, 'bad', reason, self.stage)
        self._fill_list()
        self.select(i)

    def add_note(self):
        """
        Record a note on the selected tone in the store's tone marks (the
        button, or T).
        """
        i = self.current()
        if i < 0:
            return
        note = self._ask_note()
        if note:
            record(self.root, self.res_idxs[i], 'note', note, self.stage)
            self.select(i)


def run_psd_overview(
    ts, result, f_reduced, bad = None, limits = None, root = None,
    stage = '', title = 'Noise spectra', ui_scale = 1.0
):
    """
    Open the PSD overview window and wait until it is closed.

    Parameters:
    ts, result, f_reduced, bad, limits, root, stage, title, ui_scale: see
        ``PsdOverviewWindow``.

    Returns:
    bad (dict): res_idx: reason, the tones marked bad.
    limits (dict): the review limits.
    """
    app = get_qapp(title)
    win = PsdOverviewWindow(ts, result, f_reduced, bad = bad,
                            limits = limits, root = root, stage = stage,
                            title = title, ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.bad, win.limits
