"""
PSD overview window: histograms over all tones of a run's noise
spectra at chosen frequencies and of the s-factor (``pipeline.psds``),
with review limits, the selected tone's spectra (every timestream of
the run together), and marking tones bad with a reason.

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
from ...qt_compat import add_help
from ...qt_compat import (
    Qt, delete_on_close, fit_window_to_screen, forward_keys, freeze_heights,
    get_qapp, widget_font_stylesheet
)
from .marks import ask_note, ask_reason, reason_and_scope, mark_label, record, registry
from .photon_thresholds import _message_box

# Histograms shown (Sxx at chosen frequencies); the s-factor is one more
N_HIST = 3
# The s-factor shown is the mean over this band, Hz
SFACTOR_BAND = (10., 100.)
# Spectrum colours per timestream shown: (raw, after common modes)
ENTRY_COLOURS = [((150, 150, 150), (80, 160, 255)),
                 ((230, 170, 90), (255, 110, 0)),
                 ((140, 210, 140), (0, 160, 60)),
                 ((210, 140, 210), (170, 0, 170))]

def _nanmean(a, axis = 1):
    """
    Average over an axis, leaving out NaN, without a warning for rows
    of nothing but NaN (they give NaN).

    Parameters:
    a (np.array): 2d values.
    axis (int): the axis averaged over. Default 1.

    Returns:
    m (np.array): the means.
    """
    ok = np.isfinite(a)
    n = ok.sum(axis = axis)
    total = np.where(ok, a, 0.).sum(axis = axis)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)

def _ts_label(ts):
    """
    Name a timestream for the legend and the status.

    Parameters:
    ts (dict): the timestream (``psds.list_timestreams``).

    Returns:
    label (str): e.g. 'ts0 dark0'.
    """
    return f"{ts.get('run', '')} {ts.get('name', '')}".strip() or 'timestream'


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
    Window for reviewing the spectra of one run's timestreams tone by
    tone: the first timestream and ``others`` (the same run at another
    decimation stage, say) are shown together, each tone's histogram
    values being the geometric mean over the timestreams that have the
    tone and did not flag it.

    Attributes:
    bad (dict): res_idx: reason, the tones marked bad (changed in place;
        a mark applies to every timestream shown).
    limits (dict): review limits, '<frequency> Hz' (and ' cn' after
        common-mode removal): [low, high] of Sxx, 1/Hz; and
        'sfactor 10-100 Hz': [low, high] of the s-factor, dB.
    entries (list of dict): per timestream shown: ``ts``, ``result``,
        ``stage``, ``label``.
    """

    def __init__(
        self, ts, result, f_reduced, bad = None, limits = None, root = None,
        stage = '', title = 'Noise spectra', ui_scale = 1.0, label = None,
        others = ()
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
        label (str or None): the timestream's name in the legend.
            Default None: its run and name.
        others (sequence of dict): more timestreams of the run shown
            with it: ``ts``, ``result``, and optionally ``stage`` and
            ``label``. Default none.
        """
        super().__init__()
        self.setWindowTitle(title)
        self._ui_scale = float(ui_scale) if ui_scale else 1.0
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.root, self.stage = root, stage
        self.entries = [dict(ts = ts, result = result, stage = stage,
                             label = label or _ts_label(ts))]
        for e in others:
            e = dict(e)
            e.setdefault('stage', '')
            e['label'] = e.get('label') or _ts_label(e['ts'])
            self.entries.append(e)
        self.stages = []
        for e in self.entries:
            if e['stage'] and e['stage'] not in self.stages:
                self.stages.append(e['stage'])
        # The tones: every timestream's, in order of first appearance
        self.res_idxs = []
        for e in self.entries:
            for r in e['ts']['res_idxs']:
                if int(r) not in self.res_idxs:
                    self.res_idxs.append(int(r))
        self.rows = list(range(len(self.res_idxs)))
        for e in self.entries:
            e['pos'] = {int(r): i for i, r in enumerate(e['ts']['res_idxs'])}
        self.result = result
        self.f_reduced = np.asarray(f_reduced, dtype = float)
        f_idx = set()
        for e in self.entries:
            f_idx |= set(valid_frequencies(e['result'],
                                           self.f_reduced).tolist())
        self.f_idx = np.array(sorted(f_idx), dtype = int)
        self.bad = {} if bad is None else bad
        self.limits = {} if limits is None else limits
        self.has_cn = any(
            'sxx_reduced_cn' in e['result']
            and e['result']['sxx_reduced_cn'] is not None
            for e in self.entries)
        self._updating = False
        self._listed = []
        self._only_outside = True
        self._include_bad = True
        self._ask_reason = lambda: ask_reason(
            self, f'Mark res {self.res_idxs[self.current()]} bad',
            scope = 'local')
        self._ask_note = lambda: ask_note(
            self, f'Note on res {self.res_idxs[self.current()]}')
        # Tones every timestream left out as bad keep their reason
        for e in self.entries:
            for i, st in enumerate(e['result']['status']):
                r = int(e['ts']['res_idxs'][i])
                if str(st).startswith('marked bad') and r not in self.bad \
                        and all(self._flagged(o, r) for o in self.entries
                                if r in o['pos']):
                    self.bad[r] = str(st)[len('marked bad: '):]
        self._build()
        self.refresh()
        if self.tone_list.count():
            self.tone_list.setCurrentRow(0)

    ############################################################################
    @staticmethod
    def _flagged(entry, r):
        """
        Say whether a timestream's analysis flagged a tone (left it out
        as bad, or could not calibrate it).

        Parameters:
        entry (dict): the timestream.
        r (int): the resonator index.

        Returns:
        flagged (bool): True if so, or if the timestream lacks the tone.
        """
        i = entry['pos'].get(int(r))
        if i is None:
            return True
        return bool(str(entry['result']['status'][i]))

    def _entry_values(self, entry, key, j):
        """
        Return one timestream's values of an array at a reduced
        frequency, per tone of the window.

        Parameters:
        entry (dict): the timestream.
        key (str): 'sxx_reduced', 'sxx_reduced_cn' or 'sfactor_reduced'.
        j (int): index into ``f_reduced``.

        Returns:
        v (np.array): per tone, NaN where the timestream lacks the tone,
            flagged it, or has no value.
        """
        out = np.full(len(self.res_idxs), np.nan)
        arr = entry['result'].get(key)
        if arr is None:
            return out
        arr = np.asarray(arr, dtype = float)
        for i, r in enumerate(self.res_idxs):
            k = entry['pos'].get(r)
            if k is not None and not self._flagged(entry, r):
                out[i] = arr[k, j]
        return out

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
        for k in range(N_HIST + 1):
            log = k < N_HIST
            if log:
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
            else:
                plot = self.hist_widget.addPlot(row = k, col = 0)
                plot.setLabel('bottom', 's-factor, parallel / perpendicular '
                              '(dB)')
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
        combos.addWidget(QtWidgets.QLabel(
            f's-factor: {SFACTOR_BAND[0]:g} to {SFACTOR_BAND[1]:g} Hz'))
        left.addLayout(combos)
        left.addWidget(self.hist_widget, stretch = 1)
        lay.addLayout(left, stretch = 1)

        right = QtWidgets.QVBoxLayout()
        self.only_outside = QtWidgets.QCheckBox(
            'Show only flagged tones (outside the limits)')
        self.only_outside.setChecked(True)
        self.only_outside.toggled.connect(self._on_only_outside)
        right.addWidget(self.only_outside)
        self.include_bad = QtWidgets.QCheckBox('Include tones marked bad')
        self.include_bad.setChecked(True)
        self.include_bad.toggled.connect(self._on_include_bad)
        right.addWidget(self.include_bad)
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
        # One pair of curves per timestream: raw (thin, dashed when there
        # are several) and after the common modes
        self.spec_curves, self.spec_cn_curves = [], []
        several = len(self.entries) > 1
        for e_i, e in enumerate(self.entries):
            raw_c, cn_c = ENTRY_COLOURS[e_i % len(ENTRY_COLOURS)]
            name = e['label'] if several else ''
            self.spec_curves.append(self.spec_plot.plot(
                pen = pg.mkPen(raw_c, style = Qt.DashLine if several
                               else Qt.SolidLine),
                name = f'{name} raw'.strip()))
            self.spec_cn_curves.append(self.spec_plot.plot(
                pen = pg.mkPen(cn_c, width = 1.5),
                name = f'{name} common modes removed'.strip()))
        self.spec_curve = self.spec_curves[0]
        self.spec_cn_curve = self.spec_cn_curves[0]
        self.spec_plot.setMinimumHeight(int(220 * self._ui_scale))
        right.addWidget(self.spec_plot, stretch = 1)
        self.bad_btn = QtWidgets.QPushButton('Mark tone bad (B)')
        self.bad_btn.clicked.connect(self.toggle_bad)
        self.note_btn = QtWidgets.QPushButton('Add note (N)')
        self.note_btn.clicked.connect(self.add_note)
        self.done_btn = QtWidgets.QPushButton('Done')
        self.done_btn.clicked.connect(self.close)
        for b in (self.bad_btn, self.note_btn, self.done_btn):
            right.addWidget(b)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setFixedHeight(int(80 * self._ui_scale))
        self.status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right.addWidget(self.status)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(int(380 * self._ui_scale))
        freeze_heights(right, skip = (self.tone_list, self.spec_plot))
        lay.addWidget(panel)
        self.setCentralWidget(central)
        add_help(self, 'Noise spectra', [
            ('Keyboard', [
                ('A / D', 'Previous / next tone'),
                ('B', 'Mark the tone bad (or clear the mark), with a reason; '
                 'the mark applies to every timestream shown'),
                ('N (or T)', 'Add a note on the tone'),
                ('H', 'Show / hide this help')]),
            ('Buttons', [
                ('After common-mode removal', 'Show the spectra after it'),
                ('Only tones outside', 'List only the tones outside the '
                 'window'),
                ('Include tones marked bad', 'List them too'),
                ('Done', 'Save and close')]),
            ('Histograms', [
                ('Sxx', 'At the frequency chosen above each; with several '
                 'timestreams, the geometric mean over the ones that have '
                 'the tone and did not flag it'),
                ('s-factor', f'The mean over {SFACTOR_BAND[0]:g} to '
                 f'{SFACTOR_BAND[1]:g} Hz of the parallel / perpendicular '
                 'noise ratio, dB (the nearest point when a timestream '
                 'has none there)')]),
            ('Mouse', [('Scroll / drag', 'Zoom / pan a plot'),
                       ('Right-click', 'Plot menu (axes, log scale, export)')]),
        ])
        self.resize(int(1100 * self._ui_scale), int(800 * self._ui_scale))

    ############################################################################
    def _key(self, k):
        """
        Return the limits key of histogram k.

        Parameters:
        k (int): histogram (``N_HIST`` is the s-factor's).

        Returns:
        key (str): e.g. '10 Hz cn', or 'sfactor 10-100 Hz'.
        """
        if k >= N_HIST:
            return f'sfactor {SFACTOR_BAND[0]:g}-{SFACTOR_BAND[1]:g} Hz'
        j = self.combos[k].currentData()
        return f'{self.f_reduced[j]:g} Hz' + (' cn' if self.cn_box.isChecked()
                                              else '')

    def _is_log(self, k):
        """
        Say whether histogram k is on a log axis (the Sxx ones).

        Parameters:
        k (int): histogram.

        Returns:
        log (bool): True for Sxx, False for the s-factor.
        """
        return k < N_HIST

    def values(self, k):
        """
        Return every tone's value for histogram k: Sxx at its frequency
        (1/Hz), or the s-factor over ``SFACTOR_BAND`` (dB); over several
        timestreams, the geometric mean (Sxx) or mean (s-factor) of the
        ones that have the tone and did not flag it.

        Parameters:
        k (int): histogram.

        Returns:
        v (np.array): per tone, NaN without a value.
        """
        n = len(self.res_idxs)
        if k >= N_HIST:
            per = []
            for e in self.entries:
                valid = valid_frequencies(e['result'], self.f_reduced)
                f = self.f_reduced[valid]
                band = valid[(f >= SFACTOR_BAND[0]) & (f <= SFACTOR_BAND[1])]
                if not len(band) and len(valid):
                    band = valid[[int(np.argmin(np.abs(f - SFACTOR_BAND[0])))]]
                if not len(band):
                    continue
                cols = np.stack([self._entry_values(e, 'sfactor_reduced', j)
                                 for j in band], axis = 1)
                per.append(_nanmean(cols))
            if not per:
                return np.full(n, np.nan)
            return _nanmean(np.stack(per, axis = 1))
        j = self.combos[k].currentData()
        if j is None:
            return np.full(n, np.nan)
        key = 'sxx_reduced_cn' if self.cn_box.isChecked() else 'sxx_reduced'
        per = []
        for e in self.entries:
            v = self._entry_values(e, key, j)
            per.append(np.where(v > 0, np.log10(np.where(v > 0, v, 1.)),
                                np.nan))
        mean = _nanmean(np.stack(per, axis = 1))
        return np.where(np.isfinite(mean), 10. ** mean, np.nan)

    def outside_limits(self):
        """
        Find the tones outside the review limits or without an Sxx value
        (a missing s-factor, as in older results, is not flagged), and
        the tones marked bad.

        Returns:
        outside (set of int): indices in ``rows``.
        """
        outside = {i for i, r in enumerate(self.res_idxs) if r in self.bad}
        for k in range(N_HIST + 1):
            v = self.values(k)
            lim = self.limits.get(self._key(k))
            bad = ~np.isfinite(v) if self._is_log(k) else \
                np.zeros(len(v), dtype = bool)
            if lim is not None:
                with np.errstate(invalid = 'ignore'):
                    bad |= (v < lim[0] - abs(lim[0]) * 1e-9) | \
                        (v > lim[1] + abs(lim[1]) * 1e-9)
            outside |= set(np.flatnonzero(bad).tolist())
        return outside

    def refresh(self):
        """
        Draw the histograms (bins even in log Sxx, or in dB for the
        s-factor) with their limits (unset limits at the edges of the
        data), and list the tones.
        """
        self._updating = True
        # The histograms leave out the tones marked bad (the list keeps
        # them, flagged)
        keep = np.array([r not in self.bad for r in self.res_idxs],
                        dtype = bool)
        for k in range(N_HIST + 1):
            log = self._is_log(k)
            v = self.values(k)[keep]
            v = v[np.isfinite(v)]
            key = self._key(k)
            lim = self.limits.get(key)
            if lim is None:
                if len(v):
                    lim = [float(v.min()), float(v.max())]
                else:
                    lim = [1e-18, 1e-14] if log else [-10., 10.]
                self.limits[key] = lim
            t = np.log10 if log else (lambda x: x)
            # The lines live in the plot's own coordinates
            self.lines[k][0].setValue(t(lim[0]))
            self.lines[k][1].setValue(t(lim[1]))
            self.plots[k].setTitle(key)
            if len(v):
                lv = t(v)
                lo = min(lv.min(), t(lim[0]))
                hi = max(lv.max(), t(lim[1]))
                span = (hi - lo) or 1.
                edges = np.linspace(lo - 0.05 * span, hi + 0.05 * span,
                                    max(10, min(60, len(v) // 2 + 10)))
                counts, _ = np.histogram(lv, edges)
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
        if not self._include_bad:
            listed = [i for i in listed if self.res_idxs[i] not in self.bad]
        if selected >= 0 and selected not in listed:
            listed = sorted(set(listed) | {selected})
        vals = [self.values(k) for k in range(N_HIST + 1)]
        self.tone_list.blockSignals(True)
        self.tone_list.clear()
        self._listed = listed
        for i in listed:
            r = self.res_idxs[i]
            parts = [f'{v[i]:.3g}' if np.isfinite(v[i]) else '-'
                     for v in vals[:N_HIST]]
            sf = vals[N_HIST][i]
            parts.append(f'{sf:.1f} dB' if np.isfinite(sf) else '-')
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

    def _on_include_bad(self, include):
        """
        List the tones marked bad, or leave them out.

        Parameters:
        include (bool): the box.
        """
        self._include_bad = bool(include)
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
        Show tone i: every timestream's spectrum of it, and its status.

        Parameters:
        i (int): index in ``rows``.
        """
        r = self.res_idxs[i]
        lines = []
        for e, raw_curve, cn_curve in zip(self.entries, self.spec_curves,
                                          self.spec_cn_curves):
            k = e['pos'].get(r)
            res = e['result']
            if k is None:
                raw_curve.setData([], [])
                cn_curve.setData([], [])
                lines.append(f"{e['label']}: not in this timestream")
                continue
            f = np.asarray(res['f_binned'], dtype = float)
            sxx = np.asarray(res['sxx_binned'], dtype = float)[k]
            ok = (f > 0) & (sxx > 0)
            raw_curve.setData(f[ok], sxx[ok])
            cn = res.get('sxx_binned_cn')
            if cn is not None:
                cn = np.asarray(cn, dtype = float)[k]
                ok = (f > 0) & (cn > 0)
                cn_curve.setData(f[ok], cn[ok])
            else:
                cn_curve.setData([], [])
            status = str(res['status'][k])
            lines.append(f"{e['label']}: {status or 'analyzed'}")
        self.spec_plot.enableAutoRange()
        self.bad_btn.setText('Clear bad mark (B)' if r in self.bad
                             else 'Mark tone bad (B)')
        head = f'BAD: {self.bad[r] or "no reason given"}\n' if r in self.bad \
            else ''
        marks = mark_label(self.root, r, self.stage)
        if marks and r not in self.bad:
            head += marks + '\n'
        if len(self.entries) == 1:
            self.status.setText(head + f'res {r}: '
                                + lines[0].split(': ', 1)[1])
        else:
            self.status.setText(head + f'res {r}\n' + '\n'.join(lines))

    def _on_line(self):
        """
        Keep the limits where the lines were dragged and list again.
        """
        if self._updating:
            return
        for k in range(N_HIST + 1):
            back = (lambda x: 10. ** x) if self._is_log(k) else (lambda x: x)
            lo, hi = sorted(back(float(ln.value())) for ln in self.lines[k])
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
        t = np.log10 if self._is_log(k) else (lambda a: a)
        with np.errstate(invalid = 'ignore'):
            i = int(np.nanargmin(np.abs(t(v) - x)))
        if i not in self._listed:
            self.only_outside.setChecked(False)
        if i not in self._listed:
            self.include_bad.setChecked(True)
        if i in self._listed:
            self.tone_list.setCurrentRow(self._listed.index(i))

    def keyPressEvent(self, event):
        """
        B marks the selected tone bad, N adds a note, A and D move
        through the listed tones.

        Parameters:
        event (QKeyEvent): the key event.
        """
        keys = {Qt.Key_B: self.toggle_bad, Qt.Key_M: self.toggle_bad,
                Qt.Key_T: self.add_note, Qt.Key_N: self.add_note,
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
        button, or B); the store's tone marks of every timestream shown
        follow.
        """
        i = self.current()
        if i < 0:
            return
        r = self.res_idxs[i]
        if r in self.bad:
            del self.bad[r]
            for stage in self.stages or ['']:
                record(self.root, r, 'clear', '', stage)
        else:
            reason, scope = reason_and_scope(self._ask_reason(), 'local')
            if reason is None:
                return
            self.bad[r] = reason
            for stage in self.stages or ['']:
                record(self.root, r, 'bad', reason, stage, scope)
        # The histograms change too: the tone leaves or rejoins them
        self.refresh()
        self.select(i)

    def add_note(self):
        """
        Record a note on the selected tone in the store's tone marks (the
        button, or N).
        """
        i = self.current()
        if i < 0:
            return
        note = self._ask_note()
        if note:
            record(self.root, self.res_idxs[i], 'note', note, self.stage)
            self.select(i)

def ask_existing_psd(path, description):
    """
    Ask what to do with a saved set of spectra that a run would make
    again: a new set (the default), compute it again with its saved
    review, or overwrite it, which asks once more to be sure.

    Parameters:
    path (str): the set's path (``ts_psd{p}``).
    description (str): the set, as ``psds.describe_group``.

    Returns:
    choice (str or None): 'new', 'load', 'overwrite', or None to stop
        (Cancel, or closing a box).
    """
    choice = _message_box(
        f'{path} exists',
        f'{path} is already saved:\n\n{description}\n\n'
        'New: another set, with the window opening on its limits. '
        'Load: compute it again with its settings, bad marks and limits, '
        'without the window. Overwrite: replace it.',
        ['New', 'Load', 'Overwrite', 'Cancel'], 'New')
    if choice in ('New', 'Load'):
        return choice.lower()
    if choice != 'Overwrite':
        return None
    sure = _message_box(
        f'Overwrite {path}?',
        f'Overwrite {path}? Its spectra, bad marks, limits and notes are '
        'lost.',
        ['Overwrite', 'Cancel'], 'Cancel',
        icon = QtWidgets.QMessageBox.Icon.Warning)
    return 'overwrite' if sure == 'Overwrite' else None

def run_psd_overview(
    ts, result, f_reduced, bad = None, limits = None, root = None,
    stage = '', title = 'Noise spectra', ui_scale = 1.0, label = None,
    others = ()
):
    """
    Open the PSD overview window and wait until it is closed.

    Parameters:
    ts, result, f_reduced, bad, limits, root, stage, title, ui_scale,
        label, others: see ``PsdOverviewWindow``.

    Returns:
    bad (dict): res_idx: reason, the tones marked bad.
    limits (dict): the review limits.
    """
    app = get_qapp(title)
    win = PsdOverviewWindow(ts, result, f_reduced, bad = bad,
                            limits = limits, root = root, stage = stage,
                            title = title, ui_scale = ui_scale,
                            label = label, others = others)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.bad, win.limits
