"""
Review window of a power sweep's NEP analysis (``nep.sweep.PowerSweep``,
``procedures/6d_nep.py``).

Left: the tone list (``*`` reviewed, ``x`` bad; the boxes above it list
only the tones outside the overview's limits, with or without the tones
marked bad), the point to edit, "Calibration..." (the calibration and IQ
fit panels of this tone at that point; when that window closes the
tone's fr and Sxx there are recomputed, without searching for glitches or
common modes again), the ``p_min`` of the photon-noise fit, and the
buttons. Right: x vs power with the responsivity fit, and NEP vs power
with the photon-noise fit and the draggable ``p_min`` line.

``run_nep_review`` opens the window, runs the final pass on the points
whose calibrations changed when it closes, and opens it again on the
final values with the overview, until it closes with nothing changed.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from ..qt_compat import Qt, add_help, delete_on_close, fg
from ..qt_compat import (
    fit_window_to_screen, forward_keys, freeze_heights, get_qapp,
    widget_font_stylesheet)
from ..pipeline.interactive import marks as _marks
from ..pipeline.interactive.photon_thresholds import ToneOverviewWindow
from ..responsivity.funcs import responsivity_int_x0
from .funcs import nep_photon

OVERVIEW = (('R0', 'R0 (1/W)'), ('P0', 'P0 (W)'), ('x0', 'x0'),
            ('eta', 'eta'), ('nep_inc', 'NEPinc (W/Hz^0.5)'),
            ('nep_abs', 'NEPabs (W/Hz^0.5)'))


class NEPReviewWindow(QtWidgets.QMainWindow):
    """
    Window to review a power sweep's NEP analysis tone by tone.

    Attributes:
    sweep (PowerSweep): the sweep (its values change in place).
    thresholds (PowerSweep): the sweep again, for ``ToneOverviewWindow``
        (``limits``, ``is_bad``).
    rows, res_idxs (list of int): tone indices and resonator indices.
    overview (ToneOverviewWindow or None): the overview, once opened.
    reviewed (set of int): resonator indices looked at.
    """

    def __init__(
        self, sweep, exclude = None, title = 'NEP review', ui_scale = 1.0,
        message = ''
    ):
        """
        Build the window.

        Parameters:
        sweep (PowerSweep): the sweep.
        exclude (dict or None): res_idx: reason, tones left out of the
            spectra (shown as bad). Default None.
        title (str): window title. Default 'NEP review'.
        ui_scale (float): font size multiplier. Default 1.0.
        message (str): text shown in the status at first, e.g. that the
            final pass ran. Default ''.
        """
        super().__init__()
        self.setWindowTitle(title)
        self._ui_scale = float(ui_scale) if ui_scale else 1.0
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.sweep = sweep
        self.thresholds = sweep
        if exclude:
            for r, reason in exclude.items():
                sweep.bad.setdefault(int(r), str(reason))
        self.rows = list(range(len(sweep.res_idxs)))
        self.res_idxs = [int(r) for r in sweep.res_idxs]
        self.reviewed = set()
        self._row = None
        self._list_rows = []
        self._only_outside = False
        self._include_bad = True
        self._updating = False
        self.overview = None
        self._cal_window = None
        self._message = message
        self._build_controls()
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._scroll)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        add_help(self, 'NEP review', [
            ('Keyboard', [
                ('A / D', 'Previous / next tone'),
                ('C', "Open the calibration and IQ fit of this tone at the "
                 'chosen point'),
                ('B', 'Mark the tone bad (or clear the mark), with a reason'),
                ('T', 'Add a note on the tone'),
                ('O', 'Open the tone overview'),
                ('R', 'Rescale the plots'),
                ('H', 'Show / hide this help')]),
            ('Buttons', [
                ('Only tones outside', 'List only the tones outside the '
                 'overview limits'),
                ('Include tones marked bad', 'List them too'),
                ('Apply p_min to all', "Every tone's photon-noise fit from "
                 'this p_min'),
                ('Done', 'Close; the points whose calibrations changed are '
                 'computed again in full, and the window opens again on '
                 'the final values')]),
            ('Mouse', [('Drag the dashed line', 'p_min of the photon-noise '
                        'fit'),
                       ('Scroll / drag', 'Zoom / pan a plot')]),
        ])
        for key, slot in (('A', lambda: self.step_tone(-1)),
                          ('D', lambda: self.step_tone(1)),
                          ('C', self.open_calibration), ('B', self.toggle_bad),
                          ('T', self.add_note), ('O', self.open_overview),
                          ('R', self.rescale)):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, slot)
        self._fill_list()
        self.tone_list.setCurrentRow(0)
        if message:
            self.status.setText(message)

    ############################################################################
    def _build_controls(self):
        """
        Build the left panel: tone list, the point, the buttons.
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
        self.point_box = QtWidgets.QComboBox()
        for p in self.sweep.points:
            self.point_box.addItem(f"{p['power_w']:.3g} W: {p['name']}")
        self.point_box.setToolTip('The point whose calibration to open')
        form.addRow('Point', self.point_box)
        self.pmin_box = QtWidgets.QDoubleSpinBox()
        self.pmin_box.setDecimals(3)
        self.pmin_box.setRange(0., 1e3)
        self.pmin_box.setSingleStep(0.1)
        self.pmin_box.setToolTip('Lowest power of the photon-noise fit, '
                                 'in units of the highest power (0: every '
                                 'point)')
        self.pmin_box.valueChanged.connect(self._on_pmin_box)
        form.addRow('p_min / P_max', self.pmin_box)
        lay.addLayout(form)
        self.cal_btn = QtWidgets.QPushButton('Calibration... (C)')
        self.cal_btn.setToolTip('Open the calibration and IQ fit panels of '
                                'this tone at the point above; its fr and '
                                'Sxx there are recomputed when it closes')
        self.cal_btn.clicked.connect(self.open_calibration)
        self.pmin_all_btn = QtWidgets.QPushButton('Apply p_min to all')
        self.pmin_all_btn.clicked.connect(self.apply_pmin_all)
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
        grid = QtWidgets.QGridLayout()
        for k, b in enumerate((self.cal_btn, self.pmin_all_btn, self.bad_btn,
                               self.note_btn, self.overview_btn,
                               self.rescale_btn, self.done_btn)):
            grid.addWidget(b, k // 2, k % 2)
        lay.addLayout(grid)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setFixedHeight(int(110 * self._ui_scale))
        self.status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.status)
        freeze_heights(lay, skip = (self.tone_list,))
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidget(self._panel)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFixedWidth(self._panel.width() + int(
            24 * self._ui_scale))
        self._scroll.setMinimumHeight(300)

    def _build_plots(self):
        """
        Build the x vs power and NEP vs power plots.
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.x_plot = self.plots.addPlot(row = 0, col = 0)
        self.x_plot.setLabel('bottom', 'Incident power (W)')
        self.x_plot.setLabel('left', 'x')
        self.x_plot.showGrid(x = True, y = True, alpha = 0.3)
        self.x_points = pg.ScatterPlotItem(size = 8, pen = None,
                                           brush = pg.mkBrush(80, 160, 255))
        self.x_plot.addItem(self.x_points)
        self.x_fit = self.x_plot.plot(pen = pg.mkPen('r', width = 2))
        self.nep_plot = self.plots.addPlot(row = 1, col = 0)
        self.nep_plot.setLabel('bottom', 'Incident power (W)')
        self.nep_plot.setLabel('left', 'NEP (W/Hz^0.5)')
        self.nep_plot.setLogMode(x = True, y = True)
        self.nep_plot.showGrid(x = True, y = True, alpha = 0.3)
        self.nep_in = pg.ScatterPlotItem(size = 8, pen = None,
                                         brush = pg.mkBrush(80, 160, 255))
        self.nep_out = pg.ScatterPlotItem(size = 8, pen = None,
                                          brush = pg.mkBrush(150, 150, 150))
        self.nep_plot.addItem(self.nep_in)
        self.nep_plot.addItem(self.nep_out)
        self.nep_fit = self.nep_plot.plot(pen = pg.mkPen('r', width = 2))
        self.pmin_line = pg.InfiniteLine(
            angle = 90, movable = True,
            pen = pg.mkPen(fg(), width = 1, style = Qt.DashLine),
            label = 'p_min', labelOpts = dict(position = 0.9))
        self.pmin_line.sigPositionChangeFinished.connect(self._on_pmin_line)
        self.nep_plot.addItem(self.pmin_line)

    ############################################################################
    def _list_text(self, i):
        """
        Return a tone's line in the list.

        Parameters:
        i (int): tone index.

        Returns:
        text (str): the line.
        """
        r = self.res_idxs[i]
        f = self.sweep.fields
        mark = 'x' if self.sweep.is_bad(r) else \
            '*' if r in self.reviewed else ' '
        if self.sweep.is_bad(r):
            return f'{mark} res {r}: bad ({self.sweep.bad[r]})'
        if np.isfinite(f['eta'][i]):
            return (f"{mark} res {r}: eta {f['eta'][i]:.2f}, NEPinc "
                    f"{f['nep_inc'][i]:.2e}")
        return f'{mark} res {r}: {self.sweep.status[i] or "no fit"}'

    def _fill_list(self):
        """
        Fill the tone list: every tone, or only the tones outside the
        overview's limits, with or without the tones marked bad, keeping
        the selection.
        """
        selected = self._list_rows[self.tone_list.currentRow()] \
            if self._list_rows and self.tone_list.currentRow() >= 0 else None
        outside = self.sweep.outside_limits() if self._only_outside else None
        if outside is not None and selected is not None:
            outside = set(outside) | {selected}
        self.tone_list.blockSignals(True)
        self.tone_list.clear()
        self._list_rows = []
        for i in self.rows:
            if outside is not None and i not in outside:
                continue
            if not self._include_bad and i != selected and \
                    self.sweep.is_bad(self.res_idxs[i]):
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
        i (int): tone index (the tone shown last if the list is empty).
        """
        li = self.tone_list.currentRow()
        if 0 <= li < len(self._list_rows):
            return self._list_rows[li]
        return self._row or 0

    def _on_row_changed(self, li):
        """
        Show the tone chosen in the list.

        Parameters:
        li (int): list row (negative: nothing chosen).
        """
        if li < 0 or li >= len(self._list_rows):
            return
        i = self._list_rows[li]
        if self._row is not None and self._row != i:
            self.reviewed.add(self.res_idxs[self._row])
            self._show_text(self._row)
        self._row = i
        self.select(i)

    def _show_text(self, i):
        """
        Put a tone's line in the list again.

        Parameters:
        i (int): tone index.
        """
        if i in self._list_rows:
            item = self.tone_list.item(self._list_rows.index(i))
            if item is not None:
                item.setText(self._list_text(i))

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
        Select a tone by its index (from the overview), listing every
        tone again if it is filtered out.

        Parameters:
        i (int): tone index.
        """
        if i not in self._list_rows:
            self.set_only_outside(False)
        if i not in self._list_rows:
            self.set_include_bad(True)
        if i in self._list_rows:
            self.tone_list.setCurrentRow(self._list_rows.index(i))

    def set_only_outside(self, only):
        """
        List every tone, or only the tones outside the overview's limits.

        Parameters:
        only (bool): True for the tones outside only.
        """
        self._only_outside = bool(only)
        _sync(self.only_outside, self._only_outside)
        self._fill_list()

    def set_include_bad(self, include):
        """
        List the tones marked bad too, or leave them out.

        Parameters:
        include (bool): True to list them.
        """
        self._include_bad = bool(include)
        _sync(self.include_bad, self._include_bad)
        self._fill_list()

    ############################################################################
    def select(self, i):
        """
        Show tone i: its values, plots and summary.

        Parameters:
        i (int): tone index.
        """
        if i < 0:
            return
        s, f = self.sweep, self.sweep.fields
        r = self.res_idxs[i]
        self._updating = True
        pmax = float(np.nanmax(s.power)) or 1.
        self.pmin_box.setValue(float(f['p_min'][i]) / pmax)
        self._updating = False
        self.bad_btn.setText('Clear bad mark (B)' if s.is_bad(r)
                             else 'Mark tone bad (B)')
        head = f'MARKED BAD: {s.bad[r]}\n' if s.is_bad(r) else ''
        if s.root is not None:
            label = _marks.mark_label(s.root, r, s.grp.path)
            if label and not s.is_bad(r):
                head += label + '\n'
        if np.isfinite(f['R0'][i]):
            text = (f"R0 {f['R0'][i]:.3g} 1/W, P0 {f['P0'][i]:.3g} W, x0 "
                    f"{f['x0'][i]:.3g}\n")
        else:
            text = ''
        if np.isfinite(f['eta'][i]):
            text += (f"eta {f['eta'][i]:.3f} ± {f['eta_err'][i]:.3f} "
                     f"({int(f['n_fit'][i])} points)\n"
                     f"NEPinc {f['nep_inc'][i]:.3g}, NEPabs "
                     f"{f['nep_abs'][i]:.3g} W/Hz^0.5")
        if s.status[i]:
            text += ('\n' if text else '') + s.status[i]
        self.status.setText(head + text)
        self.update_plots()

    def update_plots(self):
        """
        Draw the selected tone's x vs power and NEP vs power.
        """
        i = self.current()
        s, f = self.sweep, self.sweep.fields
        P = s.power
        x = s.x[i]
        ok = np.isfinite(x)
        self.x_points.setData(P[ok], x[ok])
        if np.isfinite(f['R0'][i]) and np.isfinite(f['P0'][i]) and \
                f['P0'][i] > 0:
            grid = np.linspace(0, P.max() * 1.05, 200)
            self.x_fit.setData(grid, responsivity_int_x0(
                grid, f['R0'][i], f['P0'][i], f['x0'][i]))
        else:
            self.x_fit.setData([], [])
        nep = s.nep[i]
        good = np.isfinite(nep) & (P > 0)
        p_min = float(f['p_min'][i])
        fit_mask = good & (P >= p_min)
        self.nep_in.setData(P[fit_mask], nep[fit_mask])
        self.nep_out.setData(P[good & ~fit_mask], nep[good & ~fit_mask])
        if np.isfinite(f['eta'][i]) and np.isfinite(s.nu[i]):
            grid = np.geomspace(max(P[P > 0].min() / 2, 1e-30),
                                P.max() * 2, 200) if good.any() else []
            self.nep_fit.setData(grid, nep_photon(grid, f['eta'][i],
                                                  s.nu[i]))
        else:
            self.nep_fit.setData([], [])
        self.pmin_line.blockSignals(True)
        self.pmin_line.setValue(np.log10(p_min) if p_min > 0
                                else np.log10(max(P[P > 0].min(), 1e-30)) - 1)
        self.pmin_line.blockSignals(False)

    def rescale(self):
        """
        Auto-range both plots (the button, or R).
        """
        self.x_plot.enableAutoRange()
        self.nep_plot.enableAutoRange()

    ############################################################################
    def _on_pmin_box(self):
        """
        Set the selected tone's p_min from the box and fit it again.
        """
        if self._updating:
            return
        pmax = float(np.nanmax(self.sweep.power)) or 1.
        self.set_pmin(self.current(), float(self.pmin_box.value()) * pmax)

    def _on_pmin_line(self):
        """
        Set the selected tone's p_min from the dragged line.
        """
        if self._updating:
            return
        self.set_pmin(self.current(), float(10 ** self.pmin_line.value()))

    def set_pmin(self, i, p_min):
        """
        Set a tone's p_min, fit it again, and refresh.

        Parameters:
        i (int): tone index.
        p_min (float): lowest power of its photon-noise fit, W (0: all).
        """
        self.sweep.fields['p_min'][i] = max(0., float(p_min))
        self.sweep.compute_tone(i)
        self.reviewed.add(self.res_idxs[i])
        self._show_text(i)
        if i == self.current():
            self.select(i)
        self._refresh_overview()

    def apply_pmin_all(self):
        """
        Give every tone the selected tone's p_min and fit them all again.
        """
        p_min = float(self.sweep.fields['p_min'][self.current()])
        for i in self.rows:
            self.sweep.fields['p_min'][i] = p_min
        self.sweep.compute_all()
        self._fill_list()
        self.select(self.current())
        self._refresh_overview()

    ############################################################################
    def open_calibration(self):
        """
        Open the calibration and IQ fit panels of the selected tone at
        the chosen point (the button, or C); when they close, the tone's
        fr and Sxx there are recomputed.
        """
        if self._cal_window is not None:
            self._cal_window.raise_()
            return
        i, j = self.current(), int(self.point_box.currentIndex())
        s = self.sweep
        row = int(s.rows[i, j])
        if row < 0:
            self.status.setText(f'res {self.res_idxs[i]} is not calibrated '
                                f"at {s.points[j]['name']}")
            return
        from ..pipeline.analysis import AnalysisRunner
        from ..pipeline.interactive.core import InteractiveAnalysisWindow
        from ..pipeline.interactive.iq_ts_analysis import iq_ts_panels
        DS = s.datasets[s.timestreams[j]['dataset']]['DS']
        AR = AnalysisRunner(DS, analysis_yaml_path = 'ts')
        win = InteractiveAnalysisWindow(
            AR, iq_ts_panels(AR), start_idx = 0, data_idxs = [row],
            title = f"Calibrate res {self.res_idxs[i]} at "
            f"{s.points[j]['name']} ({DS.root.path})",
            ui_scale = self._ui_scale)
        delete_on_close(win)
        self._cal_window = win
        win.show()
        # Watched rather than connected to its deletion, which can come
        # late or not at all while saves are flushed
        self._cal_timer = QtCore.QTimer(self)
        self._cal_timer.setInterval(200)
        self._cal_timer.timeout.connect(lambda: self._check_calibration(i, j))
        self._cal_timer.start()
        self.status.setText(f"Calibrating res {self.res_idxs[i]} at "
                            f"{s.points[j]['name']}...")

    def _check_calibration(self, i, j):
        """
        Recompute the tone once the calibration window has closed.

        Parameters:
        i (int): tone index.
        j (int): point index.
        """
        win = self._cal_window
        try:
            open_ = win is not None and win.isVisible()
        except RuntimeError:       # deleted already
            open_ = False
        if open_:
            return
        self._cal_timer.stop()
        self._after_calibration(i, j)

    def _after_calibration(self, i, j):
        """
        Recompute a tone at a point once its calibration window closed.

        Parameters:
        i (int): tone index.
        j (int): point index.
        """
        self._cal_window = None
        self.sweep.recompute(i, j)
        self.reviewed.add(self.res_idxs[i])
        self._show_text(i)
        if i == self.current():
            self.select(i)
        self._refresh_overview()
        self.status.setText(self.status.text() + '\n(recomputed; the final '
                            'pass runs on Done)')

    ############################################################################
    def toggle_bad(self):
        """
        Mark the selected tone bad with a reason, or clear its mark (the
        button, or B).
        """
        i = self.current()
        r = self.res_idxs[i]
        s = self.sweep
        if s.is_bad(r):
            del s.bad[r]
            self._record(r, 'clear', '')
        else:
            reason, scope = _marks.reason_and_scope(
                _marks.ask_reason(self, f'Mark res {r} bad', scope = 'local'),
                'local')
            if reason is None:
                return
            s.bad[r] = reason
            self._record(r, 'bad', reason, scope)
        self.reviewed.add(r)
        self._fill_list()
        self.select(i)
        self._refresh_overview()

    def add_note(self):
        """
        Record a note on the selected tone (the button, or T).
        """
        r = self.res_idxs[self.current()]
        note = _marks.ask_note(self, f'Note on res {r}')
        if note:
            self._record(r, 'note', note)
            self.select(self.current())

    def _record(self, r, kind, text, scope = None):
        """
        Write a mark to the store's tone marks, if the sweep is in one.

        Parameters:
        r (int): resonator index.
        kind (str): 'bad', 'note' or 'clear'.
        text (str): the reason or note.
        scope (str or None): 'global' or 'local'. Default None: local.
        """
        s = self.sweep
        if s.root is None or s.grp is None:
            return
        _marks.record(s.root, r, kind, text, s.grp.path, scope)

    ############################################################################
    def overview_values(self):
        """
        Return every tone's overview quantities (``PowerSweep``).

        Returns:
        values (dict): key: np.array per tone.
        """
        return self.sweep.overview_values()

    def outside_limits(self, values = None):
        """
        List the tones outside the review limits (``PowerSweep``).

        Parameters:
        values (dict or None): ``overview_values``. Default None.

        Returns:
        outside (list of int): tone indices.
        """
        return self.sweep.outside_limits(values)

    def open_overview(self):
        """
        Open the tone overview, or bring it to the front (the button, or
        O).
        """
        if self.overview is None:
            self.overview = ToneOverviewWindow(self, items = OVERVIEW)
        self.overview.refresh()
        self.overview.show()
        self.overview.raise_()
        self.overview.activateWindow()

    def _refresh_overview(self):
        """
        Refresh the overview if it is open, and the list if it shows only
        the tones outside the limits.
        """
        if self.overview is not None and self.overview.isVisible():
            self.overview.refresh()
        if self._only_outside:
            self._fill_list()

    def closeEvent(self, event):
        """
        Count the tone shown as reviewed and close the overview too.

        Parameters:
        event (QCloseEvent): the close event.
        """
        if self._row is not None:
            self.reviewed.add(self.res_idxs[self._row])
        if self.overview is not None:
            self.overview.close()
        super().closeEvent(event)


def _sync(box, checked):
    """
    Set a check box without firing its signal.

    Parameters:
    box (QCheckBox): the box.
    checked (bool): the state.
    """
    box.blockSignals(True)
    box.setChecked(bool(checked))
    box.blockSignals(False)


def run_nep_review(sweep, exclude = None, title = 'NEP review',
                   ui_scale = 1.0, verbose = True):
    """
    Open the review window; when it closes, run the final pass on the
    points whose calibrations changed and open it again on the final
    values with the overview, until it closes with nothing changed.

    Parameters:
    sweep (PowerSweep): the sweep (changed in place and saved by the
        caller).
    exclude (dict or None): res_idx: reason, tones left out of the
        spectra. Default None.
    title (str): window title. Default 'NEP review'.
    ui_scale (float): font size multiplier. Default 1.0.
    verbose (bool): If True (default), print the final pass's progress.

    Returns:
    passes (int): final passes run.
    """
    app = get_qapp(title)
    message, passes = '', 0
    while True:
        win = NEPReviewWindow(sweep, exclude = exclude, title = title,
                              ui_scale = ui_scale, message = message)
        delete_on_close(win)
        fit_window_to_screen(win)
        win.show()
        if passes:
            win.open_overview()
        (getattr(app, 'exec', None) or app.exec_)()
        if not sweep.changed:
            return passes
        done = sweep.final_pass(exclude = exclude, verbose = verbose)
        passes += 1
        message = (f'Final pass done on {len(done)} point(s): '
                   f"{', '.join(sweep.points[j]['name'] for j in done)}. "
                   'Check the overview; Done again to finish.')
