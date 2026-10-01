"""
Interactive sweep fitter for fitting IQ loops across a parameter sweep.

For each resonator (data_idx), displays:
  - Left:  scatter plot of a user-defined y value vs the sweep parameter, and
           a ``|S21|`` waterfall for all sweep indices.
  - Right: GainFitPanel + FitIQPanel driven by the selected sweep index.

One :class:`~citkid.pipeline_v2.analysis.AnalysisRunner` is created per sweep
index.  Each runner manages its own DataSet, storing results in a dedicated
zarr subgroup (``sweep_000/``, ``sweep_001/``, …).

Unlike pipeline.interactive (which uses runs), pipeline_v2 has a single active
state per sweep index. Re-running a step automatically invalidates downstream
outputs.

Usage
-----
::

    import zarr
    from citkid.pipeline_v2.interactive.sweep_fitter import run_sweep_fitter

    def make_custom_steps(sweep_idx):
        # Return list[plStep] that load data for this sweep index.
        ...

    run_sweep_fitter(
        make_custom_steps=make_custom_steps,
        cal_yaml_path='iq',
        analysis_yaml_path='iq',
        root=zarr.open_group('analysis.zarr', 'a'),
        n_sweep=7,
        x_param_name='ares',
        x_name='Power (dBm)',
        y_param_name='a',
        data_idxs=[0, 3, 7],  # optional subset of resonators; None = all
    )

Keyboard shortcuts
------------------
A / ←       previous resonator
D / →       next resonator
W / ↑       previous sweep index
S / ↓       next sweep index
R           auto-scale all plots
B           mark current sweep as bad
⇧B          mark all sweeps bad
Ctrl+⇧B     mark the selected sweep point and all points with larger x bad
1, 2, …     run panel N and all following panels (same as its "Run +")
Shift+N     run panel N only
"""

import sys
import threading
import numpy as np
import zarr
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..analysis import AnalysisRunner
from ..dataset import DataSet, _io_lock_for, _retry_io
from ..sweep_xy_fit import SweepXYFit, SweepXYFitStore  # noqa: F401 (SweepXYFit re-exported)
from ...qt_compat import (
    TITLE_BAR_MARGIN,
    Qt as _Qt,
    fit_window_to_screen,
    get_qapp,
    scroll_area_content_height,
    scroll_area_min_width,
    vbox_height_for_width,
)
from . import gain      # noqa: F401 — registers GainFitPanel
from . import fit_iq    # noqa: F401 — registers FitIQPanel
from .core import (
    _SectionHeader,
    get_panel_class,
    scale_plot_fonts,
    widget_font_stylesheet,
)


# Attribute on the state group holding viewed / pre-fitted rows.
_STATE_ATTR = 'sweep_fitter'

# Panel grouping matching iq_analysis_template.yaml
_IQ_PANELS = [
    ('fit_gain',),
    ('fit_iq',),
]

# Viridis colour stops (t=0 → 1)
_VIRIDIS_STOPS = [
    (0.00, (68,   1,  84)),
    (0.25, (59,  82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201,  98)),
    (1.00, (253, 231,  37)),
]


def _confirm_overwrite_xy_fit(message):
    """
    Ask whether to overwrite saved xy fits from a different fit definition.

    Parameters:
    message (str): Description of the saved fits, shown in the popup.

    Returns:
    overwrite (bool): True if the user chose Overwrite, False if they chose
        Cancel or closed the popup.
    """
    box = QtWidgets.QMessageBox()
    box.setWindowTitle('Existing XY Fits Found')
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setText(message + '\n\nOverwrite them with the new fit, or cancel?')
    overwrite_btn = box.addButton(
        'Overwrite', QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = box.addButton(
        'Cancel', QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel_btn)
    (getattr(box, 'exec', None) or box.exec_)()
    return box.clickedButton() is overwrite_btn


def _viridis_rgb(t):
    """Return an (R, G, B) tuple from a simple viridis approximation."""
    stops = _VIRIDIS_STOPS
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for j in range(len(stops) - 1):
        t0, c0 = stops[j]
        t1, c1 = stops[j + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + alpha * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


def _sweep_color(i, n):
    """Return an (R, G, B) tuple for sweep index *i* of *n*."""
    return _viridis_rgb(i / max(n - 1, 1))


################################################################################
# Main window
################################################################################

class SweepFitterWindow(QtWidgets.QMainWindow):
    _prefetch_status_changed = QtCore.pyqtSignal(str)
    """
    Main window for interactive IQ fitting across a parameter sweep.

    Left panel: scatter of ``y_func(AR, data_idx)`` vs ``x_param_name`` for
    all sweep indices, plus a ``|S21|`` waterfall for the current resonator.

    Right panel: GainFitPanel + FitIQPanel for the currently selected sweep
    index and resonator.

    Parameters:
    ARs (list of AnalysisRunner): One per sweep index.
    x_param_name (str): Name of the pipeline parameter to use as the x value
        on the sweep scatter.  Loaded as ``AR.DS.<x_param_name>[data_idx]``
        for each (sweep_idx, data_idx) pair, so x can vary per resonator.
    x_name (str): Label for the x-axis of the sweep plot.
    y_func (callable): ``y_func(AR, data_idx) -> float | None``. Called for
        each (sweep_idx, data_idx) pair to get the scatter y value.  Return
        ``None`` if the result is not yet available.
    y_name (str): Label for the y-axis of the sweep plot.
    start_sweep_idx (int): Initial sweep index. Default 0.
    start_idx (int or None): Position in ``data_idxs`` to start at (not a
        ``data_idx``), clamped to the valid range. None (default) starts at
        the first entry of ``data_idxs`` that hasn't been viewed yet (see
        ``state_group``), or at position 0 if all have been viewed or there
        is no saved state.
    data_idxs (list of int or None): Ordered data indices to review.
        Navigation, prefetching and background fitting only visit these rows.
        None (default) uses every row, ``0 .. nrows - 1``.
    title (str): Window title. Default 'Sweep Fitter'.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.
    parent (QWidget or None): Parent widget.
    state_group (zarr.Group or None): Group whose ``sweep_fitter`` attribute
        stores session state, so a later session can resume: the data
        indices the user has viewed (left by navigating away, or open when
        the window was closed) and the rows already pre-fitted (including
        failed fits, which are not retried). None (default) keeps state in
        memory only. Delete ``state_group.attrs['sweep_fitter']`` to reset.
    xy_fit (SweepXYFit or None): Optional fit of the sweep plot's y vs x
        data, one per resonator. Each resonator is fitted once its sweeps
        are pre-fitted, and again whenever its plotted x or y data changes.
        The fitted curve is drawn on the sweep plot and the outputs shown in
        its title. Fits are saved to ``xy_fit.group``, or to
        ``state_group/xy_fit`` if that is None (in memory if there is no
        state group either). None (default) disables fitting.

    Raises:
    ValueError: If ``data_idxs`` is empty or contains indices outside
        ``0 .. nrows - 1``.
    RuntimeError: If the xy fit group holds fits from a different fit
        definition and the user cancels the overwrite popup.
    """

    def __init__(
        self,
        ARs,
        x_param_name,
        x_name,
        y_func,
        y_name,
        start_sweep_idx=0,
        start_idx=None,
        data_idxs=None,
        title="Sweep Fitter",
        ui_scale=1.0,
        plot_scale=1.0,
        parent=None,
        state_group=None,
        xy_fit=None,
    ):
        """
        Build the window, toolbar, sweep plots and IQ panels.

        See the class docstring for parameter descriptions.
        """
        super().__init__(parent)
        self._ARs = list(ARs)
        self._x_param_name = x_param_name
        self._x_name = x_name
        self._y_func = y_func
        self._y_name = y_name
        self._ui_scale = ui_scale
        self._plot_scale = plot_scale
        self._n_sweep = len(self._ARs)

        self._sweep_idx: int | None = None  # nothing selected until user clicks

        try:
            self._nrows = int(self._ARs[0].DS.nrows)
        except Exception:
            self._nrows = 1

        # Ordered rows to review; navigation moves by position in this list.
        if data_idxs is None:
            self._data_idxs = list(range(self._nrows))
        else:
            self._data_idxs = [int(di) for di in np.atleast_1d(data_idxs)]
        if not self._data_idxs:
            raise ValueError('data_idxs must contain at least one data index')
        bad = [di for di in self._data_idxs if not 0 <= di < self._nrows]
        if bad:
            raise ValueError(
                f'data_idxs must be in 0 .. {self._nrows - 1}; got {bad}'
            )

        # Persistent session state (viewed and pre-fitted rows).
        self._state_group = state_group
        self._state_lock = threading.Lock()
        state = self._load_state()
        self._viewed_data_idxs: set[int] = set(state.get('viewed_data_idxs', []))
        # Rows already attempted for every sweep (fitted, loaded, or failed).
        # They are not fitted again automatically; the user can rerun panels.
        self._initialized_data_idxs: set[int] = set(state.get('prefit_attempted', []))

        # Optional y vs x fit, one per resonator
        self._xy_fit = xy_fit
        self._xy_store = None
        if xy_fit is not None:
            group = xy_fit.group
            if group is None:
                group = (state_group.require_group('xy_fit') if state_group is not None
                         else zarr.open_group(zarr.storage.MemoryStore(), mode='w'))
            store = SweepXYFitStore(group, xy_fit, self._nrows, self._n_sweep)
            if not store.matches_definition():
                if not _confirm_overwrite_xy_fit(store.describe_existing()):
                    raise RuntimeError('User cancelled operation')
                store.clear()
            self._xy_store = store

        if start_idx is None:
            unviewed = [
                pos for pos, di in enumerate(self._data_idxs)
                if di not in self._viewed_data_idxs
            ]
            start_idx = unviewed[0] if unviewed else 0
        self._nav_pos = max(0, min(int(start_idx), len(self._data_idxs) - 1))
        self._data_idx = self._data_idxs[self._nav_pos]

        # x and y value caches: {data_idx: np.ndarray of shape (n_sweep,)}
        self._x_cache: dict = {}
        self._y_cache: dict = {}

        # Prefetch state
        self._prefetch_thread: threading.Thread | None = None
        self._prefetching_idx: int | None = None
        self._prefetched_idx: int | None = None
        self._init_all_thread: threading.Thread | None = None
        self._init_all_stop = threading.Event()
        self._init_all_current_di: int | None = None
        self._worker_ars = None  # built once by the background thread

        # Background save state
        self._save_thread: threading.Thread | None = None
        self._prefetch_status_changed.connect(self._on_prefetch_status)
        QtWidgets.QApplication.instance().installEventFilter(self)

        self.setWindowTitle(title)

        # ---- central layout ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        outer.addWidget(self._build_toolbar(), 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        self._splitter = splitter

        splitter.addWidget(self._build_sweep_plots())

        # Right: panel stack inside a scroll area
        self._right_scroll = QtWidgets.QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._panel_container = QtWidgets.QWidget()
        self._panel_layout = QtWidgets.QVBoxLayout(self._panel_container)
        self._panel_layout.setAlignment(_Qt.AlignTop)
        self._panel_layout.setSpacing(6)
        self._right_scroll.setWidget(self._panel_container)
        splitter.addWidget(self._right_scroll)

        splitter.setSizes([480, 720])

        # Apply font scaling
        central.setStyleSheet(widget_font_stylesheet(ui_scale))

        # Build GainFitPanel + FitIQPanel — use ARs[0] as placeholder until
        # the user selects a sweep point by clicking the scatter.
        self.panels = []
        AR = self._ARs[0]
        for i, step_names_tuple in enumerate(_IQ_PANELS):
            cls = get_panel_class(step_names_tuple)
            panel = cls(
                AR, step_names_tuple,
                data_idx=self._data_idx,
                ui_scale=ui_scale,
                plot_scale=plot_scale,
                parent=self,
            )
            panel.panel_index = i
            panel.downstream_rerun.connect(self._on_panel_rerun)
            panel.run_from_here.connect(
                lambda p=panel: self._run_through_panel(p.panel_index)
            )
            self._add_panel(panel)

        # ---- keyboard shortcuts ----
        for seq in ('D', 'Right'):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(lambda: self._advance_resonator(+1))
        for seq in ('A', 'Left'):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(lambda: self._advance_resonator(-1))
        for seq in ('S', 'Down'):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(lambda: self._advance_sweep(+1))
        for seq in ('W', 'Up'):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(lambda: self._advance_sweep(-1))
        sc_r = QtGui.QShortcut(QtGui.QKeySequence('R'), self)
        sc_r.activated.connect(self._autoscale_all)
        sc_b = QtGui.QShortcut(QtGui.QKeySequence('B'), self)
        sc_b.activated.connect(self._mark_all_bad)
        # N: run panel N and all following ("Run +"). Shift+N: panel N only.
        for idx in range(1, 10):
            _sc = QtGui.QShortcut(QtGui.QKeySequence(str(idx)), self)
            _sc.activated.connect(lambda _i=idx - 1: self._run_through_panel(_i))
            _sc_s = QtGui.QShortcut(QtGui.QKeySequence(f'Shift+{idx}'), self)
            _sc_s.activated.connect(lambda _i=idx - 1: self._run_panel_by_index(_i))

        self._fit_to_screen(round(1400 * ui_scale))
        QtCore.QTimer.singleShot(0, self._auto_initialize_all)

    def _fit_to_screen(self, width: int):
        """
        Size the window to show every panel without scrolling, and center it.

        The width is set first, since the word-wrapped toolbar hints make the
        needed height depend on it. The splitter then gives the panels 60% of
        the width, or at least the width they need to avoid a horizontal
        scrollbar, with the plots taking the rest. The height is capped at
        the screen height, beyond which the panels scroll.

        Parameters:
        width (int): Preferred width in logical pixels (shrunk to fit the
            screen).
        """
        # Apply styles and fonts, and the layout's minimum size, before
        # measuring; otherwise a hidden window measures text at the wrong
        # font and width.
        self.ensurePolished()
        self.layout().activate()
        fit_window_to_screen(self, frac=0.9, size=(width, 1), height_margin=TITLE_BAR_MARGIN)
        self._splitter.setSizes(self._splitter_sizes())
        fit_window_to_screen(self, frac=0.9, size=(self.width(), self._content_height()),
                             height_margin=TITLE_BAR_MARGIN)

    def _splitter_sizes(self) -> list:
        """
        Return [plots, panels] splitter widths for the window's current width.

        Returns:
        sizes (list of int): Widths in logical pixels. The panels get 60% of
            the space, or at least the width they need to avoid a horizontal
            scrollbar, but never so much that the plots drop below their
            minimum width.
        """
        margins = self.centralWidget().layout().contentsMargins()
        total = (self.width() - margins.left() - margins.right()
                 - self._splitter.handleWidth())
        # The splitter honours an explicit minimum width as well as the hint.
        plots_min = max(self._gw.minimumSizeHint().width(), self._gw.minimumWidth())
        panels = max(round(0.6 * total), scroll_area_min_width(self._right_scroll))
        panels = min(panels, total - plots_min)
        return [total - panels, panels]

    def _content_height(self) -> int:
        """
        Return the window height that shows every panel without scrolling,
        at the window's current width.

        The sweep plots on the left stretch to any height, so only their
        minimum height counts.

        Returns:
        height (int): Height of the toolbar (wrapped at the current width)
            plus the taller of the full panel stack and the plots' minimum
            height, in logical pixels.
        """
        splitter_height = max(
            scroll_area_content_height(self._right_scroll, width=self._splitter_sizes()[1]),
            max(self._gw.minimumSizeHint().height(), self._gw.minimumHeight()),
        )
        return vbox_height_for_width(
            self.centralWidget().layout(), self.width(),
            {self._splitter: splitter_height},
        )

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QtWidgets.QWidget:
        """
        Build the two-row toolbar.

        Row 1 holds navigation, sweep selection and status labels. Row 2
        holds the shortcut hints and the batch-action buttons. The hints
        label word-wraps, so the toolbar never forces the window wider than
        the screen.

        Returns:
        w (QWidget): The toolbar widget.
        """
        w = QtWidgets.QWidget()
        rows = QtWidgets.QVBoxLayout(w)
        rows.setContentsMargins(4, 2, 4, 2)
        rows.setSpacing(2)
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(8)
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)
        rows.addLayout(layout)
        rows.addLayout(row2)

        # Resonator navigation
        self._res_prev_btn = QtWidgets.QPushButton('◀')
        self._res_prev_btn.setFixedWidth(30)
        self._res_prev_btn.setToolTip('Previous resonator  (A / ←)')
        self._res_prev_btn.clicked.connect(lambda: self._advance_resonator(-1))
        layout.addWidget(self._res_prev_btn)

        self._res_label = QtWidgets.QLabel()
        self._res_label.setMinimumWidth(60)
        self._res_label.setAlignment(_Qt.AlignCenter)
        layout.addWidget(self._res_label)

        self._res_next_btn = QtWidgets.QPushButton('▶')
        self._res_next_btn.setFixedWidth(30)
        self._res_next_btn.setToolTip('Next resonator  (D / →)')
        self._res_next_btn.clicked.connect(lambda: self._advance_resonator(+1))
        layout.addWidget(self._res_next_btn)

        layout.addWidget(QtWidgets.QLabel('data_idx:'))
        self._data_idx_spin = QtWidgets.QSpinBox()
        self._data_idx_spin.setMinimum(min(self._data_idxs))
        self._data_idx_spin.setMaximum(max(self._data_idxs))
        self._data_idx_spin.setValue(self._data_idx)
        self._data_idx_spin.valueChanged.connect(self._on_data_idx_spin_changed)
        layout.addWidget(self._data_idx_spin)

        layout.addSpacing(16)

        # Sweep selection — dropdown showing index + x value for each sweep
        layout.addWidget(QtWidgets.QLabel(f'sweep index, {self._x_name}:'))
        self._sweep_combo = QtWidgets.QComboBox()
        self._sweep_combo.setMinimumWidth(160)
        for i in range(self._n_sweep):
            self._sweep_combo.addItem(f'{i + 1}, —')
        self._sweep_combo.setCurrentIndex(-1)  # nothing selected on startup
        self._sweep_combo.currentIndexChanged.connect(self._on_sweep_combo_changed)
        layout.addWidget(self._sweep_combo)

        layout.addStretch()

        # Row 2: word-wrapping shortcut hints, then batch actions
        hints = QtWidgets.QLabel(
            '[A/D] resonator   [W/S] sweep   [R] rescale   [B] mark bad'
            '   [⇧B] mark all bad   [Ctrl+⇧B] mark bad above   [N] run+following'
            '   [⇧N] run panel only   [⇧A] apply to all'
        )
        hints.setWordWrap(True)
        hints.setStyleSheet('color: palette(mid); font-style: italic;')
        row2.addWidget(hints, 1)

        apply_all_btn = QtWidgets.QPushButton('Apply to All')
        apply_all_btn.setToolTip(
            'Apply current panel settings to every dataset in the active sweep and save'
        )
        apply_all_btn.clicked.connect(self._apply_to_all)
        row2.addWidget(apply_all_btn)

        mark_all_sweeps_bad_btn = QtWidgets.QPushButton('Mark All Sweeps Bad')
        mark_all_sweeps_bad_btn.setToolTip(
            'Mark all sweep indices for the current resonator as bad (Shift+B)'
        )
        mark_all_sweeps_bad_btn.clicked.connect(self._mark_all_sweeps_bad)
        row2.addWidget(mark_all_sweeps_bad_btn)

        mark_bad_above_btn = QtWidgets.QPushButton('Mark Bad Above')
        mark_bad_above_btn.setToolTip(
            'Mark the selected sweep point and every point with a larger x '
            f'({self._x_name}) as bad for the current resonator (Ctrl+Shift+B)'
        )
        mark_bad_above_btn.clicked.connect(self._mark_bad_above)
        row2.addWidget(mark_bad_above_btn)

        self._apply_status_label = QtWidgets.QLabel('')
        self._apply_status_label.setMinimumWidth(120)
        layout.addWidget(self._apply_status_label)

        self._prefetch_label = QtWidgets.QLabel('')
        self._prefetch_label.setMinimumWidth(110)
        self._prefetch_label.setToolTip('Background prefetch status for the next resonator')
        layout.addWidget(self._prefetch_label)

        self._update_res_label()
        return w

    def _build_sweep_plots(self) -> QtWidgets.QWidget:
        """Build the left-side sweep-plot widget."""
        gw = pg.GraphicsLayoutWidget()
        gw.setMinimumWidth(380)

        # Top: y vs x scatter
        self._plot_sweep = gw.addPlot(row=0, col=0, title='Sweep')
        self._plot_sweep.setLabel('left', self._y_name)
        self._plot_sweep.setLabel('bottom', self._x_name)
        self._plot_sweep.showGrid(x=True, y=True, alpha=0.3)

        # One ScatterPlotItem per sweep index so each gets its own colour.
        # The spot's `data` field carries the sweep index for click handling.
        self._scatter_items = []
        for i in range(self._n_sweep):
            r, g, b = _sweep_color(i, self._n_sweep)
            si = pg.ScatterPlotItem(
                size=10,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(r, g, b, 200),
            )
            si.sigClicked.connect(self._on_scatter_clicked)
            self._plot_sweep.addItem(si)
            self._scatter_items.append(si)

        # Ring marker for the currently selected sweep index
        self._selected_marker = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen('w', width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self._plot_sweep.addItem(self._selected_marker)

        # Optional fitted y(x) curve. ignoreBounds keeps it out of
        # auto-ranging, so the axes follow the data even if the fit
        # extrapolates far outside it.
        self._xy_fit_curve = None
        if self._xy_fit is not None:
            self._xy_fit_curve = pg.PlotDataItem(
                [], [], pen=pg.mkPen('w', width=1.5, style=_Qt.DashLine))
            self._plot_sweep.addItem(self._xy_fit_curve, ignoreBounds=True)

        # Bottom: |S21| waterfall
        self._plot_waterfall = gw.addPlot(row=1, col=0, title='|S21| Waterfall')
        self._plot_waterfall.setLabel('left', '|S21| + offset (dB)')
        self._plot_waterfall.setLabel('bottom', 'Frequency (Hz)')
        self._plot_waterfall.showGrid(x=True, y=True, alpha=0.3)
        self._waterfall_curves: list = []

        scale_plot_fonts(self._ui_scale, self._plot_sweep, self._plot_waterfall)

        self._gw = gw
        return gw

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.KeyPress and self.isActiveWindow():
            if self._handle_modified_letter_shortcut(event.key(), event.modifiers()):
                return True
        return super().eventFilter(obj, event)

    def _handle_modified_letter_shortcut(self, key, modifiers) -> bool:
        """
        Handle shifted letter shortcuts that collide with plain-letter bindings.

        Ctrl+Shift+B marks the selected sweep point and all points above it
        bad. Shift+A and Shift+B fire only without Ctrl.

        Parameters:
        key (int): Qt key code.
        modifiers (Qt.KeyboardModifiers): Modifiers held with the key.

        Returns:
        handled (bool): True if the key press was handled.
        """
        shift = bool(modifiers & _Qt.ShiftModifier)
        ctrl = bool(modifiers & _Qt.ControlModifier)
        if shift and ctrl and key == QtCore.Qt.Key.Key_B:
            self._mark_bad_above()
            return True
        if not shift or ctrl:
            return False
        if key == QtCore.Qt.Key.Key_A:
            self._apply_to_all()
            return True
        if key == QtCore.Qt.Key.Key_B:
            self._mark_all_sweeps_bad()
            return True
        return False

    def _add_panel(self, panel):
        if self.panels:
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            self._panel_layout.addWidget(sep)
        header = _SectionHeader(panel)
        self._panel_layout.addWidget(header)
        self.panels.append(panel)

    # ------------------------------------------------------------------
    # Persistent session state
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        """
        Read saved session state from ``state_group``.

        Returns:
        state (dict): Keys ``viewed_data_idxs`` and ``prefit_attempted``
            (lists of int). Empty if there is no state group or saved state.
        """
        if self._state_group is None:
            return {}
        try:
            with _io_lock_for(self._state_group):
                return dict(self._state_group.attrs.get(_STATE_ATTR, {}))
        except Exception as exc:
            print(f'Warning: could not read sweep fitter state: {exc}')
            return {}

    def _save_state(self):
        """
        Write the viewed and pre-fitted rows to ``state_group``.

        Safe to call from the UI thread and the background worker.
        """
        if self._state_group is None:
            return
        with self._state_lock:
            state = {
                'viewed_data_idxs': sorted(self._viewed_data_idxs),
                'prefit_attempted': sorted(self._initialized_data_idxs),
            }
            try:
                with _io_lock_for(self._state_group):
                    _retry_io(self._state_group.attrs.__setitem__, _STATE_ATTR, state)
            except Exception as exc:
                print(f'Warning: could not save sweep fitter state: {exc}')

    def _mark_viewed(self, data_idx: int):
        """
        Record that the user has viewed a data index, and save the state.

        Parameters:
        data_idx (int): Data index the user is leaving or closing.
        """
        with self._state_lock:
            if int(data_idx) in self._viewed_data_idxs:
                return
            self._viewed_data_idxs.add(int(data_idx))
        self._save_state()

    def _mark_attempted(self, data_idx: int):
        """
        Record that a row has been pre-fitted for every sweep, and save it.

        Parameters:
        data_idx (int): Row that was fitted, loaded, or failed.
        """
        with self._state_lock:
            if int(data_idx) in self._initialized_data_idxs:
                return
            self._initialized_data_idxs.add(int(data_idx))
        self._save_state()

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _update_res_label(self):
        self._res_label.setText(f'{self._nav_pos + 1} / {len(self._data_idxs)}')

    def _update_sweep_combo_items(self, data_idx: int):
        """Repopulate every combo item text with the x value for *data_idx*."""
        self._sweep_combo.blockSignals(True)
        for i in range(self._n_sweep):
            xi = self._get_x_value(i, data_idx)
            label = f'{i + 1}, {xi:.4g}' if xi is not None else f'{i + 1}, —'
            self._sweep_combo.setItemText(i, label)
        self._sweep_combo.blockSignals(False)

    def _update_sweep_combo_selection(self):
        """Sync the combo's selected index to self._sweep_idx."""
        self._sweep_combo.blockSignals(True)
        self._sweep_combo.setCurrentIndex(
            -1 if self._sweep_idx is None else self._sweep_idx
        )
        self._sweep_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _advance_resonator(self, delta: int):
        """
        Move ``delta`` positions through ``data_idxs``, clamped to its ends.

        Parameters:
        delta (int): Number of positions to move (negative moves back).
        """
        new_pos = max(0, min(len(self._data_idxs) - 1, self._nav_pos + delta))
        new_di = self._data_idxs[new_pos]
        if new_di != self._data_idx:
            self._set_data_idx(new_di)

    def _advance_sweep(self, delta: int):
        base = self._sweep_idx if self._sweep_idx is not None else (
            -1 if delta > 0 else self._n_sweep
        )
        new_si = max(0, min(self._n_sweep - 1, base + delta))
        if new_si != self._sweep_idx:
            self._set_sweep_idx(new_si)

    def _on_data_idx_spin_changed(self, value: int):
        """
        Jump to the data index typed or stepped to in the spin box.

        Values not in ``data_idxs`` snap to the nearest member in the
        direction of the change, so the arrows skip rows outside the subset.

        Parameters:
        value (int): New spin box value.
        """
        if value not in self._data_idxs:
            if value > self._data_idx:
                candidates = [di for di in self._data_idxs if di >= value]
                value = min(candidates) if candidates else self._data_idx
            else:
                candidates = [di for di in self._data_idxs if di <= value]
                value = max(candidates) if candidates else self._data_idx
        if value != self._data_idx:
            self._set_data_idx(value)
        else:
            self._data_idx_spin.blockSignals(True)
            self._data_idx_spin.setValue(self._data_idx)
            self._data_idx_spin.blockSignals(False)

    def _on_sweep_combo_changed(self, value: int):
        if value != self._sweep_idx:
            self._set_sweep_idx(value)

    def _set_data_idx(self, new_di: int):
        """Change the active resonator, load/run data, and refresh the scatter."""
        if not self._confirm_stale_downstream_before_leave():
            self._data_idx_spin.blockSignals(True)
            self._data_idx_spin.setValue(self._data_idx)
            self._data_idx_spin.blockSignals(False)
            return
        self._save_dirty_panels()  # persist results for the outgoing resonator
        self._mark_viewed(self._data_idx)
        self._data_idx = new_di
        self._nav_pos = self._data_idxs.index(new_di)
        self._data_idx_spin.blockSignals(True)
        self._data_idx_spin.setValue(new_di)
        self._data_idx_spin.blockSignals(False)
        self._update_res_label()

        # Reset sweep selection so no point is pre-selected on the new dataset.
        self._sweep_idx = None
        self._update_sweep_combo_selection()

        for panel in self.panels:
            panel.data_idx = new_di
            panel.on_data_idx_changing()
            panel.clear_plots()
            panel._needs_run = False
            panel._dirty = False
            panel._has_run = False
            # Reset status labels when changing data_idx
            if hasattr(panel, '_status_label'):
                panel._status_label.setText('')
            if hasattr(panel, '_marked_bad_label'):
                panel._marked_bad_label.setText('')

        # Silently run (or load) the full pipeline for every sweep AR at the
        # new data_idx, so the scatter and waterfall are fully populated and
        # the active panels can show results immediately.
        self._ensure_data_idx_initialized(new_di)
        self._update_sweep_combo_items(new_di)
        self._update_sweep_scatter()
        self._update_waterfall()
        self._autoscale_all()
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    def _set_sweep_idx(self, new_si: int):
        """Change the active sweep index, swap ARs in panels, and re-run."""
        if not self._confirm_stale_downstream_before_leave():
            self._update_sweep_combo_selection()
            return
        self._save_dirty_panels()  # persist results for the outgoing sweep point
        self._sweep_idx = new_si
        self._update_sweep_combo_selection()

        new_AR = self._ARs[new_si]
        for panel in self.panels:
            panel.AR = new_AR
            panel.on_data_idx_changing()

        # Satisfy global prerequisites (e.g. make_fr_spans) for the new AR
        # before running panels.  All panels share the same AR so one pass
        # through the first panel is sufficient.
        if self.panels:
            self.panels[0]._ensure_global_prerequisites()

        self._run_all_panels()
        self._update_selected_marker()

    def _on_scatter_clicked(self, plot_item, spots):
        """Select a sweep index by clicking its scatter point."""
        if not spots:
            return
        si = spots[0].data()
        if si is not None:
            self._set_sweep_idx(int(si))

    # ------------------------------------------------------------------
    # Panel execution
    # ------------------------------------------------------------------

    def _batch_init_all_sweeps(self):
        """
        Silently run or load all pipeline steps for every AR at self._data_idx.

        For each sweep AR: if ``iq_popt`` already exists in zarr for the
        current data_idx it is loaded as-is; otherwise the full pipeline
        (make_fr_spans → fit_gain → fit_iq) is executed and saved to disk.
        Pre-populates the y-cache so the scatter is fully drawn on startup.
        """
        di = self._data_idx
        self._initialize_all_sweeps_for_data_idx(di)
        self._mark_attempted(int(di))

    def _initialize_all_sweeps_for_data_idx(self, data_idx: int):
        """
        Silently run or load all pipeline steps for every AR at data_idx.
        """
        for AR in self._ARs:
            if not self._runner_outputs_exist(AR, data_idx):
                try:
                    self._initialize_runner_outputs(AR, data_idx)
                except Exception as exc:
                    print(f"Warning: batch init failed for sweep AR: {exc}")

    def _auto_initialize_all(self):
        self._batch_init_all_sweeps()
        self._update_sweep_combo_items(self._data_idx)
        self._update_sweep_scatter()
        self._update_waterfall()
        self._start_background_initialize_remaining()
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_next(self):
        """
        Pre-compute read-only caches for the next resonator.
        """
        import threading as _threading
        next_pos = self._nav_pos + 1
        if next_pos >= len(self._data_idxs):
            return
        next_di = self._data_idxs[next_pos]
        if next_di == self._prefetched_idx:
            return
        if (self._prefetch_thread is not None
                and self._prefetch_thread.is_alive()
                and self._prefetching_idx == next_di):
            return

        self._prefetching_idx = next_di
        self._prefetch_status_changed.emit(f'\u27f3 prefetch {next_di}')
        ARs = list(self._ARs)

        def _worker():
            try:
                self._get_x_array(next_di)
                self._get_y_array(next_di)
                if self._sweep_idx is not None:
                    active_ar = ARs[self._sweep_idx]
                    for panel in self.panels:
                        old_ar = panel.AR
                        old_di = panel.data_idx
                        try:
                            panel.AR = active_ar
                            panel.data_idx = next_di
                            panel.prefetch_plot_data(next_di)
                        finally:
                            panel.AR = old_ar
                            panel.data_idx = old_di
                self._prefetched_idx = next_di
                self._prefetch_status_changed.emit(f'\u2713 prefetch {next_di}')
            except Exception as exc:
                self._prefetch_status_changed.emit('')
                print(f'[prefetch] data_idx={next_di} failed: {exc}')

        self._prefetch_thread = _threading.Thread(
            target=_worker, daemon=True, name=f'prefetch-{next_di}'
        )
        self._prefetch_thread.start()

    def _on_prefetch_status(self, msg: str):
        """Update the prefetch label on the UI thread via signal."""
        self._prefetch_label.setText(msg)
        if msg.startswith('\u2713'):
            QtCore.QTimer.singleShot(
                4000, lambda: self._prefetch_label.setText('')
            )

    def _save_dirty_panels(self):
        """Save any panels that have unsaved results for the current state."""
        if self._sweep_idx is None:
            return
        for panel in self.panels:
            if panel._dirty:
                try:
                    panel.save_outputs()
                except Exception as exc:
                    print(f"Auto-save failed for {panel.step_names}: {exc}")

    def closeEvent(self, event):
        """
        Save, record the open data index as viewed, and consolidate storage.

        The background worker is stopped first (it finishes at most its
        current fit), so nothing writes while buffered rows are merged into
        the sharded arrays.

        Parameters:
        event (QCloseEvent): Close event.
        """
        if not self._confirm_stale_downstream_before_leave():
            event.ignore()
            return
        self._init_all_stop.set()
        self._save_dirty_panels()
        self._mark_viewed(self._data_idx)

        dialog = self._make_busy_dialog('Finishing background fit and saving…', 'Closing')
        dialog.show()
        QtWidgets.QApplication.processEvents()
        try:
            if self._init_all_thread is not None:
                while self._init_all_thread.is_alive():
                    self._init_all_thread.join(timeout=0.05)
                    QtWidgets.QApplication.processEvents()
            self._consolidate_storage()
        finally:
            dialog.close()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        super().closeEvent(event)

    def _consolidate_storage(self):
        """
        Merge buffered rows into the sharded zarr arrays for every sweep.
        """
        for si, AR in enumerate(self._ARs):
            try:
                AR.DS.consolidate_storage()
            except Exception as exc:
                print(f'Warning: consolidating storage for sweep_idx={si} failed: {exc}. '
                      'Call DS.consolidate_storage() to retry.')

    def _run_all_panels(self):
        for panel in self.panels:
            ok = panel.run_steps()
            if not ok:
                break
        # Refresh the scatter point for the current sweep index after running
        self._update_sweep_point(self._sweep_idx)

    def _apply_to_all(self):
        """
        Apply the current panel settings to every sweep index for the current
        data_idx and save results to zarr.

        Each panel's ``get_params_for_step`` is called with the current widget
        state (e.g. span_mult, iq_mask) so the same settings are used for
        every sweep index.  Results are saved after each sweep index.
        """
        if self._sweep_idx is None:
            return

        di = self._data_idx

        # Snapshot current settings from each panel before iterating.
        # Each entry is a list of (step, params_dict) pairs.
        panel_params = []
        for panel in self.panels:
            step_params = [(step, panel.get_params_for_step(step))
                           for step in panel.steps]
            panel_params.append(step_params)

        total = self._n_sweep
        errors = []
        for si in range(total):
            self._apply_status_label.setText(f'Applying {si + 1}/{total}…')
            QtWidgets.QApplication.processEvents()
            AR = self._ARs[si]
            try:
                for (step_params_list, panel) in zip(panel_params, self.panels):
                    for step, params in step_params_list:
                        step_di = (
                            None
                            if step.func_type in ('global', 'global-res')
                            else di
                        )
                        AR.execute_step(
                            step, data_idx=step_di,
                            user_params=params, save=True,
                        )
            except Exception as exc:
                errors.append((si, exc))
                print(f'Apply to all: error at sweep_idx={si}: {exc}')

        # Invalidate y-cache for this data_idx so scatter refreshes.
        self._y_cache.pop(di, None)
        self._update_sweep_scatter()

        if errors:
            self._apply_status_label.setText(
                f'Done with {len(errors)} error(s)'
            )
        else:
            self._apply_status_label.setText(f'Applied to all {total} ✓')

    def _run_panel_by_index(self, index: int):
        if index >= len(self.panels):
            return
        self.panels[index].run_current()

    def _run_through_panel(self, index: int):
        for panel in self.panels[index:]:
            panel.prepare_run()
            if hasattr(panel, '_status_label'):
                panel._status_label.setText('Running…')
                QtWidgets.QApplication.processEvents()
            ok = panel.run_steps()
            if ok:
                if hasattr(panel, '_status_label'):
                    panel._status_label.setText('Done ✓')
            else:
                break
        self._update_sweep_point(self._sweep_idx)

    def _on_panel_rerun(self, source_panel):
        """Mark downstream panels stale, then refresh the current sweep point."""
        try:
            src_idx = self.panels.index(source_panel)
        except ValueError:
            return
        for panel in self.panels[src_idx + 1:]:
            panel.mark_stale()
        if self._sweep_idx is not None:
            self._update_sweep_point(self._sweep_idx)

    def _mark_all_bad(self):
        """Mark every panel's outputs as NaN, clear their plots, and refresh scatter."""
        for panel in self.panels:
            panel._write_nan_outputs()
            panel.clear_plots()
        if self._sweep_idx is not None:
            self._update_sweep_point(self._sweep_idx)

    def _mark_all_sweeps_bad(self):
        """Mark every panel's outputs as NaN for all sweeps at current data_idx."""
        self._mark_sweeps_bad(range(self._n_sweep))

    def _mark_bad_above(self):
        """
        Mark the selected sweep point and every point with a larger x as bad.

        Uses the x values on the sweep scatter for the current data_idx.
        Points whose x is unavailable are left alone. Does nothing (and says
        so in the status label) if no sweep point is selected.
        """
        if self._sweep_idx is None:
            self._apply_status_label.setText('Select a sweep point first')
            return
        x = self._get_x_array(self._data_idx)
        x_sel = x[self._sweep_idx]
        sweep_idxs = [
            si for si in range(self._n_sweep)
            if si == self._sweep_idx or (np.isfinite(x[si]) and x[si] > x_sel)
        ]
        self._mark_sweeps_bad(sweep_idxs)
        for panel in self.panels:
            panel.clear_plots()
        self._apply_status_label.setText(f'Marked {len(sweep_idxs)} sweep(s) bad')

    def _mark_sweeps_bad(self, sweep_idxs):
        """
        Mark every panel's outputs as NaN for some sweeps at the current data_idx.

        Parameters:
        sweep_idxs (iterable of int): Sweep indices to mark bad.
        """
        di = self._data_idx
        old_AR = self.panels[0].AR
        for si in sweep_idxs:
            AR = self._ARs[si]
            for panel in self.panels:
                panel.AR = AR
            for panel in self.panels:
                panel._write_nan_outputs()
        for panel in self.panels:
            panel.AR = old_AR
        # Invalidate y-cache so scatter refreshes without these points
        self._y_cache.pop(di, None)
        self._update_sweep_scatter()

    def _runner_outputs_exist(self, AR, data_idx: int) -> bool:
        """
        Return True when the final analysis-step outputs already exist for data_idx.
        """
        if not getattr(AR, 'path', None):
            return False
        final_step = AR.path[-1]['task']
        for name in final_step.return_names:
            try:
                attr = getattr(AR.DS, name)
                value = attr[data_idx]
                if value is None:
                    return False
            except Exception:
                return False
        return True

    def _saved_output_rows(self, AR):
        """
        Return which rows have the final analysis-step outputs saved.

        Parameters:
        AR (AnalysisRunner): Runner for one sweep index.

        Returns:
        mask (np.ndarray or None): Boolean array of length ``nrows``, or None
            if it can't be determined (callers then check rows one at a
            time).
        """
        try:
            final_step = AR.path[-1]['task']
            masks = [AR.DS.saved_row_mask(name) for name in final_step.return_names]
        except Exception:
            return None
        if not masks or not all(isinstance(mask, np.ndarray) for mask in masks):
            return None
        return np.logical_and.reduce(masks)

    def _all_sweeps_outputs_exist(self, data_idx: int) -> bool:
        """
        Return True when every sweep runner already has final outputs for data_idx.
        """
        return all(self._runner_outputs_exist(AR, data_idx) for AR in self._ARs)

    def _initialize_runner_outputs(self, AR, data_idx: int):
        """
        Ensure global-prefix analysis steps exist once, then run the per-row suffix.
        """
        start_from_idx = self._global_prefix_length(AR)
        for step_dict in AR.path[:start_from_idx]:
            step = step_dict['task']
            step_data_idx = None if step.func_type == 'global' else data_idx
            if self._step_outputs_exist(AR, step, data_idx=step_data_idx):
                continue
            AR.execute_step(step, data_idx=None, save=True)

        if start_from_idx < len(AR.path):
            AR.execute_path(data_idx=data_idx, start_from_idx=start_from_idx, save=True, verbose=False)

    def _start_background_initialize_remaining(self):
        """
        Initialize every unattempted entry of ``data_idxs`` in the background
        using worker ARs.

        Rows after the current position are done first (in navigation order),
        then rows before it. Whether outputs already exist is checked in the
        worker thread, so the UI isn't blocked scanning zarr.
        """
        if not hasattr(self, '_worker_root'):
            return
        if self._init_all_thread is not None and self._init_all_thread.is_alive():
            return

        pos = self._nav_pos
        ordered = self._data_idxs[pos + 1:] + self._data_idxs[:pos]
        # Also revisit pre-fitted rows that still lack an xy fit (e.g. the
        # fit was added or overwritten after they were pre-fitted).
        xy_fitted = self._xy_store.fitted_rows() if self._xy_store is not None else None
        remaining = [
            di for di in ordered
            if di not in self._initialized_data_idxs
            or (xy_fitted is not None and not xy_fitted[di])
        ]
        if not remaining:
            return

        self._init_all_stop.clear()

        def _worker():
            """
            Build the worker runners once, then fit the remaining rows.
            """
            try:
                if self._worker_ars is None:
                    self._worker_ars = self._make_worker_ars()
                self._initialize_remaining_data_indices(self._worker_ars, remaining)
            except Exception as exc:
                print(f'[init-all] stopped: {type(exc).__name__}: {exc}')

        self._init_all_thread = threading.Thread(
            target=_worker,
            daemon=True,
            name='sweep-fitter-init-all',
        )
        self._init_all_thread.start()

    def _make_worker_ars(self):
        """
        Build background worker AnalysisRunner instances for each sweep subgroup.
        """
        worker_ars = []
        for i in range(self._n_sweep):
            group = self._worker_root.require_group(f'sweep_{i:03d}')
            ds = DataSet(
                zarr_path=group,
                cal_yaml_path=self._worker_cal_yaml_path,
                custom_cal_steps=self._worker_make_custom_steps(i),
                write_buffer=True,
            )
            ar = AnalysisRunner(ds, analysis_yaml_path=self._worker_analysis_yaml_path)
            worker_ars.append(ar)
        return worker_ars

    def _initialize_remaining_data_indices(self, worker_ars, data_indices):
        """
        Synchronously initialize remaining resonators for all sweep runners.

        A failure for one (row, sweep) is printed and skipped rather than
        stopping the loop. The stop flag is checked before each sweep, so a
        stop request waits for at most one fit. A row is marked attempted only
        once every sweep has been tried. Then its xy fit (if any) is computed
        and saved, and its cached per-row data is released so memory doesn't
        grow over the session. Rows attempted in an earlier session only get
        the xy fit, if they lack one.

        Parameters:
        worker_ars (list of AnalysisRunner): One worker runner per sweep.
        data_indices (list of int): Rows to initialize, in order.
        """
        # One zarr read per sweep says which rows are already fitted.
        done_masks = [self._saved_output_rows(ar) for ar in worker_ars]
        xy_fitted = self._xy_store.fitted_rows() if self._xy_store is not None else None
        for di in data_indices:
            di = int(di)
            if di in self._initialized_data_idxs:
                # Pre-fitted in an earlier session; it may still need an xy fit.
                if xy_fitted is not None and not xy_fitted[di]:
                    if self._init_all_stop.is_set():
                        return
                    self._fit_xy_in_background(worker_ars, di)
                continue
            self._init_all_current_di = di
            for si, ar in enumerate(worker_ars):
                if self._init_all_stop.is_set():
                    self._init_all_current_di = None
                    return
                done = done_masks[si]
                if done is not None and done[di]:
                    continue
                if done is None and self._runner_outputs_exist(ar, di):
                    continue
                try:
                    self._initialize_runner_outputs(ar, di)
                except Exception as exc:
                    print(f'[init-all] data_idx={di}, sweep_idx={si} failed: '
                          f'{type(exc).__name__}: {exc}')
            self._mark_attempted(di)
            self._fit_xy_in_background(worker_ars, di)
            for ar in worker_ars:
                try:
                    ar.release_rows(di)
                except Exception:
                    pass
        self._init_all_current_di = None

    def _ensure_data_idx_initialized(self, data_idx: int):
        """
        Ensure that data_idx has been fully initialized before the user edits it.

        Rows that were already attempted (including ones whose fits failed)
        are not fitted again, so revisiting a bad resonator doesn't block.

        Parameters:
        data_idx (int): Row to initialize.
        """
        di = int(data_idx)
        if di in self._initialized_data_idxs:
            return
        if self._all_sweeps_outputs_exist(di):
            self._mark_attempted(di)
            return

        self._init_all_stop.set()
        dialog = self._make_fitting_dialog(di)
        dialog.show()
        QtWidgets.QApplication.processEvents()
        try:
            if self._init_all_thread is not None and self._init_all_thread.is_alive():
                while self._init_all_thread.is_alive():
                    self._init_all_thread.join(timeout=0.05)
                    QtWidgets.QApplication.processEvents()
            self._initialize_all_sweeps_for_data_idx(di)
            self._mark_attempted(di)
        finally:
            dialog.close()
            self._init_all_stop.clear()
            self._start_background_initialize_remaining()

    def _make_fitting_dialog(self, data_idx: int):
        """
        Create a simple modal progress dialog for synchronous row initialization.

        Parameters:
        data_idx (int): Row being fitted.

        Returns:
        dialog (QProgressDialog): The dialog, not yet shown.
        """
        return self._make_busy_dialog(
            f'Fitting data_idx {data_idx} across all sweeps…', 'Initializing Sweeps'
        )

    def _make_busy_dialog(self, text: str, title: str):
        """
        Create a modal busy dialog with no cancel or close button.

        Parameters:
        text (str): Message shown in the dialog.
        title (str): Window title.

        Returns:
        dialog (QProgressDialog): The dialog, not yet shown.
        """
        dialog = QtWidgets.QProgressDialog(text, None, 0, 0, self)
        dialog.setWindowTitle(title)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        flags = dialog.windowFlags()
        flags &= ~QtCore.Qt.WindowType.WindowCloseButtonHint
        flags &= ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        dialog.setWindowFlags(flags)
        return dialog

    def _global_prefix_length(self, AR) -> int:
        """
        Return the number of leading analysis steps with global/global-res func_type.
        """
        if not getattr(AR, 'path', None):
            return 0
        prefix_len = 0
        for step_dict in AR.path:
            if step_dict['task'].func_type not in ('global', 'global-res'):
                break
            prefix_len += 1
        return prefix_len

    def _step_outputs_exist(self, AR, step, data_idx):
        """
        Return True when every output for step already exists for the requested scope.
        """
        for name in step.return_names:
            try:
                attr = getattr(AR.DS, name)
                if step.func_type == 'global':
                    _ = attr
                    continue
                if data_idx is None:
                    return False
                _ = attr[data_idx]
            except Exception:
                return False
        return True

    def _stale_panels_after(self, source_panel=None):
        """
        Return downstream panels whose outputs were invalidated and not rerun.
        """
        start_index = 0
        if source_panel is not None:
            try:
                start_index = self.panels.index(source_panel) + 1
            except ValueError:
                start_index = 0
        return [panel for panel in self.panels[start_index:] if getattr(panel, '_needs_run', False)]

    def _confirm_stale_downstream_before_leave(self):
        """
        Ask before leaving the current state while downstream panels remain stale.
        """
        stale = self._stale_panels_after()
        if not stale:
            return True
        names = ', '.join(' + '.join(panel.step_names) for panel in stale)
        reply = QtWidgets.QMessageBox.question(
            self,
            'Panels Need Run',
            'Earlier panel changes invalidated later panel outputs for the current selection.\n\n'
            f'Panels needing a rerun: {names}\n\nLeave anyway?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _confirm_stale_downstream_before_save(self, source_panel):
        """
        Ask before saving when later panels remain stale.
        """
        stale = self._stale_panels_after(source_panel)
        if not stale:
            return True
        names = ', '.join(' + '.join(panel.step_names) for panel in stale)
        reply = QtWidgets.QMessageBox.question(
            self,
            'Downstream Panels Need Run',
            'Saving now will keep later panel outputs missing for the current sweep selection.\n\n'
            f'Panels needing a rerun: {names}\n\nContinue saving?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _autoscale_all(self):
        for panel in self.panels:
            panel.autoscale_plots()
        self._plot_sweep.autoRange()
        self._plot_waterfall.autoRange()

    # ------------------------------------------------------------------
    # Sweep plot
    # ------------------------------------------------------------------

    def _get_x_value(self, sweep_idx: int, data_idx: int):
        """Return the x value for (sweep_idx, data_idx), or None on failure."""
        return self._x_of(self._ARs[sweep_idx], data_idx)

    def _x_of(self, AR, data_idx: int):
        """
        Return the sweep-plot x value of one runner at a data index.

        Parameters:
        AR (AnalysisRunner): Runner for one sweep index (UI or worker).
        data_idx (int): Resonator index.

        Returns:
        x (float or None): ``AR.DS.<x_param_name>[data_idx]``, or None if it
            is unavailable or not finite.
        """
        try:
            attr = getattr(AR.DS, self._x_param_name)
            val = float(attr[data_idx])
            return None if not np.isfinite(val) else val
        except Exception:
            return None

    def _get_x_array(self, data_idx: int) -> np.ndarray:
        """Return x values for all sweep indices at *data_idx* (NaN if unavailable)."""
        if data_idx not in self._x_cache:
            self._x_cache[data_idx] = np.full(self._n_sweep, np.nan)
        x = self._x_cache[data_idx]
        for i in range(self._n_sweep):
            if np.isnan(x[i]):
                v = self._get_x_value(i, data_idx)
                x[i] = np.nan if v is None else v
        return x

    def _get_y_value(self, sweep_idx: int, data_idx: int) -> float:
        """Return the y value for (sweep_idx, data_idx), or NaN on failure."""
        return self._y_of(self._ARs[sweep_idx], data_idx)

    def _y_of(self, AR, data_idx: int) -> float:
        """
        Return the sweep-plot y value of one runner at a data index.

        Parameters:
        AR (AnalysisRunner): Runner for one sweep index (UI or worker).
        data_idx (int): Resonator index.

        Returns:
        y (float): ``y_func(AR, data_idx)``, or NaN if it is unavailable.
        """
        try:
            val = self._y_func(AR, data_idx)
            return np.nan if val is None else float(val)
        except Exception:
            return np.nan

    def _get_y_array(self, data_idx: int) -> np.ndarray:
        """
        Return y values for all sweep indices at *data_idx*.

        Values already in the cache are returned as-is; NaN slots are filled
        by calling ``y_func``.
        """
        if data_idx not in self._y_cache:
            self._y_cache[data_idx] = np.full(self._n_sweep, np.nan)
        y = self._y_cache[data_idx]
        for i in range(self._n_sweep):
            if np.isnan(y[i]):
                y[i] = self._get_y_value(i, data_idx)
        return y

    def _update_sweep_point(self, sweep_idx: int):
        """Recompute x and y for *sweep_idx* at the current data_idx."""
        di = self._data_idx
        for cache in (self._x_cache, self._y_cache):
            if di not in cache:
                cache[di] = np.full(self._n_sweep, np.nan)
        v = self._get_x_value(sweep_idx, di)
        self._x_cache[di][sweep_idx] = np.nan if v is None else v
        self._y_cache[di][sweep_idx] = self._get_y_value(sweep_idx, di)
        self._update_sweep_scatter()

    def _update_sweep_scatter(self):
        """Redraw all scatter points for the current data_idx."""
        x = self._get_x_array(self._data_idx)
        y = self._get_y_array(self._data_idx)
        for i, (xi, yi, si_item) in enumerate(zip(x, y, self._scatter_items)):
            if np.isnan(xi) or np.isnan(yi):
                si_item.setData([], [])
            else:
                si_item.setData([xi], [yi], data=[i])
        self._update_selected_marker()
        self._refresh_xy_fit()

    # ------------------------------------------------------------------
    # Optional y vs x fit
    # ------------------------------------------------------------------

    def _refresh_xy_fit(self):
        """
        Fit the current resonator's sweep plot data if it changed, and draw it.

        The saved fit is reused when its inputs match the plotted x and y
        exactly. The outputs (or the failure) are shown in the plot title.
        """
        if self._xy_fit is None:
            return
        di = self._data_idx
        x = self._get_x_array(di).copy()
        y = self._get_y_array(di).copy()
        outputs, error = self._fit_and_save_xy(di, x, y)
        xs, ys = self._xy_fit.curve(x, outputs)
        self._xy_fit_curve.setData([] if xs is None else xs, [] if ys is None else ys)
        if error is not None:
            detail = f'fit failed ({error})'
            self._apply_status_label.setText('xy fit failed')
        elif outputs is None:
            detail = 'no fit'
        else:
            detail = self._xy_fit.describe(outputs)
        self._plot_sweep.setTitle(f'Sweep  |  {self._xy_fit.name}: {detail}')

    def _fit_and_save_xy(self, data_idx: int, x, y):
        """
        Fit one resonator's sweep data and save it, unless the saved fit used
        the same inputs.

        Safe to call from the UI thread and the background worker.

        Parameters:
        data_idx (int): Resonator index.
        x (np.ndarray): x value of each sweep index (NaN if unavailable).
        y (np.ndarray): y value of each sweep index (NaN if unavailable).

        Returns:
        outputs (list of np.ndarray or None): Fit outputs, or None if the fit
            failed or there were too few points.
        error (str or None): Error message if the fit raised, else None.
        """
        store = self._xy_store
        if store.is_current(data_idx, x, y):
            return store.load(data_idx)[2], None
        try:
            outputs, error = self._xy_fit.run(x, y), None
        except Exception as exc:
            outputs, error = None, f'{type(exc).__name__}: {exc}'
        store.save(data_idx, x, y, outputs)
        return outputs, error

    def _fit_xy_in_background(self, worker_ars, data_idx: int):
        """
        Fit and save one resonator's sweep data from the worker runners.

        Parameters:
        worker_ars (list of AnalysisRunner): One worker runner per sweep.
        data_idx (int): Resonator index whose sweeps are all pre-fitted.
        """
        if self._xy_fit is None:
            return
        x = np.full(len(worker_ars), np.nan)
        for si, ar in enumerate(worker_ars):
            value = self._x_of(ar, data_idx)
            if value is not None:
                x[si] = value
        y = np.array([self._y_of(ar, data_idx) for ar in worker_ars])
        try:
            _, error = self._fit_and_save_xy(data_idx, x, y)
            if error is not None:
                print(f'[init-all] xy fit for data_idx={data_idx} failed: {error}')
        except Exception as exc:
            print(f'[init-all] saving xy fit for data_idx={data_idx} failed: '
                  f'{type(exc).__name__}: {exc}')

    def _update_selected_marker(self):
        """Draw a white ring around the currently selected sweep point."""
        if self._sweep_idx is None:
            self._selected_marker.setData([], [])
            return
        x = self._get_x_array(self._data_idx)
        y = self._get_y_array(self._data_idx)
        xi = x[self._sweep_idx]
        yi = y[self._sweep_idx]
        if np.isnan(xi) or np.isnan(yi):
            self._selected_marker.setData([], [])
        else:
            self._selected_marker.setData([xi], [yi])

    def _update_waterfall(self):
        """Reload and redraw the ``|S21|`` waterfall for the current data_idx."""
        for curve in self._waterfall_curves:
            self._plot_waterfall.removeItem(curve)
        self._waterfall_curves.clear()

        offset = 0.0
        for i, AR in enumerate(self._ARs):
            r, g, b = _sweep_color(i, self._n_sweep)
            pen = pg.mkPen(color=(r, g, b), width=1)
            try:
                ff = np.asarray(AR.DS.ff[self._data_idx])
                zf = np.asarray(AR.DS.zf[self._data_idx])
                dB = 20.0 * np.log10(np.abs(zf))
                dB += offset - dB.min()
                offset = dB.max()
                curve = self._plot_waterfall.plot(ff, dB, pen=pen)
                self._waterfall_curves.append(curve)
            except Exception:
                pass  # data not yet available for this sweep index / resonator


################################################################################
# Convenience entry point
################################################################################

def run_sweep_fitter(
    make_custom_steps,
    cal_yaml_path,
    analysis_yaml_path,
    root,
    n_sweep,
    x_param_name,
    x_name,
    y_param_name,
    start_sweep_idx=0,
    start_idx=None,
    data_idxs=None,
    title="Sweep Fitter",
    ui_scale=1.0,
    plot_scale=1.0,
    xy_fit=None,
):
    """
    Build one DataSet + AnalysisRunner per sweep index, then launch the
    SweepFitterWindow.

    Parameters:
    make_custom_steps (callable): ``make_custom_steps(sweep_idx) ->
        list[plStep]``.  Called once per sweep index to produce the custom
        calibration steps that load data for that sweep point.
    cal_yaml_path (str): Path to the calibration YAML file, or one of the
        shorthand aliases ``'iq'``, ``'ts'``, ``'ts_offres'``.
    analysis_yaml_path (str): Path to the analysis YAML file, or one of the
        shorthand aliases ``'iq'``, ``'ts'``, ``'ts_offres'``.
    root (zarr.Group): Parent zarr group.  Each sweep index is written to a
        subgroup ``sweep_{i:03d}`` (e.g. ``sweep_000``, ``sweep_001``, …).
    n_sweep (int): Number of sweep indices.  Determines how many subgroups
        are created and how many times ``make_custom_steps`` is called.
    x_param_name (str): Name of the pipeline parameter to use as x on the
        sweep scatter.  Loaded as ``AR.DS.<x_param_name>[data_idx]`` for each
        (sweep_idx, data_idx) pair so x can vary per resonator (e.g.
        ``'ares'``).
    x_name (str): Label for the x-axis of the sweep plot (e.g.
        ``'Power (dBm)'``).
    y_param_name (str): Name of the fit parameter to plot on the sweep scatter.
        Must be one of ``['fr', 'Qr', 'amp', 'phi', 'a', 'Qc', 'Qi']``.
        ``Qc = Qr / amp`` and ``Qi = 1 / (1/Qr - 1/Qc)``.
    start_sweep_idx (int): Initial sweep index. Default 0.
    start_idx (int or None): Position in ``data_idxs`` to start at (not a
        ``data_idx``). None (default) resumes at the first entry of
        ``data_idxs`` not viewed in an earlier session. A data index counts
        as viewed once the user navigates away from it or closes the window
        with it open.
    data_idxs (list of int or None): Ordered data indices to review and fit.
        Only these rows are visited and fitted in the background. None
        (default) uses every row.
    title (str): Window title. Default ``'Sweep Fitter'``.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.
    xy_fit (SweepXYFit or None): Optional fit of the sweep plot's
        ``y_param_name`` vs x data, one per resonator. Each resonator is
        fitted once all its sweeps are pre-fitted in the background, and
        again whenever its plotted data changes. Fits are saved to
        ``xy_fit.group``, or ``root/xy_fit`` if that is None. None (default)
        disables fitting.

    Returns:
    win (SweepFitterWindow): The created (and already shown) window.

    Raises:
    ValueError: If ``y_param_name`` is not a supported fit parameter.
    RuntimeError: If saved xy fits come from a different fit definition
        and the user cancels the overwrite popup.

    Notes:
    Session state (viewed and pre-fitted rows) is stored in
    ``root.attrs['sweep_fitter']``; delete that attribute to start over.
    Results are written to a fast unsharded buffer during the session and
    merged into the sharded arrays when the window closes. If Python
    crashes first, the data is still readable, and the next session merges
    it on close (or call ``DS.consolidate_storage()`` on each sweep's
    DataSet).
    """
    _DIRECT = {'fr': 0, 'Qr': 1, 'amp': 2, 'phi': 3, 'a': 4}
    _VALID = list(_DIRECT) + ['Qc', 'Qi']
    if y_param_name not in _VALID:
        raise ValueError(
            f'y_param_name must be one of {_VALID}; got {y_param_name!r}.'
        )
    y_name = y_param_name

    def y_func(AR, data_idx):
        try:
            popt = np.asarray(AR.DS.iq_popt[data_idx], dtype=float)
            if y_param_name in _DIRECT:
                val = popt[_DIRECT[y_param_name]]
            elif y_param_name == 'Qc':
                val = popt[1] / popt[2]          # Qr / amp
            else:  # Qi
                Qc = popt[1] / popt[2]
                val = 1.0 / (1.0 / popt[1] - 1.0 / Qc)
            return None if not np.isfinite(val) else float(val)
        except Exception:
            return None
    ARs = []
    for i in range(n_sweep):
        group = root.require_group(f'sweep_{i:03d}')
        DS = DataSet(
            zarr_path=group,
            cal_yaml_path=cal_yaml_path,
            custom_cal_steps=make_custom_steps(i),
            write_buffer=True,
        )
        AR = AnalysisRunner(DS, analysis_yaml_path=analysis_yaml_path)
        ARs.append(AR)

    app = get_qapp(title)

    win = SweepFitterWindow(
        ARs=ARs,
        x_param_name=x_param_name,
        x_name=x_name,
        y_func=y_func,
        y_name=y_name,
        start_sweep_idx=start_sweep_idx,
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
        state_group=root,
        xy_fit=xy_fit,
    )
    win._worker_root = root
    win._worker_make_custom_steps = make_custom_steps
    win._worker_cal_yaml_path = cal_yaml_path
    win._worker_analysis_yaml_path = analysis_yaml_path
    win.show()
    app.exec()
    return win
