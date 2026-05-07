"""
Interactive sweep fitter for fitting IQ loops across a parameter sweep.

For each resonator (data_idx), displays:
  - Left:  scatter plot of a user-defined y value vs the sweep parameter, and
           a ``|S21|`` waterfall for all sweep indices.
  - Right: GainFitPanel + FitIQPanel driven by the selected sweep index.

One :class:`~citkid.pipeline.analysis.AnalysisRunner` is created per sweep
index.  Each runner writes its results to a dedicated subgroup of a shared
zarr file (``sweep_000/``, ``sweep_001/``, …), so all outputs are persisted
in a single file.

Usage
-----
::

    import zarr
    from citkid.pipeline.interactive.sweep_fitter import run_sweep_fitter

    def make_custom_steps(sweep_idx):
        # Return list[plStep] that load data for this sweep index.
        ...

    def y_func(AR, data_idx):
        # Return a scalar y value from a fitted AR, or None if not yet available.
        try:
            return float(AR.DS.iq_popt[data_idx][4])  # e.g. nonlinearity a
        except Exception:
            return None

    run_sweep_fitter(
        make_custom_steps=make_custom_steps,
        cal_yaml_path='iq',
        analysis_yaml_path='iq',
        root=zarr.open_group('analysis.zarr', 'a'),
        n_sweep=7,
        x_param_name='ares',
        x_name='Power (dBm)',
        y_param_name='a',
    )

Keyboard shortcuts
------------------
A / ←       previous resonator
D / →       next resonator
W / ↑       previous sweep index
S / ↓       next sweep index
R           auto-scale all plots
1, 2, …     run panel N
Shift+N     run panel N and all following panels
"""

import sys
import numpy as np
import zarr
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..analysis import AnalysisRunner
from ..dataset import DataSet
from ...qt_compat import Qt as _Qt
from . import gain      # noqa: F401 — registers GainFitPanel
from . import fit_iq    # noqa: F401 — registers FitIQPanel
from .core import get_panel_class, _SectionHeader


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
    start_data_idx (int): Initial resonator index (data_idx). Default 0.
    title (str): Window title. Default 'Sweep Fitter'.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.
    parent (QWidget or None): Parent widget.
    """

    def __init__(
        self,
        ARs,
        x_param_name,
        x_name,
        y_func,
        y_name,
        start_sweep_idx=0,
        start_data_idx=0,
        title="Sweep Fitter",
        ui_scale=1.0,
        plot_scale=1.0,
        parent=None,
    ):
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
        self._data_idx = int(start_data_idx)

        try:
            self._nrows = int(self._ARs[0].DS.nrows)
        except Exception:
            self._nrows = 1

        # x and y value caches: {data_idx: np.ndarray of shape (n_sweep,)}
        self._x_cache: dict = {}
        self._y_cache: dict = {}

        # Prefetch state
        self._prefetch_thread: __import__('threading').Thread | None = None
        self._prefetching_idx: int | None = None
        self._prefetched_idx: int | None = None
        self._prefetch_status_changed.connect(self._on_prefetch_status)

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
        if ui_scale != 1.0:
            base_pt = round(10 * ui_scale)
            central.setStyleSheet(f"* {{ font-size: {base_pt}pt; }}")

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
        for idx in range(1, 10):
            _sc = QtGui.QShortcut(QtGui.QKeySequence(str(idx)), self)
            _sc.activated.connect(lambda _i=idx - 1: self._run_panel_by_index(_i))
            _sc_s = QtGui.QShortcut(QtGui.QKeySequence(f'Shift+{idx}'), self)
            _sc_s.activated.connect(lambda _i=idx - 1: self._run_through_panel(_i))

        self.resize(round(1400 * ui_scale), round(900 * ui_scale))
        QtCore.QTimer.singleShot(0, self._auto_initialize_all)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

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
        self._data_idx_spin.setMinimum(0)
        self._data_idx_spin.setMaximum(max(self._nrows - 1, 0))
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

        layout.addSpacing(16)

        hints = QtWidgets.QLabel(
            '[A/D] resonator   [W/S] sweep   [R] rescale   [B] mark bad'
            '   [N] run panel   [⇧N] run+'
        )
        hints.setStyleSheet('color: palette(mid); font-style: italic;')
        layout.addWidget(hints)

        layout.addStretch()

        run_all_btn = QtWidgets.QPushButton('Run All')
        run_all_btn.clicked.connect(self._run_all)
        layout.addWidget(run_all_btn)

        apply_all_btn = QtWidgets.QPushButton('Apply to All')
        apply_all_btn.setToolTip(
            'Apply current panel settings to every dataset in the active sweep and save'
        )
        apply_all_btn.clicked.connect(self._apply_to_all)
        layout.addWidget(apply_all_btn)

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

        # Bottom: |S21| waterfall
        self._plot_waterfall = gw.addPlot(row=1, col=0, title='|S21| Waterfall')
        self._plot_waterfall.setLabel('left', '|S21| + offset (dB)')
        self._plot_waterfall.setLabel('bottom', 'Frequency (Hz)')
        self._plot_waterfall.showGrid(x=True, y=True, alpha=0.3)
        self._waterfall_curves: list = []

        self._gw = gw
        return gw

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
    # Label helpers
    # ------------------------------------------------------------------

    def _update_res_label(self):
        self._res_label.setText(f'{self._data_idx + 1} / {self._nrows}')

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
        new_di = max(0, min(self._nrows - 1, self._data_idx + delta))
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
        if value != self._data_idx:
            self._set_data_idx(value)

    def _on_sweep_combo_changed(self, value: int):
        if value != self._sweep_idx:
            self._set_sweep_idx(value)

    def _set_data_idx(self, new_di: int):
        """Change the active resonator, load/run data, and refresh the scatter."""
        self._save_dirty_panels()  # persist results for the outgoing resonator
        self._data_idx = new_di
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

        # Silently run (or load) the full pipeline for every sweep AR at the
        # new data_idx, so the scatter and waterfall are fully populated and
        # the active panels can show results immediately.
        if self._prefetched_idx == new_di:
            # Background thread already has results in memory — skip re-run.
            pass
        else:
            # If the prefetch thread is still running for this index, wait.
            if (self._prefetch_thread is not None
                    and self._prefetch_thread.is_alive()
                    and self._prefetching_idx == new_di):
                while self._prefetch_thread.is_alive():
                    self._prefetch_thread.join(timeout=0.05)
                    QtWidgets.QApplication.processEvents()
            if self._prefetched_idx != new_di:
                self._batch_init_all_sweeps()
        self._update_sweep_combo_items(new_di)
        self._update_sweep_scatter()
        self._update_waterfall()
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    def _set_sweep_idx(self, new_si: int):
        """Change the active sweep index, swap ARs in panels, and re-run."""
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

        For each sweep AR: if ``iq_popt`` already exists in memory or zarr for
        the current data_idx it is used as-is; otherwise the full pipeline
        (make_fr_spans → fit_gain → fit_iq) is executed without saving to
        disk.  Pre-populates the y-cache so the scatter is fully drawn on
        startup.
        """
        di = self._data_idx
        for AR in self._ARs:
            needs_run = True
            try:
                val = AR.DS.iq_popt[di]
                if val is not None:
                    needs_run = False
            except Exception:
                pass
            if needs_run:
                try:
                    AR.execute_path(
                        data_idx=di, save_override=False, verbose=False
                    )
                except Exception as exc:
                    print(f"Warning: batch init failed for sweep AR: {exc}")

    def _auto_initialize_all(self):
        self._batch_init_all_sweeps()
        self._update_sweep_combo_items(self._data_idx)
        self._update_sweep_scatter()
        self._update_waterfall()
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_next(self):
        """
        Pre-run the full pipeline for every sweep AR at ``data_idx + 1`` on a
        background daemon thread so that navigating forward feels instant.
        """
        import threading as _threading
        next_di = self._data_idx + 1
        if next_di >= self._nrows:
            return
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
                for AR in ARs:
                    AR.execute_path(
                        data_idx=next_di, save_override=False, verbose=False
                    )
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
        """Auto-save dirty panels before closing."""
        self._save_dirty_panels()
        super().closeEvent(event)

    def _run_all_panels(self):
        for panel in self.panels:
            ok = panel.run_steps()
            if not ok:
                break
        # Refresh the scatter point for the current sweep index after running
        self._update_sweep_point(self._sweep_idx)

    def _run_all(self):
        self._run_through_panel(0)

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
        ok = self.panels[index].run_steps()
        if ok:
            self.panels[index].trigger_downstream()

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
        """Cascade re-runs after source_panel, then refresh the scatter point.

        If a downstream panel fails to run (e.g. because source_panel was
        marked bad and its outputs are NaN), write NaN outputs for that panel
        and all subsequent ones so the scatter point is removed cleanly.
        """
        try:
            src_idx = self.panels.index(source_panel)
        except ValueError:
            return
        for i, panel in enumerate(self.panels[src_idx + 1:], start=src_idx + 1):
            panel.prepare_run()
            ok = panel.run_steps()
            if not ok:
                # Upstream data was bad — NaN-ify this and all remaining panels.
                for p in self.panels[i:]:
                    p._write_nan_outputs()
                break
        self._update_sweep_point(self._sweep_idx)

    def _mark_all_bad(self):
        """Mark every panel's outputs as NaN, clear their plots, and refresh scatter."""
        for panel in self.panels:
            panel._write_nan_outputs()
            panel.clear_plots()
        self._update_sweep_point(self._sweep_idx)

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
        try:
            attr = getattr(self._ARs[sweep_idx].DS, self._x_param_name)
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
        try:
            val = self._y_func(self._ARs[sweep_idx], data_idx)
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
    start_data_idx=0,
    title="Sweep Fitter",
    ui_scale=1.0,
    plot_scale=1.0,
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
    start_data_idx (int): Initial resonator index (data_idx). Default 0.
    title (str): Window title. Default ``'Sweep Fitter'``.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.

    Returns:
    win (SweepFitterWindow): The created (and already shown) window.
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
        )
        AR = AnalysisRunner(DS, analysis_yaml_path=analysis_yaml_path)
        ARs.append(AR)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp(title)

    win = SweepFitterWindow(
        ARs=ARs,
        x_param_name=x_param_name,
        x_name=x_name,
        y_func=y_func,
        y_name=y_name,
        start_sweep_idx=start_sweep_idx,
        start_data_idx=start_data_idx,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
    )
    win.show()
    app.exec()
    return win
