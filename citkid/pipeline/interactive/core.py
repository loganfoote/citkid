"""Modular, stackable interactive analysis framework for pipeline steps.

Core concepts
-------------
StepPanel
    One "card" in the stacked UI.  Owns one or more consecutive pipeline
    steps and the widgets needed to control their parameters and display
    their outputs.  Sub-class and override :meth:`~StepPanel.setup_ui`,
    :meth:`~StepPanel.get_params_for_step`, and
    :meth:`~StepPanel.update_plots` to create custom panels.

register_panel
    Class decorator that binds a :class:`StepPanel` subclass to a tuple of
    step names so the framework can look it up automatically.

InteractiveAnalysisWindow
    :class:`~PyQt5.QtWidgets.QMainWindow` that stacks panels top-to-bottom
    inside a scroll area.  When panel *i* emits
    :attr:`~StepPanel.downstream_rerun`, panels *i+1*, *i+2*, … are
    re-run in order (stopping on the first failure).

run_interactive
    Convenience entry point: builds the window, shows it, and starts the
    Qt event loop.

Typical usage
-------------
Run with default (unregistered) panels — every step gets a simple
"Run" button::

    from citkid.pipeline.interactive import run_interactive
    run_interactive(AR, panels=[('make_fr_spans', 'fit_gain'), ('fit_iq',)],
                    data_idx=0)

Register a custom panel for gain fitting::

    from citkid.pipeline.interactive import register_panel, StepPanel

    @register_panel('make_fr_spans', 'fit_gain')
    class GainFitPanel(StepPanel):
        def setup_ui(self):
            ...
        def get_params_for_step(self, step):
            if step.name == 'fit_gain':
                return {'span_mult': self._span_spin.value()}
            return {}
        def update_plots(self):
            ...

Cascade behaviour
-----------------
* User changes ``span_mult`` in GainFitPanel → ``run_steps()`` is called →
  on success the panel calls ``trigger_downstream()`` → the window reruns
  every panel below GainFitPanel.
* Changing the ``data_idx`` spinner reruns every per-row panel from the top.
"""

import concurrent.futures
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import threading

from ..analysis import AnalysisRunner
from ..dependencies import get_most_recent_run
from ...signal.iq import density_subsample as _density_subsample


################################################################################
# Panel registry
################################################################################

_PANEL_REGISTRY: dict[tuple, type] = {}


def register_panel(*step_names):
    """
    Class decorator that registers a StepPanel subclass for a group of step
    names.

    Parameters:
    *step_names (str): One or more pipeline step names this panel handles,
        in execution order.

    Example::

        @register_panel('make_fr_spans', 'fit_gain')
        class GainFitPanel(StepPanel):
            ...
    """
    def decorator(cls):
        _PANEL_REGISTRY[tuple(step_names)] = cls
        return cls
    return decorator


def get_panel_class(step_names):
    """
    Return the StepPanel subclass registered for step_names.

    Lookup order:
    1. Exact tuple match in the registry.
    2. Single-step fallback: (step_names[0],) in the registry.
    3. DefaultStepPanel (always available).

    Parameters:
    step_names (tuple of str): Step name tuple to look up.

    Returns:
    type: A StepPanel subclass.
    """
    if step_names in _PANEL_REGISTRY:
        return _PANEL_REGISTRY[step_names]
    if len(step_names) >= 1 and (step_names[0],) in _PANEL_REGISTRY:
        return _PANEL_REGISTRY[(step_names[0],)]
    return DefaultStepPanel


################################################################################
# Base panel
################################################################################

class StepPanel(QtWidgets.QWidget):
    """
    Abstract base class for an interactive analysis step panel.

    Each panel owns one or more pipeline steps. Override setup_ui,
    get_params_for_step, and update_plots in subclasses.

    Signals:
    downstream_rerun (pyqtSignal(object)): Emitted (with self as payload) when
        trigger_downstream is called after a successful run.
        InteractiveAnalysisWindow connects to this to cascade re-runs through
        all panels below this one.

    Parameters:
    AR (AnalysisRunner): The analysis runner whose execute_step will be called.
    step_names (tuple of str): Names of the steps this panel executes, in order.
    data_idx (int or None): Initial data index for per-row steps. None until set.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.
    parent (QWidget or None): Parent widget.
    """

    #: Emitted when downstream panels should re-run.  Payload is *self*.
    downstream_rerun = QtCore.pyqtSignal(object)
    #: Emitted when the user clicks "Run +" to run this panel and all following.
    run_from_here = QtCore.pyqtSignal()

    def __init__(
        self,
        AR: AnalysisRunner,
        step_names: tuple,
        data_idx=None,
        ui_scale: float = 1.0,
        plot_scale: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.AR = AR
        self.step_names = tuple(step_names)
        self.data_idx = data_idx
        self.ui_scale = ui_scale
        self.plot_scale = plot_scale
        self.panel_index: int | None = None  # set by InteractiveAnalysisWindow
        self._has_run = False
        self._last_error: Exception | None = None
        self._dirty: bool = False  # True after run_steps succeeds; cleared after save
        self._autorange_next: bool = True  # auto-scale plots on first update_plots per index

        # Resolve plStep objects from the runner
        available = {s.name: s for s in AR.analysis_steps}
        self.steps = []
        for name in step_names:
            if name not in available:
                raise ValueError(
                    f"Step '{name}' not found in AR.analysis_steps. "
                    f"Available: {sorted(available)}"
                )
            self.steps.append(available[name])

        self.setup_ui()

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------

    def setup_ui(self):
        """
        Build the panel's widgets. Called once during __init__.

        The base implementation adds a single status label. Override to
        replace this with domain-specific controls and plots.
        """
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self._status_label = QtWidgets.QLabel("Not run")
        layout.addWidget(self._status_label)

    def get_params_for_step(self, step):
        """
        Return user-controlled parameters for step as {name: value}.

        Called by run_steps just before each step executes. Override to
        return widget values (e.g. spinbox, combobox).

        Parameters:
        step (plStep): The step about to be executed.

        Returns:
        params (dict): Mapping of parameter name to value. Only
            user-controlled parameters need to be included; pipeline-produced
            parameters are resolved automatically.
        """
        return {}

    def update_plots(self):
        """
        Refresh all plots and displays after the steps have run successfully.

        Override in subclasses to read results from self.AR.DS and update
        plot widgets.
        """
        pass

    def _scale_plot_fonts(self, *plot_items):
        """
        Apply self.ui_scale to tick labels, axis labels, and titles on each
        of the supplied pyqtgraph PlotItem objects.

        Call at the end of setup_ui after all plots have been created.
        No-op when ui_scale == 1.0.
        """
        if self.ui_scale == 1.0:
            return
        tick_pt  = max(6, round(9  * self.ui_scale))
        label_pt = max(7, round(11 * self.ui_scale))
        tick_font  = QtGui.QFont()
        tick_font.setPointSize(tick_pt)
        label_font = QtGui.QFont()
        label_font.setPointSize(label_pt)
        for plot in plot_items:
            for ax_name in ('bottom', 'left', 'top', 'right'):
                ax = plot.getAxis(ax_name)
                ax.setStyle(tickFont=tick_font)
                ax.label.setFont(label_font)
            plot.titleLabel.item.setFont(label_font)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run_steps(self, save=False):
        """
        Execute each step owned by this panel, then call update_plots.

        Global and global-res steps always receive data_idx=None; per-row
        and vectorized steps receive self.data_idx.

        Parameters:
        save (bool): Passed directly to AnalysisRunner.execute_step.
            Default False.

        Returns:
        ok (bool): True if all steps succeeded, False if any step raised.
        """
        for step in self.steps:
            step_di = (
                None
                if step.func_type in ("global", "global-res")
                else self.data_idx
            )
            params = self.get_params_for_step(step)
            try:
                self.AR.execute_step(
                    step, data_idx=step_di, user_params=params, save=save
                )
            except Exception as exc:
                self._last_error = exc
                self._on_step_error(step, exc)
                return False

        self._has_run = True
        self._last_error = None
        self._dirty = True
        try:
            self.update_plots()
        except Exception as exc:
            print(f"Warning: update_plots() raised in {self.step_names}: {exc}")
        return True

    def prepare_run(self):
        """
        Called by _run_through_panel immediately before run_steps on each
        panel.

        Base implementation is a no-op. Sub-classes may override to perform
        pre-run setup that is normally handled inside their _on_run_clicked
        handler (e.g. rebuilding a mask from the current region state).
        """

    def autoscale_plots(self):
        """
        Auto-range every plot widget owned by this panel.

        Base implementation is a no-op. Sub-classes should override this
        and call plot.autoRange() on each PlotItem. Called by the window
        when the user presses the rescale shortcut.
        """

    def trigger_downstream(self):
        """
        Emit downstream_rerun to ask the window to re-run all panels that
        come after this one.

        Call this at the end of a user-triggered action (e.g. button click
        or parameter change) after run_steps succeeds.
        """
        self.downstream_rerun.emit(self)

    def save_outputs(self):
        """
        Persist the most recent in-memory outputs of every step owned by this
        panel to the zarr file. Does not re-run the steps. Raises if any step
        has no cached results yet.
        """
        for step in self.steps:
            step_di = (
                None
                if step.func_type in ("global", "global-res")
                else self.data_idx
            )
            self.AR.save_step_outputs(step, data_idx=step_di)
        self._dirty = False

    def _write_nan_outputs(self):
        """
        Write NaN-filled placeholder values for every per-row output of this
        panel's steps, then mark the panel dirty.

        Sub-classes must override _nan_outputs to return the dict of
        {return_name: nan_value} appropriate for their step(s).

        Returns:
        ok (bool): True on success, False if no override is provided or an
            error occurs.
        """
        nan_vals = self._nan_outputs()
        if not nan_vals:
            return False
        try:
            import numpy as np
            DS = self.AR.DS
            di = self.data_idx
            data_idx_arr = np.atleast_1d(np.asarray(di, dtype=np.int32))
            if di not in DS.deps_maps:
                DS.deps_maps[di] = {}
            for name, val in nan_vals.items():
                # next run after whatever is already stored (0 if nothing yet)
                run_idx = get_most_recent_run(name, DS.deps_maps[di]) + 1
                DS._store_param(
                    name, [val], run_idx, deps={},
                    is_global=False, data_idx=data_idx_arr,
                )
            self._has_run = True
            self._dirty = True
            try:
                self.update_plots()
            except Exception:
                pass
            return True
        except Exception as exc:
            print(f"_write_nan_outputs failed in {self.step_names}: {exc}")
            return False

    def _nan_outputs(self):
        """
        Return a dict {return_name: nan_array} for each per-row output.

        Base implementation returns {} (no-op). Sub-classes should override
        this to provide NaN-filled arrays of the correct shape.

        Returns:
        nan_vals (dict): Mapping of output name to NaN array.
        """
        return {}

    def prefetch_plot_data(self, di):
        """
        Pre-compute numpy arrays needed by update_plots for di.

        Called from the background prefetch thread before the user navigates
        to di. Implementations must be thread-safe: only read from AR.DS,
        only write to self._plot_cache. Never touch any Qt object here.

        The base implementation is a no-op. Sub-classes with expensive numpy
        work inside update_plots (e.g. large array loads, np.polyval,
        subsampling) should override this method and store results in
        self._plot_cache[di]. update_plots can then consume the cache and
        skip the numpy work entirely.

        Parameters:
        di (int): Data index to prefetch.
        """

    def on_data_idx_changing(self):
        """
        Called by InteractiveAnalysisWindow just before run_steps when the
        active data index changes.

        Override in subclasses to reset panel-local state that is tied to a
        specific data index (e.g. an interactive mask region).
        """
        pass

    # ------------------------------------------------------------------
    # Parameter initialisation helpers
    # ------------------------------------------------------------------

    def _get_initial_user_param(self, step_name, param_name, data_idx,
                                fallback=None):
        """
        Return the best available initial value for a user-controlled
        parameter.

        Priority: (1) value stored in AR.DS from a previous run, (2) default
        from the analysis YAML, (3) fallback.

        Parameters:
        step_name (str): Name of the step that owns the parameter.
        param_name (str): Name of the parameter / DS attribute.
        data_idx (int or None): Data index currently active on this panel.
        fallback: Value to return when neither DS nor YAML provide a value.
            Default None.
        """
        # 1. Try DS attribute
        try:
            attr = getattr(self.AR.DS, param_name)
            val = attr[data_idx] if data_idx is not None else attr
            if val is not None:
                return val
        except Exception:
            pass
        # 2. Try YAML
        try:
            step = next(s for s in self.steps if s.name == step_name)
            yaml_params = self.AR._get_yaml_params(step)
            if param_name in yaml_params and yaml_params[param_name] is not None:
                return yaml_params[param_name]
        except Exception:
            pass
        return fallback

    def set_data_idx(self, data_idx):
        """Set a new data index and immediately re-run the panel."""
        self.data_idx = data_idx
        self.run_steps()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _outputs_exist(self):
        """
        Return True if every output produced by this panel's steps is already
        present (cached in memory or stored on disk) for the current data_idx.

        Only the last step's outputs are checked — earlier steps are
        prerequisites whose results are implicitly required for the final
        step to exist.

        Returns:
        exists (bool): True if all outputs are available.
        """
        DS = self.AR.DS
        check_steps = self.steps[-1:]  # last step's outputs are sufficient
        for step in check_steps:
            step_di = (
                None
                if step.func_type in ("global", "global-res")
                else self.data_idx
            )
            for name in step.return_names:
                try:
                    attr = getattr(DS, name)
                    val = attr[step_di]
                    if val is None:
                        return False
                except Exception:
                    return False
        return True

    def _auto_initialize(self):
        """
        Called once (via a zero-delay timer) after construction.

        If all output data already exist in the DataSet the plots are drawn
        immediately without re-running the step. Otherwise the steps are
        executed with default parameters (save=False) so that initial results
        are available for the first render.
        """
        if self._outputs_exist():
            self.update_plots()
        else:
            self.run_steps(save=False)

    def _on_step_error(self, step, exc):
        """
        Handle a step-level execution error.

        Writes a short message to _status_label if it exists and prints to
        stdout. Override to add custom error UI (e.g. a dialog).
        """
        msg = f"Error in '{step.name}': {exc}"
        if hasattr(self, "_status_label"):
            self._status_label.setText(msg)
        print(msg)


################################################################################
# Default panel (fallback for unregistered steps)
################################################################################

class DefaultStepPanel(StepPanel):
    """
    Generic fallback panel for steps that have no registered custom panel.

    Displays the step names, a *Run* button, and a status indicator.
    On success it calls :meth:`trigger_downstream` to cascade re-runs.
    """

    def setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        names_str = " + ".join(self.step_names)
        layout.addWidget(QtWidgets.QLabel(f"<i>{names_str}</i>"))
        layout.addStretch()

        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._run_btn)

        self._run_through_btn = QtWidgets.QPushButton("Run+")
        self._run_through_btn.setToolTip("Run this panel and all following panels")
        self._run_through_btn.clicked.connect(lambda: self.run_from_here.emit())
        layout.addWidget(self._run_through_btn)

        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_btn)

        self._status_label = QtWidgets.QLabel("—")
        self._status_label.setMinimumWidth(130)
        layout.addWidget(self._status_label)

    def _on_run_clicked(self):
        self._status_label.setText("Running…")
        QtWidgets.QApplication.processEvents()
        ok = self.run_steps()
        if ok:
            self._status_label.setText("Done ✓")
            self.trigger_downstream()
        else:
            self._status_label.setText("Error ✗")

    def _on_save_clicked(self):
        try:
            self.save_outputs()
            self._status_label.setText("Saved ✓")
        except Exception as exc:
            self._status_label.setText("Save error ✗")
            print(f"Save error: {exc}")

    def _on_step_error(self, step, exc: Exception):
        msg = f"Error in '{step.name}': {exc}"
        if hasattr(self, "_status_label"):
            self._status_label.setText(msg)
        print(msg)


################################################################################
# Collapsible section header
################################################################################

class _SectionHeader(QtWidgets.QWidget):
    """
    Wraps a :class:`StepPanel` with a collapsible toggle button.

    The button text shows the panel number and step names joined by ``'→'``.
    Clicking the button hides or shows the wrapped panel.
    """

    def __init__(self, panel: StepPanel, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(0)

        font_pt = round(11 * panel.ui_scale)
        self._btn = QtWidgets.QPushButton(self._label_text("▼", panel))
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setFlat(True)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            "  font-weight: bold;"
            f"  font-size: {font_pt}pt;"
            "  padding: 4px 8px;"
            "  border-bottom: 1px solid palette(mid);"
            "}"
        )
        self._btn.clicked.connect(self._on_toggle)
        layout.addWidget(self._btn)

        self._panel = panel
        layout.addWidget(panel)

    @staticmethod
    def _label_text(arrow: str, panel: StepPanel) -> str:
        idx = panel.panel_index
        num_str = f"{idx + 1}: " if idx is not None else ""
        names_str = "  →  ".join(panel.step_names)
        if idx is not None:
            n = idx + 1
            hints = f"[{n}] run   [⇧{n}] run+following"
            return f"{arrow}  {num_str}{names_str}    —    {hints}"
        return f"{arrow}  {num_str}{names_str}"

    def _on_toggle(self, checked: bool):
        self._panel.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self._btn.setText(self._label_text(arrow, self._panel))


################################################################################
# Main window
################################################################################

class InteractiveAnalysisWindow(QtWidgets.QMainWindow):
    """
    Main window that vertically stacks step panels and orchestrates cascade
    re-runs.

    When panel i emits StepPanel.downstream_rerun, panels i+1, i+2, ... are
    re-run in sequence (stopping on the first failure).

    Parameters:
    AR (AnalysisRunner): Runner with loaded steps.
    panels (list of tuple of str): One tuple per panel group. Each tuple
        contains the step names that belong to that panel.
    start_idx (int): Index into data_idxs to start at. Default 0.
    data_idxs (list of int or None): Ordered sequence of data indices to
        navigate. None uses all rows.
    title (str): Window title.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.
    parent (QWidget or None): Parent widget.
    """

    #: Emitted from the prefetch worker thread to update the status label.
    _prefetch_status_changed = QtCore.pyqtSignal(str)
    #: Emitted from the background save thread to update the save status label.
    _save_status_changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        AR: AnalysisRunner,
        panels: list,
        start_idx: int = 0,
        data_idxs=None,
        title: str = "Interactive Analysis",
        ui_scale: float = 1.0,
        plot_scale: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.AR = AR
        self._ui_scale = ui_scale
        self._plot_scale = plot_scale
        self.setWindowTitle(title)

        # Prefetch state — background thread pre-runs the next data index
        self._prefetch_thread: threading.Thread | None = None
        self._prefetching_idx: int | None = None  # index currently being prefetched
        self._prefetched_idx: int | None = None   # index whose prefetch completed
        self._prefetch_status_changed.connect(self._on_prefetch_status)

        # Background save state — single-worker thread pool serialises zarr writes
        self._save_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="citkid-save"
        )
        self._save_count_lock = threading.Lock()
        self._saves_in_flight: int = 0
        self._save_status_changed.connect(self._on_save_status)

        # Navigation index list
        if data_idxs is None:
            try:
                n = int(AR.DS.nrows)
            except Exception:
                n = 1
            data_idxs = list(range(n))
        self._data_idxs: list = list(data_idxs)
        # Resolve starting position: start_idx is an index *into* data_idxs
        self._nav_pos: int = max(0, min(int(start_idx), len(self._data_idxs) - 1)) if self._data_idxs else 0
        start_di = self._data_idxs[self._nav_pos] if self._data_idxs else 0

        # Central widget + outer layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Toolbar: data_idx selector + navigation + "Run All" button
        outer.addWidget(self._build_toolbar(start_di))

        # Scrollable panel area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self._container = QtWidgets.QWidget()
        self._panel_layout = QtWidgets.QVBoxLayout(self._container)
        self._panel_layout.setAlignment(QtCore.Qt.AlignTop)
        self._panel_layout.setSpacing(6)
        scroll.setWidget(self._container)
        outer.addWidget(scroll)

        # Apply font scaling via stylesheet
        if ui_scale != 1.0:
            base_pt = round(10 * ui_scale)
            central.setStyleSheet(f"* {{ font-size: {base_pt}pt; }}")

        # Build panels
        self.panels: list[StepPanel] = []
        for i, step_names_tuple in enumerate(panels):
            cls = get_panel_class(step_names_tuple)
            panel = cls(AR, step_names_tuple, data_idx=start_di,
                        ui_scale=ui_scale, plot_scale=plot_scale, parent=self)
            panel.panel_index = i
            panel.downstream_rerun.connect(self._on_panel_rerun)
            self._add_panel(panel)

        # Keyboard shortcuts: navigate resonator index
        # Forward:  D key  or  Right arrow
        # Backward: A key  or  Left arrow
        for key in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda: self._advance(+1))
        for key in (QtCore.Qt.Key_A, QtCore.Qt.Key_Left):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda: self._advance(-1))

        # R: auto-scale all plot axes
        _sc_r = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_R), self)
        _sc_r.activated.connect(self._autoscale_all)

        # Number keys 1-9: run the corresponding panel (1-indexed)
        _digit_keys = [
            QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3,
            QtCore.Qt.Key_4, QtCore.Qt.Key_5, QtCore.Qt.Key_6,
            QtCore.Qt.Key_7, QtCore.Qt.Key_8, QtCore.Qt.Key_9,
        ]
        for idx, key in enumerate(_digit_keys):
            _sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            _sc.activated.connect(
                lambda _i=idx: self._run_panel_by_index(_i)
            )
            _sc_shift = QtWidgets.QShortcut(
                QtGui.QKeySequence(QtCore.Qt.ShiftModifier | key), self
            )
            _sc_shift.activated.connect(
                lambda _i=idx: self._run_through_panel(_i)
            )

        self.resize(round(1200 * ui_scale), round(900 * ui_scale))
        # Initialise panels in order once the event loop is running so that
        # the window is fully laid out and panels are initialised sequentially
        # (guaranteeing upstream data is ready before downstream panels run).
        QtCore.QTimer.singleShot(0, self._auto_initialize_all)
    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_toolbar(self, data_idx) -> QtWidgets.QWidget:
        """Build the top toolbar with navigation, a ``data_idx`` spinbox, and Run All."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)

        # Previous button
        self._prev_btn = QtWidgets.QPushButton("\u25c0")
        self._prev_btn.setFixedWidth(30)
        self._prev_btn.setToolTip(
            "Previous index \u2014 saves all panels then steps back  "
            "(A or \u2190)"
        )
        self._prev_btn.clicked.connect(lambda: self._advance(-1))
        layout.addWidget(self._prev_btn)

        # Position indicator  "3 / 128"
        self._nav_label = QtWidgets.QLabel()
        self._nav_label.setMinimumWidth(60)
        self._nav_label.setAlignment(QtCore.Qt.AlignCenter)
        self._update_nav_label()
        layout.addWidget(self._nav_label)

        # Next button
        self._next_btn = QtWidgets.QPushButton("\u25b6")
        self._next_btn.setFixedWidth(30)
        self._next_btn.setToolTip(
            "Next index \u2014 saves all panels then steps forward  "
            "(D or \u2192)"
        )
        self._next_btn.clicked.connect(lambda: self._advance(+1))
        layout.addWidget(self._next_btn)

        layout.addSpacing(12)
        layout.addWidget(QtWidgets.QLabel("data_idx:"))

        self._idx_spin = QtWidgets.QSpinBox()
        self._idx_spin.setMinimum(0)
        try:
            max_idx = max(int(self.AR.DS.nrows) - 1, 0)
        except Exception:
            max_idx = 9999
        self._idx_spin.setMaximum(max_idx)
        if data_idx is not None:
            self._idx_spin.setValue(int(data_idx))
        self._idx_spin.valueChanged.connect(self._on_data_idx_changed)
        layout.addWidget(self._idx_spin)

        layout.addSpacing(8)

        _hints = QtWidgets.QLabel(
            "[\u2190/A  \u2192/D] navigate    [R] rescale    [N] run panel    [\u21e7N] run+following"
        )
        _hints.setStyleSheet("color: palette(mid); font-style: italic;")
        layout.addWidget(_hints)

        layout.addStretch()

        self._prefetch_label = QtWidgets.QLabel("")
        self._prefetch_label.setMinimumWidth(110)
        self._prefetch_label.setToolTip(
            "Background prefetch status for the next data index"
        )
        layout.addWidget(self._prefetch_label)

        self._save_label = QtWidgets.QLabel("")
        self._save_label.setMinimumWidth(90)
        self._save_label.setToolTip("Background save status")
        layout.addWidget(self._save_label)

        run_all_btn = QtWidgets.QPushButton("Run All")
        run_all_btn.clicked.connect(self.run_all)
        layout.addWidget(run_all_btn)

        return w

    def _update_nav_label(self):
        n = len(self._data_idxs)
        self._nav_label.setText(f"{self._nav_pos + 1} / {n}")

    def _submit_save(self, steps, AR, di, panel_names):
        """Submit a save task for *di* to the single-worker background thread."""
        with self._save_count_lock:
            self._saves_in_flight += 1
            count = self._saves_in_flight
        self._save_status_changed.emit(f"saving… ({count})")

        def _do_save():
            try:
                for step in steps:
                    step_di = (
                        None
                        if step.func_type in ("global", "global-res")
                        else di
                    )
                    AR.save_step_outputs(step, data_idx=step_di)
            except Exception as exc:
                print(
                    f"Warning: background save failed for {panel_names}: {exc}"
                )
            finally:
                with self._save_count_lock:
                    self._saves_in_flight -= 1
                    remaining = self._saves_in_flight
                self._save_status_changed.emit(
                    "" if remaining == 0 else f"saving… ({remaining})"
                )

        self._save_executor.submit(_do_save)

    def _advance(self, delta: int):
        """Submit dirty-panel saves to the background thread, then navigate."""
        # Clear dirty flag immediately and snapshot data_idx before it changes.
        for panel in self.panels:
            if not panel._dirty:
                continue
            panel._dirty = False
            self._submit_save(
                list(panel.steps), panel.AR, panel.data_idx, panel.step_names
            )

        new_pos = max(0, min(len(self._data_idxs) - 1, self._nav_pos + delta))
        if new_pos == self._nav_pos:
            return  # already at boundary
        self._nav_pos = new_pos
        new_di = self._data_idxs[new_pos]

        # Update spinbox without double-firing _on_data_idx_changed
        self._idx_spin.blockSignals(True)
        self._idx_spin.setValue(new_di)
        self._idx_spin.blockSignals(False)

        self._update_nav_label()
        self._on_data_idx_changed(new_di)

    def _add_panel(self, panel: StepPanel):
        """Add *panel* to the scroll area, separated by a horizontal line."""
        if self.panels:
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.HLine)
            sep.setFrameShadow(QtWidgets.QFrame.Sunken)
            self._panel_layout.addWidget(sep)
        header = _SectionHeader(panel)
        self._panel_layout.addWidget(header)
        self.panels.append(panel)
        panel.run_from_here.connect(
            lambda p=panel: self._run_through_panel(p.panel_index)
        )

    # ------------------------------------------------------------------
    # Panel run helpers
    # ------------------------------------------------------------------

    def _run_panel_by_index(self, index: int):
        """Run the panel at position *index* (0-based). Triggers downstream."""
        if index >= len(self.panels):
            return
        panel = self.panels[index]
        ok = panel.run_steps()
        if ok:
            panel.trigger_downstream()

    def _autoscale_all(self):
        """Auto-range every plot in every panel."""
        for panel in self.panels:
            panel.autoscale_plots()

    def _run_through_panel(self, index: int):
        """Run panel *index* and all panels that follow it, stopping on first failure."""
        if index >= len(self.panels):
            return
        for panel in self.panels[index:]:
            panel.prepare_run()
            if hasattr(panel, '_status_label'):
                panel._status_label.setText("Running\u2026")
                QtWidgets.QApplication.processEvents()
            try:
                ok = panel.run_steps()
            except Exception as exc:
                print(f"_run_through_panel: unexpected error in "
                      f"{panel.step_names}: {exc}")
                ok = False
            if ok:
                if hasattr(panel, '_status_label'):
                    panel._status_label.setText("Done \u2713")
            else:
                break

    # ------------------------------------------------------------------
    # Cascade logic
    # ------------------------------------------------------------------

    def _on_panel_rerun(self, source_panel: StepPanel):
        """Re-run every panel that comes after *source_panel*."""
        try:
            src_idx = self.panels.index(source_panel)
        except ValueError:
            return
        for panel in self.panels[src_idx + 1:]:
            ok = panel.run_steps()
            if not ok:
                break

    def _auto_initialize_all(self):
        """Initialise every panel in order so upstream data is always ready."""
        for panel in self.panels:
            panel._auto_initialize()
        # Start prefetching the next index while the user examines this one.
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    def _on_data_idx_changed(self, value: int):
        """
        Propagate a new ``data_idx`` to all panels, then re-run every panel
        that contains at least one per-row or vectorized step (stopping on
        the first failure).

        If the next index was fully pre-computed by the background prefetch
        thread, panels skip ``run_steps`` and go straight to ``update_plots``.
        """
        # Sync nav position when spinbox is changed manually
        if value in self._data_idxs:
            self._nav_pos = self._data_idxs.index(value)
            self._update_nav_label()
        for panel in self.panels:
            panel.data_idx = value

        # If the prefetch thread is currently running for this index, wait
        # for it to finish while keeping the UI responsive.
        if (self._prefetch_thread is not None
                and self._prefetch_thread.is_alive()
                and self._prefetching_idx == value):
            while self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=0.05)
                QtWidgets.QApplication.processEvents()

        is_prefetched = (self._prefetched_idx == value)

        for panel in self.panels:
            has_per_row = any(
                s.func_type in ("per-row", "vectorized") for s in panel.steps
            )
            if has_per_row:
                panel.on_data_idx_changing()
                if is_prefetched and panel._outputs_exist():
                    # Results are already in the DS cache — just render them.
                    # Mark dirty so _advance saves these unsaved prefetch results.
                    panel._has_run = True
                    panel._dirty = True
                    panel.update_plots()
                else:
                    ok = panel.run_steps()
                    if not ok:
                        break

        # Kick off prefetch of the next index in the navigation sequence.
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_all(self):
        """Run all panels from top to bottom, stopping on the first failure."""
        for panel in self.panels:
            ok = panel.run_steps()
            if not ok:
                break

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_next(self):
        """
        Start a background daemon thread that pre-runs ``AR.execute_path``
        for the next data index in the navigation sequence.

        When the user subsequently navigates to that index,
        ``_on_data_idx_changed`` skips ``run_steps`` and calls
        ``update_plots`` directly because the results already live in the
        DS memory cache.
        """
        next_pos = self._nav_pos + 1
        if next_pos >= len(self._data_idxs):
            return
        next_di = self._data_idxs[next_pos]

        # Nothing to do if already done or in progress for this index.
        if next_di == self._prefetched_idx:
            return
        if (self._prefetch_thread is not None
                and self._prefetch_thread.is_alive()
                and self._prefetching_idx == next_di):
            return

        # Only prefetch when a pipeline path is available.
        if not (hasattr(self.AR, 'path') and self.AR.path):
            return

        self._prefetching_idx = next_di
        self._prefetch_status_changed.emit(f"⟳ prefetch {next_di}")

        def _worker():
            try:
                self.AR.execute_path(data_idx = next_di, save_override = False, 
                                     verbose = False)
                self._prefetched_idx = next_di
                # Pre-compute numpy plot data for each panel so update_plots
                # only has to call setData() when the user navigates here.
                for panel in self.panels:
                    try:
                        panel.prefetch_plot_data(next_di)
                    except Exception as exc:
                        print(f"[prefetch] prefetch_plot_data failed for "
                              f"{panel.step_names}: {exc}")
                self._prefetch_status_changed.emit(f"✓ prefetch {next_di}")
            except Exception as exc:
                self._prefetch_status_changed.emit("")
                print(f"[prefetch] data_idx={next_di} failed: {exc}")

        self._prefetch_thread = threading.Thread(
            target=_worker, daemon=True, name=f"prefetch-{next_di}"
        )
        self._prefetch_thread.start()

    def _on_prefetch_status(self, msg: str):
        """Update the prefetch status label (always called on the UI thread via signal)."""
        self._prefetch_label.setText(msg)
        # Auto-clear the "done" message after 4 seconds.
        if msg.startswith("✓"):
            QtCore.QTimer.singleShot(
                4000, lambda: self._prefetch_label.setText("")
            )

    def _on_save_status(self, msg: str):
        """Update the save status label (always called on the UI thread via signal)."""
        self._save_label.setText(msg)

    def closeEvent(self, event):
        """Submit any remaining dirty panels, then flush all background saves."""
        for panel in self.panels:
            if not panel._dirty:
                continue
            panel._dirty = False
            self._submit_save(
                list(panel.steps), panel.AR, panel.data_idx, panel.step_names
            )

        with self._save_count_lock:
            pending = self._saves_in_flight
        if pending > 0:
            orig_title = self.windowTitle()
            self.setWindowTitle(f"{orig_title} — flushing saves…")
            QtWidgets.QApplication.processEvents()
        self._save_executor.shutdown(wait=True)
        self._save_executor = None
        super().closeEvent(event)


################################################################################
# Public entry point
################################################################################

def run_interactive(
    AR,
    panels=None,
    start_idx=0,
    data_idxs=None,
    title="Interactive Analysis",
    ui_scale=1.0,
    plot_scale=1.0,
):
    """
    Build and show an InteractiveAnalysisWindow, then start the Qt event loop.

    Parameters:
    AR (AnalysisRunner): Runner with loaded steps. Must have AR.DS accessible
        for data_idx spinbox bounds.
    panels (list or None): Grouping of steps into panels. Each element is
        either a str (single step) or a list/tuple of strings (multi-step
        panel). If None, one panel per step is created from AR.path (when
        loaded from a YAML), or from AR.analysis_steps as a fallback.
    start_idx (int): Index into data_idxs to start at. Default 0.
    data_idxs (list of int or None): Ordered sequence of data indices to step
        through with the navigation buttons. None uses all rows.
    title (str): Window title. Default 'Interactive Analysis'.
    ui_scale (float): Scales text and widget chrome. 1.0 is the default size.
        Increase (e.g. 1.2) when text is too small; decrease (e.g. 0.85)
        when the interface is too large.
    plot_scale (float): Scales the minimum height of every plot area
        independently of text. 1.0 is the default.

    Returns:
    win (InteractiveAnalysisWindow): The created (and already shown) window.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp(title)

    # Default: one panel per step, derived from AR.path when available
    if panels is None:
        if hasattr(AR, "path") and AR.path:
            panels = [(sd["task"].name,) for sd in AR.path]
        else:
            panels = [(s.name,) for s in AR.analysis_steps]

    # Normalise: str → (str,), list → tuple
    normalized = []
    for p in panels:
        if isinstance(p, str):
            normalized.append((p,))
        else:
            normalized.append(tuple(p))

    win = InteractiveAnalysisWindow(
        AR, normalized, start_idx=start_idx, data_idxs=data_idxs, title=title,
        ui_scale=ui_scale, plot_scale=plot_scale,
    )
    win.show()
    app.exec_()
    return win
