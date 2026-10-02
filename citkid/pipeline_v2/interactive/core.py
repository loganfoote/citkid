"""Modular, stackable interactive analysis framework for pipeline_v2 steps.

This module is adapted from pipeline.interactive to work with pipeline_v2's
simplified single-output model. The key difference is that re-running a step
deletes (rather than marks as NaN) any downstream outputs.

Core concepts  are the same as pipeline.interactive:
- StepPanel: one "card" in the stacked UI
- register_panel: decorator to bind panels to step names  
- InteractiveAnalysisWindow: main window that stacks panels
- run_interactive: convenience entry point

See pipeline.interactive for full documentation.
"""

import concurrent.futures
import traceback
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import threading

from ..analysis import AnalysisRunner
from ...qt_compat import (
    responsive_busy,
    run_responsive,
    TITLE_BAR_MARGIN,
    Qt as _Qt,
    delete_on_close,
    fit_window_to_screen,
    get_qapp,
    scroll_area_content_height,
    vbox_height_for_width,
)
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
# Fonts
################################################################################

# Base font sizes in points at ui_scale = 1. Fonts are always set explicitly,
# so text size depends only on ui_scale, not on the system default font.
WIDGET_FONT_PT = 10
TICK_FONT_PT = 9
LABEL_FONT_PT = 11


def widget_font_stylesheet(ui_scale):
    """
    Return a stylesheet that sets the font size of every widget.

    Parameters:
    ui_scale (float): Font size multiplier.

    Returns:
    stylesheet (str): ``'* { font-size: Npt; }'`` with
        N = ``WIDGET_FONT_PT * ui_scale`` (at least 6).
    """
    return f"* {{ font-size: {max(6, round(WIDGET_FONT_PT * ui_scale))}pt; }}"


def scale_plot_fonts(ui_scale, *plot_items):
    """
    Set tick, axis-label and title fonts of pyqtgraph plots from ui_scale.

    Plot text isn't affected by widget stylesheets, so it is set here.

    Parameters:
    ui_scale (float): Font size multiplier.
    *plot_items (pg.PlotItem): Plots to style.
    """
    tick_font = QtGui.QFont()
    tick_font.setPointSize(max(6, round(TICK_FONT_PT * ui_scale)))
    label_font = QtGui.QFont()
    label_font.setPointSize(max(7, round(LABEL_FONT_PT * ui_scale)))
    for plot in plot_items:
        for ax_name in ('bottom', 'left', 'top', 'right'):
            ax = plot.getAxis(ax_name)
            ax.setStyle(tickFont=tick_font)
            ax.label.setFont(label_font)
        plot.titleLabel.item.setFont(label_font)


################################################################################
# Base panel
################################################################################

class NanOutputsWrite:
    """
    The zarr side of marking one row of one DataSet bad.

    Built by ``StepPanel._store_nan_outputs`` after the bad values are in
    memory, so it can run later (e.g. in a background thread).

    Attributes:
    DS (DataSet): Dataset to write to.
    rows (np.ndarray): Rows affected.
    delete (list of str): Parameters whose saved rows are deleted.
    write (list of str): Parameters whose in-memory rows are written.
    """

    def __init__(self, DS, rows, delete, write):
        """
        Store the work. See the class docstring for parameters.
        """
        self.DS = DS
        self.rows = rows
        self.delete = list(delete)
        self.write = list(write)

    def run(self):
        """
        Delete, then write. One delete call, so the zarr store is listed once.
        """
        if self.delete:
            self.DS.delete_saved_params(self.delete, data_idx=self.rows)
        for name in self.write:
            self.DS._write_per_row_param(name, data_idx=self.rows)


class StepPanel(QtWidgets.QWidget):
    """
    Abstract base class for an interactive analysis step panel.

    Each panel owns one or more pipeline steps. Override setup_ui,
    get_params_for_step, and update_plots in subclasses.

    Signals:
    downstream_rerun (pyqtSignal(object)): Emitted (with self as payload) when
        trigger_downstream is called after a successful run.
        InteractiveAnalysisWindow connects to this to clear and mark all
        downstream panels as needing a re-run.

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
        self._needs_run: bool = False  # True when an upstream panel invalidated this panel
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

    def clear_plots(self):
        """
        Blank all plot displays without re-running steps.

        Override in subclasses to set all plot curves to empty data, reset
        status labels, etc.  Called by the series windows when the user
        navigates to a different resonator so the panels do not show stale
        results.
        """
        pass

    def refresh_plots(self):
        """
        Call ``update_plots``, then blank curves that have no finite points.

        Data marked bad is NaN. An all-NaN curve has no bounds, so pyqtgraph
        autoscaling would warn ("All-NaN slice encountered"); an empty curve
        is simply skipped.
        """
        self.update_plots()
        self._blank_nonfinite_items()

    def blank_plots(self):
        """
        Call ``clear_plots``, then drop data pyqtgraph keeps for empty curves.
        """
        self.clear_plots()
        self._blank_nonfinite_items()

    def _blank_nonfinite_items(self):
        """
        Empty every plot data item in this panel with no finite (x, y) point.

        Notes:
        Given empty data, a pyqtgraph 0.13 ``PlotDataItem`` only hides its
        symbol scatter, which keeps the old points. Qt still asks that hidden
        scatter for its bounds, so old all-NaN points warn too. The scatter is
        therefore cleared as well.
        """
        views = self.findChildren(pg.GraphicsLayoutWidget) + self.findChildren(pg.PlotWidget)
        for view in views:
            plots = view.ci.items if isinstance(view, pg.GraphicsLayoutWidget) else [view.getPlotItem()]
            for plot in plots:
                if not hasattr(plot, 'listDataItems'):
                    continue
                for item in plot.listDataItems():
                    x, y = item.getData()
                    if x is not None and len(x):
                        finite = np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(np.asarray(y, dtype=float))
                        if finite.any():
                            continue
                        item.setData([], [])
                    item.scatter.clear()

    def _scale_plot_fonts(self, *plot_items):
        """
        Apply self.ui_scale to tick labels, axis labels, and titles on each
        of the supplied pyqtgraph PlotItem objects.

        Call at the end of setup_ui after all plots have been created.

        Parameters:
        *plot_items (pg.PlotItem): Plots to style.
        """
        scale_plot_fonts(self.ui_scale, *plot_items)

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

        Notes:
        The steps run in a worker thread (``run_responsive``) so the window
        keeps responding during long fits; widget values are read before and
        the plots updated after.
        """
        # Read the widgets here, on the GUI thread.
        plan = [
            (step,
             None if step.func_type in ("global", "global-res") else self.data_idx,
             self.get_params_for_step(step))
            for step in self.steps
        ]

        def execute():
            """
            Run the steps in order, stopping at the first error or failed row.

            Returns:
            step (plStep or None): step that failed, or None.
            exc (Exception or None): the exception it raised.
            failures (dict or None): its failed rows, ``{data_idx: traceback}``.
            """
            for step, step_di, params in plan:
                try:
                    self.AR.execute_step(
                        step, data_idx=step_di, user_params=params, save=save
                    )
                except Exception as exc:
                    return step, exc, None
                failures = getattr(self.AR, '_last_failures', None)
                if failures:
                    return step, None, dict(failures)
            return None, None, None

        step, exc, failures = run_responsive(execute)
        if exc is not None:
            self._last_error = exc
            self._on_step_error(step, exc)
            return False
        if failures:
            failed_idxs = sorted(failures.keys())
            err = RuntimeError(
                f"Step '{step.name}' failed for row(s): {failed_idxs}"
            )
            self._last_error = err
            msg = f"Error in '{step.name}': {err}"
            if hasattr(self, "_status_label"):
                self._status_label.setText(msg)
            for di, tb_str in sorted(failures.items()):
                print(f"--- Row {di} ---\n{tb_str}")
            return False

        self._has_run = True
        self._last_error = None
        self._dirty = True
        self._needs_run = False
        try:
            self.refresh_plots()
        except Exception as exc:
            print(f"Warning: update_plots() raised in {self.step_names}: {exc}")
        return True

    def run_current(self, save=False):
        """
        Run only this panel and mark downstream panels stale.

        Parameters:
        save (bool): Passed through to run_steps.

        Returns:
        bool: True when the panel ran successfully.
        """
        self.prepare_run()
        if hasattr(self, "_status_label"):
            self._status_label.setText("Running…")
            QtWidgets.QApplication.processEvents()
        ok = self.run_steps(save=save)
        if ok:
            if hasattr(self, "_status_label"):
                self._status_label.setText("Done ✓")
            self.trigger_downstream()
        return ok

    def prepare_run(self):
        """
        Called by _run_through_panel immediately before run_steps on each
        panel.

        Base implementation is a no-op. Sub-classes may override to perform
        pre-run setup (e.g. rebuilding a mask from the current region state).
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
        Emit downstream_rerun to ask the window to mark all following panels
        stale because their outputs were invalidated by this panel.

        Call this at the end of a user-triggered action (e.g. button click
        after run_steps succeeds.
        """
        self.downstream_rerun.emit(self)

    def save_outputs(self):
        """
        Persist the most recent in-memory outputs of every step owned by this
        panel to the zarr file. Does not re-run the steps. Raises if any step
        has no cached results yet.

        The widgets are read here; the zarr writes run in a worker thread
        (``run_responsive``), so the window keeps responding on slow drives.
        """
        AR = self.AR
        plan = [
            (step,
             None if step.func_type in ("global", "global-res") else self.data_idx,
             self.get_params_for_step(step))
            for step in self.steps
        ]

        def write():
            """
            Save the user parameters, then the outputs (no Qt calls).
            """
            for step, step_di, user_params in plan:
                if not user_params:
                    continue
                pipeline_scope, step_index = AR._resolve_step_scope(step)
                AR._add_user_params(
                    step,
                    user_params,
                    data_idx=step_di,
                    save=True,
                    pipeline_scope=pipeline_scope,
                    step_index=step_index,
                )
            for step, step_di, _ in plan:
                AR.save_step_outputs(step, data_idx=step_di)

        run_responsive(write)
        self._dirty = False

    def _write_nan_outputs(self, redraw=True, save=False, keep=()):
        """
        Write placeholder "bad data" outputs for the current row only.

        Everything computed from these outputs for the row is invalidated
        too, as when the step is rerun: later analysis outputs and
        calibration products (e.g. ``xt`` and ``sxx_*`` from ``poly_x``), in
        memory and in zarr. Otherwise stale values (e.g. a series-plot point)
        would survive the bad mark.

        Sub-classes should override _nan_outputs to return the dict of
        {return_name: bad_value} appropriate for their step(s).

        Parameters:
        redraw (bool): If True (default), redraw the plots afterwards. The
            series windows pass False when marking several series points,
            and redraw once at the end.
        save (bool): If True, write the bad outputs to zarr now. If False
            (default), they are held in memory and the panel is marked dirty,
            to be saved with the panel's other outputs.
        keep (iterable of str): Downstream names not to delete from zarr,
            because the caller writes them next (e.g. the bad outputs of the
            later panels).

        Returns:
        ok (bool): True on success, False if no override is provided or an
            error occurs.
        """
        job = self._store_nan_outputs(save=save, keep=keep)
        if job is None:
            return False
        try:
            job.run()
        except Exception as exc:
            print(f"_write_nan_outputs failed in {self.step_names}: {exc}")
            return False
        if redraw:
            try:
                self.refresh_plots()
            except Exception:
                pass
        return True

    def _store_nan_outputs(self, save=False, keep=()):
        """
        Mark the current row bad in memory, and return the matching zarr work.

        The bad outputs are stored in memory, and everything computed from
        them for the row is invalidated in memory (see ``_write_nan_outputs``).
        Invalidated rows are never read back from zarr, so the returned job
        can run later, e.g. in a background thread.

        Parameters:
        save (bool): If True, the job writes the bad outputs to zarr and the
            panel is not marked dirty. If False (default), the job deletes the
            saved outputs and the panel is marked dirty, to save the bad
            outputs with the panel's other outputs.
        keep (iterable of str): Downstream names the job doesn't delete,
            because the caller writes them (e.g. the bad outputs of the later
            panels).

        Returns:
        job (NanOutputsWrite or None): The zarr deletes and writes, or None if
            no override of ``_nan_outputs`` is provided or an error occurs.
        """
        nan_vals = self._nan_outputs()
        if not nan_vals:
            return None
        try:
            DS = self.AR.DS
            di = int(self.data_idx)
            data_idx_arr = np.atleast_1d(np.asarray([di], dtype=np.int32))
            producers = {name: self._find_output_step(name) for name in nan_vals}
            outputs = [name for name, step in producers.items() if step is not None]
            downstream, delete = self._downstream_of(list(nan_vals), data_idx_arr)
            DS.invalidate_memory_params(sorted(downstream) + outputs, data_idx=data_idx_arr)
            for name in outputs:
                producer, value = producers[name], nan_vals[name]
                pipeline_scope, step_index = self.AR._resolve_step_scope(producer)
                DS._store_param(
                    name,
                    [value],
                    is_global=False,
                    data_idx=data_idx_arr,
                    pipeline_scope=pipeline_scope,
                    step_name=producer.name,
                    step_index=step_index,
                    save=False,
                )
            delete.difference_update(keep)
            self._has_run = True
            self._dirty = not save
            self._needs_run = False
            # Saved outputs are overwritten when saving, so they are only
            # deleted when the bad values stay in memory for now.
            return NanOutputsWrite(
                DS, data_idx_arr,
                delete=sorted(delete) + ([] if save else outputs),
                write=outputs if save else [],
            )
        except Exception as exc:
            print(f"_write_nan_outputs failed in {self.step_names}: {exc}")
            return None

    def _downstream_of(self, names, data_idx_arr):
        """
        Find everything computed from some outputs, for some rows.

        Uses the same invalidation plan as rerunning the producing steps:
        later analysis outputs and dependent calibration products.

        Parameters:
        names (list of str): Output names being replaced.
        data_idx_arr (np.ndarray): Rows affected.

        Returns:
        downstream (set of str): Names to drop from memory, excluding
            ``names``.
        delete (set of str): Names to delete from zarr, excluding ``names``.
        """
        downstream, delete = set(), set()
        producers = {}
        for name in names:
            step = self._find_output_step(name)
            if step is not None:
                producers[step.name] = step
        for step in producers.values():
            pipeline_scope, step_index = self.AR._resolve_step_scope(step)
            plan = self.AR._build_invalidation_plan(
                step, {}, data_idx_arr, pipeline_scope, step_index)
            downstream.update(plan["memory_invalidate"])
            delete.update(plan["zarr_delete"])
        downstream.difference_update(names)
        delete.difference_update(names)
        return downstream, delete

    def _nan_outputs(self):
        """
        Return a dict {return_name: bad_value} for each per-row output.

        Base implementation returns {} (no-op). Sub-classes should override
        this to provide placeholder values of the correct shape and dtype.

        Returns:
        dict: Mapping of output name to placeholder bad-data value.
        """
        return {}

    def mark_stale(self):
        """
        Mark this panel stale after an upstream rerun invalidated its outputs.
        """
        self._has_run = False
        self._dirty = False
        self._needs_run = True
        self._last_error = None
        self._autorange_next = True
        try:
            self.clear_plots()
        except Exception:
            pass
        if hasattr(self, "_status_label"):
            self._status_label.setText("Needs run")

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

    def _find_output_step(self, output_name):
        """
        Return the step in this panel that produces output_name.
        """
        for step in self.steps:
            if output_name in step.return_names:
                return step
        return None

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
        self._ensure_global_prerequisites()
        if self._outputs_exist():
            self.refresh_plots()
        else:
            self.run_steps(save=False)

    def _ensure_global_prerequisites(self):
        """
        Silently run any global analysis steps in AR.path whose outputs are
        needed by this panel's steps but are not yet available in AR.DS.

        For example, ``make_fr_spans`` must run before ``fit_gain`` because
        ``fr_spans`` is a required input.  This method is called once during
        ``_auto_initialize`` so the dependency is satisfied automatically.
        """
        if not hasattr(self.AR, 'path'):
            return
        # Collect parameter names consumed by this panel's steps
        needed = set()
        for step in self.steps:
            needed.update(step.param_names)
        # Walk AR.path in order and run any global step whose outputs are both
        # needed and not yet available in DS.
        for step_dict in self.AR.path:
            step = step_dict['task']
            if step.func_type not in ('global', 'global-res'):
                continue
            if not needed.intersection(step.return_names):
                continue
            ds = self.AR.DS
            already_done = True
            for name in step.return_names:
                try:
                    val = getattr(ds, name)
                    if val is None:
                        already_done = False
                        break
                except Exception:
                    already_done = False
                    break
            if not already_done:
                try:
                    run_responsive(self.AR.execute_step, step, data_idx=None, save=False)
                except Exception as exc:
                    print(
                        f"Warning: auto-prerequisite '{step.name}' failed: {exc}"
                    )

    def _on_step_error(self, step, exc):
        """
        Handle a step-level execution error.

        Writes a short message to _status_label if it exists and prints the
        full traceback to stdout. Override to add custom error UI (e.g. a
        dialog).
        """
        msg = f"Error in '{step.name}': {exc}"
        if hasattr(self, "_status_label"):
            self._status_label.setText(msg)
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        print(tb)

    def _on_save_clicked(self):
        window = self.window()
        if hasattr(window, '_confirm_stale_downstream_before_save'):
            if not window._confirm_stale_downstream_before_save(self):
                return
        try:
            self.save_outputs()
            self._status_label.setText("Saved ✓")
        except Exception as exc:
            self._status_label.setText("Save error ✗")
            print(f"Save error: {exc}")

    def _on_run_clicked(self):
        self.run_current()

    def _on_bad_data_clicked(self):
        self._status_label.setText("Marking bad…")
        QtWidgets.QApplication.processEvents()
        ok = self._write_nan_outputs()
        if ok:
            self._status_label.setText("Bad data marked ✓")
            self.trigger_downstream()
        else:
            self._status_label.setText("Bad data failed ✗")


################################################################################
# Default panel (fallback for unregistered steps)
################################################################################

class DefaultStepPanel(StepPanel):
    """
    Generic fallback panel for steps that have no registered custom panel.

    Displays the step names, a *Run+* button, and a status indicator.
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

    def _on_save_clicked(self):
        super()._on_save_clicked()

    def _on_step_error(self, step, exc: Exception):
        msg = f"Error in '{step.name}': {exc}"
        if hasattr(self, "_status_label"):
            self._status_label.setText(msg)
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        print(tb)


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
            hints = f"[{n}] run+following"
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
        delete_on_close(self)  # destroy on the GUI thread when closed
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

        # Toolbar: data_idx selector + navigation
        outer.addWidget(self._build_toolbar(start_di))

        # Scrollable panel area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self._container = QtWidgets.QWidget()
        self._panel_layout = QtWidgets.QVBoxLayout(self._container)
        self._panel_layout.setAlignment(_Qt.AlignTop)
        self._panel_layout.setSpacing(6)
        scroll.setWidget(self._container)
        outer.addWidget(scroll)
        self._scroll = scroll

        # Apply font scaling via stylesheet
        central.setStyleSheet(widget_font_stylesheet(ui_scale))

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
        for seq in ("D", "Right"):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(lambda: self._advance(+1))
        for seq in ("A", "Left"):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(lambda: self._advance(-1))

        # R: auto-scale all plot axes
        _sc_r = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        _sc_r.activated.connect(self._autoscale_all)

        # Number keys 1-9: run panel N and all following ("Run +", 1-indexed).
        # Shift+N: run panel N only.
        for idx in range(1, 10):
            seq = str(idx)
            _sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            _sc.activated.connect(lambda _i=idx-1: self._run_through_panel(_i))
            _sc_shift = QtGui.QShortcut(QtGui.QKeySequence(f"Shift+{seq}"), self)
            _sc_shift.activated.connect(lambda _i=idx-1: self._run_panel_by_index(_i))

        self._fit_to_screen(round(1200 * ui_scale))
        # Initialise panels in order once the event loop is running so that
        # the window is fully laid out and panels are initialised sequentially
        # (guaranteeing upstream data is ready before downstream panels run).
        QtCore.QTimer.singleShot(0, self._auto_initialize_all)
    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _fit_to_screen(self, width: int):
        """
        Size the window to show every panel without scrolling, and center it.

        The width is set first, since word-wrapped toolbar text makes the
        needed height depend on it. The height is capped at the screen height,
        beyond which the panels scroll.

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
        fit_window_to_screen(self, frac=0.9, size=(self.width(), self._content_height()),
                             height_margin=TITLE_BAR_MARGIN)

    def _scroll_width(self) -> int:
        """
        Return the width the panel scroll area gets at the current window width.

        Returns:
        width (int): Window width minus the central layout's margins, in
            logical pixels.
        """
        margins = self.centralWidget().layout().contentsMargins()
        return self.width() - margins.left() - margins.right()

    def _content_height(self) -> int:
        """
        Return the window height that shows every panel without scrolling,
        at the window's current width.

        Returns:
        height (int): Height of the toolbar (wrapped at the current width)
            plus the full height of the panels, in logical pixels.
        """
        return vbox_height_for_width(
            self.centralWidget().layout(), self.width(),
            {self._scroll: scroll_area_content_height(self._scroll, width=self._scroll_width())},
        )

    def _build_toolbar(self, data_idx) -> QtWidgets.QWidget:
        """Build the top toolbar with navigation and a ``data_idx`` spinbox."""
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
        self._nav_label.setAlignment(_Qt.AlignCenter)
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
            "[\u2190/A  \u2192/D] navigate    [R] rescale    [N] run+following    [\u21e7N] run panel only"
        )
        # Word wrap lets the hints shrink so the toolbar never forces the
        # window wider than the screen.
        _hints.setWordWrap(True)
        _hints.setStyleSheet("color: palette(mid); font-style: italic;")
        layout.addWidget(_hints, 1)

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
        """Save dirty panels synchronously, then navigate.

        Saves are done on the main thread *before* the data index changes so
        that the in-memory cache still contains the old index's results.
        A background save would race against ``_on_data_idx_changed`` which
        overwrites the memory cache for the new index.
        """
        if not self._confirm_stale_downstream_before_leave():
            return
        for panel in self.panels:
            if not panel._dirty:
                continue
            panel._dirty = False
            try:
                panel.save_outputs()
            except Exception as exc:
                print(
                    f"Warning: save failed for {panel.step_names}: {exc}"
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
            sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
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
        """Run only the panel at position *index* (0-based)."""
        if index >= len(self.panels):
            return
        panel = self.panels[index]
        panel.run_current()

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
        """Mark every panel that comes after *source_panel* as needing a rerun."""
        try:
            src_idx = self.panels.index(source_panel)
        except ValueError:
            return
        for panel in self.panels[src_idx + 1:]:
            panel.mark_stale()

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
        current_di = self._data_idxs[self._nav_pos] if self._data_idxs else value
        if value != current_di and not self._confirm_stale_downstream_before_leave():
            self._idx_spin.blockSignals(True)
            self._idx_spin.setValue(current_di)
            self._idx_spin.blockSignals(False)
            return

        if value != current_di:
            for panel in self.panels:
                if not panel._dirty:
                    continue
                panel._dirty = False
                try:
                    panel.save_outputs()
                except Exception as exc:
                    print(f"Warning: save failed for {panel.step_names}: {exc}")

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
                    # Do not mark dirty here: prefetch is read-only and should
                    # not create new unsaved pipeline outputs.
                    panel._has_run = True
                    panel.refresh_plots()
                else:
                    ok = panel.run_steps()
                    if not ok:
                        break

        # Kick off prefetch of the next index in the navigation sequence.
        QtCore.QTimer.singleShot(200, self._prefetch_next)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_next(self):
        """
        Start a background daemon thread that pre-computes panel plot caches
        for the next data index in the navigation sequence.

        Prefetch intentionally avoids executing pipeline steps because that
        mutates DS run history and can interfere with explicit save-on-nav
        semantics in interactive sessions.
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
                # Pre-compute numpy plot data for each panel so update_plots
                # only has to call setData() when the user navigates here.
                for panel in self.panels:
                    try:
                        panel.prefetch_plot_data(next_di)
                    except Exception as exc:
                        print(f"[prefetch] prefetch_plot_data failed for "
                              f"{panel.step_names}: {exc}")
                self._prefetched_idx = next_di
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
            # A bound slot (not a lambda) ties the timer to the label, so it
            # is cancelled if the window is deleted first.
            QtCore.QTimer.singleShot(4000, self._prefetch_label.clear)

    def _on_save_status(self, msg: str):
        """Update the save status label (always called on the UI thread via signal)."""
        self._save_label.setText(msg)

    def closeEvent(self, event):
        """Submit any remaining dirty panels, then flush all background saves."""
        if responsive_busy():
            event.ignore()  # work in progress; close when it has finished
            return
        if not self._confirm_stale_downstream_before_leave():
            event.ignore()
            return
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
        # Wait in a worker thread so the window keeps responding.
        run_responsive(self._save_executor.shutdown, wait=True)
        self._save_executor = None
        # The window (and its panels) is deleted right after it closes, so
        # let a running prefetch finish first.
        prefetch = getattr(self, '_prefetch_thread', None)
        while prefetch is not None and prefetch.is_alive():
            prefetch.join(timeout=0.05)
            QtWidgets.QApplication.processEvents()
        super().closeEvent(event)

    def _stale_panels_after(self, source_panel=None):
        """
        Return panels whose outputs were invalidated and have not been rerun.
        """
        start_index = 0
        if source_panel is not None:
            try:
                start_index = self.panels.index(source_panel) + 1
            except ValueError:
                start_index = 0
        return [panel for panel in self.panels[start_index:] if panel._needs_run]

    def _confirm_stale_downstream_before_save(self, source_panel):
        """
        Ask before saving when later panels remain stale.
        """
        stale = self._stale_panels_after(source_panel)
        if not stale:
            return True
        names = ", ".join(" + ".join(panel.step_names) for panel in stale)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Downstream Panels Need Run",
            "Saving now will keep later panel outputs missing for this data index.\n\n"
            f"Panels needing a rerun: {names}\n\nContinue saving?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _confirm_stale_downstream_before_leave(self):
        """
        Ask before leaving a data index or closing while panels remain stale.
        """
        stale = self._stale_panels_after()
        if not stale:
            return True
        names = ", ".join(" + ".join(panel.step_names) for panel in stale)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Panels Need Run",
            "Earlier panel changes invalidated later panel outputs for this data index.\n\n"
            f"Panels needing a rerun: {names}\n\nLeave anyway?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes


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
    app = get_qapp(title)

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
    app.exec()
    return win
