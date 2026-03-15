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

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..analysis import AnalysisRunner


################################################################################
# Panel registry
################################################################################

_PANEL_REGISTRY: dict[tuple, type] = {}


def register_panel(*step_names: str):
    """
    Class decorator that registers a :class:`StepPanel` subclass for a group
    of step names.

    Parameters
    ----------
    *step_names : str
        One or more pipeline step names this panel handles, in execution order.

    Example
    -------
    ::

        @register_panel('make_fr_spans', 'fit_gain')
        class GainFitPanel(StepPanel):
            ...
    """
    def decorator(cls):
        _PANEL_REGISTRY[tuple(step_names)] = cls
        return cls
    return decorator


def get_panel_class(step_names: tuple) -> type:
    """
    Return the :class:`StepPanel` subclass registered for *step_names*.

    Lookup order:

    1. Exact tuple match in the registry.
    2. Single-step fallback: ``(step_names[0],)`` in the registry.
    3. :class:`DefaultStepPanel` (always available).

    Parameters
    ----------
    step_names : tuple of str

    Returns
    -------
    type
        A :class:`StepPanel` subclass.
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

    Each panel owns one or more pipeline steps.  Override
    :meth:`setup_ui`, :meth:`get_params_for_step`, and
    :meth:`update_plots` in subclasses.

    Signals
    -------
    downstream_rerun : pyqtSignal(object)
        Emitted (with *self* as payload) when
        :meth:`trigger_downstream` is called after a successful run.
        :class:`InteractiveAnalysisWindow` connects to this to cascade
        re-runs through all panels below this one.

    Parameters
    ----------
    AR : AnalysisRunner
        The analysis runner whose ``execute_step`` will be called.
    step_names : tuple of str
        Names of the steps this panel executes, in order.
    data_idx : int or None
        Initial data index for per-row steps.  ``None`` until set.
    parent : QWidget, optional
    """

    #: Emitted when downstream panels should re-run.  Payload is *self*.
    downstream_rerun = QtCore.pyqtSignal(object)

    def __init__(
        self,
        AR: AnalysisRunner,
        step_names: tuple,
        data_idx=None,
        parent=None,
    ):
        super().__init__(parent)
        self.AR = AR
        self.step_names = tuple(step_names)
        self.data_idx = data_idx
        self._has_run = False
        self._last_error: Exception | None = None

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
        Build the panel's widgets.  Called once during ``__init__``.

        The base implementation adds a single status label.  Override to
        replace this with domain-specific controls and plots.
        """
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self._status_label = QtWidgets.QLabel("Not run")
        layout.addWidget(self._status_label)

    def get_params_for_step(self, step) -> dict:
        """
        Return user-controlled parameters for *step* as ``{name: value}``.

        Called by :meth:`run_steps` just before each step executes.
        Override to return widget values (e.g. spinbox, combobox).

        Parameters
        ----------
        step : plStep
            The step about to be executed.

        Returns
        -------
        dict
            Mapping of parameter name → value.  Only user-controlled
            parameters need to be included; pipeline-produced parameters
            are resolved automatically.
        """
        return {}

    def update_plots(self):
        """
        Refresh all plots and displays after the steps have run successfully.

        Override in subclasses to read results from ``self.AR.DS`` and
        update plot widgets.
        """
        pass

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run_steps(self, save: bool = False) -> bool:
        """
        Execute each step owned by this panel, then call :meth:`update_plots`.

        Global and global-res steps always receive ``data_idx=None``; per-row
        and vectorized steps receive ``self.data_idx``.

        Parameters
        ----------
        save : bool
            Passed directly to :meth:`AnalysisRunner.execute_step`.

        Returns
        -------
        bool
            ``True`` if all steps succeeded, ``False`` if any step raised.
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
        self.update_plots()
        return True

    def trigger_downstream(self):
        """
        Emit :attr:`downstream_rerun` to ask the window to re-run all panels
        that come after this one.

        Call this at the end of a user-triggered action (e.g. button click or
        parameter change) after :meth:`run_steps` succeeds.
        """
        self.downstream_rerun.emit(self)

    def set_data_idx(self, data_idx: int):
        """Set a new data index and immediately re-run the panel."""
        self.data_idx = data_idx
        self.run_steps()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_step_error(self, step, exc: Exception):
        """
        Handle a step-level execution error.

        Writes a short message to ``_status_label`` if it exists and prints
        to stdout.  Override to add custom error UI (e.g. a dialog).
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

    The button text shows the step names joined by ``'→'``.  Clicking the
    button hides or shows the wrapped panel.
    """

    def __init__(self, panel: StepPanel, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(0)

        names_str = "  →  ".join(panel.step_names)
        self._btn = QtWidgets.QPushButton(f"▼  {names_str}")
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setFlat(True)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            "  font-weight: bold;"
            "  font-size: 11pt;"
            "  padding: 4px 8px;"
            "  border-bottom: 1px solid palette(mid);"
            "}"
        )
        self._btn.clicked.connect(self._on_toggle)
        layout.addWidget(self._btn)

        self._panel = panel
        layout.addWidget(panel)

    def _on_toggle(self, checked: bool):
        self._panel.setVisible(checked)
        arrow = "▼" if checked else "▶"
        names_str = "  →  ".join(self._panel.step_names)
        self._btn.setText(f"{arrow}  {names_str}")


################################################################################
# Main window
################################################################################

class InteractiveAnalysisWindow(QtWidgets.QMainWindow):
    """
    Main window that vertically stacks step panels and orchestrates cascade
    re-runs.

    When panel *i* emits :attr:`StepPanel.downstream_rerun`, panels
    *i+1*, *i+2*, … are re-run in sequence (stopping on the first
    failure).

    Parameters
    ----------
    AR : AnalysisRunner
        Runner with loaded steps.
    panels : list of tuple of str
        One tuple per panel group.  Each tuple contains the step names
        that belong to that panel.
        Example::

            [('make_fr_spans', 'fit_gain'), ('fit_iq',)]

    data_idx : int or None
        Starting data index for per-row steps.
    title : str
        Window title.
    parent : QWidget, optional
    """

    def __init__(
        self,
        AR: AnalysisRunner,
        panels: list,
        data_idx=None,
        title: str = "Interactive Analysis",
        parent=None,
    ):
        super().__init__(parent)
        self.AR = AR
        self.setWindowTitle(title)

        # Central widget + outer layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Toolbar: data_idx selector + "Run All" button
        outer.addWidget(self._build_toolbar(data_idx))

        # Scrollable panel area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self._container = QtWidgets.QWidget()
        self._panel_layout = QtWidgets.QVBoxLayout(self._container)
        self._panel_layout.setAlignment(QtCore.Qt.AlignTop)
        self._panel_layout.setSpacing(6)
        scroll.setWidget(self._container)
        outer.addWidget(scroll)

        # Build panels
        self.panels: list[StepPanel] = []
        for step_names_tuple in panels:
            cls = get_panel_class(step_names_tuple)
            panel = cls(AR, step_names_tuple, data_idx=data_idx, parent=self)
            panel.downstream_rerun.connect(self._on_panel_rerun)
            self._add_panel(panel)

        self.resize(1200, 900)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_toolbar(self, data_idx) -> QtWidgets.QWidget:
        """Build the top toolbar with a ``data_idx`` spinbox and Run All."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)

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

        layout.addStretch()

        run_all_btn = QtWidgets.QPushButton("Run All")
        run_all_btn.clicked.connect(self.run_all)
        layout.addWidget(run_all_btn)

        return w

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

    def _on_data_idx_changed(self, value: int):
        """
        Propagate a new ``data_idx`` to all panels, then re-run every panel
        that contains at least one per-row or vectorized step (stopping on
        the first failure).
        """
        for panel in self.panels:
            panel.data_idx = value
        for panel in self.panels:
            has_per_row = any(
                s.func_type in ("per-row", "vectorized") for s in panel.steps
            )
            if has_per_row:
                ok = panel.run_steps()
                if not ok:
                    break

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_all(self):
        """Run all panels from top to bottom, stopping on the first failure."""
        for panel in self.panels:
            ok = panel.run_steps()
            if not ok:
                break


################################################################################
# Public entry point
################################################################################

def run_interactive(
    AR: AnalysisRunner,
    panels=None,
    data_idx=None,
    title: str = "Interactive Analysis",
):
    """
    Build and show an :class:`InteractiveAnalysisWindow`, then start the Qt
    event loop.

    Parameters
    ----------
    AR : AnalysisRunner
        Runner with loaded steps.  Must have ``AR.DS`` accessible for
        ``data_idx`` spinbox bounds.
    panels : list or None
        Grouping of steps into panels.  Each element is either a ``str``
        (single step) or a ``list``/``tuple`` of strings (multi-step panel).
        If ``None``, one panel per step is created from ``AR.path`` (when
        loaded from a YAML), or from ``AR.analysis_steps`` as a fallback.

        Example::

            panels = [
                ('make_fr_spans', 'fit_gain'),   # one panel for two steps
                ('fit_iq',),                     # separate panel
            ]

    data_idx : int or None
        Starting per-row data index.
    title : str
        Window title.

    Returns
    -------
    InteractiveAnalysisWindow
        The created (and already shown) window.
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
        AR, normalized, data_idx=data_idx, title=title
    )
    win.show()
    app.exec_()
    return win
