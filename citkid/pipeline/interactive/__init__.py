"""
citkid.pipeline.interactive: modular, stackable interactive analysis UI.

Public API:
- ``StepPanel``: base class for all interactive panels. Override
  ``setup_ui``, ``get_params_for_step``, and ``update_plots``.
- ``register_panel(*step_names)``: class decorator that associates a
  ``StepPanel`` subclass with one or more pipeline step names.
- ``get_panel_class(step_names)``: look up the registered panel class for a
  tuple of step names.
- ``InteractiveAnalysisWindow``: main window that stacks panels and
  orchestrates cascade re-runs.
- ``run_interactive(AR, panels, start_idx, data_idxs, title, ...)``: build
  and show a window, then start the Qt event loop.

Built-in panels:
- ``gain.GainFitPanel``: handles *fit_gain*.
- ``fit_iq.IQFitPanel``: handles *fit_iq* with interactive mask selection.
- ``circ.CircleFitPanel``: handles *fit_iq_circle* + *get_idx_t* +
  *get_theta_phase_offset*.
- ``xcal.XCalPanel``: handles *get_xcal_mask* + *fit_x_theta*.

Pipeline-specific launchers:
- ``iq_analysis.run_iq_analysis``: one-liner for the IQ analysis YAML
  pipeline.
- ``ts_analysis.run_ts_analysis``: one-liner for the ts_analysis YAML
  pipeline.
- ``iq_ts_analysis.run_iq_ts_analysis``: the ts_analysis panels followed by
  the IQ fit panel.
"""

# Core framework — always imported
from .core import (
    StepPanel,
    DefaultStepPanel,
    register_panel,
    get_panel_class,
    InteractiveAnalysisWindow,
    run_interactive,
)

# Built-in panels — importing them triggers @register_panel registration
from . import gain      # noqa: F401
from . import fit_iq    # noqa: F401
from . import circ      # noqa: F401
from . import xcal      # noqa: F401

from .iq_series import IQSeriesWindow, SeriesValues, SeriesXYFit

# Pipeline-specific assemblers
from .iq_analysis import run_iq_analysis  # noqa: F401
from .ts_analysis import run_ts_analysis  # noqa: F401
from .iq_ts_analysis import run_iq_ts_analysis  # noqa: F401
from .gain_only_analysis import run_gain_only_analysis  # noqa: F401
from .iq_series import run_iq_series  # noqa: F401
from .ts_series import TSSeriesWindow, run_ts_series  # noqa: F401

__all__ = [
    # framework
    "StepPanel",
    "DefaultStepPanel",
    "register_panel",
    "get_panel_class",
    "InteractiveAnalysisWindow",
    "run_interactive",
    # panels
    "gain",
    "fit_iq",
    "circ",
    "xcal",
    # assemblers
    "run_iq_analysis",
    "run_gain_only_analysis",
    "run_ts_analysis",
    "run_iq_ts_analysis",
    "run_iq_series",
    "run_ts_series",
    "IQSeriesWindow",
    "TSSeriesWindow",
    "SeriesXYFit",
    "SeriesValues",
]
