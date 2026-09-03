"""citkid.pipeline.interactive — modular, stackable interactive analysis UI.

Public API
----------
StepPanel
    Base class for all interactive panels.  Override ``setup_ui``,
    ``get_params_for_step``, and ``update_plots``.

register_panel(*step_names)
    Class decorator that associates a :class:`StepPanel` subclass with one
    or more pipeline step names.

get_panel_class(step_names)
    Look up the registered panel class for a tuple of step names.

InteractiveAnalysisWindow
    Main window that stacks panels and orchestrates cascade re-runs.

run_interactive(AR, panels, data_idx, title)
    Build and show a window, then start the Qt event loop.

Built-in panels
---------------
:class:`~.gain.GainFitPanel`
    Handles *make_fr_spans* + *fit_gain*.

:class:`~.fit_iq.FitIQPanel`
    Handles *fit_iq* with interactive mask selection.

:class:`~.circ.CircleFitPanel`
    Handles *fit_iq_circle* + *get_idx_t* + *get_theta_phase_offset*.

:class:`~.xcal.XCalPanel`
    Handles *get_xcal_mask* + *fit_x_theta*.

Pipeline-specific launchers
---------------------------
:func:`~.iq_analysis.run_iq_analysis`
    One-liner for the IQ analysis YAML pipeline.

:func:`~.ts_analysis.run_ts_analysis`
    One-liner for the ts_analysis YAML pipeline.
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

from .sweep_fitter import SweepFitterWindow, run_sweep_fitter

# Pipeline-specific assemblers
from .iq_analysis import run_iq_analysis  # noqa: F401
from .ts_analysis import run_ts_analysis  # noqa: F401
from .gain_only_analysis import run_gain_only_analysis  # noqa: F401
from .sweep_fitter import run_sweep_fitter  # noqa: F401

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
    "run_sweep_fitter",
]
