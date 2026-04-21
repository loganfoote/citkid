"""
Pre-assembled interactive session for the ts_analysis pipeline.

This module wires together the panels registered for the steps defined in
``pipeline/templates/ts_analysis.yaml``:

    1 + 2. make_fr_spans + fit_gain          → :class:`~.gain.GainFitPanel`
    3–5.   fit_iq_circle + get_idx_t
           + get_theta_phase_offset          → :class:`~.circ.CircleFitPanel`
    6–7.   get_xcal_mask + fit_x_theta       → :class:`~.xcal.XCalPanel`

Usage
-----
::

    from citkid.pipeline.interactive.ts_analysis import run_ts_analysis
    run_ts_analysis(AR, data_idx=0)

The function is equivalent to calling :func:`~.core.run_interactive` with
the correct panel grouping for the ts_analysis YAML, but gives a cleaner
one-liner interface.
"""

from .core import run_interactive
# Importing the panel modules triggers their @register_panel decorators.
from . import gain   # noqa: F401 — registers GainFitPanel
from . import circ   # noqa: F401 — registers CircleFitPanel
from . import xcal   # noqa: F401 — registers XCalPanel

# Step grouping that matches ts_analysis.yaml
_TS_PANELS = [
    ('fit_gain',),
    ('fit_iq_circle', 'get_idx_t', 'get_theta_phase_offset'),
    ('get_xcal_mask', 'fit_x_theta'),
]


def run_ts_analysis(AR, start_idx=0, data_idxs=None, title="TS Analysis",
                    ui_scale=1.0, plot_scale=1.0):
    """
    Launch the interactive ts_analysis window.

    This is a convenience wrapper for run_interactive that pre-sets the panel
    grouping to match pipeline/templates/ts_analysis.yaml.

    Parameters:
    AR (AnalysisRunner): Runner whose DS already has the calibration pipeline
        loaded and whose analysis_steps include the ts_analysis steps
        (make_fr_spans through fit_x_theta).
    start_idx (int): Index into data_idxs to start at. Default 0.
    data_idxs (list of int or None): Ordered sequence of data indices to step
        through with the navigation buttons. None uses all rows.
    title (str): Window title. Default 'TS Analysis'.
    ui_scale (float): Scales text and widget chrome. 1.0 is the default size.
    plot_scale (float): Scales the minimum height of plot areas. 1.0 is the
        default.

    Returns:
    win (InteractiveAnalysisWindow): The created (and already shown) window.
    """
    return run_interactive(AR, panels=_TS_PANELS, start_idx=start_idx,
                           data_idxs=data_idxs, title=title, ui_scale=ui_scale,
                           plot_scale=plot_scale)
