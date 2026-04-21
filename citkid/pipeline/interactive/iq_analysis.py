"""
Pre-assembled interactive session for the IQ analysis pipeline.

This module wires together the panels registered for the steps defined in
``pipeline/templates/iq_analysis_template.yaml``:

    1. fit_gain  →  :class:`~.gain.GainFitPanel`
    2. fit_iq    →  :class:`~.fit_iq.FitIQPanel`

Usage
-----
::

    from citkid.pipeline.interactive.iq_analysis import run_iq_analysis
    run_iq_analysis(AR, data_idx=0)

The function is equivalent to calling :func:`~.core.run_interactive` with
the correct panel grouping for the IQ analysis YAML, but gives a cleaner
one-liner interface.
"""

from .core import run_interactive
# Importing the panel modules triggers their @register_panel decorators.
from . import gain      # noqa: F401 — registers GainFitPanel
from . import fit_iq    # noqa: F401 — registers FitIQPanel

# Step grouping that matches iq_analysis_template.yaml
_IQ_PANELS = [
    ('fit_gain',),
    ('fit_iq',),
]


def run_iq_analysis(AR, start_idx=0, data_idxs=None, title="IQ Analysis",
                    ui_scale=1.0, plot_scale=1.0):
    """
    Launch the interactive IQ analysis window.

    This is a convenience wrapper for run_interactive that pre-sets the panel
    grouping to match pipeline/templates/iq_analysis_template.yaml.

    Parameters:
    AR (AnalysisRunner): Runner whose DS already has the calibration pipeline
        loaded and whose analysis_steps include at least fit_gain and fit_iq.
    start_idx (int): Index into data_idxs to start at. Default 0.
    data_idxs (list of int or None): Ordered sequence of data indices to step
        through with the navigation buttons. None uses all rows.
    title (str): Window title. Default 'IQ Analysis'.
    ui_scale (float): Scales text and widget chrome. 1.0 is the default size.
    plot_scale (float): Scales the minimum height of plot areas. 1.0 is the
        default.

    Returns:
    win (InteractiveAnalysisWindow): The created (and already shown) window.
    """
    return run_interactive(AR, panels=_IQ_PANELS, start_idx=start_idx,
                           data_idxs=data_idxs, title=title, ui_scale=ui_scale,
                           plot_scale=plot_scale)
