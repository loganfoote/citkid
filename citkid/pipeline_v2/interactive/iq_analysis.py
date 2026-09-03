"""
Pre-assembled interactive session for the IQ analysis pipeline.

This module wires together the panels registered for the steps defined in
``pipeline_v2/templates/iq_analysis.yaml``:

    1. fit_gain  →  :class:`~.gain.GainFitPanel`
    2. fit_iq    →  :class:`~.fit_iq.FitIQPanel`

Usage
-----
::

    from citkid.pipeline_v2.interactive.iq_analysis import run_iq_analysis
    run_iq_analysis(AR, data_idx=0)

The function is equivalent to calling :func:`~.core.run_interactive` with
the correct panel grouping for the IQ analysis YAML, but gives a cleaner
one-liner interface.
"""

from .core import run_interactive
# Importing the panel modules triggers their @register_panel decorators.
from . import gain      # noqa: F401 — registers GainFitPanel
from . import fit_iq    # noqa: F401 — registers FitIQPanel

# Step grouping that matches iq_analysis.yaml
_IQ_PANELS = [
    ('fit_gain',),
    ('fit_iq',),
]


def run_iq_analysis(AR, start_idx=0, data_idxs=None, title="IQ Analysis",
                    ui_scale=1.0, plot_scale=1.0):
    """
    Launch the interactive IQ analysis window.

    This is a convenience wrapper for run_interactive that pre-sets the panel
    grouping to match pipeline_v2/templates/iq_analysis.yaml.

    Parameters
    ----------
    AR : AnalysisRunner
        The analysis runner whose DS already has the calibration pipeline
        loaded and whose analysis_steps include fit_gain and fit_iq.
    start_idx : int, optional
        Index into ``data_idxs`` to start at. Default 0.
    data_idxs : list of int or None, optional
        Ordered sequence of data indices to step through with the navigation
        buttons. None (default) uses all rows (0 to num_rows - 1).
    title : str, optional
        Window title. Default 'IQ Analysis'.
    ui_scale : float, optional
        Scaling factor for UI elements. Default 1.0.
    plot_scale : float, optional
        Scaling factor for plot heights. Default 1.0.
    """
    return run_interactive(
        AR,
        panels=_IQ_PANELS,
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
    )
