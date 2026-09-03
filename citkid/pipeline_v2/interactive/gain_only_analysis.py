"""
Pre-assembled interactive session for gain fitting only.

This module provides a single-panel launcher that runs only the gain fitting
step (fit_gain → :class:`~.gain.GainFitPanel`), with no IQ or x-calibration
panels.

Usage
-----
::

    from citkid.pipeline_v2.interactive.gain_only_analysis import run_gain_only_analysis
    run_gain_only_analysis(AR, data_idx=0)
"""

from .core import run_interactive
# Importing the panel module triggers its @register_panel decorator.
from . import gain  # noqa: F401 — registers GainFitPanel

_GAIN_ONLY_PANELS = [
    ('fit_gain',),
]


def run_gain_only_analysis(AR, start_idx=0, data_idxs=None,
                            title="Gain Analysis",
                            ui_scale=1.0, plot_scale=1.0):
    """
    Launch the interactive gain-only analysis window.

    Only the gain fitting panel (fit_gain) is shown; IQ and x-calibration
    panels are omitted.

    Parameters
    ----------
    AR : AnalysisRunner
        Runner whose DS already has the calibration pipeline
        loaded and whose analysis_steps include fit_gain.
    start_idx : int, optional
        Index into ``data_idxs`` to start at. Default 0.
    data_idxs : list of int or None, optional
        Ordered sequence of data indices to step through with the navigation
        buttons. None (default) uses all rows (0 to num_rows - 1).
    title : str, optional
        Window title. Default 'Gain Analysis'.
    ui_scale : float, optional
        Scaling factor for UI elements. Default 1.0.
    plot_scale : float, optional
        Scaling factor for plot heights. Default 1.0.
    """
    return run_interactive(
        AR,
        panels=_GAIN_ONLY_PANELS,
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
    )
