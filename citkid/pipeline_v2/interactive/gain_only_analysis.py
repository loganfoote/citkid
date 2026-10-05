"""
Launch an interactive session for gain fitting only.

The window has a single panel, ``fit_gain`` (``GainFitPanel``), with no IQ or
x-calibration panels.

Examples:
    from citkid.pipeline_v2 import run_gain_only_analysis
    run_gain_only_analysis(AR, start_idx=0)
"""

from .core import run_interactive
# Importing the panel module triggers its @register_panel decorator.
from . import gain  # noqa: F401 — registers GainFitPanel

_GAIN_ONLY_PANELS = [
    ('fit_gain',),
]


def run_gain_only_analysis(
    AR, start_idx=0, data_idxs=None, title="Gain Analysis", ui_scale=1.0,
    plot_scale=1.0
):
    """
    Launch the interactive gain-only analysis window.

    Only the gain fitting panel (``fit_gain``) is shown; IQ and x-calibration
    panels are omitted.

    Parameters:
    AR (AnalysisRunner): runner whose ``DS`` has the calibration pipeline
        loaded and whose ``analysis_steps`` include ``fit_gain``.
    start_idx (int): index into ``data_idxs`` to start at. Default is 0.
    data_idxs (list of int or None): data indices to step through with the
        navigation buttons, in order. None (default) uses every row.
    title (str): window title. Default is 'Gain Analysis'.
    ui_scale (float): scaling factor for UI elements. Default is 1.0.
    plot_scale (float): scaling factor for plot heights. Default is 1.0.

    Returns:
    window (InteractiveAnalysisWindow): the window, returned after it is
        closed (the Qt event loop blocks until then).
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
