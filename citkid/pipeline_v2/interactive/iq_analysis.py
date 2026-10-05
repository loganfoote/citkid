"""
Launch an interactive session for the IQ analysis pipeline.

The panels match the steps in ``pipeline_v2/templates/iq_analysis.yaml``:

    1. ``fit_gain`` -> ``GainFitPanel``
    2. ``fit_iq`` -> ``FitIQPanel``

``run_iq_analysis`` is ``core.run_interactive`` with this panel grouping.

Examples:
    from citkid.pipeline_v2 import run_iq_analysis
    run_iq_analysis(AR, start_idx=0)
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


def run_iq_analysis(
    AR, start_idx=0, data_idxs=None, title="IQ Analysis", ui_scale=1.0,
    plot_scale=1.0
):
    """
    Launch the interactive IQ analysis window.

    This wraps ``run_interactive`` with the panel grouping that matches
    ``pipeline_v2/templates/iq_analysis.yaml``.

    Parameters:
    AR (AnalysisRunner): runner whose ``DS`` has the calibration pipeline
        loaded and whose ``analysis_steps`` include ``fit_gain`` and ``fit_iq``.
    start_idx (int): index into ``data_idxs`` to start at. Default is 0.
    data_idxs (list of int or None): data indices to step through with the
        navigation buttons, in order. None (default) uses every row.
    title (str): window title. Default is 'IQ Analysis'.
    ui_scale (float): scaling factor for UI elements. Default is 1.0.
    plot_scale (float): scaling factor for plot heights. Default is 1.0.

    Returns:
    window (InteractiveAnalysisWindow): the window, returned after it is
        closed (the Qt event loop blocks until then).
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
