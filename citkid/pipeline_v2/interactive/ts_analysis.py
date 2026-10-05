"""
Launch an interactive session for the ts_analysis pipeline.

The panels match the steps in ``pipeline_v2/templates/ts_analysis.yaml``
(``make_fr_spans`` runs without a panel):

    ``fit_gain`` -> ``GainFitPanel``
    ``fit_iq_circle``, ``get_idx_t``, ``get_theta_phase_offset``
        -> ``CircleFitPanel``
    ``get_xcal_mask``, ``fit_x_theta`` -> ``XCalPanel``

``run_ts_analysis`` is ``core.run_interactive`` with this panel grouping.

Examples:
    from citkid.pipeline_v2 import run_ts_analysis
    run_ts_analysis(AR, start_idx=0)
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


def run_ts_analysis(
    AR, start_idx=0, data_idxs=None, title="TS Analysis", ui_scale=1.0,
    plot_scale=1.0
):
    """
    Launch the interactive TS analysis window.

    This wraps ``run_interactive`` with the panel grouping that matches
    ``pipeline_v2/templates/ts_analysis.yaml``.

    Parameters:
    AR (AnalysisRunner): runner whose ``DS`` has the calibration pipeline
        loaded and whose ``analysis_steps`` include ``fit_gain``,
        ``fit_iq_circle``, ``get_idx_t``, ``get_theta_phase_offset``,
        ``get_xcal_mask``, and ``fit_x_theta``.
    start_idx (int): index into ``data_idxs`` to start at. Default is 0.
    data_idxs (list of int or None): data indices to step through with the
        navigation buttons, in order. None (default) uses every row.
    title (str): window title. Default is 'TS Analysis'.
    ui_scale (float): scaling factor for UI elements. Default is 1.0.
    plot_scale (float): scaling factor for plot heights. Default is 1.0.

    Returns:
    window (InteractiveAnalysisWindow): the window, returned after it is
        closed (the Qt event loop blocks until then).
    """
    return run_interactive(
        AR,
        panels=_TS_PANELS,
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
    )
