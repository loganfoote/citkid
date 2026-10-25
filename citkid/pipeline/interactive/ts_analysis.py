"""
Launch an interactive session for the ts_analysis pipeline.

The panels match the steps in ``pipeline/templates/ts_analysis.yaml``
(``make_fr_spans`` runs without a panel):

    ``fit_gain`` -> ``GainFitPanel``
    ``fit_iq_circle``, ``get_idx_t``, ``get_theta_phase_offset``
        -> ``CircleFitPanel``
    ``get_xcal_mask``, ``fit_x_theta`` -> ``XCalPanel``

With the 'ts-fixed_theta' analysis
(``pipeline/templates/ts-fixed_theta_analysis.yaml``), the last panel is
``XCalFixedThetaPanel`` (``get_xcal_mask_fixed_theta``, ``fit_x_theta``).

``run_ts_analysis`` is ``core.run_interactive`` with this panel grouping.

Examples:
    from citkid.pipeline import run_ts_analysis
    run_ts_analysis(AR, start_idx=0)
"""

from .core import run_interactive
# Importing the panel modules triggers their @register_panel decorators.
from . import gain   # noqa: F401 — registers GainFitPanel
from . import circ   # noqa: F401 — registers CircleFitPanel
from . import xcal   # noqa: F401 — registers XCalPanel
from . import xcal_fixed_theta  # noqa: F401 — registers XCalFixedThetaPanel

# Step grouping that matches ts_analysis.yaml
_TS_PANELS = [
    ('fit_gain',),
    ('fit_iq_circle', 'get_idx_t', 'get_theta_phase_offset'),
    ('get_xcal_mask', 'fit_x_theta'),
]
# ... and ts-fixed_theta_analysis.yaml
_TS_FIXED_THETA_PANELS = _TS_PANELS[:2] + [
    ('get_xcal_mask_fixed_theta', 'fit_x_theta'),
]


def ts_panels(AR):
    """
    Return the TS panel grouping for a runner's analysis path.

    Parameters:
    AR (AnalysisRunner): runner with the 'ts' or 'ts-fixed_theta' analysis.

    Returns:
    panels (list of tuple of str): ``_TS_FIXED_THETA_PANELS`` if the path
        has ``get_xcal_mask_fixed_theta``, otherwise ``_TS_PANELS``.
    """
    try:
        names = [step_dict['task'].name for step_dict in AR.path]
    except Exception:
        names = []
    if 'get_xcal_mask_fixed_theta' in names:
        return list(_TS_FIXED_THETA_PANELS)
    return list(_TS_PANELS)


def run_ts_analysis(
    AR, start_idx=0, data_idxs=None, title="TS Analysis", ui_scale=1.0,
    plot_scale=1.0
):
    """
    Launch the interactive TS analysis window.

    This wraps ``run_interactive`` with the panel grouping that matches
    the runner's analysis, ``pipeline/templates/ts_analysis.yaml`` or
    ``ts-fixed_theta_analysis.yaml`` (see ``ts_panels``).

    Parameters:
    AR (AnalysisRunner): runner whose ``DS`` has the calibration pipeline
        loaded and whose ``analysis_steps`` include ``fit_gain``,
        ``fit_iq_circle``, ``get_idx_t``, ``get_theta_phase_offset``,
        ``get_xcal_mask`` (or ``get_xcal_mask_fixed_theta``), and
        ``fit_x_theta``.
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
        panels=ts_panels(AR),
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
    )
