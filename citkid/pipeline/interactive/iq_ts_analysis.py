"""
Launch an interactive session for the ts_analysis pipeline plus the
nonlinear IQ fit.

The panels are those of ``run_ts_analysis`` (see ``ts_analysis.ts_panels``)
followed by ``fit_iq`` -> ``FitIQPanel``:

    ``fit_gain`` -> ``GainFitPanel``
    ``fit_iq_circle``, ``get_idx_t``, ``get_theta_phase_offset``
        -> ``CircleFitPanel``
    ``get_xcal_mask``, ``fit_x_theta`` -> ``XCalPanel``
        (``XCalFixedThetaPanel`` with the 'ts-fixed_theta' analysis)
    ``fit_iq`` -> ``FitIQPanel``

``fit_iq`` is not a step of ``ts_analysis.yaml``. Every runner has it in
``analysis_steps``, so the panel works, but ``execute_path`` does not run it
and the DataSet's invalidation does not tie it to ``fit_gain``. Inside this
window, re-running a panel marks the panels below it stale (the window asks
before leaving a row with stale panels), so after a gain refit run the IQ
panel again. To fit many rows at once outside the window, run the step
directly::

    step = next(s for s in AR.analysis_steps if s.name == 'fit_iq')
    AR.execute_step(step, data_idx=rows, user_params={'iq_mask': None})

``run_iq_ts_analysis`` is ``core.run_interactive`` with this panel grouping.

Examples:
    from citkid.pipeline import run_iq_ts_analysis
    run_iq_ts_analysis(AR, start_idx=0)
"""

from .core import run_interactive
# Importing the panel modules triggers their @register_panel decorators.
from . import fit_iq    # noqa: F401 — registers FitIQPanel
from .ts_analysis import ts_panels


def iq_ts_panels(AR):
    """
    Return the TS panel grouping for a runner's analysis path, followed by
    the IQ fit panel.

    Parameters:
    AR (AnalysisRunner): runner with the 'ts' or 'ts-fixed_theta' analysis.

    Returns:
    panels (list of tuple of str): ``ts_panels(AR)`` plus ``('fit_iq',)``.
    """
    return ts_panels(AR) + [('fit_iq',)]


def run_iq_ts_analysis(
    AR, start_idx=0, data_idxs=None, title="TS + IQ Analysis", ui_scale=1.0,
    plot_scale=1.0
):
    """
    Launch the interactive TS analysis window with the IQ fit panel added.

    This wraps ``run_interactive`` with the TS panel grouping (see
    ``ts_analysis.ts_panels``) followed by the ``fit_iq`` panel.

    Parameters:
    AR (AnalysisRunner): runner whose ``DS`` has the calibration pipeline
        loaded and whose ``analysis_steps`` include the TS steps (see
        ``run_ts_analysis``) and ``fit_iq``.
    start_idx (int): index into ``data_idxs`` to start at. Default is 0.
    data_idxs (list of int or None): data indices to step through with the
        navigation buttons, in order. None (default) uses every row.
    title (str): window title. Default is 'TS + IQ Analysis'.
    ui_scale (float): scaling factor for UI elements. Default is 1.0.
    plot_scale (float): scaling factor for plot heights. Default is 1.0.

    Returns:
    window (InteractiveAnalysisWindow): the window, returned after it is
        closed (the Qt event loop blocks until then).
    """
    return run_interactive(
        AR,
        panels=iq_ts_panels(AR),
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
    )
