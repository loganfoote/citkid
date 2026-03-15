"""
Pre-assembled interactive session for the IQ analysis pipeline.

This module wires together the panels registered for the steps defined in
``pipeline/templates/iq_analysis_template.yaml``:

    1. make_fr_spans + fit_gain  →  :class:`~.gain.GainFitPanel`
    2. fit_iq                    →  :class:`~.fit_iq.FitIQPanel`

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
    ('make_fr_spans', 'fit_gain'),
    ('fit_iq',),
]


def run_iq_analysis(AR, data_idx=0, title="IQ Analysis"):
    """
    Launch the interactive IQ analysis window.

    This is a convenience wrapper for :func:`~.core.run_interactive` that
    pre-sets the panel grouping to match
    ``pipeline/templates/iq_analysis_template.yaml``.

    Parameters
    ----------
    AR : AnalysisRunner
        Runner whose ``DS`` already has the calibration pipeline loaded and
        whose ``analysis_steps`` include at least *make_fr_spans*,
        *fit_gain*, and *fit_iq*.
    data_idx : int
        Starting per-row data index.  Can be changed in the toolbar.
    title : str
        Window title.

    Returns
    -------
    InteractiveAnalysisWindow
        The created (and already shown) window.

    Notes
    -----
    To run only a *subset* of panels (e.g. gain fitting only) or to attach
    extra custom panels, call :func:`~.core.run_interactive` directly::

        from citkid.pipeline.interactive import run_interactive
        run_interactive(AR, panels=[('make_fr_spans', 'fit_gain')], data_idx=0)
    """
    return run_interactive(AR, panels=_IQ_PANELS, data_idx=data_idx,
                           title=title)
