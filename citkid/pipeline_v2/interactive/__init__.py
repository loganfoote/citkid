"""Interactive entry points for pipeline_v2.

This module provides the same interactive UI framework as pipeline.interactive,
adapted for pipeline_v2's simplified architecture (single active state, automatic
downstream invalidation instead of multiple runs with NaN marking).

The key adaptations:
- core.py: v2-specific implementation of StepPanel._write_nan_outputs() that
  deletes outputs instead of writing NaNs
- Panels (gain.py, fit_iq.py, sweep_fitter.py): v2-specific versions that work
  with the deletion-based marking system
"""

from .core import (
    DefaultStepPanel,
    InteractiveAnalysisWindow,
    StepPanel,
    get_panel_class,
    register_panel,
)

# For now, re-use v1 pipeline-specific launchers that don't depend on
# _write_nan_outputs. Adapt as needed if specific launchers require changes.
try:
    from ...pipeline.interactive import run_interactive
except ImportError:
    # Fallback if pipeline.interactive is not available
    run_interactive = None

__all__ = [
    "StepPanel",
    "DefaultStepPanel",
    "register_panel",
    "get_panel_class",
    "InteractiveAnalysisWindow",
    "run_interactive",
]