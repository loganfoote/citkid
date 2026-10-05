"""
Provide the single-version calibration and analysis pipeline.

``DataSet``: zarr-backed parameter store plus calibration-pipeline execution.
``AnalysisRunner``: analysis-pipeline executor built on top of a ``DataSet``.
``plStep``: definition of one pipeline step.
``run_*``: interactive launchers from ``citkid.pipeline_v2.interactive``.
"""

from .dataset import DataSet
from .analysis import AnalysisRunner
from .framework import plStep
from .interactive import (
    run_interactive,
    run_iq_analysis,
    run_ts_analysis,
    run_gain_only_analysis,
    run_iq_series,
    run_ts_series,
)

__all__ = [
    "DataSet",
    "AnalysisRunner",
    "plStep",
    "run_interactive",
    "run_iq_analysis",
    "run_ts_analysis",
    "run_gain_only_analysis",
    "run_iq_series",
    "run_ts_series",
]
