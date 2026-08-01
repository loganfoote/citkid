"""Simplified single-version pipeline API.

This package provides a replacement for :mod:`citkid.pipeline` that removes
run history and dependency backtracking.  The public interface mirrors the
original high-level entry points:

``DataSet``
	Zarr-backed parameter store plus calibration-pipeline execution.

``AnalysisRunner``
	Analysis-pipeline executor built on top of a :class:`DataSet`.
"""

from .dataset import DataSet
from .analysis import AnalysisRunner

__all__ = ["DataSet", "AnalysisRunner"]