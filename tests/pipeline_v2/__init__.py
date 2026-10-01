"""Tests for pipeline_v2 module.

This package contains comprehensive tests for citkid.pipeline_v2, adapted from
citkid.pipeline tests but customized for the v2 architecture.

Test Organization:
- test_framework_v2.py: Tests for plStep, LazyAttr, and utility functions
- test_default_steps_v2.py: Tests for default_cal_steps and default_analysis_steps format
- test_execution_options_v2.py: Tests for the vectorize and path_per_row execution options
- test_pipeline_v2.py: Integration tests for full pipeline workflows

NOTE: LazyAttrCollection tests are NOT included here (v1-only multi-run feature).
All other framework tests are fully ported.
"""
