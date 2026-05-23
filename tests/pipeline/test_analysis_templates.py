"""Tests for analysis YAML templates shipped with the pipeline."""

from pathlib import Path

import yaml

import citkid.pipeline as pipeline_pkg
from citkid.pipeline.interactive.ts_analysis import _TS_PANELS


def _load_template(name: str) -> dict:
    templates_dir = Path(pipeline_pkg.__file__).resolve().parent / 'templates'
    with open(templates_dir / name, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_ts_analysis_fit_x_theta_has_no_xcal_mask_override():
    """fit_x_theta should consume get_xcal_mask output by default."""
    ts = _load_template('ts_analysis.yaml')
    step7 = ts['ANALYSIS_STEPS'][7]

    assert step7['task'] == 'fit_x_theta'
    assert 'params' not in step7 or 'xcal_mask' not in (step7.get('params') or {})


def test_ts_analysis_xcal_mask_step_has_expected_defaults():
    """Template should include explicit user-tunable defaults for mask finding."""
    ts = _load_template('ts_analysis.yaml')
    step6 = ts['ANALYSIS_STEPS'][6]

    assert step6['task'] == 'get_xcal_mask'
    assert step6['params']['xcal_idx0_offset'] == 3
    assert step6['params']['xcal_idx1_offset'] == 9
    assert step6['params']['xcal_std_cutoff'] == 16.0


def test_ts_analysis_interactive_groups_xcal_steps_together():
    """Interactive TS workflow should keep xcal mask+fit in one panel."""
    assert ('get_xcal_mask', 'fit_x_theta') in _TS_PANELS
