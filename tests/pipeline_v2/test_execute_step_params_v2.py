"""
Tests for the user parameters of AnalysisRunner.execute_step (default: YAML).
"""
import pytest
import zarr

from citkid.pipeline_v2 import AnalysisRunner, DataSet

from .synthetic_steps import synthetic_data, ts_steps

YAML_PARAMS = {'xcal_idx0_offset': 3, 'xcal_idx1_offset': 9, 'xcal_std_cutoff': 16.0}


@pytest.fixture(scope='module')
def runner():
    """
    A 'ts' AnalysisRunner with the analysis path run for row 0.

    Returns:
    AR (AnalysisRunner): the runner.
    """
    root = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    DS = DataSet(zarr_path=root, cal_yaml_path='ts', custom_cal_steps=ts_steps(synthetic_data()))
    AR = AnalysisRunner(DS, analysis_yaml_path='ts')
    AR.execute_path(data_idx=0, save=False, verbose=False)
    return AR


def _step(AR, name):
    """
    Return an analysis-path step by name.
    """
    return next(s['task'] for s in AR.path if s['task'].name == name)


def test_execute_step_uses_yaml_params_by_default(runner):
    runner.execute_step(_step(runner, 'get_xcal_mask'), data_idx=0, save=False)

    assert runner._step_state['get_xcal_mask']['user_params'] == YAML_PARAMS


def test_execute_step_explicit_params_replace_yaml(runner):
    params = dict(YAML_PARAMS, xcal_std_cutoff=4.0)
    runner.execute_step(_step(runner, 'get_xcal_mask'), data_idx=0, user_params=params,
                        save=False)

    assert runner._step_state['get_xcal_mask']['user_params'] == params


def test_execute_step_from_yaml_for_a_calibration_step_has_no_params(runner):
    """Steps outside the analysis path (e.g. calibration steps) have no YAML params."""
    cal_step = next(s for s in runner.DS.cal_steps if s.name == 'load_data_f')

    assert runner._get_yaml_params(cal_step) == {}
