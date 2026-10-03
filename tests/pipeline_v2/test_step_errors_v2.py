"""
Tests for tracing errors in (custom) pipeline steps: error context notes,
structure checks, recorded failure causes, and DataSet.check_cal.
"""
import numpy as np
import pytest
import zarr

from citkid.pipeline_v2 import DataSet
from citkid.pipeline_v2 import framework as pf
from citkid.pipeline_v2.framework import plStep

from .synthetic_steps import FRES, NROWS, QR, synthetic_data, ts_steps

DATA = synthetic_data()


def _with_step(name, func):
    """
    Return the synthetic 'ts' loading steps with one step's function replaced.

    Parameters:
    name (str): step to replace.
    func (callable): new function.

    Returns:
    steps (list of plStep): loading steps.
    """
    return [plStep(s.name, func, s.param_names, s.return_names, s.func_type) if s.name == name
            else s for s in ts_steps(DATA)]


def _dataset(steps):
    """
    Build an in-memory 'ts' DataSet from loading steps.

    Parameters:
    steps (list of plStep): loading steps.

    Returns:
    DS (DataSet): the dataset.
    """
    root = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    return DataSet(zarr_path=root, cal_yaml_path='ts', custom_cal_steps=steps)


def _load_data_f_three_outputs(data_idx):
    """Return one output too many (a common mistake)."""
    return DATA[int(data_idx)]['ff'], DATA[int(data_idx)]['zf'], 0


def _load_zt_one_row(data_idx):
    """Return a single row's array from a vectorized step (a common mistake)."""
    return DATA[int(np.atleast_1d(data_idx)[0])]['zt']


def _load_data_f_raises(data_idx):
    """Fail inside the user's function."""
    raise KeyError('missing_file.npy')


# ---------------------------------------------------------------------------
# Errors while running steps
# ---------------------------------------------------------------------------

def test_per_row_failure_cause_is_reported():
    """The real error replaces 'missing rows ... cannot be produced'."""
    DS = _dataset(_with_step('load_data_f', _load_data_f_three_outputs))

    with pytest.raises(ValueError) as info:
        DS.ff[0]

    message = str(info.value)
    assert "Parameter 'ff' is missing rows [0]" in message
    assert "because calibration step 'load_data_f' failed for data_idx 0" in message
    assert "returned 3 output(s) but has 2 return names ['ff', 'zf']" in message
    assert "In calibration step 'load_data_f' (per-row; function _load_data_f_three_outputs" in message


def test_error_in_user_function_keeps_its_traceback():
    DS = _dataset(_with_step('load_data_f', _load_data_f_raises))

    with pytest.raises(ValueError) as info:
        DS.zf[1]

    message = str(info.value)
    assert "KeyError: 'missing_file.npy'" in message
    assert "_load_data_f_raises" in message          # the traceback reaches the user's code
    assert "data_idx = ndarray (1,) int32 = 1" in message


def test_vectorized_step_returning_one_row_is_explained():
    DS = _dataset(_with_step('load_zt', _load_zt_one_row))

    with pytest.raises(ValueError, match="Vectorized step 'load_zt' was called for 2 row") as info:
        DS.zt[[0, 1]]

    notes = "\n".join(getattr(info.value, '__notes__', []))
    assert "In calibration step 'load_zt' (vectorized; function _load_zt_one_row" in notes
    assert "Expected outputs (in order): zt" in notes


def test_global_res_outputs_need_one_entry_per_resonator():
    DS = _dataset(_with_step(
        'load_global_res_data',
        lambda: (FRES[:1], np.full(1, QR), np.zeros(1), np.arange(1))))

    with pytest.raises(ValueError, match=r"Global-res step 'load_global_res_data' output 'fres' "
                                         r"is ndarray \(1,\) float64.*length nrows = 2"):
        DS.fres[0]


@pytest.mark.parametrize('nrows, ok', [(2, True), (2.0, True), (np.int64(2), True),
                                       (2.5, False), (np.array([2]), False)])
def test_nrows_must_be_a_whole_number(nrows, ok):
    DS = _dataset(_with_step(
        'load_global_data', lambda: (FRES, np.full(NROWS, QR), 1e-3, nrows)))
    if ok:
        assert int(DS.nrows) == 2
    else:
        with pytest.raises(ValueError, match="nrows must be a whole number"):
            DS.nrows


def test_describe_value_and_function_location():
    assert pf.describe_value(np.zeros((2, 3), complex)) == "ndarray (2, 3) complex128"
    assert pf.describe_value(np.array([5], dtype=np.int64)) == "ndarray (1,) int64 = 5"
    assert pf.describe_value(3) == "int 3"
    assert pf.describe_value(None) == "None"
    assert pf.describe_value([1, 2]) == "list of 2"
    assert pf.function_location(_load_zt_one_row).startswith("_load_zt_one_row (test_step_errors_v2.py:")


# ---------------------------------------------------------------------------
# DataSet.check_cal
# ---------------------------------------------------------------------------

def _by_step(results):
    """Index check_cal results by step name."""
    return {r['step']: r for r in results}


def test_check_cal_good_steps(capsys):
    DS = _dataset(ts_steps(DATA))

    results = _by_step(DS.check_cal(0))

    for name in ('load_global_data', 'load_global_res_data', 'load_ft', 'load_zt',
                 'load_data_f', 'load_data_g'):
        assert results[name]['status'] == 'ok', results[name]
    assert results['load_data_f']['outputs'] == {
        'ff': 'ndarray (200,) float64', 'zf': 'ndarray (200,) complex128'}
    assert results['rmv_gain_f']['status'] == 'needs analysis'      # needs p_amp
    assert results['center_f']['status'] == 'skipped'
    assert not any(r['status'] == 'failed' for r in results.values())
    report = capsys.readouterr().out
    assert report.startswith('Calibration check for data_idx 0')
    assert 'Summary: ' in report
    assert list(DS.root.array_keys()) == []                       # nothing written


def test_check_cal_reports_failures_and_keeps_checking_other_branches(capsys):
    DS = _dataset(_with_step('load_zt', _load_zt_one_row))

    results = _by_step(DS.check_cal(1, verbose=False))

    assert results['load_zt']['status'] == 'failed'
    assert "Vectorized step 'load_zt'" in results['load_zt']['error']
    assert results['load_zt']['traceback'] is None    # found by the pipeline: message suffices
    assert results['rmv_gain_t']['status'] == 'skipped'
    assert results['rmv_gain_t']['messages'] == ["after failed step 'load_zt'"]
    assert results['load_data_f']['status'] == 'ok'   # other branches still checked
    assert capsys.readouterr().out == ''


def test_check_cal_traceback_reaches_user_code():
    DS = _dataset(_with_step('load_data_f', _load_data_f_raises))

    result = _by_step(DS.check_cal(0, verbose=False))['load_data_f']

    assert result['status'] == 'failed'
    assert result['error'] == "KeyError: 'missing_file.npy'"
    assert '_load_data_f_raises' in result['traceback']
    assert 'framework.py' not in result['traceback']


@pytest.mark.parametrize('func, message', [
    (lambda data_idx: (DATA[int(data_idx)]['ff'], DATA[int(data_idx)]['zf'][:-5]),
     "'ff' (200,) and 'zf' (195,) should have the same shape."),
    (lambda data_idx: (DATA[int(data_idx)]['ff'][::-1], DATA[int(data_idx)]['zf']),
     "'ff' should be sorted in increasing order"),
    (lambda data_idx: (DATA[int(data_idx)]['ff'], np.abs(DATA[int(data_idx)]['zf'])),
     "'zf' should be a complex 1-D array"),
])
def test_check_cal_warns_about_template_shapes(func, message):
    DS = _dataset(_with_step('load_data_f', func))

    result = _by_step(DS.check_cal(0, verbose=False))['load_data_f']

    assert result['status'] == 'warning'
    assert any(message in m for m in result['messages'])


def test_check_cal_uses_analysis_outputs_the_dataset_has():
    """With the gain fit saved for the row, the gain-removal steps are checked too."""
    from citkid.pipeline_v2 import AnalysisRunner
    DS = _dataset(ts_steps(DATA))
    AnalysisRunner(DS, analysis_yaml_path='ts').execute_path(data_idx=0, save=False, verbose=False)

    results = _by_step(DS.check_cal(0, verbose=False))

    assert results['rmv_gain_f']['status'] == 'ok'
    assert results['rmv_gain_f']['inputs']['p_amp'].endswith('(from the dataset)')
    assert results['get_sxx_reduced']['status'] == 'ok'
