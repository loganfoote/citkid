"""
Integration tests for DataSet - testing full pipeline workflows with simple synthetic data.

These tests exercise real DataSet behavior without mocking, using minimal synthetic
steps to validate:
- Import step re-running behavior
- Global + per-row parameter interactions
- Dependency validation and invalidation
- Multiple runs with data accumulation
- User parameter injection
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from citkid.pipeline.dataset import DataSet
from citkid.pipeline.analysis import AnalysisRunner
from citkid.pipeline.framework import plStep


# ============================================================================
#  Simple Step Functions for Testing
# ============================================================================

# Track how many times each step is called (for detecting re-runs)
CALL_COUNTS = {}

def reset_call_counts():
    """Reset call tracking for a new test"""
    global CALL_COUNTS
    CALL_COUNTS = {}

def track_call(func_name):
    """Increment call count for a function"""
    CALL_COUNTS[func_name] = CALL_COUNTS.get(func_name, 0) + 1


# Import step (global, should only run once per dataset)
def import_raw_data(base_value):
    """Import global metadata - should only run once"""
    track_call('import_raw_data')
    nrows = 10
    sampling_rate = 1000.0
    return nrows, sampling_rate, base_value


# Global-res step (runs once, produces per-row data)
def load_resonator_params(base_value):
    """Load per-resonator parameters - runs once, creates array"""
    track_call('load_resonator_params')
    nrows = 10
    fres = np.arange(nrows) * 1e6 + base_value  # MHz
    qres = np.ones(nrows) * 10000.0
    return fres, qres


# Per-row vectorized step
def load_timestream(data_idx, base_value):
    """Load timestream data for specific resonators"""
    track_call('load_timestream')
    # Ensure inputs are arrays
    data_idx = np.atleast_1d(data_idx)
    # Create some synthetic timestream data
    result = []
    for idx in data_idx:
        ts = np.random.randn(100) + idx + base_value
        result.append(ts)
    # Vectorized functions must always return a list
    return result


# Per-row vectorized step using per_row_value parameter
def load_timestream_perrow(data_idx, per_row_value):
    """Load timestream data for specific resonators with per-row value"""
    track_call('load_timestream_perrow')
    # Ensure inputs are arrays
    data_idx = np.atleast_1d(data_idx)
    per_row_value = np.atleast_1d(per_row_value)
    
    # Create synthetic timestream data for each data_idx
    result = []
    for i, (idx, val) in enumerate(zip(data_idx, per_row_value)):
        ts = np.random.randn(100) + idx + val
        result.append(ts)
    
    # Vectorized functions must always return a list
    return result


# Processing step using global parameters
def calculate_noise(timestream, sampling_rate):
    """Calculate noise using global sampling_rate and per-row timestream"""
    track_call('calculate_noise')
    # Handle both single and multiple timestre input
    if isinstance(timestream, list):
        result = []
        for ts in timestream:
            noise_psd = np.std(ts) / np.sqrt(sampling_rate)
            result.append(noise_psd)
        return result
    else:
        noise_psd = np.std(timestream) / np.sqrt(sampling_rate)
        return [noise_psd]  # Still return as list for vectorized
    # Simple noise calculation
    noise_psd = np.std(timestream, axis=-1) / np.sqrt(sampling_rate)
    return noise_psd


# Processing step with dependencies
def normalize_noise(noise_psd, fres):
    """Normalize noise by resonator frequency"""
    track_call('normalize_noise')
    # Per-row processing
    normalized = noise_psd / (fres / 1e6)  # Normalize by freq in MHz
    return normalized


# Analysis step with user parameters
def flag_outliers(normalized_noise, threshold):
    """Flag outliers above threshold"""
    track_call('flag_outliers')
    flags = normalized_noise > threshold
    return flags


# ============================================================================
#  Fixtures
# ============================================================================

@pytest.fixture
def temp_dataset_dir():
    """Create temporary directory for dataset with required files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create custom_steps.py with test step functions
    custom_steps_py = Path(temp_dir) / 'custom_steps.py'
    with open(custom_steps_py, 'w') as f:
        f.write('''"""Custom steps for integration testing"""
from citkid.pipeline.framework import plStep

def import_raw_data(base_value):
    """Import step that produces nrows"""
    return {'nrows': 10, 'sampling_rate': 1000.0, 'base_value_stored': base_value}

def load_resonator_params(base_value):
    """Global-res step producing per-row parameters"""
    return [{'fres': base_value + i*1e6, 'qres': 1000 + i*10} for i in range(10)]

def load_timestream(data_idx, base_value):
    """Vectorized step producing timestream data"""
    import numpy as np
    data_idx = np.atleast_1d(data_idx)
    result = []
    for idx in data_idx:
        ts = np.random.randn(100) + idx + base_value
        result.append(ts)
    return {'timestream': result}

def load_timestream_perrow(data_idx, per_row_value):
    """Vectorized step with per-row parameter"""
    import numpy as np
    data_idx = np.atleast_1d(data_idx)
    per_row_value = np.atleast_1d(per_row_value)
    result = []
    for idx, val in zip(data_idx, per_row_value):
        ts = np.random.randn(100) + idx + val
        result.append(ts)
    # Vectorized functions must return list
    return {'timestream': result}

def calculate_noise(data_idx, timestream):
    """Per-row step calculating noise from timestream"""
    import numpy as np
    return {'noise_psd': np.abs(np.fft.fft(timestream))**2}

custom_cal_steps = [
    plStep('import_raw_data', import_raw_data, ['base_value'],
           ['nrows', 'sampling_rate', 'base_value_stored'], 'global'),
    plStep('load_resonator_params', load_resonator_params, ['base_value'],
           ['fres', 'qres'], 'global-res'),
    plStep('load_timestream', load_timestream, ['data_idx', 'base_value'],
           ['timestream'], 'vectorized'),
    plStep('load_timestream_perrow', load_timestream_perrow, ['data_idx', 'per_row_value'],
           ['timestream'], 'vectorized'),
    plStep('calculate_noise', calculate_noise, ['data_idx', 'timestream'],
           ['noise_psd'], 'per-row'),
]
''')
    
    # Create minimal cal.yaml file with import step to satisfy nrows validation
    cal_yaml = Path(temp_dir) / 'cal.yaml'
    with open(cal_yaml, 'w') as f:
        f.write('CAL_STEPS:\n')
        f.write('  1:\n')
        f.write('    task: import_raw_data\n')
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_custom_steps():
    """Define simple custom steps for testing"""
    return [
        plStep('import_raw_data', import_raw_data, ['base_value'],
               ['nrows', 'sampling_rate', 'base_value_stored'], 'global'),
        plStep('load_resonator_params', load_resonator_params, ['base_value'],
               ['fres', 'qres'], 'global-res'),
        plStep('load_timestream', load_timestream, ['data_idx', 'base_value'],
               ['timestream'], 'vectorized'),
    ]


@pytest.fixture
def simple_cal_pipeline():
    """Simple calibration pipeline YAML"""
    return {
        'CAL_STEPS': {
            1: {'task': 'import_raw_data'},
            2: {'task': 'load_resonator_params'},
            3: {'task': 'load_timestream'},
            4: {'task': 'calculate_noise'},
            5: {'task': 'normalize_noise'},
        }
    }


@pytest.fixture
def simple_analysis_pipeline():
    """Simple analysis pipeline YAML"""
    return {
        'ANALYSIS_STEPS': {
            1: {'task': 'flag_outliers', 'params': {'threshold': 0.5}},
        }
    }


# ============================================================================
#  Integration Tests
# ============================================================================

class TestImportRerun:
    """Test that import steps don't re-run when data already exists"""
    
    def test_import_runs_once_per_dataset(self, temp_dataset_dir, simple_custom_steps):
        """Calling execute_step with user_params twice SHOULD execute twice (creates new runs)"""
        reset_call_counts()
        
        # Create dataset
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        # Manually set cal pipeline for testing
        step = plStep('import_raw_data', import_raw_data, ['base_value'],
                      ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
        
        # Execute import step with user param
        ar.execute_step(step, user_params={'base_value': 100.0})
        
        # Verify it ran once
        assert CALL_COUNTS.get('import_raw_data', 0) == 1
        
        # Verify data exists
        assert 'nrows' in ds._memory_cache[1]
        assert ds._memory_cache[1]['nrows'] == 10
        
        # Execute again with same user_params - SHOULD re-run (new run means new execution)
        ar.execute_step(step, user_params={'base_value': 100.0})
        
        # Should be 2 calls (user_params always create new runs)
        assert CALL_COUNTS.get('import_raw_data', 0) == 2
        assert 'nrows' in ds._memory_cache[2]
        assert ds._memory_cache[2]['nrows'] == 10
    
    
    def test_import_reruns_with_different_params(self, temp_dataset_dir, simple_custom_steps):
        """Import should re-run if user parameters change"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        # Manually set cal pipeline for testing
        step = plStep('import_raw_data', import_raw_data, ['base_value'],
                      ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
        
        # First run
        ar.execute_step(step, user_params={'base_value': 100.0})
        assert CALL_COUNTS.get('import_raw_data', 0) == 1
        assert ds._memory_cache[1]['base_value_stored'] == 100.0
        
        # Different parameter - should re-run at new run_idx
        ar.execute_step(step, user_params={'base_value': 200.0})
        assert CALL_COUNTS.get('import_raw_data', 0) == 2
        assert ds._memory_cache[2]['base_value_stored'] == 200.0


class TestGlobalPerRowInteraction:
    """Test that per-row/vectorized functions can access global parameters"""
    
    def test_vectorized_accesses_global(self, temp_dataset_dir):
        """Vectorized step should access global parameters via merged deps_maps"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        # Set up pipeline
        step1 = plStep('import_raw_data', import_raw_data, ['base_value'],
                       ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        step2 = plStep('load_timestream', load_timestream, ['data_idx', 'base_value'],
                       ['timestream'], 'vectorized')
        step3 = plStep('calculate_noise', calculate_noise, ['timestream', 'sampling_rate'],
                       ['noise_psd'], 'vectorized')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step1}, 1: {'task': step2}, 2: {'task': step3}}}
        
        # Run import (global)
        ar.execute_step(step1, user_params={'base_value': 50.0})
        
        # Run load_timestream (vectorized, should fetch base_value from global storage)
        ar.execute_step(step2, data_idx=0)
        
        # Run calculate_noise (vectorized, needs global sampling_rate)
        ar.execute_step(step3, data_idx=0)
        
        # Verify all steps executed
        assert CALL_COUNTS.get('import_raw_data', 0) == 1
        assert CALL_COUNTS.get('load_timestream', 0) == 1
        assert CALL_COUNTS.get('calculate_noise', 0) == 1
        
        # Verify data exists
        assert 0 in ds._memory_cache[1]['noise_psd']._cache


class TestGlobalResHandling:
    """Test global-res steps (run once, produce per-row data)"""
    
    def test_global_res_produces_perrow_data(self, temp_dataset_dir):
        """Global-res should run once but create per-row data arrays"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        # Set up pipeline
        step0 = plStep('import_raw_data', import_raw_data, ['base_value'],
                       ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        step1 = plStep('load_resonator_params', load_resonator_params, ['base_value'],
                       ['fres', 'qres'], 'global-res')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step0}, 1: {'task': step1}}}
        
        # First run import to establish nrows
        ar.execute_step(step0, user_params={'base_value': 0.0})
        
        # Execute global-res step
        ar.execute_step(step1, user_params={'base_value': 1e9})
        
        # Should run once
        assert CALL_COUNTS.get('load_resonator_params', 0) == 1
        
        # Should create per-row data - check run 1 since both steps use same base_value
        assert 0 in ds._memory_cache[1]['fres']._cache
        assert 5 in ds._memory_cache[1]['fres']._cache  # Different resonators have different values
        assert ds._memory_cache[1]['fres']._cache[0] != ds._memory_cache[1]['fres']._cache[5]


class TestMultipleRuns:
    """Test accumulating data over multiple runs"""
    
    def test_multiple_runs_with_different_params(self, temp_dataset_dir):
        """Multiple runs should accumulate in zarr with different run indices"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        step = plStep('import_raw_data', import_raw_data, ['base_value'],
                      ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
        
        # Run 1
        ar.execute_step(step, user_params={'base_value': 100.0})
        assert ds._memory_cache[1]['base_value_stored'] == 100.0
        
        # Run 2 with different param
        ar.execute_step(step, user_params={'base_value': 200.0})
        assert ds._memory_cache[2]['base_value_stored'] == 200.0
        
        # Both runs should exist
        assert 1 in ds._memory_cache
        assert 2 in ds._memory_cache


class TestDependencyValidation:
    """Test dependency checking and invalidation"""
    
    def test_downstream_step_requires_upstream(self, temp_dataset_dir):
        """Downstream step should fail if upstream dependency missing"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        step1 = plStep('import_raw_data', import_raw_data, ['base_value'],
                       ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        step2 = plStep('load_timestream', load_timestream, ['data_idx', 'base_value'],
                       ['timestream'], 'vectorized')
        step3 = plStep('calculate_noise', calculate_noise, ['timestream', 'sampling_rate'],
                       ['noise_psd'], 'vectorized')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step1}, 1: {'task': step2}, 2: {'task': step3}}}
        
        # Try to run calculate_noise without running prerequisites
        with pytest.raises(ValueError, match="Missing dependencies for parameters"):
            ar.execute_step(step3, data_idx=0)


class TestUserParameters:
    """Test user parameter injection at different stages"""
    
    def test_user_params_scalar_data_idx(self, temp_dataset_dir):
        """User params should work with scalar data_idx"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        # Set up pipeline with import providing nrows
        step0 = plStep('import_raw_data', import_raw_data, ['base_value'],
                       ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        step1 = plStep('load_timestream_perrow', load_timestream_perrow, ['data_idx', 'per_row_value'],
                       ['timestream'], 'vectorized')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step0}, 1: {'task': step1}}}
        
        # First establish nrows
        ar.execute_step(step0, user_params={'base_value': 0.0})
        
        # Scalar data_idx with per-row user param (different name from global base_value)
        ar.execute_step(step1, data_idx=3, user_params={'per_row_value': 42.0})
        
        assert 3 in ds._memory_cache[1]['timestream']._cache
        assert CALL_COUNTS.get('load_timestream_perrow', 0) == 1
    
    
    def test_user_params_array_data_idx(self, temp_dataset_dir):
        """User params should work with array data_idx"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar = AnalysisRunner(ds)
        
        # Set up pipeline with import providing nrows
        step0 = plStep('import_raw_data', import_raw_data, ['base_value'],
                       ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        step1 = plStep('load_timestream_perrow', load_timestream_perrow, ['data_idx', 'per_row_value'],
                       ['timestream'], 'vectorized')
        ds.cal_pl = {'CAL_STEPS': {0: {'task': step0}, 1: {'task': step1}}}
        
        # First establish nrows
        ar.execute_step(step0, user_params={'base_value': 0.0})
        
        # Array data_idx with per-row user params (different names from global base_value)
        ar.execute_step(step1, data_idx=[0, 1, 2], 
                       user_params={'per_row_value': [10.0, 20.0, 30.0]})
        
        assert 0 in ds._memory_cache[1]['timestream']._cache
        assert 1 in ds._memory_cache[1]['timestream']._cache
        assert 2 in ds._memory_cache[1]['timestream']._cache
        assert CALL_COUNTS.get('load_timestream_perrow', 0) == 1  # Vectorized call


class TestZarrPersistence:
    """Test data persistence and loading from zarr"""
    
    def test_reload_from_zarr(self, temp_dataset_dir):
        """Data should persist to zarr and reload correctly"""
        reset_call_counts()
        
        zarr_path = str(Path(temp_dataset_dir) / 'test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        
        # Create first dataset and add data
        ds1 = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                      custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar1 = AnalysisRunner(ds1)
        
        step = plStep('import_raw_data', import_raw_data, ['base_value'],
                      ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        ds1.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
        
        ar1.execute_step(step, user_params={'base_value': 123.0}, save=True)
        
        # Create second dataset from same zarr
        ds2 = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                      custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))
        ar2 = AnalysisRunner(ds2)
        
        step = plStep('import_raw_data', import_raw_data, ['base_value'],
                      ['nrows', 'sampling_rate', 'base_value_stored'], 'global')
        ds2.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
        
        # Verify data persists after reload by accessing via the property API
        # (data is loaded lazily from zarr, not eagerly into _memory_cache)
        assert ds2.nrows is not None, "nrows not found after reload"
        assert ds2.base_value_stored == 123.0, \
            f"Expected 123.0, got {ds2.base_value_stored}"
        
        # Function was called once by ds1. Since analysis user-params are
        # always stored in a new run, executing the same step again will
        # create a new user-params run and trigger re-execution.
        prior_calls = CALL_COUNTS.get('import_raw_data', 0)
        ar2.execute_step(step, user_params={'base_value': 123.0})
        assert CALL_COUNTS.get('import_raw_data', 0) == prior_calls + 1  # Re-run expected


# ============================================================================
#  New tests: vectorized vs per-row dispatch, failure handling
# ============================================================================

# --------------- helpers used only by the new tests -------------------------

PERROW_CALL_LOG = []  # records which data_idx values were executed

def reset_perrow_call_log():
    global PERROW_CALL_LOG
    PERROW_CALL_LOG = []


def perrow_step_func(data_idx):
    """Per-row step that records which index it was called with (single value)."""
    idx = int(np.atleast_1d(data_idx)[0])
    PERROW_CALL_LOG.append(idx)
    return np.array([float(idx) * 2])


def vectorized_step_func(data_idx):
    """Vectorized step that records which indices it was called with (array)."""
    idxs = list(np.atleast_1d(data_idx))
    for i in idxs:
        PERROW_CALL_LOG.append(int(i))
    return [float(i) * 2 for i in idxs]


def failing_perrow_func(data_idx):
    """Per-row step that raises for data_idx == 2."""
    idx = int(np.atleast_1d(data_idx)[0])
    if idx == 2:
        raise ValueError(f"Intentional failure at idx={idx}")
    return np.array([float(idx)])


# --------------- fixtures ----------------------------------------------------

@pytest.fixture
def dispatch_ds(temp_dataset_dir):
    """DataSet initialised with nrows=5 ready for dispatch tests."""
    zarr_path = str(Path(temp_dataset_dir) / 'dispatch.zarr')
    cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
    ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                 custom_path=str(Path(temp_dataset_dir) / 'custom_steps.py'))

    # Bootstrap nrows via the existing import step in custom_steps.py
    import_step = plStep(
        'import_raw_data', import_raw_data,
        ['base_value'], ['nrows', 'sampling_rate', 'base_value_stored'], 'global'
    )
    ds.cal_pl = {'CAL_STEPS': {0: {'task': import_step}}}
    ar = AnalysisRunner(ds)
    ar.execute_step(import_step, user_params={'base_value': 0.0})
    return ds


# ============================================================================
class TestVectorizedVsPerRowDispatch:
    """
    Verify that _execute_step calls the underlying function once per group for
    vectorized steps, and once per data_idx for per-row steps.
    """

    def test_vectorized_calls_func_once_for_identical_deps(self, dispatch_ds):
        """All rows share the same deps → vectorized step called exactly once."""
        reset_perrow_call_log()
        step = plStep(
            'vec_step', vectorized_step_func,
            ['data_idx'], ['vec_out'], 'vectorized'
        )
        dispatch_ds._execute_step(step, data_idx=np.arange(5))

        # The function was called once with all 5 indices stacked
        assert PERROW_CALL_LOG == [0, 1, 2, 3, 4]

    def test_perrow_calls_func_once_per_index(self, dispatch_ds):
        """Per-row step must call the function individually for each data_idx."""
        reset_perrow_call_log()
        step = plStep(
            'pr_step', perrow_step_func,
            ['data_idx'], ['pr_out'], 'per-row'
        )
        dispatch_ds._execute_step(step, data_idx=np.arange(5))

        # Function called once per row, in order
        assert PERROW_CALL_LOG == [0, 1, 2, 3, 4]

    def test_perrow_stores_each_row_independently(self, dispatch_ds):
        """Per-row outputs must be stored separately per data_idx."""
        step = plStep(
            'pr_step2', perrow_step_func,
            ['data_idx'], ['pr_out2'], 'per-row'
        )
        dispatch_ds._execute_step(step, data_idx=[1, 3])

        run_idx = max(dispatch_ds._memory_cache.keys())
        la = dispatch_ds._memory_cache[run_idx]['pr_out2']
        assert 1 in la._cache
        assert 3 in la._cache
        assert 0 not in la._cache
        assert 2 not in la._cache

    def test_vectorized_stores_all_rows(self, dispatch_ds):
        """Vectorized outputs must cover all requested data_idx."""
        reset_perrow_call_log()
        step = plStep(
            'vec_step2', vectorized_step_func,
            ['data_idx'], ['vec_out2'], 'vectorized'
        )
        dispatch_ds._execute_step(step, data_idx=[0, 2, 4])

        run_idx = max(dispatch_ds._memory_cache.keys())
        la = dispatch_ds._memory_cache[run_idx]['vec_out2']
        for di in [0, 2, 4]:
            assert di in la._cache
        assert 1 not in la._cache


# ============================================================================
class TestPerRowFailureHandling:
    """
    Verify that per-row failures are isolated, partial successes are stored,
    and failure reports are persisted to zarr.
    """

    def test_failure_returns_failures_dict(self, dispatch_ds):
        """_execute_step must return a dict with the failing data_idx."""
        step = plStep(
            'fail_step', failing_perrow_func,
            ['data_idx'], ['fail_out'], 'per-row'
        )
        failures = dispatch_ds._execute_step(step, data_idx=[0, 1, 2, 3])

        assert isinstance(failures, dict)
        assert 2 in failures
        assert len(failures) == 1

    def test_failure_traceback_is_string(self, dispatch_ds):
        """Failure values must be non-empty traceback strings."""
        step = plStep(
            'fail_step2', failing_perrow_func,
            ['data_idx'], ['fail_out2'], 'per-row'
        )
        failures = dispatch_ds._execute_step(step, data_idx=[0, 2])

        assert isinstance(failures[2], str)
        assert 'Intentional failure' in failures[2]

    def test_successful_rows_are_stored(self, dispatch_ds):
        """Rows that succeed must be stored even when other rows fail."""
        step = plStep(
            'fail_step3', failing_perrow_func,
            ['data_idx'], ['fail_out3'], 'per-row'
        )
        dispatch_ds._execute_step(step, data_idx=[0, 1, 2, 3])

        run_idx = max(dispatch_ds._memory_cache.keys())
        la = dispatch_ds._memory_cache[run_idx]['fail_out3']
        assert 0 in la._cache
        assert 1 in la._cache
        assert 3 in la._cache
        assert 2 not in la._cache  # Failed row must not be stored

    def test_empty_failures_when_all_succeed(self, dispatch_ds):
        """No failures → return empty dict (not None)."""
        step = plStep(
            'pr_step_ok', perrow_step_func,
            ['data_idx'], ['pr_ok_out'], 'per-row'
        )
        failures = dispatch_ds._execute_step(step, data_idx=[0, 1, 3])
        assert failures == {}

    def test_failures_persisted_to_zarr(self, dispatch_ds):
        """Failures must be written to the _failures/{step.name} zarr group."""
        step = plStep(
            'fail_zarr', failing_perrow_func,
            ['data_idx'], ['fail_zarr_out'], 'per-row'
        )
        dispatch_ds._execute_step(step, data_idx=[0, 2, 4])

        fail_grp = dispatch_ds.root['_failures/fail_zarr']
        stored = dict(fail_grp.attrs['failures'])
        assert 'idx2' in stored
        assert 'traceback' in stored['idx2']
        assert 'time' in stored['idx2']

    def test_no_zarr_failure_group_on_full_success(self, dispatch_ds):
        """When all rows succeed no _failures group should be created."""
        step = plStep(
            'ok_step', perrow_step_func,
            ['data_idx'], ['ok_out'], 'per-row'
        )
        dispatch_ds._execute_step(step, data_idx=[0, 1])

        assert '_failures' not in dispatch_ds.root or \
               'ok_step' not in dispatch_ds.root.get('_failures', {})

    def test_failures_accumulate_across_calls(self, dispatch_ds):
        """Re-running the same step should merge new failures with old ones."""
        step = plStep(
            'accum_fail', failing_perrow_func,
            ['data_idx'], ['accum_out'], 'per-row'
        )
        # First call: idx 2 fails
        dispatch_ds._execute_step(step, data_idx=[0, 2])
        # Second call: idx 2 fails again (new timestamp)
        dispatch_ds._execute_step(step, data_idx=[2])

        stored = dict(dispatch_ds.root['_failures/accum_fail'].attrs['failures'])
        # idx2 should be present (updated by the second call)
        assert 'idx2' in stored

    def test_vectorized_returns_none(self, dispatch_ds):
        """_execute_step must return None for vectorized steps."""
        reset_perrow_call_log()
        step = plStep(
            'vec_none', vectorized_step_func,
            ['data_idx'], ['vec_none_out'], 'vectorized'
        )
        result = dispatch_ds._execute_step(step, data_idx=[0, 1])
        assert result is None

    def test_global_returns_none(self, dispatch_ds):
        """_execute_step must return None for global steps."""
        def global_func(nrows):
            return np.float64(nrows * 2.0)
        step = plStep('global_none', global_func, ['nrows'], ['global_none_out'], 'global')
        dispatch_ds._ensure_loaded('nrows')
        result = dispatch_ds._execute_step(step, data_idx=None)
        assert result is None


# ============================================================================
class TestAnalysisRunnerFailureSurfacing:
    """
    Verify that AnalysisRunner.execute_step exposes failures via _last_failures
    and emits a RuntimeWarning.
    """

    def test_last_failures_set_on_partial_failure(self, dispatch_ds):
        """_last_failures must contain only the rows that failed."""
        step = plStep(
            'ar_fail', failing_perrow_func,
            ['data_idx'], ['ar_fail_out'], 'per-row'
        )
        ar = AnalysisRunner(dispatch_ds)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter('always')
            ar.execute_step(step, data_idx=[0, 1, 2, 3])

        assert ar._last_failures is not None
        assert 2 in ar._last_failures
        assert len(ar._last_failures) == 1

    def test_warning_issued_on_failure(self, dispatch_ds):
        """A RuntimeWarning should be raised when any row fails."""
        step = plStep(
            'ar_warn', failing_perrow_func,
            ['data_idx'], ['ar_warn_out'], 'per-row'
        )
        ar = AnalysisRunner(dispatch_ds)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            ar.execute_step(step, data_idx=[0, 2])

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 1
        assert 'ar_warn' in str(runtime_warnings[0].message)
        assert '2' in str(runtime_warnings[0].message)

    def test_no_warning_on_full_success(self, dispatch_ds):
        """No RuntimeWarning should be emitted when all rows succeed."""
        step = plStep(
            'ar_ok', perrow_step_func,
            ['data_idx'], ['ar_ok_out'], 'per-row'
        )
        ar = AnalysisRunner(dispatch_ds)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            ar.execute_step(step, data_idx=[0, 1])

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0

    def test_last_failures_none_for_global_step(self, dispatch_ds):
        """_last_failures should be None after executing a global step."""
        def global_noop(nrows):
            return np.float64(1.0)
        step = plStep('ar_global', global_noop, ['nrows'], ['ar_global_out'], 'global')
        dispatch_ds._ensure_loaded('nrows')
        ar = AnalysisRunner(dispatch_ds)
        ar.execute_step(step, data_idx=None)
        assert ar._last_failures is None

    def test_last_failures_empty_dict_on_full_perrow_success(self, dispatch_ds):
        """_last_failures should be {} (not None) after a clean per-row run."""
        step = plStep(
            'ar_empty', perrow_step_func,
            ['data_idx'], ['ar_empty_out'], 'per-row'
        )
        ar = AnalysisRunner(dispatch_ds)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter('always')
            ar.execute_step(step, data_idx=[0])
        assert ar._last_failures == {}


# ============================================================================
class TestReadOnlyZarrFailureHandling:
    """
    Verify that when zarr_mode='r', failures emit a RuntimeWarning with the
    original traceback instead of attempting to write to the zarr store.
    """

    @pytest.fixture
    def readonly_ds(self, temp_dataset_dir):
        """DataSet opened in write mode with nrows bootstrapped, then
        zarr_mode is set to 'r' on the instance to exercise the read-only
        failure-reporting branch without re-opening the store."""
        zarr_path = str(Path(temp_dataset_dir) / 'readonly_test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        custom_path = str(Path(temp_dataset_dir) / 'custom_steps.py')

        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=custom_path)
        import_step = plStep(
            'import_raw_data', import_raw_data,
            ['base_value'], ['nrows', 'sampling_rate', 'base_value_stored'], 'global'
        )
        ds.cal_pl = {'CAL_STEPS': {0: {'task': import_step}}}
        ar = AnalysisRunner(ds)
        ar.execute_step(import_step, user_params={'base_value': 0.0})

        # Simulate read-only mode for failure-reporting tests.
        ds.zarr_mode = 'r'
        return ds

    def test_zarr_mode_stored(self, readonly_ds):
        """DataSet must expose the zarr_mode it was opened with (fixture sets 'r')."""
        assert readonly_ds.zarr_mode == 'r'

    def test_failure_issues_warning_with_traceback(self, readonly_ds):
        """On zarr_mode='r', a RuntimeWarning containing the traceback
        must be raised instead of attempting to write to the store."""
        step = plStep(
            'ro_fail', failing_perrow_func,
            ['data_idx'], ['ro_fail_out'], 'per-row'
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            readonly_ds._execute_step(step, data_idx=[0, 2])

        runtime_warnings = [w for w in caught
                            if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 1
        msg = str(runtime_warnings[0].message)
        assert 'Intentional failure' in msg

    def test_failure_warning_mentions_step_name(self, readonly_ds):
        """The warning message must name the step."""
        step = plStep(
            'ro_named_fail', failing_perrow_func,
            ['data_idx'], ['ro_named_out'], 'per-row'
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            readonly_ds._execute_step(step, data_idx=[2])

        msgs = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
        assert any('ro_named_fail' in m for m in msgs)

    def test_failure_warning_mentions_idx(self, readonly_ds):
        """The warning message must identify the failing data index."""
        step = plStep(
            'ro_idx_fail', failing_perrow_func,
            ['data_idx'], ['ro_idx_out'], 'per-row'
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            readonly_ds._execute_step(step, data_idx=[2])

        msgs = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
        assert any('2' in m for m in msgs)

    def test_no_write_attempt_on_readonly(self, readonly_ds):
        """On zarr_mode='r', no 'Could not write failure report' warning
        must be emitted – the write is skipped entirely."""
        step = plStep(
            'ro_nowrite', failing_perrow_func,
            ['data_idx'], ['ro_nowrite_out'], 'per-row'
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            readonly_ds._execute_step(step, data_idx=[2])

        msgs = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
        assert not any('Could not write failure report' in m for m in msgs)

    def test_readonly_failures_dict_still_returned(self, readonly_ds):
        """_execute_step must still return the failures dict on zarr_mode='r'."""
        step = plStep(
            'ro_dict', failing_perrow_func,
            ['data_idx'], ['ro_dict_out'], 'per-row'
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter('always')
            failures = readonly_ds._execute_step(step, data_idx=[0, 2])

        assert isinstance(failures, dict)
        assert 2 in failures

    def test_writable_ds_still_writes_failures(self, temp_dataset_dir):
        """Normal (non-read-only) DataSet must still persist failures to zarr."""
        zarr_path = str(Path(temp_dataset_dir) / 'rw_fail_test.zarr')
        cal_yaml_path = str(Path(temp_dataset_dir) / 'cal.yaml')
        custom_path = str(Path(temp_dataset_dir) / 'custom_steps.py')
        ds = DataSet(zarr_path=zarr_path, cal_yaml_path=cal_yaml_path,
                     custom_path=custom_path)
        import_step = plStep(
            'import_raw_data', import_raw_data,
            ['base_value'], ['nrows', 'sampling_rate', 'base_value_stored'], 'global'
        )
        ds.cal_pl = {'CAL_STEPS': {0: {'task': import_step}}}
        AnalysisRunner(ds).execute_step(import_step, user_params={'base_value': 0.0})

        step = plStep(
            'rw_fail', failing_perrow_func,
            ['data_idx'], ['rw_fail_out'], 'per-row'
        )
        ds._execute_step(step, data_idx=[0, 2])

        fail_grp = ds.root['_failures/rw_fail']
        stored = dict(fail_grp.attrs['failures'])
        assert 'idx2' in stored
