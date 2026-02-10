import pytest
import numpy as np 
from citkid.pipeline import framework as pf 
from citkid.pipeline import dataset as dataset

# Dummy DataSet class 
class DummyDS():
    def __init__(self):
        self.cal_pl = {}
        self.nrows = 10
        self.execute_path = lambda path, rows: None

class DummyDSWithExecute(dataset.DataSet):
    def __init__(self, cal_pl):
        self.cal_pl = cal_pl
        self.nrows = 10

class DummyDSWithCounter(DummyDS):
    def __init__(self, cal_pl):
        super().__init__()
        self.cal_pl = cal_pl
        self._execute_calls = []
    def execute_path(self, path, rows):
        self._execute_calls.append((path, list(rows)))

################################################################################
#################################### Steps #####################################
################################################################################
# __init__
@pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
    ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
    ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
    ('step3', lambda: 42, [], ['y'], 'global'),
    ('step4', lambda: np.arange(10), [], ['z'], 'global-res'),
])
def test_steps_init_basic(name, func, param_names, return_names, func_type):
    """Test basic __init__ functionality with all func_types."""
    step = pf.plStep(name, func, param_names, return_names, func_type)
    
    # Check that all attributes are set correctly
    assert step.name == name
    assert step.func == func
    assert step.param_names == param_names
    assert step.return_names == return_names
    assert step.func_type == func_type

def test_steps_init_default_func_type():
    """Test that func_type defaults to 'per-row' when not specified."""
    step = pf.plStep('test_step', lambda x: x, ['x'], ['y'])
    assert step.func_type == 'per-row'

def test_steps_init_func_callable():
    """Test that the function is callable and functions correctly."""
    func = lambda x: x + 10
    step = pf.plStep('test', func, ['x'], ['y'], 'per-row')
    
    assert step.func == func
    assert step.func(5) == 15
    assert step.func(100) == 110

@pytest.mark.parametrize("param_names,return_names", [
    ([], []),  # Both empty
    (['x'], []),  # Empty return_names
    ([], ['y']),  # Empty param_names
    (['a', 'b', 'c'], ['x', 'y', 'z']),  # Multiple parameters
])
def test_steps_init_various_list_lengths(param_names, return_names):
    """Test __init__ with various lengths of param_names and return_names."""
    step = pf.plStep('test', lambda: None, param_names, return_names, 'global')
    assert step.param_names == param_names
    assert step.return_names == return_names

def test_steps_init_lists_are_copied():
    """Test that param_names and return_names are copied, not referenced."""
    original_params = ['x', 'y', 'z']
    original_returns = ['a', 'b']
    
    step = pf.plStep('test', lambda: None, original_params, original_returns, 'global')
    
    # Verify the lists contain the same values
    assert step.param_names == original_params
    assert step.return_names == original_returns
    
    # Modify the step's lists
    step.param_names[0] = 'modified_x'
    step.param_names.append('new_param')
    step.return_names[1] = 'modified_b'
    step.return_names.append('new_return')
    
    # Verify the original lists are unchanged
    assert original_params == ['x', 'y', 'z']
    assert original_returns == ['a', 'b']
    
    # Verify the step's lists were modified
    assert 'modified_x' in step.param_names
    assert 'new_param' in step.param_names
    assert 'modified_b' in step.return_names
    assert 'new_return' in step.return_names

def test_steps_init_lists_not_same_object():
    """Test that the stored lists are different objects from the input lists."""
    original_params = ['x', 'y']
    original_returns = ['a', 'b']
    
    step = pf.plStep('test', lambda: None, original_params, original_returns, 'global')
    
    # Lists should not be the same object
    assert step.param_names is not original_params
    assert step.return_names is not original_returns

@pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
    (123, lambda x: x+1, ['x'], ['y'], 'per-row'),  # name not str
    (None, lambda x: x+1, ['x'], ['y'], 'per-row'),  # name is None
    ('step', "not_a_function", ['x'], ['y'], 'vectorized'), # func not callable
    ('step', None, ['x'], ['y'], 'vectorized'), # func is None
    ('step', "lambda x: x", ['x'], ['y'], 'vectorized'), # func is string
    ('step', lambda: 42, 'not_a_list', ['y'], 'global'), # param_names not list
    ('step', lambda: 42, ('x', 'y'), ['y'], 'global'), # param_names is tuple
    ('step', lambda: 42, None, ['y'], 'global'), # param_names is None
    ('step', lambda: 42, ['a', 1], ['y'], 'global'),  # param_names contains int
    ('step', lambda: 42, ['a', None], ['y'], 'global'),  # param_names contains None
    ('step', lambda: 42, ['x'], 'not_a_list', 'global'), # return_names not list
    ('step', lambda: 42, ['x'], ('y',), 'global'), # return_names is tuple
    ('step', lambda: 42, ['x'], None, 'global'), # return_names is None
    ('step', lambda: 42, ['x'], ['a', 1], 'global'), # return_names contains int
    ('step', lambda: 42, ['x'], ['a', None], 'global'), # return_names contains None
    ('step', lambda x: x+1, ['x'], ['y'], 'invalid_type'), # invalid func_type
    ('step', lambda x: x+1, ['x'], ['y'], 'per_row'), # func_type with underscore instead of hyphen
    ('step', lambda x: x+1, ['x'], ['y'], 'Global'), # func_type wrong case
    ('step', lambda x: x+1, ['x'], ['y'], None), # func_type is None
    ('step', lambda x: x+1, ['x'], ['y'], ''), # func_type is empty string
])
def test_steps_init_invalid_inputs(name, func, param_names, return_names, func_type):
    """Test that __init__ raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        pf.plStep(name, func, param_names, return_names, func_type)

def test_steps_init_empty_string_name_is_valid():
    """Test that empty string for name is actually valid (it's just a string)."""
    # This test documents that empty strings ARE valid names
    step = pf.plStep('', lambda: None, [], [], 'global')
    assert step.name == ''

def test_steps_init_empty_string_in_lists_is_valid():
    """Test that empty strings in param_names and return_names are valid."""
    # This test documents that empty strings ARE valid in the lists
    step = pf.plStep('test', lambda: None, ['', 'x'], ['', 'y'], 'global')
    assert step.param_names == ['', 'x']
    assert step.return_names == ['', 'y']

# __repr__, __str__
@pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
    ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
    ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
    ('step3', lambda: 42, [], ['y'], 'global'),
    ('step4', lambda: np.arange(10), [], ['z'], 'global-res'),
])
def test_steps_repr_basic(name, func, param_names, return_names, func_type):
    """Test __repr__ returns correct format for all func_types."""
    step = pf.plStep(name, func, param_names, return_names, func_type)
    
    expected_repr = f"Pipeline Step: {name}, Function: {func.__module__}.{func.__name__}"
    assert repr(step) == expected_repr
    
    # Verify it's a single line (no newlines)
    assert '\n' not in repr(step)
    
    # Verify it contains the key components
    assert name in repr(step)
    assert func.__name__ in repr(step)

@pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
    ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
    ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
    ('step3', lambda: 42, [], ['y'], 'global'),
    ('step4', lambda: np.arange(10), [], ['z'], 'global-res'),
])
def test_steps_str_basic(name, func, param_names, return_names, func_type):
    """Test __str__ returns correct detailed format for all func_types."""
    step = pf.plStep(name, func, param_names, return_names, func_type)
    
    expected_str = f"Pipeline Step: {name}"
    expected_str += f"\n\tFunction: {func.__module__}.{func.__name__}"
    expected_str += f"\n\tInput Parameters: {param_names}"
    expected_str += f"\n\tOutput Parameters: {return_names}"
    expected_str += f"\n\tFunction Type: {func_type}"
    assert str(step) == expected_str
    
    # Verify it's multi-line
    assert '\n' in str(step)
    
    # Verify it contains all required components
    assert name in str(step)
    assert func.__name__ in str(step)
    assert str(param_names) in str(step)
    assert str(return_names) in str(step)
    assert func_type in str(step)

def test_steps_repr_with_special_characters():
    """Test __repr__ handles names with special characters."""
    special_names = ['step-1', 'step_2', 'step.3', 'step:4', 'step with spaces']
    
    for name in special_names:
        step = pf.plStep(name, lambda: None, [], [], 'global')
        assert name in repr(step)
        assert repr(step).startswith('Pipeline Step:')

def test_steps_str_with_special_characters():
    """Test __str__ handles names with special characters."""
    special_names = ['step-1', 'step_2', 'step.3', 'step:4', 'step with spaces']
    
    for name in special_names:
        step = pf.plStep(name, lambda: None, [], [], 'global')
        assert name in str(step)
        assert 'Pipeline Step:' in str(step)

def test_steps_repr_with_empty_lists():
    """Test __repr__ with empty param_names and return_names."""
    step = pf.plStep('test_step', lambda: 42, [], [], 'global')
    result = repr(step)
    
    assert 'test_step' in result
    assert 'lambda' in result
    assert isinstance(result, str)

def test_steps_str_with_empty_lists():
    """Test __str__ with empty param_names and return_names."""
    step = pf.plStep('test_step', lambda: 42, [], [], 'global')
    result = str(step)
    
    assert 'test_step' in result
    assert 'Input Parameters: []' in result
    assert 'Output Parameters: []' in result
    assert 'global' in result

def test_steps_repr_with_multiple_params():
    """Test __repr__ with multiple parameters."""
    step = pf.plStep('multi_param', lambda a, b, c: a+b+c, 
                     ['a', 'b', 'c'], ['result1', 'result2'], 'per-row')
    result = repr(step)
    
    assert 'multi_param' in result
    # __repr__ doesn't show params, just name and function
    assert 'Pipeline Step:' in result

def test_steps_str_with_multiple_params():
    """Test __str__ with multiple parameters."""
    step = pf.plStep('multi_param', lambda a, b, c: a+b+c, 
                     ['a', 'b', 'c'], ['result1', 'result2'], 'per-row')
    result = str(step)
    
    assert 'multi_param' in result
    assert "['a', 'b', 'c']" in result
    assert "['result1', 'result2']" in result
    assert 'per-row' in result

def test_steps_repr_vs_str_difference():
    """Test that __repr__ is concise and __str__ is detailed."""
    step = pf.plStep('test', lambda x: x, ['x'], ['y'], 'per-row')
    
    repr_result = repr(step)
    str_result = str(step)
    
    # __repr__ should be shorter (single line)
    assert len(repr_result) < len(str_result)
    
    # __str__ should have newlines and tabs
    assert '\n' not in repr_result
    assert '\n' in str_result
    assert '\t' in str_result

def test_steps_repr_str_consistency():
    """Test that repr and str both contain the step name."""
    test_names = ['step1', 'my_step', 'TEST', '123', '']
    
    for name in test_names:
        step = pf.plStep(name, lambda: None, [], [], 'global')
        assert name in repr(step)
        assert name in str(step)

def test_steps_repr_str_with_named_function():
    """Test __repr__ and __str__ with a named function (not lambda)."""
    def my_function(x):
        return x * 2
    
    step = pf.plStep('named_func_step', my_function, ['x'], ['y'], 'per-row')
    
    # Check __repr__
    repr_result = repr(step)
    assert 'named_func_step' in repr_result
    assert 'my_function' in repr_result  # Named function should appear
    
    # Check __str__
    str_result = str(step)
    assert 'named_func_step' in str_result
    assert 'my_function' in str_result
    assert "['x']" in str_result
    assert "['y']" in str_result

################################################################################
################################### _run #######################################
################################################################################
# Input validation tests
def test_run_params_must_be_list():
    """Test that params must be a list."""
    step = pf.plStep('test', lambda: 42, [], ['result'], 'global')
    
    with pytest.raises(ValueError, match="params must be a list"):
        step._run("not a list", [])
    
    with pytest.raises(ValueError, match="params must be a list"):
        step._run(None, [])
    
    with pytest.raises(ValueError, match="params must be a list"):
        step._run((1, 2), [])

def test_run_param_is_global_must_be_list():
    """Test that param_is_global must be a list."""
    step = pf.plStep('test', lambda: 42, [], ['result'], 'global')
    
    with pytest.raises(ValueError, match="param_is_global must be a list"):
        step._run([], "not a list")
    
    with pytest.raises(ValueError, match="param_is_global must be a list"):
        step._run([], None)
    
    with pytest.raises(ValueError, match="param_is_global must be a list"):
        step._run([], (True, False))

def test_run_params_length_must_match():
    """Test that params length must match param_names length."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'global')
    
    # Too few params
    with pytest.raises(ValueError, match="Length of params does not match"):
        step._run([], [True])
    
    # Too many params
    with pytest.raises(ValueError, match="Length of params does not match"):
        step._run([1, 2], [True, True])

def test_run_param_is_global_length_must_match():
    """Test that param_is_global length must match param_names length."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'global')
    
    # Too few
    with pytest.raises(ValueError, match="Length of param_is_global does not match"):
        step._run([1], [])
    
    # Too many
    with pytest.raises(ValueError, match="Length of param_is_global does not match"):
        step._run([1], [True, True])

def test_run_param_is_global_must_contain_booleans():
    """Test that param_is_global must contain only booleans."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'global')
    
    with pytest.raises(ValueError, match="All elements of param_is_global must be booleans"):
        step._run([1], [1])
    
    with pytest.raises(ValueError, match="All elements of param_is_global must be booleans"):
        step._run([1], ["True"])
    
    with pytest.raises(ValueError, match="All elements of param_is_global must be booleans"):
        step._run([1], [None])

# Global function tests
def test_run_global_no_params():
    """Test global function with no parameters."""
    step = pf.plStep('test', lambda: 42, [], ['result'], 'global')
    result = step._run([], [])
    
    assert result == {'result': 42}

def test_run_global_single_param():
    """Test global function with single parameter."""
    step = pf.plStep('test', lambda x: x * 2, ['x'], ['result'], 'global')
    result = step._run([5], [True])
    
    assert result == {'result': 10}

def test_run_global_multiple_params():
    """Test global function with multiple parameters."""
    step = pf.plStep('test', lambda x, y, z: x + y + z, 
                     ['x', 'y', 'z'], ['result'], 'global')
    result = step._run([1, 2, 3], [True, True, True])
    
    assert result == {'result': 6}

def test_run_global_multiple_returns():
    """Test global function with multiple return values."""
    step = pf.plStep('test', lambda x: (x * 2, x * 3), 
                     ['x'], ['result1', 'result2'], 'global')
    result = step._run([5], [True])
    
    assert result == {'result1': 10, 'result2': 15}

def test_run_global_all_params_must_be_global():
    """Test that global functions require all params to be global."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'global')
    
    with pytest.raises(ValueError, match="All parameters must be global"):
        step._run([[1, 2, 3]], [False])

def test_run_global_with_array_param():
    """Test global function can accept array as global parameter."""
    step = pf.plStep('test', lambda x: np.sum(x), ['x'], ['result'], 'global')
    result = step._run([np.array([1, 2, 3, 4])], [True])
    
    assert result == {'result': 10}

# Global-res function tests
def test_run_global_res_no_params():
    """Test global-res function with no parameters."""
    step = pf.plStep('test', lambda: np.array([1, 2, 3]), [], ['result'], 'global-res')
    result = step._run([], [])
    
    assert np.array_equal(result['result'], np.array([1, 2, 3]))

def test_run_global_res_returns_array():
    """Test global-res function returns array."""
    step = pf.plStep('test', lambda: np.arange(10), [], ['result'], 'global-res')
    result = step._run([], [])
    
    assert np.array_equal(result['result'], np.arange(10))
    assert len(result['result']) == 10

def test_run_global_res_all_params_must_be_global():
    """Test that global-res functions require all params to be global."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'global-res')
    
    with pytest.raises(ValueError, match="All parameters must be global"):
        step._run([[1, 2, 3]], [False])

# Vectorized function tests
def test_run_vectorized_single_non_global_param():
    """Test vectorized function with single non-global parameter."""
    step = pf.plStep('test', lambda x: np.array(x) * 2, 
                     ['x'], ['result'], 'vectorized')
    result = step._run([[1, 2, 3, 4]], [False])
    
    assert np.array_equal(result['result'], np.array([2, 4, 6, 8]))

def test_run_vectorized_mixed_params():
    """Test vectorized function with mix of global and non-global params."""
    step = pf.plStep('test', lambda x, factor: np.array(x) * factor, 
                     ['x', 'factor'], ['result'], 'vectorized')
    result = step._run([[1, 2, 3], 10], [False, True])
    
    assert np.array_equal(result['result'], np.array([10, 20, 30]))

def test_run_vectorized_multiple_non_global():
    """Test vectorized function with multiple non-global parameters."""
    step = pf.plStep('test', lambda x, y: np.array(x) + np.array(y), 
                     ['x', 'y'], ['result'], 'vectorized')
    result = step._run([[1, 2, 3], [4, 5, 6]], [False, False])
    
    assert np.array_equal(result['result'], np.array([5, 7, 9]))

def test_run_vectorized_multiple_returns():
    """Test vectorized function with multiple return values."""
    def func(x):
        x = np.array(x)
        return x * 2, x * 3
    
    step = pf.plStep('test', func, ['x'], ['result1', 'result2'], 'vectorized')
    result = step._run([[1, 2, 3]], [False])
    
    assert np.array_equal(result['result1'], np.array([2, 4, 6]))
    assert np.array_equal(result['result2'], np.array([3, 6, 9]))

def test_run_vectorized_requires_non_global_param():
    """Test vectorized function requires at least one non-global parameter."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'vectorized')
    
    with pytest.raises(ValueError, match="At least one parameter must be non-global"):
        step._run([5], [True])

def test_run_vectorized_non_global_lengths_must_match():
    """Test that all non-global params must have same length."""
    step = pf.plStep('test', lambda x, y: np.array(x) + np.array(y), 
                     ['x', 'y'], ['result'], 'vectorized')
    
    with pytest.raises(ValueError, match="All non-global parameters must have the same number of rows"):
        step._run([[1, 2, 3], [1, 2]], [False, False])

def test_run_vectorized_output_length_must_match():
    """Test that vectorized output length must match input length."""
    # Function that returns wrong length
    step = pf.plStep('test', lambda x: np.array([1, 2]),  # Always returns 2 elements
                     ['x'], ['result'], 'vectorized')
    
    with pytest.raises(ValueError, match="Vectorized function output length does not match parameter length"):
        step._run([[1, 2, 3, 4]], [False])  # Input has 4 elements

# Per-row function tests
def test_run_per_row_single_non_global_param():
    """Test per-row function with single non-global parameter."""
    step = pf.plStep('test', lambda x: x * 2, ['x'], ['result'], 'per-row')
    result = step._run([[1, 2, 3, 4]], [False])
    
    assert np.array_equal(result['result'], np.array([2, 4, 6, 8]))

def test_run_per_row_mixed_params():
    """Test per-row function with mix of global and non-global params."""
    step = pf.plStep('test', lambda x, factor: x * factor, 
                     ['x', 'factor'], ['result'], 'per-row')
    result = step._run([[1, 2, 3], 10], [False, True])
    
    assert np.array_equal(result['result'], np.array([10, 20, 30]))

def test_run_per_row_multiple_non_global():
    """Test per-row function with multiple non-global parameters."""
    step = pf.plStep('test', lambda x, y: x + y, 
                     ['x', 'y'], ['result'], 'per-row')
    result = step._run([[1, 2, 3], [4, 5, 6]], [False, False])
    
    assert np.array_equal(result['result'], np.array([5, 7, 9]))

def test_run_per_row_multiple_returns():
    """Test per-row function with multiple return values."""
    step = pf.plStep('test', lambda x: (x * 2, x * 3), 
                     ['x'], ['result1', 'result2'], 'per-row')
    result = step._run([[1, 2, 3]], [False])
    
    assert np.array_equal(result['result1'], np.array([2, 4, 6]))
    assert np.array_equal(result['result2'], np.array([3, 6, 9]))

def test_run_per_row_requires_non_global_param():
    """Test per-row function requires at least one non-global parameter."""
    step = pf.plStep('test', lambda x: x, ['x'], ['result'], 'per-row')
    
    with pytest.raises(ValueError, match="At least one parameter must be non-global"):
        step._run([5], [True])

def test_run_per_row_non_global_lengths_must_match():
    """Test that all non-global params must have same length."""
    step = pf.plStep('test', lambda x, y: x + y, 
                     ['x', 'y'], ['result'], 'per-row')
    
    with pytest.raises(ValueError, match="All non-global parameters must have the same number of rows"):
        step._run([[1, 2, 3], [1, 2]], [False, False])

def test_run_per_row_function_called_for_each_row():
    """Test that per-row function is called once for each row."""
    call_count = []
    
    def func(x):
        call_count.append(x)
        return x * 2
    
    step = pf.plStep('test', func, ['x'], ['result'], 'per-row')
    result = step._run([[1, 2, 3]], [False])
    
    assert len(call_count) == 3
    assert call_count == [1, 2, 3]
    assert np.array_equal(result['result'], np.array([2, 4, 6]))

# Return value validation tests
def test_run_return_count_must_match():
    """Test that number of returns must match return_names."""
    # Function returns 1 value, but 2 expected
    step = pf.plStep('test', lambda: 42, [], ['result1', 'result2'], 'global')
    
    with pytest.raises(ValueError, match="Function return length does not match number of return names"):
        step._run([], [])

def test_run_return_count_must_match_multiple():
    """Test return count validation with multiple returns."""
    # Function returns 2 values, but 1 expected
    step = pf.plStep('test', lambda: (1, 2), [], ['result'], 'global')
    
    with pytest.raises(ValueError, match="Function return length does not match number of return names"):
        step._run([], [])

# Edge cases
def test_run_empty_params_and_returns():
    """Test with empty params and returns."""
    step = pf.plStep('test', lambda: None, [], [], 'global')
    
    with pytest.raises(ValueError, match="Function return length does not match number of return names"):
        step._run([], [])

def test_run_single_row():
    """Test per-row and vectorized with single row."""
    # Per-row
    step = pf.plStep('test', lambda x: x * 2, ['x'], ['result'], 'per-row')
    result = step._run([[5]], [False])
    assert np.array_equal(result['result'], np.array([10]))
    
    # Vectorized - need to handle array properly
    step = pf.plStep('test', lambda x: np.array(x) * 2, ['x'], ['result'], 'vectorized')
    result = step._run([[5]], [False])
    assert np.array_equal(result['result'], np.array([10]))

def test_run_numpy_arrays_as_params():
    """Test that numpy arrays work as non-global params."""
    step = pf.plStep('test', lambda x: x * 2, ['x'], ['result'], 'per-row')
    result = step._run([np.array([1, 2, 3])], [False])
    
    assert np.array_equal(result['result'], np.array([2, 4, 6]))

def test_run_lists_as_params():
    """Test that lists work as non-global params."""
    step = pf.plStep('test', lambda x: x * 2, ['x'], ['result'], 'per-row')
    result = step._run([[1, 2, 3]], [False])
    
    assert np.array_equal(result['result'], np.array([2, 4, 6]))

def test_run_complex_per_row_scenario():
    """Test complex scenario with multiple global and non-global params."""
    def func(a, b, c, d, e):
        return a * b + c * d + e
    
    step = pf.plStep('test', func, ['a', 'b', 'c', 'd', 'e'], ['result'], 'per-row')
    # a and c are non-global arrays, b, d, e are global scalars
    result = step._run([[1, 2, 3], 10, [4, 5, 6], 2, 5], [False, True, False, True, True])
    
    # Expected: [1*10 + 4*2 + 5, 2*10 + 5*2 + 5, 3*10 + 6*2 + 5]
    #         = [10+8+5, 20+10+5, 30+12+5]
    #         = [23, 35, 47]
    assert np.array_equal(result['result'], np.array([23, 35, 47]))

def test_run_complex_vectorized_scenario():
    """Test complex scenario with vectorized function."""
    def func(a, b, c):
        a = np.array(a)
        c = np.array(c)
        return a * b + c
    
    step = pf.plStep('test', func, ['a', 'b', 'c'], ['result'], 'vectorized')
    # a and c are non-global arrays, b is global scalar
    result = step._run([[1, 2, 3], 10, [4, 5, 6]], [False, True, False])
    
    # Expected: [1*10+4, 2*10+5, 3*10+6] = [14, 25, 36]
    assert np.array_equal(result['result'], np.array([14, 25, 36]))

################################################################################
################################### LazyAttr ###################################
################################################################################

class MockDataSetForLazyAttr:
    """Mock DataSet for testing LazyAttr with _fetch_rows method."""
    def __init__(self, nrows=10):
        self.nrows = nrows
        self._fetch_calls = []  # Track calls to _fetch_rows
        
    def _fetch_rows(self, name, run_idx, rows, enforced_max_runs={}):
        """Mock _fetch_rows that returns computed data."""
        self._fetch_calls.append((name, run_idx, rows))
        # Return simple computed values: row_idx * 10 + run_idx
        return [np.array([r * 10 + run_idx]) for r in rows]


# __init__, __repr__, __str__
def test_lazyattr_init():
    """Test LazyAttr initialization."""
    DS = MockDataSetForLazyAttr()
    name = 'test_attr'
    run_idx = 5
    LA = pf.LazyAttr(DS, name, run_idx)
    
    assert LA.DS == DS  
    assert LA.name == name
    assert LA.run_idx == run_idx
    assert LA._cache == {}
    assert LA.shape == ()
    
    # Test __repr__
    r = f"LazyAttr({name}, 0 cached rows)" 
    assert repr(LA) == r, "__repr__ output incorrect"
    
    # Test __str__
    s = f"Lazy Attribute: {name}\n\tCached Rows: []"
    assert str(LA) == s, "__str__ output incorrect"
    
    # Add cache and test again
    LA._cache = {i: i*2 for i in range(10)}
    r = f"LazyAttr({name}, 10 cached rows)" 
    s = f"Lazy Attribute: {name}\n\tCached Rows: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
    assert repr(LA) == r, "__repr__ output incorrect with cached rows"
    assert str(LA) == s, "__str__ output incorrect with cached rows"


def test_lazyattr_init_invalid():
    """Test LazyAttr initialization with invalid inputs."""
    DS = MockDataSetForLazyAttr()
    
    # incorrect name datatype
    with pytest.raises(ValueError, match="name must be a string"): 
        pf.LazyAttr(DS, 123, 0)
    
    # run_idx should be convertible to int (test it works with string)
    LA = pf.LazyAttr(DS, 'test', '5')
    assert LA.run_idx == 5


def test_lazyattr_len():
    """Test __len__ method returns nrows."""
    DS = MockDataSetForLazyAttr(nrows=15)
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    assert len(LA) == 15
    
    DS2 = MockDataSetForLazyAttr(nrows=100)
    LA2 = pf.LazyAttr(DS2, 'test_attr', 0)
    assert len(LA2) == 100


# _normalize_key tests
def test_lazyattr_normalize_key_single_int():
    """Test _normalize_key with single integer."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test', 0)
    
    rows, return_array, inner_key = LA._normalize_key(5)
    assert rows == [5]
    assert return_array == False
    assert inner_key is None


def test_lazyattr_normalize_key_negative_int():
    """Test _normalize_key with negative integer."""
    DS = MockDataSetForLazyAttr(nrows=10)
    LA = pf.LazyAttr(DS, 'test', 0)
    
    rows, return_array, inner_key = LA._normalize_key(-1)
    assert rows == [9]
    assert return_array == False
    
    rows, return_array, inner_key = LA._normalize_key(-3)
    assert rows == [7]


def test_lazyattr_normalize_key_slice():
    """Test _normalize_key with slice."""
    DS = MockDataSetForLazyAttr(nrows=10)
    LA = pf.LazyAttr(DS, 'test', 0)
    
    rows, return_array, inner_key = LA._normalize_key(slice(2, 5))
    assert rows == [2, 3, 4]
    assert return_array == True
    assert inner_key is None
    
    # Slice with step
    rows, return_array, inner_key = LA._normalize_key(slice(0, 8, 2))
    assert rows == [0, 2, 4, 6]
    
    # Negative slice
    rows, return_array, inner_key = LA._normalize_key(slice(-3, None))
    assert rows == [7, 8, 9]


def test_lazyattr_normalize_key_list():
    """Test _normalize_key with list."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test', 0)
    
    rows, return_array, inner_key = LA._normalize_key([1, 3, 5])
    assert rows == [1, 3, 5]
    assert return_array == True
    assert inner_key is None
    
    # List with negative indices
    rows, return_array, inner_key = LA._normalize_key([0, -1, 5])
    assert rows == [0, 9, 5]


def test_lazyattr_normalize_key_ndarray():
    """Test _normalize_key with numpy array."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test', 0)
    
    rows, return_array, inner_key = LA._normalize_key(np.array([2, 4, 6]))
    assert rows == [2, 4, 6]
    assert return_array == True


def test_lazyattr_normalize_key_tuple():
    """Test _normalize_key with tuple for sub-indexing."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test', 0)
    
    # Tuple with 2 elements
    rows, return_array, inner_key = LA._normalize_key((5, 10))
    assert rows == [5]
    assert return_array == False
    assert inner_key == 10
    
    # Tuple with 3+ elements
    rows, return_array, inner_key = LA._normalize_key((3, 1, 5))
    assert rows == [3]
    assert inner_key == (1, 5)


def test_lazyattr_normalize_key_out_of_bounds():
    """Test _normalize_key with out of bounds indices."""
    DS = MockDataSetForLazyAttr(nrows=10)
    LA = pf.LazyAttr(DS, 'test', 0)
    
    with pytest.raises(IndexError, match="out of bounds"):
        LA._normalize_key(10)
    
    with pytest.raises(IndexError, match="out of bounds"):
        LA._normalize_key(-11)
    
    with pytest.raises(IndexError, match="out of bounds"):
        LA._normalize_key([1, 2, 15])


def test_lazyattr_normalize_key_invalid_type():
    """Test _normalize_key with invalid key type."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test', 0)
    
    with pytest.raises(TypeError, match="Invalid index type"):
        LA._normalize_key("invalid")
    
    with pytest.raises(TypeError, match="Invalid index type"):
        LA._normalize_key({'key': 'value'})


# __getitem__ tests
def test_lazyattr_getitem_single_index():
    """Test __getitem__ with single index fetches from DataSet."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=5)
    
    # First access should fetch from DataSet
    result = LA[2]
    assert np.array_equal(result, [25])  # 2*10 + 5
    assert (2,) in [tuple(r) if isinstance(r, list) else r 
                    for _, _, r in DS._fetch_calls]
    assert 2 in LA._cache
    
    # Second access should use cache
    DS._fetch_calls.clear()
    result = LA[2]
    assert np.array_equal(result, [25])
    assert len(DS._fetch_calls) == 0  # No new fetch


def test_lazyattr_getitem_list():
    """Test __getitem__ with list of indices."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=3)
    
    result = LA[[1, 3, 5]]
    expected = np.array([[13], [33], [53]])  # [1*10+3, 3*10+3, 5*10+3]
    assert np.array_equal(result, expected)
    assert result.shape == (3, 1)
    assert 1 in LA._cache
    assert 3 in LA._cache
    assert 5 in LA._cache


def test_lazyattr_getitem_slice():
    """Test __getitem__ with slice."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=2)
    
    result = LA[2:5]
    expected = np.array([[22], [32], [42]])
    assert np.array_equal(result, expected)
    assert result.shape == (3, 1)


def test_lazyattr_getitem_negative_index():
    """Test __getitem__ with negative indices."""
    DS = MockDataSetForLazyAttr(nrows=10)
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=1)
    
    result = LA[-1]
    assert np.array_equal(result, [91])  # 9*10 + 1
    assert 9 in LA._cache
    
    result = LA[[-2, -1]]
    expected = np.array([[81], [91]])
    assert np.array_equal(result, expected)


def test_lazyattr_getitem_partial_cache():
    """Test __getitem__ when some rows are cached."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=0)
    
    # Pre-populate cache
    LA._cache = {1: np.array([100]), 3: np.array([300])}
    
    # Request rows where some are cached
    result = LA[[1, 2, 3]]
    
    # Should only fetch row 2
    fetch_calls = [call[2] for call in DS._fetch_calls]
    assert [2] in fetch_calls
    assert len(fetch_calls) == 1
    
    # All rows should be in result
    assert result.shape == (3, 1)
    assert np.array_equal(result[0], [100])  # from cache
    assert np.array_equal(result[1], [20])   # fetched
    assert np.array_equal(result[2], [300])  # from cache


def test_lazyattr_getitem_repeated_indices():
    """Test __getitem__ with repeated indices."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=0)
    
    result = LA[[2, 2, 3, 2]]
    # Should fetch 2 and 3 once each
    assert result.shape == (4, 1)
    assert np.array_equal(result, [[20], [20], [30], [20]])


def test_lazyattr_getitem_tuple_subindexing():
    """Test __getitem__ with tuple for sub-indexing."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=0)
    
    # Manually set cache with multi-dimensional data
    LA._cache = {0: np.array([10, 20, 30, 40])}
    
    result = LA[0, 2]
    assert result == 30
    
    # 2D array
    LA._cache = {0: np.array([[1, 2], [3, 4], [5, 6]])}
    result = LA[0, 1, 0]
    assert result == 3


def test_lazyattr_getitem_updates_shape():
    """Test that __getitem__ updates shape on first fetch."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=0)
    
    assert LA.shape == ()
    _ = LA[0]
    assert LA.shape == (10, 1)  # (nrows, data_shape...)


def test_lazyattr_getitem_numpy_array_indices():
    """Test __getitem__ with numpy array of indices."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=1)
    
    indices = np.array([1, 4, 7])
    result = LA[indices]
    expected = np.array([[11], [41], [71]])
    assert np.array_equal(result, expected)


def test_lazyattr_getitem_out_of_bounds():
    """Test __getitem__ with out of bounds index raises error."""
    DS = MockDataSetForLazyAttr(nrows=10)
    LA = pf.LazyAttr(DS, 'test_attr', run_idx=0)
    
    with pytest.raises(IndexError, match="out of bounds"):
        LA[10]
    
    with pytest.raises(IndexError, match="out of bounds"):
        LA[-11]


# __setitem__ tests (keep existing structure, update for nrows)
@pytest.mark.parametrize("rows,values,expected_cache", [
    (slice(0, 2), [np.array([10]), np.array([20])], 
     {0: np.array([10]), 1: np.array([20])}),  # slice
    ([2, 4, 6], [np.array([30]), np.array([50]), np.array([70])], 
     {2: np.array([30]), 4: np.array([50]), 6: np.array([70])}),  # list of idx
    (np.array([3, 5], dtype=np.int32), [np.array([40]), np.array([60])], 
     {3: np.array([40]), 5: np.array([60])}),  # np.ndarray of idx
    (0, np.array([5]), {0: np.array([5])}),  # single index
    ([1, 3, 2], [np.array([20]), np.array([40]), np.array([30])], 
     {1: np.array([20]), 2: np.array([30]), 3: np.array([40])}),  # out of order
    (-1, np.array([100]), {9: np.array([100])}),  # negative index 
    ([0, -2], [np.array([1]), np.array([2])], 
     {0: np.array([1]), 8: np.array([2])}),  # negative index in list 
    (slice(-3, None), [np.array([7]), np.array([8]), np.array([9])], 
     {7: np.array([7]), 8: np.array([8]), 9: np.array([9])}),  # negative slice 
])
def test_lazyattr_setitem(rows, values, expected_cache):
    """Test __setitem__ with various index types."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    LA[rows] = values
    
    # Compare cache contents
    assert set(LA._cache.keys()) == set(expected_cache.keys())
    for key in expected_cache:
        assert np.array_equal(LA._cache[key], expected_cache[key])


def test_lazyattr_setitem_scalar_broadcast():
    """Test __setitem__ setting multiple rows to single non-iterable value."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    
    # Test with raw integer (non-iterable, should broadcast)
    LA[[1, 2, 3]] = 10
    assert LA._cache[1] == 10
    assert LA._cache[2] == 10
    assert LA._cache[3] == 10
    
    # Test with slice
    LA2 = pf.LazyAttr(DS, 'test_attr2', 0)
    LA2[1:3] = 99
    assert LA2._cache[1] == 99
    assert LA2._cache[2] == 99


def test_lazyattr_setitem_updates_shape():
    """Test that __setitem__ updates shape."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    
    assert LA.shape == ()
    LA[0] = np.array([1, 2, 3])
    assert LA.shape == (10, 3)


def test_lazyattr_setitem_tuple():
    """Test __setitem__ with tuple for sub-indexing."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    
    LA[0] = np.array([1, 2, 3])
    LA[0, 1] = 5  
    assert np.array_equal(LA._cache[0], [1, 5, 3])
    
    # 2D array 
    LA[1] = np.array([[0, 1], [2, 3], [4, 5]]) 
    LA[1, 1, 0] = 10
    assert np.array_equal(LA._cache[1], np.array([[0, 1], [10, 3], [4, 5]]))


@pytest.mark.parametrize("rows,values,error_type", [
    ("invalid", [10, 20], ValueError),  # invalid rows type
    ([1, 'a', 3], [10, 20, 30], ValueError),  # non-int in rows
    ([0, 10], [np.array([10]), np.array([20])], ValueError),  # row out of bounds
    ([1, 2], [np.array([10]), np.array([20]), np.array([30])], ValueError),  # value length mismatch
    (slice(0, 3), [np.array([10]), np.array([20])], ValueError),  # value length mismatch
])
def test_lazyattr_setitem_invalid(rows, values, error_type):
    """Test __setitem__ with invalid inputs."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    with pytest.raises(error_type):
        LA[rows] = values


def test_lazyattr_setitem_then_getitem():
    """Test that __setitem__ values can be retrieved with __getitem__."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'test_attr', 0)
    
    # Set values
    LA[0] = np.array([100])
    LA[1] = np.array([200])
    LA[2] = np.array([300])
    
    # Get values - should not fetch from DataSet
    assert np.array_equal(LA[0], [100])
    assert np.array_equal(LA[[0, 1, 2]], [[100], [200], [300]])
    assert len(DS._fetch_calls) == 0  # No fetches


# Integration tests
def test_lazyattr_integration_fetch_and_cache():
    """Integration test: fetch, cache, mix with manual sets."""
    DS = MockDataSetForLazyAttr()
    LA = pf.LazyAttr(DS, 'my_data', run_idx=7)
    
    # Fetch some rows
    first_fetch = LA[[1, 3, 5]]
    assert np.array_equal(first_fetch, [[17], [37], [57]])
    assert len(DS._fetch_calls) == 1
    
    # Manually set a row
    LA[2] = np.array([999])
    
    # Fetch mixed cached and new
    DS._fetch_calls.clear()
    mixed = LA[[1, 2, 4]]  # 1 cached, 2 manual, 4 new
    assert np.array_equal(mixed, [[17], [999], [47]])
    
    # Only row 4 should have been fetched
    assert len(DS._fetch_calls) == 1
    assert DS._fetch_calls[0][2] == [4]


def test_lazyattr_different_run_indices():
    """Test that different run indices produce different data."""
    DS = MockDataSetForLazyAttr()
    LA1 = pf.LazyAttr(DS, 'data', run_idx=1)
    LA2 = pf.LazyAttr(DS, 'data', run_idx=5)
    
    val1 = LA1[3]  # 3*10 + 1 = 31
    val2 = LA2[3]  # 3*10 + 5 = 35
    
    assert np.array_equal(val1, [31])
    assert np.array_equal(val2, [35])


def test_lazyattr_repr():
    """Test __repr__ method shows name and cached row count."""
    DS = MockDataSetForLazyAttr(nrows=20)
    LA = pf.LazyAttr(DS, 'test_param', run_idx=3)
    
    # Initially no cached rows
    result = repr(LA)
    assert "LazyAttr" in result
    assert "test_param" in result
    assert "0 cached rows" in result
    
    # Add some cached rows
    LA[5] = np.array([100])
    LA[10] = np.array([200])
    LA[15] = np.array([300])
    
    result = repr(LA)
    assert "LazyAttr" in result
    assert "test_param" in result
    assert "3 cached rows" in result


def test_lazyattr_str():
    """Test __str__ method shows detailed info with cached row indices."""
    DS = MockDataSetForLazyAttr(nrows=15)
    LA = pf.LazyAttr(DS, 'my_data', run_idx=1)
    
    # Initially no cached rows
    result = str(LA)
    assert "Lazy Attribute: my_data" in result
    assert "Cached Rows: []" in result
    
    # Add cached rows in non-sequential order
    LA[7] = np.array([70])
    LA[2] = np.array([20])
    LA[12] = np.array([120])
    
    result = str(LA)
    assert "Lazy Attribute: my_data" in result
    # Should be sorted in the output
    assert "Cached Rows: [2, 7, 12]" in result


################################################################################
########################### LazyAttrCollection #################################
################################################################################

class MockDataSetForCollection:
    """Mock DataSet for testing LazyAttrCollection."""
    def __init__(self, nrows=10):
        self.nrows = nrows
        # deps_maps: {data_idx: {param_name: run_idx}}
        # Default: all data_idx point to run 1 for any param
        self.deps_maps = {
            i: {} for i in range(nrows)
        }
        self.deps_maps['global'] = {}
        

def test_lazyattrcollection_init():
    """Test LazyAttrCollection initialization."""
    DS = MockDataSetForCollection()
    name = 'test_param'
    collection = pf.LazyAttrCollection(DS, name)
    
    assert collection.DS == DS
    assert collection.name == name
    assert collection._lazy_attrs == {}
    assert len(collection) == DS.nrows


def test_lazyattrcollection_add_run():
    """Test adding LazyAttr to collection."""
    DS = MockDataSetForCollection()
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Create and add LazyAttrs
    la1 = pf.LazyAttr(DS, 'param', run_idx=1)
    la2 = pf.LazyAttr(DS, 'param', run_idx=3)
    
    collection.add_run(1, la1)
    collection.add_run(3, la2)
    
    assert 1 in collection._lazy_attrs
    assert 3 in collection._lazy_attrs
    assert collection._lazy_attrs[1] == la1
    assert collection._lazy_attrs[3] == la2


def test_lazyattrcollection_add_run_duplicate_error():
    """Test that adding duplicate run_idx raises error."""
    DS = MockDataSetForCollection()
    collection = pf.LazyAttrCollection(DS, 'param')
    
    la = pf.LazyAttr(DS, 'param', run_idx=1)
    collection.add_run(1, la)
    
    # Try to add same run_idx again
    la2 = pf.LazyAttr(DS, 'param', run_idx=1)
    with pytest.raises(ValueError, match="Run 1 already exists"):
        collection.add_run(1, la2)


def test_lazyattrcollection_at_run():
    """Test accessing specific run."""
    DS = MockDataSetForCollection()
    collection = pf.LazyAttrCollection(DS, 'param')
    
    la1 = pf.LazyAttr(DS, 'param', run_idx=5)
    la2 = pf.LazyAttr(DS, 'param', run_idx=10)
    collection.add_run(5, la1)
    collection.add_run(10, la2)
    
    assert collection.at_run(5) == la1
    assert collection.at_run(10) == la2
    
    # Non-existent run should raise KeyError
    with pytest.raises(KeyError):
        collection.at_run(99)


def test_lazyattrcollection_normalize_index_scalar():
    """Test _normalize_index with scalar integer."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    data_idx_arr, is_scalar = collection._normalize_index(5)
    assert np.array_equal(data_idx_arr, [5])
    assert is_scalar == True


def test_lazyattrcollection_normalize_index_negative():
    """Test _normalize_index with negative indices."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    data_idx_arr, is_scalar = collection._normalize_index(-1)
    assert np.array_equal(data_idx_arr, [9])
    assert is_scalar == True
    
    data_idx_arr, is_scalar = collection._normalize_index(-3)
    assert np.array_equal(data_idx_arr, [7])


def test_lazyattrcollection_normalize_index_slice():
    """Test _normalize_index with slices."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Simple slice
    data_idx_arr, is_scalar = collection._normalize_index(slice(2, 5))
    assert np.array_equal(data_idx_arr, [2, 3, 4])
    assert is_scalar == False
    
    # Slice with step
    data_idx_arr, is_scalar = collection._normalize_index(slice(0, 8, 2))
    assert np.array_equal(data_idx_arr, [0, 2, 4, 6])
    
    # Negative slice
    data_idx_arr, is_scalar = collection._normalize_index(slice(-3, None))
    assert np.array_equal(data_idx_arr, [7, 8, 9])


def test_lazyattrcollection_normalize_index_list():
    """Test _normalize_index with lists."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    data_idx_arr, is_scalar = collection._normalize_index([1, 3, 5])
    assert np.array_equal(data_idx_arr, [1, 3, 5])
    assert is_scalar == False
    
    # List with negative indices
    data_idx_arr, is_scalar = collection._normalize_index([0, -1, 5])
    assert np.array_equal(data_idx_arr, [0, 9, 5])


def test_lazyattrcollection_normalize_index_array():
    """Test _normalize_index with numpy arrays."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    data_idx_arr, is_scalar = collection._normalize_index(np.array([2, 4, 6]))
    assert np.array_equal(data_idx_arr, [2, 4, 6])
    assert is_scalar == False


def test_lazyattrcollection_normalize_index_boolean_mask():
    """Test _normalize_index with boolean masks."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    mask = np.array([True, False, True, False, True, False, False, False, False, False])
    data_idx_arr, is_scalar = collection._normalize_index(mask)
    assert np.array_equal(data_idx_arr, [0, 2, 4])
    assert is_scalar == False


def test_lazyattrcollection_normalize_index_boolean_mask_wrong_length():
    """Test _normalize_index with wrong length boolean mask."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    mask = np.array([True, False, True])  # Wrong length
    with pytest.raises(IndexError, match="Boolean index length"):
        collection._normalize_index(mask)


def test_lazyattrcollection_normalize_index_out_of_bounds():
    """Test _normalize_index with out of bounds indices."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    with pytest.raises(IndexError, match="out of bounds"):
        collection._normalize_index(10)
    
    with pytest.raises(IndexError, match="out of bounds"):
        collection._normalize_index(-11)
    
    with pytest.raises(IndexError, match="out of bounds"):
        collection._normalize_index([1, 2, 15])


def test_lazyattrcollection_normalize_index_invalid_type():
    """Test _normalize_index with invalid types."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    with pytest.raises(TypeError, match="Invalid index type"):
        collection._normalize_index("invalid")
    
    with pytest.raises(TypeError, match="Invalid index type"):
        collection._normalize_index({'key': 'value'})


def test_lazyattrcollection_getitem_single_index():
    """Test __getitem__ with single index, single run."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up deps_maps so data_idx 5 uses run 3
    DS.deps_maps[5] = {3: {'param': {}}}
    
    # Create LazyAttr for run 3 with pre-cached data
    la = pf.LazyAttr(DS, 'param', run_idx=3)
    la._cache = {5: np.array([100])}
    collection.add_run(3, la)
    
    # Access data_idx 5
    result = collection[5]
    assert np.array_equal(result, [100])


def test_lazyattrcollection_getitem_list():
    """Test __getitem__ with list of indices from same run."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up deps_maps: data_idx 1, 3, 5 all use run 2
    for di in [1, 3, 5]:
        DS.deps_maps[di] = {2: {'param': {}}}
    
    # Create LazyAttr with cached data
    la = pf.LazyAttr(DS, 'param', run_idx=2)
    la._cache = {1: np.array([10]), 3: np.array([30]), 5: np.array([50])}
    collection.add_run(2, la)
    
    result = collection[[1, 3, 5]]
    expected = np.array([[10], [30], [50]])
    assert np.array_equal(result, expected)


def test_lazyattrcollection_getitem_mixed_runs():
    """Test __getitem__ with indices from different runs."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up deps_maps: different data_idx use different runs
    DS.deps_maps[0] = {1: {'param': {}}}
    DS.deps_maps[1] = {1: {'param': {}}}
    DS.deps_maps[2] = {3: {'param': {}}}
    DS.deps_maps[3] = {3: {'param': {}}}
    DS.deps_maps[4] = {5: {'param': {}}}
    
    # Create LazyAttrs for different runs
    la1 = pf.LazyAttr(DS, 'param', run_idx=1)
    la1._cache = {0: np.array([100]), 1: np.array([110])}
    collection.add_run(1, la1)
    
    la3 = pf.LazyAttr(DS, 'param', run_idx=3)
    la3._cache = {2: np.array([300]), 3: np.array([330])}
    collection.add_run(3, la3)
    
    la5 = pf.LazyAttr(DS, 'param', run_idx=5)
    la5._cache = {4: np.array([500])}
    collection.add_run(5, la5)
    
    # Request indices from mixed runs
    result = collection[[0, 2, 4, 1, 3]]
    expected = np.array([[100], [300], [500], [110], [330]])
    assert np.array_equal(result, expected)


def test_lazyattrcollection_getitem_preserves_order():
    """Test that __getitem__ preserves input order with mixed runs."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up different runs for different indices
    DS.deps_maps[5] = {1: {'param': {}}}
    DS.deps_maps[2] = {3: {'param': {}}}
    DS.deps_maps[7] = {1: {'param': {}}}
    
    la1 = pf.LazyAttr(DS, 'param', run_idx=1)
    la1._cache = {5: np.array([50]), 7: np.array([70])}
    collection.add_run(1, la1)
    
    la3 = pf.LazyAttr(DS, 'param', run_idx=3)
    la3._cache = {2: np.array([20])}
    collection.add_run(3, la3)
    
    # Request in specific order
    result = collection[[5, 2, 7]]
    expected = np.array([[50], [20], [70]])
    assert np.array_equal(result, expected)


def test_lazyattrcollection_getitem_slice():
    """Test __getitem__ with slice."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # All use run 1
    for i in range(10):
        DS.deps_maps[i] = {1: {'param': {}}}
    
    la = pf.LazyAttr(DS, 'param', run_idx=1)
    la._cache = {i: np.array([i * 10]) for i in range(10)}
    collection.add_run(1, la)
    
    result = collection[2:5]
    expected = np.array([[20], [30], [40]])
    assert np.array_equal(result, expected)


def test_lazyattrcollection_getitem_negative_index():
    """Test __getitem__ with negative indices."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    DS.deps_maps[9] = {1: {'param': {}}}
    DS.deps_maps[8] = {1: {'param': {}}}
    
    la = pf.LazyAttr(DS, 'param', run_idx=1)
    la._cache = {8: np.array([80]), 9: np.array([90])}
    collection.add_run(1, la)
    
    result = collection[-1]
    assert np.array_equal(result, [90])
    
    result = collection[[-2, -1]]
    expected = np.array([[80], [90]])
    assert np.array_equal(result, expected)


def test_lazyattrcollection_getitem_boolean_mask():
    """Test __getitem__ with boolean mask."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up for indices 0, 2, 4
    for i in [0, 2, 4]:
        DS.deps_maps[i] = {1: {'param': {}}}
    
    la = pf.LazyAttr(DS, 'param', run_idx=1)
    la._cache = {0: np.array([0]), 2: np.array([20]), 4: np.array([40])}
    collection.add_run(1, la)
    
    mask = np.array([True, False, True, False, True, False, False, False, False, False])
    result = collection[mask]
    expected = np.array([[0], [20], [40]])
    assert np.array_equal(result, expected)


def test_lazyattrcollection_getitem_scalar_returns_scalar():
    """Test that scalar index returns scalar, not array."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    DS.deps_maps[3] = {1: {'param': {}}}
    
    la = pf.LazyAttr(DS, 'param', run_idx=1)
    la._cache = {3: np.array([300])}
    collection.add_run(1, la)
    
    result = collection[3]
    # Should be scalar, not wrapped in array
    assert np.array_equal(result, [300])
    assert not isinstance(result, list)


def test_lazyattrcollection_getitem_defaults_to_run1():
    """Test that __getitem__ defaults to run 1 when parameter doesn't exist yet for a data_idx.
    
    When get_most_recent_run returns -1 (parameter not in deps_maps for that data_idx),
    the system should default to run 1, not run 0 (which is reserved as a special case).
    """
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up run 1 as the default run with data for all indices
    la1 = pf.LazyAttr(DS, 'param', run_idx=1)
    la1._cache = {0: np.array([10]), 1: np.array([11]), 2: np.array([12])}
    collection.add_run(1, la1)
    
    # Set up run 3 with data for some indices that have been updated
    la3 = pf.LazyAttr(DS, 'param', run_idx=3)
    la3._cache = {1: np.array([31])}
    collection.add_run(3, la3)
    
    # data_idx 0: not in deps_maps → should default to run 1
    # data_idx 1: in deps_maps with run 3 → should use run 3
    # data_idx 2: not in deps_maps → should default to run 1
    
    # Only data_idx 1 is in deps_maps (at run 3)
    DS.deps_maps[1] = {3: {'param': {}}}
    
    # Test single index that's not in deps_maps (should use run 1)
    result = collection[0]
    assert np.array_equal(result, [10])
    
    # Test single index that IS in deps_maps (should use run 3)
    result = collection[1]
    assert np.array_equal(result, [31])
    
    # Test mixed: some in deps_maps, some not (should use appropriate runs)
    result = collection[[0, 1, 2]]
    assert np.array_equal(result, [[10], [31], [12]])


def test_lazyattrcollection_getitem_all_default_to_run1():
    """Test when no data_idx are in deps_maps, all should default to run 1."""
    DS = MockDataSetForCollection(nrows=10)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    # Set up run 1 with data
    la1 = pf.LazyAttr(DS, 'param', run_idx=1)
    la1._cache = {0: np.array([100]), 1: np.array([200]), 2: np.array([300])}
    collection.add_run(1, la1)
    
    # Don't add anything to deps_maps - all should default to run 1
    # (deps_maps entries would be empty dicts by default)
    
    result = collection[[0, 1, 2]]
    assert np.array_equal(result, [[100], [200], [300]])


def test_lazyattrcollection_repr():
    """Test __repr__ method."""
    DS = MockDataSetForCollection()
    collection = pf.LazyAttrCollection(DS, 'my_param')
    
    collection.add_run(1, pf.LazyAttr(DS, 'my_param', 1))
    collection.add_run(5, pf.LazyAttr(DS, 'my_param', 5))
    collection.add_run(3, pf.LazyAttr(DS, 'my_param', 3))
    
    result = repr(collection)
    assert "LazyAttrCollection" in result
    assert "my_param" in result
    assert "[1, 3, 5]" in result  # Sorted


def test_lazyattrcollection_str():
    """Test __str__ method."""
    DS = MockDataSetForCollection(nrows=15)
    collection = pf.LazyAttrCollection(DS, 'param_x')
    
    collection.add_run(2, pf.LazyAttr(DS, 'param_x', 2))
    collection.add_run(7, pf.LazyAttr(DS, 'param_x', 7))
    
    result = str(collection)
    assert "Lazy Attribute Collection: param_x" in result
    assert "Available runs: [2, 7]" in result
    assert "Number of data indices: 15" in result


def test_lazyattrcollection_len():
    """Test __len__ method returns nrows from DataSet."""
    DS = MockDataSetForCollection(nrows=25)
    collection = pf.LazyAttrCollection(DS, 'param')
    
    assert len(collection) == 25
    
    # Test with different nrows
    DS2 = MockDataSetForCollection(nrows=100)
    collection2 = pf.LazyAttrCollection(DS2, 'other_param')
    
    assert len(collection2) == 100
    
    # Adding runs shouldn't change length (length is nrows, not number of runs)
    collection2.add_run(1, pf.LazyAttr(DS2, 'other_param', 1))
    collection2.add_run(5, pf.LazyAttr(DS2, 'other_param', 5))
    assert len(collection2) == 100


################################################################################
################################# find_pl_path #################################
################################################################################ 
step1 = {'task': pf.plStep('step1', lambda x: x + 1, ['a'], ['b'], 'per-row')}
step2 = {'task': pf.plStep('step2', lambda x: x + 1, ['b'], ['c'], 'per-row')}
step3 = {'task': pf.plStep('step3', lambda x: x + 1, ['b'], ['d'], 'per-row')}
step4 = {'task': pf.plStep('step4', lambda x: x + 1, ['d'], ['e'], 'per-row'),
         'delete_input': ['x']}
step5 = {'task': pf.plStep('step5', lambda x, y: (x + y, x - y), 
                           ['a', 'b'], ['f', 'g'], 'per-row'),
         'delete_input': ['a']}

@pytest.mark.parametrize("tree,return_name,expected_path", [
    ({'CAL_STEPS': {1: step1}}, 'b', ['step1']),
    ({'CAL_STEPS': {1: {'task': step1['task'], 'A_STEPS': {1: step2}}}}, 
     'b', ['step1']),
     ({'CAL_STEPS': {1: {'task': step1['task'], 'A_STEPS': {1: step2}}}}, 
     'c', ['step1', 'step2']),
    ({'CAL_STEPS': {1: {'task': step1['task'], 'A_STEPS': {1: step2, 2: step3}}, 
                    2: step5}}, 
     'c', ['step1', 'step2']),
     ({'CAL_STEPS': {1: {'task': step1['task'], 'A_STEPS': {1: step2, 2: step3}}, 
                     2: step5}}, 
     'g', ['step1', 'step5']),
])
def test_find_pl_path(tree, return_name, expected_path):
    path = pf.find_pl_path(tree, return_name)
    if expected_path is None:
        assert path is None
    else:
        assert path is not None
        assert [step.name for step in path] == expected_path 

def test_find_pl_path_not_found_returns_none():
    tree = {'CAL_STEPS': {1: step1, 2: step2}}
    assert pf.find_pl_path(tree, 'missing') is None

# don't need to check invalid tree input, handled in check_pl_tree_structure
def test_find_pl_path_invalid_input():
    with pytest.raises(ValueError):
        pf.find_pl_path({}, 123) # name not str
    with pytest.raises(ValueError):
        pf.find_pl_path({1: step1}, 'test') # root keys must end in '_STEPS'

################################################################################
###################### find_pl_path - Additional Edge Cases ####################
################################################################################
# Create additional steps for edge case testing
step6 = {'task': pf.plStep('step6', lambda x: x + 1, ['c'], ['h'], 'per-row')}
step7 = {'task': pf.plStep('step7', lambda x: x + 1, ['h'], ['i'], 'per-row')}
step8 = {'task': pf.plStep('step8', lambda x: x + 1, ['d'], ['j'], 'per-row')}
step9 = {'task': pf.plStep('step9', lambda x, y: (x + y, x * y), ['a', 'b'], ['k', 'm'], 'per-row')}
step10 = {'task': pf.plStep('step10', lambda x: x, ['b'], ['n', 'o', 'p'], 'per-row')}
step11 = {'task': pf.plStep('step11', lambda: 'global', [], ['q'], 'global')}

@pytest.mark.parametrize("tree,return_name,expected_path", [
    # Test finding the first step in a sequence (no predecessors)
    ({'CAL_STEPS': {1: step1, 2: step2}}, 'b', ['step1']),
    
    # Test deeply nested structure (multiple levels of child sequences)
    ({'CAL_STEPS': {1: {'task': step1['task'], 
                        'A_STEPS': {1: {'task': step2['task'],
                                       'B_STEPS': {1: step6, 2: step7}}}}}}, 
     'i', ['step1', 'step2', 'step6', 'step7']),
    
    # Test finding intermediate output in deeply nested structure
    ({'CAL_STEPS': {1: {'task': step1['task'], 
                        'A_STEPS': {1: {'task': step2['task'],
                                       'B_STEPS': {1: step6}}}}}}, 
     'h', ['step1', 'step2', 'step6']),
     
    # Test finding output at middle level (not deepest)
    ({'CAL_STEPS': {1: {'task': step1['task'], 
                        'A_STEPS': {1: {'task': step2['task'],
                                       'B_STEPS': {1: step6}}}}}}, 
     'c', ['step1', 'step2']),
    
    # Test multiple root sequences, finding in second sequence
    ({'CAL_STEPS': {1: step1, 2: step2}, 
      'ANALYSIS_STEPS': {1: step11}}, 
     'q', ['step11']),
     
    # Test multiple root sequences, finding in first sequence
    ({'CAL_STEPS': {1: step1, 2: step2}, 
      'ANALYSIS_STEPS': {1: step11}}, 
     'c', ['step1', 'step2']),
    
    # Test step with multiple return values (finding first return value)
    ({'CAL_STEPS': {1: step9}}, 'k', ['step9']),
    
    # Test step with multiple return values (finding second return value)
    ({'CAL_STEPS': {1: step9}}, 'm', ['step9']),
    
    # Test step with three return values
    ({'CAL_STEPS': {1: step10}}, 'n', ['step10']),
    ({'CAL_STEPS': {1: step10}}, 'o', ['step10']),
    ({'CAL_STEPS': {1: step10}}, 'p', ['step10']),
    
    # Test finding in parallel branches at same level (first branch)
    ({'CAL_STEPS': {1: {'task': step1['task'], 
                        'A_STEPS': {1: step2}, 
                        'B_STEPS': {1: step3}}}}, 
     'c', ['step1', 'step2']),
     
    # Test finding in parallel branches at same level (second branch)
    ({'CAL_STEPS': {1: {'task': step1['task'], 
                        'A_STEPS': {1: step2}, 
                        'B_STEPS': {1: step3}}}}, 
     'd', ['step1', 'step3']),
    
    # Test complex: multiple sequences with nested children, finding deep output
    ({'CAL_STEPS': {1: {'task': step1['task'],
                        'A_STEPS': {1: step2, 2: step3}},
                    2: {'task': step5['task'],
                        'B_STEPS': {1: step6}}}},
     'h', ['step1', 'step5', 'step6']),
     
    # Test finding output from parent task when it has children
    ({'CAL_STEPS': {1: {'task': step1['task'],
                        'A_STEPS': {1: step2, 2: step3}},
                    2: {'task': step5['task'],
                        'B_STEPS': {1: step6}}}},
     'g', ['step1', 'step5']),
     
    # Test sequence with single step
    ({'CAL_STEPS': {1: step1}}, 'b', ['step1']),
    
    # Test long linear sequence (step1 -> step2 -> step6 -> step7)
    ({'CAL_STEPS': {1: step1, 2: step2, 3: step6, 4: step7}}, 
     'i', ['step1', 'step2', 'step6', 'step7']),
     
    # Test finding intermediate in long sequence
    ({'CAL_STEPS': {1: step1, 2: step2, 3: step6, 4: step7}}, 
     'h', ['step1', 'step2', 'step6']),
     
    # Test output from second to last step in sequence
    ({'CAL_STEPS': {1: step1, 2: step2, 3: step6, 4: step7}}, 
     'h', ['step1', 'step2', 'step6']),
    
    # Test mixed: some steps with children, some without
    ({'CAL_STEPS': {1: step1, 
                    2: {'task': step2['task'], 'A_STEPS': {1: step6}},
                    3: step11}}, 
     'h', ['step1', 'step2', 'step6']),
     
    # Test finding global step output
    ({'CAL_STEPS': {1: step11, 2: step1}}, 'q', ['step11']),
    
    # Test multiple nested levels with multiple children at each level
    ({'CAL_STEPS': {1: {'task': step1['task'],
                        'A_STEPS': {1: {'task': step2['task'],
                                       'B_STEPS': {1: step6, 2: step7}},
                                   2: {'task': step3['task'],
                                       'C_STEPS': {1: step4}}}}}},
     'i', ['step1', 'step2', 'step6', 'step7']),
     
    # Test accessing parallel branch stored in separate named sequence
    ({'CAL_STEPS': {1: {'task': step1['task'],
                        'A_STEPS': {1: {'task': step2['task'],
                                       'B_STEPS': {1: step6}}},
                        'C_STEPS': {1: {'task': step3['task'],
                                       'D_STEPS': {1: step4}}}}}},
     'e', ['step1', 'step3', 'step4']),
])
def test_find_pl_path_edge_cases(tree, return_name, expected_path):
    """Test edge cases and complex scenarios for find_pl_path."""
    path = pf.find_pl_path(tree, return_name)
    if expected_path is None:
        assert path is None
    else:
        assert path is not None
        assert [step.name for step in path] == expected_path

def test_find_pl_path_multiple_outputs_not_found():
    """Test that non-existent output returns None even with multi-output steps."""
    tree = {'CAL_STEPS': {1: step9}}
    assert pf.find_pl_path(tree, 'nonexistent') is None

def test_find_pl_path_empty_root_sequence():
    """Test that empty sequences are handled (though they're unusual)."""
    tree = {'CAL_STEPS': {1: step1}, 'ANALYSIS_STEPS': {}}
    # Should still find in CAL_STEPS
    path = pf.find_pl_path(tree, 'b')
    assert path is not None
    assert [step.name for step in path] == ['step1']

################################################################################
########################### check_pl_tree_structure ############################
################################################################################
@pytest.mark.parametrize("path", [
    {},
    {'CAL_STEPS': {1: step1}},
    {'CAL_STEPS': {1: step1, 
                   2: {'task': step2['task'], 'delete_input': ['b']}}},
    # Valid params in analysis pipeline (cal=False, default)
    {'ANALYSIS_STEPS': {1: {'task': step1['task'], 
                            'params': {'span_mult': 2}}}},
    {'ANALYSIS_STEPS': {1: {'task': step1['task'], 
                            'params': {'name': 'test', 'value': 3.14, 
                                      'flag': True, 'opt': None}}}},
    # Valid params with delete_input
    {'ANALYSIS_STEPS': {1: {'task': step2['task'], 
                            'params': {'x': 1}, 
                            'delete_input': ['b']}}},
]) 
def test_check_pl_tree_structure(path):
    pf.check_pl_tree_structure(path) # should not raise any exceptions

@pytest.mark.parametrize("path", [
    {1: 1}, # node must be dict or plStep
    # if any node keys are int, all must be an int sequence starting from 1
    {1: step1, 'a': step2}, 
    {'test': {1: step1, 'a': step2}}, 
    {2: step1, 'a': step2}, 
    {'a': step1, 1: step2, 2: step3}, 
    {1: step2, 2: step3, 4: step4}, 
    # int sequences must have a 'task' key
    {1: {'not_task': step1}},
    {'test': {1: {'not_task': step1}}},
    # If keys are not an int sequence, they must all be strings
    {'task1': step1, 2.: step2},
    {1: {}},
    # 'task' is not plStep 
    {'task': 'not_a_step'},
    # delete_input in keys, but 'task' not in keys 
    {'delete_input': ['zf']},
    # delete_input value is invalid 
    {'task': step1['task'], 'delete_input': 'not_a_list_or_all'}, 
    {'task': step1['task'], 'delete_input': [1.]}, # list contents must be str 
    {'task': step1['task'], 'delete_input': ['z']}, # 'z' not in param_names 
    {'task': step1['task'], 'CAL_steps': {}}, # invalid step list name
    # params in keys, but 'task' not in keys
    {'params': {'x': 1}},
    # params value is not a dict
    {'task': step1['task'], 'params': 'not_a_dict'},
    {'task': step1['task'], 'params': [1, 2, 3]},
    # params keys are not strings
    {'task': step1['task'], 'params': {1: 'value'}},
    # params values are not simple types
    {'task': step1['task'], 'params': {'x': {'nested': 'dict'}}},
    {'task': step1['task'], 'params': {'x': [1, 2, 3]}},

]) 
def test_check_pl_tree_structure_invalid(path):
    with pytest.raises(ValueError):
        pf.check_pl_tree_structure(path)

# Test that params are rejected when cal=True
def test_check_pl_tree_structure_params_rejected_in_cal():
    path = {'CAL_STEPS': {1: {'task': step1['task'], 
                               'params': {'x': 1}}}}
    with pytest.raises(ValueError, match='params.*not allowed.*calibration'):
        pf.check_pl_tree_structure(path, cal=True)

################################################################################
############################# print_cal_pl Tests ###############################
################################################################################

def test_print_cal_pl_basic_dict(capsys):
    """Test printing a simple dictionary with plStep values."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    step_b = pf.plStep('step_b', lambda x: x * 2, ['in2'], ['out2'], 'per-row')
    cal_pl = {'key1': step_a, 'key2': step_b}
    
    pf.print_cal_pl(cal_pl)
    captured = capsys.readouterr()
    
    # Should print keys and then the step string representations
    assert 'key1:' in captured.out
    assert 'key2:' in captured.out
    assert 'step_a' in captured.out
    assert 'step_b' in captured.out

def test_print_cal_pl_nested_dict(capsys):
    """Test printing a nested dictionary structure."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    step_b = pf.plStep('step_b', lambda x: x * 2, ['in2'], ['out2'], 'per-row')
    cal_pl = {
        'level1': {
            'level2a': step_a,
            'level2b': step_b
        }
    }
    
    pf.print_cal_pl(cal_pl)
    captured = capsys.readouterr()
    
    # Check hierarchy is present
    assert 'level1:' in captured.out
    assert 'level2a:' in captured.out
    assert 'level2b:' in captured.out
    assert 'step_a' in captured.out
    assert 'step_b' in captured.out

def test_print_cal_pl_deeply_nested(capsys):
    """Test printing a deeply nested dictionary (3+ levels)."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    cal_pl = {
        'level1': {
            'level2': {
                'level3': step_a
            }
        }
    }
    
    pf.print_cal_pl(cal_pl)
    captured = capsys.readouterr()
    
    assert 'level1:' in captured.out
    assert 'level2:' in captured.out
    assert 'level3:' in captured.out
    assert 'step_a' in captured.out

def test_print_cal_pl_non_dict_input(capsys):
    """Test printing when input is not a dict (e.g., plStep directly)."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    
    pf.print_cal_pl(step_a)
    captured = capsys.readouterr()
    
    # Should print the step's string representation
    assert 'step_a' in captured.out
    assert 'in1' in captured.out
    assert 'out1' in captured.out

def test_print_cal_pl_with_custom_indent(capsys):
    """Test that custom indent parameter works correctly."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    cal_pl = {'key1': step_a}
    
    pf.print_cal_pl(cal_pl, indent=2)
    captured = capsys.readouterr()
    
    # Should have 4 spaces (2 * 2) before key1
    assert '    key1:' in captured.out

def test_print_cal_pl_empty_dict(capsys):
    """Test printing an empty dictionary."""
    cal_pl = {}
    
    pf.print_cal_pl(cal_pl)
    captured = capsys.readouterr()
    
    # Should produce no output (or just whitespace)
    assert captured.out.strip() == ''

def test_print_cal_pl_mixed_nesting(capsys):
    """Test printing a dict with both nested dicts and direct plStep values."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    step_b = pf.plStep('step_b', lambda x: x * 2, ['in2'], ['out2'], 'per-row')
    step_c = pf.plStep('step_c', lambda x: x - 1, ['in3'], ['out3'], 'per-row')
    
    cal_pl = {
        'direct': step_a,
        'nested': {
            'inner1': step_b,
            'inner2': step_c
        }
    }
    
    pf.print_cal_pl(cal_pl)
    captured = capsys.readouterr()
    
    assert 'direct:' in captured.out
    assert 'nested:' in captured.out
    assert 'inner1:' in captured.out
    assert 'inner2:' in captured.out
    assert 'step_a' in captured.out
    assert 'step_b' in captured.out
    assert 'step_c' in captured.out

def test_print_cal_pl_indentation_levels(capsys):
    """Test that indentation increases correctly at each nesting level."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    cal_pl = {
        'level1': {
            'level2': step_a
        }
    }
    
    pf.print_cal_pl(cal_pl)
    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    
    # Find the lines with keys
    level1_line = [l for l in lines if 'level1:' in l][0]
    level2_line = [l for l in lines if 'level2:' in l][0]
    
    # level1 should have no leading spaces, level2 should have 2 spaces
    assert level1_line.startswith('level1:')
    assert level2_line.startswith('  level2:')

def test_print_cal_pl_multiline_step_repr(capsys):
    """Test that multiline plStep representations are indented correctly."""
    # Create a step with multiple inputs/outputs to produce multiline output
    step_a = pf.plStep('long_step_name', lambda x, y: x + y, 
                       ['input1', 'input2'], ['output1'], 'per-row')
    cal_pl = {'key1': step_a}
    
    pf.print_cal_pl(cal_pl, indent=1)
    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    
    # All step detail lines should be indented (at least 4 spaces for indent=1+1)
    step_lines = [l for l in lines if l.strip() and 'key1:' not in l]
    for line in step_lines:
        assert line.startswith('    ')  # 2 spaces * 2 indent levels

################################################################################
############################## print_path Tests ################################
################################################################################

def test_print_path_basic_list(capsys):
    """Test printing a basic list of plSteps."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    step_b = pf.plStep('step_b', lambda x: x * 2, ['in2'], ['out2'], 'per-row')
    path = [step_a, step_b]
    
    pf.print_path(path)
    captured = capsys.readouterr()
    
    assert 'step_a' in captured.out
    assert 'step_b' in captured.out
    assert 'in1' in captured.out
    assert 'in2' in captured.out

def test_print_path_empty_list(capsys):
    """Test printing an empty path list."""
    path = []
    
    pf.print_path(path)
    captured = capsys.readouterr()
    
    # Should produce no output (or just whitespace)
    assert captured.out.strip() == ''

def test_print_path_single_step(capsys):
    """Test printing a path with a single step."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    path = [step_a]
    
    pf.print_path(path)
    captured = capsys.readouterr()
    
    assert 'step_a' in captured.out
    assert 'in1' in captured.out
    assert 'out1' in captured.out

def test_print_path_with_custom_indent(capsys):
    """Test that custom indent parameter works correctly."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    path = [step_a]
    
    pf.print_path(path, indent=2)
    captured = capsys.readouterr()
    
    # First line has 2*(indent+1)-1 = 5 spaces, subsequent lines have 6 spaces
    lines = [l for l in captured.out.split('\n') if l.strip()]
    assert lines[0].startswith('     ')  # 5 spaces
    for line in lines[1:]:
        assert line.startswith('      ')  # 6 spaces

def test_print_path_multiple_steps(capsys):
    """Test printing a path with multiple steps to verify order."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    step_b = pf.plStep('step_b', lambda x: x * 2, ['out1'], ['out2'], 'per-row')
    step_c = pf.plStep('step_c', lambda x: x - 1, ['out2'], ['out3'], 'per-row')
    path = [step_a, step_b, step_c]
    
    pf.print_path(path)
    captured = capsys.readouterr()
    
    # Check all steps are present
    assert 'step_a' in captured.out
    assert 'step_b' in captured.out
    assert 'step_c' in captured.out
    
    # Verify order by checking positions
    pos_a = captured.out.index('step_a')
    pos_b = captured.out.index('step_b')
    pos_c = captured.out.index('step_c')
    assert pos_a < pos_b < pos_c

def test_print_path_multiline_step_repr(capsys):
    """Test that multiline plStep representations are indented correctly."""
    step_a = pf.plStep('long_step_name', lambda x, y: x + y, 
                       ['input1', 'input2'], ['output1'], 'per-row')
    path = [step_a]
    
    pf.print_path(path, indent=1)
    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    
    # First line has 2*(indent+1)-1 = 3 spaces, subsequent lines have 4 spaces
    non_empty_lines = [l for l in lines if l.strip()]
    assert non_empty_lines[0].startswith('   ')  # 3 spaces
    for line in non_empty_lines[1:]:
        assert line.startswith('    ')  # 4 spaces

def test_print_path_different_func_types(capsys):
    """Test printing steps with different func_type values."""
    step_per_row = pf.plStep('step_per_row', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    step_vectorized = pf.plStep('step_vec', lambda x: x * 2, ['out1'], ['out2'], 'vectorized')
    step_global = pf.plStep('step_global', lambda: 42, [], ['out3'], 'global')
    path = [step_per_row, step_vectorized, step_global]
    
    pf.print_path(path)
    captured = capsys.readouterr()
    
    assert 'step_per_row' in captured.out
    assert 'step_vec' in captured.out
    assert 'step_global' in captured.out
    assert 'per-row' in captured.out
    assert 'vectorized' in captured.out
    assert 'global' in captured.out

def test_print_path_steps_with_special_names(capsys):
    """Test printing steps with special characters in names."""
    step_a = pf.plStep('step-with-dashes', lambda x: x, ['in'], ['out'], 'per-row')
    step_b = pf.plStep('step_with_underscores', lambda x: x, ['in'], ['out'], 'per-row')
    step_c = pf.plStep('step123', lambda x: x, ['in'], ['out'], 'per-row')
    path = [step_a, step_b, step_c]
    
    pf.print_path(path)
    captured = capsys.readouterr()
    
    assert 'step-with-dashes' in captured.out
    assert 'step_with_underscores' in captured.out
    assert 'step123' in captured.out

def test_print_path_zero_indent(capsys):
    """Test that default indent=0 works correctly."""
    step_a = pf.plStep('step_a', lambda x: x + 1, ['in1'], ['out1'], 'per-row')
    path = [step_a]
    
    pf.print_path(path, indent=0)
    captured = capsys.readouterr()
    
    # First line has 2*(indent+1)-1 = 1 space, subsequent lines have 2 spaces
    lines = [l for l in captured.out.split('\n') if l.strip()]
    assert lines[0].startswith(' ')  # 1 space
    assert not lines[0].startswith('  ')  # Not 2 spaces
    for line in lines[1:]:
        assert line.startswith('  ')  # 2 spaces
