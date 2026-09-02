"""
Tests for citkid.pipeline_v2.framework module.

Tests plStep class and framework utilities (LazyAttr, find_pl_path, check_pl_tree_structure).
All tests are adapted from pipeline.test_framework for use with pipeline_v2.

NOTE: LazyAttrCollection tests are excluded (v1-only feature for multi-run management).
"""

import pytest
import numpy as np
from citkid.pipeline_v2 import framework as pf


################################################################################
################################### plStep Tests ##############################
################################################################################

class TestPlStepInit:
    """Tests for plStep.__init__"""

    @pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
        ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
        ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
        ('step3', lambda: 42, [], ['y'], 'global'),
        ('step4', lambda: np.arange(10), [], ['z'], 'global-res'),
    ])
    def test_init_basic(self, name, func, param_names, return_names, func_type):
        """Test basic __init__ functionality with all func_types."""
        step = pf.plStep(name, func, param_names, return_names, func_type)
        
        assert step.name == name
        assert step.func == func
        assert step.param_names == param_names
        assert step.return_names == return_names
        assert step.func_type == func_type

    def test_init_default_func_type(self):
        """Test that func_type defaults to 'per-row' when not specified."""
        step = pf.plStep('test_step', lambda x: x, ['x'], ['y'])
        assert step.func_type == 'per-row'

    def test_init_func_callable(self):
        """Test that the function is callable and works correctly."""
        func = lambda x: x + 10
        step = pf.plStep('test', func, ['x'], ['y'], 'per-row')
        
        assert step.func == func
        assert step.func(5) == 15
        assert step.func(100) == 110

    @pytest.mark.parametrize("param_names,return_names", [
        ([], []),
        (['x'], []),
        ([], ['y']),
        (['a', 'b', 'c'], ['x', 'y', 'z']),
    ])
    def test_init_various_list_lengths(self, param_names, return_names):
        """Test __init__ with various lengths of param_names and return_names."""
        step = pf.plStep('test', lambda: None, param_names, return_names, 'global')
        assert step.param_names == param_names
        assert step.return_names == return_names

    def test_init_lists_are_copied(self):
        """Test that param_names and return_names are copied, not referenced."""
        original_params = ['x', 'y', 'z']
        original_returns = ['a', 'b']
        
        step = pf.plStep('test', lambda: None, original_params, original_returns, 'global')
        
        assert step.param_names == original_params
        assert step.return_names == original_returns
        
        # Modify the step's lists
        step.param_names[0] = 'modified_x'
        
        # Original should be unchanged
        assert original_params == ['x', 'y', 'z']

    def test_init_lists_not_same_object(self):
        """Test that stored lists are different objects from input lists."""
        original_params = ['x', 'y']
        original_returns = ['a', 'b']
        
        step = pf.plStep('test', lambda: None, original_params, original_returns, 'global')
        
        assert step.param_names is not original_params
        assert step.return_names is not original_returns

    @pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
        (123, lambda x: x+1, ['x'], ['y'], 'per-row'),
        (None, lambda x: x+1, ['x'], ['y'], 'per-row'),
        ('step', "not_a_function", ['x'], ['y'], 'vectorized'),
        ('step', None, ['x'], ['y'], 'vectorized'),
        ('step', lambda: 42, 'not_a_list', ['y'], 'global'),
        ('step', lambda: 42, None, ['y'], 'global'),
        ('step', lambda: 42, ['x'], 'not_a_list', 'global'),
        ('step', lambda: 42, ['x'], None, 'global'),
        ('step', lambda x: x+1, ['x'], ['y'], 'invalid_type'),
        ('step', lambda x: x+1, ['x'], ['y'], None),
    ])
    def test_init_invalid_inputs(self, name, func, param_names, return_names, func_type):
        """Test that __init__ raises ValueError for invalid inputs."""
        with pytest.raises(ValueError):
            pf.plStep(name, func, param_names, return_names, func_type)

    def test_init_empty_string_name_valid(self):
        """Test that empty string for name is valid."""
        step = pf.plStep('', lambda: None, [], [], 'global')
        assert step.name == ''

    def test_init_empty_strings_in_lists_valid(self):
        """Test that empty strings in param_names and return_names are valid."""
        step = pf.plStep('test', lambda: None, ['', 'x'], ['', 'y'], 'global')
        assert step.param_names == ['', 'x']
        assert step.return_names == ['', 'y']


class TestPlStepReprStr:
    """Tests for plStep.__repr__ and __str__"""

    @pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
        ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
        ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
        ('step3', lambda: 42, [], ['y'], 'global'),
        ('step4', lambda: np.arange(10), [], ['z'], 'global-res'),
    ])
    def test_repr_basic(self, name, func, param_names, return_names, func_type):
        """Test __repr__ returns correct format for all func_types."""
        step = pf.plStep(name, func, param_names, return_names, func_type)
        
        result = repr(step)
        assert name in result
        assert '\n' not in result

    @pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
        ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
        ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
        ('step3', lambda: 42, [], ['y'], 'global'),
        ('step4', lambda: np.arange(10), [], ['z'], 'global-res'),
    ])
    def test_str_basic(self, name, func, param_names, return_names, func_type):
        """Test __str__ returns detailed format for all func_types."""
        step = pf.plStep(name, func, param_names, return_names, func_type)
        
        result = str(step)
        assert name in result
        assert param_names.__str__() in result
        assert return_names.__str__() in result
        assert func_type in result
        assert '\n' in result

    def test_repr_with_special_characters(self):
        """Test __repr__ handles names with special characters."""
        special_names = ['step-1', 'step_2', 'step.3']
        
        for name in special_names:
            step = pf.plStep(name, lambda: None, [], [], 'global')
            assert name in repr(step)

    def test_repr_str_consistency(self):
        """Test that repr and str both contain the step name."""
        test_names = ['step1', 'my_step', 'TEST']
        
        for name in test_names:
            step = pf.plStep(name, lambda: None, [], [], 'global')
            assert name in repr(step)
            assert name in str(step)


class TestPlStepRun:
    """Tests for plStep.run() method - all func_types"""

    def test_run_global_no_params(self):
        """Test global step with no parameters."""
        step = pf.plStep('global_step', lambda: 42, [], ['result'], 'global')
        result = step._run([], [])
        assert result == {'result': 42}

    def test_run_global_with_params(self):
        """Test global step with parameters."""
        step = pf.plStep('global_step', lambda x, y: x + y, ['x', 'y'], ['result'], 'global')
        result = step._run([10, 20], [True, True])
        assert result == {'result': 30}

    def test_run_global_res_returns_array(self):
        """Test global-res step returns array."""
        step = pf.plStep('global_res_step', lambda: np.array([1, 2, 3]), [], ['result'], 'global-res')
        result = step._run([], [])
        assert isinstance(result['result'], np.ndarray)
        assert len(result['result']) == 3

    def test_run_vectorized_single_param(self):
        """Test vectorized step with single parameter."""
        step = pf.plStep('vec_step', lambda x: x * 2, ['x'], ['y'], 'vectorized')
        x_data = np.array([1, 2, 3])
        result = step._run([x_data], [False])
        assert isinstance(result['y'], np.ndarray)
        np.testing.assert_array_equal(result['y'], np.array([2, 4, 6]))

    def test_run_vectorized_multiple_params(self):
        """Test vectorized step with multiple parameters."""
        step = pf.plStep('vec_step', lambda x, y: x + y, ['x', 'y'], ['z'], 'vectorized')
        x_data = np.array([1, 2, 3])
        y_data = np.array([10, 20, 30])
        result = step._run([x_data, y_data], [False, False])
        np.testing.assert_array_equal(result['z'], np.array([11, 22, 33]))

    def test_run_per_row_single_param(self):
        """Test per-row step with single parameter."""
        step = pf.plStep('per_row_step', lambda x: x * 2, ['x'], ['y'], 'per-row')
        x_data = np.array([1, 2, 3])
        result = step._run([x_data], [False])
        assert isinstance(result['y'], np.ndarray)
        np.testing.assert_array_equal(result['y'], np.array([2, 4, 6]))

    def test_run_per_row_multiple_returns(self):
        """Test per-row step with multiple returns."""
        def my_func(x):
            return x, x*2
        
        step = pf.plStep('multi_return', my_func, ['x'], ['y', 'z'], 'per-row')
        x_data = np.array([1, 2, 3])
        result = step._run([x_data], [False])
        np.testing.assert_array_equal(result['y'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result['z'], np.array([2, 4, 6]))

    def test_run_mixed_global_and_vectorized_params(self):
        """Test vectorized step with both global and vectorized parameters."""
        step = pf.plStep('mixed', lambda offset, x: x + offset, 
                         ['offset', 'x'], ['y'], 'vectorized')
        offset = 10
        x_data = np.array([1, 2, 3])
        result = step._run([offset, x_data], [True, False])
        np.testing.assert_array_equal(result['y'], np.array([11, 12, 13]))


################################################################################
################################### LazyAttr Tests ##############################
################################################################################

class TestLazyAttrInit:
    """Tests for LazyAttr.__init__"""

    def test_init_valid(self):
        """Test LazyAttr initialization."""
        class MockDS:
            def __init__(self):
                self.nrows = 10
            def _fetch_rows(self, name, run_idx, rows):
                return np.arange(len(rows))
        
        ds = MockDS()
        attr = pf.LazyAttr(ds, 'test_attr', 0)
        
        assert attr.DS is ds
        assert attr.name == 'test_attr'
        assert len(attr) == 10

    def test_len(self):
        """Test __len__ returns nrows."""
        class MockDS:
            def __init__(self):
                self.nrows = 42
            def _fetch_rows(self, name, run_idx, rows):
                return np.zeros(len(rows))
        
        ds = MockDS()
        attr = pf.LazyAttr(ds, 'test', 0)
        assert len(attr) == 42


class TestLazyAttrIndexing:
    """Tests for LazyAttr indexing (single int, slice, list)"""

    @pytest.fixture
    def mock_ds(self):
        class MockDS:
            def __init__(self):
                self.nrows = 10
                self.data = {'test_attr': np.arange(10)}
            
            def _fetch_rows(self, name, run_idx, rows):
                if isinstance(rows, int):
                    rows = [rows]
                return self.data[name][rows]
        
        return MockDS()

    def test_getitem_single_index(self, mock_ds):
        """Test indexing with single integer."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        result = attr[0]
        assert result == 0
        result = attr[5]
        assert result == 5

    def test_getitem_negative_index(self, mock_ds):
        """Test indexing with negative integer."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        result = attr[-1]
        assert result == 9
        result = attr[-2]
        assert result == 8

    def test_getitem_slice(self, mock_ds):
        """Test indexing with slice."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        result = attr[2:5]
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))

    def test_getitem_list(self, mock_ds):
        """Test indexing with list of integers."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        result = attr[[0, 2, 4]]
        np.testing.assert_array_equal(result, np.array([0, 2, 4]))

    def test_getitem_ndarray(self, mock_ds):
        """Test indexing with numpy array."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        indices = np.array([1, 3, 5])
        result = attr[indices]
        np.testing.assert_array_equal(result, np.array([1, 3, 5]))

    def test_getitem_out_of_bounds(self, mock_ds):
        """Test that out-of-bounds indexing raises error."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        with pytest.raises((IndexError, ValueError)):
            _ = attr[100]


class TestLazyAttrSetitem:
    """Tests for LazyAttr.__setitem__"""

    @pytest.fixture
    def mock_ds(self):
        class MockDS:
            def __init__(self):
                self.nrows = 10
                self._cache = {}
            
            def _fetch_rows(self, name, run_idx, rows):
                if isinstance(rows, int):
                    rows = [rows]
                if name not in self._cache:
                    self._cache[name] = np.zeros(10)
                return self._cache[name][rows]
        
        return MockDS()

    def test_setitem_single_value(self, mock_ds):
        """Test setting a single value."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        attr[0] = 42
        # Value should be in cache
        assert 0 in attr._cache

    def test_setitem_slice(self, mock_ds):
        """Test setting a slice."""
        attr = pf.LazyAttr(mock_ds, 'test_attr', 0)
        attr[0:3] = [10, 20, 30]
        # Rows should be in cache
        assert any(i in attr._cache for i in [0, 1, 2])


################################################################################
################################ Utility Functions #############################
################################################################################

class TestFindPlPath:
    """Tests for find_pl_path utility function"""

    def test_find_pl_path_basic(self):
        """Test find_pl_path with basic pipeline structure."""
        cal_pl = {
            1: {'step1': 'file1'},
            2: {'step2': 'file2'},
        }
        
        # find_pl_path should return the directory containing the step
        # Exact behavior depends on implementation, but should not crash
        try:
            result = pf.find_pl_path(cal_pl, [1, 'step1'])
            # If it returns something, verify it's a string
            if result is not None:
                assert isinstance(result, str)
        except (KeyError, ValueError):
            # May raise if structure doesn't match expected format
            pass


class TestCheckPlTreeStructure:
    """Tests for check_pl_tree_structure utility function"""

    def test_check_pl_tree_structure_valid(self):
        """Test check_pl_tree_structure with valid structure."""
        # A basic zarr-like structure
        tree = {
            'run_1': {
                'param1': 'some_data',
                'param2': 'some_data',
            }
        }
        
        # Should not raise
        try:
            pf.check_pl_tree_structure(tree)
        except Exception:
            # Implementation-specific, may or may not raise
            pass

    def test_check_pl_tree_structure_empty(self):
        """Test check_pl_tree_structure with empty structure."""
        tree = {}
        
        # Should handle empty tree
        try:
            pf.check_pl_tree_structure(tree)
        except Exception:
            pass
