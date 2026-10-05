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
            def _fetch_rows(self, name, rows):
                return np.arange(len(rows))
        
        ds = MockDS()
        attr = pf.LazyAttr(ds, 'test_attr')
        
        assert attr.DS is ds
        assert attr.name == 'test_attr'
        assert len(attr) == 10

    def test_len(self):
        """Test __len__ returns nrows."""
        class MockDS:
            def __init__(self):
                self.nrows = 42
            def _fetch_rows(self, name, rows):
                return np.zeros(len(rows))
        
        ds = MockDS()
        attr = pf.LazyAttr(ds, 'test')
        assert len(attr) == 42


class TestLazyAttrIndexing:
    """Tests for LazyAttr indexing (single int, slice, list)"""

    @pytest.fixture
    def mock_ds(self):
        class MockDS:
            def __init__(self):
                self.nrows = 10
                self.data = {'test_attr': np.arange(10)}
            
            def _fetch_rows(self, name, rows):
                if isinstance(rows, int):
                    rows = [rows]
                return self.data[name][rows]
        
        return MockDS()

    def test_getitem_single_index(self, mock_ds):
        """Test indexing with single integer."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        result = attr[0]
        assert result == 0
        result = attr[5]
        assert result == 5

    def test_getitem_negative_index(self, mock_ds):
        """Test indexing with negative integer."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        result = attr[-1]
        assert result == 9
        result = attr[-2]
        assert result == 8

    def test_getitem_slice(self, mock_ds):
        """Test indexing with slice."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        result = attr[2:5]
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))

    def test_getitem_list(self, mock_ds):
        """Test indexing with list of integers."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        result = attr[[0, 2, 4]]
        np.testing.assert_array_equal(result, np.array([0, 2, 4]))

    def test_getitem_ndarray(self, mock_ds):
        """Test indexing with numpy array."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        indices = np.array([1, 3, 5])
        result = attr[indices]
        np.testing.assert_array_equal(result, np.array([1, 3, 5]))

    def test_getitem_out_of_bounds(self, mock_ds):
        """Test that out-of-bounds indexing raises error."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        with pytest.raises((IndexError, ValueError)):
            _ = attr[100]

    def test_getitem_uses_cache_for_repeated_rows(self):
        """Repeated row access should not fetch rows that are already cached."""
        class MockDS:
            def __init__(self):
                self.nrows = 6
                self.data = {'test_attr': np.arange(6)}
                self.fetch_calls = []

            def _fetch_rows(self, name, rows):
                rows = list(rows)
                self.fetch_calls.append(rows)
                return self.data[name][rows]

        mock_ds = MockDS()
        attr = pf.LazyAttr(mock_ds, 'test_attr')

        np.testing.assert_array_equal(attr[[1, 3]], np.array([1, 3]))
        np.testing.assert_array_equal(attr[[3, 4]], np.array([3, 4]))

        assert mock_ds.fetch_calls == [[1, 3], [4]]


class TestLazyAttrSetitem:
    """Tests for LazyAttr.__setitem__"""

    @pytest.fixture
    def mock_ds(self):
        class MockDS:
            def __init__(self):
                self.nrows = 10
                self._cache = {}
            
            def _fetch_rows(self, name, rows):
                if isinstance(rows, int):
                    rows = [rows]
                if name not in self._cache:
                    self._cache[name] = np.zeros(10)
                return self._cache[name][rows]
        
        return MockDS()

    def test_setitem_single_value(self, mock_ds):
        """Test setting a single value."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        attr[0] = 42
        # Value should be in cache
        assert 0 in attr._cache

    def test_setitem_slice(self, mock_ds):
        """Test setting a slice."""
        attr = pf.LazyAttr(mock_ds, 'test_attr')
        attr[0:3] = [10, 20, 30]
        # Rows should be in cache
        assert any(i in attr._cache for i in [0, 1, 2])


################################################################################
################################ Utility Functions #############################
################################################################################

def _tree_steps():
    """
    Build a small calibration tree with a nested sequence.

    Returns:
    tree (dict): tree with ``CAL_STEPS`` -> load (with ``X_STEPS`` child,
        sib) -> after.
    steps (dict): plStep instances by name.
    """
    steps = {
        'load': pf.plStep('load', lambda: 1, [], ['a'], 'global'),
        'child': pf.plStep('child', lambda a: a, ['a'], ['b'], 'per-row'),
        'sib': pf.plStep('sib', lambda a: a, ['a'], ['c'], 'per-row'),
        'after': pf.plStep('after', lambda a: a, ['a'], ['d'], 'per-row'),
    }
    tree = {'CAL_STEPS': {
        1: {'task': steps['load'], 'X_STEPS': {
            1: {'task': steps['child']},
            2: {'task': steps['sib']},
        }},
        2: {'task': steps['after']},
    }}
    return tree, steps


@pytest.mark.parametrize('name, expected', [
    ('a', ['load']),
    ('b', ['load', 'child']),
    ('c', ['load', 'child', 'sib']),
    ('d', ['load', 'after']),
])
def test_find_pl_path_returns_steps_in_execution_order(name, expected):
    """
    Check that the path includes ancestors and earlier siblings in order.
    """
    tree, steps = _tree_steps()
    path = pf.find_pl_path(tree, name)
    assert [step.name for step in path] == expected
    assert all(step is steps[step.name] for step in path)


def test_find_pl_path_unknown_output_returns_none():
    """
    Check that an output no step produces gives None.
    """
    tree, _ = _tree_steps()
    assert pf.find_pl_path(tree, 'missing') is None


@pytest.mark.parametrize('tree, name', [
    ({'CAL_STEPS': {}}, ['a']),
    ({'CAL': {}}, 'a'),
])
def test_find_pl_path_rejects_bad_input(tree, name):
    """
    Check that a non-string name or a root key without _STEPS raises.
    """
    with pytest.raises(ValueError):
        pf.find_pl_path(tree, name)


def test_check_pl_tree_structure_accepts_valid_trees():
    """
    Check that valid calibration and analysis trees pass.
    """
    tree, steps = _tree_steps()
    pf.check_pl_tree_structure(tree, cal=True)
    pf.check_pl_tree_structure({})
    analysis = {'ANALYSIS_STEPS': {1: {
        'task': steps['child'], 'params': {'k': 1.5, 'm': None},
        'delete_input': ['a'],
    }}}
    pf.check_pl_tree_structure(analysis, cal=False)


def _bad_trees():
    """
    Build invalid trees, each paired with a substring of its error message.

    Returns:
    cases (list): tuples (tree, cal, match).
    """
    _, s = _tree_steps()
    node = {'task': s['child']}
    return [
        ({'CAL_STEPS': {1: node, 3: node}}, False, 'start from 1'),
        ({'CAL_STEPS': {1: node, 'X': node}}, False, 'other key types'),
        ({'CAL_STEPS': {1: {'params': {}}}}, False, '"task" key'),
        ({'CAL_STEPS': {1: {'task': 'child'}}}, False, 'not a valid'),
        ({'CAL_STEPS': {1: {'task': s['child'], 'parms': {}}}}, False,
         'invalid key: parms'),
        ({'CAL_STEPS': {1: {'task': s['child'], 'params': {}}}}, True,
         'not allowed'),
        ({'CAL_STEPS': {1: {'task': s['child'], 'params': {'k': [1]}}}},
         False, 'simple type'),
        ({'CAL_STEPS': {1: {'task': s['child'], 'delete_input': ['z']}}},
         False, 'not found'),
        ({'CAL_STEPS': {1: {'task': s['child'], 'delete_input': 'some'}}},
         False, 'not a valid list'),
        ([], False, 'must be a dict'),
    ]


@pytest.mark.parametrize('case', range(len(_bad_trees())))
def test_check_pl_tree_structure_rejects_invalid_trees(case):
    """
    Check that each structural error raises with a descriptive message.
    """
    tree, cal, match = _bad_trees()[case]
    with pytest.raises(ValueError, match=match):
        pf.check_pl_tree_structure(tree, cal=cal)


@pytest.mark.parametrize("func", [
    lambda x: {"a": x, "b": x},     # a dict is a single output
    lambda x: (x, x, x),            # too many outputs
    lambda x: x,                    # too few outputs
])
def test_per_row_step_with_wrong_output_count_raises(func):
    """A per-row step must return exactly one value per return name."""
    step = pf.plStep("bad", func, ["x"], ["a", "b"], "per-row")

    with pytest.raises(ValueError, match="returned .* output"):
        step._run([np.array([1.0, 2.0])], [False])
