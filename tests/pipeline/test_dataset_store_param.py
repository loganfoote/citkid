"""
Tests for DataSet._store_param method.

This file tests the _store_param helper method which centralizes parameter 
storage logic for both global and per-row parameters.
"""

import pytest
import numpy as np
import citkid.pipeline.dataset as pds


################################################################################
# Input validation tests
################################################################################

def test_store_param_name_must_be_string():
    """Test _store_param raises TypeError if name is not a string."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    with pytest.raises(TypeError, match="name must be a string"):
        DS._store_param(123, 'value', 1, {}, True)


def test_store_param_deps_must_be_dict():
    """Test _store_param raises TypeError if deps is not a dict."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    with pytest.raises(TypeError, match="deps must be a dictionary"):
        DS._store_param('param', 'value', 1, ['not', 'dict'], True)


def test_store_param_is_global_must_be_bool():
    """Test _store_param raises TypeError if is_global is not bool."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    with pytest.raises(TypeError, match="is_global must be a boolean"):
        DS._store_param('param', 'value', 1, {}, 'not_bool')


def test_store_param_reserved_name_collision():
    """Test _store_param raises ValueError for reserved attribute names."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Try to store parameter with reserved name
    with pytest.raises(ValueError, match="reserved DataSet attribute name"):
        DS._store_param('write_data', 'value', 1, {}, True)


################################################################################
# Global parameter tests
################################################################################

def test_store_param_global_basic():
    """Test storing a basic global parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    DS._store_param('param', 42, 1, {'dep1': 1}, True)
    
    # Check memory cache
    assert 1 in DS._memory_cache
    assert 'param' in DS._memory_cache[1]
    assert DS._memory_cache[1]['param'] == 42
    
    # Check deps_maps
    assert 'global' in DS.deps_maps
    assert 1 in DS.deps_maps['global']
    assert 'param' in DS.deps_maps['global'][1]
    assert DS.deps_maps['global'][1]['param'] == {'dep1': 1}
    
    # Check is_global_cache
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == True


def test_store_param_global_with_array():
    """Test storing a global parameter with array value."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    value = np.array([1, 2, 3, 4, 5])
    DS._store_param('param', value, 1, {}, True)
    
    assert np.array_equal(DS._memory_cache[1]['param'], value)


def test_store_param_global_data_idx_must_be_none():
    """Test global parameter requires data_idx=None."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    with pytest.raises(ValueError, match="data_idx must be None for global"):
        DS._store_param('param', 42, 1, {}, True, data_idx=0)


def test_store_param_global_cannot_overwrite():
    """Test cannot overwrite existing global parameter at same run."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param': 42}}
    DS.deps_maps = {'global': {1: {'param': {}}}}
    DS._is_global_cache = {'param': True}
    DS._lazy_collections = {}
    
    with pytest.raises(ValueError, match="Attempting to overwrite"):
        DS._store_param('param', 99, 1, {}, True)


def test_store_param_global_different_runs():
    """Test can store same global parameter at different runs."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    DS._store_param('param', 42, 1, {}, True)
    DS._store_param('param', 99, 2, {}, True)
    
    assert DS._memory_cache[1]['param'] == 42
    assert DS._memory_cache[2]['param'] == 99


################################################################################
# Per-row parameter tests
################################################################################

def test_store_param_per_row_basic():
    """Test storing a basic per-row parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    data_idx = np.array([0, 1, 2])
    values = np.array([10, 20, 30])
    deps = {'dep1': 1}
    
    DS._store_param('param', values, 1, deps, False, data_idx=data_idx)
    
    # Check memory cache has LazyAttr
    assert 1 in DS._memory_cache
    assert 'param' in DS._memory_cache[1]
    assert isinstance(DS._memory_cache[1]['param'], pds.pf.LazyAttr)
    
    # Check LazyAttr has cached values
    lazy_attr = DS._memory_cache[1]['param']
    assert lazy_attr._cache[0] == 10
    assert lazy_attr._cache[1] == 20
    assert lazy_attr._cache[2] == 30
    
    # Check deps_maps for each data_idx
    for di in data_idx:
        assert di in DS.deps_maps
        assert 1 in DS.deps_maps[di]
        assert 'param' in DS.deps_maps[di][1]
        assert DS.deps_maps[di][1]['param'] == deps
    
    # Check is_global_cache
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == False
    
    # Check LazyAttrCollection registration
    assert 'param' in DS._lazy_collections
    assert isinstance(DS._lazy_collections['param'], pds.pf.LazyAttrCollection)


def test_store_param_per_row_requires_data_idx():
    """Test per-row parameter requires data_idx."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    with pytest.raises(ValueError, match="data_idx required for per-row"):
        DS._store_param('param', [1, 2, 3], 1, {}, False, data_idx=None)


def test_store_param_per_row_length_mismatch():
    """Test per-row parameter requires value length to match data_idx."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    with pytest.raises(ValueError, match="Length mismatch"):
        DS._store_param('param', [1, 2, 3], 1, {}, False, data_idx=[0, 1])


def test_store_param_per_row_multiple_calls_same_run():
    """Test can store per-row parameter multiple times at same run."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # First call: store indices 0, 1
    DS._store_param('param', [10, 20], 1, {}, False, data_idx=[0, 1])
    
    # Second call: store indices 2, 3
    DS._store_param('param', [30, 40], 1, {}, False, data_idx=[2, 3])
    
    # Check all values are stored
    lazy_attr = DS._memory_cache[1]['param']
    assert lazy_attr._cache[0] == 10
    assert lazy_attr._cache[1] == 20
    assert lazy_attr._cache[2] == 30
    assert lazy_attr._cache[3] == 40


def test_store_param_per_row_different_runs():
    """Test can store per-row parameter at different runs."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Run 1
    DS._store_param('param', [10, 20], 1, {'dep': 1}, False, data_idx=[0, 1])
    
    # Run 2
    DS._store_param('param', [30, 40], 2, {'dep': 2}, False, data_idx=[0, 1])
    
    # Check both runs exist
    assert 1 in DS._memory_cache
    assert 2 in DS._memory_cache
    assert DS._memory_cache[1]['param']._cache[0] == 10
    assert DS._memory_cache[2]['param']._cache[0] == 30
    
    # Check LazyAttrCollection tracks both runs
    assert 'param' in DS._lazy_collections
    collection = DS._lazy_collections['param']
    assert 1 in collection._lazy_attrs
    assert 2 in collection._lazy_attrs


def test_store_param_per_row_with_multidim_arrays():
    """Test storing per-row parameter with multi-dimensional values."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Store 2D arrays for each row
    data_idx = np.array([0, 1, 2])
    values = np.array([[1, 2], [3, 4], [5, 6]])
    
    DS._store_param('param', values, 1, {}, False, data_idx=data_idx)
    
    lazy_attr = DS._memory_cache[1]['param']
    assert np.array_equal(lazy_attr._cache[0], [1, 2])
    assert np.array_equal(lazy_attr._cache[1], [3, 4])
    assert np.array_equal(lazy_attr._cache[2], [5, 6])


################################################################################
# Integration tests
################################################################################

def test_store_param_mixed_global_and_per_row():
    """Test storing both global and per-row parameters."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Store global parameter
    DS._store_param('global_param', 42, 1, {}, True)
    
    # Store per-row parameter
    DS._store_param('per_row_param', [10, 20], 1, {}, False, data_idx=[0, 1])
    
    # Both should coexist in same run
    assert 'global_param' in DS._memory_cache[1]
    assert 'per_row_param' in DS._memory_cache[1]
    
    # Check types
    assert DS._is_global_cache['global_param'] == True
    assert DS._is_global_cache['per_row_param'] == False
    
    # Check deps_maps structure
    assert 'global' in DS.deps_maps
    assert 0 in DS.deps_maps
    assert 1 in DS.deps_maps


def test_store_param_lazy_attr_persists():
    """Test LazyAttr persists and can be accessed after storage."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    data_idx = np.array([0, 1, 2])
    values = np.array([10, 20, 30])
    
    DS._store_param('param', values, 1, {}, False, data_idx=data_idx)
    
    # Access via LazyAttr indexing
    lazy_attr = DS._memory_cache[1]['param']
    assert lazy_attr[0] == 10
    assert lazy_attr[1] == 20
    assert lazy_attr[2] == 30
    
    # Access via LazyAttrCollection
    collection = DS._lazy_collections['param']
    assert collection[0] == 10
    assert collection[1] == 20
    assert collection[2] == 30


def test_store_param_run_idx_conversion():
    """Test run_idx is converted to int."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Pass run_idx as float
    DS._store_param('param', 42, 1.0, {}, True)
    
    # Should be stored with int key
    assert 1 in DS._memory_cache
    assert isinstance(list(DS._memory_cache.keys())[0], int)


def test_store_param_data_idx_array_conversion():
    """Test data_idx is converted to numpy array."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Pass data_idx as list
    DS._store_param('param', [10, 20], 1, {}, False, data_idx=[0, 1])
    
    # Should work correctly
    lazy_attr = DS._memory_cache[1]['param']
    assert lazy_attr._cache[0] == 10
    assert lazy_attr._cache[1] == 20
