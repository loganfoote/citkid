import pytest
import numpy as np 
from citkid.pipeline import framework as pf 
from citkid.pipeline import dataset as dataset

# Dummy DataSet class 
class DummyDS():
    def __init__(self):
        self.cal_pl = {}
        self.nres = 10
        self.execute_path = lambda path, rows: None

class DummyDSWithExecute(dataset.DataSet):
    def __init__(self, cal_pl):
        self.cal_pl = cal_pl
        self.nres = 10

################################################################################
#################################### Steps #####################################
################################################################################
# __init__, __repr__, __str__
@pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
    ('step1', lambda x: x+1, ['x'], ['y'], 'per-row'),
    ('step2', lambda x: x*2, ['x'], ['y'], 'vectorized'),
    ('step3', lambda x: 42, [], ['y'], 'global'),
])
def test_steps_init(name, func, param_names, return_names, func_type):
    step = pf.plStep(name, func, param_names, return_names, func_type)
    # Check that attributes are set correctly
    assert step.name == name
    assert step.param_names == param_names
    assert step.return_names == return_names
    assert step.func_type == func_type
    # Check that func is set appropriately 
    assert step.func == func
    assert step.func(1) == func(1)
    assert step.func(10) == func(10)
    # Modify attributes to ensure copies were made
    if len(param_names):
        step.param_names[0] = 'modified'
        assert step.param_names != param_names
    if len(return_names):
        step.return_names[0] = 'modified'
        assert step.return_names != return_names
    # Test __repr__
    s = f"Pipeline Step: {name}"
    s += f", Function: {func.__module__}.{func.__name__}"
    assert repr(step) == s
    # Test __str__
    s = f"Pipeline Step: {name}"
    s += f"\n\tFunction: {func.__module__}.{func.__name__}"
    s += f"\n\tInput Parameters: {step.param_names}"
    s += f"\n\tOutput Parameters: {step.return_names}"
    s += f"\n\tFunction Type: {func_type}"
    assert str(step) == s

@pytest.mark.parametrize("name,func,param_names,return_names,func_type", [
    (123, lambda x: x+1, ['x'], ['y'], 'per-row'),  # name not str
    ('step2', "not_a_function", ['x'], ['y'], 'vectorized'), # func not callable
    ('step3', lambda: 42, 'not_a_list', ['y'], 'global'), # param_names not list
    ('step4', lambda: 42, ['a', 1], ['y'], 'global'),  # param_names not all str
    ('step5', lambda: 42, ['x'], ['a', 1], 'global'), # return_names not all str
    ('step4', lambda: 42, [], 'not_a_list', 'global'), # return_names not list
    ('step5', lambda x: x+1, ['x'], ['y'], 'invalid_type'), # invalid func_type
])
def test_steps_init_invalid(name, func, param_names, return_names, func_type):
    with pytest.raises(AssertionError):
        pf.plStep(name, func, param_names, return_names, func_type)

# test run method for different func_types
def test_steps_run_global():
    # same for global and global-res
    for func_type in 'global', 'global-res':
        DS = DummyDS()
        step = pf.plStep('global_step', lambda: 42, [], ['result'], func_type)
        
        # should be independent of data_idx 
        for data_idx in [None, 0, 1, [1,2,3]]:
            step.run(DS, data_idx = data_idx) 
            assert DS.result == 42 
            del DS.result 

        # Should take inputs successfully 
        step.param_names = ['x']
        step.func = lambda x: x
        for x in [100, [100, 200, 300]]:
            DS.x = x 
            step.run(DS) 
            assert DS.result == x 
            del DS.result

def test_steps_run_global_bad_input():
    for func_type in ['global', 'global-res']:
        DS = DummyDS()
        # Missing DS attribute
        step = pf.plStep('global_step', lambda x: x + 3, 
                         ['x'], ['result'], func_type)
        with pytest.raises(AttributeError):
            step.run(DS)

        # Input of wrong length should raise error (no vectorized support)
        DS.x = [1,2,3]
        with pytest.raises(TypeError):
            step.run(DS)

        # Wrong number of parameters
        DS.y = 5
        step = pf.plStep('global_step', lambda x: x * 3, 
                         ['x', 'y'], ['result'], func_type)
        with pytest.raises(TypeError):
            step.run(DS)
        step = pf.plStep('global_step', lambda x: x * 3, 
                         [], ['result'], func_type)
        with pytest.raises(TypeError):
            step.run(DS)
        step = pf.plStep('global_step', lambda x: x * 3, 
                         ['x'], [], func_type)
        with pytest.raises(ValueError):
            step.run(DS)

        step = pf.plStep('global_step', lambda x: x * 3, 
                         ['x'], ['result1', 'result2'], func_type)
        with pytest.raises(ValueError):
            step.run(DS)

def test_steps_run_vectorized():
    DS = DummyDS()
    step = pf.plStep('vectorized_step', lambda x: np.asarray(x) * 2, 
                     ['x'], ['result'], 'vectorized')
    
    # Test with list input
    DS.x = [1, 2, 3, 4]
    data_idx = [0, 1, 2, 3]
    step.run(DS, data_idx = data_idx) 
    assert type(DS.result) == pf.LazyAttr 
    assert np.array_equal(DS.result[data_idx], [2, 4, 6, 8])
    del DS.result

    # Test with numpy array input
    DS.x = np.array([10, 20, 30])
    step.run(DS, data_idx = [0, 1, 2])
    assert False, 'Error in LazyAttr._ensure_loaded, need to fix that first'
    assert type(DS.result) == pf.LazyAttr 
    assert np.array_equal(DS.result[data_idx], np.array([20, 40, 60]))
    # del DS.result

    # different order of data_idx
    # DS.x = np.array([10, 20, 30])
    # step.run(DS, data_idx = [0, 2, 1])
    # assert type(DS.result) == pf.LazyAttr 
    # assert np.array_equal(DS.result[[0, 1, 2]], np.array([20, 40, 60]))
    # del DS.result

    # # Test with single index
    # DS.x = [5, 10, 15]
    # step.run(DS, data_idx = 1)
    # assert type(DS.result) == pf.LazyAttr
    # assert DS.result[1] == 20 

    # # Test with all indices
    # DS.x = [3, 6, 9, 12]
    # DS.nres = 4
    # step.run(DS, data_idx = None)
    # assert type(DS.result) == pf.LazyAttr
    # assert np.array_equal(DS.result[[0, 1, 2, 3]], np.array([6, 12, 18, 24]))
    # step.run(DS, data_idx = [1, 2, 3, 0])
    # assert type(DS.result) == pf.LazyAttr
    # assert np.array_equal(DS.result[[1, 2, 3, 0]], np.array([24, 6, 12, 18]))


def test_steps_vectorized_bad_input():
    pass



        

################################################################################
################################### LazyAttr ###################################
################################################################################
# __init__, __repr__, __str__
DS = DummyDS()

def test_lazyattr_init():
    name = 'test_attr'
    LA = pf.LazyAttr(DS, name)
    assert LA.DS == DS  
    assert LA.name == name
    assert LA._cache == {}
    r = f"LazyAttr({name}, 0 cached rows)" 
    s = f"Lazy Attribute: {name}\n\tCached Rows: []"
    assert repr(LA) == r, "__repr__ output incorrect"
    assert str(LA) == s, "__str__ output incorrect"
    LA._cache = {i: i*2 for i in range(10)}
    r = f"LazyAttr({name}, 10 cached rows)" 
    s = f"Lazy Attribute: {name}\n\tCached Rows: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
    assert repr(LA) == r, "__repr__ output incorrect with cached rows"
    assert str(LA) == s, "__str__ output incorrect with cached rows"

def test_lazyattr_init_invalid():
    with pytest.raises(AssertionError): # incorrect DS datatype
        pf.LazyAttr("not_a_dataset", 'test_attr')
    with pytest.raises(AssertionError): # incorrect name datatype
        pf.LazyAttr(DS, 123)

# _ensure_loaded 
step1 = {'task': pf.plStep('step1', lambda x: x + 1, 
                           ['data_idx'], ['b'], 'per-row')}
step2 = {'task': pf.plStep('step2', lambda x: x + 1, 
                           ['b'], ['c'], 'per-row')}
cal_pl = {'CAL_STEPS': {1: step1, 2: step2}}
def test_lazyattr_ensure_loaded():
    # assert False, "working on this"
    DS = DummyDSWithExecute(cal_pl) 
    rows = [1, 2] 
    DS.c._ensure_loaded(rows) # c and b are created, data_idx 1, 2 are cached
    assert isinstance(DS.b, pf.LazyAttr)
    assert isinstance(DS.c, pf.LazyAttr)
    assert DS.c._cache == {1: 3, 2: 4}
    assert DS.b._cache == {1: 2, 2: 3}
    DS.c._ensure_loaded(4)
    assert DS.c._cache == {1: 3, 2: 4, 4: 6}
    assert DS.b._cache == {1: 2, 2: 3, 4: 5}

def test_lazyattr_ensure_loaded_invalid():
    DS = DummyDSWithExecute(cal_pl) 
    LA = pf.LazyAttr(DS, 'c')
    with pytest.raises(ValueError): # rows not list or int
        LA._ensure_loaded("not_a_list_or_int")
    with pytest.raises(ValueError): # rows contain non-int
        LA._ensure_loaded([1, 'a', 3])
    with pytest.raises(AssertionError): # rows out of bounds
        LA._ensure_loaded([-15, 2])
    with pytest.raises(AssertionError): # rows out of bounds
        LA._ensure_loaded([0, 10])
    with pytest.raises(AttributeError):
        DS.d._ensure_loaded([1,2]) # no path to d

# __setitem__
@pytest.mark.parametrize("rows,values,expected_cache", [
    (slice(0, 2), [10, 20], {0: 10, 1: 20}), # slice
    ([2, 4, 6], [30, 50, 70], {2: 30, 4: 50, 6: 70}), # list of idx
    (np.array([3, 5], dtype = np.int32), [40, 60], 
     {3: 40, 5: 60}), # np.ndarray of idx
    (0, 5, {0: 5}), # single index
    ([1, 2, 3], 10, {1: 10, 2: 10, 3: 10}), # multiple idx set to single value
    (slice(1, 3), 10, {1: 10, 2: 10}), # multiple idx set to single value)
    (-1, 100, {9: 100}), # negative index 
    ([0, -2], [1, 2], {0: 1, 8: 2}), # negative index in list 
    (slice(-3, None), [7, 8, 9], {7: 7, 8: 8, 9: 9}), # negative slice 
    (slice(0, None), np.arange(10), {i: i for i in range(10)}), # full slice 
    (slice(0, 4, 2), [11, 13], {0: 11, 2: 13}), # slice with step   
])
def test_lazyattr_setitem(rows, values, expected_cache):
    LA = pf.LazyAttr(DS, 'test_attr')
    # slice 
    LA[rows] = values
    assert LA._cache == expected_cache

def test_lazy_attr_tuple():
    LA = pf.LazyAttr(DS, 'test_attr')
    LA[0] = [1, 2] 
    LA[0, 1] = 3  
    k = list(LA._cache.keys())
    assert k == [0]
    assert np.array_equal(LA._cache[0], [1, 3]) 
    # 2D array 
    LA[0] = np.array([[0], [1], [2]]) 
    LA[0, 1, 0] = 3 
    k = list(LA._cache.keys())
    assert k == [0] 
    assert np.array_equal(LA._cache[0], np.array([[0], [3], [2]]))

@pytest.mark.parametrize("rows,values", [
    ("invalid", [10, 20]), # invalid rows type
    ([1, 'a', 3], [10, 20, 30]), # non-int in rows
    ([0, 10], [10, 20]), # row out of bounds
    ([1, 2], [10, 20, 30]), # values length does not match rows length
    (slice(0, 3), [10, 20]), # values length does not match rows length
    ((0, 1), [10, 20]), # tuple called when internal array isn't initialized
])
def test_lazyattr_setitem_invalid(rows, values):
    LA = pf.LazyAttr(DS, 'test_attr')
    with pytest.raises((AssertionError, ValueError, TypeError, AttributeError)):
        LA[rows] = values

# __getitem__
def test_lazyattr_getitem():
    DS = DummyDSWithExecute(cal_pl) 
    DS.c = pf.LazyAttr(DS, 'c')
    # set up LazyAttr with some cached values
    LA = pf.LazyAttr(DS, 'test_attr')
    LA._cache = {0: 10, 1: 20, 2: 30, 3: 40, 4: 50}
    assert False, 'Working on this'
    # get single index

    # get list of indices

    # get list with repeated indices

    # Ensure cached item is not recomputed
    


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
])
def test_find_pl_path(tree, return_name, expected_path):
    path = pf.find_pl_path(tree, return_name)
    if expected_path is None:
        assert path is None
    else:
        assert path is not None
        assert [step.name for step in path] == expected_path 

# don't need to check invalid tree input, handled in check_pl_tree_structure
def test_find_pl_path_invalid_input():
    with pytest.raises(AssertionError):
        pf.find_pl_path({}, 123) # name not str
    with pytest.raises(ValueError):
        pf.find_pl_path({1: step1}, 'test') # root keys must end in '_STEPS'

################################################################################
########################### check_pl_tree_structure ############################
################################################################################
@pytest.mark.parametrize("path", [
    {},
    {'CAL_STEPS': {1: step1}},
    {'CAL_STEPS': {1: step1, 
                   2: {'task': step2['task'], 'delete_input': ['b']}}}
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
    {'task': step1['task'], 'CAL_steps': {}} # invalid step list name

]) 
def test_check_pl_tree_structure_invalid(path):
    with pytest.raises(ValueError):
        pf.check_pl_tree_structure(path)

# find_pl_path
# print_pl_path -> don't really need to test this 
