import pytest
import numpy as np 
from citkid.pipeline import framework as pf 

################################################################################
################################### LazyAttr ###################################
################################################################################
# __init__, __repr__, __str__
class DummyDS(pf.dataset):
        def __init__(self):
            self.cal_pl = {}
            self.nres = 10
            self.execute_path = lambda path, rows: None

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
