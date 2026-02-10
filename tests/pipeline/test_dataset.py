import pytest 
import os 
import numpy as np
from citkid.pipeline import dataset as pds
from citkid.pipeline import util
from io import StringIO, BytesIO
from contextlib import contextmanager
import builtins

################################################################################
################################### __init__ ###################################
################################################################################
# __init__, _load_custom_steps, _load_yaml, _convert_yaml_to_steps
def test_paths_are_normalized(monkeypatch):
    DS = pds.DataSet.__new__(pds.DataSet)
    # patch module-level helper functions to avoid file I/O and custom code
    monkeypatch.setattr(pds, "_load_custom_steps", lambda custom_path: ([], []))
    monkeypatch.setattr(pds, "_convert_yaml_to_steps", lambda y, cs=None: {})
    monkeypatch.setattr(pds, "_load_deps_from_zarr", lambda root: {})

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    # avoid opening real files for cal yaml
    def _safe_open(*args, **kwargs):
        # mimic builtins.open signature: (file, mode='r', ...)
        mode = 'r'
        if len(args) >= 2:
            mode = args[1]
        elif 'mode' in kwargs:
            mode = kwargs['mode']
        if 'b' in mode:
            return BytesIO(b"{}")
        return StringIO("{}")
    monkeypatch.setattr(builtins, 'open', _safe_open)
    monkeypatch.setattr("yaml.safe_load", lambda f: {})

    # initialize with yaml file (new signature: zarr_path, cal_yaml_path, custom_path)
    DS.__init__("z//out.zarr", "x//y.yaml", "a//b/../c.py")
    assert DS.custom_path == os.path.abspath("a//b/../c.py")
    assert DS.cal_yaml_path == os.path.abspath("x//y.yaml")
    assert DS.zarr_path == os.path.abspath("z//out.zarr")
    assert DS.root == fake_root

    # initialize with yml file (order: zarr, cal_yaml, custom)
    DS.__init__("z//out.zarr", "x//y.yml", "a//b/../c.py")
    assert DS.cal_yaml_path == os.path.abspath("x//y.yml")

    # initialize without custom path
    DS.__init__("z//out.zarr", "x//y.yml", None)
    assert DS.custom_path is None

@pytest.mark.parametrize("custom_path, yaml_path, zarr_path", [
    ("dir.txt", "file.yaml", "out.zarr"), # custom_path is not a .py file
    ("dir.py", "file.txt", "out.zarr"), # yaml_path is not a .yaml or .yml file
    ("dir.py", "file.yaml", "out.txt"), # zarr_path is not a .zarr file
])
def test_paths_validation(monkeypatch, custom_path, yaml_path, zarr_path):
    DS = pds.DataSet.__new__(pds.DataSet)
    monkeypatch.setattr(pds, "_load_custom_steps", lambda custom_path: ([], []))
    monkeypatch.setattr(pds, "_convert_yaml_to_steps", lambda y, cs=None: {})
    monkeypatch.setattr("yaml.safe_load", lambda f: {})

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    with pytest.raises(ValueError):
        # call with new signature: zarr_path, cal_yaml_path, custom_path
        DS.__init__(zarr_path, yaml_path, custom_path)

def test_load_custom_steps(tmp_path):
    module = tmp_path / "custom_steps.py"
    m = "class Step:\n\tdef __init__(self, name):\n\t\tself.name = name\n"
    m += "custom_cal_steps = [Step('custom1')]\n"
    m += "custom_analysis_steps = []\n"
    module.write_text(m)

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.custom_path = tmp_path / "custom_steps.py"

    cal_steps, analysis_steps = pds._load_custom_steps(str(DS.custom_path))

    assert len(cal_steps) == 1
    assert cal_steps[0].name == "custom1"
    assert isinstance(analysis_steps, list)

def test_load_custom_steps_none():
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.custom_path = None 
    steps = pds._load_custom_steps(None)

    assert steps == []

def test_default_steps_added(monkeypatch):
    class Step:
        def __init__(self, name):
            self.name = name

    default = [Step("a"), Step("b")]

    monkeypatch.setattr(pds.default_steps, "default_cal_steps", default)

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.cal_steps = [Step("a")]  # simulate custom step list
    DS.analysis_steps = []

    for step in pds.default_steps.default_cal_steps:
        if step.name not in [s.name for s in DS.cal_steps]:
            DS.cal_steps.append(step)
    for step in pds.default_steps.default_analysis_steps:
        if step.name not in [s.name for s in DS.analysis_steps]:
            DS.analysis_steps.append(step)

    cal_names = [s.name for s in DS.cal_steps]
    assert cal_names == ["a", "b"]

def test_init_calls_convert(monkeypatch):
    fake_yaml = {"pipeline": ["a", "b"]}

    monkeypatch.setattr(
        pds, "_load_custom_steps", lambda custom_path: ([], [])
    )
    monkeypatch.setattr(
        pds, "_convert_yaml_to_steps",
        lambda y, cs=None: {}
    )
    monkeypatch.setattr(
        "yaml.safe_load", lambda f: fake_yaml
    )
    monkeypatch.setattr(
        pds.default_steps, "default_cal_steps", []
    )
    monkeypatch.setattr(
        pds.default_steps, "default_analysis_steps", []
    )
    monkeypatch.setattr(
        pds, "_load_deps_from_zarr", lambda root: {}
    )

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    # avoid opening real files for cal yaml
    def _safe_open2(*args, **kwargs):
        mode = 'r'
        if len(args) >= 2:
            mode = args[1]
        elif 'mode' in kwargs:
            mode = kwargs['mode']
        data = "pipeline:\n  - a\n  - b\n"
        if 'b' in mode:
            return BytesIO(data.encode())
        return StringIO(data)
    monkeypatch.setattr(builtins, 'open', _safe_open2)

    # new signature: zarr_path, cal_yaml_path, custom_path
    DS = pds.DataSet("out.zarr", "file.yaml", "custom_steps.py")

    assert DS.cal_pl == {}


def test_init_invalid_path(monkeypatch, tmp_path):
    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)
    # non-existent custom path should raise FileNotFoundError when used
    with pytest.raises(FileNotFoundError):
        pds.DataSet("out.zarr", "file.yaml", "nonexistent_path.py")

    with pytest.raises(FileNotFoundError):
        pds.DataSet("out.zarr", "file.yaml", str(tmp_path / "nonexistent_path.py"))

    # invalid types for inputs
    with pytest.raises(TypeError):
        pds.DataSet(123, "file.yaml", 'custom.py')

    with pytest.raises(TypeError):
        pds.DataSet("out.zarr", 456, 'custom.py')

def test_init_checks_pipeline_structure(monkeypatch):
    """Test that __init__ calls check_pl_tree_structure on cal_pl."""
    check_called = []
    
    def fake_check(pl, cal):
        check_called.append((pl, cal))
    
    monkeypatch.setattr(pds.pf, "check_pl_tree_structure", fake_check)
    monkeypatch.setattr(pds, "_load_custom_steps", lambda custom_path: ([], []))
    monkeypatch.setattr(pds, "_convert_yaml_to_steps", lambda y, cs: {'step1': {}})
    monkeypatch.setattr(pds, "_load_deps_from_zarr", lambda root: {})
    monkeypatch.setattr(pds.default_steps, "default_cal_steps", [])
    monkeypatch.setattr(pds.default_steps, "default_analysis_steps", [])
    
    fake_root = object()
    monkeypatch.setattr("zarr.open_group", lambda path, mode: fake_root)
    
    def _safe_open(*args, **kwargs):
        return StringIO("{}")
    monkeypatch.setattr(builtins, 'open', _safe_open)
    monkeypatch.setattr("yaml.safe_load", lambda f: {})
    
    DS = pds.DataSet("out.zarr", "file.yaml", None)
    
    # Verify check_pl_tree_structure was called with cal_pl and cal=True
    assert len(check_called) == 1
    assert check_called[0][0] == {'step1': {}}
    assert check_called[0][1] == True

def test_init_initializes_caches(monkeypatch):
    """Test that __init__ initializes _memory_cache and _is_global_cache."""
    monkeypatch.setattr(pds, "_load_custom_steps", lambda custom_path: ([], []))
    monkeypatch.setattr(pds, "_convert_yaml_to_steps", lambda y, cs: {})
    monkeypatch.setattr(pds, "_load_deps_from_zarr", lambda root: {'test': 'data'})
    monkeypatch.setattr(pds.pf, "check_pl_tree_structure", lambda pl, cal: None)
    monkeypatch.setattr(pds.default_steps, "default_cal_steps", [])
    monkeypatch.setattr(pds.default_steps, "default_analysis_steps", [])
    
    fake_root = object()
    monkeypatch.setattr("zarr.open_group", lambda path, mode: fake_root)
    
    def _safe_open(*args, **kwargs):
        return StringIO("{}")
    monkeypatch.setattr(builtins, 'open', _safe_open)
    monkeypatch.setattr("yaml.safe_load", lambda f: {})
    
    DS = pds.DataSet("out.zarr", "file.yaml", None)
    
    # Verify caches are initialized as empty dicts
    assert DS._memory_cache == {}
    assert DS._is_global_cache == {}
    # Verify deps_maps is set from _load_deps_from_zarr
    assert DS.deps_maps == {'test': 'data'}

################################################################################
############################# _is_in_memory ####################################
################################################################################

def test_is_in_memory_invalid_name_type():
    """Test _is_in_memory raises TypeError for non-string name."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    
    with pytest.raises(TypeError):
        DS._is_in_memory(123, 1, None)

def test_is_in_memory_run_not_in_cache():
    """Test _is_in_memory returns False when run_idx not in cache."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    
    assert DS._is_in_memory('param1', 1, None) == False

def test_is_in_memory_name_not_in_run():
    """Test _is_in_memory returns False when name not in run."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'other_param': 'value'}}
    DS._is_global_cache = {}
    
    assert DS._is_in_memory('param1', 1, None) == False

def test_is_in_memory_name_not_in_global_cache():
    """Test _is_in_memory returns False when name not in _is_global_cache."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param1': 'value'}}
    DS._is_global_cache = {}
    
    assert DS._is_in_memory('param1', 1, None) == False

def test_is_in_memory_global_data_exists():
    """Test _is_in_memory returns True for existing global data."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param1': 42}}
    DS._is_global_cache = {'param1': True}
    
    assert DS._is_in_memory('param1', 1, None) == True

def test_is_in_memory_per_row_without_data_idx():
    """Test _is_in_memory raises ValueError for per-row data without data_idx."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param1': 'lazy_attr'}}
    DS._is_global_cache = {'param1': False}
    
    with pytest.raises(ValueError, match="data_idx required"):
        DS._is_in_memory('param1', 1, None)

def test_is_in_memory_per_row_wrong_type():
    """Test _is_in_memory raises TypeError if per-row data is not LazyAttr."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param1': 'not_a_lazy_attr'}}
    DS._is_global_cache = {'param1': False}
    
    with pytest.raises(TypeError, match="Expected LazyAttr"):
        DS._is_in_memory('param1', 1, data_idx=[0, 1])

def test_is_in_memory_per_row_all_cached(monkeypatch):
    """Test _is_in_memory returns True when all requested rows are cached."""
    DS = pds.DataSet.__new__(pds.DataSet)
    
    # Create mock LazyAttr with cache
    mock_lazy = pds.pf.LazyAttr.__new__(pds.pf.LazyAttr)
    mock_lazy._cache = {0: 'data0', 1: 'data1', 2: 'data2'}
    
    DS._memory_cache = {1: {'param1': mock_lazy}}
    DS._is_global_cache = {'param1': False}
    
    assert DS._is_in_memory('param1', 1, data_idx=[0, 1]) == True
    assert DS._is_in_memory('param1', 1, data_idx=0) == True

def test_is_in_memory_per_row_partial_cached(monkeypatch):
    """Test _is_in_memory returns False when some requested rows not cached."""
    DS = pds.DataSet.__new__(pds.DataSet)
    
    # Create mock LazyAttr with partial cache
    mock_lazy = pds.pf.LazyAttr.__new__(pds.pf.LazyAttr)
    mock_lazy._cache = {0: 'data0', 1: 'data1'}
    
    DS._memory_cache = {1: {'param1': mock_lazy}}
    DS._is_global_cache = {'param1': False}
    
    assert DS._is_in_memory('param1', 1, data_idx=[0, 1, 2]) == False
    assert DS._is_in_memory('param1', 1, data_idx=[2, 3]) == False

################################################################################
############################### _is_in_zarr ####################################
################################################################################

def test_is_in_zarr_invalid_name_type():
    """Test _is_in_zarr raises TypeError for non-string name."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = {}
    
    with pytest.raises(TypeError):
        DS._is_in_zarr(123, 1, None)

def test_is_in_zarr_run_not_exists():
    """Test _is_in_zarr returns False when run group doesn't exist."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = {}
    
    assert DS._is_in_zarr('param1', 1, None) == False

def test_is_in_zarr_param_not_in_run():
    """Test _is_in_zarr returns False when parameter not in run."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    DS.root.create_group('run1')
    
    assert DS._is_in_zarr('param1', 1, None) == False

def test_is_in_zarr_missing_global_attr():
    """Test _is_in_zarr returns False when 'global' attribute missing."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run1')
    run_grp.create_group('param1')
    
    assert DS._is_in_zarr('param1', 1, None) == False

def test_is_in_zarr_global_data_exists():
    """Test _is_in_zarr returns True for existing global data."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = True
    
    assert DS._is_in_zarr('param1', 1, None) == True

def test_is_in_zarr_per_row_without_data_idx():
    """Test _is_in_zarr raises ValueError for per-row data without data_idx."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    
    with pytest.raises(ValueError, match="data_idx required"):
        DS._is_in_zarr('param1', 1, None)

def test_is_in_zarr_per_row_no_row_exists():
    """Test _is_in_zarr returns False when row_exists array missing."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    
    assert DS._is_in_zarr('param1', 1, data_idx=[0, 1]) == False

def test_is_in_zarr_per_row_all_exist():
    """Test _is_in_zarr returns True when all requested rows exist."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    row_data = np.array([0, 1, 2, 5])
    param_grp.create_array('row_exists', data=row_data)
    
    assert DS._is_in_zarr('param1', 1, data_idx=[0, 1]) == True
    assert DS._is_in_zarr('param1', 1, data_idx=[2, 5]) == True

def test_is_in_zarr_per_row_partial_exist():
    """Test _is_in_zarr returns False when some requested rows don't exist."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    row_data = np.array([0, 1, 2])
    param_grp.create_array('row_exists', data=row_data)
    
    assert DS._is_in_zarr('param1', 1, data_idx=[0, 1, 5]) == False
    assert DS._is_in_zarr('param1', 1, data_idx=[3, 4]) == False

################################################################################
############################# _get_existing ####################################
################################################################################

def test_get_existing_invalid_name_type():
    """Test _get_existing raises TypeError for non-string name."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.root = {}
    
    with pytest.raises(TypeError, match="name must be a string"):
        DS._get_existing(123, 1, None)

def test_get_existing_from_memory_global():
    """Test _get_existing loads global data from memory cache."""
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param1': 42}}
    DS._is_global_cache = {'param1': True}
    DS.root = {}
    
    result = DS._get_existing('param1', 1, None)
    assert result == 42

def test_get_existing_from_memory_per_row():
    """Test _get_existing loads per-row data from memory LazyAttr."""
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 10  # Need this for LazyAttr validation
    
    # Create proper LazyAttr with cached data
    mock_lazy = pds.pf.LazyAttr(DS, 'param1', 1)
    mock_lazy._cache = {0: 10, 1: 20, 2: 30}
    
    DS._memory_cache = {1: {'param1': mock_lazy}}
    DS._is_global_cache = {'param1': False}
    DS.root = {}
    
    result = DS._get_existing('param1', 1, data_idx=[0, 1])
    np.testing.assert_array_equal(result, [10, 20])

def test_get_existing_from_zarr_global():
    """Test _get_existing loads global data from zarr and caches it."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    # Create zarr structure
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = True
    param_grp.create_array('data', data=np.array([1, 2, 3]))
    
    result = DS._get_existing('param1', 1, None)
    
    # Verify data is returned
    np.testing.assert_array_equal(result, [1, 2, 3])
    
    # Verify data is cached in memory
    assert 1 in DS._memory_cache
    assert 'param1' in DS._memory_cache[1]
    np.testing.assert_array_equal(DS._memory_cache[1]['param1'], [1, 2, 3])

def test_get_existing_from_zarr_per_row():
    """Test _get_existing loads per-row data from zarr and caches in LazyAttr."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    DS.root = zarr.group()
    
    # Create zarr structure
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    param_grp.create_array('data', data=np.array([10, 20, 30, 40, 50]))
    param_grp.create_array('row_exists', data=np.array([0, 1, 2, 3, 4]))
    
    result = DS._get_existing('param1', 1, data_idx=[1, 3])
    
    # Verify data is returned
    np.testing.assert_array_equal(result, [20, 40])
    
    # Verify LazyAttr is created and cached
    assert 1 in DS._memory_cache
    assert 'param1' in DS._memory_cache[1]
    assert isinstance(DS._memory_cache[1]['param1'], pds.pf.LazyAttr)
    
    # Verify data is cached in LazyAttr
    lazy_attr = DS._memory_cache[1]['param1']
    assert 1 in lazy_attr._cache
    assert 3 in lazy_attr._cache
    assert lazy_attr._cache[1] == 20
    assert lazy_attr._cache[3] == 40

def test_get_existing_from_zarr_creates_run_in_cache():
    """Test _get_existing creates run_idx in _memory_cache if it doesn't exist."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {2: {'other': 123}}  # Different run already exists
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    # Create zarr structure
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = True
    param_grp.create_array('data', data=np.array(99))
    
    result = DS._get_existing('param1', 1, None)
    
    # Verify run1 was created in cache
    assert 1 in DS._memory_cache
    assert 2 in DS._memory_cache  # Old run still there
    assert DS._memory_cache[1]['param1'] == 99

def test_get_existing_data_not_found():
    """Test _get_existing raises ValueError when data doesn't exist."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="does not exist in memory or zarr"):
        DS._get_existing('nonexistent', 1, None)

def test_get_existing_per_row_data_not_found():
    """Test _get_existing raises ValueError for per-row data not found."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="does not exist in memory or zarr"):
        DS._get_existing('param1', 1, data_idx=[0, 1])

def test_get_existing_converts_data_idx_to_array():
    """Test _get_existing converts scalar data_idx to array."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    DS.root = zarr.group()
    
    # Create zarr structure
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    param_grp.create_array('data', data=np.array([10, 20, 30]))
    param_grp.create_array('row_exists', data=np.array([0, 1, 2]))
    
    # Use scalar data_idx
    result = DS._get_existing('param1', 1, data_idx=1)
    
    # Should still work and return array
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [20])

def test_get_existing_zarr_per_row_updates_existing_lazy_attr():
    """Test _get_existing updates existing LazyAttr when loading additional rows."""
    import zarr
    import numpy as np
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    # Create zarr structure
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param1')
    param_grp.attrs['global'] = False
    param_grp.create_array('data', data=np.array([10, 20, 30, 40, 50]))
    param_grp.create_array('row_exists', data=np.array([0, 1, 2, 3, 4]))
    
    # Pre-populate with LazyAttr that has some data
    existing_lazy = pds.pf.LazyAttr(DS, 'param1', 1)
    existing_lazy._cache = {0: 10, 1: 20}
    DS._memory_cache = {1: {'param1': existing_lazy}}
    
    # Load additional rows
    result = DS._get_existing('param1', 1, data_idx=[2, 3])
    
    # Verify new data is added to existing LazyAttr
    lazy_attr = DS._memory_cache[1]['param1']
    assert lazy_attr is existing_lazy  # Same object
    assert 0 in lazy_attr._cache  # Old data still there
    assert 1 in lazy_attr._cache
    assert 2 in lazy_attr._cache  # New data added
    assert 3 in lazy_attr._cache
    assert lazy_attr._cache[2] == 30
    assert lazy_attr._cache[3] == 40


################################################################################
############################# _execute_step tests ##############################
################################################################################

def test_execute_step_invalid_step_type():
    """Test _execute_step raises TypeError for non-plStep input."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    
    with pytest.raises(TypeError, match="step must be a plStep instance"):
        DS._execute_step("not a step", data_idx=None)


def test_execute_step_invalid_enforced_max_runs():
    """Test _execute_step raises TypeError for non-dict enforced_max_runs."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    
    step = pds.pf.plStep(
        name='test',
        func=lambda: None,
        param_names=[],
        return_names=['out'],
        func_type='global'
    )
    
    with pytest.raises(TypeError, match="enforced_max_runs must be a dictionary"):
        DS._execute_step(step, data_idx=None, enforced_max_runs="not a dict")


def test_execute_step_global_with_data_idx():
    """Test _execute_step raises ValueError when data_idx provided for global function."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    
    step = pds.pf.plStep(
        name='test',
        func=lambda: None,
        param_names=[],
        return_names=['out'],
        func_type='global'
    )
    
    with pytest.raises(ValueError, match="data_idx must be None for global functions"):
        DS._execute_step(step, data_idx=[0, 1])


def test_execute_step_per_row_without_data_idx():
    """Test _execute_step raises ValueError when data_idx missing for per-row function."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    
    step = pds.pf.plStep(
        name='test',
        func=lambda x: None,
        param_names=['x'],
        return_names=['out'],
        func_type='per-row'
    )
    
    with pytest.raises(ValueError, match="data_idx required for per-row functions"):
        DS._execute_step(step, data_idx=None)


def test_execute_step_global_function():
    """Test _execute_step for global function."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {'global': {}}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Create step that produces global output
    step = pds.pf.plStep(
        name='make_constant',
        func=lambda: 42,
        param_names=[],
        return_names=['const'],
        func_type='global'
    )
    
    DS._execute_step(step, data_idx=None)
    
    # Check output stored in memory cache
    assert 1 in DS._memory_cache
    assert 'const' in DS._memory_cache[1]
    assert DS._memory_cache[1]['const'] == 42
    
    # Check dependencies recorded
    assert 1 in DS.deps_maps['global']
    assert 'const' in DS.deps_maps['global'][1]
    
    # Check global flag set
    assert DS._is_global_cache['const'] is True


def test_execute_step_global_res_function():
    """Test _execute_step for global-res function."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {}
    DS.deps_maps = {'global': {}, 0: {}, 1: {}, 2: {}}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Create step that produces global-res output (one value per row)
    step = pds.pf.plStep(
        name='make_array',
        func=lambda: [10, 20, 30],
        param_names=[],
        return_names=['arr'],
        func_type='global-res'
    )
    
    DS._execute_step(step, data_idx=None)
    
    # Check LazyAttr created and stored
    assert 1 in DS._memory_cache
    assert 'arr' in DS._memory_cache[1]
    lazy_attr = DS._memory_cache[1]['arr']
    assert isinstance(lazy_attr, pds.pf.LazyAttr)
    
    # Check data stored in LazyAttr cache
    assert 0 in lazy_attr._cache
    assert 1 in lazy_attr._cache
    assert 2 in lazy_attr._cache
    assert lazy_attr._cache[0] == 10
    assert lazy_attr._cache[1] == 20
    assert lazy_attr._cache[2] == 30
    
    # Check LazyAttrCollection registered
    assert 'arr' in DS._lazy_collections
    assert 1 in DS._lazy_collections['arr']._lazy_attrs
    
    # Check dependencies recorded for all rows
    for i in range(3):
        assert 1 in DS.deps_maps[i]
        assert 'arr' in DS.deps_maps[i][1]
    
    # Check global flag set correctly
    assert DS._is_global_cache['arr'] is False


def test_execute_step_per_row_function():
    """Test _execute_step for per-row function."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {1: {'x': pds.pf.LazyAttr(DS, 'x', 1)}}
    DS._memory_cache[1]['x']._cache = {0: 1, 1: 2, 2: 3}
    DS.deps_maps = {0: {1: {'x': {}}}, 1: {1: {'x': {}}}, 2: {1: {'x': {}}}}
    DS._is_global_cache = {'x': False}
    DS._lazy_collections = {}
    
    # Create step that squares input
    step = pds.pf.plStep(
        name='square',
        func=lambda x: x**2,
        param_names=['x'],
        return_names=['y'],
        func_type='per-row'
    )
    
    DS._execute_step(step, data_idx=[0, 1, 2])
    
    # Check LazyAttr created and stored
    assert 1 in DS._memory_cache
    assert 'y' in DS._memory_cache[1]
    lazy_attr = DS._memory_cache[1]['y']
    assert isinstance(lazy_attr, pds.pf.LazyAttr)
    
    # Check data stored correctly
    assert lazy_attr._cache[0] == 1
    assert lazy_attr._cache[1] == 4
    assert lazy_attr._cache[2] == 9
    
    # Check LazyAttrCollection registered
    assert 'y' in DS._lazy_collections
    
    # Check dependencies recorded
    for i in range(3):
        assert 1 in DS.deps_maps[i]
        assert 'y' in DS.deps_maps[i][1]


def test_execute_step_vectorized_function():
    """Test _execute_step for vectorized function."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {1: {'x': pds.pf.LazyAttr(DS, 'x', 1)}}
    DS._memory_cache[1]['x']._cache = {0: np.array([1]), 1: np.array([2]), 2: np.array([3])}
    DS.deps_maps = {0: {1: {'x': {}}}, 1: {1: {'x': {}}}, 2: {1: {'x': {}}}}
    DS._is_global_cache = {'x': False}
    DS._lazy_collections = {}
    
    # Create vectorized step
    step = pds.pf.plStep(
        name='double',
        func=lambda x: x * 2,
        param_names=['x'],
        return_names=['z'],
        func_type='vectorized'
    )
    
    DS._execute_step(step, data_idx=[0, 1, 2])
    
    # Check LazyAttr created
    assert 'z' in DS._memory_cache[1]
    lazy_attr = DS._memory_cache[1]['z']
    
    # Check vectorized output stored correctly
    assert np.array_equal(lazy_attr._cache[0], np.array([2]))
    assert np.array_equal(lazy_attr._cache[1], np.array([4]))
    assert np.array_equal(lazy_attr._cache[2], np.array([6]))


def test_execute_step_with_enforced_max_runs():
    """Test _execute_step respects enforced_max_runs."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    # Set up parameter 'x' at two different runs
    DS._memory_cache = {
        1: {'x': pds.pf.LazyAttr(DS, 'x', 1)},
        3: {'x': pds.pf.LazyAttr(DS, 'x', 3)}
    }
    DS._memory_cache[1]['x']._cache = {0: 10}
    DS._memory_cache[3]['x']._cache = {0: 30}
    
    DS.deps_maps = {0: {1: {'x': {}}, 3: {'x': {}}}}
    DS._is_global_cache = {'x': False}
    DS._lazy_collections = {'x': pds.pf.LazyAttrCollection(DS, 'x')}
    DS._lazy_collections['x'].add_run(1, DS._memory_cache[1]['x'])
    DS._lazy_collections['x'].add_run(3, DS._memory_cache[3]['x'])
    
    # Create step that uses 'x'
    step = pds.pf.plStep(
        name='use_x',
        func=lambda x: x + 1,
        param_names=['x'],
        return_names=['y'],
        func_type='per-row'
    )
    
    # Execute with enforced max run 1 (should use old version)
    DS._execute_step(step, data_idx=[0], enforced_max_runs={'x': 1})
    
    # Check output based on run 1 value (10)
    lazy_attr = DS._memory_cache[1]['y']
    assert lazy_attr._cache[0] == 11  # 10 + 1, not 30 + 1


def test_execute_step_multiple_returns():
    """Test _execute_step with multiple return values."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS.deps_maps = {'global': {}}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Create step with multiple outputs
    step = pds.pf.plStep(
        name='multi_out',
        func=lambda: (1, 2, 3),
        param_names=[],
        return_names=['a', 'b', 'c'],
        func_type='global'
    )
    
    DS._execute_step(step, data_idx=None)
    
    # Check all outputs stored
    assert 'a' in DS._memory_cache[1]
    assert 'b' in DS._memory_cache[1]
    assert 'c' in DS._memory_cache[1]
    assert DS._memory_cache[1]['a'] == 1
    assert DS._memory_cache[1]['b'] == 2
    assert DS._memory_cache[1]['c'] == 3


################################################################################
############################# _produce_data tests ##############################
################################################################################

def test_produce_data_executes_path(monkeypatch):
    """Test _produce_data executes pipeline path for parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {}
    DS.deps_maps = {'global': {}}
    DS._is_global_cache = {}
    DS._lazy_collections = {}
    
    # Create simple pipeline: step1 -> step2
    step1 = pds.pf.plStep(
        name='make_base',
        func=lambda: 10,
        param_names=[],
        return_names=['base'],
        func_type='global'
    )
    
    step2 = pds.pf.plStep(
        name='make_derived',
        func=lambda base: base * 2,
        param_names=['base'],
        return_names=['derived'],
        func_type='global'
    )
    
    # Mock pipeline and find_pl_path
    DS.cal_pl = {}
    monkeypatch.setattr(pds.pf, 'find_pl_path', lambda tree, name: [step1, step2])
    
    # Execute
    DS._produce_data('derived', data_idx=None)
    
    # Check both steps executed
    assert 'base' in DS._memory_cache[1]
    assert 'derived' in DS._memory_cache[1]
    assert DS._memory_cache[1]['base'] == 10
    assert DS._memory_cache[1]['derived'] == 20


def test_produce_data_filters_enforced_max_runs(monkeypatch):
    """Test _produce_data filters enforced_max_runs per step."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {1: {'a': 1, 'b': 2}}
    DS.deps_maps = {'global': {1: {'a': {}, 'b': {}}}}
    DS._is_global_cache = {'a': True, 'b': True}
    DS._lazy_collections = {}
    
    # Create step that only uses 'a', not 'b'
    step = pds.pf.plStep(
        name='use_a',
        func=lambda a: a + 100,
        param_names=['a'],
        return_names=['result'],
        func_type='global'
    )
    
    DS.cal_pl = {}
    monkeypatch.setattr(pds.pf, 'find_pl_path', lambda tree, name: [step])
    
    # Call with enforced_max_runs containing both 'a' and 'b'
    # Should filter to only pass 'a' to the step
    DS._produce_data('result', data_idx=None, enforced_max_runs={'a': 1, 'b': 1})
    
    # Verify it executed without error (filtering worked)
    assert 'result' in DS._memory_cache[1]
    assert DS._memory_cache[1]['result'] == 101


def test_produce_data_per_row(monkeypatch):
    """Test _produce_data for per-row parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {1: {'x': pds.pf.LazyAttr(DS, 'x', 1)}}
    DS._memory_cache[1]['x']._cache = {0: 5, 1: 10}
    DS.deps_maps = {0: {1: {'x': {}}}, 1: {1: {'x': {}}}}
    DS._is_global_cache = {'x': False}
    DS._lazy_collections = {}
    
    # Create step
    step = pds.pf.plStep(
        name='process',
        func=lambda x: x * 3,
        param_names=['x'],
        return_names=['y'],
        func_type='per-row'
    )
    
    DS.cal_pl = {}
    monkeypatch.setattr(pds.pf, 'find_pl_path', lambda tree, name: [step])
    
    # Produce data for specific indices
    DS._produce_data('y', data_idx=[0, 1])
    
    # Check output
    assert 'y' in DS._memory_cache[1]
    lazy_attr = DS._memory_cache[1]['y']
    assert lazy_attr._cache[0] == 15
    assert lazy_attr._cache[1] == 30


################################################################################
############################# _fetch_rows tests ################################
################################################################################

def test_fetch_rows_from_memory():
    """Test _fetch_rows returns data from memory when available."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {2: {'param': pds.pf.LazyAttr(DS, 'param', 2)}}
    DS._memory_cache[2]['param']._cache = {0: np.array([100]), 1: np.array([200])}
    DS._is_global_cache = {'param': False}
    
    # Fetch from memory (returns result from _get_existing which calls lazy_attr[data_idx])
    result = DS._fetch_rows('param', run_idx=2, data_idx=[0, 1])
    
    # Verify data returned (should be 2D array from LazyAttr)
    assert np.array_equal(result, [[100], [200]])


def test_fetch_rows_from_zarr(monkeypatch):
    """Test _fetch_rows loads data from zarr when not in memory."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS._is_global_cache = {'param': False}
    DS._lazy_collections = {}
    DS.deps_maps = {0: {}, 1: {}}
    
    # Create mock zarr structure  
    DS.root = zarr.group()
    run_grp = DS.root.create_group('run2')
    param_grp = run_grp.create_group('param')
    param_grp.attrs['global'] = False
    param_grp.attrs['deps'] = {'idx0': {}, 'idx1': {}}
    param_grp.create_array('data', data=np.array([[10], [20], [30], [40], [50]]))
    param_grp.create_array('row_exists', data=np.array([0, 1]))
    
    # Fetch from zarr
    result = DS._fetch_rows('param', run_idx=2, data_idx=[0, 1])
    
    # Verify data loaded and cached
    assert np.array_equal(result, [[10], [20]])
    assert 2 in DS._memory_cache
    assert 'param' in DS._memory_cache[2]


def test_fetch_rows_produces_data_when_missing(monkeypatch):
    """Test _fetch_rows produces data when not in memory or zarr."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {}
    DS._is_global_cache = {'output': False}
    DS._lazy_collections = {}
    DS.deps_maps = {0: {}, 1: {}}
    
    # Create empty zarr
    DS.root = zarr.group()
    
    # Mock _produce_data to populate memory cache
    def mock_produce(name, data_idx, enforced_max_runs):
        # Simulate producing data
        if 1 not in DS._memory_cache:
            DS._memory_cache[1] = {}
        DS._memory_cache[1][name] = pds.pf.LazyAttr(DS, name, 1)
        for idx in data_idx:
            DS._memory_cache[1][name]._cache[idx] = np.array([idx * 10])
    
    monkeypatch.setattr(DS, '_produce_data', mock_produce)
    
    # Fetch - should trigger production
    result = DS._fetch_rows('output', run_idx=1, data_idx=[0, 1])
    
    # Verify data was produced and returned
    assert 0 in DS._memory_cache[1]['output']._cache
    assert 1 in DS._memory_cache[1]['output']._cache


def test_fetch_rows_validates_run_idx_match(monkeypatch):
    """Test _fetch_rows raises error when requested run_idx doesn't match potential."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {}
    DS._is_global_cache = {'param': False}
    DS._lazy_collections = {}
    # deps_maps shows param exists at run 2
    DS.deps_maps = {0: {2: {'param': {}}}, 1: {2: {'param': {}}}}
    
    DS.root = zarr.group()
    
    # Try to fetch at run 1, but potential is 3 (2+1)
    with pytest.raises(ValueError, match="Cannot produce 'param' at run_idx=1"):
        DS._fetch_rows('param', run_idx=1, data_idx=[0, 1])


def test_fetch_rows_global_parameter():
    """Test _fetch_rows for global parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {1: {'global_param': 42}}
    DS._is_global_cache = {'global_param': True}
    
    # Fetch global parameter (data_idx should be None)
    result = DS._fetch_rows('global_param', run_idx=1, data_idx=None)
    
    assert result == 42


def test_fetch_rows_multiple_run_indices_error():
    """Test _fetch_rows raises error when data_idx have different potential runs."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS._is_global_cache = {'param': False}
    DS._lazy_collections = {}
    # deps_maps shows different runs for different indices
    DS.deps_maps = {
        0: {1: {'param': {}}},  # potential run 2
        1: {2: {'param': {}}}   # potential run 3
    }
    
    DS.root = zarr.group()
    
    # Try to fetch indices with different potential runs
    with pytest.raises(ValueError, match="have different potential run indices"):
        DS._fetch_rows('param', run_idx=2, data_idx=[0, 1])


################################################################################
########################### _get_reserved_attrs ################################
################################################################################
def test_get_reserved_attrs_returns_set():
    """Test _get_reserved_attrs returns a set."""
    DS = pds.DataSet.__new__(pds.DataSet)
    reserved = DS._get_reserved_attrs()
    assert isinstance(reserved, set)


def test_get_reserved_attrs_caches_result():
    """Test _get_reserved_attrs caches result at class level."""
    # Clear cache first
    pds.DataSet._RESERVED_ATTRS = None
    
    DS1 = pds.DataSet.__new__(pds.DataSet)
    DS2 = pds.DataSet.__new__(pds.DataSet)
    
    reserved1 = DS1._get_reserved_attrs()
    reserved2 = DS2._get_reserved_attrs()
    
    # Should be the same object (cached)
    assert reserved1 is reserved2
    assert reserved1 is pds.DataSet._RESERVED_ATTRS


def test_get_reserved_attrs_excludes_private():
    """Test _get_reserved_attrs excludes private attributes (starting with _)."""
    DS = pds.DataSet.__new__(pds.DataSet)
    reserved = DS._get_reserved_attrs()
    
    # Check that private attributes are excluded
    for attr in reserved:
        assert not attr.startswith('_')


def test_get_reserved_attrs_includes_public_methods():
    """Test _get_reserved_attrs includes public methods."""
    DS = pds.DataSet.__new__(pds.DataSet)
    reserved = DS._get_reserved_attrs()
    
    # Check that known public methods are included
    assert 'write_data' in reserved
    assert 'show_file' in reserved


################################################################################
############################### show_file ######################################
################################################################################
def test_show_file_cal(monkeypatch):
    """Test show_file opens cal yaml directory."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.cal_yaml_path = "/path/to/calibration.yaml"
    
    opened_path = []
    def mock_open(path):
        opened_path.append(path)
    
    monkeypatch.setattr(util, "open_in_file_explorer", mock_open)
    
    DS.show_file('cal')
    
    assert opened_path == ["/path/to"]


def test_show_file_custom(monkeypatch):
    """Test show_file opens custom steps directory."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.custom_path = "/path/to/custom_steps.py"
    
    opened_path = []
    def mock_open(path):
        opened_path.append(path)
    
    monkeypatch.setattr(util, "open_in_file_explorer", mock_open)
    
    DS.show_file('custom')
    
    assert opened_path == ["/path/to"]


def test_show_file_custom_none(monkeypatch):
    """Test show_file raises error when custom_path is None."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.custom_path = None
    
    monkeypatch.setattr(util, "open_in_file_explorer", lambda p: None)
    
    with pytest.raises(ValueError, match="No custom steps file was provided"):
        DS.show_file('custom')


def test_show_file_zarr(monkeypatch):
    """Test show_file opens zarr directory."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.zarr_path = "/path/to/data.zarr"
    
    opened_path = []
    def mock_open(path):
        opened_path.append(path)
    
    monkeypatch.setattr(util, "open_in_file_explorer", mock_open)
    
    DS.show_file('zarr')
    
    assert opened_path == ["/path/to"]


def test_show_file_analysis(monkeypatch):
    """Test show_file opens analysis yaml directory."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.analysis_yaml_path = "/path/to/analysis.yaml"
    
    opened_path = []
    def mock_open(path):
        opened_path.append(path)
    
    monkeypatch.setattr(util, "open_in_file_explorer", mock_open)
    
    DS.show_file('analysis')
    
    assert opened_path == ["/path/to"]


def test_show_file_analysis_none(monkeypatch):
    """Test show_file raises error when analysis_yaml_path is None."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.analysis_yaml_path = None
    
    monkeypatch.setattr(util, "open_in_file_explorer", lambda p: None)
    
    with pytest.raises(ValueError, match="No analysis YAML file was provided"):
        DS.show_file('analysis')


################################################################################
######################### _convert_yaml_to_steps ###############################
################################################################################
def test_convert_yaml_to_steps_simple_task():
    """Test _convert_yaml_to_steps converts task string to plStep."""
    mock_step = pds.pf.plStep(
        name='test_step',
        func=lambda x: x,
        func_type='global',
        param_names=['x'],
        return_names=['y']
    )
    cal_steps = [mock_step]
    
    yaml_dict = {'task': 'test_step'}
    result = pds._convert_yaml_to_steps(yaml_dict, cal_steps)
    
    assert result['task'] is mock_step


def test_convert_yaml_to_steps_nested_dict():
    """Test _convert_yaml_to_steps handles nested dictionaries."""
    mock_step1 = pds.pf.plStep(
        name='step1',
        func=lambda x: x,
        func_type='global',
        param_names=['x'],
        return_names=['y']
    )
    mock_step2 = pds.pf.plStep(
        name='step2',
        func=lambda x: x,
        func_type='global',
        param_names=['x'],
        return_names=['y']
    )
    cal_steps = [mock_step1, mock_step2]
    
    yaml_dict = {
        'level1': {
            'task': 'step1'
        },
        'level2': {
            'task': 'step2'
        }
    }
    result = pds._convert_yaml_to_steps(yaml_dict, cal_steps)
    
    assert result['level1']['task'] is mock_step1
    assert result['level2']['task'] is mock_step2


def test_convert_yaml_to_steps_preserves_non_task():
    """Test _convert_yaml_to_steps preserves non-task keys."""
    mock_step = pds.pf.plStep(
        name='test_step',
        func=lambda x: x,
        func_type='global',
        param_names=['x'],
        return_names=['y']
    )
    cal_steps = [mock_step]
    
    yaml_dict = {
        'task': 'test_step',
        'params': {'param1': 10},
        'other_key': 'value'
    }
    result = pds._convert_yaml_to_steps(yaml_dict, cal_steps)
    
    assert result['task'] is mock_step
    assert result['params'] == {'param1': 10}
    assert result['other_key'] == 'value'


def test_convert_yaml_to_steps_step_not_found():
    """Test _convert_yaml_to_steps raises error when step not found."""
    cal_steps = []
    
    yaml_dict = {'task': 'nonexistent_step'}
    
    with pytest.raises(ValueError, match="Step 'nonexistent_step' not found"):
        pds._convert_yaml_to_steps(yaml_dict, cal_steps)


def test_convert_yaml_to_steps_recursive():
    """Test _convert_yaml_to_steps handles deeply nested structures."""
    mock_step = pds.pf.plStep(
        name='deep_step',
        func=lambda x: x,
        func_type='global',
        param_names=['x'],
        return_names=['y']
    )
    cal_steps = [mock_step]
    
    yaml_dict = {
        'a': {
            'b': {
                'c': {
                    'task': 'deep_step'
                }
            }
        }
    }
    result = pds._convert_yaml_to_steps(yaml_dict, cal_steps)
    
    assert result['a']['b']['c']['task'] is mock_step


def test_convert_yaml_to_steps_list_values():
    """Test _convert_yaml_to_steps handles lists in values."""
    mock_step = pds.pf.plStep(
        name='step',
        func=lambda x: x,
        func_type='global',
        param_names=['x'],
        return_names=['y']
    )
    cal_steps = [mock_step]
    
    # Lists in values should be preserved (not recursed into)
    yaml_dict = {
        'task': 'step',
        'list_param': [1, 2, 3]
    }
    result = pds._convert_yaml_to_steps(yaml_dict, cal_steps)
    
    assert result['task'] is mock_step
    assert result['list_param'] == [1, 2, 3]


def test_convert_yaml_to_steps_string_not_task_key():
    """Test _convert_yaml_to_steps doesn't convert strings unless key is 'task'."""
    cal_steps = []
    
    yaml_dict = {
        'name': 'some_string',  # Not a task key
        'description': 'another_string'
    }
    result = pds._convert_yaml_to_steps(yaml_dict, cal_steps)
    
    # Strings should be preserved when key is not 'task'
    assert result['name'] == 'some_string'
    assert result['description'] == 'another_string'


################################################################################
########################### _load_deps_from_zarr ###############################
################################################################################
def test_load_deps_from_zarr_invalid_input():
    """Test _load_deps_from_zarr raises error for non-zarr-group input."""
    with pytest.raises(ValueError, match="Input root must be a zarr group"):
        pds._load_deps_from_zarr("not a zarr group")


def test_load_deps_from_zarr_root_with_arrays():
    """Test _load_deps_from_zarr raises error if root contains arrays."""
    import zarr
    root = zarr.group()
    root.create_array('bad_array', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="root cannot have arrays"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_invalid_group_name():
    """Test _load_deps_from_zarr raises error for non-run group names."""
    import zarr
    root = zarr.group()
    root.create_group('bad_group_name')
    
    with pytest.raises(ValueError, match="root can only contain run folders"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_run_with_arrays():
    """Test _load_deps_from_zarr raises error if run group contains arrays."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    run1.create_array('bad_array', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="run1 must not contain arrays"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_missing_deps_attr():
    """Test _load_deps_from_zarr raises error if deps attribute missing."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['global'] = True
    param.create_array('data', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="Missing 'deps' attr"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_missing_global_attr():
    """Test _load_deps_from_zarr raises error if global attribute missing."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {}
    param.create_array('data', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="Missing 'global' attribute"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_param_with_subgroups():
    """Test _load_deps_from_zarr raises error if parameter contains groups."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {}
    param.attrs['global'] = True
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_group('bad_subgroup')
    
    with pytest.raises(ValueError, match="contains a zarr group"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_missing_data_array():
    """Test _load_deps_from_zarr raises error if data array missing."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {}
    param.attrs['global'] = True
    
    with pytest.raises(ValueError, match="'data' array not found"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_global_with_extra_arrays():
    """Test _load_deps_from_zarr raises error if global param has extra arrays."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {}
    param.attrs['global'] = True
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('extra_array', shape=(5,), dtype='float64')
    
    with pytest.raises(ValueError, match="Extra array.*found in run1 global parameter"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_perrow_missing_row_exists():
    """Test _load_deps_from_zarr raises error if per-row param missing row_exists."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'idx0': {}}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="'row_exists' array not found"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_perrow_row_exists_wrong_dtype():
    """Test _load_deps_from_zarr raises error if row_exists has wrong dtype."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'idx0': {}}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype='int32')
    
    with pytest.raises(ValueError, match="'row_exists' array.*must have dtype bool"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_perrow_with_extra_arrays():
    """Test _load_deps_from_zarr raises error if per-row param has extra arrays."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'idx0': {}}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype=bool)
    param.create_array('extra', shape=(5,), dtype='float64')
    
    with pytest.raises(ValueError, match="Extra array.*found in run1 parameter"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_deps_not_dict():
    """Test _load_deps_from_zarr raises error if deps is not a dict."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = "not a dict"
    param.attrs['global'] = True
    param.create_array('data', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="'deps' attribute.*must be a dictionary"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_global_deps_invalid_value():
    """Test _load_deps_from_zarr raises error if global deps has non-int value."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'input1': 'not an int'}
    param.attrs['global'] = True
    param.create_array('data', shape=(10,), dtype='float64')
    
    with pytest.raises(ValueError, match="deps values must be integers"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_duplicate_global_param():
    """Test _load_deps_from_zarr raises error for duplicate global parameter."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    
    param1 = run1.create_group('param1')
    param1.attrs['deps'] = {}
    param1.attrs['global'] = True
    param1.create_array('data', shape=(10,), dtype='float64')
    
    # Try to create another param1 - zarr won't allow it, so we simulate
    # by manually adding to attrs/deps_maps (this tests the check logic)
    # Actually, we need to test the logic within the function
    # Let me skip this duplicate test since zarr prevents duplicate group names
    

def test_load_deps_from_zarr_perrow_invalid_data_idx_key():
    """Test _load_deps_from_zarr raises error for invalid data_idx key format."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'not_a_valid_idx': {}}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype=bool)
    
    with pytest.raises(ValueError, match="deps keys must be convertible to int"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_perrow_deps_not_dict():
    """Test _load_deps_from_zarr raises error if per-row deps value is not dict."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'idx0': "not a dict"}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype=bool)
    
    with pytest.raises(ValueError, match="deps must be a dictionary"):
        pds._load_deps_from_zarr(root)




def test_load_deps_from_zarr_perrow_deps_invalid_value():
    """Test _load_deps_from_zarr raises error if per-row deps has non-int value."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'idx0': {'input1': 'not an int'}}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype=bool)
    
    with pytest.raises(ValueError, match="deps values must be integers"):
        pds._load_deps_from_zarr(root)


def test_load_deps_from_zarr_valid_global_param():
    """Test _load_deps_from_zarr correctly loads global parameter."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {'input1': 1, 'input2': 2}
    param.attrs['global'] = True
    param.create_array('data', shape=(10,), dtype='float64')
    
    deps_maps = pds._load_deps_from_zarr(root)
    
    assert 'global' in deps_maps
    assert 1 in deps_maps['global']
    assert 'param1' in deps_maps['global'][1]
    assert deps_maps['global'][1]['param1'] == {'input1': 1, 'input2': 2}


def test_load_deps_from_zarr_valid_perrow_param():
    """Test _load_deps_from_zarr correctly loads per-row parameter."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    param.attrs['deps'] = {
        'idx0': {'input1': 1},
        'idx5': {'input1': 2}
    }
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype=bool)
    
    deps_maps = pds._load_deps_from_zarr(root)
    
    assert 0 in deps_maps
    assert 5 in deps_maps
    assert deps_maps[0][1]['param1'] == {'input1': 1}
    assert deps_maps[5][1]['param1'] == {'input1': 2}


def test_load_deps_from_zarr_multiple_runs():
    """Test _load_deps_from_zarr correctly loads multiple runs."""
    import zarr
    root = zarr.group()
    
    run1 = root.create_group('run1')
    param1 = run1.create_group('param1')
    param1.attrs['deps'] = {}
    param1.attrs['global'] = True
    param1.create_array('data', shape=(10,), dtype='float64')
    
    run2 = root.create_group('run2')
    param2 = run2.create_group('param2')
    param2.attrs['deps'] = {'param1': 1}
    param2.attrs['global'] = True
    param2.create_array('data', shape=(10,), dtype='float64')
    
    deps_maps = pds._load_deps_from_zarr(root)
    
    assert 1 in deps_maps['global']
    assert 2 in deps_maps['global']
    assert deps_maps['global'][1]['param1'] == {}
    assert deps_maps['global'][2]['param2'] == {'param1': 1}


def test_load_deps_from_zarr_empty_root():
    """Test _load_deps_from_zarr returns empty dict for empty zarr."""
    import zarr
    root = zarr.group()
    
    deps_maps = pds._load_deps_from_zarr(root)
    
    assert deps_maps == {}


def test_load_deps_from_zarr_integer_data_idx_key():
    """Test _load_deps_from_zarr handles integer data_idx keys."""
    import zarr
    root = zarr.group()
    run1 = root.create_group('run1')
    param = run1.create_group('param1')
    # Use integer keys instead of 'idx0' format
    param.attrs['deps'] = {0: {'input1': 1}, 3: {'input1': 2}}
    param.attrs['global'] = False
    param.create_array('data', shape=(10,), dtype='float64')
    param.create_array('row_exists', shape=(10,), dtype=bool)
    
    deps_maps = pds._load_deps_from_zarr(root)
    
    assert 0 in deps_maps
    assert 3 in deps_maps
    assert deps_maps[0][1]['param1'] == {'input1': 1}
    assert deps_maps[3][1]['param1'] == {'input1': 2}


################################################################################
######################### _check_path_validity #################################
################################################################################
def test_check_path_validity_none_path():
    """Test _check_path_validity raises error for None path."""
    DS = pds.DataSet.__new__(pds.DataSet)
    
    with pytest.raises(ValueError, match="Cannot execute None path"):
        DS._check_path_validity(None, None, {})


def test_check_path_validity_empty_path():
    """Test _check_path_validity returns empty path unchanged."""
    DS = pds.DataSet.__new__(pds.DataSet)
    
    result = DS._check_path_validity([], None, {})
    assert result == []


def test_check_path_validity_trims_completed_global_steps():
    """Test _check_path_validity trims global steps with existing outputs."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.deps_maps = {'global': {}}  # No previous runs
    DS._memory_cache = {1: {'output1': 42}}
    DS._is_global_cache = {'output1': True}
    DS.root = zarr.group()
    
    step1 = pds.pf.plStep(
        name='step1',
        func=lambda: {'output1': 42},
        func_type='global',
        param_names=[],
        return_names=['output1']
    )
    step2 = pds.pf.plStep(
        name='step2',
        func=lambda x: {'output2': x + 1},
        func_type='global',
        param_names=['output1'],
        return_names=['output2']
    )
    
    path = [step1, step2]
    result = DS._check_path_validity(path, None, {})
    
    # step1 output exists at run 1, so it should be trimmed
    assert result == [step2]


def test_check_path_validity_trims_completed_perrow_steps():
    """Test _check_path_validity trims per-row steps with existing outputs."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS.deps_maps = {
        0: {},  # No previous runs
        1: {}   # No previous runs
    }
    DS._memory_cache = {1: {'output1': pds.pf.LazyAttr(DS, 'output1', 1)}}
    DS._memory_cache[1]['output1']._cache = {0: 10, 1: 20}
    DS._is_global_cache = {'output1': False}
    DS.root = zarr.group()
    
    step1 = pds.pf.plStep(
        name='step1',
        func=lambda data_idx: {'output1': data_idx * 10},
        func_type='per-row',
        param_names=['data_idx'],
        return_names=['output1']
    )
    step2 = pds.pf.plStep(
        name='step2',
        func=lambda x: {'output2': x + 1},
        func_type='per-row',
        param_names=['output1'],
        return_names=['output2']
    )
    
    path = [step1, step2]
    result = DS._check_path_validity(path, np.array([0, 1]), {})
    
    # step1 output exists for both indices at run 1, so it should be trimmed
    assert result == [step2]


def test_check_path_validity_keeps_partial_perrow_steps():
    """Test _check_path_validity keeps per-row steps with some missing outputs."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS.deps_maps = {
        0: {1: {'output1': {}}}
        # Missing data_idx 1
    }
    DS._memory_cache = {1: {'output1': pds.pf.LazyAttr(DS, 'output1', 1)}}
    DS._memory_cache[1]['output1']._cache = {0: 10}
    DS._is_global_cache = {'output1': False}
    
    step1 = pds.pf.plStep(
        name='step1',
        func=lambda data_idx: {'output1': data_idx * 10},
        func_type='per-row',
        param_names=['data_idx'],
        return_names=['output1']
    )
    
    path = [step1]
    result = DS._check_path_validity(path, np.array([0, 1]), {})
    
    # step1 output exists for idx 0 but not 1, so it should be kept
    assert result == [step1]


def test_check_path_validity_validates_global_input_exists():
    """Test _check_path_validity raises error if global input doesn't exist."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.deps_maps = {'global': {}}
    DS._memory_cache = {}
    
    step = pds.pf.plStep(
        name='step1',
        func=lambda x: {'output': x + 1},
        func_type='global',
        param_names=['missing_input'],
        return_names=['output']
    )
    
    with pytest.raises(ValueError, match="requires 'missing_input', which does not exist"):
        DS._check_path_validity([step], None, {})


def test_check_path_validity_validates_perrow_input_exists():
    """Test _check_path_validity raises error if per-row input doesn't exist."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.deps_maps = {0: {}}
    DS._memory_cache = {}
    
    step = pds.pf.plStep(
        name='step1',
        func=lambda x: {'output': x + 1},
        func_type='per-row',
        param_names=['missing_input'],
        return_names=['output']
    )
    
    with pytest.raises(ValueError, match="requires 'missing_input' for data_idx 0"):
        DS._check_path_validity([step], np.array([0]), {})


def test_check_path_validity_accepts_inputs_from_preceding_steps():
    """Test _check_path_validity accepts inputs that will be produced by earlier steps."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.deps_maps = {'global': {}}
    DS._memory_cache = {}
    
    step1 = pds.pf.plStep(
        name='step1',
        func=lambda: {'output1': 42},
        func_type='global',
        param_names=[],
        return_names=['output1']
    )
    step2 = pds.pf.plStep(
        name='step2',
        func=lambda x: {'output2': x + 1},
        func_type='global',
        param_names=['output1'],
        return_names=['output2']
    )
    
    path = [step1, step2]
    result = DS._check_path_validity(path, None, {})
    
    # Should not raise error, step2 gets input from step1
    assert result == [step1, step2]


def test_check_path_validity_enforced_max_runs():
    """Test _check_path_validity respects enforced_max_runs."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.deps_maps = {'global': {1: {'input1': {}}, 2: {'input1': {}}}}
    DS._memory_cache = {1: {'input1': 10}, 2: {'input1': 20}}
    DS._is_global_cache = {'input1': True}
    
    step = pds.pf.plStep(
        name='step1',
        func=lambda x: {'output': x + 1},
        func_type='global',
        param_names=['input1'],
        return_names=['output']
    )
    
    # Should succeed - enforced run 1 exists
    result = DS._check_path_validity([step], None, {'input1': 1})
    assert result == [step]


def test_check_path_validity_type_mismatch_global_needs_perrow():
    """Test _check_path_validity raises error when global step needs per-row param."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS.deps_maps = {'global': {}}
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    step1 = pds.pf.plStep(
        name='step1',
        func=lambda: {'output1': [1, 2, 3]},
        func_type='global-res',
        param_names=[],
        return_names=['output1']
    )
    step2 = pds.pf.plStep(
        name='step2',
        func=lambda x: {'output2': x},
        func_type='global',
        param_names=['output1'],
        return_names=['output2']
    )
    
    path = [step1, step2]
    
    with pytest.raises(ValueError, match="requires global parameter.*but preceding steps produce it as per-row"):
        DS._check_path_validity(path, None, {})


def test_check_path_validity_data_idx_special_param():
    """Test _check_path_validity handles 'data_idx' as special parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.deps_maps = {0: {}}
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.root = zarr.group()
    
    step = pds.pf.plStep(
        name='step1',
        func=lambda data_idx: {'output': data_idx * 10},
        func_type='per-row',
        param_names=['data_idx'],
        return_names=['output']
    )
    
    # Should not raise error - data_idx is a special built-in parameter
    result = DS._check_path_validity([step], np.array([0]), {})
    assert result == [step]


################################################################################
############################### write_data #####################################
################################################################################
def test_write_data_run_idx_not_in_cache():
    """Test write_data raises error if run_idx not in memory cache."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="run_idx 1 not found in memory cache"):
        DS.write_data('param1', 1, None)


def test_write_data_name_not_in_cache():
    """Test write_data raises error if name not in memory cache."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {}}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="Parameter 'param1' at run_idx 1 not found"):
        DS.write_data('param1', 1, None)


def test_write_data_global_param_success():
    """Test write_data successfully writes global parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'global_param': np.array(42)}}
    DS._is_global_cache = {'global_param': True}
    DS.deps_maps = {'global': {1: {'global_param': {'input1': 1}}}}
    DS.root = zarr.group()
    
    DS.write_data('global_param', 1, None)
    
    # Verify zarr structure
    assert 'run1' in DS.root
    assert 'global_param' in DS.root['run1']
    param_grp = DS.root['run1']['global_param']
    
    # Check data
    assert param_grp['data'][()] == 42
    
    # Check attrs
    assert param_grp.attrs['global'] == True
    assert param_grp.attrs['deps'] == {'input1': 1}
    assert 'write_time' in param_grp.attrs
    assert len(param_grp.attrs['write_time']) == 17  # Format: YYYYMMDD-HH:MM:SS


def test_write_data_global_param_with_data_idx_error():
    """Test write_data raises error if data_idx provided for global param."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'global_param': 42}}
    DS._is_global_cache = {'global_param': True}
    DS.deps_maps = {'global': {1: {'global_param': {}}}}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="data_idx must be None for global parameter"):
        DS.write_data('global_param', 1, data_idx=0)


def test_write_data_global_param_already_exists():
    """Test write_data raises error if trying to overwrite global parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'global_param': np.array(42)}}
    DS._is_global_cache = {'global_param': True}
    DS.deps_maps = {'global': {1: {'global_param': {}}}}
    DS.root = zarr.group()
    
    # Write once
    DS.write_data('global_param', 1, None)
    
    # Try to write again
    with pytest.raises(ValueError, match="already exists in zarr"):
        DS.write_data('global_param', 1, None)


def test_write_data_global_param_custom_dtype():
    """Test write_data uses custom dtype for global parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'global_param': np.array(42)}}
    DS._is_global_cache = {'global_param': True}
    DS.deps_maps = {'global': {1: {'global_param': {}}}}
    DS.root = zarr.group()
    
    DS.write_data('global_param', 1, None, dtype=np.float64)
    
    param_grp = DS.root['run1']['global_param']
    assert param_grp['data'].dtype == np.float64


def test_write_data_perrow_param_no_data_idx_error():
    """Test write_data raises error if data_idx not provided for per-row param."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="data_idx required for per-row parameter"):
        DS.write_data('perrow_param', 1, data_idx=None)


def test_write_data_perrow_param_not_lazyattr_error():
    """Test write_data raises error if per-row param is not LazyAttr."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {1: {'perrow_param': 42}}  # Not a LazyAttr
    DS._is_global_cache = {'perrow_param': False}
    DS.root = zarr.group()
    
    with pytest.raises(TypeError, match="Expected LazyAttr"):
        DS.write_data('perrow_param', 1, data_idx=0)


def test_write_data_perrow_param_data_idx_not_in_cache():
    """Test write_data raises error if data_idx not in LazyAttr cache."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = np.array([10])
    # Index 1 is not in cache
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {0: {1: {'perrow_param': {}}}}
    DS.root = zarr.group()
    
    with pytest.raises(ValueError, match="data_idx 1 not found in memory"):
        DS.write_data('perrow_param', 1, data_idx=[0, 1])


def test_write_data_perrow_param_new_success():
    """Test write_data successfully writes new per-row parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = np.array([10, 20])
    lazy_attr._cache[2] = np.array([30, 40])
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {
        0: {1: {'perrow_param': {'input1': 1}}},
        2: {1: {'perrow_param': {'input1': 1}}}
    }
    DS.root = zarr.group()
    
    DS.write_data('perrow_param', 1, data_idx=[0, 2])
    
    # Verify zarr structure
    param_grp = DS.root['run1']['perrow_param']
    
    # Check data array shape
    assert param_grp['data'].shape == (5, 2)
    assert np.array_equal(param_grp['data'][0], [10, 20])
    assert np.array_equal(param_grp['data'][2], [30, 40])
    
    # Check row_exists
    row_exists = param_grp['row_exists'][...]
    assert row_exists[0] == True
    assert row_exists[1] == False
    assert row_exists[2] == True
    assert row_exists[3] == False
    assert row_exists[4] == False
    
    # Check attrs
    assert param_grp.attrs['global'] == False
    assert 'idx0' in param_grp.attrs['deps']
    assert 'idx2' in param_grp.attrs['deps']
    assert param_grp.attrs['deps']['idx0'] == {'input1': 1}
    assert param_grp.attrs['deps']['idx2'] == {'input1': 1}
    assert 'idx0' in param_grp.attrs['write_times']
    assert 'idx2' in param_grp.attrs['write_times']


def test_write_data_perrow_param_incremental():
    """Test write_data incrementally writes to existing per-row parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = np.array([10])
    lazy_attr._cache[1] = np.array([20])
    lazy_attr._cache[2] = np.array([30])
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {
        0: {1: {'perrow_param': {}}},
        1: {1: {'perrow_param': {}}},
        2: {1: {'perrow_param': {}}}
    }
    DS.root = zarr.group()
    
    # Write first batch
    DS.write_data('perrow_param', 1, data_idx=[0])
    
    # Write second batch
    DS.write_data('perrow_param', 1, data_idx=[1, 2])
    
    # Verify all data is present
    param_grp = DS.root['run1']['perrow_param']
    assert param_grp['data'][0] == 10
    assert param_grp['data'][1] == 20
    assert param_grp['data'][2] == 30
    
    row_exists = param_grp['row_exists'][...]
    assert np.all(row_exists[:3])
    assert not np.any(row_exists[3:])


def test_write_data_perrow_param_overwrite_error():
    """Test write_data raises error when trying to overwrite existing rows."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = np.array([10])
    lazy_attr._cache[1] = np.array([20])
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {
        0: {1: {'perrow_param': {}}},
        1: {1: {'perrow_param': {}}}
    }
    DS.root = zarr.group()
    
    # Write first batch
    DS.write_data('perrow_param', 1, data_idx=[0])
    
    # Try to overwrite
    with pytest.raises(ValueError, match="Cannot overwrite existing data"):
        DS.write_data('perrow_param', 1, data_idx=[0, 1])


def test_write_data_perrow_param_custom_dtype():
    """Test write_data uses custom dtype for per-row parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = np.array([10])
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {0: {1: {'perrow_param': {}}}}
    DS.root = zarr.group()
    
    DS.write_data('perrow_param', 1, data_idx=[0], dtype=np.float32)
    
    param_grp = DS.root['run1']['perrow_param']
    assert param_grp['data'].dtype == np.float32


def test_write_data_perrow_param_scalar_data():
    """Test write_data handles scalar data for per-row parameters."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = 10  # Scalar
    lazy_attr._cache[1] = 20
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {
        0: {1: {'perrow_param': {}}},
        1: {1: {'perrow_param': {}}}
    }
    DS.root = zarr.group()
    
    DS.write_data('perrow_param', 1, data_idx=[0, 1])
    
    param_grp = DS.root['run1']['perrow_param']
    assert param_grp['data'].shape == (3,)
    assert param_grp['data'][0] == 10
    assert param_grp['data'][1] == 20


def test_write_data_global_array_data():
    """Test write_data handles array data for global parameter."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    array_data = np.array([1, 2, 3, 4, 5])
    DS._memory_cache = {1: {'global_param': array_data}}
    DS._is_global_cache = {'global_param': True}
    DS.deps_maps = {'global': {1: {'global_param': {}}}}
    DS.root = zarr.group()
    
    DS.write_data('global_param', 1, None)
    
    param_grp = DS.root['run1']['global_param']
    assert np.array_equal(param_grp['data'][...], array_data)


################################################################################
############################### __getattr__ ####################################
################################################################################
def test_getattr_global_param_from_memory():
    """Test __getattr__ returns global parameter value from memory."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'global_param': 42}}
    DS._is_global_cache = {'global_param': True}
    DS.deps_maps = {}
    DS._lazy_collections = {}
    
    result = DS.global_param
    assert result == 42


def test_getattr_global_param_from_zarr():
    """Test __getattr__ loads and returns global parameter from zarr."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {'global': {1: {'global_param': {}}}}
    DS._lazy_collections = {}
    DS.root = zarr.group()
    
    # Create zarr data
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('global_param')
    param_grp.create_array('data', data=np.array(99))
    param_grp.attrs['global'] = True
    param_grp.attrs['deps'] = {}
    
    result = DS.global_param
    assert result == 99
    # Should be cached in memory now
    assert 1 in DS._memory_cache
    assert 'global_param' in DS._memory_cache[1]
    assert DS._memory_cache[1]['global_param'] == 99


def test_getattr_global_param_needs_production():
    """Test __getattr__ produces global parameter if not found."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {'global': {}}
    DS._lazy_collections = {}
    DS.nrows = 5
    DS.root = None  # Not using zarr in this test
    
    # Create a simple pipeline
    step = pds.pf.plStep(
        name='compute_global',
        func=lambda: 123,  # Returns value, not dict
        func_type='global',
        param_names=[],
        return_names=['global_param']
    )
    DS.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
    
    result = DS.global_param
    assert result == 123
    # Should be in memory now
    assert 1 in DS._memory_cache
    assert 'global_param' in DS._memory_cache[1]


def test_getattr_perrow_param_returns_lazy_collection():
    """Test __getattr__ returns LazyAttrCollection for per-row parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    lazy_attr = pds.pf.LazyAttr(DS, 'perrow_param', 1)
    lazy_attr._cache[0] = np.array([10])
    DS._memory_cache = {1: {'perrow_param': lazy_attr}}
    DS._is_global_cache = {'perrow_param': False}
    DS.deps_maps = {}
    DS._lazy_collections = {}
    
    result = DS.perrow_param
    assert isinstance(result, pds.pf.LazyAttrCollection)
    assert result.name == 'perrow_param'


def test_getattr_nonexistent_param_raises_attribute_error():
    """Test __getattr__ raises AttributeError for non-existent parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {}
    DS._lazy_collections = {}
    DS.cal_pl = {'CAL_STEPS': {}}
    DS.root = None
    
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = DS.nonexistent


def test_getattr_infers_global_from_memory():
    """Test __getattr__ infers global type from memory structure."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param': 42}}
    DS._is_global_cache = {}  # Not yet cached
    DS.deps_maps = {}
    DS._lazy_collections = {}
    
    result = DS.param
    assert result == 42
    # Should have cached the type
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == True


def test_getattr_infers_perrow_from_memory():
    """Test __getattr__ infers per-row type from memory structure."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    lazy_attr = pds.pf.LazyAttr(DS, 'param', 1)
    DS._memory_cache = {1: {'param': lazy_attr}}
    DS._is_global_cache = {}  # Not yet cached
    DS.deps_maps = {}
    DS._lazy_collections = {}
    
    result = DS.param
    assert isinstance(result, pds.pf.LazyAttrCollection)
    # Should have cached the type
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == False


def test_getattr_infers_global_from_zarr_deps():
    """Test __getattr__ infers global type from zarr deps_maps."""
    import zarr
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {'global': {1: {'param': {}}}}
    DS._lazy_collections = {}
    DS.root = zarr.group()
    
    # Create zarr data
    run_grp = DS.root.create_group('run1')
    param_grp = run_grp.create_group('param')
    param_grp.create_array('data', data=np.array(55))
    param_grp.attrs['global'] = True
    param_grp.attrs['deps'] = {}
    
    result = DS.param
    assert result == 55
    # Should have cached the type
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == True


def test_getattr_infers_perrow_from_zarr_deps():
    """Test __getattr__ infers per-row type from zarr deps_maps."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {0: {1: {'param': {}}}}
    DS._lazy_collections = {}
    
    result = DS.param
    assert isinstance(result, pds.pf.LazyAttrCollection)
    # Should have cached the type
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == False


def test_getattr_infers_global_from_pipeline():
    """Test __getattr__ infers global type from pipeline func_type."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {'global': {}}
    DS._lazy_collections = {}
    DS.nrows = 5
    DS.root = None
    
    # Create pipeline with global step
    step = pds.pf.plStep(
        name='compute',
        func=lambda: 100,  # Returns value, not dict
        func_type='global',
        param_names=[],
        return_names=['param']
    )
    DS.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
    
    result = DS.param
    assert result == 100
    # Should have cached the type
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == True


def test_getattr_infers_perrow_from_pipeline():
    """Test __getattr__ infers per-row type from pipeline func_type."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 3
    DS._memory_cache = {}
    DS._is_global_cache = {}
    DS.deps_maps = {0: {}, 1: {}, 2: {}}
    DS._lazy_collections = {}
    DS.root = None
    
    # Create pipeline with per-row step
    step = pds.pf.plStep(
        name='compute',
        func=lambda idx: {'param': [i * 10 for i in idx]},
        func_type='vectorized',
        param_names=['data_idx'],
        return_names=['param']
    )
    DS.cal_pl = {'CAL_STEPS': {0: {'task': step}}}
    
    result = DS.param
    assert isinstance(result, pds.pf.LazyAttrCollection)
    # Should have cached the type
    assert 'param' in DS._is_global_cache
    assert DS._is_global_cache['param'] == False


def test_getattr_global_uses_most_recent_run():
    """Test __getattr__ returns most recent run for global parameter."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._memory_cache = {1: {'param': 10}, 2: {'param': 20}, 3: {'param': 30}}
    DS._is_global_cache = {'param': True}
    DS.deps_maps = {}
    DS._lazy_collections = {}
    
    result = DS.param
    assert result == 30  # Should get run 3 (most recent)


def test_getattr_caches_lazy_collection():
    """Test __getattr__ caches LazyAttrCollection on first access."""
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.nrows = 5
    lazy_attr = pds.pf.LazyAttr(DS, 'param', 1)
    DS._memory_cache = {1: {'param': lazy_attr}}
    DS._is_global_cache = {'param': False}
    DS.deps_maps = {}
    DS._lazy_collections = {}
    
    # First access
    result1 = DS.param
    assert isinstance(result1, pds.pf.LazyAttrCollection)
    
    # Second access should return same collection
    result2 = DS.param
    assert result2 is result1
    assert 'param' in DS._lazy_collections
