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

