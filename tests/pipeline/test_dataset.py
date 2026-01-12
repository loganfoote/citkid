import pytest 
import os 
from citkid.pipeline import dataset as pds

################################################################################
################################### __init__ ###################################
################################################################################
# __init__, _load_custom_steps, _load_yaml, _convert_yaml_to_steps
def test_paths_are_normalized(monkeypatch):
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._load_custom_steps = lambda: []  # Mock to avoid file dependency
    DS._load_yaml = lambda: {}  # Mock to avoid file dependency
    DS._convert_yaml_to_steps = lambda y: {} # Mock to avoid file dependency

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    # initialize with yaml file
    DS.__init__("a//b/../c.py", "x//y.yaml", "z//out.zarr")
    assert DS.custom_path == os.path.normpath("a//b/../c.py")
    assert DS.yaml_path == os.path.normpath("x//y.yaml")
    assert DS.zarr_path == os.path.normpath("z//out.zarr")
    assert DS.root == fake_root

    # initialize with yml file
    DS.__init__("a//b/../c.py", "x//y.yml", "z//out.zarr")
    assert DS.yaml_path == os.path.normpath("x//y.yml")

    # initialize without custom path
    DS.__init__(None, "x//y.yml", "z//out.zarr")
    assert DS.custom_path is None

@pytest.mark.parametrize("custom_path, yaml_path, zarr_path", [
    ("dir.txt", "file.yaml", "out.zarr"), # custom_path is not a .py file
    ("dir.py", "file.txt", "out.zarr"), # yaml_path is not a .yaml or .yml file
    ("dir.py", "file.yaml", "out.txt"), # zarr_path is not a .zarr file
])
def test_paths_validation(monkeypatch, custom_path, yaml_path, zarr_path):
    DS = pds.DataSet.__new__(pds.DataSet)
    DS._load_custom_steps = lambda: []
    DS._load_yaml = lambda: {}
    DS._convert_yaml_to_steps = lambda self, y: {}

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    with pytest.raises(ValueError):
        DS.__init__(custom_path, yaml_path, zarr_path)

def test_load_custom_steps(tmp_path):
    module = tmp_path / "custom_steps.py"
    m = "class Step:\n\tdef __init__(self, name):\n\t\tself.name = name\n" 
    m += "custom_cal_steps = [Step('custom1')]"
    module.write_text(m)

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.custom_path = tmp_path / "custom_steps.py"

    steps = DS._load_custom_steps()

    assert len(steps) == 1
    assert steps[0].name == "custom1"

def test_load_custom_steps_none():
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.custom_path = None 
    steps = DS._load_custom_steps()

    assert steps == []

def test_default_steps_added(monkeypatch):
    class Step:
        def __init__(self, name):
            self.name = name

    default = [Step("a"), Step("b")]

    monkeypatch.setattr(pds.pf, "default_cal_steps", default)

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.steps = [Step("a")]  # simulate custom step

    for step in pds.pf.default_cal_steps:
        if step.name not in [s.name for s in DS.steps]:
            DS.steps.append(step)

    names = [s.name for s in DS.steps]
    assert names == ["a", "b"]


def test_load_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("x: 1\ny: 2")

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.yaml_path = p

    data = DS._load_yaml()
    assert data == {"x": 1, "y": 2}


def test_init_calls_convert(monkeypatch):
    fake_yaml = {"pipeline": ["a", "b"]}

    monkeypatch.setattr(
        pds.DataSet, "_load_custom_steps", lambda self: []
    )
    monkeypatch.setattr(
        pds.DataSet, "_load_yaml", lambda self: fake_yaml
    )
    monkeypatch.setattr(
        pds.DataSet, "_convert_yaml_to_steps",
        lambda self, y: {}
    )
    monkeypatch.setattr(
        pds.pf, "default_cal_steps", []
    )

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    DS = pds.DataSet("custom_steps.py", "file.yaml", "out.zarr")

    assert DS.cal_pl == {}


def test_init_invalid_path(monkeypatch, tmp_path):
    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    with pytest.raises(FileNotFoundError):
        pds.DataSet("nonexistent_path.py", "nonexistent.yaml", 
                    'nonexistent_dir.zarr')
    
    with pytest.raises(FileNotFoundError):
        pds.DataSet(str(tmp_path / "nonexistent_path.py"), "nonexistent.yaml", 
                    str(tmp_path / 'fake_out.zarr'))
    
    with pytest.raises(FileNotFoundError):
        pds.DataSet("nonexistent_path.py", "file.yaml", 'fake_out.zarr')

    with pytest.raises(TypeError):
        pds.DataSet(123, "file.yaml", 'fake_out.zarr')

    with pytest.raises(TypeError):
        pds.DataSet(tmp_path, 456, 'fake_out.zarr')

################################################################################
############################## confirm_valid_path ##############################
################################################################################
def test_confirm_valid_path():
    pass 

################################################################################
################################# execute_path #################################
################################################################################
def test_execute_path():
    pass 

################################################################################
################################# __getattr__ ##################################
################################################################################
def test_get_attr():
    pass