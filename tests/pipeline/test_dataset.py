import pytest 
import os 
from citkid.pipeline import dataset as pds

################################################################################
################################### __init__ ###################################
################################################################################
# __init__, _load_custom_steps, _load_yaml, _convert_yaml_to_steps
def test_paths_are_normalized(monkeypatch):
    DS = pds.DataSet.__new__(pds.DataSet)
    # patch subfunctions to avoid file I/O and custom code
    monkeypatch.setattr(pds.DataSet, "_load_custom_steps", lambda self: ([], []))
    monkeypatch.setattr(pds.DataSet, "_load_yaml", lambda self, path: {})
    monkeypatch.setattr(pds.DataSet, "_convert_yaml_to_steps", lambda self, y: {})

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    # initialize with yaml file (new signature: zarr_path, cal_yaml_path, custom_path)
    DS.__init__("z//out.zarr", "x//y.yaml", "a//b/../c.py")
    assert DS.custom_path == os.path.normpath("a//b/../c.py")
    assert DS.cal_yaml_path == os.path.normpath("x//y.yaml")
    assert DS.zarr_path == os.path.normpath("z//out.zarr")
    assert DS.root == fake_root

    # initialize with yml file (order: zarr, cal_yaml, custom)
    DS.__init__("z//out.zarr", "x//y.yml", "a//b/../c.py")
    assert DS.cal_yaml_path == os.path.normpath("x//y.yml")

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
    monkeypatch.setattr(pds.DataSet, "_load_custom_steps", lambda self: ([], []))
    monkeypatch.setattr(pds.DataSet, "_load_yaml", lambda self, path: {})
    monkeypatch.setattr(pds.DataSet, "_convert_yaml_to_steps", lambda self, y: {})

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

    cal_steps, analysis_steps = DS._load_custom_steps()

    assert len(cal_steps) == 1
    assert cal_steps[0].name == "custom1"
    assert isinstance(analysis_steps, list)

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
    DS.cal_steps = [Step("a")]  # simulate custom step list
    DS.analysis_steps = []

    for step in pds.pf.default_cal_steps:
        if step.name not in [s.name for s in DS.cal_steps]:
            DS.cal_steps.append(step)
    for step in pds.pf.default_analysis_steps:
        if step.name not in [s.name for s in DS.analysis_steps]:
            DS.analysis_steps.append(step)

    cal_names = [s.name for s in DS.cal_steps]
    assert cal_names == ["a", "b"]


def test_load_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("x: 1\ny: 2")

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.cal_yaml_path = p

    data = DS._load_yaml(str(p))
    assert data == {"x": 1, "y": 2}


def test_init_calls_convert(monkeypatch):
    fake_yaml = {"pipeline": ["a", "b"]}

    monkeypatch.setattr(
        pds.DataSet, "_load_custom_steps", lambda self: ([], [])
    )
    monkeypatch.setattr(
        pds.DataSet, "_load_yaml", lambda self, path: fake_yaml
    )
    monkeypatch.setattr(
        pds.DataSet, "_convert_yaml_to_steps",
        lambda self, y: {}
    )
    monkeypatch.setattr(
        pds.pf, "default_cal_steps", []
    )
    monkeypatch.setattr(
        pds.pf, "default_analysis_steps", []
    )

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

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