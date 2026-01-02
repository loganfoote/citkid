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
    DS._convert_yaml_to_steps = lambda y: []  # Mock to avoid file dependency

    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    DS.__init__("a//b/../c", "x//y.yaml", "z//out.zarr")

    assert DS.directory == os.path.normpath("a//b/../c")
    assert DS.yaml_path == os.path.normpath("x//y.yaml")
    assert DS.zarr_path == os.path.normpath("z//out.zarr")
    assert DS.root == fake_root

def test_load_custom_steps(tmp_path):
    module = tmp_path / "custom_steps.py"
    module.write_text("""
class Step:
    def __init__(self, name):
        self.name = name

custom_steps = [Step("custom1")]
""")

    DS = pds.DataSet.__new__(pds.DataSet)
    DS.directory = tmp_path

    steps = DS._load_custom_steps()

    assert len(steps) == 1
    assert steps[0].name == "custom1"

def test_no_custom_steps(tmp_path):
    DS = pds.DataSet.__new__(pds.DataSet)
    DS.directory = tmp_path

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
        lambda self, y: f"converted-{y}"
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

    DS = pds.DataSet("dir", "file.yaml", "out.zarr")

    assert DS.cal_pl == f"converted-{fake_yaml}" 


def test_init_invalid_path(monkeypatch, tmp_path):
    # patch zarr.open_group loading
    fake_root = object()
    def fake_open_group(path, mode):
        # optional: assert inside the fake
        assert mode == "a"
        return fake_root
    monkeypatch.setattr("zarr.open_group", fake_open_group)

    with pytest.raises(FileNotFoundError):
        pds.DataSet("nonexistent_dir", "nonexistent.yaml", 'nonexistent_dir')
    
    with pytest.raises(FileNotFoundError):
        pds.DataSet(str(tmp_path), "nonexistent.yaml", 'fake_out.zarr')
    
    with pytest.raises(FileNotFoundError):
        pds.DataSet("nonexistent_dir", "file.yaml", 'fake_out.zarr')

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