import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr

from citkid.pipeline_v2 import dataset as dataset_module
from citkid.pipeline_v2.analysis import AnalysisRunner
from citkid.pipeline_v2.dataset import DataSet


@pytest.fixture
def pipeline_v2_files(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def import_base(base_value):\n"
        "    return 3, base_value\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('import_base', import_base, ['base_value'], ['nrows', 'base_value_stored'], 'global'),\n"
        "]\n",
        encoding="utf-8",
    )

    analysis_custom = tmp_path / "custom_analysis_steps.py"
    analysis_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def step1(data_idx, base_value_stored):\n"
        "    return base_value_stored + data_idx\n"
        "\n"
        "def step2(x, offset):\n"
        "    return x + offset\n"
        "\n"
        "def step3(y):\n"
        "    return y * 2\n"
        "\n"
        "custom_analysis_steps = [\n"
        "    plStep('step1', step1, ['data_idx', 'base_value_stored'], ['x'], 'per-row'),\n"
        "    plStep('step2', step2, ['x', 'offset'], ['y'], 'per-row'),\n"
        "    plStep('step3', step3, ['y'], ['z'], 'per-row'),\n"
        "]\n",
        encoding="utf-8",
    )

    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: import_base\n",
        encoding="utf-8",
    )

    analysis_yaml = tmp_path / "analysis.yaml"
    analysis_yaml.write_text(
        "ANALYSIS_STEPS:\n"
        "  1:\n"
        "    task: step1\n"
        "  2:\n"
        "    task: step2\n"
        "    params:\n"
        "      offset: 10\n"
        "  3:\n"
        "    task: step3\n",
        encoding="utf-8",
    )

    return {
        "zarr_path": tmp_path / "analysis.zarr",
        "cal_yaml": cal_yaml,
        "cal_custom": cal_custom,
        "analysis_yaml": analysis_yaml,
        "analysis_custom": analysis_custom,
    }


def test_rerunning_step_invalidates_downstream_outputs(pipeline_v2_files):
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    import_step = next(step for step in ds.cal_steps if step.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 5}, save=True)
    ar.execute_path(data_idx=0, verbose=False, save=True)

    assert ds.y[0] == 15
    assert ds.z[0] == 30

    step2 = next(step for step in ar.analysis_steps if step.name == "step2")
    ar.execute_step(step2, data_idx=0, user_params={"offset": 20}, save=True)

    assert ds.y[0] == 25
    with pytest.raises(AttributeError):
        _ = ds.z


def test_storing_one_row_keeps_other_rows_invalidated(pipeline_v2_files):
    """
    Check that writing one row after an all-row invalidation keeps the rest
    of the rows unavailable instead of exposing stale zarr data.
    """
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )
    import_step = next(s for s in ds.cal_steps if s.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 5}, save=True)
    for di in range(3):
        ar.execute_path(data_idx=di, verbose=False, save=True)
    assert ds.z[1] == 32

    ds.invalidate_memory_params(["z"])
    assert ds._invalidated_rows["z"] is None
    ds._store_param("z", 99, is_global=False, data_idx=0)

    assert ds.z[0] == 99
    assert ds._invalidated_rows["z"] == {1, 2}
    with pytest.raises(Exception):
        _ = ds.z[1]


def test_embedded_definitions_allow_reload_without_paths(pipeline_v2_files):
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )
    import_step = next(step for step in ds.cal_steps if step.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 7}, save=True)
    ar.execute_path(data_idx=0, verbose=False, save=True)

    ds_reloaded = DataSet(zarr_path=str(files["zarr_path"]))
    ar_reloaded = AnalysisRunner(ds_reloaded)

    assert int(ds_reloaded.nrows) == 3
    assert ds_reloaded.y[0] == 17
    assert ds_reloaded.z[0] == 34
    assert [step_dict["task"].name for step_dict in ar_reloaded.path] == [
        "step1",
        "step2",
        "step3",
    ]


def test_execute_step_does_not_rerun_earlier_analysis_steps(pipeline_v2_files):
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    import_step = next(step for step in ds.cal_steps if step.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 5}, save=True)

    step2 = next(step for step in ar.analysis_steps if step.name == "step2")
    with pytest.raises(ValueError, match="step1"):
        ar.execute_step(step2, data_idx=0, save=True)


def test_dataset_reloads_embedded_cal_definition_and_rejects_mismatch(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 2\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "]\n",
        encoding="utf-8",
    )
    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: load_data\n",
        encoding="utf-8",
    )
    zarr_path = tmp_path / "reload_test.zarr"

    ds = DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path=str(cal_yaml),
        custom_path=str(cal_custom),
    )
    assert int(ds.nrows) == 2

    ds_reloaded = DataSet(zarr_path=str(zarr_path))
    assert int(ds_reloaded.nrows) == 2
    assert ds_reloaded.cal_yaml_text == ds.cal_yaml_text
    assert ds_reloaded.cal_custom_source == ds.cal_custom_source

    mismatch_custom = tmp_path / "mismatch_custom_steps.py"
    mismatch_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 5\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="custom steps"):
        DataSet(
            zarr_path=str(zarr_path),
            cal_yaml_path=str(cal_yaml),
            custom_path=str(mismatch_custom),
        )


def _write_main_dir_files(directory):
    """
    Write calibration and analysis definitions that expose ``main_dir``.

    Parameters:
    directory (pathlib.Path): Directory to write the files into.

    Returns:
    files (dict): Paths keyed by ``cal_custom``, ``cal_yaml``,
        ``analysis_custom`` and ``analysis_yaml``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    cal_custom = directory / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "main_dir = '/old/raw/path'\n"
        "\n"
        "def load_data():\n"
        "    main_dir = 'local value'\n"
        "    return 1, globals()['main_dir']\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows', 'raw_dir'], 'global'),\n"
        "]\n",
        encoding="utf-8",
    )
    cal_yaml = directory / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: load_data\n",
        encoding="utf-8",
    )
    analysis_custom = directory / "custom_analysis_steps.py"
    analysis_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "main_dir = '/old/raw/path'\n"
        "\n"
        "def get_dir(raw_dir):\n"
        "    return main_dir\n"
        "\n"
        "custom_analysis_steps = [\n"
        "    plStep('get_dir', get_dir, ['raw_dir'], ['analysis_dir'], 'global'),\n"
        "]\n",
        encoding="utf-8",
    )
    analysis_yaml = directory / "analysis.yaml"
    analysis_yaml.write_text(
        "ANALYSIS_STEPS:\n"
        "  1:\n"
        "    task: get_dir\n",
        encoding="utf-8",
    )
    return {
        "cal_custom": cal_custom,
        "cal_yaml": cal_yaml,
        "analysis_custom": analysis_custom,
        "analysis_yaml": analysis_yaml,
    }


@pytest.mark.parametrize("new_dir", ["D:/new/raw/path", r"D:\new\raw\path"])
def test_moved_dataset_reloads_with_only_main_dir_overwrite(tmp_path, new_dir):
    files = _write_main_dir_files(tmp_path / "original")
    zarr_path = tmp_path / "original" / "main_dir_test.zarr"
    DS = DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    AnalysisRunner(
        DS,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    metadata = DS._read_metadata()
    assert not any(key.endswith("_path") for key in metadata)

    moved_zarr_path = tmp_path / "moved" / "main_dir_test.zarr"
    moved_zarr_path.parent.mkdir()
    shutil.move(str(zarr_path), str(moved_zarr_path))
    shutil.rmtree(tmp_path / "original")

    DS_moved = DataSet(zarr_path=str(moved_zarr_path), custom_main_dir_overwrite=new_dir)
    AR_moved = AnalysisRunner(DS_moved)
    AR_moved.execute_path()

    assert DS_moved.raw_dir == new_dir
    assert DS_moved.analysis_dir == new_dir
    assert DS_moved.cal_yaml_path is None
    assert DS_moved.custom_path is None


def test_dataset_accepts_custom_source_that_differs_only_in_main_dir(tmp_path):
    files = _write_main_dir_files(tmp_path)
    zarr_path = tmp_path / "main_dir_test.zarr"
    DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    source = files["cal_custom"].read_text(encoding="utf-8")
    files["cal_custom"].write_text(
        source.replace("main_dir = '/old/raw/path'", "main_dir = '/new/raw/path'"),
        encoding="utf-8",
    )

    DS = DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )

    assert DS.raw_dir == "/new/raw/path"


def test_dataset_warns_when_main_dir_overwrite_has_no_target(pipeline_v2_files, tmp_path):
    zarr_path = tmp_path / "no_main_dir.zarr"
    DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path=str(pipeline_v2_files["cal_yaml"]),
        custom_path=str(pipeline_v2_files["cal_custom"]),
    )

    with pytest.warns(UserWarning, match="main_dir"):
        DataSet(zarr_path=str(zarr_path), custom_main_dir_overwrite="/new/raw/path")


def test_moved_dataset_from_iq_template_and_default_yamls(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nrows, npts = 2, 5
    raw = zarr.open_group(str(raw_dir / "example.zarr"), mode="w")
    f = np.linspace(1e8, 1.1e8, nrows * npts).reshape(nrows, npts)
    raw.create_array("f", data=f)
    raw.create_array("z", data=np.ones((nrows, npts), dtype=np.complex128))
    for name in ("fres", "ares", "qres"):
        raw.create_array(name, data=np.ones(nrows))
    raw.create_array("res_idxs", data=np.arange(nrows))
    (raw_dir / "fres_init").mkdir()
    np.save(raw_dir / "fres_init" / "fres.npy", np.ones(nrows))

    template = Path(dataset_module.__file__).parent / "templates" / "custom_steps_iqonly_template.py"
    zarr_path = raw_dir / "output.zarr"
    DS = DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path="iq",
        custom_path=str(template),
        custom_main_dir_overwrite=str(raw_dir),
    )
    AnalysisRunner(DS, analysis_yaml_path="iq")
    assert DS.nrows == nrows

    templates_dir = template.parent
    metadata = DS._read_metadata()
    assert metadata["cal_yaml"] == (templates_dir / "cal-iqonly.yaml").read_text(encoding="utf-8")
    assert metadata["analysis_yaml"] == (templates_dir / "iq_analysis.yaml").read_text(encoding="utf-8")
    assert metadata["cal_custom_source"] == template.read_text(encoding="utf-8")
    assert not any(key.endswith("_path") for key in metadata)

    moved_dir = tmp_path / "moved_raw"
    shutil.move(str(raw_dir), str(moved_dir))
    DS_moved = DataSet(
        zarr_path=str(moved_dir / "output.zarr"),
        custom_main_dir_overwrite=str(moved_dir),
    )
    AR_moved = AnalysisRunner(DS_moved)

    assert DS_moved.nrows == nrows
    assert np.allclose(DS_moved.ff[1], f[1])
    assert len(AR_moved.path) > 0


def test_apply_cal_uses_replacements_without_storing_intermediates(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 3\n"
        "\n"
        "def load_zt(data_idx):\n"
        "    return data_idx + 1\n"
        "\n"
        "def calc_xt(zt):\n"
        "    return zt * 2\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "    plStep('load_zt', load_zt, ['data_idx'], ['zt'], 'per-row'),\n"
        "    plStep('calc_xt', calc_xt, ['zt'], ['xt'], 'vectorized'),\n"
        "]\n",
        encoding="utf-8",
    )
    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: load_data\n"
        "  2:\n"
        "    task: load_zt\n"
        "  3:\n"
        "    task: calc_xt\n",
        encoding="utf-8",
    )

    ds = DataSet(
        zarr_path=str(tmp_path / "apply_cal.zarr"),
        cal_yaml_path=str(cal_yaml),
        custom_path=str(cal_custom),
    )

    result = ds.apply_cal(
        data_indices=[0, 1],
        outputs=['xt'],
        replacements={'zt': np.array([10.0, 20.0])},
    )

    np.testing.assert_array_equal(result['xt'], np.array([20.0, 40.0]))
    assert 'xt' not in ds._param_meta
    assert 'zt' not in ds._param_meta


def test_apply_cal_single_data_idx_accepts_scalar_replacements(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 3\n"
        "\n"
        "def calc_xt(zt):\n"
        "    return zt * 2\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "    plStep('calc_xt', calc_xt, ['zt'], ['xt'], 'vectorized'),\n"
        "]\n",
        encoding="utf-8",
    )
    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: load_data\n"
        "  2:\n"
        "    task: calc_xt\n",
        encoding="utf-8",
    )

    ds = DataSet(
        zarr_path=str(tmp_path / "apply_cal_scalar.zarr"),
        cal_yaml_path=str(cal_yaml),
        custom_path=str(cal_custom),
    )

    result = ds.apply_cal(
        data_indices=1,
        outputs=['xt'],
        replacements={'zt': 10.0},
    )

    assert result['xt'] == 20.0


def test_apply_cal_single_data_idx_returns_consistent_multi_output_shapes(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 3\n"
        "\n"
        "def calc_outputs(zt):\n"
        "    return zt, zt * 2\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "    plStep('calc_outputs', calc_outputs, ['zt'], ['zt_out', 'xt'], 'vectorized'),\n"
        "]\n",
        encoding="utf-8",
    )
    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: load_data\n"
        "  2:\n"
        "    task: calc_outputs\n",
        encoding="utf-8",
    )

    ds = DataSet(
        zarr_path=str(tmp_path / "apply_cal_multi_output.zarr"),
        cal_yaml_path=str(cal_yaml),
        custom_path=str(cal_custom),
    )

    result = ds.apply_cal(
        data_indices=1,
        outputs=['zt_out', 'xt'],
        replacements={'zt': np.array([10.0, 11.0])},
    )

    np.testing.assert_array_equal(result['zt_out'], np.array([10.0, 11.0]))
    np.testing.assert_array_equal(result['xt'], np.array([20.0, 22.0]))


@pytest.fixture
def pipeline_v2_dependency_files(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def import_base(base_value):\n"
        "    return 3, base_value\n"
        "\n"
        "def build_final(y):\n"
        "    return y * 100\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('import_base', import_base, ['base_value'], ['nrows', 'base_value_stored'], 'global'),\n"
        "    plStep('build_final', build_final, ['y'], ['final'], 'per-row'),\n"
        "]\n",
        encoding="utf-8",
    )

    analysis_custom = tmp_path / "custom_analysis_steps.py"
    analysis_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def step1(data_idx, base_value_stored):\n"
        "    return base_value_stored + data_idx\n"
        "\n"
        "def step2(x, offset):\n"
        "    return x + offset\n"
        "\n"
        "def step3(y):\n"
        "    return y * 2\n"
        "\n"
        "custom_analysis_steps = [\n"
        "    plStep('step1', step1, ['data_idx', 'base_value_stored'], ['x'], 'per-row'),\n"
        "    plStep('step2', step2, ['x', 'offset'], ['y'], 'per-row'),\n"
        "    plStep('step3', step3, ['y'], ['z'], 'per-row'),\n"
        "]\n",
        encoding="utf-8",
    )

    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: import_base\n"
        "  2:\n"
        "    task: build_final\n",
        encoding="utf-8",
    )

    analysis_yaml = tmp_path / "analysis.yaml"
    analysis_yaml.write_text(
        "ANALYSIS_STEPS:\n"
        "  1:\n"
        "    task: step1\n"
        "  2:\n"
        "    task: step2\n"
        "    params:\n"
        "      offset: 10\n"
        "  3:\n"
        "    task: step3\n",
        encoding="utf-8",
    )

    return {
        "zarr_path": tmp_path / "analysis_dependency.zarr",
        "cal_yaml": cal_yaml,
        "cal_custom": cal_custom,
        "analysis_yaml": analysis_yaml,
        "analysis_custom": analysis_custom,
    }


def test_save_step_outputs_saves_inputs_and_deletes_targeted_downstream_rows(
    pipeline_v2_dependency_files,
):
    files = pipeline_v2_dependency_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    import_step = next(step for step in ds.cal_steps if step.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 5}, save=True)
    ar.execute_path(data_idx=[0, 1, 2], verbose=False, save=True)

    assert int(ds.offset[0]) == 10
    assert int(ds.y[2]) == 17

    _ = ds.final[[0, 1, 2]]
    ds.write_params(["final"], data_idx=[0, 1, 2])
    assert bool(ds.root["final"]["row_exists"][0])
    assert bool(ds.root["final"]["row_exists"][2])

    step2 = next(step for step in ar.analysis_steps if step.name == "step2")
    ar.execute_step(step2, data_idx=[0, 1], user_params={"offset": 20}, save=False)

    assert ds._has_rows("final", [2])
    assert not ds._has_rows("final", [0])
    assert bool(ds.root["final"]["row_exists"][0])

    ar.save_step_outputs("step2", data_idx=[0, 1])

    assert not bool(ds.root["final"]["row_exists"][0])
    assert not bool(ds.root["final"]["row_exists"][1])
    assert bool(ds.root["final"]["row_exists"][2])

    ds_reloaded = DataSet(zarr_path=str(files["zarr_path"]))
    ar_reloaded = AnalysisRunner(ds_reloaded)

    assert int(ds_reloaded.offset[0]) == 20
    assert int(ds_reloaded.offset[2]) == 10
    assert int(ds_reloaded.y[0]) == 25
    assert not bool(ds_reloaded.root["final"]["row_exists"][0])
    assert bool(ds_reloaded.root["final"]["row_exists"][2])
    assert [step_dict["task"].name for step_dict in ar_reloaded.path] == [
        "step1",
        "step2",
        "step3",
    ]


def test_per_row_saved_arrays_use_sharding(pipeline_v2_files):
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    import_step = next(step for step in ds.cal_steps if step.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 5}, save=True)
    ar.execute_path(data_idx=[0, 1, 2], verbose=False, save=True)

    y_data = ds.root["y"]["data"]
    assert y_data.chunks == (1,)
    assert y_data.shards == (256,)


@pytest.fixture
def pipeline_v2_global_res_files(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 4\n"
        "\n"
        "def build_final(vector):\n"
        "    return vector + 1\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "    plStep('build_final', build_final, ['vector'], ['final'], 'vectorized'),\n"
        "]\n",
        encoding="utf-8",
    )

    analysis_custom = tmp_path / "custom_analysis_steps.py"
    analysis_custom.write_text(
        "import numpy as np\n"
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_vector(multiplier):\n"
        "    return np.arange(4) * multiplier\n"
        "\n"
        "custom_analysis_steps = [\n"
        "    plStep('load_vector', load_vector, ['multiplier'], ['vector'], 'global-res'),\n"
        "]\n",
        encoding="utf-8",
    )

    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n"
        "    task: load_data\n"
        "  2:\n"
        "    task: build_final\n",
        encoding="utf-8",
    )

    analysis_yaml = tmp_path / "analysis.yaml"
    analysis_yaml.write_text(
        "ANALYSIS_STEPS:\n"
        "  1:\n"
        "    task: load_vector\n",
        encoding="utf-8",
    )

    return {
        "zarr_path": tmp_path / "analysis_global_res.zarr",
        "cal_yaml": cal_yaml,
        "cal_custom": cal_custom,
        "analysis_yaml": analysis_yaml,
        "analysis_custom": analysis_custom,
    }


def test_save_step_outputs_uses_last_step_rows_when_data_idx_omitted(
    pipeline_v2_dependency_files,
):
    files = pipeline_v2_dependency_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    import_step = next(step for step in ds.cal_steps if step.name == "import_base")
    ar.execute_step(import_step, user_params={"base_value": 5}, save=True)
    ar.execute_path(data_idx=[0, 1, 2], verbose=False, save=True)
    _ = ds.final[[0, 1, 2]]
    ds.write_params(["final"], data_idx=[0, 1, 2])

    step2 = next(step for step in ar.analysis_steps if step.name == "step2")
    ar.execute_step(step2, data_idx=[0, 1], user_params={"offset": 20}, save=False)
    ar.save_step_outputs("step2")

    ds_reloaded = DataSet(zarr_path=str(files["zarr_path"]))
    AnalysisRunner(ds_reloaded)

    assert int(ds_reloaded.offset[0]) == 20
    assert int(ds_reloaded.offset[1]) == 20
    assert int(ds_reloaded.offset[2]) == 10
    assert not bool(ds_reloaded.root["final"]["row_exists"][0])
    assert not bool(ds_reloaded.root["final"]["row_exists"][1])
    assert bool(ds_reloaded.root["final"]["row_exists"][2])


def test_global_res_save_overwrites_inputs_outputs_and_deletes_downstream(
    pipeline_v2_global_res_files,
):
    files = pipeline_v2_global_res_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )

    import_step = next(step for step in ds.cal_steps if step.name == "load_data")
    ar.execute_step(import_step, save=True)

    load_vector_step = next(step for step in ar.analysis_steps if step.name == "load_vector")
    ar.execute_step(load_vector_step, user_params={"multiplier": 2}, save=True)

    np.testing.assert_array_equal(ds.vector[[0, 1, 2, 3]], np.array([0, 2, 4, 6]))
    np.testing.assert_array_equal(ds.final[[0, 1, 2, 3]], np.array([1, 3, 5, 7]))
    ds.write_params(["final"], data_idx=[0, 1, 2, 3])
    assert bool(ds.root["final"]["row_exists"][3])

    with pytest.raises(ValueError, match="allow_global_step_overwrite"):
        ar.execute_step(load_vector_step, user_params={"multiplier": 5}, save=False)

    ar.execute_step(
        load_vector_step,
        user_params={"multiplier": 5},
        save=False,
        allow_global_step_overwrite=True,
    )

    assert not ds._has_rows("final", [0, 1, 2, 3])
    assert bool(ds.root["final"]["row_exists"][3])

    ar.save_step_outputs("load_vector")

    ds_reloaded = DataSet(zarr_path=str(files["zarr_path"]))
    AnalysisRunner(ds_reloaded)

    assert int(ds_reloaded.multiplier) == 5
    np.testing.assert_array_equal(ds_reloaded.vector[[0, 1, 2, 3]], np.array([0, 5, 10, 15]))
    assert "final" not in ds_reloaded.root


def test_get_downstream_calibration_params_is_recursive(pipeline_v2_global_res_files):
    files = pipeline_v2_global_res_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )

    assert ds.get_downstream_calibration_params(["vector"]) == ["final"]
    assert ds.get_downstream_calibration_params(["multiplier"]) == []

def _make_io_dataset(tmp_path, nrows=40):
    """
    Build a DataSet with global, global-res and per-row calibration steps.

    Parameters:
    tmp_path (pathlib.Path): Directory for the files and zarr store.
    nrows (int): Number of rows.

    Returns:
    ds (DataSet): The dataset.
    """
    cal_custom = tmp_path / "io_cal_steps.py"
    cal_custom.write_text(
        "import numpy as np\n"
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        f"def load_n():\n    return {nrows}\n"
        "\n"
        f"def load_all():\n    return np.arange({nrows}) * 10.0\n"
        "\n"
        "def load_x(data_idx):\n    return data_idx * 1.0\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_n', load_n, [], ['nrows'], 'global'),\n"
        "    plStep('load_all', load_all, [], ['g'], 'global-res'),\n"
        "    plStep('load_x', load_x, ['data_idx'], ['x'], 'per-row'),\n"
        "]\n",
        encoding="utf-8",
    )
    cal_yaml = tmp_path / "io_cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n"
        "  1:\n    task: load_n\n"
        "  2:\n    task: load_all\n"
        "  3:\n    task: load_x\n",
        encoding="utf-8",
    )
    return DataSet(
        zarr_path=str(tmp_path / "io.zarr"),
        cal_yaml_path=str(cal_yaml),
        custom_path=str(cal_custom),
    )


def _save_rows(ds, name, rows, offset=0.0):
    """
    Store and save per-row values ``row + offset`` for a parameter.

    Parameters:
    ds (DataSet): Dataset to write to.
    name (str): Parameter name.
    rows (list of int): Rows to write.
    offset (float): Value offset.
    """
    rows = np.asarray(rows, dtype=np.int32)
    ds._store_param(
        name, rows + offset, is_global=False, data_idx=rows,
        pipeline_scope="analysis", step_name="test", step_index=1, save=True,
    )


def test_concurrent_writes_from_two_datasets_on_one_store(tmp_path):
    ds_a = _make_io_dataset(tmp_path)
    group = ds_a.root
    ds_b = DataSet(zarr_path=group)
    nrows = int(ds_a.nrows)
    errors = []

    def writer(ds, rows):
        try:
            for di in rows:
                _save_rows(ds, "p", [di], offset=0.5)
                ds._has_rows("p", [di])
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(ds_a, range(0, nrows, 2))),
        threading.Thread(target=writer, args=(ds_b, range(1, nrows, 2))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert bool(np.all(group["p"]["row_exists"][...]))
    np.testing.assert_array_equal(group["p"]["data"][...], np.arange(nrows) + 0.5)


def test_dataset_sees_parameters_saved_by_another_dataset(tmp_path):
    ds_a = _make_io_dataset(tmp_path)
    ds_b = DataSet(zarr_path=ds_a.root)
    assert "p" not in ds_b._param_meta

    _save_rows(ds_a, "p", [3, 4])

    assert ds_b.p[4] == 4.0
    assert ds_b._param_meta["p"]["step_name"] == "test"


def test_write_per_row_param_batches_non_contiguous_rows(tmp_path):
    ds = _make_io_dataset(tmp_path)

    _save_rows(ds, "p", [7, 2, 11, 2])
    _save_rows(ds, "p", [3, 4, 5], offset=100.0)

    exists = ds.root["p"]["row_exists"][...]
    assert sorted(np.flatnonzero(exists)) == [2, 3, 4, 5, 7, 11]
    data = ds.root["p"]["data"][...]
    np.testing.assert_array_equal(data[[2, 7, 11]], [2.0, 7.0, 11.0])
    np.testing.assert_array_equal(data[[3, 4, 5]], [103.0, 104.0, 105.0])
    assert ds.root["p"].attrs["step_name"] == "test"
    assert "write_time" in ds.root["p"].attrs


def test_delete_saved_param_ignores_unsaved_rows(tmp_path):
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])

    ds.delete_saved_params(["p"], data_idx=[5, 6])
    assert sorted(np.flatnonzero(ds.root["p"]["row_exists"][...])) == [1, 2]

    ds.delete_saved_params(["p"], data_idx=[1])
    assert sorted(np.flatnonzero(ds.root["p"]["row_exists"][...])) == [2]

    ds.delete_saved_params(["p"], data_idx=[2])
    assert "p" not in ds.root


def test_release_rows_drops_row_cache_but_keeps_global_res(tmp_path):
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])
    assert ds.x[1] == 1.0
    assert ds.g[1] == 10.0

    ds.release_rows([1])

    assert 1 not in ds._per_row_cache["p"]._cache
    assert 1 not in ds._per_row_cache["x"]._cache
    assert 1 in ds._per_row_cache["g"]._cache
    assert ds.p[1] == 1.0
    assert ds.x[1] == 1.0


def test_retry_io_retries_permission_errors(monkeypatch):
    monkeypatch.setattr(dataset_module, "_IO_RETRY_DELAY", 0.0)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("locked")
        return "ok"

    assert dataset_module._retry_io(flaky) == "ok"
    assert len(calls) == 3

    def always_locked():
        raise PermissionError("locked")

    with pytest.raises(PermissionError):
        dataset_module._retry_io(always_locked)


def test_write_buffer_defers_rows_until_consolidated(tmp_path):
    ds = DataSet(zarr_path=_make_io_dataset(tmp_path).root, write_buffer=True)
    _save_rows(ds, "p", [2, 5, 20])

    group = ds.root["p"]
    assert not np.any(group["row_exists"][...])
    assert sorted(np.flatnonzero(group["pending_row_exists"][...])) == [2, 5, 20]
    assert group["data"].shards == (256,)
    assert group["pending_data"].shards is None

    reopened = DataSet(zarr_path=ds.root)
    assert reopened.p[5] == 5.0
    assert reopened._has_rows("p", [2, 5])
    assert list(np.flatnonzero(reopened.saved_row_mask("p"))) == [2, 5, 20]

    assert ds.consolidate_storage() == 3
    assert "pending_data" not in group
    assert "pending_row_exists" not in group
    assert sorted(np.flatnonzero(group["row_exists"][...])) == [2, 5, 20]
    np.testing.assert_array_equal(group["data"][[2, 5]], [2.0, 5.0])
    assert ds.consolidate_storage() == 0


def test_pending_rows_override_and_are_cleared_by_direct_writes(tmp_path):
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])
    buffered = DataSet(zarr_path=ds.root, write_buffer=True)
    _save_rows(buffered, "p", [1], offset=100.0)

    assert DataSet(zarr_path=ds.root).p[1] == 101.0

    direct = DataSet(zarr_path=ds.root)
    _save_rows(direct, "p", [1], offset=200.0)

    assert not bool(ds.root["p"]["pending_row_exists"][1])
    assert DataSet(zarr_path=ds.root).p[1] == 201.0


def test_delete_saved_params_skips_unsaved_names(tmp_path, monkeypatch):
    """Several names: one store listing, and no lookups for names never saved."""
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])
    looked_up = []
    original = ds._delete_saved_param
    monkeypatch.setattr(ds, "_delete_saved_param",
                        lambda name, data_idx=None: (looked_up.append(name), original(name, data_idx)))

    ds.delete_saved_params(["never_saved", "p", "also_never"], data_idx=[1])

    assert looked_up == ["p"]
    assert sorted(np.flatnonzero(ds.root["p"]["row_exists"][...])) == [2]


def test_delete_saved_params_without_listing_checks_each_name(tmp_path, monkeypatch):
    """If the store can't be listed, every name is still handled."""
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])
    monkeypatch.setattr(ds, "_child_names", lambda: None)

    ds.delete_saved_params(["never_saved", "p"], data_idx=[1])

    assert sorted(np.flatnonzero(ds.root["p"]["row_exists"][...])) == [2]


def test_delete_saved_param_clears_pending_rows(tmp_path):
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1])
    buffered = DataSet(zarr_path=ds.root, write_buffer=True)
    _save_rows(buffered, "p", [2])

    buffered.delete_saved_params(["p"], data_idx=[2])
    assert not bool(ds.root["p"]["pending_row_exists"][2])
    assert bool(ds.root["p"]["row_exists"][1])

    buffered.delete_saved_params(["p"], data_idx=[1])
    assert "p" not in ds.root


def test_consolidate_merges_rows_across_shards(tmp_path):
    ds = DataSet(zarr_path=_make_io_dataset(tmp_path, nrows=600).root, write_buffer=True)
    rows = [0, 255, 256, 511, 599]
    _save_rows(ds, "p", rows)

    assert ds.consolidate_storage() == len(rows)
    np.testing.assert_array_equal(ds.root["p"]["data"][rows], np.asarray(rows, dtype=float))
    assert list(np.flatnonzero(ds.root["p"]["row_exists"][...])) == rows


@pytest.mark.parametrize("write_buffer", [False, True])
def test_non_contiguous_rows_of_2d_parameter(tmp_path, write_buffer):
    ds = DataSet(zarr_path=_make_io_dataset(tmp_path, nrows=300).root, write_buffer=write_buffer)
    rows = np.array([0, 1, 3, 4, 5, 260, 262], dtype=np.int32)
    values = np.stack([np.full(3, r, dtype=float) for r in rows])
    ds._store_param(
        "q", values, is_global=False, data_idx=rows,
        pipeline_scope="analysis", step_name="test", step_index=1, save=True,
    )
    ds.consolidate_storage()

    group = ds.root["q"]
    assert "pending_data" not in group
    assert list(np.flatnonzero(group["row_exists"][...])) == list(rows)
    np.testing.assert_array_equal(group["data"][list(rows)], values)
    assert not np.any(group["data"][2])


def test_write_params_produces_unloaded_calibration_params(tmp_path):
    ds = _make_io_dataset(tmp_path, nrows=6)
    assert "x" not in ds._per_row_cache  # never accessed

    ds.write_params(["x"], data_idx=[0, 2])
    ds.write_params(["nrows"])
    ds.write_params(["g"], data_idx=[4])

    root = ds.root
    assert list(np.flatnonzero(root["x"]["row_exists"][...])) == [0, 2]
    np.testing.assert_array_equal(root["x"]["data"][[0, 2]], [0.0, 2.0])
    assert int(root["nrows"]["data"][...]) == 6
    assert root["x"].attrs["pipeline_scope"] == "cal"
    assert float(root["g"]["data"][4]) == 40.0


def test_write_params_without_rows_saves_every_calibration_row(tmp_path):
    ds = _make_io_dataset(tmp_path, nrows=5)

    ds.write_params(["x"])

    assert bool(np.all(ds.root["x"]["row_exists"][...]))
    np.testing.assert_array_equal(ds.root["x"]["data"][...], np.arange(5.0))


def test_write_params_loads_saved_analysis_rows(tmp_path):
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])
    reopened = DataSet(zarr_path=ds.root)

    reopened.write_params(["p"], data_idx=[1])  # not in memory: loaded from zarr

    np.testing.assert_array_equal(ds.root["p"]["data"][[1, 2]], [1.0, 2.0])
    with pytest.raises(ValueError, match="cannot be produced"):
        reopened.write_params(["p"], data_idx=[5])
    with pytest.raises(ValueError, match="not available to save"):
        reopened.write_params(["unknown"])


def test_copy_of_embedded_dataset_shares_data_not_memory(tmp_path):
    ds = _make_io_dataset(tmp_path)
    _save_rows(ds, "p", [1, 2])
    ds.write_buffer = True
    ds.custom_main_dir_overwrite = None

    copy = ds.copy()

    assert copy is not ds
    assert copy.root.path == ds.root.path
    assert copy.write_buffer is True
    assert copy.p[2] == 2.0                      # stored data is shared
    assert copy._per_row_cache is not ds._per_row_cache
    _save_rows(copy, "p", [3])
    assert ds._has_rows("p", [3])                # and writes go to the same store


def test_copy_of_dataset_built_from_custom_cal_steps(tmp_path):
    from citkid.pipeline_v2.framework import plStep

    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text(
        "CAL_STEPS:\n  1:\n    task: load_n\n  2:\n    task: load_x\n", encoding="utf-8")
    steps = [
        plStep("load_n", lambda: 4, [], ["nrows"], "global"),
        plStep("load_x", lambda data_idx: data_idx * 3.0, ["data_idx"], ["x"], "per-row"),
    ]
    root = zarr.open_group(str(tmp_path / "steps.zarr"), mode="w")
    ds = DataSet(zarr_path=root, cal_yaml_path=str(cal_yaml), custom_cal_steps=steps)

    copy = ds.copy()

    assert int(copy.nrows) == 4
    assert copy.x[2] == 6.0
    assert copy.write_buffer is False


def test_analysis_yaml_with_misspelled_key_is_rejected(pipeline_v2_files):
    """
    Check that a typo such as ``parms:`` raises instead of being ignored.
    """
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    bad_yaml = files["analysis_yaml"].with_name("bad.yaml")
    bad_yaml.write_text(
        "ANALYSIS_STEPS:\n"
        "  1:\n"
        "    task: step1\n"
        "  2:\n"
        "    task: step2\n"
        "    parms:\n"
        "      offset: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parms"):
        AnalysisRunner(
            ds,
            analysis_yaml_path=str(bad_yaml),
            custom_path=str(files["analysis_custom"]),
        )


def test_analysis_runner_has_no_failures_before_running(pipeline_v2_files):
    """
    Check that ``_last_failures`` exists (empty) before any step has run.
    """
    files = pipeline_v2_files
    ds = DataSet(
        zarr_path=str(files["zarr_path"]),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )
    assert ar._last_failures == {}


@pytest.mark.parametrize("cal_alias, analysis_alias, template", [
    ("ts", "ts", "custom_steps_template.py"),
    ("ts_offres", "ts_offres", "custom_steps_template.py"),
    ("iq", "iq", "custom_steps_iqonly_template.py"),
])
def test_every_template_pair_builds_a_runner(
    tmp_path, cal_alias, analysis_alias, template
):
    """
    Check that each shipped YAML alias pair and its custom-steps template
    parse, pass the structure checks, and build a DataSet and runner.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nrows, npts = 2, 5
    raw = zarr.open_group(str(raw_dir / "example.zarr"), mode="w")
    raw.create_array("f", data=np.ones((nrows, npts)))
    raw.create_array("z", data=np.ones((2, nrows, npts)))
    for name in ("fres", "fres_all", "ares", "qres"):
        raw.create_array(name, data=np.ones(nrows))
    raw.create_array("res_idxs", data=np.arange(nrows))
    raw.create_array("ts_00/dt", data=np.array(1e-3))
    (raw_dir / "fres_init").mkdir()
    np.save(raw_dir / "fres_init" / "fres.npy", np.ones(nrows))

    templates = Path(dataset_module.__file__).parent / "templates"
    DS = DataSet(
        zarr_path=str(raw_dir / "output.zarr"),
        cal_yaml_path=cal_alias,
        custom_path=str(templates / template),
        custom_main_dir_overwrite=str(raw_dir),
    )
    AR = AnalysisRunner(DS, analysis_yaml_path=analysis_alias)

    assert DS.nrows == nrows
    assert [s["task"].name for s in AR.path][:2] == [
        "make_fr_spans", "fit_gain"]
