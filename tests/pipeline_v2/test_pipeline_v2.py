from pathlib import Path

import numpy as np
import pytest

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


def test_dataset_custom_main_directory_overwrite_applies_to_embedded_source(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "main_directory = '/old/raw/path'\n"
        "\n"
        "def load_data():\n"
        "    return 1, main_directory\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows', 'raw_dir'], 'global'),\n"
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
    zarr_path = tmp_path / "main_dir_test.zarr"

    DataSet(
        zarr_path=str(zarr_path),
        cal_yaml_path=str(cal_yaml),
        custom_path=str(cal_custom),
    )
    ds_reloaded = DataSet(
        zarr_path=str(zarr_path),
        custom_main_directory_overwrite="D:/new/raw/path",
    )

    assert ds_reloaded.raw_dir == "D:/new/raw/path"


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