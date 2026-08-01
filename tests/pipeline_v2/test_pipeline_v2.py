from pathlib import Path

import pytest

from citkid.pipeline_v2.analysis import AnalysisRunner
from citkid.pipeline_v2.dataset import DataSet


@pytest.fixture
def pipeline_v2_files(tmp_path):
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline.framework import plStep\n"
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
        "from citkid.pipeline.framework import plStep\n"
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
    ar.execute_path(data_idx=0, verbose=False, save_override=True)

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
    ar.execute_path(data_idx=0, verbose=False, save_override=True)

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