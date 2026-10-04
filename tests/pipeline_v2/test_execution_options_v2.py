"""
Tests for the pipeline_v2 execution options of AnalysisRunner.

- ``vectorize`` (execute_path, execute_step): if False, steps with func_type
  'vectorized' run one row at a time instead of in one call.
- ``path_per_row`` (execute_path): if True, the full remaining path runs for
  one row before the next row starts.
"""

import pytest
import numpy as np

from citkid.pipeline_v2.analysis import AnalysisRunner
from citkid.pipeline_v2.dataset import DataSet


@pytest.fixture
def options_fixture(tmp_path):
    """
    Create a pipeline with global-res, vectorized and per-row analysis steps.

    The vectorized step logs the rows of each call to ``call_log`` and raises
    for rows listed in ``fail_rows``.

    Parameters:
    tmp_path (pathlib.Path): pytest temporary directory.

    Returns:
    files (dict): Paths keyed by ``zarr_path``, ``cal_yaml``, ``cal_custom``,
        ``analysis_yaml`` and ``analysis_custom``.
    """
    cal_custom = tmp_path / "custom_cal_steps.py"
    cal_custom.write_text(
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_data():\n"
        "    return 10  # 10 rows\n"
        "\n"
        "custom_cal_steps = [\n"
        "    plStep('load_data', load_data, [], ['nrows'], 'global'),\n"
        "]\n",
        encoding="utf-8",
    )

    analysis_custom = tmp_path / "custom_analysis_steps.py"
    analysis_custom.write_text(
        "import numpy as np\n"
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "call_log = []\n"
        "fail_rows = set()\n"
        "\n"
        "def load_vector():\n"
        "    return np.arange(10) * 2\n"
        "\n"
        "def vectorized_multiply(vector, data_idx):\n"
        "    rows = [int(r) for r in np.atleast_1d(data_idx)]\n"
        "    call_log.append(rows)\n"
        "    if fail_rows & set(rows):\n"
        "        raise ValueError('bad row')\n"
        "    return vector * 3\n"
        "\n"
        "def per_row_add(result, data_idx):\n"
        "    return result + data_idx\n"
        "\n"
        "custom_analysis_steps = [\n"
        "    plStep('load_vector', load_vector, [], ['vector'], 'global-res'),\n"
        "    plStep('vec_mult', vectorized_multiply, ['vector', 'data_idx'], ['result'], 'vectorized'),\n"
        "    plStep('row_add', per_row_add, ['result', 'data_idx'], ['final'], 'per-row'),\n"
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

    analysis_yaml = tmp_path / "analysis.yaml"
    analysis_yaml.write_text(
        "ANALYSIS_STEPS:\n"
        "  1:\n"
        "    task: load_vector\n"
        "  2:\n"
        "    task: vec_mult\n"
        "  3:\n"
        "    task: row_add\n",
        encoding="utf-8",
    )

    return {
        "zarr_path": tmp_path / "analysis.zarr",
        "cal_yaml": cal_yaml,
        "cal_custom": cal_custom,
        "analysis_yaml": analysis_yaml,
        "analysis_custom": analysis_custom,
    }


def _make_runner(files, zarr_name="analysis.zarr"):
    """
    Build a DataSet and AnalysisRunner from the fixture files.

    Parameters:
    files (dict): Output of ``options_fixture``.
    zarr_name (str): Name of the zarr store in the fixture directory.

    Returns:
    ds (DataSet): The dataset.
    ar (AnalysisRunner): The runner.
    module_globals (dict): Globals of the custom analysis module, holding
        ``call_log`` and ``fail_rows``.
    """
    ds = DataSet(
        zarr_path=str(files["zarr_path"].parent / zarr_name),
        cal_yaml_path=str(files["cal_yaml"]),
        custom_path=str(files["cal_custom"]),
    )
    ar = AnalysisRunner(
        ds,
        analysis_yaml_path=str(files["analysis_yaml"]),
        custom_path=str(files["analysis_custom"]),
    )
    vec_mult = next(step for step in ar.analysis_steps if step.name == "vec_mult")
    return ds, ar, vec_mult.func.__globals__


def _step(ar, name):
    """
    Return an analysis step by name.

    Parameters:
    ar (AnalysisRunner): Runner.
    name (str): Step name.

    Returns:
    step (plStep): The step.
    """
    return next(step for step in ar.analysis_steps if step.name == name)


class TestVectorize:
    """Tests for the ``vectorize`` option."""

    def test_vectorized_step_runs_once_by_default(self, options_fixture):
        ds, ar, module = _make_runner(options_fixture)

        ar.execute_path(data_idx=np.arange(10), verbose=False, save=True)

        assert module["call_log"] == [list(range(10))]
        np.testing.assert_array_equal(ds.final[np.arange(10)], np.arange(10) * 7)

    def test_vectorize_false_runs_one_row_at_a_time(self, options_fixture):
        ds, ar, module = _make_runner(options_fixture)

        ar.execute_path(data_idx=np.arange(10), vectorize=False, verbose=False, save=True)

        assert module["call_log"] == [[row] for row in range(10)]
        np.testing.assert_array_equal(ds.final[np.arange(10)], np.arange(10) * 7)

    def test_execute_step_vectorize_false(self, options_fixture):
        ds, ar, module = _make_runner(options_fixture)
        ar.execute_step(_step(ar, "load_vector"), save=True)

        ar.execute_step(_step(ar, "vec_mult"), data_idx=[0, 1], vectorize=False, save=True)

        assert module["call_log"] == [[0], [1]]
        np.testing.assert_array_equal(ds.result[[0, 1]], np.array([0, 6]))

    def test_vectorize_false_records_failing_rows(self, options_fixture):
        ds, ar, module = _make_runner(options_fixture)
        ar.execute_step(_step(ar, "load_vector"), save=True)
        module["fail_rows"].add(1)

        with pytest.raises(ValueError, match="bad row"):
            ar.execute_step(_step(ar, "vec_mult"), data_idx=[0, 1, 2], save=True)

        with pytest.warns(RuntimeWarning, match=r"failed for 1 row\(s\): \[1\]"):
            ar.execute_step(_step(ar, "vec_mult"), data_idx=[0, 1, 2], vectorize=False, save=True)
        assert ds._has_rows("result", [0, 2])
        assert not ds._has_rows("result", [1])

    def test_failure_warning_shows_the_first_error(self, options_fixture):
        ds, ar, module = _make_runner(options_fixture)
        ar.execute_step(_step(ar, "load_vector"), save=True)
        module["fail_rows"].add(1)

        with pytest.warns(RuntimeWarning, match=r"(?s)First failure \(data_idx 1\):.*bad row"):
            ar.execute_step(_step(ar, "vec_mult"), data_idx=[0, 1, 2], vectorize=False, save=True)
        assert "In analysis step 'vec_mult'" in ar._last_failures[1]

    @pytest.mark.parametrize("path_per_row", [False, True])
    def test_execute_path_drops_failed_rows_from_later_steps(self, options_fixture, path_per_row):
        ds, ar, module = _make_runner(options_fixture)
        module["fail_rows"].add(1)

        with pytest.warns(RuntimeWarning, match="bad row"):
            ar.execute_path(data_idx=[0, 1, 2], vectorize=False, path_per_row=path_per_row,
                            save=True, verbose=False)

        np.testing.assert_array_equal(ds.final[[0, 2]], np.array([0, 14]))  # 7 * data_idx
        assert not ds._has_rows("final", [1])

    @pytest.mark.parametrize("path_per_row", [False, True])
    def test_execute_path_raises_when_every_row_fails(self, options_fixture, path_per_row):
        _, ar, module = _make_runner(options_fixture)
        module["fail_rows"].update({0, 1})

        with pytest.warns(RuntimeWarning):
            with pytest.raises(RuntimeError, match=r"(?s)Every requested row failed\. First "
                                                   r"failure: step 'vec_mult' for data_idx 0:"
                                                   r".*bad row"):
                ar.execute_path(data_idx=[0, 1], vectorize=False, path_per_row=path_per_row,
                                verbose=False)

    @pytest.mark.parametrize("start, end, expected", [
        (0, None, ["load_vector", "vec_mult", "row_add"]),
        (1, None, ["vec_mult", "row_add"]),
        (0, 2, ["load_vector", "vec_mult"]),
        (1, 2, ["vec_mult"]),
    ])
    def test_execute_path_step_range(self, options_fixture, start, end, expected):
        _, ar, _ = _make_runner(options_fixture)
        if start > 0:
            ar.execute_step(_step(ar, "load_vector"), save=True)
        ran = []
        original = ar.execute_step
        ar.execute_step = lambda step, **kw: (ran.append(step.name), original(step, **kw))[1]

        ar.execute_path(step_start_idx=start, step_end_idx=end, verbose=False)

        assert ran == expected

    @pytest.mark.parametrize("start, end", [(3, None), (2, 2), (5, 7)])
    def test_execute_path_empty_step_range_raises(self, options_fixture, start, end):
        _, ar, _ = _make_runner(options_fixture)
        with pytest.raises(ValueError, match="select no steps.*not data indices"):
            ar.execute_path(step_start_idx=start, step_end_idx=end, verbose=False)

    @pytest.mark.parametrize("old_kwarg", [{"execution_mode": "per-row"}, {"execute_per_row": True},
                                           {"start_from_idx": 1}])
    def test_old_keyword_names_are_rejected(self, options_fixture, old_kwarg):
        _, ar, _ = _make_runner(options_fixture)

        with pytest.raises(TypeError):
            ar.execute_path(data_idx=np.arange(10), verbose=False, **old_kwarg)


class TestPathPerRow:
    """Tests for the ``path_per_row`` option."""

    def test_path_per_row_runs_complete_row_before_next(self, tmp_path):
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
        analysis_custom = tmp_path / "custom_analysis_steps.py"
        analysis_custom.write_text(
            "import numpy as np\n"
            "from citkid.pipeline_v2.framework import plStep\n"
            "\n"
            "call_log = []\n"
            "\n"
            "def vec_step(data_idx):\n"
            "    call_log.append(('vec', int(np.atleast_1d(data_idx)[0])))\n"
            "    return np.atleast_1d(data_idx) * 10\n"
            "\n"
            "def row_step(vec_out, data_idx):\n"
            "    call_log.append(('row', int(data_idx)))\n"
            "    return vec_out + data_idx\n"
            "\n"
            "custom_analysis_steps = [\n"
            "    plStep('vec_step', vec_step, ['data_idx'], ['vec_out'], 'vectorized'),\n"
            "    plStep('row_step', row_step, ['vec_out', 'data_idx'], ['final'], 'per-row'),\n"
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
        analysis_yaml = tmp_path / "analysis.yaml"
        analysis_yaml.write_text(
            "ANALYSIS_STEPS:\n"
            "  1:\n"
            "    task: vec_step\n"
            "  2:\n"
            "    task: row_step\n",
            encoding="utf-8",
        )

        ds = DataSet(
            zarr_path=str(tmp_path / "path_per_row.zarr"),
            cal_yaml_path=str(cal_yaml),
            custom_path=str(cal_custom),
        )
        ar = AnalysisRunner(
            ds,
            analysis_yaml_path=str(analysis_yaml),
            custom_path=str(analysis_custom),
        )

        ar.execute_path(data_idx=[0, 1], path_per_row=True, verbose=False, save=True)

        log = ar.analysis_steps[0].func.__globals__['call_log']
        assert log == [('vec', 0), ('row', 0), ('vec', 1), ('row', 1)]
        np.testing.assert_array_equal(ds.final[[0, 1]], [0, 11])

    def test_path_per_row_runs_global_res_once(self, options_fixture):
        ds, ar, module = _make_runner(options_fixture)

        ar.execute_path(data_idx=np.arange(10), path_per_row=True, verbose=False, save=True)

        assert module["call_log"] == [[row] for row in range(10)]
        np.testing.assert_array_equal(ds.final[np.arange(10)], np.arange(10) * 7)


class TestPartialRowGlobalSteps:
    """Tests for partial-row runs that involve global-res steps."""

    def test_execute_path_partial_rows_allow_first_global_res_run(self, options_fixture):
        """A first partial execute_path may run a global-res step if nothing exists yet."""
        ds, ar, _ = _make_runner(options_fixture)

        ar.execute_path(data_idx=0, verbose=False, save=True)

        assert ds.final[0] == 0

    def test_execute_path_partial_rows_raises_for_global_res_rerun(self, options_fixture):
        """Partial execute_path should raise before rerunning an existing global-res step."""
        _, ar, _ = _make_runner(options_fixture)

        ar.execute_path(data_idx=0, verbose=False, save=True)

        with pytest.raises(ValueError, match="load_vector"):
            ar.execute_path(data_idx=1, verbose=False, save=True)

    def test_execute_step_raises_when_partial_rows_need_global_res_prerequisite(self, options_fixture):
        """A later step should not silently rerun a missing global-res prerequisite for one row."""
        _, ar, _ = _make_runner(options_fixture)

        with pytest.raises(ValueError, match="load_vector"):
            ar.execute_step(_step(ar, "vec_mult"), data_idx=0, save=True)

    def test_execute_step_global_res_rerun_requires_overwrite_flag(self, options_fixture):
        """Re-running a global-res step should require explicit overwrite permission."""
        _, ar, _ = _make_runner(options_fixture)
        load_vec_step = _step(ar, "load_vector")
        ar.execute_step(load_vec_step, save=True)

        with pytest.raises(ValueError, match="allow_global_step_overwrite"):
            ar.execute_step(load_vec_step, save=False)

        ar.execute_step(load_vec_step, save=False, allow_global_step_overwrite=True)
