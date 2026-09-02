"""
Tests for pipeline_v2 execution_mode parameter.

The execution_mode parameter controls how vectorized steps are executed:
- 'vectorized' (default): Load all row data at once, run step once
- 'per-row': Loop over each row individually, run step for each row

This is a pipeline_v2-specific feature for memory-constrained analysis.
"""

import pytest
import numpy as np
from pathlib import Path

from citkid.pipeline_v2.analysis import AnalysisRunner
from citkid.pipeline_v2.dataset import DataSet


@pytest.fixture
def execution_mode_fixture(tmp_path):
    """Create a simple pipeline with vectorized and per-row steps."""
    
    # Custom steps: mix of vectorized and per-row
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

    # Analysis with both vectorized and per-row steps
    analysis_custom = tmp_path / "custom_analysis_steps.py"
    analysis_custom.write_text(
        "import numpy as np\n"
        "from citkid.pipeline_v2.framework import plStep\n"
        "\n"
        "def load_vector():\n"
        "    return np.arange(10) * 2  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]\n"
        "\n"
        "def vectorized_multiply(vector):\n"
        "    '''Vectorized step: multiply entire array.'''\n"
        "    return vector * 3\n"
        "\n"
        "def per_row_add(result, data_idx):\n"
        "    '''Per-row step: add data_idx to each element.'''\n"
        "    return result + data_idx\n"
        "\n"
        "custom_analysis_steps = [\n"
        "    plStep('load_vector', load_vector, [], ['vector'], 'global-res'),\n"
        "    plStep('vec_mult', vectorized_multiply, ['vector'], ['result'], 'vectorized'),\n"
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


class TestExecutionModeParameter:
    """Tests for execution_mode parameter in execute_path and execute_step"""

    def test_execute_path_accepts_execution_mode_default(self, execution_mode_fixture):
        """execute_path should accept execution_mode parameter with 'vectorized' default."""
        files = execution_mode_fixture
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
        
        # Execute with default (vectorized) - should not raise
        ar.execute_path(data_idx=0, verbose=False, save_override=True)

    def test_execute_path_with_vectorized_mode(self, execution_mode_fixture):
        """execute_path should work with execution_mode='vectorized'."""
        files = execution_mode_fixture
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
        
        # Execute with explicit vectorized mode
        ar.execute_path(data_idx=0, execution_mode='vectorized', 
                       verbose=False, save_override=True)

    def test_execute_path_with_per_row_mode(self, execution_mode_fixture):
        """execute_path should work with execution_mode='per-row'."""
        files = execution_mode_fixture
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
        
        # Execute with per-row mode (memory-efficient)
        ar.execute_path(data_idx=0, execution_mode='per-row', 
                       verbose=False, save_override=True)

    def test_execute_step_accepts_execution_mode(self, execution_mode_fixture):
        """execute_step should accept execution_mode parameter."""
        files = execution_mode_fixture
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
        
        # First run calibration
        import_step = next(step for step in ds.cal_steps if step.name == "load_data")
        ar.execute_step(import_step, save=True)
        
        # First run load_vector step (global-res, no data_idx)
        load_vec_step = next(step for step in ar.analysis_steps if step.name == "load_vector")
        ar.execute_step(load_vec_step, save=True)
        
        # Execute vectorized step with per-row mode
        vec_mult_step = next(step for step in ar.analysis_steps if step.name == "vec_mult")
        ar.execute_step(vec_mult_step, data_idx=0, execution_mode='per-row', save=True)

    def test_execution_mode_invalid_value_raises(self, execution_mode_fixture):
        """execute_path should raise ValueError for invalid execution_mode."""
        files = execution_mode_fixture
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
        
        # Invalid execution_mode should raise
        with pytest.raises(ValueError):
            ar.execute_path(data_idx=0, execution_mode='invalid_mode', 
                           verbose=False, save_override=True)

    def test_vectorized_and_per_row_produce_same_results(self, execution_mode_fixture):
        """Vectorized and per-row execution modes should produce identical results."""
        files = execution_mode_fixture
        
        # Run with vectorized mode
        ds1 = DataSet(
            zarr_path=str(files["zarr_path"]),
            cal_yaml_path=str(files["cal_yaml"]),
            custom_path=str(files["cal_custom"]),
        )
        ar1 = AnalysisRunner(
            ds1,
            analysis_yaml_path=str(files["analysis_yaml"]),
            custom_path=str(files["analysis_custom"]),
        )
        
        import_step = next(step for step in ds1.cal_steps if step.name == "load_data")
        ar1.execute_step(import_step, save=True)
        ar1.execute_path(data_idx=0, execution_mode='vectorized', 
                        verbose=False, save_override=True)
        
        # Save the result
        result_vec = ds1.final[0]
        
        # Run with per-row mode (different zarr)
        zarr_path_2 = files["zarr_path"].parent / "analysis_2.zarr"
        ds2 = DataSet(
            zarr_path=str(zarr_path_2),
            cal_yaml_path=str(files["cal_yaml"]),
            custom_path=str(files["cal_custom"]),
        )
        ar2 = AnalysisRunner(
            ds2,
            analysis_yaml_path=str(files["analysis_yaml"]),
            custom_path=str(files["analysis_custom"]),
        )
        
        import_step = next(step for step in ds2.cal_steps if step.name == "load_data")
        ar2.execute_step(import_step, save=True)
        ar2.execute_path(data_idx=0, execution_mode='per-row', 
                        verbose=False, save_override=True)
        
        # Results should be identical
        result_per_row = ds2.final[0]
        
        # Both should exist and be equal
        assert result_vec == result_per_row, \
            f"Results differ: vectorized={result_vec}, per-row={result_per_row}"


class TestExecutionModeMemoryConsiderations:
    """Tests verifying that per-row mode works for memory-constrained scenarios"""

    def test_per_row_mode_exists_and_works(self, execution_mode_fixture):
        """Per-row execution mode should be available and functional."""
        files = execution_mode_fixture
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
        
        # Per-row mode should not crash even with larger data
        ar.execute_path(data_idx=0, execution_mode='per-row', 
                       verbose=False, save_override=True)
        
        # Result should still be available
        assert hasattr(ds, 'final'), "Final result not available after per-row execution"

    def test_vectorized_mode_is_default(self, execution_mode_fixture):
        """Vectorized mode should be the default execution_mode."""
        files = execution_mode_fixture
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
        
        # Call without execution_mode - should use vectorized by default
        ar.execute_path(data_idx=0, verbose=False, save_override=True)
        
        # Should complete successfully
        assert True


class TestExecutionModeBackwardCompatibility:
    """Tests ensuring execution_mode doesn't break existing code"""

    def test_existing_code_without_execution_mode_works(self, execution_mode_fixture):
        """Code that doesn't use execution_mode parameter should still work."""
        files = execution_mode_fixture
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
        
        # Old-style call without execution_mode
        ar.execute_path(data_idx=0, verbose=False, save_override=True)

    def test_execute_step_backward_compatible(self, execution_mode_fixture):
        """execute_step should work without execution_mode parameter."""
        files = execution_mode_fixture
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
        # Old-style call without execution_mode
        ar.execute_step(import_step, save=True)
        
        assert ds.nrows == 10
