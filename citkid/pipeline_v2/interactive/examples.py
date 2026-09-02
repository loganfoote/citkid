"""
Example: Using pipeline_v2.interactive with a simple IQ analysis.

This script demonstrates how to use the interactive UI with pipeline_v2's
simplified analysis architecture.

Key differences from v1:
- No concept of "runs" - only one active state per parameter
- Re-running a step automatically deletes downstream outputs
- Panels call _write_nan_outputs() which now deletes instead of writing NaNs
"""

def example_basic_usage():
    """Basic usage example."""
    from citkid.pipeline_v2.analysis import AnalysisRunner
    from citkid.pipeline_v2.dataset import DataSet
    from citkid.pipeline_v2.interactive import run_interactive
    
    # Create a dataset and analysis runner
    DS = DataSet(
        zarr_path="my_analysis.zarr",
        cal_yaml_path="iq",  # or path to cal.yaml
    )
    AR = AnalysisRunner(
        DS,
        analysis_yaml_path="iq",  # or path to iq_analysis.yaml
    )
    
    # Launch the interactive window
    # This will stack panels for each group of steps defined in the YAML
    run_interactive(
        AR,
        data_idx=0,  # Start at first resonator
        title="IQ Analysis - pipeline_v2",
    )


def example_sweep_fitter():
    """Example using the sweep fitter for parameter sweeps."""
    import zarr
    from citkid.pipeline_v2.analysis import AnalysisRunner
    from citkid.pipeline_v2.dataset import DataSet
    from citkid.pipeline_v2.interactive.sweep_fitter import run_sweep_fitter
    
    def make_custom_steps(sweep_idx):
        """Create calibration steps for loading data at a specific sweep point.
        
        This could load data from different files, apply different offsets, etc.
        """
        from citkid.pipeline_v2 import framework as pf
        
        # Example: just use default calibration
        # In practice, you'd customize per sweep_idx
        return []
    
    def y_func(AR, data_idx):
        """Extract a scalar y value to plot on the sweep scatter."""
        try:
            # Example: nonlinearity 'a' parameter from IQ fit
            return float(AR.DS.iq_popt[data_idx][4])
        except Exception:
            return None
    
    # Open the zarr store
    root = zarr.open_group("sweep_analysis.zarr", mode="a")
    
    # Launch the sweep fitter
    run_sweep_fitter(
        make_custom_steps=make_custom_steps,
        cal_yaml_path="iq",
        analysis_yaml_path="iq",
        root=root,
        n_sweep=7,  # Number of sweep points
        x_param_name="power",  # Parameter to plot on x-axis (e.g. pump power)
        x_name="Power (dBm)",
        y_param_name="a",  # Fit parameter to plot on y-axis
        title="Sweep Analysis - pipeline_v2",
    )


def example_custom_panel():
    """Example creating a custom panel for pipeline_v2.interactive."""
    from citkid.pipeline_v2.interactive import StepPanel, register_panel
    from pyqtgraph.Qt import QtWidgets
    
    @register_panel("make_fr_spans", "fit_gain")
    class CustomGainPanel(StepPanel):
        """Custom panel that handles make_fr_spans and fit_gain steps."""
        
        def setup_ui(self):
            """Build the panel UI."""
            layout = QtWidgets.QVBoxLayout(self)
            
            # Add custom widgets here
            self._span_spin = QtWidgets.QDoubleSpinBox()
            self._span_spin.setValue(2.0)
            self._span_spin.setLabel("Span multiplier")
            layout.addWidget(self._span_spin)
            
            self._run_btn = QtWidgets.QPushButton("Run")
            self._run_btn.clicked.connect(self.run_steps)
            layout.addWidget(self._run_btn)
            
            self._status_label = QtWidgets.QLabel("Ready")
            layout.addWidget(self._status_label)
        
        def get_params_for_step(self, step):
            """Return custom parameters for the given step."""
            if step.name == "fit_gain":
                return {"span_mult": self._span_spin.value()}
            return {}
        
        def update_plots(self):
            """Update displays after steps run successfully."""
            self._status_label.setText("Done ✓")
        
        def _nan_outputs(self):
            """Return list of output names to delete when marking bad."""
            # In pipeline_v2, we return the parameter names to delete
            return ["p_amp", "p_phase", "gain_mask"]


def example_differences_from_v1():
    """Key differences when adapting v1 code to v2.interactive."""
    
    print("""
    Key changes when migrating from pipeline.interactive to pipeline_v2.interactive:
    
    1. PARAMETER DELETION vs NaN MARKING
       v1: _write_nan_outputs() writes NaN-filled arrays to zarr
       v2: _write_nan_outputs() deletes parameters from the dataset
       
       To adapt your panel:
       - v1: def _nan_outputs(self): return {'param': np.full(..., np.nan)}
       - v2: def _nan_outputs(self): return ['param']  # just the names
    
    2. NO RUN-BASED TRACKING
       v1: Each analysis step creates multiple run folders (run_000, run_001, ...)
       v2: Single active state - re-running a step deletes downstream outputs
       
       Implication: No need to manage run indices or call get_most_recent_run()
    
    3. AUTOMATIC DOWNSTREAM INVALIDATION
       v1: Manually mark bad data with _write_nan_outputs()
       v2: Re-running a step automatically invalidates downstream (via AnalysisRunner)
       
       Implication: Simpler logic in panels - they don't need to manage invalidation
    
    4. SWEEP FITTER CHANGES
       Both v1 and v2 create one AR per sweep index
       v2: Each AR has a simpler, single-state dataset (no runs within each sweep)
    """)


if __name__ == "__main__":
    print("See function examples above for usage patterns")
    print("Execute with: python -m citkid.pipeline_v2.interactive.examples")
