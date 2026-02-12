
import os 
import yaml 
import numpy as np 
from tqdm.auto import tqdm
import importlib.util

from .dependencies import get_most_recent_run
from . import default_steps
from . import framework as pf
from .dataset import _convert_yaml_to_steps

class AnalysisRunner:
    """
    Runner for executing analysis steps on a DataSet with user parameters.
    
    Provides a framework for running analysis pipelines that use user-provided
    parameters alongside computed calibration parameters. Analysis steps are
    executed on the parent DataSet, with results stored in the DataSet's cache.
    
    Attributes:
        DS (DataSet): The DataSet instance this runner operates on.
        analysis_yaml_path (str or None): Path to analysis YAML configuration.
        analysis_yaml_dict (dict or None): Loaded YAML configuration.
    """
    
    def __init__(self, DS, analysis_yaml_path = None, custom_path = None):
        """
        Initialize the analysis runner with a dataset.
        
        Parameters:
        DS (DataSet): The DataSet instance to run analysis on.
        analysis_yaml_path (str or None): Path to analysis YAML file defining
            analysis pipeline and parameters. If None, runner can still be
            used to execute individual steps programmatically.
        custom_path (str or None): Optional path to custom steps .py file, 
            defining custom plStep objects. If None, no custom steps are loaded.
            Must contain a variable 'custom_analysis_steps' which is a list of 
            plStep objects.
        
        Raises:
        ValueError: If analysis_yaml_path is not a .yaml or .yml file.
        FileNotFoundError: If analysis_yaml_path doesn't exist.
        """
        self.DS = DS

        # Load custom steps 
        self.analysis_steps = _load_custom_steps(custom_path)

        # Add default analysis steps if not already present
        for step in default_steps.default_analysis_steps:
            if step.name not in [s.name for s in self.analysis_steps]:
                self.analysis_steps.append(step)
        
        # Load analysis YAML
        if analysis_yaml_path is not None:
            self.analysis_yaml_path = os.path.abspath(analysis_yaml_path)
            
            # Validate file type
            is_yaml = self.analysis_yaml_path.endswith('.yaml') \
                or self.analysis_yaml_path.endswith('.yml')
            if not is_yaml:
                raise ValueError(
                    "analysis_yaml_path must point to a .yaml or .yml file."
                )
            
            # Load YAML file
            with open(self.analysis_yaml_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.analysis_pl = _convert_yaml_to_steps(
                yaml_dict, self.analysis_steps
                )
            if list(self.analysis_pl.keys()) != ['ANALYSIS_STEPS']:
                raise ValueError(
                    "analysis YAML must contain only 'ANALYSIS_STEPS' key"
                    )
            path_dict = self.analysis_pl['ANALYSIS_STEPS']
            task_idxs = path_dict.keys()
            max_task = _validate_task_idxs(task_idxs)
            self.path = [path_dict[i] for i in range(1, max_task + 1)]
            for step_dict in self.path:
                if 'task' not in step_dict:
                    raise ValueError("Each analysis step must include 'task'")
                for k in step_dict.keys():
                    if k not in ['task', 'params', 'save']:
                        raise ValueError(
                            f"Invalid key '{k}' found in analysis YAML"
                        ) 
        else:
            self.analysis_yaml_path = None
            self.analysis_yaml_dict = None
    
    def execute_path(self, data_idx = None, enforced_max_runs = None,
                     start_from_idx = 0, verbose = True):
        """
        Execute each step in the path specified by the anlysis YAML file. 

        Parameters:
        data_idx (int, array-like, or None): Data indices to process for per-row
            steps. If None, for per-row and vectorized functions, runs the path 
            on all rows. Must be None for global steps.
        enforced_max_runs (dict or None): Optional dict mapping parameter names 
            to maximum run indices for dependency resolution. If provided, these 
            constraints will be applied when producing any missing parameters
            required by the steps in the path. 
            Format: {param_name: max_run_idx}.
        start_from_idx (int): Optional index into the path to start from.
            Default is 0 (start from the beginning). 
        verbose (bool): If True, displays a progress bar while executing the 
        path.
        """
        pbar = self.path[start_from_idx:]
        if verbose:
            pbar = tqdm(pbar, leave = False, 
                        bar_format = "{desc}: {n_fmt}/{total_fmt}  |{bar}|")
        for step_dict in pbar:
            pbar.set_description(f"Executing step: {step_dict['task'].name}")
            # Collect info from YAML step dict
            step = step_dict['task'] 
            params = step_dict.get('params', {})
            # save defaults to True if not specified
            save = step_dict.get('save', True) 
            # Execute step 
            if step.func_type in ['global', 'global-res']:
                step_data_idx = None  
            else:
                step_data_idx = data_idx
            self.execute_step(
                step, data_idx = step_data_idx, user_params = params, 
                enforced_max_runs = enforced_max_runs, save = save
            )

    def execute_step(self, step, data_idx = None, user_params = None, 
                     enforced_max_runs = None, save = False):
        """
        Execute a single analysis step with optional user-provided parameters.
        
        This method allows running analysis steps with user parameters that are
        not computed from the pipeline. User parameters are stored in the 
        DataSet before executing the step.
        
        Parameters:
            step (plStep): The analysis step to execute.
            data_idx (int, array-like, or None): Data indices to process. 
                Required for per-row/vectorized steps, must be None for global 
                steps.
            user_params (dict or None): Dictionary mapping parameter names to 
                values. These parameters will be stored in the DataSet before 
                executing the step. Format: {param_name: value} for global 
                params, or {param_name: array} for per-row params (length must 
                match data_idx).
            enforced_max_runs (dict or None): Optional dict mapping parameter 
                names to maximum run indices for dependency resolution.
            save (bool): If True, write the step outputs to the zarr file on 
                disk after execution. Default is False.
        
        Returns:
            None (results stored in self.DS._memory_cache and optionally written
                 to zarr)
        
        Raises:
            ValueError: If user parameter constraints are violated.
            TypeError: If step is not a plStep instance.
        
        Examples:
            >>> runner = AnalysisRunner(DS)
            >>> step = pf.plStep(...)
            >>> # Execute with user parameters and save to disk
            >>> runner.execute_step(step, data_idx=[0,1,2], 
            ...                     user_params={'threshold': 5.0}, save=True)
        """
        if user_params is None:
            user_params = {}
        if enforced_max_runs is None:
            enforced_max_runs = {}

        if step.func_type in ['per-row', 'vectorized'] and data_idx is None:
            data_idx = np.arange(self.DS.nrows)
        elif step.func_type in ['global', 'global-res'] and data_idx is not None:
            raise ValueError(
                f"data_idx must be None for global func_type '{step.func_type}'"
            )
        
        # Validate that all required parameters are available
        missing_params = []
        for param_name in step.param_names:
            if param_name == 'data_idx':
                continue  # Special built-in parameter
            
            # Check if parameter can be produced from calibration pipeline
            path = pf.find_pl_path(self.DS.cal_pl, param_name)
            if path is not None:
                continue  # Can be produced
            
            # Check if parameter is provided as user_param
            if param_name in user_params:
                continue  # Will be added before execution
            
            # Check if parameter already exists in memory or zarr
            param_exists = False
            
            # Check global deps_maps
            if 'global' in self.DS.deps_maps:
                for run_idx in self.DS.deps_maps['global']:
                    if param_name in self.DS.deps_maps['global'][run_idx]:
                        param_exists = True
                        break
            
            # Check per-row deps_maps if needed
            if not param_exists and data_idx is not None:
                data_idx_array = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
                for di in data_idx_array:
                    if di in self.DS.deps_maps:
                        for run_idx in self.DS.deps_maps[di]:
                            if param_name in self.DS.deps_maps[di][run_idx]:
                                param_exists = True
                                break
                    if param_exists:
                        break
            
            if not param_exists:
                missing_params.append(param_name)
        
        if missing_params:
            raise ValueError(
                f"Step '{step.name}' requires parameters {missing_params} which "
                f"cannot be produced from the calibration pipeline, do not exist "
                f"in memory or zarr, and were not provided in user_params. Please "
                f"provide these parameters via the user_params argument."
            )
        
        # Store user parameters in DataSet before executing step
        if user_params:
            self._add_user_params(user_params, step.func_type, data_idx, save=save)
        
        # Ensure all required input parameters exist by producing them if needed
        self._ensure_inputs_exist(step, data_idx, enforced_max_runs)
        
        # Execute the step using DataSet's _execute_step method
        try:
            self.DS._execute_step(step, data_idx = data_idx, 
                                  enforced_max_runs = enforced_max_runs,
                                  save = save)
        except Exception as e:
            raise RuntimeError(
                f"Error executing step '{step.name}': {str(e)}"
            ) from e
    
    def _add_user_params(
            self, user_params, func_type, data_idx = None, save = False
            ):
        """
        Add user-provided parameters to the DataSet cache.
        
        User parameters are stored with empty dependencies (deps={}) to prevent
        backtracking by _check_path_validity. Run indices are determined by
        get_most_recent_run + 1 for each parameter.
        
        Parameters:
            user_params (dict): Mapping of parameter names to values.
            func_type (str): The function type of the step 
                (e.g., 'global', 'global-res', 'per-row', 'vectorized').
            data_idx (int, array-like, or None): Data indices for per-row 
                parameters. Required for per-row steps, must be None for global.
            save (bool): Whether to save user parameters to zarr. Only user
                parameters are saved during analysis runs.
        
        Raises:
            TypeError: If user_params is not a dict or func_type is not a 
                string.
            ValueError: If func_type is invalid, data_idx requirements are 
                violated, or per-row value length doesn't match data_idx.
        
        Notes:
            For global parameters:
            - Stores value directly for all data (is_global=True)
            - Run index determined from global deps_map
            
            For per-row parameters:
            - Stores array of values, one per data_idx (is_global=False)
            - Each data_idx gets its own run_idx based on its history
            - Data indices with same run_idx are grouped and stored together
            - Value must be array-like with length matching data_idx
        """
        # Input validation
        if not isinstance(user_params, dict):
            raise TypeError("user_params must be a dictionary")
        if not isinstance(func_type, str):
            raise TypeError("func_type must be a string")
        
        # Validate and determine is_global from func_type
        if func_type in ['global', 'global-res']:
            is_global = True
        elif func_type in ['per-row', 'vectorized']:
            is_global = False
        else:
            raise ValueError(
                f"Invalid func_type '{func_type}'. Must be one of: "
                "'global', 'global-res', 'per-row', 'vectorized'"
            )
        
        # Validate data_idx requirements
        if is_global:
            if data_idx is not None:
                raise ValueError(
                    f"data_idx must be None for global func_type '{func_type}'"
                )
        else:
            if data_idx is None:
                raise ValueError(
                    f"data_idx required for per-row func_type '{func_type}'"
                )
            # Track if data_idx was originally a scalar
            data_idx_was_scalar = isinstance(data_idx, (int, np.integer))
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        
        # Store each user parameter
        if is_global:
            # Global parameters: store directly
            if 'global' not in self.DS.deps_maps:
                self.DS.deps_maps['global'] = {}
            deps_map = self.DS.deps_maps['global']
            
            for name, val in user_params.items():
                # Always create a new run for user parameters
                most_recent_run = get_most_recent_run(name, deps_map)
                run_idx = most_recent_run + 1 if most_recent_run > 0 else 1
                self.DS._store_param(
                    name, val, run_idx, deps={}, is_global=True, data_idx=None
                )

                # Save user parameter to zarr if requested
                if save:
                    self.DS.write_data(name, run_idx, data_idx=None)
        else:
            # Per-row parameters: store array of values
            for name, val in user_params.items():
                # If data_idx was scalar, wrap value in a list so it's treated
                # as a single value (which could be an array)
                if data_idx_was_scalar:
                    val = [val]
                elif not isinstance(val, (list, np.ndarray)):
                    # Broadcast scalar to all data_idx
                    val = [val] * len(data_idx)  
                
                # Convert value to array and validate length
                val_array = np.atleast_1d(np.asarray(val))
                if len(val_array) != len(data_idx):
                    raise ValueError(
                        f"Length mismatch for user parameter '{name}': "
                        f"value has {len(val_array)} elements but data_idx has "
                        f"{len(data_idx)} elements"
                    )
                
                # Determine run_idx for each data_idx based on its history
                run_indices = []
                for di in data_idx:
                    if di not in self.DS.deps_maps:
                        self.DS.deps_maps[di] = {}
                    deps_map = self.DS.deps_maps[di]
                    run_idx = get_most_recent_run(name, deps_map)
                    run_idx = run_idx + 1 if run_idx > 0 else 1
                    run_indices.append(run_idx)
                
                # Group data_idx by their run_idx
                run_idx_groups = {}
                for i, (di, run_idx) in enumerate(zip(data_idx, run_indices)):
                    if run_idx not in run_idx_groups:
                        run_idx_groups[run_idx] = []
                    run_idx_groups[run_idx].append(i)
                
                # Store for each unique run_idx
                for run_idx, indices in run_idx_groups.items():
                    group_data_idx = data_idx[indices]
                    group_val = val_array[indices]
                    
                    self.DS._store_param(
                        name, group_val, run_idx, deps={}, is_global=False, 
                        data_idx=group_data_idx
                    )
                    
                    # Save user parameter to zarr if requested
                    if save:
                        self.DS.write_data(name, run_idx, data_idx=group_data_idx)
    
    def _ensure_inputs_exist(self, step, data_idx, enforced_max_runs):
        """
        Ensure all required input parameters exist before executing a step.
        
        For each parameter required by the step, checks if it exists in the
        DataSet. If not, attempts to produce it from the calibration pipeline
        using _produce_data.
        
        Parameters:
            step (plStep): The step whose inputs need to be validated.
            data_idx (int, array-like, or None): Data indices being processed.
            enforced_max_runs (dict): Optional max run constraints.
        
        Raises:
            ValueError: If a required parameter cannot be produced.
        """
        # Check each parameter (except 'data_idx' which is special)
        for param_name in step.param_names:
            if param_name == 'data_idx':
                continue
            
            # Check if parameter exists in deps_maps
            param_exists = False
            
            # Check global deps_maps
            if 'global' in self.DS.deps_maps:
                for run_idx in self.DS.deps_maps['global']:
                    if param_name in self.DS.deps_maps['global'][run_idx]:
                        param_exists = True
                        break
            
            # Check per-row deps_maps if needed and not found in global
            if not param_exists and data_idx is not None:
                data_idx_array = np.atleast_1d(np.asarray(data_idx, 
                                                          dtype=np.int32))
                for di in data_idx_array:
                    if di in self.DS.deps_maps:
                        for run_idx in self.DS.deps_maps[di]:
                            if param_name in self.DS.deps_maps[di][run_idx]:
                                param_exists = True
                                break
                    if param_exists:
                        break
            
            # If parameter doesn't exist, try to produce it
            if not param_exists:
                # Determine appropriate data_idx for production
                if step.func_type in ['global', 'global-res']:
                    produce_data_idx = None
                else:
                    produce_data_idx = data_idx
                
                # Attempt to produce the parameter
                self.DS._produce_data(
                    param_name, 
                    data_idx=produce_data_idx,
                    enforced_max_runs=enforced_max_runs
                )

################################################################################ 
############################# Custom steps loader ##############################
################################################################################
def _load_custom_steps(custom_path):
    """
    Load custom steps from 'custom_steps.py' in the dataset directory.

    Returns:
    list: A list of custom plStep objects.
    """
    if custom_path is None:
        return []
    
    if not os.path.exists(custom_path):
        m = f"Custom path '{custom_path}' does not exist."
        raise FileNotFoundError(m)
    
    spec = importlib.util.spec_from_file_location("custom_analysis_steps", 
                                                    custom_path)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    custom_analysis_steps = cs.custom_analysis_steps
    
    return custom_analysis_steps

def _validate_task_idxs(task_idxs):
    if not task_idxs:
        raise ValueError("task_idxs is empty")

    # must all be integers ≥1
    if not all(isinstance(i, int) and i >= 1 for i in task_idxs):
        raise ValueError("task_idxs must contain only integers ≥ 1")

    # must be exactly [1, 2, ..., max]
    max_idx = max(task_idxs)
    expected = set(range(1, max_idx + 1))

    if set(task_idxs) != expected or len(task_idxs) != max_idx:
        raise ValueError(
            "task_idxs must be consecutive integers starting at 1 with "
            "no gaps or duplicates"
            )

    return max_idx