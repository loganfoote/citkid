import os
import yaml
import importlib.util 
import numpy as np
import zarr
import re 

from .dependencies import get_most_recent_run, get_deps
from . import framework as pf
from . import util 
from . import default_steps

class Analyzer:
    def __init__(self, DS, analysis_yaml_path = None):
        if analysis_yaml_path is not None:
            self.analysis_yaml_path = os.path.abspath(analysis_yaml_path)
        else:
            self.analysis_yaml_path = None
            return 

        # analysis_yaml_path (if provided) must be .yaml or .yml
        if self.analysis_yaml_path is not None:
            is_yaml = self.analysis_yaml_path.endswith('.yaml') \
                or self.analysis_yaml_path.endswith('.yml')
            if not is_yaml:
                raise ValueError(
                    "analysis_yaml_path must point to a .yaml or .yml file."
                )
            
        # Load analysis YAML and user-specified analysis parameters
        with open(self.analysis_yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        self.analysis_pl = _convert_yaml_to_steps(
            yaml_dict, self.analysis_steps
            )
        pf.check_pl_tree_structure(self.analysis_pl, cal = False)

        # Collect data from analysis YAML
        self.analysis_user_params = _collect_user_params(yaml_dict)
        self._add_user_params_to_cache_and_deps(self.analysis_user_params)

    def run_analysis_step(self, name, data_idx=None, save_to_zarr=True):
        """
        Run an analysis step and save the output to zarr.
        
        Parameters:
        name (str): The name of the analysis step.
        data_idx (int or array-like): Data index (or indices) to
            run the step on.
        save_to_zarr (bool): If True, save the outputs to the
            zarr store at Analyzer.dataset.root.

        Returns:
        None
        """
        x = [d for d in self.analysis_steps if d.name == name]
        if not len(x):
            m = f"Step '{name}' not found in available analysis steps."
            raise ValueError(m)
        
        step = x[0]
        step.run(self, data_idx)
        
        if save_to_zarr:
                        
            for return_name in step.return_names:
                value = getattr(self, return_name)             
                    
                self.write_data(return_name, step,
                                 value, data_idx) 

    def _add_user_params_to_cache_and_deps(self, user_params):
        """
        Add user-specified parameters to the global dependencies map,
        and to the global cache as 'global'-type attributes.
        
        Parameters:
        user_params (dictionary): Keys are function names, and values are 
            dictionaries of parameter names and their values.
        
        Returns:
        None
        """
        if 1 not in self.deps_map['global']:
            self.deps_map['global'][1] = {}
        for d in user_params.values():
            for param_name in d.keys():
                self.global_cache[param_name] = True
                self.deps_map['global'][1][param_name] = {}

class DataSet:
    # Reserved attribute names - computed dynamically from class attributes
    _RESERVED_ATTRS = None
    
    def __init__(self, 
        zarr_path, 
        cal_yaml_path, 
        custom_path = None
        ):
        """
        Initialize the dataset with a calibration pipeline defined by a YAML 
        file.  
        
        Parameters:
        zarr_path (str): The path to the zarr file containing the analysis 
            outputs.
        cal_yaml_path (str): The path to the calibration YAML file.
        custom_path (str or None): Directory to the .py file containing custom 
            calibration functions, or None if no custom functions are used.
        """
        ### Input validation and path normalization
        if custom_path is not None:
            self.custom_path = os.path.abspath(custom_path)
        else:
            self.custom_path = None
        self.zarr_path = os.path.abspath(zarr_path)
        self.root = zarr.open_group(self.zarr_path, mode = 'a')
        self.cal_yaml_path = os.path.abspath(cal_yaml_path) 

        # Validate file types 
        # custom_path must be .py
        if self.custom_path is not None and \
            not self.custom_path.endswith('.py'):
            raise ValueError("custom_path must point to a .py file.")
        # zarr_path must be .zarr
        if not self.zarr_path.endswith('.zarr'):
            raise ValueError("zarr_path must point to a .zarr file.")
        # cal_yaml_path must be .yaml or .yml
        is_yaml = self.cal_yaml_path.endswith('.yaml') \
               or self.cal_yaml_path.endswith('.yml')
        if not is_yaml:
            raise ValueError(
                "cal_yaml_path must point to a .yaml or .yml file."
            )

        ### Load steps from custom_steps.py
        self.cal_steps, self.analysis_steps = _load_custom_steps(
            self.custom_path
            )
        # Add default calibration steps if not already present
        for step in default_steps.default_cal_steps:
            if step.name not in [s.name for s in self.cal_steps]:
                self.cal_steps.append(step)
        # Add default analysis steps if not already present
        for step in default_steps.default_analysis_steps:
            if step.name not in [s.name for s in self.analysis_steps]:
                self.analysis_steps.append(step)
                
        # Load calibration YAML file and convert it to calibration pipeline
        with open(self.cal_yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f) 
        self.cal_pl = _convert_yaml_to_steps(yaml_dict, self.cal_steps)

        # confirm that the cal_pl structure is valid
        pf.check_pl_tree_structure(self.cal_pl, cal = True) 

        # Load deps_map from zarr - also validates zarr structure
        self.deps_maps = _load_deps_from_zarr(self.root)

        # start is_global cache, which maps param names to boolean indicating 
        # whether or not the parameter is global 
        self._is_global_cache = {}

        # start memory_cache, where data is stored referenced to run_idx 
        self._memory_cache = {}
        
        # start lazy_collections, which tracks LazyAttrCollections for per-row 
        # params
        self._lazy_collections = {}

    def _get_reserved_attrs(self):
        """Get all non-private DataSet attributes as reserved names.
        
        This dynamically computes the set of attribute names that cannot be used
        as pipeline parameter names. The result is cached at the class level.
        
        Returns:
            set: Set of reserved attribute names.
        """
        if DataSet._RESERVED_ATTRS is None:
            # Get all non-private attributes from DataSet class
            DataSet._RESERVED_ATTRS = {
                name for name in dir(type(self)) 
                if not name.startswith('_')
            }
        return DataSet._RESERVED_ATTRS

    def _execute_step(self, step, data_idx=None, enforced_max_runs={}):
        """
        Execute a single pipeline step, storing results and dependencies in 
        memory.
        
        All input parameters must already exist in memory or on disk - this 
        method will NOT execute additional steps to generate missing inputs.
        
        Parameters:
        step (plStep): The pipeline step to execute.
        data_idx (int, array-like, or None): Data indices to process. Required
            for 'per-row' and 'vectorized' steps, must be None for 'global' and 
            'global-res' steps.
        enforced_max_runs (dict): Optional dict mapping parameter names to 
            maximum run indices for dependency resolution.
        
        Returns:
        None (results are stored in self._memory_cache)
        
        Raises:
        TypeError: If step is not a plStep instance.
        ValueError: If required inputs don't exist or data_idx is invalid.
        """
        # Input validation
        if not isinstance(step, pf.plStep):
            raise TypeError("step must be a plStep instance")
        if not isinstance(enforced_max_runs, dict):
            raise TypeError("enforced_max_runs must be a dictionary")
        
        # Handle data_idx based on function type
        if step.func_type in ['global', 'global-res']:
            if data_idx is not None:
                raise ValueError(
                    f"data_idx must be None for {step.func_type} functions"
                )
        else:
            if data_idx is None:
                raise ValueError(
                    f"data_idx required for {step.func_type} functions"
                )
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))

        # global and global-res have only global parameters
        if step.func_type in ['global', 'global-res']:
            # Determine run_idx deps for outputs
            # all global inputs -> single deps_map and run_idx per return name
            deps_map = self.deps_maps['global']
            run_idxs = [get_most_recent_run(name, deps_map) + 1 
                        for name in step.return_names]
            # run that doesn't exist yet should be 1 
            run_idxs = [1 if r == 0 else r for r in run_idxs]
            deps = get_deps(
                step.param_names, 
                deps_map, 
                enforced_max_runs=enforced_max_runs
            )

            # Collect parameters 
            params = []
            param_is_global = []
            for p in step.param_names:
                # Special handling for 'data_idx' parameter
                if p == 'data_idx':
                    params.append(None)
                    param_is_global.append(True)  # Treat as global-like
                    continue
                
                val = self._get_existing(p, deps[p], data_idx=None)
                param_is_global.append(self._is_global_cache[p])
                params.append(val)

            # Run step
            out = step._run(params, param_is_global)

            # Store outputs in memory cache, each with its own run_idx
            for (name, val), run_idx in zip(out.items(), run_idxs):
                # Check for reserved attribute name collision
                reserved = self._get_reserved_attrs()
                if name in reserved:
                    raise ValueError(
                        f"Cannot create parameter '{name}' - this is a reserved "
                        f"DataSet attribute name. Reserved names: {sorted(reserved)}"
                    )
                
                # Ensure run_idx exists in memory cache
                if run_idx not in self._memory_cache:
                    self._memory_cache[run_idx] = {}
                    
                # Check for overwrite
                if name in self._memory_cache[run_idx]:
                    raise ValueError(
                        f"Attempting to overwrite '{name}' at run {run_idx}"
                    )
                    
                # For global-res, wrap outputs in LazyAttr
                if step.func_type == 'global-res':
                    self._memory_cache[run_idx][name] = pf.LazyAttr(
                        self, name, run_idx
                    )
                    # Store data in LazyAttr cache
                    for i, v in enumerate(val):
                        self._memory_cache[run_idx][name]._cache[i] = v
                    
                    # Register with LazyAttrCollection
                    if name not in self._lazy_collections:
                        self._lazy_collections[name] = pf.LazyAttrCollection(
                            self, name
                            )
                    self._lazy_collections[name].add_run(
                        run_idx, 
                        self._memory_cache[run_idx][name]
                    )
                    
                    # Store dependencies for all rows
                    for i in range(self.nrows):
                        if run_idx not in self.deps_maps[i]:
                            self.deps_maps[i][run_idx] = {}
                        self.deps_maps[i][run_idx][name] = deps
                    self._is_global_cache[name] = False
                else:
                    # For global, assign parameter directly 
                    self._memory_cache[run_idx][name] = val
                    if run_idx not in self.deps_maps['global']:
                        self.deps_maps['global'][run_idx] = {}
                    self.deps_maps['global'][run_idx][name] = deps
                    self._is_global_cache[name] = True

        else:  # vectorized or per-row 
            # For vectorized/per-row, need to find run-idx for each data_idx 
            run_deps = []  # (run_idxs_tuple, deps) for each data_idx
            for di in data_idx:
                deps_map = self.deps_maps[di]
                run_idxs = [get_most_recent_run(name, deps_map) + 1 
                                  for name in step.return_names]
                # run that doesn't exist yet should be 1
                run_idxs = tuple([1 if r == 0 else r for r in run_idxs])
                deps = get_deps(
                    step.param_names, 
                    deps_map, 
                    enforced_max_runs=enforced_max_runs
                )
                run_deps.append((run_idxs, deps))
            
            # Group by unique run_idxs/deps combinations
            unique_run_deps, indices_groups = util.group_unique_tuples(run_deps)
            
            for (run_idxs, deps), local_idx in zip(
                unique_run_deps, indices_groups
                ):
                # Collect parameters 
                params = []
                param_is_global = []
                for p in step.param_names:
                    # Special handling for 'data_idx' parameter
                    if p == 'data_idx':
                        params.append(data_idx[local_idx])
                        param_is_global.append(False)
                        continue
                    
                    param_is_global.append(self._is_global_cache[p])
                    if param_is_global[-1]:
                        val = self._get_existing(p, deps[p], data_idx=None)
                    else:
                        val = self._get_existing(
                            p, deps[p], data_idx=data_idx[local_idx]
                        )
                    params.append(val)
                
                # Run step 
                out = step._run(params, param_is_global)

                # Store outputs in memory cache, each with its own run_idx
                for (name, val), run_idx in zip(out.items(), run_idxs):
                    # Check for reserved attribute name collision
                    reserved = self._get_reserved_attrs()
                    if name in reserved:
                        raise ValueError(
                            f"Cannot create parameter '{name}' - this is a reserved "
                            f"DataSet attribute name. Reserved names: {sorted(reserved)}"
                        )
                    
                    # Ensure run_idx exists in memory cache
                    if run_idx not in self._memory_cache:
                        self._memory_cache[run_idx] = {}
                    
                    # Create LazyAttr if needed
                    if name not in self._memory_cache[run_idx]:
                        self._memory_cache[run_idx][name] = pf.LazyAttr(
                            self, name, run_idx
                        )
                        
                        # Register with LazyAttrCollection
                        if name not in self._lazy_collections:
                            self._lazy_collections[name]= pf.LazyAttrCollection(
                                self, name
                                )
                        self._lazy_collections[name].add_run(
                            run_idx, 
                            self._memory_cache[run_idx][name]
                        )
                    
                    # Store data in LazyAttr cache
                    for idx, v in zip(data_idx[local_idx], val):
                        self._memory_cache[run_idx][name]._cache[int(idx)] = v
                    
                    # Store dependencies for each data_idx
                    for di in data_idx[local_idx]:
                        if run_idx not in self.deps_maps[di]:
                            self.deps_maps[di][run_idx] = {}
                        self.deps_maps[di][run_idx][name] = deps
                    
                    self._is_global_cache[name] = False

    def __getattr__(self, name):
        """
        Dynamic attribute access for pipeline parameters.
        
        Provides transparent access to parameters defined in the calibration
        pipeline. Global parameters are computed and returned directly.
        Per-row parameters return a LazyAttrCollection that supports indexing.
        
        Priority: memory → zarr → produce (if in pipeline) → AttributeError
        
        Parameters:
        name (str): The name of the parameter to access.
        
        Returns:
        For global parameters: The computed value (any type).
        For per-row parameters: LazyAttrCollection instance.
        
        Raises:
        AttributeError: If the parameter doesn't exist anywhere.
        
        Examples:
        >>> DS.some_global_param  # Returns the value directly
        42
        >>> DS.some_per_row_param  # Returns LazyAttrCollection
        LazyAttrCollection(some_per_row_param, runs=[1, 3])
        >>> DS.some_per_row_param[0]  # Access specific row
        array([...]) 
        >>> DS.some_per_row_param[[0, 1, 2]]  # Access multiple rows
        array([[...], [...], [...]])
        """
        # First check if we already know about this parameter
        if name in self._is_global_cache:
            is_global = self._is_global_cache[name]
        else:
            # Check memory cache to see if it exists there
            runs_in_memory = [
                run for run in self._memory_cache 
                if name in self._memory_cache[run]
            ]
            
            if runs_in_memory:
                # Found in memory - infer type from structure
                # Global params are stored as direct values
                # Per-row params are stored as LazyAttr
                run_idx = max(runs_in_memory)
                val = self._memory_cache[run_idx][name]
                is_global = not isinstance(val, pf.LazyAttr)
                self._is_global_cache[name] = is_global
            else:
                # Check zarr to see if it exists there
                found_in_zarr = False
                is_global = None
                
                # Check global deps_maps
                if 'global' in self.deps_maps:
                    for run_idx in self.deps_maps['global']:
                        if name in self.deps_maps['global'][run_idx]:
                            found_in_zarr = True
                            is_global = True
                            break
                
                # Check per-row deps_maps if not found in global
                if not found_in_zarr:
                    for data_idx in self.deps_maps:
                        if data_idx == 'global':
                            continue
                        for run_idx in self.deps_maps[data_idx]:
                            if name in self.deps_maps[data_idx][run_idx]:
                                found_in_zarr = True
                                is_global = False
                                break
                        if found_in_zarr:
                            break
                
                if found_in_zarr:
                    self._is_global_cache[name] = is_global
                else:
                    # Not in memory or zarr - check if it can be produced
                    path = pf.find_pl_path(self.cal_pl, name)
                    if path is None:
                        raise AttributeError(
                            f"'{type(self).__name__}' object has no attribute '{name}'"
                        )
                    
                    # Determine type from pipeline
                    final_step = path[-1]
                    is_global = final_step.func_type in ['global', 'global-res']
                    self._is_global_cache[name] = is_global
        
        # Now handle based on whether it's global or per-row
        if is_global:
            # For global parameters: return the value directly
            # Check memory first
            runs_in_memory = [
                run for run in self._memory_cache 
                if name in self._memory_cache[run]
            ]
            if runs_in_memory:
                run_idx = max(runs_in_memory)
                return self._memory_cache[run_idx][name]
            
            # Check zarr
            if 'global' in self.deps_maps:
                runs_in_zarr = [
                    run for run in self.deps_maps['global']
                    if name in self.deps_maps['global'][run]
                ]
                if runs_in_zarr:
                    run_idx = max(runs_in_zarr)
                    deps = self.deps_maps['global'][run_idx][name]
                    return self._get_existing(name, deps, data_idx=None)
            
            # Not found - produce it
            self._produce_data(name, data_idx=None)
            
            # Retrieve from memory (should exist now)
            run_idx = get_most_recent_run(name, self.deps_maps['global'])
            return self._memory_cache[run_idx][name]
        
        else:
            # For per-row parameters: return LazyAttrCollection
            # The collection handles lazy loading when indexed
            if name not in self._lazy_collections:
                self._lazy_collections[name] = pf.LazyAttrCollection(self, name)
            return self._lazy_collections[name]

    def _is_in_memory(self, name, run_idx, data_idx=None):
        """
        Check if data exists in memory cache.
        
        Parameters:
        name (str): The parameter name to check.
        run_idx (int): The run index to check.
        data_idx (int, array-like, or None): The data index/indices to check 
            for per-row data. None for global data.
        
        Returns:
        bool: True if data exists in memory cache, False otherwise.
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        run_idx = int(run_idx)
        if data_idx is not None:
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int32))

        # Check if run exists in cache
        if run_idx not in self._memory_cache:
            return False
        
        # Check if name exists in this run
        if name not in self._memory_cache[run_idx]:
            return False
        
        # Check if name is known to be global
        if name not in self._is_global_cache:
            # If not yet tracked, cannot determine - assume not in memory
            return False
        
        # For global data, just check existence
        if self._is_global_cache[name]:
            return True
        
        # For per-row data, check if requested rows are cached in LazyAttr
        if data_idx is None:
            # Requesting per-row data without specifying rows is ambiguous
            raise ValueError(
                f"data_idx required for per-row parameter '{name}'"
            )
        
        lazy_attr = self._memory_cache[run_idx][name]
        if not isinstance(lazy_attr, pf.LazyAttr):
            raise TypeError(
                f"Expected LazyAttr for per-row parameter '{name}', "
                f"got {type(lazy_attr)}"
            )
        
        return np.all(np.isin(data_idx, list(lazy_attr._cache.keys())))
    
    def _is_in_zarr(self, name, run_idx, data_idx=None):
        """
        Check if data exists in zarr file on disk.
        
        Parameters:
        name (str): The parameter name to check.
        run_idx (int): The run index to check.
        data_idx (int, array-like, or None): The data index/indices to check 
            for per-row data. None for global data.
        
        Returns:
        bool: True if data exists in zarr file, False otherwise.
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        run_idx = int(run_idx)
        if data_idx is not None:
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int32))
        
        # Check if run group exists
        run_str = f'run{run_idx:d}'
        if run_str not in self.root:
            return False
        
        run_grp = self.root[run_str]
        
        # Check if parameter exists in this run
        if name not in run_grp:
            return False
        
        # Check if it's global data
        if 'global' not in run_grp[name].attrs:
            # Missing metadata, assume False
            return False
        
        is_global = run_grp[name].attrs['global']
        
        if is_global:
            return True
        
        # For per-row data, check if requested rows exist
        if data_idx is None:
            raise ValueError(
                f"data_idx required for per-row parameter '{name}'"
            )
        
        if 'row_exists' not in run_grp[name]:
            # Missing row tracking, cannot verify
            return False
        
        return np.all(np.isin(data_idx, run_grp[name]['row_exists'][...]))
    
    def _get_existing(self, name, run_idx, data_idx = None):
        """
        Get data that already exists in memory or on disk.
        Does NOT execute pipeline steps - raises error if data requires 
        computation.
        
        This is the "safe" access method that ensures no side effects from
        pipeline execution during dependency resolution.
        
        Parameters:
        name (str): The parameter name to retrieve.
        run_idx (int): The run index to retrieve.
        data_idx (int, array-like, or None): The data index/indices to retrieve
            for per-row data. None for global data.
        
        Returns:
        np.ndarray or scalar: The requested data from memory or disk.
        
        Raises:
        ValueError: If data does not exist and cannot be retrieved without 
            computation.
        TypeError: If input types are invalid.
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        run_idx = int(run_idx)
        if data_idx is not None:
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int32))
        
        # Try memory first
        if self._is_in_memory(name, run_idx, data_idx):
            if self._is_global_cache[name]:
                return self._memory_cache[run_idx][name]
            else:
                # Return indexed data from LazyAttr
                lazy_attr = self._memory_cache[run_idx][name]
                return lazy_attr[data_idx]
        
        # Try disk second
        if self._is_in_zarr(name, run_idx, data_idx):
            run_str = f'run{run_idx:d}'
            run_grp = self.root[run_str]
            
            if run_grp[name].attrs['global']:
                # Load global data from zarr
                data = run_grp[name]['data'][...]
                
                # Cache it in memory for future access
                if run_idx not in self._memory_cache:
                    self._memory_cache[run_idx] = {}
                self._memory_cache[run_idx][name] = data
                
                return data
            else:
                # Load per-row data from zarr
                data = run_grp[name]['data'][data_idx]
                
                # Cache it in LazyAttr for future access
                if run_idx not in self._memory_cache:
                    self._memory_cache[run_idx] = {}
                if name not in self._memory_cache[run_idx]:
                    self._memory_cache[run_idx][name] = pf.LazyAttr(
                        self, name, run_idx
                    )
                    
                    # Register with LazyAttrCollection when first creating
                    if name not in self._lazy_collections:
                        self._lazy_collections[name] = pf.LazyAttrCollection(self, name)
                    self._lazy_collections[name].add_run(
                        run_idx,
                        self._memory_cache[run_idx][name]
                    )
                    
                    # Update _is_global_cache
                    self._is_global_cache[name] = False
                
                # Update LazyAttr cache with loaded data
                lazy_attr = self._memory_cache[run_idx][name]
                for idx, val in zip(np.atleast_1d(data_idx), data):
                    lazy_attr._cache[int(idx)] = val
                
                return data
        
        # Data doesn't exist - raise error
        raise ValueError(
            f"Parameter '{name}' at run_idx={run_idx} "
            f"{'with data_idx=' + str(data_idx) if data_idx is not None else ''} "
            f"does not exist in memory or zarr and cannot be computed here. "
            f"Use _execute_path to generate missing data."
        )
    
    def _produce_data(self, name, data_idx = None, enforced_max_runs = {}):
        """
        Produce data for a parameter by executing its pipeline path.
        
        Finds the pipeline path needed to produce the named parameter and 
        executes each step in sequence. Each step's output is stored in memory
        with appropriate run indices and dependency tracking.
        
        Parameters:
        name (str): Name of the parameter to produce.
        data_idx (int, array-like, or None): Data indices to produce. Required
            for per-row/vectorized parameters, must be None for global parameters.
        enforced_max_runs (dict): Optional dict mapping parameter names to 
            maximum run indices. Used to ensure specific input versions are used
            when producing the output. Format: {param_name: max_run_idx}.
        
        Returns:
        None (results are stored in self._memory_cache and registered with
             LazyAttrCollections for per-row parameters)
        
        Raises:
        ValueError: If no pipeline path exists for the parameter.
        
        Notes:
        - Only executes steps needed for the specific parameter
        - Filters enforced_max_runs per step to only relevant parameters
        - Works with _execute_step which handles actual computation and storage
        """
        path = pf.find_pl_path(self.cal_pl, name) 
        for step in path:
            enforced_max_runs_i = {
                k: v for k, v in enforced_max_runs.items() 
                if k in step.param_names
            }
            self._execute_step(
                step, 
                data_idx = data_idx, 
                enforced_max_runs = enforced_max_runs_i)
                
    def _fetch_rows(
            self, name, run_idx, data_idx = None, enforced_max_runs = {}
            ):
        """
        Given a parameter name, run index, and data index (if not global), 
        return the data location in order of loading priority: 
        memory -> disk -> potential -> None, 
        where memory means the data exists at an attribute of DataSet, 
        disk means the data exists in the zarr file, and potential means the
        data can be generated from the calibration pipeline. 

        Parameters:
        name (str): The name of the parameter to look for.
        run_idx (int): The run index to look for in the zarr file.
        data_idx (int or None): The data index to look for if the parameter is 
            not global, or None if the parameter is global.

        Returns:
        data (misc): 
        """
        # if not in memory or zarr -> find and execute path
        if not ( 
            self._is_in_memory(name, run_idx, data_idx) or \
            self._is_in_zarr(name, run_idx, data_idx)
        ):
            if self._is_global_cache[name]:
                run_idx_potential = get_most_recent_run(
                    name, self.deps_maps['global']) + 1
                if run_idx_potential == 0:
                    run_idx_potential = 1 
            else:
                runs = []
                for di in data_idx:
                    run = get_most_recent_run(name, self.deps_maps[di]) + 1 
                    if run == 0:
                        run = 1
                    runs.append(run)
                run_idx_potential = runs[0] 
                if not all(r == run_idx_potential for r in runs):
                    raise ValueError(
                        f"Data indices {data_idx} for parameter '{name}' "
                        f"have different potential run indices: {runs}"
                    )
            
            if run_idx_potential != run_idx:
                msg = {'with data_idx=' + '' if data_idx is None 
                       else str(data_idx)}
                raise ValueError(
                    f"Cannot produce '{name}' at run_idx={run_idx} "
                    f"{msg} "
                    f"because it would require producing run_idx="
                    f"{run_idx_potential}."
                )
            self._produce_data(
                name, 
                data_idx = data_idx, 
                enforced_max_runs = enforced_max_runs
                )
                
        # Return from memory or disk
        return self._get_existing(name, run_idx, data_idx)
    
    # Methods below are older code that needs to be refactored. 
    def write_data(self, return_name, step, value, data_idx, dtype = None):
        """
        Write data attribute 'name' for dataset.

        Parameters:
        return_name (str): The name of the data attribute to write.
        func_type (str): The type of function that produced the data.
            Can be "per-row", "vectorized", or "global-res".
        param_names (array-like): Parameter names used when calling the 
            function.
        dependencies (dictionary): Dictionary giving the dependencies for the 
            input parameters, where keys are parameter names and values are run 
            indices.
        value (np.ndarray or scalar): The data to write.
        data_idx (int or list): The data index or indices to write.
        run_idx (int): run index that produced the output.
        dtype (np.dtype, optional): The data type to use when writing. Defaults 
            to None (use value's dtype).

        Returns:
        None
        """
        if step.func_type in ['vectorized', 'per-row'] and data_idx is None:
            data_idx = list(range(self.nrows))

        # set dtype if not provided
        if dtype is None:
            dtype = value[data_idx].dtype

        data_idx = np.atleast_1d(data_idx)
            
        root = self.root
        
        if step.func_type in ['global', 'global-res']:
            
            deps_map = self.deps_maps['global']
            run_idx = get_most_recent_run(return_name, deps_map)
            deps = get_deps(step.return_names, deps_map)
            names_to_remove = [name for name in step.return_names
                               if name != return_name]
            for name in names_to_remove:
                deps.pop(name)
            
            if str(run_idx) not in root:
                root.create_group(str(run_idx))
            run_grp = root[str(run_idx)]
            return_grp = run_grp.create_group(return_name)
            
            # If the func_type is global, we can't index it by row with data_idx
            # So we always just write the full thing to zarr.
            shape = value.shape
            chunks = value.shape
            
            values_arr = return_grp.create_array(
                name = 'data',
                shape = shape,
                dtype = dtype,
                chunks = chunks,
            )
            
            values_arr[...] = value
            # Store parameter dependencies and 'global' status.
            return_grp.attrs['deps'] = deps
            return_grp.attrs['global'] = self._is_global_cache[return_name]
            
        elif step.func_type in ['per-row', 'vectorized']:
            for local_idx, di in enumerate(data_idx):
                # di = di.item()
                di_str = f'idx{di}'
                deps_map = self.deps_maps[di]
                run_idx = get_most_recent_run(return_name, deps_map)
                deps = get_deps(step.return_names, deps_map)
                names_to_remove = [name for name in step.return_names
                                if name != return_name]
                for name in names_to_remove:
                    deps.pop(name)
                
                if str(run_idx) not in root:
                    root.create_group(str(run_idx))
                run_grp = root[str(run_idx)]
                if return_name not in run_grp:
                    run_grp.create_group(return_name)
                return_grp = run_grp[return_name]
                
                if 'data' not in return_grp:                    
                    # Initialize the full empty array with all rows.
                    # The array is chunked per-row.
                    shape = (self.nrows, *value.shape[1:])
                    chunks = (1, *value.shape[1:])
                    
                    values_arr = return_grp.create_array(
                        name = 'data',
                        shape = shape,
                        dtype = dtype,
                        chunks = chunks
                    )
                    
                    # Initialize an array to tell us which data indices
                    # have been saved to values_arr.
                    exists_arr = return_grp.create_array(
                        name = 'row_exists',
                        data = np.full(self.nrows, False)
                    )
                else:
                    values_arr = return_grp['data']
                    exists_arr = return_grp['row_exists']
                    
                values_arr[di] = value[di]
                exists_arr[di] = True
                
                # Store parameter dependencies and 'global(-res)' status.
                if 'deps' not in return_grp.attrs:
                    return_grp.attrs['deps'] = {}
                zarr_deps = return_grp.attrs['deps']
                zarr_deps[di_str] = deps
                return_grp.attrs['deps'] = zarr_deps
                return_grp.attrs['global'] = self._is_global_cache[return_name]
 
    def show_file(self, ftype):
        """
        Opens the file explorer to the location of a file associated with this 
        dataset.

        Parameters:
        ftype (str): The type of file to show. Can be 'cal', 'analysis', 
            'custom', or 'zarr'.

        Raises:
        ValueError: If ftype is not one of the allowed values, or if the 
            specified file does not exist for this dataset.
        """
        if ftype == 'cal': 
            util.open_in_file_explorer(
                os.path.dirname(self.cal_yaml_path)
                ) 
        elif ftype == 'analysis':
            if self.analysis_yaml_path is None:
                m = "No analysis YAML file was provided for this dataset."
                raise ValueError(m)
            else:
                util.open_in_file_explorer(
                    os.path.dirname(self.analysis_yaml_path)
                    )
        elif ftype == 'custom':
            if self.custom_path is None:
                m = "No custom steps file was provided for this dataset."
                raise ValueError(m)
            else:
                util.open_in_file_explorer(
                    os.path.dirname(self.custom_path)
                    )
        elif ftype == 'zarr':
            util.open_in_file_explorer(
                os.path.dirname(self.zarr_path)
                )

################################################################################ 
# Custom steps 
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

    spec = importlib.util.spec_from_file_location("custom_cal_steps", 
                                                    custom_path)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    custom_cal_steps = cs.custom_cal_steps
    
    spec = importlib.util.spec_from_file_location("custom_analysis_steps", 
                                                    custom_path)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    custom_analysis_steps = cs.custom_analysis_steps
    
    return custom_cal_steps, custom_analysis_steps

def _load_deps_from_zarr(root):
    """
    Given the root on an analysis output file, load the dependencies map for 
    each data_idx while validating the file structure.

    Parameters:
    root (zarr group): The root of the zarr file to load from.

    Returns:
    deps_maps (dictionary): A dictionary mapping data indices to their 
        dependencies maps. The key-value map is: 
        data_idx (int or 'global') -> run_idx (int) -> parameter name (str) 
        -> dependencies (dict), 
        where the dependencies dict represents parameter: run_idx pairs for 
        each other parameter that the parameter depends on. Global parameters
        use the key 'global' instead of a data_idx.
    """
    ### Input validation 
    if not isinstance(root, zarr.core.group.Group):
        raise ValueError("Input root must be a zarr group.")
    if any(root.arrays()):
        raise ValueError("root cannot have arrays.")
    
    ### Parse root to create deps_maps and validate file structure.
    # deps_maps: data_idx (or 'global') -> run_idx -> param name -> dependencies 
    deps_maps = {} 
    # Iterate through runs
    for run_str, grp in root.groups():
        # Input validation
        if any(grp.arrays()):
            raise ValueError(f"{run_str} must not contain arrays.")
        if re.fullmatch(r"run\d+", run_str) is None:
            raise ValueError(f"root can only contain run folders: {run_str} group found") 
        run_idx = int(run_str.replace('run', ''))
    
        # Iterate through params in the run
        for name, subgrp in grp.groups():
            # Input validation 
            if 'deps' not in subgrp.attrs.keys():
                raise ValueError(f"Missing 'deps' attr for {run_str} parameter {name}") 
            if 'global' not in subgrp.attrs.keys():
                raise ValueError(f"Missing 'global' attribute for {run_str} parameter {name}")
            if any(subgrp.groups()):
                raise ValueError(f"{run_str} parameter {name} contains a zarr group") 
            array_names = [a[0] for a in subgrp.arrays()]
            if "data" not in array_names:
                raise ValueError(
                    f"'data' array not found in {run_str} parameter {name}"
                )
            array_names = [d for d in array_names if d != 'data']
            if subgrp.attrs['global']:
                if len(array_names) != 0:
                    raise ValueError(
                        f"Extra array(s) found in {run_str} global parameter {name}: {array_names}"
                    )
            else:
                if "row_exists" not in array_names:
                    raise ValueError(f"'row_exists' array not found in {run_str} parameter {name}")
                # Validate row_exists dtype
                row_exists = subgrp['row_exists']
                if row_exists.dtype != np.bool_:
                    raise ValueError(
                        f"'row_exists' array in {run_str} parameter {name} must have dtype bool, "
                        f"got {row_exists.dtype}"
                    )
                array_names = [d for d in array_names if d != 'row_exists']
                if len(array_names) != 0:
                    raise ValueError(
                        f"Extra array(s) found in {run_str} parameter {name}: {array_names}"
                    )
            
            # Validate and process deps attribute based on global status
            deps_attr = subgrp.attrs['deps']
            if not isinstance(deps_attr, dict):
                raise ValueError(
                    f"'deps' attribute in {run_str} parameter {name} must be a dictionary"
                )
            
            if subgrp.attrs['global']:
                # For global parameters, deps should be {str: int}
                for key, value in deps_attr.items():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"Global parameter {name} in {run_str}: deps keys must be strings, got {type(key)}"
                        )
                    if not isinstance(value, (int, np.integer)):
                        raise ValueError(
                            f"Global parameter {name} in {run_str}: deps values must be integers, got {type(value)}"
                        )
                
                # Add to deps_maps under 'global' key
                if 'global' not in deps_maps:
                    deps_maps['global'] = {}
                if run_idx not in deps_maps['global']:
                    deps_maps['global'][run_idx] = {}
                if name in deps_maps['global'][run_idx]:
                    raise ValueError(
                        f"Duplicate parameter {name} found in {run_str}"
                    )
                deps_maps['global'][run_idx][name] = deps_attr
                
            else:
                # For non-global parameters, deps should be {data_idx_str: {str: int}}
                for data_idx_str, deps in deps_attr.items():
                    # Validate data_idx_str can be converted to int
                    try:
                        data_idx = int(data_idx_str.replace('idx', '')) if isinstance(data_idx_str, str) and data_idx_str.startswith('idx') else int(data_idx_str)
                    except (ValueError, AttributeError):
                        raise ValueError(
                            f"Non-global parameter {name} in {run_str}: deps keys must be convertible to int "
                            f"(format 'idx<int>' or int), got {data_idx_str}"
                        )
                    
                    # Validate deps structure
                    if not isinstance(deps, dict):
                        raise ValueError(
                            f"Non-global parameter {name} in {run_str} data_idx {data_idx}: "
                            f"deps must be a dictionary, got {type(deps)}"
                        )
                    for key, value in deps.items():
                        if not isinstance(key, str):
                            raise ValueError(
                                f"Non-global parameter {name} in {run_str} data_idx {data_idx}: "
                                f"deps keys must be strings, got {type(key)}"
                            )
                        if not isinstance(value, (int, np.integer)):
                            raise ValueError(
                                f"Non-global parameter {name} in {run_str} data_idx {data_idx}: "
                                f"deps values must be integers, got {type(value)}"
                            )
                    
                    # Add to deps_maps
                    if data_idx not in deps_maps:
                        deps_maps[data_idx] = {}
                    if run_idx not in deps_maps[data_idx]:
                        deps_maps[data_idx][run_idx] = {}
                    if name in deps_maps[data_idx][run_idx]:
                        raise ValueError(
                            f"Duplicate parameter {name} found in {run_str} for data_idx {data_idx}"
                        )
                    deps_maps[data_idx][run_idx][name] = deps
    
    return deps_maps
################################################################################
# Untested functions 
################################################################################
def _convert_yaml_to_steps(pl_dict, cal_steps, key = None):
    """
    Converts YAML-defined path dictionary leaves to plStep objects.

    Parameters:
    pl_dict (dict or str): The YAML-defined path dictionary or leaf.
    cal_steps (list of plStep): The list of available plStep objects to 
        match against.
    key (str): The key associated with the current pl_dict, used to identify
        task names.

    Returns:
    dict or plStep: The converted paths with plStep objects.
    """
    if isinstance(pl_dict, dict):
        for key, val in pl_dict.items():
            pl_dict[key] = _convert_yaml_to_steps(val, cal_steps, key)
    if isinstance(pl_dict, str) and key == 'task':
        x = [d for d in cal_steps if d.name == pl_dict]
        if not len(x):
            m = f"Step '{pl_dict}' not found in available steps."
            raise ValueError(m)
        return x[0]
    return pl_dict
       
def _collect_user_params(obj, result = None):
    """
    Recurcsively loads user-specified parameters from the analysis YAML 
    file.
    
    Parameters:
    obj: Dictionary from loading the YAML file.
    
    Returns:
    result (dictionary): Keys are function names, and values are 
        dictionaries of parameter names and their values.
    """
    if result is None:
        result = {}

    if isinstance(obj, dict):
        # Check current level
        if "params" in obj:
            if "task" not in obj:
                raise KeyError("Found 'params' without a sibling 'task'")
            params_dict = obj["params"]
            for param_name, param_val in params_dict.items():
                if param_val in ['None', 'none']:
                    params_dict[param_name] = None
            result[obj["task"]] = params_dict

        # Recurse
        for v in obj.values():
            _collect_user_params(v, result)

    elif isinstance(obj, list):
        for item in obj:
            _collect_user_params(item, result)

    return result

################################################################################
######################## Functions not currently in use ########################
################################################################################

def _validate_nrows(cal_pl, nrows):
    "Not currently in use"
    path = pf.find_pl_path(cal_pl, 'nrows')
    step = path[-1] 

    # Check that nrows is the only returned name.
    if len(step.return_names) != 1:
        m = f"The function named '{step.name}' in "
        m += "custom_steps.py must only return 'nrows'."
        raise ValueError(m)
    
    # Check that the step returning nrows has func_type = 'global'.
    if step.func_type != 'global':
        m = f"The function named '{step.name}' in "
        m += "custom_steps.py must have return_type = 'global'."
        raise ValueError(m)
    
    # Load nrows, and check that it is integer-valued and > 0.
    if not (type(nrows) is int and nrows > 0):
        m = "The return parameter 'nrows' from the step named "
        m += f"'{step.name}' in custom_steps.py must be "
        m += "integer-valued and > 0."
        raise ValueError(m)