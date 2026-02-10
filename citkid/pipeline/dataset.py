import os
import yaml
import importlib.util 
import numpy as np
import zarr
import re
from datetime import datetime 

from .dependencies import get_most_recent_run, get_deps
from . import framework as pf
from . import util 
from . import default_steps

class Analyzer:
    def __init__(self, DS, analysis_yaml_path = None):
        raise NotImplementedError(
        "This function is not fully implemented yet and may need to be "
        "refactored."
        )
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
        raise NotImplementedError(
        "This function is not fully implemented yet and may need to be "
        "refactored."
        )
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
        raise NotImplementedError(
        "This function is not fully implemented yet and may need to be "
        "refactored."
        )
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
                raise ValueError("data_idx must be None for global functions")
            data_idx = None 
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
            if 'global' not in self.deps_maps:
                self.deps_maps['global'] = {}
            deps_map = self.deps_maps['global']
            run_idxs = [get_most_recent_run(name, deps_map) + 1 
                        for name in step.return_names]
            # run that doesn't exist yet should be 1 
            run_idxs = [1 if r == 0 else r for r in run_idxs]
            deps = get_deps(
                [p for p in step.param_names if p != 'data_idx'], 
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
                        f"Cannot create parameter '{name}' - this is a reserved"
                        f" DataSet attribute name. Reserved names: "
                        f"{sorted(reserved)}"
                    )
                
                # Ensure run_idx exists in memory cache
                if run_idx not in self._memory_cache:
                    self._memory_cache[run_idx] = {}
                    
                # For global-res, wrap outputs in LazyAttr
                if step.func_type == 'global-res':
                    # Create LazyAttr if needed, otherwise reuse existing
                    if name not in self._memory_cache[run_idx]:
                        self._memory_cache[run_idx][name] = pf.LazyAttr(
                            self, name, run_idx
                        )
                        
                        # Register with LazyAttrCollection 
                        # (only on first creation)
                        if name not in self._lazy_collections:
                            self._lazy_collections[name] =pf.LazyAttrCollection(
                                self, name
                                )
                        self._lazy_collections[name].add_run(
                            run_idx, 
                            self._memory_cache[run_idx][name]
                        )
                    
                    # Store data in LazyAttr cache
                    for i, v in enumerate(val):
                        self._memory_cache[run_idx][name]._cache[i] = v
                    
                    # Store dependencies for all rows
                    for i in range(self.nrows):
                        if i not in self.deps_maps:
                            self.deps_maps[i] = {}
                        if run_idx not in self.deps_maps[i]:
                            self.deps_maps[i][run_idx] = {}
                        self.deps_maps[i][run_idx][name] = deps
                    self._is_global_cache[name] = False
                else:
                    # For global, assign parameter directly
                    # Check for overwrite (global params should only be 
                    # created once)
                    if name in self._memory_cache[run_idx]:
                        raise ValueError(
                            f"Attempting to overwrite '{name}' at run {run_idx}"
                        )
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
                    [p for p in step.param_names if p != 'data_idx'], 
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
                            f"Cannot create parameter '{name}' - this is a "
                            f"reserved DataSet attribute name. Reserved names: "
                            f"{sorted(reserved)}"
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
                        if name not in self._lazy_collections:
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
                        raise AttributeError((
                            f"'{type(self).__name__}' object has no attribute "
                            f"'{name}'"
                        ))
                    
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
                    return self._get_existing(name, run_idx, data_idx=None)
            
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
        # Check if root exists (important for tests that create minimal 
        # DS objects)
        if 'root' not in self.__dict__:
            return False
        if self.root is None:
            return False
        
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
                        self._lazy_collections[name] = pf.LazyAttrCollection(
                            self, name
                            )
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
            f"{'with data_idx=' + str(data_idx) 
               if data_idx is not None else ''}"
            f" does not exist in memory or zarr and cannot be computed here. "
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
            for per-row/vectorized parameters, must be None for global 
            parameters.
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
        path = self._check_path_validity(path, data_idx, enforced_max_runs)
        for step in path:
            enforced_max_runs_i = {
                k: v for k, v in enforced_max_runs.items() 
                if k in step.param_names
            }
            if step.func_type in ['global', 'global-res']:
                step_data_idx = None
            else:
                step_data_idx = data_idx
            self._execute_step(
                step, 
                data_idx = step_data_idx, 
                enforced_max_runs = enforced_max_runs_i)
            
    def _check_path_validity(self, path, data_idx, enforced_max_runs):
        """
        Validate pipeline path and optimize by removing steps with existing 
        outputs.
        
        Checks that all required inputs are available (in memory, disk, or from
        preceding steps) and removes leading steps whose outputs already exist.
        
        Parameters:
        path (list of plStep): Pipeline steps to validate.
        data_idx (int, array-like, or None): Data indices being processed.
        enforced_max_runs (dict): Parameter name to max run_idx constraints.
        
        Returns:
        list of plStep: Optimized path with unnecessary leading steps removed.
        
        Raises:
        ValueError: If path is invalid or required inputs are unavailable.
        """
        if path is None:
            raise ValueError("Cannot execute None path")
        if len(path) == 0:
            return path
            
        # Convert data_idx to array for per-row steps
        if data_idx is not None:
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        
        # Find first step that needs execution (others have outputs already)
        first_step_idx = 0
        for step_idx, step in enumerate(path):
            # Determine what data_idx this step operates on
            if step.func_type in ['global', 'global-res']:
                step_data_idx = None
            else:
                step_data_idx = data_idx
            
            # Check if all outputs already exist at expected run indices
            all_outputs_exist = True
            for return_name in step.return_names:
                if step.func_type in ['global', 'global-res']:
                    # Global output
                    deps_map = self.deps_maps.get('global', {})
                    expected_run = get_most_recent_run(
                        return_name, deps_map
                        ) + 1
                    if expected_run == 0:
                        expected_run = 1
                    
                    if not (self._is_in_memory(return_name, expected_run, None) 
                         or self._is_in_zarr(return_name, expected_run, None)):
                        all_outputs_exist = False
                        break
                else:
                    # Per-row output - check all data_idx
                    for di in step_data_idx:
                        if di not in self.deps_maps:
                            all_outputs_exist = False
                            break
                        deps_map = self.deps_maps[di]
                        expected_run = get_most_recent_run(
                            return_name, deps_map
                            ) + 1
                        if expected_run == 0:
                            expected_run = 1
                        
                        if not (self._is_in_memory(
                                     return_name, expected_run, np.array([di])
                                ) or 
                                self._is_in_zarr(
                                    return_name, expected_run, np.array([di])
                                )
                            ):
                            all_outputs_exist = False
                            break
                    if not all_outputs_exist:
                        break
            
            if all_outputs_exist:
                first_step_idx = step_idx + 1
            else:
                break
        
        # Trim path to start at first step needing execution
        path = path[first_step_idx:]
        
        if len(path) == 0:
            return path
        
        # Track what will be produced by preceding steps in the trimmed path
        # Format: {param_name: run_idx} for global, {param_name: {di: run_idx}} 
        # for per-row
        produced_params = {}
        
        # Validate each remaining step
        for step_idx, step in enumerate(path):
            # Determine data_idx for this step
            if step.func_type in ['global', 'global-res']:
                step_data_idx = None
            else:
                step_data_idx = data_idx
            
            # Filter enforced_max_runs to this step's parameters
            step_enforced = {k: v for k, v in enforced_max_runs.items() 
                           if k in step.param_names}
            
            # Validate each input parameter
            for param_name in step.param_names:
                if param_name == 'data_idx':
                    continue  # Special built-in parameter
                
                # Determine required run_idx for this input
                if step.func_type in ['global', 'global-res']:
                    # Global step accessing inputs
                    deps_map = self.deps_maps.get('global', {})
                    
                    if param_name in step_enforced:
                        required_run = step_enforced[param_name]
                    else:
                        required_run = get_most_recent_run(param_name, deps_map)
                    
                    # Check if input is available
                    if param_name in produced_params:
                        # Will be produced by preceding step - just verify type 
                        # compatibility
                        produced_run = produced_params[param_name]
                        if isinstance(produced_run, dict):
                            raise ValueError(
                                f"Step '{step.name}' (global) requires global "
                                f"parameter '{param_name}', "
                                f"but preceding steps produce it as per-row"
                            )
                        # If enforced, verify the preceding step produces the 
                        # enforced run
                        if param_name in step_enforced and \
                            produced_run != step_enforced[param_name]:
                            raise ValueError(
                                f"Step '{step.name}' requires '{param_name}' at"
                                f" run {step_enforced[param_name]} "
                                f"(enforced), but preceding steps will produce "
                                f"run {produced_run}"
                            )
                    else:
                        # Must exist in memory or disk
                        if required_run < 0:
                            raise ValueError(
                                f"Step '{step.name}' requires '{param_name}', "
                                f"which does not exist "
                                f"and is not produced by preceding steps"
                            )
                        if not (self._is_in_memory(
                                     param_name, required_run, None
                                     ) or 
                                self._is_in_zarr(
                                    param_name, required_run, None
                                    )
                                ):
                            raise ValueError(
                                f"Step '{step.name}' requires '{param_name}' at"
                                 f" run {required_run}, "
                                f"which is not available in memory or disk"
                            )
                else:
                    # Per-row step accessing inputs
                    for di in step_data_idx:
                        if di not in self.deps_maps:
                            self.deps_maps[di] = {}
                        deps_map = self.deps_maps[di]
                        
                        if param_name in step_enforced:
                            required_run = step_enforced[param_name]
                        else:
                            required_run = get_most_recent_run(
                                param_name, deps_map
                                )
                        
                        # Check if input is available
                        if param_name in produced_params:
                            produced_run = produced_params[param_name]
                            if isinstance(produced_run, dict):
                                # Per-row parameter from preceding step
                                if di not in produced_run:
                                    raise ValueError(
                                    f"Step '{step.name}' requires "
                                    f"'{param_name}' at data_idx {di}, "
                                    f"but it will not be produced for this "
                                    "index"
                                    )
                                # If enforced, verify the preceding step 
                                # produces the enforced run
                                if param_name in step_enforced and \
                                produced_run[di] != step_enforced[param_name]:
                                    raise ValueError(
                                    f"Step '{step.name}' requires "
                                    f"'{param_name}' at run "
                                    f"{step_enforced[param_name]} "
                                    f"(enforced) for data_idx {di}, but "
                                    f"preceding steps will produce run"
                                    f" {produced_run[di]}"
                                    )
                            else:
                                # Global parameter from preceding step - just 
                                # verify enforced if needed
                                if param_name in step_enforced and \
                                    produced_run != step_enforced[param_name]:
                                    raise ValueError(
                                        f"Step '{step.name}' requires "
                                        f"'{param_name}' at run "
                                        f"{step_enforced[param_name]} "
                                        f"(enforced), but preceding steps will "
                                        f"produce run {produced_run}"
                                    )
                        else:
                            # Must exist in memory or disk
                            if required_run < 0:
                                raise ValueError(
                                    f"Step '{step.name}' requires "
                                    f"'{param_name}' for data_idx {di}, "
                                    f"which does not exist and is not produced "
                                    "by preceding steps"
                                )
                            if not (self._is_in_memory(
                                        param_name, required_run, np.array([di])
                                        ) or 
                                    self._is_in_zarr(
                                        param_name, required_run, np.array([di])
                                        )
                                    ):
                                raise ValueError(
                                    f"Step '{step.name}' requires "
                                    f"'{param_name}' at run {required_run} "
                                    f"for data_idx {di}, which is not available"
                                    f" in memory or disk"
                                )
            
            # Record what this step will produce (for subsequent steps)
            for return_name in step.return_names:
                if step.func_type in ['global', 'global-res']:
                    deps_map = self.deps_maps.get('global', {})
                    expected_run = get_most_recent_run(
                        return_name, deps_map
                        ) + 1
                    if expected_run == 0:
                        expected_run = 1
                    produced_params[return_name] = expected_run
                else:
                    # Per-row output
                    produced_params[return_name] = {}
                    for di in step_data_idx:
                        if di not in self.deps_maps:
                            self.deps_maps[di] = {}
                        deps_map = self.deps_maps[di]
                        expected_run = get_most_recent_run(
                            return_name, deps_map
                            ) + 1
                        if expected_run == 0:
                            expected_run = 1
                        produced_params[return_name][di] = expected_run
        
        return path
                
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
 
    def write_data(self, name, run_idx, data_idx = None, dtype = None):
        """
        Write data from memory to zarr file on disk.
        
        Writes parameter data and metadata (dependencies, global status, 
        row existence) to the zarr store. For per-row parameters, supports 
        incremental writing of individual rows.

        Parameters:
        name (str): The parameter name to write.
        run_idx (int): The run index to write.
        data_idx (int, array-like, or None): Data indices to write for per-row
            parameters. Must be None for global parameters.
        dtype (np.dtype, optional): Data type for zarr array. If None, infers
            from the data.

        Returns:
        None

        Raises:
        ValueError: If parameter doesn't exist in memory or if trying to 
            overwrite existing rows.
        """
        # Validate run_idx exists
        if run_idx not in self._memory_cache:
            raise ValueError(f"run_idx {run_idx} not found in memory cache")
        if name not in self._memory_cache[run_idx]:
            raise ValueError(
                f"Parameter '{name}' at run_idx {run_idx} not found in memory "
                "cache"
            )
        
        # Get data from memory
        is_global = self._is_global_cache[name]
        
        # Create run group if needed
        run_str = f'run{run_idx}'
        run_grp = self.root.require_group(run_str)
        
        if is_global:
            # Global parameter
            if data_idx is not None:
                raise ValueError(
                    f"data_idx must be None for global parameter '{name}'"
                )
            
            # Get data and dependencies
            data = self._memory_cache[run_idx][name]
            deps = self.deps_maps['global'][run_idx][name]
            
            # Convert data to array with specified dtype
            data = np.asarray(data, dtype=dtype)
            
            # Create parameter group
            if name in run_grp:
                raise ValueError(
                    f"Parameter '{name}' at run {run_idx} already exists in "
                    "zarr"
                )
            param_grp = run_grp.create_group(name)
            
            # Write data as single array
            data_arr = param_grp.create_array(
                name='data',
                data=data
            )
            
            # Write metadata
            param_grp.attrs['global'] = True
            param_grp.attrs['deps'] = deps
            param_grp.attrs['write_time'] = datetime.now().strftime(
                                                               '%Y%m%d-%H:%M:%S'
                                                                   )
            
        else:
            # Per-row parameter (including global-res)
            if data_idx is None:
                raise ValueError(
                    f"data_idx required for per-row parameter '{name}'"
                )
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
            
            # Get LazyAttr from memory
            lazy_attr = self._memory_cache[run_idx][name]
            if not isinstance(lazy_attr, pf.LazyAttr):
                raise TypeError(
                    f"Expected LazyAttr for per-row parameter '{name}', "
                    f"got {type(lazy_attr)}"
                )
            
            # Collect data for requested indices
            data_list = []
            for di in data_idx:
                if di not in lazy_attr._cache:
                    raise ValueError(
                        f"data_idx {di} not found in memory for '{name}' at run"
                        f" {run_idx}"
                    )
                data_list.append(lazy_attr._cache[di])
            
            # Convert to array with specified dtype
            data = np.asarray(data_list, dtype=dtype)
            if dtype is None:
                dtype = data.dtype
            
            # Create or access parameter group
            if name not in run_grp:
                # Create new group and arrays
                param_grp = run_grp.create_group(name)
                
                # Determine shape and chunks
                shape = (self.nrows, *data.shape[1:])
                chunks = (1, *data.shape[1:])
                
                # Create data array
                data_arr = param_grp.create_array(
                    name='data',
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks
                )
                
                # Create row_exists array
                exists_arr = param_grp.create_array(
                    name='row_exists',
                    shape=(self.nrows,),
                    dtype=np.bool_
                )
                
                # Write metadata
                param_grp.attrs['global'] = False
                param_grp.attrs['deps'] = {}
                param_grp.attrs['write_times'] = {}
                
            else:
                # Access existing arrays
                param_grp = run_grp[name]
                data_arr = param_grp['data']
                exists_arr = param_grp['row_exists']
                
                # Check for overwrites
                existing_rows = exists_arr[...]
                conflicts = np.isin(data_idx, np.where(existing_rows)[0])
                if np.any(conflicts):
                    conflict_indices = data_idx[conflicts]
                    raise ValueError(
                        f"Cannot overwrite existing data for '{name}' at run"
                        f" {run_idx} for data_idx: {conflict_indices}"
                    )
            
            # Write data for each index
            for i, di in enumerate(data_idx):
                data_arr[di] = data[i]
                exists_arr[di] = True
            
            # Update dependencies and write times in attrs
            deps_dict = param_grp.attrs['deps']
            write_times_dict = param_grp.attrs.get('write_times', {})
            timestamp = datetime.now().strftime('%Y%m%d-%H:%M:%S')
            
            for di in data_idx:
                di_str = f'idx{di}'
                deps = self.deps_maps[di][run_idx][name]
                deps_dict[di_str] = deps
                write_times_dict[di_str] = timestamp
            
            param_grp.attrs['deps'] = deps_dict
            param_grp.attrs['write_times'] = write_times_dict

 
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
            raise ValueError(
                f"root can only contain run folders: {run_str} group found"
                ) 
        run_idx = int(run_str.replace('run', ''))
    
        # Iterate through params in the run
        for name, subgrp in grp.groups():
            # Input validation 
            if 'deps' not in subgrp.attrs.keys():
                raise ValueError(
                    f"Missing 'deps' attr for {run_str} parameter {name}"
                    ) 
            if 'global' not in subgrp.attrs.keys():
                raise ValueError(
                    f"Missing 'global' attribute for {run_str} parameter {name}"
                    )
            if any(subgrp.groups()):
                raise ValueError(
                    f"{run_str} parameter {name} contains a zarr group"
                    ) 
            array_names = [a[0] for a in subgrp.arrays()]
            if "data" not in array_names:
                raise ValueError(
                    f"'data' array not found in {run_str} parameter {name}"
                )
            array_names = [d for d in array_names if d != 'data']
            if subgrp.attrs['global']:
                if len(array_names) != 0:
                    raise ValueError(
                        f"Extra array(s) found in {run_str} global parameter"
                        f" {name}: {array_names}"
                    )
            else:
                if "row_exists" not in array_names:
                    raise ValueError(
                        f"'row_exists' array not found in {run_str} parameter "
                        f"{name}"
                        )
                # Validate row_exists dtype
                row_exists = subgrp['row_exists']
                if row_exists.dtype != np.bool_:
                    raise ValueError(
                        f"'row_exists' array in {run_str} parameter {name} "
                        f"must have dtype bool, got {row_exists.dtype}"
                    )
                array_names = [d for d in array_names if d != 'row_exists']
                if len(array_names) != 0:
                    raise ValueError(
                        f"Extra array(s) found in {run_str} parameter {name}: "
                        f"{array_names}"
                    )
            
            # Validate and process deps attribute based on global status
            deps_attr = subgrp.attrs['deps']
            if not isinstance(deps_attr, dict):
                raise ValueError(
                    f"'deps' attribute in {run_str} parameter {name} must be a "
                    "dictionary"
                )
            
            if subgrp.attrs['global']:
                # For global parameters, deps should be {str: int}
                for key, value in deps_attr.items():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"Global parameter {name} in {run_str}: deps keys "
                            f"must be strings, got {type(key)}"
                        )
                    if not isinstance(value, (int, np.integer)):
                        raise ValueError(
                            f"Global parameter {name} in {run_str}: deps values"
                            f" must be integers, got {type(value)}"
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
                # For non-global parameters, deps should be 
                # {data_idx_str: {str: int}}
                for data_idx_str, deps in deps_attr.items():
                    # Validate data_idx_str can be converted to int
                    try:
                        if isinstance(data_idx_str, str) and \
                           data_idx_str.startswith('idx'):
                            data_idx = int(data_idx_str.replace('idx', ''))  
                        else:
                            data_idx = int(data_idx_str)
                    except (ValueError, AttributeError):
                        raise ValueError(
                            f"Non-global parameter {name} in {run_str}: deps "
                            f"keys must be convertible to int "
                            f"(format 'idx<int>' or int), got {data_idx_str}"
                        )
                    
                    # Validate deps structure
                    if not isinstance(deps, dict):
                        raise ValueError(
                            f"Non-global parameter {name} in {run_str} data_idx"
                            f" {data_idx}: deps must be a dictionary, got "
                            f"{type(deps)}"
                        )
                    for key, value in deps.items():
                        if not isinstance(key, str):
                            raise ValueError(
                                f"Non-global parameter {name} in {run_str} "
                                f"data_idx {data_idx}: deps keys must be "
                                f"strings, got {type(key)}"
                            )
                        if not isinstance(value, (int, np.integer)):
                            raise ValueError(
                                f"Non-global parameter {name} in {run_str} "
                                f"data_idx {data_idx}: deps values must be "
                                f"integers, got {type(value)}"
                            )
                    
                    # Add to deps_maps
                    if data_idx not in deps_maps:
                        deps_maps[data_idx] = {}
                    if run_idx not in deps_maps[data_idx]:
                        deps_maps[data_idx][run_idx] = {}
                    if name in deps_maps[data_idx][run_idx]:
                        raise ValueError(
                            f"Duplicate parameter {name} found in {run_str} for"
                            f" data_idx {data_idx}"
                        )
                    deps_maps[data_idx][run_idx][name] = deps
    
    return deps_maps

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

################################################################################
# Analysis helper
################################################################################    
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
    raise NotImplementedError(
        "This function is not fully implemented yet and may need to be "
        "refactored."
        )
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
