import os
from sqlalchemy import func
import yaml
import importlib.util 
import numpy as np
import zarr
import re
from datetime import datetime 
import matplotlib.pyplot as plt

from .dependencies import get_most_recent_run, get_deps
from . import framework as pf
from . import util 
from . import default_steps
from ..xcal.plot import default_plot_funcs
from ..util import combine_figs_horz, combine_figs_vert


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
        self.cal_steps = _load_custom_steps(
            self.custom_path
            )
        # Add default calibration steps if not already present
        for step in default_steps.default_cal_steps:
            if step.name not in [s.name for s in self.cal_steps]:
                self.cal_steps.append(step)
                
        # Load calibration YAML file and convert it to calibration pipeline
        with open(self.cal_yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f) 
        self.cal_pl = _convert_yaml_to_steps(yaml_dict, self.cal_steps)

        # confirm that the cal_pl structure is valid
        pf.check_pl_tree_structure(self.cal_pl, cal = True) 

        # Load deps_map and is_global_cache from zarr - also validates zarr 
        # structure
        self.deps_maps, self._is_global_cache = _load_deps_from_zarr(self.root)

        # Load existing data from zarr into memory cache
        # self._memory_cache = _load_existing_data_from_zarr(
        #     self.root, self.deps_maps, self._is_global_cache
        # )
        self._memory_cache = {}
        
        # start lazy_collections, which tracks LazyAttrCollections for per-row 
        # params
        self._lazy_collections = {}
        
        # Validate that nrows can be produced and is a global parameter
        nrows_path = pf.find_pl_path(self.cal_pl, 'nrows')
        if nrows_path is None:
            raise ValueError(
                "Calibration pipeline must be able to produce 'nrows' "
                "as a global parameter. No step found that produces 'nrows'."
            )
        # Check that the step producing nrows is global
        nrows_step = nrows_path[-1]  # Last step in path produces nrows
        if nrows_step.func_type not in ['global', 'global-res']:
            raise ValueError(
                f"Parameter 'nrows' must be produced by a global or global-res "
                f"step, but is produced by step '{nrows_step.name}' with "
                f"func_type '{nrows_step.func_type}'"
            )
        if nrows_step.func_type == 'global-res':
            raise ValueError(
                "Parameter 'nrows' must be produced by a 'global' step, not "
                "'global-res'. nrows should be a single integer value, not a "
                "per-row array."
            )
        
        # Initialize global entry in deps_maps
        if 'global' not in self.deps_maps:
            self.deps_maps['global'] = {}
        # Per-data_idx entries are initialised lazily (in _store_param and
        # LazyAttrCollection.__getitem__) so we do NOT call self.nrows here.
        # Eagerly computing nrows would execute the pipeline, which may require
        # user-provided parameters that are not yet available.

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
        # Guard: prevent infinite recursion when the instance is partially
        # initialised (e.g. created via __new__ in tests, or during early
        # __init__ before all essential attrs exist).  None of these attrs
        # should ever be pipeline parameters, so raising AttributeError
        # here is always correct.
        if not all(a in self.__dict__
                   for a in ('_is_global_cache', '_memory_cache', 'deps_maps')):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

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
                    if 'cal_pl' not in self.__dict__:
                        raise AttributeError(
                            f"'{type(self).__name__}' object has no attribute "
                            f"'{name}'"
                        )
                    path = pf.find_pl_path(self.cal_pl, name)
                    if path is None:
                        raise AttributeError((
                            f"'{type(self).__name__}' object has no attribute "
                            f"'{name}'"
                        ))
                    
                    # Determine type from pipeline
                    # global-res stores data per-row (not in deps_maps['global']),
                    # so it must be treated as per-row here.
                    final_step = path[-1]
                    is_global = final_step.func_type == 'global'
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
                    return self._get_existing(name, run_idx, data_idx = None)
            
            # Not found - produce it
            self._ensure_loaded(name, data_idx = None)
            
            # Retrieve from memory (should exist now)
            run_idx = get_most_recent_run(name, self.deps_maps['global'])
            return self._memory_cache[run_idx][name]
        
        else:
            # For per-row parameters: return LazyAttrCollection
            # The collection handles lazy loading when indexed
            if name not in self._lazy_collections:
                self._lazy_collections[name] = pf.LazyAttrCollection(self, name)
            return self._lazy_collections[name]

    def _sync_to_zarr(self):
        """
        Renumber in-memory run indices so they are consistent with zarr.

        For every parameter that exists in memory, this method ensures that the
        highest in-memory run index is exactly one greater than the highest
        zarr run index (or 1 if the parameter is not yet saved).  All
        intermediate in-memory runs that sit above the zarr ceiling are
        purged, leaving only the single most-recent in-memory run, renamed to
        the correct next index.  Parameters whose highest in-memory run index
        is already at or below the zarr ceiling are left completely untouched
        (they represent zarr-backed data loaded into the cache and have no
        stale indices).

        This must be called before write_data so that the deps stored on disk
        reference run indices that will be reproducible on the next session
        load.

        After this call the complete in-memory/deps state is coherent with the
        zarr file: every dep entry pointing at an ephemeral parameter will use
        run index 1 (or zarr_max+1 if some runs were previously saved).

        Returns:
        None  (mutates self._memory_cache, self.deps_maps,
               self._lazy_collections in-place)
        """
        # Guard: skip if the instance is not fully initialised.  This can
        # happen when tests create DS via __new__ without running __init__.
        required = ('_memory_cache', 'deps_maps', '_lazy_collections', 'root')
        if not all(a in self.__dict__ for a in required):
            return

        # ------------------------------------------------------------------
        # Phase 1: compute rename targets (read-only pass)
        # ------------------------------------------------------------------
        # rename_map : param -> (old_mem_max, target)
        # purge_map  : param -> set of run indices to delete from memory
        rename_map = {}  # {param: (old_mem_max, target)}
        purge_map  = {}  # {param: set of run_idxs to purge}

        # Collect every param present in memory
        all_mem_params = set()
        for run_cache in self._memory_cache.values():
            all_mem_params.update(run_cache.keys())

        # Build zarr_max_per_param once (O(n_runs) zarr I/O)
        zarr_max_per_param = {}
        for run_str, run_grp in self.root.groups():
            r = int(run_str.replace('run', ''))
            for pname, _ in run_grp.groups():
                prev = zarr_max_per_param.get(pname, -1)
                zarr_max_per_param[pname] = max(prev, r)

        for param in all_mem_params:
            # All run indices for this param that are in memory
            mem_runs = sorted(
                r for r in self._memory_cache if param in self._memory_cache[r]
            )
            if not mem_runs:
                continue
            mem_max = mem_runs[-1]

            # Highest run index for this param that is saved in zarr
            zarr_max = zarr_max_per_param.get(param, -1)

            # If mem_max is at or below the zarr ceiling the data came from
            # zarr itself — don't touch it.
            if mem_max <= zarr_max:
                continue

            # max(1, ...) because the pipeline always starts at run 1;
            # zarr_max + 1 would be 0 when nothing has been saved yet.
            target = max(1, zarr_max + 1)

            # Runs above the zarr ceiling that are NOT the one we keep
            above_zarr = [r for r in mem_runs if r > zarr_max]
            purge_runs = set(above_zarr[:-1])  # everything except mem_max

            rename_map[param] = (mem_max, target)
            purge_map[param]  = purge_runs

        # ------------------------------------------------------------------
        # Phase 2: apply purges and renames
        # ------------------------------------------------------------------
        for param, purge_runs in purge_map.items():
            old_max, target = rename_map[param]

            # 2a. Delete purged runs from _memory_cache
            for r in purge_runs:
                del self._memory_cache[r][param]

            # 2b. Rename old_max -> target in _memory_cache (no-op if equal)
            if old_max != target:
                val = self._memory_cache[old_max].pop(param)
                if target not in self._memory_cache:
                    self._memory_cache[target] = {}
                self._memory_cache[target][param] = val

                # Update LazyAttr.run_idx on the stored value if needed
                if isinstance(self._memory_cache[target][param], pf.LazyAttr):
                    self._memory_cache[target][param].run_idx = target

            # 2c. Mirror in _lazy_collections
            if param in self._lazy_collections:
                lac = self._lazy_collections[param]
                for r in purge_runs:
                    lac._lazy_attrs.pop(r, None)
                if old_max != target and old_max in lac._lazy_attrs:
                    la = lac._lazy_attrs.pop(old_max)
                    la.run_idx = target
                    lac._lazy_attrs[target] = la

            # 2d. Mirror in deps_maps (outer keys = run indices where param
            #     was produced).  deps_maps is keyed by scope (either
            #     'global' or an integer data_idx).
            for scope in list(self.deps_maps.keys()):
                scope_map = self.deps_maps[scope]
                for r in purge_runs:
                    if r in scope_map and param in scope_map[r]:
                        del scope_map[r][param]
                if old_max != target:
                    if old_max in scope_map and param in scope_map[old_max]:
                        deps_entry = scope_map[old_max].pop(param)
                        if target not in scope_map:
                            scope_map[target] = {}
                        scope_map[target][param] = deps_entry

        # Remove now-empty run dicts from _memory_cache and deps_maps
        for r in list(self._memory_cache.keys()):
            if not self._memory_cache[r]:
                del self._memory_cache[r]
        for scope in list(self.deps_maps.keys()):
            for r in list(self.deps_maps[scope].keys()):
                if not self.deps_maps[scope][r]:
                    del self.deps_maps[scope][r]

        # ------------------------------------------------------------------
        # Phase 3: update inner dep entries
        # ------------------------------------------------------------------
        # Any dep dict that references an old membrane run index (e.g.
        # {zf_rmv: 16}) must be updated to the new target (e.g. {zf_rmv: 1}).
        for scope in self.deps_maps:
            for run_idx in self.deps_maps[scope]:
                for prod_param, inner_deps in \
                        self.deps_maps[scope][run_idx].items():
                    for dep_param, dep_run in list(inner_deps.items()):
                        if dep_param in rename_map:
                            old_max, target = rename_map[dep_param]
                            if dep_run == old_max:
                                inner_deps[dep_param] = target

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
        # Renumber in-memory run indices to be consistent with zarr before
        # writing, so that persisted dependency entries are session-independent.
        self._sync_to_zarr()

        # After sync, run_idx may have changed — resolve the correct run_idx
        # for this parameter (the most recent one now in memory).
        if run_idx not in self._memory_cache or \
                name not in self._memory_cache[run_idx]:
            matching = [
                r for r in self._memory_cache
                if name in self._memory_cache[r]
            ]
            if matching:
                run_idx = max(matching)
            # If no matching run found, leave run_idx unchanged so the
            # validation block below raises the correct descriptive error.

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
                    chunks=chunks,
                    shards=shape,
                    dtype=dtype
                )
                
                # Create row_exists array
                exists_arr = param_grp.create_array(
                    name='row_exists',
                    shape=(self.nrows,),
                    shards=(self.nrows,),
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

    
    ############################################################################
    ###################### Store param in memory utility #######################
    ############################################################################
    def _store_param(self, name, value, run_idx, deps, is_global, 
                     data_idx = None):
        """
        Store a parameter in memory cache with dependency tracking.
        
        Handles both global and per-row parameters, including LazyAttr creation
        and collection registration for per-row data. This method centralizes
        the storage logic used by _execute_step and can be reused by other
        components (e.g., add_user_params).
        
        Parameters:
        name (str): Parameter name to store.
        value: The data to store. For global parameters, this is the direct 
            value. For per-row parameters, this should be an array-like where 
            each element corresponds to a data_idx.
        run_idx (int): Run index for this parameter version.
        deps (dict): Dependencies dict mapping parameter names to their run 
            indices.
        is_global (bool): True for global parameters, False for per-row 
            parameters.
        data_idx (array-like or None): Data indices for per-row parameters. 
            Must be None for global parameters. Required for per-row parameters.
        
        Returns:
        None (stores in self._memory_cache, self.deps_maps, 
              self._is_global_cache, and self._lazy_collections)
        
        Raises:
        ValueError: If name is reserved, if trying to overwrite global 
            parameter, or if data_idx doesn't match is_global.
        TypeError: If input types are invalid.
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        run_idx = int(run_idx)
        if not isinstance(deps, dict):
            raise TypeError("deps must be a dictionary")
        if not isinstance(is_global, bool):
            raise TypeError("is_global must be a boolean")
        
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
        
        # Store based on parameter type
        if is_global:
            # Global parameter
            if data_idx is not None:
                raise ValueError(
                    f"data_idx must be None for global parameter '{name}'"
                )
            
            # Check for overwrite (global params should only be created once)
            if name in self._memory_cache[run_idx]:
                raise ValueError(
                    f"Attempting to overwrite global parameter '{name}' at run "
                    f"{run_idx}"
                )
            
            # Store value directly
            self._memory_cache[run_idx][name] = value
            
            # Update deps_maps
            if 'global' not in self.deps_maps:
                self.deps_maps['global'] = {}
            if run_idx not in self.deps_maps['global']:
                self.deps_maps['global'][run_idx] = {}
            self.deps_maps['global'][run_idx][name] = deps
            
            # Update is_global cache
            self._is_global_cache[name] = True
            
        else:
            # Per-row parameter
            if data_idx is None:
                raise ValueError(
                    f"data_idx required for per-row parameter '{name}'"
                )
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
            value = np.atleast_1d(np.asarray(value))
            
            # Validate value length matches data_idx
            if len(value) != len(data_idx):
                raise ValueError(
                    f"Length mismatch: value has {len(value)} elements but "
                    f"data_idx has {len(data_idx)} elements"
                )
            
            # Create LazyAttr if needed
            if name not in self._memory_cache[run_idx]:
                self._memory_cache[run_idx][name] = pf.LazyAttr(
                    self, name, run_idx
                )
                
                # Register with LazyAttrCollection (only on first creation)
                if name not in self._lazy_collections:
                    self._lazy_collections[name] = pf.LazyAttrCollection(
                        self, name
                    )
                if run_idx not in \
                    self._lazy_collections[name]._lazy_attrs.keys():
                    self._lazy_collections[name].add_run(
                        run_idx, 
                        self._memory_cache[run_idx][name]
                    )
            
            # Store data in LazyAttr cache
            for idx, v in zip(data_idx, value):
                self._memory_cache[run_idx][name]._cache[int(idx)] = v
            
            # Store dependencies for each data_idx
            for di in data_idx:
                if di not in self.deps_maps:
                    self.deps_maps[di] = {}
                if run_idx not in self.deps_maps[di]:
                    self.deps_maps[di][run_idx] = {}
                self.deps_maps[di][run_idx][name] = deps
            
            # Update is_global cache
            self._is_global_cache[name] = False
    
    def _collect_step_parameters_global(self, step, deps):
        """
        Collect parameters for global/global-res step execution.
        
        Returns (params, param_is_global) tuples.
        """
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
        
        return params, param_is_global
    
    def _store_step_outputs_global(self, step, out, run_idxs, deps, save):
        """
        Store outputs from global/global-res step execution.
        """
        for (name, val), run_idx in zip(out.items(), run_idxs):
            if step.func_type == 'global-res':
                # Global-res: store for all rows
                self._store_param(
                    name, val, run_idx, deps, 
                    is_global = False, 
                    data_idx = range(self.nrows)
                )
            else:
                # Global: store directly
                self._store_param(
                    name, val, run_idx, deps, 
                    is_global = True
                )
        
        # Write to disk if requested
        if save:
            for (name, _), run_idx in zip(out.items(), run_idxs):
                if step.func_type == 'global-res':
                    self.write_data(
                        name, run_idx, data_idx=range(self.nrows)
                        )
                else:
                    self.write_data(name, run_idx, data_idx=None)

    ############################################################################
    ############################## Step execution ##############################
    ############################################################################
    def _execute_step(self, step, data_idx = None, enforced_max_runs = {}, 
                      save = False):
        """
        Execute a single pipeline step, storing results and dependencies in 
        memory at the most recent run idx + 1.
        
        All input parameters must already exist in memory or on disk - this 
        method will NOT execute additional steps to generate missing inputs.
        
        Parameters:
        step (plStep): The pipeline step to execute.
        data_idx (int, array-like, or None): Data indices to process. Required
            for 'per-row' and 'vectorized' steps, must be None for 'global' and 
            'global-res' steps.
        enforced_max_runs (dict): Optional dict mapping parameter names to 
            maximum run indices for dependency resolution.
        save (bool): If True, write outputs to zarr file on disk after 
            execution. Default is False.
        
        Returns:
        None (results are stored in self._memory_cache and optionally written
             to zarr)
        
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
            if step.func_type == 'global':
                deps_map = self.deps_maps['global']
            else:
                # all data_idx will share the same deps for returns from a 
                # global-res step, so it is ok to merge with data_idx = 0
                deps_map = self.merge_deps_maps(0)

            
            # Compute current dependencies for inputs
            input_params = [p for p in step.param_names if p != 'data_idx']
            deps = get_deps(
                input_params, deps_map, enforced_max_runs=enforced_max_runs
                )
            
            # compute new run indices
            run_idxs = [get_most_recent_run(name, deps_map) + 1 
                        for name in step.return_names]
            # run that doesn't exist yet should be 1 
            run_idxs = [1 if r == 0 else r for r in run_idxs]
            # Collect parameters 
            params, param_is_global = self._collect_step_parameters_global(
                step, deps
                )

            # Run step
            out = step._run(params, param_is_global)

            # Store outputs in memory cache
            self._store_step_outputs_global(step, out, run_idxs, deps, save)

        else:  # vectorized or per-row                      
            # For vectorized/per-row, need to find run-idx for each data_idx
            run_deps = []  # (run_idxs_tuple, deps) for each data_idx
            for di in data_idx:
                merged_deps_map = self.merge_deps_maps(di)
                run_idxs = [get_most_recent_run(name, merged_deps_map) + 1
                           for name in step.return_names]
                # run that doesn't exist yet should be 1
                run_idxs = tuple([1 if r == 0 else r for r in run_idxs])
                deps = get_deps(
                    [p for p in step.param_names if p != 'data_idx'],
                    merged_deps_map,
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
                    # Per-row: store for specific data indices
                    self._store_param(
                        name, val, run_idx, deps,
                        is_global=False,
                        data_idx=data_idx[local_idx]
                    ) 
                
                # Write to disk if requested
                if save:
                    for (name, _), run_idx in zip(out.items(), run_idxs):
                        self.write_data(
                            name, run_idx, data_idx=data_idx[local_idx]
                            )
                        
    def merge_deps_maps(self, di):
        """
        Merge global and per-row deps_maps for a given data_idx.
        Returns a dict mapping run_idx to deps.
        """
        merged_deps_map = {}
        if 'global' in self.deps_maps:
            for run_idx_i, params in self.deps_maps['global'].items():
                merged_deps_map[run_idx_i] = params.copy()
        if di in self.deps_maps:
            for run_idx_i, params in self.deps_maps[di].items():
                if run_idx_i not in merged_deps_map:
                    merged_deps_map[run_idx_i] = {}
                merged_deps_map[run_idx_i].update(params)
        return merged_deps_map
    ############################################################################
    ########################## Data fetching methods ###########################
    ############################################################################
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
    
    def _get_needs_execution_global(self, step, enforced_max_runs):
        # First, return True if any return doesn't exist yet=
        step_data_idx = None 
        curr_deps_map = self.deps_maps['global']
        if any(get_most_recent_run(name, curr_deps_map) < 0
               for name in step.return_names):
            return True

        # If return_names do exist, check if their deps match the 
        # deps they would have if the step is executed again now
        new_deps = {}
        for name in step.param_names: 
            if name == 'data_idx':
                continue 
            if name in enforced_max_runs:
                new_deps[name] = enforced_max_runs[name]
            else:
                ri_curr = get_most_recent_run(name, curr_deps_map)
                ri_curr = 1 if ri_curr == -1 else ri_curr
                new_deps[name] = ri_curr

        needs_execution = True
        # Do any run indices have deps that are equal to new_deps?
        for run_idx, curr_deps in curr_deps_map.items():
            deps_match = [] 
            for key, dep in new_deps.items():
                    if key in curr_deps and curr_deps[key] == dep:
                        deps_match.append(True)
                    else:
                        deps_match.append(False)
            if all(deps_match):
                needs_execution = False
                break
        return needs_execution
    
    def _get_needs_execution_nonglobal(self, step, data_idx, enforced_max_runs):
        step_data_idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int32))
        
        row_needs_execution = [False for _ in range(step_data_idx.shape[0])]
        for local_idx, di in enumerate(step_data_idx): 
            curr_deps_map = self.merge_deps_maps(di)
            # First, assign True if any return doesn't exist yet
            if any(get_most_recent_run(name, curr_deps_map) < 0
                for name in step.return_names):
                row_needs_execution[local_idx] = True
                continue 

            # If return_names do exist, check if their deps match the 
            # deps they would have if the step is executed again now
            new_deps = {}
            for name in step.param_names:
                if name == 'data_idx':
                    continue
                if name in enforced_max_runs:
                    new_deps[name] = enforced_max_runs[name]
                else:
                    new_deps[name] = get_most_recent_run(name, curr_deps_map)

            needs_execution = True
            # Do any run indices have deps that are equal to new_deps?
            for run_idx, curr_deps in curr_deps_map.items():
                curr_deps = curr_deps_map[1]
                deps_match = [] 
                for name in step.return_names:
                    runs_match = [] 
                    if name not in curr_deps:
                        deps_match.append(False)
                        continue
                    for param, ri in new_deps.items():
                        ri_curr = curr_deps[name][param]
                        runs_match.append(ri == ri_curr)
                    deps_match.append(all(runs_match))
                if all(deps_match):
                    needs_execution = False
                    break
            row_needs_execution[local_idx] = needs_execution 

        # Filter step_data_idx by needs_execution 
        step_data_idx = step_data_idx[row_needs_execution] 
        needs_execution = bool(len(step_data_idx))
        return step_data_idx, needs_execution
    
    def _ensure_loaded(self, name, data_idx = None, enforced_max_runs = {}):
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
        cal_pl = self.__dict__.get('cal_pl')
        if cal_pl is None:
            return
        path = pf.find_pl_path(cal_pl, name)
        if path is None:
            return
        for step in path:
            enforced_max_runs_i = {
                k: v for k, v in enforced_max_runs.items() 
                if k in step.param_names
            }

            # Check if step inputs most recent run idxs match dependencies of 
            # already existing data. If so, skip execution. 
            if step.func_type == 'global':
                needs_execution = \
                    self._get_needs_execution_global(
                        step, enforced_max_runs_i
                        )
                step_data_idx = None
            elif step.func_type == 'global-res':
                # Sufficient to check index 0, since global-res will create 
                # data for all data_idx and they will all share the same deps
                _, needs_execution = \
                    self._get_needs_execution_nonglobal(
                        step, 0, enforced_max_runs_i
                        ) 
                step_data_idx = None
            else: # vectorized or per-row
                step_data_idx, needs_execution = \
                    self._get_needs_execution_nonglobal(
                        step, data_idx, enforced_max_runs_i
                        )

            # Execute step 
            if not needs_execution:
                continue
            self._execute_step(
                step, 
                data_idx = step_data_idx, 
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
        # Only convert data_idx to an array when it is not None; converting
        # None with dtype=int32 raises TypeError in modern NumPy versions.
        if data_idx is not None:
            data_idx = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
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
                    # Merge global and per-row deps_maps for this data_idx
                    merged_deps_map = {}
                    if 'global' in self.deps_maps:
                        for run_idx_i, params in \
                            self.deps_maps['global'].items():
                            merged_deps_map[run_idx_i] = params.copy()
                    if di in self.deps_maps:
                        for run_idx_i, params in self.deps_maps[di].items():
                            if run_idx_i not in merged_deps_map:
                                merged_deps_map[run_idx_i] = {}
                            merged_deps_map[run_idx_i].update(params)
                    run = get_most_recent_run(name, merged_deps_map) + 1 
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
            self._ensure_loaded(
                name, 
                data_idx = data_idx, 
                enforced_max_runs = enforced_max_runs
                )
                
        # Return from memory or disk
        return self._get_existing(name, run_idx, data_idx)
    
    def _is_in_memory(self, name, run_idx, data_idx = None):
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
    
    def _is_in_zarr(self, name, run_idx, data_idx = None):
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
        
        return np.all(run_grp[name]['row_exists'][data_idx])
    
    ############################################################################
    ####################### Path validity check utility ########################
    ############################################################################
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
        # return path  # Placeholder - implement validation and optimization logic here
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
            
            # Check if all outputs already exist at their most recent run indices
            # AND if dependencies match what would be computed now
            all_outputs_exist = True
            for return_name in step.return_names:
                if step.func_type in ['global', 'global-res']:
                    # Global output
                    deps_map = self.deps_maps.get('global', {})
                    most_recent_run = get_most_recent_run(return_name, deps_map)
                    
                    # If no run exists yet, data doesn't exist
                    if most_recent_run == -1:
                        all_outputs_exist = False
                        break
                    
                    # Check if data exists at the most recent run
                    if not (self._is_in_memory(
                        return_name, most_recent_run, None
                        ) 
                        or self._is_in_zarr(
                        return_name, most_recent_run, None
                        )
                    ):
                        all_outputs_exist = False
                        break
                    
                    # Check if dependencies match what would be computed now
                    # (only if step has input parameters)
                    input_params = [p for p in step.param_names 
                                    if p != 'data_idx']
                    if input_params:
                        stored_deps = deps_map[most_recent_run][return_name]
                        current_deps = get_deps(
                            input_params, 
                            deps_map, 
                            enforced_max_runs=enforced_max_runs
                        )
                        if stored_deps != current_deps:
                            # Dependencies changed, need new run
                            all_outputs_exist = False
                            break
                else:
                    # Per-row output - check all data_idx
                    for di in step_data_idx:
                        if di not in self.deps_maps:
                            all_outputs_exist = False
                            break
                        
                        # Merge global and per-row deps_maps for this data_idx
                        merged_deps_map = {}
                        if 'global' in self.deps_maps:
                            for run_idx_i, params in \
                                self.deps_maps['global'].items():
                                merged_deps_map[run_idx_i] = params.copy()
                        if di in self.deps_maps:
                            for run_idx_i, params in self.deps_maps[di].items():
                                if run_idx_i not in merged_deps_map:
                                    merged_deps_map[run_idx_i] = {}
                                merged_deps_map[run_idx_i].update(params)
                        
                        deps_map = self.deps_maps[di]
                        most_recent_run = get_most_recent_run(
                            return_name, deps_map
                            )
                        
                        # If no run exists yet, data doesn't exist
                        if most_recent_run == -1:
                            all_outputs_exist = False
                            break
                        
                        # Check if data exists at the most recent run
                        if not (self._is_in_memory(
                                     return_name, most_recent_run, 
                                     np.array([di])
                                ) or 
                                self._is_in_zarr(
                                    return_name, most_recent_run, np.array([di])
                                )
                            ):
                            all_outputs_exist = False
                            break
                        
                        # Check if dependencies match what would be computed now
                        # (only if step has input parameters)
                        input_params = [p for p in step.param_names 
                                        if p != 'data_idx']
                        if input_params:
                            stored_deps = deps_map[most_recent_run][return_name]
                            current_deps = get_deps(
                                input_params, 
                                merged_deps_map, 
                                enforced_max_runs=enforced_max_runs
                            )
                            if stored_deps != current_deps:
                                # Dependencies changed, need new run
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
                        
                        # Merge global and per-row deps_maps for this data_idx
                        merged_deps_map = {}
                        if 'global' in self.deps_maps:
                            for run_idx_i, params in \
                                self.deps_maps['global'].items():
                                merged_deps_map[run_idx_i] = params.copy()
                        if di in self.deps_maps:
                            for run_idx_i, params in self.deps_maps[di].items():
                                if run_idx_i not in merged_deps_map:
                                    merged_deps_map[run_idx_i] = {}
                                merged_deps_map[run_idx_i].update(params)
                        
                        if param_name in step_enforced:
                            required_run = step_enforced[param_name]
                        else:
                            required_run = get_most_recent_run(
                                param_name, merged_deps_map
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
                if step.func_type == 'global':
                    # Global: produces single value for all data
                    deps_map = self.deps_maps.get('global', {})
                    expected_run = get_most_recent_run(
                        return_name, deps_map
                        ) + 1
                    if expected_run == 0:
                        expected_run = 1
                    produced_params[return_name] = expected_run
                else:
                    # Per-row, vectorized, and global-res: all produce 
                    # per-row data
                    produced_params[return_name] = {}
                    if step.func_type == 'global-res':
                        # Global-res runs once but produces for all rows
                        if data_idx is None:
                            step_data_idx = range(self.nrows)  
                        else:
                            step_data_idx = data_idx
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
    
    def plot(self, data_idx, plot_type, title = None, **kwargs):
        """
        Generate a plot for the specified data index and plot type.
        
        Parameters:
            data_idx (int): The index of the data to plot.
            plot_type (str): The type of plot to generate.
            title (str, optional): The title of the plot. Defaults to None.
            **kwargs: Additional keyword arguments for the plot function.
        
        Returns:
        fig, axs: The figure and axes objects from the plot function.
        """
        ### Input validation
        if plot_type not in default_plot_funcs:
            raise ValueError(
                f"plot_type '{plot_type}' not recognized. Available types: "
                f"{list(default_plot_funcs.keys())}"
            )
        title, func, param_names, user_params = default_plot_funcs[plot_type]
        for kwarg in kwargs:
            if kwarg in user_params:
                user_params[kwarg] = kwargs[kwarg]
            
        ### Collect parameters
        params = [getattr(self, name)[data_idx] for name in param_names]
        fig, axs = func(*params, **user_params)
        if title is not None:
            fig.suptitle(title)

        return fig, axs
    
    def plot_full_cal(self, data_idx, **kwargs):
        """
        Generate a comprehensive set of plots for the specified data index,
        including raw data, gain fit, S21 removal, circle fit, sparper, and xcal
        plots. The resulting plots are combined into a single image for easy
        viewing.
        
        Parameters:
        data_idx (int): The index of the data to plot.
        **kwargs: Additional keyword arguments for the individual plot 
            functions.

        Returns:
        bytes: A PNG image in bytes format containing the combined plots.
        """
        plot_types = [
            'raw_data', 'gain_fit', 's21_rmv', 'circfit', 'sparper', 'xcal'
            ] 
        figs = []
        for plot_type in plot_types:
            try:
                fig, axs = self.plot(data_idx, plot_type, **kwargs)
                figs.append(fig)
            except Exception as e:
                msg = f"Error generating plot '{plot_type}' for data_idx "
                msg += f"{data_idx}: {e}"
                raise RuntimeError(msg) from e
        png_bytes = combine_figs_vert([
            combine_figs_horz(figs[:3]), 
            combine_figs_horz(figs[3:])
        ])
        return png_bytes

    ############################################################################
    ########################## Other utility methods ###########################
    ############################################################################
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

    spec = importlib.util.spec_from_file_location("custom_cal_steps", 
                                                    custom_path)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    custom_cal_steps = cs.custom_cal_steps
    
    return custom_cal_steps

################################################################################ 
########################### Zarr dependencies loader ###########################
################################################################################
def _load_existing_data_from_zarr(root, deps_maps, is_global_cache):
    """
    Load existing data from zarr into memory cache.
    
    Parameters:
    root (zarr.Group): Root zarr group
    deps_maps (dict): Dependencies maps loaded from zarr
    is_global_cache (dict): Global parameter cache
    
    Returns:
    dict: Memory cache with loaded data
    """
    memory_cache = {}
    
    # Iterate through all runs in zarr
    for run_str, run_grp in root.groups():
        run_idx = int(run_str.replace('run', ''))
        memory_cache[run_idx] = {}
        
        # Load each parameter in this run
        for param_name, param_grp in run_grp.groups():
            is_global = param_grp.attrs['global']
            data_array = param_grp['data']
            
            if is_global:
                # Global parameter - load directly
                memory_cache[run_idx][param_name] = data_array[()]
            else:
                # Per-row parameter - create LazyAttr with existing data
                row_exists = param_grp['row_exists'][:]
                data_shape = data_array.shape
                
                # Create LazyAttr and populate with existing data
                lazy_attr = pf.LazyAttr(data_shape[0], data_array.dtype)
                lazy_attr._shape = data_shape[1:] if len(data_shape) > 1 else ()
                
                # Load data for rows that exist
                for idx in range(len(row_exists)):
                    if row_exists[idx]:
                        lazy_attr._cache[idx] = data_array[idx]
                
                memory_cache[run_idx][param_name] = lazy_attr
    
    return memory_cache

def _load_deps_from_zarr(root):
    """
    Given the root of an analysis output file, load the dependencies map for 
    each data_idx and the is_global_cache while validating the file structure.

    Parameters:
    root (zarr group): The root of the zarr file to load from.

    Returns:
    tuple: (deps_maps, is_global_cache) where:
        - deps_maps (dict): Maps data indices to their dependencies maps. The 
          key-value map is: data_idx (int or 'global') -> run_idx (int) -> 
          parameter name (str) -> dependencies (dict), where the dependencies 
          dict represents parameter: run_idx pairs for each other parameter 
          that the parameter depends on. Global parameters use the key 'global' 
          instead of a data_idx.
        - is_global_cache (dict): Maps parameter names to boolean indicating 
          whether the parameter is global (True) or per-row (False).
    """
    ### Input validation 
    if not isinstance(root, zarr.core.group.Group):
        raise ValueError("Input root must be a zarr group.")
    if any(root.arrays()):
        raise ValueError("root cannot have arrays.")
    
    ### Parse root to create deps_maps, is_global_cache and validate file 
    ### structure.
    # deps_maps: data_idx (or 'global') -> run_idx -> param name -> dependencies 
    deps_maps = {} 
    # is_global_cache: param name -> bool (True if global, False if per-row)
    is_global_cache = {}
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
                
                # Update is_global_cache
                if name in is_global_cache and not is_global_cache[name]:
                    raise ValueError(
                        f"Parameter {name} found as both global and per-row "
                        f"in zarr file"
                    )
                is_global_cache[name] = True
                
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
                    
                    # Update is_global_cache
                    if name in is_global_cache and is_global_cache[name]:
                        raise ValueError(
                            f"Parameter {name} found as both global and per-row"
                            f" in zarr file"
                        )
                    is_global_cache[name] = False
    
    return deps_maps, is_global_cache

################################################################################ 
########################### YAML to dict conversion ############################
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

################################################################################
##################### Legacy zarr dep-index migration #########################
################################################################################
def fix_legacy_deps(DS):
    """
    Fix dep run indices stored in a zarr file that was created before
    _sync_to_zarr was introduced.

    Before _sync_to_zarr, write_data could persist dep entries that reference
    ephemeral, session-specific run indices (e.g. {zf_rmv: 16}) that can never
    be satisfied on a fresh load.  This function scans every dep entry stored
    in the zarr file and replaces any run index that does not exist in zarr
    with the canonical value that a fresh run would produce:

        max(1, zarr_max_for_dep_param + 1)

    Both the zarr attrs and the in-memory DS.deps_maps are updated so the
    DataSet instance stays coherent.

    Parameters:
    DS (DataSet): An open DataSet instance connected to the zarr file to fix.

    Returns:
    None  (mutates zarr attrs and DS.deps_maps in-place)
    """
    # ------------------------------------------------------------------
    # Build a lookup of the highest run index per param that is in zarr
    # ------------------------------------------------------------------
    zarr_max_per_param = {}
    for run_str, run_grp in DS.root.groups():
        r = int(run_str.replace('run', ''))
        for param_name, _ in run_grp.groups():
            prev = zarr_max_per_param.get(param_name, -1)
            zarr_max_per_param[param_name] = max(prev, r)

    # Precompute set of (run_str, param_name) pairs present in zarr so
    # _corrected never needs a zarr lookup in the hot path.
    zarr_present = set()
    for run_str, run_grp in DS.root.groups():
        for pname, _ in run_grp.groups():
            zarr_present.add((run_str, pname))

    def _corrected(dep_name, dep_run_idx):
        """Return the fixed run index, or dep_run_idx unchanged if valid."""
        if (f'run{dep_run_idx}', dep_name) in zarr_present:
            return dep_run_idx  # exists in zarr — valid reference
        # Stale: compute canonical next run (mirrors _sync_to_zarr logic)
        zarr_max = zarr_max_per_param.get(dep_name, -1)
        return max(1, zarr_max + 1)

    # ------------------------------------------------------------------
    # Walk every param in every run in zarr and patch stale dep entries
    # ------------------------------------------------------------------
    for run_str, run_grp in DS.root.groups():
        run_idx = int(run_str.replace('run', ''))
        # Note: zarr_present already populated above; no extra I/O needed here
        for param_name, param_grp in run_grp.groups():
            is_global_param = param_grp.attrs.get('global', False)
            deps_attr = dict(param_grp.attrs.get('deps', {}))
            changed = False

            if is_global_param:
                # Global deps format: {dep_name: dep_run_idx}
                for dep_name, dep_run_idx in list(deps_attr.items()):
                    fixed = _corrected(dep_name, dep_run_idx)
                    if fixed != dep_run_idx:
                        deps_attr[dep_name] = fixed
                        changed = True
                        # Mirror in memory
                        if ('global' in DS.deps_maps
                                and run_idx in DS.deps_maps['global']
                                and param_name in
                                    DS.deps_maps['global'][run_idx]):
                            DS.deps_maps['global'][run_idx][
                                param_name][dep_name] = fixed
            else:
                # Per-row deps format: {f'idx{di}': {dep_name: dep_run_idx}}
                for idx_str, row_deps in list(deps_attr.items()):
                    di = int(idx_str.replace('idx', ''))
                    for dep_name, dep_run_idx in list(row_deps.items()):
                        fixed = _corrected(dep_name, dep_run_idx)
                        if fixed != dep_run_idx:
                            deps_attr[idx_str][dep_name] = fixed
                            changed = True
                            # Mirror in memory
                            if (di in DS.deps_maps
                                    and run_idx in DS.deps_maps[di]
                                    and param_name in
                                        DS.deps_maps[di][run_idx]):
                                DS.deps_maps[di][run_idx][
                                    param_name][dep_name] = fixed

            if changed:
                param_grp.attrs['deps'] = deps_attr

################################################################################
# Analysis helpers -> in progress
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
