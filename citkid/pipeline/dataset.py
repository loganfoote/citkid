import os
import yaml
import importlib.util 
import numpy as np
import zarr
import copy

from .dependencies import get_most_recent_run, get_deps
from . import framework as pf
from ..zarr_util import deep_union

class DataSet:
    def __init__(self, custom_path, zarr_path, cal_yaml_path, analysis_yaml_path=None):
        """
        Initialize the dataset with a calibration pipeline defined by a YAML 
        file.  
        
        Parameters:
        custom_path (str): Directory to the .py file containing custom 
            calibration functions.
        zarr_path (str): The path to the zarr file containing the analysis 
            outputs.
        cal_yaml_path (str): The path to the YAML configuration file.
        analysis_yaml_path (str): The path to the YAML analysis file.
        """
        # Normalize paths 
        if custom_path is not None:
            self.custom_path = os.path.normpath(custom_path)
        else:
            self.custom_path = None
        self.zarr_path = os.path.normpath(zarr_path)
        self.root = zarr.open_group(self.zarr_path, mode = 'a')
        self.cal_yaml_path = os.path.normpath(cal_yaml_path)
        self.analysis_yaml_path = analysis_yaml_path
        self.global_cache = {}
        self.deps_map = {'global': {}}

        # Input validation 
        if self.custom_path is not None and \
            not self.custom_path.endswith('.py'):
            raise ValueError("custom_path must point to a .py file.")
        if not self.zarr_path.endswith('.zarr'):
            raise ValueError("zarr_path must point to a .zarr file.")
        is_yaml = self.cal_yaml_path.endswith('.yaml') 
        is_yaml = is_yaml or self.cal_yaml_path.endswith('.yml')
        if not is_yaml:
            raise ValueError("cal_yaml_path must point to a .yaml or .yml file.")

        # Load steps from custom_steps.py if it exists
        self.cal_steps, self.analysis_steps = self._load_custom_steps()
        # Add default calibration steps if not already present
        for step in pf.default_cal_steps:
            if step.name not in [s.name for s in self.cal_steps]:
                self.cal_steps.append(step)
        # Add default analysis steps if not already present
        for step in pf.default_analysis_steps:
            if step.name not in [s.name for s in self.analysis_steps]:
                self.analysis_steps.append(step)

        # hard-coded nres for now
        self.nres = 1600
                
        # Load YAML and convert to calibration pipeline
        yaml_dict = self._load_yaml(self.cal_yaml_path)
        self.cal_pl = self._convert_yaml_to_steps(yaml_dict)

        # confirm that the cal_pl structure is valid
        pf.check_pl_tree_structure(self.cal_pl) 
        
        if self.analysis_yaml_path is not None:
            self.analysis_yaml_path = os.path.normpath(analysis_yaml_path)
            
            is_yaml = self.analysis_yaml_path.endswith('.yaml')
            is_yaml = is_yaml or self.analysis_yaml_path.endswith('.yml')
            if not is_yaml:
                raise ValueError("analysis_yaml_path must point to a .yaml or .yml file.")
            
            # Load analysis YAML and user-specified analysis parameters
            yaml_dict = self._load_yaml(self.analysis_yaml_path)
            yaml_dict = yaml_dict['ANALYSIS_STEPS']
            self.analysis_params = {}
            step_names = np.array([s.name for s in self.analysis_steps])
            for step_dict in yaml_dict.values():
                name = step_dict['task']
                ixs = np.where(step_names == name)[0]
                if len(ixs):
                    step = self.analysis_steps[ixs[0]]
                    if 'params' in step_dict.keys():
                        params_dict = step_dict['params']
                        for param_name, param_val in params_dict.items():
                            if param_name in step.param_names:
                                if param_val == 'None':
                                    param_val = None
                                self.analysis_params[param_name] = param_val
                                self.global_cache[param_name] = True
                                if 1 not in self.deps_map['global']:
                                    self.deps_map['global'][1] = {}
                                self.deps_map['global'][1][param_name] = {}
        
    def _load_custom_steps(self):
        """
        Load custom steps from 'custom_steps.py' in the dataset directory.

        Returns:
        list: A list of custom plStep objects.
        """
        if self.custom_path is None:
            return []
        
        if not os.path.exists(self.custom_path):
            m = f"Custom path '{self.custom_path}' does not exist."
            raise FileNotFoundError(m)

        spec = importlib.util.spec_from_file_location("custom_cal_steps", 
                                                      self.custom_path)
        cs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cs)
        custom_cal_steps = cs.custom_cal_steps
        
        spec = importlib.util.spec_from_file_location("custom_analysis_steps", 
                                                      self.custom_path)
        cs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cs)
        custom_analysis_steps = cs.custom_analysis_steps
        
        return custom_cal_steps, custom_analysis_steps
    
    def _load_yaml(self, path):
        """
        Load the YAML configuration file. 

        Parameters:
        path: the path to the yaml file.

        Returns:
        dict: The loaded YAML configuration as a dictionary.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)        
            
    def _convert_yaml_to_steps(self, pl_dict, key = None):
        """
        Converts YAML-defined path dictionary leaves to plStep objects.

        Parameters:
        pl_dict (dict or str): The YAML-defined path dictionary or leaf.
        key (str): The key associated with the current pl_dict, used to identify
            task names.

        Returns:
        dict or plStep: The converted paths with plStep objects.
        """
        if isinstance(pl_dict, dict):
            for key, val in pl_dict.items():
                pl_dict[key] = self._convert_yaml_to_steps(val, key)
        if isinstance(pl_dict, str) and key == 'task':
            x = [d for d in self.cal_steps if d.name == pl_dict]
            if not len(x):
                m = f"Step '{pl_dict}' not found in available steps."
                raise ValueError(m)
            return x[0]
        return pl_dict
        
    def confirm_valid_path(self, path, raise_error = True):
        """
        Confirm that a list of plStep objects forms a valid path.

        All inputs must exist or be generatable from the Zarr file or path.

        Parameters:
        path (list): List of plStep objects forming the path.
        raise_error (bool): Set to True to raise an error message,
            False to return the input which was not found and
            the step for which it was not found.

        Raises (if raise_error = True):
        ValueError: If an input for any step is not found
            and raise_error = True.
            
        Returns (if raise_error = False):
        missing_input (str or None): Missing input name, or None
            if there were no missing input names.
        step (plStep or None): Step where input was missing, or None.

        Returns:
        None
        """
        # Need to modify this to check if values are in zarr
        valid_inputs = [d for d in dir(self) if '__' not in d]
        for step in path:
            for inp in step.param_names:
                if inp not in valid_inputs and inp != 'data_idx':
                    
                    run = self.get_attr_version(inp)
                    
                    if run is None:
                        if raise_error:
                            m = (
                                f"Invalid path, input '{inp}' for step "
                                f"'{step.name}'"
                            )
                            m += f" not found."
                            raise ValueError(m)
                        else:
                            missing_input = inp
                            return missing_input, step
                            
            valid_inputs.extend(step.return_names)
            
        if not raise_error:
            missing_input = None
            step = None
            return missing_input, step

    def execute_path(self, path, data_idx):
        """
        Execute a list of plStep objects in sequence for given data indices.

        Parameters:
        path (list): List of plStep objects forming the path.
        data_idx (int or list): The data index or indices to process.

        Returns:
        None
        """
        self.confirm_valid_path(path)
        for step in path:
            this_data_idx = data_idx
            if step.func_type in ['global', 'global-res']:
                this_data_idx = None
            step.run(self, this_data_idx)
        

    def read_data(self, name, data_idx, run_idx = None):
        """
        Read data attribute 'name' for data indices 'data_idx' from dataset.

        Parameters:
        name (str): The name of the data attribute to read.
        data_idx (int or list): The data index or indices to read.
        run_idx (int or None): run index, or None for most recent run.
            that produced the desired data attribute.

        Returns:
        np.ndarray or scalar: The requested data attribute value(s).
        """
        data_idx = np.atleast_1d(data_idx)
        if run_idx is None:
            run_idx = self.get_attr_version(name)
            # If the run index was not specified, and the
            # data does not exist in any run, raise an error.
            if run_idx is None:
                m = f"'{name}' was not found under any run index "
                m += "in the zarr file."
                raise ValueError(m)
        else:
            # If the data does not exist in the user-specified
            # run index, raise an error.
            if not self.run_exists(name, run_idx):
                m = f"'{name}' was not found under run index {run_idx} "
                m += "in the zarr file."
                raise ValueError(m)
                
        grp = self.root[str(run_idx)]
        return grp[name].oindex[data_idx]
        
    
    def write_data(self, return_name, step, value, data_idx, dtype = None):
        """
        Write data attribute 'name' for dataset.

        Parameters:
        return_name (str): The name of the data attribute to write.
        func_type (str): The type of function that produced the data.
            Can be "per-row", "vectorized", or "global-res".
        param_names (array-like): Parameter names used when calling the function.
        dependencies (dictionary): Dictionary giving the dependencies for the input
            parameters, where keys are parameter names and values are run indices.
        value (np.ndarray or scalar): The data to write.
        data_idx (int or list): The data index or indices to write.
        run_idx (int): run index that produced the output.
        dtype (np.dtype, optional): The data type to use when writing. Defaults 
            to None (use value's dtype).

        Returns:
        None
        """
        # set dtype if not provided
        if dtype is None:
            dtype = value[data_idx].dtype
            
        root = self.root
        
        if step.func_type in ['global', 'global-res']:
            
            deps_map = self.deps_map['global']
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
            
            # If the func_type is global, we can't index it by row with data_idx.
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
            # Store parameter dependencies and 'global(-res)' status.
            return_grp.attrs['deps'] = deps
            return_grp.attrs['global'] = self.global_cache[return_name]
            
        elif step.func_type in ['per-row', 'vectorized']:
            
            for local_idx, di in enumerate(data_idx):
                di = di.item()
                di_str = f'idx{di}'
                deps_map = self.deps_map[di_str]
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
                    shape = (self.nres, *value.shape[1:])
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
                        data = np.full(self.nres, False)
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
                return_grp.attrs['global'] = self.global_cache[return_name]
                
        
    def get_attr_version(self, name):
        """
        Finds the most recent run version of a given attribute.
        Returns None if the attribute does not exist in any run.
        
        Parameters:
        name (str): Name of the attribute to search for.
        
        Returns:
        attr_version (int): Most recent run version containing the attribute.
        """
        folders = list(self.root.keys())
        runs = []
        for folder in folders:
            try:
                int(folder)
                runs.append(folder)
            except ValueError:
                pass
        runs = np.array(runs, dtype=int)
        runs = np.flip(np.sort(runs))
        
        attr_version = None
        for run in runs:
            grp = self.root[str(run)]
            attrs = list(grp.keys())
            if name in attrs:
                attr_version = run
                return attr_version
    
    def run_exists(self, name, run):
        """
        Returns True if data under "name" exists for a specified run,
        otherwise return False.
        
        Parameters:
        name (str): Name of the attribute to search for.
        run (int): run index.
        """
        data_exists = False
        try:
            self.root[f'{run}/{name}']
            data_exists = True
        except:
            pass
        return data_exists
    

    # ***EK - The zarr loading is causing inconsistent behavior where
    # the outputs of a plStep.run call will be stored as a LazyAttr,
    # but loading the same result from the zarr file is just 
    # stored as a zarr file.
    def __getattr__(self, name):
        """
        Custom attribute getter to handle LazyAttr creation for per-row
        attributes.

        Parameters:
        name (str): The name of the attribute to get.

        Returns:
        Any: The requested attribute value or LazyAttr.
        """
        # Look up the attribute in the zarr file.
        run_idx = self.get_attr_version(name)
        if run_idx is not None:
            grp = self.root[f'{run_idx}/{name}']
            attr = grp['data']
            object.__setattr__(self, name, attr)
            self.global_cache[name] = grp.attrs['global']
            if self.global_cache[name]:
                deps_to_add = grp.attrs['deps']
                self.update_deps_map_after_load(name, deps_to_add, di=None)
                
            else:
                row_exists = grp['row_exists'][...]
                dis = np.where(row_exists)[0]
                for di in dis:
                    deps_to_add = grp.attrs['deps'][f'idx{di}']
                    self.update_deps_map_after_load(name, deps_to_add, di=di)
                
            return attr

        if name in self.analysis_params.keys():
            return self.analysis_params[name]

        # Only run when normal lookup fails
        cal_pl = object.__getattribute__(self, "cal_pl")

        # Find the path to produce this attribute
        path = pf.find_pl_path(cal_pl, name)
        if path is None:
            raise AttributeError(name)

        # Check if all steps are global or global-res 
        if all(step.func_type in ["global", "global-res"] for step in path):
            # Execute immediately, store result directly
            self.execute_path(path, data_idx = None)
            return object.__getattribute__(self, name)
        
        # Otherwise create LazyAttr for per-row/vectorized output
        attr = pf.LazyAttr(self, name)
        object.__setattr__(self, name, attr)
        self.global_cache[name] = False
        
        attr = object.__getattribute__(self, name)
        
        return attr
            
    def has_attr(obj, name):
        """
        Function to check if an attribute or method is present, without
        calling __getattr__.
        """
        # 1. Instance attributes
        if name in obj.__dict__:
            return True

        # 2. Class + base classes (methods live here)
        for cls in type(obj).__mro__:
            if name in cls.__dict__:
                return True

        return False

    
    def _validate_nres(self):
        path = pf.find_pl_path(self.cal_pl, 'nres')
        step = path[-1] 

        # Check that nres is the only returned name.
        if len(step.return_names) != 1:
            m = f"The function named '{step.name}' in "
            m += "custom_steps.py must only return 'nres'."
            raise ValueError(m)
        
        # Check that the step returning nres has func_type = 'global'.
        if step.func_type != 'global':
            m = f"The function named '{step.name}' in "
            m += "custom_steps.py must have return_type = 'global'."
            raise ValueError(m)
        
        # Load nres, and check that it is integer-valued and > 0.
        if not (type(self.nres) is int and self.nres > 0):
            m = "The return parameter 'nres' from the step named "
            m += f"'{step.name}' in custom_steps.py must be "
            m += "integer-valued and > 0."
            raise ValueError(m)

    def update_deps_map_after_run(self, param_names, return_names, data_idx):
        """
        Updates the dependencies map after running a plStep.
        
        Parameters:
        param_names (array-like (str)): The list of input parameter names
            of the plStep: plStep.param_names
        return_names (array-like (str)): The list of return parameter names
            of the plStep: plStep.return_names
        data_idx: The data indices that the plStep was run on, or None if
            the function was 'global'-type.
            
        Returns:
        None
        """
        def update_deps(param_names, return_names, deps_map):
            """
            Updates individual leaves of the deps_map corresponding
            to each return name in return_names.
            """
            param_names = [param_name for param_name in param_names
                           if param_name != 'data_idx']
            deps_to_add = get_deps(param_names, deps_map)
            for name in return_names:
                # This conditional is needed to distinguish between
                # steps where an input parameter can change, so that 
                # the run_idx can increase, and those where it can't,
                # so that it should always be run_idx = 1.
                if param_names:
                    run_idx = get_most_recent_run(name, deps_map)
                    run_idx = max(run_idx+1, 1)
                else:
                    run_idx = 1
                if run_idx not in deps_map:
                    deps_map[run_idx] = {}
                deps_map[run_idx][name] = {}
                for dep_name, dep_run_idx in deps_to_add.items():
                    if dep_name != name:
                        deps_map[run_idx][name][dep_name] = dep_run_idx
        
        if data_idx is None: # "global" case
            if 'global' not in self.deps_map:
                self.deps_map['global'] = {}
            deps_map = self.deps_map['global']
            update_deps(param_names, return_names, deps_map)

        else: # "non-global" case
            # Load the global deps_map so we can copy it over to 
            # the deps_maps for each data index.
            deps_map_global = {}
            if 'global' in self.deps_map:
                deps_map_global = copy.deepcopy(self.deps_map['global'])
            for di in data_idx:
                di_str = f'idx{di}'
                if di_str not in self.deps_map:
                    self.deps_map[di_str] = {}
                deps_map = self.deps_map[di_str]
                # Unite global deps_map with the per-data_index deps_map
                self.deps_map[di_str] = deep_union(deps_map, deps_map_global)
                deps_map = self.deps_map[di_str]
                
                update_deps(param_names, return_names, deps_map)
                       
    def update_deps_map_after_load(self, name, deps_to_add, di=None):
        """
        Updates the dependencies map after loading data from the zarr file.
        
        Parameters:
        name (str): The name of the analysis output.
        deps_to_add (dictionary): Keys are return names, and values are run indices.
            'name' should be included in the list of keys.
        di (int or None): The data index of the output. If None, the 
            output is of 'global' type and has no data index.
        
        Returns:
        None
        """
        # Get the branch of self.deps_map that corresponds to 'global' attributes,
        # or to the specified data index.
        if di is None:
            if 'global' not in self.deps_map:
                self.deps_map['global'] = {}
            deps_map = self.deps_map['global']
        else:
            di_str = f'idx{di}'
            if di_str not in self.deps_map:
                self.deps_map[di_str] = {}
            deps_map = self.deps_map[di_str]
        
        # Initialize the dependencies leaf for this output name.
        run_idx = deps_to_add[name]
        if run_idx not in deps_map:
            deps_map[run_idx] = {}
        deps_map[run_idx][name] = {}
        
        # Add the dependencies to this leaf.
        for key, value in deps_to_add.items():
            if key != name:
                deps_map[run_idx][name][key] = value
                
                
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