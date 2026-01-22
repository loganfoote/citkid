import os
import yaml
import importlib.util 
import numpy as np
import zarr
from .dependencies import get_most_recent_run, get_dependencies
from . import framework as pf
    
class DataSet:
    def __init__(self, custom_path, yaml_path, zarr_path):
        """
        Initialize the dataset with a calibration pipeline defined by a YAML 
        file.  
        
        Parameters:
        custom_path (str): Directory to the .py file containing custom 
            calibration functions.
        zarr_path (str): The path to the zarr file containing the analysis 
            outputs.
        yaml_path (str): The path to the YAML configuration file.
        """
        # Normalize paths 
        if custom_path is not None:
            self.custom_path = os.path.normpath(custom_path)
        else:
            self.custom_path = None
        self.zarr_path = os.path.normpath(zarr_path)
        self.root = zarr.open_group(self.zarr_path, mode = 'a')
        self.yaml_path = os.path.normpath(yaml_path)

        # Input validation 
        if self.custom_path is not None and \
            not self.custom_path.endswith('.py'):
            raise ValueError("custom_path must point to a .py file.")
        if not self.zarr_path.endswith('.zarr'):
            raise ValueError("zarr_path must point to a .zarr file.")
        is_yaml = self.yaml_path.endswith('.yaml') 
        is_yaml = is_yaml or self.yaml_path.endswith('.yml')
        if not is_yaml:
            raise ValueError("yaml_path must point to a .yaml or .yml file.")

        # Load steps from custom_steps.py if it exists
        self.steps = self._load_custom_steps()
        # Add default calibration steps if not already present
        for step in pf.default_cal_steps:
            if step.name not in [s.name for s in self.steps]:
                self.steps.append(step)

        # hard-coded nres for now
        self.nres = 1600
                
        # Load YAML and convert to calibration pipeline
        yaml_dict = self._load_yaml()
        self.cal_pl = self._convert_yaml_to_steps(yaml_dict)

        # confirm that the cal_plstructure is valid
        pf.check_pl_tree_structure(self.cal_pl) 
        
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
        return cs.custom_cal_steps
    
    def _load_yaml(self):
        """
        Load the YAML configuration file. 

        Returns:
        dict: The loaded YAML configuration as a dictionary.
        """
        with open(self.yaml_path, 'r') as f:
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
            x = [d for d in self.steps if d.name == pl_dict]
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
            step.run(self, data_idx)
        

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
            dtype = value.dtype
            
        root = self.root

        if step.func_type in ['global', 'global-res']:
            
            if 'saved-global' in root.attrs:
                saved_global = root.attrs['saved-global']
                run_idx = get_most_recent_run(return_name, saved_global) + 1
                run_idx = max(run_idx, 1)
            else:
                self.root.attrs['saved-global'] = {0: {}, 1: {return_name: {}}}
                saved_global = root.attrs['saved-global']
                for param_name in step.param_names:
                    saved_global[0][param_name] = {}
                run_idx = 1
            
            if str(run_idx) not in root:
                root.create_group(str(run_idx))
            grp = root[str(run_idx)]
                
            shape = value.shape
            chunks = value.shape
            
            values_arr = grp.create_array(
                name = return_name,
                shape = shape,
                dtype = dtype,
                chunks = chunks,
            )
            
            values_arr[...] = value
            
            dependencies = get_dependencies(step.param_names, saved_global)
            
            for param_name, param_run_idx in dependencies.items():
                saved_global[run_idx][return_name][param_name] = param_run_idx
                
            if 'output_type' not in root.attrs:
                root.attrs['output_type'] = {}
            output_type = root.attrs['output_type']
            if run_idx not in root.attrs['output_type']:
                output_type[run_idx] = {}
            output_type[run_idx][return_name] = 'global'
            root.attrs['output_type'] = output_type
            
        elif step.func_type in ['per-row', 'vectorized']:
            
            if 'saved-global' in root.attrs:
                saved_start = root.attrs['saved-global'].copy()
            else:
                saved_start = {0: {}}
                for param_name in step.param_names:
                    saved_start[0][param_name] = {}
            
            if 'saved' not in root.attrs:
                root.attrs['saved'] = {}
                root.attrs['saved_idx'] = '[]' 
                            
            saved = root.attrs['saved']
            saved_idx = np.fromstring(root.attrs['saved_idx'].strip('[]'), dtype=int, sep=' ')
                
            for local_idx, idx in enumerate(data_idx):
                local_saved_idxs = np.where(saved_idx == idx)[0]
                if local_saved_idxs.shape != (0,):
                    local_saved_idx = local_saved_idxs[0]
                else:
                    saved[idx] = saved_start
                    local_saved_idx = len(saved) - 1
                    saved_idx.append(idx)
                    root.attrs['saved_idx'] = str(saved_idx)

                this_saved = saved[local_saved_idx]
                dependencies = get_dependencies(step.param_names, this_saved)
                run_idx = get_most_recent_run(return_name, this_saved) + 1
                run_idx = max(run_idx, 1)
                    
                if str(run_idx) not in root:
                    root.create_group(str(run_idx))
                grp = root[str(run_idx)]
                
                this_value = value[local_idx]
                
                if return_name not in grp:
                    
                    shape = (1, *this_value.shape)
                    chunks = (1, *this_value.shape)
                    
                    values_arr = grp.create_array(
                        name = return_name,
                        shape = shape,
                        dtype = dtype,
                        chunks = chunks,
                    )
                    values_arr[...] =  [this_value]
                    
                    data_idx_arr = grp.create_array(
                        name = f'{return_name}_idx',
                        dtype=int,
                        data = [idx]
                    )
                    
                    this_saved[run_idx] = {}
                    
                else:
                    
                    values_arr = grp[return_name]
                    data_idx_arr = grp[f'{return_name}_idx']
                    n = values_arr.shape[0]
                    data_idx_arr.resize((n+1))
                    data_idx_arr[-1] = data_idx[local_idx]
                    new_shape = (n+1, *values_arr.shape[1:])
                    values_arr.resize(new_shape)
                    values_arr[-1, ...] = this_value
                    
                for param_name, param_run_idx in dependencies.items():
                    saved[run_idx][return_name][param_name] = param_run_idx
                    
                if 'output_type' not in root.attrs:
                    root.attrs['output_type'] = {}
                output_type = root.attrs['output_type']
                if run_idx not in root.attrs['output_type']:
                    output_type[run_idx] = {}
                output_type[run_idx][return_name] = 'indexable'
                root.attrs['output_type'] = output_type

        
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
            attr = pf.LazyAttr(self, name)
            object.__setattr__(self, name, attr)
            return attr

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
        
        attr = object.__getattribute__(self, name)
        
        return attr
    
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
