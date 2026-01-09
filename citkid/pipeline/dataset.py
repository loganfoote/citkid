import os
import yaml
import importlib.util 
import numpy as np
import zarr
from . import framework as pf

class LazyZarrArray:
    def __init__(self, data, exists):
        self.data = data          # zarr.Array
        self.exists = exists      # zarr.Array (bool)

    def __getitem__(self, key):
        if self.exists[key].all():
            return self.data[key]
        return None
    
class DataSet:
    def __init__(self, directory, yaml_path, zarr_path):
        """
        Initialize the dataset with a calibration pipeline defined by a YAML file.  
        
        Parameters:
        directory (str): The base directory for the dataset.
        zarr_path (str): The path to the zarr file containing the analysis 
            outputs.
        yaml_path (str): The path to the YAML configuration file.
        """
        # Normalize paths 
        self.directory = os.path.normpath(directory)
        self.zarr_path = os.path.normpath(zarr_path)
        self.root = zarr.open_group(self.zarr_path, mode = 'a')
        self.yaml_path = os.path.normpath(yaml_path)

        # Load steps from custom_steps.py if it exists
        self.steps = self._load_custom_steps()
        # Add default calibration steps if not already present
        for step in pf.default_cal_steps:
            if step.name not in [s.name for s in self.steps]:
                self.steps.append(step)

        # Check that we can load nres (number of tones in the data set).
        steps_returning_nres = []
        for step in self.steps:
            if 'nres' in step.return_names:
                steps_returning_nres.append(step)
        
        # Check that there is only one step that returns nres.
        if len(steps_returning_nres) != 1:
            m = "There must be exactly one plStep object in the "
            m += "custom_cal_steps list within custom_steps.py "
            m += "which returns a parameter named 'nres'. "
            m += f"{len(steps_returning_nres)} such plStep objects "
            m += "were provided."
            raise ValueError(m)
        
        # Check that nres is the only returned name.
        step = steps_returning_nres[0]
        if step.return_names != ['nres']:
            m = f"The function named '{step.name}' in "
            m += "custom_steps.py must only return 'nres'."
            raise ValueError(m)
        
        # Check that the step returning nres has func_type = 'global'.
        if step.func_type != 'global':
            m = f"The function named '{step.name}' in "
            m += "custom_steps.py must have return_type = 'global'."
            raise ValueError(m)
        
        # Load nres, and check that it is integer-valued and > 0.
        step.run(self)
        if not (type(self.nres) is int and self.nres > 0):
            m = "The return parameter 'nres' from the step named "
            m += f"'{step.name}' in custom_steps.py must be "
            m += "integer-valued and > 0."
            raise ValueError(m)
                
        # Load YAML and convert to calibration pipeline
        yaml_dict = self._load_yaml()
        self.cal_pl = self._convert_yaml_to_steps(yaml_dict)
        

    def _load_custom_steps(self):
        """
        Load custom steps from 'custom_steps.py' in the dataset directory.

        Returns:
        list: A list of custom plStep objects.
        """
        custom_module_path = os.path.join(self.directory, 'custom_steps.py')
        if not os.path.exists(custom_module_path):
            return []

        spec = importlib.util.spec_from_file_location("custom_cal_steps", 
                                                      custom_module_path)
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
        Confirm that a list of plStep objects forms a valid path, where all 
        inputs exist or can be generated from the Zarr file or from the path.

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
        step (plStep or None): Step at which there was a
            missing input name, or None if there were not missing
            input names at any step.
        """
        # Need to modify this to check if values are in zarr
        valid_inputs = [d for d in dir(self) if '__' not in d]
        for step in path:
            for inp in step.param_names:
                if inp not in valid_inputs and inp != 'data_idx':
                    
                    run = self.get_attr_version(inp)
                    
                    if run is None:
                        if raise_error:
                            m = f"Invalid path, input '{inp}' for step '{step.name}'"
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
        run_idx (int): run index, or None to find most recent run index
            that produced the desired data attribute.

        Returns:
        np.ndarray or scalar: The requested data attribute value(s).
        """
        data_idx = np.atleast_1d(data_idx)
        if run_idx is None:
            run_idx = self.get_attr_version(name)
        grp = self.root[str(run_idx)]
        return grp[name].oindex[data_idx]
    
    def write_data(self, name, value, data_idx, run_idx, dtype = None):
        """
        Write data attribute 'name' for dataset.

        Parameters:
        name (str): The name of the data attribute to write.
        value (np.ndarray or scalar): The data to write.
        data_idx (int or list): The data index or indices to write.
        run_idx (int): run index.
        dtype (np.dtype, optional): The data type to use when writing. Defaults 
            to None (use value's dtype).
        """
        # Convert data_idx and value to numpy arrays
        data_idx = np.atleast_1d(data_idx)
        value = np.atleast_1d(value)
            
        # Check if the length of value equals the length of data_idx.
        if data_idx.shape[0] != value.shape[0]:
            raise ValueError('value and data_idx must have the same length.')
        
        # ensure run group exists (create if missing)
        key = str(run_idx)
        grp = self.root.require_group(key)

        # set dtype if not provided
        if dtype is None:
            dtype = value.dtype

        # Get the per-element shape of the new value to be added.
        element_shape = value.shape[1:]

        # ensure dataset exists (create if missing)
        # dataset shape: row index at the start + all value dims
        if name not in grp:
            shape = (0, *element_shape)
            chunks = (1, *element_shape)  # first axis = 1 for single-row writes
            values_arr = grp.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
            data_idxs = grp.create_array(f"{name}_idx", shape=(0,), dtype=int)
            expected_shape = element_shape
        else:
            values_arr = grp[name]
            data_idxs = grp[f"{name}_idx"]
            expected_shape = values_arr.shape[1:]

        # Check that shape of the elements of value match the
        # shape of the elements of values_arr.
        if element_shape != expected_shape:
            raise ValueError(f"The shape of 'value' must match the shape of the existing zarr array, {name}.")

        # write data at specified indices
        n = values_arr.shape[0]
        n_add = data_idx.shape[0]
        n_new = n + n_add
        new_shape = (n_new, *expected_shape)
        values_arr.resize(new_shape)
        data_idxs.resize((n_new))
        values_arr[-n_add:, ...] = value
        data_idxs[-n_add:] = data_idx
                
        
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
    
    def __getattr__(self, name):
        """
        Custom attribute getter to handle LazyAttr creation for per-row
        attributes.

        Parameters:
        name (str): The name of the attribute to get.

        Returns:
        Any: The requested attribute value or LazyAttr.
        """
        # run = 0
        # grp = self.root[f'run{run:d}']

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

