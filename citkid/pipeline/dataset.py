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

        # Load YAML and convert to calibration pipeline
        yaml_dict = self._load_yaml()
        self.cal_pl = self._convert_yaml_to_steps(yaml_dict)
        self.cal_pl_list = _convert_dict_to_list(self.cal_pl)

        # Set nres 
        # self.nres = len(self.res_idxs) # must be able to produce from pipeline
        self.nres = 1600  # temporary hardcode until pipeline can produce res_idxs

    def _load_custom_steps(self):
        """
        Load custom steps from 'custom_steps.py' in the dataset directory.

        Returns:
        list: A list of custom plStep objects.
        """
        custom_module_path = os.path.join(self.directory, 'custom_steps.py')
        if not os.path.exists(custom_module_path):
            return []

        spec = importlib.util.spec_from_file_location("custom_steps", 
                                                      custom_module_path)
        cs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cs)
        return cs.custom_steps
    
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
    
    def confirm_valid_path(self, path):
        """
        Confirm that a list of plStep objects forms a valid path, where all 
        inputs exist or can be generated from the Zarr file or from the path.

        Parameters:
        path (list): List of plStep objects forming the path.

        Raises:
        ValueError: If an input for any step is not found.
        """
        # Need to modify this to check if values are in zarr
        valid_inputs = [d for d in dir(self) if '__' not in d]
        for step in path:
            for inp in step.param_names:
                if inp not in valid_inputs and inp != 'data_idx':
                    m = f"Invalid path, input '{inp}' for step '{step.name}'"
                    m += f" not found."
                    raise ValueError(m)
            valid_inputs.extend(step.return_names)

    def execute_path(self, path, data_idx):
        """
        Execute a list of plStep objects in sequence for given data indices.

        Parameters:
        path (list): List of plStep objects forming the path.
        data_idx (int or list): The data index or indices to process.
        """
        self.confirm_valid_path(path)
        for step in path:
            print(step)
            step.run(self, data_idx)

    def read_data(self, name, data_idx, run):
        """
        Read data attribute 'name' for data indices 'data_idx' from dataset.

        Parameters:
        name (str): The name of the data attribute to read.
        data_idx (int or list): The data index or indices to read.
        run (int): run index (placeholder for future use).

        Returns:
        np.ndarray or scalar: The requested data attribute value(s).
        """
        attr_version = self.get_attr_version(name)
        grp = self.root[str(attr_version)]
        return grp[name][:, data_idx]
    
    def write_data(self, name, value, data_idx, run, dtype = None):
        """
        Write data attribute 'name' for dataset.

        Parameters:
        name (str): The name of the data attribute to write.
        value (np.ndarray or scalar): The data to write.
        data_idx (int or list): The data index or indices to write.
        run (int): run index (placeholder for future use).
        dtype (np.dtype, optional): The data type to use when writing. Defaults 
            to None (use value's dtype).
        """
        # ensure run group exists (create if missing)
        key = str(run)
        grp = self.root.require_group(key)

        # scalar → wrap as array
        if not hasattr(value, 'shape'):
            value = np.array([value])

        # set dtype if not provided
        if dtype is None:
            dtype = value.dtype

        # ensure dataset exists (create if missing)
        # dataset shape: all value dims + row index at the end
        shape = (*value.shape, 0)
        chunks = (*value.shape, 1)  # last axis = 1 for single-row writes
        grp.require_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
        exists = grp.require_dataset(f"{name}_exists", shape=shape, dtype=bool,
                                     chunks=chunks, fill_value=False)

        # write data at specified indices
        grp[name][..., data_idx] = value
        grp[f"{name}_exists"][..., data_idx] = True
        
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
        
        # attr = object.__getattribute__(self, name)
        
        return attr

def _convert_dict_to_list(pl_dict, key = None):
        """
        Converts a path dictionary of plStep objects to a 1-D list.
        
        Parameters:
        pl_dict (dict or plStep): The path dictionary or plStep object.
        key (str): The key associated with the current pl_dict, used to identify
            task names.

        Returns:
        list: The list of plStep objects.
        """
        pl_list = []
        
        if isinstance(pl_dict, dict):
            for key, val in pl_dict.items():
                pl_list.extend(_convert_dict_to_list(val, key))
        else:#if isinstance(pl_dict, pf.plStep) and key == 'task':
            pl_list = [pl_dict]
        return pl_list