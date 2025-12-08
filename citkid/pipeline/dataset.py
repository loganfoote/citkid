
import os
import yaml
import importlib.util 
from .framework import default_cal_steps, find_pl_path, LazyAttr

class DataSet:
    def __init__(self, directory, yaml_path):
        """
        Initialize the dataset with a calibration pipeline defined by a YAML file.  
        
        Parameters:
        directory (str): The base directory for the dataset.
        yaml_path (str): The path to the YAML configuration file.
        """
        # On import of yaml, check that all paths are valid
        self.directory = os.path.normpath(directory)
        self.yaml_path = os.path.normpath(yaml_path)

        # Create list of possible steps
        custom_module_path = os.path.join(self.directory, 'custom_steps.py')
        if os.path.exists(custom_module_path):
            spec = importlib.util.spec_from_file_location("custom_steps", 
                                                          custom_module_path)
            cs = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cs)
            custom_steps = cs.custom_steps
        else:
            custom_steps = []

        self.steps = custom_steps 
        for step in default_cal_steps: 
            if step.name not in [s.name for s in self.steps]:
                self.steps.append(step) 
        
        # Load YAML configuration
        with open(self.yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        self.cal_pl = self.convert_yaml_to_steps(yaml_dict)

        # Set nres 
        # self.nres = len(self.res_idxs) # must be able to produce from pipeline
        self.nres = 1600  # temporary hardcode until pipeline can produce res_idxs
            
    def convert_yaml_to_steps(self, pl_dict, key = None):
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
                pl_dict[key] = self.convert_yaml_to_steps(val, key)
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
            step.run(self, data_idx)

    def __getattr__(self, name):
        """
        Custom attribute getter to handle LazyAttr creation for per-row
        attributes.

        Parameters:
        name (str): The name of the attribute to get.

        Returns:
        Any: The requested attribute value or LazyAttr.
        """
        # Only run when normal lookup fails
        cal_pl = object.__getattribute__(self, "cal_pl")

        # Find the path to produce this attribute
        path = find_pl_path(cal_pl, name)
        if path is None:
            raise AttributeError(name)

        # Check if all steps are global or global-res 
        if all(step.func_type in ["global", "global-res"] for step in path):
            # Execute immediately, store result directly
            self.execute_path(path, data_idx = None)
            return object.__getattribute__(self, name)
        
        # Otherwise create LazyAttr for per-row/vectorized output
        attr = LazyAttr(self, name)
        object.__setattr__(self, name, attr)
        return attr
    
    def _extract_param(ds, name, data_idx):
        """
        Extract parameter 'name' for data indices 'data_idx' from dataset 'ds'.
        If the parameter is a LazyAttr (per-row), extract only relevant rows.
        Otherwise, return the global scalar / non-row attribute.

        Parameters:
        ds (dataset): The dataset instance.
        name (str): The name of the parameter to extract.
        data_idx (int or list): The data index or indices to extract.

        Returns:
        np.ndarray or scalar: The extracted parameter value(s).
        """
        val = getattr(ds, name)
        # If val is LazyAttr (per-row), extract only relevant rows
        if isinstance(val, LazyAttr):
            return val[data_idx]
        else:
            # global scalar / non-row attribute
            return val