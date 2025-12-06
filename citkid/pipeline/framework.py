import numpy as np
from .pl_steps import find_pl_path 
import os
import yaml 
import importlib.util
from .pl_steps import default_cal_steps, find_pl_path

class dataset:
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
        
################################################################################
############################### Lazy Attribute #################################
################################################################################
class LazyAttr:
    def __init__(self, DS, name):
        """
        Class to represent a lazily-loaded attribute of a DataSet.

        Parameters:
        DS (DataSet): The DataSet instance this attribute belongs to.
        name (str): The name of the attribute.
        """
        assert isinstance(DS, dataset), "DS must be a dataset instance"
        assert isinstance(name, str), "name must be a string"
        self.DS = DS
        self.name = name
        self._cache = {}        # maps row -> np.ndarray

    def _ensure_loaded(self, rows):
        """
        Load all rows in 'rows' that are not cached.

        Parameters:
        rows (list of int): List of row indices to ensure are loaded.
        """
        missing = [r for r in rows if r not in self._cache]
        if not missing:
            return

        # Generate path and ensure it is valid
        path = find_pl_path(self.DS.cal_pl, self.name)
        if path is None:
            raise AttributeError(f"No processing path for {self.name}")

        # execute_path handles multiple indices at once
        # return shape: (len(missing), ...)
        self.DS.execute_path(path, missing)

    def __getitem__(self, key):
        """
        Get item(s) from the LazyAttr cache, loading from pipeline if needed.

        Parameters:
        key (int, slice, list, tuple): Index or indices to retrieve.

        Returns:
        np.ndarray (N) or (M, N)): Retrieved value(s), where N is the length of 
        the data for a single row, and M is the number of rows requested.
        """
        if isinstance(key, tuple):
            row_key, inner_key = key[0], key[1]
            row = self[row_key]          # load row normally
            return row[inner_key]        # then index inside the row
        elif isinstance(key, slice):
            rows = list(range(*key.indices(self.DS.nres)))
            return_array = True
        elif isinstance(key, (list, np.ndarray)):
            rows = list(key)
            return_array = True
        else:
            rows = [int(key)]
            return_array = False

        self._ensure_loaded(rows)

        # Fetch data from cache
        out = [self._cache[r] for r in rows]

        if not return_array:
            return out[0]  # single row, return 1D array
        else:
            return np.stack(out, axis=0)  # preserve requested shape
        
    def __setitem__(self, key, value):
        """
        Set item(s) in the LazyAttr cache.
        
        Parameters:
        key (int, slice, list, tuple): Index or indices to set.
        value (np.ndarray or list of np.ndarray): Value(s) to set.
        """
        # Allow assignment to one row or multiple rows
        if isinstance(key, slice):
            rows = list(range(*key.indices(self.DS.nres)))
        elif isinstance(key, (list, np.ndarray)):
            rows = list(key)
        elif isinstance(key, tuple):
            row_key, inner_key = key[0], key[1]
            row = self[row_key]          # load row normally
            row[inner_key] = value       # then set inside the row
            self._cache[int(row_key)] = row  # update cache
            return
        else:
            rows = [int(key)]
            value = [value]  # wrap single value for iteration

        # If value is not iterable and multiple rows, broadcast
        if len(rows) != len(value):
            value = [value[0]] * len(rows)

        for r, v in zip(rows, value):
            self._cache[r] = v

    def __repr__(self):
        return f"LazyAttr({self.name}, {len(self._cache.keys()):d} cached rows)"
    
    def __str__(self):
        s = f"Lazy Attribute: {self.name}\n"
        s += f"\tCached Rows: {sorted(self._cache.keys())}"
        return s

################################################################################
########################### Index Mapped Parameter #############################
################################################################################
class IndexMappedParam:
    def __init__(self, base_attr, index_map):
        """
        Wrapper to map indices from a base attribute to another set of indices.

        Parameters:
        base_attr (LazyAttr or np.ndarray): The base attribute to map from.
        index_map (list or np.ndarray): The mapping from desired indices to base
            attribute indices.
        """
        self.base = base_attr
        self.map = index_map

    def __getitem__(self, idx):
        """
        Get item(s) from the mapped attribute.
        
        Parameters:
        idx (int, list, tuple, or np.ndarray): Index or indices to retrieve.
        
        Returns:
        np.ndarray: Mapped value(s) from the base attribute.
        """
        if isinstance(idx, (list, tuple, np.ndarray)):
            return np.array([self.base[self.map[i]] for i in idx])
        else:
            return self.base[self.map[int(idx)]]