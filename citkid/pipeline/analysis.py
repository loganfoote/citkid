import os
import yaml
import importlib.util 
import numpy as np
import zarr
from . import framework as pf
from .run_validation import get_most_recent_run, get_dependencies
from .dataset import DataSet

class Analyzer():
    def __init__(self, directory, cal_yaml_path, analysis_yaml_path,
                 zarr_path):
        """
        Initialize the Analyzer class, which 
        """
        # Normalize paths
        self.directory = os.path.normpath(directory)
        self.zarr_path = os.path.normpath(zarr_path)
        self.cal_yaml_path = os.path.normpath(cal_yaml_path)
        self.analysis_yaml_path = os.path.normpath(analysis_yaml_path)
        
        self.dataset = DataSet(self.directory, self.cal_yaml_path, 
                               self.zarr_path)
        
        # Load analysis steps from custom_steps.py if it exists
        self.steps = self._load_custom_steps()
        # Add default calibration steps if not already present
        for step in pf.default_analysis_steps:
            if step.name not in [s.name for s in self.steps]:
                self.steps.append(step)
        
        # Load YAML and convert to list of analysis steps
        # yaml_dict = self._load_yaml()
        # self.analysis_list = self._convert_yaml_to_steps_list(yaml_dict)
        
    def _load_custom_steps(self):
        """
        Load custom steps from 'custom_steps.py' in the dataset directory.

        Returns:
        list: A list of custom analysisStep objects.
        """
        custom_module_path = os.path.join(self.directory, 'custom_steps.py')
        if not os.path.exists(custom_module_path):
            return []

        spec = importlib.util.spec_from_file_location("custom_analysis_steps", 
                                                      custom_module_path)
        cs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cs)
        return cs.custom_analysis_steps
        
    def _load_yaml(self):
        """
        Load the YAML configuration file. 

        Returns:
        dict: The loaded YAML configuration as a dictionary.
        """
        with open(self.analysis_yaml_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _convert_yaml_to_steps_list(self, pl_dict, key = None):
        """
        Converts YAML-defined path dictionary leaves to analysisStep objects.

        Parameters:
        pl_dict (dict or str): The YAML-defined path dictionary or leaf.
        key (str): The key associated with the current pl_dict, used to identify
            task names.

        Returns:
        dict or plStep: The converted paths with plStep objects.
        """
        steps_list = []
        if isinstance(pl_dict, dict):
            for key, val in pl_dict.items():
                steps_list = np.append(steps_list,
                            self._convert_yaml_to_steps_list(val, key))
        if isinstance(pl_dict, str) and key == 'task':
            x = [d for d in self.steps if d.name == pl_dict]
            if not len(x):
                m = f"Step '{pl_dict}' not found in available steps."
                raise ValueError(m)
            steps_list = np.append(steps_list, x[0])
        return steps_list
        
    def run_analysis_step(self, name, data_idx=None, save_to_zarr=True):
        """
        Runs an analysis step and saves the output to zarr.
        
        Parameters:
        name (str): The name of the analysis step.
        data_idx (int or array-like): Data index (or indices) to
            run the step on.
        save_to_zarr (bool): If True, save the outputs to the 
            zarr store at Analyzer.dataset.root.
        """
        x = [d for d in self.steps if d.name == name]
        if not len(x):
            m = f"Step '{name}' not found in available steps."
            raise ValueError(m)
        
        step = x[0]
        step.run(self.dataset, data_idx)
        
        if save_to_zarr:
            
            if 'saved' in self.dataset.root.attrs:
                saved = self.dataset.root.attrs['saved']
                dependencies = get_dependencies(step.param_names, saved)
                run_idxs = [get_most_recent_run(rname, saved)+1 for rname in step.return_names]
                run_idxs[run_idxs == 0] = 1
            else:
                dependencies = {}
                run_idxs = [1 for _ in step.return_names]
            
            param_run_idxs = []
            for ii, return_name in enumerate(step.return_names):
                value = getattr(self.dataset, return_name)
                if data_idx is not None:
                    value = value[data_idx]
                    
                for param_name in step.param_names:
                    if param_name in dependencies.keys():
                        param_run_idx = dependencies[param_name]
                    else:
                        # If the parameter name is not in the list of dependencies,
                        # then it must not be an analysis output. 
                        # I.e., it has run_idx = 0.
                        param_run_idx = 0
                    param_run_idxs.append(param_run_idx)
                        
                    
                self.dataset.write_data(return_name, step.func_type,
                                        step.param_names, param_run_idxs,
                                        value, data_idx, run_idxs[ii])
            
            