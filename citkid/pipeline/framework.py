import numpy as np
from ..xcal import gain, circle, xcal
from .dependencies import get_deps, get_most_recent_run
import copy
        
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
        for a in ['cal_pl', 'execute_path', 'nres']:
            if not DS.has_attr(a):
                raise AttributeError(f"DS must have '{a}' attribute")
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        self.DS = DS
        self.name = name
        self._cache = {}        # maps row -> np.ndarray
        self.shape = ()
        
        ### Check if the attribute exists in the zarr file.
        # self.run_idx = DS.get_attr_version(name)
        # if self.run_idx is not None:
        #     grp = DS.root[str(self.run_idx)]
        #     self.data = grp[name]
        #     self.data_idx = grp[f'{name}_idx']
        # else:
        #     self.run_idx = 1

    def _ensure_loaded(self, rows):
        """
        Load all rows in 'rows' that are not cached.

        Parameters:
        rows (list of int): List of row indices to ensure are loaded.
        """
        # Type checks
        if isinstance(rows, np.ndarray):
            rows = rows.tolist()
        if isinstance(rows, list):
            if not all(isinstance(r, (int, np.integer)) for r in rows):
                raise ValueError("all rows must be integers") 
        else:
            if not isinstance(rows, (int, np.integer)):   
                raise ValueError("row must be an integer")
            rows = [rows] # wrap single int in list

        for r in rows:
            if not isinstance(r, (int, np.integer)):
                raise ValueError("all rows must be integers")
            if not 0 <= r < self.DS.nres:
                raise ValueError("row index out of bounds")

        # Determine which rows are missing
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
        return_array = True
        # Allow assignment to one row or multiple rows
        if isinstance(key, slice):
            rows = list(range(*key.indices(self.DS.nres)))
        elif isinstance(key, (list, np.ndarray)):
            rows = list(key)
        elif isinstance(key, tuple):
            row_key, inner_key = key[0], key[1:]
            if len(inner_key) == 1:
                inner_key = inner_key[0]
            return self[row_key][inner_key]
        else:
            rows = [key]
            return_array = False

        # key type checks
        if not all([isinstance(r, (int, np.integer)) \
                    for r in rows]):
            raise ValueError("all rows must be integers")
        for idx, r in enumerate(rows):
            if r < 0:
                rows[idx] = self.DS.nres + r
            if not 0 <= rows[idx] < self.DS.nres:
                raise ValueError(f"row index {r} out of bounds")

        # # If the data was found in the zarr, then just load it from the zarr array.
        # if self.run_idx is not None:
        #     local_idxs = np.where(np.isin(self.data_idx, rows))[0]
        #     out = self.data[local_idxs]
                
        # else: # Otherwise, load data into the cache, and return it from the cache.
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
            row_key, inner_key = key[0], key[1:]
            if len(inner_key) == 1:
                inner_key = inner_key[0]
            row = self[row_key]          # load row normally
            row[inner_key] = value       # then set inside the row
            self._cache[int(row_key)] = row  # update cache
            return
        else:
            rows = [key]
            value = [value]  # wrap single value for iteration

        # key type checks
        if not all([isinstance(r, (int, np.integer)) \
                    for r in rows]):
            raise ValueError("all rows must be integers")
        for idx, r in enumerate(rows):
            if r < 0:
                rows[idx] = self.DS.nres + r
            if not 0 <= rows[idx] < self.DS.nres:
                raise ValueError(f"row index {r} out of bounds")

        # Ensure value is iterable and matches length of rows
        try:
            iter(value)
        except TypeError:
            value = [value] * len(rows)
        if len(value) != len(rows):
            raise ValueError("Length of value does not match number of rows.") 

        for r, v in zip(rows, value):
            self._cache[r] = v
            
        # Update the shape of the LazyAttr if needed.
        if self.shape == ():
            self.shape = (self.DS.nres, *value[0].shape)

    def __repr__(self):
        return f"LazyAttr({self.name}, {len(self._cache.keys()):d} cached rows)"
    
    def __str__(self):
        s = f"Lazy Attribute: {self.name}\n"
        s += f"\tCached Rows: {sorted(self._cache.keys())}"
        return s

################################################################################
#################################### Steps #####################################
################################################################################
class plStep:
    def __init__(self, name, func, param_names, return_names,
                 func_type = "per-row"):
        """
        Class to represent a step in the analysis or calibration pipeline.

        Parameters:
        name (str): Name of the pipeline step.
        func (callable): Function to execute for this step.
        param_names (list of str): Names of the parameters to pass to the
            function.
        return_names (list of str): Names of the attributes to store the
            function's return values.
        func_type (str): Type of function execution.
            "per-row": function acts on one data_idx.
            "vectorized": function can act on one or multiple data_idx.
            "global": function acts on global data only.
            "global-res": function loads per-resonator data only all-at-once.

        Returns:
        None
        """
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        if not callable(func):
            raise ValueError("func must be callable")
        if not isinstance(param_names, list):
            raise ValueError("param_names must be a list")
        if not isinstance(return_names, list):
            raise ValueError("return_names must be a list")
        if not all([isinstance(p, str) for p in param_names]):
            raise ValueError("all param_names must be strings")
        if not all([isinstance(r, str) for r in return_names]):
            raise ValueError("all return_names must be strings")
        if func_type not in ["per-row", "vectorized", "global", "global-res"]:
            m = "func_type must be one of 'per-row', 'vectorized', "
            m += "'global', 'global-res'"
            raise ValueError(m)
        
        self.name = name
        self.func = func
        self.param_names = param_names[:]
        self.return_names = return_names[:]
        self.func_type = func_type
        
    def run(self, DS, data_idx = None):
        """
        Run the pipeline step on the given dataset.

        Parameters:
        DS (.dataset.DataSet): The dataset to operate on.
        data_idx (int or list of int, optional): Data index or indices to
            process. For func_type == "global" or "global-res", data_idx is
            ignored. For func_type == "vectorized", the function is called on
            data indexed by data_idx. For func_type == "per-row", the function
            is called separately for each data index in data_idx if data_idx is
            a list. If data_idx is None, all indices (0 to DS.nres - 1) are
            processed.

        Returns:
        None
        """
        if self.func_type in ["per-row", "vectorized"]:
            if not DS.has_attr('nres'):
                raise ValueError("DS must have 'nres' attribute")
            if data_idx is None:
                data_idx = list(range(DS.nres))
            else:
                data_idx = np.atleast_1d(data_idx)
        elif self.func_type in ['global', 'global-res']:
            if data_idx is not None:
                m = "data_idx must be None for global or global-res functions."
                raise ValueError(m)         
                
        # --- 1. Collect parameters ---
        params = []
        param_is_global = []
        for p in self.param_names:
            if p == 'data_idx':
                params.append(data_idx)
                continue
            
            val = getattr(DS, p)
            param_is_global.append(DS.global_cache[p])
            # Only slice input for per-row or vectorized functions
            if self.func_type in ["per-row", "vectorized"]:
                if not param_is_global[-1]:
                    val = val[data_idx]
            params.append(val)

        # --- 2. Execute function based on func_type ---
        if self.func_type == "global" or self.func_type == "global-res":
            # No difference here, but kept for clarity
            results = self.func(*params)
            if not isinstance(results, tuple):
                results = (results,)
            if len(self.return_names) != len(results):
                raise ValueError("Function return length does not match "
                                 "number of return names.")
                
            for name, val in zip(self.return_names, results):
                setattr(DS, name, val)  # store as normal attribute, no LazyAttr
                DS.global_cache[name] = True
                
        elif self.func_type == "vectorized":
            results = self.func(*params)
            if not isinstance(results, tuple):
                results = (results,)
            if len(self.return_names) != len(results):
                raise ValueError("Function return length does not match "
                                 "number of return names.")
            for name, val in zip(self.return_names, results):
                if not DS.has_attr(name):
                    setattr(DS, name, LazyAttr(DS, name))
                    DS.global_cache[name] = False
                getattr(DS, name)[data_idx] = val

        elif self.func_type == "per-row":
            results_per_row = [[] for _ in self.return_names]

            # params are already sliced to match data_idx
            for local_idx in range(len(data_idx)):
                args_i = []
                for p, is_global in zip(params, param_is_global):
                    if not is_global:
                        p = p[local_idx]
                    args_i.append(p)
                out_i = self.func(*args_i)
                if not isinstance(out_i, tuple):
                    out_i = (out_i,)
                for r, v in enumerate(out_i):
                    results_per_row[r].append(v)

            # Assign to LazyAttr
            if len(self.return_names) != len(results_per_row):
                raise ValueError("Function return length does not match "
                                 "number of return names.")
            for name, val in zip(self.return_names, results_per_row):
                if not DS.has_attr(name):
                    setattr(DS, name, LazyAttr(DS, name))
                    DS.global_cache[name] = False
                getattr(DS, name)[data_idx] = val
                
        DS.update_deps_map(self.param_names, self.return_names, data_idx)

    def __repr__(self):
        s = f"Pipeline Step: {self.name}"
        s += f", Function: {self.func.__module__}.{self.func.__name__}"
        return s
    
    def __str__(self):
        s = f"Pipeline Step: {self.name}"
        s += f"\n\tFunction: {self.func.__module__}.{self.func.__name__}"
        s += f"\n\tInput Parameters: {self.param_names}"
        s += f"\n\tOutput Parameters: {self.return_names}"
        s += f"\n\tFunction Type: {self.func_type}"
        return s

################################ Default steps #################################
# name, function, input parameter names, output parameter names, 
# save, func_vectorized
default_cal_steps =\
(# calibration steps
 ('rmv_gain_f', gain.remove_gain, 
  ['ff', 'zf', 'p_amp', 'p_phase'], ['zf_rmv'], 'per-row'),

 ('rmv_gain_t', gain.remove_gain, 
  ['ft', 'zt', 'p_amp', 'p_phase'], ['zt_rmv'], 'per-row'),

 ('center_f', circle.cent_rot_s21, 
  ['zf_rmv', 'circ_origin', 'theta_phase_offset'], ['zf_cent'], 'per-row'),

 ('center_t', circle.cent_rot_s21, 
  ['zt_rmv', 'circ_origin', 'theta_phase_offset'], ['zt_cent'], 'per-row'),

 ('get_thetaf', circle.convert_to_theta, 
  ['zf_cent', 'unwrap_thetaf'], ['thetaf'], 'per-row'),

 ('get_thetat', circle.convert_to_theta, 
  ['zt_cent', 'unwrap_thetat'], ['thetat'], 'per-row'),

  ('cut_xf', lambda x, t, mask: (x[mask], t[mask]), 
  ['xf', 'thetaf', 'xcal_mask'], ['xf_cut', 'thetaf_cut'], 'per-row'),

 ('get_xf', lambda ff, ft: 1 - ff / ft, 
  ['ff', 'ft'], ['xf'], 'per-row'),

 ('get_xt', np.polyval, 
  ['thetat', 'poly_x'], ['xt'], 'per-row'),
 # extra steps
 ('get_At', circle.convert_to_A, 
  ['zt_cent'], ['At'], 'per-row'),
 ('get_sparper', circle.get_spar_sper, 
  ['thetat', 'At', 'circ_radius', 'dt', 'sparper_get_freqs'], 
  ['spar', 'sper'], 'per-row'),
  
)
default_analysis_steps =\
(# analysis steps
 ('make_fr_spans', gain.make_fr_spans, 
  ['fres_all', 'qres_all'], ['fr_spans'], 'global'),

 ('fit_gain', gain.fit_gain, 
  ['fg', 'zg', 'fr_spans'], ['p_amp', 'p_phase', 'gain_mask'], 'per-row'),

 ('fit_circ', circle.fit_iq_circle, 
  ['zf_rmv', 'idx_circfit'], ['circ_origin', 'circ_radius'], 'per-row'),

 ('get_theta_phase_offset', np.median, 
  ['zt_rmv'], ['theta_phase_offset'], 'per-row'),

 ('get_xcal_mask', xcal.get_xcal_mask,
  ['ff', 'thetaf', 'thetat', 'xcal_idx0_offset', 'xcal_idx1_offset', 
   'xcal_std_cutoff'], ['xcal_mask'], 'per-row'),

 ('fit_x_theta', np.polyfit, 
  ['xf_cut', 'thetaf_cut', 'poly_x_deg'], ['poly_x'], 'per-row'),
  # This will probably be concatenated with 'cut_xf' and take the mask as input
)


default_cal_steps = [plStep(*cs) for cs in default_cal_steps]
default_analysis_steps = [plStep(*ans) for ans in default_analysis_steps]


# ('rmv_cm_offres', corr.remove_cm_complex, 
# ['zt_rmv', 'aI', 'aQ', 'AI', 'AQ', 'cm_offres_idx', 'theta_cm_offres'],
# ['zt_cm_rmv']),

# ('calc_cm_offres', corr.calc_cm_complex, 
# ['zt_rmv', 'theta_cm_offres', 'N_comp_offres', 'N_iter_offres', 'dt', 
# 'lowpass_params_offres', 'highpass_params_offres', 'verbose'],
# ['aI', 'aQ', 'AI', 'AQ', 'sigI_iter', 'sigQ_iter', 'aI_full', 'aQ_full', 
# 'theta_cm_offres']),

################################################################################
#################################### Paths #####################################
################################################################################
def find_pl_path(tree, return_name):
    """
    Given a pipeline tree (as imported from calibration yaml file) and an output
    name, return the execution path that produces the output.

    Parameters:
    tree (dict): Pipeline tree structure that passes check_pl_tree_structure().
    return_name (str): Name of the desired output.

    Returns:
    path (list[plStep] or None): If found, list of plStep instances that must be
        executed in order to produce the desired output. If not found, returns
        None.
    """
    def is_seq(d):
        """
        Check if the given dictionary represents a sequence, i.e., all keys 
        are digits.

        Parameters:
        d (dict): Dictionary to check.

        Returns:
        bool: True if d is a sequence, False otherwise.
        """
        isd = isinstance(d, dict)
        return isd and all(k is not None and str(k).isdigit() for k in d.keys())

    def execute_full_stepnode(node):
        """
        Return full execution expansion of this stepnode: step + all descendant 
        work.

        Parameters:
        node (dict): stepnode dictionary with "task" key and possibly child 
            sequences.

        Returns:
        path (list[plStep]): Full execution path of this stepnode and 
            descendants.
        """
        step = node["task"]
        path = [step]
        sort_key = lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0
        for k, child in sorted(node.items(), key = sort_key):
            if k == "task":
                continue
            if is_seq(child):
                sort_key = lambda kv: int(kv[0])
                for _, child_stepnode in sorted(child.items(), key = sort_key):
                    path.extend(execute_full_stepnode(child_stepnode))
        return path

    def contains_path_stepnode(node):
        """
        If this stepnode contains the output in itself or descendants,
        return path from this stepnode to the matching leaf (list[plStep]).
        Otherwise return None.
        """
        step = node["task"]
        if return_name in getattr(step, "return_names", []):
            return [step]

        # search each child sequence in natural order; if found, prepend 
        # this step
        def sort_key(kv):
            if kv[0] == "task":
                return 0
            elif str(kv[0]).isdigit():
                return int(kv[0])
            else:
                return 9999
            
        for k, child in sorted(node.items(), key = sort_key):
            if k == "task":
                continue
            if is_seq(child):
                seq_res = contains_path_sequence(child)
                if seq_res is not None:
                    return [step] + seq_res
        return None

    def contains_path_sequence(seq):
        """
        If this sequence contains the output in itself or descendants,
        return path from this sequence to the matching leaf (list[plStep]).
        Otherwise return None.

        Parameters:
        seq (dict): sequence dictionary with digit keys and stepnode values.

        Returns:
        path (list[plStep] or None): Execution path to matching leaf, or None
            if not found.
        """
        keys = sorted(seq.items(), key = lambda kv: int(kv[0]))
        for idx, (key, stepnode) in enumerate(keys):
            res = contains_path_stepnode(stepnode)
            if res is not None:
                # prepend full execution of all earlier steps in this sequence
                pref = []
                for prev_key, prev_node in keys[:idx]:
                    pref.extend(execute_full_stepnode(prev_node))
                return pref + res
        return None

    # Check input datatypes
    if type(return_name) != str:
        raise ValueError("return_name must be a string")
    # check_pl_tree_structure(tree) # confirm that the structure is valid
    if not all([type(k) == str and k.endswith('_STEPS') for k in tree.keys()]):
        raise ValueError('root keys must end with "_STEPS"')
    
    # Top-level: walk root keys
    for root_val in tree.values():
        if is_seq(root_val):
            out = contains_path_sequence(root_val)
            if out is not None:
                return out
    return None

def check_pl_tree_structure(tree):
    """
    Recursively check that pipeline tree structure is valid.

    Parameters:
    tree (dict): Pipeline tree node to check.

    Raises:
    ValueError: If the tree structure is invalid.
    """
    # this function only excepts dictionary arguments
    if not isinstance(tree, dict):
        m = 'Pipeline tree node must be a dict or plStep instance.'
        raise ValueError(m)
    
    # check if keys are integer sequence starting from 1
    keys = list(tree.keys()) 
    key_types = [type(k) for k in keys]
    if any([k == int for k in key_types]):
        if not all([k == int for k in key_types]):
            m = 'Pipeline tree node with integer keys cannot '
            m += 'have other key types.'
            raise ValueError(m)
        if not (sorted(keys) == list(range(1, len(keys) + 1))):
            m = 'Pipeline tree node integer keys must start from 1 '
            m += 'with a step size of 1.'
            raise ValueError(m)
        for key, val in tree.items():
            if type(val) != dict or 'task' not in val.keys():
                m = 'Pipeline tree stepnode must have a "task" key.'
                raise ValueError(m)
            check_pl_tree_structure(val)

    # Otherwise keys must all be strings
    elif any([k == str for k in key_types]):
        # If any keys are strings, they must all be strings
        if not all([k == str for k in key_types]):
            m = 'Pipeline tree node with string keys cannot '
            m += 'have other key types.'
            raise ValueError(m)
        # Check if task key is present, and corresponds to valid plStep instance
        if 'task' in keys:
            if not isinstance(tree['task'], plStep):
                m = f"Pipeline task <{tree['task']}> is not a valid "
                m += "plStep instance."
                raise ValueError(m)
        # Check if delete_input key is present, 
        # and corresponds to 'all' or list of strings
        # Need to check that list elements are valid for task
        if 'delete_input' in keys:
            if 'task' not in keys:
                m = 'Pipeline tree "delete_input" key requires '
                m += 'a corresponding "task" key.'
                raise ValueError(m)
            val = tree['delete_input']
            if isinstance(val, list):
                for v in val:
                    if not isinstance(v, str):
                        m = 'Pipeline tree "delete_input" list elements '
                        m += 'must be strings.'
                        raise ValueError(m)
                    if v not in tree['task'].param_names:
                        m = f'Pipeline tree "delete_input" value "{v}" '
                        m += f'not found in "{tree["task"].name}" input names.'
                        raise ValueError(m)
            elif val != 'all':
                m = f"Pipeline tree <delete_input: {val}> is not a valid list"
                m += " of input names or 'all'."
                raise ValueError(m)
            
        # Otherwise, key should be <>_STEPS sequence 
        checked_keys = ['task', 'delete_input']
        for k in checked_keys:
            if k in keys:
                keys.remove(k)
        for key in keys:
            if not key.endswith('_STEPS'):
                m = f'Pipeline tree string keys must be in {checked_keys}'
                m += f' or end with "_STEPS". '
                m += f'Found invalid key: {key}'
                raise ValueError(m)
            check_pl_tree_structure(tree[key])

def print_pl_path(paths, indent = 0):
    """
    Print calibration paths in a readable format.

    Parameters:
    paths (dict or list): The calibration paths to print.
    indent (int): Current indentation level for nested structures.
    """
    if isinstance(paths, dict):
        for key, val in paths.items():
            print(f"{'  ' * indent}{key}:")
            print_pl_path(val, indent + 1)
    else:
        p = '\n' + paths.__str__() 
        p = p.replace('\n', f'\n{"  " * (indent + 1)}')[2:]
