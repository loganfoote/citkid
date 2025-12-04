import numpy as np
from citkid.xcal import gain, corr, circle, xcal

################################################################################
#################################### Steps #####################################
################################################################################
class plStep:
    def __init__(self, name, func, param_names, return_names, save = False,
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
        save (bool): Whether to save the results after running the step.
        func_type (str): Type of function execution.
            "per-row": function acts on one data_idx.
            "vectorized": function can act on one or multiple data_idx.
            "global": function acts on global data only.
            "global-res": function loads per-resonator data only all-at-once.
        """
        assert type(name) == str 
        assert callable(func)
        assert type(param_names) == list
        assert type(return_names) == list
        assert type(param_names[0]) == str
        assert type(return_names[0]) == str
        assert func_type in ["per-row", "vectorized", "global", "global-res"]
        self.name = name
        self.func = func
        self.param_names = param_names
        self.return_names = return_names
        self.save = save
        self.func_type = func_type
        

    def run(self, ds, data_idx = None):
        if isinstance(data_idx, int):
            data_idx = [data_idx]

        # --- 1. Collect parameters ---
        params = []
        for p in self.param_names:
            if p == 'data_idx':
                params.append(data_idx)
                continue
            val = getattr(ds, p)
            # Only slice LazyAttr for per-row or vectorized functions
            if isinstance(val, LazyAttr) and self.func_type in ["per-row", "vectorized"]:
                val = val[data_idx]
            params.append(val)

        # --- 2. Execute function based on func_type ---
        if self.func_type == "global" or self.func_type == "global-res":
            results = self.func(*params)
            if not isinstance(results, tuple):
                results = (results,)
            for name, val in zip(self.return_names, results):
                setattr(ds, name, val)  # store as normal attribute, no LazyAttr

        elif self.func_type == "vectorized":
            results = self.func(*params)
            if not isinstance(results, tuple):
                results = (results,)
            for name, val in zip(self.return_names, results):
                attr = getattr(ds, name, None)
                if isinstance(attr, LazyAttr):
                    # assign per-row into LazyAttr
                    for i, v in zip(data_idx, val):
                        attr._cache[i] = v
                else:
                    # create new LazyAttr for per-row output
                    setattr(ds, name, LazyAttr.from_partial_array(ds, name, data_idx, val))

        elif self.func_type == "per-row":
            results_per_row = [[] for _ in self.return_names]

            # params are already sliced to match data_idx
            for local_idx in range(len(data_idx)):
                args_i = [p[local_idx] if isinstance(p, (list, np.ndarray)) else p for p in params]
                out_i = self.func(*args_i)
                if not isinstance(out_i, tuple):
                    out_i = (out_i,)
                for r, v in enumerate(out_i):
                    results_per_row[r].append(v)

            # Assign to LazyAttr
            for name, val in zip(self.return_names, results_per_row):
                attr = getattr(ds, name, None)
                if isinstance(attr, LazyAttr):
                    for r, v in zip(data_idx, val):
                        attr._cache[r] = v
                else:
                    setattr(ds, name, LazyAttr.from_partial_array(ds, name, data_idx, val))

    def __repr__(self):
        s = f"Pipeline Step: {self.name}"
        s += f", Function: {self.func.__module__}.{self.func.__name__}"
        return s
    
    def __str__(self):
        s = f"Pipeline Step: {self.name}"
        s += f"\n\tFunction: {self.func.__module__}.{self.func.__name__}"
        s += f"\n\tInput Parameters: {self.param_names}"
        s += f"\n\tOutput Parameters: {self.return_names}"
        s += f"\n\tSave Results: {self.save}"
        s += f"\n\tFunction Type: {self.func_type}"
        return s

# Problem -> how to deal with the fact that offres cm removal needs to be run on 
# bins of frequencies 
# How to deal with parameters that need to be None on the first run, but maybe 
# should be enforced later

################################ Default steps #################################
# name, function, input parameter names, output parameter names, 
# save, func_vectorized
default_cal_steps =\
(# calibration steps
 ('rmv_gain_f', gain.remove_gain, 
  ['ff', 'zf', 'p_amp', 'p_phase'], ['zf_rmv'], False, 'per-row'),

 ('rmv_gain_t', gain.remove_gain, 
  ['ft', 'zt', 'p_amp', 'p_phase'], ['zt_rmv'], False, 'per-row'),

 ('center_f', circle.cent_rot_s21, 
  ['zf_rmv', 'circ_origin', 'theta_phase_offset'], ['zf_cent'], False, 'per-row'),

 ('center_t', circle.cent_rot_s21, 
  ['zt_rmv', 'circ_origin', 'theta_phase_offset'], ['zt_cent'], False, 'per-row'),

 ('get_theta_f', circle.convert_to_theta, 
  ['zf_cent', 'unwrap_theta_f'], ['theta_f'], False, 'per-row'),

 ('get_theta_t', circle.convert_to_theta, 
  ['zt_cent', 'unwrap_theta_t'], ['theta_t'], False, 'per-row'),

 ('get_x_f', lambda ff, ft: 1 - ff / ft, 
  ['ff', 'ft'], ['x_f'], False, 'per-row'),

 ('get_x_t', np.polyval, 
  ['theta_t', 'poly_x'], ['x_t'], True, 'per-row'),
# analysis steps
 ('make_fr_spans', gain.make_fr_spans, 
  ['fres_all', 'qres_all', 'fg'], ['fr_spans'], True, 'per-row'),

 ('fit_gain', gain.fit_gain, 
  ['fg', 'zg', 'fr_spans'], ['pamp', 'pphase', 'gain_mask'], True, 'per-row'),

 ('fit_circ', circle.fit_iq_circle, 
  ['zf_rmv', 'idx_circfit'], ['circ_origin', 'circ_radius'], True, 'per-row'),

 ('get_theta_phase_offset', np.median, 
  ['zt_rmv'], ['theta_phase_offset'], True, 'per-row'),

 ('get_xcal_idx', xcal.get_xcal_idx,
  ['ff', 'theta_f', 'theta_t', 'xcal_idx0_offset', 'xcal_idx1_offset', 
   'xcal_std_cutoff'], ['xcal_idx'], False, 'per-row'),

 ('cut_xf', lambda x, t, idx: (x[idx], t[idx]), 
  ['x_f', 'theta_f', 'xcal_idx'], ['x_f_cut', 'theta_f_cut'], False, 'per-row'),

 ('fit_x_theta', np.polyfit, 
  ['x_f_cut', 'theta_f_cut', 'poly_x_deg'], ['poly_x'], True, 'per-row'),
# extra steps
 ('get_A_t', circle.convert_to_A, 
  ['zt_cent'], ['A_t'], False, 'per-row'),
 ('get_sparper', circle.get_spar_sper, 
  ['theta_t', 'A_t', 'circ_radius', 'dt', 'sparper_get_freqs'], 
  ['spar', 'sper'], False, 'per-row')
)

default_cal_steps = [plStep(*cs) for cs in default_cal_steps]

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
    check_pl_tree_structure(tree) # confirm that the structure is valid
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
        keys = sorted(seq.items(), key=lambda kv: int(kv[0]))
        for idx, (key, stepnode) in enumerate(keys):
            res = contains_path_stepnode(stepnode)
            if res is not None:
                # prepend full execution of all earlier steps in this sequence
                pref = []
                for prev_key, prev_node in keys[:idx]:
                    pref.extend(execute_full_stepnode(prev_node))
                return pref + res
        return None

    # Top-level: walk root keys that are sequences (like "CAL_STEPS")
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
        if not (sorted(keys) == list(range(1, len(keys) + 1))):
            m = 'Pipeline tree node integer keys must start from 1 '
            m += 'and increase by 1 without gaps.'
            raise ValueError(m)
        if not all([k == int for k in key_types]):
            m = 'Pipeline tree node with integer keys cannot '
            m += 'have other key types.'
            raise ValueError(m)
        for val in tree.values():
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

################################################################################
############################### Lazy Attributes ################################
################################################################################

class LazyAttr:
    def __init__(self, ds, name):
        self.ds = ds
        self.name = name
        self._cache = {}        # maps row -> np.ndarray
        self._missing = set()   # track which rows we know are absent (optional)

    def _ensure_loaded(self, rows):
        """Load all rows in 'rows' that are not cached."""
        missing = [r for r in rows if r not in self._cache]
        if not missing:
            return

        # Generate path and ensure it is valid
        path = find_pl_path(self.ds.cal_pl, self.name)
        if path is None:
            raise AttributeError(f"No processing path for {self.name}")

        # execute_path handles multiple indices at once
        # return shape: (len(missing), ...)
        self.ds.execute_path(path, missing)

    def __getitem__(self, key):
        if isinstance(key, slice):
            rows = list(range(*key.indices(self.ds.nres)))
            return_array = True
        elif isinstance(key, (list, tuple, np.ndarray)):
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
        # Allow assignment to one row or multiple rows
        if isinstance(key, slice):
            rows = list(range(*key.indices(self.ds.nres)))
        elif isinstance(key, (list, tuple, np.ndarray)):
            rows = list(key)
        else:
            rows = [int(key)]
            value = [value]  # wrap single value for iteration

        # If value is not iterable and multiple rows, broadcast
        if len(rows) != len(value):
            value = [value[0]] * len(rows)

        for r, v in zip(rows, value):
            self._cache[r] = v