import numpy as np
from citkid.pipeline.dependencies import get_most_recent_run

################################################################################
############################### Lazy Attribute #################################
################################################################################
class LazyAttr:
    def __init__(self, DS, name, run_idx):
        """
        Class to represent a lazily-loaded attribute of a DataSet.
        Provides lazy row-wise indexing that delegates to DataSet for data 
        fetching.

        Parameters:
        DS (DataSet): The DataSet instance this attribute belongs to.
        name (str): The name of the attribute.
        run_idx (int): The run index corresponding to the version of the 
            attribute to load when fetching data.
        """
        # Input validation 
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        run_idx = int(run_idx)

        # store DS and name for later use, and initialize cache and shape
        self.DS = DS
        self.name = name
        self.run_idx = run_idx
        self._cache = {}        # maps row -> np.ndarray
        self.shape = ()

    def _normalize_key(self, key):
        """
        Normalize various key types to a list of row indices.
        
        Parameters:
        key (int, slice, list, np.ndarray, tuple): Index or indices to 
            normalize.
        
        Returns:
        tuple: (rows, return_array, inner_key) where rows is list of ints,
               return_array is bool, inner_key is None or the sub-indexing key.
        """
        inner_key = None
        return_array = True
        
        # Handle tuple for fancy indexing like lazy_attr[5, 10:20]
        if isinstance(key, tuple):
            row_key = key[0]
            inner_key = key[1:] if len(key) > 2 else key[1]
            key = row_key
        
        # Convert key to list of rows
        if isinstance(key, slice):
            rows = list(range(*key.indices(self.DS.nrows)))
        elif isinstance(key, (list, np.ndarray)):
            rows = [int(r) for r in key]
        elif isinstance(key, (int, np.integer)):
            rows = [int(key)]
            return_array = False
        else:
            raise TypeError(f"Invalid index type: {type(key)}")
        
        # Handle negative indices and validate bounds
        normalized_rows = []
        for r in rows:
            if r < 0:
                r = self.DS.nrows + r
            if not 0 <= r < self.DS.nrows:
                raise IndexError(
                    f"Row index {r} out of bounds [0, {self.DS.nrows})"
                    )
            normalized_rows.append(r)
        
        return normalized_rows, return_array, inner_key

    def __getitem__(self, key):
        """
        Get item(s) from the LazyAttr, delegating to DataSet for fetching.

        Parameters:
        key (int, slice, list, tuple): Index or indices to retrieve.

        Returns:
        np.ndarray (N,) or (M, N): Retrieved value(s), where N is the length of 
        the data for a single row, and M is the number of rows requested.
        """
        rows, return_array, inner_key = self._normalize_key(key)
        
        # Handle sub-indexing (e.g., lazy_attr[5, 10:20])
        if inner_key is not None:
            return self[rows[0]][inner_key]
        
        # Check cache first, fetch missing rows from DataSet
        missing = [r for r in rows if r not in self._cache]
        if missing:
            # Delegate to DataSet to fetch the missing rows
            fetched_data = self.DS._fetch_rows(
                self.name, self.run_idx, missing
                )
            # Update cache with fetched data
            for r, data in zip(missing, fetched_data):
                self._cache[r] = data
            # Update shape if first time loading
            if self.shape == () and len(fetched_data) > 0:
                self.shape = (self.DS.nrows, *fetched_data[0].shape)
        
        # Retrieve from cache
        out = [self._cache[r] for r in rows]
        
        if not return_array:
            return out[0]  # single row, return 1D array
        else:
            return np.stack(out, axis=0)  # multiple rows, stack into 2D array
        
    def __setitem__(self, key, value):
        """
        Set item(s) in the LazyAttr cache.
        
        Parameters:
        key (int, slice, list, tuple): Index or indices to set.
        value (np.ndarray or list of np.ndarray): Value(s) to set.
        """
        # Allow assignment to one row or multiple rows
        if isinstance(key, slice):
            rows = list(range(*key.indices(self.DS.nrows)))
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
                rows[idx] = self.DS.nrows + r
            if not 0 <= rows[idx] < self.DS.nrows:
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
            first_val = value[0]
            if hasattr(first_val, 'shape'):
                self.shape = (self.DS.nrows, *first_val.shape)
            else:
                # Scalar value, shape is just (nrows,)
                self.shape = (self.DS.nrows,)

    def __len__(self):
        """
        Return the number of rows (nrows) for this attribute.
        """
        return self.DS.nrows

    def __repr__(self):
        """
        Return a concise string representation of the LazyAttr, showing its name
        and number of cached rows.
        """
        return f"LazyAttr({self.name}, {len(self._cache.keys()):d} cached rows)"
    
    def __str__(self):
        """
        Return a detailed string representation of the LazyAttr, showing its 
        name and the specific cached rows.
        """
        s = f"Lazy Attribute: {self.name}\n"
        s += f"\tCached Rows: {sorted(self._cache.keys())}"
        return s


################################################################################
######################### Lazy Attribute Collection ############################
################################################################################
class LazyAttrCollection:
    """
    Collection of LazyAttr objects for a single parameter across multiple runs.
    
    Manages access to per-row parameters that may exist at different run indices
    for different data indices. Provides convenient access to the most recent
    run for each data_idx, while also allowing access to specific historical runs.
    
    Typical usage:
        # Get most recent run for data indices 0 and 1
        data = DS.x[[0, 1]]
        
        # Access specific historical run
        data = DS.x.at_run(5)[[0, 1]]
    
    Attributes:
        DS (DataSet): The DataSet instance this collection belongs to.
        name (str): The name of the parameter.
        _lazy_attrs (dict): Maps run_idx → LazyAttr for that run.
    """
    
    def __init__(self, DS, name):
        """
        Initialize a LazyAttrCollection for tracking parameter versions.
        
        Parameters:
            DS (DataSet): The DataSet instance this collection belongs to.
            name (str): The name of the parameter being tracked.
        """
        self.DS = DS
        self.name = name
        self._lazy_attrs = {}  # {run_idx: LazyAttr}
        
    def _normalize_index(self, key):
        """
        Normalize various key types to array of data indices.
        
        Handles slices, negative indices, boolean masks, scalar ints, and arrays.
        
        Parameters:
            key (int, slice, list, np.ndarray): Index or indices to normalize.
        
        Returns:
            tuple: (data_idx_arr, is_scalar) where data_idx_arr is np.ndarray of 
                   int32, and is_scalar indicates if input was scalar.
        """
        is_scalar = False
        
        # Handle slices
        if isinstance(key, slice):
            indices = list(range(*key.indices(self.DS.nrows)))
            data_idx_arr = np.array(indices, dtype=np.int32)
        
        # Handle boolean masks
        elif isinstance(key, (list, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                # Boolean indexing
                if len(arr) != self.DS.nrows:
                    raise IndexError(
                        f"Boolean index length {len(arr)} doesn't match nrows {self.DS.nrows}"
                    )
                indices = np.where(arr)[0]
                data_idx_arr = np.array(indices, dtype=np.int32)
            else:
                # Integer array indexing
                data_idx_arr = np.asarray(arr, dtype=np.int32)
        
        # Handle scalar integer
        elif isinstance(key, (int, np.integer)):
            data_idx_arr = np.array([int(key)], dtype=np.int32)
            is_scalar = True
        
        else:
            raise TypeError(
                f"Invalid index type: {type(key)}. "
                f"Expected int, slice, list, or array."
            )
        
        # Handle negative indices
        data_idx_arr = np.where(
            data_idx_arr < 0, 
            data_idx_arr + self.DS.nrows, 
            data_idx_arr
        )
        
        # Validate bounds
        if np.any((data_idx_arr < 0) | (data_idx_arr >= self.DS.nrows)):
            raise IndexError(
                f"Index out of bounds [0, {self.DS.nrows})"
            )
        
        return data_idx_arr, is_scalar
        
    def __getitem__(self, data_idx):
        """
        Get data from the most recent run for each requested data_idx.
        
        For each data_idx, determines which run_idx is most recent according to
        the dependency maps, then retrieves data from that run's LazyAttr.
        Different data indices may return data from different runs.
        
        If the parameter doesn't exist yet for a data_idx (run_idx == -1),
        the data will be produced and stored in run 1 (run 0 is reserved).
        
        Parameters:
            data_idx (int, slice, list, array): Data index or indices to retrieve.
                Supports numpy-like indexing: scalars, slices, lists, arrays, 
                boolean masks, and negative indices.
        
        Returns:
            scalar or np.ndarray: Data from most recent runs. Returns scalar if
                input was scalar, array if input was array-like.
        
        Raises:
            KeyError: If no run exists for a required run_idx.
            IndexError: If indices are out of bounds.
            
        Examples:
            >>> collection[0]           # Single index
            >>> collection[-1]          # Negative index
            >>> collection[0:5]         # Slice
            >>> collection[[0, 1, 5]]   # List of indices
            >>> collection[mask]        # Boolean mask
        """
        # Normalize input to array of indices
        data_idx_arr, is_scalar = self._normalize_index(data_idx)

        # Map each data_idx to its run_idx and preserve order
        run_idx_to_data_idx = {}
        di_to_position = {}  # Track original position
        for pos, di in enumerate(data_idx_arr):
            if di not in self.DS.deps_maps:
                self.DS.deps_maps[di] = {}
            run_idx = get_most_recent_run(self.name, self.DS.deps_maps[di])
            # If parameter doesn't exist yet for this data_idx, default to run 1
            # (run 0 is reserved as a special case)
            if run_idx == -1:
                run_idx = 1
                # Ensure run 1 LazyAttr exists in the collection
                if run_idx not in self._lazy_attrs:
                    self._lazy_attrs[run_idx] = LazyAttr(
                        self.DS, self.name, run_idx
                        )
            if run_idx not in run_idx_to_data_idx:
                run_idx_to_data_idx[run_idx] = []
            run_idx_to_data_idx[run_idx].append(di)
            di_to_position[di] = pos
        
        # Fetch data in groups
        results = [None] * len(data_idx_arr)  # Pre-allocate in correct order
        for run_idx, data_idxs in run_idx_to_data_idx.items():
            lazy_attr = self._lazy_attrs[run_idx]
            fetched = lazy_attr[data_idxs]  # Batch call - returns 2D array
            # fetched is always 2D: (n_rows, ...), iterate through rows
            for i, di in enumerate(data_idxs):
                results[di_to_position[di]] = fetched[i]
        
        # Return scalar or array based on input
        if is_scalar:
            return results[0]
        return np.array(results)
        
    def at_run(self, run_idx):
        """
        Access the LazyAttr for a specific run index.
        
        Useful for accessing historical versions of a parameter when multiple
        runs exist. The returned LazyAttr can be indexed normally.
        
        Parameters:
            run_idx (int): The run index to access.
        
        Returns:
            LazyAttr: The LazyAttr object for the specified run.
        
        Raises:
            KeyError: If the specified run_idx doesn't exist.
            
        Examples:
            >>> collection.at_run(5)[0]        # Get data_idx 0 from run 5
            >>> collection.at_run(3)[[0, 1]]   # Get multiple indices from run 3
        """
        return self._lazy_attrs[run_idx]
        
    def add_run(self, run_idx, lazy_attr):
        """
        Register a LazyAttr for a specific run index.
        
        Called internally when new data is produced or loaded from zarr.
        Should not typically be called by users directly.
        
        Parameters:
        run_idx (int): The run index being registered.
        lazy_attr (LazyAttr): The LazyAttr object for this run.
        
        Raises:
        ValueError: If run_idx already exists (prevents accidental 
            overwrites).
        """
        if run_idx in self._lazy_attrs:
            raise ValueError(
                f"Run {run_idx} already exists for parameter '{self.name}'"
            )
        self._lazy_attrs[run_idx] = lazy_attr
    
    def __len__(self):
        """
        Return the number of data indices (nrows) in the dataset.
        """
        return self.DS.nrows
    
    def __repr__(self):
        """
        Return a concise string representation showing parameter name and 
        available runs.
        """
        runs = sorted(self._lazy_attrs.keys())
        return f"LazyAttrCollection({self.name}, runs={runs})"
    
    def __str__(self):
        """
        Return a detailed string representation.
        """
        runs = sorted(self._lazy_attrs.keys())
        s = f"Lazy Attribute Collection: {self.name}\n"
        s += f"\tAvailable runs: {runs}\n"
        s += f"\tNumber of data indices: {self.DS.nrows}"
        return s

################################################################################
#################################### Steps #####################################
################################################################################
class plStep:
    def __init__(
            self, name, func, param_names, return_names, func_type = "per-row"
            ):
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
        
    def _run(self, params, param_is_global):
        """
        Run the pipeline step with the given parameters, and return the output
        as a dictionary mapping return_names to values.

        Parameters: 
        params (list): List of parameter values to pass to the function, in the
            same order as self.param_names. For func_type == "per-row" or 
            "vectorized", each parameter can be either a global value or an 
            array of values corresponding to each value in data_idx, as 
            indicated by param_is_global. 
        param_is_global (list of bool): List of booleans indicating whether each
            parameter is global (True) or per-row (False). 

        Returns:
        dict: Dictionary mapping return_names to their corresponding output 
            values.
        """ 
        ### Input validation 
        if not isinstance(params, list):
            raise ValueError("params must be a list")
        if not isinstance(param_is_global, list):
            raise ValueError("param_is_global must be a list")
        if len(params) != len(self.param_names):
            raise ValueError("Length of params does not match number of "
                             "parameter names.")
        if len(param_is_global) != len(self.param_names):
            raise ValueError("Length of param_is_global does not match number "
                             "of parameter names.")
        if not all(isinstance(pg, bool) for pg in param_is_global):
            raise ValueError(
                "All elements of param_is_global must be booleans."
                )
        
        # Determine number of rows for per-row/vectorized functions
        if self.func_type in ['per-row', 'vectorized']:
            non_global_lengths = []
            for p, is_global in zip(params, param_is_global):
                if not is_global:
                    non_global_lengths.append(len(p))
            
            if not non_global_lengths:
                raise ValueError(
                    f"At least one parameter must be non-global for "
                    f"func_type '{self.func_type}'."
                )
            
            nrows = non_global_lengths[0]
            if any(n != nrows for n in non_global_lengths[1:]):
                raise ValueError("All non-global parameters must have the same "
                                 "number of rows.")
        
        # For global/global-res functions, all params must be global
        if self.func_type in ['global', 'global-res']:
            if not all(param_is_global):
                raise ValueError(f"All parameters must be global for func_type "
                                 f"'{self.func_type}'.")
            
        ### Execute function according to func_type
        if self.func_type in ["global", "global-res", "vectorized"]:
            # Execute function once 
            results = self.func(*params)
            if not isinstance(results, tuple):
                results = (results,)

            # For vectorized, confirm that output length matches expected length
            if self.func_type == 'vectorized':
                for res in results:
                    if len(res) != nrows:
                        raise ValueError("Vectorized function output length "
                                         "does not match parameter length.")

        else: # per-row
            results = [[] for _ in self.return_names]

            # execute function separately for each data index
            for local_idx in range(nrows):
                # collect local parameters
                params_i = []
                for p, is_global in zip(params, param_is_global):
                    # only index into p for non-global parameters
                    if not is_global:
                        p = p[local_idx]
                    params_i.append(p)
                # execute function 
                results_i = self.func(*params_i)
                # collect results 
                if not isinstance(results_i, tuple):
                    results_i = (results_i,)
                for r, v in enumerate(results_i):
                    results[r].append(v)
            results = tuple([np.array(r) for r in results])

        # Confirm that output length matches return_names length, and assign 
        # to output dict
        if len(self.return_names) != len(results):
            raise ValueError("Function return length does not match "
                             "number of return names.")
        
        out = {}
        for name, val in zip(self.return_names, results):
            out[name] = val
        return out
    
    def __repr__(self):
        """
        Return a concise string representation of the pipeline step.
        
        Returns:
        str: A single-line string containing the step name and function info.
        """
        s = f"Pipeline Step: {self.name}"
        s += f", Function: {self.func.__module__}.{self.func.__name__}"
        return s
    
    def __str__(self):
        """
        Return a detailed string representation of the pipeline step.
        
        Returns:
        str: A multi-line string with step name, function, parameters, and type.
        """
        s = f"Pipeline Step: {self.name}"
        s += f"\n\tFunction: {self.func.__module__}.{self.func.__name__}"
        s += f"\n\tInput Parameters: {self.param_names}"
        s += f"\n\tOutput Parameters: {self.return_names}"
        s += f"\n\tFunction Type: {self.func_type}"
        return s

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
                # prepend only the tasks from earlier steps in this sequence
                pref = []
                for prev_key, prev_node in keys[:idx]:
                    pref.append(prev_node["task"])
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

def check_pl_tree_structure(tree, cal = False):
    """
    Recursively check that pipeline tree structure is valid.

    Parameters:
    tree (dict): Pipeline tree node to check.
    cal (bool): If True, 'params' key is not allowed. If False, 'params' 
        key is allowed. Default is False.

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
            check_pl_tree_structure(val, cal=cal)

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
        
        # Check if params key is present, and corresponds to valid dict
        if 'params' in keys:
            if cal:
                m = 'Pipeline tree "params" key is not allowed for '
                m += 'calibration pipelines.'
                raise ValueError(m)
            if 'task' not in keys:
                m = 'Pipeline tree "params" key requires '
                m += 'a corresponding "task" key.'
                raise ValueError(m)
            val = tree['params']
            if not isinstance(val, dict):
                m = 'Pipeline tree "params" value must be a dict.'
                raise ValueError(m)
            for param_name, param_val in val.items():
                if not isinstance(param_name, str):
                    m = 'Pipeline tree "params" keys must be strings.'
                    raise ValueError(m)
                # Validate that param_val is a simple type 
                # (str, bool, float, int, None)
                allowed_types = (str, bool, float, int, type(None))
                if not isinstance(param_val, allowed_types):
                    m = f'Pipeline tree "params" value for "{param_name}" '
                    m += f'must be a simple type (str, bool, float, int, None), '
                    m += f'got {type(param_val).__name__}.'
                    raise ValueError(m)
            
        # Otherwise, key should be <>_STEPS sequence 
        checked_keys = ['task', 'delete_input', 'params']
        for k in checked_keys:
            if k in keys:
                keys.remove(k)
        for key in keys:
            if not key.endswith('_STEPS'):
                m = f'Pipeline tree string keys must be in {checked_keys}'
                m += f' or end with "_STEPS". '
                m += f'Found invalid key: {key}'
                raise ValueError(m)
            check_pl_tree_structure(tree[key], cal=cal)

################################################################################ 
###### Helper functions for printing paths and cal_pl in readable formats ######
################################################################################
def print_cal_pl(cal_pl, indent = 0):
    """
    Print calibration dictionary in a readable format.

    Parameters:
    cal_pl (dict or list): The calibration dictionary to print.
    indent (int): Current indentation level for nested structures.
    """
    if isinstance(cal_pl, dict):
        for key, val in cal_pl.items():
            print(f"{'  ' * indent}{key}:")
            print_cal_pl(val, indent + 1)
    else:
        p = '\n' + cal_pl.__str__() 
        p = p.replace('\n', f'\n{"  " * (indent + 1)}')[2:]
        print(p)

def print_path(path, indent = 0):
    """
    Print PL path in a readable format.

    Parameters:
    path (list): PL path to print.
    indent (int): Current indentation level for nested structures.
    """
    for step in path:
        p = '\n' + step.__str__() 
        p = p.replace('\n', f'\n{"  " * (indent + 1)}')[2:]
        print(p)
