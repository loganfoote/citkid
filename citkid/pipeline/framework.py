import numpy as np
from .pl_steps import find_pl_path

################################################################################
############################### Lazy Attribute #################################
################################################################################
class LazyAttr:
    def __init__(self, ds, name):
        """
        Class to represent a lazily-loaded attribute of a DataSet.

        Parameters:
        ds (DataSet): The DataSet instance this attribute belongs to.
        name (str): The name of the attribute.
        """
        self.ds = ds
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
        path = find_pl_path(self.ds.cal_pl, self.name)
        if path is None:
            raise AttributeError(f"No processing path for {self.name}")

        # execute_path handles multiple indices at once
        # return shape: (len(missing), ...)
        self.ds.execute_path(path, missing)

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
            rows = list(range(*key.indices(self.ds.nres)))
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
            rows = list(range(*key.indices(self.ds.nres)))
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