import numpy as np
import zarr
import copy

def create_zarr_param(root, name, value, nres, dtype = None):
    """
    Create a Zarr array suitable for row-wise writes along the last axis.
 
    Parameters:
    root (zarr.core.group.Group): zarr file root.
    name (str): parameter name.
    value (np.ndarray): data to write, corresponding to idx.
    nres (int): number of resonators in the dataset.
    dtype (type or None): data type, or None to inherit the data type from
        value.
 
    Returns:
    None
    """
    if name in root:
        return
 
    value = np.asarray(value)
 
    if dtype is None:
        dtype = value.dtype
 
    # data dims + row dimension
    shape = (nres, *value.shape)
    chunks = (1, *value.shape)
 
    root.create_array(name, shape = shape, chunks = chunks, dtype = dtype)
 
 
def write_zarr_row(root, name, idx, value):
    """
    Write a single row (chunk) into an existing Zarr array.
 
    Parameters:
    root (zarr.core.group.Group): zarr file root.
    name (str): parameter name.
    idx (int): data index.
    value (np.ndarray): data to write, corresponding to idx.
 
    Returns:
    None
    """
    arr = root[name]
    value = np.asarray(value)
 
    # Optional but strongly recommended safety check
    expected_shape = arr.shape[1:]
    if value.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: expected {expected_shape}, got {value.shape}"
        )
 
    arr[idx] = value
 
 
def write_zarr(root, name, idx, value, nres, dtype = None):
    """
    Create the array if needed and write one row.
 
    Parameters:
    root (zarr.core.group.Group): zarr file root.
    name (str): parameter name.
    idx (int): data index.
    value (np.ndarray): data to write, corresponding to idx.
    nres (int): number of resonators in the dataset.
    dtype (type or None): data type, or None to inherit the data type from
        value.
 
    Returns:
    None
    """
    create_zarr_param(root, name, value, nres, dtype=dtype)
    write_zarr_row(root, name, idx, value)
 
def write_single_array(root, name, value, dtype = None):
    """
    Write a single un-chunked array to a zarr file.
 
    Parameters:
    root (zarr.core.group.Group): zarr file root.
    name (str): parameter name.
    value (np.ndarray): data to write, corresponding to idx.
    dtype (type or None): data type, or None to inherit the data type from
        value.
 
    Returns:
    None
    """
    if dtype is None:
        dtype = value.dtype
 
    value = np.asarray(value, dtype = dtype)
 
    if name in root:
        arr = root[name]
        arr[...] = value
        return
 
    root.create_array(name, data = value) # no chunking
    
    
def deep_union(a, b):
    """
    Returns the union of two dictionaries a and b, both of which
    may have nested sub-dictionaries.
    This function is used for merging dependencies dictionaries
    when writing to DataSet.deps_map.
    
    Parameters:
    a (dictionary): Dictionary 1
    b (dictionary): Dictionary 2
    
    Returns:
    out (dictionary): Merged dictionary
    """
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_union(out[k], v)
        else:
            out[k] = v
    return out