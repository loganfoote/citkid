import copy

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