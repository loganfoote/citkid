import copy
import os
import subprocess
import sys

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

def open_in_file_explorer(path):
    """
    Opens the given path in the system's file explorer.
    
    Parameters:
    path (str): The path to open in the file explorer.
    
    Raises:
    RuntimeError: If the operating system is unsupported.
    """
    path = os.path.abspath(path)
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", path])
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", path])
    else:
        raise RuntimeError("Unsupported OS") 