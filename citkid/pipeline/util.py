import os
import subprocess
import sys
from collections import defaultdict

def open_in_file_explorer(path):
    """
    Opens the given path in the system's file explorer.
    
    Parameters:
    path (str): The path to open in the file explorer.
    
    Raises:
    RuntimeError: If the operating system is unsupported.
    """
    path = os.path.abspath(path)
    # Ensure the path exists before attempting to open it.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' not found")

    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", path])
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", path])
    else:
        raise RuntimeError("Unsupported OS")


def group_unique_tuples(tuples):
    """
    Group tuples of (hashable, dict) by unique values, comparing dicts by content.
    
    Returns both the unique tuples and the indices in the original list where
    each unique tuple appears. The first element can be an int, tuple, or any 
    hashable type.
    
    Parameters:
    tuples (list): List of (hashable, dict) tuples to group.
    
    Returns:
    tuple: (unique_tuples, indices) where:
        - unique_tuples (list): List of unique (hashable, dict) tuples
        - indices (list of lists): indices[i] contains all original indices 
          that match unique_tuples[i]
    
    Example:
    >>> tuples = [(1, {'a': 1}), (2, {'b': 2}), (1, {'a': 1})]
    >>> unique, indices = group_unique_tuples(tuples)
    >>> unique
    [(1, {'a': 1}), (2, {'b': 2})]
    >>> indices
    [[0, 2], [1]]
    """
    # Map each unique key to its indices
    groups = defaultdict(list)
    for idx, (key_val, d) in enumerate(tuples):
        # Use frozenset of items to make dict hashable by content
        # key_val can be int, tuple, or any hashable type
        key = (key_val, frozenset(d.items()))
        groups[key].append(idx)
    
    # Convert to lists
    unique_tuples = [tuples[idx_list[0]] for idx_list in groups.values()]
    indices = list(groups.values())
    
    return unique_tuples, indices
