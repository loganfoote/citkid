import copy
import warnings

def _get_most_recent_run(name, saved): 
    """
    Given a parameter name and saved runs dictionary, finds the most recent run 
    index where that parameter was exists. Returns the run index, or -1 if not 
    found.

    Parameters:
    name (str): The parameter name to search for.
    saved (dict): A dictionary where keys are run indices and values are lists
        of tuples (parameter name, dependencies dict). The dependencies 
        dictionary maps parameter names to their run indices, and represents 
        all the other parameters with run index that the given parameter 
        depends on.

    Returns:
    int: The most recent run index where the parameter exists, or -1 if not 
        found.
    """
    if not isinstance(name, str):
        raise ValueError("Parameter name must be a string") 
    if not isinstance(saved, dict):
        raise ValueError("Saved runs must be a dictionary")

    for k in sorted(saved, reverse = True):
        if any(t[0] == name for t in saved[k]):
            return k
    return -1

def _get_sub_dependencies(name, run_idx, saved):
    """
    Given a parameter name, a run index, and a saved runs dictionary, retrieves 
    the dependencies for that parameter from the saved runs and returns as a 
    dictionary.

    Parameters:
    name (str): The parameter name whose dependencies are to be retrieved.
    run_idx (int): The run index from which to retrieve the dependencies.
    saved (dict): A dictionary where keys are run indices and values are lists
        of tuples (parameter name, dependencies dict). The dependencies 
        dictionary maps parameter names to their run indices, and represents 
        all the other parameters with run index that the given parameter 
        depends on.

    Returns:
    dict: A dictionary of dependencies for the given parameter, where keys are
        parameter names and values are their corresponding run indices.
    """
    if not isinstance(name, str):
        raise ValueError("Parameter name must be a string")
    if not isinstance(run_idx, int):
        raise ValueError("Run index must be an integer")
    if not isinstance(saved, dict):
        raise ValueError("Saved runs must be a dictionary")

    prev_deps = [s for s in saved[run_idx] if s[0] == name] 
    if len(prev_deps) == 0:
        raise LookupError(f"No entry for {name} in run {run_idx}")
    if len(prev_deps) != 1:
        raise ValueError(f"Multiple entries for {name} in run {run_idx}")
    # prev_deps is a list with one element: (name, dependencies dict)
    return prev_deps[0][1] 

def _get_lowest_runs(sub_dependencies):
    """
    Given a dictionary of sub_dependencies (name: {param: run}), finds the 
    lowest run for each parameter across all sub_dependencies. Also returns 
    boolean conflicts, which is True if any parameter has different runs in 
    different sub_dependencies. 

    Parameters:
    sub_dependencies (dict): A dictionary where keys are parameter names and 
        values are dictionaries of dependencies for that parameter, where keys 
        are parameter names and values are their corresponding run indices. 

    Returns:
    lowest_runs (dict): A dictionary where keys are parameter names and values 
        are the lowest run indices found across all sub_dependencies. 
    conflicts (bool): True if any parameter has different runs in different 
        sub_dependencies, False otherwise.
    """
    if not isinstance(sub_dependencies, dict):
        raise ValueError("Sub-dependencies must be a dictionary")
    for d in sub_dependencies.values():
        if not isinstance(d, dict):
            raise ValueError("Each sub-dependency must be a dictionary")
    
    lowest_runs, conflicts = {}, False
    for d in sub_dependencies.values(): 
        for k, v in d.items():
            if k not in lowest_runs:
                # append if new
                lowest_runs[k] = v
            elif v < lowest_runs[k]:
                # replace if lower, and set conflicts to True
                lowest_runs[k] = v
                conflicts = True
    return lowest_runs, conflicts

def _get_dependencies(param_names, saved):
    """
    Given a list of function parameter names and saved runs, determines the 
    appropriate run versions for each input parameter by first selecting the 
    most recent run number, then reducing run numbers if necessary to resolve 
    conflicts in dependencies. Returns a dictionary of all dependencies 
    (including param_names and sub-dependencies of parameters).

    Parameters:
    param_names (list): A list of parameter names (str) for which to find 
        dependencies.
    saved (dict): A dictionary where keys are run indices and values are lists
        of tuples (parameter name, dependencies dict). The dependencies 
        dictionary maps parameter names to their run indices, and represents 
        all the other parameters with run index that the given parameter
        depends on.

    Returns:
    dependencies (dict): A dictionary where keys are parameter names and values 
        are their corresponding run indices, including both the input parameters 
        and their sub-dependencies.
    """
    # don't overwrite saved
    saved = copy.deepcopy(saved)

    # start with most recent run for each parameter
    dependencies = {k: _get_most_recent_run(k, saved) for k in param_names}
    dependencies0 = dependencies.copy()

    # raise error if any parameters are missing from saved 
    if any(v == -1 for v in dependencies.values()):
        missing = [k for k, v in dependencies.items() if v == -1]
        raise ValueError(f"Missing dependencies for parameters: {missing}")

    # Get sub-dependencies for each parameter
    sub_dependencies = {}
    for name, run_idx in dependencies.copy().items():
        prev_deps = _get_sub_dependencies(name, run_idx, saved)
        sub_dependencies[name] = prev_deps

    # Resolve conflicts in sub-dependencies by backtracking runs
    lowest_runs, conflicts = _get_lowest_runs(sub_dependencies)
    conflicts = True
    mod_runs = {}
    while conflicts:
        for name, dep in sub_dependencies.items():
            for n, run_idx in dep.items():
                if lowest_runs[n] < run_idx:
                    # backtrack this dependency
                    dependencies[name] -= 1 
                    sub_dependencies[name] = _get_sub_dependencies(name, 
                                                    dependencies[name], saved)
                    mod_runs[name] = dependencies[name]
        lowest_runs, conflicts = _get_lowest_runs(sub_dependencies)

    # Warn if any runs were modified
    if len(mod_runs):
        msg = f"Backtracking params {list(mod_runs.keys())} from runs ["
        for r0, r1 in zip([dependencies0[k] for k in mod_runs.keys()], 
                           mod_runs.values()):
            msg += f"{r0} -> {r1}, "
        msg = msg[:-2] + "] to resolve dependency conflicts."
        warnings.warn(msg)

    # Sanity check: ensure no input names depend on subfunction input names
    if any([k in dependencies.keys() for k in lowest_runs.keys()]):
        m = "Function input names cannot depend on subfunction input names"
        raise ValueError(m)

    # Merge sub-dependencies of each parameter into dependencies
    dependencies.update(lowest_runs)
    return dependencies