import copy
import warnings
import numpy as np

def print_deps_map(deps_map):
    """
    Prints a user-readable diagram of a dependency map.
    """
    return
    

def get_most_recent_run(name, deps_map):
    """
    Find the most recent run index for a parameter name.

    Parameters:
    name (str): The parameter name to search for.
    deps_map (dict): A dictionary where keys are run indices and values are
        dictionaries mapping parameter name (str) to its dependencies dict.
        The dependencies dictionary maps parameter names to their run indices,
        and represents all the other parameters with run index that the given
        parameter depends on.

    Returns:
    int: Most recent run index, or -1 if not found.
    """
    if not isinstance(name, str):
        raise ValueError("Parameter name must be a string")
    if not isinstance(deps_map, dict):
        raise ValueError("deps_map must be a dictionary")

    for k in sorted(deps_map, reverse=True):
        # Expect dict per run: {param_name: deps_dict}
        if isinstance(deps_map[k], dict) and name in deps_map[k]:
            return k
    return -1

def _get_sub_deps(name, run_idx, deps_map):
    """
    Retrieve dependencies for a parameter at a specific run index.

    Parameters:
    name (str): The parameter name whose dependencies are to be retrieved.
    run_idx (int): The run index from which to retrieve the dependencies.
    deps_map (dict): A dictionary where keys are run indices and values are
        dictionaries mapping parameter name (str) to its dependencies dict.
        The dependencies dictionary maps parameter names to their run indices,
        and represents all the other parameters with run index that the given
        parameter depends on.

    Returns:
    dict: Dependencies for the parameter with run indices.
    """
    if not isinstance(name, str):
        raise ValueError("Parameter name must be a string")
    if not isinstance(run_idx, int):
        raise ValueError("Run index must be an integer")
    if not isinstance(deps_map, dict):
        raise ValueError("deps_map must be a dictionary")

    # Expect dict structure per run
    run_dict = deps_map.get(run_idx)
    if not isinstance(run_dict, dict):
        raise LookupError(f"No entry for run {run_idx}")
    if name not in run_dict:
        raise LookupError(f"No entry for {name} in run {run_idx}")
    deps = run_dict[name]
    if not isinstance(deps, dict):
        raise ValueError("Dependencies for a parameter must be a dictionary")
    return deps

def _get_lowest_runs(sub_deps):
    """
    Find lowest runs across sub-dependencies and detect conflicts.

    Parameters:
    sub_deps (dict): A dictionary where keys are parameter names and 
        values are dictionaries of dependencies for that parameter, where keys 
        are parameter names and values are their corresponding run indices. 

    Returns:
    lowest_runs (dict): Lowest run indices found across dependencies.
    conflicts (bool): True if any parameter has different runs.
    """
    if not isinstance(sub_deps, dict):
        raise ValueError("Sub-deps must be a dictionary")
    for d in sub_deps.values():
        if not isinstance(d, dict):
            raise ValueError("Each sub-dependency must be a dictionary")
    
    lowest_runs, conflicts = {}, False
    for d in sub_deps.values():
        for k, v in d.items():
            if k not in lowest_runs:
                # append if new
                lowest_runs[k] = v
            elif v < lowest_runs[k]:
                # replace if lower, and set conflicts to True
                lowest_runs[k] = v
                conflicts = True
    return lowest_runs, conflicts

def get_deps(param_names, deps_map):
    """
    Determine appropriate run versions and resolve dependency conflicts.

    Parameters:
    param_names (list): Parameter names (str) for which to find dependencies.
    deps_map (dict): A dictionary where keys are run indices and values are
        dictionaries mapping parameter name (str) to its dependencies dict.
        The dependencies dictionary maps parameter names to their run indices,
        and represents all the other parameters with run index that the given
        parameter depends on.

    Returns:
    deps (dict): Run indices for input parameters and sub-dependencies.
    """
    # Input validation
    if not isinstance(param_names, (list, np.ndarray)):
        raise ValueError("param_names must be a list or numpy array")
    if any(not isinstance(p, str) for p in param_names):
        raise ValueError("All elements in param_names must be strings")
    if not isinstance(deps_map, dict):
        raise ValueError("deps_map must be a dictionary")
    
    # don't overwrite deps_map
    deps_map = copy.deepcopy(deps_map)

    # start with most recent run for each parameter
    deps = {k: get_most_recent_run(k, deps_map) for k in param_names}
    deps0 = deps.copy()

    # Raise error if any parameters are missing from deps_map
    if any(v == -1 for v in deps.values()):
        missing = [k for k, v in deps.items() if v == -1]
        raise ValueError(f"Missing dependencies for parameters: {missing}")

    # Get sub-dependencies for each parameter
    sub_deps = {}
    for name, run_idx in deps.copy().items():
        prev_deps = _get_sub_deps(name, run_idx, deps_map)
        sub_deps[name] = prev_deps

    # Resolve conflicts in sub-dependencies by backtracking runs
    lowest_runs, conflicts = _get_lowest_runs(sub_deps)
    conflicts = True
    mod_runs = {}
    while conflicts:
        for name, dep in sub_deps.items():
            for n, run_idx in dep.items():
                if lowest_runs[n] < run_idx:
                    # backtrack this dependency
                    deps[name] -= 1 
                    sub_deps[name] = _get_sub_deps(
                        name,
                        deps[name],
                        deps_map,
                    )
                    mod_runs[name] = deps[name]
        lowest_runs, conflicts = _get_lowest_runs(sub_deps)

    # Warn if any runs were modified
    if len(mod_runs):
        msg = f"Backtracking params {list(mod_runs.keys())} from runs ["
        for r0, r1 in zip(
            [deps0[k] for k in mod_runs.keys()],
            mod_runs.values(),
        ):
            msg += f"{r0} -> {r1}, "
        msg = msg[:-2] + "] to resolve dependency conflicts."
        warnings.warn(msg)

    # Sanity check: ensure no input names depend on subfunction input names
    if any([k in deps.keys() for k in lowest_runs.keys()]):
        m = "Function input names cannot depend on subfunction input names"
        raise ValueError(m)

    # Merge sub-dependencies of each parameter into dependencies
    deps.update(lowest_runs)
    return deps