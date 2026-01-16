import copy
import warnings
import numpy as np

def get_most_recent_run(name, saved):
    """
    Find the most recent run index for a parameter name.

    Parameters:
    name (str): The parameter name to search for.
    saved (dict): A dictionary where keys are run indices and values are
        dictionaries mapping parameter name (str) to its dependencies dict.
        The dependencies dictionary maps parameter names to their run indices,
        and represents all the other parameters with run index that the given
        parameter depends on.

    Returns:
    int: Most recent run index, or -1 if not found.
    """
    if not isinstance(name, str):
        raise ValueError("Parameter name must be a string")
    if not isinstance(saved, dict):
        raise ValueError("Saved runs must be a dictionary")

    for k in sorted(saved, reverse=True):
        # Expect dict per run: {param_name: deps_dict}
        if isinstance(saved[k], dict) and name in saved[k]:
            return k
    return -1

def _get_sub_dependencies(name, run_idx, saved):
    """
    Retrieve dependencies for a parameter at a specific run index.

    Parameters:
    name (str): The parameter name whose dependencies are to be retrieved.
    run_idx (int): The run index from which to retrieve the dependencies.
    saved (dict): A dictionary where keys are run indices and values are
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
    if not isinstance(saved, dict):
        raise ValueError("Saved runs must be a dictionary")

    # Expect dict structure per run
    run_dict = saved.get(run_idx)
    if not isinstance(run_dict, dict):
        raise LookupError(f"No entry for run {run_idx}")
    if name not in run_dict:
        raise LookupError(f"No entry for {name} in run {run_idx}")
    deps = run_dict[name]
    if not isinstance(deps, dict):
        raise ValueError("Dependencies for a parameter must be a dictionary")
    return deps

def _get_lowest_runs(sub_dependencies):
    """
    Find lowest runs across sub-dependencies and detect conflicts.

    Parameters:
    sub_dependencies (dict): A dictionary where keys are parameter names and 
        values are dictionaries of dependencies for that parameter, where keys 
        are parameter names and values are their corresponding run indices. 

    Returns:
    lowest_runs (dict): Lowest run indices found across dependencies.
    conflicts (bool): True if any parameter has different runs.
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

def get_dependencies(param_names, saved):
    """
    Determine appropriate run versions and resolve dependency conflicts.

    Parameters:
    param_names (list): Parameter names (str) for which to find dependencies.
    saved (dict): A dictionary where keys are run indices and values are
        dictionaries mapping parameter name (str) to its dependencies dict.
        The dependencies dictionary maps parameter names to their run indices,
        and represents all the other parameters with run index that the given
        parameter depends on.

    Returns:
    dependencies (dict): Run indices for input parameters and sub-dependencies.
    """
    # Input validation
    if not isinstance(param_names, (list, np.ndarray)):
        raise ValueError("param_names must be a list or numpy array")
    if any(not isinstance(p, str) for p in param_names):
        raise ValueError("All elements in param_names must be strings")
    if not isinstance(saved, dict):
        raise ValueError("saved must be a dictionary")
    
    # don't overwrite saved
    saved = copy.deepcopy(saved)

    # start with most recent run for each parameter
    dependencies = {k: get_most_recent_run(k, saved) for k in param_names}
    dependencies0 = dependencies.copy()

    # Raise error if any parameters are missing from saved
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
                    sub_dependencies[name] = _get_sub_dependencies(
                        name,
                        dependencies[name],
                        saved,
                    )
                    mod_runs[name] = dependencies[name]
        lowest_runs, conflicts = _get_lowest_runs(sub_dependencies)

    # Warn if any runs were modified
    if len(mod_runs):
        msg = f"Backtracking params {list(mod_runs.keys())} from runs ["
        for r0, r1 in zip(
            [dependencies0[k] for k in mod_runs.keys()],
            mod_runs.values(),
        ):
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