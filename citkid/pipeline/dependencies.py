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

def _flatten_all_deps(deps, sub_deps):
    """
    Flatten all dependencies into a single view showing all parameters and their 
    run indices.
    
    Parameters:
    deps (dict): Input parameter names mapped to their run indices.
    sub_deps (dict): Input parameters mapped to their dependencies
        {dep_name: dep_run}.
    
    Returns:
    dict: {param_name: [(run_idx, source), ...]} where source is the input param 
        that requires it.
    """
    flattened = {}
    
    # Add all sub-dependencies
    for source, deps_dict in sub_deps.items():
        for param_name, run_idx in deps_dict.items():
            if param_name not in flattened:
                flattened[param_name] = []
            flattened[param_name].append((run_idx, source))
    
    # Also add input parameters themselves (they might appear in sub_deps)
    for param_name, run_idx in deps.items():
        if param_name not in flattened:
            flattened[param_name] = []
        flattened[param_name].append((run_idx, param_name))
    
    return flattened

def _find_conflicting_params(flattened_deps):
    """
    Find parameters that have conflicting run indices.
    
    Parameters:
    flattened_deps (dict): Output from _flatten_all_deps.
    
    Returns:
    set: Parameter names that have different run indices.
    """
    conflicts = set()
    for param_name, runs_list in flattened_deps.items():
        if len(runs_list) > 1:
            # Check if all run indices are the same
            run_indices = [r for r, _ in runs_list]
            if len(set(run_indices)) > 1:
                conflicts.add(param_name)
    return conflicts

def _identify_params_to_backtrack(
        deps, sub_deps, conflicting_params, flattened_deps
        ):
    """
    Identify which input parameters need to be backtracked to resolve conflicts.
    
    Parameters:
    deps (dict): Input parameter names mapped to their run indices.
    sub_deps (dict): Input parameters mapped to their dependencies.
    conflicting_params (set): Parameter names with conflicts.
    flattened_deps (dict): Output from _flatten_all_deps.
    
    Returns:
    set: Input parameter names that should be backtracked.
    """
    to_backtrack = set()
    
    for conflict_param in conflicting_params:
        # Find the lowest required run for this parameter
        runs_list = flattened_deps[conflict_param]
        lowest_run = min(r for r, _ in runs_list)
        
        # Find all sources that require a higher run
        for run_idx, source in runs_list:
            if run_idx > lowest_run and source in deps:
                to_backtrack.add(source)
    
    # Also backtrack parameters that depend on backtracked parameters
    # This handles the case where one input depends on another input
    changed = True
    while changed:
        changed = False
        for param_name, param_deps in sub_deps.items():
            if param_name not in to_backtrack and param_name in deps:
                # Check if this parameter depends on any backtracked parameter
                for dep_name in param_deps.keys():
                    if dep_name in to_backtrack:
                        to_backtrack.add(param_name)
                        changed = True
                        break
    
    return to_backtrack

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
    mod_runs = {}
    max_iterations = 100  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Flatten all dependencies to see the full picture
        flattened = _flatten_all_deps(deps, sub_deps)
        
        # Find parameters with conflicting run indices
        conflicts = _find_conflicting_params(flattened)
        
        if not conflicts:
            break  # No conflicts, we're done
        
        # Identify which input parameters need to be backtracked
        to_backtrack = _identify_params_to_backtrack(
            deps, sub_deps, conflicts, flattened
            )
        
        if not to_backtrack:
            # No input parameters can be backtracked, something is wrong
            break
        
        # Backtrack the identified parameters
        for param_name in to_backtrack:
            deps[param_name] -= 1
            sub_deps[param_name] = _get_sub_deps(
                param_name,
                deps[param_name],
                deps_map,
            )
            mod_runs[param_name] = deps[param_name]

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

    # Merge all dependencies - add any from sub_deps that aren't already in deps
    flattened = _flatten_all_deps(deps, sub_deps)
    for param_name in flattened.keys():
        if param_name not in deps:
            # Get the run index 
            # (should all be the same after conflict resolution)
            runs = [r for r, _ in flattened[param_name]]
            deps[param_name] = runs[0]
    
    return deps