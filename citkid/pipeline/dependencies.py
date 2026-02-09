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
            elif v != lowest_runs[k]:
                # Different value found - mark as conflict
                conflicts = True
                if v < lowest_runs[k]:
                    # Replace with lower value
                    lowest_runs[k] = v
    return lowest_runs, conflicts

def get_deps(param_names, deps_map, enforced_max_runs = {}):
    """
    Determine appropriate run versions and resolve dependency conflicts.

    Parameters:
    param_names (list): Parameter names (str) for which to find dependencies.
    deps_map (dict): A dictionary where keys are run indices and values are
        dictionaries mapping parameter name (str) to its dependencies dict.
        The dependencies dictionary maps parameter names to their run indices,
        and represents all the other parameters with run index that the given
        parameter depends on.
    enforced_max_runs (dict): Optional dict of parameter names to run indices
        that should be enforced as maximum runs for those parameters. Parameters 
        names that are not enforced will default to the most recent run in 
        deps_map. Backtracking below enforced values is allowed if needed to 
        resolve conflicts.

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
    if not isinstance(enforced_max_runs, dict):
        raise ValueError("enforced_max_runs must be a dictionary")
    for key, val in enforced_max_runs.items():
        if not isinstance(key, str):
            raise ValueError("Keys in enforced_max_runs must be strings")
        if not isinstance(val, int):
            raise ValueError("Values in enforced_max_runs must be integers")

    # Validate enforced runs don't exceed most recent available
    for param_name, enforced_run in enforced_max_runs.items():
        most_recent = get_most_recent_run(param_name, deps_map)
        if most_recent != -1 and enforced_run > most_recent:
            raise ValueError(
                f"Enforced run {enforced_run} for '{param_name}' exceeds "
                f"most recent available run {most_recent}"
            ) 

    # Initialize: enforced runs or most recent for each parameter
    param_deps = enforced_max_runs.copy()  
    for param in param_names:
        if param not in param_deps:
            run = get_most_recent_run(param, deps_map)
            if run == -1:
                raise ValueError(
                    f"Missing dependencies for parameters: [{param}]"
                    )
            param_deps[param] = run
    param_deps0 = param_deps.copy()



    # Raise error if any parameters are missing from deps_map
    if any(v == -1 for v in param_deps.values()):
        missing = [k for k, v in param_deps.items() if v == -1]
        raise ValueError(f"Missing dependencies for parameters: {missing}")

    # Use recursive search to find optimal solution with highest run_sum
    solution = _find_best_deps_recursive(
        param_deps, deps_map, param_names, enforced_max_runs
        )
    
    if solution is None:
        raise ValueError('dependencies could not be resolved')
    
    deps, run_sum, backtracked = solution

    # Warn if backtracking 
    if backtracked:
        msg = f"Backtracking params {list(backtracked.keys())} from runs ["
        for r0, r1 in zip(
            [param_deps0[k] for k in backtracked.keys()],
            backtracked.values(),
        ):
            msg += f"{r0} -> {r1}, "
        msg = msg[:-2] + "] to resolve dependency conflicts."
        warnings.warn(msg)

    return deps

def _find_best_deps_recursive(
        current_deps, 
        deps_map, 
        param_names, 
        enforced_max_runs, 
        memo=None
        ):
    """
    Recursively try all backtracking options and return the one with highest 
    run_sum.
    
    Parameters:
    current_deps (dict): Current parameter run assignments 
        {param_name: run_idx}.
    deps_map (dict): The full dependency map.
    param_names (list): Original input parameter names.
    enforced_max_runs (dict): Maximum run indices for specific parameters.
    memo (dict): Memoization cache.
    
    Returns:
    tuple: (all_deps_dict, run_sum, backtracked_dict) or None if no valid 
        solution.
    """
    if memo is None:
        memo = {}
    
    # State key for memoization
    state_key = tuple(sorted(current_deps.items()))
    if state_key in memo:
        return memo[state_key]
    
    # Helper to get sub-deps with error handling
    def _get_sub_deps_local(param_deps):
        sub_deps = {}
        for name, run_idx in param_deps.items():
            try:
                sub_dep = _get_sub_deps(name, run_idx, deps_map).copy()
                sub_dep[name] = run_idx
                sub_deps[name] = sub_dep
            except LookupError:
                # Parameter doesn't exist at this run
                return None
        return sub_deps
    
    # Get sub-deps and check for conflicts
    sub_deps = _get_sub_deps_local(current_deps)
    if sub_deps is None:
        # Invalid state
        memo[state_key] = None
        return None
        
    to_backtrack = _detect_conflicts(sub_deps)
    
    if not to_backtrack:
        # No conflicts! Calculate result
        all_deps = {k: v for d in sub_deps.values() for k, v in d.items()}
        
        # Check enforced_max_runs constraints
        for param, max_run in enforced_max_runs.items():
            if param in all_deps and all_deps[param] > max_run:
                # Violates enforced max - invalid solution
                memo[state_key] = None
                return None
        
        run_sum = sum(all_deps.values())
        # Track which params were backtracked (empty dict for this leaf)
        result = (all_deps, run_sum, {})
        memo[state_key] = result
        return result
    
    # Has conflicts - try backtracking each possible param in the list
    best_solution = None
    best_run_sum = -1
    
    for param in to_backtrack:
        # Try backtracking this param
        new_deps = current_deps.copy()
        new_run = current_deps[param] - 1
        
        # Find valid run (skip missing runs)
        while new_run >= 0:
            if new_run in deps_map and param in deps_map[new_run]:
                break
            new_run -= 1
        
        if new_run < 0:
            continue  # Can't backtrack further
        
        new_deps[param] = new_run
        
        # Recursively solve from this new state
        solution = _find_best_deps_recursive(
            new_deps, deps_map, param_names, enforced_max_runs, memo
            )
        
        if solution is not None:
            deps_result, run_sum, child_backtracked = solution
            if run_sum > best_run_sum:
                best_run_sum = run_sum
                # Update backtracked dict to include this param
                updated_backtracked = child_backtracked.copy()
                if param not in updated_backtracked:
                    updated_backtracked[param] = new_run
                best_solution = (deps_result, run_sum, updated_backtracked)
    
    memo[state_key] = best_solution
    return best_solution


def _detect_conflicts(sub_deps):
    """
    Detect conflicts across sub-dependencies and identify parameters to 
    backtrack.
    
    Parameters:
    sub_deps (dict): A dictionary where keys are parameter names and 
        values are dictionaries of dependencies for that parameter, where keys 
        are parameter names and values are their corresponding run indices.
        
    Returns:
    list: List of parameter names that should be backtracked to resolve 
        conflicts. If no conflicts are detected, returns an empty list.
    """
    flattened = {} # name: (param, run_idx) 
    to_backtrack = [] # list of param names to backtrack
    for param, sub_dep in sub_deps.items():
        for k, v in sub_dep.items():
            # k not in flattened: add key
            if k not in flattened: 
                flattened[k] = (param, v) 
            # current run_idx < flattened run_idx -> return flattened name
            elif v < flattened[k][1]: 
                to_backtrack.append(flattened[k][0])
            # flattened run_idx < current run_idx -> return current param
            elif v > flattened[k][1]:
                to_backtrack.append(param)
    # no conflicts -> return None
    return to_backtrack