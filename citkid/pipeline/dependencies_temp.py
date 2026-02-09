import copy
import warnings
import numpy as np

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
    deps (dict): Run indices for input parameters and all transitive sub-dependencies.
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
        if key not in param_names:
            raise ValueError(f"Enforced parameter '{key}' not in param_names")
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
    input_deps = enforced_max_runs.copy()  
    for param in param_names:
        if param not in input_deps:
            run = get_most_recent_run(param, deps_map)
            if run == -1:
                raise ValueError(f"Missing dependencies for parameters: [{param}]")
            input_deps[param] = run
    
    # Iteratively resolve conflicts 
    input_deps0 = input_deps.copy()  # For warning message
    mod_runs = {}
    backtracked_enforced = {}
    had_conflicts = False  # Track if we had any conflicts
    
    for iteration in range(100):  # Safety limit
        # Recursively collect ALL dependencies (input + transitive)
        all_deps, requirements = _collect_all_dependencies(input_deps, deps_map)
        
        # Check for conflicts (same param at multiple runs)
        conflicts = _find_conflicts_in_deps(requirements)
        
        if not conflicts:
            break  # Done!
        
        had_conflicts = True  # Mark that we had conflicts
        
        # Resolve conflicts by backtracking input params
        backtracked = _backtrack_to_resolve_conflicts(
            input_deps, all_deps, conflicts, enforced_max_runs, deps_map
        )
        
        if not backtracked:
            break  # Can't resolve further
        
        # Track changes for warning
        for param, (old_run, new_run) in backtracked.items():
            mod_runs[param] = new_run
            if param in enforced_max_runs and new_run < enforced_max_runs[param]:
                backtracked_enforced[param] = (enforced_max_runs[param], new_run)
    
    # Final optimization: prefer earlier runs with identical dependencies
    # (for version consistency when there's no functional difference)
    # Apply to: 1) params that were backtracked, 2) non-enforced params that depend
    # on backtracked params AT THE BACKTRACKED RUN (version consistency)
    if had_conflicts and mod_runs:
        # Find params that should be considered for optimization
        params_to_optimize = set(mod_runs.keys())
        
        # Also include non-enforced params that depend on backtracked params at the backtracked run
        for param in input_deps.keys():
            if param in enforced_max_runs or param in params_to_optimize:
                continue
            try:
                sub_deps = _get_sub_deps(param, input_deps[param], deps_map)
                # Check if this param depends on any backtracked param AT the backtracked run
                for dep_name, dep_run in sub_deps.items():
                    if dep_name in mod_runs and dep_run == input_deps[dep_name]:
                        # This param depends on a backtracked param at its new run
                        params_to_optimize.add(param)
                        break
            except LookupError:
                pass
        
        # Optimize each identified param
        for param in params_to_optimize:
            current_run = input_deps[param]
            try:
                current_deps = _get_sub_deps(param, current_run, deps_map)
                # Try earlier runs to find one with same dependencies
                for earlier_run in range(current_run - 1, 0, -1):
                    try:
                        earlier_deps = _get_sub_deps(param, earlier_run, deps_map)
                        if earlier_deps == current_deps:
                            # Found earlier version with same deps, use it
                            input_deps[param] = earlier_run
                            if param not in mod_runs:
                                mod_runs[param] = earlier_run
                            current_run = earlier_run
                        else:
                            break  # Dependencies differ, stop looking
                    except LookupError:
                        break  # Param doesn't exist at this run
            except LookupError:
                pass
        
        # Recollect dependencies after optimization
        all_deps, _ = _collect_all_dependencies(input_deps, deps_map)
    
    # Warn if any runs were modified
    if mod_runs:
        msg = f"Backtracking params {list(mod_runs.keys())} from runs ["
        for param in mod_runs.keys():
            msg += f"{input_deps0[param]} -> {mod_runs[param]}, "
        msg = msg[:-2] + "] to resolve dependency conflicts."
        warnings.warn(msg)
    
    # Warn specifically about enforced params that were backtracked
    for param, (enforced, actual) in backtracked_enforced.items():
        msg = (
            f"Enforced parameter '{param}' was backtracked from "
            f"enforced run {enforced} to run {actual} to resolve conflicts"
        )
        warnings.warn(msg)
    
    return all_deps

def _get_lowest_runs(sub_deps):
    """
    Collect lowest run for each dependency and detect conflicts.
    
    Parameters
    ----------
    sub_deps : dict[str, dict[str, int]]
        Dictionary mapping param names to their dependencies.
        
    Returns
    -------
    lowest : dict[str, int]
        Lowest run for each dependency param.
    conflicts : bool
        True if any param had conflicting run requirements.
    """
    if not isinstance(sub_deps, dict):
        raise ValueError("sub_deps must be a dict")
    
    lowest = {}
    conflicts = False
    
    for param_deps in sub_deps.values():
        if not isinstance(param_deps, dict):
            raise ValueError("Each value in sub_deps must be a dict")
        for dep_name, dep_run in param_deps.items():
            if dep_name in lowest:
                if lowest[dep_name] != dep_run:
                    conflicts = True
                    lowest[dep_name] = min(lowest[dep_name], dep_run)
            else:
                lowest[dep_name] = dep_run
    
    return lowest, conflicts

def _flatten_all_deps(deps, sub_deps):
    """
    Flatten dependencies, tracking sources.
    
    Returns dict[str, list[(run, source)]] where each param maps to
    list of (run_idx, source_param) tuples showing where it appears.
    """
    flattened = {}
    
    # Add input params
    for param, run in deps.items():
        if param not in flattened:
            flattened[param] = []
        flattened[param].append((run, param))
    
    # Add transitive deps
    for source_param, param_deps in sub_deps.items():
        for dep_name, dep_run in param_deps.items():
            if dep_name not in flattened:
                flattened[dep_name] = []
            flattened[dep_name].append((dep_run, source_param))
    
    return flattened

def _find_conflicting_params(flattened):
    """
    Find params that appear at multiple different runs.
    
    Returns set of param names that have conflicts.
    """
    conflicts = set()
    for param, appearances in flattened.items():
        runs = {run for run, _ in appearances}
        if len(runs) > 1:
            conflicts.add(param)
    return conflicts

def _identify_params_to_backtrack(deps, sub_deps, conflicting_params, flattened):
    """
    Identify which input params to backtrack to resolve conflicts.
    
    For each conflicting param, backtrack input params that depend on it
    at a higher run than needed.
    """
    to_backtrack = set()
    
    for conflict in conflicting_params:
        # Get all runs this param appears at
        appearances = flattened[conflict]
        runs = sorted({run for run, _ in appearances})
        min_run = runs[0]
        
        # Find input params that use this conflict at a higher run
        for run, source in appearances:
            if run > min_run and source in deps:
                # This input param needs backtracking
                to_backtrack.add(source)
    
    return to_backtrack

def _collect_all_dependencies(input_deps, deps_map):
    """
    Recursively collect all transitive dependencies, tracking ALL requirements.
    
    Parameters:
    input_deps (dict): {param_name: run_idx} for input parameters.
    deps_map (dict): Run indices mapped to parameter dependencies.
    
    Returns:
    tuple: (all_deps, requirements)
        - all_deps (dict): {param_name: run_idx} using first encountered run
        - requirements (dict): {param_name: set of run_idx} tracking ALL requirements
    """
    all_deps = {}
    requirements = {}  # Track ALL required runs for each param
    visited = set()  # Track (param, run) to avoid infinite loops
    
    def add_param(param_name, run_idx):
        """Recursively add a parameter and its dependencies."""
        # Avoid infinite loops
        key = (param_name, run_idx)
        if key in visited:
            return
        visited.add(key)
        
        # Record this requirement
        if param_name not in requirements:
            requirements[param_name] = set()
        requirements[param_name].add(run_idx)
        
        # Keep first encountered run in all_deps
        if param_name not in all_deps:
            all_deps[param_name] = run_idx
        
        # Get and add its dependencies
        try:
            sub_deps = _get_sub_deps(param_name, run_idx, deps_map)
            for sub_param, sub_run in sub_deps.items():
                add_param(sub_param, sub_run)
        except LookupError:
            pass  # No dependencies or doesn't exist at this run
    
    # Start with input parameters
    for param_name, run_idx in input_deps.items():
        add_param(param_name, run_idx)
    
    return all_deps, requirements

def _find_conflicts_in_deps(requirements):
    """
    Find parameters that have conflicting run requirements.
    
    Parameters:
    requirements (dict): {param_name: set of required run indices}.
    
    Returns:
    dict: {param_name: set of run indices} for conflicting params only.
    """
    conflicts = {}
    for param_name, runs in requirements.items():
        if len(runs) > 1:
            conflicts[param_name] = runs
    
    return conflicts

def _backtrack_to_resolve_conflicts(input_deps, all_deps, conflicts, enforced_max_runs, deps_map):
    """
    Backtrack input parameters to resolve conflicts.
    
    Strategy: For each conflicting parameter, backtrack ALL input params
    that could be contributing to the conflict by being at too high a run.
    
    Parameters:
    input_deps (dict): {param_name: run_idx} for input parameters (will be modified).
    all_deps (dict): {param_name: run_idx} for all parameters.
    conflicts (dict): {param_name: {required runs}}.
    enforced_max_runs (dict): Enforced maximum runs.
    deps_map (dict): Dependencies map.
    
    Returns:
    dict: {param_name: (old_run, new_run)} for backtracked params.
    """
    backtracked = {}
    
    # For each conflict, find params to backtrack
    for conflict_param, conflict_runs in conflicts.items():
        min_run = min(conflict_runs)
        max_run = max(conflict_runs)
        
        # If the conflicting param IS an input param at too high a run, backtrack it
        if conflict_param in input_deps and input_deps[conflict_param] > min_run:
            old_run = input_deps[conflict_param]
            # Try to backtrack to minimum required run
            for new_run in range(min_run, old_run):
                try:
                    _get_sub_deps(conflict_param, new_run, deps_map)
                    input_deps[conflict_param] = new_run
                    backtracked[conflict_param] = (old_run, new_run)
                    break
                except LookupError:
                    continue
        
        # For other input params, check if they depend (directly or indirectly)
        # on the conflicting param at too high a run
        for input_param, input_run in list(input_deps.items()):
            if input_param == conflict_param:
                continue  # Already handled above
            
            # Check if this input param depends on the conflict at the max run
            try:
                sub_deps = _get_sub_deps(input_param, input_run, deps_map)
                if conflict_param in sub_deps and sub_deps[conflict_param] >= max_run:
                    # This input is using the conflict at too high a run
                    # Backtrack input param one step
                    old_run = input_deps[input_param]
                    new_run = old_run - 1
                    if new_run >= 0:
                        try:
                            _get_sub_deps(input_param, new_run, deps_map)
                            input_deps[input_param] = new_run
                            if input_param not in backtracked:
                                backtracked[input_param] = (old_run, new_run)
                        except LookupError:
                            pass
            except LookupError:
                pass
    
    return backtracked

