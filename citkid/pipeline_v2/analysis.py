import os
import warnings

import numpy as np
import yaml
from tqdm.auto import tqdm

from . import default_steps
from . import framework as pf
from .dataset import _convert_yaml_to_steps, _read_text_file


_ANALYSIS_YAML_ALIASES = {
    "iq": "iq_analysis.yaml",
    "ts": "ts_analysis.yaml",
    "ts_offres": "ts_offres_analysis.yaml",
}

_MASK_SHAPE_SOURCES = {
    "gain_mask": "fg",
    "xcal_mask": "ff",
    "iq_mask": "ff",
    "circ_mask": "ff",
}


class AnalysisRunner:
    """
    Runner for executing analysis steps on a pipeline_v2 DataSet.

    This version keeps only one active set of outputs. Re-running an earlier
    step invalidates downstream products rather than creating new run folders.
    """

    def __init__(self, DS, analysis_yaml_path=None, custom_path=None):
        """
        Initialize the analysis runner and load its pipeline definition.

        Parameters:
        DS (DataSet): Dataset that stores inputs and outputs for the analysis
            pipeline.
        analysis_yaml_path (str or None): Path or alias for the analysis YAML.
            When None, an embedded analysis definition is used if present.
        custom_path (str or None): Path to the Python file defining
            ``custom_analysis_steps``.

        Raises:
        ValueError: If the supplied analysis definition is invalid or conflicts
            with the embedded dataset definition.
        """
        self.DS = DS

        resolved = self._resolve_analysis_definition(analysis_yaml_path, custom_path)
        self.analysis_yaml_path = resolved["yaml_path"]
        self.custom_path = resolved["custom_path"]
        self.analysis_yaml_text = resolved["yaml_text"]
        self.analysis_custom_source = resolved["custom_source"]

        self.analysis_steps = list(_load_custom_analysis_steps_from_source(self.analysis_custom_source))
        for step in default_steps.default_analysis_steps:
            if step.name not in [s.name for s in self.analysis_steps]:
                self.analysis_steps.append(step)

        self.path = []
        self.step_indices = {}
        if self.analysis_yaml_text is not None:
            yaml_dict = yaml.safe_load(self.analysis_yaml_text) or {}
            self.analysis_pl = _convert_yaml_to_steps(yaml_dict, self.analysis_steps)
            if list(self.analysis_pl.keys()) != ["ANALYSIS_STEPS"]:
                raise ValueError("analysis YAML must contain only 'ANALYSIS_STEPS' key")
            path_dict = self.analysis_pl["ANALYSIS_STEPS"]
            max_task = _validate_task_idxs(path_dict.keys())
            self.path = [path_dict[i] for i in range(1, max_task + 1)]
            self.step_indices = {
                step_dict["task"].name: index
                for index, step_dict in enumerate(self.path, start=1)
            }
            self.DS.set_analysis_step_names({index: step_dict["task"].name for index, step_dict in enumerate(self.path, start=1)})

    def execute_path(self, data_idx=None, start_from_idx=0, verbose=True, save_override=None, execution_mode='vectorized'):
        """
        Execute the loaded analysis path in order.

        Parameters:
        data_idx (int, array-like, or None): Rows to process for per-row and
            vectorized steps. When None, all rows are processed.
        start_from_idx (int): Zero-based index into ``self.path`` from which to
            begin execution.
        verbose (bool): If True, show a progress bar.
        save_override (bool or None): If not None, overrides the YAML ``save``
            flag for every executed step.
        execution_mode (str): How to execute vectorized steps. Can be 'vectorized'
            (default, loads all data at once) or 'per-row' (loops over each
            data_idx one at a time, using less memory).
        """
        if execution_mode not in ('vectorized', 'per-row'):
            raise ValueError(f"execution_mode must be 'vectorized' or 'per-row', got '{execution_mode}'")
        path_iter = self.path[start_from_idx:]
        if verbose:
            path_iter = tqdm(
                path_iter,
                leave=False,
                bar_format="{desc}: {n_fmt}/{total_fmt}  |{bar}|",
            )
        for step_dict in path_iter:
            if verbose:
                path_iter.set_description(f"Executing step: {step_dict['task'].name}")
            step = step_dict["task"]
            params = step_dict.get("params", {})
            save = step_dict.get("save", True) if save_override is None else save_override
            step_data_idx = None if step.func_type in ("global", "global-res") else data_idx
            self.execute_step(step, data_idx=step_data_idx, user_params=params, save=save, execution_mode=execution_mode)

    def execute_step(self, step, data_idx=None, user_params=None, save=False, execution_mode='vectorized'):
        """
        Execute a single analysis or calibration step.

        Parameters:
        step (plStep): Step to execute.
        data_idx (int, array-like, or None): Rows to process for per-row and
            vectorized steps.
        user_params (dict, None, or 'from_yaml'): Explicit user parameters for
            the step, or the sentinel ``'from_yaml'`` to reuse parameters from
            the loaded analysis YAML.
        save (bool): If True, persist outputs immediately after execution.
        execution_mode (str): How to execute this step. Can be 'vectorized'
            (default, loads all data at once) or 'per-row' (loops over each
            data_idx one at a time). 'per-row' is useful for memory-constrained
            scenarios where vectorized execution would load too much data.

        Raises:
        ValueError: If inputs are missing or step/data_idx constraints are
            violated.
        """
        if execution_mode not in ('vectorized', 'per-row'):
            raise ValueError(f"execution_mode must be 'vectorized' or 'per-row', got '{execution_mode}'")
        if user_params == "from_yaml":
            user_params = self._get_yaml_params(step)
        if user_params is None:
            user_params = {}

        if step.func_type in ("per-row", "vectorized") and data_idx is None:
            data_idx = np.arange(int(self.DS.nrows), dtype=np.int32)
        elif step.func_type in ("global", "global-res") and data_idx is not None:
            raise ValueError(
                f"data_idx must be None for global func_type '{step.func_type}'"
            )

        pipeline_scope, step_index = self._resolve_step_scope(step)
        self._invalidate_downstream(pipeline_scope, step_index)

        user_params = self._expand_none_masks(user_params, step.func_type, data_idx)
        if user_params:
            self._add_user_params(
                step,
                user_params,
                data_idx=data_idx,
                save=save,
                pipeline_scope=pipeline_scope,
                step_index=step_index,
            )

        self._ensure_inputs_exist(
            step,
            data_idx=data_idx,
            pipeline_scope=pipeline_scope,
            step_index=step_index,
            save=save,
            execution_mode=execution_mode,
        )
        failures = self.DS._execute_step(
            step,
            data_idx=data_idx,
            save=save,
            pipeline_scope=pipeline_scope,
            step_index=step_index,
            execution_mode=execution_mode,
        )
        self._last_failures = failures
        if failures:
            warnings.warn(
                f"Step '{step.name}' failed for {len(failures)} row(s): {sorted(failures)}",
                RuntimeWarning,
                stacklevel=2,
            )

    def save_step_outputs(self, step, data_idx=None):
        """
        Persist the latest outputs of a step without re-running it.

        Parameters:
        step (plStep): Step whose outputs should be written.
        data_idx (int, array-like, or None): Rows to save for per-row outputs.
        """
        self.DS.save_step_outputs(step, data_idx=data_idx)

    def _invalidate_downstream(self, pipeline_scope, step_index):
        """
        Delete products from steps that depend on the current execution point.

        Parameters:
        pipeline_scope (str or None): Scope of the step about to run.
        step_index (int or None): Execution index of the step about to run.
        """
        if step_index is None or pipeline_scope is None:
            return
        self.DS.invalidate_after(pipeline_scope, step_index)

    def _ensure_inputs_exist(self, step, data_idx, pipeline_scope, step_index, save, execution_mode='vectorized'):
        """
        Ensure that every input required by a step is available.

        Parameters:
        step (plStep): Step whose inputs should be validated.
        data_idx (int, array-like, or None): Rows being processed.
        pipeline_scope (str or None): Scope of the step being executed.
        step_index (int or None): Execution index of the step.
        save (bool): Save flag propagated when earlier analysis steps must be
            executed to satisfy dependencies.
        execution_mode (str): Execution mode to use for prerequisite steps.

        Raises:
        ValueError: If an input cannot be loaded or produced.
        """
        for param_name in step.param_names:
            if param_name == "data_idx":
                continue

            param_scope_data_idx = None if step.func_type in ("global", "global-res") else data_idx
            if self._param_available(param_name, param_scope_data_idx):
                continue

            if pf.find_pl_path(self.DS.cal_pl, param_name) is not None:
                self.DS._ensure_loaded(param_name, data_idx=param_scope_data_idx)
                continue

            if pipeline_scope == "analysis" and step_index is not None:
                self._ensure_analysis_param(param_name, step_index, param_scope_data_idx, save=save, execution_mode=execution_mode)
                if self._param_available(param_name, param_scope_data_idx):
                    continue

            raise ValueError(
                f"Step '{step.name}' requires parameter '{param_name}', but it is not available"
            )

    def _ensure_analysis_param(self, param_name, before_step_index, data_idx, save, execution_mode='vectorized'):
        """
        Produce an analysis parameter by re-running earlier analysis steps.

        Parameters:
        param_name (str): Required parameter name.
        before_step_index (int): First step index that is not allowed to run.
        data_idx (int, array-like, or None): Rows needed for per-row steps.
        save (bool): Save flag to propagate to any executed prerequisite steps.
        execution_mode (str): Execution mode to use for prerequisite steps.
        """
        producer_idx = None
        for index, step_dict in enumerate(self.path, start=1):
            if index >= before_step_index:
                break
            if param_name in step_dict["task"].return_names:
                producer_idx = index
                break
        if producer_idx is None:
            return

        for index, step_dict in enumerate(self.path[:producer_idx], start=1):
            step = step_dict["task"]
            step_data_idx = None if step.func_type in ("global", "global-res") else data_idx
            if self._step_outputs_exist(step, step_data_idx):
                continue
            params = step_dict.get("params", {})
            self.execute_step(step, data_idx=step_data_idx, user_params=params, save=save, execution_mode=execution_mode)

    def _step_outputs_exist(self, step, data_idx):
        """
        Check whether a step's outputs are already available.

        Parameters:
        step (plStep): Step whose outputs should be checked.
        data_idx (int, array-like, or None): Relevant rows for per-row steps.

        Returns:
        bool: True if all outputs exist.
        """
        for name in step.return_names:
            if not self._param_available(name, data_idx if step.func_type not in ("global", "global-res") else None):
                return False
        return True

    def _param_available(self, name, data_idx):
        """
        Check whether a parameter exists for the requested scope or rows.

        Parameters:
        name (str): Parameter name.
        data_idx (int, array-like, or None): Rows to verify for per-row
            parameters.

        Returns:
        bool: True if the parameter is available.
        """
        if name not in self.DS._param_meta:
            return False
        meta = self.DS._param_meta[name]
        if meta["global"]:
            return self.DS._has_global(name)
        if data_idx is None:
            return name in self.DS.root or name in self.DS._per_row_cache
        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        return self.DS._has_rows(name, rows)

    def _get_yaml_params(self, step):
        """
        Retrieve user parameters for a step from the loaded analysis YAML.

        Parameters:
        step (plStep): Step whose YAML parameters should be returned.

        Returns:
        dict: Parameter mapping defined in the YAML.

        Raises:
        ValueError: If the step is not present in the loaded path.
        """
        for step_dict in self.path:
            if step_dict["task"].name == step.name:
                return dict(step_dict.get("params", {}) or {})
        raise ValueError(f"Step '{step.name}' not found in analysis path")

    def _expand_none_masks(self, user_params, func_type, data_idx):
        """
        Replace ``None`` mask parameters with full-True arrays.

        Parameters:
        user_params (dict): User parameter mapping.
        func_type (str): Step function type.
        data_idx (int, array-like, or None): Rows being processed.

        Returns:
        dict: User parameter mapping with concrete mask arrays.
        """
        def is_null(value):
            return value is None or (isinstance(value, str) and value.strip().lower() == "none")

        if not any(is_null(value) and name in _MASK_SHAPE_SOURCES for name, value in user_params.items()):
            return user_params

        expanded = dict(user_params)
        for name, value in list(expanded.items()):
            if is_null(value) and name in _MASK_SHAPE_SOURCES:
                expanded[name] = self._expand_none_mask(name, func_type, data_idx)
        return expanded

    def _expand_none_mask(self, mask_name, func_type, data_idx):
        """
        Build a full-True mask array for a known mask parameter.

        Parameters:
        mask_name (str): Name of the mask parameter.
        func_type (str): Step function type.
        data_idx (int, array-like, or None): Rows being processed.

        Returns:
        np.ndarray: Mask array with the correct per-row inner shape.
        """
        source_name = _MASK_SHAPE_SOURCES[mask_name]
        source = getattr(self.DS, source_name)
        shape = source.shape
        if not shape:
            probe_idx = 0 if data_idx is None else int(np.atleast_1d(np.asarray(data_idx, dtype=np.int32))[0])
            shape = np.asarray(source[probe_idx]).shape
        else:
            shape = shape[1:]

        if func_type in ("global", "global-res"):
            return np.ones(shape, dtype=bool)
        if np.ndim(data_idx) == 0:
            return np.ones(shape, dtype=bool)
        return np.ones((len(np.atleast_1d(np.asarray(data_idx, dtype=np.int32))), *shape), dtype=bool)

    def _add_user_params(self, step, user_params, data_idx, save, pipeline_scope, step_index):
        """
        Store user-provided parameters in the dataset before step execution.

        Parameters:
        step (plStep): Step that will consume the parameters.
        user_params (dict): Parameter mapping supplied by the caller or YAML.
        data_idx (int, array-like, or None): Rows being processed.
        save (bool): If True, persist the parameters immediately.
        pipeline_scope (str or None): Owning pipeline scope.
        step_index (int or None): Execution index of the owning step.
        """
        is_global = step.func_type in ("global", "global-res")
        for name, value in user_params.items():
            if is_global:
                self.DS._store_param(
                    name,
                    value,
                    is_global=True,
                    pipeline_scope=pipeline_scope,
                    step_name=step.name,
                    step_index=step_index,
                    save=save,
                )
                continue

            rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
            if isinstance(data_idx, (int, np.integer)):
                row_value = [value]
            elif not isinstance(value, (list, tuple, np.ndarray)):
                row_value = [value] * len(rows)
            else:
                row_value = value
            self.DS._store_param(
                name,
                row_value,
                is_global=False,
                data_idx=rows,
                pipeline_scope=pipeline_scope,
                step_name=step.name,
                step_index=step_index,
                save=save,
            )

    def _resolve_step_scope(self, step):
        """
        Identify whether a step belongs to the calibration or analysis path.

        Parameters:
        step (plStep): Step to classify.

        Returns:
        tuple: ``(pipeline_scope, step_index)`` or ``(None, None)`` when the
            step is not part of a known path.
        """
        if step.name in self.step_indices:
            return "analysis", self.step_indices[step.name]
        if step.name in self.DS.cal_step_indices:
            return "cal", self.DS.cal_step_indices[step.name]
        return None, None

    def _resolve_analysis_definition(self, analysis_yaml_path, custom_path):
        """
        Resolve the analysis definition from explicit inputs or dataset metadata.

        Parameters:
        analysis_yaml_path (str or None): Requested analysis YAML path or
            alias.
        custom_path (str or None): Requested custom analysis step file.

        Returns:
        dict: Normalized definition containing paths, YAML text, and custom
            source code.

        Raises:
        ValueError: If the supplied definition conflicts with the dataset's
            embedded definition.
        """
        metadata = self.DS._read_metadata()
        stored_yaml = metadata.get("analysis_yaml")
        stored_custom = metadata.get("analysis_custom_source")
        stored_yaml_path = metadata.get("analysis_yaml_path")
        stored_custom_path = metadata.get("analysis_custom_path")

        yaml_path = None
        yaml_text = None
        custom_source = None
        resolved_custom_path = None

        if analysis_yaml_path is not None:
            yaml_path = _resolve_analysis_yaml_path(analysis_yaml_path)
            yaml_text = _read_text_file(yaml_path)
        elif stored_yaml is not None:
            yaml_path = stored_yaml_path
            yaml_text = stored_yaml

        if custom_path is not None:
            if not custom_path.endswith(".py"):
                raise ValueError("custom_path must point to a .py file")
            resolved_custom_path = os.path.abspath(custom_path)
            custom_source = _read_text_file(resolved_custom_path)
        elif stored_custom is not None:
            resolved_custom_path = stored_custom_path
            custom_source = stored_custom

        if stored_yaml is not None and yaml_text is not None and yaml_text != stored_yaml:
            raise ValueError("Provided analysis YAML does not match the dataset definition")
        if stored_custom is not None and custom_source != stored_custom:
            raise ValueError("Provided analysis custom steps do not match the dataset definition")

        if yaml_text is not None:
            self.DS.register_analysis_definition(
                yaml_text,
                custom_source,
                analysis_yaml_path=os.path.abspath(yaml_path) if yaml_path else None,
                analysis_custom_path=resolved_custom_path,
            )

        return {
            "yaml_path": os.path.abspath(yaml_path) if yaml_path else None,
            "custom_path": resolved_custom_path,
            "yaml_text": yaml_text,
            "custom_source": custom_source,
        }


def _resolve_analysis_yaml_path(analysis_yaml_path):
    """
    Resolve an analysis YAML alias or validate a filesystem path.

    Parameters:
    analysis_yaml_path (str): Alias or path supplied by the caller.

    Returns:
    str: Resolved YAML file path.

    Raises:
    ValueError: If the path does not point to a YAML file.
    """
    if analysis_yaml_path in _ANALYSIS_YAML_ALIASES:
        return os.path.join(
            os.path.dirname(__file__),
            "templates",
            _ANALYSIS_YAML_ALIASES[analysis_yaml_path],
        )
    if not (analysis_yaml_path.endswith(".yaml") or analysis_yaml_path.endswith(".yml")):
        raise ValueError("analysis_yaml_path must point to a .yaml or .yml file")
    return analysis_yaml_path


def _load_custom_analysis_steps_from_source(source):
    """
    Execute custom analysis step source and extract ``custom_analysis_steps``.

    Parameters:
    source (str or None): Python source code defining a
        ``custom_analysis_steps`` list.

    Returns:
    list: Custom analysis steps, or an empty list when no source is given.
    """
    if source is None:
        return []
    namespace = {}
    exec(compile(source, "<pipeline_v2_custom_analysis>", "exec"), namespace)
    return list(namespace.get("custom_analysis_steps", []))


def _validate_task_idxs(task_idxs):
    """
    Validate that analysis step indices are consecutive integers from 1.

    Parameters:
    task_idxs (iterable): Step indices from the YAML mapping.

    Returns:
    int: Largest valid task index.

    Raises:
    ValueError: If the indices are empty, non-integer, or non-consecutive.
    """
    if not task_idxs:
        raise ValueError("task_idxs is empty")
    if not all(isinstance(index, int) and index >= 1 for index in task_idxs):
        raise ValueError("task_idxs must contain only integers >= 1")
    max_idx = max(task_idxs)
    expected = set(range(1, max_idx + 1))
    if set(task_idxs) != expected or len(task_idxs) != max_idx:
        raise ValueError(
            "task_idxs must be consecutive integers starting at 1 with no gaps or duplicates"
        )
    return max_idx