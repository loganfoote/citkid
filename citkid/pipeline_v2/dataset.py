import importlib.util
import os
from datetime import datetime

import numpy as np
import yaml
import zarr

from . import default_steps
from . import framework as pf


_CAL_YAML_ALIASES = {
    "ts": "cal.yaml",
    "iq": "cal-iqonly.yaml",
    "ts_offres": "cal-offres.yaml",
}


class DataSet:
    """
    Simplified zarr-backed dataset for calibration and analysis products.

    Unlike :class:`citkid.pipeline.dataset.DataSet`, this version stores only
    a single active version of each parameter.  When an earlier analysis step
    is re-run, downstream products are deleted instead of being preserved in
    separate run folders.
    """

    _RESERVED_ATTRS = None
    _METADATA_ATTR = "pipeline_v2"
    _SCHEMA_VERSION = 1

    def __init__(
        self,
        zarr_path,
        cal_yaml_path=None,
        custom_path=None,
        zarr_mode="a",
        custom_cal_steps=None,
    ):
        """
        Initialize the dataset and load its calibration definition.

        Parameters:
        zarr_path (str or zarr.Group): Path to the output zarr store or an
            already-open zarr group.
        cal_yaml_path (str or None): Path or alias for the calibration YAML.
            When None, the YAML must already be embedded in the zarr store.
        custom_path (str or None): Path to the Python file defining
            ``custom_cal_steps``. When None, embedded custom step source is
            used if available.
        zarr_mode (str): Mode used when opening ``zarr_path`` if a path was
            provided.
        custom_cal_steps (list of plStep or None): Custom calibration steps to
            use directly instead of loading them from ``custom_path``.

        Raises:
        TypeError: If the supplied path arguments have invalid types.
        ValueError: If the zarr path or calibration definition is invalid.
        """
        if custom_path is not None and not isinstance(custom_path, str):
            raise TypeError("custom_path must be a string or None")
        if cal_yaml_path is not None and not isinstance(cal_yaml_path, str):
            raise TypeError("cal_yaml_path must be a string or None")

        if isinstance(zarr_path, zarr.Group):
            self.root = zarr_path
            self.zarr_path = None
            self.zarr_mode = None
        elif isinstance(zarr_path, str):
            self.zarr_path = os.path.abspath(zarr_path)
            self.zarr_mode = zarr_mode
            if ".zarr" not in self.zarr_path:
                raise ValueError("zarr_path must point to a .zarr file")
            self.root = zarr.open_group(self.zarr_path, mode=zarr_mode)
        else:
            raise TypeError("zarr_path must be a zarr.Group or string path")

        try:
            self.root.require_group("_failures")
        except Exception:
            pass

        self._global_cache = {}
        self._per_row_cache = {}
        self._param_meta = {}
        self._analysis_step_names = {}

        self._metadata = self._read_metadata()
        cal_def = self._resolve_cal_definition(
            cal_yaml_path=cal_yaml_path,
            custom_path=custom_path,
            custom_cal_steps=custom_cal_steps,
        )
        self.cal_yaml_path = cal_def["yaml_path"]
        self.custom_path = cal_def["custom_path"]
        self.cal_yaml_text = cal_def["yaml_text"]
        self.cal_custom_source = cal_def["custom_source"]

        self.cal_steps = list(cal_def["custom_steps"])
        for step in default_steps.default_cal_steps:
            if step.name not in [s.name for s in self.cal_steps]:
                self.cal_steps.append(step)

        yaml_dict = yaml.safe_load(self.cal_yaml_text) or {}
        self.cal_pl = _convert_yaml_to_steps(yaml_dict, self.cal_steps)
        pf.check_pl_tree_structure(self.cal_pl, cal=True)
        self.cal_step_indices = _step_indices(self.cal_pl)

        self._load_param_registry()

        nrows_path = pf.find_pl_path(self.cal_pl, "nrows")
        if nrows_path is None:
            raise ValueError(
                "Calibration pipeline must be able to produce 'nrows' as a global parameter"
            )
        if nrows_path[-1].func_type != "global":
            raise ValueError("Parameter 'nrows' must be produced by a global step")

    def register_analysis_definition(
        self,
        analysis_yaml_text,
        analysis_custom_source,
        analysis_yaml_path=None,
        analysis_custom_path=None,
    ):
        """
        Persist the analysis definition inside the zarr metadata.

        Parameters:
        analysis_yaml_text (str): Full text of the analysis YAML file.
        analysis_custom_source (str or None): Source code defining
            ``custom_analysis_steps``.
        analysis_yaml_path (str or None): Original filesystem path of the YAML
            definition, stored for provenance only.
        analysis_custom_path (str or None): Original filesystem path of the
            custom analysis step module, stored for provenance only.

        Raises:
        ValueError: If the dataset already contains a conflicting analysis
            definition.
        """
        existing = self._read_metadata()
        stored_yaml = existing.get("analysis_yaml")
        stored_custom = existing.get("analysis_custom_source")

        if stored_yaml is not None and analysis_yaml_text != stored_yaml:
            raise ValueError(
                "The dataset already contains a different analysis YAML definition"
            )
        if stored_custom is not None and analysis_custom_source != stored_custom:
            raise ValueError(
                "The dataset already contains different analysis custom step code"
            )

        update = {
            "analysis_yaml": analysis_yaml_text,
            "analysis_yaml_path": analysis_yaml_path,
            "analysis_custom_source": analysis_custom_source,
            "analysis_custom_path": analysis_custom_path,
        }
        self._write_metadata(update)

    def set_analysis_step_names(self, step_names_by_index):
        """
        Cache analysis step names by execution index.

        Parameters:
        step_names_by_index (dict): Mapping of integer step indices to step
            names.
        """
        self._analysis_step_names = dict(step_names_by_index)

    def invalidate_after(self, pipeline_scope, step_index):
        """
        Delete products created by downstream steps.

        Parameters:
        pipeline_scope (str): Either ``'cal'`` or ``'analysis'`` for the step
            that is about to be re-run.
        step_index (int): Execution index of the step being re-run within its
            pipeline scope.

        Notes:
        Any parameter whose recorded producing step is downstream of the given
        scope/index pair is removed from memory and from the zarr store.
        """
        to_delete = []
        for name, meta in list(self._param_meta.items()):
            scope = meta.get("pipeline_scope")
            index = meta.get("step_index")
            if scope is None or index is None:
                continue
            if self._is_downstream(
                current_scope=pipeline_scope,
                current_index=step_index,
                other_scope=scope,
                other_index=index,
            ):
                to_delete.append(name)

        for name in to_delete:
            self._delete_param(name)

    def save_step_outputs(self, step, data_idx=None):
        """
        Persist the latest outputs of a step without re-running it.

        Parameters:
        step (plStep): Step whose outputs should be written to the zarr store.
        data_idx (int, array-like, or None): Rows to save for per-row outputs.
            Ignored for global outputs.

        Raises:
        ValueError: If the step outputs are not currently available in memory.
        """
        for name in step.return_names:
            if name not in self._param_meta:
                raise ValueError(
                    f"Step '{step.name}' output '{name}' is not available to save"
                )
            meta = self._param_meta[name]
            if meta["global"]:
                self._write_global_param(name)
            else:
                self._write_per_row_param(name, data_idx=data_idx)

    def __getattr__(self, name):
        """
        Lazily access a stored or calibratable parameter.

        Parameters:
        name (str): Parameter name to retrieve.

        Returns:
        misc: Global parameters are returned directly; per-row parameters are
            returned as :class:`citkid.pipeline.framework.LazyAttr` objects.

        Raises:
        AttributeError: If the parameter is neither stored nor producible from
            the calibration pipeline.
        """
        if name in self._param_meta:
            meta = self._param_meta[name]
            if meta["global"]:
                return self._get_global(name)
            return self._get_lazy_attr(name)

        path = pf.find_pl_path(self.cal_pl, name)
        if path is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if path[-1].func_type == "global":
            self._ensure_loaded(name, data_idx=None)
            return self._get_global(name)

        self._param_meta[name] = {
            "global": False,
            "pipeline_scope": "cal",
            "step_name": path[-1].name,
            "step_index": self.cal_step_indices.get(path[-1].name),
        }
        return self._get_lazy_attr(name)

    def _get_global(self, name):
        """
        Retrieve a global parameter from memory, zarr, or the calibration path.

        Parameters:
        name (str): Global parameter name.

        Returns:
        misc: Stored value for the parameter.

        Raises:
        AttributeError: If the parameter cannot be found or produced.
        """
        if name in self._global_cache:
            return self._global_cache[name]
        if name not in self.root:
            self._ensure_loaded(name, data_idx=None)
        if name not in self.root:
            raise AttributeError(f"Global parameter '{name}' is not available")
        data = self.root[name]["data"][...]
        data = _unwrap_scalar(data)
        self._global_cache[name] = data
        return data

    def _get_lazy_attr(self, name):
        """
        Return the lazy per-row accessor for a parameter.

        Parameters:
        name (str): Per-row parameter name.

        Returns:
        LazyAttr: Accessor that loads rows on demand.
        """
        if name not in self._per_row_cache:
            self._per_row_cache[name] = pf.LazyAttr(self, name, 1)
        return self._per_row_cache[name]

    def _fetch_rows(self, name, run_idx, data_idx=None, enforced_max_runs=None):
        """
        Load per-row data into the lazy cache and return the requested rows.

        Parameters:
        name (str): Per-row parameter name.
        run_idx (int): Requested run index. Must be 1 in pipeline_v2.
        data_idx (int, array-like, or None): Row indices to load.
        enforced_max_runs (dict or None): Unused compatibility argument kept to
            match the original LazyAttr interface.

        Returns:
        np.ndarray: Requested rows stacked into an array.

        Raises:
        ValueError: If a non-existent run index is requested or the parameter
            cannot be produced.
        """
        if run_idx != 1:
            raise ValueError("pipeline_v2 does not support multiple run indices")
        if data_idx is None:
            raise ValueError(f"data_idx required for per-row parameter '{name}'")

        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        lazy_attr = self._get_lazy_attr(name)
        missing = [int(di) for di in rows if int(di) not in lazy_attr._cache]

        if missing and name in self.root:
            group = self.root[name]
            exists = group["row_exists"][missing]
            if np.any(exists):
                found_rows = [row for row, exists_i in zip(missing, exists) if exists_i]
                data = group["data"][found_rows]
                for di, value in zip(found_rows, data):
                    lazy_attr._cache[int(di)] = value
                if lazy_attr._shape == () and len(data):
                    first = np.asarray(data[0])
                    lazy_attr._shape = (int(self.nrows), *first.shape) if first.shape else (int(self.nrows),)

        still_missing = [int(di) for di in rows if int(di) not in lazy_attr._cache]
        if still_missing:
            self._ensure_loaded(name, data_idx=np.asarray(still_missing, dtype=np.int32))

        unresolved = [int(di) for di in rows if int(di) not in lazy_attr._cache]
        if unresolved:
            raise ValueError(
                f"Parameter '{name}' is missing rows {unresolved} and cannot be produced"
            )

        return np.array([lazy_attr._cache[int(di)] for di in rows])

    def _ensure_loaded(self, name, data_idx=None):
        """
        Ensure that a calibratable parameter exists.

        Parameters:
        name (str): Parameter name to produce if missing.
        data_idx (int, array-like, or None): Row indices required for per-row
            parameters.

        Notes:
        This method executes only the calibration steps needed to produce the
        requested parameter and skips steps whose outputs already exist.
        """
        path = pf.find_pl_path(self.cal_pl, name)
        if path is None:
            return

        for step in path:
            step_index = self.cal_step_indices.get(step.name)
            if step.func_type == "global":
                if self._step_outputs_exist(step, data_idx=None):
                    continue
                self._execute_step(
                    step,
                    data_idx=None,
                    save=False,
                    pipeline_scope="cal",
                    step_index=step_index,
                )
                continue

            if step.func_type == "global-res":
                if self._step_outputs_exist(step, data_idx=np.arange(int(self.nrows), dtype=np.int32)):
                    continue
                self._execute_step(
                    step,
                    data_idx=None,
                    save=False,
                    pipeline_scope="cal",
                    step_index=step_index,
                )
                continue

            if data_idx is None:
                raise ValueError(
                    f"data_idx is required to produce per-row parameter '{name}'"
                )
            rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
            missing_rows = self._step_missing_rows(step, rows)
            if len(missing_rows) == 0:
                continue
            self._execute_step(
                step,
                data_idx=missing_rows,
                save=False,
                pipeline_scope="cal",
                step_index=step_index,
            )

    def _execute_step(self, step, data_idx=None, save=False, pipeline_scope=None, step_index=None, execution_mode='vectorized'):
        """
        Execute a single calibration or analysis step.

        Parameters:
        step (plStep): Step to execute.
        data_idx (int, array-like, or None): Row indices for per-row or
            vectorized steps. Must be None for global and global-res steps.
        save (bool): If True, persist outputs immediately after execution.
        pipeline_scope (str or None): Scope that owns the step, typically
            ``'cal'`` or ``'analysis'``.
        step_index (int or None): Execution index of the step within its
            pipeline scope.
        execution_mode (str): How to execute vectorized steps. Can be 'vectorized'
            (default, loads all data at once) or 'per-row' (loops over each
            data_idx one at a time, using less memory).

        Returns:
        dict or None: Failure mapping for per-row steps, or None otherwise.

        Raises:
        TypeError: If ``step`` is not a :class:`plStep`.
        ValueError: If the supplied ``data_idx`` is incompatible with the step
            type.
        """
        if not isinstance(step, pf.plStep):
            raise TypeError("step must be a plStep instance")
        
        if execution_mode not in ('vectorized', 'per-row'):
            raise ValueError(f"execution_mode must be 'vectorized' or 'per-row', got '{execution_mode}'")

        if step.func_type == "global":
            params, param_is_global = self._collect_params(step, None)
            out = step._run(params, param_is_global)
            for name, value in out.items():
                self._store_param(
                    name,
                    value,
                    is_global=True,
                    pipeline_scope=pipeline_scope,
                    step_name=step.name,
                    step_index=step_index,
                    save=save,
                )
            return None

        if step.func_type == "global-res":
            params, param_is_global = self._collect_params(step, None)
            out = step._run(params, param_is_global)
            all_rows = np.arange(int(self.nrows), dtype=np.int32)
            for name, value in out.items():
                self._store_param(
                    name,
                    value,
                    is_global=False,
                    data_idx=all_rows,
                    pipeline_scope=pipeline_scope,
                    step_name=step.name,
                    step_index=step_index,
                    save=save,
                )
            return None

        if data_idx is None:
            raise ValueError(f"data_idx required for step '{step.name}'")
        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))

        if step.func_type == "vectorized" and execution_mode == 'vectorized':
            params, param_is_global = self._collect_params(step, rows)
            out = step._run(params, param_is_global)
            for name, value in out.items():
                self._store_param(
                    name,
                    value,
                    is_global=False,
                    data_idx=rows,
                    pipeline_scope=pipeline_scope,
                    step_name=step.name,
                    step_index=step_index,
                    save=save,
                )
            return None

        failures = {}
        for di in rows:
            try:
                params, param_is_global = self._collect_params(
                    step, np.atleast_1d(np.asarray([int(di)], dtype=np.int32))
                )
                out = step._run(params, param_is_global)
                for name, value in out.items():
                    self._store_param(
                        name,
                        value,
                        is_global=False,
                        data_idx=np.atleast_1d(np.asarray([int(di)], dtype=np.int32)),
                        pipeline_scope=pipeline_scope,
                        step_name=step.name,
                        step_index=step_index,
                        save=save,
                    )
            except Exception as exc:
                failures[int(di)] = str(exc)

        if failures:
            self._record_failures(step.name, failures)
        return failures

    def _collect_params(self, step, data_idx):
        """
        Collect concrete function arguments for a step execution.

        Parameters:
        step (plStep): Step whose input parameters should be resolved.
        data_idx (np.ndarray or None): Row indices for per-row access.

        Returns:
        tuple: ``(params, param_is_global)`` matching
            :meth:`citkid.pipeline.framework.plStep._run`.

        Raises:
        ValueError: If a global step attempts to consume a per-row parameter.
        """
        params = []
        param_is_global = []
        for param_name in step.param_names:
            if param_name == "data_idx":
                if data_idx is None:
                    params.append(None)
                    param_is_global.append(True)
                else:
                    params.append(data_idx)
                    param_is_global.append(False)
                continue

            value = getattr(self, param_name)
            is_global = not isinstance(value, pf.LazyAttr)
            param_is_global.append(is_global)
            if is_global:
                params.append(value)
            else:
                if data_idx is None:
                    raise ValueError(
                        f"Global step '{step.name}' cannot consume per-row parameter '{param_name}'"
                    )
                params.append(value[data_idx])
        return params, param_is_global

    def _store_param(
        self,
        name,
        value,
        is_global,
        data_idx=None,
        pipeline_scope=None,
        step_name=None,
        step_index=None,
        save=False,
    ):
        """
        Store a parameter in memory and optionally write it to zarr.

        Parameters:
        name (str): Parameter name.
        value (misc): Value to store. For per-row parameters, this should hold
            one entry per row in ``data_idx``.
        is_global (bool): True for global parameters, False for per-row
            parameters.
        data_idx (int, array-like, or None): Row indices for per-row storage.
        pipeline_scope (str or None): Producing scope, e.g. ``'cal'`` or
            ``'analysis'``.
        step_name (str or None): Name of the producing step.
        step_index (int or None): Execution index of the producing step.
        save (bool): If True, persist the parameter immediately.

        Raises:
        ValueError: If ``name`` collides with a reserved DataSet attribute.
        """
        if name in self._get_reserved_attrs():
            raise ValueError(f"Cannot create parameter '{name}' because the name is reserved")

        meta = {
            "global": bool(is_global),
            "pipeline_scope": pipeline_scope,
            "step_name": step_name,
            "step_index": step_index,
        }
        self._param_meta[name] = meta

        if is_global:
            self._global_cache[name] = _unwrap_scalar(np.asarray(value)) if np.asarray(value).shape == () else value
            if save:
                self._write_global_param(name)
            return

        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        lazy_attr = self._get_lazy_attr(name)
        row_values = _normalize_row_values(value, rows)
        for di, row_value in zip(rows, row_values):
            lazy_attr._cache[int(di)] = row_value
        if lazy_attr._shape == () and row_values:
            first = np.asarray(row_values[0])
            lazy_attr._shape = (int(self.nrows), *first.shape) if first.shape else (int(self.nrows),)
        if save:
            self._write_per_row_param(name, data_idx=rows)

    def _write_global_param(self, name):
        """
        Write a global parameter to the zarr store.

        Parameters:
        name (str): Name of the global parameter to persist.
        """
        meta = self._param_meta[name]
        if name in self.root:
            del self.root[name]
        group = self.root.create_group(name)
        group.create_array("data", data=np.asarray(self._global_cache[name]))
        _write_group_metadata(group, meta)

    def _write_per_row_param(self, name, data_idx=None):
        """
        Write per-row parameter data to the zarr store.

        Parameters:
        name (str): Name of the per-row parameter to persist.
        data_idx (int, array-like, or None): Specific rows to save. When None,
            all cached rows are written.

        Raises:
        ValueError: If requested rows are not currently cached in memory.
        """
        meta = self._param_meta[name]
        lazy_attr = self._get_lazy_attr(name)
        if data_idx is None:
            if not lazy_attr._cache:
                raise ValueError(f"Parameter '{name}' has no cached rows to save")
            rows = np.array(sorted(lazy_attr._cache.keys()), dtype=np.int32)
        else:
            rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        if len(rows) == 0:
            return

        if name not in self.root:
            first = np.asarray(lazy_attr._cache[int(rows[0])])
            shape = (int(self.nrows), *first.shape)
            group = self.root.create_group(name)
            group.create_array("data", shape=shape, chunks=(1, *first.shape), dtype=first.dtype)
            group.create_array("row_exists", shape=(int(self.nrows),), dtype=np.bool_)
        else:
            group = self.root[name]

        for di in rows:
            if int(di) not in lazy_attr._cache:
                raise ValueError(f"Parameter '{name}' row {int(di)} is not cached in memory")
            group["data"][int(di)] = lazy_attr._cache[int(di)]
            group["row_exists"][int(di)] = True
        _write_group_metadata(group, meta)

    def _load_param_registry(self):
        """
        Load parameter metadata from the existing zarr store.

        Reads zarr group attributes for previously-saved parameters and
        reconstructs ``self._param_meta``.
        """
        for name, group in self.root.groups():
            if name.startswith("_"):
                continue
            if "global" not in group.attrs:
                continue
            self._param_meta[name] = {
                "global": bool(group.attrs.get("global")),
                "pipeline_scope": group.attrs.get("pipeline_scope"),
                "step_name": group.attrs.get("step_name"),
                "step_index": group.attrs.get("step_index"),
            }

    def _delete_param(self, name):
        """
        Remove a parameter from memory and from the zarr store.

        Parameters:
        name (str): Parameter name to delete.
        """
        self._global_cache.pop(name, None)
        self._per_row_cache.pop(name, None)
        self._param_meta.pop(name, None)
        if name in self.root:
            del self.root[name]

    def _step_outputs_exist(self, step, data_idx):
        """
        Check whether all outputs for a step already exist.

        Parameters:
        step (plStep): Step whose outputs should be checked.
        data_idx (int, array-like, or None): Relevant rows for per-row steps.

        Returns:
        bool: True if every output required for the given scope/rows exists.
        """
        if step.func_type == "global":
            return all(self._has_global(name) for name in step.return_names)
        if step.func_type == "global-res":
            rows = np.arange(int(self.nrows), dtype=np.int32)
            return all(self._has_rows(name, rows) for name in step.return_names)
        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        return all(self._has_rows(name, rows) for name in step.return_names)

    def _step_missing_rows(self, step, data_idx):
        """
        Return the subset of rows whose outputs are missing for a step.

        Parameters:
        step (plStep): Step whose outputs should be checked.
        data_idx (int or array-like): Candidate rows.

        Returns:
        np.ndarray: Row indices that still need execution.
        """
        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        missing = []
        for di in rows:
            if not all(self._has_rows(name, np.atleast_1d(np.asarray([int(di)], dtype=np.int32))) for name in step.return_names):
                missing.append(int(di))
        return np.asarray(missing, dtype=np.int32)

    def _has_global(self, name):
        """
        Check whether a global parameter exists in memory or zarr.

        Parameters:
        name (str): Parameter name.

        Returns:
        bool: True if the global parameter is available.
        """
        return name in self._global_cache or (name in self.root and "data" in self.root[name])

    def _has_rows(self, name, rows):
        """
        Check whether specific rows exist for a per-row parameter.

        Parameters:
        name (str): Parameter name.
        rows (int or array-like): Rows to verify.

        Returns:
        bool: True if every requested row is available.
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int32))
        if name in self._per_row_cache:
            lazy_attr = self._per_row_cache[name]
            if all(int(di) in lazy_attr._cache for di in rows):
                return True
        if name not in self.root:
            return False
        group = self.root[name]
        if "row_exists" not in group:
            return False
        return bool(np.all(group["row_exists"][rows]))

    def _resolve_cal_definition(self, cal_yaml_path, custom_path, custom_cal_steps):
        """
        Resolve the calibration definition from inputs and embedded metadata.

        Parameters:
        cal_yaml_path (str or None): Requested calibration YAML path or alias.
        custom_path (str or None): Requested custom calibration step file.
        custom_cal_steps (list of plStep or None): Explicit custom step list.

        Returns:
        dict: Normalized definition containing YAML text, source text, file
            paths, and loaded custom steps.

        Raises:
        ValueError: If no calibration definition is available or if the
            supplied definition conflicts with embedded metadata.
        """
        metadata = self._metadata
        stored_yaml = metadata.get("cal_yaml")
        stored_custom = metadata.get("cal_custom_source")
        stored_yaml_path = metadata.get("cal_yaml_path")
        stored_custom_path = metadata.get("cal_custom_path")

        if custom_path is not None and not custom_path.endswith(".py"):
            raise ValueError("custom_path must point to a .py file")

        if custom_cal_steps is not None:
            if cal_yaml_path is None:
                raise ValueError("cal_yaml_path is required when custom_cal_steps is provided")
            yaml_path = _resolve_cal_yaml_path(cal_yaml_path)
            yaml_text = _read_text_file(yaml_path)
            return {
                "yaml_path": os.path.abspath(yaml_path),
                "custom_path": custom_path,
                "yaml_text": yaml_text,
                "custom_source": stored_custom,
                "custom_steps": list(custom_cal_steps),
            }

        yaml_text = None
        custom_source = None
        yaml_path = None
        resolved_custom_path = None

        if cal_yaml_path is not None:
            yaml_path = _resolve_cal_yaml_path(cal_yaml_path)
            yaml_text = _read_text_file(yaml_path)
        elif stored_yaml is not None:
            yaml_text = stored_yaml
            yaml_path = stored_yaml_path

        if yaml_text is None:
            raise ValueError(
                "No calibration YAML was provided and no embedded definition exists in the dataset"
            )

        if custom_path is not None:
            resolved_custom_path = os.path.abspath(custom_path)
            custom_source = _read_text_file(resolved_custom_path)
        elif stored_custom is not None:
            resolved_custom_path = stored_custom_path
            custom_source = stored_custom

        if stored_yaml is not None and yaml_text != stored_yaml:
            raise ValueError("Provided calibration YAML does not match the dataset definition")
        if stored_custom is not None and custom_source != stored_custom:
            raise ValueError("Provided calibration custom steps do not match the dataset definition")

        if stored_yaml is None:
            self._write_metadata(
                {
                    "cal_yaml": yaml_text,
                    "cal_yaml_path": os.path.abspath(yaml_path) if yaml_path else None,
                    "cal_custom_source": custom_source,
                    "cal_custom_path": resolved_custom_path,
                }
            )

        return {
            "yaml_path": os.path.abspath(yaml_path) if yaml_path else None,
            "custom_path": resolved_custom_path,
            "yaml_text": yaml_text,
            "custom_source": custom_source,
            "custom_steps": _load_custom_cal_steps_from_source(custom_source),
        }

    def _read_metadata(self):
        """
        Read the pipeline_v2 metadata block from the zarr root.

        Returns:
        dict: Metadata dictionary stored under ``self._METADATA_ATTR``.
        """
        return dict(self.root.attrs.get(self._METADATA_ATTR, {}))

    def _write_metadata(self, update):
        """
        Update the pipeline_v2 metadata block in the zarr root.

        Parameters:
        update (dict): Key-value pairs to merge into the existing metadata.
        """
        current = self._read_metadata()
        current.update(update)
        current["schema_version"] = self._SCHEMA_VERSION
        self.root.attrs[self._METADATA_ATTR] = current
        self._metadata = current

    def _record_failures(self, step_name, failures):
        """
        Persist per-row execution failures to the ``_failures`` zarr group.

        Parameters:
        step_name (str): Name of the failing step.
        failures (dict): Mapping ``data_idx -> error message``.
        """
        try:
            group = self.root.require_group(f"_failures/{step_name}")
            payload = dict(group.attrs.get("failures", {}))
            payload.update(
                {
                    f"idx{di}": {
                        "error": message,
                        "time": datetime.now().strftime("%Y%m%d-%H:%M:%S"),
                    }
                    for di, message in failures.items()
                }
            )
            group.attrs["failures"] = payload
        except Exception:
            pass

    def _is_downstream(self, current_scope, current_index, other_scope, other_index):
        """
        Determine whether one producing step is downstream of another.

        Parameters:
        current_scope (str): Scope of the step being re-run.
        current_index (int): Execution index of the step being re-run.
        other_scope (str): Scope of the stored parameter's producing step.
        other_index (int): Execution index of the stored parameter's producing
            step.

        Returns:
        bool: True if the stored parameter should be invalidated.
        """
        order = {"cal": 0, "analysis": 1}
        if current_scope not in order or other_scope not in order:
            return False
        if order[other_scope] > order[current_scope]:
            return True
        if order[other_scope] < order[current_scope]:
            return False
        return other_index > current_index

    def _get_reserved_attrs(self):
        """
        Return the set of reserved public DataSet attribute names.

        Returns:
        set: Attribute names that cannot be reused as parameter names.
        """
        if DataSet._RESERVED_ATTRS is None:
            DataSet._RESERVED_ATTRS = {name for name in dir(type(self)) if not name.startswith("_")}
        return DataSet._RESERVED_ATTRS


def _resolve_cal_yaml_path(cal_yaml_path):
    """
    Resolve a calibration YAML alias or validate a filesystem path.

    Parameters:
    cal_yaml_path (str): Alias or path supplied by the caller.

    Returns:
    str: Resolved YAML file path.

    Raises:
    ValueError: If the path does not point to a YAML file.
    """
    if cal_yaml_path in _CAL_YAML_ALIASES:
        return os.path.join(
            os.path.dirname(__file__),
            "templates",
            _CAL_YAML_ALIASES[cal_yaml_path],
        )
    if not (cal_yaml_path.endswith(".yaml") or cal_yaml_path.endswith(".yml")):
        raise ValueError("cal_yaml_path must point to a .yaml or .yml file")
    return cal_yaml_path


def _read_text_file(path):
    """
    Read a UTF-8 text file.

    Parameters:
    path (str): File path to read.

    Returns:
    str: Entire file contents.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_custom_cal_steps_from_source(source):
    """
    Execute custom calibration step source and extract ``custom_cal_steps``.

    Parameters:
    source (str or None): Python source code defining a
        ``custom_cal_steps`` list.

    Returns:
    list: Custom calibration steps, or an empty list when no source is given.
    """
    if source is None:
        return []
    namespace = {}
    exec(compile(source, "<pipeline_v2_custom_cal>", "exec"), namespace)
    return list(namespace.get("custom_cal_steps", []))


def _normalize_row_values(value, rows):
    """
    Normalize per-row outputs into one value per requested row.

    Parameters:
    value (misc): Raw step output value.
    rows (np.ndarray): Row indices that the output corresponds to.

    Returns:
    list: Output values aligned one-to-one with ``rows``.

    Raises:
    ValueError: If the output length does not match the number of rows.
    """
    if len(rows) == 1:
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
            return [value[0]]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return [value[0]]
        return [value]

    value_array = np.asarray(value)
    if len(value_array) != len(rows):
        raise ValueError(
            f"Length mismatch: value has {len(value_array)} elements but rows has {len(rows)}"
        )
    return list(value_array)


def _unwrap_scalar(value):
    """
    Convert a 0-D numpy array into a Python scalar.

    Parameters:
    value (misc): Value to normalize.

    Returns:
    misc: Python scalar for 0-D arrays, otherwise the original value.
    """
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _write_group_metadata(group, meta):
    """
    Write pipeline_v2 provenance metadata onto a zarr parameter group.

    Parameters:
    group (zarr.Group): Parameter group being updated.
    meta (dict): Stored parameter metadata.
    """
    group.attrs["global"] = bool(meta["global"])
    group.attrs["pipeline_scope"] = meta.get("pipeline_scope")
    group.attrs["step_name"] = meta.get("step_name")
    group.attrs["step_index"] = meta.get("step_index")
    group.attrs["write_time"] = datetime.now().strftime("%Y%m%d-%H:%M:%S")


def _step_indices(tree):
    """
    Build a name-to-index mapping for a pipeline tree.

    Parameters:
    tree (dict): Pipeline definition tree.

    Returns:
    dict: Mapping ``step.name -> execution index``.
    """
    steps = _flatten_pipeline_steps(tree)
    return {step.name: index for index, step in enumerate(steps, start=1)}


def _flatten_pipeline_steps(tree):
    """
    Flatten a pipeline tree into execution order.

    Parameters:
    tree (dict): Pipeline definition tree.

    Returns:
    list: Sequence of :class:`plStep` objects in execution order.
    """
    steps = []

    def is_seq(node):
        return isinstance(node, dict) and all(str(key).isdigit() for key in node.keys())

    def walk(node):
        if is_seq(node):
            for _, child in sorted(node.items(), key=lambda item: int(item[0])):
                walk(child)
            return
        if isinstance(node, dict) and "task" in node:
            steps.append(node["task"])
            for key, child in sorted(node.items(), key=lambda item: 0 if item[0] == "task" else 1):
                if key == "task":
                    continue
                if is_seq(child):
                    walk(child)
            return
        if isinstance(node, dict):
            for child in node.values():
                walk(child)

    walk(tree)
    return steps


def _convert_yaml_to_steps(pl_dict, cal_steps, key=None):
    """
    Replace YAML ``task`` strings with matching :class:`plStep` objects.

    Parameters:
    pl_dict (dict or str): YAML-derived pipeline structure.
    cal_steps (list of plStep): Available steps.
    key (str or None): Parent key used during recursion.

    Returns:
    dict or plStep or str: Converted pipeline structure.

    Raises:
    ValueError: If a referenced task name is not found.
    """
    if isinstance(pl_dict, dict):
        for inner_key, value in pl_dict.items():
            pl_dict[inner_key] = _convert_yaml_to_steps(value, cal_steps, inner_key)
    if isinstance(pl_dict, str) and key == "task":
        matches = [step for step in cal_steps if step.name == pl_dict]
        if not matches:
            raise ValueError(f"Step '{pl_dict}' not found in available steps.")
        return matches[0]
    return pl_dict