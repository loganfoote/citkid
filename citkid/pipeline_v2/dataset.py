import os
from datetime import datetime

import numpy as np
import yaml
import zarr

from . import default_steps
from . import framework as pf


_CAL_YAML_ALIASES = {
    "ts": "cal-ts.yaml",
    "iq": "cal-iqonly.yaml",
    "ts_offres": "cal-offres.yaml",
}

_PER_ROW_CHUNK_ROWS = 1
_PER_ROW_SHARD_ROWS = 256


class DataSet:
    """
    Zarr-backed dataset for calibration and analysis products.

    Parameters are cached in memory, loaded lazily from zarr, and produced on
    demand from the calibration pipeline when possible.
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
        self._invalidated_globals = set()
        self._invalidated_rows = {}

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
        self._cal_dependency_graph = _build_param_dependency_graph(self.cal_pl)

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

        self._write_metadata(
            {
                "analysis_yaml": analysis_yaml_text,
                "analysis_yaml_path": analysis_yaml_path,
                "analysis_custom_source": analysis_custom_source,
                "analysis_custom_path": analysis_custom_path,
            }
        )

    def set_analysis_step_names(self, step_names_by_index):
        """
        Cache analysis step names by execution index.
        """
        self._analysis_step_names = dict(step_names_by_index)

    def get_downstream_calibration_params(self, source_names):
        """
        Return calibration parameters that depend on any of ``source_names``.
        """
        downstream = []
        seen = set()
        queue = [name for name in source_names if isinstance(name, str)]
        while queue:
            source = queue.pop(0)
            for target in sorted(self._cal_dependency_graph.get(source, ())):
                if target in seen:
                    continue
                seen.add(target)
                downstream.append(target)
                queue.append(target)
        return downstream

    def invalidate_memory_params(self, names, data_idx=None):
        """
        Mark parameters as unavailable in memory for the requested scope.
        """
        rows = self._normalize_rows(data_idx)
        for name in names:
            meta = self._param_meta.get(name) or self._infer_param_meta(name)
            if meta is None:
                continue
            if meta["global"]:
                self._global_cache.pop(name, None)
                self._invalidated_globals.add(name)
                continue

            lazy_attr = self._per_row_cache.get(name)
            if rows is None:
                if lazy_attr is not None:
                    lazy_attr._cache.clear()
                self._invalidated_rows[name] = None
                continue

            if lazy_attr is not None:
                for di in rows:
                    lazy_attr._cache.pop(int(di), None)
            if name in self._invalidated_rows and self._invalidated_rows[name] is None:
                continue
            invalid_rows = self._invalidated_rows.setdefault(name, set())
            invalid_rows.update(int(di) for di in rows)

    def delete_saved_params(self, names, data_idx=None):
        """
        Delete persisted parameter data from zarr for the requested scope.
        """
        rows = self._normalize_rows(data_idx)
        for name in names:
            self._delete_saved_param(name, data_idx=rows)

    def write_params(self, names, data_idx=None):
        """
        Persist parameters that already exist in memory.
        """
        rows = self._normalize_rows(data_idx)
        for name in names:
            if name not in self._param_meta:
                raise ValueError(f"Parameter '{name}' is not available to save")
            meta = self._param_meta[name]
            if meta["global"]:
                self._write_global_param(name)
            else:
                self._write_per_row_param(name, data_idx=rows)

    def __getattr__(self, name):
        """
        Lazily access a stored or calibratable parameter.
        """
        if name in self._param_meta:
            meta = self._param_meta[name]
            if meta["global"]:
                return self._get_global(name)
            if not self._parameter_has_any_available_row(name):
                if pf.find_pl_path(self.cal_pl, name) is None:
                    raise AttributeError(f"Per-row parameter '{name}' is not available")
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
        """
        if name in self._global_cache:
            return self._global_cache[name]
        if name in self._invalidated_globals:
            self._ensure_loaded(name, data_idx=None)
            if name in self._invalidated_globals:
                raise AttributeError(f"Global parameter '{name}' is not available")
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
        """
        if name not in self._per_row_cache:
            self._per_row_cache[name] = pf.LazyAttr(self, name)
        return self._per_row_cache[name]

    def _fetch_rows(self, name, data_idx=None):
        """
        Load per-row data into the lazy cache and return the requested rows.
        """
        if data_idx is None:
            raise ValueError(f"data_idx required for per-row parameter '{name}'")

        rows = self._normalize_rows(data_idx)
        lazy_attr = self._get_lazy_attr(name)
        invalid_rows = self._invalidated_rows.get(name, set())
        if invalid_rows is None:
            invalid_rows = set(int(di) for di in rows)
        missing = [
            int(di)
            for di in rows
            if int(di) not in lazy_attr._cache and int(di) not in invalid_rows
        ]

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

    def _normalize_rows(self, data_idx):
        """
        Normalize a row selector into a 1-D int32 numpy array.
        """
        if data_idx is None:
            return None
        rows = np.atleast_1d(np.asarray(data_idx, dtype=np.int32))
        if rows.ndim != 1:
            raise ValueError("data_idx must resolve to a 1-D array of row indices")
        return rows

    def _infer_param_meta(self, name):
        """
        Infer metadata for a calibratable parameter that has not been loaded.
        """
        path = pf.find_pl_path(self.cal_pl, name)
        if path is None:
            return None
        step = path[-1]
        self._param_meta.setdefault(
            name,
            {
                "global": step.func_type == "global",
                "pipeline_scope": "cal",
                "step_name": step.name,
                "step_index": self.cal_step_indices.get(step.name),
            },
        )
        return self._param_meta[name]

    def _parameter_has_any_available_row(self, name):
        """
        Check whether a per-row parameter has at least one available row.
        """
        invalid_rows = self._invalidated_rows.get(name, set())
        if invalid_rows is None:
            return False
        lazy_attr = self._per_row_cache.get(name)
        if lazy_attr is not None:
            for di in lazy_attr._cache.keys():
                if int(di) not in invalid_rows:
                    return True
        if name not in self.root or "row_exists" not in self.root[name]:
            return False
        exists = np.asarray(self.root[name]["row_exists"][...], dtype=bool)
        if invalid_rows:
            exists[list(invalid_rows)] = False
        return bool(np.any(exists))

    def _ensure_loaded(self, name, data_idx=None):
        """
        Ensure that a calibratable parameter exists.
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
                all_rows = np.arange(int(self.nrows), dtype=np.int32)
                if self._step_outputs_exist(step, data_idx=all_rows):
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
            rows = self._normalize_rows(data_idx)
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
        rows = self._normalize_rows(data_idx)

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
                step_rows = np.asarray([int(di)], dtype=np.int32)
                params, param_is_global = self._collect_params(step, step_rows)
                out = step._run(params, param_is_global)
                for name, value in out.items():
                    self._store_param(
                        name,
                        value,
                        is_global=False,
                        data_idx=step_rows,
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
        """
        if name in self._get_reserved_attrs():
            raise ValueError(f"Cannot create parameter '{name}' because the name is reserved")

        self._param_meta[name] = {
            "global": bool(is_global),
            "pipeline_scope": pipeline_scope,
            "step_name": step_name,
            "step_index": step_index,
        }

        if is_global:
            self._invalidated_globals.discard(name)
            scalar_array = np.asarray(value)
            self._global_cache[name] = _unwrap_scalar(scalar_array) if scalar_array.shape == () else value
            if save:
                self._write_global_param(name)
            return

        rows = self._normalize_rows(data_idx)
        lazy_attr = self._get_lazy_attr(name)
        row_values = _normalize_row_values(value, rows)
        invalid_rows = self._invalidated_rows.get(name)
        if invalid_rows is None:
            invalid_rows = set()
            self._invalidated_rows[name] = invalid_rows
        if invalid_rows is not None:
            invalid_rows.difference_update(int(di) for di in rows)
            if not invalid_rows:
                self._invalidated_rows.pop(name, None)
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
        """
        meta = self._param_meta[name]
        lazy_attr = self._get_lazy_attr(name)
        if data_idx is None:
            if not lazy_attr._cache:
                raise ValueError(f"Parameter '{name}' has no cached rows to save")
            rows = np.array(sorted(lazy_attr._cache.keys()), dtype=np.int32)
        else:
            rows = self._normalize_rows(data_idx)
        if len(rows) == 0:
            return

        if name not in self.root:
            first = np.asarray(lazy_attr._cache[int(rows[0])])
            shape = (int(self.nrows), *first.shape)
            chunk_shape = (_PER_ROW_CHUNK_ROWS, *first.shape)
            shard_shape = (_PER_ROW_SHARD_ROWS, *first.shape)
            group = self.root.create_group(name)
            group.create_array(
                "data",
                shape=shape,
                chunks=chunk_shape,
                shards=shard_shape,
                dtype=first.dtype,
            )
            group.create_array("row_exists", data=np.zeros((int(self.nrows),), dtype=np.bool_))
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

    def _delete_saved_param(self, name, data_idx=None):
        """
        Delete saved parameter data from zarr.
        """
        if name not in self.root:
            return

        meta = self._param_meta.get(name) or self._infer_param_meta(name)
        if meta is None:
            return

        if meta["global"] or data_idx is None:
            del self.root[name]
            return

        group = self.root[name]
        if "row_exists" not in group:
            del self.root[name]
            return

        rows = self._normalize_rows(data_idx)
        group["row_exists"][rows] = False
        if not bool(np.any(group["row_exists"][...])):
            del self.root[name]

    def _delete_param(self, name):
        """
        Remove a parameter from memory and from the zarr store.
        """
        self._global_cache.pop(name, None)
        self._per_row_cache.pop(name, None)
        self._param_meta.pop(name, None)
        self._invalidated_globals.discard(name)
        self._invalidated_rows.pop(name, None)
        if name in self.root:
            del self.root[name]

    def _step_outputs_exist(self, step, data_idx):
        """
        Check whether all outputs for a step already exist.
        """
        if step.func_type == "global":
            return all(self._has_global(name) for name in step.return_names)
        if step.func_type == "global-res":
            rows = np.arange(int(self.nrows), dtype=np.int32)
            return all(self._has_rows(name, rows) for name in step.return_names)
        rows = self._normalize_rows(data_idx)
        return all(self._has_rows(name, rows) for name in step.return_names)

    def _step_missing_rows(self, step, data_idx):
        """
        Return the subset of rows whose outputs are missing for a step.
        """
        rows = self._normalize_rows(data_idx)
        missing = []
        for di in rows:
            one_row = np.asarray([int(di)], dtype=np.int32)
            if not all(self._has_rows(name, one_row) for name in step.return_names):
                missing.append(int(di))
        return np.asarray(missing, dtype=np.int32)

    def _has_global(self, name):
        """
        Check whether a global parameter exists in memory or zarr.
        """
        if name in self._invalidated_globals:
            return False
        return name in self._global_cache or (name in self.root and "data" in self.root[name])

    def _has_rows(self, name, rows):
        """
        Check whether specific rows exist for a per-row parameter.
        """
        rows = self._normalize_rows(rows)
        if name in self._invalidated_rows and self._invalidated_rows[name] is None:
            return False
        invalid_rows = self._invalidated_rows.get(name)
        if invalid_rows and any(int(di) in invalid_rows for di in rows):
            return False
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
        """
        return dict(self.root.attrs.get(self._METADATA_ATTR, {}))

    def _write_metadata(self, update):
        """
        Update the pipeline_v2 metadata block in the zarr root.
        """
        current = self._read_metadata()
        current.update(update)
        current["schema_version"] = self._SCHEMA_VERSION
        self.root.attrs[self._METADATA_ATTR] = current
        self._metadata = current

    def _record_failures(self, step_name, failures):
        """
        Persist per-row execution failures to the ``_failures`` zarr group.
        """
        try:
            group = self.root.require_group(f"_failures/{step_name}")
            payload = dict(group.attrs.get("failures", {}))
            payload.update(
                {
                    f"idx{di}": {
                        "error": message,
                        "traceback": message,
                        "time": datetime.now().strftime("%Y%m%d-%H:%M:%S"),
                    }
                    for di, message in failures.items()
                }
            )
            group.attrs["failures"] = payload
        except Exception:
            pass

    def _get_reserved_attrs(self):
        """
        Return the set of reserved public DataSet attribute names.
        """
        if DataSet._RESERVED_ATTRS is None:
            DataSet._RESERVED_ATTRS = {name for name in dir(type(self)) if not name.startswith("_")}
        return DataSet._RESERVED_ATTRS


def _resolve_cal_yaml_path(cal_yaml_path):
    """
    Resolve a calibration YAML alias or validate a filesystem path.
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
    """
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_custom_cal_steps_from_source(source):
    """
    Execute custom calibration step source and extract ``custom_cal_steps``.
    """
    if source is None:
        return []
    namespace = {}
    exec(compile(source, "<pipeline_v2_custom_cal>", "exec"), namespace)
    return list(namespace.get("custom_cal_steps", []))


def _normalize_row_values(value, rows):
    """
    Normalize per-row outputs into one value per requested row.
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
    """
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _write_group_metadata(group, meta):
    """
    Write pipeline_v2 provenance metadata onto a zarr parameter group.
    """
    group.attrs["global"] = bool(meta["global"])
    group.attrs["pipeline_scope"] = meta.get("pipeline_scope")
    group.attrs["step_name"] = meta.get("step_name")
    group.attrs["step_index"] = meta.get("step_index")
    group.attrs["write_time"] = datetime.now().strftime("%Y%m%d-%H:%M:%S")


def _step_indices(tree):
    """
    Build a name-to-index mapping for a pipeline tree.
    """
    steps = _flatten_pipeline_steps(tree)
    return {step.name: index for index, step in enumerate(steps, start=1)}


def _flatten_pipeline_steps(tree):
    """
    Flatten a pipeline tree into execution order.
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


def _build_param_dependency_graph(tree):
    """
    Build a parameter dependency graph from the calibration pipeline.
    """
    graph = {}
    for step in _flatten_pipeline_steps(tree):
        input_names = [name for name in step.param_names if name != "data_idx"]
        for input_name in input_names:
            graph.setdefault(input_name, set()).update(step.return_names)
    return graph


def _convert_yaml_to_steps(pl_dict, cal_steps, key=None):
    """
    Replace YAML ``task`` strings with matching :class:`plStep` objects.
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