import os
import re
import threading
import time
import warnings
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

# Unsharded write buffer used when DataSet(write_buffer=True). Pending rows
# override the sharded ``data`` rows until consolidate_storage merges them.
_PENDING_DATA = "pending_data"
_PENDING_EXISTS = "pending_row_exists"

# One re-entrant lock per zarr store, shared by every DataSet on that store.
# Zarr has no locking of its own, and on Windows a write fails with
# PermissionError if another thread has the same chunk file open.
_IO_LOCKS = {}
_IO_LOCKS_GUARD = threading.Lock()
_IO_RETRIES = 5
_IO_RETRY_DELAY = 0.05


def _io_lock_for(group):
    """
    Return the I/O lock shared by all DataSets on the same zarr store.

    Parameters:
    group (zarr.Group): Group whose store determines the lock.

    Returns:
    lock (threading.RLock): Lock for the store.
    """
    store = group.store
    store_root = getattr(store, "root", None)
    key = os.path.normcase(os.path.abspath(str(store_root))) if store_root is not None else id(store)
    with _IO_LOCKS_GUARD:
        if key not in _IO_LOCKS:
            _IO_LOCKS[key] = threading.RLock()
        return _IO_LOCKS[key]


def _row_masks(group):
    """
    Read the saved-row masks of a per-row parameter group.

    Call with the DataSet I/O lock held.

    Parameters:
    group (zarr.Group): Parameter group.

    Returns:
    main (np.ndarray or None): ``row_exists`` of the sharded array, or None
        if missing.
    pending (np.ndarray or None): ``pending_row_exists`` of the write
        buffer, or None if there is no buffer.
    """
    main = np.asarray(group["row_exists"][...], dtype=bool) if "row_exists" in group else None
    pending = (
        np.asarray(group[_PENDING_EXISTS][...], dtype=bool)
        if _PENDING_EXISTS in group else None
    )
    return main, pending


def _retry_io(func, *args, **kwargs):
    """
    Call a zarr I/O function, retrying briefly on PermissionError.

    Other processes (e.g. antivirus, indexing or sync tools on Windows) can
    briefly hold chunk files open, which makes zarr's atomic rename fail.

    Parameters:
    func (callable): Function performing the I/O.
    *args, **kwargs: Arguments passed to ``func``.

    Returns:
    result: Return value of ``func``.

    Raises:
    PermissionError: If every attempt fails.
    """
    for attempt in range(_IO_RETRIES):
        try:
            return func(*args, **kwargs)
        except PermissionError:
            if attempt == _IO_RETRIES - 1:
                raise
            time.sleep(_IO_RETRY_DELAY * (attempt + 1))


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
        custom_main_dir_overwrite=None,
        write_buffer=False,
    ):
        """
        Initialize the dataset and load its calibration definition.

        Parameters:
        zarr_path (str or zarr.Group): Path to the output zarr store or an
            already-open zarr group.
        cal_yaml_path (str or None): Path or alias for the calibration YAML.
            When None, the YAML must already be embedded in the zarr store.
            On first use, the YAML contents (not the path) are embedded in
            the zarr store.
        custom_path (str or None): Path to the Python file defining
            ``custom_cal_steps``. When None, embedded custom step source is
            used if available. On first use, the file contents (not the path)
            are embedded in the zarr store.
        zarr_mode (str): Mode used when opening ``zarr_path`` if a path was
            provided.
        custom_cal_steps (list of plStep or None): Custom calibration steps to
            use directly instead of loading them from ``custom_path``. These
            are not embedded in the zarr store, so the calibration definition
            is not written and the dataset cannot be reopened without them.
        custom_main_dir_overwrite (str or None): Replacement value for the
            top-level ``main_dir = ...`` assignment in the calibration and
            analysis custom step source. Use this when the raw data has moved.
            The embedded source is not modified, so pass this each time the
            dataset is opened. None keeps the ``main_dir`` in the source.
        write_buffer (bool): If True, per-row writes go to an unsharded
            pending buffer next to each sharded array (``pending_data`` and
            ``pending_row_exists`` in the parameter group). Single-row writes
            are much faster this way, but they create one file per row, so
            call ``consolidate_storage`` when done to merge them into the
            shards. Reads through a DataSet always include pending rows, but
            readers that open the zarr arrays directly only see consolidated
            rows. If False (default), rows are written straight to the
            sharded arrays.

        Raises:
        TypeError: If the supplied path arguments have invalid types.
        ValueError: If the zarr path or calibration definition is invalid.
        """
        if custom_path is not None and not isinstance(custom_path, str):
            raise TypeError("custom_path must be a string or None")
        if cal_yaml_path is not None and not isinstance(cal_yaml_path, str):
            raise TypeError("cal_yaml_path must be a string or None")
        if custom_main_dir_overwrite is not None and not isinstance(custom_main_dir_overwrite, str):
            raise TypeError("custom_main_dir_overwrite must be a string or None")

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

        self._io_lock = _io_lock_for(self.root)
        self.write_buffer = bool(write_buffer)
        try:
            with self._io_lock:
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
            custom_main_dir_overwrite=custom_main_dir_overwrite,
        )
        self.cal_yaml_path = cal_def["yaml_path"]
        self.custom_path = cal_def["custom_path"]
        self.cal_yaml_text = cal_def["yaml_text"]
        self.cal_custom_source = cal_def["custom_source"]
        self.custom_main_dir_overwrite = custom_main_dir_overwrite

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

    def register_analysis_definition(self, analysis_yaml_text, analysis_custom_source):
        """
        Persist the analysis definition inside the zarr metadata.

        Only the contents are stored, not file paths, so the zarr store can be
        moved.

        Parameters:
        analysis_yaml_text (str): Contents of the analysis YAML.
        analysis_custom_source (str or None): Source code defining
            ``custom_analysis_steps``, or None if there are no custom steps.

        Raises:
        ValueError: If the dataset already contains a different analysis
            definition. Differences only in the top-level ``main_dir``
            assignment are allowed.
        """
        existing = self._read_metadata()
        stored_yaml = existing.get("analysis_yaml")
        stored_custom = existing.get("analysis_custom_source")

        if stored_yaml is not None and analysis_yaml_text != stored_yaml:
            raise ValueError(
                "The dataset already contains a different analysis YAML definition"
            )
        if stored_custom is not None and not _custom_sources_match(analysis_custom_source, stored_custom):
            raise ValueError(
                "The dataset already contains different analysis custom step code"
            )

        update = {}
        if stored_yaml is None:
            update["analysis_yaml"] = analysis_yaml_text
        if stored_custom is None:
            update["analysis_custom_source"] = analysis_custom_source
        if update:
            self._write_metadata(update)

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

    def apply_cal(self, data_indices, outputs, replacements=None):
        """
        Evaluate calibration outputs with temporary input replacements.

        Parameters:
        data_indices (int, array-like, or None): Rows to evaluate for per-row
            and vectorized calibration steps.
        outputs (str or iterable[str]): Calibration outputs to produce.
        replacements (dict or None): Temporary parameter overrides used in
            place of the dataset-backed values.

        Returns:
        dict: Mapping of requested output name to the computed value.
        """
        requested_outputs = [outputs] if isinstance(outputs, str) else list(outputs)
        if not requested_outputs:
            raise ValueError("outputs must contain at least one calibration output")
        replacements = {} if replacements is None else dict(replacements)

        rows = self._normalize_rows(data_indices)
        step_paths = []
        for output_name in requested_outputs:
            path = pf.find_pl_path(self.cal_pl, output_name)
            if path is None:
                raise ValueError(f"Calibration pipeline cannot produce output '{output_name}'")
            step_paths.append(path)

        steps = _merge_step_paths(step_paths)
        local_values = {}
        replacement_values, replacement_is_global = self._normalize_apply_cal_replacements(replacements, rows)

        for step in steps:
            if all(name in replacement_values or name in local_values for name in step.return_names):
                continue
            step_rows = None if step.func_type in ("global", "global-res") else rows
            params, param_is_global = self._collect_apply_cal_params(
                step,
                step_rows,
                local_values,
                replacement_values,
                replacement_is_global,
            )
            out = step._run(params, param_is_global)
            for name, value in out.items():
                local_values[name] = value

        return {
            output_name: self._finalize_apply_cal_output(
                output_name,
                replacement_values[output_name] if output_name in replacement_values else local_values[output_name],
                rows,
            )
            for output_name in requested_outputs
        }

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
        Save parameters to zarr, producing them first if needed.

        Values that aren't in memory are loaded from zarr, and calibration
        parameters that aren't saved either are produced by running the
        calibration path, so you don't need to access a parameter before
        saving it.

        Parameters:
        names (iterable of str): Parameters to save.
        data_idx (int, array-like, or None): Rows to save for per-row
            parameters (ignored for global ones). None saves every row of a
            calibration parameter, or every row in memory of an analysis
            parameter (which can't be recomputed).

        Raises:
        ValueError: If a parameter is unknown, or requested rows of it are
            neither in memory, in zarr, nor producible by the calibration
            path.
        """
        rows = self._normalize_rows(data_idx)
        for name in names:
            calibratable = pf.find_pl_path(self.cal_pl, name) is not None
            meta = self._param_meta.get(name)
            if meta is None and calibratable:
                meta = self._infer_param_meta(name)
            if meta is None:
                meta = self._refresh_param_meta(name)
            if meta is None:
                raise ValueError(f"Parameter '{name}' is not available to save")
            if meta["global"]:
                if name not in self._global_cache:
                    self._get_global(name)  # loads from zarr or runs the cal path
                self._write_global_param(name)
                continue
            write_rows = rows
            if write_rows is None and calibratable:
                write_rows = np.arange(int(self.nrows), dtype=np.int32)
            if write_rows is not None:
                self._fetch_rows(name, write_rows)  # load or produce missing rows
            self._write_per_row_param(name, data_idx=write_rows)

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
            if name.startswith("_") or self._refresh_param_meta(name) is None:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            return getattr(self, name)

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
        with self._io_lock:
            stored = name in self.root
        if not stored:
            self._ensure_loaded(name, data_idx=None)
            if name in self._global_cache:
                return self._global_cache[name]
            with self._io_lock:
                stored = name in self.root
        if not stored:
            raise AttributeError(f"Global parameter '{name}' is not available")
        with self._io_lock:
            data = _retry_io(lambda: self.root[name]["data"][...])
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

        if missing:
            with self._io_lock:
                found_rows, data = _retry_io(self._read_saved_rows, name, missing)
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

    def _read_saved_rows(self, name, rows):
        """
        Read the saved subset of ``rows`` for a per-row parameter from zarr.

        Call with ``self._io_lock`` held.

        Parameters:
        name (str): Parameter name.
        rows (list of int): Rows to read.

        Returns:
        found_rows (list of int): Rows that are saved, in the order given.
        data (np.ndarray or list): Saved values for ``found_rows`` (empty
            list if none are saved).
        """
        if name not in self.root:
            return [], []
        group = self.root[name]
        main, pending = _row_masks(group)
        pending_rows = [row for row in rows if pending is not None and pending[row]]
        pending_set = set(pending_rows)
        main_rows = [
            row for row in rows
            if row not in pending_set and main is not None and main[row]
        ]
        if not pending_rows and not main_rows:
            return [], []
        values = {}
        if pending_rows:
            values.update(zip(pending_rows, group[_PENDING_DATA][pending_rows]))
        if main_rows:
            values.update(zip(main_rows, group["data"][main_rows]))
        found_rows = [row for row in rows if row in values]
        return found_rows, [values[row] for row in found_rows]

    def _saved_mask(self, name):
        """
        Return which rows of a per-row parameter are saved in zarr.

        Pending (buffered) rows count as saved.

        Parameters:
        name (str): Parameter name.

        Returns:
        mask (np.ndarray or None): Boolean mask of length ``nrows``, or None
            if the parameter has no saved per-row data.
        """
        with self._io_lock:
            if name not in self.root:
                return None
            main, pending = _retry_io(_row_masks, self.root[name])
        if main is None and pending is None:
            return None
        if main is None:
            return pending
        return main if pending is None else (main | pending)

    def saved_row_mask(self, name):
        """
        Return which rows of a per-row parameter are saved in zarr.

        Rows in the pending write buffer count as saved. Use this to check
        many rows at once instead of reading them one at a time.

        Parameters:
        name (str): Parameter name.

        Returns:
        mask (np.ndarray): Boolean array of length ``nrows``. All False if the
            parameter has no saved per-row data.
        """
        mask = self._saved_mask(name)
        if mask is None:
            return np.zeros(int(self.nrows), dtype=bool)
        return mask

    def _refresh_param_meta(self, name):
        """
        Load metadata for a parameter saved to zarr after this DataSet opened.

        Another DataSet on the same store (e.g. a background worker) may have
        created the parameter.

        Parameters:
        name (str): Parameter name.

        Returns:
        meta (dict or None): The parameter metadata, or None if the parameter
            is not saved.
        """
        with self._io_lock:
            if name not in self.root:
                return None
            attrs = dict(self.root[name].attrs)
        if "global" not in attrs:
            return None
        self._param_meta[name] = {
            "global": bool(attrs.get("global")),
            "pipeline_scope": attrs.get("pipeline_scope"),
            "step_name": attrs.get("step_name"),
            "step_index": attrs.get("step_index"),
        }
        return self._param_meta[name]

    def release_rows(self, data_idx, keep=()):
        """
        Drop cached per-row values for rows that are no longer needed.

        Values that were saved are reloaded from zarr on the next access, and
        calibration products are recomputed. Parameters produced by
        global-res calibration steps are kept, since those steps produce every
        row at once.

        Parameters:
        data_idx (int or array-like): Rows to drop from memory.
        keep (iterable of str): Additional parameter names to keep, e.g.
            outputs of global-res analysis steps.
        """
        rows = self._normalize_rows(data_idx)
        keep = set(keep) | {
            name
            for step in _flatten_pipeline_steps(self.cal_pl)
            if step.func_type == "global-res"
            for name in step.return_names
        }
        for name, lazy_attr in self._per_row_cache.items():
            if name in keep:
                continue
            for di in rows:
                lazy_attr._cache.pop(int(di), None)

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
        exists = self._saved_mask(name)
        if exists is None:
            return False
        exists = exists.copy()
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

    def _normalize_apply_cal_replacements(self, replacements, rows):
        """
        Normalize replacement values for apply_cal.
        """
        values = {}
        is_global = {}
        for name, value in replacements.items():
            replacement_is_global = self._is_known_global_param(name)
            is_global[name] = replacement_is_global
            if replacement_is_global:
                values[name] = value
                continue
            if rows is None:
                raise ValueError(
                    f"Replacement '{name}' is per-row and requires data_indices"
                )
            if len(rows) == 1:
                if isinstance(value, np.ndarray):
                    if value.ndim > 0 and len(value) == 1:
                        values[name] = value
                    else:
                        values[name] = np.asarray([value])
                elif isinstance(value, (list, tuple)) and len(value) == 1:
                    values[name] = np.asarray(value)
                else:
                    values[name] = np.asarray([value])
                continue
            if isinstance(value, np.ndarray) and len(value) == len(rows):
                values[name] = value
                continue
            if isinstance(value, (list, tuple)) and len(value) == len(rows):
                values[name] = np.asarray(value)
                continue
            if not isinstance(value, (list, tuple, np.ndarray)):
                values[name] = np.asarray([value] * len(rows))
                continue
            raise ValueError(
                f"Replacement '{name}' must provide one value per requested row"
            )
        return values, is_global

    def _finalize_apply_cal_output(self, name, value, rows):
        """
        Convert internal apply_cal results back to caller-facing shapes.
        """
        if self._is_known_global_param(name) or rows is None:
            return value

        if len(rows) != 1:
            return value

        if isinstance(value, np.ndarray):
            if value.ndim > 0 and len(value) == 1:
                return value[0]
            return value
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    def _collect_apply_cal_params(
        self,
        step,
        data_idx,
        local_values,
        replacement_values,
        replacement_is_global,
    ):
        """
        Collect concrete arguments for apply_cal without mutating dataset state.
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

            if param_name in replacement_values:
                params.append(replacement_values[param_name])
                param_is_global.append(replacement_is_global[param_name])
                continue

            if param_name in local_values:
                value = local_values[param_name]
                value_is_global = self._is_known_global_param(param_name)
                params.append(value)
                param_is_global.append(value_is_global)
                continue

            value = getattr(self, param_name)
            value_is_global = not isinstance(value, pf.LazyAttr)
            if value_is_global:
                params.append(value)
                param_is_global.append(True)
                continue
            if data_idx is None:
                raise ValueError(
                    f"Global step '{step.name}' cannot consume per-row parameter '{param_name}'"
                )
            params.append(value[data_idx])
            param_is_global.append(False)
        return params, param_is_global

    def _is_known_global_param(self, name):
        """
        Return True when name is known to be produced as a global parameter.
        """
        meta = self._param_meta.get(name)
        if meta is not None:
            return bool(meta["global"])
        path = pf.find_pl_path(self.cal_pl, name)
        if path is not None:
            return path[-1].func_type == "global"
        return False

    def _execute_step(self, step, data_idx=None, save=False, pipeline_scope=None, step_index=None, vectorize=True):
        """
        Execute a single calibration or analysis step.

        Parameters:
        step (plStep): Step to execute.
        data_idx (array-like or None): Rows for per-row and vectorized steps.
        save (bool): If True, write outputs to zarr.
        pipeline_scope (str or None): 'cal' or 'analysis', stored as metadata.
        step_index (int or None): Execution index, stored as metadata.
        vectorize (bool): If True (default), a 'vectorized' step runs on all
            rows in one call. If False, it runs one row at a time like a
            'per-row' step, recording failing rows instead of raising.

        Returns:
        failures (dict or None): Error message by row for row-at-a-time
            execution, or None for global and vectorized calls.
        """
        if not isinstance(step, pf.plStep):
            raise TypeError("step must be a plStep instance")

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

        if step.func_type == "vectorized" and vectorize:
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
        value = np.asarray(self._global_cache[name])

        def write():
            """
            Replace the parameter group with the cached global value.
            """
            if name in self.root:
                del self.root[name]
            group = self.root.create_group(name)
            group.create_array("data", data=value)
            _write_group_metadata(group, meta)

        with self._io_lock:
            _retry_io(write)

    def _write_per_row_param(self, name, data_idx=None):
        """
        Write per-row parameter data to the zarr store.

        All requested rows are written in one zarr call per array, so each
        shard and the ``row_exists`` array are rewritten once rather than once
        per row. With ``write_buffer`` the rows go to the unsharded pending
        buffer instead; otherwise they go to the sharded array and any
        pending copies of those rows are discarded.

        Parameters:
        name (str): Parameter name.
        data_idx (int, array-like, or None): Rows to write. None writes every
            cached row.

        Raises:
        ValueError: If there is nothing cached to save or a requested row is
            not cached.
        """
        meta = self._param_meta[name]
        lazy_attr = self._get_lazy_attr(name)
        if data_idx is None:
            if not lazy_attr._cache:
                raise ValueError(f"Parameter '{name}' has no cached rows to save")
            rows = np.array(sorted(lazy_attr._cache.keys()), dtype=np.int32)
        else:
            rows = np.unique(self._normalize_rows(data_idx))
        if len(rows) == 0:
            return

        for di in rows:
            if int(di) not in lazy_attr._cache:
                raise ValueError(f"Parameter '{name}' row {int(di)} is not cached in memory")
        values = np.stack([np.asarray(lazy_attr._cache[int(di)]) for di in rows])

        def write():
            """
            Create the parameter arrays if needed and write the rows.
            """
            try:
                group = self.root[name]
                data_array = group["data"]
                exists_array = group["row_exists"]
            except KeyError:
                group = self.root.require_group(name)
                if "data" in group:
                    data_array = group["data"]
                else:
                    first = values[0]
                    data_array = group.create_array(
                        "data",
                        shape=(int(self.nrows), *first.shape),
                        chunks=(_PER_ROW_CHUNK_ROWS, *first.shape),
                        shards=(_PER_ROW_SHARD_ROWS, *first.shape),
                        dtype=first.dtype,
                    )
                if "row_exists" in group:
                    exists_array = group["row_exists"]
                else:
                    exists_array = group.create_array(
                        "row_exists", data=np.zeros((int(self.nrows),), dtype=np.bool_)
                    )
            if self.write_buffer:
                if _PENDING_DATA in group:
                    pending_data = group[_PENDING_DATA]
                else:
                    pending_data = group.create_array(
                        _PENDING_DATA,
                        shape=data_array.shape,
                        chunks=(_PER_ROW_CHUNK_ROWS, *data_array.shape[1:]),
                        dtype=data_array.dtype,
                    )
                if _PENDING_EXISTS in group:
                    pending_exists = group[_PENDING_EXISTS]
                else:
                    pending_exists = group.create_array(
                        _PENDING_EXISTS, data=np.zeros((int(self.nrows),), dtype=np.bool_)
                    )
                _write_rows(pending_data, rows, values)
                pending_exists.oindex[rows] = True
            else:
                _write_rows(data_array, rows, values)
                exists_array.oindex[rows] = True
                if _PENDING_EXISTS in group:
                    # Stale buffered rows would otherwise shadow the new values.
                    pending_exists = group[_PENDING_EXISTS]
                    pending = np.asarray(pending_exists[...], dtype=bool)
                    if np.any(pending[rows]):
                        pending[rows] = False
                        pending_exists[...] = pending
            _write_group_metadata(group, meta)

        with self._io_lock:
            _retry_io(write)

    def _load_param_registry(self):
        """
        Load parameter metadata from the existing zarr store.
        """
        with self._io_lock:
            groups = [(name, dict(group.attrs)) for name, group in self.root.groups()]
        for name, attrs in groups:
            if name.startswith("_"):
                continue
            if "global" not in attrs:
                continue
            self._param_meta[name] = {
                "global": bool(attrs.get("global")),
                "pipeline_scope": attrs.get("pipeline_scope"),
                "step_name": attrs.get("step_name"),
                "step_index": attrs.get("step_index"),
            }

    def _delete_saved_param(self, name, data_idx=None):
        """
        Delete saved parameter data from zarr.

        Rows that are not saved are left alone, so nothing is written when
        there is nothing to delete.

        Parameters:
        name (str): Parameter name.
        data_idx (array-like or None): Rows to delete. None deletes the whole
            parameter.
        """
        meta = self._param_meta.get(name) or self._infer_param_meta(name)
        if meta is None:
            return
        rows = None if data_idx is None else self._normalize_rows(data_idx)

        def delete():
            """
            Delete the rows, or the group if no saved rows would remain.
            """
            if name not in self.root:
                return
            if meta["global"] or rows is None:
                del self.root[name]
                return
            group = self.root[name]
            main, pending = _row_masks(group)
            if main is None and pending is None:
                del self.root[name]
                return
            changed = []
            for mask, key in ((main, "row_exists"), (pending, _PENDING_EXISTS)):
                if mask is not None and np.any(mask[rows]):
                    mask[rows] = False
                    changed.append((mask, key))
            if not changed:
                return
            if not any(np.any(mask) for mask in (main, pending) if mask is not None):
                del self.root[name]
                return
            for mask, key in changed:
                group[key][...] = mask

        with self._io_lock:
            _retry_io(delete)

    def consolidate_storage(self):
        """
        Merge buffered (pending) rows into the sharded arrays.

        Run this after a session with ``write_buffer=True``, and before
        copying the zarr store, to bring the file count back down. It is safe
        to run at any time: stores without pending rows are left untouched,
        and an interrupted run leaves every row readable.

        Returns:
        nrows_merged (int): Number of pending (parameter, row) entries merged.
        """
        with self._io_lock:
            names = [
                name for name, group in self.root.groups()
                if not name.startswith("_") and _PENDING_EXISTS in group
            ]
        merged = 0
        for name in names:
            with self._io_lock:
                merged += _retry_io(_consolidate_group, self.root[name])
        return merged

    def _delete_param(self, name):
        """
        Remove a parameter from memory and from the zarr store.
        """
        self._global_cache.pop(name, None)
        self._per_row_cache.pop(name, None)
        self._param_meta.pop(name, None)
        self._invalidated_globals.discard(name)
        self._invalidated_rows.pop(name, None)
        with self._io_lock:
            if name in self.root:
                _retry_io(self.root.__delitem__, name)

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
        if name in self._global_cache:
            return True
        with self._io_lock:
            return name in self.root and "data" in self.root[name]

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
        exists = self._saved_mask(name)
        if exists is None:
            return False
        return bool(np.all(exists[rows]))

    def _resolve_cal_definition(self, cal_yaml_path, custom_path, custom_cal_steps, custom_main_dir_overwrite):
        """
        Resolve the calibration definition from inputs and embedded metadata.

        Only YAML and custom step contents are embedded in the zarr metadata,
        not file paths, so the zarr store can be moved. The only path that
        needs updating after a move is ``main_dir`` in the custom step source,
        via ``custom_main_dir_overwrite``.

        Parameters:
        cal_yaml_path (str or None): Path or alias for the calibration YAML,
            or None to use the embedded YAML.
        custom_path (str or None): Path to the custom calibration step file,
            or None to use the embedded source.
        custom_cal_steps (list of plStep or None): Custom steps to use
            directly. When given, nothing is embedded.
        custom_main_dir_overwrite (str or None): Replacement for the
            top-level ``main_dir`` assignment in the custom source, or None to
            keep it.

        Returns:
        dict: Definition with keys ``yaml_path`` (path read this session, or
            None if embedded), ``custom_path`` (same), ``yaml_text``,
            ``custom_source`` (source that is executed, after the ``main_dir``
            overwrite) and ``custom_steps``.

        Raises:
        ValueError: If no YAML is available, a path has the wrong extension,
            or the supplied definition does not match the embedded one.
        """
        metadata = self._metadata
        stored_yaml = metadata.get("cal_yaml")
        stored_custom = metadata.get("cal_custom_source")

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

        if yaml_text is None:
            raise ValueError(
                "No calibration YAML was provided and no embedded definition exists in the dataset"
            )

        if custom_path is not None:
            resolved_custom_path = os.path.abspath(custom_path)
            custom_source = _read_text_file(resolved_custom_path)
        elif stored_custom is not None:
            custom_source = stored_custom

        if stored_yaml is not None and yaml_text != stored_yaml:
            raise ValueError("Provided calibration YAML does not match the dataset definition")
        if (
            stored_custom is not None
            and custom_path is not None
            and not _custom_sources_match(custom_source, stored_custom)
        ):
            raise ValueError("Provided calibration custom steps do not match the dataset definition")

        if stored_yaml is None:
            self._write_metadata(
                {
                    "cal_yaml": yaml_text,
                    "cal_custom_source": custom_source,
                }
            )

        if (
            custom_main_dir_overwrite is not None
            and custom_source is not None
            and not _has_main_dir_assignment(custom_source)
        ):
            warnings.warn(
                "custom_main_dir_overwrite was given, but the calibration custom "
                "step source has no top-level 'main_dir = ...' assignment",
                stacklevel=3,
            )
        custom_source = _overwrite_main_dir_in_source(custom_source, custom_main_dir_overwrite)

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
        with self._io_lock:
            return dict(self.root.attrs.get(self._METADATA_ATTR, {}))

    def _write_metadata(self, update):
        """
        Update the pipeline_v2 metadata block in the zarr root.
        """
        with self._io_lock:
            current = self._read_metadata()
            current.update(update)
            current["schema_version"] = self._SCHEMA_VERSION
            _retry_io(self.root.attrs.__setitem__, self._METADATA_ATTR, current)
        self._metadata = current

    def _record_failures(self, step_name, failures):
        """
        Persist per-row execution failures to the ``_failures`` zarr group.
        """
        try:
            with self._io_lock:
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
                _retry_io(group.attrs.__setitem__, "failures", payload)
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


_MAIN_DIR_PATTERN = re.compile(r"^(main_dir\s*=\s*).*$", re.MULTILINE)


def _has_main_dir_assignment(source):
    """
    Check whether source code has a top-level ``main_dir = ...`` assignment.

    Parameters:
    source (str): Python source code.

    Returns:
    has_main_dir (bool): True if an unindented ``main_dir = ...`` line exists.
    """
    return _MAIN_DIR_PATTERN.search(source) is not None


def _overwrite_main_dir_in_source(source, main_dir_overwrite):
    """
    Replace the first top-level ``main_dir = ...`` assignment when requested.

    Only unindented assignments are replaced, so a local ``main_dir`` inside a
    function is left alone.

    Parameters:
    source (str or None): Python source code.
    main_dir_overwrite (str or None): New value for ``main_dir``. None leaves
        the source unchanged.

    Returns:
    source (str or None): Source with the assignment replaced, or the input
        unchanged if ``source`` or ``main_dir_overwrite`` is None or there is
        no top-level assignment.
    """
    if source is None or main_dir_overwrite is None:
        return source
    return _MAIN_DIR_PATTERN.sub(
        lambda match: f"{match.group(1)}{main_dir_overwrite!r}",
        source,
        count=1,
    )


def _custom_sources_match(source_a, source_b):
    """
    Compare custom step sources, ignoring the top-level ``main_dir`` value.

    Parameters:
    source_a (str or None): First source.
    source_b (str or None): Second source.

    Returns:
    match (bool): True if the sources are identical apart from the value
        assigned to ``main_dir``.
    """
    return _overwrite_main_dir_in_source(source_a, "") == _overwrite_main_dir_in_source(source_b, "")


def _consolidate_group(group):
    """
    Merge a parameter group's pending rows into its sharded array.

    The sharded data is written first, then ``row_exists``, and the pending
    arrays are deleted last, so an interruption at any point leaves every row
    readable. Rows are merged one shard at a time to bound memory use: each
    affected shard's rows are read, patched, and written back in one call,
    so every shard is rewritten once. (Rows that were never saved keep the
    fill value, which zarr doesn't store.)

    Call with the DataSet I/O lock held.

    Parameters:
    group (zarr.Group): Parameter group with a pending buffer.

    Returns:
    nrows_merged (int): Number of rows merged.
    """
    pending = np.asarray(group[_PENDING_EXISTS][...], dtype=bool)
    rows = np.flatnonzero(pending)
    if _PENDING_DATA not in group:
        # An earlier run already merged the rows and deleted the data buffer.
        del group[_PENDING_EXISTS]
        return 0
    if len(rows):
        data_array = group["data"]
        pending_data = group[_PENDING_DATA]
        shard_rows = data_array.shards[0] if data_array.shards else _PER_ROW_SHARD_ROWS
        for shard_idx in np.unique(rows // shard_rows):
            batch = rows[rows // shard_rows == shard_idx]
            start = int(batch[0])
            stop = int(batch[-1]) + 1
            block = data_array[start:stop]
            block[batch - start] = pending_data.oindex[batch]
            data_array[start:stop] = block
        exists = np.asarray(group["row_exists"][...], dtype=bool)
        exists[rows] = True
        group["row_exists"][...] = exists
    del group[_PENDING_DATA]
    del group[_PENDING_EXISTS]
    return len(rows)


def _write_rows(array, rows, values):
    """
    Write values into rows of a zarr array, one slice per contiguous run.

    Slices are used instead of ``array.oindex[rows] = values`` because zarr
    3.1.5 fails on integer-array writes into sharded arrays with more than
    one dimension.

    Parameters:
    array (zarr.Array): Array whose first axis is the row axis.
    rows (np.ndarray): Sorted, unique row indices.
    values (np.ndarray): Values, one per row along the first axis.
    """
    rows = np.asarray(rows)
    values = np.asarray(values)
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    for run_rows, run_values in zip(np.split(rows, breaks), np.split(values, breaks)):
        array[int(run_rows[0]):int(run_rows[-1]) + 1] = run_values


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

    All attributes are set in one ``update_attributes`` call, so the group
    metadata file is rewritten once. (``group.attrs.update`` would rewrite it
    once per key.)

    Parameters:
    group (zarr.Group): Parameter group.
    meta (dict): Metadata with keys ``global``, ``pipeline_scope``,
        ``step_name`` and ``step_index``.
    """
    group.update_attributes(
        {
            "global": bool(meta["global"]),
            "pipeline_scope": meta.get("pipeline_scope"),
            "step_name": meta.get("step_name"),
            "step_index": meta.get("step_index"),
            "write_time": datetime.now().strftime("%Y%m%d-%H:%M:%S"),
        }
    )


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


def _merge_step_paths(paths):
    """
    Merge multiple step paths into one ordered list without duplicates.
    """
    merged = []
    seen = set()
    for path in paths:
        for step in path:
            if step.name in seen:
                continue
            seen.add(step.name)
            merged.append(step)
    return merged


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