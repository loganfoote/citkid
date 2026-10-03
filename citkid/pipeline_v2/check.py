"""
Check a DataSet's calibration steps (including custom steps) on one row.

``check_calibration`` (also ``DataSet.check_cal``) runs every calibration step
for one data index in execution order, using only the step functions (it
doesn't read or write the DataSet's caches or zarr store), and reports for each
step the inputs and outputs (type, shape, dtype), problems with the structure
the pipeline expects, and, for a failing step, the error with a traceback into
the step's function. Steps after a failed step that need its outputs are
skipped; independent branches still run, so several problems are reported at
once.
"""

import os
import time
import traceback

import numpy as np

from . import framework as pf
from .dataset import _flatten_pipeline_steps

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_INTERNAL_FILES = {os.path.normcase(os.path.join(_PIPELINE_DIR, name))
                   for name in ("framework.py", "check.py", "dataset.py")}

# Template parameter names (see templates/custom_steps_template.py) and what
# the default calibration steps expect of them.
_REAL_SWEEP_FREQS = ("ff", "fg")
_COMPLEX_1D = ("zf", "zg", "zt")
_PAIRS = (("ff", "zf"), ("fg", "zg"))


def _user_traceback(exc):
    """
    Format an exception with the pipeline's own frames left out.

    Parameters:
    exc (BaseException): the exception.

    Returns:
    text (str or None): the traceback, starting at the step's function, or
        None if every frame is internal (the error was found by the pipeline,
        e.g. a wrong number of outputs, and its message says what's wrong).
    """
    tbe = traceback.TracebackException.from_exception(exc)
    frames = [f for f in tbe.stack
              if os.path.normcase(os.path.abspath(f.filename)) not in _INTERNAL_FILES]
    if not frames:
        return None
    tbe.stack = traceback.StackSummary.from_list(frames)
    return "".join(tbe.format())


def _structure_problems(step, out, nrows):
    """
    Return errors in outputs whose structure the pipeline relies on.

    Parameters:
    step (plStep): the step.
    out (dict): outputs by name.
    nrows (int or None): number of rows, if known.

    Returns:
    problems (list of str): error messages (empty if fine).
    """
    problems = []
    if "nrows" in out:
        value = out["nrows"]
        if not (np.ndim(value) == 0 and np.isreal(value)
                and float(np.real(value)).is_integer() and value >= 0):
            problems.append(f"nrows = {pf.describe_value(value)}, but nrows must be a whole "
                            f"number (the number of rows).")
    if step.func_type == "global-res" and nrows is not None:
        for name, value in out.items():
            length = len(value) if np.ndim(value) > 0 else None
            if length != nrows:
                problems.append(
                    f"'{name}' is {pf.describe_value(value)}, but a global-res step must "
                    f"return one entry per resonator: length nrows = {nrows} along axis 0.")
    return problems


def _template_warnings(row_values):
    """
    Return warnings about template parameters that look wrong for one row.

    Parameters:
    row_values (dict): the row's values of the step's outputs, by name.

    Returns:
    warnings (list of str): warning messages (empty if fine).
    """
    found = []
    for name in _REAL_SWEEP_FREQS:
        if name in row_values:
            f = np.asarray(row_values[name])
            if f.ndim != 1 or np.iscomplexobj(f):
                found.append(f"'{name}' should be a real 1-D frequency array; got "
                             f"{pf.describe_value(f)}.")
            elif len(f) > 1 and not np.all(np.diff(f) > 0):
                found.append(f"'{name}' should be sorted in increasing order (the default "
                             f"calibration steps assume sorted sweeps).")
    for name in _COMPLEX_1D:
        if name in row_values:
            z = np.asarray(row_values[name])
            if z.ndim != 1 or not np.iscomplexobj(z):
                found.append(f"'{name}' should be a complex 1-D array; got "
                             f"{pf.describe_value(z)}.")
    for f_name, z_name in _PAIRS:
        if f_name in row_values and z_name in row_values:
            f_shape = np.shape(row_values[f_name])
            z_shape = np.shape(row_values[z_name])
            if f_shape != z_shape:
                found.append(f"'{f_name}' {f_shape} and '{z_name}' {z_shape} should have the "
                             f"same shape.")
    return found


def _row_value(value, data_idx):
    """
    Return the value of a DataSet parameter for one row (or a global value).

    Parameters:
    value: ``getattr(DS, name)``.
    data_idx (int): row.

    Returns:
    value: the row's value.
    """
    if isinstance(value, pf.LazyAttr):
        return value[data_idx]
    return value


def check_calibration(DS, data_idx=0, verbose=True):
    """
    Run every calibration step for one row and report what each produced.

    Only the step functions are called, with each step's inputs taken from
    the outputs of earlier steps; the DataSet's caches and zarr store are not
    read or written, except for inputs no calibration step produces (outputs
    of the analysis, e.g. ``p_amp``), which are read from the DataSet if it
    has them for ``data_idx``.

    Parameters:
    DS (DataSet): dataset whose calibration (custom and default steps) to
        check.
    data_idx (int): row to check. Default 0.
    verbose (bool): If True (default), print the report.

    Returns:
    results (list of dict): one entry per step, in execution order, with
        keys ``step`` (name), ``func_type``, ``function`` (name and source
        location), ``status`` ('ok', 'warning', 'failed', 'needs analysis'
        (an input is an analysis output the dataset doesn't have yet) or
        'skipped' (an input comes from a failed or unchecked step)),
        ``inputs`` and ``outputs`` (name -> description of the row's value),
        ``messages`` (list of str: problems, warnings, or why the step was
        skipped), ``error`` (str or None), ``traceback`` (str or None; into
        the step's function) and ``seconds`` (float).
    """
    data_idx = int(data_idx)
    steps = _flatten_pipeline_steps(DS.cal_pl)
    produced_by = {name: step.name for step in steps for name in step.return_names}
    global_values, row_values = {}, {}
    unavailable = {}  # output name -> (root step, its status) that blocks it
    results = []

    for step in steps:
        result = {"step": step.name, "func_type": step.func_type,
                  "function": pf.function_location(step.func), "status": "ok",
                  "inputs": {}, "outputs": {}, "messages": [], "error": None,
                  "traceback": None, "seconds": 0.0}
        results.append(result)
        params, is_global, reason = [], [], None
        for name in step.param_names:
            if name == "data_idx":
                if step.func_type in ("global", "global-res"):
                    params.append(None)
                    is_global.append(True)
                else:
                    params.append(np.asarray([data_idx], dtype=np.int32))
                    is_global.append(False)
                result["inputs"][name] = f"int {data_idx}"
                continue
            if name in global_values:
                params.append(global_values[name])
                is_global.append(True)
                result["inputs"][name] = pf.describe_value(global_values[name])
                continue
            if name in row_values:
                value = row_values[name]
                params.append([value])
                is_global.append(False)
                result["inputs"][name] = pf.describe_value(value)
                continue
            if name in unavailable:
                root, root_status = unavailable[name]
                status = "skipped"
                reason = (f"after failed step '{root}'" if root_status == "failed"
                          else f"after '{root}' (needs analysis)")
                break
            if name not in produced_by:
                try:
                    value = _row_value(getattr(DS, name), data_idx)
                except Exception:
                    root, status = step.name, "needs analysis"
                    reason = (f"needs '{name}', an analysis output the dataset doesn't have "
                              f"for data_idx {data_idx}; run the analysis for this row to "
                              f"check this step")
                    break
                params.append([value])
                is_global.append(False)
                result["inputs"][name] = pf.describe_value(value) + " (from the dataset)"
                continue
            root, status = step.name, "failed"
            reason = f"needs '{name}', which step '{produced_by[name]}' produces later"
            break
        if reason is not None:
            result["status"] = status
            result["messages"].append(reason)
            # Later steps needing these outputs are blocked by the same root step.
            blocker = (root, root_status) if status == "skipped" else (step.name, status)
            unavailable.update({name: blocker for name in step.return_names})
            continue

        start = time.perf_counter()
        try:
            out = step._run(params, is_global)
            problems = _structure_problems(step, out, global_values.get("nrows"))
            if problems:
                raise ValueError(" ".join(problems))
        except Exception as exc:
            result["seconds"] = time.perf_counter() - start
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = _user_traceback(exc)
            unavailable.update({name: (step.name, "failed") for name in step.return_names})
            continue
        result["seconds"] = time.perf_counter() - start

        produced = {}
        for name, value in out.items():
            if step.func_type == "global":
                global_values[name] = value
                produced[name] = value
            elif step.func_type == "global-res":
                row_values[name] = value[data_idx]
                produced[name] = value[data_idx]
                result["outputs"][name] = (f"{pf.describe_value(value[data_idx])} "
                                           f"(row {data_idx} of {pf.describe_value(value)})")
                continue
            else:
                row_values[name] = value[0]
                produced[name] = value[0]
            result["outputs"][name] = pf.describe_value(produced[name])
        warnings_found = _template_warnings(produced)
        if warnings_found:
            result["status"] = "warning"
            result["messages"].extend(warnings_found)

    if verbose:
        print(format_check_report(results, data_idx))
    return results


def format_check_report(results, data_idx):
    """
    Format the results of ``check_calibration`` as a readable report.

    Parameters:
    results (list of dict): output of ``check_calibration``.
    data_idx (int): row that was checked.

    Returns:
    text (str): the report.
    """
    labels = {"ok": "ok", "warning": "WARNING", "failed": "FAILED",
              "needs analysis": "analysis", "skipped": "skipped"}
    name_width = max((len(r["step"]) for r in results), default=10)
    lines = [f"Calibration check for data_idx {data_idx} ({len(results)} steps)"]
    for r in results:
        seconds = f"{r['seconds']:6.2f} s" if r["status"] in ("ok", "warning", "failed") else " " * 8
        if r["status"] == "skipped":
            lines.append(f"  {labels[r['status']]:<8} {r['step']:<{name_width}}  "
                         f"{r['func_type']:<10} {seconds}  ({r['messages'][0]})")
            continue
        lines.append(f"  {labels[r['status']]:<8} {r['step']:<{name_width}}  "
                     f"{r['func_type']:<10} {seconds}  {r['function']}")
        indent = " " * 11
        if r["status"] in ("failed", "warning") and r["inputs"]:
            lines.append(indent + "inputs:  " + ", ".join(f"{k} = {v}" for k, v in r["inputs"].items()))
        if r["outputs"]:
            lines.append(indent + "outputs: " + ", ".join(f"{k} = {v}" for k, v in r["outputs"].items()))
        for message in r["messages"]:
            lines.append(indent + message)
        if r["error"]:
            lines.append(indent + r["error"])
            if r["traceback"]:
                lines.extend(indent + "  " + line for line in r["traceback"].rstrip().splitlines())
    counts = {status: sum(r["status"] == status for r in results) for status in labels}
    lines.append(f"Summary: {counts['ok']} ok, {counts['warning']} with warnings, "
                 f"{counts['failed']} failed, {counts['needs analysis']} need analysis "
                 f"outputs, {counts['skipped']} skipped")
    return "\n".join(lines)
