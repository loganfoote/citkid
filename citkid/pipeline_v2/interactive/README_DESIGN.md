# pipeline_v2.interactive Design

## Overview

`pipeline_v2.interactive` provides the same interactive UI framework as `pipeline.interactive`, adapted for pipeline_v2's simplified architecture:

- **Single active state**: Only one set of outputs per parameter, not multiple runs
- **Automatic invalidation**: Re-running a step automatically deletes downstream outputs
- **Simpler deletion model**: No NaN marking, just deletion

## Key Adaptations from v1

### 1. Core Framework (`core.py`)

**Changes from v1:**
- Import `AnalysisRunner` from `pipeline_v2.analysis` instead of `pipeline.analysis`
- Removed `get_most_recent_run` dependency (v1 run-tracking specific)
- Adapted `StepPanel._write_nan_outputs()` to delete parameters instead of writing NaNs:
  - v1: Called `DS._store_param(name, nan_val, run_idx, ...)` to write NaN placeholders
  - v2: Calls `DS._delete_param(name)` for each output to delete
- Updated `_nan_outputs()` interface:
  - v1: Returns `dict[name, nan_array]`
  - v2: Returns `list[name]` (names to delete)

### 2. Panel Implementations

**gain.py**
- Updated `_nan_outputs()` to return list of parameter names to delete
- Logic unchanged - "Mark Bad" button still works, just deletes instead of writes NaNs

**fit_iq.py**
- Updated `_nan_outputs()` to return list of parameter names
- Logic unchanged

### 3. Sweep Fitter (`sweep_fitter.py`)

- Docstring updated to reflect v2 architecture
- Imports through relative paths, so automatically uses pipeline_v2.analysis
- `_mark_all_sweeps_bad()` calls `panel._write_nan_outputs()` which now deletes
- `_mark_all_bad()` calls `panel._write_nan_outputs()` which now deletes

## Compatibility Matrix

| Feature | v1 pipeline.interactive | v2 pipeline_v2.interactive |
|---------|------------------------|---------------------------|
| Multiple runs per dataset | ✓ (zarr subgroups) | ✗ (single active state) |
| Mark step as bad | Write NaN array | Delete parameters |
| Downstream invalidation | Manual (mark as bad) | Automatic (on re-run) |
| AnalysisRunner | v1 architecture | v2 architecture |

## Usage Example

```python
from citkid.pipeline_v2.interactive import run_interactive
from citkid.pipeline_v2.analysis import AnalysisRunner
from citkid.pipeline_v2.dataset import DataSet

DS = DataSet("output.zarr", cal_yaml_path="iq")
AR = AnalysisRunner(DS, analysis_yaml_path="iq")

run_interactive(
    AR,
    panels=[
        ("make_fr_spans", "fit_gain"),
        ("fit_iq",),
    ],
    data_idx=0,
    title="IQ Analysis",
)
```

## Breaking Changes from v1

1. **No NaN marking**: Steps don't produce NaN outputs. When marked bad, outputs are deleted.
2. **No runs per dataset**: Each parameter has only one active version (per sweep index)
3. **Automatic downstream deletion**: Re-running a step immediately deletes all downstream outputs (you don't need to manually mark them)

## When to Use v1 vs v2 Interactive

**Use v1 (`pipeline.interactive`)** if you:
- Need to preserve multiple "runs" of analysis results
- Want to compare different parameter choices on the same data
- Are using `pipeline.analysis.AnalysisRunner`

**Use v2 (`pipeline_v2.interactive`)** if you:
- Prefer a simpler, single-state model
- Want automatic downstream invalidation
- Are using `pipeline_v2.analysis.AnalysisRunner`
