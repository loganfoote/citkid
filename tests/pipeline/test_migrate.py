"""Tests for citkid.pipeline.migrate."""
from pathlib import Path

import numpy as np
import pytest
import zarr

from citkid.pipeline import dataset as dataset_module
from citkid.pipeline import migrate_step_names
from citkid.pipeline.dataset import DataSet
from citkid.pipeline.migrate import (
    RENAMED_STEPS,
    RENAMED_TEMPLATE_STEPS,
    _METADATA_ATTR,
    rename_tasks_in_yaml,
)

TEMPLATES = Path(dataset_module.__file__).parent / "templates"


def _raw_dir(tmp_path):
    """
    Write a tiny raw zarr store that the custom-steps templates can load.

    Parameters:
    tmp_path (Path): pytest temporary directory.

    Returns:
    raw_dir (Path): directory holding ``example.zarr``.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nrows, npts = 2, 5
    raw = zarr.open_group(str(raw_dir / "example.zarr"), mode="w")
    raw.create_array("f", data=np.ones((nrows, npts)))
    raw.create_array("z", data=np.ones((2, nrows, npts)))
    for name in ("fres", "fres_all", "ares", "qres"):
        raw.create_array(name, data=np.ones(nrows))
    raw.create_array("res_idxs", data=np.arange(nrows))
    raw.create_array("ts_00/dt", data=np.array(1e-3))
    return raw_dir


def _legacy_offres_dataset(tmp_path):
    """
    Make a 'ts_offres' dataset and rewrite it as citkid stored it before the
    step renames: old task names in the stored YAML, an old ``step_name``
    label on a saved parameter, and failures under the old step name.

    Parameters:
    tmp_path (Path): pytest temporary directory.

    Returns:
    zarr_path (str): path to the dataset.
    raw_dir (Path): raw data directory (for ``custom_main_dir_overwrite``).
    old_yaml (str): the stored calibration YAML with old names.
    """
    raw_dir = _raw_dir(tmp_path)
    zarr_path = str(raw_dir / "output.zarr")
    DataSet(zarr_path=zarr_path, cal_yaml_path="ts_offres",
            custom_path=str(TEMPLATES / "custom_steps_ts.py"),
            custom_main_dir_overwrite=str(raw_dir))

    root = zarr.open_group(zarr_path, mode="r+")
    meta = dict(root.attrs[_METADATA_ATTR])
    old_yaml = meta["cal_yaml"]
    for old, new in RENAMED_STEPS.items():
        old_yaml = old_yaml.replace(new, old)
    meta["cal_yaml"] = old_yaml
    root.attrs[_METADATA_ATTR] = meta
    root.create_group("zt_cent").attrs["step_name"] = "rotate_offres_t"
    root.require_group("_failures/rotate_offres_t").attrs["failures"] = {
        "idx1": {"error": "old"}}
    root.require_group("_failures/rotate_t_offres").attrs["failures"] = {
        "idx0": {"error": "new"}}
    return zarr_path, raw_dir, old_yaml


def test_metadata_attr_matches_dataset():
    """
    The migration reads the same attribute DataSet stores its definition in.
    """
    assert _METADATA_ATTR == DataSet._METADATA_ATTR


def test_rename_tasks_in_yaml_keeps_format_and_comments():
    """
    Only ``task:`` values change; comments, spacing and other text stay.
    """
    text = ("CAL_STEPS:\n"
            "  1 :\n"
            "    task: rotate_offres_t   # was rotate_offres_t\n"
            "  2:\n"
            "    task: get_thetat\n"
            "# rotate_offres_f in a comment\n")
    new, renamed = rename_tasks_in_yaml(text, RENAMED_STEPS)
    assert renamed == ["rotate_offres_t"]
    assert new == text.replace("task: rotate_offres_t",
                               "task: rotate_t_offres")


def test_rename_tasks_in_yaml_refuses_unsupported_forms():
    """
    A task written in a form the line-based rename misses is caught by the
    parsed-tree check instead of being left half renamed.
    """
    text = "CAL_STEPS:\n  1: {task: rotate_offres_t}\n"
    with pytest.raises(ValueError, match="nothing was changed"):
        rename_tasks_in_yaml(text, RENAMED_STEPS)


def test_old_dataset_errors_until_migrated(tmp_path):
    """
    An old dataset raises an error pointing to migrate_step_names; a dry run
    changes nothing; the migration fixes every stored use of the old names,
    keeps a backup of the original YAML, and is idempotent.
    """
    zarr_path, raw_dir, old_yaml = _legacy_offres_dataset(tmp_path)

    with pytest.raises(ValueError, match="renamed to 'rotate_t_offres'.*"
                       "migrate_step_names"):
        DataSet(zarr_path=zarr_path, custom_main_dir_overwrite=str(raw_dir))

    planned = migrate_step_names(zarr_path, dry_run=True, verbose=False)
    assert len(planned) == 4  # 2 YAML tasks, 1 label, 1 failure group
    root = zarr.open_group(zarr_path, mode="r")
    assert root.attrs[_METADATA_ATTR]["cal_yaml"] == old_yaml

    assert migrate_step_names(zarr_path, verbose=False) == planned

    DS = DataSet(zarr_path=zarr_path, custom_main_dir_overwrite=str(raw_dir))
    assert "rotate_t_offres" in DS.cal_step_indices
    root = zarr.open_group(zarr_path, mode="r")
    meta = root.attrs[_METADATA_ATTR]
    assert meta["before_step_rename_cal_yaml"] == old_yaml
    assert "rotate_offres_t" not in meta["cal_yaml"]
    assert root["zt_cent"].attrs["step_name"] == "rotate_t_offres"
    assert "rotate_offres_t" not in root["_failures"]
    assert set(root["_failures/rotate_t_offres"].attrs["failures"]) == {
        "idx0", "idx1"}

    assert migrate_step_names(zarr_path, verbose=False) == []


def test_old_yaml_file_with_new_template_loader_names(tmp_path):
    """
    A YAML file that still uses an old template loader name, with custom
    steps that define the new name, says which name to use.
    """
    raw_dir = _raw_dir(tmp_path)
    old_yaml = tmp_path / "cal_old.yaml"
    old_yaml.write_text(
        (TEMPLATES / "cal_iq.yaml").read_text(encoding="utf-8")
        .replace("load_sweep_f", "load_data_f"), encoding="utf-8")
    with pytest.raises(ValueError, match="renamed to 'load_sweep_f' in the "
                       "templates"):
        DataSet(zarr_path=str(raw_dir / "output.zarr"),
                cal_yaml_path=str(old_yaml),
                custom_path=str(TEMPLATES / "custom_steps_iq.py"),
                custom_main_dir_overwrite=str(raw_dir))


def test_old_custom_steps_file_with_new_template_says_what_to_rename(
    tmp_path
):
    """
    A custom-steps file copied from an old template, used with the current
    template YAML, gets an error telling the user to rename the step.
    """
    raw_dir = _raw_dir(tmp_path)
    source = (TEMPLATES / "custom_steps_iq.py").read_text(
        encoding="utf-8")
    for old, new in RENAMED_TEMPLATE_STEPS.items():
        source = source.replace(new, old)
    old_custom = tmp_path / "custom_old.py"
    old_custom.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="define 'load_data_f'.*Rename"):
        DataSet(zarr_path=str(raw_dir / "output.zarr"), cal_yaml_path="iq",
                custom_path=str(old_custom),
                custom_main_dir_overwrite=str(raw_dir))
