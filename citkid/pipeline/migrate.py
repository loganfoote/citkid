"""
Update datasets made with older versions of citkid so they open with the
current one.

Run ``migrate_step_names`` once per old dataset, while nothing else (a
notebook, a series window) has the dataset open.
"""
import re

import yaml
import zarr

# Built-in steps citkid has renamed (old name -> new name). Only pure renames
# belong here: the new step must compute exactly what the old one did, or
# migrating would change results for old data.
RENAMED_STEPS = {
    "rotate_offres_t": "rotate_t_offres",
    "rotate_offres_f": "rotate_f_offres",
}

# Loader steps renamed in the custom-steps templates (old name -> new name).
# Each dataset stores its own custom-steps code, which still defines the old
# names, so datasets don't need migrating. The table is only used to explain
# errors for custom-steps files copied from an old template.
RENAMED_TEMPLATE_STEPS = {
    "load_data_f": "load_sweep_f",
    "load_data_g": "load_sweep_g",
}

# Must match DataSet._METADATA_ATTR (checked by the tests)
_METADATA_ATTR = "pipeline"
_YAML_KEYS = ("cal_yaml", "analysis_yaml")
_BACKUP_PREFIX = "before_step_rename_"


def rename_tasks_in_yaml(yaml_text, renames):
    """
    Rename ``task:`` values in pipeline YAML text, keeping its formatting and
    comments.

    Parameters:
    yaml_text (str): Pipeline YAML text.
    renames (dict): Old task name -> new task name.

    Returns:
    new_text (str): YAML text with the tasks renamed.
    renamed (list of str): Old names that were found, in order.

    Raises:
    ValueError: If the text and the renamed text don't parse to the same
        tree apart from the renamed tasks (e.g. a task written in a form the
        line-based rename doesn't handle), so nothing would be changed
        safely.
    """
    if not renames:
        return yaml_text, []
    names = '|'.join(re.escape(name) for name in renames)
    pattern = re.compile(
        r'^(\s*task\s*:\s*)(' + names + r')(\s*(?:#.*)?)$', re.MULTILINE)
    renamed = []

    def replace(match):
        """
        Return the matched line with its task renamed.
        """
        renamed.append(match.group(2))
        return match.group(1) + renames[match.group(2)] + match.group(3)

    new_text = pattern.sub(replace, yaml_text)
    expected = _rename_tasks_in_tree(yaml.safe_load(yaml_text), renames)
    if yaml.safe_load(new_text) != expected:
        raise ValueError(
            "Renaming the tasks in this YAML changed more than the task "
            "names, or missed some; nothing was changed.")
    return new_text, renamed


def _rename_tasks_in_tree(node, renames):
    """
    Return a copy of a parsed YAML tree with ``task`` values renamed.

    Parameters:
    node (dict, list, or other): Parsed YAML node.
    renames (dict): Old task name -> new task name.

    Returns:
    node (dict, list, or other): Copy with renamed tasks.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == 'task' and isinstance(value, str):
                out[key] = renames.get(value, value)
            else:
                out[key] = _rename_tasks_in_tree(value, renames)
        return out
    if isinstance(node, list):
        return [_rename_tasks_in_tree(value, renames) for value in node]
    return node


def migrate_step_names(zarr_path, dry_run = False, verbose = True):
    """
    Rename built-in steps in a dataset made before citkid renamed them
    (``RENAMED_STEPS``), so it opens with the current citkid.

    Updates the stored calibration and analysis YAML, the ``step_name``
    label of saved parameters, and the ``_failures`` records. The original
    YAML text is kept in the dataset metadata under
    ``'before_step_rename_cal_yaml'`` (and ``..._analysis_yaml``). The
    metadata is written last, so if the migration stops partway, the dataset
    still reports the old names and the migration can be run again. Running
    it on a dataset that is already up to date changes nothing. Datasets
    migrated this way can't be opened by citkid versions from before the
    rename.

    Parameters:
    zarr_path (str or zarr.Group): Path to the dataset's zarr store, or the
        open group.
    dry_run (bool): If True, only report the changes. Default is False.
    verbose (bool): If True (default), print the changes.

    Returns:
    changes (list of str): Description of each change (made, or planned if
        ``dry_run``). Empty if the dataset is already up to date.

    Raises:
    ValueError: If the stored YAML can't be renamed safely (nothing is
        written in that case).
    """
    if isinstance(zarr_path, zarr.Group):
        root = zarr_path
    else:
        root = zarr.open_group(str(zarr_path),
                               mode = 'r' if dry_run else 'r+')

    # Plan every change before writing anything
    metadata = dict(root.attrs.get(_METADATA_ATTR, {}))
    new_metadata = dict(metadata)
    changes = []
    for key in _YAML_KEYS:
        text = metadata.get(key)
        if text is None:
            continue
        new_text, renamed = rename_tasks_in_yaml(text, RENAMED_STEPS)
        if renamed:
            new_metadata[key] = new_text
            new_metadata.setdefault(_BACKUP_PREFIX + key, text)
            changes += [f"{key}: task '{old}' -> '{RENAMED_STEPS[old]}'"
                        for old in renamed]

    labeled = []
    for name, group in root.groups():
        if name.startswith('_'):
            continue
        step_name = group.attrs.get('step_name')
        if step_name in RENAMED_STEPS:
            labeled.append((group, step_name))
            changes.append(f"parameter '{name}': step_name '{step_name}' -> "
                           f"'{RENAMED_STEPS[step_name]}'")

    failures = root['_failures'] if '_failures' in root else None
    failed_steps = []
    if failures is not None:
        for old in RENAMED_STEPS:
            if old in failures:
                failed_steps.append(old)
                changes.append(f"_failures/{old} -> "
                               f"_failures/{RENAMED_STEPS[old]}")

    if verbose:
        if not changes:
            print('Already up to date; nothing to change.')
        for change in changes:
            print(('Would change ' if dry_run else 'Changed ') + change)
    if dry_run or not changes:
        return changes

    # Labels and failure records first, metadata last
    for group, step_name in labeled:
        group.attrs['step_name'] = RENAMED_STEPS[step_name]
    for old in failed_steps:
        old_group = failures[old]
        new_group = failures.require_group(RENAMED_STEPS[old])
        merged = dict(old_group.attrs.get('failures', {}))
        merged.update(new_group.attrs.get('failures', {}))
        new_group.attrs['failures'] = merged
        del failures[old]
    if new_metadata != metadata:
        root.attrs[_METADATA_ATTR] = new_metadata
    return changes
