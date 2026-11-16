"""
05: calibrate timestreams (gain, IQ circle, x), first in a window, then
in batch, into ts_cal{c}. With CALIBRATE empty, lists the runs. No CRS
needed.

Usage:
    python 05_calibrate_streams.py --device deviceA
    python 05_calibrate_streams.py --device deviceA --no-gui
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.pipeline.analysis import AnalysisRunner
from citkid.pipeline.interactive import run_ts_analysis
from citkid.pipeline.stream_sources import (
    open_stream_dataset, plan_stream_datasets
)
from citkid.zarr_util import to_jsonable

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'
COOLDOWN = 'cooldown'
DEVICE = 'device'
SUFFIX = ''
CALIBRATE = [
]
F_AVG = 1
INTERACTIVE = True

NOTES = """
"""

################################################################################
################################### Script #####################################
################################################################################
def parse_args(argv = None):
    """
    Parse the command-line options (defaults from the settings above).

    Parameters:
    argv (list of str or None): arguments. None (default): sys.argv.

    Returns:
    args (argparse.Namespace): the options.
    """
    p = argparse.ArgumentParser(description = 'Calibrate timestreams.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--new', action = 'store_true',
                   help = 'start a new ts_cal group')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'calibrate in batch only')
    p.add_argument('--notes', default = NOTES,
                   help = 'notes saved with the data (quote multi-line text)')
    return p.parse_args(argv)

def save_notes(grp, notes):
    """
    Save notes with a group the script made (attribute 'notes'), if any.

    Parameters:
    grp (zarr.Group): the group.
    notes (str): the notes (``--notes`` or ``NOTES``); blank is not saved.
    """
    if notes and notes.strip():
        grp.attrs['notes'] = notes.strip()

def store_path(args):
    """
    Return the device store's path.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str): ``--store``, or
        ``<data dir>/<cooldown>/<device>[_<suffix>].zarr``.
    """
    if args.store:
        return os.path.expanduser(args.store)
    name = args.device + (f'_{args.suffix}' if args.suffix else '')
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{name}.zarr')

def cal_group(root, selection, new, notes = None):
    """
    Return the ts_cal group for a choice of streams: the latest one made with
    the same choice, or a new one.

    Parameters:
    root (zarr.Group): the store root.
    selection (list of dict): ``CALIBRATE``.
    new (bool): If True, always start a new one.
    notes (str or None): notes saved with a new group. Default None.

    Returns:
    cal (zarr.Group): the group.
    """
    selection = to_jsonable(list(selection))
    idxs = dl.list_indices(root, 'ts_cal')
    if idxs and not new:
        last = root[dl.set_name(idxs[-1], 'ts_cal')]
        if last.attrs.get('selection') == selection:
            print(f'Continuing {last.path}')
            return last
    cal, _ = dl.new_indexed_group(root, 'ts_cal', 'analysis',
                                  attrs = dict(selection = selection))
    save_notes(cal, notes)
    print(f'Calibrating into {cal.path}')
    return cal

def main(args):
    """
    Calibrate the chosen streams, or list the runs.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    datasets (dict or None): name: DataSet, or None if only listed.
    """
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    print(dl.describe_stream_runs(root))
    if not CALIBRATE:
        print('Set CALIBRATE at the top of the script to choose streams.')
        return None
    groups = plan_stream_datasets(root, CALIBRATE)
    cal = cal_group(root, CALIBRATE, args.new, args.notes)
    datasets = {}
    for name, g in groups.items():
        res = g['resonators']
        fres_all, qres_all = f'{res}/fres_all', f'{res}/qres_all'
        grp = cal.require_group(name)
        grp.attrs.update(entries = to_jsonable(g['entries']),
                         resonators = res, f_avg = F_AVG)
        DS = open_stream_dataset(grp, g['sources'], fres_all, qres_all,
                                 f_avg = F_AVG)
        datasets[name] = DS
        AR = AnalysisRunner(DS, analysis_yaml_path = 'ts')
        nrows = len(DS.res_idxs[:])
        print(f'{name}: {nrows} rows from {len(g["sources"])} sources')
        if INTERACTIVE and not args.no_gui:
            run_ts_analysis(AR, title = f'Calibrate {cal.path}/{name}')
        AR.execute_remaining(verbose = True)
        has_cal_tones = any(
            (root[s['group']]['res_idxs'][:] < 0).any() for s in g['sources'])
        if has_cal_tones:
            DSc = open_stream_dataset(
                cal.require_group(f'{name}_caltones'), g['sources'],
                fres_all, qres_all, rows = 'calibration', f_avg = F_AVG)
            datasets[f'{name}_caltones'] = DSc
            AnalysisRunner(DSc, analysis_yaml_path = 'ts_offres') \
                .execute_remaining(verbose = True)
    print(f'Done: {cal.path}')
    return datasets

if __name__ == '__main__':
    main(parse_args())
