"""
05: calibrate a tuning's timestreams.

Choose the streams in ``CALIBRATE`` (below): each entry is a run of
04_stream.py by name (the latest run with that name) or index, a window of
zarr chunks along time (``chunks``: all of it by default, n for the first
n chunks, or (first, stop)), and optionally some tone sets. Loading only a
few chunks keeps loading fast. With ``CALIBRATE`` empty, the script lists
the tuning's runs (name, stage, length, chunks, notes) and stops.

The chosen streams are loaded one after another into one DataSet
(``pipeline.stream_sources``) per decimation stage and window length:
each row is one tone of one run, with ``source_idx`` (which entry and tone
set) and ``write_idx`` (which tones were streamed together, for
common-mode removal). The rows are calibrated with the 'ts' path (gain
fit, IQ circle fit, x calibration): first in the interactive window, then
in batch for the rows not done there. Calibration tones (res_idxs < 0), if
the tuning has any, get their own DataSet with the 'ts_offres' path, in
batch.

Everything goes in ``tune{k}/cal{c}`` (an analysis group: delete it to
start over). Running again with the same ``CALIBRATE`` continues in the
same ``cal{c}``; a different choice, or ``--new``, starts the next one.
No CRS is needed.

Usage:
    python 05_calibrate_streams.py --device deviceA
    python 05_calibrate_streams.py --device deviceA --no-gui

Edit the settings below; the command-line options override them for one
run.
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.pipeline.analysis import AnalysisRunner
from citkid.pipeline.interactive import run_ts_analysis
from citkid.pipeline.stream_sources import open_stream_dataset
from citkid.zarr_util import to_jsonable

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
# Streams to calibrate. Each entry: name (latest run with that name) or
# run (index), chunks (None: all; n: the first n zarr chunks; (first,
# stop)), tone_sets (None: all). Leave empty to list the runs.
CALIBRATE = [
    # dict(name = 'dark_dec6', chunks = 1),
    # dict(name = 'dark_dec0', chunks = 2),
]
F_AVG = 1                   # average this many fine sweep points
INTERACTIVE = True          # review in the window before the batch

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
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--tune', type = int, default = None,
                   help = 'tuning (default: the latest with streams)')
    p.add_argument('--new', action = 'store_true',
                   help = 'start a new cal group')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'calibrate in batch only')
    return p.parse_args(argv)

def store_path(args):
    """
    Return the device store's path.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str): ``--store``, or ``<data dir>/<cooldown>/<device>.zarr``.
    """
    if args.store:
        return os.path.expanduser(args.store)
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{args.device}.zarr')

def find_tuning(root, k = None):
    """
    Return the tuning whose streams to calibrate.

    Parameters:
    root (zarr.Group): the store root.
    k (int or None): tuning index. None: the latest with stream runs.

    Returns:
    tune (zarr.Group): the tuning.

    Raises:
    KeyError: if there is no such tuning.
    """
    if k is not None:
        return root[dl.set_name(k, 'tune')]
    done = [j for j in dl.list_indices(root, 'tune')
            if dl.stream_runs(root[dl.set_name(j, 'tune')])]
    if not done:
        raise KeyError('no tuning has streams yet: run 04_stream.py')
    return root[dl.set_name(done[-1], 'tune')]

def cal_group(tune, selection, new):
    """
    Return the cal group for a choice of streams: the latest one made with
    the same choice, or a new one.

    Parameters:
    tune (zarr.Group): the tuning.
    selection (list of dict): ``CALIBRATE``.
    new (bool): If True, always start a new one.

    Returns:
    cal (zarr.Group): the group.
    """
    selection = to_jsonable(list(selection))
    idxs = dl.list_indices(tune, 'cal')
    if idxs and not new:
        last = tune[dl.set_name(idxs[-1], 'cal')]
        if last.attrs.get('selection') == selection:
            print(f'Continuing {last.path}')
            return last
    cal, _ = dl.new_indexed_group(tune, 'cal', 'analysis',
                                  attrs = dict(selection = selection))
    print(f'Calibrating into {cal.path}')
    return cal

def plan_datasets(tune, selection):
    """
    Resolve the chosen streams and group them into DataSets: one per
    decimation stage and window length.

    Parameters:
    tune (zarr.Group): the tuning.
    selection (list of dict): ``CALIBRATE``.

    Returns:
    groups (dict): name: dict(sources, entries), in order of first use.
    """
    groups = {}
    for i, entry in enumerate(selection):
        sources, run = dl.select_streams(tune, **entry)
        if not sources:
            raise ValueError(f'entry {i} ({entry}) selects no tone sets')
        key = (run['dec_stage'], run['stop'] - run['start'])
        name = f"stage{key[0]}_n{key[1]}"
        g = groups.setdefault(name, dict(sources = [], entries = []))
        g['sources'] += sources
        g['entries'].append(dict(
            entry = i, run = run['run'], name = run['name'],
            start = run['start'], stop = run['stop'],
            sources = [s['group'] for s in sources]))
    return groups

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
    tune = find_tuning(root, args.tune)
    print(dl.describe_stream_runs(tune))
    if not CALIBRATE:
        print('Set CALIBRATE at the top of the script to choose streams.')
        return None
    groups = plan_datasets(tune, CALIBRATE)
    cal = cal_group(tune, CALIBRATE, args.new)
    res = tune.attrs['resonators']
    fres_all, qres_all = f'{res}/fres_all', f'{res}/qres_all'
    datasets = {}
    for name, g in groups.items():
        grp = cal.require_group(name)
        grp.attrs['entries'] = to_jsonable(g['entries'])
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
