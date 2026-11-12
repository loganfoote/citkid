"""
05: calibrate timestreams.

Choose the streams in ``CALIBRATE`` (below): each entry is a timestream
run of 04_stream.py (``ts{n}`` in a tuning's run list, its streams under
the calibration sweep they belong to) by name (the latest run with that
name) or index (``run = n``), a window of zarr chunks along time
(``chunks``: all of it by default, n for the first n chunks, or (first,
stop)), and optionally some tone sets. Loading only a few chunks keeps
loading fast. With ``CALIBRATE`` empty, the script lists the runs (name,
tuning, stage, length, tone sets, chunks, notes) and stops.

The chosen streams are loaded into one DataSet per tone set
(``pipeline.stream_sources``): each row is one tone, and its timestream is
its windows from every chosen run, one after another (``source_bounds``
says where each run's samples are; ``write_idx`` says which tones were
streamed together, for common-mode removal). So the calibration is done
once, on everything chosen, whatever the runs' decimation stages: it
takes no spectra (the 'ts_cal' path: gain fit, IQ circle fit, x
calibration), first in the interactive window, then in batch for the
rows not done there. Calibration tones (res_idxs < 0), if the runs have any,
get their own DataSet with the 'ts_offres' path, in batch.

The calibration is applied to whole runs by the scripts that come after
this one (e.g. photon detection), with
``pipeline.stream_sources.apply_stream_calibration``: every tone's x (or
other outputs) over a run's whole length, chunk by chunk, with the run's
own sample period.

Everything goes in ``ts_cal{c}`` at the store root (an analysis group:
delete it to start over). Running again with the same ``CALIBRATE``
continues in the same ``ts_cal{c}``; a different choice, or ``--new``, starts
the next one.
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
# Timestreams to calibrate. Each entry selects a run by name (the latest
# run with that name) or by run (its index n in ts{n}); chunks, the zarr
# chunks to load (None: all; n: the first n; (first, stop): a range); and
# tone_sets (None: all). Leave the list empty to list the runs.
CALIBRATE = [
    # dict(name = 'dark_dec6', chunks = 1),
    # dict(name = 'dark_dec0', chunks = 2),
]
F_AVG = 1                   # number of fine sweep points averaged together
INTERACTIVE = True          # review in the interactive window before the
                            # batch calibration

# Notes saved with the data this script makes (attribute 'notes'), e.g.
# the filters and attenuators on the DAC and ADC lines. Multi-line is fine.
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
    path (str): ``--store``, or ``<data dir>/<cooldown>/<device>.zarr``.
    """
    if args.store:
        return os.path.expanduser(args.store)
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{args.device}.zarr')

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

def plan_datasets(root, selection):
    """
    Resolve the chosen streams and group them into DataSets: one per tone
    set (and resonator list), with the chosen runs' windows of that set
    one after another, in the order of ``selection``.

    Parameters:
    root (zarr.Group): the store root.
    selection (list of dict): ``CALIBRATE``.

    Returns:
    groups (dict): name: dict(sources, entries, resonators, tone_set), in
        order of first use.
    """
    groups = {}
    for i, entry in enumerate(selection):
        sources, run = dl.select_streams(root, **entry)
        if not sources:
            raise ValueError(f'entry {i} ({entry}) selects no tone sets')
        res = run.get('resonators')
        for src in sources:
            s = int(root[src['group']].attrs['tone_set'])
            name = f'set{s}'
            if any(g['resonators'] != res for g in groups.values()) or \
                    name in groups and groups[name]['resonators'] != res:
                name = f'{name}_{res}'
            g = groups.setdefault(name, dict(sources = [], entries = [],
                                             resonators = res, tone_set = s))
            g['sources'].append(src)
            g['entries'].append(dict(
                entry = i, run = run['group'], name = run.get('name'),
                start = run['start'], stop = run['stop'],
                source = src['group']))
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
    print(dl.describe_stream_runs(root))
    if not CALIBRATE:
        print('Set CALIBRATE at the top of the script to choose streams.')
        return None
    groups = plan_datasets(root, CALIBRATE)
    cal = cal_group(root, CALIBRATE, args.new, args.notes)
    datasets = {}
    for name, g in groups.items():
        res = g['resonators']
        fres_all, qres_all = f'{res}/fres_all', f'{res}/qres_all'
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
