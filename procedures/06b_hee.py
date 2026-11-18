"""
06b: lifetimes from high-energy event (HEE) tails for devices without
photons:
the pulses are lined up where each falls through the tail level,
averaged, and the average is fitted; check the level in a window; save
one result per timestream (hee/<name>). No CRS needed.

Usage:
    python 06b_hee.py --device deviceA
    python 06b_hee.py --device deviceA --events triggers
    python 06b_hee.py --device deviceA --continue
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline import hee_finder as cf
from citkid.pipeline import photon_finder as pf
from citkid.pipeline.stream_sources import open_calibration
from citkid.signal.photons import DetectionSettings

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'
COOLDOWN = 'cooldown'
DEVICE = 'device'
SUFFIX = ''
CALIBRATION = None
TIMESTREAMS = [
]
NAME = ''
EVENTS = 'auto'
PHOTONS = None
INTERACTIVE = True
TRIG_NSIGMA = 6.
COINC_WIN = 2
HEE_MIN_MULT = 3
HMIN = 0.
HMAX = 1000.
LEVEL = 10.
FIT_S = 2.e-3
PRE_S = 0.5e-3
POST_S = 5.e-3
SEARCH_S = 0.3e-3
HIGHPASS_HZ = 10.
MEMORY_GB = 12.

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
    p = argparse.ArgumentParser(
        description = 'Fit the lifetimes of HEE tails.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--cal', type = int, default = CALIBRATION,
                   help = 'ts_cal index (default: the latest)')
    p.add_argument('--name', default = NAME,
                   help = 'name of the result of a single timestream')
    p.add_argument('--events', choices = ('auto', 'photons', 'triggers'),
                   default = EVENTS, help = 'where the events come from')
    p.add_argument('--photons', default = PHOTONS,
                   help = 'photon result to take the events from (name)')
    p.add_argument('--continue', dest = 'cont', action = 'store_true',
                   default = False,
                   help = "go on from each timestream's saved result in "
                   'the window')
    p.add_argument('--overwrite', action = 'store_true',
                   default = False,
                   help = "replace a timestream's saved result without "
                   'asking')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'skip the window')
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
    return dl.store_path(args.data_dir, args.cooldown, args.device,
                         args.suffix)

def tail_settings():
    """
    Return the analysis settings from the settings above.

    Returns:
    settings (TailSettings): the settings (the thresholds are given
        separately, by ``TailThresholds``).
    """
    return cf.TailSettings(
        highpass_hz = HIGHPASS_HZ, pre_s = PRE_S, post_s = POST_S,
        search_s = SEARCH_S, trig_nsigma = TRIG_NSIGMA,
        coinc_win = COINC_WIN, min_mult = HEE_MIN_MULT)

def ask_existing(name, grp):
    """
    Ask, in a popup, what to do with a timestream's saved result.

    Parameters:
    name (str): the result's name.
    grp (zarr.Group): the result.

    Returns:
    choice (str or None): 'continue', 'overwrite', or None to skip.
    """
    from citkid.pipeline.interactive.photon_thresholds import \
        ask_existing_result
    return ask_existing_result(name, cf.describe_result(grp))

def existing_choice(name, grp, args, interactive):
    """
    Decide what to do with a timestream's saved result: continue from
    it, overwrite it, or skip the timestream. With the window a popup
    asks; otherwise ``--overwrite`` replaces it and anything else skips
    it.

    Parameters:
    name (str): the result's name.
    grp (zarr.Group): the result.
    args (argparse.Namespace): options from ``parse_args``.
    interactive (bool): If True, the window is in use.

    Returns:
    choice (str or None): 'continue', 'overwrite', or None to skip.
    """
    if args.cont:
        return 'continue'
    if args.overwrite:
        return 'overwrite'
    if interactive:
        return ask_existing(name, grp)
    print(f'hee/{name} exists: skipped (run with --continue to go on '
          'from it, or --overwrite to replace it)')
    return None

def pick_events(root, ts, data, name, settings, args):
    """
    Get a timestream's events, from the photon result or from triggers.

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    name (str): the result's name (the photon result's, unless
        ``--photons``).
    settings (TailSettings): trigger settings.
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    events (dict): 't0', 'mult', 'write_idx'.
    mode (str): 'triggers', or 'photons/<name>'.
    """
    ph_name = args.photons if args.photons else name
    have = pf.PHOTONS_GROUP in root and ph_name in root[pf.PHOTONS_GROUP]
    # A result saved without its HEEs (interrupted) has no events
    if have and 'hee/events' not in root[pf.PHOTONS_GROUP][ph_name]:
        print(f'photons/{ph_name} holds no HEEs (saved '
              'incompletely): run 06a_find_photons.py on it again')
        have = False
    if args.events == 'photons' and not have:
        raise KeyError(f'no photon result photons/{ph_name} with HEEs '
                       'to take the events from: run '
                       "06a_find_photons.py, or use EVENTS = 'triggers'")
    if args.events != 'triggers' and have:
        grp = root[pf.PHOTONS_GROUP][ph_name]
        return cf.events_from_photons(grp), f'photons/{ph_name}'
    return cf.find_events_timestream(root, ts, data, settings), 'triggers'

def review(
    ts, results, thresholds, interactive, root = None, stage = '',
    data = None
):
    """
    Open the HEE tail window on a timestream's results, if asked.

    Parameters:
    ts (dict): the timestream.
    results (dict): ``analyze_timestream`` output.
    thresholds (TailThresholds): the thresholds (the window may change
        them).
    interactive (bool): If True, open the window.
    data (TimestreamData or None): the detection timestreams, for the
        window's timestream plot. Default None.
    root (zarr.Group or None): the store root, for the tone-marks
        registry. Default None.
    stage (str): the result's path, recorded with the marks. Default ''.

    Returns:
    results (dict): the results with the chosen thresholds.
    thresholds (TailThresholds): the thresholds.
    """
    if not interactive:
        return results, thresholds
    from citkid.pipeline.interactive.hee_tails import run_hee_tails

    def refit(rows, th):
        """
        Fit some tones' tails again.

        Parameters:
        rows (list of int): DataSet rows.
        th (TailThresholds): thresholds.

        Returns:
        results (dict): row: result.
        """
        return cf.refit_timestream(ts, results, th, rows)

    def tone_data(row):
        """
        Return one tone's detection timestream.

        Parameters:
        row (int): DataSet row.

        Returns:
        d (np.array or str): the timestream, or the error message.
        """
        return next(iter(data.get([row])))[1]

    thresholds, results = run_hee_tails(
        ts, results, thresholds, refit,
        data = None if data is None else tone_data,
        title = f"HEE tails: {ts['run']} ({ts['name']}), tone set "
        f"{ts['tone_set']}", root = root, stage = stage)
    return results, thresholds

def report(ts, results, events):
    """
    Print a summary of one timestream's tails.

    Parameters:
    ts (dict): the timestream.
    results (dict): ``analyze_timestream`` output.
    events (dict): the events.
    """
    ok = [r for r in ts['rows'] if isinstance(results.get(int(r)), dict)]
    tau = np.array([results[int(r)]['tau'] for r in ok])
    n_used = sum(results[int(r)]['n_used'] for r in ok)
    good = tau[np.isfinite(tau)]
    print(f"{ts['run']} ({ts['name']}), tone set {ts['tone_set']}: "
          f"{len(events['t0'])} events, {len(ok)} of {len(ts['rows'])} "
          f'tones analyzed, {n_used} pulses averaged, {len(good)} tails '
          f'fitted; median lifetime {np.median(good) * 1e6:.3g} us'
          if len(good) else
          f"{ts['run']} ({ts['name']}), tone set {ts['tone_set']}: "
          f"{len(events['t0'])} events, no tails fitted")
    for r in ts['rows']:
        res = results.get(int(r))
        if not isinstance(res, dict):
            print(f"  res {ts['res_idxs'][int(r)]}: failed: {res}")

def main(args):
    """
    Fit the HEE tails of the chosen timestreams, or list them.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    out (dict or None): result name: dict(``results``, ``events``,
        ``thresholds``), or None if only listed.
    """
    root = dl.open_store(store_path(args), mode = 'a')
    dl.check_store(root)
    cals = dl.list_indices(root, 'ts_cal')
    if not cals:
        raise KeyError('the store has no ts_cal group: run '
                       '05_calibrate_streams.py first')
    c = cals[-1] if args.cal is None else int(args.cal)
    cal = root[dl.set_name(c, 'ts_cal')]
    datasets = open_calibration(root, cal)
    timestreams = pf.list_timestreams(root, datasets)
    print(f'{cal.path}:')
    print(pf.describe_timestreams(timestreams))
    if not TIMESTREAMS:
        print(cf.describe_results(root))
        print('Set TIMESTREAMS at the top of the script to choose '
              'timestreams.')
        return None
    chosen = pf.select_timestreams(timestreams, TIMESTREAMS)
    if args.name and len(chosen) != 1:
        raise ValueError('NAME names the result of a single timestream; '
                         f'{len(chosen)} are chosen')
    settings = tail_settings()
    results_grp = cf.results_group(root)
    interactive = INTERACTIVE and not args.no_gui
    out = {}
    for ts in chosen:
        name = args.name.strip() if args.name else \
            pf.result_name(ts, timestreams)
        DS = datasets[ts['dataset']]['DS']
        data = pf.TimestreamData(
            root, ts, DS, DetectionSettings(highpass_hz = HIGHPASS_HZ),
            max_bytes = MEMORY_GB * 1e9)
        thresholds = cf.TailThresholds(hmin = HMIN, hmax = HMAX,
                                       level = LEVEL, fit_s = FIT_S)
        continued = None
        if name in results_grp:
            choice = existing_choice(name, results_grp[name], args,
                                     interactive)
            if choice is None:
                print(f'Skipped hee/{name}')
                del data
                continue
            if choice == 'continue':
                continued = f'hee/{name}'
                thresholds = cf.load_thresholds(results_grp[name])
        elif args.cont:
            raise KeyError(f"{ts['run']} tone set {ts['tone_set']} has no "
                           f'saved result hee/{name} to continue')
        events, mode = pick_events(root, ts, data, name, settings, args)
        print(f"{len(events['t0'])} events from {mode}")
        marked = pf.apply_tone_marks(root, ts, thresholds)
        if marked:
            print(f"{len(marked)} tones marked bad in the store's tone "
                  f'marks are left out: res {marked}')
        results = cf.analyze_timestream(root, ts, data, events, settings,
                                        thresholds)
        results, thresholds = review(ts, results, thresholds, interactive,
                                     root = root, stage = f'hee/{name}',
                                     data = data)
        if name in results_grp:
            print(f'Replacing hee/{name}')
        grp = dl.create_group(results_grp, name, 'analysis',
                              overwrite = True)
        cf.save_timestream(grp, ts, results, events, thresholds, settings,
                           mode)
        grp.attrs.update(dl.to_jsonable(dict(
            calibration = cal.path, continued_from = continued)))
        save_notes(grp, args.notes)
        report(ts, results, events)
        out[name] = dict(results = results, events = events,
                         thresholds = thresholds)
        del data
    print(f"Done: {', '.join(f'hee/{n}' for n in out)}")
    return out

if __name__ == '__main__':
    main(parse_args())
