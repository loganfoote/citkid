"""
06b: lifetimes from high-energy event (HEE) tails for devices without
photons:
the pulses are lined up where each falls through the tail level,
averaged, and the average is fitted; check the level in a window; save
one result per timestream (hee/<name>). No CRS needed.

Usage:
    python 6b_hee.py --device deviceA
    python 6b_hee.py --device deviceA --events triggers
    python 6b_hee.py --device deviceA --continue
"""
import argparse
import copy
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline import hee_finder as cf
from citkid.pipeline import photon_finder as pf
from citkid.signal.photons import DetectionSettings
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX, MEMORY_GB, \
    N_WORKERS, LIGHT_MODE

# Timestreams
CALIBRATION_NAME = None
# index as listed or run name: result name (None: named after the run)
TIMESTREAMS = {
    'dark_dec0': 'dark_dec0'
}
EVENTS = 'auto'
PHOTONS_NAME = None
INTERACTIVE = True
TONES_BEFORE_WINDOW = None
WINDOW_FIRST_ONLY = False

# Triggers
TRIG_NSIGMA = 6.
COINC_WIN = 2
HEE_MIN_MULT = 3

# Tail fits
HMIN = 0.
HMAX = 1000.
LEVEL = 10.
FIT_S = 4.e-3
FIT_START_S = None
PRE_S = 0.5e-3
POST_S = 5.e-3
SEARCH_S = 0.3e-3
HIGHPASS_HZ = 10.

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
    p.add_argument('--cal', default = CALIBRATION_NAME,
                   help = 'calibration to use when a sweep has several, by '
                   "name or index (default: each timestream's own)")
    p.add_argument('--events', choices = ('auto', 'photons', 'triggers'),
                   default = EVENTS, help = 'where the events come from')
    p.add_argument('--photons', default = PHOTONS_NAME,
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
              'incompletely): run 6a_find_photons.py on it again')
        have = False
    if args.events == 'photons' and not have:
        raise KeyError(f'no photon result photons/{ph_name} with HEEs '
                       'to take the events from: run '
                       "6a_find_photons.py, or use EVENTS = 'triggers'")
    if args.events != 'triggers' and have:
        grp = root[pf.PHOTONS_GROUP][ph_name]
        return cf.events_from_photons(grp), f'photons/{ph_name}'
    return cf.find_events_timestream(root, ts, data, settings), 'triggers'

def review(
    ts, results, thresholds, interactive, root = None, stage = '',
    data = None, events = None, settings = None
):
    """
    Open the HEE tail window on a timestream's results, if asked. Tones
    without a result (``TONES_BEFORE_WINDOW``) are fitted in the window
    when selected or when values are applied to all tones; those still
    not run when it closes are fitted with its thresholds.

    Parameters:
    ts (dict): the timestream.
    results (dict): ``analyze_timestream`` output, of all tones or of
        the first ones.
    thresholds (TailThresholds): the thresholds (the window may change
        them).
    interactive (bool): If True, open the window.
    data (TimestreamData or None): the detection timestreams, for the
        window's timestream plot. Default None.
    root (zarr.Group or None): the store root, for the tone-marks
        registry and the tones not run. Default None.
    stage (str): the result's path, recorded with the marks. Default ''.
    events (dict or None): the events, for the tones not run. Default
        None: every tone has a result.
    settings (DetectionSettings or None): the tail settings, for the
        tones not run. Default None.

    Returns:
    results (dict): the results with the chosen thresholds.
    thresholds (TailThresholds): the thresholds.
    """
    def analyze(rows, th, progress):
        """
        Fit tones without a result from scratch.

        Parameters:
        rows (list of int): DataSet rows.
        th (TailThresholds): thresholds.
        progress (bool): If True, show a progress bar.

        Returns:
        results (dict): row: result.
        """
        return cf.analyze_timestream(root, ts, data, events, settings, th,
                                     rows = rows, progress = progress,
                                     n_workers = workers())

    if not interactive:
        return results, thresholds
    from citkid.pipeline.interactive.hee_tails import run_hee_tails
    known = dict(results)

    def refit(rows, th):
        """
        Fit some tones' tails again, from scratch for the tones not run
        yet.

        Parameters:
        rows (list of int): DataSet rows.
        th (TailThresholds): thresholds.

        Returns:
        results (dict): row: result.
        """
        new = [int(r) for r in rows if int(r) not in known]
        out = analyze(new, th, False) if new else {}
        known.update(out)
        old = [int(r) for r in rows if int(r) not in out]
        if old:
            out.update(cf.refit_timestream(ts, known, th, old))
        known.update(out)
        return out

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
    rows = [int(r) for r in ts['rows'] if int(r) not in results]
    if rows:
        print(f'Fitting the {len(rows)} tones not run in the window')
        results.update(analyze(rows, thresholds, True))
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

def workers():
    """
    Return the number of tail-fit processes (``N_WORKERS``).

    Returns:
    n (int): at least 1.
    """
    from citkid.parallel import resolve_workers
    return resolve_workers(N_WORKERS)

def main(args):
    """
    Fit the HEE tails of the chosen timestreams, or list them.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    out (dict or None): result name: dict(``results``, ``events``,
        ``thresholds``), or None if only listed.
    """
    set_theme(LIGHT_MODE)
    root = dl.open_store(store_path(args), mode = 'a')
    dl.check_store(root)
    timestreams, datasets = pf.list_all_timestreams(root)
    if not timestreams:
        raise KeyError('the store has no calibrated timestreams: run '
                       '5_calibrate_streams.py first')
    print(pf.describe_timestreams(timestreams))
    if not TIMESTREAMS:
        print(cf.describe_results(root))
        print('Set TIMESTREAMS at the top of the script to choose '
              'timestreams.')
        return None
    named = pf.resolve_named(timestreams, TIMESTREAMS,
                             calibration = args.cal, root = root)
    settings = tail_settings()
    results_grp = cf.results_group(root)
    interactive = INTERACTIVE and not args.no_gui
    out = {}
    first = None     # (name, thresholds) of the first window
    for name, ts in named:
        DS = datasets[ts['cal']][ts['dataset']]['DS']
        data = pf.TimestreamData(
            root, ts, DS, DetectionSettings(highpass_hz = HIGHPASS_HZ),
            max_bytes = MEMORY_GB * 1e9)
        thresholds = cf.TailThresholds(hmin = HMIN, hmax = HMAX,
                                       level = LEVEL, fit_s = FIT_S,
                                       fit_start_s = FIT_START_S)
        window = interactive
        taken_from = None
        if first is not None and WINDOW_FIRST_ONLY:
            # The window's values (global and per tone), without its
            # marks, and no window
            taken_from, th0 = first
            thresholds = cf.TailThresholds(
                hmin = th0.hmin, hmax = th0.hmax, level = th0.level,
                fit_s = th0.fit_s, fit_start_s = th0.fit_start_s,
                overrides = copy.deepcopy(th0.overrides))
            window = False
            print(f'Thresholds of the window on {taken_from}, no window')
        continued = None
        if name in results_grp:
            choice = existing_choice(name, results_grp[name], args,
                                     window)
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
        marked = pf.apply_tone_marks(root, ts, thresholds, f'hee/{name}')
        if marked:
            print(f"{len(marked)} tones marked bad in the store's tone "
                  f'marks are left out: res {marked}')
        rows = None
        if window and TONES_BEFORE_WINDOW is not None and \
                continued is None:
            rows = [int(r) for r in ts['rows'][:int(TONES_BEFORE_WINDOW)]]
        results = cf.analyze_timestream(root, ts, data, events, settings,
                                        thresholds, rows = rows,
                                        n_workers = workers())
        results, thresholds = review(ts, results, thresholds, window,
                                     root = root, stage = f'hee/{name}',
                                     data = data, events = events,
                                     settings = settings)
        if window and first is None and continued is None:
            first = (f'hee/{name}', thresholds)
        if name in results_grp:
            print(f'Replacing hee/{name}')
        grp = dl.create_group(results_grp, name, 'analysis',
                              overwrite = True)
        cf.save_timestream(grp, ts, results, events, thresholds, settings,
                           mode)
        grp.attrs.update(dl.to_jsonable(dict(
            calibration = ts['cal'], continued_from = continued,
            thresholds_from = taken_from)))
        save_notes(grp, args.notes)
        report(ts, results, events)
        out[name] = dict(results = results, events = events,
                         thresholds = thresholds)
        del data
    print(f"Done: {', '.join(f'hee/{n}' for n in out)}")
    return out

if __name__ == '__main__':
    main(parse_args())
