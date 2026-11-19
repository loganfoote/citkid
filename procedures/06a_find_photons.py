"""
06a: find photons and high-energy events (HEEs) in calibrated
timestreams with an iterative optimal filter, check the thresholds in
a window, and save one result per timestream (photons/<name>). No CRS
needed.

Usage:
    python 06a_find_photons.py --device deviceA
    python 06a_find_photons.py --device deviceA --continue
    python 06a_find_photons.py --device deviceA --reference dark --reuse filter
    python 06a_find_photons.py --device deviceA --refit-tau
"""
import argparse
import dataclasses
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline import photon_finder as pf
from citkid.pipeline.stream_sources import open_calibration
from citkid.signal.photons import DetectionSettings

################################################################################
################################### Settings ###################################
################################################################################
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX, MEMORY_GB

# Timestreams
CALIBRATION = None
TIMESTREAMS = [
]
NAME = ''
REFERENCE = None
REUSE = 'settings'
INTERACTIVE = True

# Pulse finding
INIT_NSIGMA = 6.
INIT_HMAX = 3.
OF_ITERATIONS = 5
HMIN = 1.
HMAX = 3.2
HMIN_FALSE_FRAC = None
HMIN_AUTO_MAX = 1.6
TEMPLATE_PRE_S = 0.5e-3
TEMPLATE_POST_S = 2.5e-3
FILTER_POST_S = 1.0e-3
TAU_SKIP = 2
TAU_FIT_S = 1.e-3
HIGHPASS_HZ = 10.

# Coincidences and HEEs
COINC_NSIGMA = 4.
COINC_WIN = 2
HEE_MIN_MULT = 3

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
        description = 'Find photons in timestreams.')
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
    p.add_argument('--reference', default = REFERENCE,
                   help = 'saved result to reuse (its name)')
    p.add_argument('--reuse', choices = pf.REUSE_OPTIONS, default = REUSE,
                   help = 'what to take from the reference')
    p.add_argument('--continue', dest = 'cont', action = 'store_true',
                   default = False,
                   help = "go on from each timestream's saved result in "
                   'the window')
    p.add_argument('--overwrite', action = 'store_true',
                   default = False,
                   help = "replace a timestream's saved result without "
                   'asking')
    p.add_argument('--refit-tau', action = 'store_true',
                   help = "only fit the decay times of each timestream's "
                   'saved result again with TAU_SKIP and TAU_FIT_S')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'skip the threshold window')
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

def detection_settings():
    """
    Return the detection settings from the settings above.

    Returns:
    settings (DetectionSettings): the settings (the thresholds are given
        separately, by ``Thresholds``).
    """
    return DetectionSettings(
        template_pre_s = TEMPLATE_PRE_S, template_post_s = TEMPLATE_POST_S,
        filter_post_s = FILTER_POST_S,
        highpass_hz = HIGHPASS_HZ, init_nsigma = INIT_NSIGMA,
        init_hmax = INIT_HMAX, of_iterations = OF_ITERATIONS, hmin = HMIN,
        hmax = HMAX, hmin_false_frac = HMIN_FALSE_FRAC,
        hmin_auto_max = HMIN_AUTO_MAX, coinc_nsigma = COINC_NSIGMA,
        tau_skip = TAU_SKIP, tau_fit_s = TAU_FIT_S)

def find_photons(
    ts, data, settings, thresholds, interactive, reference = None,
    results = None, root = None, stage = ''
):
    """
    Detect one timestream's photons, with the window if asked.

    Parameters:
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    settings (DetectionSettings): detection settings.
    thresholds (Thresholds): thresholds (the window may change them).
    interactive (bool): If True, open the threshold window.
    reference (dict or None): res_idx: result whose filter to apply
        (``detect_timestream``). Default None.
    results (dict or None): results to start from (a previous result's,
        ``photon_finder.load_results``) instead of detecting first.
        Default None.

    Returns:
    results (dict): row: result or error message.
    thresholds (Thresholds): the thresholds used.
    """
    if results is None:
        results = pf.detect_timestream(ts, data, settings, thresholds,
                                       reference = reference)
    if interactive:
        from citkid.pipeline.interactive.photon_thresholds import \
            run_photon_thresholds

        def rerun(rows, th):
            """
            Run the detection again on some tones.

            Parameters:
            rows (list of int): DataSet rows.
            th (Thresholds): thresholds.

            Returns:
            results (dict): row: result or error message.
            """
            return pf.detect_timestream(ts, data, settings, th,
                                        rows = rows, progress = False)

        def tone_data(row):
            """
            Return one tone's detection timestream for the data snippet.

            Parameters:
            row (int): DataSet row.

            Returns:
            d (np.array or str): the timestream, or the error message.
            """
            return next(iter(data.get([row])))[1]

        thresholds, results = run_photon_thresholds(
            ts, results, thresholds, rerun, data = tone_data,
            settings = settings,
            title = f"Photon thresholds: {ts['run']} ({ts['name']}), "
            f"tone set {ts['tone_set']}", root = root, stage = stage)
    return results, thresholds

def report(ts, results, summary):
    """
    Print a summary of one timestream's photons.

    Parameters:
    ts (dict): the timestream.
    results (dict): detection results.
    summary (dict): ``finish_timestream`` output.
    """
    ok = [r for r in ts['rows'] if isinstance(results.get(int(r)), dict)]
    n_used = sum(len(summary['used'].get(int(r), [])) for r in ok)
    duration = (ts['stop'] - ts['start']) * ts['dt']
    n_cr = len(summary['hee']['events']['t0'])
    left_out = ts.get('n_uncalibrated', 0)
    print(f"{ts['run']} ({ts['name']}), tone set {ts['tone_set']}: "
          f"{len(ok)} of {len(ts['rows'])} tones detected"
          + (f' ({left_out} uncalibrated left out)' if left_out else '')
          + f', {n_used} photons averaged, {n_cr} HEEs '
          f'({n_cr / duration:.3g}/s)')
    for k, r in enumerate(ts['rows']):
        res = results.get(int(r))
        if not isinstance(res, dict):
            print(f"  res {ts['res_idxs'][k]}: failed: {res}")

def reference_for(ref_grp, settings):
    """
    Read a reference result's thresholds and filters.

    Parameters:
    ref_grp (zarr.Group): the reference (``photons/<name>``).
    settings (DetectionSettings): detection settings.

    Returns:
    thresholds (Thresholds): its thresholds.
    filters (dict): res_idx: its filter (``photon_finder.load_reference``).
    """
    return pf.load_reference(ref_grp, settings)

def refit_taus(root, chosen, timestreams):
    """
    Fit the decay times of the chosen timestreams' saved results again,
    with the script's ``TAU_SKIP`` and ``TAU_FIT_S``.

    Parameters:
    root (zarr.Group): the store root, writable.
    chosen (list of dict): the timestreams.
    timestreams (list of dict): every timestream (for the names).

    Returns:
    taus (dict): result name: the new decay times (np.array).

    Raises:
    KeyError: if a timestream has no saved result.
    """
    out = {}
    for ts in chosen:
        name = pf.result_name(ts, timestreams)
        if name not in pf.list_results(root):
            raise KeyError(f'{ts["run"]} tone set {ts["tone_set"]} has no '
                           f'saved result photons/{name}')
        grp = root[pf.PHOTONS_GROUP][name]
        tau = pf.refit_tau(grp, TAU_SKIP, TAU_FIT_S)
        out[name] = tau
        ok = np.isfinite(tau)
        print(f'photons/{name}: {ok.sum()} of {len(tau)} decay times fit '
              f'(median {np.nanmedian(tau) * 1e6 if ok.any() else np.nan:.3g}'
              f' us; skip {TAU_SKIP}, {TAU_FIT_S} s)')
    return out

def continue_from(grp, ts, data, settings):
    """
    Load a timestream's saved result to go on from: its results and
    thresholds, or, for a result without detection detail (an earlier
    version), its thresholds with a fresh detection.

    Parameters:
    grp (zarr.Group): the saved result.
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    settings (DetectionSettings): detection settings.

    Returns:
    results (dict): row: result or error message.
    thresholds (Thresholds): its thresholds.
    """
    try:
        return pf.load_results(grp, settings)
    except KeyError:
        print(f'{grp.path} holds no detection detail: detecting again with '
              'its thresholds')
        thresholds = pf.load_reference(grp, settings)[0]
        return pf.detect_timestream(ts, data, settings, thresholds), \
            thresholds

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
    return ask_existing_result(name, pf.describe_result(grp))

def existing_choice(name, grp, args, interactive):
    """
    Decide what to do with a timestream's saved result: continue from it,
    overwrite it, or skip the timestream. With the window (and no
    reference) a popup asks; otherwise ``--overwrite`` replaces it and
    anything else skips it.

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
    if interactive and args.reference is None:
        return ask_existing(name, grp)
    print(f'photons/{name} exists: skipped (run with --continue to go on '
          'from it, or --overwrite to replace it)')
    return None

def main(args):
    """
    Find the photons of the chosen timestreams, or list them.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    out (dict or None): result name: dict(``results``, ``summary``,
        ``thresholds``), or None if only listed; with ``--refit-tau``,
        the new decay times per result.
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
        print(pf.describe_results(root))
        print('Set TIMESTREAMS at the top of the script to choose '
              'timestreams.')
        return None
    chosen = pf.select_timestreams(timestreams, TIMESTREAMS)
    if args.refit_tau:
        return refit_taus(root, chosen, timestreams)
    if args.name and len(chosen) != 1:
        raise ValueError('NAME names the result of a single timestream; '
                         f'{len(chosen)} are chosen')
    settings = detection_settings()
    ref_grp = None
    if args.reference is not None and not args.cont:
        ref_grp = root[pf.PHOTONS_GROUP][str(args.reference)]
    results_grp = pf.results_group(root)
    interactive = INTERACTIVE and not args.no_gui
    out = {}
    for ts in chosen:
        name = args.name.strip() if args.name else \
            pf.result_name(ts, timestreams)
        DS = datasets[ts['dataset']]['DS']
        data = pf.TimestreamData(root, ts, DS, settings,
                                 max_bytes = MEMORY_GB * 1e9)
        continued = None
        cont = args.cont
        if name in results_grp:
            choice = existing_choice(name, results_grp[name], args,
                                     interactive)
            if choice is None:
                print(f'Skipped photons/{name}')
                del data
                continue
            cont = choice == 'continue'
        elif cont:
            raise KeyError(f"{ts['run']} tone set {ts['tone_set']} has "
                           f'no saved result photons/{name} to continue')
        if cont:
            continued = results_grp[name]
            results, thresholds = continue_from(continued, ts, data,
                                                settings)
            marked = pf.apply_tone_marks(root, ts, thresholds,
                                        f'photons/{name}')
            results, thresholds = find_photons(ts, data, settings,
                                               thresholds, interactive,
                                               results = results,
                                               root = root,
                                               stage = f'photons/{name}')
            mode = f'continued from photons/{name}'
        elif ref_grp is None:
            thresholds = pf.Thresholds(hmin = HMIN, hmax = HMAX,
                                       init_hmax = INIT_HMAX,
                                       tau_skip = TAU_SKIP,
                                       tau_fit_s = TAU_FIT_S)
            marked = pf.apply_tone_marks(root, ts, thresholds,
                                        f'photons/{name}')
            results, thresholds = find_photons(ts, data, settings,
                                               thresholds, interactive,
                                               root = root,
                                               stage = f'photons/{name}')
            mode = 'full'
        else:
            thresholds, filters = reference_for(ref_grp, settings)
            reference = None
            if args.reuse == 'filter':
                reference = filters
                if any(abs(f['dt'] - ts['dt']) > 1e-12 * ts['dt']
                       for f in filters.values()):
                    raise ValueError(
                        f"the reference's filters are for another sample "
                        f"period than {ts['run']}'s: use REUSE = 'settings'")
            marked = pf.apply_tone_marks(root, ts, thresholds,
                                        f'photons/{name}')
            results, _ = find_photons(ts, data, settings, thresholds, False,
                                      reference = reference)
            mode = f'{args.reuse} from {ref_grp.path}'
        if marked:
            print(f"{len(marked)} tones marked bad in the store's tone "
                  f'marks are left out: res {marked}')
        summary = pf.finish_timestream(root, ts, data, results, settings,
                                       coinc_win = COINC_WIN,
                                       hee_min_mult = HEE_MIN_MULT,
                                       thresholds = thresholds)
        if name in results_grp:
            print(f'Replacing photons/{name}')
        grp = dl.create_group(results_grp, name, 'analysis',
                              overwrite = True)
        pf.save_timestream(grp, ts, results, summary, thresholds, settings,
                           mode, COINC_WIN, HEE_MIN_MULT)
        grp.attrs.update(dl.to_jsonable(dict(
            calibration = cal.path,
            reference = None if ref_grp is None else ref_grp.path,
            reuse = None if ref_grp is None else args.reuse,
            continued_from = None if continued is None else mode[15:])))
        save_notes(grp, args.notes)
        report(ts, results, summary)
        out[name] = dict(results = results, summary = summary,
                         thresholds = thresholds)
        del data
    print(f"Done: {', '.join(f'photons/{n}' for n in out)}")
    return out

if __name__ == '__main__':
    main(parse_args())
