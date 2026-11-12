"""
06: find photons in calibrated timestreams.

Works on a stream calibration made by 05_calibrate_streams.py
(``ts_cal{c}``, the latest by default, or ``--cal c``). A timestream is
one entry of that calibration (one window of one run, with all its
calibrated tones). With ``TIMESTREAMS`` empty, the script lists them and
stops; otherwise it finds the photons of each chosen timestream in turn.

For the first timestream (and for every one with ``REUSE = 'none'``):

1. Every tone is run through the iterative optimal filter
   (``signal.photons.run_detection``): first-pass peaks above
   ``INIT_NSIGMA`` noise sigma and at most ``INIT_HMAX`` times their
   median height seed the template; then the filter is rebuilt
   ``OF_ITERATIONS`` times from the pulses with heights in [``HMIN``,
   ``HMAX``] (in units of the mean photon).
2. The thresholds are checked in the photon threshold window (height
   histograms with draggable lines), globally or per tone, rerunning the
   tones they change (skipped with ``INTERACTIVE = False`` or
   ``--no-gui``).

The later timestreams reuse the first one's work, as ``REUSE`` says:

- 'settings' (default): the first timestream's thresholds (per-tone ones
  included), with the full detection but no window;
- 'filter': the first timestream's optimal filter of each tone (matched by
  resonator index) and its thresholds, applied without iterating; tones
  without one are detected from scratch. The decimation stage must match;
- 'none': each timestream from scratch, with the window.

Then, per tone write (the tones streamed together), a pulse is coincident
when another tone has a pulse within +-``COINC_WIN`` samples of it (pulses
above ``COINC_NSIGMA`` filter sigma count, so small pulses of cosmic rays
on other tones do too). Each tone's mean photon pulse averages the pulses
in the height window that are not coincident, and its decay time is fit.
Events seen by at least ``COSMIC_MIN_MULT`` tones are saved as cosmic rays,
with each tone's pulse height.

Everything goes in ``ts_photons{p}`` at the store root (an analysis group,
attrs ``source`` = the calibration), one ``timestream{j}`` group per
chosen timestream (``pipeline.photon_finder.save_timestream`` describes
the arrays). Each run of the script makes a new ``ts_photons{p}``.
No CRS is needed.

Usage:
    python 06_find_photons.py --device deviceA
    python 06_find_photons.py --device deviceA --cal 1 --no-gui

Edit the settings below; the command-line options override them for one
run.
"""
import argparse
import dataclasses
import os
from citkid import daq_layout as dl
from citkid.pipeline import photon_finder as pf
from citkid.pipeline.stream_sources import open_calibration
from citkid.signal.photons import DetectionSettings

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
CALIBRATION = None          # ts_cal index (None: the latest)
# Timestreams to search, by index (as listed) or run name. The first one
# sets the thresholds; leave the list empty to list the timestreams.
TIMESTREAMS = [
    # 0,
    # 'dark_dec0',
]
# How the timestreams after the first reuse its work: 'settings' (its
# thresholds, full detection, no window), 'filter' (its optimal filters
# and thresholds, no iterations), or 'none' (from scratch, with the window)
REUSE = 'settings'
INTERACTIVE = True          # check the thresholds in the window
# First pass
INIT_NSIGMA = 6.            # lower threshold, noise sigma of the boxcar
INIT_HMAX = 3.              # upper cut, times the median first-pass height
# Optimal filter
OF_ITERATIONS = 5           # filter iterations
HMIN = 0.5                  # height window, in units of the mean photon
HMAX = 1.6
# Choose HMIN automatically: the lowest value (from HMIN up to
# HMIN_AUTO_MAX) with at most this fraction of expected noise pulses in the
# height window. None: use HMIN as it is
HMIN_FALSE_FRAC = None
HMIN_AUTO_MAX = 0.8
TEMPLATE_PRE_S = 0.5e-3     # template length before the pulse maximum, s
TEMPLATE_POST_S = 2.5e-3    # template length after the pulse maximum, s
HIGHPASS_HZ = 10.           # drift removal before detection, Hz
# Coincidences and cosmic rays
COINC_NSIGMA = 4.           # pulses counted for coincidences, filter sigma
COINC_WIN = 2               # coincidence window, +- samples
COSMIC_MIN_MULT = 3         # fewest tones of a cosmic ray

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
    p = argparse.ArgumentParser(
        description = 'Find photons in calibrated timestreams.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--cal', type = int, default = CALIBRATION,
                   help = 'ts_cal index (default: the latest)')
    p.add_argument('--reuse', choices = pf.REUSE_OPTIONS, default = REUSE,
                   help = 'how later timestreams reuse the first one')
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
    path (str): ``--store``, or ``<data dir>/<cooldown>/<device>.zarr``.
    """
    if args.store:
        return os.path.expanduser(args.store)
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{args.device}.zarr')

def detection_settings():
    """
    Return the detection settings from the settings above.

    Returns:
    settings (DetectionSettings): the settings (the thresholds are given
        separately, by ``Thresholds``).
    """
    return DetectionSettings(
        template_pre_s = TEMPLATE_PRE_S, template_post_s = TEMPLATE_POST_S,
        highpass_hz = HIGHPASS_HZ, init_nsigma = INIT_NSIGMA,
        init_hmax = INIT_HMAX, of_iterations = OF_ITERATIONS, hmin = HMIN,
        hmax = HMAX, hmin_false_frac = HMIN_FALSE_FRAC,
        hmin_auto_max = HMIN_AUTO_MAX, coinc_nsigma = COINC_NSIGMA)

def find_photons(ts, DS, settings, thresholds, interactive, reference = None):
    """
    Detect one timestream's photons, with the window if asked.

    Parameters:
    ts (dict): the timestream.
    DS (DataSet): its DataSet.
    settings (DetectionSettings): detection settings.
    thresholds (Thresholds): thresholds (the window may change them).
    interactive (bool): If True, open the threshold window.
    reference (dict or None): res_idx: result whose filter to apply
        (``detect_timestream``). Default None.

    Returns:
    results (dict): row: result or error message.
    thresholds (Thresholds): the thresholds used.
    """
    results = pf.detect_timestream(ts, DS, settings, thresholds,
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
            return pf.detect_timestream(ts, DS, settings, th, rows = rows,
                                        progress = False)

        thresholds, results = run_photon_thresholds(
            ts, results, thresholds, rerun,
            title = f"Photon thresholds: {ts['run']} ({ts['name']})")
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
    n_cr = len(summary['cosmic']['events']['t0'])
    print(f"{ts['run']} ({ts['name']}): {len(ok)} of {len(ts['rows'])} "
          f'tones detected, {n_used} photons averaged, {n_cr} cosmic rays '
          f'({n_cr / duration:.3g}/s)')
    for r in ts['rows']:
        res = results.get(int(r))
        if not isinstance(res, dict):
            print(f'  row {int(r)}: failed: {res}')

def main(args):
    """
    Find the photons of the chosen timestreams, or list them.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    out (dict or None): timestream group name: dict(``results``,
        ``summary``, ``thresholds``), or None if only listed.
    """
    root = dl.open_store(store_path(args), mode = 'a')
    dl.check_store(root)
    cals = dl.list_indices(root, 'ts_cal')
    if not cals:
        raise KeyError('the store has no ts_cal group: run '
                       '05_calibrate_streams.py first')
    c = cals[-1] if args.cal is None else int(args.cal)
    cal_name = dl.set_name(c, 'ts_cal')
    datasets = open_calibration(root, cal_name)
    timestreams = pf.list_timestreams(datasets)
    print(f'{cal_name}:')
    print(pf.describe_timestreams(timestreams))
    if not TIMESTREAMS:
        print('Set TIMESTREAMS at the top of the script to choose '
              'timestreams.')
        return None
    chosen = pf.select_timestreams(timestreams, TIMESTREAMS)
    if args.reuse == 'filter' and len({t['dt'] for t in chosen}) > 1:
        raise ValueError("REUSE = 'filter' needs timestreams of one "
                         'decimation stage')
    settings = detection_settings()
    out_grp, p = dl.new_indexed_group(root, 'ts_photons', 'analysis',
                                      attrs = dict(
                                          source = cal_name,
                                          timestreams = list(TIMESTREAMS),
                                          reuse = args.reuse))
    save_notes(out_grp, args.notes)
    print(f'Saving to {out_grp.path}')
    interactive = INTERACTIVE and not args.no_gui
    out = {}
    first = None
    for j, ts in enumerate(chosen):
        DS = datasets[ts['dataset']]['DS']
        if first is None or args.reuse == 'none':
            thresholds = pf.Thresholds(hmin = HMIN, hmax = HMAX,
                                       init_hmax = INIT_HMAX)
            results, thresholds = find_photons(ts, DS, settings, thresholds,
                                               interactive)
            mode = 'full'
        else:
            thresholds = dataclasses.replace(
                first['thresholds'],
                overrides = dict(first['thresholds'].overrides))
            reference = None
            if args.reuse == 'filter':
                reference = {res['res_idx']: res
                             for res in first['results'].values()
                             if isinstance(res, dict)}
            results, _ = find_photons(ts, DS, settings, thresholds, False,
                                      reference = reference)
            mode = f"{args.reuse} from {first['name']}"
        summary = pf.finish_timestream(ts, DS, results, settings,
                                       coinc_win = COINC_WIN,
                                       cosmic_min_mult = COSMIC_MIN_MULT)
        name = f'timestream{j}'
        pf.save_timestream(dl.create_group(out_grp, name, 'analysis'), ts,
                           results, summary, thresholds, settings, mode,
                           COINC_WIN, COSMIC_MIN_MULT)
        report(ts, results, summary)
        out[name] = dict(results = results, summary = summary,
                         thresholds = thresholds)
        if first is None:
            first = dict(out[name], name = name)
    print(f'Done: {out_grp.path}')
    return out

if __name__ == '__main__':
    main(parse_args())
