"""
6d: the NEP of every tone from a blackbody series (4c_ts_series.py):
calibrate each point's timestream and fit its resonances, compute the
spectra with peak removal and common-mode removal, fit the responsivity
and the photon-noise NEP, review in a window, and save nep/<NAME>.
Replaces 5 and 6c for a sweep. No CRS needed.

Usage:
    python 6d_nep.py --device deviceA
    python 6d_nep.py --device deviceA --no-gui
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.nep import sweep as sw
from citkid.pipeline import psds
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX, MEMORY_GB, \
    N_WORKERS, LIGHT_MODE

# Sweep
NAME = 'bb_sweep'
POINTS = {
}
TUNING_NAME = None
CALIBRATION_NAME = None
NU = 150e9
FREQ_HZ = 10.
F_AVG = 1
INTERACTIVE = True
BAD_TONES = []

# Spectra
NBINS = 100
FMIN_MULT = 10.
FMAX_DIV = 10.
CN_REMOVE = True
CN_N_COMP = 3
CN_N_ITER = 5
CN_LOWPASS_HZ = None
CN_HIGHPASS_HZ = None
CN_FILTER_ORDER = 2
MODES_RATE_HZ = 400.

# Peak removal
DEGLITCH = 'peaks'
GLITCH_PRE_S = 0.2e-3
GLITCH_POST_S = 2.e-3
GLITCH_SIGN = -1
GLITCH_NSTD = 12.
GLITCH_BASELINE_S = 1.
GLITCH_DISTANCE_S = 1.e-3
GLITCH_ROUNDS = 2

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
    p = argparse.ArgumentParser(description = 'NEP from a power sweep.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--name', default = NAME, help = 'result name')
    p.add_argument('--tuning', default = TUNING_NAME,
                   help = 'tuning whose runs are the points, by name or '
                   'index (default: every tuning)')
    p.add_argument('--cal', default = CALIBRATION_NAME,
                   help = 'stream calibration to continue, by name or index '
                   '(default: one named after the result)')
    p.add_argument('--overwrite', action = 'store_true',
                   help = 'compute a saved result again instead of '
                   'resuming its review')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'skip the review window')
    p.add_argument('--notes', default = NOTES,
                   help = 'notes saved with the result (quote multi-line '
                   'text)')
    return p.parse_args(argv)

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

def spectra_settings():
    """
    Return the spectra settings from the settings above.

    Returns:
    settings (dict): for ``pipeline.psds.analyze_timestream``, with
        ``modes_rate_hz`` for the review.
    """
    return dict(
        nbins = NBINS, fmin_mult = FMIN_MULT, fmax_div = FMAX_DIV,
        cn_remove = CN_REMOVE, cn_n_comp = CN_N_COMP, cn_n_iter = CN_N_ITER,
        cn_lowpass_hz = CN_LOWPASS_HZ, cn_highpass_hz = CN_HIGHPASS_HZ,
        cn_filter_order = CN_FILTER_ORDER, deglitch = DEGLITCH,
        glitch_pre_s = GLITCH_PRE_S, glitch_post_s = GLITCH_POST_S,
        glitch_sign = GLITCH_SIGN, glitch_nstd = GLITCH_NSTD,
        glitch_baseline_s = GLITCH_BASELINE_S,
        glitch_distance_s = GLITCH_DISTANCE_S, glitch_rounds = GLITCH_ROUNDS,
        glitch_pulses = 'hee', coinc_win = 2, hee_min_mult = 3,
        memory_gb = MEMORY_GB, modes_rate_hz = MODES_RATE_HZ,
        n_workers = N_WORKERS)

def calibration_group(root, points, args):
    """
    Return the stream calibration of the sweep: ``--cal`` (CALIBRATION_NAME),
    or one named after the result (continued when it exists).

    Parameters:
    root (zarr.Group): the store root.
    points (list of dict): the points.
    args (argparse.Namespace): the options.

    Returns:
    cal (zarr.Group): the group.

    Raises:
    KeyError: if ``--cal`` names no calibration.
    """
    selection = [dict(run = p['run'], tone_sets = [p['tone_set']])
                 for p in points]
    which = args.cal if args.cal is not None else args.name
    cal = dl.find_calibration(root, which)
    if cal is None:
        if args.cal is not None:
            raise KeyError(f'no calibration {args.cal!r}')
        cal, _ = dl.new_indexed_group(root, 'ts_cal', 'analysis', attrs = dict(
            selection = dl.to_jsonable(selection), name = args.name))
        print(f'Calibrating into {cal.path} ({args.name!r})')
    else:
        cal.attrs['selection'] = dl.to_jsonable(selection)
        print(f'Continuing {cal.path} ({dl.calibration_name(cal)!r})')
    return cal

def excluded_tones(root, name, points):
    """
    Return the tones left out: the tone marks of this result's stage and
    the global marks of the points' tunings, and ``BAD_TONES``.

    Parameters:
    root (zarr.Group): the store root.
    name (str): the result name.
    points (list of dict): the points.

    Returns:
    exclude (dict): res_idx: reason.
    """
    out = {}
    for tuning in sorted({p['tuning'] for p in points}):
        for r, e in dl.bad_tones(root, f'{sw.NEP_GROUP}/{name}',
                                 tuning = tuning).items():
            out[int(r)] = dl.bad_reason(e)
    out.update({int(r): 'BAD_TONES' for r in BAD_TONES})
    return out

def compute(root, points, args):
    """
    Calibrate the points, compute their spectra, and fit every tone
    (the batch, before the review).

    Parameters:
    root (zarr.Group): the store root, writable.
    points (list of dict): the points.
    args (argparse.Namespace): the options.

    Returns:
    sweep (PowerSweep): the sweep, saved to nep/<name>.
    """
    cal = calibration_group(root, points, args)
    datasets = sw.calibrate_points(root, cal, points, f_avg = F_AVG)
    timestreams = sw.point_timestreams(root, points, datasets)
    settings = spectra_settings()
    f_reduced = psds.union_freqs(timestreams, FMIN_MULT, FMAX_DIV)
    sw.reduced_column(f_reduced, FREQ_HZ)
    psd_grp, _ = dl.new_indexed_group(
        root, psds.PREFIX, 'analysis', attrs = dict(
            name = args.name, timestreams = [p['name'] for p in points],
            settings = settings, reference = None))
    dl.write_array(psd_grp, 'f_reduced', f_reduced)
    print(f'Spectra into {psd_grp.path}')
    exclude = excluded_tones(root, args.name, points)
    results = sw.psd_points(root, psd_grp, points, timestreams, datasets,
                            settings, exclude = exclude)
    grp = dl.create_group(sw.results_group(root), args.name, 'analysis',
                          overwrite = True)
    sweep = sw.PowerSweep(root, grp, points, timestreams, datasets, FREQ_HZ,
                          NU, settings)
    sweep.bad = dict(exclude)
    sweep.fill(results, psd_grp.path)
    sweep.save(args.notes)
    return sweep

def review(sweep, args):
    """
    Open the review window, run the final pass on the points whose
    calibrations changed, and save.

    Parameters:
    sweep (PowerSweep): the sweep.
    args (argparse.Namespace): the options.
    """
    from citkid.nep.review import run_nep_review
    exclude = excluded_tones(sweep.root, args.name, sweep.points)
    run_nep_review(sweep, exclude = exclude,
                   title = f"NEP: {args.name} ({len(sweep.points)} points)")
    sweep.save(args.notes)

def report(sweep):
    """
    Print a summary of the sweep.

    Parameters:
    sweep (PowerSweep): the sweep.
    """
    f = sweep.fields
    ok = np.isfinite(f['eta'])
    bad = [int(r) for r in sweep.res_idxs if sweep.is_bad(r)]
    print(f'{sweep.grp.path}: {len(sweep.points)} points, '
          f'{len(sweep.res_idxs)} tones, {int(ok.sum())} with a photon-noise '
          f'fit, {len(bad)} marked bad')
    if ok.any():
        print(f"  median eta {np.nanmedian(f['eta'][ok]):.3g}, NEPinc "
              f"{np.nanmedian(f['nep_inc'][ok]):.3g} W/Hz^0.5, NEPabs "
              f"{np.nanmedian(f['nep_abs'][ok]):.3g} W/Hz^0.5")
    failed = [(int(r), s) for r, s in zip(sweep.res_idxs, sweep.status) if s]
    for r, s in failed[:20]:
        print(f'  res {r}: {s}')
    if len(failed) > 20:
        print(f'  ... {len(failed) - 20} more')

def main(args):
    """
    Compute or resume the sweep, review it, and save it.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    sweep (PowerSweep or None): the sweep, or None if only listed.
    """
    set_theme(LIGHT_MODE)
    root = dl.open_store(store_path(args), mode = 'a')
    dl.check_store(root)
    tuning = None
    if args.tuning is not None:
        tune = dl.find_tuning(root, args.tuning)
        if tune is None:
            raise KeyError(f'no tuning {args.tuning!r}')
        tuning = tune.basename
    print(sw.describe_results(root))
    if not POINTS:
        listed = sw.list_points(root, tuning)
        if not any(p['power_w'] is not None for p in listed):
            names = sorted({p['point'] or p['name'] for p in listed})
            print('Set POINTS: the point names with their incident '
                  'powers in W, e.g. POINTS = {' + ', '.join(
                      f'{n!r}: 1e-15' for n in names[:3]) + ', ...}'
                  if names else 'No run has a point name (4c) or a '
                  'power: set POINTS with run names and powers.')
            return None
    points = sw.select_points(root, POINTS, tuning)
    print(sw.describe_points(points))
    existing = sw.NEP_GROUP in root and args.name in root[sw.NEP_GROUP]
    if existing and not args.overwrite:
        print(f'Resuming nep/{args.name} (--overwrite computes it again)')
        sweep = sw.PowerSweep.load(root, root[sw.NEP_GROUP][args.name])
    else:
        sweep = compute(root, points, args)
    if INTERACTIVE and not args.no_gui:
        review(sweep, args)
    elif sweep.changed:
        sweep.final_pass(excluded_tones(root, args.name, sweep.points))
        sweep.save(args.notes)
    report(sweep)
    return sweep

if __name__ == '__main__':
    main(parse_args())
