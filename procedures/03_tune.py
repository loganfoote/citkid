"""
03: tune a device: plan the tones, sweep the power, choose each tone's
power and frequency in two windows, and save the tuned tones (tune{k}).
TUNING names the tuning: an existing name (or index) continues it,
a new name starts a new tuning. Steps already done are skipped; --redo
repeats an analysis step. With PORT_TUNE (or --port-tune), a new
tuning keeps that tuning's powers and only optimizes the frequencies
and spans again, from PREDICT's guess of them (and of the powers).
Needs crs_init.py first.

Usage:
    python 03_tune.py --device deviceA --modules 1
    python 03_tune.py --device deviceA --modules 1 --redo power_fits
    python 03_tune.py --device deviceA --modules 1 --tuning warm
    python 03_tune.py --device deviceA --modules 1 --tuning warm2 \
        --port-tune warm
"""
import argparse
import asyncio
import os
import re
import numpy as np
from citkid import daq_layout as dl
from citkid.crs.instrument import CRS
from citkid.crs.procedures import stitched_power_sweep, stitched_target_sweep
from citkid.crs.util import max_stream_tones
from citkid.crs.sim import simulated_crs
from citkid.pipeline.interactive import run_iq_series
from citkid.pipeline.sweep_series import a_vs_power_fit, open_series_datasets
from citkid.tone_plan import place_cal_tones, plan_tone_sets, select_isolated
from citkid.vna.fq_finder import run_fqfinder
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import (SERIAL_NUMBER, INTERFACE, MODULES, DATA_DIR, COOLDOWN,
                      DEVICE, SUFFIX, LIGHT_MODE)

# Tuning
TUNING = None

# Tone plan
RESONATORS = None
MIN_SEP_LW = 5.
PAIRS = 'drop_both'
EDGE_MARGIN = 10e6
STREAM_DEC = 6
N_CAL_TONES = 0
CAL_MIN_SEP_LW = 10.
WINDOWS = None
FMIN = None
FMAX = None

# Power sweep
POWERS = np.arange(-60., -35., 3)
SHUFFLE = False
SEED = 0
NPOINTS_FINE = 1000
NPOINTS_GAIN = 50
GAIN_SPAN_FACTOR = 10
NSAMPS = 200

# Power fits
F_AVG = 4
A_FIT_RANGE = (0.05, 0.8)
ANL = 0.5
ARES_LIMITS = (-60., -35.)

# fq finder
FRES_UPDATE_METHOD = 'spacing'
RMV_GAIN_SIMPLE = True

# Port the powers of an earlier tuning
PORT_STORE = None
PORT_TUNE = None
PREDICT = None

STEPS = ('plan', 'power_sweep', 'power_fits', 'fq_sweep', 'fq_finder',
         'tones')
REDOABLE = ('power_fits', 'fq_finder', 'tones')

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
    p = argparse.ArgumentParser(description = 'Tune a device.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--resonators', type = int, default = RESONATORS,
                   help = 'resonator list index (default: the latest)')
    p.add_argument('--tuning', default = TUNING,
                   help = 'the tuning, by name or index: continued if it '
                   'exists, else made (default: the latest, or a first '
                   'one)')
    p.add_argument('--port-tune', default = PORT_TUNE,
                   help = 'a new tuning keeps the powers of this tuning '
                   '(of --port-store, or this store) and only optimizes '
                   'the frequencies and spans')
    p.add_argument('--port-store', default = PORT_STORE,
                   help = 'the store of --port-tune (default: this one)')
    p.add_argument('--redo', default = None, choices = REDOABLE,
                   help = 'repeat this step and the ones after it')
    p.add_argument('--simulate', action = 'store_true',
                   help = "sweep the store's simulated device")
    p.add_argument('--windows', type = int, nargs = '+', default = WINDOWS,
                   help = 'NCO windows to use (default: all)')
    p.add_argument('--list-windows', action = 'store_true',
                   help = 'print the NCO windows of a new tuning and stop')
    p.add_argument('--fmin', type = float, default = FMIN,
                   help = 'lowest tone frequency, Hz (default: no limit)')
    p.add_argument('--fmax', type = float, default = FMAX,
                   help = 'highest tone frequency, Hz (default: no limit)')
    p.add_argument('--notes', default = NOTES,
                   help = 'notes saved with the data (quote multi-line text)')
    return p.parse_args(argv)

def predict_tones(fres, qres, ares):
    """
    Predict the tones of a ported tuning (set ``PREDICT =
    predict_tones`` to use it; edit the body): where the resonances
    are now and the powers to use. The fq sweep of the new tuning is
    taken around the predicted frequencies at the predicted powers.

    Parameters:
    fres (np.array): the earlier tuning's tone frequencies in Hz.
    qres (np.array): its spans, as quality factors.
    ares (np.array): its tone powers in dBm.

    Returns:
    fres_pred (np.array): predicted frequencies in Hz.
    qres_pred (np.array): predicted spans.
    ares_pred (np.array): the powers to use, dBm.
    """
    return fres, qres, ares

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

def sweep_kwargs():
    """
    Return the target sweep settings.

    Returns:
    kwargs (dict): ``target_sweep`` parameters.
    """
    return dict(npoints_fine = NPOINTS_FINE, npoints_gain = NPOINTS_GAIN,
                nsamps = NSAMPS,
                gain_span_factor = GAIN_SPAN_FACTOR)

async def connect(args, path, offset = 0):
    """
    Attach to this session's modules (or the simulated board).

    Parameters:
    args (argparse.Namespace): options.
    path (str): store path (for ``--simulate``).
    offset (int): noise seed offset of the simulated board. Default 0.

    Returns:
    crs (CRS or SimulatedCRS): the attached instrument.
    """
    if args.simulate:
        crs = simulated_crs(path, args.modules, seed = 100 + offset)
    else:
        crs = CRS(serial_number = args.serial, interface = args.iface,
                  module_idxs = args.modules, global_control = False)
    # Attaching sets the other module settings to their defaults
    await crs.attach(full_scale_dbm = 7)
    return crs

def done(tune, step):
    """
    Decide whether a step's output exists (and is complete).

    Parameters:
    tune (zarr.Group): the tuning.
    step (str): one of ``STEPS``.

    Returns:
    done (bool): True if the step is done.
    """
    if step == 'plan':
        return 'tones_init' in tune
    ported = bool(tune.attrs.get('ported_powers'))
    if step == 'power_sweep':
        return ported or 'power_sweep' in tune and \
            bool(tune['power_sweep'].attrs.get('complete'))
    if step == 'power_fits':
        return 'power_choice' in tune
    if step == 'fq_sweep':
        return 'fq_sweep' in tune and \
            bool(tune['fq_sweep'].attrs.get('complete'))
    if step == 'fq_finder':
        fq = tune['fq_sweep'] if 'fq_sweep' in tune else {}
        if 'fres_opt' not in fq or 'reject_reason' not in fq:
            return False
        kept = fq['reject_reason'][:] == ''
        return bool(np.all(np.isfinite(fq['fres_opt'][:][kept])))
    return 'tones' in tune

def plan(root, args, path):
    """
    Choose the tones and tone sets of a new tuning.

    Parameters:
    root (zarr.Group): the store root.
    args (argparse.Namespace): options.
    path (str): store path.

    Returns:
    tune (zarr.Group or None): the new tuning (None with
        ``--list-windows``, which only prints the windows).
    """
    rk = dl.latest_index(root, dl.RESONATORS_PREFIX) \
        if args.resonators is None else args.resonators
    name = dl.set_name(rk, dl.RESONATORS_PREFIX)
    res = dl.load_resonators(root, rk)
    strength = None
    if PAIRS == 'keep_stronger':
        strength = res['depth_all']
        if not np.isfinite(strength).any():
            raise ValueError(
                f"{name} has no dip depths for PAIRS = 'keep_stronger' "
                '(it was saved before they were); rerun 02_find_resonances.py'
                " --redo resonators, or use PAIRS = 'drop_both'")
    keep, _ = select_isolated(res['fres_all'],
                              res['fres_all'] / res['Qr_all'],
                              min_sep_lw = MIN_SEP_LW, pairs = PAIRS,
                              strength = strength)
    n_close = int((~keep).sum())
    # Record the close tones in the store's tone marks, as notes (the
    # tone may be measurable later, with another rule)
    for r in res['res_idxs_all'][~keep].tolist():
        marks = dl.tone_marks(root, r)
        note = f'dropped by the plan of {name}: too close to another ' \
            'resonance'
        if not (marks and marks[-1]['text'] == note):
            dl.add_tone_mark(root, r, 'note', note, f'{name}/tones_init')
    in_range = frequency_range(res['fres_all'], args.fmin, args.fmax)
    n_out = int((keep & ~in_range).sum())
    keep &= in_range
    if not keep.any():
        raise ValueError(f'no tones: none of the {len(keep)} resonators of '
                         f'{name} is isolated and in {range_text(args)}')
    crs = asyncio.run(connect(args, path))
    dl.check_crs_identity(root, crs)
    fres = res['fres_all'][keep]
    max_tones = max_stream_tones(STREAM_DEC, 1)[0]
    kw = dict(edge_margin = EDGE_MARGIN, extended_bw = bool(crs.extended_bw),
              max_tones = max_tones)
    tplan = plan_tone_sets(fres, crs.active_module_idxs, **kw)
    window = tone_windows(fres, tplan)
    if args.list_windows:
        print(describe_windows(fres, tplan, window))
        return None
    if args.windows is not None:
        bad = sorted(set(args.windows) - set(range(len(tplan['window_ncos']))))
        if bad:
            raise ValueError(f'no windows {bad}: there are '
                             f"{len(tplan['window_ncos'])} (--list-windows)")
        chosen = np.isin(window, args.windows)
        keep[np.flatnonzero(keep)[~chosen]] = False
        fres = fres[chosen]
        tplan = plan_tone_sets(fres, crs.active_module_idxs, **kw)
    qres, res_idxs = res['qres_all'][keep], res['res_idxs_all'][keep]
    tone_set, tone_module = tplan['tone_set'], tplan['tone_module']
    fcal, scal, mcal = cal_tones(res, tplan, max_tones)
    tune, k = dl.new_tuning(root, name, name = args.tuning)
    save_notes(tune, args.notes)
    dl.write_tune_config(tune, dl.tuning_config(crs, tplan))
    init = dl.create_group(tune, 'tones_init', 'decision', attrs = dict(
        source = name, min_sep_lw = MIN_SEP_LW, pairs = PAIRS,
        edge_margin = EDGE_MARGIN, n_dropped = n_close,
        stream_dec = STREAM_DEC, max_tones = max_tones,
        n_cal_tones = len(fcal), cal_min_sep_lw = CAL_MIN_SEP_LW,
        windows = None if args.windows is None else list(args.windows),
        fmin = args.fmin, fmax = args.fmax, n_out_of_range = n_out))
    dl.save_tones(
        init, np.concatenate((fres, fcal)),
        np.concatenate((qres, np.full(len(fcal), np.median(qres)))), np.nan,
        np.concatenate((res_idxs, -1 - np.arange(len(fcal)))),
        tone_set = np.concatenate((tone_set, scal)),
        tone_module = np.concatenate((tone_module, mcal)),
        source = dict(fres = f'{name}/fres_all', qres = f'{name}/qres_all'))
    print(f'plan: tune{k}: {keep.sum()} tones of {len(keep)} resonators '
          f'({n_close} too close'
          + ('' if args.fmin is None and args.fmax is None else
             f', {n_out} outside {range_text(args)}')
          + ('' if args.windows is None else
             f', windows {list(args.windows)}')
          + f'), {len(tplan["ncos"])} tone set(s)'
          + (', extended bandwidth' if tplan['extended_bw'] else '')
          + (f', {len(fcal)} calibration tones' if len(fcal) else ''))
    return tune

def plan_from_tuning(root, args, path):
    """
    Start a tuning from an earlier one's tones (``--port-tune``, in
    this store or ``--port-store``): its powers are kept
    (``power_choice``, or ``PREDICT``'s) and the plan, power sweep and
    power fits are skipped, so only the fq sweep, fq_finder and tones
    steps run.
    ``tones_init`` holds the earlier tones at the frequencies, spans and
    powers ``PREDICT`` returns (None: unchanged). Another store's tuning is
    copied in first (``daq_layout.port_tuning``, with its resonator
    list and CRS settings).

    Parameters:
    root (zarr.Group): the store root.
    args (argparse.Namespace): options.
    path (str): store path.

    Returns:
    tune (zarr.Group): the new tuning.

    Raises:
    KeyError: if the earlier tuning has no tones.
    ValueError: if ``PREDICT`` returns the wrong shapes or non-finite
        values.
    """
    if args.port_store is None:
        src_root, store = root, None
    else:
        store = os.path.expanduser(args.port_store)
        src_root = dl.open_store(store)
    src = dl.find_tuning(src_root, args.port_tune)
    if src is None:
        raise KeyError(f'no tuning {args.port_tune!r} to port'
                       f"{'' if store is None else ' in ' + store}")
    if 'tones' not in src:
        raise KeyError(f'{src.basename} has no tones to port')
    ported = dict(store = store, group = src.basename)
    if store is None:
        tune, k = dl.new_tuning(root, src.attrs['resonators'], attrs = dict(
            ported_from = ported, ported_powers = True),
            name = args.tuning)
        dl.write_tune_config(tune, src['crs_config'].attrs['config'])
    else:
        k = dl.port_tuning(src_root, int(src.basename[4:]), path)
        tune = root[dl.set_name(k, 'tune')]
        del tune['tones']    # the tones step makes the new ones
        tune.attrs['ported_powers'] = True
        tune.attrs['name'] = tune.basename if args.tuning is None \
            else str(args.tuning)
    save_notes(tune, args.notes)
    t = dl.load_tones(src['tones'])
    fres, qres, ares = t['fres'], t['qres'], t['ares']
    if PREDICT is not None:
        out = PREDICT(fres.copy(), qres.copy(), ares.copy())
        if len(out) != 3:
            raise ValueError('PREDICT must return fres_pred, qres_pred and '
                             'ares_pred')
        fres, qres, ares = (np.asarray(v, dtype = float) for v in out)
        if any(v.shape != t[k].shape for v, k in
               ((fres, 'fres'), (qres, 'qres'), (ares, 'ares'))):
            raise ValueError('PREDICT must return fres_pred, qres_pred and '
                             'ares_pred of the same shapes as fres, qres '
                             'and ares')
        if not all(np.isfinite(v).all() for v in (fres, qres, ares)):
            raise ValueError('PREDICT returned non-finite values')
    init = dl.create_group(tune, 'tones_init', 'decision', attrs = dict(
        source = f'{src.basename}/tones', ported_from = ported,
        predicted = PREDICT is not None, stream_dec = STREAM_DEC))
    dl.save_tones(
        init, fres, qres, np.nan, t['res_idxs'], tone_set = t['tone_set'],
        tone_module = t['tone_module'],
        source = dict(fres = f'{src.basename}/tones/fres',
                      qres = f'{src.basename}/tones/qres'))
    grp = dl.create_group(tune, 'power_choice', 'decision', attrs = dict(
        notes = (args.notes or '').strip(), ported_from = ported,
        source = f'{src.basename}/tones/ares'))
    n = len(ares)
    for name, value in (('ares', ares), ('res_idxs', t['res_idxs']),
                        ('fallback', np.zeros(n, bool)),
                        ('user_chosen', np.zeros(n, bool))):
        dl.write_array(grp, name, value)
    print(f'plan: tune{k} from {src.basename} of '
          f"{store or 'this store'}: {n} tones at its powers"
          + (', frequencies, spans and powers predicted'
             if PREDICT is not None else ''))
    return tune

def frequency_range(fres, fmin = None, fmax = None):
    """
    Return which frequencies are in a range.

    Parameters:
    fres (np.array): frequencies, Hz.
    fmin (float or None): lowest frequency (inclusive), Hz. None: no limit.
    fmax (float or None): highest frequency (inclusive), Hz. None: no limit.

    Returns:
    in_range (np.array, bool): True where fmin <= fres <= fmax.

    Raises:
    ValueError: if fmin > fmax.
    """
    if fmin is not None and fmax is not None and fmin > fmax:
        raise ValueError(f'FMIN ({fmin:.6g} Hz) is above FMAX ({fmax:.6g} Hz)')
    fres = np.asarray(fres, dtype = float)
    in_range = np.ones(len(fres), dtype = bool)
    if fmin is not None:
        in_range &= fres >= fmin
    if fmax is not None:
        in_range &= fres <= fmax
    return in_range

def range_text(args):
    """
    Describe the tone frequency range, for messages.

    Parameters:
    args (argparse.Namespace): options (``fmin``, ``fmax``).

    Returns:
    text (str): e.g. '1-1.5 GHz', 'below 1.5 GHz', 'above 1 GHz', or
        'any frequency'.
    """
    lo = None if args.fmin is None else f'{args.fmin / 1e9:.6g}'
    hi = None if args.fmax is None else f'{args.fmax / 1e9:.6g}'
    if lo is not None and hi is not None:
        return f'{lo}-{hi} GHz'
    if hi is not None:
        return f'below {hi} GHz'
    if lo is not None:
        return f'above {lo} GHz'
    return 'any frequency'

def tone_windows(fres, tplan):
    """
    Return each tone's NCO window (index in frequency order).

    Parameters:
    fres (np.array): tone frequencies in Hz.
    tplan (dict): the tone-set plan (``tone_plan.plan_tone_sets``).

    Returns:
    window (np.array, int): window index per tone.
    """
    lo = np.asarray(tplan['window_edges'])[:, 0]
    return np.searchsorted(lo, fres, side = 'right') - 1

def describe_windows(fres, tplan, window):
    """
    Describe a plan's NCO windows, one line each, for choosing ``WINDOWS``.

    Parameters:
    fres (np.array): tone frequencies in Hz.
    tplan (dict): the tone-set plan.
    window (np.array, int): each tone's window (``tone_windows``).

    Returns:
    text (str): the description.
    """
    lines = [f"NCO windows (STREAM_DEC = {STREAM_DEC}: at most "
             f"{max_stream_tones(STREAM_DEC, 1)[0]} tones each):"]
    for w, ((lo, hi), nco) in enumerate(zip(tplan['window_edges'],
                                            tplan['window_ncos'])):
        lines.append(f'  {w}: NCO {nco / 1e6:.1f} MHz, tones '
                     f'{lo / 1e6:.1f}-{hi / 1e6:.1f} MHz, '
                     f'{int(np.sum(window == w))} tones')
    return '\n'.join(lines)

def cal_tones(res, tplan, max_tones):
    """
    Place ``N_CAL_TONES`` calibration tones in each NCO window of the plan
    (``tone_plan.place_cal_tones``), within the windows' tone limits.

    Parameters:
    res (dict): the resonator list (``daq_layout.load_resonators``).
    tplan (dict): the tone-set plan (``tone_plan.plan_tone_sets``).
    max_tones (int): most tones per window.

    Returns:
    fcal (np.array): tone frequencies in Hz.
    tone_set, tone_module (np.array, int): their tone sets and modules.
    """
    out = ([], [], [])
    if not N_CAL_TONES:
        return tuple(np.zeros(0, dtype = t) for t in (float, int, int))
    half = (575e6 / 2 if tplan['extended_bw'] else 250e6) - EDGE_MARGIN
    lw = res['fres_all'] / res['Qr_all']
    for s, ncos in enumerate(tplan['ncos']):
        for m, nco in ncos.items():
            used = np.sum((tplan['tone_set'] == s) &
                          (tplan['tone_module'] == int(m)))
            n = min(N_CAL_TONES, max_tones - int(used))
            f = place_cal_tones(res['fres_all'], lw, nco - half, nco + half,
                                n, min_sep_lw = CAL_MIN_SEP_LW)
            out[0].extend(f)
            out[1].extend([s] * len(f))
            out[2].extend([int(m)] * len(f))
    return (np.asarray(out[0], dtype = float),
            np.asarray(out[1], dtype = int), np.asarray(out[2], dtype = int))

def sweep(root, tune, args, path, step):
    """
    Take the power sweep or the fq sweep (stitched over the tone sets).

    Parameters:
    root (zarr.Group): the store root.
    tune (zarr.Group): the tuning.
    args (argparse.Namespace): options.
    path (str): store path.
    step (str): 'power_sweep' or 'fq_sweep'.
    """
    t = dl.load_tones(tune['tones_init'])
    ncos = tune['crs_config'].attrs['config']['tone_sets']

    async def run():
        """
        Connect, check the board against the tuning, and sweep.
        """
        crs = await connect(args, path, offset = STEPS.index(step))
        dl.check_crs_identity(root, crs)
        dl.check_tune_config(tune, crs)
        if step == 'power_sweep':
            grp = dl.require_group(tune, 'power_sweep', 'raw')
            save_notes(grp, args.notes)
            grp.attrs.update(dict(powers = [float(p) for p in POWERS],
                                  shuffle = SHUFFLE, seed = SEED))
            await stitched_power_sweep(
                crs, t['fres'], POWERS, t['qres'], t['res_idxs'],
                t['tone_set'], t['tone_module'], ncos, grp,
                shuffle = SHUFFLE, seed = SEED, **sweep_kwargs())
            grp.attrs['complete'] = True
            return
        if 'fq_sweep' in tune:
            del tune['fq_sweep']    # an empty group left by a failed pass
        grp = dl.create_group(tune, 'fq_sweep', 'raw', attrs = dict(
            notes = (args.notes or '').strip(),
            ares_source = f'{tune.basename}/power_choice/ares'))
        await stitched_target_sweep(
            crs, t['fres'], tune['power_choice/ares'][:], t['qres'],
            t['res_idxs'], t['tone_set'], t['tone_module'], ncos, grp,
            **sweep_kwargs())

    asyncio.run(run())
    print(f'{step}: done')

def power_fits(tune, args):
    """
    Fit the power sweep in the IQ series window, then save the chosen
    powers (``power_choice``).

    Parameters:
    tune (zarr.Group): the tuning.
    """
    res_name = tune.attrs['resonators']
    out = dl.require_group(tune, 'power_fits', 'analysis')
    datasets = open_series_datasets(
        out_root = out, group = f'{tune.basename}/power_sweep',
        fres_all = f'{res_name}/fres_all', qres_all = f'{res_name}/qres_all',
        f_avg = F_AVG)
    # Calibration tones (res_idxs < 0) have no resonance to fit: they get
    # the median power below
    rows = np.flatnonzero(tune['tones_init/res_idxs'][:] >= 0)
    run_iq_series(
        datasets = datasets, analysis_yaml_path = 'iq', root = out,
        data_idxs = [int(r) for r in rows],
        x_param_name = 'ares', x_name = 'Tone power (dBm)',
        y_param_name = 'a', xy_fit = a_vs_power_fit(a_min = A_FIT_RANGE[0],
                                                    a_max = A_FIT_RANGE[1]),
        y_target = ANL)
    res_idxs = tune['tones_init/res_idxs'][:]
    chosen = out['xy_fit/x_chosen'][:] if 'xy_fit/x_chosen' in out else \
        np.full(len(res_idxs), np.nan)
    user = out['xy_fit/x_chosen_user'][:] \
        if 'xy_fit/x_chosen_user' in out else np.zeros(len(chosen), bool)
    chosen = np.where(res_idxs >= 0, chosen, np.nan)
    fallback = ~np.isfinite(chosen)
    if fallback.all():
        raise ValueError('no power was chosen for any tone: check the '
                         'power fits (--redo power_fits)')
    median = float(np.median(chosen[~fallback]))
    unclipped = np.where(fallback, median, chosen)
    ares = np.clip(unclipped, *ARES_LIMITS)
    grp = dl.create_group(tune, 'power_choice', 'decision', attrs = dict(
        notes = (args.notes or '').strip(),
        source = f'{tune.basename}/power_fits/xy_fit/x_chosen', anl = ANL,
        a_fit_range = list(A_FIT_RANGE), fallback = median,
        limits = list(ARES_LIMITS)), overwrite = True)
    for name, value in (('ares', ares), ('res_idxs', res_idxs),
                        ('fallback', fallback), ('user_chosen', user)):
        dl.write_array(grp, name, value)
    print(f'power_fits: {len(ares)} powers chosen ({int(user.sum())} by '
          f'hand, {int(fallback.sum())} fallback, '
          f'{int(np.sum(ares != unclipped))} clipped)')

def fq_finder(tune):
    """
    Set each tone's frequency and span in the fq_finder window.

    Parameters:
    tune (zarr.Group): the tuning.
    """
    fq = tune['fq_sweep']
    run_fqfinder(
        f = fq['fine_sweep/f'][:], z = fq['fine_sweep/z'][:],
        fres = fq['fres'][:], qres = fq['qres'][:],
        res_idxs = fq['res_idxs'][:], zarr_grp = fq,
        title = f'FQ Finder: {tune.basename}',
        fres_update_method = FRES_UPDATE_METHOD,
        rmv_gain_simple = RMV_GAIN_SIMPLE)

def tones(tune, args):
    """
    Save the final tone list (``tones``): the tones not rejected in
    fq_finder, at the chosen powers.

    Parameters:
    tune (zarr.Group): the tuning.

    Raises:
    ValueError: if a tone moved outside its tone set's window.
    """
    fq, name = tune['fq_sweep'], tune.basename
    keep = (fq['reject_reason'][:] == '') & np.isfinite(fq['fres_opt'][:])
    # Tones marked bad globally in this tuning (the windows of 03, or the
    # store browser) and not cleared since are left out of the tuned set
    root = dl.store_root(tune)
    bad = dl.bad_tones(root, f'{name}/tones')
    marked = np.array([int(r) in bad for r in fq['res_idxs'][:]])
    if (keep & marked).any():
        print(f"{int((keep & marked).sum())} tones marked bad in the store's "
              'tone marks are left out: res '
              f'{fq["res_idxs"][:][keep & marked].tolist()}')
    keep &= ~marked
    fres = fq['fres_opt'][:][keep]
    tone_set = fq['tone_set'][:][keep]
    tone_module = fq['tone_module'][:][keep]
    config = tune['crs_config'].attrs['config']
    for f, s, m in zip(fres, tone_set, tone_module):
        nco = config['tone_sets'][s][str(m)]
        if abs(f - nco) > config['bw'] / 2:
            raise ValueError(f'tone at {f / 1e6:.3f} MHz moved outside tone '
                             f'set {s} (NCO {nco / 1e6:.3f} MHz)')
    grp = dl.create_group(tune, 'tones', 'decision', overwrite = True)
    save_notes(grp, args.notes)
    dl.save_tones(
        grp, fres, fq['qres_opt'][:][keep],
        tune['power_choice/ares'][:][keep], fq['res_idxs'][:][keep],
        tone_set = tone_set, tone_module = tone_module, source = dict(
            fres = f'{name}/fq_sweep/fres_opt',
            qres = f'{name}/fq_sweep/qres_opt',
            ares = f'{name}/power_choice/ares',
            res_idxs = f'{name}/fq_sweep/res_idxs',
            tone_set = f'{name}/tones_init/tone_set'))
    print(f'tones: {keep.sum()} tones saved in {name}/tones '
          f'({(~keep).sum()} rejected)')

def main(args):
    """
    Run the steps that aren't done yet.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    k (int or None): the tuning index (None with ``--list-windows``).
    """
    set_theme(LIGHT_MODE)
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    tune = dl.find_tuning(root, args.tuning)
    if tune is None:
        if args.redo:
            raise ValueError('--redo applies to an existing tuning')
        if args.tuning is not None and \
                re.fullmatch(r'(?:tune)?\d+', str(args.tuning).strip()):
            raise KeyError(f'no tuning {args.tuning!r}')
        tune = plan_from_tuning(root, args, path) \
            if args.port_tune is not None else plan(root, args, path)
        if tune is None:
            return None
    else:
        if args.port_tune is not None:
            raise ValueError(
                f'tuning {dl.tuning_name(tune)!r} ({tune.basename}) '
                'exists; porting powers makes a new tuning: give '
                'TUNING (--tuning) a new name')
        print(f'{path}: continuing {tune.basename} '
              f'({dl.tuning_name(tune)!r})')
    redo = STEPS.index(args.redo) if args.redo else len(STEPS)
    if args.redo == 'power_fits' and 'fq_sweep' in tune:
        raise ValueError('the fq sweep was taken at the chosen powers; '
                         'start a new tuning (a new TUNING name) to '
                         'choose them again')
    for step in STEPS[1:]:
        if done(tune, step) and STEPS.index(step) < redo:
            print(f'{step}: done already')
            continue
        if step in ('power_sweep', 'fq_sweep'):
            sweep(root, tune, args, path, step)
        elif step == 'power_fits':
            power_fits(tune, args)
        elif step == 'fq_finder':
            fq_finder(tune)
            if not done(tune, 'fq_finder'):
                print('fq_finder: not finished; run again to continue')
                break
        else:
            tones(tune, args)
    return int(tune.basename[4:])

if __name__ == '__main__':
    main(parse_args())
