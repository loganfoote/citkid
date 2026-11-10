"""
03: tune a device: choose tones, find each tone's power and frequency, and
save the final tone list.

Steps, in a tuning ``tune{k}`` of the device store:

1. plan: start from a resonator list (``resonators{j}``, from step 02),
   drop close tones (``tone_plan.select_isolated``), split the rest into
   tone sets (``tone_plan.plan_tone_sets``), and save ``tones_init`` and
   ``crs_config`` (with each tone set's NCOs);
2. power_sweep: a target sweep at each power in ``POWERS``, stitched over
   the tone sets (``power_sweep/set{i}``);
3. power_fits: fit the IQ loops of every power in the IQ series window,
   with a fit of a vs power per resonator; the window marks the power
   where a reaches ``ANL``, and Shift+click (or dragging the line) picks
   another one. The chosen powers, with the median of the others where
   there is none, and then clipped to ``ARES_LIMITS``, are saved in
   ``power_choice``;
4. fq_sweep: a stitched target sweep at the chosen powers;
5. fq_finder: set each tone's frequency and span (or reject it) in the
   fq_finder window;
6. tones: save the final tone list in ``tones`` (``fres``, ``qres``,
   ``ares``, ``res_idxs``, ``tone_set``, ``tone_module``). With the
   tuning's ``crs_config`` and resonator list, this is everything
   ``daq_layout.port_tuning`` copies to measure with the tuning in another
   store.

Steps already done are skipped, so the script picks up where it stopped
(an interrupted power sweep continues at the next power). ``--redo``
repeats one of the analysis steps (power_fits, fq_finder, tones) and the
ones after it; the sweeps are raw data and are never redone in place: use
``--new`` for a new tuning.

The sweep steps attach to this session's modules of a board configured by
crs_init.py; the window steps need no CRS. ``--simulate`` sweeps the
simulated device of a store made by simulate_store.py instead.

Usage:
    python 03_tune.py --device deviceA --modules 1
    python 03_tune.py --device deviceA --modules 1 --redo power_fits
    python 03_tune.py --device deviceA --modules 1 --new

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import asyncio
import os
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

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46          # CRS serial number
INTERFACE = 'enp5s0'        # Ethernet interface connected to the CRS
DATA_DIR = '~/data'         # stores are in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
MODULES = [1]               # modules connected to this device
FULL_SCALE_DBM = 7          # DAC full scale of these modules, 0-7 dBm
# Tones
RESONATORS = None           # resonator list index (None: the latest)
MIN_SEP_LW = 5.             # drop tones closer than this many linewidths
PAIRS = 'drop_both'         # close pairs: 'drop_both' or 'keep_stronger'
EDGE_MARGIN = 10e6          # smallest distance of a tone from a window edge
# Fastest decimation stage the tuning will stream at. Tone sets get at most
# as many tones per module as can stream at once there (128 at stages 0-1,
# 1024 at 2-6 with the current limits), so streaming never changes NCOs
STREAM_DEC = 6
# Calibration tones (off resonance, res_idxs < 0) per NCO window, at the
# middles of the widest gaps between resonances, at least CAL_MIN_SEP_LW
# linewidths from any; streams of them are calibrated with 'ts_offres'
N_CAL_TONES = 0
# NCO windows to use (indices in frequency order, as --list-windows prints
# them), to look at part of the array; None: all. E.g. STREAM_DEC = 0 and
# WINDOWS = [2] gives the up-to-128 tones of one window
WINDOWS = None
CAL_MIN_SEP_LW = 10.
# Sweeps
POWERS = np.arange(-60., -35., 3)   # power sweep, dBm
SHUFFLE = False             # shuffle each tone's powers across the sets
SEED = 0                    # shuffle seed
NPOINTS_FINE = 1000         # fine sweep points per tone
NPOINTS_GAIN = 50           # gain sweep points per tone
NPOINTS_ROUGH = None        # rough sweep points per tone (None: none)
GAIN_SPAN_FACTOR = 10       # gain sweep span = 10 x fine sweep span
NSAMPS = 200                # samples averaged per point
# Power choice
F_AVG = 4                   # average this many fine sweep points in fits
A_FIT_RANGE = (0.05, 0.8)   # a range used in the a vs power fits
ANL = 0.5                   # target nonlinearity a
# Chosen powers: where no fit reaches ANL, the median of the others; then
# every power is clipped to ARES_LIMITS, dBm
ARES_LIMITS = (-55., -35.)
# fq_finder
FRES_UPDATE_METHOD = 'spacing'  # 'none', 'spacing', 'minS21', 'distance'
RMV_GAIN_SIMPLE = True      # remove the gain simply for display

STEPS = ('plan', 'power_sweep', 'power_fits', 'fq_sweep', 'fq_finder',
         'tones')
REDOABLE = ('power_fits', 'fq_finder', 'tones')

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
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--full-scale', type = float, default = FULL_SCALE_DBM)
    p.add_argument('--resonators', type = int, default = RESONATORS,
                   help = 'resonator list index (default: the latest)')
    p.add_argument('--tune', type = int, default = None,
                   help = 'tuning to continue (default: the latest)')
    p.add_argument('--new', action = 'store_true', help = 'start a tuning')
    p.add_argument('--redo', default = None, choices = REDOABLE,
                   help = 'repeat this step and the ones after it')
    p.add_argument('--simulate', action = 'store_true',
                   help = "sweep the store's simulated device")
    p.add_argument('--windows', type = int, nargs = '+', default = WINDOWS,
                   help = 'NCO windows to use (default: all)')
    p.add_argument('--list-windows', action = 'store_true',
                   help = 'print the NCO windows of a new tuning and stop')
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

def sweep_kwargs():
    """
    Return the target sweep settings.

    Returns:
    kwargs (dict): ``target_sweep`` parameters.
    """
    return dict(npoints_fine = NPOINTS_FINE, npoints_gain = NPOINTS_GAIN,
                npoints_rough = NPOINTS_ROUGH, nsamps = NSAMPS,
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
    await crs.attach(full_scale_dbm = args.full_scale)
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
    if step == 'power_sweep':
        return 'power_sweep' in tune and \
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
    keep, _ = select_isolated(res['fres_all'],
                              res['fres_all'] / res['Qr_all'],
                              min_sep_lw = MIN_SEP_LW, pairs = PAIRS)
    crs = asyncio.run(connect(args, path))
    dl.check_crs_identity(root, crs)
    fres = res['fres_all'][keep]
    n_close = int((~keep).sum())
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
    tune, k = dl.new_tuning(root, name)
    dl.write_tune_config(tune, dl.tuning_config(crs, tplan))
    init = dl.create_group(tune, 'tones_init', 'decision', attrs = dict(
        source = name, min_sep_lw = MIN_SEP_LW, pairs = PAIRS,
        edge_margin = EDGE_MARGIN, n_dropped = n_close,
        stream_dec = STREAM_DEC, max_tones = max_tones,
        n_cal_tones = len(fcal), cal_min_sep_lw = CAL_MIN_SEP_LW,
        windows = None if args.windows is None else list(args.windows)))
    dl.save_tones(
        init, np.concatenate((fres, fcal)),
        np.concatenate((qres, np.full(len(fcal), np.median(qres)))), np.nan,
        np.concatenate((res_idxs, -1 - np.arange(len(fcal)))),
        tone_set = np.concatenate((tone_set, scal)),
        tone_module = np.concatenate((tone_module, mcal)),
        source = dict(fres = f'{name}/fres_all', qres = f'{name}/qres_all'))
    print(f'plan: tune{k}: {keep.sum()} tones of {len(keep)} resonators '
          f'({n_close} too close'
          + ('' if args.windows is None else
             f', windows {list(args.windows)}')
          + f'), {len(tplan["ncos"])} tone set(s)'
          + (', extended bandwidth' if tplan['extended_bw'] else '')
          + (f', {len(fcal)} calibration tones' if len(fcal) else ''))
    return tune

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
            ares_source = f'{tune.basename}/power_choice/ares'))
        await stitched_target_sweep(
            crs, t['fres'], tune['power_choice/ares'][:], t['qres'],
            t['res_idxs'], t['tone_set'], t['tone_module'], ncos, grp,
            **sweep_kwargs())

    asyncio.run(run())
    print(f'{step}: done')

def power_fits(tune):
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

def tones(tune):
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
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    tunes = dl.list_indices(root, 'tune')
    if args.new or not tunes:
        if args.redo:
            raise ValueError('--redo applies to an existing tuning')
        tune = plan(root, args, path)
        if tune is None:
            return None
    else:
        k = tunes[-1] if args.tune is None else args.tune
        tune = root[dl.set_name(k, 'tune')]
        print(f'{path}: continuing {tune.basename}')
    redo = STEPS.index(args.redo) if args.redo else len(STEPS)
    if args.redo == 'power_fits' and 'fq_sweep' in tune:
        raise ValueError('the fq sweep was taken at the chosen powers; '
                         'start a new tuning (--new) to choose them again')
    for step in STEPS[1:]:
        if done(tune, step) and STEPS.index(step) < redo:
            print(f'{step}: done already')
            continue
        if step in ('power_sweep', 'fq_sweep'):
            sweep(root, tune, args, path, step)
        elif step == 'power_fits':
            power_fits(tune)
        elif step == 'fq_finder':
            fq_finder(tune)
            if not done(tune, 'fq_finder'):
                print('fq_finder: not finished; run again to continue')
                break
        else:
            tones(tune)
    return int(tune.basename[4:])

if __name__ == '__main__':
    main(parse_args())
