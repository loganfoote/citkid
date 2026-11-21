"""
07: noise spectra (Sxx) of calibrated timestreams, before and after
common-mode removal, reviewed in an overview window (histograms of Sxx
over the tones; mark tones bad), saved as a set in ts_psd{p}. When the
set exists, a popup offers a new set, computing it again with its saved
review, or overwriting it. With TIMESTREAMS empty, lists the timestreams.
No CRS needed.

Usage:
    python 6c_psds.py --device deviceA --name dark
    python 6c_psds.py --device deviceA --name dark --load --no-gui
    python 6c_psds.py --device deviceA --deglitch photons --photons dark
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline import psds
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX, MEMORY_GB, LIGHT_MODE

# Timestreams
TIMESTREAMS = [
]
NAME = ''
INTERACTIVE = True

# Spectra
NBINS = 100
FMIN_MULT = 10.
FMAX_DIV = 10.

# Common-mode removal
CN_REMOVE = True
CN_N_COMP = 3
CN_N_ITER = 5
CN_LOWPASS_HZ = None
CN_HIGHPASS_HZ = None
CN_FILTER_ORDER = 2

# Tones left out
BAD_TONES = []

# Deglitching
DEGLITCH = None
GLITCH_PRE_S = 0.2e-3
GLITCH_POST_S = 2.e-3
GLITCH_SIGN = -1
GLITCH_NSTD = 5.
GLITCH_BASELINE_S = 1.
GLITCH_DISTANCE_S = 1.e-3
GLITCH_ROUNDS = 2
PHOTONS = None
GLITCH_PULSES = 'hee'
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
        description = 'Noise spectra of calibrated timestreams.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--deglitch', default = DEGLITCH,
                   choices = [o for o in psds.DEGLITCH_OPTIONS if o],
                   help = 'take out glitches first')
    p.add_argument('--photons', default = PHOTONS,
                   help = 'photon result whose filters find the pulses '
                   '(the name of a photons/<name>), with --deglitch '
                   'photons')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'skip the overview window')
    p.add_argument('--new', action = 'store_true',
                   help = 'another set, when one with this name exists')
    p.add_argument('--load', action = 'store_true',
                   help = 'compute the existing set again with its saved '
                   'settings and review, without the window')
    p.add_argument('--overwrite', action = 'store_true',
                   help = 'replace the existing set without asking')
    p.add_argument('--no-cn', action = 'store_true',
                   help = 'keep the common modes')
    p.add_argument('--name', default = NAME,
                   help = 'name of this set of spectra')
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

def analysis_settings(args):
    """
    Return the analysis settings from the settings above and the options.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    settings (dict): for ``pipeline.psds.analyze_timestream``.
    """
    return dict(
        nbins = NBINS, fmin_mult = FMIN_MULT, fmax_div = FMAX_DIV,
        cn_remove = CN_REMOVE and not args.no_cn, cn_n_comp = CN_N_COMP,
        cn_n_iter = CN_N_ITER, cn_lowpass_hz = CN_LOWPASS_HZ,
        cn_highpass_hz = CN_HIGHPASS_HZ, cn_filter_order = CN_FILTER_ORDER,
        deglitch = args.deglitch, glitch_pre_s = GLITCH_PRE_S,
        glitch_post_s = GLITCH_POST_S, glitch_sign = GLITCH_SIGN,
        glitch_nstd = GLITCH_NSTD, glitch_baseline_s = GLITCH_BASELINE_S,
        glitch_distance_s = GLITCH_DISTANCE_S, glitch_rounds = GLITCH_ROUNDS,
        glitch_pulses = GLITCH_PULSES, coinc_win = COINC_WIN,
        hee_min_mult = HEE_MIN_MULT, memory_gb = MEMORY_GB,
        photons = args.photons)

def review(ts, result, f_reduced, root, stage, limits = None):
    """
    Open the overview window on a timestream's raw spectra.

    Parameters:
    ts (dict): the timestream.
    result (dict): ``analyze_timestream`` output without common-mode
        removal.
    f_reduced (np.array): the group's reduced frequencies.
    root (zarr.Group): the store root (the tone-marks registry).
    stage (str): the result's path, recorded with the marks.
    limits (dict or None): review limits to open with. Default None.

    Returns:
    bad (dict): res_idx: reason, the tones marked bad.
    limits (dict): the overview's review limits.
    """
    from citkid.pipeline.interactive.psd_overview import run_psd_overview
    return run_psd_overview(
        ts, result, f_reduced, root = root, stage = stage,
        limits = dict(limits or {}),
        title = f"Noise spectra: {ts['run']} ({ts['name']}), tone set "
        f"{ts['tone_set']}")

def existing_choice(grp, args, interactive):
    """
    Decide what to do with the saved set a run would make again: a new
    set, compute it again with its saved review, overwrite it, or stop.
    With the window a popup asks; otherwise ``--new``, ``--load`` or
    ``--overwrite`` decide and anything else stops.

    Parameters:
    grp (zarr.Group): the saved set.
    args (argparse.Namespace): options from ``parse_args``.
    interactive (bool): If True, the window is in use.

    Returns:
    choice (str or None): 'new', 'load', 'overwrite', or None to stop.
    """
    if args.new:
        return 'new'
    if args.load:
        return 'load'
    if args.overwrite:
        return 'overwrite'
    if interactive:
        from citkid.pipeline.interactive.psd_overview import \
            ask_existing_psd
        return ask_existing_psd(grp.path, psds.describe_group(grp))
    print(f'{grp.path} exists: skipped (run with --new for another set, '
          '--load to compute it again with its saved review, or '
          '--overwrite to replace it)')
    return None

def report(ts, result, settings, bad = None):
    """
    Print a summary of one timestream's spectra.

    Parameters:
    ts (dict): the timestream.
    result (dict): ``analyze_timestream`` output.
    settings (dict): the settings used.
    bad (dict or None): res_idx: reason, the tones marked bad. Default
        None.
    """
    ok = np.asarray([s == '' for s in result['status']])
    text = (f"{ts['run']} ({ts['name']}), tone set {ts['tone_set']}, "
            f"{ts['cal']}: {ok.sum()} of {len(ok)} tones, "
            f"{len(result['f_binned'])} bins, "
            f"{int(result['valid_reduced'].sum())} reduced frequencies")
    if settings['deglitch']:
        text += f", {int(result['n_glitches'][ok].clip(0).sum())} glitches"
    if settings['cn_remove']:
        text += ', common modes ' + ', '.join(
            f"{w['name']}: {w['n_comp']}" for w in result['writes'])
    print(text + (f', {len(bad)} marked bad in the window' if bad else ''))
    for r, s in zip(ts['res_idxs'], result['status']):
        if s:
            print(f'  res {r}: {s}')

def main(args):
    """
    Compute the spectra of the chosen timestreams, or list them.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    out (dict or None): timestream group name: ``analyze_timestream``
        output, or None if only listed.
    """
    set_theme(LIGHT_MODE)
    root = dl.open_store(store_path(args), mode = 'a')
    dl.check_store(root)
    timestreams, datasets = psds.list_timestreams(root)
    print(psds.describe_timestreams(timestreams))
    if not TIMESTREAMS:
        print(psds.describe_results(root))
        print('Set TIMESTREAMS at the top of the script to choose '
              'timestreams.')
        return None
    chosen = psds.select_timestreams(timestreams, TIMESTREAMS)
    interactive = INTERACTIVE and not args.no_gui
    name = args.name.strip() or None
    existing = psds.matching_group(root, name, list(TIMESTREAMS))
    choice = 'new' if existing is None else \
        existing_choice(existing, args, interactive)
    if choice is None:
        print(f'Skipped {existing.path}')
        return None
    settings = analysis_settings(args)
    photons = args.photons
    saved = {} if existing is None else psds.saved_review(existing)
    if choice == 'load':
        # The saved settings and review, no window
        settings = {**existing.attrs['settings'], 'memory_gb': MEMORY_GB}
        photons = settings.get('photons')
        interactive = False
    ref_grp = None
    if settings['deglitch'] == 'photons':
        if photons is None:
            raise ValueError("DEGLITCH = 'photons' needs PHOTONS (or "
                             '--photons): the name of a photon result')
        from citkid.pipeline.photon_finder import PHOTONS_GROUP
        ref_grp = root[PHOTONS_GROUP][str(photons)]
    f_reduced = psds.union_freqs(chosen, settings['fmin_mult'],
                                 settings['fmax_div'])
    attrs = dict(name = name, timestreams = list(TIMESTREAMS),
                 settings = settings,
                 reference = None if ref_grp is None else ref_grp.path)
    if choice == 'load':
        # The saved spectra stay until each timestream's are replaced
        out_grp = existing
        out_grp.attrs.update(dl.to_jsonable(attrs))
        if 'f_reduced' in out_grp:
            del out_grp['f_reduced']
        print(f'Computing {out_grp.path} again with its saved review')
    elif choice == 'overwrite':
        out_grp = dl.create_group(root, existing.path, 'analysis',
                                  attrs = attrs, overwrite = True)
        save_notes(out_grp, args.notes)
        saved = {}
        print(f'Overwriting {out_grp.path}')
    else:
        out_grp, _ = dl.new_indexed_group(root, psds.PREFIX, 'analysis',
                                          attrs = attrs)
        save_notes(out_grp, args.notes)
    dl.write_array(out_grp, 'f_reduced', f_reduced)
    print(f'Saving to {out_grp.path}; reduced frequencies '
          f'{f_reduced.tolist()} Hz')
    out = {}
    try:
        for j, ts in enumerate(chosen):
            out[f'timestream{j}'] = analyze_one(
                root, ts, j, datasets, out_grp, f_reduced, settings,
                ref_grp, saved, choice, interactive)
    finally:
        if not out and choice != 'load':
            # Nothing was saved: no empty set is left behind
            del root[out_grp.path]
            print(f'{out_grp.path} removed: nothing was saved')
    print(f'Done: {out_grp.path}')
    return out

def analyze_one(
    root, ts, j, datasets, out_grp, f_reduced, settings, ref_grp, saved,
    choice, interactive
):
    """
    Compute, review and save one timestream's spectra.

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    j (int): its position in the set (``timestream{j}``).
    datasets (dict): ``psds.list_timestreams`` output.
    out_grp (zarr.Group): the set's group.
    f_reduced (np.array): the set's reduced frequencies.
    settings (dict): the analysis settings.
    ref_grp (zarr.Group or None): the photon result for deglitching.
    saved (dict): ``psds.saved_review`` of the set being made again.
    choice (str): 'new', 'load' or 'overwrite'.
    interactive (bool): If True, the overview window opens.

    Returns:
    result (dict): ``analyze_timestream`` output.
    """
    DS = datasets[ts['cal']][ts['dataset']]['DS']
    filters, ph_settings = None, None
    if ref_grp is not None:
        filters, ph_settings = psds.load_photon_reference(ref_grp)
    name = f'timestream{j}'
    exclude = {int(r): dl.bad_reason(e) for r, e in dl.bad_tones(
        root, f'{out_grp.path}/{name}',
        tuning = str(ts['source']['group']).split('/')[0]).items()}
    exclude.update({int(r): 'BAD_TONES' for r in BAD_TONES})
    left_out = sorted(set(exclude) & set(ts['res_idxs'].tolist()))
    if left_out:
        print(f'{len(left_out)} tones left out (marked bad): res '
              f'{left_out}')
    bad, limits = saved.get(name, ({}, {}))
    if choice == 'load':
        exclude = {**exclude, **{int(r): v for r, v in bad.items()}}
    elif interactive:
        # The raw spectra first, for the overview window; the tones
        # marked bad there stay out of the common-mode fit
        raw = psds.analyze_timestream(
            root, ts, DS, f_reduced, dict(settings, cn_remove = False),
            filters = filters, photon_settings = ph_settings,
            progress = print, exclude = exclude)
        bad, limits = review(ts, raw, f_reduced, root,
                             f'{out_grp.path}/{name}', limits)
        exclude = {**exclude, **{int(r): v for r, v in bad.items()}}
    else:
        bad, limits = {}, {}
    result = psds.analyze_timestream(
        root, ts, DS, f_reduced, settings, filters = filters,
        photon_settings = ph_settings, progress = print,
        exclude = exclude)
    psds.save_timestream(
        dl.create_group(out_grp, name, 'analysis', overwrite = True),
        ts, result, settings, bad = bad, limits = limits)
    report(ts, result, settings, bad)
    return result

if __name__ == '__main__':
    main(parse_args())
