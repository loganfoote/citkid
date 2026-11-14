"""
07: noise spectra (Sxx) of calibrated timestreams, before and after
common-mode removal.

Choose the timestreams in ``TIMESTREAMS``. A timestream is one tone set of
one run (``ts{n}``), over the whole run, with a stream calibration
(``ts_cal{c}``, made by 05_calibrate_streams.py) that covers its tone
set; every (run, tone set, calibration) is listed, so each chosen
timestream can use a different calibration. With ``TIMESTREAMS`` empty,
the script lists the timestreams and the previous results and stops.

The timestreams chosen in one run of the script go together: they are
saved in one group (``ts_psd{p}``), with one grid of reduced frequencies
for all of them, so they can be plotted and compared as a set. Make
another run (or a copy of this script with other settings) for another
set.

For each timestream, tone write by tone write (the tones streamed
together), x is computed from the raw streams in memory
(``stream_sources.apply_stream_calibration``; calibrated timestreams are
never saved) and:

1. Glitches (cosmic rays) are optionally taken out (``DEGLITCH``):
   'peaks' finds the peaks of each tone above ``GLITCH_NSTD`` standard
   deviations (``signal.deglitch``); 'photons' applies the optimal
   filters of a photon result (``PHOTONS``, a ``ts_photons{p}`` of
   06_find_photons.py) and takes out the cosmic rays (events seen by at
   least ``COSMIC_MIN_MULT`` tones; ``GLITCH_PULSES = 'all'`` for every
   pulse). Either way the samples from ``GLITCH_PRE_S`` before to
   ``GLITCH_POST_S`` after each glitch are replaced with Gaussian noise.
2. Each tone's Sxx (``signal.psd.get_psd``) is reduced to its mean within
   20% of each frequency of the grid 0.1, 0.3, 1, 3, 10, ... Hz that the
   run allows (from ``FMIN_MULT`` times 1 / duration up to half the
   Nyquist frequency over ``FMAX_DIV``, well inside the spectrum), and
   binned logarithmically (``signal.psd.bin_psd``: ``NBINS`` bins from
   ``FMIN_MULT`` / duration, the lower frequencies unbinned; optionally
   with the pulse tube harmonics filtered, ``FILTER_PT_N``).
3. With ``CN_REMOVE``, the common modes of the write are found with the
   iterative algorithm of ``xcal.corr.calc_cm`` (``CN_N_COMP``
   components, ``CN_N_ITER`` iterations; a highpass on the timestreams
   while finding the weights, so the lowest 1/f noise does not dominate
   them, and a lowpass on the modes while iterating, for convergence:
   neither is applied to the modes that are removed) and removed; step 2
   is repeated on the cleaned tones (``sxx_reduced_cn``,
   ``sxx_binned_cn``), and the modes get binned spectra of their own
   (``common_modes/``).

The group's reduced frequencies are the union over its timestreams: each
timestream has values at its own frequencies and NaN elsewhere
(``valid_reduced``). ``pipeline.psds.load_group`` reads a result back.
No CRS is needed.

Usage:
    python 07_psds.py --device deviceA
    python 07_psds.py --device deviceA --deglitch photons --photons 0

Edit the settings below; the command-line options override them for one
run.
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline import psds

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in
                            # DATA_DIR/COOLDOWN/DEVICE[_SUFFIX].zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
SUFFIX = ''                 # store name suffix: the store is
                            # DEVICE_SUFFIX.zarr ('': DEVICE.zarr)
# Timestreams to analyze, by index (as listed), run name (every tone set
# of the latest run with that name, each with the latest calibration that
# covers it), or dict(name = ..., or run = n, and optionally tone_sets =
# [...], cal = c (a ts_cal index)). Leave empty to list the timestreams.
TIMESTREAMS = [
    # 'dark_dec6',
    # dict(name = 'dark_dec0', cal = 1),
]
# Spectra
NBINS = 100                 # logarithmic bins of the binned spectra
FMIN_MULT = 10.             # lowest binned and reduced frequency, times
                            # 1 / duration
FMAX_DIV = 10.              # highest reduced frequency: half the Nyquist
                            # frequency over this
FILTER_PT_N = None          # pulse tube harmonics to filter before
                            # binning (None: no filtering)
PT_FREQUENCY = 1.39296      # pulse tube frequency, Hz
# Common modes
CN_REMOVE = True            # find and remove the common modes
CN_N_COMP = 3               # components
CN_N_ITER = 5               # iterations
CN_LOWPASS_HZ = None        # lowpass on the modes while iterating, Hz
                            # (None: a tenth of the Nyquist frequency)
CN_HIGHPASS_HZ = None       # highpass on the timestreams while finding
                            # the weights, Hz (None: FMIN_MULT /
                            # duration; 0: no highpass)
CN_FILTER_ORDER = 2         # order of both filters
# Glitches
DEGLITCH = None             # None, 'peaks', or 'photons'
GLITCH_PRE_S = 0.2e-3       # replaced before each glitch, s
GLITCH_POST_S = 2.e-3       # replaced after each glitch, s
# 'peaks'
GLITCH_SIGN = -1            # sign of a glitch in x (photon pulses lower
                            # the resonance frequency: -1)
GLITCH_NSTD = 5.            # peak threshold, standard deviations
GLITCH_BASELINE_S = 1.      # baseline bin subtracted before peak finding
GLITCH_DISTANCE_S = 1.e-3   # fewest seconds between peaks
GLITCH_ROUNDS = 2           # rounds of peak finding
# 'photons'
PHOTONS = None              # p of the ts_photons{p} whose optimal filters
                            # find the pulses
GLITCH_PULSES = 'cosmic'    # 'cosmic' (events on at least COSMIC_MIN_MULT
                            # tones) or 'all' (every pulse)
COINC_WIN = 2               # coincidence window, +- samples
COSMIC_MIN_MULT = 3         # fewest tones of a cosmic ray
MEMORY_GB = 4.              # timestreams kept in memory per tone write, GB

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
    p.add_argument('--photons', type = int, default = PHOTONS,
                   help = "photon result whose filters find the pulses "
                   "(p of ts_photons{p}), with --deglitch photons")
    p.add_argument('--no-cn', action = 'store_true',
                   help = 'keep the common modes')
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
    name = args.device + (f'_{args.suffix}' if args.suffix else '')
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{name}.zarr')

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
        filter_pt_n = FILTER_PT_N, pt_frequency = PT_FREQUENCY,
        cn_remove = CN_REMOVE and not args.no_cn, cn_n_comp = CN_N_COMP,
        cn_n_iter = CN_N_ITER, cn_lowpass_hz = CN_LOWPASS_HZ,
        cn_highpass_hz = CN_HIGHPASS_HZ, cn_filter_order = CN_FILTER_ORDER,
        deglitch = args.deglitch, glitch_pre_s = GLITCH_PRE_S,
        glitch_post_s = GLITCH_POST_S, glitch_sign = GLITCH_SIGN,
        glitch_nstd = GLITCH_NSTD, glitch_baseline_s = GLITCH_BASELINE_S,
        glitch_distance_s = GLITCH_DISTANCE_S, glitch_rounds = GLITCH_ROUNDS,
        glitch_pulses = GLITCH_PULSES, coinc_win = COINC_WIN,
        cosmic_min_mult = COSMIC_MIN_MULT, memory_gb = MEMORY_GB,
        photons = args.photons)

def report(ts, result, settings):
    """
    Print a summary of one timestream's spectra.

    Parameters:
    ts (dict): the timestream.
    result (dict): ``analyze_timestream`` output.
    settings (dict): the settings used.
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
    print(text)
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
    settings = analysis_settings(args)
    ref_grp = None
    if settings['deglitch'] == 'photons':
        if args.photons is None:
            raise ValueError("DEGLITCH = 'photons' needs PHOTONS (or "
                             '--photons): a ts_photons result')
        ref_grp = root[dl.set_name(int(args.photons), 'ts_photons')]
    f_reduced = psds.union_freqs(chosen, FMIN_MULT, FMAX_DIV)
    out_grp, _ = dl.new_indexed_group(
        root, psds.PREFIX, 'analysis', attrs = dict(
            timestreams = list(TIMESTREAMS), settings = settings,
            reference = None if ref_grp is None else ref_grp.path))
    dl.write_array(out_grp, 'f_reduced', f_reduced)
    save_notes(out_grp, args.notes)
    print(f'Saving to {out_grp.path}; reduced frequencies '
          f'{f_reduced.tolist()} Hz')
    out = {}
    for j, ts in enumerate(chosen):
        DS = datasets[ts['cal']][ts['dataset']]['DS']
        filters, ph_settings = None, None
        if ref_grp is not None:
            filters, ph_settings = psds.load_photon_reference(
                ref_grp, ts['tone_set'])
        result = psds.analyze_timestream(
            root, ts, DS, f_reduced, settings, filters = filters,
            photon_settings = ph_settings, progress = print)
        name = f'timestream{j}'
        psds.save_timestream(dl.create_group(out_grp, name, 'analysis'), ts,
                             result, settings)
        report(ts, result, settings)
        out[name] = result
    print(f'Done: {out_grp.path}')
    return out

if __name__ == '__main__':
    main(parse_args())
