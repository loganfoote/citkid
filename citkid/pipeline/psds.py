"""
Noise spectra of calibrated timestreams, before and after common-mode
removal: the multi-tone bookkeeping around ``signal.psd``,
``xcal.reduced_params``, ``xcal.corr`` and ``signal.deglitch`` for
07_psds.py.

A timestream is one tone set of one run (``ts{n}``) over the whole run,
with a stream calibration (``ts_cal{c}``, made by 05_calibrate_streams.py)
that covers its tone set; ``list_timestreams`` lists every such pair over
every calibration in the store, so each chosen timestream can use a
different calibration. Its x comes from the raw streams in memory
(``stream_sources.apply_stream_calibration``), one tone write at a time
(the tones streamed together, over which the common modes are taken), and
only reduced results are saved (``save_timestream``). Per write:

1. Optionally, glitches (high-energy events, HEEs) are taken out of
   every tone and replaced with noise (``deglitch_peaks``, or
   ``deglitch_photons`` with the optimal filters of a photon result).
2. Every tone's Sxx (``signal.psd.get_psd``) is reduced to its mean
   around each frequency of a 1-3-10 grid that the run's length and
   sample period allow (``reduced_freqs``, ``xcal.reduced_params``), and
   binned (``signal.psd.bin_psd``): ``psd_summary``.
3. The common modes of the write are found and removed
   (``remove_common_modes``: the iterative algorithm of ``xcal.corr``),
   step 2 is repeated on the cleaned tones, and the common modes get
   binned spectra of their own.
"""
import json
import numpy as np
from .. import daq_layout as dl
from ..signal import deglitch as dg
from ..signal import photons as ph
from ..signal.psd import bin_psd, get_psd
from ..xcal.corr import calc_cm, remove_cm
from ..xcal.reduced_params import get_sxx_reduced
from ..signal.basic_filt import decimate
from ..zarr_util import to_jsonable
from . import photon_finder as pf
from .stream_sources import apply_stream_calibration, open_calibration

# Group prefix of the results at the store root
PREFIX = 'ts_psd'
DEGLITCH_OPTIONS = (None, 'peaks', 'photons')
PULSE_OPTIONS = ('hee', 'all')

################################################################################
################################## Timestreams #################################
################################################################################
def list_timestreams(root):
    """
    List the timestreams of every stream calibration in the store: every
    tone set of every run with streams, with each ``ts_cal{c}`` that has a
    DataSet for its calibration sweep (``photon_finder.list_timestreams``
    per calibration).

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    timestreams (list of dict): ``photon_finder.list_timestreams`` entries
        (``run``, ``run_index``, ``name``, ``tone_set``, ``dataset``,
        ``source``, ``start``, ``stop``, ``dt``, ``rows``, ``res_idxs``),
        with ``index`` over all calibrations, ``cal`` (the calibration's
        path, e.g. 'ts_cal0') and ``cal_index`` (c), in calibration then
        run order.
    datasets (dict): calibration path: ``stream_sources.open_calibration``
        output. Calibrations made by an earlier version (no DataSet per
        tone set) are left out.
    """
    timestreams, datasets = [], {}
    for c in dl.list_indices(root, 'ts_cal'):
        cal = root[dl.set_name(c, 'ts_cal')]
        try:
            ds = open_calibration(root, cal)
        except KeyError:
            continue
        datasets[cal.path] = ds
        for ts in pf.list_timestreams(root, ds):
            timestreams.append(dict(ts, index = len(timestreams),
                                    cal = cal.path, cal_index = c))
    return timestreams, datasets

def describe_timestreams(timestreams):
    """
    Describe the timestreams, one line each.

    Parameters:
    timestreams (list of dict): ``list_timestreams`` output.

    Returns:
    text (str): the description.
    """
    lines = ['Timestreams (index: run, name, tone set, duration, sample '
             'rate, tones, calibration):']
    for t in timestreams:
        n = t['stop'] - t['start']
        lines.append(
            f"  {t['index']}: {t['run']}, {t['name']!r}, set "
            f"{t['tone_set']}, {n * t['dt']:.4g} s at {1 / t['dt']:.4g} Hz, "
            f"{len(t['rows'])} tones, {t['cal']}/{t['dataset']}")
    if len(lines) == 1:
        lines.append('  none (run 05_calibrate_streams.py first)')
    return '\n'.join(lines)

def select_timestreams(timestreams, choice):
    """
    Pick timestreams by index, run name, or a description.

    Parameters:
    timestreams (list of dict): ``list_timestreams`` output.
    choice (list): each entry an index (int), a run name (str: every tone
        set of the latest run with that name, each with the latest
        calibration that covers it), or a dict with ``name`` or ``run``
        (the run index n), and optionally ``tone_sets`` (a list; default
        all) and ``cal`` (a ts_cal index; default the latest that covers
        each tone set).

    Returns:
    chosen (list of dict): the timestreams, in the order of ``choice``.

    Raises:
    KeyError: if a choice matches no timestream.
    """
    out = []
    for c in choice:
        if isinstance(c, (int, np.integer)):
            match = [t for t in timestreams if t['index'] == int(c)]
        else:
            spec = dict(name = c) if isinstance(c, str) else dict(c)
            if spec.get('run') is not None:
                match = [t for t in timestreams
                         if t['run_index'] == int(spec['run'])]
            else:
                named = [t for t in timestreams
                         if t['name'] == spec.get('name')]
                match = [t for t in named
                         if named and t['run_index'] == named[-1]['run_index']]
            if spec.get('tone_sets') is not None:
                keep = [int(s) for s in spec['tone_sets']]
                match = [t for t in match if t['tone_set'] in keep]
            if spec.get('cal') is not None:
                match = [t for t in match
                         if t['cal_index'] == int(spec['cal'])]
            else:
                latest = {}
                for t in match:
                    latest[t['tone_set']] = t
                match = [t for t in match
                         if t is latest[t['tone_set']]]
        if not match:
            raise KeyError(f'no timestream {c!r} (see the list)')
        out += match
    return out

################################################################################
################################ Reduced grid ##################################
################################################################################
def reduced_freqs(dt, nsamples, lo_mult = 10., hi_div = 10.):
    """
    Return the frequencies at which a timestream's PSD is reduced: every
    1 and 3 times a power of ten (0.1, 0.3, 1, 3, 10, ...) well inside the
    PSD's range.

    Parameters:
    dt (float): sample period, s.
    nsamples (int): samples.
    lo_mult (float): lowest frequency, times the PSD's lowest frequency
        (1 / duration). Default 10.
    hi_div (float): the highest frequency is half the Nyquist frequency
        divided by this. Default 10.

    Returns:
    freqs (np.array): the frequencies in Hz, ascending (empty if the range
        allows none).
    """
    lo = lo_mult / (nsamples * dt)
    hi = 0.25 / (dt * hi_div)
    if lo > hi:
        return np.zeros(0)
    k0, k1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    grid = np.array([float(f'{m}e{k}') for k in range(k0, k1 + 1)
                     for m in (1, 3)])
    return grid[(grid >= lo) & (grid <= hi)]

def union_freqs(timestreams, lo_mult = 10., hi_div = 10.):
    """
    Return the reduced frequencies of a group of timestreams: the union of
    each one's ``reduced_freqs``.

    Parameters:
    timestreams (list of dict): the timestreams.
    lo_mult, hi_div (float): see ``reduced_freqs``.

    Returns:
    freqs (np.array): the frequencies in Hz, ascending.
    """
    grids = [reduced_freqs(t['dt'], t['stop'] - t['start'], lo_mult,
                           hi_div) for t in timestreams]
    return np.unique(np.concatenate(grids)) if grids else np.zeros(0)

################################################################################
################################### Spectra ####################################
################################################################################
# The spectra are kept up to this fraction of the Nyquist frequency: the
# CRS decimation filters roll off above it (the correction grows large)
SPECTRUM_CUTOFF = 0.8

def dec_stage_from_dt(dt):
    """
    Return the CRS decimation stage of a sample period.

    Parameters:
    dt (float): sample period, s.

    Returns:
    stage (int or None): the stage (0-6), or None if the period is not a
        CRS one (e.g. another instrument's).
    """
    from rfmux.core.transferfunctions import decimation_to_sampling
    for stage in range(7):
        if abs(1 / dt - decimation_to_sampling(stage)) < 1e-3 / dt:
            return stage
    return None

def cic_correction(f, dt, cutoff = SPECTRUM_CUTOFF):
    """
    Return the factor that undoes the CRS decimation filters' droop in a
    PSD (``rfmux.core.transferfunctions.compensate_psd_for_cics``), NaN
    above the cutoff. The filters act on the streamed I and Q; the noise
    is small, so the calibrated timestreams (x, the loop components) are
    linear in it and their PSDs take the same factor.

    Parameters:
    f (np.array): frequencies, Hz.
    dt (float): sample period, s. A period that is not a CRS stage's
        gives a factor of 1 everywhere.
    cutoff (float): fraction of the Nyquist frequency kept. Default
        ``SPECTRUM_CUTOFF``.

    Returns:
    factor (np.array): per frequency (multiply the PSD by it).
    """
    f = np.asarray(f, dtype = float)
    stage = dec_stage_from_dt(dt)
    if stage is None or not len(f):
        return np.ones(len(f))
    from rfmux.core.transferfunctions import compensate_psd_for_cics
    _, factor = compensate_psd_for_cics(f, np.ones(len(f)),
                                        dec_stage = stage,
                                        spectrum_cutoff = 1.0)
    factor = np.asarray(factor, dtype = float)
    factor[f > cutoff * 0.5 / dt] = np.nan
    return factor

def psd_summary(x, dt, freqs, valid = None, nbins = 100, fmin = None):
    """
    Reduce and bin the PSD of each of several timestreams.

    Parameters:
    x (np.array, (N, T) or (T,)): timestreams. A row with a non-finite
        value gets NaN results.
    dt (float): sample period, s.
    freqs (np.array): frequencies to reduce the PSD at
        (``xcal.reduced_params.get_sxx_reduced``: the mean within 20%).
    valid (np.array, bool, or None): which of ``freqs`` to evaluate (the
        others are NaN). None (default): all of them. A frequency outside
        the PSD's range is NaN too.
    nbins (int): bins (``signal.psd.bin_psd``). Default 100.
    fmin (float or None): lowest binned frequency, Hz; below it the PSD is
        kept unbinned. None (default): 10 / duration.

    The CRS decimation filters' droop is undone (``cic_correction``) and
    the spectra end at ``SPECTRUM_CUTOFF`` of the Nyquist frequency (NaN
    above it).

    Returns:
    f_binned (np.array): binned frequencies, Hz (empty if no row is
        finite).
    binned (np.array, (N, nb)): binned PSD of each row.
    reduced (np.array, (N, len(freqs))): reduced PSD of each row.
    """
    x = np.atleast_2d(np.asarray(x, dtype = np.float64))
    n, T = x.shape
    freqs = np.asarray(freqs, dtype = np.float64)
    valid = np.ones(len(freqs), bool) if valid is None else \
        np.asarray(valid, dtype = bool)
    fmin = 10. / (T * dt) if fmin is None else float(fmin)
    f = np.fft.rfftfreq(T, d = dt)
    reduced = np.full((n, len(freqs)), np.nan)
    f_binned, binned = np.zeros(0), None
    corr = cic_correction(f, dt)
    for i in range(n):
        if not np.isfinite(x[i]).all():
            continue
        psd = get_psd(x[i], dt) * corr
        for j in np.flatnonzero(valid):
            try:
                reduced[i, j] = get_sxx_reduced(f, psd, freqs[j])
            except ValueError:
                pass
        fb, sb = bin_psd(f, [f, psd], nbins = nbins, fmin = fmin)
        if binned is None:
            f_binned, binned = fb, np.full((n, len(fb)), np.nan)
        binned[i] = sb
    if binned is None:
        binned = np.zeros((n, 0))
    return f_binned, binned, reduced

def remove_common_modes(x, dt, n_comp, n_iter, lowpass, highpass):
    """
    Find and remove the common modes of the tones streamed together
    (``xcal.corr.calc_cm`` and ``remove_cm``). Only the rows with finite
    data are used; the number of components is capped at one fewer than
    those rows (nothing is removed with fewer than two).

    Parameters:
    x (np.array, (N, T)): the tones' timestreams.
    dt (float): sample period, s.
    n_comp (int): common modes to remove.
    n_iter (int): iterations.
    lowpass (tuple): (frequency in Hz, order) of the lowpass filter
        applied to the common modes while iterating (for convergence; the
        removed modes are not filtered).
    highpass (tuple or None): (frequency in Hz, order) of the highpass
        filter applied to the timestreams while finding the weights (not
        to the removed modes). None: no highpass.

    Returns:
    out (dict): ``y`` ((N, T): x with the modes removed; NaN rows kept),
        ``a`` ((N, n_comp used): weights, NaN for unused rows), ``A``
        ((n_comp used, T): the common modes), ``sigma_iter`` ((n_iter,
        N): each row's residual sigma after each iteration), ``n_comp``
        (the components used), ``fit_rows`` (the rows used).
    """
    x = np.atleast_2d(np.asarray(x, dtype = np.float64))
    n, T = x.shape
    ok = np.isfinite(x).all(axis = 1)
    fit_rows = np.flatnonzero(ok)
    m = len(fit_rows)
    k = max(0, min(int(n_comp), m - 1))
    a = np.full((n, k), np.nan)
    sigma_iter = np.full((int(n_iter), n), np.nan)
    A = np.zeros((0, T))
    if k:
        # The rows used, without a copy when every row is finite
        xf = x if ok.all() else x[ok]
        a_fit, A, sig_iter, _ = calc_cm(xf, k, int(n_iter), dt,
                                        tuple(lowpass),
                                        None if highpass is None
                                        else tuple(highpass))
        yf = remove_cm(xf, a_fit, A, np.arange(m))
        del xf
        if ok.all():
            y = yf
        else:
            y = np.full((n, T), np.nan)
            y[ok] = yf
        del yf
        a[ok] = a_fit
        sigma_iter[:, ok] = sig_iter[:, :, 0]
    else:
        y = x.copy()
        y[~ok] = np.nan
    return dict(y = y, a = a, A = np.asarray(A), sigma_iter = sigma_iter,
                n_comp = k, fit_rows = fit_rows)

################################################################################
################################### Glitches ###################################
################################################################################
def _replace(x, idxs, pre, post):
    """
    Replace the samples around some glitches of one timestream with noise
    (``signal.deglitch.replace_glitches_with_gaussian_noise``).

    Parameters:
    x (np.array, (T,)): the timestream.
    idxs (np.array): glitch samples.
    pre, post (int): samples replaced before and after each.

    Returns:
    clean (np.array): the median-subtracted, cleaned timestream.
    frac (float): fraction of samples replaced.
    """
    clean, masked = dg.replace_glitches_with_gaussian_noise(
        x, np.asarray(idxs, dtype = np.int64), int(pre), int(post))
    return clean, len(masked) / len(x)

def deglitch_peaks(
    x, dt, sign = -1, nstd = 5., baseline_s = 1., distance = 1, width = None,
    nrounds = 2, pre = 1, post = 1
):
    """
    Find the glitches of each timestream by peak finding
    (``signal.deglitch.find_glitch_idxs``) and replace them with noise.

    Parameters:
    x (np.array, (N, T)): the timestreams. Non-finite rows are skipped.
    dt (float): sample period, s.
    sign (int): sign of a glitch in x (-1, the default: photon pulses
        lower the resonance frequency).
    nstd (float): peak threshold, standard deviations. Default 5.
    baseline_s (float): bin length of the baseline subtracted before peak
        finding, s (capped at the duration). Default 1.
    distance (int): fewest samples between peaks. Default 1.
    width (int, array-like or None): peak width requirement, samples.
        Default None: none.
    nrounds (int): rounds of peak finding. Default 2.
    pre, post (int): samples replaced before and after each glitch.
        Default 1 each.

    Returns:
    clean (np.array, (N, T)): the cleaned timestreams (median subtracted;
        sign as x).
    n_glitches (np.array, (N,)): glitches per row.
    frac (np.array, (N,)): fraction of samples replaced per row.
    idxs (list of np.array): the glitch samples of each row (empty for
        a row skipped), for ``reapply_glitches``.
    """
    x = np.atleast_2d(np.asarray(x, dtype = np.float64))
    n, T = x.shape
    clean = x.copy()
    n_glitches = np.zeros(n, dtype = np.int64)
    frac = np.zeros(n)
    found = [np.zeros(0, dtype = np.int64) for _ in range(n)]
    dtbin = min(float(baseline_s), T * dt)
    for i in range(n):
        if not np.isfinite(x[i]).all():
            continue
        ts = sign * x[i]
        idxs, _ = dg.find_glitch_idxs(ts, dt, dtbin, nstd, int(distance),
                                      width, int(nrounds), int(pre),
                                      int(post))
        c, frac[i] = _replace(ts, idxs, pre, post)
        clean[i] = sign * c
        n_glitches[i] = len(idxs)
        found[i] = np.asarray(idxs, dtype = np.int64)
    return clean, n_glitches, frac, found

def reapply_glitches(x, idxs, sign = -1, pre = 1, post = 1):
    """
    Replace the samples around glitches found earlier (``deglitch_peaks``)
    in a timestream computed again, e.g. with a changed calibration:
    the pulses sit at the same samples, so no search is needed.

    Parameters:
    x (np.array, (T,)): the timestream.
    idxs (array-like): the glitch samples.
    sign (int): sign of a glitch in x. Default -1.
    pre, post (int): samples replaced before and after each glitch.
        Default 1 each.

    Returns:
    clean (np.array, (T,)): the median-subtracted, cleaned timestream
        (sign as x).
    frac (float): fraction of samples replaced.
    """
    c, frac = _replace(sign * np.asarray(x, dtype = np.float64), idxs,
                       pre, post)
    return sign * c, frac

def load_photon_reference(ref_grp):
    """
    Read the optimal filters of a photon result (``photons/<name>``, made
    by 06a_find_photons.py), for finding the pulses of other timestreams.

    Parameters:
    ref_grp (zarr.Group): the result.

    Returns:
    filters (dict): res_idx: filter (``photon_finder.load_reference``).
    settings (DetectionSettings): the result's settings.
    """
    saved = dict(ref_grp.attrs['settings'])
    saved.pop('tau_lifetimes', None)
    settings = ph.DetectionSettings(**saved)
    return pf.load_reference(ref_grp, settings)[1], settings

def deglitch_photons(
    x, dt, res_idxs, filters, settings, pulses = 'hee', coinc_win = 2,
    min_mult = 3, pre = 1, post = 1
):
    """
    Find the pulses of the tones streamed together with given optimal
    filters (``signal.photons.detect_with_filter``) and replace the HEEs
    (or every pulse) with noise.

    Parameters:
    x (np.array, (N, T)): the tones' timestreams. Non-finite rows, and
        tones without a filter, are left as they are.
    dt (float): sample period, s.
    res_idxs (array-like, (N,)): the tones' resonator indices.
    filters (dict): res_idx: filter (``load_photon_reference``).
    settings (DetectionSettings): detection settings (``highpass_hz``,
        ``coinc_nsigma``, ...).
    pulses (str): 'hee' (default): replace the events seen by at least
        ``min_mult`` tones within +-``coinc_win`` samples
        (``signal.photons.hee_events``); 'all': every pulse candidate
        of every tone.
    coinc_win (int): coincidence window, samples. Default 2.
    min_mult (int): fewest tones of an HEE. Default 3.
    pre, post (int): samples replaced before and after each pulse.
        Default 1 each.

    Returns:
    clean (np.array, (N, T)): the cleaned timestreams (median subtracted
        where cleaned).
    n_glitches (np.array, (N,)): pulses replaced per row (-1 for a tone
        without a filter or whose detection failed).
    frac (np.array, (N,)): fraction of samples replaced per row.

    Raises:
    ValueError: if a filter is for another sample period, or ``pulses``
        is unknown.
    """
    if pulses not in PULSE_OPTIONS:
        raise ValueError(f'pulses must be one of {PULSE_OPTIONS}')
    x = np.atleast_2d(np.asarray(x, dtype = np.float64))
    n, T = x.shape
    clean = x.copy()
    n_glitches = np.full(n, -1, dtype = np.int64)
    frac = np.zeros(n)
    results = []
    for i, r in enumerate(res_idxs):
        ref = filters.get(int(r))
        if ref is None or not np.isfinite(x[i]).all():
            results.append(None)
            continue
        if abs(ref['dt'] - dt) > 1e-12 * dt:
            raise ValueError(f'the filter of resonator {int(r)} is for a '
                             f"sample period of {ref['dt']:.4g} s, not "
                             f'{dt:.4g} s')
        d = ph.detection_timestream(x[i], dt, settings.highpass_hz)
        try:
            results.append(ph.detect_with_filter(d, dt, ref, settings))
        except (ph.DetectionError, ValueError):
            results.append(None)
    good = [i for i, res in enumerate(results) if res is not None]
    idxs = {i: np.zeros(0, dtype = np.int64) for i in good}
    if pulses == 'all':
        for i in good:
            idxs[i] = np.asarray(results[i]['cand_idx'], dtype = np.int64)
    elif good:
        _, hits = ph.hee_events([results[i] for i in good], coinc_win,
                                   min_mult)
        for tone, idx in zip(hits['tone'], hits['idx']):
            idxs[good[tone]] = np.append(idxs[good[tone]], int(idx))
    for i in good:
        clean[i], frac[i] = _replace(x[i], np.sort(idxs[i]), pre, post)
        n_glitches[i] = len(idxs[i])
    return clean, n_glitches, frac

################################################################################
################################## Analysis ####################################
################################################################################
def analyze_timestream(
    root, ts, DS, f_reduced, settings, filters = None,
    photon_settings = None, progress = None, exclude = None
):
    """
    Compute one timestream's spectra, one tone write at a time: optional
    deglitching, reduced and binned Sxx, common-mode removal, and the same
    spectra after it.

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream (``list_timestreams``).
    DS (DataSet): its calibration DataSet.
    f_reduced (np.array): the group's reduced frequencies
        (``union_freqs``); this timestream's own (``reduced_freqs``) are
        evaluated, the others are NaN.
    settings (dict): ``nbins``, ``fmin_mult``, ``fmax_div``,
        ``cn_remove`` (bool),
        ``cn_n_comp``, ``cn_n_iter``, ``cn_lowpass_hz`` (None: a tenth of
        the Nyquist frequency), ``cn_highpass_hz`` (None: ``fmin_mult`` /
        duration; 0: none), ``cn_filter_order``, ``deglitch`` (None,
        'peaks' or 'photons'), ``glitch_pre_s``, ``glitch_post_s``,
        ``glitch_sign``, ``glitch_nstd``, ``glitch_baseline_s``,
        ``glitch_distance_s``, ``glitch_rounds``, ``glitch_pulses``,
        ``coinc_win``, ``hee_min_mult``, ``memory_gb``; optionally
        ``modes_rate_hz``: keep each write's common modes decimated to
        this sample rate (``modes_dec``, ``dt_dec`` per write), for
        ``nep.sweep`` to take out of a tone computed again.
    filters (dict or None): res_idx: optimal filter, for ``deglitch`` =
        'photons' (``load_photon_reference``). Default None.
    photon_settings (DetectionSettings or None): with ``filters``.
        Default None.
    progress (callable or None): called with a message per write. Default
        None.
    exclude (dict or None): res_idx: reason, tones left out (marked bad):
        their timestreams are set to NaN before anything is computed, so
        they neither get spectra nor enter the common-mode fit, and their
        ``status`` says why. Default None.

    Returns:
    result (dict): ``f_binned``, ``valid_reduced`` (mask of
        ``f_reduced``), ``fmin``, ``lowpass_hz``, ``highpass_hz``,
        ``write_idx`` (per tone: its write's index in the source's
        writes, -1 if none), ``status`` (per tone: '' or a message),
        ``n_glitches``, ``glitch_frac``, ``sxx_reduced``, ``sxx_binned``,
        the spectra of the loop components ``spar_binned`` (parallel to
        the loop, r theta) and ``sper_binned`` (perpendicular, A), in normalized
        units squared per Hz, and ``sfactor_reduced`` (10 log10 of their
        ratio at ``f_reduced``, dB), and with ``cn_remove``:
        ``sxx_reduced_cn``, ``sxx_binned_cn``,
        ``glitch_idxs`` (per tone: the glitch samples found with
        ``deglitch`` = 'peaks', else empty),
        ``cn_weights`` ((ntones, n_comp), NaN-padded), ``writes`` (list
        per write: ``index``, ``name``, ``rows``, ``n_comp``,
        ``sxx_binned`` of the common modes, ``sigma_iter``).

    Raises:
    MemoryError: if a write's timestreams would exceed ``memory_gb``.
    ValueError: if ``deglitch`` is 'photons' without ``filters``.
    """
    s = settings
    dt, T = float(ts['dt']), int(ts['stop'] - ts['start'])
    duration = T * dt
    n = len(ts['rows'])
    nf = len(f_reduced)
    own = reduced_freqs(dt, T, s['fmin_mult'], s['fmax_div'])
    valid = np.isin(f_reduced, own)
    fmin = s['fmin_mult'] / duration
    lowpass_hz = 0.05 / dt if s.get('cn_lowpass_hz') is None else \
        float(s['cn_lowpass_hz'])
    highpass_hz = fmin if s.get('cn_highpass_hz') is None else \
        float(s['cn_highpass_hz'])
    order = int(s.get('cn_filter_order', 2))
    highpass = None if highpass_hz <= 0 else (highpass_hz, order)
    pre = max(1, int(round(s['glitch_pre_s'] / dt)))
    post = max(1, int(round(s['glitch_post_s'] / dt)))
    if s['deglitch'] == 'photons' and filters is None:
        raise ValueError("deglitch = 'photons' needs the filters of a "
                         'photon result')
    out = dict(
        f_binned = np.zeros(0), valid_reduced = valid, fmin = fmin,
        lowpass_hz = lowpass_hz, highpass_hz = highpass_hz,
        write_idx = np.full(n, -1, dtype = np.int64),
        status = np.array(['no stream'] * n, dtype = object),
        n_glitches = np.zeros(n, dtype = np.int64), glitch_frac = np.zeros(n),
        sxx_reduced = np.full((n, nf), np.nan), sxx_binned = None,
        spar_binned = None, sper_binned = None,
        sfactor_reduced = np.full((n, nf), np.nan), writes = [],
        glitch_idxs = [np.zeros(0, dtype = np.int64) for _ in range(n)])
    if s['cn_remove']:
        k_max = max(0, min(int(s['cn_n_comp']), n - 1))
        out.update(sxx_reduced_cn = np.full((n, nf), np.nan),
                   sxx_binned_cn = None,
                   cn_weights = np.full((n, k_max), np.nan))
    bins = dict(nbins = int(s['nbins']), fmin = fmin)
    # The per-tone arrays follow ts['rows'] (the calibrated tones), not
    # the DataSet's row numbers
    pos = {int(r): i for i, r in enumerate(ts['rows'])}
    for w, name, rows in pf.tone_writes(root, ts):
        rows = np.asarray(rows, dtype = np.int64)
        idx = np.array([pos[int(r)] for r in rows], dtype = np.int64)
        # About 4 float64 arrays of the write's timestreams at the peak
        # (the calibration's three outputs plus a stream chunk, or the
        # common-mode removal's working copies)
        need = 4 * len(rows) * T * 8
        if need > float(s['memory_gb']) * 1e9:
            raise MemoryError(
                f"{ts['run']} write {name}: {len(rows)} tones of {T} "
                f'samples need about {need / 1e9:.1f} GB; raise MEMORY_GB '
                'or analyze a shorter run')
        if progress is not None:
            progress(f"{ts['run']} ({ts['name']}), tone set "
                     f"{ts['tone_set']}, {name}: {len(rows)} tones")
        cal = apply_stream_calibration(DS, root, ts['source'], rows = rows,
                                       outputs = ('xt', 'thetat', 'At'))
        x = cal['xt']
        ok = np.isfinite(x).all(axis = 1)
        # The loop components: r theta parallel to the loop, A
        # perpendicular to it (no
        # deglitching, no common-mode removal)
        radius = np.asarray(DS.circ_radius[rows.tolist()], dtype = float)
        par = cal['thetat']
        par *= radius[:, None]
        perp = cal['At']
        del cal
        out['write_idx'][idx] = w
        out['status'][idx] = np.where(
            ok, '', 'the calibrated timestream is not finite')
        if exclude:
            for k, i in enumerate(idx):
                reason = exclude.get(int(ts['res_idxs'][i]))
                if reason is not None:
                    x[k] = par[k] = perp[k] = np.nan
                    out['status'][i] = f'marked bad: {reason}'
        fb, sb, sr_par = psd_summary(par, dt, f_reduced, valid, **bins)
        _fill(out, 'spar_binned', idx, fb, sb, n)
        fb, sb, sr_per = psd_summary(perp, dt, f_reduced, valid, **bins)
        _fill(out, 'sper_binned', idx, fb, sb, n)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out['sfactor_reduced'][idx] = 10 * np.log10(sr_par / sr_per)
        del par, perp
        if s['deglitch'] == 'peaks':
            x, ng, fr, found = deglitch_peaks(
                x, dt, sign = int(s['glitch_sign']),
                nstd = float(s['glitch_nstd']),
                baseline_s = float(s['glitch_baseline_s']),
                distance = max(1, int(round(s['glitch_distance_s'] / dt))),
                nrounds = int(s['glitch_rounds']), pre = pre, post = post)
            out['n_glitches'][idx], out['glitch_frac'][idx] = ng, fr
            for i, g in zip(idx, found):
                out['glitch_idxs'][i] = g
        elif s['deglitch'] == 'photons':
            x, ng, fr = deglitch_photons(
                x, dt, ts['res_idxs'][idx], filters, photon_settings,
                pulses = s['glitch_pulses'], coinc_win = int(s['coinc_win']),
                min_mult = int(s['hee_min_mult']), pre = pre, post = post)
            out['n_glitches'][idx], out['glitch_frac'][idx] = ng, fr
        fb, sb, sr = psd_summary(x, dt, f_reduced, valid, **bins)
        _fill(out, 'sxx_binned', idx, fb, sb, n)
        out['sxx_reduced'][idx] = sr
        if not s['cn_remove']:
            del x
            continue
        cm = remove_common_modes(x, dt, s['cn_n_comp'], s['cn_n_iter'],
                                 (lowpass_hz, order), highpass)
        del x
        fb, sb, sr = psd_summary(cm['y'], dt, f_reduced, valid, **bins)
        _fill(out, 'sxx_binned_cn', idx, fb, sb, n)
        out['sxx_reduced_cn'][idx] = sr
        out['cn_weights'][idx, :cm['n_comp']] = cm['a']
        if cm['n_comp']:
            fbA, sA, _ = psd_summary(cm['A'], dt, np.zeros(0), **bins)
        else:
            fbA, sA = np.zeros(0), np.zeros((0, 0))
        write = dict(
            index = w, name = name, rows = rows, n_comp = cm['n_comp'],
            f_binned = fbA, sxx_binned = sA, sigma_iter = cm['sigma_iter'])
        if s.get('modes_rate_hz') and cm['n_comp']:
            # The modes at a low sample rate, to take out of a tone
            # computed again later without finding them again
            A, dt_dec = decimate(cm['A'], dt, float(s['modes_rate_hz']))
            write.update(modes_dec = A.astype(np.float32), dt_dec = dt_dec)
        out['writes'].append(write)
    for key in ('sxx_binned', 'sxx_binned_cn', 'spar_binned', 'sper_binned'):
        if key in out and out[key] is None:
            out[key] = np.zeros((n, 0))
    return out

def _fill(out, key, rows, f_binned, binned, n):
    """
    Put a write's binned spectra into the timestream's array, making it
    on the first write (every write bins the same frequencies).

    Parameters:
    out (dict): the result.
    key (str): array name.
    rows (np.array): the write's rows.
    f_binned (np.array): the binned frequencies.
    binned (np.array): the write's binned spectra.
    n (int): tones in the timestream.
    """
    if out[key] is None:
        out[key] = np.full((n, len(f_binned)), np.nan)
        if len(f_binned):
            out['f_binned'] = f_binned
    if binned.shape[1] == out[key].shape[1]:
        out[key][rows] = binned

################################################################################
################################### Saving #####################################
################################################################################
def save_timestream(grp, ts, result, settings, bad = None, limits = None):
    """
    Save one timestream's spectra.

    Layout of ``grp``: attrs describe the timestream (``run``, ``name``,
    ``tone_set``, ``calibration``, ``dataset``, ``source``, ``writes``,
    ``start``, ``stop``, ``dt``, ``duration_s``), the binning (``fmin``,
    ``nbins``), the common-mode settings used (``cn_removed``,
    ``cn_n_comp``, ``cn_n_iter``, ``cn_lowpass_hz``, ``cn_highpass_hz``,
    ``cn_filter_order``) and ``deglitch``; arrays ``f_binned`` (Hz),
    ``valid_reduced`` (which of the group's ``f_reduced`` this
    timestream covers), and per tone (in the order of ``rows``):
    ``rows``, ``res_idxs``, ``write_idx``, ``status`` ('' if analyzed),
    ``bad``, ``bad_reason`` (marked in the overview window),
    ``n_glitches``, ``glitch_frac``, ``sxx_reduced`` (1/Hz, at
    ``f_reduced``), ``sxx_binned`` (1/Hz, at ``f_binned``),
    ``spar_binned``, ``sper_binned`` (the loop components parallel and
    perpendicular to the loop, normalized units squared per Hz),
    ``sfactor_reduced``
    (10 log10 of their ratio at ``f_reduced``, dB); with
    common-mode removal also ``sxx_reduced_cn``, ``sxx_binned_cn``,
    ``cn_weights`` ((ntones, n_comp)), and ``common_modes/write{k}/``
    per tone write (attrs ``name``, ``n_comp``; arrays ``rows``,
    ``f_binned``, ``sxx_binned`` ((n_comp, nb): the modes' spectra),
    ``sigma_iter`` ((n_iter, tones): residual sigma per iteration)).

    Parameters:
    grp (zarr.Group): the (new, empty) group.
    ts (dict): the timestream.
    result (dict): ``analyze_timestream`` output.
    settings (dict): the settings used (``analyze_timestream``).
    bad (dict or None): res_idx: reason, the tones marked bad in the
        overview window (arrays ``bad``, ``bad_reason``). Default None.
    limits (dict or None): the overview's review limits (attribute
        ``limits``). Default None.
    """
    T = int(ts['stop'] - ts['start'])
    bad = {} if bad is None else {int(k): str(v) for k, v in bad.items()}
    grp.attrs.update(to_jsonable(dict(
        dataset = ts['dataset'], run = ts['run'], name = ts['name'],
        tone_set = ts['tone_set'], calibration = ts['cal'],
        source = ts['source']['group'],
        writes = list(ts['source']['writes']), start = ts['start'],
        stop = ts['stop'], dt = ts['dt'], duration_s = T * ts['dt'],
        fmin = result['fmin'], nbins = settings['nbins'],
        deglitch = settings['deglitch'], cn_removed = settings['cn_remove'],
        cn_n_comp = settings['cn_n_comp'], cn_n_iter = settings['cn_n_iter'],
        cn_lowpass_hz = result['lowpass_hz'],
        cn_highpass_hz = result['highpass_hz'],
        cn_filter_order = settings.get('cn_filter_order', 2),
        limits = {} if limits is None else dict(limits))))
    dl.write_array(grp, 'f_binned', result['f_binned'])
    dl.write_array(grp, 'bad', np.array(
        [int(r) in bad for r in ts['res_idxs']], dtype = bool))
    dl.write_array(grp, 'bad_reason', np.array(
        [bad.get(int(r), '') for r in ts['res_idxs']]))
    dl.write_array(grp, 'valid_reduced', result['valid_reduced'])
    dl.write_array(grp, 'rows', np.asarray(ts['rows'], dtype = np.int64))
    dl.write_array(grp, 'res_idxs', np.asarray(ts['res_idxs'],
                                               dtype = np.int64))
    dl.write_array(grp, 'write_idx', result['write_idx'])
    dl.write_array(grp, 'status', np.asarray(result['status'], dtype = str))
    for key in ('n_glitches', 'glitch_frac', 'sxx_reduced', 'sxx_binned',
                'spar_binned', 'sper_binned', 'sfactor_reduced'):
        dl.write_array(grp, key, result[key])
    if not settings['cn_remove']:
        return
    for key in ('sxx_reduced_cn', 'sxx_binned_cn', 'cn_weights'):
        dl.write_array(grp, key, result[key])
    cm = dl.create_group(grp, 'common_modes', 'analysis')
    for w in result['writes']:
        g = dl.create_group(cm, f"write{w['index']}", 'analysis',
                            attrs = dict(name = w['name'],
                                         n_comp = w['n_comp']))
        dl.write_array(g, 'rows', np.asarray(w['rows'], dtype = np.int64))
        for key in ('f_binned', 'sxx_binned', 'sigma_iter'):
            dl.write_array(g, key, w[key])

def load_timestream(grp):
    """
    Read one saved timestream's spectra (``save_timestream``).

    Parameters:
    grp (zarr.Group): a ``ts_psd{p}/timestream{j}`` group.

    Returns:
    out (dict): the group's attributes, every array (by name), and
        ``common_modes`` (list of dict per write: its attributes and
        arrays), if any.
    """
    out = dict(grp.attrs)
    for name, arr in grp.arrays():
        out[name] = arr[:]
    if 'common_modes' in grp:
        cm = grp['common_modes']
        out['common_modes'] = [
            dict(cm[w].attrs, **{k: v[:] for k, v in cm[w].arrays()})
            for w in sorted(cm.group_keys(), key = lambda t: int(t[5:]))]
    return out

def find_group(root, p):
    """
    Find a result of 07_psds.py by index or name.

    Parameters:
    root (zarr.Group): the store root.
    p (int or str): the result's index (``ts_psd{p}``), or its name
        (attribute ``name``: the latest result with that name).

    Returns:
    grp (zarr.Group): the result.

    Raises:
    KeyError: if there is no such result.
    """
    if isinstance(p, str):
        named = [q for q in dl.list_indices(root, PREFIX)
                 if root[dl.set_name(q, PREFIX)].attrs.get('name') == p]
        if not named:
            raise KeyError(f'the store has no {PREFIX} result named {p!r}')
        p = named[-1]
    return root[dl.set_name(int(p), PREFIX)]

def matching_group(root, name, timestreams):
    """
    Find the latest result of 07_psds.py that a run would make again.

    Parameters:
    root (zarr.Group): the store root.
    name (str or None): the run's name (attribute ``name``). None: the
        run has no name, and a result matches when it has none and the
        same ``timestreams``.
    timestreams (list): the run's ``TIMESTREAMS``.

    Returns:
    grp (zarr.Group or None): the latest matching result, or None.
    """
    want = json.loads(json.dumps(dl.to_jsonable(timestreams)))
    for q in reversed(dl.list_indices(root, PREFIX)):
        g = root[dl.set_name(q, PREFIX)]
        if name is not None:
            if g.attrs.get('name') == name:
                return g
        elif g.attrs.get('name') is None and \
                g.attrs.get('timestreams') == want:
            return g
    return None

def saved_review(grp):
    """
    Read the overview window's review back from a result of 07_psds.py.

    Parameters:
    grp (zarr.Group): the result (``ts_psd{p}``).

    Returns:
    review (dict): timestream group name ('timestream{j}'): tuple
        (bad, limits): the tones marked bad (res_idx: reason) and the
        review limits (as ``save_timestream`` took them).
    """
    out = {}
    for t in grp.group_keys():
        if not t.startswith('timestream'):
            continue
        g = grp[t]
        res_idxs = g['res_idxs'][:].tolist()
        flags = g['bad'][:].tolist() if 'bad' in g else [False] * len(res_idxs)
        reasons = g['bad_reason'][:].tolist() if 'bad_reason' in g else \
            [''] * len(res_idxs)
        bad = {int(r): str(s) for r, b, s in zip(res_idxs, flags, reasons)
               if b}
        out[t] = (bad, dict(g.attrs.get('limits', {})))
    return out

def describe_group(grp):
    """
    Describe a result of 07_psds.py: its timestreams, review and notes.

    Parameters:
    grp (zarr.Group): the result.

    Returns:
    text (str): one line per timestream, and the notes if any.
    """
    s = grp.attrs.get('settings', {})
    cn = f"{s.get('cn_n_comp')} modes" if s.get('cn_remove') else 'kept'
    lines = [f"{grp.path}: {grp.attrs.get('name')!r}; deglitch "
             f"{s.get('deglitch')}; common modes {cn}"]
    for t, (bad, limits) in sorted(saved_review(grp).items()):
        a = grp[t].attrs
        lines.append(f"  {t}: {a['run']} set {a['tone_set']} "
                     f"({a['calibration']}), "
                     f"{grp[t]['res_idxs'].shape[0]} tones, "
                     f'{len(bad)} marked bad, limits at {len(limits)} '
                     'frequencies')
    if grp.attrs.get('notes'):
        lines.append(f"  notes: {grp.attrs['notes']}")
    return '\n'.join(lines)

def load_group(root, p):
    """
    Read a result of 07_psds.py: its reduced frequencies and every
    timestream's spectra.

    Parameters:
    root (zarr.Group): the store root.
    p (int or str): the result's index (``ts_psd{p}``) or name
        (``find_group``).

    Returns:
    out (dict): the group's attributes, ``f_reduced`` (Hz), and
        ``timestreams`` (list of ``load_timestream`` outputs, in order).
    """
    grp = find_group(root, p)
    names = sorted((t for t in grp.group_keys() if t.startswith('timestream')),
                   key = lambda t: int(t[len('timestream'):]))
    return dict(grp.attrs, f_reduced = grp['f_reduced'][:],
                timestreams = [load_timestream(grp[t]) for t in names])

def describe_results(root):
    """
    Describe the previous results of 07_psds.py, one line each.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    text (str): the description.
    """
    lines = ['Previous results (ts_psd{p}: name, timestreams, '
             'deglitching, common modes):']
    for p in dl.list_indices(root, PREFIX):
        g = root[dl.set_name(p, PREFIX)]
        parts = [f"{g[t].attrs['run']} set {g[t].attrs['tone_set']} "
                 f"({g[t].attrs['calibration']})"
                 for t in sorted(g.group_keys())
                 if t.startswith('timestream')]
        s = g.attrs.get('settings', {})
        cn = f"{s.get('cn_n_comp')} modes" if s.get('cn_remove') else 'kept'
        lines.append(f"  {PREFIX}{p}: {g.attrs.get('name')!r}, "
                     f"{', '.join(parts)}; deglitch {s.get('deglitch')}; "
                     f'common modes {cn}')
    if len(lines) == 1:
        lines.append('  none')
    return '\n'.join(lines)
