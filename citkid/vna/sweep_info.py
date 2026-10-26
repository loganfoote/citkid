"""
Describe a full VNA sweep: prepare it for fitting, measure its quality
(noise, gain ripple, phase jumps, sampling), find where the resonances are
and how they are arranged in banks, and write it all up as a short text
report (``describe_full_sweep``; ``run_sweep_info`` prints it for a
fitted sweep saved in a zarr group).
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks
from ..res.funcs import product_s21
from ..res.multifit import block_average

# Note thresholds: roughly the worst ~10% of the real sweeps get each note
# (5% for 'very noisy'). Medians there: S/N 45 dB, ripple 11 dB p-p, sharp
# features 5 dB p-p, gain range 29 dB.
NOTE_THRESHOLDS = dict(
    noise_db_very = 22., noise_db = 27.,      # baseline S/N per quadrature
    ripple_db = 20., features_db = 14.,       # gain ripple / sharp features
    gain_range_db = 55.,                      # total gain variation
    phase_jump_frac = 0.01,                   # point-to-point phase jumps
    pts_per_lw = 5.,                          # undersampled below this
    frac_poor_fit = 0.15,                     # many poor fits
    frac_overlap = 0.5,                       # very crowded
)
# Note distorted line shapes when at least max(5, 10%) of the resonances
# have a narrow core
NARROW_MIN = 5
NARROW_FRAC = 0.1

################################################################################
############################## Sweep preparation ###############################
################################################################################
def noise_correlation_length(
    z, lags = (1, 2, 4, 8, 16, 32, 64), ratio = 1.15
):
    """
    Estimate over how many samples the noise of a sweep is correlated.

    Uses second differences at lag L, z[i+L] - 2 z[i] + z[i-L], which
    remove a smooth gain to first order. For white noise their spread does
    not depend on L; for noise correlated over ~l samples it grows with L
    until L ~ l. The median absolute value is robust to the resonances.

    Parameters:
    z (np.array): complex data on a (roughly) uniform or fractional grid.
    lags (tuple): lags tried, ascending. Default powers of 2 up to 64.
    ratio (float): the spread has saturated once doubling the lag grows it
        by less than this factor. Default 1.15.

    Returns:
    length (int): the smallest lag at which the spread saturates (1 for
        white noise, or if it never saturates within ``lags``: then the
        growth comes from signal, not correlated noise).
    """
    z = np.asarray(z, dtype = np.complex128)
    s = []
    for L in lags:
        if 2 * L >= len(z):
            break
        d2 = z[2 * L:] - 2 * z[L:-L] + z[:-2 * L]
        s.append(np.median(np.abs(d2)))
    for k in range(len(s) - 1):
        if s[k + 1] < ratio * s[k]:
            return int(lags[k])
    return 1

def prepare_full_sweep(f, z, decorrelate = True, max_step = 20e3):
    """
    Prepare a full VNA sweep for fitting.

    Sorts by frequency and averages repeated frequencies. Then, if
    ``decorrelate``, block averages to the noise correlation length
    (``noise_correlation_length``): sweeps stepped much finer than the IF
    bandwidth have noise correlated over several points, which breaks the
    fitter's white-noise error estimates and costs time. Nothing is
    averaged if the averaged step would exceed ``max_step`` (the measured
    correlation is then more likely signal). Typical sweeps are unchanged.

    Parameters:
    f (np.array): frequencies in Hz (any order; repeats are averaged).
    z (np.array): complex S21 data.
    decorrelate (bool): If True (default), average to the noise correlation
        length.
    max_step (float): largest allowed averaged step in Hz. Default 20 kHz.

    Returns:
    f (np.array): frequencies in Hz, ascending, no repeats.
    z (np.array): complex S21 data.
    corr_length (int): noise correlation length in points (1: white), as
        measured before averaging.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    f, inv, counts = np.unique(f, return_inverse = True, return_counts = True)
    z = (np.bincount(inv, z.real, len(f))
         + 1j * np.bincount(inv, z.imag, len(f))) / counts
    k = noise_correlation_length(z)
    if decorrelate and k > 1 and k * np.median(np.diff(f)) <= max_step:
        f, z = block_average(f, z, k)
    return f, z, k

def resample_fractional(f, z, df_frac = None):
    """
    Resample a sweep onto a grid with constant fractional step df/f.

    On this grid a resonance's width in samples depends only on Qr
    (lw / step = 1 / (Qr * df_frac)), not on where it sits in the band.
    Complex data are interpolated linearly; grid points inside gaps of the
    input (a step more than 3x the local typical spacing) are flagged.

    Parameters:
    f (np.array): frequencies in Hz, ascending, no repeats.
    z (np.array): complex S21 data.
    df_frac (float or None): fractional step. If None, the 25th percentile
        of diff(f) / f, so the grid is at least as fine as most of the data.

    Returns:
    fg (np.array): grid frequencies in Hz.
    zg (np.array): interpolated complex data.
    valid (np.array, bool): False where the grid point falls in a gap of
        the input data.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    df = np.diff(f)
    if df_frac is None:
        df_frac = np.quantile(df / f[:-1], 0.25)
    n = int(np.log(f[-1] / f[0]) / np.log1p(df_frac)) + 1
    fg = f[0] * (1 + df_frac) ** np.arange(n)
    zg = np.interp(fg, f, z.real) + 1j * np.interp(fg, f, z.imag)
    local = median_filter(df, min(len(df), 1001) | 1, mode = 'nearest')
    gap = df > 3 * local
    i = np.clip(np.searchsorted(f, fg), 1, len(f) - 1)
    valid = ~gap[i - 1]
    return fg, zg, valid

def has_phase(z, tol = 1e-9, frac = 0.99):
    """
    Check whether a sweep has phase information.

    Some sweeps are saved without phase: the q channel is all zeros, so z
    is real. Such data cannot be fit with the complex resonator model.

    Parameters:
    z (np.array): complex S21 data.
    tol (float): |Im z| below ``tol`` x max |z| counts as zero. Default
        1e-9.
    frac (float): the sweep has no phase if at least this fraction of
        points has zero imaginary part. Default 0.99.

    Returns:
    ok (bool): False if the phase is missing.
    """
    z = np.asarray(z)
    if len(z) == 0:
        return False
    scale = max(float(np.max(np.abs(z))), 1e-300)
    return bool(np.mean(np.abs(z.imag) <= tol * scale) < frac)

def _iq_circularity(z, n_dips = 30):
    """
    Measure whether the IQ data at the deepest dips traces circles.

    A resonance traces a circle in the complex plane; a sweep whose phase
    was lost or replaced (e.g. |S21| times a constant phase) traces lines.
    For each of the ``n_dips`` deepest dips (fractional depth below a
    running baseline), the IQ points within +-(full width at half depth)
    give sqrt(smallest / largest eigenvalue) of their covariance: ~0.5-0.8
    for resonance circles, ~0 for a line.

    Parameters:
    z (np.array): complex S21 data.
    n_dips (int): number of dips used. Default 30.

    Returns:
    circularity (float): median over the dips (NaN if no dips).
    n (int): number of dips measured.
    """
    a = np.abs(z)
    w = max(51, len(z) // 2000)
    base = uniform_filter1d(maximum_filter1d(a, w), w)
    depth = a / np.maximum(base, 1e-300)
    pk, pr = find_peaks(-depth, prominence = 0.3, distance = w)
    if len(pk) == 0:
        return np.nan, 0
    pk = pk[np.argsort(pr['prominences'])[::-1][:n_dips]]
    out = []
    for p in pk:
        # Extent where the dip is below half its depth
        thr = 0.5 * (1 + depth[p])
        lo = p
        while lo > 0 and depth[lo] < thr and p - lo < 5 * w:
            lo -= 1
        hi = p
        while hi < len(z) - 1 and depth[hi] < thr and hi - p < 5 * w:
            hi += 1
        span = max(hi - lo, 6)
        lo, hi = max(0, p - span), min(len(z), p + span + 1)
        zz = z[lo:hi] / base[lo:hi]
        if len(zz) < 8:
            continue
        x = np.c_[zz.real - zz.real.mean(), zz.imag - zz.imag.mean()]
        ev = np.linalg.eigvalsh(x.T @ x)
        out.append(np.sqrt(max(ev[0], 0.) / max(ev[1], 1e-300)))
    return (float(np.median(out)) if out else np.nan), len(out)

def phase_problem(z, max_circularity = 0.25, max_span_rad = 20.):
    """
    Check whether a sweep's phase can be used for fitting.

    'missing': the q channel is zero (``has_phase``). 'fake': q is not
    zero but the phase carries no resonance information: the IQ data at
    the deepest dips traces lines, not circles, and the unwrapped phase
    barely changes over the sweep (< ``max_span_rad``; a sweep with cable
    delay winds by hundreds of rad). Both are required, so a sweep with
    its cable delay removed is not mistaken for one without phase. On 117
    real sweeps: circularity >= 0.29 and span >= 170 rad with phase;
    0.05-0.16 and ~1.2 rad for the two with unusable phase.

    Parameters:
    z (np.array): complex S21 data, sorted by frequency.
    max_circularity (float): circularity below this counts as lines.
        Default 0.25.
    max_span_rad (float): phase span below this counts as no winding.
        Default 20.

    Returns:
    problem (str or None): None (phase usable), 'missing' or 'fake'.
    info (dict): ``circularity``, ``n_dips`` and ``phase_span_rad`` (NaN
        when not computed).
    """
    info = dict(circularity = np.nan, n_dips = 0, phase_span_rad = np.nan)
    if not has_phase(z):
        return 'missing', info
    stride = max(1, len(z) // 200000)
    ph = np.unwrap(np.angle(z[::stride]))
    info['phase_span_rad'] = float(ph.max() - ph.min())
    if info['phase_span_rad'] >= max_span_rad:
        return None, info
    c, n = _iq_circularity(z)
    info['circularity'], info['n_dips'] = c, n
    if n >= 5 and c < max_circularity:
        return 'fake', info
    return None, info

################################################################################
############################ Sweep quality metrics #############################
################################################################################
def _robust_sigma(z):
    """
    Estimate the noise per quadrature from second differences.

    Parameters:
    z (np.array): complex data (smooth apart from noise and rare features).

    Returns:
    sigma (float): noise std per quadrature.
    """
    d2 = z[2:] - 2 * z[1:-1] + z[:-2]
    # |d2| of circular Gaussian noise: median = sqrt(6) s sqrt(2 ln 2)
    return float(np.median(np.abs(d2))
                 / (np.sqrt(6) * np.sqrt(2 * np.log(2))))

def narrow_core(f, mag, fr, Qr, amp, phi, min_excess_db = 6., max_points = 3):
    """
    Check whether a fitted resonance has a core the model cannot follow.

    Signature: within +-1 fitted linewidth, the gain-removed data dips at
    least ``min_excess_db`` below the deepest point of the fitted model, on
    at most ``max_points`` points. The fit then follows only the broad part
    (a resonance narrower than the frequency step, or a distorted line
    shape), so its Qr, Qi and Qc are unreliable. When many resonances of a
    sweep do this, the sweep itself is distorted.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    mag (np.array): |S21| with the gain removed, linear.
    fr, Qr, amp, phi (float): the fitted resonance.
    min_excess_db (float): how much deeper the data must dip. Default 6.
    max_points (int): at most this many points that deep. Default 3.

    Returns:
    flag (bool): True if it has a narrow core (False if it cannot be
        judged).
    """
    if not all(np.isfinite(v) for v in (fr, Qr, amp, phi)) or Qr <= 0:
        return False
    lw = fr / Qr
    i0, i1 = np.searchsorted(f, [fr - lw, fr + lw])
    if i1 - i0 < 3:
        return False
    data = 20 * np.log10(np.maximum(mag[i0:i1], 1e-300))
    model = 20 * np.log10(np.maximum(np.abs(product_s21(
        f[i0:i1], [fr], [Qr], [amp], [phi])), 1e-300))
    deep = data < model.min() - min_excess_db
    return bool(deep.any() and deep.sum() <= max_points)

def sweep_metrics(f, z, gain, res = None, corr_length = 1):
    """
    Measure the quality of a full VNA sweep.

    Parameters:
    f (np.array): frequencies in Hz, ascending (as fit).
    z (np.array): complex S21 data, same.
    gain (np.array): smooth complex gain at ``f`` (``vna.gain``).
    res (pd.DataFrame or None): fit table (``fr``, ``Qr``, ``snr``,
        ``flags``) for the fit-based metrics. Default None.
    corr_length (int): noise correlation length of the raw sweep in points
        (``prepare_full_sweep``). Default 1.

    Returns:
    m (dict): ``noise_db`` (baseline S/N per quadrature, dB: -20 log10 of
        the noise of z / gain), ``gain_range_db`` (1-99% range of |gain| in
        dB), ``ripple_db`` (peak-to-peak of the gain on scales of ~20-300
        MHz, dB), ``features_db`` (peak-to-peak of the gain on scales below
        ~20 MHz, dB), ``phase_jump_frac`` (share of steps whose phase change
        differs from the typical one, e.g. the cable delay, by > 0.5 rad,
        away from resonances), ``corr_length``, and, with ``res``:
        ``pts_per_lw`` (median), ``snr_median``, ``frac_poor_fit``,
        ``frac_overlap`` (shares flagged).
    """
    f = np.asarray(f, dtype = np.float64)
    rm = z / gain
    m = dict(noise_db = float(-20 * np.log10(max(_robust_sigma(rm), 1e-12))),
             corr_length = int(corr_length))
    gdb = 20 * np.log10(np.maximum(np.abs(gain), 1e-12))
    m['gain_range_db'] = float(np.subtract(*np.percentile(gdb, [99, 1])))
    # Gain on a uniform grid; smooth at ~20 MHz and ~300 MHz scales
    span = f[-1] - f[0]
    fu = np.linspace(f[0], f[-1], 6000)
    gu = np.interp(fu, f, gdb)
    step = fu[1] - fu[0]
    w20 = max(3, int(min(20e6, span / 20) / step))
    w300 = max(w20 + 2, int(min(300e6, span / 3) / step))
    mid = uniform_filter1d(gu, w20, mode = 'nearest')
    trend = uniform_filter1d(gu, w300, mode = 'nearest')
    p2p = lambda x: float(np.subtract(*np.percentile(x, [99, 1])))
    # Moving averages are biased near the ends (rolloffs): skip half a
    # window there
    e300, e20 = w300 // 2, w20 // 2
    m['ripple_db'] = p2p((mid - trend)[e300:len(gu) - e300]
                         if len(gu) > 2 * e300 + 10 else mid - trend)
    m['features_db'] = p2p((gu - mid)[e20:len(gu) - e20])
    # Phase jumps between consecutive points, away from resonances: the
    # step-to-step phase change minus its typical value (cable delay)
    dph = np.angle(z[1:] / z[:-1])
    dph = np.abs(np.angle(np.exp(1j * (dph - np.median(dph)))))
    away = np.ones(len(dph), dtype = bool)
    if res is not None and len(res):
        lw = (res.fr / res.Qr).values
        for c, w in zip(res.fr.values, lw):
            if np.isfinite(c) and np.isfinite(w):
                i0, i1 = np.searchsorted(f, [c - 3 * w, c + 3 * w])
                away[max(i0 - 1, 0):i1] = False
    m['phase_jump_frac'] = float(np.mean(dph[away] > 0.5)) if away.any() \
        else float('nan')
    if res is not None and len(res):
        ok = np.isfinite(res.Qr.values)
        step_at = np.interp(res.fr.values[ok], f[1:], np.diff(f))
        m['pts_per_lw'] = float(np.median((res.fr / res.Qr).values[ok]
                                          / step_at)) if ok.any() else np.nan
        m['snr_median'] = float(np.nanmedian(res.snr)) if 'snr' in res \
            else np.nan
        flags = res['flags'].astype(str) if 'flags' in res else None
        m['frac_poor_fit'] = (float(flags.str.contains('poor fit').mean())
                              if flags is not None else np.nan)
        m['frac_overlap'] = (float(flags.str.contains('overlap').mean())
                             if flags is not None else np.nan)
    return m

def sweep_notes(m, n_narrow = 0, n_res = 0, thresholds = None):
    """
    Turn sweep metrics into short caveats.

    Parameters:
    m (dict): ``sweep_metrics`` output.
    n_narrow (int): resonances with a narrow core (``narrow_core``).
        Default 0.
    n_res (int): number of resonances. Default 0.
    thresholds (dict or None): overrides of ``NOTE_THRESHOLDS``. Default
        None.

    Returns:
    notes (list of str): caveats (empty if none), e.g. 'very noisy (S/N
        18 dB)', 'large gain ripple (26.1 dB p-p)'.
    """
    t = dict(NOTE_THRESHOLDS, **(thresholds or {}))
    notes = []
    g = lambda k: m.get(k, np.nan)
    if np.isfinite(g('noise_db')):
        if g('noise_db') < t['noise_db_very']:
            notes.append(f'very noisy (S/N {g("noise_db"):.0f} dB)')
        elif g('noise_db') < t['noise_db']:
            notes.append(f'noisy (S/N {g("noise_db"):.0f} dB)')
    if g('ripple_db') > t['ripple_db']:
        notes.append(f'large gain ripple ({g("ripple_db"):.1f} dB p-p)')
    if g('features_db') > t['features_db']:
        notes.append(f'sharp gain features ({g("features_db"):.1f} dB p-p)')
    if g('gain_range_db') > t['gain_range_db']:
        notes.append(f'gain varies by {g("gain_range_db"):.0f} dB')
    if g('phase_jump_frac') > t['phase_jump_frac']:
        notes.append(f'phase jumps between points '
                     f'({100 * g("phase_jump_frac"):.1f}% of steps)')
    if m.get('corr_length', 1) > 1:
        notes.append(f'oversampled: noise correlated over '
                     f'{m["corr_length"]} points')
    if g('pts_per_lw') < t['pts_per_lw']:
        notes.append(f'undersampled ({g("pts_per_lw"):.1f} points per '
                     'linewidth)')
    if g('frac_poor_fit') > t['frac_poor_fit']:
        notes.append(f'many poor fits ({100 * g("frac_poor_fit"):.0f}%)')
    if g('frac_overlap') > t['frac_overlap']:
        notes.append(f'very crowded ({100 * g("frac_overlap"):.0f}% '
                     'overlapping)')
    if n_narrow and n_narrow >= max(NARROW_MIN, NARROW_FRAC * n_res):
        notes.append(f'sweep distorted: {n_narrow} dips have a narrow core '
                     'the resonator model cannot fit')
    return notes

################################################################################
############################ Resonance arrangement #############################
################################################################################
def _fmt_mhz(v):
    """
    Format a frequency in Hz as whole MHz with thousands separators.

    Parameters:
    v (float): frequency in Hz.

    Returns:
    text (str): e.g. '4,568'.
    """
    return f'{v / 1e6:,.0f}'

def _fmt_around(v):
    """
    Format a frequency for 'around ...' (GHz above 1 GHz, else MHz).

    Parameters:
    v (float): frequency in Hz.

    Returns:
    text (str): e.g. '2.0 GHz' or '450 MHz'.
    """
    return f'{v / 1e9:.1f} GHz' if v >= 1e9 else f'{v / 1e6:.0f} MHz'

def _fmt_q(v):
    """
    Format a quality factor with two significant digits, in k / M.

    Parameters:
    v (float): quality factor.

    Returns:
    text (str): e.g. '60k', '1.2M', '950'; 'inf' or 'n/a'.
    """
    if not np.isfinite(v):
        return 'inf' if v > 0 else 'n/a'
    r = float(f'{v:.2g}')
    if abs(r) >= 1e6:
        return f'{r / 1e6:g}M'
    if abs(r) >= 1e3:
        return f'{r / 1e3:g}k'
    return f'{r:g}'

def describe_range(
    fr, gap_frac = 0.05, gap_factor = 10., band_frac = 0.05, band_min = 5,
    tight_frac = 0.05, gap_note_factor = 25.
):
    """
    Describe where the resonances are, concisely.

    The sorted frequencies are split into bands at gaps larger than both
    ``gap_frac`` of the frequency and ``gap_factor`` x the local typical
    spacing (median of the ~10 spacings on each side), so that random gaps
    in a sparse band do not split it while isolated groups do. Bands with
    at least ``band_frac`` of the resonances (and ``band_min``) form the
    main range; the rest are stragglers, reported as '+ n low' / '+ n high'
    (with their range) or '+ n around X' when tightly grouped. Gaps inside
    the main range are mentioned.

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    gap_frac (float): minimum fractional gap that splits bands. Default
        0.05.
    gap_factor (float): the gap must also exceed this many local typical
        spacings. Default 10.
    band_frac (float): minimum share of the resonances in a main band.
        Default 0.05.
    band_min (int): minimum count in a main band. Default 5.
    tight_frac (float): straggler groups spanning less than this fraction
        of their center frequency are reported as 'around X'. Default 0.05.
    gap_note_factor (float): a gap between two main bands is kept (and
        mentioned) only if it exceeds this many local spacings; otherwise
        the bands are merged. Default 25.

    Returns:
    text (str): e.g. '200-4,568 MHz + 10 low (120-180 MHz)', '158-1,405
        MHz + 2 around 2.0 GHz', or 'none'.
    info (dict): ``fmin``, ``fmax`` (main range, Hz), ``n_main``,
        ``n_low``, ``n_high``, ``n_between``, ``gaps`` (list of (lo, hi)
        in Hz inside the main range).
    """
    fr = np.sort(np.asarray(fr, dtype = np.float64)[np.isfinite(fr)])
    n = len(fr)
    if n == 0:
        return 'none', dict(fmin = np.nan, fmax = np.nan, n_main = 0,
                            n_low = 0, n_high = 0, n_between = 0, gaps = [])
    if n < 2 * band_min:
        return (f'{_fmt_mhz(fr[0])}-{_fmt_mhz(fr[-1])} MHz',
                dict(fmin = fr[0], fmax = fr[-1], n_main = n, n_low = 0,
                     n_high = 0, n_between = 0, gaps = []))
    gaps = np.diff(fr)
    # Local typical spacing: median of the ~10 gaps on each side
    local = np.array([np.median(np.r_[gaps[max(0, k - 10):k],
                                      gaps[k + 1:k + 11]])
                      for k in range(len(gaps))])
    split = np.flatnonzero((gaps > gap_frac * fr[1:])
                           & (gaps > gap_factor * local))
    is_big = lambda b: len(b) >= max(band_min, band_frac * n)
    # Two large bands stay one band unless the gap between them is much
    # larger (gap_note_factor) than the local spacing
    while True:
        edges = np.r_[0, split + 1, n]
        bands = [fr[a:b] for a, b in zip(edges[:-1], edges[1:])]
        weak = [j for j, s in enumerate(split)
                if is_big(bands[j]) and is_big(bands[j + 1])
                and gaps[s] < gap_note_factor * local[s]]
        if not weak:
            break
        split = np.delete(split, weak[0])
    big = [is_big(b) for b in bands]
    if not any(big):  # no dominant band: report everything
        big = [True] * len(bands)
    first = big.index(True)
    last = len(big) - 1 - big[::-1].index(True)
    main = bands[first:last + 1]
    lo, hi = main[0][0], main[-1][-1]
    parts = [f'{_fmt_mhz(lo)}-{_fmt_mhz(hi)} MHz']
    # Gaps inside the main range (between main bands)
    main_gaps = []
    for k in range(first, last):
        if big[k] and any(big[k + 1:last + 1]):
            nxt = next(j for j in range(k + 1, last + 1) if big[j])
            main_gaps.append((bands[k][-1], bands[nxt][0]))
    main_gaps = [g for g in main_gaps if g[1] - g[0] > 0]
    if main_gaps:
        parts[0] += ' (gap ' + ', '.join(
            f'{_fmt_mhz(a)}-{_fmt_mhz(b)}' for a, b in main_gaps) + ' MHz)'
    low = np.concatenate(bands[:first]) if first > 0 else np.array([])
    high = (np.concatenate(bands[last + 1:]) if last < len(bands) - 1
            else np.array([]))
    between = [b for k, b in enumerate(bands[first:last + 1], first)
               if not big[k]]

    def straggle(x, word):
        if len(x) == 0:
            return None
        c = np.median(x)
        if np.ptp(x) < tight_frac * c:
            return f'+ {len(x)} around {_fmt_around(c)}'
        return (f'+ {len(x)} {word} ({_fmt_mhz(x.min())}-'
                f'{_fmt_mhz(x.max())} MHz)')

    for b in between:
        parts.append(f'+ {len(b)} around {_fmt_around(np.median(b))}')
    for x, w in ((low, 'low'), (high, 'high')):
        s = straggle(x, w)
        if s:
            parts.append(s)
    n_between = int(sum(len(b) for b in between))
    return ' '.join(parts), dict(
        fmin = float(lo), fmax = float(hi),
        n_main = int(sum(len(b) for b in main)) - n_between,
        n_low = len(low), n_high = len(high), n_between = n_between,
        gaps = [(float(a), float(b)) for a, b in main_gaps])

def find_banks(
    fr, n_banks = 16, min_bank_frac = 0.25, sep_factor = 2.,
    search_frac = 0.3, gap_factor = 3., prior_width = 0.5
):
    """
    Find a device's banks: groups of resonances repeating at a nearly
    constant spacing in log frequency.

    The bank spacing P and alignment come from the gaps between banks: the
    2 ``n_banks`` widest spacings between neighboring resonances (and above
    ``gap_factor`` times the median) are weighted by log(size) and their
    midpoints' periodicity is measured (weighted Rayleigh periodogram, P
    between 1/24 and 1/3 of the span), times a broad log-normal prior that
    the bulk of the resonances spans ``n_banks`` banks. Each predicted
    boundary is moved to the widest empty stretch within
    +-``search_frac`` P. Slots with fewer than ``min_bank_frac`` of the
    typical bank count are empty. With more than ``n_banks`` banks, the
    least separated neighbors are merged. A boundary separates its banks if
    its empty stretch is at least ``sep_factor`` times the widest spacing
    found in equal windows at the bank centers; isolated resonances next to
    separating boundaries are counted as between banks.

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    n_banks (int): expected (maximum) number of banks. Default 16.
    min_bank_frac (float): smallest bank, as a fraction of the typical
        bank. Default 0.25.
    sep_factor (float): see above. Default 2.
    search_frac (float): boundary search half-width in periods. Default
        0.3.
    gap_factor (float): spacings above this times the median are gap
        candidates for the period. Default 3.
    prior_width (float): width (in ln P) of the prior on the spacing.
        Default 0.5.

    Returns:
    label (np.array, int): bank index per resonance (order of ``fr``; -1
        outside the banks or between separated banks).
    banks (list of tuple): (log10 f start, log10 f end, count) per bank,
        ascending, empty slots inside left out.
    info (dict): ``period`` (log10 f), ``comb_strength`` (0-1: how periodic
        the gaps are), ``slot`` (slot index per bank, so gaps of empty
        slots are missing banks), ``frac_separated``, ``n_between``,
        ``gap_spacings``, ``gap_vs_center``.
    """
    fr = np.asarray(fr, dtype = float)
    x = np.log10(fr)
    o = np.argsort(x)
    xs = x[o]
    label = np.full(len(x), -1)
    info = dict(period = np.nan, comb_strength = 0., slot = [],
                frac_separated = 0., n_between = 0)
    if len(xs) < 2 * n_banks:
        return label, [], info
    span = xs[-1] - xs[0]
    dx = np.diff(xs)
    mids = 0.5 * (xs[1:] + xs[:-1])
    med = np.median(dx[dx > 0]) if (dx > 0).any() else 0.
    # Gap candidates: the 2 n_banks widest spacings (well above the median)
    big = np.zeros(len(dx), bool)
    big[np.argsort(dx)[-2 * n_banks:]] = True
    big &= dx > gap_factor * med
    if med <= 0 or big.sum() < 3:
        return label, [], info
    w = np.log(np.minimum(dx[big] / med, 50.))

    def comb(periods):
        return (w[None, :] * np.exp(2j * np.pi * mids[big][None, :]
                                    / periods[:, None])).sum(axis = 1) \
            / w.sum()

    periods = np.geomspace(span / 24, span / 3, 4000)
    r = np.abs(comb(periods))
    # Prior: the bulk of the resonances spans about n_banks banks
    bulk = np.diff(np.percentile(xs, [0.5, 99.5]))[0]
    prior = np.exp(-0.5 * (np.log(periods * n_banks / bulk)
                           / prior_width) ** 2)
    k = int(np.argmax(r * prior))
    # Refine around the chosen spacing
    fine = np.linspace(periods[max(k - 2, 0)], periods[min(k + 2, 3999)],
                       400)
    sf = comb(fine)
    kf = int(np.argmax(np.abs(sf)))
    P, s0 = fine[kf], sf[kf]
    info['period'] = float(P)
    info['comb_strength'] = float(np.abs(s0))
    b0 = np.angle(s0) / (2 * np.pi) * P
    # Predicted boundaries over the data, each moved to the widest empty
    # stretch nearby
    first = np.floor((xs[0] - b0) / P) - 1
    pred = b0 + P * np.arange(first, first + span / P + 4)
    bounds, widths = [], []
    for b in pred:
        m = np.abs(mids - b) < search_frac * P
        if m.any():
            j = np.flatnonzero(m)[np.argmax(dx[m])]
            bounds.append(mids[j])
            widths.append(dx[j])
        else:
            bounds.append(b)
            widths.append(2 * search_frac * P)

    def slots_of(bounds):
        counts = np.diff(np.searchsorted(xs, bounds))
        occ = counts[counts > 0]
        typical = np.median(occ) if len(occ) else 0
        return typical, counts >= max(3, min_bank_frac * typical)

    typical, real = slots_of(bounds)
    # Too many banks: merge the least separated neighbors
    while real.sum() > n_banks:
        cand = [b for b in range(1, len(bounds) - 1)
                if real[b - 1] and real[b]]
        if not cand:
            break
        b = min(cand, key = lambda b: widths[b])
        bounds.pop(b)
        widths.pop(b)
        typical, real = slots_of(bounds)
    if not real.any():
        return label, [], info
    bounds, widths = np.array(bounds), np.array(widths)
    slots = np.flatnonzero(real)
    banks = []
    for i, sl in enumerate(slots):
        a, b = np.searchsorted(xs, bounds[sl:sl + 2])
        banks.append((xs[a], xs[b - 1]))
        label[o[a:b]] = i
    info['slot'] = [int(sl - slots[0]) for sl in slots]
    # Separation at the boundaries between neighboring banks: the
    # boundary's empty stretch against the widest spacing in an equal
    # window at the banks' centers
    d_avg = P / typical
    centers = 0.5 * (bounds[slots] + bounds[slots + 1])
    ref = [dx[np.abs(mids - c) < search_frac * P].max(initial = 0.)
           for c in centers]
    ref = np.median(ref) if len(ref) else np.inf
    inner = [sl + 1 for sl, nx in zip(slots[:-1], slots[1:]) if nx == sl + 1]
    sep = [widths[b] >= sep_factor * ref for b in inner]
    info['frac_separated'] = float(np.mean(sep)) if sep else 0.
    info['gap_spacings'] = (float(np.median(widths[inner]) / d_avg)
                            if inner else 0.)
    info['gap_vs_center'] = (float(np.median(widths[inner]) / ref)
                             if inner and ref > 0 else 0.)
    # Between banks: isolated resonances (both neighbors far) next to
    # separating boundaries
    iso = np.r_[np.inf, np.minimum(dx[:-1], dx[1:]), np.inf]
    between = np.zeros(len(xs), bool)
    for b, ok in zip(inner, sep):
        if ok:
            between |= ((np.abs(xs - bounds[b]) < search_frac * P)
                        & (iso >= 3 * d_avg))
    label[o[between]] = -1
    info['n_between'] = int(between.sum())
    banks = [(lo, hi, int((label == i).sum())) for i, (lo, hi) in
             enumerate(banks)]
    return label, banks, info

def _groups(fr, label = None, banks = None, n_groups = 10):
    """
    Group resonances for the frequency check: by bank where banks were
    found, else into equal-count stretches of frequency.

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    label (np.array, int, or None): bank per resonance (``find_banks``;
        -1 outside the banks). Default None.
    banks (list or None): ``find_banks`` banks. Default None.
    n_groups (int): number of frequency stretches without banks. Default
        10.

    Returns:
    group (np.array, int): group per resonance (-1: none).
    """
    if banks is not None and len(banks) >= 3:
        return np.asarray(label)
    o = np.argsort(fr)
    group = np.empty(len(fr), int)
    group[o] = np.arange(len(fr)) * n_groups // max(len(fr), 1)
    return group

def q_vs_frequency(
    fr, q, label = None, banks = None, min_n = 10, p_max = 1e-6,
    min_ratio = 2., min_eta2 = 0.25, min_rho = 0.8
):
    """
    Check whether a quality factor varies with frequency across the band.

    The resonances are grouped by bank (or, without banks, into 10
    equal-count stretches of frequency). Only clear, large variation
    counts: the group medians differ (Kruskal-Wallis p < ``p_max``), the
    groups explain at least ``min_eta2`` of the variance of log Q, and the
    largest / smallest group median is at least ``min_ratio``. It is a
    trend if the group medians rise or fall with frequency (Spearman |rho|
    >= ``min_rho``).

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    q (np.array): quality factors (NaN: unreliable).
    label (np.array, int, or None): bank per resonance (``find_banks``).
        Default None.
    banks (list or None): ``find_banks`` banks. Default None (no banks).
    min_n (int): groups with fewer finite values are left out. Default 10.
    p_max (float): significance threshold. Default 1e-6.
    min_ratio (float): smallest notable ratio of group medians. Default 2.
    min_eta2 (float): smallest notable fraction of variance between groups.
        Default 0.25.
    min_rho (float): smallest |Spearman rho| for a trend. Default 0.8.

    Returns:
    result (dict or None): None if Q does not clearly vary with frequency;
        else ``trend`` ('rises', 'falls', or None: varies without a
        trend), ``low`` and ``high`` (Q of the lowest- and
        highest-frequency group for a trend, else the smallest and largest
        group median).
    """
    from scipy.stats import kruskal, spearmanr
    fr = np.asarray(fr, dtype = float)
    q = np.asarray(q, dtype = float)
    group = _groups(fr, label, banks)
    ok = np.isfinite(q) & (q > 0) & (group >= 0)
    lq = np.log10(np.where(ok, q, 1.))
    ids = [k for k in np.unique(group[ok])
           if (ok & (group == k)).sum() >= min_n]
    if len(ids) < 3:
        return None
    vals = [lq[ok & (group == k)] for k in ids]
    fmid = np.array([np.median(fr[ok & (group == k)]) for k in ids])
    med = np.array([np.median(v) for v in vals])
    allv = np.concatenate(vals)
    eta2 = (sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in vals)
            / max(((allv - allv.mean()) ** 2).sum(), 1e-300))
    if not (kruskal(*vals).pvalue < p_max and eta2 >= min_eta2
            and 10 ** (med.max() - med.min()) >= min_ratio):
        return None
    rho = spearmanr(fmid, med).correlation
    o = np.argsort(fmid)
    if abs(rho) >= min_rho:
        return dict(trend = 'rises' if rho > 0 else 'falls',
                    low = float(10 ** med[o[0]]),
                    high = float(10 ** med[o[-1]]))
    return dict(trend = None, low = float(10 ** med.min()),
                high = float(10 ** med.max()))

def q_vs_bank_position(
    fr, q, label, banks, n_bins = 8, min_n = 10, min_banks = 4,
    min_eta2 = 0.1, min_consistency = 0.8, min_rho = 0.8
):
    """
    Check whether a quality factor varies with position within the banks,
    with the same pattern in every bank.

    Each resonance's log Q relative to its bank's median is binned by its
    position in the bank (0 at the bank's low-frequency end, 1 at its high
    end). A pattern counts if the position explains at least ``min_eta2``
    of the variance of these values and repeats from bank to bank: the
    profiles of the even- and the odd-numbered banks correlate by at least
    ``min_consistency``. On real devices with separated banks, Qi patterns
    explained 13-80% with consistency 0.8-1.0 on about half of them; Qc
    never more than 8%.

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    q (np.array): quality factors (NaN: unreliable).
    label (np.array, int): bank per resonance (``find_banks``).
    banks (list): ``find_banks`` banks.
    n_bins (int): position bins. Default 8.
    min_n (int): banks with fewer finite values are left out. Default 10.
    min_banks (int): fewest banks needed. Default 4.
    min_eta2 (float): smallest notable fraction of variance explained by
        position. Default 0.1.
    min_consistency (float): smallest correlation between the even- and
        odd-bank profiles. Default 0.8.
    min_rho (float): smallest |Spearman rho| of the profile for a trend.
        Default 0.8.

    Returns:
    result (dict or None): None if there is no clear pattern; else
        ``trend``: 'rises' or 'falls' toward the bank's high-frequency end
        (``low``, ``high``: typical Q at the bank's low and high end);
        'peaks' or 'dips' in the middle of the bank (the extreme is inside
        the bank and both ends are at least halfway back from it; ``low``,
        ``high``: typical Q at the ends and in the middle); or None,
        another repeating shape (``low``, ``high``: the profile's smallest
        and largest). Also ``eta2`` and ``consistency``.
    """
    from scipy.stats import spearmanr
    fr = np.asarray(fr, dtype = float)
    q = np.asarray(q, dtype = float)
    x = np.log10(fr)
    ok = np.isfinite(q) & (q > 0) & (label >= 0)
    lq = np.log10(np.where(ok, q, 1.))
    r = np.full(len(q), np.nan)
    u = np.full(len(q), np.nan)
    use = []
    for k, (lo, hi, _) in enumerate(banks):
        m = ok & (label == k)
        if m.sum() < min_n or hi <= lo:
            continue
        r[m] = lq[m] - np.median(lq[m])
        u[m] = (x[m] - lo) / (hi - lo)
        use.append(k)
    if len(use) < min_banks:
        return None
    m = np.isfinite(r)
    b = np.clip((np.nan_to_num(u) * n_bins).astype(int), 0, n_bins - 1)

    def profile(sel):
        return np.array([np.median(r[sel & (b == i)])
                         if (sel & (b == i)).any() else np.nan
                         for i in range(n_bins)])

    p = profile(m)
    fin = np.isfinite(p)
    eta2 = (sum((m & (b == i)).sum() * p[i] ** 2 for i in range(n_bins)
                if fin[i]) / max(np.sum(r[m] ** 2), 1e-300))
    pe = profile(m & np.isin(label, use[0::2]))
    po = profile(m & np.isin(label, use[1::2]))
    both = np.isfinite(pe) & np.isfinite(po)
    cons = (np.corrcoef(pe[both], po[both])[0, 1] if both.sum() >= 4
            else np.nan)
    if not (eta2 >= min_eta2 and cons >= min_consistency):
        return None
    base = np.median(lq[m] - r[m])  # typical bank median
    rho = spearmanr(np.flatnonzero(fin), p[fin]).correlation
    out = dict(eta2 = float(eta2), consistency = float(cons))
    pf = p[fin]
    ends = pf[[0, -1]]
    if abs(rho) >= min_rho:
        return dict(out, trend = 'rises' if rho > 0 else 'falls',
                    low = float(10 ** (base + ends[0])),
                    high = float(10 ** (base + ends[1])))
    half = 0.5 * (pf.max() - pf.min())
    for trend, k, sign in (('peaks', np.argmax(pf), 1),
                           ('dips', np.argmin(pf), -1)):
        # Extreme inside the bank, both ends at least halfway back
        if 0 < k < len(pf) - 1 and np.all(sign * (pf[k] - ends) >= half):
            return dict(out, trend = trend,
                        low = float(10 ** (base + ends.mean())),
                        high = float(10 ** (base + pf[k])))
    return dict(out, trend = None, low = float(10 ** (base + pf.min())),
                high = float(10 ** (base + pf.max())))

def describe_banks(
    fr, f_lo = None, f_hi = None, n_banks = 16, min_resonances = 600,
    min_comb = 0.15, min_gap_vs_center = 2.2, min_gap_spacings = 9.
):
    """
    Describe how a device's resonances are arranged in banks.

    Only for high-yield devices (at least ``min_resonances``); otherwise
    ''. Banks from ``find_banks``. The banks are separated if the typical
    boundary gap is clearly emptier than the banks' centers (at least
    ``min_gap_vs_center`` times the widest spacing there) or at least
    ``min_gap_spacings`` average spacings wide; otherwise (or if the gaps
    are not periodic, comb strength < ``min_comb``) they overlap: touching
    and overlapping banks cannot be told apart from frequencies alone, and
    the bank count is then not reliable. For separated banks: the number
    found, resonances between banks, banks missing inside the range, and
    banks missing at an end ('cut off' if the data ends within 0.75 bank
    spacing of the outermost bank there). Thresholds set on 84 real devices
    with >= 600 resonances.

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    f_lo, f_hi (float or None): the swept frequency range in Hz (for
        cut-off banks). None: unknown ('missing' instead of 'cut off').
    n_banks (int): expected number of banks. Default 16.
    min_resonances (int): smaller devices get no bank description.
        Default 600.
    min_comb (float): minimum periodicity of the gaps (0-1). Default 0.15.
    min_gap_vs_center (float): see above. Default 2.2.
    min_gap_spacings (float): see above. Default 9.

    Returns:
    text (str): e.g. '16 banks, separated', '14 banks, mostly separated
        (12 between); 2 cut off at the high end', 'banks overlap'; '' below
        ``min_resonances``.
    info (dict): ``n_found``, ``n_missing_inside``, ``n_cut_low``,
        ``n_cut_high``, ``n_between``, ``overlapping``, ``label`` (bank
        per resonance, order of ``fr``, -1 outside) and ``banks``
        (``find_banks``; both None unless the banks are separated), and
        ``find_banks`` diagnostics.
    """
    fr = np.asarray(fr, dtype = float)
    keep = np.isfinite(fr)
    fr = fr[keep]
    info = dict(n_found = 0, n_missing_inside = 0, n_cut_low = 0,
                n_cut_high = 0, n_between = 0, overlapping = False,
                label = None, banks = None)
    if len(fr) < min_resonances:
        return '', info
    label, banks, fb = find_banks(fr, n_banks)
    info.update(fb)
    separated = (fb['comb_strength'] >= min_comb and len(banks) >= 3
                 and (fb.get('gap_vs_center', 0) >= min_gap_vs_center
                      or fb.get('gap_spacings', 0) >= min_gap_spacings))
    if not separated:
        info['overlapping'] = True
        return 'banks overlap', info
    info['n_found'] = len(banks)
    info['n_between'] = fb['n_between']
    lab = np.full(len(keep), -1)
    lab[keep] = label
    info['label'], info['banks'] = lab, banks
    parts = [f'{len(banks)} banks, '
             + ('separated' if fb['n_between'] <= max(2, 0.005 * len(fr))
                else f'mostly separated ({fb["n_between"]} between)')]
    # Missing inside: empty comb slots between banks
    slot = np.array(fb['slot'])
    for (lo0, hi0, _), (lo1, hi1, _), s0, s1 in zip(
            banks[:-1], banks[1:], slot[:-1], slot[1:]):
        k = int(min(s1 - s0 - 1,
                    n_banks - len(banks) - info['n_missing_inside']))
        if k > 0:
            info['n_missing_inside'] += k
            parts.append(f'{k} missing near '
                         f'{_fmt_around(10 ** (0.5 * (hi0 + lo1)))}')
    short = n_banks - len(banks) - info['n_missing_inside']
    if short > 0:
        P = fb['period']
        low = high = None
        if f_lo is not None and f_hi is not None:
            low = banks[0][0] - np.log10(f_lo) < 0.75 * P
            high = np.log10(f_hi) - banks[-1][1] < 0.75 * P
        if low and not high:
            info['n_cut_low'] = short
            parts.append(f'{short} cut off at the low end')
        elif high and not low:
            info['n_cut_high'] = short
            parts.append(f'{short} cut off at the high end')
        elif low and high:
            parts.append(f'{short} cut off')
        else:
            parts.append(f'{short} missing')
    return '; '.join(parts), info

################################################################################
#################################### Report ####################################
################################################################################
def describe_full_sweep(f, res, metrics, notes, n_banks = 16):
    """
    Write a short text description of a fitted full VNA sweep.

    Parameters:
    f (np.array): the fitted frequencies in Hz, ascending.
    res (pd.DataFrame): fit table (``vna.full_sweep.fit_full_sweep`` with
        flags): ``fr``, ``Qr``, ``Qi``, ``Qc``, ``fit_status``, ``flags``.
    metrics (dict): ``sweep_metrics`` output.
    notes (list of str): ``sweep_notes`` output.
    n_banks (int): expected number of banks. Default 16.

    Returns:
    text (str): several lines, e.g.
        Full VNA sweep: 1,005 resonances, 480-1,971 MHz
        Fits: 962 ok, 12 Stage A only, 31 failed (fr only)
        Flags: 60 overlap, 20 poor fit, 3 distorted
        Banks: 16 banks, separated
        Quality factors (reliable fits): Qi~130k (within each bank
            rises from 90k at the low end to 180k at the high end),
            Qc~42k (falls with frequency, 60k to 30k), Qr~31k
        Sampling: 22 points per linewidth; median SNR 101
        Noise and gain: S/N 39.7 dB; gain range 17.5 dB, ripple 12.3 dB
            p-p, sharp features 2.1 dB p-p; phase jumps 0.03% of steps
        Notes: none
    info (dict): ``range`` and ``banks`` (the ``describe_range`` and
        ``describe_banks`` info dicts), and ``q``: per quality factor, the
        ``q_vs_frequency`` and ``q_vs_bank_position`` results.
    """
    status = res.fit_status.astype(str)
    flags = res['flags'].astype(str)
    range_text, range_info = describe_range(res.fr.values)
    reliable = ~status.str.startswith('failed') \
        & ~flags.str.contains('poor fit|distorted|bifurcated')
    Qi = np.where(reliable, res.Qi.values.astype(float), np.nan)
    Qc = np.where(reliable & ~flags.str.contains('overlap'),
                  res.Qc.values.astype(float), np.nan)
    Qr = np.where(reliable & ~flags.str.contains('overlap'),
                  res.Qr.values.astype(float), np.nan)
    bank_text, bank_info = describe_banks(res.fr.values, f[0], f[-1],
                                          n_banks)
    label, banks = bank_info['label'], bank_info['banks']
    lines = [f'Full VNA sweep: {len(res):,} resonances, {range_text}']
    n_ok = int((status == 'ok').sum())
    n_a = int((status == 'stage_a').sum())
    n_fail = int(status.str.startswith('failed').sum())
    lines.append(f'Fits: {n_ok:,} ok'
                 + (f', {n_a:,} Stage A only' if n_a else '')
                 + (f', {n_fail:,} failed (fr only)' if n_fail else ''))
    counts = [(k, int(flags.str.contains(k).sum()))
              for k in ('overlap', 'poor fit', 'distorted', 'bifurcated')]
    counts = [f'{n:,} {k}' for k, n in counts if n]
    lines.append('Flags: ' + (', '.join(counts) if counts else 'none'))
    if bank_text:
        lines.append(f'Banks: {bank_text}')

    fr = res.fr.values.astype(float)
    q_info = {}

    def q_part(name, q):
        if not np.isfinite(q).any():
            return f'{name}: none reliable'
        vf = q_vs_frequency(fr, q, label, banks)
        vp = (q_vs_bank_position(fr, q, label, banks)
              if banks is not None else None)
        q_info[name] = dict(frequency = vf, bank_position = vp)
        parts = []
        if vf is not None:
            parts.append(
                f'{vf["trend"]} with frequency, {_fmt_q(vf["low"])} to '
                f'{_fmt_q(vf["high"])}' if vf['trend'] else
                f'varies across the band, {_fmt_q(vf["low"])}-'
                f'{_fmt_q(vf["high"])}')
        if vp is not None:
            lo, hi, tr = _fmt_q(vp['low']), _fmt_q(vp['high']), vp['trend']
            parts.append({
                'rises': f'within each bank rises from {lo} at the low end '
                         f'to {hi} at the high end',
                'falls': f'within each bank falls from {lo} at the low end '
                         f'to {hi} at the high end',
                'peaks': f'highest in the middle of each bank: {hi} there, '
                         f'{lo} at the ends',
                'dips': f'lowest in the middle of each bank: {hi} there, '
                        f'{lo} at the ends'}.get(
                tr, f'repeating pattern within each bank, {lo}-{hi}'))
        text = f'{name}~{_fmt_q(np.nanmedian(q))}'
        return text + (f' ({"; ".join(parts)})' if parts else '')

    lines.append('Quality factors (reliable fits): ' + ', '.join(
        q_part(n, q) for n, q in (('Qi', Qi), ('Qc', Qc), ('Qr', Qr))))
    g = lambda k: metrics.get(k, np.nan)
    if np.isfinite(g('pts_per_lw')):
        lines.append(f'Sampling: {g("pts_per_lw"):.0f} points per linewidth;'
                     f' median SNR {g("snr_median"):.0f}')
    lines.append(f'Noise and gain: S/N {g("noise_db"):.1f} dB; gain range '
                 f'{g("gain_range_db"):.1f} dB, ripple {g("ripple_db"):.1f} '
                 f'dB p-p, sharp features {g("features_db"):.1f} dB p-p; '
                 f'phase jumps {100 * g("phase_jump_frac"):.2f}% of steps')
    lines.append('Notes: ' + ('; '.join(notes) if notes else 'none'))
    bank_info = {k: v for k, v in bank_info.items()
                 if k not in ('label', 'banks')}
    return '\n'.join(lines), dict(range = range_info, banks = bank_info,
                                  q = q_info)

################################################################################
############################ Bookkeeping functions #############################
################################################################################
def run_sweep_info(zarr_grp):
    """
    Print the description of a fitted full VNA sweep, from the results that
    ``vna.full_sweep.run_full_sweep_fit`` saved in
    ``zarr_grp['full_sweep_fit']`` (no refitting).

    Parameters:
    zarr_grp (zarr.Group or str): the zarr group (or its path) used by the
        res finders and the fitter.

    Returns:
    report (str): the printed description (``describe_full_sweep``).

    Raises:
    KeyError: if the group has no 'full_sweep_fit'.
    """
    import os
    import pandas as pd
    import zarr
    if isinstance(zarr_grp, (str, os.PathLike)):
        zarr_grp = zarr.open_group(os.path.expanduser(str(zarr_grp)),
                                   mode = 'r')
    if 'full_sweep_fit' not in zarr_grp:
        raise KeyError("zarr_grp has no 'full_sweep_fit': run "
                       "run_full_sweep_fit first")
    fit = zarr_grp['full_sweep_fit']
    res = pd.DataFrame({c: fit[c][:] for c in ('fr', 'Qr', 'Qi', 'Qc',
                                               'fit_status', 'flags')})
    fg = fit['gain/f']
    f = np.array([fg[0], fg[fg.shape[0] - 1]])
    # Saved attributes hold None for NaN
    metrics = {k: (np.nan if v is None else v)
               for k, v in fit.attrs['sweep_metrics'].items()}
    report = describe_full_sweep(f, res, metrics, fit.attrs['notes'])[0]
    print(report)
    return report
