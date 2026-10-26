"""
Dips in |S21| of a full VNA sweep: how significant a dip is compared with
the surrounding data, and a simple magnitude-only dip finder.

These use |S21| only, so they are not affected by phase noise.
Significance is measured empirically: the dip's mean is compared with the
spread of the means of equally wide windows in the surrounding data, so
correlated noise, ripple and phase noise are accounted for without a noise
model (as a person judging a dip by eye does).
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

def _window_means(y, n, step):
    """
    Return the means of windows of ``n`` points, every ``step`` points.

    Parameters:
    y (np.array): data.
    n (int): window length in points.
    step (int): stride in points.

    Returns:
    means (np.array): one per window (empty if ``y`` is shorter than ``n``).
    """
    if len(y) < n:
        return np.zeros(0)
    c = np.concatenate(([0.], np.cumsum(y)))
    starts = np.arange(0, len(y) - n + 1, max(1, step))
    return (c[starts + n] - c[starts]) / n

def dip_significance(
    f, mag, fr, lw, core_lw = 0.5, ring_lw = (2., 20.), exclude = None,
    exclude_lw = 2., min_windows = 6
):
    """
    Measure how significant the dip at ``fr`` is in |S21|.

    The dip's mean magnitude within +-``core_lw`` linewidths is compared
    with windows of the same number of points over the ring ``ring_lw``
    linewidths away on both sides (excluding the cores of the ``exclude``
    resonances): significance = (median of the window means - dip mean) /
    their robust standard deviation.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    mag (np.array): |S21| with the gain removed (or locally normalized),
        linear.
    fr (float): dip frequency in Hz.
    lw (float): its linewidth fr / Qr in Hz.
    core_lw (float): half-width of the dip window in linewidths. Default
        0.5.
    ring_lw (tuple): (inner, outer) distance of the comparison region in
        linewidths. Default (2, 20).
    exclude (tuple or None): (fr, lw) arrays of other resonances whose
        +-``exclude_lw`` linewidths are left out of the comparison region.
        Default None.
    exclude_lw (float): half-width excluded around each of ``exclude``, in
        their linewidths. Default 2.
    min_windows (int): minimum comparison windows; fewer gives NaN.
        Default 6.

    Returns:
    sig (float): significance (NaN if it cannot be measured).
    depth_db (float): depth of the dip mean below the median window mean,
        in dB (NaN if it cannot be measured).
    """
    i0, i1 = np.searchsorted(f, [fr - ring_lw[1] * lw, fr + ring_lw[1] * lw])
    fw, yw = f[i0:i1], mag[i0:i1]
    core = np.abs(fw - fr) < core_lw * lw
    n = int(core.sum())
    if n < 1:
        return np.nan, np.nan
    ring = np.abs(fw - fr) > ring_lw[0] * lw
    if exclude is not None:
        for e_fr, e_lw in zip(*exclude):
            if e_fr == fr:
                continue
            ring &= np.abs(fw - e_fr) > exclude_lw * e_lw
    # Windows within contiguous stretches of the ring only
    means = []
    edges = np.flatnonzero(np.diff(np.r_[0, ring.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        means.append(_window_means(yw[a:b], n, max(1, n // 2)))
    means = np.concatenate(means) if means else np.zeros(0)
    if len(means) < min_windows:
        return np.nan, np.nan
    base = np.median(means)
    spread = 1.4826 * np.median(np.abs(means - base))
    dip = yw[core].mean()
    if not (base > 0 and dip > 0):
        return np.nan, np.nan
    sig = (base - dip) / spread if spread > 0 else np.inf
    return float(sig), float(20 * np.log10(base / dip))

def _smooth(y, w):
    """
    Return a centered running mean of ``y`` over ``w`` points.

    Parameters:
    y (np.array): data.
    w (int): window in points (1: unchanged).

    Returns:
    ys (np.array): smoothed data, same length.
    """
    return y if w <= 1 else uniform_filter1d(y, w, mode = 'nearest')

def find_dips(
    f, mag, lw = None, prom_db = 1., nsig = 5., max_lw_frac = 0.05,
    return_widths = False
):
    """
    Find dips in |S21| (a simple magnitude-only resonance finder).

    The dB trace is smoothed over ~0.3 linewidths; dips with a prominence
    (within +-10 linewidths) above both ``prom_db`` and ``nsig`` times the
    smoothed trace's noise are returned. Without ``lw``, the typical
    linewidth is estimated from the half-prominence widths of the clearest
    dips.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    mag (np.array): |S21| (linear; only the local prominence matters).
    lw (float or None): typical linewidth in Hz. Default None (estimated).
    prom_db (float): minimum prominence in dB. Default 1.
    nsig (float): minimum prominence in units of the smoothed trace's
        noise. Default 5.
    max_lw_frac (float): largest linewidth considered, as a fraction of
        the span. Default 0.05.
    return_widths (bool): If True, also return each dip's width. Default
        False.

    Returns:
    fr (np.array): dip frequencies in Hz, ascending.
    prom (np.array): their prominences in dB.
    lw (float): the linewidth used, in Hz (NaN if no dips).
    if return_widths:
        width (np.array): each dip's full width at half prominence, Hz.
    """
    empty = (np.zeros(0), np.zeros(0), np.nan)
    if len(f) < 10:
        return empty + (np.zeros(0),) if return_widths else empty
    db = 20 * np.log10(np.maximum(mag, 1e-300))
    step = float(np.median(np.diff(f)))
    if lw is None:
        # Clear dips at light smoothing; their half-prominence widths
        y = -_smooth(db, 3)
        wlen = max(3, int(max_lw_frac * len(f)))
        pk, pr = find_peaks(y, prominence = 3., wlen = wlen)
        if len(pk) == 0:
            return empty + (np.zeros(0),) if return_widths else empty
        top = pk[np.argsort(pr['prominences'])[::-1][:100]]
        w = peak_widths(y, top, rel_height = 0.5, wlen = wlen)[0]
        lw = float(np.median(w)) * step
    lw = max(lw, 2 * step)
    w = max(1, int(round(0.3 * lw / step)))
    ys = _smooth(db, w)
    noise = 1.4826 * np.median(np.abs(np.diff(ys[::w]))) / np.sqrt(2)
    wlen = max(3, int(20 * lw / step))
    pk, pr = find_peaks(-ys, prominence = max(prom_db, nsig * noise),
                        wlen = wlen, distance = max(1, int(0.5 * lw / step)))
    # Report the deepest raw point near each smoothed minimum
    fr = np.empty(len(pk))
    for k, p in enumerate(pk):
        a, b = max(0, p - w), min(len(f), p + w + 1)
        fr[k] = f[a + int(np.argmin(db[a:b]))]
    if not return_widths:
        return fr, pr['prominences'], lw
    width = (peak_widths(-ys, pk, rel_height = 0.5, wlen = wlen)[0] * step
             if len(pk) else np.zeros(0))
    return fr, pr['prominences'], lw, width

def unexplained_dips(
    f, mag, fr, Qr, prom_db = 3., nsig = 10., near_lw = 3., min_sig = 5.,
    own_width = False, strong_db = None, max_strong = None,
    return_info = False
):
    """
    Find clear dips in |S21| with no resonance nearby (possible misses).

    Dips from ``find_dips`` (at the resonances' median linewidth, with the
    stricter ``prom_db`` and ``nsig``) farther than ``near_lw`` linewidths
    from every resonance, and significant against their surroundings
    (``dip_significance`` >= ``min_sig``, the resonances' cores excluded).

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    mag (np.array): |S21| with the gain removed, linear.
    fr, Qr (np.array): the resonances found (Qr NaN: median linewidth).
    prom_db (float): minimum prominence in dB. Default 3.
    nsig (float): minimum prominence in noise units. Default 10.
    near_lw (float): a dip within this many linewidths of a resonance is
        explained. Default 3.
    min_sig (float): minimum dip significance. Default 5.
    own_width (bool): If True, each dip's significance is measured at its
        own width (half-prominence width, 1-20 typical linewidths), so broad
        dips are compared with their surroundings, not their own wings.
        Default False (the typical linewidth).
    strong_db (float or None): dips at least this prominent (dB) count even
        if not significant (a very deep dip with broad wings fails the
        significance test). Default None (off).
    max_strong (int or None): if more dips than this are kept only by
        ``strong_db``, none of them are kept (a sweep full of deep spikes:
        glitches, not resonances). Default None (no limit).
    return_info (bool): If True, also return a dict ``n_strong`` (kept by
        prominence alone) and ``capped`` (True if ``max_strong`` dropped
        them). Default False.

    Returns:
    f_dip (np.array): unexplained dip frequencies in Hz, ascending.
    prom (np.array): their prominences in dB.
    if return_info:
        info (dict): see ``return_info``.
    """
    fr = np.asarray(fr, dtype = float)
    lw = fr / np.asarray(Qr, dtype = float)
    lw_med = np.nanmedian(lw) if np.isfinite(lw).any() else None
    lw = np.where(np.isfinite(lw), lw, lw_med if lw_med else 0.)
    fd, prom, lw_used, width = find_dips(f, mag, lw_med, prom_db, nsig,
                                         return_widths = True)
    w_sig = (np.clip(width, lw_used, 20 * lw_used) if own_width
             else np.full(len(fd), lw_used))
    info = dict(n_strong = 0, capped = False)
    if len(fd) == 0:
        return (fd, prom, info) if return_info else (fd, prom)
    if len(fr):
        o = np.argsort(fr)
        fs, ls = fr[o], lw[o]
        j = np.clip(np.searchsorted(fs, fd), 1, max(len(fs) - 1, 1))
        if len(fs) > 1:
            j = np.where(np.abs(fs[j - 1] - fd) < np.abs(fs[j] - fd),
                         j - 1, j)
        else:
            j = np.zeros(len(fd), int)
        far = np.abs(fs[j] - fd) > near_lw * np.maximum(ls[j], lw_used)
    else:
        far = np.ones(len(fd), bool)
    keep, strong = [], []
    for k in np.flatnonzero(far):
        sig, _ = dip_significance(f, mag, fd[k], w_sig[k], exclude = (fr, lw))
        if np.isfinite(sig) and sig >= min_sig:
            keep.append(k)
        elif strong_db is not None and prom[k] >= strong_db:
            strong.append(k)
    if max_strong is not None and len(strong) > max_strong:
        info['capped'] = True
        strong = []
    info['n_strong'] = len(strong)
    keep = np.sort(np.array(keep + strong, int))
    if return_info:
        return fd[keep], prom[keep], info
    return fd[keep], prom[keep]
