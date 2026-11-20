import numpy as np
from scipy.optimize import curve_fit

def update_fres(fs, zs, fres, qres, res_idxs, method = 'distance',
                npoints_update = None):
    """
    Update resonant frequencies given fine sweep data.

    Calibration tones (``res_idxs[i] < 0``) are left at their input
    frequency and are not passed to the update algorithm.

    Parameters:
    fs (array-like): Fine sweep frequency data in Hz, shape (M, N).
    zs (array-like): Fine sweep complex S21 data, shape (M, N).
    fres (array-like): Resonant frequencies in Hz, length M.
    qres (array-like): Q-factors, length M.
    res_idxs (array-like): Resonator indices, length M.  Entries with
        values < 0 are treated as calibration tones.
    method (str): Algorithm used to locate the resonance.  'minS21' finds
        the minimum of |S21| after subtracting a linear baseline.  'spacing'
        finds the point of maximum adjacent IQ spacing.  'spacing_fit' fits
        a Lorentzian to the adjacent IQ spacing around that point
        (``update_fr_spacing_fit``; the point itself when the fit fails).
        'distance' finds the point furthest from the off-resonance IQ
        value.  'none' returns fres unchanged.
    npoints_update (int or None): Only the sweep points within this many
        points of the previous frequency (the point nearest ``fres``) are
        considered. None (default): the whole sweep.

    Returns:
    fres_new (np.ndarray): Updated resonant frequencies in Hz, length M.
    """
    # Input validation
    fres = np.asarray(fres, dtype = float)
    if method == 'none':  # Return fres unchanged
        return fres.copy()

    fs = np.asarray(fs, dtype = float)
    zs = np.asarray(zs)
    qres = np.asarray(qres, dtype = float)
    res_idxs = np.asarray(res_idxs)

    # Select update method
    methods = dict(minS21 = update_fr_min_s21, spacing = update_fr_spacing,
                   spacing_fit = update_fr_spacing_fit,
                   distance = update_fr_distance)
    if method not in methods:
        raise ValueError("method must be 'minS21', 'distance', 'spacing', "
                         "'spacing_fit', or 'none'.")
    update = methods[method]

    # Apply update to on-resonance tones only
    res_mask = res_idxs >= 0  # True for resonators
    fres_new = fres.copy()
    if res_mask.any():
        fres_new[res_mask] = update(fs[res_mask], zs[res_mask],
                                    fres = fres[res_mask],
                                    npoints_update = npoints_update)
    return fres_new

def _as_batch(f, z, fres):
    """
    Put sweep data in batch form, shape (M, N).

    Parameters:
    f (array-like): frequency data in Hz, shape (N,) or (M, N).
    z (array-like): complex S21 data, shape (N,) or (M, N).
    fres (float, array-like or None): the previous frequency per sweep.

    Returns:
    f (np.ndarray): frequencies, shape (M, N).
    z (np.ndarray): S21, shape (M, N).
    fres (np.ndarray or None): previous frequencies, shape (M,), or None.
    single (bool): True if the input was one sweep.
    """
    f = np.asarray(f, dtype = float)
    z = np.asarray(z)
    single = f.ndim == 1
    if single:
        f = f[np.newaxis]
        z = z[np.newaxis]
    if fres is not None:
        fres = np.atleast_1d(np.asarray(fres, dtype = float))
    return f, z, fres, single

def update_window(f, fres, npoints_update):
    """
    Mask the sweep points considered by an update: those within
    ``npoints_update`` points of the point nearest the previous frequency.

    Parameters:
    f (np.ndarray): frequency data in Hz, shape (M, N).
    fres (np.ndarray or None): previous frequencies, shape (M,).
    npoints_update (int or None): half-width of the window in points.
        None: every point.

    Returns:
    inside (np.ndarray, bool): shape (M, N), True where a point is
        considered. Every point when ``npoints_update`` or ``fres`` is
        None.
    """
    inside = np.ones(f.shape, dtype = bool)
    if npoints_update is None or fres is None:
        return inside
    n = int(npoints_update)
    if n < 0:
        raise ValueError('npoints_update must be a non-negative integer')
    centre = np.argmin(np.abs(f - fres[:, None]), axis = 1)
    k = np.arange(f.shape[1])[None, :]
    return np.abs(k - centre[:, None]) <= n

def update_fr_min_s21(f, z, fres = None, npoints_update = None):
    """
    Return the resonance frequency estimated as the minimum of |S21| after
    subtracting a linear baseline.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).
    fres (float, array-like or None): the previous frequency of each
        sweep, for ``npoints_update``. Default None.
    npoints_update (int or None): see ``update_window``. Default None:
        the whole sweep.

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    """
    f, z, fres, single = _as_batch(f, z, fres)
    dB = 20.0 * np.log10(np.abs(z))                          # M x N
    # Vectorised linear detrend along axis=1
    f_c = f - f.mean(axis = 1, keepdims = True)
    denom = np.sum(f_c ** 2, axis = 1, keepdims = True)
    slope = np.sum(f_c * dB, axis = 1, keepdims = True) / denom
    trend = slope * f_c + dB.mean(axis = 1, keepdims = True)
    score = np.where(update_window(f, fres, npoints_update), dB - trend,
                     np.inf)
    ix = np.argmin(score, axis = 1)                          # M
    fr = f[np.arange(len(f)), ix]                            # M
    return float(fr[0]) if single else fr

def spacing_score(z):
    """
    Return the adjacent IQ spacing score of a sweep: the sum of the
    distances to the two neighbouring points, 0 at the ends.

    Parameters:
    z (np.ndarray): Complex S21 data, shape (M, N).

    Returns:
    score (np.ndarray): shape (M, N).
    """
    d = np.abs(np.diff(z, axis = 1))                 # M x (N-1)
    score = d[:, 1:] + d[:, :-1]                     # M x (N-2)
    return np.pad(score, ((0, 0), (1, 1)))           # M x N, edges = 0

def update_fr_spacing(f, z, fres = None, npoints_update = None):
    """
    Return the resonance frequency estimated as the point at which the sum of
    adjacent IQ segment lengths is largest.  This quantity peaks where the IQ
    trajectory moves fastest, which corresponds to the resonance.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).
    fres (float, array-like or None): the previous frequency of each
        sweep, for ``npoints_update``. Default None.
    npoints_update (int or None): see ``update_window``. Default None:
        the whole sweep.

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    """
    f, z, fres, single = _as_batch(f, z, fres)
    score = np.where(update_window(f, fres, npoints_update),
                     spacing_score(z), -np.inf)
    ix = np.argmax(score, axis = 1)                   # M
    fr = f[np.arange(len(f)), ix]                     # M
    return float(fr[0]) if single else fr

def _lorentzian(f, a, f0, w, c):
    """
    Evaluate a Lorentzian peak on a constant.

    Parameters:
    f (np.ndarray): frequencies.
    a (float): height above the constant.
    f0 (float): centre.
    w (float): half width at half maximum.
    c (float): the constant.

    Returns:
    y (np.ndarray): the values.
    """
    return c + a / (1. + ((f - f0) / w) ** 2)

def fit_spacing_peak(f, score, ix, inside = None, fit_widths = 3.):
    """
    Fit a Lorentzian to the spacing score around its peak and return the
    fitted centre.

    The half-maximum crossings of the smoothed score on either side of
    ``ix`` give the starting width; the points within ``fit_widths`` half
    widths of the peak, at least 8 points either side, inside the window,
    are fit. The fit is rejected, and None
    returned, when it fails, when the centre leaves the fitted points,
    or when the fitted half width is under one point step or over half
    the sweep.

    Parameters:
    f (np.ndarray): frequencies of one sweep, shape (N,).
    score (np.ndarray): spacing score, shape (N,).
    ix (int): index of the peak.
    inside (np.ndarray, bool or None): points that may be fit. Default
        None: all.
    fit_widths (float): half-width of the fit window in starting half
        widths. Default 3.

    Returns:
    f0 (float or None): the fitted centre, or None if the fit is not
        trusted.
    """
    n = len(f)
    if inside is None:
        inside = np.ones(n, dtype = bool)
    base = np.median(score[inside])
    peak = score[ix]
    if not peak > base:
        return None
    # The starting width from a smoothed score, so that noise does not
    # stop the half-maximum walk at the first dip
    k = min(5, n)
    smooth = np.convolve(score, np.ones(k) / k, mode = 'same')
    half = base + 0.5 * (smooth[ix] - base)
    lo = ix
    while lo > 0 and smooth[lo - 1] > half and inside[lo - 1]:
        lo -= 1
    hi = ix
    while hi < n - 1 and smooth[hi + 1] > half and inside[hi + 1]:
        hi += 1
    step = np.median(np.abs(np.diff(f)))
    w0 = max(0.5 * abs(f[hi] - f[lo]), step)
    sel = inside & (np.abs(f - f[ix]) <= max(fit_widths * w0, 8 * step))
    if sel.sum() < 7:
        return None
    try:
        popt, _ = curve_fit(_lorentzian, f[sel], score[sel],
                            p0 = (peak - base, f[ix], w0, base),
                            maxfev = 2000)
    except (RuntimeError, ValueError):
        return None
    a, f0, w, c = popt
    w = abs(w)
    fsel = f[sel]
    if not (a > 0 and fsel.min() <= f0 <= fsel.max()):
        return None
    if w < step or w > 0.5 * abs(f[-1] - f[0]):
        return None
    return float(f0)

def update_fr_spacing_fit(f, z, fres = None, npoints_update = None,
                          return_fallback = False):
    """
    Return the resonance frequency as the centre of a Lorentzian fit to
    the adjacent IQ spacing around its peak (``fit_spacing_peak``): the
    spacing peaks at resonance and falls off over the linewidth, so the
    fit averages the noise of ``update_fr_spacing`` and lands between
    points. A sweep whose fit is not trusted (a double or truncated
    peak, a failed fit) gets the peak point instead, as 'spacing'.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).
    fres (float, array-like or None): the previous frequency of each
        sweep, for ``npoints_update``. Default None.
    npoints_update (int or None): see ``update_window``. Default None:
        the whole sweep.
    return_fallback (bool): If True, also return which sweeps got the
        peak point. Default False.

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    fallback (bool or np.ndarray): only with ``return_fallback``: True
        where the fit was not trusted.
    """
    f, z, fres, single = _as_batch(f, z, fres)
    inside = update_window(f, fres, npoints_update)
    score = spacing_score(z)
    ix = np.argmax(np.where(inside, score, -np.inf), axis = 1)
    fr = f[np.arange(len(f)), ix].copy()
    fallback = np.ones(len(f), dtype = bool)
    for i in range(len(f)):
        f0 = fit_spacing_peak(f[i], score[i], int(ix[i]), inside[i])
        if f0 is not None:
            fr[i], fallback[i] = f0, False
    if single:
        return (float(fr[0]), bool(fallback[0])) if return_fallback \
            else float(fr[0])
    return (fr, fallback) if return_fallback else fr

def update_fr_distance(f, z, fres = None, npoints_update = None):
    """
    Return the resonance frequency estimated as the point of greatest distance
    from the off-resonance IQ value.  The off-resonance reference is the mean
    of the first and last ``min(10, N//4)`` samples of each sweep.  Performance
    improves when cable delay has been removed beforehand.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).
    fres (float, array-like or None): the previous frequency of each
        sweep, for ``npoints_update``. Default None.
    npoints_update (int or None): see ``update_window``. Default None:
        the whole sweep.

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    """
    f, z, fres, single = _as_batch(f, z, fres)
    N = z.shape[1]
    n_edge = max(1, min(10, N // 4))
    offres = np.concatenate(
        [z[:, :n_edge], z[:, -n_edge:]], axis = 1
    ).mean(axis = 1, keepdims = True)                 # M x 1
    score = np.where(update_window(f, fres, npoints_update),
                     np.abs(z - offres), -np.inf)
    ix = np.argmax(score, axis = 1)                   # M
    fr = f[np.arange(len(f)), ix]                     # M
    return float(fr[0]) if single else fr
