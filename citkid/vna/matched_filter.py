"""
Classical resonance detector for full VNA sweeps: a matched filter with a
polynomial nuisance subspace.

Near a resonance, z(u) = G(u) (1 - A L(u)) with u = ln f, L the Lorentzian
and G the slowly varying gain, so locally z = poly(u) + c L(u) with an
unknown complex c. For each width scale, the template L is projected
orthogonal to low-order polynomials over its support; the correlation of z
with the projected, unit-norm template, divided by the noise, is the
generalized likelihood ratio for a resonance with unknown amplitude in
white noise and an unknown local quadratic gain. Peaks of the best-scale
response above a threshold are the candidates, with Qr from the best
scale. The CNN finder (``vna.res_finder_cnn``) uses it next to the
network: it catches resonances the network misses.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve, find_peaks
from .sweep_info import resample_fractional

def lorentz_template(width, half_len = 6., deg = 2):
    """
    Build the matched-filter template for one width.

    L = 1 / (1 + 2j u / width) on +-``half_len`` linewidths, with the
    projection onto polynomials of degree ``deg`` (the nuisance gain)
    removed, normalized to unit norm.

    Parameters:
    width (float): linewidth in samples.
    half_len (float): template half-length in linewidths. Default 6.
    deg (int): degree of the nuisance polynomial. Default 2.

    Returns:
    t (np.array): complex template, unit norm, centered.
    """
    m = max(deg + 2, int(np.ceil(half_len * width)))
    u = np.arange(-m, m + 1) / width
    t = 1 / (1 + 2j * u)
    q = np.linalg.qr(np.vander(u / u.max(), deg + 1))[0]
    t = t - q @ (q.T @ t)
    return t / np.linalg.norm(t)

def _template_overlap(w1, w2, d):
    """
    Predict the sidelobe ratio of a resonance in another template.

    A resonance of width ``w2`` (full Lorentzian, including its long tails)
    gives response r2 in its own template; this returns the response of a
    template of width ``w1`` offset by ``d`` samples, relative to r2.

    Parameters:
    w1 (float): width of the offset template, samples.
    w2 (float): width of the resonance and its own template, samples.
    d (int): offset in samples.

    Returns:
    rho (float): |<t1(n - d), L2(n)>| / |<t2(n), L2(n)>|.
    """
    t1, t2 = lorentz_template(w1), lorentz_template(w2)
    m1, m2 = len(t1) // 2, len(t2) // 2
    h = int(max(m1 + d, m2, 60 * w2)) + 1
    n = np.arange(-h, h + 1)
    lor = 1 / (1 + 2j * n / w2)
    a = np.zeros(len(n), complex)
    i1 = n - d + m1
    ok1 = (i1 >= 0) & (i1 < len(t1))
    a[ok1] = t1[i1[ok1]]
    b = np.zeros(len(n), complex)
    i2 = n + m2
    ok2 = (i2 >= 0) & (i2 < len(t2))
    b[ok2] = t2[i2[ok2]]
    return abs(np.vdot(a, lor)) / max(abs(np.vdot(b, lor)), 1e-300)

def _detect_chunk(
    f, z, Qr_range, n_scales, threshold, min_sep_lw, suppress_sidelobes
):
    """
    Run the matched filter on one stretch of a sweep (see
    ``find_candidates_mf``).

    Parameters:
    f (np.array): frequencies in Hz, ascending, no repeats.
    z (np.array): complex S21 data.
    Qr_range (tuple): (min, max) Qr to search.
    n_scales (int): number of log-spaced widths.
    threshold (float): detection threshold on the normalized response
        (noise units; ~Rayleigh with sigma ~1 under noise only).
    min_sep_lw (float): candidates closer than this many (best-scale)
        linewidths are merged.
    suppress_sidelobes (bool): If True, drop weaker candidates explained by
        a stronger one's predicted sidelobe.

    Returns:
    fr (np.array): candidate frequencies in Hz.
    Qr (np.array): candidate quality factors (best scale).
    score (np.array): peak response (noise units).
    """
    fg, zg, valid = resample_fractional(f, z)
    step = np.log(fg[1] / fg[0])
    Q = np.geomspace(*Qr_range, n_scales)
    widths = 1 / (Q * step)  # linewidth in samples
    keep = (widths >= 0.7) & (widths <= len(fg) / 20)
    Q, widths = Q[keep], widths[keep]
    if not len(Q):
        return np.array([]), np.array([]), np.array([])
    # Noise of z from second differences of the original samples (var =
    # 6 sigma^2 for white noise; interpolation would bias it), running
    # median over many linewidths of the widest scale, then onto the grid
    d2 = z[2:] - 2 * z[1:-1] + z[:-2]
    win = max(3, int(min(len(d2), max(51, int(20 * widths.max() * len(f)
                                               / len(fg))))) | 1)
    sig_o = median_filter(np.abs(d2), win, mode = 'reflect') \
        / np.sqrt(2 * np.log(2)) / np.sqrt(6)
    sigma = np.interp(fg, f[1:-1], sig_o)
    sigma = np.maximum(sigma, 1e-12 * np.median(np.abs(zg)))
    resp = np.empty((len(widths), len(zg)))
    for k, width in enumerate(widths):
        # Correlation = convolution with the reversed conjugate template
        t = lorentz_template(width)
        resp[k] = np.abs(fftconvolve(zg, np.conj(t[::-1]), mode = 'same')) \
            / sigma
    resp[:, ~valid] = 0
    # Zero the edges, where zero padding looks like a step
    for k, w in enumerate(widths):
        e = int(np.ceil(6 * w)) + 1
        resp[k, :e] = 0
        resp[k, -e:] = 0
    # Normalize each scale to unit noise (correlated noise changes the scale)
    med = np.median(resp, axis = 1, keepdims = True)
    resp /= np.where(med > 0, med, 1) / np.sqrt(np.log(4))
    best = resp.argmax(axis = 0)
    peak = resp.max(axis = 0)
    idx, _ = find_peaks(peak, height = threshold,
                        distance = max(1, int(min_sep_lw * widths.min())))
    # Strongest first; drop peaks closer than min_sep_lw of their width, and
    # sidelobes explained by a stronger candidate's template overlap
    order = idx[np.argsort(-peak[idx])]
    taken = []
    for i in order:
        wi = widths[best[i]]
        keep_i = True
        for j in taken:
            wj = widths[best[j]]
            d = abs(i - j)
            if d < min_sep_lw * wi:
                keep_i = False
                break
            if suppress_sidelobes and d < 6 * wi + 60 * wj:
                rho = _template_overlap(wi, wj, d)
                if peak[i] < 1.5 * rho * peak[j] + threshold:
                    keep_i = False
                    break
        if keep_i:
            taken.append(i)
    taken = np.sort(np.array(taken, dtype = int))
    return fg[taken], Q[best[taken]], peak[taken]

def find_candidates_mf(
    f, z, Qr_range = (1e3, 3e5), n_scales = 24, threshold = 6.,
    min_sep_lw = 0.5, chunk_ratio = 1.1, overlap = 0.01,
    suppress_sidelobes = False
):
    """
    Find resonance candidates in a full VNA sweep with the matched filter.

    The sweep is processed in chunks of fractional span ``chunk_ratio``
    (overlapping by ``overlap`` on each side), each resampled at its own
    local fractional step; candidates are kept from each chunk's core.

    Parameters:
    f (np.array): frequencies in Hz, ascending, no repeats.
    z (np.array): complex S21 data.
    Qr_range (tuple): (min, max) Qr to search. Default (1e3, 3e5).
    n_scales (int): number of log-spaced widths. Default 24.
    threshold (float): detection threshold (noise units). Default 6.
    min_sep_lw (float): candidates closer than this many (best-scale)
        linewidths are merged. Default 0.5.
    chunk_ratio (float): f_hi / f_lo of each chunk core. Default 1.1.
    overlap (float): fractional overlap added on each side. Default 0.01.
    suppress_sidelobes (bool): If True, drop weaker candidates explained by
        a stronger one's predicted sidelobe: removes junk, but also genuine
        neighbors of strong resonances. Default False.

    Returns:
    fr (np.array): candidate frequencies in Hz, ascending.
    Qr (np.array): candidate quality factors (best scale).
    score (np.array): peak response (noise units).
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    edges = [f[0]]
    while edges[-1] * chunk_ratio < f[-1]:
        edges.append(edges[-1] * chunk_ratio)
    edges.append(f[-1] * (1 + 1e-12))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo * (1 - overlap)) & (f <= hi * (1 + overlap))
        if m.sum() < 64:
            continue
        fr, Qr, sc = _detect_chunk(f[m], z[m], Qr_range, n_scales, threshold,
                                   min_sep_lw, suppress_sidelobes)
        core = (fr >= lo) & (fr < hi)
        out.append((fr[core], Qr[core], sc[core]))
    if not out:
        return np.array([]), np.array([]), np.array([])
    fr, Qr, sc = (np.concatenate(v) for v in zip(*out))
    o = np.argsort(fr)
    return fr[o], Qr[o], sc[o]
