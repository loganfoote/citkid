"""
Smooth gain of a full VNA sweep from its fitted resonances.

The data is divided by the fitted resonator model (product of the
single-resonator S21s), leaving gain times noise everywhere, including on
the resonances. A cubic least-squares spline in log-amplitude and phase,
weighted by |model|^2 (dividing by a deep dip amplifies the noise), then
gives the gain. Compared with masking the resonances and fitting a spline
to what is left, this uses all the data and needs no mask widths; where a
resonance has no fit (NaN parameters), its neighborhood is masked.
"""
import numpy as np
from scipy.interpolate import make_lsq_spline
from ..res.funcs import product_s21

def resonator_model(
    f, fr, Qr, amp, phi, a = None, downward = True, n_lw = 200.,
    n_coarse = 4000
):
    """
    Evaluate the product-model S21 of many resonances over a full sweep.

    Each resonance is evaluated exactly within +-``n_lw`` linewidths. Its
    far tail (each ~amp / (2 n_lw) or less, but hundreds of them add up to
    a few percent) varies slowly, so the sum of all far tails, in log S21,
    is evaluated on a coarse grid and interpolated. The cost stays
    proportional to the sweep length plus resonances x ``n_coarse``.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    fr, Qr, amp, phi (np.array): resonance parameters (finite).
    a (np.array or None): nonlinearity parameters (only used within
        ``n_lw`` linewidths). Default None (all linear).
    downward (bool): sweep direction for nonlinear resonances. Default True.
    n_lw (float): exact-evaluation half-width in linewidths. Default 200.
    n_coarse (int): coarse grid points for the far tails. Default 4000.

    Returns:
    s21 (np.array): complex model transmission.
    """
    f = np.asarray(f, dtype = np.float64)
    fr, Qr, amp, phi = (np.atleast_1d(np.asarray(v, dtype = np.float64))
                        for v in (fr, Qr, amp, phi))
    a = np.zeros(len(fr)) if a is None else np.atleast_1d(a)
    s21 = np.ones(len(f), dtype = np.complex128)
    if len(fr) == 0:
        return s21
    fc = np.linspace(f[0], f[-1], n_coarse)
    tail = np.zeros(n_coarse, dtype = np.complex128)
    for p in zip(fr, Qr, amp, phi, a):
        w = n_lw * p[0] / p[1]
        i0, i1 = np.searchsorted(f, [p[0] - w, p[0] + w])
        if i1 > i0:
            s21[i0:i1] *= product_s21(f[i0:i1], *p, downward = downward)
        far = np.abs(fc - p[0]) >= w
        tail[far] += np.log(product_s21(fc[far], *p[:4]))
    # Linear interpolation of the summed tails; the steps where a
    # resonance's own window starts are ~amp / (2 n_lw) and smooth enough
    tail_f = np.interp(f, fc, tail.real) + 1j * np.interp(f, fc, tail.imag)
    return s21 * np.exp(tail_f)

def _lsq_spline(x, y, w, t_inner, k = 3):
    """
    Fit a weighted least-squares spline, dropping knots without data.

    Parameters:
    x (np.array): abscissae, ascending.
    y (np.array): values.
    w (np.array): weights (> 0).
    t_inner (np.array): interior knots, ascending; those outside (x[0],
        x[-1]) are ignored (the data used can stop short of the sweep's
        ends, e.g. when an edge is masked).
    k (int): degree. Default 3.

    Returns:
    spl (BSpline): fitted spline.
    """
    t_inner = np.asarray(t_inner, dtype = float)
    t_inner = t_inner[(t_inner > x[0]) & (t_inner < x[-1])]
    # Keep interior knots only where every knot interval holds enough
    # points (Schoenberg-Whitney conditions)
    edges = np.r_[x[0], t_inner, x[-1]]
    counts = np.histogram(x, edges)[0]
    keep, run = [], 0
    for knot, c in zip(t_inner, counts[:-1]):
        run += c
        if run >= 2 * (k + 1):
            keep.append(knot)
            run = 0
    t = np.r_[[x[0]] * (k + 1), keep, [x[-1]] * (k + 1)]
    while True:
        try:
            # Normal equations: ~10x faster than the default QR here, same
            # result to ~1e-10
            return make_lsq_spline(x, y, t, k = k, w = w, method = 'norm-eq')
        except (np.linalg.LinAlgError, ValueError):
            if len(t) <= 2 * (k + 1):
                raise
            t = np.r_[t[:k + 1], t[k + 2:-k - 1][::2], t[-k - 1:]]

def fit_gain_spline(
    f, z, fr, Qr, amp, phi, a = None, downward = True, knot_spacing = None,
    exclude = None, exclude_lw = 5., min_model = 0.3, n_robust = 2, clip = 5.,
    core_lw = 3.
):
    """
    Estimate the smooth complex gain of a full VNA sweep.

    Parameters:
    f (np.array): frequencies in Hz, ascending, no repeats.
    z (np.array): complex S21 data.
    fr, Qr, amp, phi (np.array): fitted resonances (product model); rows
        with any NaN are not modeled and are masked instead.
    a (np.array or None): fitted nonlinearity parameters (NaN: 0). Default
        None (linear).
    downward (bool): sweep direction for nonlinear resonances. Default True.
    knot_spacing (float or None): spline knot spacing in Hz. None: 20 times
        the median resonance linewidth, at least 50 frequency steps (the
        gain varies on scales of MHz, resonances on ~0.1 MHz).
    exclude (np.array or None): extra frequencies (Hz) whose neighborhoods
        are masked (e.g. resonances without a fit). Default None.
    exclude_lw (float): mask half-width in (median) linewidths around
        unmodeled resonances. Default 5.
    min_model (float): points where |model| is below this (deep dips) are
        left out (they would also break the phase unwrapping). Default 0.3.
    n_robust (int): refits after dropping outliers (missed or badly fit
        resonances). Default 2.
    clip (float): outlier threshold in robust noise units. Default 5.
    core_lw (float): +- this many linewidths around each modeled resonance
        are left out of the fit (0 to use them). Default 3.

    Returns:
    gain (np.array): complex gain at ``f`` (held constant beyond the first
        and last points used).
    info (dict): ``knot_spacing`` (Hz), ``n_knots``, ``n_modeled``
        (resonances divided out), ``n_masked`` (resonances masked),
        ``frac_dropped`` (share of the used points dropped as outliers).
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    p = [np.asarray(v, dtype = np.float64) for v in (fr, Qr, amp, phi)]
    good = np.all([np.isfinite(v) for v in p], axis = 0)
    a = None if a is None else np.nan_to_num(
        np.asarray(a, dtype = np.float64))[good]
    model = resonator_model(f, *(v[good] for v in p), a = a,
                            downward = downward)
    lw_med = (np.median(p[0][good] / p[1][good]) if good.any()
              else 50 * np.median(np.diff(f)))
    w = np.abs(model) ** 2
    # Points used for the phase unwrapping: all but deep dips (division by
    # a small model amplifies the noise)
    use = (w > min_model ** 2) & np.isfinite(z) & (z != 0)
    # Fit weights: resonance cores and unmodeled resonances get zero weight
    # (kept for the unwrapping, so masked gaps can't skip a 2 pi): the gain
    # comes from the tails and the baseline, so imperfect fits near the
    # resonances don't bend the spline
    if core_lw:
        for c, q in zip(p[0][good], p[1][good]):
            i0, i1 = np.searchsorted(f, [c - core_lw * c / q,
                                         c + core_lw * c / q])
            w[i0:i1] = 0
    masked = list(p[0][~good & np.isfinite(p[0])])
    if exclude is not None:
        masked += list(np.atleast_1d(exclude))
    for c in masked:
        w[np.abs(f - c) < exclude_lw * lw_med] = 0
    g_raw = z[use] / model[use]
    x = (f - f[0]) / (f[-1] - f[0])
    xu = x[use]
    if knot_spacing is None:
        knot_spacing = max(20 * lw_med, 50 * np.median(np.diff(f)))
    n_int = max(int((f[-1] - f[0]) / knot_spacing), 1)
    t_inner = np.linspace(0, 1, n_int + 1)[1:-1]
    ya, yp = np.log(np.abs(g_raw)), np.unwrap(np.angle(g_raw))
    wu = w[use].copy()
    for _ in range(n_robust + 1):
        sel = wu > 0
        la = _lsq_spline(xu[sel], ya[sel], wu[sel], t_inner)
        ph = _lsq_spline(xu[sel], yp[sel], wu[sel], t_inner)
        # Robust reweighting: drop points far from the fit (missed or badly
        # fit resonances, glitches), in units of the noise scaled by the
        # model depth
        r = (np.abs(ya - la(xu)) + np.abs(yp - ph(xu))) * np.sqrt(w[use])
        # Noise scale from the weighted points only (on crowded sweeps most
        # points can have zero weight, which would make the median 0)
        s = 1.4826 * np.median(r[w[use] > 0])
        out = r > clip * s
        keep = (w[use] > 0) & ~out
        if not out.any() or keep.sum() < 4 * (3 + 1):
            break
        wu = np.where(out, 0., w[use])
    # No extrapolation past the data used
    xe = np.clip(x, xu[0], xu[-1])
    gain = np.exp(la(xe) + 1j * ph(xe))
    return gain, dict(knot_spacing = float(knot_spacing),
                      n_knots = int(len(la.t) - 8),
                      n_modeled = int(good.sum()), n_masked = len(masked),
                      frac_dropped = float(np.mean((wu == 0) & (w[use] > 0))))

def normalize_edges(z, n_edge = None):
    """
    Normalize the gain amplitude and phase offset of each sweep from its edge
    samples.

    Parameters:
    z (np.array, complex, (M, N)): complex S21 data, one sweep per row.
    n_edge (int or None): number of off-resonance samples used at each end
        of each row. None (default) uses max(1, N // 100).

    Returns:
    z (np.array, complex, (M, N)): normalized complex S21 data (a new
        array).

    Notes:
    1. Amplitude: divide each row by the median |z| of the edge samples
       (rows whose median is 0 are left unscaled).
    2. Phase offset: rotate each row so the mean of the edge samples lies on
       the positive real axis.
    Cable delay is not removed.
    """
    z = np.asarray(z)
    n = max(1, z.shape[1] // 100) if n_edge is None else n_edge

    # Amplitude normalization from the off-resonance edge samples
    offres = np.concatenate([z[:, :n], z[:, -n:]], axis = 1)
    amp_ref = np.median(np.abs(offres), axis = 1, keepdims = True)
    amp_ref = np.where(amp_ref == 0, 1.0, amp_ref)
    z = z / amp_ref

    # Rotate the off-resonance mean onto the positive real axis
    offres = np.concatenate([z[:, :n], z[:, -n:]], axis = 1)
    phase_ref = np.angle(offres.mean(axis = 1, keepdims = True))
    return z * np.exp(-1j * phase_ref)
