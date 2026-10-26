"""
Fit several resonators in one frequency window (e.g. of a full VNA sweep).

Two stages:

Stage A (``fit_poles``): within a window, fit

    z(x) = P(x) + sum_i e_i L_i(x),   L_i = 1 / (1 + 2j Qr_i (f - fr_i) / fr_i)

with x = (f - fc) / fs, P a complex polynomial (the gain, including cable
delay) and e_i complex residues. P and e_i are solved by linear least
squares (variable projection), so the nonlinear search is only over
(fr_i, log10 Qr_i). For a polynomial gain G times the product model
G(f) * prod_i (1 - A_i L_i) this form is exact; ``poles_to_params``
converts (P, e) to G and each resonator's (amp, phi). No initial guess of
amp, phi or the gain is needed.

Stage B (``fit_resonators``): fit G(x) * product_s21 directly, the gain
again solved linearly, the search over (fr, log10 Qr, amp, phi) per
resonator (plus the nonlinearity a, optionally). This gives the
parameter errors.

Resonances outside the window whose tails reach into it enter as a fixed
product-model factor (``ext_*`` arguments): free residues for them would
be nearly collinear with the gain polynomial.
"""
import numpy as np
from scipy import optimize
from scipy.ndimage import uniform_filter1d
from .funcs import product_s21

_LSQ = dict(x_scale = 'jac', ftol = 1e-12, xtol = 1e-12, gtol = 1e-12)

################################################################################
############################ Windows and grouping ##############################
################################################################################
def group_resonances(fr, Qr, n_lw):
    """
    Group resonances whose fit windows overlap.

    Each resonance's window is fr +- n_lw * fr / Qr.

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    Qr (np.array): total quality factors.
    n_lw (float): window half-width in linewidths.

    Returns:
    groups (list of np.array): indices into ``fr`` for each group, each
        sorted by frequency; groups are in frequency order.
    """
    fr = np.asarray(fr, dtype = np.float64)
    Qr = np.asarray(Qr, dtype = np.float64)
    order = np.argsort(fr)
    half = n_lw * fr / Qr
    groups, end = [], -np.inf
    for i in order:
        if fr[i] - half[i] < end:
            groups[-1].append(i)
        else:
            groups.append([i])
        end = max(end, fr[i] + half[i]) if len(groups[-1]) > 1 \
            else fr[i] + half[i]
    return [np.array(g) for g in groups]

def window_mask(f, fr, Qr, n_lw, min_points = 20):
    """
    Select the fit window around a group of resonances.

    Parameters:
    f (np.array): frequencies in Hz, sorted ascending.
    fr (array-like): resonance frequencies of the group in Hz.
    Qr (array-like): total quality factors of the group.
    n_lw (float): window half-width in linewidths.
    min_points (int): minimum number of points; the window is widened
        symmetrically (in points) if it has fewer. Default 20.

    Returns:
    mask (np.array, bool): True inside the window.
    """
    fr, Qr = np.atleast_1d(fr), np.atleast_1d(Qr)
    lo = np.min(fr - n_lw * fr / Qr)
    hi = np.max(fr + n_lw * fr / Qr)
    i0, i1 = np.searchsorted(f, [lo, hi])
    if i1 - i0 < min_points:
        mid = (i0 + i1) // 2
        i0 = max(0, mid - min_points // 2)
        i1 = min(len(f), i0 + min_points)
        i0 = max(0, i1 - min_points)
    mask = np.zeros(len(f), dtype = bool)
    mask[i0:i1] = True
    return mask

def gain_degree(span):
    """
    Choose the gain polynomial degree for a fit window.

    Parameters:
    span (float): window span in Hz.

    Returns:
    deg (int): 2 for windows up to 2 MHz, +1 per doubling, at most 5.
    """
    return int(np.clip(2 + np.ceil(np.log2(max(span, 1) / 2e6)), 2, 5))

def block_average(f, z, k):
    """
    Average consecutive blocks of ``k`` points (the last block may be short).

    Averaging over a small fraction of a linewidth changes the resonance
    shape negligibly, reduces the noise per point by ~sqrt(k), and speeds
    up fits of densely sampled windows.

    Parameters:
    f (np.array): frequencies in Hz, sorted ascending.
    z (np.array): complex S21 data.
    k (int): block size; ``k`` <= 1 returns the inputs unchanged.

    Returns:
    f_avg (np.array): block-mean frequencies.
    z_avg (np.array): block-mean data.
    """
    if k <= 1:
        return f, z
    idx = np.arange(0, len(f), int(k))
    count = np.diff(np.append(idx, len(f)))
    return np.add.reduceat(f, idx) / count, np.add.reduceat(z, idx) / count

def resid_noise_sigma(f, resid, fr, lw, exclude_lw = 3., min_points = 20):
    """
    Estimate the noise per quadrature from a fit residual off resonance.

    Uses the RMS of the residual outside +-``exclude_lw`` linewidths of the
    fitted resonances, so it includes correlated noise and is not inflated
    by misfit near the resonances. Falls back to the whole residual if
    fewer than ``min_points`` points remain.

    Parameters:
    f (np.array): frequencies in Hz (one window).
    resid (np.array): complex residual, data - model.
    fr (array-like): fitted resonance frequencies in Hz.
    lw (array-like): their linewidths in Hz.
    exclude_lw (float): exclusion half-width in linewidths. Default 3.
    min_points (int): minimum number of off-resonance points. Default 20.

    Returns:
    sigma (float): noise standard deviation per quadrature.
    """
    off = np.ones(len(f), dtype = bool)
    for c, w in zip(np.atleast_1d(fr), np.atleast_1d(lw)):
        off &= np.abs(f - c) > exclude_lw * w
    r = resid[off] if off.sum() >= min_points else resid
    return np.sqrt(np.mean(np.abs(r) ** 2) / 2)

################################################################################
################################ Pole model ####################################
################################################################################
def _poles(fr, Qr, fc, fs):
    """
    Compute the complex poles in x and the scale factors of L_i, where
    L_i(x) = kappa_i / (x - q_i).

    Parameters:
    fr (np.array): resonance frequencies in Hz.
    Qr (np.array): total quality factors.
    fc (float): window center in Hz.
    fs (float): window half-width in Hz.

    Returns:
    q (np.array): complex poles in x.
    kappa (np.array): complex scale factors.
    """
    q = (fr - fc + 1j * fr / (2 * Qr)) / fs
    kappa = fr / (2j * Qr * fs)
    return q, kappa

def _lorentzian_basis(x, fr, Qr, fc, fs):
    """
    Build the Lorentzian columns L_i(x) of the pole model.

    Parameters:
    x (np.array): normalized frequencies, (f - fc) / fs.
    fr (np.array): resonance frequencies in Hz, shape (N,).
    Qr (np.array): total quality factors, shape (N,).
    fc (float): window center in Hz.
    fs (float): window half-width in Hz.

    Returns:
    basis (np.array): complex matrix, shape (len(x), N).
    """
    q, kappa = _poles(np.asarray(fr, dtype = np.float64),
                      np.asarray(Qr, dtype = np.float64), fc, fs)
    return kappa[None, :] / (x[:, None] - q[None, :])

def _poly_basis(x, deg):
    """
    Build polynomial columns x^deg, ..., x^0.

    Parameters:
    x (np.array): normalized frequencies.
    deg (int): polynomial degree.

    Returns:
    basis (np.array): matrix, shape (len(x), deg + 1), ``np.polyval`` order.
    """
    return x[:, None] ** np.arange(deg, -1, -1)[None, :]

def solve_pole_linear(x, z, fr, Qr, fc, fs, gain_deg, m_ext = None):
    """
    Solve the linear part of the pole model for fixed poles:
    z = m_ext(x) * (P(x) + sum_i e_i L_i(x)).

    Parameters:
    x (np.array): normalized frequencies, (f - fc) / fs.
    z (np.array): complex S21 data.
    fr (np.array): resonance frequencies in Hz, shape (N,).
    Qr (np.array): total quality factors, shape (N,).
    fc (float): window center in Hz.
    fs (float): window half-width in Hz.
    gain_deg (int): degree of the polynomial P.
    m_ext (np.array or None): fixed multiplicative factor (e.g. the S21 of
        resonances outside the window). Default None (1).

    Returns:
    p_coeffs (np.array): complex coefficients of P, highest power first.
    e (np.array): complex residues, shape (N,).
    resid (np.array): complex residual, z - model.
    """
    basis = np.hstack((_poly_basis(x, gain_deg),
                       _lorentzian_basis(x, fr, Qr, fc, fs)))
    if m_ext is not None:
        basis = basis * m_ext[:, None]
    coeffs = np.linalg.lstsq(basis, z, rcond = None)[0]
    resid = z - basis @ coeffs
    return coeffs[:gain_deg + 1], coeffs[gain_deg + 1:], resid

def _cov_err(jac, sigma):
    """
    Compute 1-sigma parameter errors from a residual Jacobian.

    Parameters:
    jac (np.array): Jacobian of the stacked real residual.
    sigma (float): noise per quadrature.

    Returns:
    err (np.array): standard errors (pseudo-inverse for singular cases).
    """
    cov = np.linalg.pinv(jac.T @ jac) * sigma ** 2
    return np.sqrt(np.abs(np.diag(cov)))

def _ext_params(ext_fr, ext_Qr, ext_amp, ext_phi):
    """
    Normalize the parameters of external resonances.

    Parameters:
    ext_fr, ext_Qr, ext_amp, ext_phi (array-like or None): parameters; None
        means none (fr, Qr) or the defaults (amp 0.5, phi 0).

    Returns:
    ext_fr, ext_Qr, ext_amp, ext_phi (np.array): float arrays of equal
        length.
    """
    as_arr = lambda v: np.atleast_1d(
        np.asarray([] if v is None else v, dtype = np.float64))
    ext_fr, ext_Qr = as_arr(ext_fr), as_arr(ext_Qr)
    ext_amp = np.full(len(ext_fr), 0.5) if ext_amp is None else \
        as_arr(ext_amp)
    ext_phi = np.zeros(len(ext_fr)) if ext_phi is None else as_arr(ext_phi)
    return ext_fr, ext_Qr, ext_amp, ext_phi

def poles_to_params(fr, Qr, p_coeffs, e, fc, fs, n_iter = 50):
    """
    Convert a pole fit to the gain and each resonator's (amp, phi).

    Assumes the product model, z = G(x) * prod_i (1 - A_i L_i) with
    A_i = amp_i exp(j phi_i) / cos(phi_i). First recovers G and B_i from P
    and e_i using
    G(x) B_i L_i = B_i G(q_i) L_i + B_i kappa_i (G(x) - G(q_i)) / (x - q_i),
    then solves prod_i (1 - A_i L_i) = 1 + sum_i B_i L_i for A_i.

    Parameters:
    fr (np.array): resonance frequencies in Hz, shape (N,).
    Qr (np.array): total quality factors, shape (N,).
    p_coeffs (np.array): complex coefficients of P, highest power first.
    e (np.array): complex residues, shape (N,).
    fc (float): window center in Hz.
    fs (float): window half-width in Hz.
    n_iter (int): maximum fixed-point iterations for G. Default 50.

    Returns:
    gain_coeffs (np.array): complex coefficients of G in x, highest power
        first.
    amp (np.array): Qr / Qc per resonator. Negative if the resonance points
        the wrong way (unphysical).
    phi (np.array): impedance-mismatch rotation per resonator in radians,
        in (-pi/2, pi/2].
    """
    fr, Qr, e = np.atleast_1d(fr), np.atleast_1d(Qr), np.atleast_1d(e)
    q, kappa = _poles(fr, Qr, fc, fs)
    n = len(fr)
    # Recover G from P = G + sum_i B_i kappa_i (G - G(q_i)) / (x - q_i)
    g = np.array(p_coeffs, dtype = np.complex128)
    for _ in range(n_iter):
        b = e / np.polyval(g, q)
        corr = np.zeros_like(g)
        for i in range(n):
            # (G - G(q_i)) / (x - q_i) by synthetic division (the quotient
            # has one fewer coefficient than G, aligned to the low powers)
            acc = 0j
            for k in range(len(g) - 1):
                acc = acc * q[i] + g[k]
                corr[k + 1] += b[i] * kappa[i] * acc
        g_new = p_coeffs - corr
        done = np.all(np.abs(g_new - g) <= 1e-14 * np.abs(g))
        g = g_new
        if done:
            break
    b = e / np.polyval(g, q)
    # Solve B_i = -A_i prod_{j != i} (1 - A_j L_j(q_i)) for A
    off = ~np.eye(n, dtype = bool)
    dq = q[:, None] - q[None, :]
    dq[~off] = 1
    lij = np.where(off, kappa[None, :] / dq, 0)

    def eqs(v):
        a = v[:n] + 1j * v[n:]
        fac = np.where(off, 1 - a[None, :] * lij, 1).prod(axis = 1)
        r = b + a * fac
        return np.concatenate((r.real, r.imag))

    v = optimize.root(eqs, np.concatenate((-b.real, -b.imag))).x
    a = v[:n] + 1j * v[n:]
    phi = np.angle(a)
    amp = np.abs(a) * np.cos(phi)
    # Keep phi in (-pi/2, pi/2]; an upside-down resonance gives amp < 0
    flip = np.abs(phi) > np.pi / 2
    phi[flip] = np.angle(-a[flip])
    amp[flip] = -np.abs(a[flip]) * np.cos(phi[flip])
    return g, amp, phi

################################################################################
############################## Stage A: pole fit ###############################
################################################################################
def fit_poles(
    f, z, fr0, Qr0 = None, gain_deg = 2, Qr_bounds = (1e2, 1e7), n_grid = 41,
    ext_fr = None, ext_Qr = None, ext_amp = None, ext_phi = None,
    max_nfev = 200, tol = 1e-12, fr_bounds = None
):
    """
    Fit resonators in a frequency window with the pole model (Stage A).

    The window should span several linewidths around the resonators, with
    enough off-resonance data to pin down the gain.

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 data.
    fr0 (array-like): initial resonance frequencies in Hz, shape (N,).
    Qr0 (array-like or None): initial total quality factors, shape (N,).
        NaN entries (or all, if None) are initialized by a grid search over
        log10(Qr), one resonator at a time with the others held at their
        current guesses.
    gain_deg (int): degree of the complex gain polynomial. Default 2.
    Qr_bounds (tuple): (min, max) bounds on Qr. Default (1e2, 1e7).
    n_grid (int): number of grid points for the Qr initialization.
        Default 41.
    ext_fr, ext_Qr, ext_amp, ext_phi (array-like or None): fixed parameters
        of resonances outside the window whose tails reach into it. Their
        product-model S21 multiplies the model; nothing about them is fit.
        ``ext_amp`` defaults to 0.5 and ``ext_phi`` to 0 if None. Default
        None (no external resonances).
    max_nfev (int): maximum number of optimizer function evaluations.
        Default 200.
    tol (float): ftol, xtol and gtol of the optimizer. Default 1e-12 (full
        convergence); looser values are faster for screening.
    fr_bounds (np.array or None): per-resonator (min, max) frequency bounds
        in Hz, shape (N, 2), intersected with the window; NaN entries mean
        unbounded. Default None (the window only).

    Returns:
    result (dict): fit results with keys
        ``fr``, ``Qr``, ``amp``, ``phi`` (np.array, shape (N,)): fitted
            resonance frequencies (Hz), quality factors, amp and phi.
        ``fr_err``, ``Qr_err`` (np.array): 1-sigma uncertainties.
        ``ext_fr``, ``ext_Qr``, ``ext_amp``, ``ext_phi`` (np.array): the
            external resonances used.
        ``gain_coeffs`` (np.array): complex gain polynomial in x, highest
            power first, with x = (f - ``fc``) / ``fs``.
        ``fc``, ``fs`` (float): window center and half-width in Hz.
        ``rss`` (float): residual sum of squares, ||z - model||^2.
        ``sigma`` (float): noise per quadrature from the residual.
        ``nrmse`` (float): ||z - model||^2 / ||z||^2.
        ``success`` (bool): optimizer success flag (False if it stopped at
            ``max_nfev``).
        ``at_bound`` (np.array, bool): True where Qr ended at a bound.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    fr0 = np.atleast_1d(np.asarray(fr0, dtype = np.float64))
    n = len(fr0)
    ext_fr, ext_Qr, ext_amp, ext_phi = _ext_params(ext_fr, ext_Qr, ext_amp,
                                                   ext_phi)
    m_ext = product_s21(f, ext_fr, ext_Qr, ext_amp, ext_phi)
    fc = 0.5 * (f.max() + f.min())
    fs = 0.5 * (f.max() - f.min())
    x = (f - fc) / fs
    # The polynomial columns don't depend on the poles: project them out
    # once, so each evaluation solves only for the N residues
    q_p = np.linalg.qr(_poly_basis(x, gain_deg) * m_ext[:, None])[0]
    perp_p = lambda a: a - q_p @ (q_p.conj().T @ a)
    z_p = perp_p(z)
    cache = {}

    def solve(p):
        # Lorentzian columns projected off the polynomial, their QR, the
        # residues and the residual (shared by resid and jac at the same p)
        key = p.tobytes()
        if key not in cache:
            cache.clear()
            fr, Qr = fc + fs * p[:n], 10 ** p[n:]
            lor = _lorentzian_basis(x, fr, Qr, fc, fs)
            q_l, r_l = np.linalg.qr(perp_p(lor * m_ext[:, None]))
            c = q_l.conj().T @ z_p
            e = np.linalg.lstsq(r_l, c, rcond = None)[0]
            cache[key] = (fr, Qr, lor, q_l, e, z_p - q_l @ c)
        return cache[key]

    def resid(p):
        r = solve(p)[5]
        return np.concatenate((r.real, r.imag))

    def jac(p):
        # Variable-projection Jacobian (Kaufman approximation):
        # J = -P_perp (dB/dp) c, where only the L_i columns depend on p
        fr, Qr, lor, q_l, e, _ = solve(p)
        rel = f[:, None] / fr[None, :]
        dfr = lor ** 2 * 2j * Qr * rel / fr * fs          # d/du
        dq = -lor ** 2 * 2j * (rel - 1) * Qr * np.log(10)  # d/dv
        d = perp_p(np.hstack((dfr * e, dq * e)) * m_ext[:, None])
        j = -(d - q_l @ (q_l.conj().T @ d))
        return np.vstack((j.real, j.imag))

    # Initial guess
    u0 = (fr0 - fc) / fs
    vmin, vmax = np.log10(Qr_bounds)
    v0 = np.full(n, np.nan) if Qr0 is None else \
        np.log10(np.atleast_1d(np.asarray(Qr0, dtype = np.float64)))
    need = np.flatnonzero(~np.isfinite(v0))
    if len(need):
        # Linewidths between ~1 sample spacing and the full window
        df = np.median(np.diff(np.sort(f)))
        lo = max(vmin, np.log10(fr0.min() / (2 * fs)))
        hi = min(vmax, np.log10(fr0.max() / max(df, 1e-300)))
        grid = np.linspace(lo, hi, n_grid)
        v0[need] = 0.5 * (lo + hi)
        for i in need:
            cost = []
            for v in grid:
                v0[i] = v
                cost.append(np.sum(resid(np.concatenate((u0, v0))) ** 2))
            v0[i] = grid[np.argmin(cost)]
    lb = np.array([-1.] * n + [vmin] * n)
    ub = np.array([1.] * n + [vmax] * n)
    if fr_bounds is not None:
        fb = (np.asarray(fr_bounds, dtype = np.float64).reshape(n, 2) - fc) \
            / fs
        lb[:n] = np.clip(np.where(np.isfinite(fb[:, 0]), fb[:, 0], -1), -1, 1)
        ub[:n] = np.clip(np.where(np.isfinite(fb[:, 1]), fb[:, 1], 1), -1, 1)
        bad = ub[:n] <= lb[:n]  # bounds outside the window: no constraint
        lb[:n][bad], ub[:n][bad] = -1, 1
    p0 = np.clip(np.concatenate((u0, v0)), lb, ub)
    sol = optimize.least_squares(resid, p0, jac = jac, bounds = (lb, ub),
                                 max_nfev = max_nfev, x_scale = 'jac',
                                 ftol = tol, xtol = tol, gtol = tol)

    fr = fc + fs * sol.x[:n]
    Qr = 10 ** sol.x[n:]
    p_coeffs, e, r = solve_pole_linear(x, z, fr, Qr, fc, fs, gain_deg, m_ext)
    gain_coeffs, amp, phi = poles_to_params(fr, Qr, p_coeffs, e, fc, fs)
    rss = np.vdot(r, r).real
    n_real = 2 * (gain_deg + 1 + n) + 2 * n
    sigma = np.sqrt(rss / max(2 * len(f) - n_real, 1))
    err = _cov_err(sol.jac, sigma)
    return dict(fr = fr, Qr = Qr, amp = amp, phi = phi,
                fr_err = fs * err[:n], Qr_err = Qr * np.log(10) * err[n:],
                ext_fr = ext_fr, ext_Qr = ext_Qr, ext_amp = ext_amp,
                ext_phi = ext_phi, gain_coeffs = gain_coeffs, fc = fc, fs = fs,
                rss = rss, sigma = sigma, nrmse = rss / np.vdot(z, z).real,
                success = sol.success,
                at_bound = ((sol.x[n:] <= vmin + 1e-6)
                            | (sol.x[n:] >= vmax - 1e-6)))

def init_qr(
    f, z, fr0, n_lw = 8., lw_min = None, lw_max = 3e6, factor = 2.,
    gain_deg = 2, min_snr = 3., lw_hint = None, hint_range = 6., tol = 1e-6,
    max_pts_lw = 10.
):
    """
    Estimate (fr, Qr) of one resonance near a frequency guess, at any scale.

    For each candidate linewidth lw_c in a geometric series from ``lw_min``
    to ``lw_max``, fits one pole (``fit_poles``) in a window of
    +-``n_lw`` * lw_c around ``fr0``, with Qr limited to linewidths in
    [lw_c / factor, lw_c * factor]. A scale is valid if the fit converged
    with Qr inside those limits, amp > 0, the fitted fr stayed within
    2 * lw_c of ``fr0``, and SNR = amp |G| / sigma >= ``min_snr``. Among
    valid scales, the one with the highest SNR is returned: windows that
    are too small or too large both lower the SNR estimate. The search
    stops after two consecutive scales, past a valid one, that are invalid
    or below half the best SNR. Fits use a loose tolerance (screening only).

    Parameters:
    f (np.array): frequencies in Hz, sorted ascending.
    z (np.array): complex S21 data.
    fr0 (float): frequency guess in Hz.
    n_lw (float): window half-width in linewidths. Default 8.
    lw_min (float or None): smallest candidate linewidth in Hz. If None,
        3 times the local frequency step.
    lw_max (float): largest candidate linewidth in Hz. Default 3 MHz.
    factor (float): ratio between candidate linewidths. Default 2.
    gain_deg (int): degree of the gain polynomial. Default 2.
    min_snr (float): minimum SNR of a valid scale. Default 3.
    lw_hint (float or None): rough linewidth in Hz (e.g. from a detector).
        If given, only scales within a factor ``hint_range`` of it are
        searched. Default None (full range).
    hint_range (float): search range around ``lw_hint``. Default 6.
    tol (float): optimizer tolerance for the screening fits. Default 1e-6.
    max_pts_lw (float or None): windows sampled more densely than this many
        points per smallest allowed linewidth (lw_c / ``factor``) are block
        averaged down to it (``block_average``) for speed; the SNR is
        corrected back to per-point noise. None disables. Default 10.

    Returns:
    fr (float): estimated resonance frequency in Hz, or NaN if no scale is
        valid.
    Qr (float): estimated total quality factor, or NaN.
    snr (float): SNR of the chosen scale, or NaN.
    """
    i = np.clip(np.searchsorted(f, fr0), 1, len(f) - 1)
    df = np.median(np.diff(f[max(0, i - 50):i + 50]))
    lw_min = 3 * df if lw_min is None else lw_min
    if lw_hint is not None and np.isfinite(lw_hint):
        lw_min = max(lw_min, lw_hint / hint_range)
        lw_max = min(lw_max, lw_hint * hint_range)
        if lw_max < lw_min:
            lw_max = lw_min
    best = (np.nan, np.nan, -np.inf)
    lw_c = lw_min
    n_worse = 0  # consecutive scales after the best that are invalid/worse
    while lw_c <= lw_max * 1.0001 and n_worse < 2:
        i0, i1 = np.searchsorted(f, [fr0 - n_lw * lw_c, fr0 + n_lw * lw_c])
        valid, snr = False, -np.inf
        if i1 - i0 >= 12:
            k = (1 if max_pts_lw is None
                 else max(1, int(lw_c / (factor * df * max_pts_lw))))
            fw, zw = block_average(f[i0:i1], z[i0:i1], k)
            try:
                res = fit_poles(fw, zw, [fr0], [fr0 / lw_c],
                                gain_deg = gain_deg,
                                Qr_bounds = (fr0 / (lw_c * factor),
                                             fr0 * factor / lw_c), tol = tol)
                x = (res['fr'][0] - res['fc']) / res['fs']
                snr = (res['amp'][0]
                       * abs(np.polyval(res['gain_coeffs'], x))
                       / (res['sigma'] * np.sqrt(k)))
                valid = (res['success'] and not res['at_bound'][0]
                         and res['amp'][0] > 0
                         and abs(res['fr'][0] - fr0) < 2 * lw_c
                         and snr >= min_snr)
                if valid and snr > best[2]:
                    best = (res['fr'][0], res['Qr'][0], snr)
            except (np.linalg.LinAlgError, ValueError):
                pass
        # SNR vs scale is roughly unimodal: stop after two worse scales
        if np.isfinite(best[2]) and (not valid or snr < 0.5 * best[2]):
            n_worse += 1
        elif valid:
            n_worse = 0
        lw_c *= factor
    return best if np.isfinite(best[2]) else (np.nan, np.nan, np.nan)

def add_missing_poles(
    f, z, result, gain_deg = 2, max_add = 2, min_dchi2 = 30.,
    min_sep_lw = 1.0, min_snr = 5., min_Qr_ratio = 0.2, distinct_ratio = 2.,
    min_sep_narrow = 2.5, min_sep_close = 1., strong_dchi2 = 1000.,
    strong_amp = 0.3, **kwargs
):
    """
    Add poles for resonances missing from a Stage A fit.

    Repeatedly places a trial pole at the largest smoothed residual (more
    than half a linewidth from existing poles), refits, and keeps it only
    if it looks like a resonance: the fit converged with no Qr at a bound,
    the chi-squared improvement exceeds ``min_dchi2``, the new resonance is
    a dip (0 < amp <= 1.2) with SNR = amp |G| / sigma >= ``min_snr``, its
    Qr is at least ``min_Qr_ratio`` times the smallest Qr in the window
    (broad poles would just absorb gain structure), it lies inside the
    window, and after the refit it is at least ``min_sep_lw`` linewidths
    (of the broader pole) from the other poles, or, if the two linewidths
    differ by at least ``distinct_ratio``, at least ``min_sep_narrow``
    linewidths of the narrower one (down to ``min_sep_close`` for a strong
    new pole: chi-squared improvement > ``strong_dchi2`` and amp >=
    ``strong_amp``). The separation rules reject a second pole that splits
    one resonance whose shape deviates slightly from the model, while
    allowing a broad resonance next to a narrow one.

    Parameters:
    f (np.array): frequencies in Hz (the window of ``result``).
    z (np.array): complex S21 data.
    result (dict): ``fit_poles`` output on (f, z).
    gain_deg (int): degree of the gain polynomial, as in ``result``.
        Default 2.
    max_add (int): maximum number of poles to add. Default 2.
    min_dchi2 (float): minimum (rss_old - rss_new) / sigma_new^2. Default
        30.
    min_sep_lw (float): see above. Default 1.
    min_snr (float): minimum SNR of an added resonance. Default 5.
    min_Qr_ratio (float): see above. Default 0.2.
    distinct_ratio (float): see above. Default 2.
    min_sep_narrow (float): see above. Default 2.5.
    min_sep_close (float): see above. Default 1.
    strong_dchi2 (float): see above. Default 1000.
    strong_amp (float): see above. Default 0.3.
    **kwargs: passed to ``fit_poles`` (e.g. ``Qr_bounds``).

    Returns:
    result (dict): updated ``fit_poles`` result, with ``n_added`` (int) and
        ``added`` (np.array, bool: True for added resonances, which come
        last).
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    n0 = len(result['fr'])
    df = np.median(np.diff(f))
    ext = {k: result['ext_' + k] for k in ('fr', 'Qr', 'amp', 'phi')}
    m_ext = product_s21(f, ext['fr'], ext['Qr'], ext['amp'], ext['phi'])
    for _ in range(max_add):
        lw = result['fr'] / result['Qr']
        x = (f - result['fc']) / result['fs']
        r = solve_pole_linear(x, z, result['fr'], result['Qr'], result['fc'],
                              result['fs'], gain_deg, m_ext)[2]
        k = max(1, int(round(0.5 * np.median(lw) / df)))
        power = uniform_filter1d(np.abs(r) ** 2, k)
        for fr_i, lw_i in zip(result['fr'], lw):
            power[np.abs(f - fr_i) < 0.5 * lw_i] = 0
        if not np.any(power > 0):
            break
        f_new = f[np.argmax(power)]
        trial = fit_poles(f, z, np.append(result['fr'], f_new),
                          np.append(result['Qr'], np.nan),
                          gain_deg = gain_deg, ext_fr = ext['fr'],
                          ext_Qr = ext['Qr'], ext_amp = ext['amp'],
                          ext_phi = ext['phi'], **kwargs)
        dchi2 = (result['rss'] - trial['rss']) / trial['sigma'] ** 2
        x_new = (trial['fr'][-1] - trial['fc']) / trial['fs']
        snr = (trial['amp'][-1]
               * np.abs(np.polyval(trial['gain_coeffs'], x_new))
               / trial['sigma'])
        lw_t = trial['fr'] / trial['Qr']
        d = np.abs(trial['fr'][:-1] - trial['fr'][-1])
        lw_hi = np.maximum(lw_t[:-1], lw_t[-1])
        lw_lo = np.minimum(lw_t[:-1], lw_t[-1])
        distinct = lw_hi >= distinct_ratio * lw_lo
        sep_ok = (d >= min_sep_lw * lw_hi) | (distinct
                                              & (d >= min_sep_narrow * lw_lo))
        close_ok = distinct & (d >= min_sep_close * lw_lo)
        strong = (dchi2 > strong_dchi2 and trial['amp'][-1] >= strong_amp)
        ok = (trial['success'] and not np.any(trial['at_bound'])
              and dchi2 > min_dchi2 and 0 < trial['amp'][-1] <= 1.2
              and snr >= min_snr and abs(x_new) < 0.95
              and trial['Qr'][-1] >= min_Qr_ratio * np.min(result['Qr'])
              and (np.all(sep_ok) or (strong and np.all(sep_ok | close_ok))))
        if not ok:
            break
        result = trial
    n = len(result['fr'])
    result['n_added'] = n - n0
    result['added'] = np.arange(n) >= n0
    return result

def remove_redundant_poles(
    f, z, result, gain_deg = 2, min_dchi2 = 30., max_sep_lw = 1.5, **kwargs
):
    """
    Remove poles that do not significantly improve a Stage A fit.

    The reverse of ``add_missing_poles``, for duplicate guesses: for every
    pole with another pole within ``max_sep_lw`` linewidths, refit without
    it; if the chi-squared increase is below ``min_dchi2``, the pole is
    redundant. The least significant redundant pole is removed and the
    check repeats.

    Parameters:
    f (np.array): frequencies in Hz (the window of ``result``).
    z (np.array): complex S21 data.
    result (dict): ``fit_poles`` output on (f, z).
    gain_deg (int): degree of the gain polynomial, as in ``result``.
        Default 2.
    min_dchi2 (float): a pole is kept if removing it raises the rss by more
        than ``min_dchi2`` sigma^2. Default 30.
    max_sep_lw (float): only poles with another pole this close (in the
        larger linewidth) are tested. Default 1.5.
    **kwargs: passed to ``fit_poles`` (e.g. ``Qr_bounds``).

    Returns:
    result (dict): updated ``fit_poles`` result (or the input if nothing
        was removed), with ``kept`` (np.array, int): indices of the input
        poles that remain, in order.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    kept = np.arange(len(result['fr']))
    ext = {'ext_' + k: result['ext_' + k] for k in ('fr', 'Qr', 'amp', 'phi')}
    while len(kept) > 1:
        fr, lw = result['fr'], result['fr'] / result['Qr']
        sep = np.abs(fr[:, None] - fr[None, :]) / np.maximum(lw[:, None],
                                                              lw[None, :])
        np.fill_diagonal(sep, np.inf)
        cand = np.flatnonzero(sep.min(axis = 1) < max_sep_lw)
        best = None
        for i in cand:
            keep = np.delete(np.arange(len(fr)), i)
            trial = fit_poles(f, z, fr[keep], result['Qr'][keep],
                              gain_deg = gain_deg, **ext, **kwargs)
            dchi2 = (trial['rss'] - result['rss']) / result['sigma'] ** 2
            if dchi2 < min_dchi2 and (best is None or dchi2 < best[0]):
                best = (dchi2, i, trial)
        if best is None:
            break
        result, kept = best[2], np.delete(kept, best[1])
    result['kept'] = kept
    return result

################################################################################
########################### Stage B: physical fit ##############################
################################################################################
def fit_resonators(
    f, z, fr0, Qr0, amp0, phi0, gain_deg = 2, ext_fr = None, ext_Qr = None,
    ext_amp = None, ext_phi = None, Qr_bounds = (1e2, 1e7), max_nfev = 200,
    fit_nonlinear = False, a0 = None, a_max = 1., downward = True
):
    """
    Fit resonator parameters with the product model (Stage B).

    Model: z = P(x) * product_s21(f, ...) * S21_ext, with P a complex gain
    polynomial in x = (f - fc) / fs solved by linear least squares and
    S21_ext the fixed product-model S21 of resonances outside the window.
    Start from a ``fit_poles`` result.

    The fitted (fr, Qr) of close resonators are the poles of the measured
    S21, which include the effect of reflections between them ("dressed"
    parameters); for well separated resonators they equal the bare values.

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 data.
    fr0, Qr0, amp0, phi0 (array-like): initial parameters, shape (N,).
    gain_deg (int): degree of the complex polynomial P. Default 2.
    ext_fr, ext_Qr, ext_amp, ext_phi (array-like or None): fixed external
        resonances, as in ``fit_poles``. Default None.
    Qr_bounds (tuple): (min, max) bounds on Qr. Default (1e2, 1e7).
    max_nfev (int): maximum optimizer function evaluations. Default 200.
    fit_nonlinear (bool): If True, also fit the nonlinearity parameter a of
        each resonator, so ``fr`` is the low-power resonance frequency.
        Default False (linear model).
    a0 (array-like or None): initial a, shape (N,). Default None (0.01).
    a_max (float): upper bound on a. Default 1. Bifurcation is at
        4 sqrt(3) / 9 ~ 0.77; a fit above it means the resonator is
        bifurcated (strongly bifurcated ones end at the bound).
    downward (bool): If True (default), uses the nonlinear solution for a
        downward sweep (decreasing frequency); if False, for an upward
        sweep.

    Returns:
    result (dict): fit results with keys
        ``fr``, ``Qr``, ``amp``, ``phi`` (np.array, shape (N,)): fitted
            parameters, in input order.
        ``fr_err``, ``Qr_err``, ``amp_err``, ``phi_err`` (np.array): 1-sigma
            uncertainties.
        ``a``, ``a_err`` (np.array): nonlinearity parameters and their
            errors (zeros if not fit).
        ``gain_coeffs`` (np.array): complex polynomial P in x.
        ``ext_fr``, ``ext_Qr``, ``ext_amp``, ``ext_phi`` (np.array): the
            external resonances used.
        ``fc``, ``fs``, ``rss``, ``sigma``, ``nrmse``, ``success``: as in
            ``fit_poles``.
        ``downward`` (bool): the sweep direction used.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    fr0, Qr0, amp0, phi0 = (np.atleast_1d(np.asarray(p, dtype = np.float64))
                            for p in (fr0, Qr0, amp0, phi0))
    n = len(fr0)
    ext_fr, ext_Qr, ext_amp, ext_phi = _ext_params(ext_fr, ext_Qr, ext_amp,
                                                   ext_phi)
    fc = 0.5 * (f.max() + f.min())
    fs = 0.5 * (f.max() - f.min())
    x = (f - fc) / fs
    bg_basis = (_poly_basis(x, gain_deg)
                * product_s21(f, ext_fr, ext_Qr, ext_amp, ext_phi)[:, None])
    n_a = n if fit_nonlinear else 0

    def unpack(p):
        a = p[4 * n:] if fit_nonlinear else None
        return (fc + fs * p[:n], 10 ** p[n:2 * n], p[2 * n:3 * n],
                p[3 * n:4 * n], a)

    def linear(p):
        fr, Qr, amp, phi, a = unpack(p)
        s21 = product_s21(f, fr, Qr, amp, phi, a, downward)
        basis = bg_basis * s21[:, None]
        coeffs = np.linalg.lstsq(basis, z, rcond = None)[0]
        return coeffs, z - basis @ coeffs

    def resid(p):
        r = linear(p)[1]
        return np.concatenate((r.real, r.imag))

    vmin, vmax = np.log10(Qr_bounds)
    lb = [-1] * n + [vmin] * n + [1e-6] * n + [-1.5] * n + [0.] * n_a
    ub = [1] * n + [vmax] * n + [1.0] * n + [1.5] * n + [a_max] * n_a
    a_start = (np.full(n, 0.01) if a0 is None
               else np.clip(np.atleast_1d(a0), 0, 0.9 * a_max))
    p0 = np.concatenate(((fr0 - fc) / fs, np.log10(Qr0),
                         np.clip(amp0, 1e-3, 0.999), np.clip(phi0, -1.4, 1.4),
                         a_start if fit_nonlinear else []))
    p0 = np.clip(p0, lb, ub)
    sol = optimize.least_squares(resid, p0, bounds = (lb, ub),
                                 max_nfev = max_nfev, **_LSQ)
    fr, Qr, amp, phi, a = unpack(sol.x)
    coeffs, r = linear(sol.x)
    rss = np.vdot(r, r).real
    n_real = 2 * bg_basis.shape[1] + len(sol.x)
    sigma = np.sqrt(rss / max(2 * len(f) - n_real, 1))
    err = _cov_err(sol.jac, sigma)
    return dict(fr = fr, Qr = Qr, amp = amp, phi = phi,
                fr_err = fs * err[:n],
                Qr_err = Qr * np.log(10) * err[n:2 * n],
                amp_err = err[2 * n:3 * n], phi_err = err[3 * n:4 * n],
                a = np.zeros(n) if a is None else a,
                a_err = err[4 * n:] if fit_nonlinear else np.zeros(n),
                gain_coeffs = coeffs, ext_fr = ext_fr, ext_Qr = ext_Qr,
                ext_amp = ext_amp, ext_phi = ext_phi, fc = fc, fs = fs,
                rss = rss, sigma = sigma, nrmse = rss / np.vdot(z, z).real,
                success = sol.success, downward = bool(downward))

def compare_single_double(
    f, z, fr0, Qr0, gain_deg = 2, splits = (0.3, 0.6, 1.0), ext = None,
    Qr_bounds = (1e2, 1e7), min_Qr_frac = 0.25
):
    """
    Compare single- and double-resonance models for one resonance.

    Single: one resonator with the nonlinear model (a up to 0.7), so small
    shape deviations are not mistaken for a second resonance. Double: two
    linear resonators, Stage A from starts split by ``splits`` linewidths
    around ``fr0`` (each narrower than the single fit), the best refined
    with Stage B. Evidence for the double model is the chi-squared
    improvement in units of the noise (``resid_noise_sigma`` of the double
    fit), penalized BIC-style for the 3 extra parameters.

    Parameters:
    f (np.array): frequencies in Hz (one window around the resonance).
    z (np.array): complex S21 data.
    fr0 (float): resonance frequency guess in Hz.
    Qr0 (float): quality factor guess.
    gain_deg (int): gain polynomial degree. Default 2.
    splits (tuple): trial pair separations in linewidths. Default
        (0.3, 0.6, 1.0).
    ext (dict or None): external resonances, keys ``ext_fr``, ``ext_Qr``,
        ``ext_amp``, ``ext_phi``. Default None.
    Qr_bounds (tuple): (min, max) bounds on Qr. Default (1e2, 1e7).
    min_Qr_frac (float): the double model's Qr lower bound is raised to
        this times the single fit's Qr, so its second pole cannot be a
        broad one absorbing gain structure. Default 0.25.

    Returns:
    result (dict): ``single`` and ``double`` (``fit_resonators`` results),
        ``sigma_noise``, ``dchi2`` = (rss_single - rss_double) /
        sigma_noise^2, ``dbic`` = dchi2 - 3 ln(2 len(f)) (positive favors
        two resonances), and ``sep_lw`` (fitted pair separation in
        linewidths of the broader one).
    """
    ext = ext or {}
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    a1 = fit_poles(f, z, [fr0], [Qr0], gain_deg = gain_deg,
                   Qr_bounds = Qr_bounds, **ext)
    single = fit_resonators(f, z, a1['fr'], a1['Qr'],
                            np.clip(a1['amp'], .01, .99), a1['phi'],
                            gain_deg = gain_deg, fit_nonlinear = True,
                            a_max = 0.7, downward = False,
                            Qr_bounds = Qr_bounds, **ext)
    lw = single['fr'][0] / single['Qr'][0]
    bounds2 = (max(Qr_bounds[0], min_Qr_frac * single['Qr'][0]),
               Qr_bounds[1])
    best = None
    for s in splits:
        fr2 = single['fr'][0] + np.array([-0.5, 0.5]) * s * lw
        try:
            a2 = fit_poles(f, z, fr2, [1.5 * single['Qr'][0]] * 2,
                           gain_deg = gain_deg, Qr_bounds = bounds2, **ext)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if best is None or a2['rss'] < best['rss']:
            best = a2
    double = fit_resonators(f, z, best['fr'], best['Qr'],
                            np.clip(best['amp'], .01, .99), best['phi'],
                            gain_deg = gain_deg, Qr_bounds = bounds2, **ext)
    sig = resid_noise_sigma(f, z - multifit_model(f, double), double['fr'],
                            double['fr'] / double['Qr'])
    dchi2 = (single['rss'] - double['rss']) / sig ** 2
    lw2 = double['fr'] / double['Qr']
    return dict(single = single, double = double, sigma_noise = sig,
                dchi2 = dchi2, dbic = dchi2 - 3 * np.log(2 * len(f)),
                sep_lw = abs(np.diff(double['fr'])[0]) / lw2.max())

def multifit_model(f, result):
    """
    Evaluate the model of a ``fit_poles`` or ``fit_resonators`` result.

    Parameters:
    f (np.array): frequencies in Hz.
    result (dict): output of ``fit_poles`` or ``fit_resonators``.

    Returns:
    z (np.array): complex model S21: gain times the product-model S21 of
        the fitted and the external resonances.
    """
    f = np.asarray(f, dtype = np.float64)
    a = result.get('a')
    s21 = product_s21(f, result['fr'], result['Qr'], result['amp'],
                      result['phi'], a, result.get('downward', True))
    s21_ext = product_s21(f, result['ext_fr'], result['ext_Qr'],
                          result['ext_amp'], result['ext_phi'])
    gain = np.polyval(result['gain_coeffs'], (f - result['fc']) / result['fs'])
    return gain * s21 * s21_ext
