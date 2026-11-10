"""
Simulate the S21 of a KID device for testing the full VNA sweep analysis
and for example notebooks.

A device (``draw_device``) is banks of resonators with fabrication
scatter, missing resonators, a few colliding pairs, spreads in Qi, Qc and
the mismatch rotation phi, and a nonlinearity that grows with power.
Resonators closer than a few linewidths reflect off each other: they are
cascaded on the feedline (``cascade_s21``), which dresses their poles (fr,
Qr, amp, phi) but not their transmission zeros. ``dressed_truth`` gives the
dressed parameters, which is what a product-model fit should recover. The
readout chain (``draw_gain``, ``gain_s21``) adds cable delay, a smooth
gain and phase, standing-wave ripple, and a few broad low-Q features
(e.g. box modes).

These are the device's physics only: noise, dithering, and the band
structure of a measurement come from the simulated CRS
(``crs.sim.SimulatedCRS``), which sweeps this device through the same
procedures as the real one.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment
from ..res.funcs import product_s21
from ..res.multifit import group_resonances, poles_to_params, \
    solve_pole_linear
from ..res.util import qi_qc

# Resonators within this many linewidths of each other are cascaded
GROUP_LW = 10.

################################################################################
############################### Feedline model #################################
################################################################################
def _star(a, b):
    """
    Cascade two 2-ports (s11, s12, s21, s22) with the Redheffer star
    product, a then b.

    Parameters:
    a, b (tuple): 2-ports as tuples of complex arrays.

    Returns:
    s (tuple): the combined 2-port.
    """
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    d = 1 / (1 - a22 * b11)
    return (a11 + a12 * b11 * a21 * d, a12 * b12 * d, a21 * b21 * d,
            b22 + b21 * a22 * b12 * d)

def cascade_s21(f, fr, Qr, amp, phi, theta, a = None, downward = True):
    """
    Compute the S21 of resonators on a shared feedline, including the
    reflections between them.

    Each resonator is a symmetric shunt 2-port with S11 = S21 - 1, where
    S21 is its own transmission (``res.funcs.product_s21`` of one
    resonator); neighbors are joined by matched lossless line sections.
    The total line phase is left out (it is part of the gain phase).

    Parameters:
    f (np.array): frequencies in Hz.
    fr, Qr, amp, phi (np.array): resonator parameters in their order along
        the feedline, shape (N,).
    theta (np.array): electrical length in radians of each of the N - 1
        sections between neighbors.
    a (np.array or None): nonlinearity parameters, shape (N,). Default None
        (linear).
    downward (bool): sweep direction for nonlinear resonators. Default
        True.

    Returns:
    s21 (np.array): complex transmission.
    """
    fr, Qr, amp, phi = (np.atleast_1d(np.asarray(v, dtype = np.float64))
                        for v in (fr, Qr, amp, phi))
    n = len(fr)
    a = np.zeros(n) if a is None else np.atleast_1d(a)
    theta = np.atleast_1d(np.asarray(theta, dtype = np.float64))
    if len(theta) != n - 1:
        raise ValueError('theta must have one entry per section (N - 1)')

    def res2port(k):
        """
        Return resonator k's 2-port.

        Parameters:
        k (int): resonator.

        Returns:
        s (tuple): 2-port.
        """
        s21 = product_s21(f, fr[k], Qr[k], amp[k], phi[k], a[k:k + 1],
                          downward)
        return s21 - 1, s21, s21, s21 - 1

    s = res2port(0)
    for k in range(1, n):
        t = np.full(len(f), np.exp(-1j * theta[k - 1]))
        zero = np.zeros(len(f), complex)
        s = _star(_star(s, (zero, t, t, zero)), res2port(k))
    return s[2] * np.exp(1j * theta.sum())

def dressed_params(fr, Qr, amp, phi, theta):
    """
    Compute the product-model parameters a fit of a cascaded group should
    recover (linear).

    The cascade's S21 is a rational function with N poles and N zeros. A
    linearized rational fit on noiseless samples gives the poles; residues
    follow by linear least squares and ``res.multifit.poles_to_params``
    converts them to amp and phi. Poles are matched to the bare resonators
    by frequency.

    Parameters:
    fr, Qr, amp, phi (np.array): bare parameters in feedline order.
    theta (np.array): section phases, shape (N - 1,).

    Returns:
    fr_d, Qr_d, amp_d, phi_d (np.array): dressed parameters, in the input
        order.
    """
    n = len(fr)
    lw = fr / Qr
    f = np.linspace(np.min(fr - 10 * lw), np.max(fr + 10 * lw),
                    max(401, 60 * n))
    s21 = cascade_s21(f, fr, Qr, amp, phi, theta)
    fc, fs = 0.5 * (f[0] + f[-1]), 0.5 * (f[-1] - f[0])
    x = (f - fc) / fs
    pw = x[:, None] ** np.arange(n + 1)[None, :]
    mat = np.hstack((pw, -s21[:, None] * pw[:, :n]))
    coef = np.linalg.lstsq(mat, s21 * x ** n, rcond = None)[0]
    q = np.roots(np.concatenate(([1.], coef[n + 1:][::-1])))
    p = fc + fs * q
    fr_d, Qr_d = p.real, p.real / (2 * np.abs(p.imag))
    p_coeffs, e, _ = solve_pole_linear(x, s21, fr_d, Qr_d, fc, fs, 0)
    _, amp_d, phi_d = poles_to_params(fr_d, Qr_d, p_coeffs, e, fc, fs)
    _, j = linear_sum_assignment(np.abs(fr[:, None] - fr_d[None, :]))
    return fr_d[j], Qr_d[j], amp_d[j], phi_d[j]

################################################################################
#################################### Device ####################################
################################################################################
def draw_device(
    rng, f_lo = 500e6, f_hi = 2300e6, n_banks = 6, n_per_bank = 100,
    bank_fill = 0.7, scatter = 0.15, yield_frac = 0.95, pair_frac = 0.02,
    pair_sep_lw = (0.5, 8.), Qi_median = 1e5, Qi_dex = 0.3,
    Qc_median = 2.5e4, Qc_dex = 0.2, phi_std = 0.3, a_median = 0.02,
    a_dex = 0.4
):
    """
    Draw the resonators of a simulated device.

    The range is split into ``n_banks`` equal log-frequency banks. Each
    bank's resonators are designed at constant fractional spacing over the
    middle ``bank_fill`` of the bank, then scattered by ``scatter`` times
    the spacing (fabrication; Gaussian) and thinned to ``yield_frac``. A
    fraction ``pair_frac`` get a close partner a log-uniform
    ``pair_sep_lw`` linewidths away. Qi and Qc are log-normal; phi is
    Gaussian (clipped to +-1.3 rad). The nonlinearity at the reference
    power (``device_s21``) is log-normal. Each resonator has a random
    position along the feedline and a random line phase to the next
    resonator there (``theta``), used when it is cascaded with a neighbor.

    Parameters:
    rng (np.random.Generator): random generator.
    f_lo, f_hi (float): device range in Hz. Default 500-2300 MHz.
    n_banks (int): number of banks. Default 6.
    n_per_bank (int): designed resonators per bank. Default 100.
    bank_fill (float): fraction of each bank (in log frequency) holding
        resonators. Default 0.7.
    scatter (float): fabrication scatter in units of the design spacing.
        Default 0.15.
    yield_frac (float): fraction of resonators that work. Default 0.95.
    pair_frac (float): fraction given a close partner. Default 0.02.
    pair_sep_lw (tuple): (min, max) partner separation in linewidths.
        Default (0.5, 8).
    Qi_median, Qi_dex (float): median and std of log10 Qi. Default 1e5,
        0.3.
    Qc_median, Qc_dex (float): median and std of log10 Qc. Default 2.5e4,
        0.2.
    phi_std (float): std of phi in radians. Default 0.3.
    a_median, a_dex (float): median and std of log10 of the nonlinearity
        at the reference power. Default 0.02, 0.4.

    Returns:
    truth (pd.DataFrame): one row per resonator, sorted by fr: ``res_id``,
        ``bank``, ``fr`` (Hz), ``Qi``, ``Qc``, ``Qr``, ``amp``, ``phi``,
        ``a_ref``, ``paired`` (bool), ``feed_pos``, ``theta``.
    """
    edges = np.geomspace(f_lo, f_hi, n_banks + 1)
    fr, bank = [], []
    for b in range(n_banks):
        lo, hi = np.log(edges[b]), np.log(edges[b + 1])
        pad = 0.5 * (1 - bank_fill) * (hi - lo)
        design = np.exp(np.linspace(lo + pad, hi - pad, n_per_bank))
        spacing = np.diff(np.log(design)).mean()
        fr.append(design * np.exp(scatter * spacing *
                                  rng.standard_normal(n_per_bank)))
        bank.append(np.full(n_per_bank, b))
    fr, bank = np.concatenate(fr), np.concatenate(bank)
    keep = rng.random(len(fr)) < yield_frac
    fr, bank = fr[keep], bank[keep]
    n = len(fr)

    def params(m):
        """
        Draw Qi, Qc, phi, a for m resonators.

        Parameters:
        m (int): number of resonators.

        Returns:
        Qi, Qc, phi, a (np.array): parameters.
        """
        Qi = 10 ** rng.normal(np.log10(Qi_median), Qi_dex, m)
        Qc = 10 ** rng.normal(np.log10(Qc_median), Qc_dex, m)
        phi = np.clip(rng.normal(0, phi_std, m), -1.3, 1.3)
        a = 10 ** rng.normal(np.log10(a_median), a_dex, m)
        return Qi, Qc, phi, a

    Qi, Qc, phi, a = params(n)
    paired = np.zeros(n, bool)
    npair = rng.binomial(n, pair_frac)
    if npair:
        src = rng.choice(n, npair, replace = False)
        Qr_src = 1 / (1 / Qi[src] + 1 / Qc[src])
        sep = np.exp(rng.uniform(*np.log(pair_sep_lw), npair)) * \
            rng.choice([-1, 1], npair) * fr[src] / Qr_src
        Qi2, Qc2, phi2, a2 = params(npair)
        fr = np.concatenate((fr, fr[src] + sep))
        bank = np.concatenate((bank, bank[src]))
        Qi, Qc, phi, a = (np.concatenate(v) for v in
                          ((Qi, Qi2), (Qc, Qc2), (phi, phi2), (a, a2)))
        paired = np.concatenate((paired, np.ones(npair, bool)))
        paired[src] = True
    Qr = 1 / (1 / Qi + 1 / Qc)
    truth = pd.DataFrame(dict(
        bank = bank, fr = fr, Qi = Qi, Qc = Qc, Qr = Qr, amp = Qr / Qc,
        phi = phi, a_ref = a, paired = paired,
        feed_pos = rng.permutation(len(fr)),
        theta = rng.uniform(0, 2 * np.pi, len(fr))))
    truth = truth.sort_values('fr', ignore_index = True)
    truth.insert(0, 'res_id', np.arange(len(truth)))
    return truth

def _groups(truth, group_lw):
    """
    Return the cascaded groups of a device, each in feedline order.

    Parameters:
    truth (pd.DataFrame): ``draw_device`` table.
    group_lw (float): resonators within this many linewidths are cascaded.

    Returns:
    groups (list of np.array): row indices, in feedline order.
    """
    groups = group_resonances(truth.fr.values, truth.Qr.values,
                              group_lw / 2)
    pos = truth.feed_pos.values
    return [g[np.argsort(pos[g])] for g in groups]

def device_s21(
    f, truth, power_dbm = -55., p_ref = -55., group_lw = GROUP_LW,
    window_lw = 1000., downward = True
):
    """
    Compute a device's S21 (resonators only, no gain or noise).

    Groups of resonators within ``group_lw`` linewidths are cascaded
    (``cascade_s21``, in feedline order with the ``theta`` of the
    resonators before each section); groups multiply. The nonlinearity is
    ``a_ref * 10^((power - p_ref) / 10)``, with each group's power taken at
    the sample nearest its resonances (so each tone of a target sweep
    drives its own resonator). Each group is evaluated only
    within ``window_lw`` linewidths, which leaves out tails below about
    1 / window_lw.

    Parameters:
    f (np.array): frequencies in Hz (any shape and order).
    truth (pd.DataFrame): ``draw_device`` table.
    power_dbm (float or np.array): readout power in dBm, one value or one
        per frequency (the shape of ``f``). Default -55.
    p_ref (float): power of ``a_ref`` in dBm. Default -55.
    group_lw (float): cascade threshold in linewidths. Default
        ``GROUP_LW`` (10).
    window_lw (float or None): evaluation half-width in linewidths. None:
        everywhere. Default 1000.
    downward (bool): sweep direction for the nonlinear response. Default
        True.

    Returns:
    s21 (np.array): complex transmission, the shape of ``f``.
    """
    f = np.asarray(f, dtype = np.float64)
    shape = f.shape
    order = np.argsort(f, axis = None)
    fs = f.ravel()[order]
    s21 = np.ones(len(fs), complex)
    cols = [truth[c].values for c in ('fr', 'Qr', 'amp', 'phi')]
    power = np.asarray(power_dbm, dtype = np.float64)
    if power.ndim:
        power = np.broadcast_to(power, shape).ravel()[order]
        near = np.clip(np.searchsorted(fs, cols[0]), 0, len(fs) - 1)
        power = power[near]
    a = truth.a_ref.values * 10 ** ((power - p_ref) / 10)
    theta = truth.theta.values
    lw = cols[0] / cols[1]
    for g in _groups(truth, group_lw):
        if window_lw is None:
            i0, i1 = 0, len(fs)
        else:
            i0, i1 = np.searchsorted(fs, [
                np.min(cols[0][g] - window_lw * lw[g]),
                np.max(cols[0][g] + window_lw * lw[g])])
        if i1 <= i0:
            continue
        fw = fs[i0:i1]
        if len(g) == 1:
            s = product_s21(fw, *(c[g] for c in cols), a[g], downward)
        else:
            s = cascade_s21(fw, *(c[g] for c in cols), theta[g[:-1]], a[g],
                            downward)
        s21[i0:i1] *= s
    out = np.empty(len(fs), complex)
    out[order] = s21
    return out.reshape(shape)

def dressed_truth(truth, group_lw = GROUP_LW):
    """
    Add the dressed (product-model) parameters to a device table: what a
    fit should recover. Isolated resonators keep their bare values.

    Parameters:
    truth (pd.DataFrame): ``draw_device`` table.
    group_lw (float): cascade threshold in linewidths. Default
        ``GROUP_LW``.

    Returns:
    truth (pd.DataFrame): a copy with ``group_size`` and the dressed
        ``fr_d``, ``Qr_d``, ``amp_d``, ``phi_d``, ``Qi_d``, ``Qc_d``.
    """
    truth = truth.copy()
    for c in ('fr', 'Qr', 'amp', 'phi'):
        truth[c + '_d'] = truth[c].values.astype(float)
    truth['group_size'] = 1
    for g in _groups(truth, group_lw):
        if len(g) == 1:
            continue
        truth.loc[g, 'group_size'] = len(g)
        d = dressed_params(*(truth[c].values[g] for c in
                             ('fr', 'Qr', 'amp', 'phi')),
                           truth.theta.values[g[:-1]])
        for c, v in zip(('fr_d', 'Qr_d', 'amp_d', 'phi_d'), d):
            truth.loc[g, c] = v
    truth['Qi_d'], truth['Qc_d'], _, _ = qi_qc(truth.Qr_d.values,
                                               truth.amp_d.values)
    return truth

################################################################################
##################################### Gain #####################################
################################################################################
def draw_gain(
    rng, f_lo = 100e6, f_hi = 5e9, tau_ns = (40., 80.), tilt_db_per_ghz = 6.,
    knots_per_ghz = 4., knot_db_std = 1.5, knot_phase_std = 0.5,
    n_ripple = (1, 3), ripple_r = (0.02, 0.1), ripple_period_mhz = (20., 300.),
    broad_per_ghz = 1., broad_Qr = (30., 300.), broad_amp = (0.1, 0.5)
):
    """
    Draw the readout chain's gain: cable delay, a smooth gain and phase
    (tilt plus spline knots), standing-wave ripple, and broad low-Q
    features.

    Parameters:
    rng (np.random.Generator): random generator.
    f_lo, f_hi (float): range in Hz the gain is defined over. Default
        100-5000 MHz.
    tau_ns (tuple): (min, max) cable delay in ns. Default (40, 80).
    tilt_db_per_ghz (float): largest |gain slope| in dB/GHz (drawn
        uniformly, either sign). Default 6.
    knots_per_ghz (float): spline knots per GHz. Default 4.
    knot_db_std, knot_phase_std (float): random dB and phase (rad) at each
        knot. Default 1.5, 0.5.
    n_ripple (tuple): (min, max) number of standing waves. Default (1, 3).
    ripple_r (tuple): (min, max) echo amplitude. Default (0.02, 0.1).
    ripple_period_mhz (tuple): (min, max) ripple period in MHz,
        log-uniform. Default (20, 300).
    broad_per_ghz (float): mean broad features per GHz. Default 1.
    broad_Qr (tuple): (min, max) their Qr, log-uniform. Default (30, 300).
    broad_amp (tuple): (min, max) their depth. Default (0.1, 0.5).

    Returns:
    params (dict): the drawn gain, for ``gain_s21``.
    """
    ghz = (f_hi - f_lo) / 1e9
    nk = max(2, int(np.ceil(knots_per_ghz * ghz)) + 1)
    fk = np.linspace(f_lo, f_hi, nk)
    tilt = rng.uniform(-tilt_db_per_ghz, tilt_db_per_ghz)
    nr = rng.integers(n_ripple[0], n_ripple[1] + 1)
    nb = rng.poisson(broad_per_ghz * ghz)

    def logu(lo, hi, m):
        """
        Draw log-uniform values.

        Parameters:
        lo, hi (float): range.
        m (int): count.

        Returns:
        x (np.array): values.
        """
        return np.exp(rng.uniform(np.log(lo), np.log(hi), m))

    return dict(
        tau = rng.uniform(*tau_ns) * 1e-9, phase0 = rng.uniform(0, 2 * np.pi),
        f_knots = fk,
        db_knots = tilt * (fk - f_lo) / 1e9 + rng.normal(0, knot_db_std, nk),
        phase_knots = rng.normal(0, knot_phase_std, nk),
        ripple_r = rng.uniform(*ripple_r, nr),
        ripple_tau = 1 / (logu(*ripple_period_mhz, nr) * 1e6),
        ripple_phase = rng.uniform(0, 2 * np.pi, nr),
        broad_fr = rng.uniform(f_lo, f_hi, nb),
        broad_Qr = logu(*broad_Qr, nb),
        broad_amp = rng.uniform(*broad_amp, nb),
        broad_phi = rng.uniform(-0.5, 0.5, nb))

def gain_s21(f, params):
    """
    Evaluate a drawn gain.

    Parameters:
    f (np.array): frequencies in Hz (any shape).
    params (dict): ``draw_gain`` output.

    Returns:
    g (np.array): complex gain, the shape of ``f``.
    """
    f = np.asarray(f, dtype = np.float64)
    p = params
    db = CubicSpline(p['f_knots'], p['db_knots'])(f)
    ph = CubicSpline(p['f_knots'], p['phase_knots'])(f)
    g = 10 ** (db / 20) * np.exp(1j * (ph + p['phase0'] -
                                       2 * np.pi * f * p['tau']))
    for r, tau, phase in zip(p['ripple_r'], p['ripple_tau'],
                             p['ripple_phase']):
        g = g * (1 + r * np.exp(-1j * (2 * np.pi * f * tau + phase)))
    if len(p['broad_fr']):
        g = g * product_s21(f.ravel(), p['broad_fr'], p['broad_Qr'],
                            p['broad_amp'], p['broad_phi']).reshape(f.shape)
    return g

################################################################################
################################## Comparison ##################################
################################################################################
def compare_to_truth(fit, truth, tol_lw = 1.):
    """
    Compare fitted resonances with a simulated device's true parameters.

    Fitted and true resonances are paired one-to-one by nearest frequency
    (``vna.match.match_nearest``) and kept as pairs if they are within
    ``tol_lw`` true linewidths. fr, Qr, Qc, and phi are compared with the
    dressed values (what a product-model fit should recover); Qi with the
    bare value, since reflections don't move the transmission zero.

    Parameters:
    fit (pd.DataFrame or dict): fitted resonances with ``fr``, ``Qr``,
        ``Qi``, ``Qc``, ``phi`` and optionally ``flags``, ``valid_fr``,
        ``valid_Qi``, ``valid_Qc``, ``valid_phi`` (e.g. the arrays of a
        ``full_sweep_fit`` group).
    truth (pd.DataFrame): ``dressed_truth`` table.
    tol_lw (float): largest pairing distance in true linewidths. Default
        1.

    Returns:
    pairs (pd.DataFrame): one row per pair: ``res_id``, ``fit_idx``, the
        true and fitted values, the errors ``dfr_lw`` (in linewidths),
        ``dQr``, ``dQi``, ``dQc`` (relative), ``dphi`` (rad), and the
        fit's ``valid_*`` and ``flags`` if given.
    missed (np.array): ``res_id`` of true resonances without a pair.
    extra (np.array): indices into ``fit`` of fits without a pair.
    """
    from .match import match_nearest
    fit = pd.DataFrame({k: np.asarray(v) for k, v in dict(fit).items()
                        if np.ndim(v) == 1})
    fr_fit = fit['fr'].values.astype(float)
    ok = np.isfinite(fr_fit)
    idx_fit = np.flatnonzero(ok)
    lw = truth.fr_d.values / truth.Qr_d.values
    pairs, _, _ = match_nearest(truth.fr_d.values, fr_fit[ok],
                                threshold = tol_lw * np.nanmax(lw))
    t_i, f_i = pairs[:, 0], idx_fit[pairs[:, 1]]
    close = np.abs(fr_fit[f_i] - truth.fr_d.values[t_i]) <= tol_lw * lw[t_i]
    t_i, f_i = t_i[close], f_i[close]
    out = pd.DataFrame(dict(res_id = truth.res_id.values[t_i],
                            fit_idx = f_i))
    for c in ('fr_d', 'Qr_d', 'Qc_d', 'phi_d', 'Qi', 'group_size',
              'paired'):
        if c in truth:
            out['true_' + c] = truth[c].values[t_i]
    for c in ('fr', 'Qr', 'Qi', 'Qc', 'phi'):
        out['fit_' + c] = fit[c].values[f_i].astype(float)
    out['dfr_lw'] = (out.fit_fr - out.true_fr_d) / (lw[t_i])
    out['dQr'] = out.fit_Qr / out.true_Qr_d - 1
    out['dQi'] = out.fit_Qi / out.true_Qi - 1
    out['dQc'] = out.fit_Qc / out.true_Qc_d - 1
    out['dphi'] = out.fit_phi - out.true_phi_d
    for c in ('flags', 'valid_fr', 'valid_Qi', 'valid_Qc', 'valid_phi',
              'fit_status'):
        if c in fit:
            out[c] = fit[c].values[f_i]
    missed = np.setdiff1d(truth.res_id.values, out.res_id.values)
    extra = np.setdiff1d(np.arange(len(fit)), f_i)
    return out, missed, extra
