"""
Fit every resonance of a full VNA sweep from a list of resonance
frequencies (e.g. ``fres_manual`` from ``run_res_finder_manual``).

The list is authoritative: every listed frequency gets one result row,
and no resonances are added or removed. Each resonance gets Qr from its
frequency alone (``res.multifit.init_qr``); resonances whose fit windows
overlap are fit together, with the tails of their neighbors as fixed
factors (Stage A pole fits, ``res.multifit.fit_poles``, in several passes
as the neighbors improve), then refined with the product model (Stage B,
``res.multifit.fit_resonators``) for the errors. The smooth gain of the
whole sweep is then fit (``vna.gain.fit_gain_spline``) and each result is
flagged where it is unreliable.

``run_full_sweep_fit`` is the notebook step: it reads ``fres_manual``
from a zarr group, fits, prints a description of the sweep
(``vna.sweep_info.describe_full_sweep``), and saves everything in the
group's ``full_sweep_fit`` subgroup.
"""
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm
from ..res.funcs import product_s21
from ..res.multifit import (
    fit_poles, fit_resonators, multifit_model, init_qr, group_resonances,
    window_mask, gain_degree, block_average, resid_noise_sigma,
    solve_pole_linear, add_missing_poles, remove_redundant_poles
)
from .gain import fit_gain_spline
from .sweep_info import (
    prepare_full_sweep, phase_problem, narrow_core, sweep_metrics,
    sweep_notes, describe_full_sweep
)

A_BIFURCATION = 4 * np.sqrt(3) / 9  # nonlinearity at bifurcation, ~0.77
KAPPA_MAX = 0.2     # coupling above which a resonance is flagged 'overlap'
MISFIT_MAX = 0.05   # rms misfit / depth above which a fit is 'poor fit'
RESID_CUTOFF = 5.   # ... if the residual variance is also 6x the noise

################################################################################
############################## Analysis functions ##############################
################################################################################
def coupling_error(fr, Qr, amp):
    """
    Estimate how far each resonance's fitted Qc and phi may be from the
    bare values because of reflections off its neighbors.

    For a pair, the worst case over the feedline phase between them of the
    relative error in Qc, and of the error in phi (radians), is about
    kappa = a_j lw_j / (2 |fr_i - fr_j|) (the neighbor's amp and
    linewidth). Neighbors' reflections arrive with effectively random
    phases, so they are combined as a root sum of squares. On a simulated
    cascade of a real 40 MHz band, the median Qc error was close to kappa
    (90th percentile 2-5x larger), while fr and Qi stayed accurate (Qi:
    median error 0.1%).

    Parameters:
    fr, Qr, amp (np.array): fitted resonances (rows with NaN are ignored as
        neighbors and get NaN).

    Returns:
    kappa (np.array): estimated relative error of Qc (and error of phi in
        radians), per resonance.
    """
    fr, Qr, amp = (np.asarray(v, dtype = np.float64) for v in (fr, Qr, amp))
    ok = np.isfinite(fr) & np.isfinite(Qr) & np.isfinite(amp)
    kappa = np.full(len(fr), np.nan)
    if ok.sum() < 2:
        kappa[ok] = 0.
        return kappa
    f, lw, a = fr[ok], fr[ok] / Qr[ok], np.clip(amp[ok], 0, 1)
    o = np.argsort(f)
    f, lw, a = f[o], lw[o], a[o]
    k = np.zeros(len(f))
    for i in range(len(f)):
        d = np.abs(f - f[i])
        d[i] = np.inf
        k[i] = np.sqrt(np.sum((a * lw / (2 * d)) ** 2))
    out = np.empty(len(f))
    out[o] = k
    kappa[ok] = out
    return kappa

def residual_excess(
    f, z, fr, Qr, amp, phi, sigma, n_lw = 6., core_lw = 2., ext_lw = 30.
):
    """
    Measure unexplained structure at each fitted resonance.

    For each resonance, refits the data in +-``n_lw`` linewidths with the
    fitted resonances there as fixed poles (residues and a gain polynomial
    free, linear least squares) and the tails of the others as a fixed
    factor, and returns the mean |residual|^2 / (2 sigma^2) - 1 within
    +-``core_lw`` linewidths. A bad fit or a resonance missing from the
    list leaves excess residual; noise alone gives ~0.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21 data.
    fr, Qr, amp, phi (np.array): fitted resonances, sorted by fr.
    sigma (np.array): noise per quadrature per point at each resonance.
    n_lw (float): window half-width in linewidths. Default 6.
    core_lw (float): core half-width in linewidths. Default 2.
    ext_lw (float): resonances within this many of their own linewidths of
        the window enter as a fixed factor. Default 30.

    Returns:
    excess (np.array): excess in the core, per resonance (NaN if the
        window has too few points).
    """
    lw = fr / Qr
    amp_c, phi_c = np.clip(amp, 0, 1), np.clip(phi, -1.5, 1.5)
    excess = np.full(len(fr), np.nan)
    for i in range(len(fr)):
        lo, hi = fr[i] - n_lw * lw[i], fr[i] + n_lw * lw[i]
        i0, i1 = np.searchsorted(f, [lo, hi])
        if i1 - i0 < 20:
            continue
        fw, zw = f[i0:i1], z[i0:i1]
        inside = (fr > lo - 2 * lw) & (fr < hi + 2 * lw)
        ext = ~inside & (fr + ext_lw * lw > lo) & (fr - ext_lw * lw < hi)
        fc, fs = 0.5 * (fw[0] + fw[-1]), 0.5 * (fw[-1] - fw[0])
        m_ext = product_s21(fw, fr[ext], Qr[ext], amp_c[ext], phi_c[ext])
        try:
            r = solve_pole_linear((fw - fc) / fs, zw, fr[inside], Qr[inside],
                                  fc, fs, gain_degree(fw[-1] - fw[0]),
                                  m_ext)[2]
        except np.linalg.LinAlgError:
            continue
        e = np.abs(r) ** 2 / (2 * sigma[i] ** 2) - 1
        core = np.abs(fw - fr[i]) < core_lw * lw[i]
        if core.any():
            excess[i] = e[core].mean()
    return excess

def qi_qc(Qr, amp, Qr_err = None, amp_err = None):
    """
    Convert (Qr, amp) arrays to internal and coupling quality factors.

    amp = Qr / Qc with 1 / Qr = 1 / Qi + 1 / Qc, so Qc = Qr / amp and
    Qi = Qr / (1 - amp) (``res.util.calc_qc_qi``, for arrays and with
    errors). amp at 1 (within 1e-6; fits are bounded there) means the
    internal loss is unresolved: Qi = inf. Other amp outside (0, 1) gives
    NaN. Errors are propagated to first order, ignoring the Qr-amp
    covariance.

    Parameters:
    Qr (np.array): total quality factors.
    amp (np.array): Qr / Qc.
    Qr_err, amp_err (np.array or None): 1-sigma errors. Default None (the
        returned errors are NaN).

    Returns:
    Qi (np.array): internal quality factors.
    Qc (np.array): coupling quality factors.
    Qi_err (np.array): 1-sigma error of Qi.
    Qc_err (np.array): 1-sigma error of Qc.
    """
    Qr = np.asarray(Qr, dtype = np.float64)
    amp = np.asarray(amp, dtype = np.float64)
    at_one = np.abs(amp - 1) <= 1e-6
    ok = (amp > 0) & (amp < 1) & ~at_one
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        Qc = np.where(ok | at_one, Qr / amp, np.nan)
        Qi = np.where(ok, Qr / (1 - amp), np.where(at_one, np.inf, np.nan))
        if Qr_err is None or amp_err is None:
            nan = np.full(Qr.shape, np.nan)
            return Qi, Qc, nan, nan.copy()
        rq = np.asarray(Qr_err, dtype = np.float64) / Qr
        ra = np.asarray(amp_err, dtype = np.float64)
        Qc_err = Qc * np.sqrt(rq ** 2 + (ra / amp) ** 2)
        Qi_err = Qi * np.sqrt(rq ** 2 + (ra / (1 - amp)) ** 2)
    return Qi, Qc, Qi_err, Qc_err

def reliability_flags(
    coupling, excess, snr, narrow, a = None, kappa_max = KAPPA_MAX,
    misfit_max = MISFIT_MAX, resid_cutoff = RESID_CUTOFF
):
    """
    Flag fitted resonances whose parameters are unreliable.

    - 'overlap': the coupling to neighbors (``coupling_error``) exceeds
      ``kappa_max``: Qc, phi, Qr and amp may be off by more than that
      fraction; fr and Qi are reliable (Qi comes from the transmission
      zero, which reflections do not move).
    - 'poor fit': the misfit is both significant (residual excess above
      ``resid_cutoff``, not just noise) and large (rms residual within 2
      linewidths relative to the resonance depth, sqrt(2 excess) / snr,
      above ``misfit_max``): only fr is reliable.
    - 'distorted': a narrow core the model cannot follow
      (``sweep_info.narrow_core``): only fr is reliable.
    - 'bifurcated': the fitted nonlinearity is above bifurcation
      (``A_BIFURCATION``, ~0.77; only with a nonlinear fit).

    Parameters:
    coupling (np.array): ``coupling_error`` per resonance.
    excess (np.array): ``residual_excess`` per resonance.
    snr (np.array): amp |gain| / noise per resonance.
    narrow (np.array, bool): ``narrow_core`` per resonance.
    a (np.array or None): fitted nonlinearity (None: linear fit).
    kappa_max (float or None): overlap threshold (None: off). Default 0.2.
    misfit_max (float or None): misfit threshold (None: off). Default 0.05.
    resid_cutoff (float): residual excess threshold. Default 5.

    Returns:
    flags (np.array, str): comma-separated flags per resonance ('' if
        none).
    misfit (np.array): rms misfit relative to the resonance depth.
    """
    excess = np.clip(np.nan_to_num(np.asarray(excess, dtype = float)), 0,
                     None)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        misfit = np.sqrt(2 * excess) / np.asarray(snr, dtype = float)
    rules = [
        ('overlap', np.zeros(len(excess), bool) if kappa_max is None
         else np.nan_to_num(coupling) > kappa_max),
        ('poor fit', np.zeros(len(excess), bool) if misfit_max is None
         else (np.nan_to_num(misfit) > misfit_max) & (excess > resid_cutoff)),
        ('distorted', np.asarray(narrow, dtype = bool)),
        ('bifurcated', np.zeros(len(excess), bool) if a is None
         else np.nan_to_num(a) > A_BIFURCATION)]
    flags = np.array([', '.join(name for name, m in rules if m[i])
                      for i in range(len(excess))], dtype = object)
    return flags, misfit

################################################################################
############################ Bookkeeping functions #############################
################################################################################
def _external(fr, Qr, members, f_lo, f_hi, ext_lw):
    """
    Select resonances outside a group whose tails reach a window.

    Parameters:
    fr, Qr (np.array): all resonances in the sweep.
    members (np.array): indices of the group.
    f_lo, f_hi (float): window edges in Hz.
    ext_lw (float): a resonance is external if it lies within ``ext_lw`` of
        its own linewidths of the window.

    Returns:
    idx (np.array): indices of external resonances.
    """
    lw = fr / Qr
    near = (fr + ext_lw * lw > f_lo) & (fr - ext_lw * lw < f_hi)
    near[members] = False
    return np.flatnonzero(near)

_WORKER = {}

def _worker_init(f, z):
    """
    Store the sweep in a worker process (``_refine`` with ``n_workers``).

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 data.
    """
    _WORKER['f'], _WORKER['z'] = f, z

def _init_task(args):
    """
    Run ``multifit.init_qr`` for a few guesses in a worker process.

    Parameters:
    args (tuple): (frequencies in Hz, linewidth hints in Hz, keyword
        arguments for ``init_qr``).

    Returns:
    out (list of tuple): (fr, Qr) per guess (NaN where none was found).
    """
    frs, hints, kw = args
    return [init_qr(_WORKER['f'], _WORKER['z'], c, lw_hint = h, **kw)[:2]
            for c, h in zip(frs, hints)]

def _stage_a(f, z, st, members, Qr_init, screen = False):
    """
    Fit one group of resonances with the pole model (``_refine`` steps 2-3).

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21 data.
    st (dict): fit state: ``fr``, ``q_win``, ``amp``, ``phi``, ``active``
        (np.array per guess), ``bounds`` (Qr bounds), ``n_lw``, ``ext_lw``,
        ``max_pts_lw``, ``max_pts_lw_final``.
    members (np.array): indices of the group.
    Qr_init (np.array): starting Qr of the members.
    screen (bool): If True, average dense windows to ``max_pts_lw`` points
        per linewidth, else to ``max_pts_lw_final``. Default False.

    Returns:
    m (np.array, bool): the window.
    ext (np.array): indices of the external resonances.
    res (dict): ``multifit.fit_poles`` result.
    fw, zw (np.array): the (averaged) window data.
    k (int): points averaged.
    """
    fr, q_win = st['fr'], st['q_win']
    m = window_mask(f, fr[members], q_win[members], st['n_lw'])
    ext = _external(fr, q_win, members, f[m][0], f[m][-1], st['ext_lw'])
    ext = ext[st['active'][ext]]
    fw, zw = f[m], z[m]
    target = st['max_pts_lw'] if screen else st['max_pts_lw_final']
    k = 1
    if target is not None:
        # Densely sampled windows are block averaged down to about
        # ``target`` points per (narrowest) linewidth
        lw = np.min(fr[members] / q_win[members])
        k = max(1, int(lw / (np.median(np.diff(fw)) * target)))
        fw, zw = block_average(fw, zw, k)
    bounds = st['bounds']
    res = fit_poles(fw, zw, fr[members],
                    np.clip(Qr_init, bounds[0] * 1.01, None),
                    gain_deg = gain_degree(f[m][-1] - f[m][0]),
                    Qr_bounds = tuple(bounds), ext_fr = fr[ext],
                    ext_Qr = q_win[ext], ext_amp = st['amp'][ext],
                    ext_phi = st['phi'][ext])
    return m, ext, res, fw, zw, k

def _final_unit(f, z, st, g, opts):
    """
    Fit one unit for good (``_refine`` step 3): Stage A, optionally the
    missing-pole search and redundant-pole removal, and Stage B. A failed
    joint fit is redone one member at a time.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21 data.
    st (dict): fit state (``_stage_a``), not modified.
    g (np.array): indices of the unit's members.
    opts (dict): ``add_poles``, ``fit_nonlinear``, ``downward``,
        ``dup_lw``.

    Returns:
    groups (list of tuple): per fit group, in order: rows (list of dict;
        ``sorted_idx`` is the index of the guess, -1 for added poles) and
        dropped guesses (list of (index, index of the guess it duplicates
        or -1)).
    """
    bounds = st['bounds']
    out = []
    queue = [g]
    while queue:
        g = queue.pop(0)
        try:
            m, ext, a, fw, zw, kb = _stage_a(f, z, st, g, st['q_win'][g])
        except Exception:
            a = None
        if len(g) > 1 and (a is None or np.any(a['at_bound'])):
            # A failed joint fit is redone one member at a time
            queue[:0] = [np.array([k]) for k in g]
            continue
        dropped = []
        try:
            if a is None:
                m, ext, a, fw, zw, kb = _stage_a(f, z, st, g, st['q_win'][g])
            deg = gain_degree(f[m][-1] - f[m][0])
            a['added'] = np.zeros(len(a['fr']), dtype = bool)
            if opts['add_poles']:
                # Missing resonances, then poles that add nothing
                # (duplicate guesses or added poles)
                a = add_missing_poles(fw, zw, a, gain_deg = deg,
                                      Qr_bounds = tuple(bounds))
                if len(a['fr']) > 1:
                    added = a['added']
                    a = remove_redundant_poles(fw, zw, a, gain_deg = deg,
                                               Qr_bounds = tuple(bounds))
                    kept = a['kept']
                    a['added'] = added[kept]
                    kept_g = kept[kept < len(g)]
                    frg = st['fr'][g]
                    for k in np.setdiff1d(np.arange(len(g)), kept_g):
                        near = (g[kept_g[np.argmin(np.abs(frg[kept_g]
                                                          - frg[k]))]]
                                if len(kept_g) else -1)
                        dropped.append((g[k], near))
                    g = g[kept_g]

            def stage_b(a):
                return fit_resonators(
                    fw, zw, a['fr'], a['Qr'], np.clip(a['amp'], 1e-3, 0.999),
                    a['phi'], gain_deg = deg, ext_fr = a['ext_fr'],
                    ext_Qr = a['ext_Qr'], ext_amp = a['ext_amp'],
                    ext_phi = a['ext_phi'],
                    Qr_bounds = (bounds[0] * 0.5, bounds[1]),
                    fit_nonlinear = opts['fit_nonlinear'],
                    downward = opts['downward'])

            def b_ok_of(a, b):
                # Converged, not worse than Stage A, and no two poles merged
                o = np.argsort(b['fr'])
                lw_s = (b['fr'] / b['Qr'])[o]
                merged = np.any(np.diff(b['fr'][o]) < opts['dup_lw']
                                * np.minimum(lw_s[1:], lw_s[:-1]))
                return (b['success'] and b['rss'] <= 1.5 * a['rss']
                        and not merged)

            b = stage_b(a)
            if not b_ok_of(a, b) and a['added'].any():
                # Retry without the added poles
                keep = ~a['added']
                a = fit_poles(fw, zw, a['fr'][keep], a['Qr'][keep],
                              gain_deg = deg, Qr_bounds = tuple(bounds),
                              ext_fr = a['ext_fr'], ext_Qr = a['ext_Qr'],
                              ext_amp = a['ext_amp'], ext_phi = a['ext_phi'])
                a['added'] = np.zeros(keep.sum(), dtype = bool)
                b = stage_b(a)
            b_ok = b_ok_of(a, b)
            ng = len(a['fr'])
            gidx = np.concatenate((g, -np.ones(ng - len(g), int)))
            if not b_ok:
                # Fall back to Stage A (same product model)
                b = dict(b, **{k: a[k] for k in ('fr', 'Qr', 'amp', 'phi',
                                                 'fr_err', 'Qr_err', 'rss',
                                                 'sigma', 'nrmse',
                                                 'gain_coeffs')},
                         amp_err = np.full(ng, np.nan),
                         phi_err = np.full(ng, np.nan), a = np.zeros(ng),
                         a_err = np.full(ng, np.nan),
                         downward = opts['downward'])
            # Noise per original point: averaging kb points reduced it by
            # ~sqrt(kb)
            sig_n = np.sqrt(kb) * resid_noise_sigma(
                fw, zw - multifit_model(fw, b), b['fr'], b['fr'] / b['Qr'])
            rows = []
            for k in range(ng):
                x_k = (b['fr'][k] - b['fc']) / b['fs']
                g_k = np.abs(np.polyval(b['gain_coeffs'], x_k))
                rows.append(dict(
                    sorted_idx = gidx[k], b_ok = b_ok, a_fr = a['fr'][k],
                    a_Qr = a['Qr'][k], a_amp = a['amp'][k],
                    a_at_bound = bool(a['at_bound'][k]),
                    fr = b['fr'][k], Qr = b['Qr'][k], amp = b['amp'][k],
                    phi = b['phi'][k], fr_err = b['fr_err'][k],
                    Qr_err = b['Qr_err'][k], amp_err = b['amp_err'][k],
                    phi_err = b['phi_err'][k], nl_a = b['a'][k],
                    nl_a_err = b['a_err'][k], nrmse = b['nrmse'],
                    snr = b['amp'][k] * g_k / (
                        sig_n if np.isfinite(sig_n)
                        else np.sqrt(kb) * b['sigma']),
                    sigma_noise = sig_n, duplicate_of = -1, error = '',
                    gain_coeffs = np.asarray(b['gain_coeffs']),
                    gain_fc = b['fc'], gain_fs = b['fs']))
        except Exception as exc:
            rows = [dict(sorted_idx = k, duplicate_of = -1,
                         error = repr(exc)) for k in g]
        out.append((rows, dropped))
    return out

def _final_task(args):
    """
    Run ``_final_unit`` on a few units in a worker process.

    Parameters:
    args (tuple): (fit state, list of units, ``_final_unit`` options).

    Returns:
    out (list): ``_final_unit`` output per unit.
    """
    st, units, opts = args
    return [_final_unit(_WORKER['f'], _WORKER['z'], st, g, opts)
            for g in units]

def _chunks(items, n):
    """
    Split a list into about ``n`` consecutive chunks.

    Parameters:
    items (list): items.
    n (int): number of chunks wanted.

    Returns:
    chunks (list of list): consecutive pieces, in order.
    """
    size = max(1, int(np.ceil(len(items) / max(n, 1))))
    return [items[i:i + size] for i in range(0, len(items), size)]

def _refine(
    f, z, fr0, Qr0 = None, fit_nonlinear = False, downward = True,
    add_poles = False, n_lw = 8., lw_max = 3e6, n_pass = 3, ext_lw = 30.,
    max_group = 8, Qr_bounds = (1e2, 1e7), min_Qr_frac = 0.2, dup_lw = 0.1,
    max_pts_lw = 10., max_pts_lw_final = 20., n_workers = None,
    progress = False
):
    """
    Fit all resonances of a sweep, starting from frequency guesses.

    Steps:
    1. (fr, Qr) for each guess from the multi-scale search
       ``multifit.init_qr`` (within a factor 6 of ``Qr0`` where given).
    2. ``n_pass`` passes of Stage A on groups whose +-``n_lw`` linewidth
       windows overlap, with neighbors outside the group as external
       factors. A group whose joint fit fails (a Qr at a bound) is fit one
       member at a time; members still at a bound alone are dropped
       (``error`` 'Qr at a bound') and retried in later passes.
    3. Per group, from the state after step 2: Stage A (with
       ``add_poles``, then the missing-pole search,
       ``multifit.add_missing_poles``, and the removal of poles that add
       nothing, ``multifit.remove_redundant_poles``), then Stage B
       (``b_ok`` if it converged, its residual is at most 1.5x Stage A's,
       and no two poles merged; otherwise the Stage A values are kept; with
       added poles that fail, retried without them). Fits from different
       groups that converged onto the same resonance are reduced to one.

    Clusters larger than ``max_group`` are split by regrouping with smaller
    windows, so strongly overlapping resonators are still fit jointly. For
    windows and grouping, linewidths are capped (robust Qr floor), so one
    bad fit can't create a huge window. Steps 1 and 3 run in
    ``n_workers`` processes (the result does not depend on it); step 2 is
    sequential (each group uses its neighbors' latest fits).

    Parameters:
    f (np.array): frequencies in Hz, sorted ascending, no repeats.
    z (np.array): complex S21 data.
    fr0 (np.array): initial resonance frequencies in Hz.
    Qr0 (np.array or None): rough quality factors (NaN: unknown), same
        order as ``fr0``. Default None.
    fit_nonlinear (bool): If True, Stage B also fits the nonlinearity a.
        Default False.
    downward (bool): sweep direction for the nonlinear model. Default True.
    add_poles (bool): If True, search for resonances missing from the
        guesses (for detector candidates; added rows have ``guess_idx``
        -1). Default False (the guesses are a complete list).
    n_lw (float): fit-window half-width in linewidths. Default 8.
    lw_max (float): largest linewidth (Hz) considered in step 1. Default
        3 MHz.
    n_pass (int): number of Stage A passes. Default 3.
    ext_lw (float): neighbors within this many of their linewidths of a
        window are external factors. Default 30.
    max_group (int): largest cluster fit jointly. Default 8.
    Qr_bounds (tuple): (min, max) bounds on Qr. Default (1e2, 1e7).
    min_Qr_frac (float): the robust Qr floor is min(``min_Qr_frac`` x
        median, 0.3 x the 2nd percentile of the step-1 Qr above 0.05 x
        median); step-1 values below it count as failed, and it is the
        lower Qr bound afterwards. Default 0.2.
    dup_lw (float): fits within this many linewidths of each other are
        duplicates. Default 0.1.
    max_pts_lw (float or None): steps 1-2 average windows sampled more
        densely than this many points per linewidth. Default 10.
    max_pts_lw_final (float or None): the same for step 3. Default 20.
    n_workers (int or None): processes for steps 1 and 3. None (default):
        up to 4 (more is slower on a typical sweep: starting processes
        costs more than it saves) for 50 or more guesses, else 1.
    progress (bool): If True, show progress bars.

    Returns:
    df (pd.DataFrame): one row per fit (and per dropped guess):
        ``guess_idx`` (index into ``fr0``), ``fr0``, ``group``, ``b_ok``,
        ``a_fr``, ``a_Qr``, ``a_amp``, ``a_at_bound`` (Stage A), ``fr``,
        ``Qr``, ``amp``, ``phi``, their ``_err``, ``nl_a``, ``nl_a_err``,
        ``nrmse``, ``snr``, ``sigma_noise``, ``duplicate_of`` (guess index
        it duplicates, else -1), ``error`` ('' if fit), and the fit's
        local gain polynomial ``gain_coeffs`` (in x = (f - ``gain_fc``) /
        ``gain_fs``, highest power first).
    lw_guess (np.array): step-1 linewidth (Hz) per guess.
    q_bounds (tuple): the Qr bounds used after step 1.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    fr0 = np.asarray(fr0, dtype = np.float64)
    fr = np.sort(fr0)
    guess_idx = np.argsort(fr0)
    n = len(fr)
    Qr = np.full(n, np.nan)
    q_win = np.full(n, np.nan)  # Qr used for windows and grouping
    amp = np.full(n, 0.5)       # latest amp, phi (for external resonances)
    phi = np.zeros(n)
    active = np.ones(n, dtype = bool)  # False for duplicates and pinned
    pinned = np.zeros(n, dtype = bool)  # Qr at a bound when fit alone
    dup_of = np.full(n, -1)
    bounds = list(Qr_bounds)  # lower bound raised to the floor after step 1
    st = dict(fr = fr, q_win = q_win, amp = amp, phi = phi, active = active,
              bounds = bounds, n_lw = n_lw, ext_lw = ext_lw,
              max_pts_lw = max_pts_lw, max_pts_lw_final = max_pts_lw_final)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4) if n >= 50 else 1

    def split(g, q, radius):
        # Split a cluster by regrouping with smaller window radii until
        # every subgroup has at most max_group members
        if len(g) <= max_group:
            return [g]
        if radius < 0.25:
            return [np.array([k]) for k in g]
        out = []
        for sub in group_resonances(fr[g], q[g], radius):
            out.extend(split(g[sub], q, radius / 2))
        return out

    def units(q):
        # Fit units: groups of overlapping windows, large clusters split
        idx = np.flatnonzero(active)
        out = []
        for g in group_resonances(fr[idx], q[idx], n_lw):
            out.extend(split(idx[g], q, n_lw / 2))
        return out

    def mark_duplicates():
        # Guesses that converged onto the same resonance: keep the first
        idx = np.flatnonzero(active)
        idx = idx[np.argsort(fr[idx])]
        lw = fr[idx] / q_win[idx]
        close = np.diff(fr[idx]) < dup_lw * np.minimum(lw[1:], lw[:-1])
        for k in np.flatnonzero(close):
            keep, drop = idx[k], idx[k + 1]
            if active[keep]:
                active[drop] = False
                dup_of[drop] = keep

    def accept(members, res):
        # Keep a fit only if it converged and no Qr ended at a bound
        if res['success'] and not np.any(res['at_bound']):
            fr[members], Qr[members] = res['fr'], res['Qr']
            amp[members] = np.clip(res['amp'], 0, 1)
            phi[members] = np.clip(res['phi'], -1.5, 1.5)
            pinned[members] = False
            return True
        return False

    def fit_unit(g):
        # Joint Stage A fit of one unit; if it fails, fit the members one at
        # a time (one bad pole otherwise drags every pole of the unit to a
        # bound). A member still at a bound on its own is pinned.
        try:
            res = _stage_a(f, z, st, g, q_win[g], screen = True)[2]
        except Exception:
            res = None
        if res is not None and accept(g, res):
            return
        if len(g) > 1:
            for k in g:
                fit_unit(np.array([k]))
        elif res is not None and np.any(res['at_bound']):
            pinned[g] = True

    def q_floor(q):
        # Robust lower bound on Qr, below the distribution's low tail
        q = q[np.isfinite(q)]
        med = np.median(q)
        core = q[q > 0.05 * med]
        return min(min_Qr_frac * med, 0.3 * np.quantile(core, 0.02))

    def update_q_win():
        # Window/grouping Qr: the fitted Qr, or the sweep median if unknown,
        # capped below at 0.5 x min_Qr_frac x median
        q_ref = np.nanmedian(Qr) if np.any(np.isfinite(Qr)) else 1e4
        q_min = max(bounds[0], 0.5 * min_Qr_frac * q_ref)
        q_win[:] = np.where(np.isfinite(Qr), np.maximum(Qr, q_min), q_ref)

    pool = (ProcessPoolExecutor(n_workers, initializer = _worker_init,
                                initargs = (f, z)) if n_workers > 1 else None)
    try:
        # 1. Qr initialization (restricted around Qr0 where given)
        hint = np.full(n, np.nan) if Qr0 is None else \
            fr / np.asarray(Qr0, dtype = np.float64)[guess_idx]
        kw = dict(n_lw = n_lw, lw_max = lw_max, max_pts_lw = max_pts_lw)
        if pool is None:
            init = [init_qr(f, z, fr[k], lw_hint = hint[k], **kw)[:2]
                    for k in tqdm(range(n), disable = not progress,
                                  leave = False, desc = 'Qr init')]
        else:
            tasks = [(fr[c], hint[c], kw)
                     for c in _chunks(list(range(n)), 4 * n_workers)]
            init = [x for out in tqdm(pool.map(_init_task, tasks),
                                      total = len(tasks),
                                      disable = not progress, leave = False,
                                      desc = 'Qr init') for x in out]
        init_fr = fr.copy()
        for k, (fr_k, Qr_k) in enumerate(init):
            if np.isfinite(Qr_k):
                init_fr[k], Qr[k] = fr_k, Qr_k
        if np.any(np.isfinite(Qr)):
            # Initializations below the robust floor count as failures
            low = Qr < q_floor(Qr)
            Qr[low] = np.nan
            moved = np.abs(init_fr - fr)
            fr[~low] = init_fr[~low]
            # Two guesses that initialized onto the same resonance: the one
            # that moved further goes back to its own guess, so the joint
            # fits can separate them
            o = np.argsort(fr)
            lw = fr[o] / np.where(np.isfinite(Qr[o]), Qr[o],
                                  np.nanmedian(Qr))
            for k in np.flatnonzero(np.diff(fr[o]) < dup_lw * lw[1:]):
                i, j = o[k], o[k + 1]
                back = i if moved[i] > moved[j] else j
                fr[back] = np.sort(fr0)[back]
            bounds[0] = max(bounds[0], q_floor(Qr))
        update_q_win()
        lw_guess = np.full(n, np.nan)
        lw_guess[guess_idx] = fr / q_win

        # 2. Regroup by linewidth and refit with external poles
        for _ in range(n_pass):
            # Dropped (pinned) guesses get another single fit once their
            # neighbors have improved; they rejoin if it succeeds
            retry = np.flatnonzero(pinned & ~active)
            for k in retry:
                fit_unit(np.array([k]))
            active[retry[~pinned[retry]]] = True
            for g in units(q_win):
                fit_unit(g)
            active[pinned] = False
            update_q_win()
        mark_duplicates()

        # 3. Final fits per unit, all from the same state
        opts = dict(add_poles = add_poles, fit_nonlinear = fit_nonlinear,
                    downward = downward, dup_lw = dup_lw)
        snap = {k: (v.copy() if isinstance(v, (np.ndarray, list)) else v)
                for k, v in st.items()}
        unit_list = units(q_win)
        if pool is None:
            results = [_final_unit(f, z, snap, g, opts)
                       for g in tqdm(unit_list, disable = not progress,
                                     leave = False, desc = 'Fitting')]
        else:
            tasks = [(snap, c, opts)
                     for c in _chunks(unit_list, 4 * n_workers)]
            results = [x for out in tqdm(pool.map(_final_task, tasks),
                                         total = len(tasks),
                                         disable = not progress,
                                         leave = False, desc = 'Fitting')
                       for x in out]
    finally:
        if pool is not None:
            pool.shutdown()
    rows = []
    gi = -1
    for unit in results:
        for grows, dropped in unit:
            gi += 1
            for k, near in dropped:
                active[k] = False
                dup_of[k] = near
            for r in grows:
                s = r.pop('sorted_idx')
                gidx = guess_idx[s] if s >= 0 else -1
                rows.append(dict(r, guess_idx = gidx, group = gi,
                                 fr0 = fr0[gidx] if gidx >= 0 else np.nan))
    # Fits from different groups that converged onto the same resonance:
    # keep the higher SNR
    fit_rows = sorted([r for r in rows if r['error'] == ''
                       and np.isfinite(r.get('fr', np.nan))],
                      key = lambda r: r['fr'])
    prev = None
    for r in fit_rows:
        if prev is not None and prev['group'] != r['group']:
            lw = min(prev['fr'] / prev['Qr'], r['fr'] / r['Qr'])
            if r['fr'] - prev['fr'] < dup_lw * lw:
                if np.nan_to_num(prev['snr']) < np.nan_to_num(r['snr']):
                    prev['error'] = 'duplicate fit'
                else:
                    r['error'] = 'duplicate fit'
                    continue
        prev = r
    for k in np.flatnonzero(~active & pinned):
        rows.append(dict(guess_idx = guess_idx[k], fr0 = fr0[guess_idx[k]],
                         duplicate_of = -1, error = 'Qr at a bound'))
    for k in np.flatnonzero(~active & ~pinned):
        dup = guess_idx[dup_of[k]] if dup_of[k] >= 0 else -1
        rows.append(dict(guess_idx = guess_idx[k], fr0 = fr0[guess_idx[k]],
                         duplicate_of = dup, error = ''))
    cols = ['guess_idx', 'fr0', 'group', 'b_ok', 'a_fr', 'a_Qr', 'a_amp',
            'a_at_bound', 'fr', 'Qr', 'amp', 'phi', 'fr_err', 'Qr_err',
            'amp_err', 'phi_err', 'nl_a', 'nl_a_err', 'nrmse', 'snr',
            'sigma_noise', 'duplicate_of', 'error', 'gain_coeffs',
            'gain_fc', 'gain_fs']
    df = pd.DataFrame(rows, columns = cols)
    df['recovered'] = False
    return df, lw_guess, tuple(bounds)

def _recover_drifted(
    f, z, df, lw_guess, n_lw = 8., ext_lw = 30., max_move_lw = 1.,
    bound_lw = 0.5, min_sep_lw = 0.3, min_dchi2 = 50.,
    amp_range = (0.03, 1.2), Qr_bounds = (1e2, 1e7)
):
    """
    Recover listed resonances whose fit drifted onto a neighbor.

    A guess whose fit ended more than ``max_move_lw`` of its linewidths from
    the listed frequency (or merged into another fit), with no fit left
    within ``bound_lw`` of it, may have lost its own resonance to a
    stronger neighbor. Its window is refit with the fits there (free) plus
    one pole held within +-``bound_lw`` linewidths of the listed frequency,
    the other fits as fixed factors. The pole is kept if the fit converged
    with no Qr at a bound, the chi-squared improvement over the fit without
    it exceeds ``min_dchi2``, its amp is in ``amp_range``, and it is at
    least ``min_sep_lw`` linewidths from the other poles. The drifted fit
    is then no longer tied to the listed frequency (``guess_idx`` -1).

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21 data.
    df (pd.DataFrame): ``_refine`` table.
    lw_guess (np.array): step-1 linewidth (Hz) per guess.
    n_lw (float): window half-width in linewidths. Default 8.
    ext_lw (float): fits within this many of their own linewidths of the
        window enter as fixed factors. Default 30.
    max_move_lw (float): drift threshold in linewidths. Default 1.
    bound_lw (float): the recovered pole stays within this many linewidths
        of the listed frequency. Default 0.5.
    min_sep_lw (float): minimum separation from the other poles, in
        linewidths. Default 0.3.
    min_dchi2 (float): minimum chi-squared improvement (in units of the
        refit's noise). Default 50.
    amp_range (tuple): allowed amp of the recovered pole. Default
        (0.03, 1.2).
    Qr_bounds (tuple): Qr bounds of the refit. Default (1e2, 1e7).

    Returns:
    df (pd.DataFrame): the table with recovered rows added (``recovered``
        True, Stage A values, ``b_ok`` False) and the drifted rows
        relabeled.
    """
    from ..res.multifit import _poly_basis
    ok_fit = (df.error == '') & df.fr.notna() & (df.duplicate_of < 0)
    fits = df[ok_fit]
    F = {k: list(fits[k].values.astype(float))
         for k in ('fr', 'Qr', 'amp', 'phi')}
    new_rows, drop, relabel = [], [], []
    group_next = int(np.nanmax(df.group)) + 1 if df.group.notna().any() else 0
    for idx, r in df[df.guess_idx >= 0].iterrows():
        g, fc = int(r.guess_idx), float(r.fr0)
        lw0 = lw_guess[g] if g < len(lw_guess) else np.nan
        if not (np.isfinite(fc) and np.isfinite(lw0)):
            continue
        if r.error not in ('', 'duplicate fit'):
            continue
        if r.duplicate_of >= 0 or not np.isfinite(r.fr):
            keep = df.fr[(df.guess_idx == r.duplicate_of) & df.fr.notna()]
            pos = keep.values[0] if len(keep) else np.nan
        else:
            pos = r.fr
        if np.isfinite(pos) and abs(pos - fc) <= max_move_lw * lw0:
            continue
        ffr, fq = np.array(F['fr']), np.array(F['Qr'])
        if np.any(np.abs(ffr - fc) < bound_lw * lw0):
            continue  # the listed frequency is covered
        i0, i1 = np.searchsorted(f, [fc - n_lw * lw0, fc + n_lw * lw0])
        if i1 - i0 < 20:
            continue
        fw, zw = f[i0:i1], z[i0:i1]
        flw = ffr / fq
        inside = (ffr > fw[0] - 2 * flw) & (ffr < fw[-1] + 2 * flw)
        ext = (~inside & (ffr + ext_lw * flw > fw[0])
               & (ffr - ext_lw * flw < fw[-1]))
        e = dict(ext_fr = ffr[ext], ext_Qr = fq[ext],
                 ext_amp = np.clip(np.array(F['amp'])[ext], 0, 1),
                 ext_phi = np.clip(np.array(F['phi'])[ext], -1.5, 1.5))
        deg = gain_degree(fw[-1] - fw[0])
        n_in = int(inside.sum())
        try:
            if n_in:
                a = fit_poles(fw, zw, ffr[inside], fq[inside], gain_deg = deg,
                              Qr_bounds = Qr_bounds, **e)
                rss_a = a['rss']
            else:
                # Gain polynomial only (times the fixed neighbors)
                x = (fw - 0.5 * (fw[0] + fw[-1])) / (0.5 * (fw[-1] - fw[0]))
                basis = _poly_basis(x, deg) * product_s21(
                    fw, e['ext_fr'], e['ext_Qr'], e['ext_amp'],
                    e['ext_phi'])[:, None]
                c = np.linalg.lstsq(basis, zw, rcond = None)[0]
                rss_a = float(np.sum(np.abs(zw - basis @ c) ** 2))
            fb = np.full((n_in + 1, 2), np.nan)
            fb[-1] = (fc - bound_lw * lw0, fc + bound_lw * lw0)
            b = fit_poles(fw, zw, np.r_[ffr[inside], fc],
                          np.r_[fq[inside], fc / lw0], gain_deg = deg,
                          Qr_bounds = Qr_bounds, fr_bounds = fb, **e)
        except (np.linalg.LinAlgError, ValueError):
            continue
        dchi2 = (rss_a - b['rss']) / b['sigma'] ** 2
        others = b['fr'][:-1]
        sep_ok = (len(others) == 0
                  or np.min(np.abs(others - b['fr'][-1])) > min_sep_lw * lw0)
        if not (b['success'] and not np.any(b['at_bound'])
                and dchi2 > min_dchi2 and sep_ok
                and amp_range[0] <= b['amp'][-1] <= amp_range[1]):
            continue
        sig_n = resid_noise_sigma(fw, zw - multifit_model(fw, b), b['fr'],
                                  b['fr'] / b['Qr'])
        x_k = (b['fr'][-1] - b['fc']) / b['fs']
        g_k = abs(np.polyval(b['gain_coeffs'], x_k))
        row = dict(guess_idx = g, fr0 = fc, group = group_next, b_ok = False,
                   a_at_bound = False, amp_err = np.nan, phi_err = np.nan,
                   nl_a = 0., nl_a_err = np.nan, nrmse = b['nrmse'],
                   snr = b['amp'][-1] * g_k / sig_n, sigma_noise = sig_n,
                   duplicate_of = -1, error = '', recovered = True,
                   gain_coeffs = np.asarray(b['gain_coeffs']),
                   gain_fc = b['fc'], gain_fs = b['fs'])
        for key in ('fr', 'Qr', 'amp'):
            row['a_' + key] = b[key][-1]
        for key in ('fr', 'Qr', 'amp', 'phi', 'fr_err', 'Qr_err'):
            row[key] = b[key][-1]
        new_rows.append(row)
        group_next += 1
        for k in ('fr', 'Qr', 'amp', 'phi'):
            F[k].append(b[k][-1])
        if np.isfinite(r.fr):
            relabel.append(idx)
        else:
            drop.append(idx)
    if not new_rows:
        return df
    df = df.copy()
    df.loc[relabel, 'guess_idx'] = -1
    df = df.drop(index = drop)
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index = True)

def _sweep_snr(table, window = 51):
    """
    Recompute SNR against a sweep-level noise estimate.

    Per-window noise estimates are unreliable both ways (second differences
    miss correlated noise; fit residuals are inflated by misfit in crowded
    windows). The noise is roughly constant across a sweep, so a running
    median over ``window`` neighboring fit groups of the per-group
    ``sigma_noise`` (from groups whose fits look healthy) is used,
    interpolated in frequency.

    Parameters:
    table (pd.DataFrame): ``_refine`` table.
    window (int): running-median window in fit groups. Default 51.

    Returns:
    table (pd.DataFrame): with ``sigma_sweep`` and ``snr`` rescaled to it.
    """
    from scipy.ndimage import median_filter
    table = table.copy()
    table['sigma_sweep'] = np.nan
    ok = table.fr.notna() & np.isfinite(table.sigma_noise.astype(float)) \
        & (table.sigma_noise.astype(float) > 0)
    if ok.sum() == 0:
        return table
    t = table[ok]
    healthy = ((t.amp > 0) & (t.amp <= 1.2)
               & ~t.a_at_bound.astype('boolean').fillna(True).astype(bool))
    good = healthy.groupby(t.group).transform('all')
    g = t[good].groupby('group').agg(fr = ('fr', 'median'),
                                     s = ('sigma_noise', 'first'))
    if len(g) == 0:
        g = t.groupby('group').agg(fr = ('fr', 'median'),
                                   s = ('sigma_noise', 'first'))
    g = g.sort_values('fr')
    w = int(min(window, len(g))) | 1
    ref = median_filter(g.s.values.astype(float), w, mode = 'reflect')
    sig = np.interp(t.fr.values, g.fr.values, ref)
    table.loc[ok, 'sigma_sweep'] = sig
    table.loc[ok, 'snr'] = table.loc[ok, 'snr'] * t.sigma_noise / sig
    return table

FIT_COLUMNS = ['fres_input', 'fr', 'fr_err', 'Qr', 'Qr_err', 'Qi', 'Qi_err',
               'Qc', 'Qc_err', 'amp', 'amp_err', 'phi', 'phi_err', 'a',
               'a_err', 'snr', 'nrmse', 'fit_status', 'flags', 'group']

def fit_full_sweep(
    f, z, fres, Qr0 = None, fit_nonlinear = False, downward = True,
    kappa_max = KAPPA_MAX, misfit_max = MISFIT_MAX, knot_spacing = None,
    progress = False, **kwargs
):
    """
    Fit every listed resonance of a full VNA sweep, then the sweep's gain.

    Every frequency in ``fres`` gets one row, in the same order; no
    resonances are added or removed. Where a fit fails, the row keeps the
    listed frequency with NaN parameters, and ``fit_status`` says why.

    Parameters:
    f (np.array): frequencies in Hz, ascending, no repeats
        (``sweep_info.prepare_full_sweep``).
    z (np.array): complex S21 data.
    fres (np.array): resonance frequencies in Hz (e.g. ``fres_manual``).
    Qr0 (np.array or None): rough Qr per listed frequency (NaN: unknown),
        e.g. from a previous fit or the CNN finder. Default None.
    fit_nonlinear (bool): If True, also fit each resonator's nonlinearity
        a (fr is then the low-power resonance frequency). Default False.
    downward (bool): sweep direction for the nonlinear model. Default True
        (downward sweep); set False for upward sweeps.
    kappa_max, misfit_max (float or None): flag thresholds
        (``reliability_flags``; None: off). Defaults 0.2, 0.05.
    knot_spacing (float or None): gain spline knot spacing in Hz (None:
        automatic, ``gain.fit_gain_spline``).
    progress (bool): If True, show progress bars. Default False.
    **kwargs: passed to the fitting loop (``n_lw`` 8, ``lw_max`` 3 MHz,
        ``n_pass`` 3, ``ext_lw`` 30, ``max_group`` 8, ``Qr_bounds``,
        ``min_Qr_frac`` 0.2, ``dup_lw`` 0.1, ``max_pts_lw`` 10,
        ``max_pts_lw_final`` 20, ``n_workers`` (processes; default up to
        4, the result does not depend on it); see ``_refine``).

    Returns:
    res (pd.DataFrame): one row per listed frequency, columns
        ``FIT_COLUMNS``: ``fres_input``, ``fr``, ``Qr``, ``Qi``, ``Qc``,
        ``amp``, ``phi``, ``a`` (0 unless ``fit_nonlinear``) and their
        ``_err``, ``snr`` (amp |gain| / noise), ``nrmse`` (of the fit
        group), ``fit_status`` ('ok'; 'stage_a': Stage B failed, Stage A
        values, no amp or phi errors; or 'failed: <reason>', e.g. 'Qr at a
        bound', 'duplicate fit', 'merged with a neighbor', 'unphysical
        amp', 'fit moved to another resonance': fr is the listed
        frequency, the rest NaN), ``flags`` (``reliability_flags``;
        the values are kept), ``group`` (resonances fit together).
    gain (np.array): smooth complex gain at ``f``.
    info (dict): ``gain`` (``gain.fit_gain_spline`` info),
        ``n_recovered`` (listed resonances recovered from a neighbor),
        ``narrow_core`` (np.array, bool, per row).
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    fres = np.asarray(fres, dtype = np.float64)
    n_lw = kwargs.get('n_lw', 8.)
    ext_lw = kwargs.get('ext_lw', 30.)
    if len(fres):
        table, lw_guess, q_bounds = _refine(
            f, z, fres, Qr0, fit_nonlinear, downward, progress = progress,
            **kwargs)
        table = _recover_drifted(f, z, table, lw_guess, n_lw = n_lw,
                                 ext_lw = ext_lw, Qr_bounds = q_bounds)
        table = _sweep_snr(table)
    else:
        table = pd.DataFrame(columns = ['guess_idx'])
    rows = []
    for k, fk in enumerate(fres):
        row = dict(fres_input = fk, fr = fk)
        status = 'failed: not fit'
        r = table[table.guess_idx == k]
        if len(r):
            r = r.iloc[0]
            lw = r.fr / r.Qr if np.isfinite(r.Qr) else np.nan
            if r.error:
                status = f'failed: {r.error[:40]}'
            elif r.duplicate_of >= 0:
                status = 'failed: merged with a neighbor'
            elif not np.isfinite(r.fr):
                status = 'failed: not fit'
            elif bool(r.a_at_bound):
                status = 'failed: Qr at a bound'
            elif not 0 < r.amp <= 1.2:
                status = 'failed: unphysical amp'
            elif abs(r.fr - fk) > 2 * lw:
                status = 'failed: fit moved to another resonance'
            else:
                status = 'ok' if bool(r.b_ok) else 'stage_a'
                row.update(fr = r.fr, fr_err = r.fr_err, Qr = r.Qr,
                           Qr_err = r.Qr_err, amp = r.amp,
                           amp_err = r.amp_err, phi = r.phi,
                           phi_err = r.phi_err, a = r.nl_a,
                           a_err = r.nl_a_err, snr = r.snr, nrmse = r.nrmse,
                           group = r.group, sigma = r.sigma_sweep)
        row['fit_status'] = status
        rows.append(row)
    res = pd.DataFrame(rows, columns = FIT_COLUMNS + ['sigma'])
    for c in res.columns:
        if c not in ('fit_status', 'flags'):
            res[c] = res[c].astype(float)
    res['Qi'], res['Qc'], res['Qi_err'], res['Qc_err'] = qi_qc(
        res.Qr, res.amp, res.Qr_err, res.amp_err)
    res['group'] = res.group.fillna(-1).astype(int)
    # Gain of the whole sweep, from the fitted resonances
    ok = np.isfinite(res.Qr.values)
    p = {c: res[c].values[ok] for c in ('fr', 'Qr', 'amp', 'phi', 'a')}
    gain, ginfo = fit_gain_spline(
        f, z, p['fr'], p['Qr'], p['amp'], p['phi'],
        a = p['a'] if fit_nonlinear else None, downward = downward,
        knot_spacing = knot_spacing, exclude = res.fr.values[~ok])
    # Reliability flags
    o = np.argsort(res.fr.values[ok])
    idx = np.flatnonzero(ok)[o]
    excess = np.full(len(res), np.nan)
    if len(idx):
        sig = res.sigma.values[idx]
        sig = np.where(np.isfinite(sig), sig, np.nanmedian(sig))
        excess[idx] = residual_excess(
            f, z, *(res[c].values[idx] for c in ('fr', 'Qr', 'amp', 'phi')),
            sig)
    mag = np.abs(z / gain)
    narrow = np.array([narrow_core(f, mag, *v) for v in
                       res[['fr', 'Qr', 'amp', 'phi']].values], dtype = bool)
    flags, _ = reliability_flags(
        coupling_error(res.fr.values, res.Qr.values, res.amp.values), excess,
        res.snr.values, narrow, res.a.values if fit_nonlinear else None,
        kappa_max, misfit_max)
    res['flags'] = flags
    n_rec = int(table.recovered.astype(bool).sum()) if 'recovered' in table \
        else 0
    return res[FIT_COLUMNS], gain, dict(gain = ginfo, n_recovered = n_rec,
                                        narrow_core = narrow)

def _jsonable(v):
    """
    Convert a value to plain Python types for zarr attributes.

    Parameters:
    v (object): value (dicts, lists, tuples, numpy scalars and arrays).

    Returns:
    out (object): JSON-serializable copy (non-finite floats as None).
    """
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.bool_):
        return bool(v)
    return v

def run_full_sweep_fit(
    f, z, zarr_grp, fit_nonlinear = False, downward = True,
    overwrite = False, progress = True, **kwargs
):
    """
    Fit the resonances in ``fres_manual`` of a full VNA sweep, print a
    description of the sweep, and save the results.

    The sweep is prepared (``sweep_info.prepare_full_sweep``), checked for
    usable phase, and fit (``fit_full_sweep``). Everything is saved in
    ``zarr_grp['full_sweep_fit']``:
        fres_input, fr, fr_err, Qr, Qr_err, Qi, Qi_err, Qc, Qc_err, amp,
        amp_err, phi, phi_err, a, a_err, snr, nrmse (float arrays, one per
        listed frequency, in the order of ``fres_manual``), fit_status,
        flags (str arrays), group (int array);
        gain/f, gain/gain: the analyzed frequencies and the smooth complex
        gain there (gain-removed S21 = z / gain after the same preparation);
        attrs: created, citkid_version, source, fit_nonlinear, downward,
        settings, sweep_metrics, notes, range, banks, q_variation (whether
        Qi, Qc, Qr vary with frequency or with position in the banks),
        report.

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 data.
    zarr_grp (zarr.Group or str): the zarr group (or its path) used by the
        res finders, with 'fres_manual'.
    fit_nonlinear (bool): If True, also fit each resonator's nonlinearity.
        Default False.
    downward (bool): sweep direction for the nonlinear model. Default True
        (downward sweep); set False for upward sweeps.
    overwrite (bool): If True, replace an existing 'full_sweep_fit'.
        Default False (raise instead).
    progress (bool): If True (default), show progress bars.
    **kwargs: passed to ``fit_full_sweep``.

    Returns:
    res (pd.DataFrame): ``fit_full_sweep`` table.
    report (str): the printed description.

    Raises:
    KeyError: if the group has no 'fres_manual'.
    FileExistsError: if 'full_sweep_fit' exists and ``overwrite`` is False.
    ValueError: if the sweep has no usable phase.
    """
    from .. import __version__
    t0 = time.time()
    if isinstance(zarr_grp, (str, os.PathLike)):
        zarr_grp = zarr.open_group(os.path.expanduser(str(zarr_grp)),
                                   mode = 'a')
    if 'fres_manual' not in zarr_grp:
        raise KeyError("zarr_grp has no 'fres_manual': run "
                       "run_res_finder_manual first")
    if 'full_sweep_fit' in zarr_grp and not overwrite:
        raise FileExistsError("zarr_grp already has 'full_sweep_fit'; pass "
                              "overwrite = True to replace it")
    fres = np.asarray(zarr_grp['fres_manual'][:], dtype = np.float64)
    f, z, corr_length = prepare_full_sweep(f, z)
    problem, pinfo = phase_problem(z)
    if problem:
        raise ValueError(f'the sweep has no usable phase ({problem}): it '
                         'cannot be fit')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res, gain, info = fit_full_sweep(
            f, z, fres, fit_nonlinear = fit_nonlinear, downward = downward,
            progress = progress, **kwargs)
    metrics = sweep_metrics(f, z, gain, res, corr_length)
    notes = sweep_notes(metrics, int(info['narrow_core'].sum()), len(res))
    report, rinfo = describe_full_sweep(f, res, metrics, notes)
    print(report)
    if 'full_sweep_fit' in zarr_grp:
        del zarr_grp['full_sweep_fit']
    out = zarr_grp.create_group('full_sweep_fit')
    for c in FIT_COLUMNS:
        if c in ('fit_status', 'flags'):
            arr = out.create_array(c, shape = (len(res),), dtype = str,
                                   fill_value = '')
            arr[:] = res[c].values.astype(str)
        elif c == 'group':
            out.create_array(c, data = res[c].values.astype(np.int64))
        else:
            out.create_array(c, data = res[c].values.astype(np.float64))
    g = out.create_group('gain')
    g.create_array('f', data = f)
    g.create_array('gain', data = gain)
    settings = dict(kwargs, corr_length = corr_length,
                    runtime_s = round(time.time() - t0, 1),
                    n_recovered = info['n_recovered'], gain = info['gain'])
    out.attrs.update(_jsonable(dict(
        created = time.strftime('%Y-%m-%d %H:%M:%S'),
        citkid_version = __version__, source = 'fres_manual',
        fit_nonlinear = fit_nonlinear, downward = downward,
        settings = settings, sweep_metrics = metrics, notes = notes,
        range = rinfo['range'],
        banks = {k: v for k, v in rinfo['banks'].items()
                 if k in ('n_found', 'n_missing_inside', 'n_cut_low',
                          'n_cut_high', 'n_between', 'overlapping',
                          'period')},
        q_variation = rinfo['q'],
        report = report)))
    return res, report
