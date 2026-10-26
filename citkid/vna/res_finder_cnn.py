"""
CNN resonance finder for full VNA sweeps: an alternative to the
interactive auto finder (``run_res_finder_auto``) that writes the same
'fres_auto' starting list for the manual finder.

Steps (``find_resonances_cnn``): a trained network (``vna.cnn``) and the
classical matched filter (``vna.matched_filter``) find candidate
resonances; all candidates are fit together (as in ``vna.full_sweep``,
plus a search for resonances missing from the candidates); fits that look
like resonances are kept (``accept_candidates``), plus weak ones on a
significant dip (``rescue_weak``); close pairs are tested for one or two
resonances (``resolve_close_pairs``); and clear dips in |S21| without a
usable fit are added as frequency-only resonances
(``dip_only_resonances``). On 28 held-out real full VNA sweeps reviewed by
hand, this missed ~1.4% of the resonances and ~2.7% of what it found were
not on the reviewed lists. Needs PyTorch
(``environment-cnn.yml``, or ``pip install citkid[cnn]``).
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import zarr
from ..res.multifit import compare_single_double, gain_degree
from .dips import dip_significance, unexplained_dips
from .gain import fit_gain_spline
from .matched_filter import find_candidates_mf
from .sweep_info import prepare_full_sweep
from .full_sweep import _refine, _recover_drifted, _sweep_snr, _jsonable

_INSTALL = ("the CNN finder needs PyTorch: create the environment from "
            "environment-cnn.yml, or run 'mamba install -n citkid -c "
            "conda-forge pytorch=2.13.0 numpy=1.26.4'")

################################################################################
############################## Analysis functions ##############################
################################################################################
def merge_candidates(fr_a, Qr_a, fr_b, Qr_b, merge_lw = 0.3):
    """
    Select the candidates of a second detector that are not duplicates of
    the first's.

    Parameters:
    fr_a, Qr_a (np.array): first detector's candidates, fr ascending.
    fr_b, Qr_b (np.array): second detector's candidates.
    merge_lw (float): a second-detector candidate within this many
        linewidths (the larger of the two estimates) of a first-detector
        candidate is a duplicate. Default 0.3.

    Returns:
    new (np.array, bool): True for the second detector's candidates to add.
    """
    fr_b = np.asarray(fr_b, dtype = float)
    if len(fr_a) == 0 or len(fr_b) == 0:
        return np.ones(len(fr_b), dtype = bool)
    j = np.clip(np.searchsorted(fr_a, fr_b), 1, max(len(fr_a) - 1, 1))
    if len(fr_a) > 1:
        j = np.where(np.abs(fr_a[j - 1] - fr_b) < np.abs(fr_a[j] - fr_b),
                     j - 1, j)
    else:
        j = np.zeros(len(fr_b), int)
    lw = np.maximum(fr_a[j] / Qr_a[j], fr_b / np.asarray(Qr_b, float))
    return np.abs(fr_a[j] - fr_b) > merge_lw * lw

def accept_candidates(
    table, f, min_snr = 5., amp_min = 0.03, amp_max = 1.2, min_pts_lw = 1.
):
    """
    Select the fits of the candidates that look like resonances.

    Parameters:
    table (pd.DataFrame): candidate fit table (``vna.full_sweep._refine``
        rows: ``fr``, ``Qr``, ``amp``, ``phi``, ``snr``, ``error``,
        ``duplicate_of``, ``a_at_bound``).
    f (np.array): the fitted frequencies in Hz, ascending.
    min_snr (float): minimum SNR (amp |gain| / noise). Default 5.
    amp_min (float): minimum amp. Default 0.03 (a 0.26 dB dip; gain
        wiggles make shallower dips that pass the SNR cut).
    amp_max (float): maximum amp. Default 1.2.
    min_pts_lw (float): minimum fitted linewidth in frequency steps
        (narrower fits are unresolved, usually junk). Default 1.

    Returns:
    keep (np.array, bool): True for rows that were fit (no error, not a
        duplicate, Qr not at a bound), with amp in [``amp_min``,
        ``amp_max``], SNR >= ``min_snr``, a linewidth of at least
        ``min_pts_lw`` steps, and not pinned at both the amp (1) and phi
        (+-1.5) bounds (a sign of model trouble, typical of junk fits).
    """
    fr = table.fr.values.astype(float)
    step = np.interp(fr, f[1:], np.diff(f))
    bound = table.a_at_bound.astype('boolean').fillna(True).astype(bool)
    with np.errstate(invalid = 'ignore'):
        pinned = (table.amp >= 0.995) & (table.phi.abs() >= 1.49)
        ok = (np.isfinite(fr) & (table.error == '').values
              & (table.duplicate_of < 0).values & ~bound.values
              & ((table.amp >= amp_min) & (table.amp <= amp_max)
                 & (table.snr >= min_snr) & ~pinned).values
              & (fr / table.Qr.values.astype(float) >= min_pts_lw * step))
    return ok

def rescue_weak(
    f, z, table, min_snr = 5., min_snr_lw = 5., min_sig = 3.,
    shoulder_lw = 1.5
):
    """
    Accept candidates that fail only the per-point SNR cut but show a
    significant dip.

    The per-point SNR ignores how many points cover a resonance: at 20-50
    points per linewidth a dip with per-point SNR 2 is obvious, and on real
    sweeps with correlated or phase noise the complex residual overstates
    the noise. A fit is rescued if it passes the other
    ``accept_candidates`` tests, its SNR integrated over a linewidth (snr x
    sqrt(points per linewidth)) is at least ``min_snr_lw``, its |S21| dip
    (gain removed with the fit's local gain polynomial) is significant
    against the surrounding data (``dips.dip_significance`` >=
    ``min_sig``), and it is not within ``shoulder_lw`` linewidths of an
    accepted resonance.

    Parameters:
    f (np.array): frequencies in Hz (as fit).
    z (np.array): complex S21 data.
    table (pd.DataFrame): candidate fit table with ``accepted``.
    min_snr (float): the per-point SNR cut. Default 5.
    min_snr_lw (float): minimum integrated SNR. Default 5.
    min_sig (float): minimum dip significance. Default 3.
    shoulder_lw (float): exclusion around accepted resonances, in the
        larger of the two linewidths. Default 1.5.

    Returns:
    table (pd.DataFrame): updated in place: rescued rows get ``accepted``
        True; columns ``rescued`` (bool) and ``dip_sig`` (significance of
        the rows tested, NaN otherwise).
    """
    table['rescued'] = False
    table['dip_sig'] = np.nan
    acc = table.accepted.values.astype(bool)
    fr_all = table.fr.values.astype(float)
    lw = fr_all / table.Qr.values.astype(float)
    step = np.interp(fr_all, f[1:], np.diff(f))
    with np.errstate(invalid = 'ignore'):
        snr_lw = table.snr.values * np.sqrt(lw / step)
    cand = (accept_candidates(table, f, -np.inf) & ~acc
            & (table.snr.values < min_snr) & (snr_lw >= min_snr_lw))
    if not cand.any():
        return table
    fa, lwa = fr_all[acc], lw[acc]
    valid = np.isfinite(lw) & (lw > 0) & np.isfinite(fr_all)
    excl = (fr_all[valid], lw[valid])
    for i in np.flatnonzero(cand):
        fr_i, lw_i = fr_all[i], lw[i]
        if len(fa) and np.any(np.abs(fa - fr_i)
                              < shoulder_lw * np.maximum(lwa, lw_i)):
            continue
        row = table.iloc[i]
        i0, i1 = np.searchsorted(f, [fr_i - 20 * lw_i, fr_i + 20 * lw_i])
        fw = f[i0:i1]
        g = np.abs(np.polyval(row.gain_coeffs, (fw - row.gain_fc)
                              / row.gain_fs))
        sig, _ = dip_significance(fw, np.abs(z[i0:i1]) / g, fr_i, lw_i,
                                  exclude = excl)
        table.iloc[i, table.columns.get_loc('dip_sig')] = sig
        if np.isfinite(sig) and sig >= min_sig:
            table.iloc[i, table.columns.get_loc('accepted')] = True
            table.iloc[i, table.columns.get_loc('rescued')] = True
    return table

def _plausible_single(single, pair, amp_min = 0.03, amp_max = 1.2,
                      q_factor = 3.):
    """
    Check that a merged single-resonance fit can replace a pair's fits.

    Parameters:
    single (dict): ``fit_resonators`` result for one resonator.
    pair (pd.DataFrame): the two rows being merged (``fr``, ``Qr``).
    amp_min, amp_max (float): allowed amp range. Defaults 0.03, 1.2.
    q_factor (float): the single Qr must be within this factor of the
        pair's Qr range. Default 3.

    Returns:
    ok (bool): True if the single fit converged, its amp is in range, its
        Qr is near the pair's, and its fr lies between the pair's
        frequencies (+- one linewidth).
    """
    fr, Qr, amp = single['fr'][0], single['Qr'][0], single['amp'][0]
    lw = (pair.fr / pair.Qr).max()
    return bool(single['success'] and amp_min <= amp <= amp_max
                and pair.Qr.min() / q_factor <= Qr <= pair.Qr.max() * q_factor
                and pair.fr.min() - lw <= fr <= pair.fr.max() + lw)

def resolve_close_pairs(
    f, z, table, max_sep_lw = 0.5, n_lw = 8., ext_lw = 30., min_dbic = 1000.,
    max_split_lw = 3.
):
    """
    Decide single vs double for accepted resonances closer than
    ``max_sep_lw``.

    For each such pair, ``multifit.compare_single_double`` compares one
    nonlinear resonator with two linear ones. Both are kept only if the
    double model is strongly favored (dbic > ``min_dbic``) with its poles
    at most ``max_split_lw`` linewidths apart; otherwise, or if the test
    fails, the pair is merged into one resonance (the higher-SNR row, given
    the single-model parameters if they are plausible). The large
    ``min_dbic`` allows for model mismatch at high SNR on real data.
    Repeats until no untested close pair remains, so chains of
    near-duplicates are merged.

    Parameters:
    f (np.array): frequencies in Hz (sorted, no repeats).
    z (np.array): complex S21 data.
    table (pd.DataFrame): candidate fit table with ``accepted`` (unique
        index).
    max_sep_lw (float): pair threshold in linewidths of the broader one.
        Default 0.5.
    n_lw (float): fit-window half-width in linewidths. Default 8.
    ext_lw (float): other accepted resonances within ``ext_lw`` of their own
        linewidths of the window are fixed external factors. Default 30.
    min_dbic (float): minimum dbic to keep both. Default 1000.
    max_split_lw (float): maximum fitted pair separation, in linewidths,
        to keep both. Default 3.

    Returns:
    table (pd.DataFrame): updated in place: columns ``pair_dbic`` (NaN if
        not tested) and ``merged`` (True for the row that absorbed a
        partner); partners are un-accepted.
    """
    table['pair_dbic'] = np.nan
    table['merged'] = False
    tested = set()
    while True:
        acc = table.index[table.accepted].to_numpy()
        acc = acc[np.argsort(table.loc[acc, 'fr'].values)]
        fr = table.loc[acc, 'fr'].values.astype(float)
        lw = fr / table.loc[acc, 'Qr'].values.astype(float)
        close = [k for k in range(len(acc) - 1)
                 if fr[k + 1] - fr[k] < max_sep_lw * max(lw[k], lw[k + 1])
                 and (acc[k], acc[k + 1]) not in tested]
        if not close:
            break
        k = close[0]
        i, j = acc[k], acc[k + 1]
        tested.add((i, j))
        w = max(lw[k], lw[k + 1])
        c = 0.5 * (fr[k] + fr[k + 1])
        m = np.abs(f - c) < n_lw * w
        # Other accepted resonances whose tails reach the window enter as
        # fixed external factors
        o = np.setdiff1d(np.arange(len(acc)), [k, k + 1])
        o = o[(fr[o] + ext_lw * lw[o] > c - n_lw * w)
              & (fr[o] - ext_lw * lw[o] < c + n_lw * w)]
        ext = dict(
            ext_fr = fr[o], ext_Qr = table.loc[acc[o], 'Qr'].values,
            ext_amp = np.clip(table.loc[acc[o], 'amp'].values, 0, 1),
            ext_phi = np.clip(table.loc[acc[o], 'phi'].values, -1.5, 1.5))
        try:
            r = compare_single_double(f[m], z[m], c, c / w,
                                      gain_deg = gain_degree(np.ptp(f[m])),
                                      ext = ext)
            dbic = r['dbic']
        except Exception:
            r, dbic = None, np.nan
        table.loc[[i, j], 'pair_dbic'] = dbic
        # Double only with strong evidence and an actual split
        double = (dbic > min_dbic) and r['sep_lw'] <= max_split_lw
        if not double:
            keep, drop = ((i, j) if table.loc[i, 'snr'] >= table.loc[j, 'snr']
                          else (j, i))
            if r is not None and _plausible_single(r['single'],
                                                   table.loc[[i, j]]):
                s = r['single']
                for col in ('fr', 'Qr', 'amp', 'phi'):
                    table.loc[keep, col] = s[col][0]
                table.loc[keep, 'nl_a'] = s['a'][0]
            table.loc[keep, 'merged'] = True
            table.loc[drop, 'accepted'] = False
    return table

def dip_only_resonances(
    f, mag, fr, Qr, table, min_sig_dip = 3., min_sig_rejected = 15.,
    near_lw = 1.5, min_near = 20e3, strong_db = 15., max_strong_frac = 0.1
):
    """
    Find resonances that show as clear dips but have no accepted fit.

    Two kinds: clear dips in the gain-removed |S21| far from every accepted
    resonance (``dips.unexplained_dips``, significance >= ``min_sig_dip``
    at each dip's own width, or prominence >= ``strong_db``, unless more
    than max(10, ``max_strong_frac`` x the accepted count) dips qualify
    only that way: a sweep full of deep spikes), often resonances narrower
    than the frequency step; and rejected candidates (not duplicates, amp
    in range) on a dip with significance >= ``min_sig_rejected``, not
    within ``near_lw`` linewidths of an accepted resonance or a dip of the
    first kind. Against hand-checked real sweeps about half were on the
    lists and, by eye, about half of the rest were real resonances the
    lists missed.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    mag (np.array): |S21| with the gain removed, linear.
    fr, Qr (np.array): accepted resonances.
    table (pd.DataFrame): candidate fit table (``accepted``, ``fr``,
        ``fr0``, ``Qr``, ``amp``, ``error``, ``duplicate_of``).
    min_sig_dip (float): significance for unexplained dips. Default 3.
    min_sig_rejected (float): significance for rejected candidates.
        Default 15.
    near_lw (float): 'near' in linewidths. Default 1.5.
    min_near (float): smallest 'near' distance in Hz. Default 20 kHz.
    strong_db (float or None): see above. Default 15 dB. None: off.
    max_strong_frac (float): see above. Default 0.1.

    Returns:
    f_add (np.array): frequencies in Hz, ascending.
    kind (np.array, str): 'dip' or 'rejected fit on a dip', per entry.
    capped (bool): True if deep isolated spikes were ignored.
    """
    fr = np.asarray(fr, dtype = float)
    lw = fr / np.asarray(Qr, dtype = float)
    lw0 = np.nanmedian(lw) if np.isfinite(lw).any() \
        else 50 * np.median(np.diff(f))
    lw = np.where(np.isfinite(lw), lw, lw0)
    fd, _, info = unexplained_dips(
        f, mag, fr, Qr, min_sig = min_sig_dip, own_width = True,
        strong_db = strong_db,
        max_strong = max(10, max_strong_frac * np.isfinite(fr).sum()),
        return_info = True)
    add, kind = list(fd), ['dip'] * len(fd)
    if len(table):
        acc = table.accepted.values.astype(bool)
        err = table.error.fillna('').astype(str).values
        amp = table.amp.values.astype(float)
        cand = (~acc & (table.duplicate_of.values < 0)
                & ~((amp < 0.03) | (amp > 1.2))
                & ~np.char.startswith(err.astype(str), 'duplicate'))
        fx = table.fr.fillna(table.fr0).values.astype(float)
        best = {}
        for i in np.flatnonzero(cand & np.isfinite(fx)):
            c = fx[i]
            if len(fr) and np.any(np.abs(fr - c)
                                  < np.maximum(near_lw * lw, min_near)):
                continue
            if add and np.min(np.abs(np.array(add) - c)) \
                    < max(near_lw * lw0, min_near):
                continue
            q = table.Qr.values[i]
            w = c / q if np.isfinite(q) and q > 0 else lw0
            w = float(np.clip(w, 0.2 * lw0, 5 * lw0))
            sig = dip_significance(f, mag, c, w, exclude = (fr, lw))[0]
            if np.isfinite(sig) and sig >= min_sig_rejected:
                key = round(c / max(near_lw * lw0, min_near))
                if key not in best or sig > best[key][1]:
                    best[key] = (c, sig)
        for c, _ in best.values():
            add.append(c)
            kind.append('rejected fit on a dip')
    o = np.argsort(add)
    return (np.asarray(add, dtype = float)[o], np.asarray(kind)[o],
            info['capped'])

################################################################################
############################ Bookkeeping functions #############################
################################################################################
def find_resonances_cnn(
    f, z, model = None, model_path = None, threshold = None,
    classical = True, fit_candidates = True, min_snr = 5., n_workers = None,
    progress = False
):
    """
    Find the resonances of a full VNA sweep with the CNN (see the module
    description for the steps).

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 data.
    model (str, UNet1D, or None): released model name, a loaded network,
        or None for ``cnn.model.DEFAULT_MODEL``.
    model_path (str or None): local model file (instead of a released
        model). Default None.
    threshold (float or None): heatmap threshold for candidates. None: the
        model's own (0.3 for det_v2).
    classical (bool): If True (default), add the matched filter's
        candidates (threshold 12, sidelobes suppressed) that are not near
        a CNN candidate.
    fit_candidates (bool): If True (default), fit the candidates and keep
        the resonances as described above. If False, return the raw
        candidates.
    min_snr (float): minimum fitted SNR to accept. Default 5.
    n_workers (int or None): processes for the fits (``full_sweep``).
        Default None (up to 4).
    progress (bool): If True, show progress bars. Default False.

    Returns:
    fres (np.array): resonance frequencies in Hz, ascending.
    qres (np.array): their Qr (fitted; NaN for frequency-only dips; the
        detectors' rough estimate if not fit).
    table (pd.DataFrame): if fit, one row per fit (``vna.full_sweep``
        fit-table columns, ``score`` and ``source`` ('cnn' or 'classical')
        of its candidate, '' for resonances added by the fitter, and
        ``accepted``), plus one row per frequency-only dip (``source``
        'dip: <kind>'); else the candidates (``fr``, ``Qr``, ``score``,
        ``source``).
    """
    try:
        from .cnn.model import load_model
        from .cnn.infer import detect
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc
    net = model if hasattr(model, 'settings') else \
        load_model(model, model_path)
    s = net.settings
    f, z, _ = prepare_full_sweep(f, z)
    fr_c, q_c, score = detect(
        net, f, z, threshold = s.get('threshold', 0.3) if threshold is None
        else threshold, chunk_ratio = s.get('chunk_ratio', 1.1),
        overlap = s.get('overlap', 0.01), min_step = s.get('min_step', 1.5e-6))
    source = np.full(len(fr_c), 'cnn', dtype = object)
    if classical:
        fm, qm, sm = find_candidates_mf(f, z, threshold = 12.,
                                        suppress_sidelobes = True)
        new = merge_candidates(fr_c, q_c, fm, qm)
        fr_c = np.concatenate((fr_c, fm[new]))
        q_c = np.concatenate((q_c, qm[new]))
        score = np.concatenate((score, sm[new]))
        source = np.concatenate((source, np.full(new.sum(), 'classical',
                                                 dtype = object)))
        o = np.argsort(fr_c)
        fr_c, q_c, score, source = fr_c[o], q_c[o], score[o], source[o]
    if not fit_candidates or len(fr_c) == 0:
        return fr_c, q_c, pd.DataFrame(dict(fr = fr_c, Qr = q_c,
                                            score = score, source = source))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        table, lw_guess, q_bounds = _refine(f, z, fr_c, q_c, add_poles = True,
                                            n_workers = n_workers,
                                            progress = progress)
        table = _recover_drifted(f, z, table, lw_guess, Qr_bounds = q_bounds)
        table = _sweep_snr(table).reset_index(drop = True)
        gi = table.guess_idx.fillna(-1).astype(int).values
        table['score'] = [score[i] if i >= 0 else np.nan for i in gi]
        table['source'] = [source[i] if i >= 0 else '' for i in gi]
        table['accepted'] = accept_candidates(table, f, min_snr)
        table = rescue_weak(f, z, table, min_snr)
        table = resolve_close_pairs(f, z, table)
        acc = table[table.accepted]
        # Clear dips without a usable fit, on the gain-removed sweep
        gain = fit_gain_spline(f, z, acc.fr.values, acc.Qr.values,
                               acc.amp.values, acc.phi.values)[0]
        add, kind, _ = dip_only_resonances(f, np.abs(z / gain),
                                           acc.fr.values, acc.Qr.values,
                                           table)
    if len(add):
        dips = pd.DataFrame(dict(fr = add, Qr = np.nan, accepted = True,
                                 source = [f'dip: {k}' for k in kind]))
        table = pd.concat([table, dips], ignore_index = True)
    acc = table[table.accepted.astype(bool)].sort_values('fr')
    return acc.fr.values.astype(float), acc.Qr.values.astype(float), table

def run_res_finder_cnn(
    f, z, zarr_grp, model = None, model_path = None, threshold = None,
    classical = True, fit_candidates = True, n_workers = None,
    overwrite = False, progress = True
):
    """
    Find the resonances of a full VNA sweep with the CNN and save them as
    the starting list for ``run_res_finder_manual``.

    Writes to ``zarr_grp``: 'fres_auto' (resonance frequencies in Hz, as
    from ``run_res_finder_auto``), 'qres_auto' (their Qr; NaN for
    frequency-only dips), and a 'res_finder_auto' subgroup whose attributes
    record the finder ('cnn'), model, settings, the number of resonances
    found, and the time.

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21 data.
    zarr_grp (zarr.Group or str): the zarr group (or its path) for the res
        finders.
    model (str or None): released model name. Default None (the default
        model).
    model_path (str or None): local model file instead. Default None.
    threshold (float or None): heatmap threshold (None: the model's).
    classical (bool): If True (default), also use the matched filter's
        candidates.
    fit_candidates (bool): If True (default), fit the candidates and keep
        the resonances (see ``find_resonances_cnn``); if False, save the
        raw candidates.
    n_workers (int or None): processes for the fits. Default None (up to
        4).
    overwrite (bool): If True, replace an existing 'fres_auto'. Default
        False (raise instead).
    progress (bool): If True (default), show progress bars.

    Returns:
    fres (np.array): resonance frequencies in Hz.

    Raises:
    FileExistsError: if 'fres_auto' exists and ``overwrite`` is False.
    ImportError: if PyTorch is not installed.
    """
    t0 = time.time()
    if isinstance(zarr_grp, (str, os.PathLike)):
        zarr_grp = zarr.open_group(os.path.expanduser(str(zarr_grp)),
                                   mode = 'a')
    if 'fres_auto' in zarr_grp and not overwrite:
        raise FileExistsError("zarr_grp already has 'fres_auto'; pass "
                              "overwrite = True to replace it")
    fres, qres, table = find_resonances_cnn(
        f, z, model, model_path, threshold, classical, fit_candidates,
        n_workers = n_workers, progress = progress)
    for name, v in (('fres_auto', fres), ('qres_auto', qres)):
        zarr_grp.create_array(name, data = np.asarray(v, dtype = np.float64),
                              overwrite = True)
    if 'res_finder_auto' in zarr_grp:
        del zarr_grp['res_finder_auto']
    g = zarr_grp.create_group('res_finder_auto')
    g.attrs.update(_jsonable(dict(
        finder = 'cnn', model = model_path or model or 'default',
        threshold = threshold, classical = classical,
        fit_candidates = fit_candidates, n_found = len(fres),
        created = time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s = round(time.time() - t0, 1))))
    print(f'CNN finder: {len(fres)} resonances ({time.time() - t0:.0f} s); '
          'saved as fres_auto')
    return fres
