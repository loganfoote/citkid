"""
Plan the tones of a tuning: which resonators get a tone, and how the
tones are split into tone sets when they don't all fit in the modules'
bandwidth at once (see ``documents/daq_layout.md``, section 7).

Analysis code: these functions only compute; saving the plan is done by
the procedures (``citkid.crs.procedures``, ``citkid.daq_layout``).
"""
import numpy as np

PAIRS_OPTIONS = ('drop_both', 'keep_stronger')
EXTENDED_OPTIONS = (False, True, 'auto')

################################################################################
############################ Close-tone selection ##############################
################################################################################
def select_isolated(
    fr, lw, min_sep_lw = 5., pairs = 'drop_both', strength = None,
    candidates = None
):
    """
    Select the resonators that are far enough from their neighbors to get a
    tone.

    Two resonators are close when they are less than ``min_sep_lw`` times
    the larger of their two linewidths apart.

    Parameters:
    fr (array-like): every resonator's frequency in Hz (e.g. ``fres_all``).
    lw (array-like): their linewidths fr / Qr in Hz. NaN (no fit) is
        replaced by the median of the finite linewidths.
    min_sep_lw (float): close threshold in linewidths. Default 5.
    pairs (str): what to do with close resonators. Can be 'drop_both'
        (default: a candidate close to any other resonator gets no tone) or
        'keep_stronger' (candidates are taken from the strongest down, and
        one is kept unless it is close to a candidate already kept; close
        resonators that get no tone are still cut out of gain fits through
        ``fres_all``).
    strength (array-like or None): per-resonator strength for
        'keep_stronger', e.g. the fitted depth or SNR (NaN counts as
        weakest). Required for 'keep_stronger'. Default None.
    candidates (array-like bool or None): resonators that may get a tone;
        the others still count as neighbors for 'drop_both'. None (default):
        all.

    Returns:
    keep (np.array, bool): True for the resonators that get a tone (always
        False for non-candidates).
    nearest_lw (np.array): distance to the nearest other resonator, in units
        of the larger of the two linewidths (inf if there is no other).

    Raises:
    ValueError: if the inputs are invalid.
    """
    fr = np.asarray(fr, dtype = np.float64)
    lw = np.asarray(lw, dtype = np.float64)
    n = len(fr)
    if fr.ndim != 1 or lw.shape != fr.shape:
        raise ValueError('fr and lw must be 1D arrays of the same length')
    if pairs not in PAIRS_OPTIONS:
        raise ValueError(f'pairs must be one of {PAIRS_OPTIONS}')
    if not min_sep_lw >= 0:
        raise ValueError('min_sep_lw must be non-negative')
    candidates = np.ones(n, dtype = bool) if candidates is None else \
        np.asarray(candidates, dtype = bool)
    if candidates.shape != (n,):
        raise ValueError('candidates must have the same length as fr')
    if pairs == 'keep_stronger':
        if strength is None:
            raise ValueError("strength is required for 'keep_stronger'")
        strength = np.asarray(strength, dtype = np.float64)
        if strength.shape != (n,):
            raise ValueError('strength must have the same length as fr')
    if n == 0:
        return np.zeros(0, dtype = bool), np.zeros(0)
    finite = np.isfinite(lw) & (lw > 0)
    if not finite.any():
        raise ValueError('no finite, positive linewidth')
    lw = np.where(finite, lw, np.median(lw[finite]))

    # Distance to every neighbor that could be close, via a window in
    # sorted order (close pairs are at most min_sep_lw * max(lw) apart)
    o = np.argsort(fr, kind = 'stable')
    fs, ls = fr[o], lw[o]
    reach = max(min_sep_lw, 1.) * ls.max()
    lo = np.searchsorted(fs, fs - reach, side = 'left')
    hi = np.searchsorted(fs, fs + reach, side = 'right')
    nearest = np.full(n, np.inf)
    close_to = [None] * n
    for i in range(n):
        j = np.r_[lo[i]:i, i + 1:hi[i]]
        d = np.abs(fs[j] - fs[i]) / np.maximum(ls[j], ls[i])
        if len(d):
            nearest[i] = d.min()
        close_to[i] = j[d < min_sep_lw]
    # Back to input order
    nearest_lw = np.empty(n)
    nearest_lw[o] = nearest
    rank = np.empty(n, dtype = int)
    rank[o] = np.arange(n)
    keep = np.zeros(n, dtype = bool)
    if pairs == 'drop_both':
        keep = candidates & ~(nearest_lw < min_sep_lw)
        return keep, nearest_lw
    # keep_stronger: strongest candidates first; NaN strength is weakest
    s = np.where(np.isfinite(strength), strength, -np.inf)
    order = np.flatnonzero(candidates)
    order = order[np.argsort(-s[order], kind = 'stable')]
    kept_sorted = np.zeros(n, dtype = bool)
    for i in order:
        if not kept_sorted[close_to[rank[i]]].any():
            keep[i] = True
            kept_sorted[rank[i]] = True
    return keep, nearest_lw

################################################################################
############################## Tone-set planning ###############################
################################################################################
def _greedy_windows(fs, width, max_tones, allowed = None):
    """
    Split sorted frequencies into the fewest consecutive windows, each
    spanning at most ``width`` and holding at most ``max_tones`` tones,
    cutting only where allowed.

    Parameters:
    fs (np.array): frequencies in Hz, ascending.
    width (float): largest span of the tones in one window, Hz.
    max_tones (int): most tones in one window.
    allowed (np.array or None): bool per gap (len(fs) - 1): True where a
        window may end. None: everywhere.

    Returns:
    starts (list of int or None): index of each window's first tone, or
        None if no split is possible.
    """
    n = len(fs)
    starts, s = [], 0
    while s < n:
        starts.append(s)
        # Last tone that fits in a window starting at s
        e = min(np.searchsorted(fs, fs[s] + width, side = 'right'),
                s + max_tones) - 1
        if e >= n - 1:
            break
        # Cut at the farthest allowed gap at or before e
        if allowed is None:
            cut = e
        else:
            ok = np.flatnonzero(allowed[s:e + 1])
            if not len(ok):
                return None
            cut = s + ok[-1]
        s = cut + 1
    return starts

def _split(fs, width, max_tones):
    """
    Split sorted frequencies into the fewest windows, with the cuts at gaps
    as wide as possible.

    Among the splits with the fewest windows, the one whose narrowest cut
    gap is widest is chosen (found by bisection over the gaps).

    Parameters:
    fs (np.array): frequencies in Hz, ascending.
    width (float): largest span of the tones in one window, Hz.
    max_tones (int): most tones in one window.

    Returns:
    starts (list of int): index of each window's first tone.
    """
    if len(fs) == 0:
        return []
    starts = _greedy_windows(fs, width, max_tones)
    k = len(starts)
    if k == 1:
        return starts
    gaps = np.diff(fs)
    levels = np.unique(gaps)
    lo, hi = 0, len(levels) - 1  # levels[lo] always feasible
    best = starts
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trial = _greedy_windows(fs, width, max_tones,
                                allowed = gaps >= levels[mid])
        if trial is not None and len(trial) <= k:
            lo, best = mid, trial
        else:
            hi = mid - 1
    return best

def plan_tone_sets(
    fres, modules, bw = 500e6, edge_margin = 10e6, extended_bw = 'auto',
    extended_span = 575e6, max_tones = 1024
):
    """
    Split tones into NCO windows and tone sets (tones live at the same
    time).

    The tones are covered with the fewest NCO windows. In a standard
    window the tones span at most ``bw - 2 * edge_margin`` (each tone at
    least ``edge_margin`` from the edges); in an extended-bandwidth window
    (600 MHz) they span at most ``extended_span``. Among the splits with
    the fewest windows, the cuts are placed at the widest gaps between
    tones, so close neighbors stay together. Each NCO is at the center of
    its window's tones. Windows, in frequency order, are given to the
    modules in turn: window w goes to ``modules[w % len(modules)]`` in tone
    set ``w // len(modules)``.

    Parameters:
    fres (array-like): tone frequencies in Hz.
    modules (list of int): modules available for the tones.
    bw (float): standard module bandwidth in Hz. Default 500 MHz.
    edge_margin (float): smallest distance of a tone from a standard
        window's edges, Hz. Default 10 MHz.
    extended_bw (bool or str): Can be False (standard windows), True
        (extended windows), or 'auto' (default: extended windows only if
        they need fewer windows than standard ones).
    extended_span (float): largest span of the tones in an extended
        window, Hz. Default 575 MHz (12.5 MHz from the 600 MHz edges).
    max_tones (int): most tones in one window (channels per module).
        Default 1024.

    Returns:
    plan (dict):
        ``tone_set`` (np.array, int32): each tone's set, in input order.
        ``tone_module`` (np.array, int32): each tone's module.
        ``ncos`` (list of dict): per tone set, {module: NCO in Hz}.
        ``extended_bw`` (bool): whether extended windows are used.
        ``window_edges`` (np.array): (nw, 2) lowest and highest tone of
            each window, Hz.
        ``window_ncos`` (np.array): (nw,) NCO of each window, Hz.

    Raises:
    ValueError: if the inputs are invalid.
    """
    fres = np.asarray(fres, dtype = np.float64)
    modules = [int(m) for m in modules]
    if fres.ndim != 1 or not np.all(np.isfinite(fres)):
        raise ValueError('fres must be a 1D array of finite frequencies')
    if not len(modules) or len(set(modules)) != len(modules):
        raise ValueError('modules must be a non-empty list of distinct '
                         'modules')
    if extended_bw not in EXTENDED_OPTIONS:
        raise ValueError(f'extended_bw must be one of {EXTENDED_OPTIONS}')
    width_std = bw - 2 * edge_margin
    if width_std <= 0 or extended_span <= 0:
        raise ValueError('bw - 2 * edge_margin and extended_span must be '
                         'positive')
    if int(max_tones) != max_tones or max_tones < 1:
        raise ValueError('max_tones must be a positive integer')
    o = np.argsort(fres, kind = 'stable')
    fs = fres[o]
    std = _split(fs, width_std, int(max_tones))
    if extended_bw is False:
        use_ext, starts = False, std
    else:
        ext = _split(fs, extended_span, int(max_tones))
        use_ext = extended_bw is True or len(ext) < len(std)
        starts = ext if use_ext else std
    ends = starts[1:] + [len(fs)]
    nw, nm = len(starts), len(modules)
    window_edges = np.array([[fs[s], fs[e - 1]] for s, e in
                             zip(starts, ends)]).reshape(nw, 2)
    window_ncos = window_edges.mean(axis = 1)
    tone_set_s = np.empty(len(fs), dtype = np.int32)
    tone_module_s = np.empty(len(fs), dtype = np.int32)
    ncos = [dict() for _ in range((nw + nm - 1) // nm)]
    for w, (s, e) in enumerate(zip(starts, ends)):
        tone_set_s[s:e] = w // nm
        tone_module_s[s:e] = modules[w % nm]
        ncos[w // nm][modules[w % nm]] = float(window_ncos[w])
    tone_set = np.empty(len(fs), dtype = np.int32)
    tone_module = np.empty(len(fs), dtype = np.int32)
    tone_set[o], tone_module[o] = tone_set_s, tone_module_s
    return dict(tone_set = tone_set, tone_module = tone_module, ncos = ncos,
                extended_bw = bool(use_ext), window_edges = window_edges,
                window_ncos = window_ncos)
