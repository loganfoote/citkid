"""
High-energy event (HEE) tails: decay times from the pulses that HEEs
leave in many tones at once, for devices without photon pulses (or
alongside them).

An HEE deposits a different energy in every tone, so its pulses
have different heights and shapes near the peak (where the response can
saturate), but their tails decay alike. So the pulses are lined up at
the point where each has fallen through a level set by the user
(``tail_start``, ``align_tails``: from there on the pulses of one tone
follow one curve), averaged (``mean_aligned``), and the average's tail
is fitted (``fit_tail`` from the alignment point). ``fit_tails`` and
``tail_summary`` fit pulses one by one instead, for a look at their
spread.

Events come either from a photon result (``photons/<name>``, its
HEEs) or, without photons, from simple triggers on the tones streamed
together (``find_events``: ``photons.triggers`` and
``photons.coincidence_clusters``). ``cut_pulses`` cuts a baseline-
subtracted snippet around each event, ``peak_heights`` measures the
pulses for the user's height window, and ``fit_tails`` fits them.
Heights and levels are in units of the timestream (x); the callers
convert from kHz/GHz (``photons.PPM``).
"""
import numpy as np
from .photons import coincidence_clusters, fit_tau, triggers

def find_events(
    ds, nsigma, win, min_mult = 3, distance = 3, sign = None, dead = 0
):
    """
    Find HEEs in the detection timestreams of tones streamed
    together: the peaks above ``nsigma`` noise sigma of each tone
    (``triggers``), clustered across tones (``coincidence_clusters``),
    keeping the clusters seen by at least ``min_mult`` tones, and
    dropping those within ``dead`` samples after a kept one (the noisy
    tail of a pulse triggers again).

    Parameters:
    ds (list of np.array): detection timestreams, one per tone, the same
        length.
    nsigma (float): trigger threshold, MAD sigma of the 3-sample boxcar.
    win (int): largest gap between triggers of one cluster, samples.
    min_mult (int): fewest tones of an event. Default 3.
    distance (int): smallest spacing of one tone's triggers, samples.
        Default 3.
    sign (int or None): pulse sign to trigger on (1 or -1). Default None:
        both.
    dead (int): dead time after a kept event, samples. Default 0.

    Returns:
    t0 (np.array, int64): first trigger of each event, ascending.
    mult (np.array, int): tones in each event.
    """
    trig = [triggers(np.asarray(d), nsigma, distance, sign) for d in ds]
    t0, mult = coincidence_clusters(trig, win)
    keep = mult >= int(min_mult)
    t0, mult = t0[keep], mult[keep]
    if dead > 0 and len(t0):
        alive = np.ones(len(t0), dtype = bool)
        last = t0[0]
        for k in range(1, len(t0)):
            if t0[k] - last <= dead:
                alive[k] = False
            else:
                last = t0[k]
        t0, mult = t0[alive], mult[alive]
    return t0, mult

def cut_pulses(d, t0, pre, post, sign = -1, gap = 3):
    """
    Cut a snippet around each event: samples [t0 - pre, t0 + post), with
    the sign corrected (pulses positive) and the baseline (the median of
    the first pre - gap samples) subtracted. Events whose snippet would
    run off the timestream are dropped.

    Parameters:
    d (np.array): detection timestream.
    t0 (np.array, int): event samples.
    pre, post (int): samples before and after t0.
    sign (int): pulse sign in d. Default -1 (x of a pulse).
    gap (int): the baseline ends this many samples before t0. Default 3.

    Returns:
    pulses (np.array): (events kept, pre + post), sign-corrected and
        baseline-subtracted, in units of d.
    kept (np.array, bool): which events were kept, aligned with ``t0``.
    """
    t0 = np.asarray(t0, dtype = np.int64)
    n = len(d)
    kept = (t0 - pre >= 0) & (t0 + post <= n)
    idx = t0[kept][:, None] + np.arange(-pre, post)[None, :]
    pulses = int(sign) * np.asarray(d, dtype = float)[idx]
    nb = max(int(pre) - int(gap), 1)
    if len(pulses):
        pulses -= np.median(pulses[:, :nb], axis = 1)[:, None]
    return pulses, kept

def peak_heights(pulses, pre, search):
    """
    Measure each pulse's peak: the largest value within ``search``
    samples from the event.

    Parameters:
    pulses (np.array): ``cut_pulses`` output.
    pre (int): samples before the event in each snippet.
    search (int): samples after the event to search.

    Returns:
    h (np.array): peak height of each pulse, units of the pulses.
    p (np.array, int): index of each peak in its snippet.
    """
    if not len(pulses):
        return np.zeros(0), np.zeros(0, dtype = np.int64)
    stop = min(int(pre) + max(int(search), 1), pulses.shape[1])
    seg = pulses[:, int(pre):stop]
    p = int(pre) + np.argmax(seg, axis = 1)
    return pulses[np.arange(len(pulses)), p], p

def tail_start(pulse, p, level):
    """
    Find where a pulse's tail begins: the first sample at or after its
    peak where it is at or below ``level``.

    Parameters:
    pulse (np.array): the pulse.
    p (int): index of its peak.
    level (float): the level, units of the pulse.

    Returns:
    i (int): the sample, or -1 if the pulse never falls to the level.
    """
    below = np.flatnonzero(np.asarray(pulse[int(p):]) <= level)
    return -1 if not len(below) else int(p) + int(below[0])

def fit_tail(pulse, p, dt, level, fit_s):
    """
    Fit one pulse's tail: an exponential (``fit_tau``) over ``fit_s``
    seconds from where it falls through ``level``.

    Parameters:
    pulse (np.array): the pulse (positive-going, baseline subtracted).
    p (int): index of its peak.
    dt (float): sample period, s.
    level (float): the tail begins where the pulse falls to this, units
        of the pulse.
    fit_s (float): length of the fit, s.

    Returns:
    tau (float): decay time, s (NaN if the pulse never reaches the level
        or the fit fails).
    popt (np.array): (A, tau, B) of the fit, t = 0 at the tail's start
        (NaN if the fit fails).
    n_fit (int): samples fitted (0 if the fit fails).
    start (int): the tail's first sample (-1 if the pulse never reaches
        the level).
    """
    i = tail_start(pulse, p, level)
    if i < 0:
        return np.nan, np.full(3, np.nan), 0, -1
    tau, popt, n_fit = fit_tau(pulse, i, dt, n_skip = 0, fit_s = fit_s)
    return tau, popt, n_fit, i

def fit_tails(pulses, p, dt, level, fit_s):
    """
    Fit every pulse's tail (``fit_tail``).

    Parameters:
    pulses (np.array): ``cut_pulses`` output.
    p (np.array, int): index of each pulse's peak (``peak_heights``).
    dt (float): sample period, s.
    level (float): where the tails begin, units of the pulses.
    fit_s (float): length of each fit, s.

    Returns:
    tau (np.array): decay time of each pulse, s (NaN where the fit
        failed).
    start (np.array, int): each tail's first sample (-1 if never below
        the level).
    n_fit (np.array, int): samples fitted per pulse.
    """
    n = len(pulses)
    tau = np.full(n, np.nan)
    start = np.full(n, -1, dtype = np.int64)
    n_fit = np.zeros(n, dtype = np.int64)
    for k in range(n):
        tau[k], _, n_fit[k], start[k] = fit_tail(pulses[k], int(p[k]), dt,
                                                 level, fit_s)
    return tau, start, n_fit

def align_tails(pulses, start, anchor):
    """
    Line pulses up at their tail starts: each is shifted so that its
    ``start`` sample lands at ``anchor``, in a frame as long as the
    pulses, with NaN where a shifted pulse has no sample.

    Parameters:
    pulses (np.array): ``cut_pulses`` output, (n, L).
    start (np.array, int): each pulse's tail start (``tail_start``).
    anchor (int): the sample the tail starts land on.

    Returns:
    aligned (np.array): (n, L), NaN-padded.
    """
    pulses = np.asarray(pulses, dtype = float)
    n, L = pulses.shape
    aligned = np.full((n, L), np.nan)
    for k in range(n):
        shift = int(anchor) - int(start[k])
        lo, hi = max(0, shift), min(L, L + shift)
        aligned[k, lo:hi] = pulses[k, lo - shift:hi - shift]
    return aligned

def mean_aligned(aligned, min_count = 1):
    """
    Average aligned pulses sample by sample, ignoring the NaN padding.

    Parameters:
    aligned (np.array): ``align_tails`` output.
    min_count (int): fewest pulses a sample needs to get a mean.
        Default 1.

    Returns:
    avg (np.array): the mean pulse (NaN where fewer than ``min_count``
        pulses have a sample).
    count (np.array, int): pulses averaged at each sample.
    """
    aligned = np.asarray(aligned, dtype = float)
    if not len(aligned):
        return np.zeros(0), np.zeros(0, dtype = int)
    count = np.isfinite(aligned).sum(axis = 0)
    with np.errstate(invalid = 'ignore'):
        avg = np.nanmean(np.where(np.isfinite(aligned), aligned, np.nan),
                         axis = 0)
    avg[count < int(min_count)] = np.nan
    return avg, count

def fit_mean_tail(avg, anchor, dt, fit_s):
    """
    Fit the tail of an aligned mean pulse: an exponential (``fit_tau``)
    over ``fit_s`` seconds from the alignment point, over the samples
    the mean has (it ends where the padding begins).

    Parameters:
    avg (np.array): ``mean_aligned`` output.
    anchor (int): the alignment point.
    dt (float): sample period, s.
    fit_s (float): length of the fit, s.

    Returns:
    tau (float): decay time, s (NaN if the fit fails).
    popt (np.array): (A, tau, B), t = 0 at the alignment point.
    n_fit (int): samples fitted (0 if the fit fails).
    """
    y = np.asarray(avg, dtype = float)[int(anchor):]
    bad = np.flatnonzero(~np.isfinite(y))
    if len(bad):
        y = y[:bad[0]]
    if y.size < 4:
        return np.nan, np.full(3, np.nan), 0
    return fit_tau(y, 0, dt, n_skip = 0, fit_s = fit_s)

def tail_summary(tau):
    """
    Summarize the decay times of one tone's pulses: their median and
    spread (a robust sigma, 1.4826 times the median absolute deviation),
    over the finite values.

    Parameters:
    tau (np.array): decay times, s (NaN where a fit failed).

    Returns:
    median (float): median decay time, s (NaN without values).
    spread (float): robust sigma of the decay times, s (NaN without
        values).
    n (int): decay times used.
    """
    t = np.asarray(tau, dtype = float)
    t = t[np.isfinite(t)]
    if not len(t):
        return np.nan, np.nan, 0
    med = float(np.median(t))
    return med, float(1.4826 * np.median(np.abs(t - med))), int(len(t))
