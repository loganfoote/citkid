"""
Find photon pulses in calibrated timestreams with an iteratively built
optimal filter, mark pulses that are coincident across tones (e.g. cosmic
rays), average the photon pulses, and fit their decay time.

Detection on one tone (``run_detection``):

1. The detection timestream is the calibrated fractional frequency shift
   x with slow drift removed (``detection_timestream``). A photon moves the
   resonance down, so pulses are negative (``pulse_sign = -1``).
2. Initial events: peaks of a 3-sample boxcar of sign * d above
   ``init_nsigma`` robust (MAD) sigma.
3. Repeat ``of_iterations`` times: average the events into a template,
   estimate the noise PSD from segments without pulses, build the optimal
   filter, filter, and take the filter peaks with heights in [hmin, hmax]
   as the new events. Heights are in units of the mean single-photon
   amplitude, so thresholds don't depend on the tone's responsivity. The
   lower threshold can be chosen automatically: the lowest hmin at which
   the expected noise events (peaks of the inverted filter output, which
   has noise but no pulses) are at most ``hmin_false_frac`` of the
   accepted events.
4. The result keeps every filter peak at or above hmin with its height
   (the noise threshold only), and the events in [hmin, hmax].

Across the tones streamed together, ``coincidence_multiplicity`` counts
how many tones have a pulse near each pulse; ``average_pulse`` averages
the pulses inside the height window that are not coincident, and the
coincident ones are kept for separate study. ``fit_tau`` fits the decay of
an averaged pulse.

Adapted from the crosstalk analysis's detection (iteration, thresholds,
noise PSD, filter, coincidence clusters).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, oaconvolve

################################################################################
################################### Settings ###################################
################################################################################
@dataclass
class DetectionSettings:
    """
    Settings of ``run_detection``.

    Attributes:
    pulse_sign (int): sign of a photon pulse in the detection timestream.
        -1 (default) for citkid's x (the resonance moves down).
    template_pre_s, template_post_s (float): template length before and
        after the pulse maximum, s. Default 0.5 ms and 2.5 ms.
    highpass_hz (float): drift removal for detection (running median over
        about 1 / highpass_hz). Default 10.
    init_nsigma (float): first-pass threshold, MAD sigma of the 3-sample
        boxcar. Default 6.
    of_iterations (int): optimal-filter iterations. Default 5.
    hmin, hmax (float): accepted height window, in units of the mean
        single-photon amplitude. Default 0.5, 1.6.
    hmin_false_frac (float or None): if set (e.g. 1e-3), hmin is chosen in
        every iteration as the lowest value in [hmin, hmin_auto_max] with at
        most this fraction of expected noise events among the accepted
        ones. Default None (fixed hmin).
    hmin_auto_max (float): largest automatic hmin. Default 0.8.
    psd_max_segments (int): most segments in the noise PSD. Default 4000.
    baseline_gap (int): the template's baseline ends this many samples
        before its pre-maximum part ends. Default 3.
    block (int): samples per processing block. Default 2**22.
    """
    pulse_sign: int = -1
    template_pre_s: float = 0.5e-3
    template_post_s: float = 2.5e-3
    highpass_hz: float = 10.
    init_nsigma: float = 6.
    of_iterations: int = 5
    hmin: float = 0.5
    hmax: float = 1.6
    hmin_false_frac: Optional[float] = None
    hmin_auto_max: float = 0.8
    psd_max_segments: int = 4000
    baseline_gap: int = 3
    block: int = 2 ** 22

class DetectionError(RuntimeError):
    """
    Detection found no usable events; the message says where it stopped.
    """

################################################################################
############################ Detection timestream ##############################
################################################################################
def mad_sigma(x):
    """
    Return the robust standard deviation 1.4826 * MAD of the finite values.

    Parameters:
    x (np.array): values.

    Returns:
    sigma (float): robust sigma (NaN if there are no finite values).
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))

def remove_baseline(d, dt, f_hp):
    """
    Remove slow drift: subtract a running median of segment medians.

    Segments are 1 / (4 f_hp) long and the running median spans 5
    segments, so pulses much shorter than 1 / f_hp don't bias the baseline.

    Parameters:
    d (np.array): timestream.
    dt (float): sample period, s.
    f_hp (float): corner frequency, Hz.

    Returns:
    out (np.array): ``d`` with the drift removed (same dtype).
    """
    d = np.asarray(d)
    n = d.shape[0]
    seg = max(int(round(1 / (4 * f_hp * dt))), 1)
    nseg = n // seg
    if nseg < 3:
        return d - np.median(d)
    meds = np.empty(nseg)
    step = max(1, 2 ** 22 // seg)
    for s0 in range(0, nseg, step):
        s1 = min(s0 + step, nseg)
        meds[s0:s1] = np.median(
            d[s0 * seg:s1 * seg].reshape(s1 - s0, seg), axis = 1)
    meds = median_filter(meds, size = 5, mode = 'nearest')
    tc = (np.arange(nseg) + 0.5) * seg
    out = np.empty_like(d)
    block = 2 ** 22
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        out[i0:i1] = d[i0:i1] - np.interp(
            np.arange(i0, i1), tc, meds).astype(d.dtype)
    return out

def detection_timestream(x, dt, f_hp = 10.):
    """
    Make the detection timestream from a calibrated x timestream: float32,
    with slow drift removed.

    Parameters:
    x (np.array): fractional frequency shift timestream (real).
    dt (float): sample period, s.
    f_hp (float): drift removal corner frequency, Hz. Default 10.

    Returns:
    d (np.array, float32): detection timestream.
    """
    return remove_baseline(np.asarray(x, dtype = np.float32), dt, f_hp)

################################################################################
################################ Optimal filter ################################
################################################################################
def noise_psd(d, nfft, exclude, max_segments = 4000):
    """
    Estimate a one-sided noise PSD from non-overlapping segments that hold
    no excluded sample.

    Parameters:
    d (np.array): timestream.
    nfft (int): segment length.
    exclude (np.array, bool): samples to keep out (pulses).
    max_segments (int): most segments used (evenly spread). Default 4000.

    Returns:
    psd (np.array): PSD, length nfft // 2 + 1 (units of d squared per
        sample-rate unit).
    nseg (int): segments used.

    Raises:
    ValueError: if no segment is clean.
    """
    n = d.shape[0]
    nseg = n // nfft
    dirty = exclude[:nseg * nfft].reshape(nseg, nfft).any(axis = 1)
    clean = np.flatnonzero(~dirty)
    if clean.size == 0:
        raise ValueError('no clean segments for the noise PSD')
    if clean.size > max_segments:
        clean = clean[np.linspace(0, clean.size - 1,
                                  max_segments).astype(int)]
    segs = d[:nseg * nfft].reshape(nseg, nfft)[clean].astype(np.float64)
    segs = segs - segs.mean(axis = 1, keepdims = True)
    X = np.abs(rfft(segs, axis = 1)) ** 2 / nfft
    X[:, 1:-1] *= 2
    return X.mean(axis = 0), clean.size

class OptimalFilter:
    """
    Optimal filter for a template and a noise PSD.

    ``apply`` returns y with y[n] = the amplitude (in units of the
    template) of a template whose maximum is at sample n.

    Attributes:
    a (np.array): template.
    p (int): index of the template's maximum.
    nfft (int): FFT length.
    norm (float): filter normalization.
    gc (np.array): centered filter (lags -c ... nfft - c - 1).
    """

    def __init__(self, a, p, psd, nfft):
        """
        Build the filter.

        Parameters:
        a (np.array): template (length at most nfft).
        p (int): index of its maximum.
        psd (np.array): noise PSD, length nfft // 2 + 1.
        nfft (int): FFT length.
        """
        self.a, self.p, self.nfft = np.asarray(a, float), int(p), int(nfft)
        apad = np.zeros(self.nfft)
        apad[:self.a.size] = self.a
        # Zero weight at DC: the segments are mean-subtracted and the
        # baseline is unknown anyway
        w = np.zeros_like(psd)
        w[1:] = 1 / psd[1:]
        g = irfft(rfft(apad) * w, n = self.nfft)
        self.norm = float(np.dot(g, apad))
        self.c = self.nfft // 2
        self.gc = np.roll(g, self.c) / self.norm

    def apply(self, d, sign = 1, block = 2 ** 22):
        """
        Filter sign * d.

        Parameters:
        d (np.array): timestream.
        sign (int): pulse sign. Default 1.
        block (int): samples per block. Default 2**22.

        Returns:
        y (np.array, float32): y[n] = amplitude of a pulse peaking at n.
        """
        n, L, c, p = d.shape[0], self.nfft, self.c, self.p
        r = self.gc[::-1] * sign
        y = np.zeros(n, dtype = np.float32)
        for b0 in range(0, n - p, block):
            b1 = min(b0 + block, n - p)
            lo, hi = b0 - c, b1 - c + L - 1
            x = np.zeros(hi - lo)
            s0, s1 = max(lo, 0), min(hi, n)
            x[s0 - lo:s1 - lo] = d[s0:s1]
            y[b0 + p:b1 + p] = oaconvolve(x, r, mode = 'valid')
        return y

################################################################################
############################# Templates and peaks ##############################
################################################################################
def build_template(d, events, pre, post, sign, gap = 3):
    """
    Average the pulses around events (index = pulse maximum).

    Parameters:
    d (np.array): detection timestream.
    events (np.array, int): event indices.
    pre, post (int): samples before and after the maximum.
    sign (int): pulse sign.
    gap (int): the baseline is the first pre - gap samples. Default 3.

    Returns:
    avg (np.array): sign-corrected mean pulse, baseline subtracted.
    p (int): index of its maximum.

    Raises:
    ValueError: if no event has a full window.
    """
    events = np.asarray(events, dtype = np.int64)
    ok = (events - pre >= 0) & (events + post <= d.shape[0])
    ev = events[ok]
    if ev.size == 0:
        raise ValueError('no events for the template')
    X = d[ev[:, None] + np.arange(-pre, post)[None, :]].astype(np.float64)
    avg = X.mean(axis = 0) * sign
    nb = max(pre - gap, 1)
    avg = avg - avg[:nb].mean()
    return avg, int(np.argmax(avg))

def exclusion_mask(n, peaks, pre, post):
    """
    Mark the samples within [p - 2 pre, p + 2 post) of any peak.

    Parameters:
    n (int): length.
    peaks (np.array, int): peak indices.
    pre, post (int): template samples before and after the maximum.

    Returns:
    mask (np.array, bool): True near a peak.
    """
    m = np.zeros(n, dtype = bool)
    for p in np.asarray(peaks, dtype = np.int64):
        m[max(p - 2 * pre, 0):min(p + 2 * post, n)] = True
    return m

def _find_peaks_blocked(y, height, distance, block = 2 ** 24, sign = 1):
    """
    Find the peaks of sign * y in overlapping blocks.

    Parameters:
    y (np.array): timestream.
    height (float): smallest peak height.
    distance (int): smallest peak spacing.
    block (int): samples per block. Default 2**24.
    sign (int): 1 (default) or -1.

    Returns:
    idx (np.array, int64): peak indices.
    heights (np.array): peak heights (of sign * y).
    """
    n = y.shape[0]
    idx_all, h_all = [], []
    ov = 2 * int(distance) + 2
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        lo, hi = max(b0 - ov, 0), min(b1 + ov, n)
        seg = y[lo:hi] if sign > 0 else -y[lo:hi]
        idx, props = find_peaks(seg, height = height, distance = distance)
        idx = idx + lo
        keep = (idx >= b0) & (idx < b1)
        idx_all.append(idx[keep])
        h_all.append(props['peak_heights'][keep])
    return np.concatenate(idx_all).astype(np.int64), np.concatenate(h_all)

def candidates(y, sigma_y, distance, floor = 0.2, nsig = 3., sign = 1):
    """
    Find the local maxima of sign * y above max(floor, nsig * sigma_y), at
    least ``distance`` apart.

    Parameters:
    y (np.array): filter output.
    sigma_y (float): its noise sigma.
    distance (int): smallest spacing.
    floor (float): smallest height. Default 0.2.
    nsig (float): smallest height in sigma. Default 3.
    sign (int): 1 (default) or -1.

    Returns:
    idx (np.array, int64): candidate indices.
    heights (np.array, float32): their heights.
    """
    h = max(floor, nsig * sigma_y)
    idx, heights = _find_peaks_blocked(y, h, distance, sign = sign)
    return idx, heights.astype(np.float32)

def _boxcar3(d, sign):
    """
    Return sign * the 3-sample moving average of d.

    Parameters:
    d (np.array): timestream.
    sign (int): pulse sign.

    Returns:
    sd (np.array, float32): smoothed timestream (the ends repeat d).
    """
    n = d.shape[0]
    sd = np.empty(n, dtype = np.float32)
    sd[1:-1] = d[:-2] + d[1:-1] + d[2:]
    sd[0], sd[-1] = 3 * d[0], 3 * d[-1]
    sd *= sign / 3
    return sd

def near(idx, times, before, after):
    """
    Find which indices are in [t - before, t + after) for some time t.

    Parameters:
    idx (np.array, int): indices.
    times (np.array, int or None): sorted times.
    before, after (int): window, samples.

    Returns:
    near (np.array, bool): True for indices near a time.
    """
    idx = np.asarray(idx, dtype = np.int64)
    if times is None or len(times) == 0:
        return np.zeros(idx.size, dtype = bool)
    times = np.asarray(times, dtype = np.int64)
    j = np.searchsorted(times, idx + before, side = 'right') - 1
    return (j >= 0) & (times[np.clip(j, 0, None)] > idx - after)

def _sub(n, nmax = 2_000_000):
    """
    Return evenly spaced indices for statistics of long timestreams.

    Parameters:
    n (int): length.
    nmax (int): most indices. Default 2,000,000.

    Returns:
    idx (np.array, int): indices.
    """
    return np.arange(0, n, max(1, n // nmax))

################################################################################
############################ Noise threshold (hmin) ############################
################################################################################
def inverted_candidates(y, sigma_y, distance, peaks, before, after):
    """
    Find the candidates of -y away from pulses: pulses are one-signed, so
    these are noise peaks with the statistics of the noise among the
    candidates of y. Peaks within [p - before, p + after) of a pulse p (its
    filter undershoot) are left out.

    Parameters:
    y (np.array): filter output.
    sigma_y (float): its noise sigma.
    distance (int): smallest spacing.
    peaks (np.array, int): pulse indices.
    before, after (int): window around each pulse, samples.

    Returns:
    neg_h (np.array, float32): heights of the inverted candidates.
    clean_frac (float): fraction of the record away from pulses.
    """
    n = y.shape[0]
    idx, h = candidates(y, sigma_y, distance, sign = -1)
    peaks = np.sort(np.asarray(peaks, dtype = np.int64))
    if peaks.size == 0:
        return h, 1.
    s0 = np.clip(peaks - before, 0, n)
    s1 = np.clip(peaks + after, 0, n)
    j = np.searchsorted(s0, idx, side = 'right') - 1
    close = (j >= 0) & (idx < s1[np.clip(j, 0, None)])
    covered = np.sum(np.minimum(s1, np.r_[s0[1:], n]) - s0)
    return h[~close], float(1 - covered / n)

def false_fraction(cand_h, neg_h, clean_frac, hmin, hmax):
    """
    Return the expected noise events among the candidates in [hmin, hmax],
    over the accepted count.

    Parameters:
    cand_h (np.array): candidate heights.
    neg_h (np.array): inverted candidate heights.
    clean_frac (float): fraction of the record they come from.
    hmin, hmax (float): height window.

    Returns:
    frac (float): expected false fraction.
    """
    n_ev = np.count_nonzero((cand_h >= hmin) & (cand_h <= hmax))
    n_false = np.count_nonzero((neg_h >= hmin) & (neg_h <= hmax)) / \
        max(clean_frac, 1e-6)
    return n_false / max(n_ev, 1)

def auto_hmin(
    cand_h, neg_h, clean_frac, hmin0, hmax, max_false, hmin_cap,
    step = 0.01
):
    """
    Return the lowest hmin in [hmin0, hmin_cap] (steps of ``step``) whose
    false fraction is at most ``max_false``; ``hmin_cap`` where none is.

    Parameters:
    cand_h, neg_h, clean_frac: see ``false_fraction``.
    hmin0 (float): lowest hmin.
    hmax (float): upper threshold.
    max_false (float): largest false fraction.
    hmin_cap (float): highest hmin.
    step (float): grid step. Default 0.01.

    Returns:
    hmin (float): the threshold.
    """
    grid = np.arange(hmin0, hmin_cap + step / 2, step)
    ff = np.array([false_fraction(cand_h, neg_h, clean_frac, h, hmax)
                   for h in grid])
    ok = np.flatnonzero(ff <= max_false)
    return float(np.round(grid[ok[0]] if ok.size else hmin_cap, 6))

################################################################################
################################### Detection ##################################
################################################################################
def run_detection(d, dt, settings = None, hmin = None, hmax = None,
                  veto = None):
    """
    Find the photon pulses of one tone with the iterative optimal filter
    (see the module description).

    Parameters:
    d (np.array): detection timestream (``detection_timestream``).
    dt (float): sample period, s.
    settings (DetectionSettings or None): settings. None: the defaults.
    hmin, hmax (float or None): height window, overriding the settings'.
        With hmin None and ``settings.hmin_false_frac`` set, hmin is chosen
        automatically.
    veto (np.array or None): sorted times (e.g. coincidences of many tones
        found with ``triggers``): candidates within the template length of
        them are not events, and the noise PSD avoids them. Default None.

    Returns:
    res (dict):
        ``cand_idx``, ``cand_h``: every filter peak at or above hmin (the
            noise threshold) and its height, in units of the mean photon;
        ``events``: those with heights in [hmin, hmax];
        ``template`` (mean pulse of the events, in units of d, times
            ``sign``), ``p`` (index of its maximum), ``pre``, ``post``,
            ``amp_ph`` (its amplitude), ``sign`` (the pulse sign);
        ``hmin``, ``hmax``, ``sigma_y`` (filter noise, photon units),
            ``rate`` (events per second);
        ``psd``, ``nfft``, ``filter`` (centered), ``history`` (template
            of each iteration);
        ``neg_h``, ``clean_frac`` (noise peaks, for the false fraction);
        ``n_vetoed``; ``log`` (one line per iteration).

    Raises:
    DetectionError: if there are no events to build a template from.
    """
    s = DetectionSettings() if settings is None else settings
    auto = hmin is None and s.hmin_false_frac is not None
    hmin = s.hmin if hmin is None else float(hmin)
    hmax = s.hmax if hmax is None else float(hmax)
    sign = int(s.pulse_sign)
    pre = max(int(round(s.template_pre_s / dt)), s.baseline_gap + 2)
    post = max(int(round(s.template_post_s / dt)), 4)
    M = pre + post
    nfft = next_fast_len(2 * M)
    n = d.shape[0]
    if s.of_iterations < 1:
        raise ValueError('of_iterations must be at least 1')
    veto = np.zeros(0, dtype = np.int64) if veto is None else \
        np.sort(np.asarray(veto, dtype = np.int64))
    sd = _boxcar3(d, sign)
    sig0 = mad_sigma(sd[_sub(n)])
    events, _ = _find_peaks_blocked(sd, s.init_nsigma * sig0, post)
    del sd
    events = events[~near(events, veto, post, post)]
    peaks_all = np.union1d(events, veto)
    history = []
    log = [f'initial: {events.size} peaks above {s.init_nsigma:g} sigma '
           'of the boxcar']
    for it in range(s.of_iterations):
        if events.size == 0:
            raise DetectionError('no events to build the template from; '
                                 + '; '.join(log))
        avg, p = build_template(d, events, pre, post, sign, s.baseline_gap)
        if p != pre:
            events = events + (p - pre)
            avg, p = build_template(d, events, pre, post, sign,
                                    s.baseline_gap)
        amp_ph = avg[p]
        history.append(avg / amp_ph)
        excl = exclusion_mask(n, peaks_all, pre, post)
        psd, nseg = noise_psd(d, nfft, excl, s.psd_max_segments)
        of = OptimalFilter(avg / amp_ph, p, psd, nfft)
        y = of.apply(d, sign, s.block) / amp_ph
        sub = _sub(n)
        sigma_y = mad_sigma(y[sub][~excl[sub]])
        del excl
        cidx, ch = candidates(y, sigma_y, distance = post)
        vetoed = near(cidx, veto, post, post)
        cidx, ch = cidx[~vetoed], ch[~vetoed]
        neg_h, clean = inverted_candidates(
            y, sigma_y, post, np.union1d(cidx[ch >= s.hmin], veto), M, nfft)
        if auto:
            hmin = auto_hmin(ch, neg_h, clean, s.hmin, hmax,
                             s.hmin_false_frac, s.hmin_auto_max)
        keep = (ch >= hmin) & (ch <= hmax)
        events = cidx[keep]
        peaks_all = np.union1d(cidx[ch >= hmin], veto)
        log.append(f'iteration {it}: template amplitude {amp_ph:.3g}, '
                   f'sigma_y {sigma_y:.3g}, {cidx.size} candidates, '
                   f'{events.size} in [{hmin:g}, {hmax:g}]')
    if events.size == 0:
        raise DetectionError('no events after the last iteration; '
                             + '; '.join(log))
    avg, p = build_template(d, events, pre, post, sign, s.baseline_gap)
    if p != pre:
        shift = p - pre
        events, cidx = events + shift, cidx + shift
        avg, p = build_template(d, events, pre, post, sign, s.baseline_gap)
    above = ch >= hmin
    return dict(
        cand_idx = cidx[above], cand_h = ch[above], events = events,
        template = avg, p = p, pre = pre, post = post, amp_ph = avg[p],
        sign = sign,
        hmin = hmin, hmax = hmax, sigma_y = sigma_y,
        rate = events.size / (n * dt), psd = psd, nfft = nfft,
        filter = of.gc, history = np.array(history), neg_h = neg_h,
        clean_frac = clean, n_vetoed = int(np.sum(vetoed)), log = log)

################################################################################
################################# Coincidences #################################
################################################################################
def triggers(d, nsigma, distance, sign = None):
    """
    Find the peaks of a 3-sample boxcar of d above ``nsigma`` MAD sigma,
    for coincidences across tones.

    Parameters:
    d (np.array): detection timestream.
    nsigma (float): threshold in MAD sigma.
    distance (int): smallest peak spacing.
    sign (int or None): 1 or -1 for one sign; None (default) for both.

    Returns:
    idx (np.array, int64): trigger indices, ascending.
    """
    out = []
    for sg in ((1, -1) if sign is None else (int(sign),)):
        sd = _boxcar3(d, sg)
        thr = nsigma * mad_sigma(sd[_sub(sd.shape[0])])
        out.append(_find_peaks_blocked(sd, thr, distance)[0])
    return np.unique(np.concatenate(out))

def coincidence_clusters(trig, win):
    """
    Group the triggers of several tones into clusters: sorted trigger
    times joined while the gap is at most ``win`` samples.

    Parameters:
    trig (list of np.array): trigger indices, one array per tone.
    win (int): largest gap within a cluster, samples.

    Returns:
    t (np.array, int64): first trigger of each cluster.
    mult (np.array, int): number of distinct tones in each cluster.
    """
    t = np.concatenate([np.asarray(x, dtype = np.int64) for x in trig])
    if t.size == 0:
        return t, np.zeros(0, dtype = int)
    tone = np.concatenate([np.full(len(x), i, dtype = np.int64)
                           for i, x in enumerate(trig)])
    o = np.argsort(t, kind = 'stable')
    t, tone = t[o], tone[o]
    cid = np.r_[0, np.cumsum(np.diff(t) > win)]
    first = np.r_[True, cid[1:] != cid[:-1]]
    key = np.unique(cid * len(trig) + tone)
    mult = np.bincount(key // len(trig), minlength = cid[-1] + 1)
    return t[first], mult

def coincidence_multiplicity(events, win):
    """
    Count, for each pulse of each tone, the tones (itself included) with a
    pulse within ``win`` samples of it. Use the pulses of tones streamed
    together (one write), e.g. every candidate at or above the noise
    threshold (``run_detection``'s ``cand_idx``).

    Parameters:
    events (list of np.array): pulse indices, one array per tone.
    win (int): coincidence window, samples.

    Returns:
    mult (list of np.array, int): per tone, the multiplicity of each pulse
        (1: only this tone).
    """
    events = [np.sort(np.asarray(e, dtype = np.int64)) for e in events]
    out = []
    for i, e in enumerate(events):
        m = np.ones(len(e), dtype = int)
        for j, other in enumerate(events):
            if j == i or not len(other):
                continue
            k = np.searchsorted(other, e - win)
            hit = (k < len(other)) & \
                (other[np.clip(k, 0, len(other) - 1)] <= e + win)
            m += hit
        out.append(m)
    return out

def average_pulse(d, res, mult = None, max_mult = 1, gap = 3):
    """
    Average the photon pulses of one tone: the events in the height window
    whose coincidence multiplicity is at most ``max_mult``.

    Parameters:
    d (np.array): detection timestream.
    res (dict): ``run_detection`` result.
    mult (np.array or None): multiplicity of each candidate
        (``coincidence_multiplicity`` of ``res['cand_idx']``). None: no
        coincidence cut.
    max_mult (int): largest multiplicity kept. Default 1 (no other tone).
    gap (int): baseline gap (``build_template``). Default 3.

    Returns:
    avg (np.array): mean pulse in units of d, sign-corrected, baseline
        subtracted.
    p (int): index of its maximum.
    used (np.array, int): the events averaged.
    coincident (np.array, int): the candidates above the noise threshold
        with multiplicity above ``max_mult`` (e.g. cosmic rays), for
        separate study.
    """
    idx, h = res['cand_idx'], res['cand_h']
    ok = (h >= res['hmin']) & (h <= res['hmax'])
    coinc = np.zeros(len(idx), bool) if mult is None else \
        np.asarray(mult) > max_mult
    used = idx[ok & ~coinc]
    avg, p = build_template(d, used, res['pre'], res['post'], res['sign'],
                            gap)
    return avg, p, used, idx[coinc]

################################################################################
###################################### Tau #####################################
################################################################################
def fit_tau(avg, p, dt, n_skip = 1):
    """
    Fit A exp(-t / tau) + B to the decay of a mean pulse after its maximum.

    Parameters:
    avg (np.array): mean pulse (positive-going).
    p (int): index of its maximum.
    dt (float): sample period, s.
    n_skip (int): samples after the maximum to skip. Default 1.

    Returns:
    tau (float): decay time in s (NaN if the fit fails).
    popt (np.array): (A, tau, B) (NaN if the fit fails).
    """
    y = np.asarray(avg[p + n_skip:], float)
    t = np.arange(y.size) * dt
    if y.size < 4 or not y[0] > 0:
        return np.nan, np.full(3, np.nan)
    # Fit in units of the first point: x pulses are ~1e-6, too small for
    # the fitter's default tolerances
    scale = y[0]
    try:
        popt, _ = curve_fit(
            lambda t, A, tau, B: A * np.exp(-t / tau) + B, t, y / scale,
            p0 = [1., t[-1] / 5, 0.],
            bounds = ([0, dt / 10, -1.], [10., 10 * t[-1] + dt, 1.]))
    except (RuntimeError, ValueError):
        return np.nan, np.full(3, np.nan)
    return float(popt[1]), popt * np.array([scale, 1., scale])
