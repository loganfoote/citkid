"""
Find photon pulses in calibrated timestreams with an iteratively built
optimal filter, mark pulses that are coincident across tones (e.g. cosmic
rays), average the photon pulses, and fit their decay time.

Detection on one tone (``run_detection``):

1. The detection timestream is the calibrated fractional frequency shift
   x with slow drift removed (``detection_timestream``). A photon moves the
   resonance down, so pulses are negative (``pulse_sign = -1``).
2. Initial events: peaks of a 3-sample boxcar of sign * d above
   ``init_nsigma`` robust (MAD) sigma, and at most ``init_hmax`` times the
   median initial peak height (a loose cut that keeps large pulses such as
   cosmic rays out of the first template).
3. Repeat ``of_iterations`` times: average the events into a template,
   estimate the noise PSD from segments without pulses, build the optimal
   filter, filter, and take the filter peaks with heights in [hmin, hmax]
   as the new events. Only isolated events go into the template
   (``isolated_events``): none with another peak within the template
   length (pile-up), and none within the filter length of a large pulse
   (above hmax, or above ``init_hmax`` in the first pass), whose filter
   side lobes would otherwise put its edges into the template. The filter
   is built from the first ``filter_post_s`` of the template after its
   maximum (short, so that it does not reach into the next pulse); the
   whole template (``template_post_s``) is kept for display and for the
   decay-time fit. Heights are in kHz/GHz (the fractional frequency
   shift x times 1e6, ``PPM``), so a threshold means the same shift on
   every tone and the data keeps its physical size; the template's
   amplitude (the mean photon, ``amp_ph``) is reported alongside. The
   lower threshold can be chosen automatically: the lowest hmin at which
   the expected noise events (peaks of the inverted filter output, which
   has noise but no pulses) are at most ``hmin_false_frac`` of the
   accepted events.
4. The result keeps every filter peak at or above hmin with its height
   (the noise threshold only), and the events in [hmin, hmax]. It also
   keeps the coincidence candidates: the filter peaks above the lower
   ``coinc_nsigma`` threshold, with their heights and raw peak values, so
   that small pulses of a cosmic ray still count as coincident.

Across the tones streamed together (one tone write, which share sample
times), a pulse is coincident when another tone has a coincidence
candidate within +-``win`` samples of it (``candidate_multiplicity``).
``average_pulse`` averages the pulses inside the height window that are
not coincident. ``cosmic_events`` groups the coincidence candidates of all
tones into events (each starting at its earliest pulse and spanning
2 ``win`` samples) and returns the events seen by at least ``min_mult``
tones, with each tone's pulse height. ``fit_tau`` fits the decay of an
averaged pulse.

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
    template_pre_s, template_post_s (float): template (mean pulse) length
        before and after the pulse maximum, s. Default 0.5 ms and 2.5 ms.
    filter_post_s (float or None): the optimal filter's template reaches
        this long after the maximum (at most ``template_post_s``): a few
        decay times, so that the filter does not run into the next pulse.
        Default 1 ms. None: the whole template.
    highpass_hz (float): drift removal for detection (running median over
        about 1 / highpass_hz). Default 10.
    init_nsigma (float): first-pass threshold, MAD sigma of the 3-sample
        boxcar. Default 6.
    init_hmax (float or None): first-pass upper cut, in units of the median
        first-pass peak height. Default 3 (loose: it only keeps large
        pulses out of the first template). None: no cut.
    of_iterations (int): optimal-filter iterations. Default 5.
    hmin, hmax (float): accepted height window of the filter output, in
        kHz/GHz (x times 1e6). Default 0.5, 1.6 (a 1 kHz/GHz photon).
    hmin_false_frac (float or None): if set (e.g. 1e-3), hmin is chosen in
        every iteration as the lowest value in [hmin, hmin_auto_max] with at
        most this fraction of expected noise events among the accepted
        ones. Default None (fixed hmin).
    hmin_auto_max (float): largest automatic hmin, kHz/GHz. Default 0.8.
    coinc_nsigma (float): threshold of the coincidence candidates, in
        sigma of the filter output. Default 4. The threshold is never above
        hmin, so every pulse candidate is also a coincidence candidate.
    raw_half_width (int): the raw peak of a pulse is the largest sign * d
        within this many samples of it. Default 2.
    psd_max_segments (int): most segments in the noise PSD. Default 4000.
    baseline_gap (int): the template's baseline ends this many samples
        before its pre-maximum part ends. Default 3.
    tau_skip (int): the decay-time fit of a mean pulse (``fit_tau``)
        starts this many samples after its maximum. Default 2.
    tau_lifetimes (float or None): the fit reaches this many decay times
        after its start (from a first fit of the whole decay). Default 5.
        None: the whole template after the maximum.
    block (int): samples per processing block. Default 2**22.
    """
    pulse_sign: int = -1
    template_pre_s: float = 0.5e-3
    template_post_s: float = 2.5e-3
    filter_post_s: Optional[float] = 1e-3
    highpass_hz: float = 10.
    init_nsigma: float = 6.
    init_hmax: Optional[float] = 3.
    of_iterations: int = 5
    hmin: float = 0.5
    hmax: float = 1.6
    hmin_false_frac: Optional[float] = None
    hmin_auto_max: float = 0.8
    coinc_nsigma: float = 4.
    raw_half_width: int = 2
    psd_max_segments: int = 4000
    baseline_gap: int = 3
    tau_skip: int = 2
    tau_lifetimes: Optional[float] = 5.
    block: int = 2 ** 22

# Filter heights and thresholds are x times this: parts per million of the
# resonance frequency (kHz per GHz)
PPM = 1e6

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

    @classmethod
    def from_kernel(cls, gc, p, nfft):
        """
        Rebuild a filter from its centered kernel (``gc``, as saved with a
        detection result), to apply it again.

        Parameters:
        gc (np.array): the centered kernel (length ``nfft``).
        p (int): index of the maximum of the template it was built for.
        nfft (int): FFT length.

        Returns:
        of (OptimalFilter): the filter (``a`` and ``norm`` unset).
        """
        of = cls.__new__(cls)
        of.a, of.p, of.nfft = None, int(p), int(nfft)
        of.norm, of.c = 1., of.nfft // 2
        of.gc = np.asarray(gc, float)
        return of

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

def isolated_events(events, peaks, big, near_len, far_len):
    """
    Keep the events clean enough to average into a template: those with
    no other peak within ``near_len`` samples (pile-up, which would put
    another pulse into the template window) and no large pulse within
    ``far_len`` samples (whose edges the filter's side lobes can turn into
    events of their own).

    Parameters:
    events (np.array, int): event indices.
    peaks (np.array, int): every peak found (the events among them).
    big (np.array, int): the large pulses (e.g. above hmax), and any
        vetoed times.
    near_len (int): half-width of the pile-up window (the template
        length).
    far_len (int): half-width of the window around a large pulse (the
        filter length).

    Returns:
    keep (np.array, bool): True for the isolated events.
    """
    ev = np.asarray(events, dtype = np.int64)
    peaks = np.sort(np.asarray(peaks, dtype = np.int64))
    n_near = np.searchsorted(peaks, ev + near_len, side = 'right') - \
        np.searchsorted(peaks, ev - near_len, side = 'left')
    # The event is one of the peaks (or none, if it isn't listed)
    own = np.isin(ev, peaks).astype(int)
    keep = n_near - own < 1
    keep &= ~near(ev, np.sort(np.asarray(big, dtype = np.int64)), far_len,
                  far_len + 1)
    return keep

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
def template_lengths(s, dt):
    """
    Return the template lengths in samples.

    Parameters:
    s (DetectionSettings): the settings.
    dt (float): sample period, s.

    Returns:
    pre (int): samples before the maximum (at least baseline_gap + 2).
    post (int): samples after it (at least 4).
    post_f (int): the filter template's samples after it (at least 4, at
        most ``post``).
    """
    pre = max(int(round(s.template_pre_s / dt)), s.baseline_gap + 2)
    post = max(int(round(s.template_post_s / dt)), 4)
    post_f = post if s.filter_post_s is None else \
        min(max(int(round(s.filter_post_s / dt)), 4), post)
    return pre, post, post_f

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
            noise threshold) and its height, in kHz/GHz (x times ``PPM``);
        ``events``: those with heights in [hmin, hmax];
        ``template`` (the mean pulse the last filter was built from: the
            isolated events of the iteration before, in units of d, times
            ``sign``), ``template_idx`` (those events), ``p`` (index of
            its maximum), ``pre``, ``post``, ``filter_post`` (the filter's
            template ends this many samples after the maximum),
            ``amp_ph`` (its amplitude, in x: the mean photon), ``sign``
            (the pulse sign);
        ``hmin``, ``hmax``, ``sigma_y`` (filter noise), in kHz/GHz,
            ``rate`` (events per second);
        ``psd``, ``nfft``, ``filter`` (the last filter's centered kernel;
            ``OptimalFilter.from_kernel(filter, p, nfft)`` applies it
            again, and its output times ``PPM`` gives the heights),
            ``history`` (template of each iteration);
        ``neg_h``, ``clean_frac`` (noise peaks, for the false fraction);
        ``coinc_idx``, ``coinc_h``, ``coinc_raw``: the coincidence
            candidates (filter peaks above ``coinc_nsigma`` sigma or hmin,
            whichever is lower; they include every ``cand_idx``), their
            heights in kHz/GHz, and their raw peaks
            (``raw_peaks``, in units of d, sign-corrected);
        ``init_idx``, ``init_h``: the first-pass peaks and their heights
            (boxcar, units of d), before the ``init_hmax`` cut;
            ``init_hmax_abs``: that cut in units of d (inf without it);
        ``n_vetoed``; ``dt``; ``log`` (one line per iteration).

    Raises:
    DetectionError: if there are no events to build a template from.
    """
    s = DetectionSettings() if settings is None else settings
    auto = hmin is None and s.hmin_false_frac is not None
    hmin = s.hmin if hmin is None else float(hmin)
    hmax = s.hmax if hmax is None else float(hmax)
    sign = int(s.pulse_sign)
    pre, post, post_f = template_lengths(s, dt)
    M = pre + post
    nfft = next_fast_len(2 * (pre + post_f))
    n = d.shape[0]
    if s.of_iterations < 1:
        raise ValueError('of_iterations must be at least 1')
    veto = np.zeros(0, dtype = np.int64) if veto is None else \
        np.sort(np.asarray(veto, dtype = np.int64))
    sd = _boxcar3(d, sign)
    sig0 = mad_sigma(sd[_sub(n)])
    init_idx, init_h = _find_peaks_blocked(sd, s.init_nsigma * sig0, post_f)
    del sd
    init_hmax_abs = np.inf
    if s.init_hmax is not None and init_h.size:
        init_hmax_abs = float(s.init_hmax * np.median(init_h))
    events = init_idx[(init_h <= init_hmax_abs) &
                      ~near(init_idx, veto, post, post)]
    # The noise PSD avoids every first-pass peak, the large ones included
    peaks_all = np.union1d(init_idx, veto)
    big = np.union1d(init_idx[init_h > init_hmax_abs], veto)
    history = []
    log = [f'initial: {init_idx.size} peaks above {s.init_nsigma:g} sigma '
           f'of the boxcar, {events.size} at most {s.init_hmax} x their '
           'median height']
    for it in range(s.of_iterations):
        if events.size == 0:
            raise DetectionError('no events to build the template from; '
                                 + '; '.join(log))
        # Only the isolated events make the template; all of them if none
        # is isolated
        iso = isolated_events(events, peaks_all, big, M, nfft)
        tmpl_idx = events[iso] if iso.any() else events
        avg, p = build_template(d, tmpl_idx, pre, post, sign,
                                s.baseline_gap)
        if p != pre:
            tmpl_idx = tmpl_idx + (p - pre)
            avg, p = build_template(d, tmpl_idx, pre, post, sign,
                                    s.baseline_gap)
        amp_ph = avg[p]
        history.append(avg / amp_ph)
        excl = exclusion_mask(n, peaks_all, pre, post)
        psd, nseg = noise_psd(d, nfft, excl, s.psd_max_segments)
        # The filter of a unit-height template (its first post_f samples
        # after the maximum) reads pulse amplitudes in units of d (x);
        # heights are in kHz/GHz
        of = OptimalFilter(avg[:pre + post_f] / amp_ph, p, psd, nfft)
        y = of.apply(d, sign, s.block) * PPM
        sub = _sub(n)
        sigma_y = mad_sigma(y[sub][~excl[sub]])
        del excl
        cidx, ch = candidates(y, sigma_y, distance = post_f)
        vetoed = near(cidx, veto, post, post)
        cidx, ch = cidx[~vetoed], ch[~vetoed]
        neg_h, clean = inverted_candidates(
            y, sigma_y, post_f, np.union1d(cidx[ch >= s.hmin], veto),
            pre + post_f, nfft)
        if auto:
            hmin = auto_hmin(ch, neg_h, clean, s.hmin, hmax,
                             s.hmin_false_frac, s.hmin_auto_max)
        keep = (ch >= hmin) & (ch <= hmax)
        events = cidx[keep]
        peaks_all = np.union1d(cidx[ch >= hmin], veto)
        big = np.union1d(cidx[ch > hmax], veto)
        log.append(f'iteration {it}: template of {tmpl_idx.size} isolated '
                   f'events, amplitude {amp_ph:.3g}, sigma_y {sigma_y:.3g}, '
                   f'{cidx.size} candidates, {events.size} in '
                   f'[{hmin:g}, {hmax:g}]')
    if events.size == 0:
        raise DetectionError('no events after the last iteration; '
                             + '; '.join(log))
    # Coincidence candidates: the lower threshold, never above hmin
    floor = min(s.coinc_nsigma * sigma_y, hmin)
    coinc_idx, coinc_h = candidates(y, sigma_y, distance = post_f,
                                    floor = floor, nsig = 0.)
    del y
    # The template stays the one the last filter was built from, so the
    # heights (in units of its amplitude), the filter, and the template
    # agree; the mean pulse of the final events is ``average_pulse``'s
    above = ch >= hmin
    coinc_idx = np.clip(coinc_idx, 0, n - 1)
    coinc_raw = raw_peaks(d, coinc_idx, sign, s.raw_half_width)
    return dict(
        cand_idx = cidx[above], cand_h = ch[above], events = events,
        template = avg, template_idx = tmpl_idx, p = p, pre = pre,
        post = post, filter_post = post_f, amp_ph = amp_ph, sign = sign,
        hmin = hmin, hmax = hmax, sigma_y = sigma_y,
        rate = events.size / (n * dt), psd = psd, nfft = nfft,
        filter = of.gc, history = np.array(history), neg_h = neg_h,
        clean_frac = clean, coinc_idx = coinc_idx, coinc_h = coinc_h,
        coinc_raw = coinc_raw, init_idx = init_idx,
        init_h = init_h.astype(np.float32), init_hmax_abs = init_hmax_abs,
        n_vetoed = int(np.sum(vetoed)), dt = dt, log = log)

def detect_with_filter(d, dt, ref, settings = None, hmin = None, hmax = None):
    """
    Find the photon pulses of one tone with a given optimal filter (e.g.
    the one ``run_detection`` built on another timestream of the same
    tone), without iterating.

    The filter is the reference's template and noise PSD. Heights are in
    kHz/GHz as always, so the reference's thresholds apply. The result has
    the keys of ``run_detection``; ``template``, ``p`` and ``amp_ph_own``
    describe this timestream's own mean pulse (of the events), while
    ``amp_ph`` stays the reference's (the filter's template amplitude)
    and ``filter_p`` is the reference template's maximum (the filter's
    alignment: ``OptimalFilter.from_kernel(filter, filter_p, nfft)``).

    Parameters:
    d (np.array): detection timestream (``detection_timestream``).
    dt (float): sample period, s.
    ref (dict): ``run_detection`` result to take the filter from
        (``template``, ``p``, ``pre``, ``post``, ``sign``, ``amp_ph``,
        ``psd``, ``nfft``, ``hmin``, ``hmax``, and ``filter_post`` and
        ``dt``, if present).
    settings (DetectionSettings or None): settings for ``coinc_nsigma``,
        ``raw_half_width``, ``baseline_gap`` and ``block``. None: the
        defaults.
    hmin, hmax (float or None): height window. None (default): the
        reference's.

    Returns:
    res (dict): as ``run_detection``, plus ``amp_ph_own``; ``history``,
        ``init_idx`` and ``init_h`` are empty and ``init_hmax_abs`` NaN.

    Raises:
    ValueError: if the sample period differs from the reference's.
    DetectionError: if no pulse is in the height window.
    """
    s = DetectionSettings() if settings is None else settings
    if 'dt' in ref and not np.isclose(float(ref['dt']), dt, rtol = 1e-9):
        raise ValueError(f"the reference filter is for dt = {ref['dt']}, "
                         f'not {dt}')
    hmin = float(ref['hmin']) if hmin is None else float(hmin)
    hmax = float(ref['hmax']) if hmax is None else float(hmax)
    sign, pre, post = int(ref['sign']), int(ref['pre']), int(ref['post'])
    post_f = int(ref.get('filter_post', post))
    nfft, amp_ph = int(ref['nfft']), float(ref['amp_ph'])
    M = pre + post
    n = d.shape[0]
    of = OptimalFilter(np.asarray(ref['template'])[:pre + post_f] / amp_ph,
                       int(ref['p']), np.asarray(ref['psd']), nfft)
    y = of.apply(d, sign, s.block) * PPM
    sub = _sub(n)
    # Filter noise away from pulses (found with a first estimate)
    sigma_y = mad_sigma(y[sub])
    cidx, ch = candidates(y, sigma_y, distance = post_f)
    excl = exclusion_mask(n, cidx[ch >= hmin], pre, post)
    sigma_y = mad_sigma(y[sub][~excl[sub]])
    del excl
    cidx, ch = candidates(y, sigma_y, distance = post_f)
    neg_h, clean = inverted_candidates(y, sigma_y, post_f,
                                       cidx[ch >= hmin], pre + post_f, nfft)
    events = cidx[(ch >= hmin) & (ch <= hmax)]
    log = [f'reference filter: sigma_y {sigma_y:.3g}, {cidx.size} '
           f'candidates, {events.size} in [{hmin:g}, {hmax:g}]']
    if events.size == 0:
        raise DetectionError('no events with the reference filter; '
                             + '; '.join(log))
    floor = min(s.coinc_nsigma * sigma_y, hmin)
    coinc_idx, coinc_h = candidates(y, sigma_y, distance = post_f,
                                    floor = floor, nsig = 0.)
    del y
    iso = isolated_events(events, cidx[ch >= hmin], cidx[ch > hmax], M,
                          nfft)
    tmpl_idx = events[iso] if iso.any() else events
    avg, p = build_template(d, tmpl_idx, pre, post, sign, s.baseline_gap)
    above = ch >= hmin
    return dict(
        cand_idx = cidx[above], cand_h = ch[above], events = events,
        template = avg, template_idx = tmpl_idx, p = p, pre = pre,
        post = post, filter_post = post_f, amp_ph = amp_ph,
        amp_ph_own = float(avg[p]), sign = sign, dt = dt,
        hmin = hmin, hmax = hmax, sigma_y = sigma_y,
        rate = events.size / (n * dt), psd = np.asarray(ref['psd']),
        nfft = nfft, filter = of.gc, filter_p = int(ref['p']),
        history = np.zeros((0, M)),
        neg_h = neg_h, clean_frac = clean, coinc_idx = coinc_idx,
        coinc_h = coinc_h,
        coinc_raw = raw_peaks(d, coinc_idx, sign, s.raw_half_width),
        init_idx = np.zeros(0, dtype = np.int64),
        init_h = np.zeros(0, dtype = np.float32), init_hmax_abs = np.nan,
        n_vetoed = 0, log = log)

################################################################################
################################# Coincidences #################################
################################################################################
def raw_peaks(d, idx, sign, half_width = 2):
    """
    Return the raw peak of each pulse: the largest sign * d within
    ``half_width`` samples of it. Unlike the filter height, this doesn't
    assume the photon pulse shape (e.g. for cosmic rays).

    Parameters:
    d (np.array): detection timestream.
    idx (np.array, int): pulse indices.
    sign (int): pulse sign.
    half_width (int): samples on each side. Default 2.

    Returns:
    raw (np.array, float32): raw peaks, in units of d, sign-corrected.
    """
    idx = np.asarray(idx, dtype = np.int64)
    n = d.shape[0]
    if idx.size == 0:
        return np.zeros(0, dtype = np.float32)
    offs = np.arange(-int(half_width), int(half_width) + 1)
    win = np.clip(idx[:, None] + offs[None, :], 0, n - 1)
    return (sign * d[win]).max(axis = 1).astype(np.float32)

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

def candidate_multiplicity(results, win):
    """
    Count, for each pulse candidate of each tone (``cand_idx``), the tones
    (itself included) with a coincidence candidate (``coinc_idx``) within
    +-``win`` samples of it. Use the tones of one tone write.

    Parameters:
    results (list of dict): ``run_detection`` results, one per tone.
    win (int): coincidence window, samples: a pulse at t is coincident
        with pulses in [t - win, t + win].

    Returns:
    mult (list of np.array, int): per tone, aligned with ``cand_idx`` (1:
        only this tone).
    """
    coinc = [np.sort(np.asarray(r['coinc_idx'], dtype = np.int64))
             for r in results]
    out = []
    for i, r in enumerate(results):
        e = np.asarray(r['cand_idx'], dtype = np.int64)
        m = np.ones(len(e), dtype = int)
        for j, other in enumerate(coinc):
            if j == i or not len(other):
                continue
            k = np.searchsorted(other, e - win)
            m += (k < len(other)) & \
                (other[np.clip(k, 0, len(other) - 1)] <= e + win)
        out.append(m)
    return out

def cosmic_events(results, win, min_mult = 3):
    """
    Group the coincidence candidates of the tones of one tone write into
    events, and return those seen by at least ``min_mult`` tones (cosmic
    rays).

    The candidates of all tones are sorted by time. An event starts at the
    earliest candidate not yet in an event, t0, and takes every candidate
    in [t0, t0 + 2 win] (the same width as the +-``win`` coincidence
    window). If a tone has more than one candidate in an event, the one
    with the largest filter height is kept.

    Parameters:
    results (list of dict): ``run_detection`` results, one per tone.
    win (int): coincidence window, samples.
    min_mult (int): fewest tones in a returned event. Default 3.

    Returns:
    events (dict): one entry per event: ``t0`` (sample of its first pulse,
        int64), ``mult`` (number of tones).
    hits (dict): one entry per tone of each event, in event order:
        ``event`` (index into ``events``), ``tone`` (index into
        ``results``), ``idx`` (pulse sample), ``h`` (filter height, kHz/GHz),
        ``x`` (filter height in units of d: h over ``PPM``), ``raw`` (raw
        peak, units of d, sign-corrected).
    """
    def column(key, dtype):
        """
        Concatenate one per-candidate quantity over the tones.

        Parameters:
        key (str or callable): result key, or a function of (i, result).
        dtype (type): output dtype.

        Returns:
        values (np.array): the concatenated values.
        """
        parts = [np.asarray(key(i, r) if callable(key) else r[key],
                            dtype = dtype) for i, r in enumerate(results)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype)

    t = column('coinc_idx', np.int64)
    tone = column(lambda i, r: np.full(len(r['coinc_idx']), i), np.int64)
    h = column('coinc_h', np.float64)
    raw = column('coinc_raw', np.float64)
    o = np.argsort(t, kind = 'stable')
    t, tone, h, raw = t[o], tone[o], h[o], raw[o]
    ev_t0, ev_mult, ev_rows = [], [], []
    i = 0
    while i < len(t):
        j = int(np.searchsorted(t, t[i] + 2 * win, side = 'right'))
        # The largest candidate of each tone in the event
        rows = np.arange(i, j)
        rows = rows[np.argsort(-h[rows], kind = 'stable')]
        _, first = np.unique(tone[rows], return_index = True)
        rows = np.sort(rows[first])
        if len(rows) >= min_mult:
            ev_t0.append(int(t[i]))
            ev_mult.append(len(rows))
            ev_rows.append(rows)
        i = j
    events = dict(t0 = np.asarray(ev_t0, dtype = np.int64),
                  mult = np.asarray(ev_mult, dtype = int))
    if ev_rows:
        rows = np.concatenate(ev_rows)
        event = np.concatenate([np.full(len(r), e, dtype = np.int64)
                                for e, r in enumerate(ev_rows)])
    else:
        rows, event = np.zeros(0, np.int64), np.zeros(0, np.int64)
    hits = dict(event = event, tone = tone[rows], idx = t[rows],
                h = h[rows], x = h[rows] / PPM, raw = raw[rows])
    return events, hits

def average_pulse(d, res, mult = None, max_mult = 1, gap = 3,
                  isolated = True):
    """
    Average the photon pulses of one tone: the events in the height window
    whose coincidence multiplicity is at most ``max_mult``, and that are
    isolated (``isolated_events``: no other candidate within the template
    length, no pulse above hmax within the filter length).

    Parameters:
    d (np.array): detection timestream.
    res (dict): ``run_detection`` result.
    mult (np.array or None): multiplicity of each candidate
        (``candidate_multiplicity``, aligned with ``res['cand_idx']``).
        None: no coincidence cut.
    max_mult (int): largest multiplicity kept. Default 1 (no other tone).
    gap (int): baseline gap (``build_template``). Default 3.
    isolated (bool): If True (default), only the isolated events.

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
    if isolated:
        ok &= isolated_events(idx, res['coinc_idx'], idx[h > res['hmax']],
                              res['pre'] + res['post'], res['nfft'])
    used = idx[ok & ~coinc]
    avg, p = build_template(d, used, res['pre'], res['post'], res['sign'],
                            gap)
    return avg, p, used, idx[coinc]

################################################################################
###################################### Tau #####################################
################################################################################
def fit_tau(avg, p, dt, n_skip = 2, lifetimes = 5.):
    """
    Fit A exp(-t / tau) + B to the decay of a mean pulse after its maximum:
    from ``n_skip`` samples after the maximum (where the pulse's rise and
    the filter's ringing are over) out to ``lifetimes`` decay times
    (first fit over the whole decay to find the decay time, then again
    over that range), so a long, flat tail doesn't weigh on the fit.

    Parameters:
    avg (np.array): mean pulse (positive-going).
    p (int): index of its maximum.
    dt (float): sample period, s.
    n_skip (int): samples after the maximum to skip. Default 2.
    lifetimes (float or None): length of the fit in decay times. Default
        5. None: the whole template after the maximum.

    Returns:
    tau (float): decay time in s (NaN if the fit fails).
    popt (np.array): (A, tau, B), with t = 0 at the first fitted sample
        (``p + n_skip``) (NaN if the fit fails).
    n_fit (int): samples fitted (0 if the fit fails).
    """
    y_all = np.asarray(avg[p + int(n_skip):], float)
    if y_all.size < 4 or not y_all[0] > 0:
        return np.nan, np.full(3, np.nan), 0
    # Fit in units of the first point: x pulses are ~1e-6, too small for
    # the fitter's default tolerances
    scale = y_all[0]

    def fit(n):
        """
        Fit the first n samples of the decay.

        Parameters:
        n (int): samples.

        Returns:
        popt (np.array or None): (A, tau, B) in units of the first point,
            or None if the fit fails.
        """
        y = y_all[:n] / scale
        t = np.arange(n) * dt
        try:
            return curve_fit(
                lambda t, A, tau, B: A * np.exp(-t / tau) + B, t, y,
                p0 = [1., t[-1] / 5, 0.],
                bounds = ([0, dt / 10, -1.], [10., 10 * t[-1] + dt, 1.]))[0]
        except (RuntimeError, ValueError):
            return None

    n = y_all.size
    popt = fit(n)
    if popt is not None and lifetimes is not None:
        n = int(min(max(round(lifetimes * popt[1] / dt), 4), y_all.size))
        popt = fit(n)
    if popt is None:
        return np.nan, np.full(3, np.nan), 0
    return float(popt[1]), popt * np.array([scale, 1., scale]), n
