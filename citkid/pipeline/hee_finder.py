"""
Fit high-energy event (HEE) tails in calibrated timestreams: the
multi-tone bookkeeping around ``signal.hee``, for devices without
photon pulses (or alongside them).

A timestream is one tone set of one run with a stream calibration, as
``photon_finder.list_timestreams`` lists them; its detection timestreams
come from ``photon_finder.TimestreamData`` (x from the raw streams in
memory, never saved). For one timestream:

1. The events come from a photon result (``events_from_photons``: the
   HEEs ``6a_find_photons.py`` saved) or from triggers on the
   tones streamed together (``find_events_timestream``).
2. ``analyze_timestream`` cuts every tone's pulses at the events, lines
   them up where each falls through the tail level, averages them and
   fits the average's tail (``analyze_tone``); the thresholds (height
   window, tail level, fit length; ``TailThresholds``) can then be
   reviewed in the HEE tail window
   (``pipeline.interactive.hee_tails``), which calls ``refit_tone``
   for the tones it changes (no data is read again).
3. ``save_timestream`` saves the result as ``hee/<name>`` at the store
   root (``photon_finder.result_name``), replaced when the timestream is
   analyzed again; ``load_thresholds`` reads a result's thresholds back
   to continue in the window.
"""
import dataclasses
import numpy as np
from tqdm.auto import tqdm
from .. import daq_layout as dl
from ..signal import hee as cr
from ..signal.photons import PPM
from . import photon_finder as pf

# The results' group at the store root, one child per timestream
HEE_GROUP = 'hee'

################################################################################
################################### Settings ###################################
################################################################################
@dataclasses.dataclass
class TailSettings:
    """
    Settings of the HEE tail analysis.

    Attributes:
    pulse_sign (int): sign of a pulse in the detection timestream.
        Default -1 (citkid's x).
    highpass_hz (float): drift removal before cutting the pulses, Hz.
        Default 10.
    pre_s, post_s (float): snippet length before and after an event, s.
        Default 0.5 ms and 5 ms (several decay times of a long tail).
    search_s (float): the peak is looked for this long after the event,
        s. Default 0.3 ms.
    baseline_gap (int): the baseline ends this many samples before the
        event. Default 3.
    trig_nsigma (float): trigger threshold for finding events without
        photons, MAD sigma of the 3-sample boxcar. Default 6.
    trig_distance (int): smallest spacing of one tone's triggers, samples.
        Default 3.
    coinc_win (int): largest gap between the triggers of one event,
        samples. Default 2.
    min_mult (int): fewest tones of an event. Default 3.
    """
    pulse_sign: int = -1
    highpass_hz: float = 10.
    pre_s: float = 0.5e-3
    post_s: float = 5e-3
    search_s: float = 0.3e-3
    baseline_gap: int = 3
    trig_nsigma: float = 6.
    trig_distance: int = 3
    coinc_win: int = 2
    min_mult: int = 3

    def __post_init__(self):
        """
        Check the trigger threshold: at zero every sample triggers and
        the dead time swallows the record.
        """
        if not self.trig_nsigma > 0:
            raise ValueError('TRIG_NSIGMA must be above zero (6 is usual): '
                             'at zero every sample triggers')

    def lengths(self, dt):
        """
        Return the snippet lengths in samples.

        Parameters:
        dt (float): sample period, s.

        Returns:
        pre, post, search (int): samples before the event, after it, and
            searched for the peak (each at least 1).
        """
        return (max(int(round(self.pre_s / dt)), 1),
                max(int(round(self.post_s / dt)), 4),
                max(int(round(self.search_s / dt)), 1))

@dataclasses.dataclass
class TailThresholds:
    """
    Thresholds of the tail fits: global values, with overrides per
    resonator (as ``photon_finder.Thresholds``).

    Attributes:
    hmin, hmax (float): height window of the pulses fitted, in kHz/GHz
        (x times 1e6). Default 0 and 1000.
    level (float): a pulse's tail begins where it falls through this,
        kHz/GHz. Default 10.
    fit_s (float): length of each tail fit, s. Default 2 ms.
    overrides (dict): res_idx: dict with any of 'hmin', 'hmax', 'level',
        'fit_s'.
    bad (dict): res_idx: reason, for the tones marked bad in the window.
    limits (dict): review limits of the tone overview: 'height', 'tau',
        'rate': [low, high] (kHz/GHz, s, 1/s).
    reviewed (set of int): resonators the user reviewed in the window
        (``mark_reviewed``), shown with ``*`` in its list.
    """
    hmin: float = 0.
    hmax: float = 1000.
    level: float = 10.
    fit_s: float = 2e-3
    overrides: dict = dataclasses.field(default_factory = dict)
    bad: dict = dataclasses.field(default_factory = dict)
    limits: dict = dataclasses.field(default_factory = dict)
    reviewed: set = dataclasses.field(default_factory = set)

    KEYS = ('hmin', 'hmax', 'level', 'fit_s')

    def mark_reviewed(self, res_idx):
        """
        Record that the user reviewed a resonator in the window: gave it
        its own values, or looked at it and moved on. Applying values to
        every tone does not count.

        Parameters:
        res_idx (int): resonator index.
        """
        self.reviewed.add(int(res_idx))

    def is_bad(self, res_idx):
        """
        Return whether a resonator is marked bad.

        Parameters:
        res_idx (int): resonator index.

        Returns:
        bad (bool): True if marked bad.
        """
        return int(res_idx) in self.bad

    def set_bad(self, res_idx, reason = ''):
        """
        Mark a resonator bad.

        Parameters:
        res_idx (int): resonator index.
        reason (str): why. Default ''.
        """
        self.bad[int(res_idx)] = str(reason)

    def clear_bad(self, res_idx):
        """
        Clear a resonator's bad mark, if any.

        Parameters:
        res_idx (int): resonator index.
        """
        self.bad.pop(int(res_idx), None)

    def has_own(self, res_idx, keys = KEYS):
        """
        Say whether a resonator has its own value of any of some keys.

        Parameters:
        res_idx (int): resonator index.
        keys (sequence of str): override keys. Default: all.

        Returns:
        own (bool): True if it has one.
        """
        return bool(set(keys) & set(self.overrides.get(int(res_idx), {})))

    def set_all(self, **values):
        """
        Set global values, dropping every resonator's own value of them.

        Parameters:
        values: any of 'hmin', 'hmax', 'level', 'fit_s'.
        """
        for key, v in values.items():
            setattr(self, key, float(v))
            for r in list(self.overrides):
                self.overrides[r].pop(key, None)
                if not self.overrides[r]:
                    del self.overrides[r]

    def for_tone(self, res_idx):
        """
        Return the thresholds of one resonator.

        Parameters:
        res_idx (int): resonator index.

        Returns:
        values (dict): 'hmin', 'hmax', 'level', 'fit_s'.
        """
        out = dict(hmin = self.hmin, hmax = self.hmax, level = self.level,
                   fit_s = self.fit_s)
        out.update(self.overrides.get(int(res_idx), {}))
        return out

    def to_dict(self):
        """
        Return the thresholds as JSON-ready values.

        Returns:
        values (dict): 'hmin', 'hmax', 'level', 'fit_s', 'overrides'
            (keys as strings), 'bad' (keys as strings), 'limits',
            'reviewed' (sorted list).
        """
        return dict(hmin = self.hmin, hmax = self.hmax, level = self.level,
                    fit_s = self.fit_s,
                    overrides = {str(k): dict(v)
                                 for k, v in self.overrides.items()},
                    bad = {str(k): str(v) for k, v in self.bad.items()},
                    limits = dict(self.limits),
                    reviewed = sorted(int(r) for r in self.reviewed))

    @classmethod
    def from_dict(cls, values):
        """
        Rebuild thresholds from ``to_dict`` output.

        Parameters:
        values (dict): the saved values.

        Returns:
        thresholds (TailThresholds): the thresholds.
        """
        return cls(
            hmin = float(values['hmin']), hmax = float(values['hmax']),
            level = float(values['level']), fit_s = float(values['fit_s']),
            overrides = {int(k): dict(v)
                         for k, v in values.get('overrides', {}).items()},
            bad = {int(k): str(v) for k, v in values.get('bad', {}).items()},
            limits = dict(values.get('limits', {})),
            reviewed = {int(r) for r in values.get('reviewed', [])})

################################################################################
#################################### Results ###################################
################################################################################
def results_group(root):
    """
    Return the results' group at the store root, making it if needed.

    Parameters:
    root (zarr.Group): the store root (writable).

    Returns:
    grp (zarr.Group): ``hee``.
    """
    if HEE_GROUP not in root:
        dl.create_group(root, HEE_GROUP, 'analysis')
    return root[HEE_GROUP]

def list_results(root):
    """
    List the saved HEE results.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    names (list of str): the result names, sorted.
    """
    if HEE_GROUP not in root:
        return []
    return sorted(root[HEE_GROUP].group_keys())

def describe_results(root):
    """
    Describe the saved HEE results, one line each.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    text (str): the description.
    """
    lines = ['Previous results (hee/<name>: run, tone set, events, '
             'created):']
    for name in list_results(root):
        a = root[HEE_GROUP][name].attrs
        n_bad = len(a.get('thresholds', {}).get('bad', {}))
        lines.append(f"  {name}: {a.get('run')} set {a.get('tone_set')}, "
                     f"{a.get('n_events')} events from {a.get('events')}, "
                     f"{a.get('created')}"
                     + (f', {n_bad} bad' if n_bad else ''))
    if len(lines) == 1:
        lines.append('  none')
    return '\n'.join(lines)

def describe_result(grp):
    """
    Describe one saved result in a few lines: when and how it was made,
    and how much of it was set by hand.

    Parameters:
    grp (zarr.Group): the result (``hee/<name>``).

    Returns:
    text (str): the description.
    """
    a = grp.attrs
    th = TailThresholds.from_dict(a.get('thresholds', {}))
    n = grp['res_idxs'].shape[0] if 'res_idxs' in grp else 0
    return (f"{a.get('run')} tone set {a.get('tone_set')}, {n} tones, "
            f"{a.get('n_events')} events from {a.get('events')}\n"
            f"created {a.get('created')}\n"
            f"{len(th.overrides)} tones with their own values, "
            f"{len(th.bad)} marked bad")

def load_thresholds(grp):
    """
    Read a saved result's thresholds.

    Parameters:
    grp (zarr.Group): the result.

    Returns:
    thresholds (TailThresholds): its thresholds, overrides, bad marks and
        review limits.
    """
    return TailThresholds.from_dict(grp.attrs['thresholds'])

################################################################################
#################################### Events ####################################
################################################################################
def events_from_photons(grp):
    """
    Read the HEEs a photon result saved, as events.

    Parameters:
    grp (zarr.Group): a ``photons/<name>`` result.

    Returns:
    events (dict): 't0' (sample, int64), 'mult' (tones), 'write_idx'
        (tone write), arrays over the events.
    """
    ev = grp['hee/events']
    return dict(t0 = np.asarray(ev['t0'][:], dtype = np.int64),
                mult = np.asarray(ev['mult'][:], dtype = np.int64),
                write_idx = np.asarray(ev['write_idx'][:], dtype = np.int64))

def find_events_timestream(root, ts, data, settings, progress = True):
    """
    Find the HEEs of a timestream from triggers, per tone write
    (``signal.hee.find_events`` on the tones streamed together).

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    settings (TailSettings): trigger settings.
    progress (bool): If True (default), show a progress bar.

    Returns:
    events (dict): 't0', 'mult', 'write_idx' (``events_from_photons``).
    """
    cols = dict(t0 = [], mult = [], write_idx = [])
    for w, _, rows in tqdm(pf.tone_writes(root, ts), disable = not progress,
                           desc = 'Finding events'):
        ds = [d for _, d in data.get(rows) if not isinstance(d, str)]
        if not ds:
            continue
        t0, mult = cr.find_events(ds, settings.trig_nsigma,
                                  settings.coinc_win, settings.min_mult,
                                  settings.trig_distance,
                                  sign = settings.pulse_sign,
                                  dead = settings.lengths(data.dt)[1])
        cols['t0'].append(t0)
        cols['mult'].append(mult)
        cols['write_idx'].append(np.full(len(t0), w, dtype = np.int64))
    return {k: (np.concatenate(v).astype(np.int64) if v
                else np.zeros(0, dtype = np.int64)) for k, v in cols.items()}

################################################################################
#################################### Analysis ##################################
################################################################################
def analyze_tone(d, t0, dt, settings, th):
    """
    Cut one tone's pulses at the events, align and average them, and
    fit the mean's tail.

    Parameters:
    d (np.array): detection timestream.
    t0 (np.array, int): the events of its tone write.
    dt (float): sample period, s.
    settings (TailSettings): analysis settings.
    th (dict): the tone's thresholds (``TailThresholds.for_tone``).

    Returns:
    res (dict): 'pulses' ((events, pre + post), x, float32), 't0' (the
        events kept), 'pre', 'post', 'h' (peak heights, kHz/GHz), 'p'
        (peak index per pulse), then ``refit_tone``'s keys.
    """
    pre, post, search = settings.lengths(dt)
    pulses, kept = cr.cut_pulses(d, t0, pre, post, settings.pulse_sign,
                                 settings.baseline_gap)
    h, p = cr.peak_heights(pulses, pre, search)
    res = dict(pulses = pulses.astype(np.float32),
               t0 = np.asarray(t0, dtype = np.int64)[kept], pre = pre,
               post = post, h = h * PPM, p = p, dt = dt,
               sign = int(settings.pulse_sign))
    return refit_tone(res, th)

def refit_tone(res, th):
    """
    Align, average and fit a tone's pulses again with other thresholds,
    from its cut pulses.

    Parameters:
    res (dict): ``analyze_tone`` output.
    th (dict): the tone's thresholds (``TailThresholds.for_tone``).

    Returns:
    res (dict): a copy with 'hmin', 'hmax', 'level', 'fit_s' (kHz/GHz,
        s), 'in_window' (pulses in the height window), 'start' (tail
        start per pulse, -1 if never below the level), 'used' (in the
        window with a tail start), 'n_used', 'anchor' (the sample the
        tail starts are lined up at: ``pre``), 'aligned' ((n_used, L),
        x, NaN-padded), 'avg' (their mean, x), 'count' (pulses per
        sample of the mean), 'tau' (decay time of the mean's tail, s;
        NaN if the fit failed), 'popt', 'n_fit'.
    """
    res = dict(res)
    res.update(hmin = float(th['hmin']), hmax = float(th['hmax']),
               level = float(th['level']), fit_s = float(th['fit_s']))
    pulses = np.asarray(res['pulses'], dtype = float)
    h, p = res['h'], res['p']
    inside = (h >= res['hmin']) & (h <= res['hmax'])
    level = res['level'] / PPM
    start = np.array([cr.tail_start(pulses[k], int(p[k]), level)
                      if inside[k] else -1 for k in range(len(h))],
                     dtype = np.int64)
    used = inside & (start >= 0)
    anchor = int(res['pre'])
    aligned = cr.align_tails(pulses[used], start[used], anchor)
    avg, count = cr.mean_aligned(aligned)
    if used.any():
        tau, popt, n_fit = cr.fit_mean_tail(avg, anchor, res['dt'],
                                            res['fit_s'])
    else:
        avg = np.full(pulses.shape[1], np.nan)
        count = np.zeros(pulses.shape[1], dtype = int)
        tau, popt, n_fit = np.nan, np.full(3, np.nan), 0
    res.update(in_window = inside, start = start, used = used,
               n_used = int(used.sum()), anchor = anchor,
               aligned = aligned.astype(np.float32), avg = avg,
               count = count, tau = tau, popt = popt, n_fit = n_fit)
    return res

def analyze_timestream(
    root, ts, data, events, settings, thresholds, rows = None,
    progress = True
):
    """
    Cut and fit every tone's pulses at the events of its tone write.

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    events (dict): the events ('t0', 'write_idx').
    settings (TailSettings): analysis settings.
    thresholds (TailThresholds): the thresholds.
    rows (list of int or None): DataSet rows to analyze. Default None:
        all.
    progress (bool): If True (default), show a progress bar.

    Returns:
    results (dict): row: ``analyze_tone`` result (with 'res_idx'), or
        the error message (str) for a tone without a detection
        timestream.
    """
    res_of = dict(zip(ts['rows'].tolist(), ts['res_idxs'].tolist()))
    write_of = {}
    for w, _, w_rows in pf.tone_writes(root, ts):
        for r in w_rows:
            write_of[int(r)] = w
    rows = [int(r) for r in (ts['rows'] if rows is None else rows)]
    out = {}
    for r, d in tqdm(data.get(rows), total = len(rows),
                     disable = not progress, desc = 'Fitting tails'):
        if isinstance(d, str):
            out[r] = d
            continue
        t0 = events['t0'][events['write_idx'] == write_of.get(r, -1)]
        res = analyze_tone(d, t0, float(ts['dt']), settings,
                           thresholds.for_tone(res_of[r]))
        res['res_idx'] = int(res_of[r])
        out[r] = res
    return out

def refit_timestream(ts, results, thresholds, rows = None):
    """
    Fit some tones' tails again with the thresholds (``refit_tone``),
    without reading data.

    Parameters:
    ts (dict): the timestream.
    results (dict): ``analyze_timestream`` output.
    thresholds (TailThresholds): the thresholds.
    rows (list of int or None): DataSet rows. Default None: all.

    Returns:
    results (dict): row: the new result, for the rows with a result.
    """
    res_of = dict(zip(ts['rows'].tolist(), ts['res_idxs'].tolist()))
    rows = [int(r) for r in (ts['rows'] if rows is None else rows)]
    out = {}
    for r in rows:
        res = results.get(r)
        if isinstance(res, dict):
            out[r] = refit_tone(res, thresholds.for_tone(res_of[r]))
    return out

################################################################################
##################################### Saving ###################################
################################################################################
def save_timestream(grp, ts, results, events, thresholds, settings, mode):
    """
    Save one timestream's HEE tails.

    Layout of ``grp``: attrs describe the timestream, the settings, the
    thresholds, ``events`` (where the events came from) and
    ``n_events``; per tone (in the order of ``ts['rows']``): ``rows``,
    ``res_idxs``, ``status`` ('' if analyzed, else the error), ``hmin``,
    ``hmax``, ``level`` (kHz/GHz), ``fit_s``, ``n_pulses`` (events with
    a snippet), ``n_used`` (pulses averaged), ``rate`` (pulses averaged
    per second), ``tau`` (decay time of the mean's tail, s), ``n_fit``,
    ``avg_pulse`` (the aligned mean, x, (ntones, pre + post); attr
    ``anchor`` is the sample the tails are lined up at), ``bad``,
    ``bad_reason``; ``events/`` (``t0``, ``time_s``, ``mult``,
    ``write_idx``); ``pulses/`` (one row per tone and event: ``row``,
    ``res_idx``, ``t0``, ``h`` (kHz/GHz), ``p``, ``in_window``,
    ``start``, ``used``).

    Parameters:
    grp (zarr.Group): the (new, empty) group.
    ts (dict): the timestream.
    results (dict): ``analyze_timestream`` output.
    events (dict): the events.
    thresholds (TailThresholds): the thresholds used.
    settings (TailSettings): the settings used.
    mode (str): where the events came from, e.g. 'triggers' or
        'photons/dark'.
    """
    rows = [int(r) for r in ts['rows']]
    ok = [isinstance(results.get(r), dict) for r in rows]
    ref = next((results[r] for r, g in zip(rows, ok) if g), None)
    M = 0 if ref is None else ref['pre'] + ref['post']
    duration = float(ts['stop'] - ts['start']) * float(ts['dt'])
    grp.attrs.update(dl.to_jsonable(dict(
        dataset = ts['dataset'], run = ts['run'], name = ts['name'],
        tone_set = ts['tone_set'], source = ts['source']['group'],
        writes = list(ts['source']['writes']), start = ts['start'],
        stop = ts['stop'], dt = ts['dt'], events = mode,
        n_events = int(len(events['t0'])),
        thresholds = thresholds.to_dict(),
        settings = dataclasses.asdict(settings),
        pre = None if ref is None else ref['pre'],
        post = None if ref is None else ref['post'],
        anchor = None if ref is None else ref['anchor'])))

    def per_tone(key, default = np.nan):
        """
        Collect one scalar per tone.

        Parameters:
        key (str or callable): result key, or a function of the result.
        default (float): value for failed tones. Default NaN.

        Returns:
        values (np.array): the values.
        """
        return np.array([
            (key(results[r]) if callable(key) else results[r][key])
            if g else default for r, g in zip(rows, ok)], dtype = float)

    dl.write_array(grp, 'rows', np.asarray(rows, dtype = np.int64))
    dl.write_array(grp, 'res_idxs', np.asarray(ts['res_idxs'],
                                               dtype = np.int64))
    dl.write_array(grp, 'status', np.array(
        ['' if g else str(results.get(r, 'not run'))
         for r, g in zip(rows, ok)]))
    for key in ('hmin', 'hmax', 'level', 'fit_s', 'tau'):
        dl.write_array(grp, key, per_tone(key))
    dl.write_array(grp, 'n_fit', per_tone('n_fit', 0))
    dl.write_array(grp, 'n_pulses', per_tone(lambda res: len(res['h']), 0))
    dl.write_array(grp, 'n_used', per_tone('n_used', 0))
    dl.write_array(grp, 'rate', per_tone(
        lambda res: res['n_used'] / duration if duration > 0 else np.nan))
    avg = np.full((len(rows), M), np.nan)
    for i, (r, g) in enumerate(zip(rows, ok)):
        if g:
            avg[i] = results[r]['avg']
    dl.write_array(grp, 'avg_pulse', avg)
    dl.write_array(grp, 'bad', np.array(
        [thresholds.is_bad(ri) for ri in ts['res_idxs']], dtype = bool))
    dl.write_array(grp, 'bad_reason', np.array(
        [thresholds.bad.get(int(ri), '') for ri in ts['res_idxs']]))
    ev = dl.create_group(grp, 'events', 'analysis')
    for k in ('t0', 'mult', 'write_idx'):
        dl.write_array(ev, k, np.asarray(events[k], dtype = np.int64))
    dl.write_array(ev, 'time_s', np.asarray(events['t0'], dtype = float)
                   * float(ts['dt']))
    cols = {k: [] for k in ('row', 'res_idx', 't0', 'h', 'p', 'in_window',
                            'start', 'used')}
    for r, g in zip(rows, ok):
        if not g:
            continue
        res = results[r]
        n = len(res['h'])
        cols['row'].append(np.full(n, r, dtype = np.int64))
        cols['res_idx'].append(np.full(n, res['res_idx'], dtype = np.int64))
        for k in ('t0', 'h', 'p', 'in_window', 'start', 'used'):
            cols[k].append(np.asarray(res[k]))
    pulses = dl.create_group(grp, 'pulses', 'analysis')
    for k, v in cols.items():
        dtype = bool if k in ('in_window', 'used') else \
            (float if k == 'h' else np.int64)
        dl.write_array(pulses, k, np.concatenate(v).astype(dtype) if v
                       else np.zeros(0, dtype))
