"""
Find photons in raw timestreams with a stream calibration (``ts_cal{c}``,
made by 05_calibrate_streams.py): the multi-tone bookkeeping around
``signal.photons``.

A timestream here is one tone set of one run (``ts{n}``), over the whole
run, whose tone set the calibration has a DataSet for (the run need not
be one the calibration was fit on). ``list_timestreams`` lists them.
Its x is computed from the raw streams with
``stream_sources.apply_stream_calibration``, in memory only
(``TimestreamData``, which keeps the detection timestreams up to a size
limit): calibrated timestreams are never saved, only the photon results.

For one timestream:

1. ``detect_timestream`` runs the detection on every tone, either with
   ``run_detection`` (thresholds given per tone, e.g. a previous result's,
   ``load_reference``) or with ``detect_with_filter`` and a previous
   result's optimal filters (matched by ``res_idx``).
2. The thresholds can then be reviewed in the photon threshold window
   (``pipeline.interactive.photon_thresholds``), which calls
   ``detect_timestream`` again for the tones it changes.
3. ``finish_timestream`` works per tone write (the tones streamed
   together): the multiplicity of each pulse (``candidate_multiplicity``),
   the cosmic rays (``cosmic_events``), and per tone the mean photon pulse
   of the pulses in the height window with multiplicity 1
   (``average_pulse``) and its decay time (``fit_tau``).
4. ``save_timestream`` saves the results in one group, from which
   ``load_reference`` reads a later run's thresholds or filters.
"""
import dataclasses
import numpy as np
from tqdm.auto import tqdm
from .. import daq_layout as dl
from ..signal import photons as ph
from .stream_sources import apply_stream_calibration

# What a run takes from a previous result (``load_reference``)
REUSE_OPTIONS = ('settings', 'filter')

################################################################################
################################## Timestreams #################################
################################################################################
def list_timestreams(root, datasets):
    """
    List the timestreams a stream calibration can calibrate: every tone set
    of every run with streams whose tone set (and resonator list) has a
    DataSet in the calibration.

    Parameters:
    root (zarr.Group): the store root.
    datasets (dict): ``stream_sources.open_calibration`` output.

    Returns:
    timestreams (list of dict): in run order: ``index``, ``run``
        ('ts{n}'), ``run_index`` (n), ``name``, ``tone_set``, ``dataset``
        (DataSet name), ``source`` (the run's tone set, whole, as
        ``daq_layout.select_streams`` makes it), ``start``, ``stop``
        (samples), ``dt``, ``rows`` (the DataSet's rows, np.array),
        ``res_idxs`` (their resonator indices).
    """
    by_set = {(int(info['tone_set']), info['resonators']): name
              for name, info in datasets.items()}
    out = []
    for run in dl.stream_runs(root):
        if not run.get('n_chunks'):
            continue
        try:
            sources, _ = dl.select_streams(root, run = run['run'])
        except (KeyError, ValueError):
            continue
        for src in sources:
            s = int(root[src['group']].attrs['tone_set'])
            dname = by_set.get((s, run.get('resonators')))
            if dname is None:
                continue
            DS = datasets[dname]['DS']
            res_idxs = np.asarray(DS.res_idxs[:])
            stream = root[src['group']][src['writes'][0]][
                src.get('stream', dl.STREAM_NAME)]
            out.append(dict(
                index = len(out), run = run['group'], run_index = run['run'],
                name = run.get('name'), tone_set = s, dataset = dname,
                source = src, start = int(src['start']),
                stop = int(src['stop']),
                dt = float(np.asarray(stream['dt'][...])),
                rows = np.arange(len(res_idxs)), res_idxs = res_idxs))
    return out

def describe_timestreams(timestreams):
    """
    Describe the timestreams of a stream calibration, one line each.

    Parameters:
    timestreams (list of dict): ``list_timestreams`` output.

    Returns:
    text (str): the description.
    """
    lines = ['Timestreams (index: run, name, tone set, duration, tones, '
             'DataSet):']
    for t in timestreams:
        n = t['stop'] - t['start']
        lines.append(
            f"  {t['index']}: {t['run']}, {t['name']!r}, set "
            f"{t['tone_set']}, {n * t['dt']:.3g} s, {len(t['rows'])} "
            f"tones, {t['dataset']}")
    return '\n'.join(lines)

def select_timestreams(timestreams, choice):
    """
    Pick timestreams by index or run name.

    Parameters:
    timestreams (list of dict): ``list_timestreams`` output.
    choice (list of int or str): timestream indices, or run names (every
        tone set of the latest run with that name).

    Returns:
    chosen (list of dict): the timestreams, in the order of ``choice``.

    Raises:
    KeyError: if a choice matches no timestream.
    """
    out = []
    for c in choice:
        if isinstance(c, str):
            named = [t for t in timestreams if t['name'] == c]
            match = [t for t in named if named and
                     t['run_index'] == named[-1]['run_index']]
        else:
            match = [t for t in timestreams if t['index'] == int(c)]
        if not match:
            raise KeyError(f'no timestream {c!r} (see the list)')
        out += match
    return out

class TimestreamData:
    """
    Detection timestreams of one timestream's tones, computed from its raw
    streams with the calibration (``apply_stream_calibration``), in memory
    only. They are kept while their total size is under ``max_bytes``;
    others are computed again when asked for, in blocks of tones that fit
    the memory (reading only those tones' chunks of the raw stream).

    Attributes:
    dt (float): sample period, s.
    nsamples (int): samples per tone.
    settings (DetectionSettings): for ``highpass_hz``.
    max_bytes (float): memory limit.
    """

    def __init__(self, root, ts, DS, settings, max_bytes = 4e9):
        """
        Set up the timestream's data.

        Parameters:
        root (zarr.Group): the store root.
        ts (dict): the timestream (``list_timestreams``).
        DS (DataSet): its calibration DataSet.
        settings (DetectionSettings): detection settings.
        max_bytes (float): memory limit, bytes. Default 4e9.
        """
        self._root, self._ts, self._DS = root, ts, DS
        self.dt = float(ts['dt'])
        self.nsamples = int(ts['stop'] - ts['start'])
        self.settings = settings
        self.max_bytes = float(max_bytes)
        self._kept = {}

    def get(self, rows):
        """
        Yield the detection timestreams of some tones.

        Parameters:
        rows (list of int): DataSet rows.

        Yields:
        row (int), d (np.array, float32 or str): each tone's detection
            timestream, or an error message if its x is not finite.
        """
        rows = [int(r) for r in rows]
        # x (float64) and the detection timestreams (float32) of a block
        block = max(1, int(self.max_bytes // max(self.nsamples * 12, 1)))
        for i in range(0, len(rows), block):
            part = rows[i:i + block]
            need = [r for r in part if r not in self._kept]
            fresh = {}
            if need:
                x = apply_stream_calibration(self._DS, self._root,
                                             self._ts['source'],
                                             rows = need)['xt']
                for r, xr in zip(need, x):
                    fresh[r] = self._detection(xr)
                    self._keep(r, fresh[r])
                del x
            for r in part:
                yield r, self._kept.get(r, fresh.get(r))

    def _detection(self, x):
        """
        Make one tone's detection timestream.

        Parameters:
        x (np.array): calibrated x.

        Returns:
        d (np.array, float32 or str): detection timestream, or the error
            message if x is not finite.
        """
        x = np.asarray(x, dtype = np.float64)
        if not np.isfinite(x).any():
            return 'the calibrated timestream is not finite (calibration ' \
                'missing?)'
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        return ph.detection_timestream(x, self.dt, self.settings.highpass_hz)

    def _keep(self, row, d):
        """
        Keep a detection timestream if it fits the memory limit.

        Parameters:
        row (int): DataSet row.
        d (np.array or str): the timestream.
        """
        size = sum(v.nbytes for v in self._kept.values()
                   if hasattr(v, 'nbytes'))
        if isinstance(d, str) or size + d.nbytes <= self.max_bytes:
            self._kept[row] = d

################################################################################
################################## Thresholds ##################################
################################################################################
@dataclasses.dataclass
class Thresholds:
    """
    Height thresholds of the detection: global values, with overrides per
    resonator.

    Attributes:
    hmin, hmax (float): height window of the filter output, in kHz/GHz (the
        fractional frequency shift x times 1e6).
    init_hmax (float or None): first-pass upper cut, in units of the median
        first-pass peak height (None: no cut).
    overrides (dict): res_idx: dict with any of 'hmin', 'hmax',
        'init_hmax'.
    """
    hmin: float = 0.5
    hmax: float = 1.6
    init_hmax: float = 3.
    overrides: dict = dataclasses.field(default_factory = dict)

    def for_tone(self, res_idx):
        """
        Return the thresholds of one resonator.

        Parameters:
        res_idx (int): resonator index.

        Returns:
        values (dict): 'hmin', 'hmax', 'init_hmax'.
        """
        out = dict(hmin = self.hmin, hmax = self.hmax,
                   init_hmax = self.init_hmax)
        out.update(self.overrides.get(int(res_idx), {}))
        return out

    def to_dict(self):
        """
        Return the thresholds as JSON-ready values.

        Returns:
        values (dict): 'hmin', 'hmax', 'init_hmax', 'overrides' (keys as
            strings).
        """
        return dict(hmin = self.hmin, hmax = self.hmax,
                    init_hmax = self.init_hmax,
                    overrides = {str(k): dict(v)
                                 for k, v in self.overrides.items()})

################################################################################
################################### Detection ##################################
################################################################################
def detect_timestream(
    ts, data, settings, thresholds, rows = None, reference = None,
    progress = True
):
    """
    Run the photon detection on the tones of one timestream.

    Parameters:
    ts (dict): the timestream (``list_timestreams``).
    data (TimestreamData): its detection timestreams.
    settings (DetectionSettings): detection settings (the thresholds come
        from ``thresholds``).
    thresholds (Thresholds): height thresholds.
    rows (list of int or None): DataSet rows to run. None (default): all
        the timestream's tones.
    reference (dict or None): res_idx: ``run_detection`` result of another
        timestream, to apply its optimal filter (``detect_with_filter``)
        instead of iterating. Tones without a reference are detected from
        scratch. None (default): every tone from scratch.
    progress (bool): If True (default), show a progress bar.

    Returns:
    results (dict): row: the detection result (dict), or the error
        message (str) if the detection failed.
    """
    rows = ts['rows'] if rows is None else rows
    res_of_row = dict(zip(ts['rows'].tolist(), ts['res_idxs'].tolist()))
    out = {}
    for row, d in tqdm(data.get(rows), total = len(rows),
                       disable = not progress, desc = 'Detecting'):
        r_idx = res_of_row[row]
        th = thresholds.for_tone(r_idx)
        if isinstance(d, str):
            out[row] = d
            continue
        try:
            ref = None if reference is None else reference.get(r_idx)
            if ref is not None:
                res = ph.detect_with_filter(d, data.dt, ref, settings,
                                            hmin = th['hmin'],
                                            hmax = th['hmax'])
            else:
                s = dataclasses.replace(settings,
                                        init_hmax = th['init_hmax'])
                res = ph.run_detection(d, data.dt, s, hmin = th['hmin'],
                                       hmax = th['hmax'])
            res['res_idx'] = r_idx
            out[row] = res
        except (ph.DetectionError, ValueError) as err:
            out[row] = str(err)
    return out

def tone_writes(root, ts):
    """
    Group a timestream's tones by tone write (the tones streamed together:
    at fast decimation stages a tone set streams in several writes).

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.

    Returns:
    writes (list of tuple): (write index in the source's ``writes``, write
        name, DataSet rows) per write.
    """
    src = ts['source']
    sweep = root[src['group']]
    pos = {int(r): i for i, r in enumerate(sweep['res_idxs'][...])}
    sweep_rows = np.array([pos[int(r)] for r in ts['res_idxs']])
    out = []
    for k, name in enumerate(src['writes']):
        wg = sweep[name]
        held = np.asarray(wg['sweep_rows'][...]) if 'sweep_rows' in wg \
            else np.arange(len(pos))
        rows = ts['rows'][np.isin(sweep_rows, held)]
        if len(rows):
            out.append((k, name, rows))
    return out

def finish_timestream(
    root, ts, data, results, settings, coinc_win = 2, cosmic_min_mult = 3,
    progress = True
):
    """
    Find coincidences and cosmic rays per tone write, and average each
    tone's photon pulses (height window, multiplicity 1).

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    results (dict): ``detect_timestream`` output for all its tones.
    settings (DetectionSettings): for ``baseline_gap``.
    coinc_win (int): coincidence window, samples: a pulse at t is
        coincident with other tones' pulses in [t - win, t + win].
        Default 2.
    cosmic_min_mult (int): fewest tones of a cosmic ray. Default 3.
    progress (bool): If True (default), show a progress bar.

    Returns:
    summary (dict): ``mult`` (row: multiplicity of each pulse candidate),
        ``avg`` (row: mean photon pulse), ``p`` (row: index of its
        maximum), ``used`` (row: the pulses averaged), ``tau`` (row: decay
        time, s), ``cosmic`` (dict of arrays: ``events`` with ``t0``,
        ``time_s``, ``mult``, ``write_idx``; ``hits`` with ``event``,
        ``row``, ``res_idx``, ``idx``, ``h``, ``x``, ``raw``).
    """
    mult, avg, peak, used, tau = {}, {}, {}, {}, {}
    ev_cols = {k: [] for k in ('t0', 'mult', 'write_idx')}
    hit_cols = {k: [] for k in ('event', 'row', 'res_idx', 'idx', 'h', 'x',
                                'raw')}
    n_events = 0
    for w, _, rows in tone_writes(root, ts):
        good = [int(r) for r in rows if isinstance(results.get(int(r)),
                                                   dict)]
        res = [results[r] for r in good]
        if not res:
            continue
        for r, m in zip(good, ph.candidate_multiplicity(res, coinc_win)):
            mult[r] = m
        events, hits = ph.cosmic_events(res, coinc_win, cosmic_min_mult)
        k = len(events['t0'])
        ev_cols['t0'].append(events['t0'])
        ev_cols['mult'].append(events['mult'])
        ev_cols['write_idx'].append(np.full(k, w))
        tone_row = np.asarray(good, dtype = np.int64)[hits['tone']]
        hit_cols['event'].append(hits['event'] + n_events)
        hit_cols['row'].append(tone_row)
        hit_cols['res_idx'].append(np.asarray(
            [results[r]['res_idx'] for r in tone_row], dtype = np.int64))
        for key in ('idx', 'h', 'x', 'raw'):
            hit_cols[key].append(hits[key])
        n_events += k
    rows = [int(r) for r in ts['rows'] if int(r) in mult]
    for r, d in tqdm(data.get(rows), total = len(rows),
                     disable = not progress, desc = 'Averaging'):
        try:
            a, p, u, _ = ph.average_pulse(d, results[r], mult[r],
                                          gap = settings.baseline_gap)
        except ValueError:
            continue
        avg[r], peak[r], used[r] = a, p, u
        tau[r] = ph.fit_tau(a, p, data.dt, settings.tau_skip,
                            settings.tau_lifetimes)[0]

    def cat(parts, dtype):
        """
        Concatenate column parts (empty if there are none).

        Parameters:
        parts (list of np.array): the parts.
        dtype (type): dtype.

        Returns:
        col (np.array): the column.
        """
        return np.concatenate(parts).astype(dtype) if parts else \
            np.zeros(0, dtype)

    events = {k: cat(v, np.int64) for k, v in ev_cols.items()}
    events['time_s'] = (ts['start'] + events['t0']) * ts['dt']
    hits = {k: cat(v, np.int64 if k in ('event', 'row', 'res_idx', 'idx')
                   else np.float64) for k, v in hit_cols.items()}
    return dict(mult = mult, avg = avg, p = peak, used = used, tau = tau,
                cosmic = dict(events = events, hits = hits))

################################################################################
##################################### Saving ###################################
################################################################################
def save_timestream(
    grp, ts, results, summary, thresholds, settings, mode, coinc_win,
    cosmic_min_mult
):
    """
    Save one timestream's photons.

    Layout of ``grp``: attrs describe the timestream and settings; per tone
    (in the order of ``ts['rows']``): ``rows``, ``res_idxs``, ``status``
    ('' if detected, else the error), ``hmin``, ``hmax`` (kHz/GHz),
    ``init_hmax``, ``sigma_y`` (kHz/GHz), ``amp_ph`` (the filter template's
    amplitude, in x: the mean photon), ``amp_ph_own`` (this
    timestream's mean photon amplitude, x), ``rate`` (pulses in
    the height window per second), ``n_used`` (pulses averaged), ``tau``,
    ``p`` (index of the maximum of ``avg_pulse``), ``template_p`` (of
    ``template``), ``template`` (mean pulse of the pulses in the height
    window, x),
    ``avg_pulse`` (mean photon pulse: height window and multiplicity 1, x),
    ``psd``, ``filter``; ``pulses/`` (every coincidence candidate: ``row``,
    ``idx`` (sample in the window), ``h`` (height, kHz/GHz),
    ``raw`` (raw peak, x), ``candidate`` (at or above hmin),
    ``in_window`` (in [hmin, hmax]), ``mult``, ``used`` (averaged)); and
    ``cosmic/events`` and ``cosmic/hits`` (``finish_timestream``).

    Parameters:
    grp (zarr.Group): the (new, empty) group.
    ts (dict): the timestream.
    results (dict): ``detect_timestream`` output.
    summary (dict): ``finish_timestream`` output.
    thresholds (Thresholds): the thresholds used.
    settings (DetectionSettings): the settings used.
    mode (str): how the timestream was detected, e.g. 'full'.
    coinc_win (int): coincidence window, samples.
    cosmic_min_mult (int): fewest tones of a cosmic ray.
    """
    rows = [int(r) for r in ts['rows']]
    ok = [isinstance(results.get(r), dict) for r in rows]
    ref = next((results[r] for r, g in zip(rows, ok) if g), None)
    M = 0 if ref is None else ref['pre'] + ref['post']
    nf = 0 if ref is None else int(ref['nfft'])
    grp.attrs.update(dl.to_jsonable(dict(
        dataset = ts['dataset'], run = ts['run'], name = ts['name'],
        tone_set = ts['tone_set'], source = ts['source']['group'],
        writes = list(ts['source']['writes']), start = ts['start'],
        stop = ts['stop'], dt = ts['dt'], mode = mode,
        thresholds = thresholds.to_dict(),
        settings = dataclasses.asdict(settings), coinc_win = coinc_win,
        cosmic_min_mult = cosmic_min_mult,
        pre = None if ref is None else ref['pre'],
        post = None if ref is None else ref['post'],
        filter_post = None if ref is None else ref['filter_post'])))

    def per_tone(key, default = np.nan):
        """
        Collect one scalar per tone.

        Parameters:
        key (str or callable): result key, or a function of (row, result).
        default (float): value for failed tones. Default NaN.

        Returns:
        values (np.array): the values.
        """
        return np.array([
            (key(r, results[r]) if callable(key) else results[r][key])
            if g else default for r, g in zip(rows, ok)], dtype = float)

    def per_tone_arr(get, length):
        """
        Collect one array per tone, NaN-padded to ``length``.

        Parameters:
        get (callable): function of (row, result) returning the array or
            None.
        length (int): row length.

        Returns:
        values (np.array): (ntones, length).
        """
        out = np.full((len(rows), length), np.nan)
        for i, (r, g) in enumerate(zip(rows, ok)):
            v = get(r, results[r]) if g else None
            if v is not None:
                v = np.asarray(v, dtype = float)[:length]
                out[i, :len(v)] = v
        return out

    dl.write_array(grp, 'rows', np.asarray(rows, dtype = np.int64))
    dl.write_array(grp, 'res_idxs', np.asarray(ts['res_idxs'],
                                               dtype = np.int64))
    dl.write_array(grp, 'status', np.array(
        ['' if g else str(results.get(r, 'not run'))
         for r, g in zip(rows, ok)]))
    for key in ('hmin', 'hmax', 'sigma_y', 'amp_ph', 'rate'):
        dl.write_array(grp, key, per_tone(key))
    dl.write_array(grp, 'amp_ph_own', per_tone(
        lambda r, res: res.get('amp_ph_own', res['amp_ph'])))
    dl.write_array(grp, 'init_hmax', np.array(
        [thresholds.for_tone(ri)['init_hmax'] or np.nan
         for ri in ts['res_idxs']], dtype = float))
    dl.write_array(grp, 'n_used', per_tone(
        lambda r, res: len(summary['used'].get(r, [])), 0))
    dl.write_array(grp, 'tau', per_tone(
        lambda r, res: summary['tau'].get(r, np.nan)))
    dl.write_array(grp, 'p', per_tone(
        lambda r, res: summary['p'].get(r, res['p'])))
    dl.write_array(grp, 'template_p', per_tone(lambda r, res: res['p']))
    dl.write_array(grp, 'template', per_tone_arr(
        lambda r, res: res['template'], M))
    dl.write_array(grp, 'avg_pulse', per_tone_arr(
        lambda r, res: summary['avg'].get(r), M))
    dl.write_array(grp, 'psd', per_tone_arr(
        lambda r, res: res['psd'], nf // 2 + 1))
    dl.write_array(grp, 'filter', per_tone_arr(
        lambda r, res: res['filter'], nf))
    cols = {k: [] for k in ('row', 'idx', 'h', 'raw', 'candidate',
                            'in_window', 'mult', 'used')}
    for r, g in zip(rows, ok):
        if not g:
            continue
        res = results[r]
        ci = np.asarray(res['coinc_idx'], dtype = np.int64)
        ch = np.asarray(res['coinc_h'], dtype = float)
        cand = np.isin(ci, res['cand_idx'])
        m = np.zeros(len(ci), dtype = np.int64)
        if r in summary['mult']:
            pos = np.searchsorted(ci, res['cand_idx'])
            m[pos] = summary['mult'][r]
        cols['row'].append(np.full(len(ci), r, dtype = np.int64))
        cols['idx'].append(ci)
        cols['h'].append(ch)
        cols['raw'].append(np.asarray(res['coinc_raw'], dtype = float))
        cols['candidate'].append(cand)
        cols['in_window'].append(cand & (ch >= res['hmin']) &
                                 (ch <= res['hmax']))
        cols['mult'].append(m)
        cols['used'].append(np.isin(ci, summary['used'].get(r, [])))
    pulses = dl.create_group(grp, 'pulses', 'analysis')
    for k, v in cols.items():
        dtype = bool if k in ('candidate', 'in_window', 'used') else \
            (float if k in ('h', 'raw') else np.int64)
        dl.write_array(pulses, k, np.concatenate(v).astype(dtype) if v
                       else np.zeros(0, dtype))
    cosmic = dl.create_group(grp, 'cosmic', 'analysis')
    for part in ('events', 'hits'):
        g = dl.create_group(cosmic, part, 'analysis')
        for k, v in summary['cosmic'][part].items():
            dl.write_array(g, k, v)

def load_reference(grp, settings = None):
    """
    Read a saved photon result (``save_timestream``) as a reference for
    another timestream: its thresholds, and per tone the optimal filter it
    found (its mean pulse in the height window and its noise PSD).

    Parameters:
    grp (zarr.Group): a ``ts_photons{p}/timestream{j}`` group.
    settings (DetectionSettings or None): for the pulse sign. None: the
        saved settings' sign.

    Returns:
    thresholds (Thresholds): its thresholds (with the per-tone ones).
    reference (dict): res_idx: dict for ``detect_with_filter``
        (``template``, ``p``, ``pre``, ``post``, ``filter_post``,
        ``sign``, ``amp_ph``, ``psd``, ``nfft``, ``hmin``, ``hmax``,
        ``dt``), for the tones it detected. The filter is normalized to
        that timestream's own mean photon, which becomes the height unit.
    """
    a = dict(grp.attrs)
    th = a['thresholds']
    thresholds = Thresholds(
        hmin = th['hmin'], hmax = th['hmax'], init_hmax = th['init_hmax'],
        overrides = {int(k): dict(v) for k, v in th['overrides'].items()})
    sign = int(a['settings']['pulse_sign']) if settings is None else \
        int(settings.pulse_sign)
    status = grp['status'][:]
    res_idxs = grp['res_idxs'][:]
    p = grp['template_p'][:] if 'template_p' in grp else grp['p'][:]
    template, psd = grp['template'][:], grp['psd'][:]
    nfft = int(grp['filter'].shape[1])
    amp = grp['amp_ph_own'][:] if 'amp_ph_own' in grp else grp['amp_ph'][:]
    hmin, hmax = grp['hmin'][:], grp['hmax'][:]
    reference = {}
    for i, r in enumerate(res_idxs):
        if str(status[i]) != '' or not np.isfinite(amp[i]):
            continue
        reference[int(r)] = dict(
            template = template[i], p = int(p[i]), pre = int(a['pre']),
            post = int(a['post']),
            filter_post = int(a.get('filter_post') or a['post']),
            sign = sign, amp_ph = float(amp[i]),
            psd = psd[i], nfft = nfft, hmin = float(hmin[i]),
            hmax = float(hmax[i]), dt = float(a['dt']))
    return thresholds, reference
