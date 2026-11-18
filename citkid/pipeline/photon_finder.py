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
   the high-energy events (HEEs, ``hee_events``), and per tone the
   mean photon pulse
   of the pulses in the height window with multiplicity 1
   (``average_pulse``) and its decay time (``fit_tau``).
4. ``save_timestream`` saves the results in one group,
   ``photons/<name>`` at the store root (``result_name``: the run's
   name, with the tone set when the run has several), replaced when the
   timestream is analyzed again; ``load_reference`` reads a later run's
   thresholds or filters from it, and ``load_results`` its results, to
   continue in the window. ``migrate_results`` moves results saved by
   earlier versions (``ts_photons{p}/timestream{j}``) into this layout.
"""
import dataclasses
import os
import warnings
import numpy as np
from tqdm.auto import tqdm
from .. import daq_layout as dl
from ..signal import photons as ph
from .stream_sources import apply_stream_calibration

# What a run takes from a previous result (``load_reference``)
REUSE_OPTIONS = ('settings', 'filter')
# The results' group at the store root, one child per timestream
PHOTONS_GROUP = 'photons'

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
        (samples), ``dt``, ``rows`` (the DataSet's calibrated rows,
        np.array: tones whose x calibration failed are left out,
        ``calibrated_rows``), ``res_idxs`` (their resonator indices),
        ``n_uncalibrated`` (tones left out).
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
            rows = calibrated_rows(DS)
            stream = root[src['group']][src['writes'][0]][
                src.get('stream', dl.STREAM_NAME)]
            out.append(dict(
                index = len(out), run = run['group'], run_index = run['run'],
                name = run.get('name'), tone_set = s, dataset = dname,
                source = src, start = int(src['start']),
                stop = int(src['stop']),
                dt = float(np.asarray(stream['dt'][...])),
                rows = rows, res_idxs = res_idxs[rows],
                n_uncalibrated = len(res_idxs) - len(rows)))
    return out

def calibrated_rows(DS):
    """
    Return the rows of a stream DataSet whose x calibration succeeded
    (``poly_x`` saved and finite), so that uncalibrated tones are left
    out of the analyses rather than listed as failed. Rows whose
    calibration failed without saving anything count as uncalibrated.

    Parameters:
    DS (DataSet): the DataSet, calibrated.

    Returns:
    rows (np.array, int): the calibrated rows.
    """
    n = len(np.asarray(DS.res_idxs[:]))
    try:
        saved = np.asarray(DS.saved_row_mask('poly_x'), dtype = bool)
    except Exception:
        saved = np.ones(n, dtype = bool)
    rows = np.flatnonzero(saved)
    if not len(rows):
        return rows
    try:
        poly = np.asarray(DS.poly_x[rows.tolist()], dtype = float)
    except Exception as exc:
        warnings.warn(f'poly_x cannot be read ({exc}): no tone is '
                      'calibrated')
        return np.zeros(0, dtype = int)
    ok = np.zeros(n, dtype = bool)
    ok[rows] = np.isfinite(poly.reshape(len(rows), -1)).all(axis = 1)
    return np.flatnonzero(ok)

def result_name(ts, timestreams):
    """
    Return the name of a timestream's result: its run's name, with the
    tone set (``<name>_set<s>``) when the listing has several timestreams
    of that run.

    Parameters:
    ts (dict): the timestream.
    timestreams (list of dict): ``list_timestreams`` output.

    Returns:
    name (str): the name.
    """
    same_run = [t for t in timestreams if t['run'] == ts['run']]
    name = str(ts['name'] or ts['run'])
    return name if len(same_run) <= 1 else f"{name}_set{ts['tone_set']}"

def results_group(root):
    """
    Return the store's photon results group (``photons/``), making it if
    needed.

    Parameters:
    root (zarr.Group): the store root, writable.

    Returns:
    grp (zarr.Group): the group.
    """
    return dl.require_group(root, PHOTONS_GROUP, 'analysis')

def list_results(root):
    """
    List the saved photon results.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    names (list of str): result names, sorted.
    """
    if PHOTONS_GROUP not in root:
        return []
    return sorted(root[PHOTONS_GROUP].group_keys())

def describe_results(root):
    """
    Describe the saved photon results, one line each.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    text (str): the description.
    """
    lines = ['Previous results (photons/<name>: run, tone set, '
             'calibration, mode, created):']
    for name in list_results(root):
        a = root[PHOTONS_GROUP][name].attrs
        n_bad = len(a.get('thresholds', {}).get('bad', []))
        lines.append(f"  {name}: {a.get('run')} set {a.get('tone_set')}, "
                     f"{a.get('calibration', a.get('source'))}, "
                     f"{a.get('mode')}, "
                     f"{a.get('created')}"
                     + (f', {n_bad} bad' if n_bad else ''))
    old = dl.list_indices(root, 'ts_photons')
    if old:
        lines.append('  earlier layout, not listed: '
                     + ', '.join(f'ts_photons{p}' for p in old)
                     + ' (photon_finder.migrate_results moves them)')
    if len(lines) == 1:
        lines.append('  none')
    return '\n'.join(lines)

def apply_tone_marks(root, ts, thresholds):
    """
    Mark the tones the store's tone-marks registry says are bad
    (``daq_layout.bad_tones``) bad in a result's thresholds, with the
    registry's reason, unless they are marked already.

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    thresholds (Thresholds or TailThresholds): the thresholds (changed in
        place).

    Returns:
    marked (list of int): the resonators marked by this call.
    """
    bad = dl.bad_tones(root)
    marked = []
    for r in ts['res_idxs'].tolist():
        if int(r) in bad and not thresholds.is_bad(r):
            thresholds.set_bad(r, dl.bad_reason(bad[int(r)]))
            marked.append(int(r))
    return marked

def describe_result(grp):
    """
    Describe one saved photon result in a few lines: when and how it was
    made, and how much of it was set by hand.

    Parameters:
    grp (zarr.Group): the result (``photons/<name>``).

    Returns:
    text (str): the description.
    """
    a = grp.attrs
    th = Thresholds.from_dict(a.get('thresholds', {}))
    n = grp['res_idxs'].shape[0] if 'res_idxs' in grp else 0
    own = sum(1 for r in th.overrides if th.has_own(r))
    fits = sum(1 for r, o in th.overrides.items()
               if 'tau_skip' in o or 'tau_fit_s' in o)
    return (f"{a.get('run')} tone set {a.get('tone_set')}, {n} tones\n"
            f"created {a.get('created')}, {a.get('mode')}\n"
            f"{own} tones with their own thresholds, {fits} with their "
            f"own decay fit, {len(th.bad)} marked bad")

def migrate_results(root, which = None, verbose = True):
    """
    Move photon results saved by earlier versions (``ts_photons{p}``,
    one ``timestream{j}`` per timestream) into the current layout
    (``photons/<name>``), by renaming their directories on disk (a local
    store) and merging the run-level attributes (``source``,
    ``reference``, ``reuse``, ``continued_from``) into each result's.
    A result whose name is taken is left where it is.

    Parameters:
    root (zarr.Group): the store root, writable, on a local store.
    which (list of int or None): the ``ts_photons{p}`` indices to move.
        Default None: all.
    verbose (bool): If True (default), print what moved.

    Returns:
    moved (dict): new name: the old path, for the results moved.
    """
    base = str(root.store.root)
    results_group(root)
    moved = {}
    for p in dl.list_indices(root, 'ts_photons'):
        if which is not None and p not in [int(q) for q in which]:
            continue
        old = root[dl.set_name(p, 'ts_photons')]
        run_attrs = {k: old.attrs.get(k) for k in
                     ('source', 'reference', 'reuse', 'continued_from')}
        names = sorted((t for t in old.group_keys()
                        if t.startswith('timestream')),
                       key = lambda t: int(t[len('timestream'):]))
        runs = [old[t].attrs.get('name') for t in names]
        left = []
        for t, run in zip(names, runs):
            g = old[t]
            name = str(run or g.attrs.get('run'))
            if runs.count(run) > 1:
                name = f"{name}_set{g.attrs.get('tone_set')}"
            if name in root[PHOTONS_GROUP]:
                left.append(t)
                if verbose:
                    print(f'{old.path}/{t} not moved: photons/{name} exists')
                continue
            src = os.path.join(base, old.path, t)
            dst = os.path.join(base, PHOTONS_GROUP, name)
            os.rename(src, dst)
            new = root[PHOTONS_GROUP][name]
            new.attrs.update(dl.to_jsonable(dict(
                run_attrs, migrated_from = f'{old.path}/{t}')))
            moved[name] = f'{old.path}/{t}'
            if verbose:
                print(f'{old.path}/{t} -> photons/{name}')
        if not left:
            del root[old.basename]
    return moved

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
        left_out = t.get('n_uncalibrated', 0)
        lines.append(
            f"  {t['index']}: {t['run']}, {t['name']!r}, set "
            f"{t['tone_set']}, {n * t['dt']:.3g} s, {len(t['rows'])} "
            f"tones" + (f' ({left_out} uncalibrated left out)' if left_out
                        else '') + f", {t['dataset']}")
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

    def __init__(self, root, ts, DS, settings, max_bytes = 12e9):
        """
        Set up the timestream's data.

        Parameters:
        root (zarr.Group): the store root.
        ts (dict): the timestream (``list_timestreams``).
        DS (DataSet): its calibration DataSet.
        settings (DetectionSettings): detection settings.
        max_bytes (float): memory limit, bytes. Default 12e9.
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
                try:
                    x = apply_stream_calibration(self._DS, self._root,
                                                 self._ts['source'],
                                                 rows = need)['xt']
                except Exception as exc:
                    # A tone without a usable calibration is skipped, not
                    # fatal (its result says why)
                    for r in need:
                        fresh[r] = f'the calibration cannot be applied: {exc}'
                        self._keep(r, fresh[r])
                    x = []
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
    tau_skip (int): the decay-time fit of a mean pulse starts this many
        samples after its maximum. Default 2.
    tau_fit_s (float or None): the fit covers this long, s. Default 1 ms.
        None: the whole template after the maximum.
    overrides (dict): res_idx: dict with any of 'hmin', 'hmax',
        'init_hmax', 'tau_skip', 'tau_fit_s'.
    bad (dict): res_idx: reason, for the tones marked bad (in the
        threshold window): left out of the coincidences, HEEs and
        mean pulses, and not used as a reference.
    limits (dict): review limits of the tone overview window: 'height',
        'tau', 'rate': [low, high] (kHz/GHz, s, 1/s), or None when unset.
        They mark tones for review and change nothing else.
    reviewed (set of int): resonators the user reviewed in the window
        (``mark_reviewed``), shown with ``*`` in its list.
    """
    hmin: float = 0.5
    hmax: float = 1.6
    init_hmax: float = 3.
    tau_skip: int = 2
    tau_fit_s: float = 1e-3
    overrides: dict = dataclasses.field(default_factory = dict)
    bad: dict = dataclasses.field(default_factory = dict)
    limits: dict = dataclasses.field(default_factory = dict)
    reviewed: set = dataclasses.field(default_factory = set)

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

    def set_fit(self, res_idx, tau_skip, tau_fit_s):
        """
        Give one resonator its own decay-fit values.

        Parameters:
        res_idx (int): resonator index.
        tau_skip (int): samples skipped after the maximum.
        tau_fit_s (float or None): fit length, s.
        """
        o = self.overrides.setdefault(int(res_idx), {})
        o['tau_skip'] = int(tau_skip)
        o['tau_fit_s'] = None if tau_fit_s is None else float(tau_fit_s)

    def set_fit_all(self, tau_skip, tau_fit_s):
        """
        Set the decay-fit values of every resonator, dropping their own.

        Parameters:
        tau_skip (int): samples skipped after the maximum.
        tau_fit_s (float or None): fit length, s.
        """
        self.tau_skip = int(tau_skip)
        self.tau_fit_s = None if tau_fit_s is None else float(tau_fit_s)
        for r in list(self.overrides):
            for key in ('tau_skip', 'tau_fit_s'):
                self.overrides[r].pop(key, None)
            if not self.overrides[r]:
                del self.overrides[r]

    def has_own(self, res_idx, keys = ('hmin', 'hmax', 'init_hmax')):
        """
        Say whether a resonator has its own value of any of some keys.

        Parameters:
        res_idx (int): resonator index.
        keys (sequence of str): override keys. Default: the thresholds.

        Returns:
        own (bool): True if it has one.
        """
        return bool(set(keys) & set(self.overrides.get(int(res_idx), {})))

    def for_tone(self, res_idx):
        """
        Return the thresholds of one resonator.

        Parameters:
        res_idx (int): resonator index.

        Returns:
        values (dict): 'hmin', 'hmax', 'init_hmax', 'tau_skip',
            'tau_fit_s'.
        """
        out = dict(hmin = self.hmin, hmax = self.hmax,
                   init_hmax = self.init_hmax, tau_skip = self.tau_skip,
                   tau_fit_s = self.tau_fit_s)
        out.update(self.overrides.get(int(res_idx), {}))
        return out

    def to_dict(self):
        """
        Return the thresholds as JSON-ready values.

        Returns:
        values (dict): 'hmin', 'hmax', 'init_hmax', 'tau_skip',
            'tau_fit_s', 'overrides' (keys as strings), 'bad' (keys as
            strings), 'limits', 'reviewed' (sorted list).
        """
        return dict(hmin = self.hmin, hmax = self.hmax,
                    init_hmax = self.init_hmax, tau_skip = self.tau_skip,
                    tau_fit_s = self.tau_fit_s,
                    overrides = {str(k): dict(v)
                                 for k, v in self.overrides.items()},
                    bad = {str(k): str(v) for k, v in self.bad.items()},
                    limits = dict(self.limits),
                    reviewed = sorted(int(r) for r in self.reviewed))

    @classmethod
    def from_dict(cls, values):
        """
        Rebuild thresholds from ``to_dict`` output (also the output of
        earlier versions, without fit values, reasons or limits).

        Parameters:
        values (dict): the saved values.

        Returns:
        thresholds (Thresholds): the thresholds.
        """
        bad = values.get('bad', [])
        if isinstance(bad, list):
            bad = {int(r): '' for r in bad}
        return cls(
            hmin = values['hmin'], hmax = values['hmax'],
            init_hmax = values['init_hmax'],
            tau_skip = int(values.get('tau_skip', 2)),
            tau_fit_s = values.get('tau_fit_s', 1e-3),
            overrides = {int(k): dict(v)
                         for k, v in values.get('overrides', {}).items()},
            bad = {int(k): str(v) for k, v in bad.items()},
            limits = dict(values.get('limits', {})),
            reviewed = {int(r) for r in values.get('reviewed', [])})

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
    root, ts, data, results, settings, coinc_win = 2, hee_min_mult = 3,
    progress = True, thresholds = None
):
    """
    Find coincidences and HEEs per tone write, and average each
    tone's photon pulses (height window, multiplicity 1). Tones marked
    bad are left out, like tones whose detection failed.

    Parameters:
    root (zarr.Group): the store root.
    ts (dict): the timestream.
    data (TimestreamData): its detection timestreams.
    results (dict): ``detect_timestream`` output for all its tones.
    settings (DetectionSettings): for ``baseline_gap`` (and the decay fit
        without ``thresholds``).
    coinc_win (int): coincidence window, samples: a pulse at t is
        coincident with other tones' pulses in [t - win, t + win].
        Default 2.
    hee_min_mult (int): fewest tones of an HEE. Default 3.
    progress (bool): If True (default), show a progress bar.
    thresholds (Thresholds or None): for the bad marks and each tone's
        decay-fit values. Default None: no bad tones, the settings' fit
        values.

    Returns:
    summary (dict): ``mult`` (row: multiplicity of each pulse candidate),
        ``avg`` (row: mean photon pulse), ``p`` (row: index of its
        maximum), ``used`` (row: the pulses averaged), ``tau`` (row: decay
        time, s), ``hee`` (dict of arrays: ``events`` with ``t0``,
        ``time_s``, ``mult``, ``write_idx``; ``hits`` with ``event``,
        ``row``, ``res_idx``, ``idx``, ``h``, ``x``, ``raw``).
    """
    mult, avg, peak, used, tau = {}, {}, {}, {}, {}
    ev_cols = {k: [] for k in ('t0', 'mult', 'write_idx')}
    hit_cols = {k: [] for k in ('event', 'row', 'res_idx', 'idx', 'h', 'x',
                                'raw')}
    n_events = 0
    bad = set() if thresholds is None else {int(r) for r in thresholds.bad}
    res_of_row = dict(zip(ts['rows'].tolist(), ts['res_idxs'].tolist()))
    for w, _, rows in tone_writes(root, ts):
        good = [int(r) for r in rows if isinstance(results.get(int(r)),
                                                   dict)
                and res_of_row[int(r)] not in bad]
        res = [results[r] for r in good]
        if not res:
            continue
        for r, m in zip(good, ph.candidate_multiplicity(res, coinc_win)):
            mult[r] = m
        events, hits = ph.hee_events(res, coinc_win, hee_min_mult)
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
        if thresholds is None:
            skip, fit_s = settings.tau_skip, settings.tau_fit_s
        else:
            fit = thresholds.for_tone(res_of_row[r])
            skip, fit_s = fit['tau_skip'], fit['tau_fit_s']
        tau[r] = ph.fit_tau(a, p, data.dt, skip, fit_s)[0]

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
                hee = dict(events = events, hits = hits))

################################################################################
##################################### Saving ###################################
################################################################################
def save_timestream(
    grp, ts, results, summary, thresholds, settings, mode, coinc_win,
    hee_min_mult
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
    ``psd``, ``filter``, ``bad`` and ``bad_reason`` (marked bad in the
    window), ``tau_skip`` and ``tau_fit_s`` (the decay fit's values);
    ``pulses/``
    (every coincidence candidate: ``row``, ``idx`` (sample in the window),
    ``h`` (height, kHz/GHz), ``raw`` (raw peak, x), ``candidate`` (at or
    above hmin), ``in_window`` (in [hmin, hmax]), ``mult``, ``used``
    (averaged)); ``hee/events`` and ``hee/hits``
    (``finish_timestream``); and ``detail/``, the rest of each tone's
    detection result, so that ``load_results`` can rebuild it for the
    threshold window: tables ``events`` (``row``, ``idx``), ``init``
    (``row``, ``idx``, ``h``: first-pass peaks), ``neg`` (``row``,
    ``h``: inverted candidates), ``template_idx`` (``row``, ``idx``),
    per tone ``n_vetoed``, ``init_hmax_abs``, ``clean_frac``,
    ``filter_p``, ``n_history``, ``history`` ((ntones, iterations, M),
    NaN-padded) and ``log`` ((ntones, lines), ''-padded).

    Parameters:
    grp (zarr.Group): the (new, empty) group.
    ts (dict): the timestream.
    results (dict): ``detect_timestream`` output.
    summary (dict): ``finish_timestream`` output.
    thresholds (Thresholds): the thresholds used.
    settings (DetectionSettings): the settings used.
    mode (str): how the timestream was detected, e.g. 'full'.
    coinc_win (int): coincidence window, samples.
    hee_min_mult (int): fewest tones of an HEE.
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
        hee_min_mult = hee_min_mult,
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
    dl.write_array(grp, 'bad', np.array(
        [thresholds.is_bad(ri) for ri in ts['res_idxs']], dtype = bool))
    dl.write_array(grp, 'bad_reason', np.array(
        [thresholds.bad.get(int(ri), '') for ri in ts['res_idxs']]))
    dl.write_array(grp, 'tau_skip', np.array(
        [thresholds.for_tone(ri)['tau_skip'] for ri in ts['res_idxs']],
        dtype = np.int64))
    dl.write_array(grp, 'tau_fit_s', np.array(
        [thresholds.for_tone(ri)['tau_fit_s'] or np.nan
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
    hee = dl.create_group(grp, 'hee', 'analysis')
    for part in ('events', 'hits'):
        g = dl.create_group(hee, part, 'analysis')
        for k, v in summary['hee'][part].items():
            dl.write_array(g, k, v)
    _save_detail(dl.create_group(grp, 'detail', 'analysis'), rows, ok,
                 results, M)

def _save_detail(grp, rows, ok, results, M):
    """
    Save the parts of each tone's detection result that the other arrays
    don't hold (``save_timestream``'s ``detail/``).

    Parameters:
    grp (zarr.Group): the (new, empty) group.
    rows (list of int): DataSet rows.
    ok (list of bool): whether each row's detection succeeded.
    results (dict): ``detect_timestream`` output.
    M (int): template length.
    """
    def table(name, columns):
        """
        Save one concatenated per-tone table with a ``row`` column.

        Parameters:
        name (str): group name.
        columns (dict): column: (result key or callable, dtype).
        """
        g = dl.create_group(grp, name, 'analysis')
        parts = {k: [] for k in ('row', *columns)}
        for r, good in zip(rows, ok):
            if not good:
                continue
            res = results[r]
            vals = {k: np.asarray(key(res) if callable(key) else res[key])
                    for k, (key, _) in columns.items()}
            n = len(next(iter(vals.values())))
            parts['row'].append(np.full(n, r, dtype = np.int64))
            for k, v in vals.items():
                parts[k].append(v)
        for k, v in parts.items():
            dtype = np.int64 if k == 'row' else columns[k][1]
            dl.write_array(g, k, np.concatenate(v).astype(dtype) if v
                           else np.zeros(0, dtype))

    table('events', dict(idx = ('events', np.int64)))
    table('init', dict(idx = ('init_idx', np.int64),
                       h = ('init_h', np.float32)))
    table('neg', dict(h = ('neg_h', np.float32)))
    table('template_idx', dict(idx = ('template_idx', np.int64)))
    n_hist = [len(results[r]['history']) if good else 0
              for r, good in zip(rows, ok)]
    n_log = [len(results[r]['log']) if good else 0
             for r, good in zip(rows, ok)]
    history = np.full((len(rows), max(n_hist, default = 0), M), np.nan)
    log = np.full((len(rows), max(n_log, default = 0)), '', dtype = object)
    for i, (r, good) in enumerate(zip(rows, ok)):
        if not good:
            continue
        h = np.asarray(results[r]['history'], dtype = float)
        if h.size:
            history[i, :len(h), :h.shape[1]] = h
        log[i, :n_log[i]] = [str(line) for line in results[r]['log']]
    for key, default in (('n_vetoed', 0), ('init_hmax_abs', np.nan),
                         ('clean_frac', np.nan), ('filter_p', -1)):
        dl.write_array(grp, key, np.array(
            [results[r].get(key, default) if good else default
             for r, good in zip(rows, ok)], dtype = float))
    dl.write_array(grp, 'n_history', np.asarray(n_hist, dtype = np.int64))
    dl.write_array(grp, 'history', history)
    dl.write_array(grp, 'log', log.astype(str))

def load_results(grp, settings = None):
    """
    Read a saved photon result (``save_timestream``) back as detection
    results and thresholds, to continue choosing thresholds in the window
    (06a_find_photons.py's ``CONTINUE``) without running the detection
    again.

    Parameters:
    grp (zarr.Group): a ``ts_photons{p}/timestream{j}`` group.
    settings (DetectionSettings or None): for the pulse sign. None: the
        saved settings' sign.

    Returns:
    results (dict): row: the detection result (the keys of
        ``run_detection``, plus ``res_idx``), or the saved error message
        for tones whose detection failed.
    thresholds (Thresholds): the saved thresholds, overrides and bad
        marks.

    Raises:
    KeyError: if the result has no ``detail/`` group (saved by an earlier
        version: run the detection again instead).
    """
    if 'detail' not in grp:
        raise KeyError(f'{grp.path} holds no detection detail (saved by an '
                       'earlier version): it cannot be continued')
    a = dict(grp.attrs)
    thresholds, _ = load_reference(grp, settings)
    sign = int(a['settings']['pulse_sign']) if settings is None else \
        int(settings.pulse_sign)
    rows = grp['rows'][:]
    status = grp['status'][:]
    res_idxs = grp['res_idxs'][:]
    d = grp['detail']
    pulses = grp['pulses']
    p_row, p_idx, p_h = pulses['row'][:], pulses['idx'][:], pulses['h'][:]
    p_raw, p_cand = pulses['raw'][:], pulses['candidate'][:]
    tables = {name: {k: d[name][k][:] for k in d[name].array_keys()}
              for name in ('events', 'init', 'neg', 'template_idx')}
    per_tone = {k: grp[k][:] for k in ('hmin', 'hmax', 'sigma_y', 'amp_ph',
                                       'amp_ph_own', 'rate', 'template_p',
                                       'template', 'psd', 'filter')}
    detail = {k: d[k][:] for k in ('n_vetoed', 'init_hmax_abs',
                                   'clean_frac', 'filter_p', 'n_history',
                                   'history', 'log')}
    results = {}
    for i, (row, r_idx) in enumerate(zip(rows, res_idxs)):
        row = int(row)
        if str(status[i]) != '':
            results[row] = str(status[i])
            continue
        sel = p_row == row
        t = {name: {k: v[tab['row'] == row] for k, v in tab.items()
                    if k != 'row'} for name, tab in tables.items()}
        n_hist = int(detail['n_history'][i])
        res = dict(
            cand_idx = p_idx[sel & p_cand], cand_h = p_h[sel & p_cand],
            events = t['events']['idx'], template = per_tone['template'][i],
            template_idx = t['template_idx']['idx'],
            p = int(per_tone['template_p'][i]), pre = int(a['pre']),
            post = int(a['post']),
            filter_post = int(a.get('filter_post') or a['post']),
            amp_ph = float(per_tone['amp_ph'][i]),
            amp_ph_own = float(per_tone['amp_ph_own'][i]), sign = sign,
            hmin = float(per_tone['hmin'][i]),
            hmax = float(per_tone['hmax'][i]),
            sigma_y = float(per_tone['sigma_y'][i]),
            rate = float(per_tone['rate'][i]), psd = per_tone['psd'][i],
            nfft = int(per_tone['filter'].shape[1]),
            filter = per_tone['filter'][i],
            history = detail['history'][i, :n_hist],
            neg_h = t['neg']['h'], clean_frac = float(detail['clean_frac'][i]),
            coinc_idx = p_idx[sel], coinc_h = p_h[sel], coinc_raw = p_raw[sel],
            init_idx = t['init']['idx'], init_h = t['init']['h'],
            init_hmax_abs = float(detail['init_hmax_abs'][i]),
            n_vetoed = int(detail['n_vetoed'][i]), dt = float(a['dt']),
            log = [str(line) for line in detail['log'][i] if line != ''],
            res_idx = int(r_idx))
        if detail['filter_p'][i] >= 0:
            res['filter_p'] = int(detail['filter_p'][i])
        results[row] = res
    return results, thresholds

def refit_tau(grp, tau_skip, tau_fit_s):
    """
    Fit the decay times of a saved result's mean photon pulses again
    (``signal.photons.fit_tau``) with other fit values for every tone,
    replacing its ``tau``, ``tau_skip`` and ``tau_fit_s`` arrays and
    recording the values in its settings and thresholds.

    Parameters:
    grp (zarr.Group): a ``ts_photons{p}/timestream{j}`` group, writable.
    tau_skip (int): samples after the maximum to skip.
    tau_fit_s (float or None): length of the fit, s (None: the whole
        template after the maximum).

    Returns:
    tau (np.array): the new decay times, s (NaN where there is no mean
        pulse or the fit fails).
    """
    avg, p, dt = grp['avg_pulse'][:], grp['p'][:], float(grp.attrs['dt'])
    tau = np.full(len(p), np.nan)
    for i in range(len(p)):
        if np.isfinite(p[i]) and np.isfinite(avg[i]).any():
            pulse = np.where(np.isfinite(avg[i]), avg[i], 0.)
            tau[i] = ph.fit_tau(pulse, int(p[i]), dt, tau_skip,
                                tau_fit_s)[0]
    for name, value in (('tau', tau),
                        ('tau_skip', np.full(len(p), int(tau_skip))),
                        ('tau_fit_s', np.full(len(p), np.nan if tau_fit_s
                                              is None else tau_fit_s))):
        if name in grp:
            del grp[name]
        dl.write_array(grp, name, value)
    settings = dict(grp.attrs.get('settings', {}))
    settings.update(tau_skip = int(tau_skip), tau_fit_s = tau_fit_s)
    th = Thresholds.from_dict(grp.attrs['thresholds'])
    th.set_fit_all(tau_skip, tau_fit_s)
    grp.attrs.update(dl.to_jsonable(dict(
        settings = settings, thresholds = th.to_dict(),
        tau_refit = dl._now())))
    return tau

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
    thresholds = Thresholds.from_dict(a['thresholds'])
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
        if str(status[i]) != '' or not np.isfinite(amp[i]) or \
                thresholds.is_bad(r):
            continue
        reference[int(r)] = dict(
            template = template[i], p = int(p[i]), pre = int(a['pre']),
            post = int(a['post']),
            filter_post = int(a.get('filter_post') or a['post']),
            sign = sign, amp_ph = float(amp[i]),
            psd = psd[i], nfft = nfft, hmin = float(hmin[i]),
            hmax = float(hmax[i]), dt = float(a['dt']))
    return thresholds, reference
