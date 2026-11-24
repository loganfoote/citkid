"""
Power sweeps: the NEP of every tone from a sweep of optical power
(``procedures/6d_nep.py``).

A sweep is a list of points: one timestream run per point, taken on a
calibration sweep at an incident power. 4c_ts_series.py names the
points (attribute ``point`` on the run and its sweep); 6d_nep.py maps
the names to powers (``select_points``). Points may share a calibration
sweep. The analysis:

1. ``calibrate_points``: each point's timestream calibration (the
   steps of 5_calibrate_streams.py, one DataSet per sweep) and IQ fit, in
   one stream calibration.
2. ``psd_points``: each timestream's spectra as 6c computes them, with
   peak removal and common-mode removal, saved as a PSD set; the glitch
   positions and the (decimated) common modes are kept for the review.
3. ``PowerSweep``: per tone, x(P) from the IQ fits' fr relative to the
   lowest power, the responsivity fit (R0, P0, x0), NEP(P) =
   sqrt(Sxx) / |R(P)| at ``freq_hz``, the photon-noise fit (eta), NEPinc
   (the NEP at the lowest power) and NEPabs = eta NEPinc.

The review window (``nep.review``) changes a tone's calibration at one
point; ``PowerSweep.recompute`` then takes the tone's new fr and Sxx (the
stored glitch positions cut again, the stored modes taken out with
weights fitted anew) without searching again; ``final_pass`` computes
the spectra of the changed points again in full.
"""
import numpy as np
from tqdm.auto import tqdm
from .. import daq_layout as dl
from ..pipeline import photon_finder as pf
from ..pipeline import psds
from ..pipeline.analysis import AnalysisRunner
from ..pipeline.stream_sources import (
    apply_stream_calibration, open_calibration, open_stream_dataset,
    plan_stream_datasets)
from ..responsivity.fitter import fit_responsivity_int_x0
from ..signal.basic_filt import decimate
from ..signal.psd import get_psd
from ..xcal.corr import calc_a
from ..xcal.reduced_params import get_sxx_reduced
from .fitter import fit_nep_photon
from .funcs import nep_from_sxx, nu_per_tone, responsivity_ok

NEP_GROUP = 'nep'
# Per-tone results saved (and loaded) by name
TONE_FIELDS = ('R0', 'P0', 'x0', 'R0_err', 'P0_err', 'x0_err', 'f1', 'eta',
               'eta_err', 'n_fit', 'p_min', 'nep_inc', 'nep_abs')

################################################################################
################################### Points #####################################
################################################################################
def list_points(root, tuning = None):
    """
    List the runs of a power sweep: those with a point name (attribute
    ``point``, written by 4c_ts_series.py) or a recorded power
    (``power_w``) that have streams.

    Parameters:
    root (zarr.Group): the store root.
    tuning (str or None): only this tuning's runs ('tune0'). Default
        None: every tuning.

    Returns:
    points (list of dict): per run, in run order: ``name`` (the run's),
        ``run`` (n), ``point`` (the point's name, or None), ``power_w``
        (recorded, or None), ``calibration`` (its sweep, e.g.
        'tune0/cal2'), ``tone_set``, ``tuning``.
    """
    out = []
    for rec in dl.stream_runs(root, tuning):
        if rec.get('point') is None and rec.get('power_w') is None:
            continue
        if not rec.get('paths'):
            continue
        out.append(dict(name = rec['name'], run = int(rec['run']),
                        point = rec.get('point'),
                        power_w = None if rec.get('power_w') is None
                        else float(rec['power_w']),
                        calibration = rec['calibration'],
                        tone_set = int(rec['tone_set']),
                        tuning = rec['tuning']))
    return out

def select_points(root, choice, tuning = None):
    """
    Choose the points of a sweep and give them their powers, in order of
    power.

    Parameters:
    root (zarr.Group): the store root.
    choice (dict, list or None): point name (or run name): incident
        power in W, as a dict; or a list of dicts with ``name`` (point or
        run name) or ``run`` (n) and ``power_w``. None or empty: every
        run with a recorded power.
    tuning (str or None): only this tuning's runs. Default None.

    Returns:
    points (list of dict): as ``list_points``, with ``power_w`` set,
        sorted by power.

    Raises:
    KeyError: if a name matches no run.
    ValueError: if a point name matches several runs (give the run
        names), a point has no power, or fewer than two points are left.
    """
    runs = dl.stream_runs(root, tuning)
    if not choice:
        points = [p for p in list_points(root, tuning)
                  if p['power_w'] is not None]
        if not points:
            raise ValueError('no run has a recorded power: give POINTS, '
                             'the point names with their powers')
    else:
        if isinstance(choice, dict):
            choice = [dict(name = k, power_w = v) for k, v in choice.items()]
        points = []
        for c in choice:
            c = dict(c)
            if 'run' in c:
                match = [r for r in runs if int(r['run']) == int(c['run'])]
            else:
                key = c.get('name')
                match = [r for r in runs if r['name'] == key]
                if not match:
                    match = [r for r in runs if r.get('point') == key
                             and r.get('paths')]
                if len(match) > 1:
                    raise ValueError(
                        f'point {key!r} has {len(match)} runs: '
                        f"{', '.join(r['name'] for r in match)}; give the "
                        'run names')
            if not match:
                raise KeyError(f"no run {c.get('name', c.get('run'))!r}")
            rec = match[-1]
            power = c.get('power_w', rec.get('power_w'))
            if power is None:
                raise ValueError(f"run {rec['name']!r} has no power: give "
                                 'power_w')
            points.append(dict(name = rec['name'], run = int(rec['run']),
                               point = rec.get('point'),
                               power_w = float(power),
                               calibration = rec['calibration'],
                               tone_set = int(rec['tone_set']),
                               tuning = rec['tuning']))
    points = sorted(points, key = lambda p: p['power_w'])
    if len(points) < 2:
        raise ValueError('a power sweep needs at least two points')
    return points

def describe_points(points):
    """
    Describe the points, one line each.

    Parameters:
    points (list of dict): ``select_points`` output.

    Returns:
    text (str): the lines.
    """
    lines = ['Points (power, point, run, calibration sweep, tone set):']
    for p in points:
        lines.append(f"  {p['power_w']:.4g} W: {p.get('point') or '-'}, "
                     f"ts{p['run']} {p['name']!r}, {p['calibration']}, set "
                     f"{p['tone_set']}")
    return '\n'.join(lines)

################################################################################
################################### Batch ######################################
################################################################################
def calibrate_points(root, cal, points, f_avg = 1, verbose = True):
    """
    Calibrate the points' timestreams (the steps of
    5_calibrate_streams.py, one DataSet per
    calibration sweep, in ``cal``) and fit each sweep's resonances
    (``fit_iq``), skipping what is saved already.

    Parameters:
    root (zarr.Group): the store root, writable.
    cal (zarr.Group): the stream calibration group (``ts_cal{c}``), with
        attribute ``selection``.
    points (list of dict): ``select_points`` output.
    f_avg (int): fine sweep points averaged in the fits. Default 1.
    verbose (bool): If True (default), show progress.

    Returns:
    datasets (dict): DataSet name: dict(``DS``, ``calibration`` (the
        sweep's path), ``tone_set``, ``resonators``, ``entries``), as
        ``stream_sources.open_calibration`` gives.
    """
    selection = [dict(run = p['run'], tone_sets = [p['tone_set']])
                 for p in points]
    groups = plan_stream_datasets(root, selection)
    datasets = {}
    for name, g in groups.items():
        res = g['resonators']
        grp = cal.require_group(name)
        grp.attrs.update(dl.to_jsonable(dict(
            entries = g['entries'], resonators = res, f_avg = f_avg)))
        DS = open_stream_dataset(grp, g['sources'], f'{res}/fres_all',
                                 f'{res}/qres_all', f_avg = f_avg)
        AR = AnalysisRunner(DS, analysis_yaml_path = 'ts')
        if verbose:
            print(f'{name}: {len(DS.res_idxs[:])} rows, {g["calibration"]}')
        AR.execute_remaining(verbose = verbose)
        step = next(s for s in AR.analysis_steps if s.name == 'fit_iq')
        rows = np.flatnonzero(~DS.saved_row_mask('iq_popt'))
        if len(rows):
            AR.execute_step(step, data_idx = rows,
                            user_params = {'iq_mask': None})
        grp.attrs['iq_fit'] = True
        datasets[name] = dict(DS = DS, entries = g['entries'],
                              resonators = res, tone_set = g['tone_set'],
                              calibration = g['calibration'])
    return datasets

def point_timestreams(root, points, datasets):
    """
    Find each point's timestream in its calibration.

    Parameters:
    root (zarr.Group): the store root.
    points (list of dict): ``select_points`` output.
    datasets (dict): ``calibrate_points`` output.

    Returns:
    timestreams (list of dict): per point, ``photon_finder.list_timestreams``
        entries, with ``cal`` (the calibration group's path).

    Raises:
    KeyError: if a point's timestream is not in the calibration.
    """
    cal_path = None
    for d in datasets.values():
        cal_path = str(d['DS'].root.path).rsplit('/', 1)[0]
        break
    found = pf.list_timestreams(root, datasets)
    out = []
    for p in points:
        match = [t for t in found if t['run'] == f"ts{p['run']}"
                 and t.get('tuning') == p['tuning']
                 and int(t['tone_set']) == p['tone_set']]
        if not match:
            raise KeyError(f"run ts{p['run']} ({p['name']!r}) tone set "
                           f"{p['tone_set']} is not calibrated")
        out.append(dict(match[0], cal = cal_path))
    return out

def psd_points(
    root, psd_grp, points, timestreams, datasets, settings, exclude = None,
    only = None, verbose = True
):
    """
    Compute and save the spectra of the points' timestreams (6c's
    ``analyze_timestream``, one group per point), keeping what the review
    needs: the glitch positions and the decimated common modes.

    Parameters:
    root (zarr.Group): the store root.
    psd_grp (zarr.Group): the PSD set (``ts_psd{p}``), with ``f_reduced``
        written by the caller.
    points (list of dict): ``select_points`` output.
    timestreams (list of dict): ``point_timestreams`` output.
    datasets (dict): ``calibrate_points`` output.
    settings (dict): ``analyze_timestream`` settings, with
        ``modes_rate_hz``.
    exclude (dict or None): res_idx: reason, tones left out. Default None.
    only (iterable of int or None): point indices to compute. Default
        None: all.
    verbose (bool): If True (default), print progress.

    Returns:
    results (dict): point index: ``analyze_timestream`` output.
    """
    f_reduced = np.asarray(psd_grp['f_reduced'][:], dtype = float)
    todo = range(len(points)) if only is None else sorted(set(only))
    out = {}
    for j in todo:
        ts = timestreams[j]
        DS = datasets[ts['dataset']]['DS']
        result = psds.analyze_timestream(
            root, ts, DS, f_reduced, settings,
            progress = print if verbose else None, exclude = exclude)
        psds.save_timestream(
            dl.create_group(psd_grp, f'timestream{j}', 'analysis',
                            overwrite = True),
            ts, result, settings)
        out[j] = result
    return out

def reduced_column(f_reduced, freq_hz):
    """
    Find the NEP frequency in the reduced grid.

    Parameters:
    f_reduced (np.array): the reduced frequencies, Hz.
    freq_hz (float): the NEP frequency, Hz.

    Returns:
    k (int): its column.

    Raises:
    ValueError: if no reduced frequency is within 20% of it.
    """
    f_reduced = np.asarray(f_reduced, dtype = float)
    k = int(np.argmin(np.abs(f_reduced - freq_hz)))
    if abs(f_reduced[k] - freq_hz) > 0.2 * freq_hz:
        raise ValueError(f'{freq_hz} Hz is not on the reduced grid '
                         f'{f_reduced.tolist()}: the spectra are the mean '
                         'within 20% of 0.1, 0.3, 1, 3, 10, ... Hz')
    return k

def _write(grp, name, data):
    """
    Write an array to a group, replacing one of the name.

    Parameters:
    grp (zarr.Group): the group.
    name (str): array name.
    data (array-like): the data.
    """
    if name in grp:
        del grp[name]
    dl.write_array(grp, name, data)

################################################################################
################################### Sweep ######################################
################################################################################
class PowerSweep:
    """
    The NEP analysis of a power sweep: per tone, the IQ fits' fr at each
    power, Sxx at the NEP frequency, the responsivity fit, NEP(P), the
    photon-noise fit and the overview quantities, with the data the
    review needs to recompute one tone at one point.

    Attributes:
    root (zarr.Group): the store root.
    grp (zarr.Group): the result group (``nep/<name>``).
    points (list of dict): the points, by power.
    timestreams (list of dict): each point's timestream.
    datasets (dict): the calibration's DataSets.
    res_idxs (np.array): the tones (the union over the points).
    power (np.array): the powers (W).
    rows (np.array, (n, m) int): each tone's DataSet row at each point,
        -1 if the tone is not calibrated there.
    fr, x, sxx, nep (np.array, (n, m)): per tone and point.
    nu (np.array): photon frequency per tone (Hz).
    freq_hz (float): the NEP frequency.
    fields (dict): per-tone results (``TONE_FIELDS``: np.array each).
    status (list of str): per tone, '' or why there is no result.
    bad (dict): res_idx: reason (tone marks and the window).
    limits (dict): the overview's review limits.
    changed (set): points whose spectra must be computed again
        (``final_pass``).
    settings (dict): the spectra settings.
    review (dict): point index: dict(``glitch_idxs`` (per DataSet row),
        ``write_idx`` (per row), ``writes`` (write index: dict
        ``modes_dec``, ``dt_dec``)).
    """

    def __init__(
        self, root, grp, points, timestreams, datasets, freq_hz, nu,
        settings
    ):
        """
        Set up the sweep (no values yet: ``fill``).

        Parameters:
        root, grp, points, timestreams, datasets, freq_hz, settings: see
            the attributes.
        nu (float or dict): photon frequency (``funcs.nu_per_tone``).
        """
        self.root, self.grp = root, grp
        self.points = list(points)
        self.timestreams = list(timestreams)
        self.datasets = datasets
        self.freq_hz = float(freq_hz)
        self.settings = dict(settings)
        self.power = np.array([p['power_w'] for p in self.points])
        res = set()
        for ts in self.timestreams:
            res |= set(int(r) for r in ts['res_idxs'] if int(r) >= 0)
        self.res_idxs = np.array(sorted(res), dtype = int)
        n, m = len(self.res_idxs), len(self.points)
        self.rows = np.full((n, m), -1, dtype = int)
        pos = {int(r): i for i, r in enumerate(self.res_idxs)}
        for j, ts in enumerate(self.timestreams):
            for row, r in zip(ts['rows'], ts['res_idxs']):
                if int(r) in pos:
                    self.rows[pos[int(r)], j] = int(row)
        self.nu = nu_per_tone(nu, self.res_idxs)
        self._nu_setting = nu
        self.fr = np.full((n, m), np.nan)
        self.x = np.full((n, m), np.nan)
        self.sxx = np.full((n, m), np.nan)
        self.nep = np.full((n, m), np.nan)
        self.fields = {k: np.full(n, np.nan) for k in TONE_FIELDS}
        self.fields['n_fit'] = np.zeros(n, dtype = int)
        self.fields['p_min'] = np.zeros(n)
        self.status = [''] * n
        self.bad = {}
        self.limits = {}
        self.changed = set()
        self.review = {}
        self.psd_path = None

    ############################################################################
    def tone_index(self, res_idx):
        """
        Return a tone's index.

        Parameters:
        res_idx (int): resonator index.

        Returns:
        i (int): index in ``res_idxs``.

        Raises:
        KeyError: if the tone is not in the sweep.
        """
        where = np.flatnonzero(self.res_idxs == int(res_idx))
        if not len(where):
            raise KeyError(f'res {res_idx} is not in the sweep')
        return int(where[0])

    def is_bad(self, res_idx):
        """
        Say whether a tone is marked bad.

        Parameters:
        res_idx (int): resonator index.

        Returns:
        bad (bool): True if so.
        """
        return int(res_idx) in self.bad

    def fill_fr(self, j = None):
        """
        Take the IQ fits' resonance frequencies from the DataSets.

        Parameters:
        j (int or None): one point. Default None: all.
        """
        for jj in (range(len(self.points)) if j is None else [j]):
            DS = self.datasets[self.timestreams[jj]['dataset']]['DS']
            saved = np.asarray(DS.saved_row_mask('iq_popt'), dtype = bool)
            self.fr[:, jj] = np.nan
            for i, row in enumerate(self.rows[:, jj]):
                if row < 0 or not saved[row]:
                    continue
                popt = np.asarray(DS.iq_popt[int(row)], dtype = float)
                if popt.ndim == 1 and len(popt) and np.isfinite(popt[0]):
                    self.fr[i, jj] = popt[0]

    def fill_sxx(self, j, result):
        """
        Take one point's Sxx at the NEP frequency from its spectra, and
        keep the review data.

        Parameters:
        j (int): the point.
        result (dict): ``analyze_timestream`` output.
        """
        ts = self.timestreams[j]
        f_reduced = np.asarray(self.root[self.psd_path]['f_reduced'][:],
                               dtype = float)
        k = reduced_column(f_reduced, self.freq_hz)
        key = 'sxx_reduced_cn' if self.settings.get('cn_remove') and \
            'sxx_reduced_cn' in result else 'sxx_reduced'
        by_row = {int(r): float(result[key][p, k])
                  for p, r in enumerate(ts['rows'])}
        self.sxx[:, j] = [by_row.get(int(row), np.nan) if row >= 0
                          else np.nan for row in self.rows[:, j]]
        glitches = {int(r): result['glitch_idxs'][p]
                    for p, r in enumerate(ts['rows'])}
        write_idx = {int(r): int(result['write_idx'][p])
                     for p, r in enumerate(ts['rows'])}
        writes = {int(w['index']): dict(modes_dec = w['modes_dec'],
                                        dt_dec = float(w['dt_dec']))
                  for w in result.get('writes', []) if 'modes_dec' in w}
        self.review[j] = dict(glitch_idxs = glitches, write_idx = write_idx,
                              writes = writes)

    ############################################################################
    def compute_tone(self, i):
        """
        Fit one tone: x(P) to the responsivity, NEP(P) from Sxx, and the
        photon-noise NEP for eta; then NEPinc and NEPabs.

        Parameters:
        i (int): tone index.
        """
        f = self.fields
        for k in TONE_FIELDS:
            f[k][i] = 0 if k == 'n_fit' else (f['p_min'][i] if k == 'p_min'
                                              else np.nan)
        self.x[i] = np.nan
        self.nep[i] = np.nan
        self.status[i] = ''
        fr = self.fr[i]
        ok = np.isfinite(fr)
        if ok.sum() < 4:
            self.status[i] = (f'{int(ok.sum())} points with an IQ fit '
                              '(4 needed)')
            return
        f1 = fr[np.flatnonzero(ok)[0]]
        f['f1'][i] = f1
        self.x[i] = (fr - f1) / f1
        try:
            _, popt, perr, _, _ = fit_responsivity_int_x0(
                self.power[ok], self.x[i][ok], f1)
        except Exception as exc:
            self.status[i] = f'responsivity fit failed: {exc}'
            return
        R0, P0, x0 = popt
        f['R0'][i], f['P0'][i], f['x0'][i] = R0, P0, x0
        f['R0_err'][i], f['P0_err'][i], f['x0_err'][i] = perr
        if not np.isfinite(popt).all():
            self.status[i] = 'responsivity fit failed'
            return
        if not responsivity_ok(R0, P0):
            self.status[i] = ('x rises with power (R0 >= 0): not a KID '
                              'response' if np.isfinite(R0) and R0 >= 0
                              else 'responsivity fit unusable (P0 <= 0)')
            return
        self.nep[i] = nep_from_sxx(self.sxx[i], self.power, R0, P0)
        good = np.isfinite(self.nep[i])
        if not good.any():
            self.status[i] = 'no Sxx at any point'
            return
        f['nep_inc'][i] = self.nep[i][np.flatnonzero(good)[0]]
        if not np.isfinite(self.nu[i]):
            self.status[i] = 'no photon frequency for this tone'
            return
        p_min = float(f['p_min'][i]) or None
        eta, eta_err, _, n_fit = fit_nep_photon(
            self.power, self.nep[i], self.nu[i], p_min = p_min)
        f['eta'][i], f['eta_err'][i], f['n_fit'][i] = eta, eta_err, n_fit
        if np.isfinite(eta):
            f['nep_abs'][i] = eta * f['nep_inc'][i]
        else:
            self.status[i] = 'photon-noise fit failed'

    def compute_all(self, progress = False):
        """
        Fit every tone.

        Parameters:
        progress (bool): If True, show a progress bar. Default False.
        """
        for i in tqdm(range(len(self.res_idxs)), disable = not progress,
                      desc = 'NEP fits', leave = False):
            self.compute_tone(i)

    def fill(self, results, psd_path):
        """
        Take every point's values (after the batch) and fit every tone.

        Parameters:
        results (dict): point index: ``analyze_timestream`` output.
        psd_path (str): the PSD set's path.
        """
        self.psd_path = psd_path
        self.fill_fr()
        for j, result in results.items():
            self.fill_sxx(j, result)
        self.compute_all()

    ############################################################################
    def recompute(self, i, j):
        """
        Take a tone's new fr and Sxx at one point after its calibration
        changed, without searching for glitches or common modes again
        (``reapply``), and fit the tone again; the point is noted for
        ``final_pass``.

        Parameters:
        i (int): tone index.
        j (int): point index.
        """
        self.fill_fr(j)
        self.sxx[i, j] = self.reapply(i, j)
        self.changed.add(int(j))
        self.compute_tone(i)

    def reapply(self, i, j):
        """
        Compute a tone's Sxx at one point from its current calibration:
        the stored glitch positions are cut again, the timestream is
        decimated, and the stored common modes are taken out with
        weights fitted anew.

        Parameters:
        i (int): tone index.
        j (int): point index.

        Returns:
        sxx (float): Sxx at the NEP frequency (1/Hz), NaN if the tone is
            not calibrated there.
        """
        row = int(self.rows[i, j])
        rev = self.review.get(j)
        if row < 0 or rev is None:
            return np.nan
        ts = self.timestreams[j]
        DS = self.datasets[ts['dataset']]['DS']
        dt = float(ts['dt'])
        try:
            x = apply_stream_calibration(DS, self.root, ts['source'],
                                         rows = [row])['xt'][0]
        except Exception:
            return np.nan
        x = np.asarray(x, dtype = float)
        if not np.isfinite(x).all():
            return np.nan
        s = self.settings
        if s.get('deglitch') == 'peaks':
            x, _ = psds.reapply_glitches(
                x, rev['glitch_idxs'].get(row, []),
                sign = int(s.get('glitch_sign', -1)),
                pre = max(1, int(round(s['glitch_pre_s'] / dt))),
                post = max(1, int(round(s['glitch_post_s'] / dt))))
        rate = s.get('modes_rate_hz')
        if rate:
            x, dt = decimate(x, dt, float(rate))
        w = rev['writes'].get(rev['write_idx'].get(row, -1))
        if w is not None and rate:
            A = np.asarray(w['modes_dec'], dtype = float)
            if A.shape[1] == len(x):
                a = calc_a(x[None, :], A)
                x = x - (a @ A)[0]
        f, p = get_psd(x - np.mean(x), dt, return_freqs = True)
        return float(get_sxx_reduced(f, p, self.freq_hz))

    def final_pass(self, exclude = None, verbose = True):
        """
        Compute the spectra of the changed points again in full (peak
        search and common modes), take their values and fit every tone.

        Parameters:
        exclude (dict or None): res_idx: reason, tones left out of the
            spectra. Default None.
        verbose (bool): If True (default), print progress.

        Returns:
        points (list of int): the points computed again.
        """
        todo = sorted(self.changed)
        if todo:
            results = psd_points(
                self.root, self.root[self.psd_path], self.points,
                self.timestreams, self.datasets, self.settings,
                exclude = exclude, only = todo, verbose = verbose)
            for j, result in results.items():
                self.fill_sxx(j, result)
        self.fill_fr()
        self.compute_all()
        self.changed = set()
        return todo

    ############################################################################
    def overview_values(self):
        """
        Return every tone's overview quantities.

        Returns:
        values (dict): 'R0' (1/W), 'P0' (W), 'x0', 'eta', 'nep_inc',
            'nep_abs' (W/Hz^0.5): np.array per tone, NaN without a result.
        """
        return {k: np.array(self.fields[k], dtype = float)
                for k in ('R0', 'P0', 'x0', 'eta', 'nep_inc', 'nep_abs')}

    def outside_limits(self, values = None):
        """
        List the tones outside the review limits, or without a value.

        Parameters:
        values (dict or None): ``overview_values``. Default None: computed.

        Returns:
        outside (list of int): tone indices.
        """
        values = self.overview_values() if values is None else values
        out = set()
        for key, v in values.items():
            lim = self.limits.get(key)
            out |= set(np.flatnonzero(~np.isfinite(v)).tolist())
            if lim is not None:
                out |= set(np.flatnonzero((v < lim[0]) | (v > lim[1]))
                           .tolist())
        return sorted(out)

    ############################################################################
    def save(self, notes = None):
        """
        Save the sweep to its group: the points, the per-point and
        per-tone values, the review data and the marks.

        Parameters:
        notes (str or None): notes saved with the group. Default None.
        """
        g = self.grp
        n, m = self.rows.shape
        attrs = dict(
            points = self.points, calibration = self.timestreams[0]['cal'],
            psd = self.psd_path, freq_hz = self.freq_hz,
            nu = self._nu_setting, settings = self.settings,
            limits = self.limits, changed = sorted(self.changed),
            timestreams = [dict(run = t['run'], name = t['name'],
                                tone_set = t['tone_set'],
                                dataset = t['dataset']) for t in
                           self.timestreams],
            n_tones = int(n), n_points = int(m))
        if notes:
            attrs['notes'] = str(notes).strip()
        g.attrs.update(dl.to_jsonable(attrs))
        _write(g, 'res_idxs', self.res_idxs)
        _write(g, 'nu', self.nu)
        _write(g, 'power_w', self.power)
        _write(g, 'rows', self.rows)
        for key in ('fr', 'x', 'sxx', 'nep'):
            _write(g, key, getattr(self, key))
        for key in TONE_FIELDS:
            _write(g, key, np.asarray(self.fields[key]))
        _write(g, 'status', np.asarray(self.status, dtype = str))
        _write(g, 'bad', np.array([int(r) in self.bad
                                           for r in self.res_idxs]))
        _write(g, 'bad_reason', np.array(
            [self.bad.get(int(r), '') for r in self.res_idxs], dtype = str))
        if 'review' in g:
            del g['review']
        rg = dl.create_group(g, 'review', 'analysis')
        for j, rev in self.review.items():
            pg = dl.create_group(rg, f'point{j}', 'analysis')
            rows = sorted(rev['glitch_idxs'])
            counts = [len(rev['glitch_idxs'][r]) for r in rows]
            _write(pg, 'glitch_rows', np.asarray(rows, dtype = int))
            _write(pg, 'glitch_offsets',
                           np.concatenate([[0], np.cumsum(counts)]).astype(int))
            _write(pg, 'glitch_idxs', np.concatenate(
                [np.asarray(rev['glitch_idxs'][r], dtype = np.int64)
                 for r in rows] or [np.zeros(0, dtype = np.int64)]))
            _write(pg, 'write_rows',
                           np.asarray(sorted(rev['write_idx']), dtype = int))
            _write(pg, 'write_idx', np.asarray(
                [rev['write_idx'][r] for r in sorted(rev['write_idx'])],
                dtype = int))
            for k, w in rev['writes'].items():
                wg = dl.create_group(pg, f'write{k}', 'analysis',
                                     attrs = dict(dt_dec = w['dt_dec']))
                _write(wg, 'modes_dec',
                               np.asarray(w['modes_dec'], dtype = np.float32))

    @classmethod
    def load(cls, root, grp):
        """
        Reopen a saved sweep (its DataSets through its calibration).

        Parameters:
        root (zarr.Group): the store root.
        grp (zarr.Group): the result group.

        Returns:
        sweep (PowerSweep): the sweep, with its saved values.
        """
        a = grp.attrs
        cal = root[a['calibration']]
        datasets = open_calibration(root, cal)
        points = [dict(p) for p in a['points']]
        timestreams = point_timestreams(root, points, datasets)
        sweep = cls(root, grp, points, timestreams, datasets, a['freq_hz'],
                    a['nu'], a['settings'])
        sweep.psd_path = a.get('psd')
        sweep.limits = dict(a.get('limits') or {})
        sweep.changed = set(int(j) for j in a.get('changed') or [])
        for key in ('fr', 'x', 'sxx', 'nep'):
            setattr(sweep, key, np.asarray(grp[key][:], dtype = float))
        for key in TONE_FIELDS:
            sweep.fields[key] = np.asarray(grp[key][:])
        sweep.status = [str(s) for s in grp['status'][:]]
        for r, b, reason in zip(grp['res_idxs'][:], grp['bad'][:],
                                grp['bad_reason'][:]):
            if b:
                sweep.bad[int(r)] = str(reason)
        if 'review' in grp:
            for name in grp['review']:
                if not name.startswith('point'):
                    continue
                pg = grp['review'][name]
                j = int(name[5:])
                rows = pg['glitch_rows'][:].tolist()
                off = pg['glitch_offsets'][:]
                idxs = pg['glitch_idxs'][:]
                glitches = {int(r): idxs[off[k]:off[k + 1]]
                            for k, r in enumerate(rows)}
                write_idx = dict(zip(pg['write_rows'][:].tolist(),
                                     pg['write_idx'][:].tolist()))
                writes = {}
                for wname in pg:
                    if wname.startswith('write') and wname[5:].isdigit():
                        wg = pg[wname]
                        writes[int(wname[5:])] = dict(
                            modes_dec = wg['modes_dec'][:],
                            dt_dec = float(wg.attrs['dt_dec']))
                sweep.review[j] = dict(glitch_idxs = glitches,
                                       write_idx = write_idx, writes = writes)
        return sweep

################################################################################
################################### Results ####################################
################################################################################
def results_group(root):
    """
    Return the store's NEP results group (``nep``), creating it.

    Parameters:
    root (zarr.Group): the store root, writable.

    Returns:
    grp (zarr.Group): the group.
    """
    if NEP_GROUP not in root:
        return dl.create_group(root, NEP_GROUP, 'analysis')
    return root[NEP_GROUP]

def list_results(root):
    """
    List the saved sweeps.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    names (list of str): the result names.
    """
    if NEP_GROUP not in root:
        return []
    return sorted(root[NEP_GROUP].group_keys())

def describe_results(root):
    """
    Describe the saved sweeps, one line each.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    text (str): the lines, or that there are none.
    """
    names = list_results(root)
    if not names:
        return 'No NEP results yet.'
    lines = ['NEP results (nep/<name>: points, tones, calibration):']
    for name in names:
        a = root[NEP_GROUP][name].attrs
        lines.append(f"  {name}: {a.get('n_points')} points, "
                     f"{a.get('n_tones')} tones, {a.get('calibration')}")
    return '\n'.join(lines)
