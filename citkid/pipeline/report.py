"""
Report of a store's timestream results as plain files: the lifetimes
(photon and high-energy event (HEE) results: pulse height and lifetime
againstresonant frequency) and the noise spectra (PSD groups: the spectra's
mean and range over the tones, Sxx and the s-factor against resonant
frequency at a few frequencies), each as a figure and a CSV table, with
a text summary. ``export_timestream_report`` writes everything.
"""
import os
import numpy as np
import pandas as pd
from .. import daq_layout as dl
from . import hee_finder as cf
from . import photon_finder as pf
from . import psds
from .stream_sources import open_calibration
from ..signal.photons import PPM

# Frequencies of the Sxx and s-factor against resonant frequency plots, Hz
# (those the group has)
POINT_FREQS = (0.1, 1., 10., 100.)
# The range shown around the mean spectrum: percentiles over the tones
RANGE = (1., 99.)

################################################################################
################################### Lookups ####################################
################################################################################
def tone_frequencies(root, grp):
    """
    Return the resonant frequencies of a result's tones: the tones of
    the calibration sweep the result's run was streamed on, matched by
    resonator index.

    Parameters:
    root (zarr.Group): the store root.
    grp (zarr.Group): a photon, HEE or PSD timestream result (attr
        ``source``, the sweep, or ``run`` whose run list names it; array
        ``res_idxs``).

    Returns:
    fres (np.array): per tone, Hz (NaN for a tone the sweep lacks).
    """
    a = grp.attrs
    sweep = None
    src = a.get('source')
    if src and src in root and 'fres' in root[src] and 'res_idxs' in \
            root[src]:
        sweep = root[src]
    else:
        runs = {r['group']: r for r in dl.stream_runs(root)}
        cal = runs.get(a.get('run'), {}).get('calibration')
        if cal and cal in root:
            sweep = root[cal]
    if sweep is None:
        # An older result: the sweep through its stream calibration's
        # timestream listing
        cal = a.get('calibration') or src
        if cal and cal in root:
            listed = pf.list_timestreams(root, open_calibration(root, cal))
            for ts in listed:
                if ts['run'] == a.get('run') and \
                        ts['tone_set'] == a.get('tone_set') and \
                        (a.get('tuning') is None or
                         ts.get('tuning') == a.get('tuning')):
                    sweep = root[ts['source']['group']]
                    break
    if sweep is None:
        raise KeyError(f'{grp.path}: no calibration sweep found for its '
                       'tones')
    idxs = np.asarray(sweep['res_idxs'][:], dtype = int).tolist()
    lookup = dict(zip(idxs,
                      np.asarray(sweep['fres'][:], dtype = float).tolist()))
    return np.array([lookup.get(int(r), np.nan)
                     for r in grp['res_idxs'][:]], dtype = float)

def _bad(root, grp):
    """
    Return a result's bad marks: its own (saved with it) and the
    registry's marks that apply to it now (``daq_layout.bad_tones`` at
    its stage), so a global mark made later, in any step, leaves the
    tone out of the report.

    Parameters:
    root (zarr.Group): the store root.
    grp (zarr.Group): the result.

    Returns:
    bad (np.array, bool), reason (np.array, str): per tone.
    """
    res_idxs = np.asarray(grp['res_idxs'][:], dtype = int)
    n = len(res_idxs)
    if 'bad' in grp:
        bad = np.asarray(grp['bad'][:], dtype = bool)
        reason = np.asarray(grp['bad_reason'][:], dtype = str)
    else:
        bad, reason = np.zeros(n, bool), np.array([''] * n, dtype = str)
    marks = dl.bad_tones(root, grp.path)
    reason = reason.astype(object)
    for i, r in enumerate(res_idxs):
        if int(r) in marks and not bad[i]:
            bad[i] = True
            reason[i] = dl.bad_reason(marks[int(r)])
    return bad, reason.astype(str)

################################################################################
################################## Lifetimes ###################################
################################################################################
def lifetime_table(root, grp, kind):
    """
    Tabulate a photon or HEE result per tone.

    Parameters:
    root (zarr.Group): the store root.
    grp (zarr.Group): ``photons/<name>`` or ``hee/<name>``.
    kind (str): 'photons' or 'hee'.

    Returns:
    table (pd.DataFrame): ``res_idx``, ``fres`` (Hz), ``height`` (the mean
        pulse's optimal-filter height, kHz/GHz; NaN for HEEs),
        ``tau`` (s), ``rate`` (pulses per second), ``n_used``,
        ``status``, ``bad``, ``bad_reason``.
    """
    bad, reason = _bad(root, grp)
    height = np.asarray(grp['amp_ph'][:], dtype = float) * PPM \
        if kind == 'photons' else np.full(len(bad), np.nan)
    rate = np.asarray(grp['rate'][:], dtype = float) if 'rate' in grp \
        else np.full(len(bad), np.nan)
    return pd.DataFrame(dict(
        res_idx = np.asarray(grp['res_idxs'][:], dtype = int),
        fres = tone_frequencies(root, grp), height = height,
        tau = np.asarray(grp['tau'][:], dtype = float), rate = rate,
        n_used = np.asarray(grp['n_used'][:], dtype = int),
        status = np.asarray(grp['status'][:], dtype = str), bad = bad,
        bad_reason = reason))

def multiplicity_table(grp, kind):
    """
    Tabulate the coincident events of a photon or HEE result.

    Parameters:
    grp (zarr.Group): ``photons/<name>`` or ``hee/<name>``.
    kind (str): 'photons' or 'hee'.

    Returns:
    table (pd.DataFrame): per event: ``time_s`` (since the start of the
        run) and ``mult`` (tones hit). Empty if the result has none.
    """
    ev = grp.get('hee/events' if kind == 'photons' else 'events')
    if ev is None or 'mult' not in ev or 'time_s' not in ev:
        return pd.DataFrame(dict(time_s = np.zeros(0), mult = np.zeros(0,
                                                                     int)))
    return pd.DataFrame(dict(
        time_s = np.asarray(ev['time_s'][:], dtype = float),
        mult = np.asarray(ev['mult'][:], dtype = int)))

def plot_multiplicity(table, duration, title = '', bin_s = 5.):
    """
    Plot the coincident events of a result: how many tones each hit, and
    their rate over the run.

    Parameters:
    table (pd.DataFrame): ``multiplicity_table`` output.
    duration (float): length of the run, s.
    title (str): figure title. Default ''.
    bin_s (float): time bin, s. Default 5.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 3.6),
                                   gridspec_kw = dict(width_ratios = [2, 3]))
    mult = table['mult'].to_numpy()
    if len(mult):
        values, counts = np.unique(mult, return_counts = True)
        ax1.bar(values, counts, width = 0.8, color = 'C0')
        ax1.set_yscale('log')
    ax1.set_xlabel('Multiplicity (tones hit per event)')
    ax1.set_ylabel('Events')
    ax1.set_title(f'{len(mult)} events', fontsize = 10)
    ax1.grid(alpha = 0.3)
    edges = np.arange(0., max(duration, bin_s) + bin_s / 2, bin_s)
    t = table['time_s'].to_numpy()
    counts, _ = np.histogram(t, edges)
    ax2.stairs(counts / bin_s, edges, color = 'C0')
    ax2.set_xlabel('Time since the start of the run (s)')
    ax2.set_ylabel('Events per second')
    ax2.set_ylim(bottom = 0)
    ax2.grid(alpha = 0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig

def _format_axes(ax, x_mhz = True):
    """
    Put thousands separators on an axis's tick labels.

    Parameters:
    ax (Axes): the axes.
    x_mhz (bool): If True (default), the x axis too.
    """
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f'{v:,.10g}')
    ax.yaxis.set_major_formatter(fmt)
    if x_mhz:
        ax.xaxis.set_major_formatter(fmt)

def outliers(v, log = False, k = 3.):
    """
    Flag the outliers of some values: beyond ``k`` interquartile ranges
    from the quartiles (of log10 of the values when ``log``).

    Parameters:
    v (np.array): the values (NaN ignored).
    log (bool): If True, in log10. Default False.
    k (float): the distance, in interquartile ranges. Default 3.

    Returns:
    out (np.array, bool): True where an outlier (False where NaN).
    """
    v = np.asarray(v, dtype = float)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        w = np.where(v > 0, np.log10(v), np.nan) if log else v
    ok = np.isfinite(w)
    if ok.sum() < 4:
        return np.zeros(len(v), bool)
    q1, q3 = np.percentile(w[ok], [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(len(v), bool)
    out = np.zeros(len(v), bool)
    out[ok] = (w[ok] < q1 - k * iqr) | (w[ok] > q3 + k * iqr)
    return out

def _draw_outliers(ax, x, v, lo, hi, color, log = False, placed = None):
    """
    Draw outliers at the edge of the y range as arrows with their values.

    Parameters:
    ax (Axes): the axes.
    x, v (np.array): the outliers' x and values.
    lo, hi (float): the y range shown.
    color (str): color.
    log (bool): If True, the y axis is logarithmic. Default False.
    placed (list or None): labels drawn so far at the same edges, tuples
        (x, up, step), so that labels of several series close in x step
        away from each other. Updated in place. None (default): a new
        list.

    Returns:
    placed (list): the labels drawn, including this call's.
    """
    if placed is None:
        placed = []
    xlo, xhi = ax.get_xlim()
    near = 0.04 * (xhi - xlo)
    order = np.argsort(np.asarray(x, dtype = float))
    for xi, vi in zip(np.asarray(x)[order], np.asarray(v)[order]):
        up = vi > hi
        if log:
            y = hi / 1.15 if up else lo * 1.15
        else:
            y = hi - 0.05 * (hi - lo) if up else lo + 0.05 * (hi - lo)
        ax.plot([xi], [y], marker = '^' if up else 'v', color = color,
                markersize = 8, linestyle = 'none')
        # Labels of arrows close in x step away from the edge so they
        # stay readable
        taken = {s for px, pu, s in placed
                 if pu == up and abs(px - xi) < near}
        step = 0
        while step in taken:
            step += 1
        placed.append((xi, up, step))
        text = f'{vi:.2e}' if log else \
            (f'{vi:,.0f}' if abs(vi) >= 100 else f'{vi:,.3g}')
        ax.annotate(text, (xi, y), textcoords = 'offset points',
                    xytext = (4, -9 * step if up else 9 * step),
                    fontsize = 7, color = color, va = 'center')
    return placed

def _hist_side(hx, vals, color, log = False, nbins = None):
    """
    Draw a horizontal histogram at the side of a scatter plot.

    Parameters:
    hx (Axes): the histogram axes (sharing y with the scatter).
    vals (np.array): the values (log10 of them when ``log``).
    color (str): color.
    log (bool): If True, the y axis is logarithmic. Default False.
    nbins (int or None): bins. Default None: from the count.
    """
    if not len(vals):
        return
    lo, hi = vals.min(), vals.max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    n = max(6, min(20, len(vals) // 6 + 6)) if nbins is None else nbins
    edges = np.linspace(lo, hi, n + 1)
    counts, _ = np.histogram(vals, edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    hx.barh(10. ** centers if log else centers, counts,
            height = np.diff(10. ** edges) if log else np.diff(edges),
            color = color, alpha = 0.5)

def _scatter_with_hist(fig, gs, row, table, column, label):
    """
    Draw one quantity against resonant frequency (MHz), with a histogram
    of it at the side sharing the y axis. Tones marked bad are left out;
    outliers are drawn at the edge as arrows with their values and left
    out of the histogram and the y range.

    Parameters:
    fig (Figure): the figure.
    gs (GridSpec): two columns, the second narrow.
    row (int): the row of the grid.
    table (pd.DataFrame): with ``fres``, ``bad`` and the column.
    column (str): the quantity.
    label (str): y label.

    Returns:
    ax (Axes): the scatter axes.
    """
    import matplotlib.pyplot as plt
    ax = fig.add_subplot(gs[row, 0])
    hx = fig.add_subplot(gs[row, 1], sharey = ax)
    keep = ~table['bad'].to_numpy()
    v = table[column].to_numpy(dtype = float)[keep]
    f = table['fres'].to_numpy(dtype = float)[keep] / 1e6
    ok = np.isfinite(v)
    out = outliers(v)
    good = ok & ~out
    ax.scatter(f[good], v[good], s = 14, color = 'C0')
    if good.any():
        lo, hi = v[good].min(), v[good].max()
        pad = 0.08 * (hi - lo) or 0.1 * abs(hi) or 1.
        lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, hi)
        _draw_outliers(ax, f[out], v[out], lo, hi, 'C3')
        _hist_side(hx, v[good], 'C0')
    ax.set_ylabel(label)
    ax.grid(True, alpha = 0.3)
    _format_axes(ax)
    hx.set_xlabel('tones')
    plt.setp(hx.get_yticklabels(), visible = False)
    hx.grid(True, alpha = 0.3)
    return ax

def plot_lifetimes(table, title = ''):
    """
    Plot pulse height and lifetime (us) against resonant frequency
    (MHz), each with a histogram at the side.

    Parameters:
    table (pd.DataFrame): ``lifetime_table`` output.
    title (str): figure title. Default ''.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    rows = [('tau_us', 'Lifetime (us)')]
    table = table.assign(tau_us = table['tau'] * 1e6)
    if np.isfinite(table['height']).any():
        rows.insert(0, ('height', 'Pulse height (kHz/GHz)'))
    if 'rate' in table and np.isfinite(table['rate']).any():
        rows.append(('rate', 'Pulses per second'))
    fig = plt.figure(figsize = (12, 3.6 * len(rows)))
    gs = GridSpec(len(rows), 2, width_ratios = [4, 1], figure = fig,
                  wspace = 0.05, hspace = 0.3)
    for k, (col, label) in enumerate(rows):
        ax = _scatter_with_hist(fig, gs, k, table, col, label)
        if k == 0:
            n_bad = int(table['bad'].sum())
            ax.set_title(title + (f' ({n_bad} tones marked bad left out)'
                                  if n_bad else ''))
    ax.set_xlabel('Resonant frequency (MHz)')
    return fig

################################################################################
################################ Noise spectra #################################
################################################################################
def _interp_log(f_to, f_from, s):
    """
    Interpolate spectra onto other binned frequencies, in log-log.

    Parameters:
    f_to (np.array): target frequencies.
    f_from (np.array): the spectra's frequencies.
    s (np.array, (N, nb)): spectra (NaN rows stay NaN).

    Returns:
    out (np.array, (N, len(f_to))): the spectra at ``f_to``.
    """
    out = np.full((len(s), len(f_to)), np.nan)
    ok_f = f_from > 0
    lt = np.log10(np.asarray(f_to, dtype = float))
    for i, row in enumerate(s):
        good = ok_f & np.isfinite(row) & (row > 0)
        if good.sum() < 2:
            continue
        out[i] = 10. ** np.interp(lt, np.log10(f_from[good]),
                                  np.log10(row[good]), left = np.nan,
                                  right = np.nan)
    return out

def psd_group_tables(root, grp):
    """
    Gather a PSD group's results over every tone of every timestream.

    Parameters:
    root (zarr.Group): the store root.
    grp (zarr.Group): ``ts_psd{p}``.

    Returns:
    points (pd.DataFrame): per tone: ``timestream``, ``run``, ``res_idx``,
        ``fres`` (Hz), ``offres`` (a calibration tone), ``status``,
        ``bad``, ``bad_reason``, and at each reduced frequency f:
        ``sxx_<f>``, ``sxx_cn_<f>`` (1/Hz) and ``sfactor_<f>`` (dB).
    spectra (dict): ``f`` (the first timestream's binned frequencies, Hz)
        and, for each of ``sxx``, ``sxx_cn``, ``spar``, ``sper``, an
        (N, nb) array of every tone's binned spectrum interpolated onto
        ``f`` (NaN where a tone has none).
    f_reduced (np.array): the group's reduced frequencies, Hz.
    """
    f_reduced = np.asarray(grp['f_reduced'][:], dtype = float)
    names = sorted((n for n in grp.group_keys() if n.startswith('timestream')),
                   key = lambda n: int(n[len('timestream'):]))
    rows, f = [], None
    spectra = {k: [] for k in ('sxx', 'sxx_cn', 'spar', 'sper')}
    keys = {'sxx': 'sxx_binned', 'sxx_cn': 'sxx_binned_cn',
            'spar': 'spar_binned', 'sper': 'sper_binned'}
    for name in names:
        t = grp[name]
        n = t['res_idxs'].shape[0]
        fb = np.asarray(t['f_binned'][:], dtype = float)
        if f is None:
            f = fb
        # Groups saved before the spectra ended below the Nyquist
        # frequency are cut the same way here
        cut = fb > psds.SPECTRUM_CUTOFF * 0.5 / float(t.attrs['dt'])
        for key, arr in keys.items():
            s = np.asarray(t[arr][:], dtype = float) if arr in t else \
                np.full((n, len(fb)), np.nan)
            s = s.copy()
            s[:, cut] = np.nan
            spectra[key].append(s if np.array_equal(fb, f) and s.shape[1]
                                == len(f) else _interp_log(f, fb, s))
        bad, reason = _bad(root, t)
        res_idxs = np.asarray(t['res_idxs'][:], dtype = int)
        d = dict(timestream = [name] * n, run = [t.attrs.get('run')] * n,
                 res_idx = res_idxs, fres = tone_frequencies(root, t),
                 offres = res_idxs < 0,
                 status = np.asarray(t['status'][:], dtype = str), bad = bad,
                 bad_reason = reason)
        sr = np.asarray(t['sxx_reduced'][:], dtype = float)
        cn = np.asarray(t['sxx_reduced_cn'][:], dtype = float) \
            if 'sxx_reduced_cn' in t else np.full_like(sr, np.nan)
        sf = np.asarray(t['sfactor_reduced'][:], dtype = float) \
            if 'sfactor_reduced' in t else np.full_like(sr, np.nan)
        for j, fr in enumerate(f_reduced):
            d[f'sxx_{fr:g}'] = sr[:, j]
            d[f'sxx_cn_{fr:g}'] = cn[:, j]
            d[f'sfactor_{fr:g}'] = sf[:, j]
        rows.append(pd.DataFrame(d))
    points = pd.concat(rows, ignore_index = True) if rows else \
        pd.DataFrame()
    spectra = {k: (np.concatenate(v) if v else np.zeros((0, 0)))
               for k, v in spectra.items()}
    spectra['f'] = np.zeros(0) if f is None else f
    return points, spectra, f_reduced

def spectra_stats(spectra, keep):
    """
    Return the mean and the range of the spectra over some tones.

    Parameters:
    spectra (dict): ``psd_group_tables`` output.
    keep (np.array, bool): the tones used.

    Returns:
    stats (pd.DataFrame): ``f`` and, per spectrum kind, ``<kind>_mean``
        (the geometric mean), ``<kind>_lo`` and ``<kind>_hi`` (the
        ``RANGE`` percentiles), NaN where no tone has a value.
    """
    out = dict(f = spectra['f'])
    for kind in ('sxx', 'sxx_cn', 'spar', 'sper'):
        s = spectra[kind][keep] if len(spectra[kind]) else np.zeros((0, 0))
        nb = len(spectra['f'])
        mean = np.full(nb, np.nan)
        lo, hi = np.full(nb, np.nan), np.full(nb, np.nan)
        if len(s):
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                ls = np.where(s > 0, np.log10(s), np.nan)
            has = np.isfinite(ls).any(axis = 0)
            mean[has] = 10. ** np.nanmean(ls[:, has], axis = 0)
            lo[has] = 10. ** np.nanpercentile(ls[:, has], RANGE[0], axis = 0)
            hi[has] = 10. ** np.nanpercentile(ls[:, has], RANGE[1], axis = 0)
        out[f'{kind}_mean'], out[f'{kind}_lo'], out[f'{kind}_hi'] = mean, lo, hi
    return pd.DataFrame(out)

def _band(ax, f, stats, kind, color, label, style = '-', db = False):
    """
    Draw a mean spectrum with its range.

    Parameters:
    ax (Axes): the axes.
    f (np.array): frequencies.
    stats (pd.DataFrame): ``spectra_stats`` output.
    kind (str): spectrum kind.
    color (str): color.
    label (str): legend label.
    style (str): line style. Default '-'.
    db (bool): If True, draw 10 log10 of the values (dBc/Hz for the
        loop components, normalized to the carrier). Default False.
    """
    m = stats[f'{kind}_mean'].to_numpy()
    if not np.isfinite(m).any():
        return
    lo, hi = stats[f'{kind}_lo'].to_numpy(), stats[f'{kind}_hi'].to_numpy()
    if db:
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            m, lo, hi = (10 * np.log10(v) for v in (m, lo, hi))
    ax.plot(f, m, style, color = color, label = label)
    ax.fill_between(f, lo, hi, color = color, alpha = 0.2, linewidth = 0)

def _points_with_hist(fig, gs, row, points, prefix, freqs, label, log):
    """
    Draw a reduced quantity against resonant frequency (MHz) at several
    frequencies, with histograms at the side sharing the y axis. Tones
    marked bad are left out; outliers (beyond the y range of the other
    values) are drawn at the edge as arrows with their values and left
    out of the histograms and the y range.

    Parameters:
    fig (Figure): the figure.
    gs (GridSpec): the grid.
    row (int): the row.
    points (pd.DataFrame): ``psd_group_tables`` points.
    prefix (str): column prefix, 'sxx', 'sxx_cn' or 'sfactor'.
    freqs (list of float): the frequencies.
    label (str): y label.
    log (bool): If True, log y.

    Returns:
    ax (Axes): the scatter axes.
    """
    import matplotlib.pyplot as plt
    ax = fig.add_subplot(gs[row, 0])
    hx = fig.add_subplot(gs[row, 1], sharey = ax)
    keep = ~points['bad'].to_numpy()
    f = points['fres'].to_numpy(dtype = float)[keep] / 1e6
    series = []
    for k, fr in enumerate(freqs):
        v = points[f'{prefix}_{fr:g}'].to_numpy(dtype = float)[keep]
        ok = np.isfinite(v) & ((v > 0) if log else True)
        out = outliers(np.where(ok, v, np.nan), log = log)
        good = ok & ~out
        ax.scatter(f[good], v[good], s = 12, color = f'C{k}',
                   label = f'{fr:g} Hz')
        series.append((k, f[out], v[out], v[good]))
    if log:
        ax.set_yscale('log')
    good_all = np.concatenate([g for _, _, _, g in series]) if series \
        else np.zeros(0)
    if len(good_all):
        lo, hi = good_all.min(), good_all.max()
        if log:
            lo, hi = lo / 1.5, hi * 1.5
        else:
            pad = 0.08 * (hi - lo) or 0.1 * abs(hi) or 1.
            lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, hi)
        placed = []
        for k, fo, vo, g in series:
            # An outlier of its own series that lies within the range
            # of the others is an ordinary point, not an arrow
            inside = (vo >= lo) & (vo <= hi)
            if inside.any():
                ax.scatter(fo[inside], vo[inside], s = 12, color = f'C{k}')
            _draw_outliers(ax, fo[~inside], vo[~inside], lo, hi, f'C{k}',
                           log, placed)
        vals = np.log10(good_all) if log else good_all
        vlo, vhi = vals.min(), vals.max()
        if vlo == vhi:
            vlo, vhi = vlo - 0.5, vhi + 0.5
        n = max(6, min(20, len(vals) // 8 + 6))
        edges = np.linspace(vlo, vhi, n + 1)
        for k, _, _, g in series:
            counts, _ = np.histogram(np.log10(g) if log else g, edges)
            centers = 0.5 * (edges[1:] + edges[:-1])
            hx.barh(10. ** centers if log else centers, counts,
                    height = np.diff(10. ** edges) if log else np.diff(edges),
                    color = f'C{k}', alpha = 0.45)
    ax.set_ylabel(label)
    ax.grid(True, alpha = 0.3)
    ax.legend(loc = 'upper right', fontsize = 8, ncol = max(len(freqs), 1))
    _format_axes(ax, x_mhz = True)
    if log:
        from matplotlib.ticker import LogFormatterSciNotation
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    hx.set_xlabel('tones')
    plt.setp(hx.get_yticklabels(), visible = False)
    hx.grid(True, alpha = 0.3)
    return ax

def point_freqs(f_reduced, points):
    """
    Choose the frequencies of the points plots: those of ``POINT_FREQS``
    the group has with a value.

    Parameters:
    f_reduced (np.array): the group's reduced frequencies.
    points (pd.DataFrame): the points.

    Returns:
    freqs (list of float): the frequencies.
    """
    out = []
    for fr in POINT_FREQS:
        j = np.flatnonzero(np.isclose(f_reduced, fr))
        if len(j) and f'sxx_{fr:g}' in points and \
                np.isfinite(points[f'sxx_{fr:g}']).any():
            out.append(fr)
    return out

def plot_psd_group(points, spectra, stats, f_reduced, title = ''):
    """
    Plot a PSD group: the Sxx mean and range over the tones (before and
    after correlated noise removal; on- and off-resonance tones apart),
    then Sxx against resonant frequency at a few frequencies with
    histograms at the side; and the same for the loop components
    (parallel and perpendicular to the loop: amplitude and dissipation
    noise) and the s-factor.

    Parameters:
    points (pd.DataFrame): ``psd_group_tables`` points.
    spectra (dict): its spectra.
    stats (dict): 'on' and, if any, 'off': ``spectra_stats`` of the on-
        and off-resonance tones.
    f_reduced (np.array): the group's reduced frequencies.
    title (str): figure title. Default ''.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    freqs = point_freqs(f_reduced, points)
    f = spectra['f']
    has_loop = any(np.isfinite(s['spar_mean']).any() for s in stats.values())
    nrows = 4 if has_loop else 2
    fig = plt.figure(figsize = (12, 3.75 * nrows))
    gs = GridSpec(nrows, 2, width_ratios = [4, 1], figure = fig,
                  wspace = 0.05, hspace = 0.35)
    ax = fig.add_subplot(gs[0, :])
    for which, style in (('on', '-'), ('off', '--')):
        if which not in stats:
            continue
        tag = '' if which == 'on' else ' (off resonance)'
        _band(ax, f, stats[which], 'sxx', 'C7', 'raw' + tag, style)
        _band(ax, f, stats[which], 'sxx_cn', 'C0',
              'correlated noise removed' + tag, style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Sxx (1/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(title + ': mean over the tones, shaded '
                 f'{RANGE[0]:g}-{RANGE[1]:g} percentile')
    ax.grid(True, alpha = 0.3, which = 'both')
    ax.legend(loc = 'upper right', fontsize = 8)
    on = points[~points['offres']] if len(points) else points
    if freqs and len(on):
        prefix = 'sxx_cn' if np.isfinite(
            on[[f'sxx_cn_{fr:g}' for fr in freqs]]).any().any() else 'sxx'
        ax = _points_with_hist(fig, gs, 1, on, prefix, freqs,
                               'Sxx (1/Hz)' + (', correlated noise removed'
                                               if prefix == 'sxx_cn'
                                               else ''), True)
        ax.set_xlabel('Resonant frequency (MHz)')
    if not has_loop:
        return fig
    ax = fig.add_subplot(gs[2, :])
    for which, style in (('on', '-'), ('off', '--')):
        if which not in stats:
            continue
        tag = '' if which == 'on' else ' (off resonance)'
        _band(ax, f, stats[which], 'spar', 'C1',
              'parallel (dissipation)' + tag, style, db = True)
        _band(ax, f, stats[which], 'sper', 'C2',
              'perpendicular (amplitude)' + tag, style, db = True)
    ax.set_xscale('log')
    ax.set_ylabel('PSD (dBc/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Amplitude / dissipation noise')
    ax.grid(True, alpha = 0.3, which = 'both')
    ax.legend(loc = 'upper right', fontsize = 8)
    if freqs and len(on):
        ax = _points_with_hist(fig, gs, 3, on, 'sfactor', freqs,
                               's-factor (dB)', False)
        ax.set_xlabel('Resonant frequency (MHz)')
    return fig

################################################################################
#################################### Report ####################################
################################################################################
def report_text(root, lifetimes, groups):
    """
    Describe the results in plain text.

    Parameters:
    root (zarr.Group): the store root.
    lifetimes (list of tuple): (kind, name, table) per lifetime result.
    groups (list of tuple): (p, points, stats) per PSD group.

    Returns:
    text (str): the description.
    """
    lines = [f"Device {root.attrs.get('device')}, cooldown "
             f"{root.attrs.get('cooldown')}", '']
    for kind, name, t in lifetimes:
        grp = root[f'{kind}/{name}']
        good = t[~t['bad'] & np.isfinite(t['tau'])]
        lines += [f"{kind}/{name}: run {grp.attrs.get('run')} "
                  f"({grp.attrs.get('name')}), tone set "
                  f"{grp.attrs.get('tone_set')}, {len(t)} tones, "
                  f"{int(t['bad'].sum())} marked bad, "
                  f"{int((t['status'] != '').sum())} failed"]
        if len(good):
            med = np.median(good['tau']) * 1e6
            lines.append(f"  lifetime: median {med:.3g} "
                         f"us, {RANGE[0]:g}-{RANGE[1]:g} percentile "
                         f"{np.percentile(good['tau'], RANGE[0]) * 1e6:.3g}-"
                         f"{np.percentile(good['tau'], RANGE[1]) * 1e6:.3g} us")
            if np.isfinite(good['height']).any():
                lines.append(f"  pulse height: median "
                             f"{np.nanmedian(good['height']):.3g} kHz/GHz")
            if 'rate' in good and np.isfinite(good['rate']).any():
                lines.append(f"  pulses per second: median "
                             f"{np.nanmedian(good['rate']):.3g}")
        events = multiplicity_table(grp, kind)
        if len(events):
            m = events['mult'].to_numpy()
            lines.append(f"  multiplicity: {len(m)} events, tones hit "
                         f"median {int(np.median(m))}, most {int(m.max())}")
        for _, r in t[t['bad']].iterrows():
            lines.append(f"  res {r['res_idx']} bad: {r['bad_reason']}")
    for p, points, stats in groups:
        grp = root[f'ts_psd{p}']
        s = grp.attrs.get('settings', {})
        n_ts = points['timestream'].nunique() if len(points) else 0
        n_bad = int(points['bad'].sum()) if len(points) else 0
        lines += ['', f"ts_psd{p} ({grp.attrs.get('name') or 'no name'}): "
                  f"{len(points)} tones in {n_ts} timestreams, {n_bad} "
                  f"marked bad; common modes "
                  f"{'removed' if s.get('cn_remove') else 'kept'}, deglitch "
                  f"{s.get('deglitch')}"]
        for fr in point_freqs(np.asarray(grp['f_reduced'][:]), points):
            v = points.loc[~points['bad'] & ~points['offres'],
                           f'sxx_cn_{fr:g}']
            v = v[np.isfinite(v)]
            if len(v):
                lines.append(f"  Sxx at {fr:g} Hz (common modes removed): "
                             f"median {np.median(v):.3g} 1/Hz, "
                             f"{RANGE[0]:g}-{RANGE[1]:g} percentile "
                             f"{np.percentile(v, RANGE[0]):.3g}-"
                             f"{np.percentile(v, RANGE[1]):.3g}")
    return '\n'.join(lines) + '\n'

def export_timestream_report(
    root, out_dir, photons = None, hee = None, groups = None, dpi = 150
):
    """
    Write the figures, tables and text of a store's timestream results.

    Parameters:
    root (zarr.Group, str, or os.PathLike): the store.
    out_dir (str or os.PathLike): output directory (created if needed).
    photons (list of str or None): photon results to report (names).
        Default None: all.
    hee (list of str or None): HEE results. Default None: all.
    groups (list of int or None): PSD groups (p). Default None: all.
    dpi (int): plot resolution. Default 150.

    Returns:
    paths (dict): file kind: path, for every file written (``info``, and
        ``lifetime_<kind>_<name>``, ``lifetime_<kind>_<name>_csv``,
        ``psd<p>``, ``psd<p>_csv``, ``psd<p>_spectra_csv``).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    root = dl.open_store(root)
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok = True)
    device = root.attrs.get('device', 'device')
    stem = os.path.join(out_dir, str(device))
    backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    paths, lifetimes, group_out = {}, [], []
    kinds = (('photons', pf.list_results(root) if photons is None
              else list(photons)),
             ('hee', cf.list_results(root) if hee is None
              else list(hee)))
    skipped = []
    for kind, names in kinds:
        for name in names:
            grp = root[f'{kind}/{name}']
            if 'res_idxs' not in grp or 'tau' not in grp:
                # Saved incompletely (an interrupted run)
                skipped.append(f'{kind}/{name}')
                continue
            table = lifetime_table(root, grp, kind)
            fig = plot_lifetimes(table, f'{device} {kind}/{name}')
            key = f'lifetime_{kind}_{name}'
            paths[key] = f'{stem}_{key}.png'
            fig.savefig(paths[key], dpi = dpi, bbox_inches = 'tight')
            plt.close(fig)
            paths[key + '_csv'] = f'{stem}_{key}.csv'
            table.to_csv(paths[key + '_csv'], index = False)
            lifetimes.append((kind, name, table))
            events = multiplicity_table(grp, kind)
            if len(events):
                a = grp.attrs
                duration = (int(a['stop']) - int(a['start'])) * float(a['dt'])
                fig = plot_multiplicity(
                    events, duration,
                    f'{device} {kind}/{name}: multiplicity')
                key = f'multiplicity_{kind}_{name}'
                paths[key] = f'{stem}_{key}.png'
                fig.savefig(paths[key], dpi = dpi, bbox_inches = 'tight')
                plt.close(fig)
                paths[key + '_csv'] = f'{stem}_{key}.csv'
                events.to_csv(paths[key + '_csv'], index = False)
    ps = dl.list_indices(root, psds.PREFIX) if groups is None else \
        [int(p) for p in groups]
    for p in ps:
        grp = root[f'ts_psd{p}']
        if 'f_reduced' not in grp or not any(
                n.startswith('timestream') and 'res_idxs' in grp[n]
                for n in grp.group_keys()):
            # Saved incompletely (an interrupted run)
            skipped.append(f'ts_psd{p}')
            continue
        points, spectra, f_reduced = psd_group_tables(root, grp)
        keep = ~points['bad'].to_numpy() if len(points) else \
            np.zeros(0, bool)
        off = points['offres'].to_numpy() if len(points) else \
            np.zeros(0, bool)
        stats = {'on': spectra_stats(spectra, keep & ~off)}
        if off.any():
            stats['off'] = spectra_stats(spectra, keep & off)
        name = grp.attrs.get('name')
        fig = plot_psd_group(points, spectra, stats, f_reduced,
                             f"{device} ts_psd{p}" + (f' ({name})' if name
                                                      else ''))
        key = f'psd{p}'
        paths[key] = f'{stem}_{key}.png'
        fig.savefig(paths[key], dpi = dpi, bbox_inches = 'tight')
        plt.close(fig)
        paths[key + '_csv'] = f'{stem}_{key}.csv'
        points.to_csv(paths[key + '_csv'], index = False)
        paths[key + '_spectra_csv'] = f'{stem}_{key}_spectra.csv'
        spectra_csv = stats['on'].rename(
            columns = lambda c: c if c == 'f' else c + '_on')
        if 'off' in stats:
            for c in stats['off'].columns:
                if c != 'f':
                    spectra_csv[c + '_off'] = stats['off'][c]
        spectra_csv.to_csv(paths[key + '_spectra_csv'], index = False)
        group_out.append((p, points, stats))
    if backend != matplotlib.get_backend():
        matplotlib.use(backend)
    paths['info'] = f'{stem}_timestreams.txt'
    text = report_text(root, lifetimes, group_out)
    if skipped:
        text += '\nSkipped (saved incompletely): ' + ', '.join(skipped) \
            + '\n'
    with open(paths['info'], 'w', encoding = 'utf-8') as fh:
        fh.write(text)
    return paths
