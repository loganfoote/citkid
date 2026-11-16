"""
Report of a store's timestream results as plain files: the lifetimes
(photon and cosmic-ray results: pulse height and decay time against
resonant frequency) and the noise spectra (PSD groups: the spectra's
mean and range over the tones, Sxx and the s-factor against resonant
frequency at a few frequencies), each as a figure and a CSV table, with
a text summary. ``export_timestream_report`` writes everything.
"""
import os
import numpy as np
import pandas as pd
from .. import daq_layout as dl
from . import cosmic_finder as cf
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
    grp (zarr.Group): a photon, cosmic or PSD timestream result (attr
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
                        ts['tone_set'] == a.get('tone_set'):
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

def _bad(grp):
    """
    Return a result's bad marks.

    Parameters:
    grp (zarr.Group): the result.

    Returns:
    bad (np.array, bool), reason (np.array, str): per tone.
    """
    n = grp['res_idxs'].shape[0]
    if 'bad' not in grp:
        return np.zeros(n, bool), np.array([''] * n)
    return (np.asarray(grp['bad'][:], dtype = bool),
            np.asarray(grp['bad_reason'][:], dtype = str))

################################################################################
################################## Lifetimes ###################################
################################################################################
def lifetime_table(root, grp, kind):
    """
    Tabulate a photon or cosmic-ray result per tone.

    Parameters:
    root (zarr.Group): the store root.
    grp (zarr.Group): ``photons/<name>`` or ``cosmic/<name>``.
    kind (str): 'photons' or 'cosmic'.

    Returns:
    table (pd.DataFrame): ``res_idx``, ``fres`` (Hz), ``height`` (the mean
        pulse's optimal-filter height, kHz/GHz; NaN for cosmic rays),
        ``tau`` (s), ``n_used``, ``status``, ``bad``, ``bad_reason``.
    """
    bad, reason = _bad(grp)
    height = np.asarray(grp['amp_ph'][:], dtype = float) * PPM \
        if kind == 'photons' else np.full(len(bad), np.nan)
    return pd.DataFrame(dict(
        res_idx = np.asarray(grp['res_idxs'][:], dtype = int),
        fres = tone_frequencies(root, grp), height = height,
        tau = np.asarray(grp['tau'][:], dtype = float),
        n_used = np.asarray(grp['n_used'][:], dtype = int),
        status = np.asarray(grp['status'][:], dtype = str), bad = bad,
        bad_reason = reason))

def _scatter_with_hist(fig, gs, row, table, column, label, log = False):
    """
    Draw one quantity against resonant frequency, with a histogram of it
    at the side sharing the y axis; bad tones hollow.

    Parameters:
    fig (Figure): the figure.
    gs (GridSpec): two columns, the second narrow.
    row (int): the row of the grid.
    table (pd.DataFrame): with ``fres``, ``bad`` and the column.
    column (str): the quantity.
    label (str): y label.
    log (bool): If True, log y. Default False.

    Returns:
    ax (Axes): the scatter axes.
    """
    import matplotlib.pyplot as plt
    ax = fig.add_subplot(gs[row, 0])
    hx = fig.add_subplot(gs[row, 1], sharey = ax)
    v = table[column].to_numpy(dtype = float)
    f = table['fres'].to_numpy(dtype = float) / 1e9
    ok = np.isfinite(v) & (v > 0 if log else True)
    good = ok & ~table['bad'].to_numpy()
    ax.scatter(f[good], v[good], s = 14, color = 'C0', label = 'tones')
    ax.scatter(f[ok & ~good], v[ok & ~good], s = 14, facecolors = 'none',
               edgecolors = 'C3', label = 'marked bad')
    if log:
        ax.set_yscale('log')
    if good.any():
        vals = np.log10(v[good]) if log else v[good]
        edges = np.linspace(vals.min(), vals.max() or vals.min() + 1,
                            max(8, min(40, good.sum() // 3 + 8)))
        if vals.min() == vals.max():
            edges = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 9)
        counts, _ = np.histogram(vals, edges)
        centers = 0.5 * (edges[1:] + edges[:-1])
        heights = np.diff(10. ** edges) if log else np.diff(edges)
        hx.barh(10. ** centers if log else centers, counts, height = heights,
                color = 'C0', alpha = 0.6)
    ax.set_ylabel(label)
    ax.grid(True, alpha = 0.3)
    hx.set_xlabel('tones')
    plt.setp(hx.get_yticklabels(), visible = False)
    hx.grid(True, alpha = 0.3)
    return ax

def plot_lifetimes(table, title = ''):
    """
    Plot pulse height and decay time against resonant frequency, each
    with a histogram at the side.

    Parameters:
    table (pd.DataFrame): ``lifetime_table`` output.
    title (str): figure title. Default ''.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    rows = [('tau_us', 'Decay time (us)')]
    table = table.assign(tau_us = table['tau'] * 1e6)
    if np.isfinite(table['height']).any():
        rows.insert(0, ('height', 'Pulse height (kHz/GHz)'))
    fig = plt.figure(figsize = (12, 3.6 * len(rows)))
    gs = GridSpec(len(rows), 2, width_ratios = [4, 1], figure = fig,
                  wspace = 0.05, hspace = 0.3)
    for k, (col, label) in enumerate(rows):
        ax = _scatter_with_hist(fig, gs, k, table, col, label)
        if k == 0:
            ax.legend(loc = 'upper right')
            ax.set_title(title)
    ax.set_xlabel('Resonant frequency (GHz)')
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
        for key, arr in keys.items():
            s = np.asarray(t[arr][:], dtype = float) if arr in t else \
                np.full((n, len(fb)), np.nan)
            spectra[key].append(s if np.array_equal(fb, f) and s.shape[1]
                                == len(f) else _interp_log(f, fb, s))
        bad, reason = _bad(t)
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

def _band(ax, f, stats, kind, color, label, style = '-'):
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
    """
    m = stats[f'{kind}_mean'].to_numpy()
    if not np.isfinite(m).any():
        return
    ax.plot(f, m, style, color = color, label = label)
    ax.fill_between(f, stats[f'{kind}_lo'].to_numpy(),
                    stats[f'{kind}_hi'].to_numpy(), color = color,
                    alpha = 0.2, linewidth = 0)

def _points_with_hist(fig, gs, row, points, prefix, freqs, label, log):
    """
    Draw a reduced quantity against resonant frequency at several
    frequencies, with histograms at the side sharing the y axis.

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
    f = points['fres'].to_numpy(dtype = float) / 1e9
    bad = points['bad'].to_numpy()
    all_vals = []
    for k, fr in enumerate(freqs):
        v = points[f'{prefix}_{fr:g}'].to_numpy(dtype = float)
        ok = np.isfinite(v) & ((v > 0) if log else True)
        good = ok & ~bad
        ax.scatter(f[good], v[good], s = 12, color = f'C{k}',
                   label = f'{fr:g} Hz')
        ax.scatter(f[ok & ~good], v[ok & ~good], s = 12, facecolors = 'none',
                   edgecolors = f'C{k}')
        all_vals.append((k, np.log10(v[good]) if log else v[good]))
    if log:
        ax.set_yscale('log')
    vals = np.concatenate([v for _, v in all_vals]) if all_vals else \
        np.zeros(0)
    if len(vals):
        lo, hi = vals.min(), vals.max()
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, max(8, min(40, len(vals) // 4 + 8)))
        for k, v in all_vals:
            counts, _ = np.histogram(v, edges)
            centers = 0.5 * (edges[1:] + edges[:-1])
            hx.barh(10. ** centers if log else centers, counts,
                    height = np.diff(10. ** edges) if log else np.diff(edges),
                    color = f'C{k}', alpha = 0.45)
    ax.set_ylabel(label)
    ax.grid(True, alpha = 0.3)
    ax.legend(loc = 'upper right', fontsize = 8, ncol = len(freqs))
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
    after common-mode removal; on- and off-resonance tones apart), then
    Sxx against resonant frequency at a few frequencies with histograms
    at the side; and the same for the loop components (along and across
    the loop) and the s-factor.

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
        s = stats[which]
        tag = '' if which == 'on' else ' (off resonance)'
        _band(ax, f, s, 'sxx', 'C7', 'raw' + tag, style)
        _band(ax, f, s, 'sxx_cn', 'C0', 'common modes removed' + tag, style)
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
        ax = _points_with_hist(fig, gs, 1, on, 'sxx_cn' if np.isfinite(
            on[[f'sxx_cn_{fr:g}' for fr in freqs]]).any().any() else 'sxx',
            freqs, 'Sxx (1/Hz)', True)
        ax.set_xlabel('Resonant frequency (GHz)')
    if not has_loop:
        return fig
    ax = fig.add_subplot(gs[2, :])
    for which, style in (('on', '-'), ('off', '--')):
        if which not in stats:
            continue
        tag = '' if which == 'on' else ' (off resonance)'
        _band(ax, f, stats[which], 'spar', 'C1', 'along the loop (r theta)'
              + tag, style)
        _band(ax, f, stats[which], 'sper', 'C2', 'across the loop (A)' + tag,
              style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('PSD (1/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Loop components')
    ax.grid(True, alpha = 0.3, which = 'both')
    ax.legend(loc = 'upper right', fontsize = 8)
    if freqs and len(on):
        ax = _points_with_hist(fig, gs, 3, on, 'sfactor', freqs,
                               's-factor (dB)', False)
        ax.set_xlabel('Resonant frequency (GHz)')
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
            lines.append(f"  decay time: median {med:.3g} "
                         f"us, {RANGE[0]:g}-{RANGE[1]:g} percentile "
                         f"{np.percentile(good['tau'], RANGE[0]) * 1e6:.3g}-"
                         f"{np.percentile(good['tau'], RANGE[1]) * 1e6:.3g} us")
            if np.isfinite(good['height']).any():
                lines.append(f"  pulse height: median "
                             f"{np.nanmedian(good['height']):.3g} kHz/GHz")
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
    root, out_dir, photons = None, cosmic = None, groups = None, dpi = 150
):
    """
    Write the figures, tables and text of a store's timestream results.

    Parameters:
    root (zarr.Group, str, or os.PathLike): the store.
    out_dir (str or os.PathLike): output directory (created if needed).
    photons (list of str or None): photon results to report (names).
        Default None: all.
    cosmic (list of str or None): cosmic-ray results. Default None: all.
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
             ('cosmic', cf.list_results(root) if cosmic is None
              else list(cosmic)))
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
    ps = dl.list_indices(root, psds.PREFIX) if groups is None else \
        [int(p) for p in groups]
    for p in ps:
        grp = root[f'ts_psd{p}']
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
