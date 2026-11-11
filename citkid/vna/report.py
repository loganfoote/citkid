"""
Export the results of a full VNA sweep screening (``vna{k}`` of a
``citkid.daq_layout`` store, after the resonance-finding step) in plain
files, for people who don't use zarr or Python:

- ``<device>_vna<k>.png``: the gain-removed sweep with the resonances
  marked at their dips (open circles where the fit is unreliable), and
  Qi, Qc, and phi vs frequency where they are reliable
  (``full_sweep.parameter_masks``);
- ``<device>_vna<k>_info.txt``: the sweep's settings (power, averaging,
  resolution, range, modules, Nyquist zones, bands), the finding and
  fitting steps, the number of resonances, the frequency schedule
  (range, gaps, banks), typical quality factors, data-quality notes, and
  any notes saved with the data;
- ``<device>_vna<k>_resonators.csv``: one row per resonator of
  ``resonators{k}`` (``res_idx`` as in the store): fitted fr, Qi, Qc, Qr,
  amp, phi with their errors, NaN where unreliable, plus the flags and the
  fit status.

``export_vna_report`` writes all three. This is a starting point for HTML
or wiki pages.
"""
import os
import numpy as np
import pandas as pd
from .. import daq_layout as dl
from .full_sweep import parameter_masks, resonator_list

# Fit columns exported, and the mask that applies to each
CSV_COLUMNS = (
    ('fr', 'valid_fr'), ('fr_err', 'valid_fr'), ('Qi', 'valid_Qi'),
    ('Qi_err', 'valid_Qi'), ('Qc', 'valid_Qc'), ('Qc_err', 'valid_Qc'),
    ('Qr', 'valid_Qc'), ('Qr_err', 'valid_Qc'), ('amp', 'valid_Qc'),
    ('amp_err', 'valid_Qc'), ('phi', 'valid_phi'), ('phi_err', 'valid_phi'),
    ('snr', 'valid_fr'))

################################################################################
################################### Tables #####################################
################################################################################
def resonator_table(fit_grp):
    """
    Make the resonator table of a full sweep fit, one row per resonator in
    ascending frequency (the order of the store's resonator list, so
    ``res_idx`` matches it), with unreliable values set to NaN.

    Parameters:
    fit_grp (zarr.Group): a ``full_sweep_fit`` group.

    Returns:
    table (pd.DataFrame): ``res_idx``, ``CSV_COLUMNS``, ``flags``,
        ``fit_status``, ``valid_fr``, ``valid_Qi``, ``valid_Qc``,
        ``valid_phi``.
    """
    status, flags = fit_grp['fit_status'][:], fit_grp['flags'][:]
    if 'valid_fr' in fit_grp:
        masks = {k: fit_grp[k][:] for k in
                 ('valid_fr', 'valid_Qi', 'valid_Qc', 'valid_phi')}
    else:
        masks = parameter_masks(status, flags)
    fres, _, _ = resonator_list(fit_grp)
    out = {}
    for col, mask in CSV_COLUMNS:
        v = np.asarray(fit_grp[col][:], dtype = float)
        out[col] = np.where(masks[mask], v, np.nan)
    # The resonator list's frequency: fitted, or the input where the fit
    # failed
    out['fr'] = fres
    out.update(flags = np.asarray(flags, dtype = str),
               fit_status = np.asarray(status, dtype = str), **masks)
    table = pd.DataFrame(out)
    table = table.iloc[np.argsort(fres, kind = 'stable')]
    table.insert(0, 'res_idx', np.arange(len(table)))
    return table.reset_index(drop = True)

def sweep_info_text(root, k, table):
    """
    Describe a VNA sweep and its screening in plain text.

    Parameters:
    root (zarr.Group): the store root.
    k (int): sweep index.
    table (pd.DataFrame): ``resonator_table``.

    Returns:
    text (str): the description.
    """
    grp = root[dl.set_name(k, 'vna')]
    a, fit = dict(grp.attrs), grp['full_sweep_fit']
    fa = dict(fit.attrs)
    lines = [f"Device: {root.attrs.get('device')}",
             f"Cooldown: {root.attrs.get('cooldown')}",
             f'VNA sweep: vna{k} (taken {a.get("created", "?")})', '']
    lines.append('Sweep settings')
    bands = {n: grp[n][:] for n in dl.VNA_BAND_ARRAYS if n in grp}
    lines += [
        f"  Tone power: {a.get('amplitude')} dBm",
        f"  Samples per point: {a.get('nsamps')}",
        f"  Range: {a.get('fmin', np.nan) / 1e6:.1f}-"
        f"{a.get('fmax', np.nan) / 1e6:.1f} MHz",
        f"  Frequency step: df/f = {a.get('step')} ({a.get('spacing')} "
        'spacing)',
        f"  Points per channel: {a.get('npoints_per_ch')}",
        f"  Points: {grp['f'].shape[0]:,}"]
    if 'chunk_modules' in bands:
        lines += [
            '  Modules: ' + ', '.join(str(m) for m in
                                      sorted(set(bands['chunk_modules']))),
            f"  Bands: {len(bands['chunk_modules'])} in "
            f"{len(set(bands['chunk_passes']))} passes"]
    if 'chunk_zones' in bands:
        lines.append('  Nyquist zones: ' + ', '.join(
            str(z) for z in sorted(set(bands['chunk_zones']))))
    if 'crs_config' in grp:
        cfg = grp['crs_config'].attrs.get('config', {})
        lines.append(f"  Module bandwidth: {cfg.get('bw', np.nan) / 1e6:.0f}"
                     ' MHz' + (' (extended)' if cfg.get('extended_bw')
                               else ''))
    setup = root.attrs.get('setup')
    if setup:
        lines.append(f'  Setup: {setup}')
    lines += ['', 'Finding and fitting']
    finder = grp['res_finder_auto'].attrs.get('finder') \
        if 'res_finder_auto' in grp else None
    lines.append(f"  Finder: {finder or 'unknown'}; checked by hand: "
                 f"{a.get('fres_manual_checked', 'unknown')}")
    lines.append(f"  Fit: {fa.get('created', '?')} (citkid "
                 f"{fa.get('citkid_version', '?')})")
    n = len(table)
    lines += ['', f'Resonances: {n}']
    for col in ('Qi', 'Qc', 'phi'):
        ok = int(table[f'valid_{col}'].sum())
        lines.append(f'  Reliable {col}: {ok} of {n}')
    rng = fa.get('range', {})
    if rng:
        lines += ['', 'Frequency schedule',
                  f"  Range: {rng.get('fmin', np.nan) / 1e6:.1f}-"
                  f"{rng.get('fmax', np.nan) / 1e6:.1f} MHz"]
        gaps = rng.get('gaps', [])
        if gaps:
            lines.append('  Gaps: ' + ', '.join(
                f'{g0 / 1e6:.0f}-{g1 / 1e6:.0f} MHz' for g0, g1 in gaps))
    banks = fa.get('banks', {})
    if banks.get('n_found'):
        lines.append('  Banks: ' + ', '.join(
            f"{key.replace('_', ' ')} {val}" for key, val in banks.items()))
    report = fa.get('report')
    if report:
        lines += ['', 'Summary (from the fit)'] + \
            ['  ' + line for line in report.splitlines()]
    for where, notes in (('sweep', a.get('notes')),
                         ('fit', fa.get('user_notes'))):
        if notes:
            lines += ['', f'Notes ({where})'] + \
                ['  ' + line for line in str(notes).splitlines()]
    lines += ['', 'Resonator table columns (CSV): res_idx is the index in '
              f'resonators{k}; values are NaN where the fit is unreliable '
              '(valid_* columns, flags and fit_status say why); frequencies '
              'in Hz, phi in rad.']
    return '\n'.join(lines) + '\n'

################################################################################
#################################### Plots #####################################
################################################################################
def _dip_points(f, ydb, fr, Qr, half_lw = 1.):
    """
    Find the deepest point of the curve within half_lw linewidths of each
    resonance.

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    ydb (np.array): curve in dB.
    fr (np.array): resonance frequencies in Hz.
    Qr (np.array): quality factors (NaN: the median linewidth).
    half_lw (float): half-width in linewidths. Default 1.

    Returns:
    x, y (np.array): frequency (Hz) and curve value (dB) of each dip.
    """
    fr = np.asarray(fr, dtype = float)
    lw = fr / np.asarray(Qr, dtype = float)
    lw = np.where(np.isfinite(lw), lw, np.nanmedian(lw)
                  if np.isfinite(lw).any() else 1e5)
    x, y = np.empty(len(fr)), np.empty(len(fr))
    for i, (c, w) in enumerate(zip(fr, lw)):
        i0, i1 = np.searchsorted(f, [c - half_lw * w, c + half_lw * w])
        if i1 <= i0:
            i0 = int(np.clip(np.searchsorted(f, c), 0, len(f) - 1))
            i1 = i0 + 1
        j = i0 + int(np.argmin(ydb[i0:i1]))
        x[i], y[i] = f[j], ydb[j]
    return x, y

def plot_vna_results(f, y, table, title = '', n_bins = 6000,
                     clip = (0.5, 99.5)):
    """
    Plot a screened VNA sweep: the sweep (min/max envelope per bin, so
    narrow dips stay visible) with the resonances marked, and Qi, Qc, and
    phi vs frequency on a shared logarithmic frequency axis, limited to
    the resonances' range (5% margin) if there are any. Unreliable
    values are NaN in ``table`` and left out; Qi and Qc outside the
    ``clip`` percentiles are left out of the plot (not the data).

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    y (np.array): complex sweep to draw (e.g. gain-removed).
    table (pd.DataFrame): ``resonator_table``.
    title (str): figure title. Default ''.
    n_bins (int): plotting bins for the sweep. Default 6000.
    clip (tuple): percentiles kept in the Qi and Qc panels. Default
        (0.5, 99.5).

    Returns:
    fig (matplotlib.figure.Figure): the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    c = plt.cm.tab10
    db = 20 * np.log10(np.maximum(np.abs(y), 1e-12))
    idx = np.unique(np.searchsorted(f, np.geomspace(f[0], f[-1], n_bins)))
    idx = idx[idx < len(f)]
    fig, axs = plt.subplots(4, 1, figsize = (14, 11), sharex = True,
                            gridspec_kw = dict(height_ratios = (2, 1, 1, 1)))
    ax = axs[0]
    lo, hi = np.minimum.reduceat(db, idx), np.maximum.reduceat(db, idx)
    x = f[idx] / 1e9
    ax.fill_between(x, lo, hi, color = c(0), lw = 0, alpha = 0.35)
    ax.plot(x, lo, color = c(0), lw = 1.)
    ax.plot(x, hi, color = c(0), lw = 1.)
    px, py = _dip_points(f, db, table.fr.values, table.Qr.values)
    good = table.valid_Qi.values.astype(bool)
    ax.plot(px[good] / 1e9, py[good], 'o', color = c(1), ms = 4.5,
            mec = 'white', mew = 0.5, zorder = 3,
            label = f'resonance ({good.sum()})')
    if (~good).any():
        ax.plot(px[~good] / 1e9, py[~good], 'o', mfc = 'none', mec = c(1),
                ms = 5, mew = 1., zorder = 3,
                label = f'unreliable fit ({(~good).sum()})')
    ax.set_ylabel('|S21 / gain| (dB)')
    ax.grid(True, alpha = 0.3)
    if len(table):
        ax.legend(loc = 'lower left', fontsize = 9, markerscale = 1.5)
    labels = {'Qi': '$Q_i$', 'Qc': '$Q_c$', 'phi': r'$\phi$ (rad)'}
    for ax, col in zip(axs[1:], ('Qi', 'Qc', 'phi')):
        v = table[col].values.astype(float)
        keep = np.isfinite(v)
        log = col != 'phi'
        if log and keep.sum() > 2:
            p0, p1 = np.percentile(v[keep], clip)
            keep &= (v >= p0) & (v <= p1)
        ax.plot(table.fr.values[keep] / 1e9, v[keep], 'o', color = c(0),
                ms = 3.5, mec = 'none')
        if log:
            ax.set_yscale('log')
        else:
            ax.axhline(0, color = '0.6', lw = 0.8)
        ax.set_ylabel(labels[col])
        ax.grid(True, which = 'major', alpha = 0.3)
    ax = axs[-1]
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(
        LogLocator(base = 10, subs = (1., 1.5, 2., 3., 4., 5., 7.)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Frequency (GHz)')
    if len(table):
        # The resonances' range (5% margin): the rest of the sweep is empty
        ax.set_xlim(table.fr.min() / 1.05 / 1e9, table.fr.max() * 1.05 / 1e9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig

################################################################################
#################################### Export ####################################
################################################################################
def export_vna_report(root, out_dir, k = None, dpi = 150):
    """
    Write the plot, the info text, and the resonator CSV of a screened VNA
    sweep to a directory (see the module description).

    Parameters:
    root (zarr.Group, str, or os.PathLike): the store (root or path; may
        be zipped).
    out_dir (str or os.PathLike): output directory (created if needed).
    k (int or None): VNA sweep index. None (default): the latest with a
        full sweep fit.
    dpi (int): plot resolution. Default 150.

    Returns:
    paths (dict): ``plot``, ``info``, ``csv``: the files written.

    Raises:
    KeyError: if the sweep has no full sweep fit (run the
        resonance-finding step first).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    root = dl.open_store(root)
    if k is None:
        done = [j for j in dl.list_indices(root, 'vna')
                if 'full_sweep_fit' in root[dl.set_name(j, 'vna')]]
        if not done:
            raise KeyError('no VNA sweep has a full sweep fit yet')
        k = done[-1]
    grp = root[dl.set_name(k, 'vna')]
    if 'full_sweep_fit' not in grp:
        raise KeyError(f'vna{k} has no full_sweep_fit: find and fit its '
                       'resonances first')
    fit = grp['full_sweep_fit']
    table = resonator_table(fit)
    device = str(root.attrs.get('device', 'device'))
    stem = os.path.join(os.path.expanduser(str(out_dir)),
                        f'{device}_vna{k}')
    os.makedirs(os.path.dirname(stem), exist_ok = True)
    f, z = dl.load_vna_sweep(root, k)
    gf, gz = fit['gain/f'][:], fit['gain/gain'][:]
    gain = np.interp(f, gf, gz.real) + 1j * np.interp(f, gf, gz.imag)
    checked = grp.attrs.get('fres_manual_checked')
    title = (f'{device}: vna{k}, {len(table)} resonances'
             + ('' if checked is not False else ' (not checked by hand)'))
    backend = matplotlib.get_backend()
    fig = plot_vna_results(f, z / gain, table, title)
    fig.savefig(stem + '.png', dpi = dpi)
    plt.close(fig)
    if backend != matplotlib.get_backend():
        matplotlib.use(backend)
    with open(stem + '_info.txt', 'w', encoding = 'utf-8') as fh:
        fh.write(sweep_info_text(root, k, table))
    table.to_csv(stem + '_resonators.csv', index = False)
    return dict(plot = stem + '.png', info = stem + '_info.txt',
                csv = stem + '_resonators.csv')
