import time
import numpy as np
import zarr
from datetime import datetime
from typing import TYPE_CHECKING
from tqdm.auto import tqdm
from citkid.res.fres import update_fres
from . import util
from .. import daq_layout as dl
from ..zarr_util import set_name

# Channels per module used by full-band sweeps
VNA_NCHANNELS = 1024
# Boundary between the first and second Nyquist zones, Hz
NYQUIST_FREQ = 2.5e9
# VNA sweep ranges by Nyquist zone ('1', '2', or 'both'), Hz
VNA_RANGES = {'1': (100e6, 2.5e9), '2': (2.5e9, 5e9), 'both': (100e6, 5e9)}
# Default VNA df/f: 25 points per linewidth at Qr 2e4 (``vna_step``)
VNA_STEP = 2e-6
# Guard band: VNA sweeps never take data within this of 2.5 GHz, where the
# point nearest the Nyquist frequency glitches, Hz
NYQUIST_GUARD = 10e6

if TYPE_CHECKING:
    from . import instrument as inst

async def target_sweep(
    crs, fres, ares, qres, res_idxs, grp, ch_map = None, gain_span_factor = 10,
    npoints_fine = 500, npoints_gain = 50, npoints_rough = None, nsamps = 100,
    fres_update_method = 'spacing', verbose = True,
):
    """
    Execute a target sweep procedure on the CRS instrument, consisting of
    a rough sweep, gain sweep, and fine sweep, saving data to the provided Zarr
    group. The tone frequencies are updated after the rough sweep according to 
    the specified method. Any of the sweeps can be disabled by setting the 
    corresponding npoints parameter to None.

    Parameters:
    crs (CRS): initialized CRS instrument class.
    fres (array-like float64): array of resonant frequencies in Hz.
    ares (array-like float64): array of tone powers in dBm.
    qres (array-like float64): array of span factors for cutting out of adjacent
        datasets. Resonances should span fres / qres.
    res_idxs (array-like int32): Array of resonator indices. Negative indices
        mark calibration tones, which are not updated after the rough sweep.
    grp (zarr.Group): Zarr group to which data is saved.
    ch_map (dict or None): Channel mapping dictionary. If None, crs.ch_map is 
        generated during the first sweep (rough or gain). 
    gain_span_factor (float): gain span is (gain_span_factor * fine_span).
    npoints_fine (int or None): number of points per resonator in the 
        fine sweep.
    npoints_gain (int or None): number of points per resonator in the 
        gain sweep.
    npoints_rough (int or None): number of points per resonator in the 
        rough sweep.
    nsamps (int): number of samples to average over per sweep point. 
    fres_update_method (str): method for updating the tone frequencies after
        the rough sweep, if npoints_rough is not None. Can be 'spacing'
        (default), 'distance', or 'minS21'. See
        ``citkid.res.fres.update_fres``.
    verbose (bool): If True, displays progress bars while taking data.

    Returns:
    fres (np.ndarray, float64): updated resonant frequencies after rough sweep.
    ch_map (dict): channel mapping dictionary after sweeps.
    """
    # Input validation 
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        crs, fres, ares, qres, res_idxs, grp, gain_span_factor, npoints_fine, 
        npoints_gain, npoints_rough, nsamps, fres_update_method, verbose
    )

    # Save input data. System configuration is written first, since it
    # checks its own attribute conflicts
    util.write_system_cfg_to_zarr(crs, grp)
    grp.create_array(name = 'ares', data = ares)
    grp.create_array(name = 'qres', data = qres)
    grp.create_array(name = 'res_idxs', data = res_idxs)
    grp.attrs['nsamps'] = nsamps
    
    if npoints_rough is not None:
        # Rough sweep 
        grpr = grp.require_group('rough_sweep')
        fres_rough = fres.copy()
        grpr.attrs['timestamp'] = datetime.now().strftime('%Y%m%d,%H:%M:%S')
        f_rough, z_rough = await crs.sweep_qres(
                fres_rough,
                ares,
                qres,
                npoints = npoints_rough,
                nsamps = nsamps,
                ch_map = ch_map,
                acq_cfg_grp = grpr,
                verbose = verbose,
                pbar_description = 'Rough sweep',
            )
        
        # Save sweep data and fres_rough
        _save_sweep_data(grpr, '', f_rough, z_rough)
        grpr.create_array(name = 'fres', data = fres_rough)

        # Save fres update method
        grpr.attrs['fres_update_method'] = fres_update_method

        # Assign ch_map. Each sweep clears the session's channels before
        # writing tones and keeps crs.ch_map afterwards, so this is the map
        # of the sweep just taken
        ch_map = crs.ch_map

        # Update fres based on rough sweep. Calibration tones (res_idxs < 0)
        # are not updated
        fres = update_fres(
            f_rough, z_rough, fres, qres, res_idxs,
            method = fres_update_method
            )
        # Some assertions until update_fres is well-tested
        assert isinstance(fres, np.ndarray)
        assert fres.shape == res_idxs.shape
        assert fres.dtype == np.float64
        
    # save fres (after potential update) 
    grp.create_array(name = f'fres', data = fres)

    # Gain sweep 
    if npoints_gain is not None:
        grpg = grp.require_group('gain_sweep')
        grpg.attrs['timestamp'] = datetime.now().strftime('%Y%m%d,%H:%M:%S')
        f_gain, z_gain = await crs.sweep_qres(
                fres,
                ares,
                qres / gain_span_factor,
                npoints = npoints_gain,
                nsamps = nsamps,
                ch_map = ch_map,
                acq_cfg_grp = grpg,
                verbose = verbose,
                pbar_description = 'Gain sweep',
            )
        _save_sweep_data(grpg, '', f_gain, z_gain)
        # If ch_map was determined within the gain sweep, 
        # # save here so fine sweep is consistent
        ch_map = crs.ch_map 

    # Fine sweep
    if npoints_fine is not None:
        grpf = grp.require_group('fine_sweep')
        grpf.attrs['timestamp'] = datetime.now().strftime('%Y%m%d,%H:%M:%S')
        f_fine, z_fine = await crs.sweep_qres(
                fres,
                ares,
                qres,
                npoints = npoints_fine,
                nsamps = nsamps,
                ch_map = ch_map,
                acq_cfg_grp = grpf,
                verbose = verbose,
                pbar_description = 'Fine sweep',
            )
        _save_sweep_data(grpf, '', f_fine, z_fine)
        # If ch_map was determined within the fine sweep, return it so later
        # sweeps (e.g. power_sweep sets) use the same channels
        ch_map = crs.ch_map

    # return updated fres and ch_map
    return fres, ch_map

async def power_sweep(
    crs, fres, aress, qres, res_idxs, root, ch_map = None, shuffle = False,
    seed = None, verbose = True, **target_sweep_kwargs
):
    """
    Execute a target sweep at each of a series of tone powers. Sweep i is
    saved to the group root/set{i} (see ``citkid.zarr_util.set_name``), in
    the target_sweep format.

    The channel map from the first sweep is reused for every later sweep, so
    tones stay on the same module and channel. If the rough sweep is enabled
    (npoints_rough is not None), each sweep starts from the fres found by the
    previous one.

    Parameters:
    crs (CRS): initialized CRS instrument class.
    fres (array-like float64): array of resonant frequencies in Hz.
    aress (array-like float64): tone powers in dBm. Either 1D, of shape
        (n_sets,), to sweep every tone at the same powers, or 2D, of shape
        (n_sets, ntones), to give each tone its own powers.
    qres (array-like float64): array of span factors. See target_sweep.
    res_idxs (array-like int32): array of resonator indices. Negative indices
        mark calibration tones. See target_sweep.
    root (zarr.Group): Zarr group to which the set groups are saved. It must
        not already contain any of the set groups.
    ch_map (dict or None): Channel mapping dictionary for the first sweep. If
        None, it is generated during the first sweep.
    shuffle (bool): If True, each tone's powers are randomly reordered across
        the sets, so that set i doesn't sweep every tone at the same power.
        The powers actually used are saved in each set's 'ares'.
    seed (int or None): random seed for shuffle.
    verbose (bool): If True, displays progress bars while taking data.
    **target_sweep_kwargs: other target_sweep parameters, e.g. npoints_fine,
        npoints_gain, npoints_rough, nsamps, gain_span_factor,
        fres_update_method.

    Returns:
    fres (np.ndarray, float64): resonant frequencies after the last sweep.
    ch_map (dict): channel mapping dictionary after the sweeps.
    """
    fres = np.asarray(fres, dtype = np.float64)
    aress = np.asarray(aress, dtype = np.float64)
    if fres.ndim != 1:
        raise ValueError("fres must be a 1D array.")
    if aress.ndim == 1:
        aress = np.repeat(aress[:, np.newaxis], len(fres), axis = 1)
    if aress.ndim != 2 or aress.shape[1] != len(fres) or not len(aress):
        raise ValueError(
            "aress must have shape (n_sets,) or (n_sets, len(fres)).")
    if not isinstance(root, zarr.core.group.Group):
        raise TypeError("root must be a zarr Group instance.")
    names = [set_name(set_idx) for set_idx in range(len(aress))]
    existing = [name for name in names if name in root]
    if existing:
        raise ValueError(f"root already contains {existing}.")
    if shuffle:
        rng = np.random.default_rng(seed)
        aress = rng.permuted(aress, axis = 0)

    pbar = tqdm(names, desc = 'Power sweep', leave = False,
                disable = not verbose)
    for name, ares in zip(pbar, aress):
        fres, ch_map = await target_sweep(
            crs, fres, ares, qres, res_idxs, root.require_group(name),
            ch_map = ch_map, verbose = verbose, **target_sweep_kwargs
        )
    return fres, ch_map

################################################################################
################################# VNA sweeps ###################################
################################################################################
def vna_step(qr_median = 2e4, pts_per_lw = 25.):
    """
    Return the fractional frequency step df/f of a VNA sweep that samples
    a resonance of the expected median Qr with ``pts_per_lw`` points per
    linewidth.

    On real sweeps, the resonance finders work best with 10-80 points per
    linewidth for a typical resonance and lose resonances below about 5;
    the CNN finder averages data finer than df/f = 1.5e-6 (30 points per
    linewidth at Qr 2e4) down to about that, so a finer step costs time
    without helping it. Qr within a sweep typically spreads by a factor of
    about 2 (0.3 dex), so the default 25 points at the median keeps
    resonances up to 5 times the median Qr at 5 or more. The defaults give
    df/f = 2e-6.

    Parameters:
    qr_median (float): expected median Qr. Default 2e4.
    pts_per_lw (float): points per linewidth at ``qr_median``. Default 25.

    Returns:
    step (float): df/f.

    Raises:
    ValueError: if an input is not positive.
    """
    if not qr_median > 0 or not pts_per_lw > 0:
        raise ValueError('qr_median and pts_per_lw must be positive')
    return 1. / (float(qr_median) * float(pts_per_lw))

def vna_range(nyquist = '1', fmin = None, fmax = None):
    """
    Return the frequency range of a VNA sweep over one or both Nyquist
    zones.

    Parameters:
    nyquist (str): '1' (default, 100-2500 MHz), '2' (2500-5000 MHz), or
        'both' (100-5000 MHz).
    fmin, fmax (float or None): override the zone's lower or upper edge,
        in Hz. Default None (the zone's).

    Returns:
    fmin, fmax (float): range in Hz.

    Raises:
    ValueError: if ``nyquist`` is not an option.
    """
    if nyquist not in VNA_RANGES:
        raise ValueError(f'nyquist must be one of {list(VNA_RANGES)}')
    lo, hi = VNA_RANGES[nyquist]
    return (lo if fmin is None else float(fmin),
            hi if fmax is None else float(fmax))

def _vna_segments(fmin, fmax, guard = NYQUIST_GUARD):
    """
    Split a VNA range at the Nyquist frequency, keeping ``guard`` clear of
    it on both sides.

    Parameters:
    fmin, fmax (float): range in Hz.
    guard (float): guard band around 2.5 GHz in Hz. Default
        ``NYQUIST_GUARD`` (10 MHz).

    Returns:
    segments (list of tuple): (lo, hi, Nyquist zone) of each non-empty
        part, in frequency order.

    Raises:
    ValueError: if the range lies entirely within the guard band.
    """
    segments = []
    if fmin < NYQUIST_FREQ:
        segments.append((fmin, min(fmax, NYQUIST_FREQ - guard), 1))
    if fmax > NYQUIST_FREQ:
        segments.append((max(fmin, NYQUIST_FREQ + guard), fmax, 2))
    segments = [seg for seg in segments if seg[0] < seg[1]]
    if not segments:
        raise ValueError(
            f'the range {fmin / 1e6:g}-{fmax / 1e6:g} MHz lies within '
            f'{guard / 1e6:g} MHz of the Nyquist frequency '
            f'({NYQUIST_FREQ / 1e6:g} MHz), where nothing is swept')
    return segments

def _segment_bands(lo, hi, step, npc, bw, max_channels, from_top = False):
    """
    Lay out one Nyquist-zone segment of a geometric VNA grid in bands.

    The segment's grid is lo * exp(i * delta), i = 0 ... n - 1, with n a
    multiple of ``npc`` and delta <= log(1 + step). Starting from one end,
    each band takes as many channels of ``npc`` consecutive points as fit
    in ``bw`` and ``max_channels``, so the last band, at the other end, may
    be small.

    Parameters:
    lo, hi (float): segment range in Hz (hi excluded).
    step (float): largest df/f.
    npc (int): points per channel.
    bw (float): largest band span in Hz.
    max_channels (int): channels per module.
    from_top (bool): If True, start from the top, else (default) from the
        bottom.

    Returns:
    if ``npc`` points fit in ``bw`` everywhere:
        n (int): number of points.
        delta (float): log step.
        bands (list of tuple): (first point index, channels) of each band.
    else:
        None
    """
    n = int(np.ceil(np.log(hi / lo) / np.log1p(step) / npc - 1e-9)) * npc
    delta = np.log(hi / lo) / n
    bands = []
    if not from_top:
        i = 0
        while i < n:
            f0 = lo * np.exp(i * delta)
            # Points from f0 up that fit within bw
            nfit = int(np.floor(np.log1p(bw / f0) / delta * (1 - 1e-9))) + 1
            nch = min(max_channels, nfit // npc, (n - i) // npc)
            if nch < 1:
                return None
            bands.append((i, nch))
            i += nch * npc
        return n, delta, bands
    j = n
    while j > 0:
        f1 = lo * np.exp((j - 1) * delta)
        # Points from f1 down that fit within bw
        nfit = j if bw >= f1 else \
            int(np.floor(-np.log1p(-bw / f1) / delta * (1 - 1e-9))) + 1
        nch = min(max_channels, nfit // npc, j // npc)
        if nch < 1:
            return None
        j -= nch * npc
        bands.append((j, nch))
    return n, delta, bands[::-1]

def plan_vna_sweep(
    fmin, fmax, bw, module_idxs, step = VNA_STEP, npoints_per_ch = None,
    pass_overhead = 50, nyquist_guard = NYQUIST_GUARD,
    max_channels = VNA_NCHANNELS
):
    """
    Plan a VNA sweep on a geometric frequency grid (constant df/f, so
    every resonance gets the same points per linewidth for its Qr).

    The range is split at the Nyquist frequency (2.5 GHz) if it crosses
    it, and nothing within ``nyquist_guard`` of 2.5 GHz is swept (the
    point nearest it glitches). Each part's grid has a step of at most
    ``step`` and is cut into
    channels of ``npoints_per_ch`` consecutive points; channels are grouped
    into bands of up to ``max_channels`` channels spanning at most ``bw``,
    each swept around an NCO at its center. Low bands are limited by the
    channel count and high bands by ``bw``, so bands get narrower toward
    low frequencies and high bands use fewer channels. Bands go to the
    modules in turn: band i is swept by module i % len(module_idxs) in
    pass i // len(module_idxs). Bands are laid out away from 2.5 GHz (up
    from the bottom in the first zone, down from the top in the second).

    With ``npoints_per_ch`` None, it is chosen to minimize the estimated
    sweep time, passes x (npoints_per_ch + ``pass_overhead``) in sweep
    steps: fewer points per channel use the channels better but need more
    bands, and each pass costs an NCO change and adds a seam.

    Parameters:
    fmin, fmax (float): frequency range in Hz (fmax excluded).
    bw (float): module bandwidth in Hz (``CRS.bw``).
    module_idxs (list of int): modules to sweep with.
    step (float): largest df/f (``vna_step``). Default ``VNA_STEP``
        (2e-6).
    npoints_per_ch (int or None): points per channel. None (default):
        chosen as above.
    pass_overhead (float): cost of a pass in sweep steps, for choosing
        ``npoints_per_ch``. Default 50.
    nyquist_guard (float): no data is taken within this of 2.5 GHz, in
        Hz. Default ``NYQUIST_GUARD`` (10 MHz).
    max_channels (int): channels per module. Default 1024.

    Returns:
    bands (dict): band arrays in frequency order: ``chunk_edges`` ((nb,
        2), first and last frequency in Hz), ``chunk_ncos`` ((nb,), Hz),
        ``chunk_modules``, ``chunk_passes``, ``chunk_zones`` (Nyquist
        zone), ``chunk_nchannels`` (channels used), each (nb,).
    freqs (list of np.array): each band's frequencies, (channels,
        npoints_per_ch), ascending along rows and down the columns.
    npoints_per_ch (int): points per channel.

    Raises:
    ValueError: if the inputs are invalid, the range lies within the
        guard band, or ``npoints_per_ch`` points span more than ``bw``.
    """
    fmin, fmax, bw, step = float(fmin), float(fmax), float(bw), float(step)
    module_idxs = [int(m) for m in module_idxs]
    if not 0 < fmin < fmax:
        raise ValueError('need 0 < fmin < fmax')
    if bw <= 0 or not len(module_idxs):
        raise ValueError('bw must be positive and module_idxs not empty')
    if not 0 < step < 1:
        raise ValueError('step must be in (0, 1)')
    if not float(nyquist_guard) >= 0:
        raise ValueError('nyquist_guard must be non-negative')
    nm = len(module_idxs)
    # Lay out each segment away from 2.5 GHz: the second zone's segment
    # that starts at the guard band is filled down from the top
    segments = [(lo, hi, zone, zone == 2 and lo > fmin)
                for lo, hi, zone in
                _vna_segments(fmin, fmax, float(nyquist_guard))]
    # Margin for rounding and the CRS dithering (50 Hz, inward at the edges)
    bw_eff = bw * (1 - 1e-6)

    def plan(npc):
        """
        Lay out every segment with ``npc`` points per channel.

        Parameters:
        npc (int): points per channel.

        Returns:
        plan (tuple or None): cost, ``npc``, and the ``_segment_bands``
            output per segment; None if ``npc`` points span more than
            ``bw``.
        """
        lay = [_segment_bands(lo, hi, step, npc, bw_eff, max_channels,
                              from_top)
               for lo, hi, _, from_top in segments]
        if any(o is None for o in lay):
            return None
        npass = -(-sum(len(b) for _, _, b in lay) // nm)
        return npass * (npc + pass_overhead), npc, lay

    if npoints_per_ch is None:
        # More points per channel than fill the lowest bw-wide band can't
        # reduce the band count further
        npc_max = max(int(np.log1p(bw / lo) / np.log1p(step) / max_channels)
                      for lo, _, _, _ in segments) + 1
        candidates = range(1, npc_max + 1)
    else:
        if not isinstance(npoints_per_ch, (int, np.integer)) or \
                npoints_per_ch <= 0:
            raise ValueError('npoints_per_ch must be a positive integer')
        candidates = [int(npoints_per_ch)]
    best = min((p for p in (plan(npc) for npc in candidates)
                if p is not None), default = None, key = lambda p: p[0])
    if best is None:
        raise ValueError(f'{candidates[0]} points per channel at df/f '
                         f'{step:g} span more than the module bandwidth')
    _, npc, lay = best
    freqs, edges, zones, nchs = [], [], [], []
    for (lo, _, zone, _), (_, delta, segment) in zip(segments, lay):
        for i0, nch in segment:
            fb = lo * np.exp((i0 + np.arange(nch * npc)) * delta)
            freqs.append(fb.reshape(nch, npc))
            edges.append((fb[0], fb[-1]))
            zones.append(zone)
            nchs.append(nch)
    idx = np.arange(len(freqs))
    edges = np.array(edges)
    bands = dict(
        chunk_edges = edges, chunk_ncos = edges.mean(axis = 1),
        chunk_modules = np.array(module_idxs)[idx % nm],
        chunk_passes = idx // nm, chunk_zones = np.array(zones),
        chunk_nchannels = np.array(nchs))
    return bands, freqs, npc

async def _set_nyquist_zones(crs, zones):
    """
    Set modules' Nyquist zones, keeping their other module settings.

    Parameters:
    crs (CRS): configured CRS instrument.
    zones (dict): module: Nyquist zone (1 or 2). Modules already in their
        zone are left alone.
    """
    for m, zone in zones.items():
        cfg = crs.module_cfg[m]
        if int(cfg['nyquist_zone']) == int(zone):
            continue
        await crs.configure_modules(
            full_scale_dbm = cfg['full_scale_dbm'],
            adc_attenuation_db = int(round(cfg['adc_attenuation_db'])),
            nyquist_zone = int(zone), cable_length_m = cfg['cable_length_m'],
            adc_autocal = bool(cfg['adc_autocal']),
            adc_calibration_mode = cfg['adc_calibration_mode'],
            module_idxs = [int(m)])

async def vna_sweep(
    crs, fmin, fmax, amplitude, nsamps, step = VNA_STEP,
    npoints_per_ch = None, pass_overhead = 50,
    nyquist_guard = NYQUIST_GUARD, verbose = True, progress = None
):
    """
    Sweep a frequency range on a geometric grid with this session's
    modules, in as many passes as needed, and stitch the result into one
    sweep.

    The grid and bands come from ``plan_vna_sweep`` with the session's
    modules in the active bank and the current module bandwidth
    (``crs.bw``, 600 MHz with extended bandwidth). Each pass sets the
    Nyquist zone and NCO of its modules and sweeps their bands at once
    (``CRS.sweep``); every channel sweeps downward. The CRS dithers each
    frequency by up to 50 Hz, so the result is geometric up to that. The
    modules' Nyquist zones are restored at the end.

    Parameters:
    crs (CRS): configured CRS instrument.
    fmin, fmax (float): frequency range in Hz (``vna_range``).
    amplitude (float): tone power in dBm, the same for every channel.
    nsamps (int): samples averaged per point.
    step, npoints_per_ch, pass_overhead, nyquist_guard: see
        ``plan_vna_sweep``.
    verbose (bool): If True (default), print the plan and show progress
        bars.
    progress (callable or None): if given, called as ``progress(done,
        total)`` with the number of points taken so far over the whole
        sweep and the total, at the start of each pass and after every
        frequency step (``CRS.sweep``). Independent of ``verbose``.

    Returns:
    f (np.array): frequencies in Hz (dithered), ascending.
    z (np.array): complex S21.
    bands (dict): ``plan_vna_sweep`` band arrays plus ``chunk_times``
        (start of each band's pass, Unix seconds).

    Raises:
    ValueError: if the inputs are invalid.
    """
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None')
    bands, freqs, npc = plan_vna_sweep(
        fmin, fmax, crs.bw, crs.active_module_idxs, step = step,
        npoints_per_ch = npoints_per_ch, pass_overhead = pass_overhead,
        nyquist_guard = nyquist_guard)
    passes = np.unique(bands['chunk_passes'])
    npoints = sum(fb.size for fb in freqs)
    bands['chunk_times'] = np.zeros(len(freqs))
    if verbose:
        print(f'VNA sweep: {npoints} points, {len(freqs)} bands in '
              f'{len(passes)} passes, {npc} points per channel')
    zones0 = {int(m): int(crs.module_cfg[m]['nyquist_zone'])
              for m in np.unique(bands['chunk_modules'])}
    fs, zs = [], []
    # Points taken before the current pass, for progress
    done_before = 0
    try:
        for p in passes:
            idx = np.flatnonzero(bands['chunk_passes'] == p)
            modules = [int(m) for m in bands['chunk_modules'][idx]]
            await _set_nyquist_zones(crs, {
                m: int(zone) for m, zone in
                zip(modules, bands['chunk_zones'][idx])})
            await crs.set_nco({m: float(nco) for m, nco in
                               zip(modules, bands['chunk_ncos'][idx])},
                              verbose = False)
            # Every channel sweeps downward
            rows = np.concatenate([freqs[i][:, ::-1] for i in idx])
            starts = np.cumsum([0] + [len(freqs[i]) for i in idx])
            ch_map = {m: list(range(starts[k], starts[k + 1]))
                      for k, m in enumerate(modules)}
            bands['chunk_times'][idx] = time.time()
            pass_progress = None
            if progress is not None:
                offset = done_before

                def pass_progress(done, total, offset = offset):
                    """
                    Report this pass's progress as sweep progress.

                    Parameters:
                    done, total (int): points taken and total in the pass.
                    offset (int): points taken before the pass.
                    """
                    progress(offset + done, npoints)
            f, z = await crs.sweep(
                rows, np.full(len(rows), float(amplitude)), nsamps,
                ch_map = ch_map, allow_missing = False, verbose = verbose,
                pbar_description = f'VNA pass {p + 1}/{len(passes)}',
                progress = pass_progress)
            fs.append(np.asarray(f).ravel())
            zs.append(np.asarray(z).ravel())
            done_before += rows.size
    finally:
        await _set_nyquist_zones(crs, zones0)
    f, z = np.concatenate(fs), np.concatenate(zs)
    o = np.argsort(f, kind = 'stable')
    return f[o], z[o], bands

async def vna_sweep_to_store(
    crs, root, fmin, fmax, amplitude, nsamps, step = VNA_STEP,
    npoints_per_ch = None, pass_overhead = 50,
    nyquist_guard = NYQUIST_GUARD, verbose = True
):
    """
    Take a VNA sweep (``vna_sweep``) and save it as the next
    ``vna/sweep{k}`` of a device store (``citkid.daq_layout``).

    Saves ``f``, ``z``, the band arrays, the sweep settings and the
    per-acquisition CRS settings (attributes), and the CRS settings it was
    taken with in ``crs_config`` (as a tuning's, without the NCOs and
    Nyquist zones, which are in ``chunk_ncos`` and ``chunk_zones``). The
    store's CRS identity (serial number, firmware, rfmux) is recorded on
    the first acquisition and checked on later ones. If the sweep fails,
    the new group is removed, so no partial sweep is left.

    Parameters:
    crs (CRS): configured CRS instrument.
    root (zarr.Group): device store root, writable
        (``daq_layout.create_store``).
    fmin, fmax, amplitude, nsamps, step, npoints_per_ch, pass_overhead,
        nyquist_guard, verbose: see ``vna_sweep``.

    Returns:
    k (int): index of the new sweep.
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21.

    Raises:
    RuntimeError: if the CRS differs from the store's.
    """
    dl.check_store(root)
    dl.check_crs_identity(root, crs)
    vna = dl.require_group(root, 'vna', 'raw')
    settings = dict(
        amplitude = float(amplitude), nsamps = int(nsamps),
        step = float(step), fmin = float(fmin), fmax = float(fmax),
        spacing = 'geometric', downward = True,
        acquisition = dl.acquisition_settings(crs))
    grp, k = dl.new_indexed_group(vna, 'sweep', 'raw', attrs = settings)
    try:
        f, z, bands = await vna_sweep(
            crs, fmin, fmax, amplitude, nsamps, step = step,
            npoints_per_ch = npoints_per_ch, pass_overhead = pass_overhead,
            nyquist_guard = nyquist_guard, verbose = verbose)
        config = dl.tune_config(crs)
        del config['ncos']
        for cfg in config['module_settings'].values():
            cfg.pop('nyquist_zone', None)
        dl.create_group(grp, 'crs_config', 'raw',
                        attrs = dict(config = config))
        grp.attrs['npoints_per_ch'] = int(
            len(f) // bands['chunk_nchannels'].sum())
        dl.save_vna_sweep(grp, f, z, **bands)
    except BaseException:
        del vna[set_name(k, 'sweep')]
        raise
    return k, f, z

################################################################################
########################### Zarr saving helper #################################
################################################################################
def _save_sweep_data(grp, prefix, f, z):
    """
    Save sweep data to zarr group.
    
    Parameters:
    grp (zarr.Group): zarr group to save data to.
    prefix (str): prefix for dataset names.
    f (np.ndarray, float64): frequency data.
    z (np.ndarray, complex128): s21 data.
    
    Returns:
    None
    """
    # Input validation
    f = np.asarray(f, dtype = np.float64) 
    z = np.asarray(z, dtype = np.complex128)
    if f.shape != z.shape:
        raise ValueError("f and z must have the same shape.")
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.") 
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string.") 
    
    # Save to group
    if prefix:
        prefix += '_'
    grp.create_array(name = f'{prefix}f', 
                     data = f, 
                     chunks = (1, f.shape[1]),
                     shards = f.shape
                    )
    grp.create_array(name = f'{prefix}z', 
                     data = z, 
                     chunks = (1, z.shape[1]),
                     shards = z.shape
                     )
    
################################################################################
########################### Input validation helpers ###########################
################################################################################
def _validate_target_sweep_inputs(
    crs, fres, ares, qres, res_idxs, grp, gain_span_factor, npoints_fine,
    npoints_gain, npoints_rough, nsamps, fres_update_method, verbose,
):
    """
    Validate inputs to target_sweep function.
    
    Parameters:
    see target_sweep function.

    Returns:
    fres, ares, qres, res_idxs: validated and converted inputs.
    """
    # Check if it's a CRS instance by verifying class name
    if type(crs).__name__ != 'CRS' and type(crs).__name__ != 'DummyCRS':
        raise TypeError("crs must be an instance of CRS class.") 
    # Validate array-like inputs
    fres = np.asarray(fres, dtype = np.float64)
    ares = np.asarray(ares, dtype = np.float64)
    qres = np.asarray(qres, dtype = np.float64)
    res_idxs = np.asarray(res_idxs, dtype = np.int32)
    if fres.ndim != 1:
        raise ValueError("fres must be a 1D array.")
    if not (fres.shape == ares.shape == qres.shape == res_idxs.shape):
        raise ValueError(
            "fres, ares, qres, and res_idxs must have the same shape."
        )
    # gain_span_factor
    if not isinstance(gain_span_factor, (int, float, np.integer, np.floating)) \
        or gain_span_factor <= 1:
        raise ValueError("gain_span_factor must be a number > 1.") 
    # npoints_fine, npoints_gain, npoints_rough
    if npoints_fine is not None and \
        (not isinstance(npoints_fine, (int, np.integer)) or npoints_fine <= 0):
        raise ValueError("npoints_fine must be a positive integer.") 
    if npoints_gain is not None and \
        (not isinstance(npoints_gain, (int, np.integer)) or npoints_gain <= 0):
        raise ValueError("npoints_gain must be a positive integer.") 
    if npoints_rough is not None and \
        (not isinstance(npoints_rough, (int, np.integer)) or npoints_rough <=0):
        raise ValueError("npoints_rough must be a positive integer.") 
    # nsamps
    if not isinstance(nsamps, (int, np.integer)) or nsamps <= 0:
        raise ValueError("nsamps must be a positive integer.") 
    # fres_update_method
    if npoints_rough is not None and \
        fres_update_method not in ['distance', 'spacing', 'minS21']:
        msg = "fres_update_method must be 'distance', 'spacing', or 'minS21'."
        raise ValueError(msg)
    # verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")
    
    # Validate zarr group and ensure that datasets do not already exist
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.")
    def zarr_key_exists_check(grp, name):
        """
        Raise an error if a Zarr group already contains a key.

        Parameters:
        grp (zarr.Group): Zarr group to check.
        name (str): key to look for.

        Returns:
        None

        Raises:
        ValueError: if ``name`` is in ``grp``.
        """
        if name in grp:
            raise ValueError(f"Zarr group already contains dataset '{name}'.")
    for name in ['ares', 'qres', 'res_idxs', 'fres']:
        zarr_key_exists_check(grp, name)
    if npoints_rough is not None:
        zarr_key_exists_check(grp, 'rough_sweep')
    if npoints_gain is not None:
        zarr_key_exists_check(grp, 'gain_sweep')
    if npoints_fine is not None:
        zarr_key_exists_check(grp, 'fine_sweep')
    if 'nsamps' in grp.attrs:
        raise ValueError("Zarr group already contains attribute 'nsamps'.")
            
    # Return validated inputs
    return fres, ares, qres, res_idxs