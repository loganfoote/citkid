import inspect
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

class SweepCancelled(Exception):
    """
    Raised by ``vna_sweep`` when its ``cancelled`` callable asks it to stop
    (between points or passes). The board's tones are cleared and the
    modules' Nyquist zones restored before it is raised.
    """

async def target_sweep(
    crs, fres, ares, qres, res_idxs, grp, ch_map = None, gain_span_factor = 10,
    npoints_fine = 500, npoints_gain = 50, npoints_rough = None, nsamps = 100,
    fres_update_method = 'spacing', verbose = True, progress = None,
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
    progress (callable or None): if given, called as ``progress(done,
        total)`` with the points taken so far and the total over the
        rough, gain, and fine sweeps (``ntones`` times the points per
        tone), from (0, total) to (total, total). Default None.

    Returns:
    fres (np.ndarray, float64): updated resonant frequencies after rough sweep.
    ch_map (dict): channel mapping dictionary after sweeps.
    """
    # Input validation 
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        crs, fres, ares, qres, res_idxs, grp, gain_span_factor, npoints_fine, 
        npoints_gain, npoints_rough, nsamps, fres_update_method, verbose
    )
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None')
    # Progress over the three sweeps: each reports its own (done, total);
    # offset them by the points of the sweeps before it
    total = len(fres) * sum(n for n in (npoints_rough, npoints_gain,
                                        npoints_fine) if n is not None)
    before = [0]

    def stage_progress(done, stage_total):
        """
        Report one sweep's progress as progress over all three.

        Parameters:
        done (int): points taken in this sweep.
        stage_total (int): points in this sweep.
        """
        progress(min(before[0] + done, total), total)

    def stage_done(npoints):
        """
        Count a finished sweep's points.

        Parameters:
        npoints (int): its points per tone.
        """
        before[0] += len(fres) * npoints

    # Only given to the sweeps when asked for, so a sweep method without
    # it (e.g. a test double) still works
    report = {} if progress is None else dict(progress = stage_progress)
    if progress is not None:
        progress(0, total)

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
                **report,
            )
        stage_done(npoints_rough)
        
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
                **report,
            )
        stage_done(npoints_gain)
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
                **report,
            )
        stage_done(npoints_fine)
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
############################ Stitched target sweeps ############################
################################################################################
def _merge_passes(dst, parts, n):
    """
    Merge the target sweep groups of several passes into one group, with
    the rows of every per-tone array back in tone order.

    An array is per-tone if its first axis has one entry per tone of its
    pass in every pass; it is assembled with each pass's rows at that
    pass's tone indices. Other arrays are taken from the first pass, except
    the channel maps (``chs_module*``), which only describe one pass (the
    stitched sweep has ``tone_module`` and ``tone_channel`` instead).
    Attributes come from the first pass, without the NCO attributes
    (``nco_module*``; the stitched sweep has ``pass_ncos``); timestamps of
    every pass are kept as ``pass_timestamps``.

    Parameters:
    dst (zarr.Group): destination group, empty.
    parts (list of tuple): (tone indices, pass group) per pass.
    n (int): number of tones.
    """
    groups = [g for _, g in parts]
    attrs = {k: v for k, v in dict(groups[0].attrs).items()
             if not k.startswith('nco_module')}
    if 'timestamp' in attrs:
        attrs['pass_timestamps'] = [g.attrs.get('timestamp')
                                    for g in groups]
    dst.attrs.update(attrs)
    for name in groups[0].array_keys():
        if name.startswith('chs_module'):
            continue
        arrays = [g[name] for g in groups]
        if all(a.ndim >= 1 and a.shape[0] == len(idx)
               for a, (idx, _) in zip(arrays, parts)):
            a0 = arrays[0]
            fill = np.nan if a0.dtype.kind in 'fc' else 0
            full = np.full((n,) + tuple(a0.shape[1:]), fill, dtype = a0.dtype)
            for a, (idx, _) in zip(arrays, parts):
                full[idx] = a[...]
        else:
            full = arrays[0][...]
        dst.create_array(name = name, data = full)
    for name in groups[0].group_keys():
        _merge_passes(dst.require_group(name),
                      [(idx, g[name]) for idx, g in parts], n)

def points_per_tone(target_sweep_kwargs):
    """
    Return the points per tone of a target sweep with these settings (the
    rough, gain, and fine sweeps together), for progress reports.

    Parameters:
    target_sweep_kwargs (dict): ``target_sweep`` keyword arguments.

    Returns:
    npoints (int): points per tone.
    """
    defaults = inspect.signature(target_sweep).parameters
    total = 0
    for key in ('npoints_rough', 'npoints_gain', 'npoints_fine'):
        n = target_sweep_kwargs.get(key, defaults[key].default)
        if n is not None:
            total += int(n)
    return total

async def stitched_target_sweep(
    crs, fres, ares, qres, res_idxs, tone_set, tone_module, ncos, grp,
    verbose = True, progress = None, **target_sweep_kwargs
):
    """
    Take a target sweep of tones split into tone sets, one pass per set,
    and save it as one target sweep.

    Each pass sets the NCOs of its tone set (``ncos``), takes a
    ``target_sweep`` of the set's tones with each tone on its module
    (``tone_module``), and keeps the result in memory. When every pass is
    done, the passes are merged into ``grp`` in the input tone order
    (``_merge_passes``), so the group has the usual target sweep layout
    (``fres``, ``ares``, ``qres``, ``res_idxs``, ``gain_sweep/``,
    ``fine_sweep/``, ...) and loads like a single sweep. It also gets
    ``tone_set``, ``tone_module``, ``tone_channel`` (channel within the
    module, per tone), and the attributes ``pass_ncos`` (per pass, {module:
    NCO in Hz}), ``pass_times`` (start of each pass, Unix seconds), and
    ``complete`` (True). If a pass fails, ``grp`` is left as it was.

    Parameters:
    crs (CRS): configured CRS instrument.
    fres, ares, qres (array-like float): tone frequencies (Hz), powers
        (dBm), and span factors.
    res_idxs (array-like int): resonator indices (negative for calibration
        tones).
    tone_set (array-like int): tone set of each tone (index into ``ncos``).
    tone_module (array-like int): module of each tone.
    ncos (list of dict): per tone set, {module: NCO in Hz} (keys may be
        strings, as saved in JSON).
    grp (zarr.Group): destination group, empty.
    verbose (bool): If True (default), show progress bars.
    progress (callable or None): if given, called as ``progress(done,
        total)`` with the points taken so far over every pass (see
        ``target_sweep``). Default None.
    **target_sweep_kwargs: other ``target_sweep`` parameters
        (``npoints_fine``, ``npoints_gain``, ``npoints_rough``, ``nsamps``,
        ``gain_span_factor``, ``fres_update_method``).

    Returns:
    fres (np.array): tone frequencies after the sweep (updated by the rough
        sweep, if any), in input order.

    Raises:
    ValueError: if the inputs are inconsistent or ``grp`` isn't empty.
    """
    fres = np.asarray(fres, dtype = np.float64)
    n = len(fres)
    ares = np.broadcast_to(np.asarray(ares, dtype = np.float64), (n,))
    qres = np.broadcast_to(np.asarray(qres, dtype = np.float64), (n,))
    res_idxs = np.asarray(res_idxs, dtype = np.int32)
    tone_set = np.asarray(tone_set, dtype = np.int32)
    tone_module = np.asarray(tone_module, dtype = np.int32)
    if fres.ndim != 1 or any(v.shape != (n,) for v in
                             (res_idxs, tone_set, tone_module)):
        raise ValueError('fres, res_idxs, tone_set and tone_module must be '
                         '1D arrays of the same length')
    if not isinstance(grp, zarr.core.group.Group) or len(list(grp.keys())):
        raise ValueError('grp must be an empty zarr group')
    ncos = [{int(m): float(f) for m, f in s.items()} for s in ncos]
    sets = np.unique(tone_set)
    if len(sets) and (sets[0] < 0 or sets[-1] >= len(ncos)):
        raise ValueError('tone_set must index ncos')
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None')
    per_tone = points_per_tone(target_sweep_kwargs)
    total, before = n * per_tone, 0
    if progress is not None:
        progress(0, total)
    fres_out = fres.copy()
    tone_channel = np.zeros(n, dtype = np.int32)
    parts, pass_ncos, pass_times = [], [], []
    for p, s in enumerate(sets):
        idx = np.flatnonzero(tone_set == s)
        mods = tone_module[idx]
        missing = sorted(set(mods.tolist()) - set(ncos[s]))
        if missing:
            raise ValueError(f'tone set {s} has no NCO for modules {missing}')
        set_ncos = {int(m): ncos[s][int(m)] for m in np.unique(mods)}
        await crs.set_nco(set_ncos, verbose = False)
        ch_map = {}
        for m in set_ncos:
            rows = np.flatnonzero(mods == m)
            ch_map[m] = rows.tolist()
            tone_channel[idx[rows]] = np.arange(len(rows))
        tmp = zarr.open_group(zarr.storage.MemoryStore(), mode = 'w')
        pass_times.append(time.time())
        pass_ncos.append({str(m): f for m, f in set_ncos.items()})
        if verbose and len(sets) > 1:
            print(f'Tone set {s} ({p + 1}/{len(sets)}): {len(idx)} tones')
        pass_progress = None
        if progress is not None:
            def pass_progress(done, pass_total, offset = before):
                """
                Report this pass's progress as progress over every pass.

                Parameters:
                done (int): points taken in this pass.
                pass_total (int): points in this pass.
                offset (int): points of the passes before it.
                """
                progress(min(offset + done, total), total)
        fres_out[idx], _ = await target_sweep(
            crs, fres[idx], ares[idx], qres[idx], res_idxs[idx], tmp,
            ch_map = ch_map, verbose = verbose, progress = pass_progress,
            **target_sweep_kwargs)
        before += len(idx) * per_tone
        parts.append((idx, tmp))
    _merge_passes(grp, parts, n)
    for name, value in (('tone_set', tone_set), ('tone_module', tone_module),
                        ('tone_channel', tone_channel)):
        grp.create_array(name = name, data = value)
    grp.attrs.update(dict(pass_ncos = pass_ncos, pass_times = pass_times,
                          complete = True))
    return fres_out

async def stitched_power_sweep(
    crs, fres, aress, qres, res_idxs, tone_set, tone_module, ncos, root,
    shuffle = False, seed = None, verbose = True, **target_sweep_kwargs
):
    """
    Take a stitched target sweep (``stitched_target_sweep``) at each of a
    series of tone powers, saving sweep i as ``root/set{i}``.

    Sets already complete (``attrs['complete']``) are skipped, so an
    interrupted power sweep can be continued; a set left empty by a failed
    pass is taken again. With a rough sweep, each set starts from the
    frequencies found by the previous one.

    Parameters:
    crs (CRS): configured CRS instrument.
    fres (array-like float): tone frequencies in Hz.
    aress (array-like float): powers in dBm, shape (n_sets,) (every tone
        the same) or (n_sets, ntones).
    qres, res_idxs, tone_set, tone_module, ncos: see
        ``stitched_target_sweep``.
    root (zarr.Group): group for the sets.
    shuffle (bool): If True, each tone's powers are randomly reordered
        across the sets (the powers used are saved in each set's
        ``ares``). Default False. Use the same ``seed`` to continue a
        shuffled sweep.
    seed (int or None): random seed for ``shuffle``. Default None.
    verbose (bool): If True (default), show progress bars.
    **target_sweep_kwargs: other ``target_sweep`` parameters.

    Returns:
    fres (np.array): tone frequencies after the last set.

    Raises:
    ValueError: if ``aress`` has the wrong shape, or a set group is not
        empty and not complete.
    """
    fres = np.asarray(fres, dtype = np.float64)
    aress = np.asarray(aress, dtype = np.float64)
    if aress.ndim == 1:
        aress = np.repeat(aress[:, np.newaxis], len(fres), axis = 1)
    if aress.ndim != 2 or aress.shape[1] != len(fres) or not len(aress):
        raise ValueError(
            'aress must have shape (n_sets,) or (n_sets, len(fres))')
    if shuffle:
        aress = np.random.default_rng(seed).permuted(aress, axis = 0)
    pbar = tqdm(range(len(aress)), desc = 'Power sweep', leave = False,
                disable = not verbose)
    for i in pbar:
        name = set_name(i)
        if name in root and root[name].attrs.get('complete'):
            fres = np.asarray(root[name]['fres'][:], dtype = np.float64)
            continue
        if name in root:
            if len(list(root[name].keys())):
                raise ValueError(f'{name} is neither empty nor complete')
            del root[name]
        fres = await stitched_target_sweep(
            crs, fres, aress[i], qres, res_idxs, tone_set, tone_module, ncos,
            root.create_group(name), verbose = verbose, **target_sweep_kwargs)
    return fres

################################################################################
################################### Streaming ##################################
################################################################################
def plan_stream_chunks(tone_module, dec_stage):
    """
    Split one tone set's tones into chunks that can each be streamed at
    once at a decimation stage, without changing NCOs.

    As many modules stream together as the frame rate allows
    (``util.max_stream_tones``), with at most the per-module and total
    tone limits; a module's tones beyond them go to later chunks. At stages
    where every tone fits (e.g. stage 6, or a tone set planned for this
    stage), there is one chunk.

    Parameters:
    tone_module (array-like int): module of each tone of the set.
    dec_stage (int): decimation stage.

    Returns:
    chunks (list of np.array): indices into ``tone_module`` per chunk, in
        order, covering every tone once.

    Raises:
    ValueError: if not even one module can stream at ``dec_stage``.
    """
    tone_module = np.asarray(tone_module, dtype = np.int32)
    mods = sorted(set(tone_module.tolist()))
    if not mods:
        return []
    k = max((n for n in range(1, len(mods) + 1)
             if util.max_stream_tones(dec_stage, n)[0] > 0), default = 0)
    if not k:
        raise ValueError(f'no module can stream at decimation stage '
                         f'{dec_stage} within the verified limits')
    chunks = []
    for g in range(0, len(mods), k):
        group = mods[g:g + k]
        per_module, total = util.max_stream_tones(dec_stage, len(group))
        per_module = min(per_module, total // len(group))
        rows = {m: np.flatnonzero(tone_module == m) for m in group}
        n_rounds = max(-(-len(r) // per_module) for r in rows.values())
        for i in range(n_rounds):
            chunk = np.concatenate([r[i * per_module:(i + 1) * per_module]
                                    for r in rows.values()])
            if len(chunk):
                chunks.append(np.sort(chunk))
    return chunks

async def write_stream_chunk(
    crs, fres, ares, tone_module, set_ncos, grp = None
):
    """
    Write a chunk of one tone set's tones for streaming, each on its
    module, with the set's NCOs (``CRS.write_tones_for_stream``). An NCO
    already at its value is left alone (``CRS.set_nco``), so that tones
    written after the set's calibration sweep measure in the sweep's
    phase frame.

    Parameters:
    crs (CRS): configured CRS instrument.
    fres, ares (array-like): the chunk's tone frequencies (Hz) and powers
        (dBm).
    tone_module (array-like int): module of each tone.
    set_ncos (dict): the tone set's {module: NCO in Hz} (keys may be
        strings).
    grp (zarr.Group or None): write group to save the tones in. Default
        None (nothing saved).

    Returns:
    fres_written (np.array): written (dithered) tone frequencies in Hz.
    """
    tone_module = np.asarray(tone_module, dtype = np.int32)
    set_ncos = {int(m): float(f) for m, f in set_ncos.items()}
    mods = sorted(set(tone_module.tolist()))
    # An NCO already there is left alone (CRS.set_nco): setting one starts
    # a new phase frame, and the stream must stay in its calibration's
    await crs.set_nco({m: set_ncos[m] for m in mods}, verbose = False)
    ch_map = {m: np.flatnonzero(tone_module == m).tolist() for m in mods}
    return await crs.write_tones_for_stream(fres, ares, grp = grp,
                                            ch_map = ch_map)

async def calibrate_tone_set(
    crs, tune, tone_set, status = None, verbose = True, progress = None,
    **target_sweep_kwargs
):
    """
    Take a calibration sweep of one tone set of a tuning and save it as
    the tuning's next ``cal{j}`` (raw; attrs ``tone_set`` and ``tones``,
    plus the sweep's own, e.g. ``nsamps`` and ``complete``).

    A set is calibrated once, and every timestream of it after that uses
    the calibration (``stream_tone_sets`` saves the streams under it), so
    a long sweep is never repeated. The sweep clears the board's tones.

    The calibration is only in frame while its modules' NCOs stay as
    they were for the sweep (``CRS.set_nco``): the ids of those settings
    are saved as attr ``nco_ids`` ({module: id}), and
    ``calibration_in_frame`` compares them with the board's. Another
    tone set's NCO, or another session, puts it out of frame, and a new
    sweep is needed.

    Parameters:
    crs (CRS): configured CRS instrument.
    tune (zarr.Group): the tuning, with ``tones`` and ``crs_config``, in a
        writable store.
    tone_set (int): the tone set.
    status (callable or None): called with a progress message. Default
        None.
    verbose (bool): If True (default), show progress bars.
    progress (callable or None): ``progress(done, total)`` in points (see
        ``stitched_target_sweep``). Default None.
    **target_sweep_kwargs: sweep settings (``target_sweep``:
        ``npoints_fine``, ``npoints_gain``, ``nsamps``, ...).

    Returns:
    grp (zarr.Group): the calibration, ``tune{k}/cal{j}``.

    Raises:
    ValueError: if the tuning has no such tone set.
    """
    say = status or (lambda message: None)
    t = dl.load_tones(tune['tones'])
    ncos = tune['crs_config'].attrs['config']['tone_sets']
    s = int(tone_set)
    idx = np.flatnonzero(t['tone_set'] == s)
    if not len(idx):
        raise ValueError(f'{tune.basename} has no tone set {s}')
    grp, j = dl.new_indexed_group(tune, dl.CAL_PREFIX, 'raw', attrs = dict(
        tone_set = s, tones = tune['tones'].path))
    say(f'Tone set {s}: calibration sweep of {len(idx)} tones, saved as '
        f'{grp.basename}')
    try:
        await stitched_target_sweep(
            crs, t['fres'][idx], t['ares'][idx], t['qres'][idx],
            t['res_idxs'][idx], t['tone_set'][idx], t['tone_module'][idx],
            ncos, grp, verbose = verbose, progress = progress,
            **target_sweep_kwargs)
    except BaseException:
        del tune[grp.basename]
        raise
    grp.attrs['nco_ids'] = {str(int(m)): crs.nco_ids.get(int(m))
                            for m in np.unique(t['tone_module'][idx])}
    return grp

def calibration_in_frame(crs, cal):
    """
    Decide whether a calibration sweep is in the board's current phase
    frame: none of its modules' NCOs was set since it was taken.

    Parameters:
    crs (CRS): the instrument (``nco_ids``).
    cal (zarr.Group): the calibration (``calibrate_tone_set``).

    Returns:
    in_frame (bool): True if it is; False if an NCO was set since, or
        the calibration records no NCO ids (taken by older code, or by
        another session).
    """
    ids = cal.attrs.get('nco_ids') or {}
    return bool(ids) and all(
        v is not None and crs.nco_ids.get(int(m)) == v for m, v in ids.items())

def tones_written(written, tone_set, rows, calibration):
    """
    Decide whether the board's tones are these (see ``write_set_tones``).

    Parameters:
    written (dict): the board's tones.
    tone_set (int): tone set.
    rows (array-like int): rows of the calibration sweep.
    calibration (str): path of the calibration sweep.

    Returns:
    same (bool): True if ``written`` describes exactly these tones.
    """
    return bool(written) and written.get('tone_set') == int(tone_set) and \
        written.get('calibration') == str(calibration) and \
        np.array_equal(written.get('rows', []), np.asarray(rows))

async def write_set_tones(crs, cal, rows, written, grp = None):
    """
    Write some tones of a tone set, from its calibration sweep, unless
    they are on the board already.

    ``written`` records the board's tones: empty if unknown (or none),
    else ``tone_set``, ``rows`` (rows of ``cal``), ``calibration`` (the
    path of ``cal``), ``fres`` (the written frequencies),
    ``fres_requested``, and ``ares``. Unless it describes exactly these
    tones, they are written (``write_stream_chunk``, which clears the
    others first) and it is updated. Tones stay on the board between
    streams; a calibration sweep clears them, so clear ``written`` after
    one taken elsewhere (``stream_tone_sets`` does).

    Parameters:
    crs (CRS): configured CRS instrument.
    cal (zarr.Group): the set's calibration sweep
        (``calibrate_tone_set``).
    rows (array-like int): rows of ``cal`` to write: all of the set, or a
        chunk (``plan_stream_chunks``).
    written (dict): the board's tones, updated in place.
    grp (zarr.Group or None): write group to save the tones in
        (``CRS.write_tones_for_stream``'s ``fres``, ``fres_requested``,
        ``ares``, and ``crs_config``; from ``written`` if nothing was
        written), with attr ``tones_reused``. Default None.

    Returns:
    fres_written (np.array): the tones' written frequencies in Hz.

    Raises:
    ValueError: if the calibration is out of frame
        (``calibration_in_frame``): a stream of its tones would be
        rotated from it.
    """
    rows = np.asarray(rows, dtype = np.int64)
    s = int(cal.attrs['tone_set'])
    if not calibration_in_frame(crs, cal):
        raise ValueError(f'{cal.path} is not in the board\'s phase frame '
                         '(an NCO was set since it was taken): calibrate '
                         f'tone set {s} again')
    if tones_written(written, s, rows, cal.path):
        if grp is not None:
            util.write_system_cfg_to_zarr(crs, grp.require_group('crs_config'))
            for key in ('fres', 'fres_requested', 'ares'):
                grp.create_array(name = key, data = np.asarray(
                    written[key], dtype = np.float64))
            grp.attrs['tones_reused'] = True
        return np.asarray(written['fres'], dtype = np.float64)
    fres, ares = cal['fres'][:][rows], cal['ares'][:][rows]
    fres_w = await write_stream_chunk(
        crs, fres, ares, cal['tone_module'][:][rows],
        cal.attrs['pass_ncos'][0], grp = grp)
    written.clear()
    written.update(tone_set = s, rows = rows, calibration = cal.path,
                   fres = np.asarray(fres_w, dtype = np.float64),
                   fres_requested = fres, ares = ares)
    if grp is not None:
        grp.attrs['tones_reused'] = False
    return fres_w

async def stream_tone_sets(
    crs, tune, name, duration_s, dec_stage, notes = '', set_idxs = None,
    calibrations = None, written = None, keep_tones = False, status = None,
    verbose = True, progress = None, **target_sweep_kwargs
):
    """
    Take a named timestream of a tuning's tones, one tone set after
    another, each on its calibration sweep: its tones are written (unless
    they are on the board already) and streamed, in chunks if the
    decimation stage needs them (``plan_stream_chunks``).

    The streams live with the calibration they belong to: a tone write
    is ``tune{k}/cal{j}/write{w}`` (``write_set_tones``: the tones as
    written, plus ``sweep_rows``, each tone's row in the sweep,
    ``res_idxs``, and ``qres``), and each stream of those tones is a
    ``write{w}/ts{n}`` under it (``CRS.stream``; attrs ``name``,
    ``notes``, ``dec_stage``, ``duration_s``, ``run``). Tones already on
    the board get no new write: the stream goes under the write that put
    them there. The run is recorded in the tuning's run list
    (``daq_layout.register_stream_run``: ``tune{k}/streams``) as
    ``ts{n}``, with ``name``, ``notes``, ``dec_stage``, ``duration_s``,
    ``tone_set`` (or ``tone_sets``), ``calibration`` (or
    ``calibrations``), ``tuning``, ``tones`` and ``resonators`` (where
    the tones came from), ``paths`` (its streams), and ``complete`` once
    every stream is done.

    Parameters:
    crs (CRS): configured CRS instrument (with global control to stream at
        another decimation stage than the board's).
    tune (zarr.Group): the tuning, with ``tones`` and ``crs_config``,
        in a writable store.
    name (str): timestream name, e.g. 'dark_dec6'. Names may repeat.
    duration_s (float): stream length in seconds.
    dec_stage (int): decimation stage.
    notes (str): free-form notes saved with the run. Default ''.
    set_idxs (list of int or None): tone sets to stream. None (default):
        all.
    calibrations (dict or None): per tone set, the calibration sweep to
        use: a ``cal{j}`` group of the tuning, or its path in the store
        (e.g. 'tune0/cal2'). A set without one, or whose calibration is
        out of frame (``calibration_in_frame``: its NCO was set since,
        e.g. by another tone set's stream), gets a new sweep
        (``calibrate_tone_set``, saved as the tuning's next ``cal{j}``).
        So a run of several sets calibrates each set anew as it gets to
        it, except the set whose NCOs are on the board already. Default
        None.
    written (dict or None): the board's tones (``write_set_tones``), kept
        by the caller across runs so that tones already written are not
        written again; updated in place (with ``write``, the path of the
        write group holding them). Default None (unknown: every chunk is
        written).
    keep_tones (bool): If True, leave the last tones on the board (and in
        ``written``) for the next stream. Default False: clear them.
    status (callable or None): called with a progress message before
        each step. Default None.
    verbose (bool): If True (default), show progress bars.
    progress (callable or None): ``progress(done, total)`` of a
        calibration sweep taken here (points) and of each stream
        (samples, from the elapsed time: ``CRS.stream``). Default None.
    **target_sweep_kwargs: settings of a calibration sweep taken here
        (``target_sweep``).

    Returns:
    paths (list of str): the streams' group paths, e.g.
        'tune0/cal0/write0/ts3'.

    Raises:
    ValueError: if ``name`` is empty.
    """
    if not str(name).strip():
        raise ValueError('the timestream needs a name')
    say = status or (lambda message: None)
    t = dl.load_tones(tune['tones'])
    sets = sorted(set(t['tone_set'].tolist())) if set_idxs is None else \
        [int(s) for s in set_idxs]
    cals = {int(s): c for s, c in (calibrations or {}).items()}
    if written is None:
        written = {}
    root = dl.store_root(tune, mode = 'r+')
    record = dict(name = str(name), notes = str(notes),
                  dec_stage = int(dec_stage), duration_s = float(duration_s),
                  tuning = tune.basename, tones = tune['tones'].path,
                  resonators = tune.attrs.get('resonators'), paths = [],
                  complete = False)
    if len(sets) == 1:
        record['tone_set'] = sets[0]
    else:
        record['tone_sets'] = sets
    n = dl.register_stream_run(tune, record)
    ts_name = set_name(n, dl.TS_PREFIX)
    paths, used = [], {}
    for s in sets:
        idx = np.flatnonzero(t['tone_set'] == s)
        cal = cals.get(s)
        if isinstance(cal, str):
            cal = root[cal]
        if cal is not None and not calibration_in_frame(crs, cal):
            say(f'Tone set {s}: {cal.path} is out of frame (an NCO was set '
                'since): calibrating again')
            cal = None
        if cal is None:
            cal = await calibrate_tone_set(
                crs, tune, s, status = status, verbose = verbose,
                progress = progress, **target_sweep_kwargs)
            written.clear()     # the sweep cleared the board's tones
        used[str(s)] = cal.path
        chunks = plan_stream_chunks(t['tone_module'][idx], dec_stage)
        for c_i, rows in enumerate(chunks):
            part = f' {c_i + 1}/{len(chunks)}' if len(chunks) > 1 else ''
            if tones_written(written, s, rows, cal.path) and \
                    written.get('write') in root:
                say(f'Tone set {s}: tones{part} already written')
                write = root[written['write']]
            else:
                say(f'Tone set {s}: writing {len(rows)} tones{part}')
                write, _ = dl.new_indexed_group(cal, 'write', 'raw')
                await write_set_tones(crs, cal, rows, written, grp = write)
                for key, value in (('sweep_rows', rows),
                                   ('res_idxs', cal['res_idxs'][:][rows]),
                                   ('qres', cal['qres'][:][rows])):
                    write.create_array(name = key, data = np.asarray(value))
                written['write'] = write.path
            say(f'Tone set {s}: stream{part} ({len(rows)} tones, '
                f'{duration_s:g} s at stage {dec_stage})')
            grp = dl.create_group(write, ts_name, 'raw', attrs = dict(
                name = str(name), notes = str(notes),
                dec_stage = int(dec_stage), duration_s = float(duration_s),
                run = n))
            paths.append(grp.path)
            dl.update_stream_run(tune, n, paths = paths)
            await crs.stream(duration_s, dec_stage, grp, verbose = verbose,
                             **({} if progress is None else
                                dict(progress = progress)))
    if not keep_tones:
        await crs.clear_tones()
        written.clear()
    more = dict(calibration = used[str(sets[0])]) if len(sets) == 1 \
        else dict(calibrations = used)
    dl.update_stream_run(tune, n, paths = paths, complete = True, **more)
    say(f'{name}: done as ts{n} ({len(paths)} streams)')
    return paths

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
    nyquist_guard = NYQUIST_GUARD, verbose = True, progress = None,
    cancelled = None
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
    cancelled (callable or None): if given, called with no arguments
        before each pass and after every frequency step; when it returns
        True, the sweep stops (cooperatively: between points or passes),
        clears the board's tones, restores the Nyquist zones, and raises
        ``SweepCancelled``. E.g. ``threading.Event().is_set`` set from a
        GUI thread. Default None: never cancelled.

    Returns:
    f (np.array): frequencies in Hz (dithered), ascending.
    z (np.array): complex S21.
    bands (dict): ``plan_vna_sweep`` band arrays plus ``chunk_times``
        (start of each band's pass, Unix seconds).

    Raises:
    ValueError: if the inputs are invalid.
    SweepCancelled: if ``cancelled()`` returned True.
    """
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None')
    if cancelled is not None and not callable(cancelled):
        raise TypeError('cancelled must be callable or None')

    def check_cancelled():
        """
        Raise ``SweepCancelled`` if the caller asked to stop.
        """
        if cancelled is not None and cancelled():
            raise SweepCancelled('VNA sweep cancelled')
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
            check_cancelled()
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
            if progress is not None or cancelled is not None:
                offset = done_before

                def pass_progress(done, total, offset = offset):
                    """
                    Report this pass's progress as sweep progress, and
                    stop the sweep if asked to (raising from the sweep's
                    progress callback ends ``CRS.sweep`` at this point).

                    Parameters:
                    done, total (int): points taken and total in the pass.
                    offset (int): points taken before the pass.
                    """
                    if progress is not None:
                        progress(offset + done, npoints)
                    check_cancelled()
            f, z = await crs.sweep(
                rows, np.full(len(rows), float(amplitude)), nsamps,
                ch_map = ch_map, allow_missing = False, verbose = verbose,
                pbar_description = f'VNA pass {p + 1}/{len(passes)}',
                progress = pass_progress)
            fs.append(np.asarray(f).ravel())
            zs.append(np.asarray(z).ravel())
            done_before += rows.size
    except SweepCancelled:
        # Stopped inside a pass: the sweep's tones are still written
        await crs.clear_tones()
        raise
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
    ``vna{k}`` of a device store (``citkid.daq_layout``).

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
    settings = dict(
        amplitude = float(amplitude), nsamps = int(nsamps),
        step = float(step), fmin = float(fmin), fmax = float(fmax),
        spacing = 'geometric', downward = True,
        acquisition = dl.acquisition_settings(crs))
    grp, k = dl.new_indexed_group(root, 'vna', 'raw', attrs = settings)
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
        del root[set_name(k, 'vna')]
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
    if type(crs).__name__ not in util.CRS_CLASS_NAMES:
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