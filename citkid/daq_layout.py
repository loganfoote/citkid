"""
Layout of a device's DAQ store: one zarr store per device per cooldown,
organized as a tree that follows the measurement procedure (see the
procedure guides in ``documents/procedures/``).

```
<device>.zarr/
├── vna{k}/                raw VNA sweeps (stitched over bands)
├── resonators{k}/         resonators found on vna{k} (res_idxs_all, ...)
├── tune{k}/               tunings: crs_config/, tones_init/, tones/, ...
│   │                      (attrs['resonators']: the list they use)
│   ├── cal{j}/            calibration sweep of one tone set (04)
│   │   └── write{w}/      tones written for streaming
│   │       └── ts{n}/     stream of run n
│   └── streams/           run list (attrs['runs'])
├── ts_cal{c}/             stream calibrations (05)
└── ts_photons{p}/         photon results (06)
```

Calibrated timestreams are never saved: analyses apply a calibration to
the raw streams in memory and save their reduced results. Runs saved by
earlier versions at the store root (``ts{n}/set{s}/write{w}/stream``) are
still read, and ``migrate_root_runs`` moves them into their tunings.

Every group has ``attrs['kind']``: 'raw' (write-once), 'decision', or
'analysis'. References between groups are paths inside the store, so the
store can be moved, renamed, or zipped.

This module is bookkeeping: it creates and checks groups, saves and loads
the layout's arrays, ports tunings between stores, and archives stores. It
does no measurement and no analysis.
"""
import os
import re
import zipfile
from datetime import datetime
import numpy as np
import zarr
from . import __version__ as citkid_version
from .zarr_util import list_sets, set_name, to_jsonable

LAYOUT_VERSION = 1
KINDS = ('raw', 'decision', 'analysis')
# Per-module CRS settings saved per tuning, and per acquisition
TUNE_MODULE_SETTINGS = ('full_scale_dbm', 'adc_attenuation_db',
                        'nyquist_zone', 'cable_length_m')
ACQ_MODULE_SETTINGS = ('adc_calibration_mode', 'adc_autocal')
VNA_BAND_ARRAYS = ('chunk_edges', 'chunk_ncos', 'chunk_modules',
                   'chunk_passes', 'chunk_zones', 'chunk_nchannels',
                   'chunk_times')
TONE_ARRAYS = ('fres', 'qres', 'ares', 'res_idxs', 'tone_set',
               'tone_module')
RESONATORS_PREFIX = 'resonators'
# Timestream runs (ts{n}: each run's streams under its calibration's tone
# writes, and runs at the store root in the earlier layout), calibration
# sweeps (cal{j}), and the earlier layout's stream name
TS_PREFIX = 'ts'
CAL_PREFIX = 'cal'
STREAM_NAME = 'stream'
RESONATOR_ARRAYS = ('res_idxs_all', 'fres_all', 'qres_all', 'Qr_all',
                    'depth_all', 'source_all')
# Largest chunk of a 1D array; longer arrays are sharded into one file
_CHUNK = 2 ** 20

################################################################################
################################ Store and groups ##############################
################################################################################
def _now():
    """
    Return the current local time as a string.

    Returns:
    now (str): 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def create_store(path, device, cooldown, setup = None):
    """
    Create a device store, or open an existing one for the same device.

    Parameters:
    path (str or os.PathLike): path of the ``.zarr`` directory.
    device (str): device name.
    cooldown (str): cooldown name.
    setup (dict or None): free-form setup description (fridge channel,
        attenuators, filters, ...). Saved only when the store is created.
        Default None (empty).

    Returns:
    root (zarr.Group): the store's root group, writable.

    Raises:
    ValueError: if the store exists but is not a layout store, or belongs
        to another device or cooldown.
    """
    path = os.path.expanduser(str(path))
    exists = os.path.exists(os.path.join(path, 'zarr.json'))
    root = zarr.open_group(path, mode = 'a')
    if exists:
        check_store(root)
        for key, value in (('device', device), ('cooldown', cooldown)):
            if root.attrs[key] != value:
                raise ValueError(
                    f"store {path} is for {key} {root.attrs[key]!r}, not "
                    f"{value!r}")
        return root
    root.attrs.update(to_jsonable(dict(
        layout_version = LAYOUT_VERSION, device = str(device),
        cooldown = str(cooldown), created = _now(),
        citkid_version = citkid_version, setup = dict(setup or {}))))
    return root

def open_store(path, mode = 'r'):
    """
    Open a device store, as a directory or as a zip archive.

    Parameters:
    path (str, os.PathLike, or zarr.Group): store path (``.zarr``
        directory or ``.zip`` archive), or an already open root group.
    mode (str): 'r' (default) or 'a'. A zip archive is always read-only.

    Returns:
    root (zarr.Group): the store's root group.

    Raises:
    ValueError: if the store is not a layout store, or ``mode`` is 'a' for
        a zip archive.
    """
    if isinstance(path, zarr.Group):
        check_store(path)
        return path
    path = os.path.expanduser(str(path))
    if path.endswith('.zip'):
        if mode != 'r':
            raise ValueError('a zipped store is read-only; unzip it to write')
        root = zarr.open_group(zarr.storage.ZipStore(path, mode = 'r'),
                               mode = 'r')
    else:
        root = zarr.open_group(path, mode = mode)
    check_store(root)
    return root

def check_store(root):
    """
    Check that a group is the root of a layout store.

    Parameters:
    root (zarr.Group): group to check.

    Raises:
    ValueError: if ``layout_version`` is missing or newer than this code.
    """
    version = root.attrs.get('layout_version')
    if version is None:
        raise ValueError('not a DAQ layout store (no layout_version)')
    if version > LAYOUT_VERSION:
        raise ValueError(f'store layout_version {version} is newer than '
                         f'this citkid ({LAYOUT_VERSION}); update citkid')

def list_indices(grp, prefix):
    """
    Return the indices of the child groups named ``prefix`` + integer.

    Parameters:
    grp (zarr.Group): parent group.
    prefix (str): name prefix, e.g. 'tune' or 'sweep'.

    Returns:
    idxs (list of int): sorted indices, e.g. [0, 1, 2, 10].
    """
    return list_sets(grp, prefix = prefix)

def next_index(grp, prefix):
    """
    Return the next free index of the child groups named ``prefix`` +
    integer.

    Parameters:
    grp (zarr.Group): parent group.
    prefix (str): name prefix.

    Returns:
    idx (int): one more than the largest index, or 0 if there is none.
    """
    idxs = list_indices(grp, prefix)
    return idxs[-1] + 1 if idxs else 0

def latest_index(grp, prefix):
    """
    Return the largest index of the child groups named ``prefix`` +
    integer.

    Parameters:
    grp (zarr.Group): parent group.
    prefix (str): name prefix.

    Returns:
    idx (int): largest index.

    Raises:
    KeyError: if there is no such group.
    """
    idxs = list_indices(grp, prefix)
    if not idxs:
        raise KeyError(f"no '{prefix}' groups in {grp.path or 'the root'}")
    return idxs[-1]

def create_group(parent, name, kind, attrs = None, overwrite = False):
    """
    Create a layout group with its ``kind`` and creation attributes.

    Parameters:
    parent (zarr.Group): parent group.
    name (str): group name (may be a path, e.g. 'tune0/tones').
    kind (str): 'raw', 'decision', or 'analysis'.
    attrs (dict or None): more attributes. Default None.
    overwrite (bool): If True, replace an existing 'decision' or 'analysis'
        group. Raw groups are never replaced. Default False.

    Returns:
    grp (zarr.Group): the new group.

    Raises:
    ValueError: if ``kind`` is unknown, or the group exists and can't be
        replaced.
    """
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    if name in parent:
        old = parent[name]
        old_kind = old.attrs.get('kind') if isinstance(old, zarr.Group) \
            else None
        if not overwrite or old_kind == 'raw' or kind == 'raw':
            raise ValueError(
                f"{_path(parent, name)} exists" + (
                    ' and is raw data (write-once)' if 'raw' in
                    (old_kind, kind) else '; pass overwrite = True'))
        del parent[name]
    grp = parent.create_group(name)
    grp.attrs.update(to_jsonable(dict(
        kind = kind, created = _now(), citkid_version = citkid_version,
        **(attrs or {}))))
    return grp

def require_group(parent, name, kind):
    """
    Open a layout group, creating it if needed (e.g. a ``power_sweep``
    that holds the ``sweep{k}`` groups).

    Parameters:
    parent (zarr.Group): parent group.
    name (str): group name.
    kind (str): 'raw', 'decision', or 'analysis'.

    Returns:
    grp (zarr.Group): the group.

    Raises:
    ValueError: if the group exists with another kind.
    """
    if name not in parent:
        return create_group(parent, name, kind)
    grp = parent[name]
    if grp.attrs.get('kind') != kind:
        raise ValueError(f"{_path(parent, name)} has kind "
                         f"{grp.attrs.get('kind')!r}, not {kind!r}")
    return grp

def new_indexed_group(parent, prefix, kind, attrs = None):
    """
    Create the next ``prefix`` + integer group (e.g. 'sweep2').

    Parameters:
    parent (zarr.Group): parent group.
    prefix (str): name prefix.
    kind (str): 'raw', 'decision', or 'analysis'.
    attrs (dict or None): more attributes. Default None.

    Returns:
    grp (zarr.Group): the new group.
    idx (int): its index.
    """
    idx = next_index(parent, prefix)
    return create_group(parent, set_name(idx, prefix), kind, attrs), idx

def _path(parent, name):
    """
    Return the store path of a child, for messages.

    Parameters:
    parent (zarr.Group): parent group.
    name (str): child name.

    Returns:
    path (str): e.g. 'tune0/tones'.
    """
    return f'{parent.path}/{name}' if parent.path else name

def write_array(grp, name, data):
    """
    Save an array, in as few files as practical: one chunk if it is short,
    else 1D chunks of 2**20 points in one shard.

    Parameters:
    grp (zarr.Group): group.
    name (str): array name; must not exist.
    data (array-like): data. Strings are saved as variable-length strings.

    Returns:
    arr (zarr.Array): the new array.
    """
    data = np.asarray(data)
    if data.dtype.kind in 'US':
        arr = grp.create_array(name, shape = data.shape, dtype = str,
                               fill_value = '')
        if data.size:
            arr[...] = data.astype(str)
        return arr
    if data.ndim == 1 and len(data) > _CHUNK:
        n = int(np.ceil(len(data) / _CHUNK)) * _CHUNK
        return grp.create_array(name, data = data, chunks = (_CHUNK,),
                                shards = (n,))
    return grp.create_array(name, data = data)

################################################################################
############################# CRS configuration ################################
################################################################################
def crs_identity(crs):
    """
    Return the CRS identity saved at the store root.

    Parameters:
    crs (CRS): configured CRS instrument.

    Returns:
    identity (dict): ``crs_serial_number``, ``firmware_version``,
        ``rfmux_version``.
    """
    return dict(crs_serial_number = int(crs.serial_number),
                firmware_version = str(crs.firmware_release.version),
                rfmux_version = str(crs.rfmux_version))

def check_crs_identity(root, crs):
    """
    Record the CRS identity on the first acquisition, and check it on later
    ones.

    Parameters:
    root (zarr.Group): store root, writable.
    crs (CRS): configured CRS instrument.

    Raises:
    RuntimeError: if the CRS differs from the one the store was measured
        with (start a new store for a new CRS or firmware).
    """
    ident = crs_identity(crs)
    saved = {k: root.attrs[k] for k in ident if k in root.attrs}
    if not saved:
        root.attrs.update(ident)
        return
    diff = {k: (saved.get(k), v) for k, v in ident.items()
            if saved.get(k) != v}
    if diff:
        raise RuntimeError(
            'the CRS differs from the one this store was measured with '
            f'(saved, now): {diff}. Start a new store for a new CRS or '
            'firmware')

def tune_config(crs):
    """
    Return the tuning-level CRS settings.

    Parameters:
    crs (CRS): configured CRS instrument.

    Returns:
    config (dict): ``ncos`` ({module: Hz}), ``modules`` (this session's
        modules in the active bank), ``analog_bank_high``,
        ``extended_bw``, ``bw`` (Hz), and ``module_settings``
        ({module: {setting: value}} for ``TUNE_MODULE_SETTINGS``).
    """
    modules = [int(m) for m in crs.active_module_idxs]
    return dict(
        ncos = {str(m): float(f) for m, f in sorted(crs.nco_freqs.items())},
        modules = modules,
        analog_bank_high = bool(crs.analog_bank_high),
        extended_bw = bool(crs.extended_bw), bw = float(crs.bw),
        module_settings = {
            str(m): {k: crs.module_cfg[m][k] for k in TUNE_MODULE_SETTINGS}
            for m in modules if m in crs.module_cfg})

def acquisition_settings(crs):
    """
    Return the CRS settings recorded on every raw acquisition group.

    Parameters:
    crs (CRS): configured CRS instrument.

    Returns:
    settings (dict): ``clock_source`` and ``module_settings``
        ({module: {setting: value}} for ``ACQ_MODULE_SETTINGS``).
    """
    return dict(
        clock_source = str(crs.clock_source),
        module_settings = {
            str(m): {k: crs.module_cfg[m][k] for k in ACQ_MODULE_SETTINGS}
            for m in sorted(crs.module_cfg)
            if m in crs.active_module_idxs})

def tuning_config(crs, plan):
    """
    Return a tuning's CRS settings with its tone-set plan: the NCOs are
    those of each tone set, not the board's current ones.

    Parameters:
    crs (CRS): configured CRS instrument.
    plan (dict): ``tone_plan.plan_tone_sets`` output.

    Returns:
    config (dict): ``tune_config`` without ``ncos``, plus ``tone_sets``
        (per set, {module: NCO in Hz}).

    Raises:
    ValueError: if the plan's ``extended_bw`` differs from the board's.
    """
    if bool(plan['extended_bw']) != bool(crs.extended_bw):
        raise ValueError(
            f"the plan uses extended_bw = {plan['extended_bw']}, the board "
            f'{crs.extended_bw}: plan with the board\'s setting, or change '
            'it with crs_init.py')
    config = tune_config(crs)
    del config['ncos']
    config['tone_sets'] = [{str(m): float(f) for m, f in s.items()}
                           for s in plan['ncos']]
    return config

def write_tune_config(tune_grp, config):
    """
    Save a tuning's CRS settings as ``tune{k}/crs_config``.

    Parameters:
    tune_grp (zarr.Group): tuning group.
    config (dict): settings (``tune_config``). May hold more keys, e.g.
        the tone-set plan.

    Returns:
    grp (zarr.Group): the ``crs_config`` group (kind 'decision').
    """
    return create_group(tune_grp, 'crs_config', 'decision',
                        attrs = dict(config = config))

def check_tune_config(tune_grp, crs, rtol = 1e-9, atol = 1e-6):
    """
    Check that the board matches a tuning's saved CRS settings.

    Only the settings of ``tune_config`` are compared; extra keys in the
    saved config are ignored. With a tone-set plan (``tuning_config``), the
    NCOs are not compared: each pass sets its tone set's NCOs from the
    saved plan.

    Parameters:
    tune_grp (zarr.Group): tuning group with ``crs_config``.
    crs (CRS): configured CRS instrument.
    rtol, atol (float): tolerances for numbers. Defaults 1e-9, 1e-6.

    Raises:
    KeyError: if the tuning has no ``crs_config``.
    RuntimeError: if a setting differs (changing one means a new tuning).
    """
    saved = tune_grp['crs_config'].attrs['config']
    now = to_jsonable(tune_config(crs))
    if 'tone_sets' in saved:
        del now['ncos']
    diffs = _diff(saved, now, rtol, atol)
    if diffs:
        raise RuntimeError(
            "the board doesn't match this tuning's CRS settings "
            f"({'; '.join(diffs)}). Changing them means a new tuning")

def _diff(saved, now, rtol, atol, prefix = ''):
    """
    List the differences between saved and current settings.

    Parameters:
    saved (dict): saved settings (JSON types).
    now (dict): current settings (JSON types); only its keys are compared.
    rtol, atol (float): tolerances for numbers.
    prefix (str): key path for messages. Default ''.

    Returns:
    diffs (list of str): one message per difference.
    """
    diffs = []
    for key, value in now.items():
        name = f'{prefix}{key}'
        if key not in saved:
            diffs.append(f'{name}: not saved, now {value!r}')
            continue
        old = saved[key]
        if isinstance(value, dict) and isinstance(old, dict):
            if set(old) != set(value):
                diffs.append(f'{name}: {sorted(old)} vs now {sorted(value)}')
            else:
                diffs += _diff(old, value, rtol, atol, f'{name}.')
        elif isinstance(value, float) or isinstance(old, float):
            if not (isinstance(old, (int, float))
                    and np.isclose(old, value, rtol = rtol, atol = atol)):
                diffs.append(f'{name}: {old!r} vs now {value!r}')
        elif old != value:
            diffs.append(f'{name}: {old!r} vs now {value!r}')
    return diffs

################################################################################
################################# VNA sweeps ###################################
################################################################################
def save_vna_sweep(
    sweep_grp, f, z, chunk_edges, chunk_ncos, chunk_modules, chunk_passes,
    chunk_zones, chunk_nchannels, chunk_times, settings = None
):
    """
    Save a stitched VNA sweep and its band arrays in a raw sweep group.

    Parameters:
    sweep_grp (zarr.Group): the raw ``vna{k}`` group, empty.
    f (np.array): frequencies in Hz.
    z (np.array): complex S21, same shape as ``f``.
    chunk_edges (np.array): (nb, 2) first and last frequency of each band
        in Hz.
    chunk_ncos (np.array): (nb,) NCO frequency of each band in Hz.
    chunk_modules (np.array): (nb,) module that swept each band.
    chunk_passes (np.array): (nb,) pass index of each band.
    chunk_zones (np.array): (nb,) Nyquist zone of each band (1 or 2).
    chunk_nchannels (np.array): (nb,) channels used by each band.
    chunk_times (np.array): (nb,) start time of each band, Unix seconds.
    settings (dict or None): sweep settings saved as attributes
        (amplitude, npoints_per_ch, nsamps, ...). Default None.

    Raises:
    ValueError: if the shapes don't match.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    if f.ndim != 1 or f.shape != z.shape:
        raise ValueError('f and z must be 1D arrays of the same length')
    o = np.argsort(f, kind = 'stable')
    bands = dict(
        chunk_edges = np.asarray(chunk_edges, dtype = np.float64),
        chunk_ncos = np.asarray(chunk_ncos, dtype = np.float64),
        chunk_modules = np.asarray(chunk_modules, dtype = np.int32),
        chunk_passes = np.asarray(chunk_passes, dtype = np.int32),
        chunk_zones = np.asarray(chunk_zones, dtype = np.int32),
        chunk_nchannels = np.asarray(chunk_nchannels, dtype = np.int32),
        chunk_times = np.asarray(chunk_times, dtype = np.float64))
    nb = len(bands['chunk_ncos'])
    if bands['chunk_edges'].shape != (nb, 2) or any(
            bands[k].shape != (nb,) for k in VNA_BAND_ARRAYS[1:]):
        raise ValueError('band arrays must have nb entries (chunk_edges: '
                         '(nb, 2))')
    write_array(sweep_grp, 'f', f[o])
    write_array(sweep_grp, 'z', z[o])
    for name, value in bands.items():
        write_array(sweep_grp, name, value)
    if settings:
        sweep_grp.attrs.update(to_jsonable(settings))

def load_vna_sweep(root, k = None, bands = False):
    """
    Load a stitched VNA sweep.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.
    k (int or None): sweep index. None (default): the latest.
    bands (bool): If True, also return the band arrays. Default False.

    Returns:
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21.
    band_arrays (dict): only if ``bands``: ``VNA_BAND_ARRAYS`` by name.
    """
    root = open_store(root)
    k = latest_index(root, 'vna') if k is None else int(k)
    grp = root[set_name(k, 'vna')]
    f, z = grp['f'][:], grp['z'][:]
    if not bands:
        return f, z
    return f, z, {name: grp[name][:] for name in VNA_BAND_ARRAYS}

################################################################################
################################# Resonators ###################################
################################################################################
def save_resonators(
    root, k, fres, qres, source, Qr = None, depth = None, overwrite = False
):
    """
    Save a resonator list as ``resonators{k}``: the resonators found on VNA
    sweep ``vna{k}`` (or a list ported from another store).

    The resonators get indices 0 ... n - 1 (``res_idxs_all``) in ascending
    frequency order (``fres``, ``qres``, ``Qr`` and ``depth`` are sorted
    together if they aren't already: the pipeline's gain cuts need a sorted
    list).
    Each tuning records which list its ``res_idxs`` refer to
    (``attrs['resonators']``, see ``new_tuning``), so a list can't be
    replaced once a tuning uses it.

    Parameters:
    root (zarr.Group): store root, writable.
    k (int): list index (the VNA sweep's index).
    fres (array-like): resonance frequencies in Hz.
    qres (array-like or float): span factors for cutting them out of gain
        fits, one per resonator or one for all.
    source (str): where they were found, e.g. 'vna0/full_sweep_fit'.
    Qr (array-like or None): fitted total quality factors (NaN where the
        fit is unreliable), used for linewidths (e.g.
        ``tone_plan.select_isolated``). None (default): all NaN.
    depth (array-like or None): resonance dip depths
        (``vna.full_sweep.measured_depths``), used as the strength when
        only the stronger of two close resonators gets a tone
        (``select_isolated`` with ``pairs = 'keep_stronger'``). None
        (default): all NaN.
    overwrite (bool): If True, replace an existing list that no tuning
        uses. Default False.

    Returns:
    grp (zarr.Group): the ``resonators{k}`` group.

    Raises:
    FileExistsError: if the list exists and ``overwrite`` is False.
    ValueError: if the list exists and a tuning uses it, or ``fres`` is
        not 1D.
    """
    fres = np.atleast_1d(np.asarray(fres, dtype = np.float64))
    if fres.ndim != 1:
        raise ValueError('fres must be 1D')
    qres = np.broadcast_to(np.asarray(qres, dtype = np.float64),
                           fres.shape).copy()
    def per_resonator(values):
        """
        Broadcast optional per-resonator values to ``fres``' shape.

        Parameters:
        values (array-like or None): the values. None: all NaN.

        Returns:
        values (np.array): float array of ``fres``' shape.
        """
        if values is None:
            return np.full(fres.shape, np.nan)
        return np.broadcast_to(np.asarray(values, dtype = np.float64),
                               fres.shape).copy()
    Qr, depth = per_resonator(Qr), per_resonator(depth)
    order = np.argsort(fres, kind = 'stable')
    fres, qres, Qr, depth = fres[order], qres[order], Qr[order], depth[order]
    name = set_name(int(k), RESONATORS_PREFIX)
    if name in root:
        users = tunings_using(root, name)
        if users:
            raise ValueError(f'{name} is used by {users}; its res_idxs '
                             'must not change')
        if not overwrite:
            raise FileExistsError(f'{name} exists; pass overwrite = True '
                                  'to replace it')
    grp = create_group(root, name, 'decision', attrs = dict(source = source),
                       overwrite = overwrite)
    arrays = dict(
        res_idxs_all = np.arange(len(fres), dtype = np.int32),
        fres_all = fres, qres_all = qres, Qr_all = Qr, depth_all = depth,
        source_all = np.full(len(fres), str(source)))
    for key in RESONATOR_ARRAYS:
        write_array(grp, key, arrays[key])
    return grp

def load_resonators(root, k = None):
    """
    Load a resonator list.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.
    k (int or str or None): list index, or a group name such as a
        tuning's ``attrs['resonators']``. None (default): the latest.

    Returns:
    resonators (dict): ``RESONATOR_ARRAYS`` by name (``source_all`` as
        str; ``Qr_all`` and ``depth_all`` NaN if the list has none, e.g.
        lists saved before ``depth_all`` existed).
    """
    root = open_store(root)
    if k is None:
        k = latest_index(root, RESONATORS_PREFIX)
    name = k if isinstance(k, str) else set_name(int(k), RESONATORS_PREFIX)
    grp = root[name]
    out = {key: grp[key][:] for key in RESONATOR_ARRAYS if key in grp}
    for key in ('Qr_all', 'depth_all'):
        if key not in out:
            out[key] = np.full(len(out['fres_all']), np.nan)
    out['source_all'] = np.asarray(out['source_all'], dtype = str)
    return out

def tunings_using(root, name):
    """
    List the tunings whose ``res_idxs`` refer to a resonator list.

    Parameters:
    root (zarr.Group): store root.
    name (str): resonator list, e.g. 'resonators0'.

    Returns:
    names (list of str): tuning group names, e.g. ['tune0'].
    """
    return [set_name(j, 'tune') for j in list_indices(root, 'tune')
            if root[set_name(j, 'tune')].attrs.get('resonators') == name]

def new_tuning(root, resonators, attrs = None):
    """
    Create the next tuning, ``tune{k}``, recording the resonator list its
    ``res_idxs`` refer to.

    Parameters:
    root (zarr.Group): store root, writable.
    resonators (int or str): resonator list index or group name.
    attrs (dict or None): more attributes. Default None.

    Returns:
    grp (zarr.Group): the new tuning.
    k (int): its index.

    Raises:
    KeyError: if the resonator list doesn't exist.
    """
    name = resonators if isinstance(resonators, str) else \
        set_name(int(resonators), RESONATORS_PREFIX)
    if name not in root:
        raise KeyError(f'no resonator list {name!r}')
    return new_indexed_group(root, 'tune', 'decision',
                             attrs = dict(attrs or {}, resonators = name))

################################################################################
################################### Tones ######################################
################################################################################
def save_tones(
    grp, fres, qres, ares, res_idxs, tone_set = None, tone_module = None,
    source = None
):
    """
    Save a tone list (``fres``, ``qres``, ``ares``, ``res_idxs``,
    ``tone_set``, and ``tone_module`` if given) in a group.

    Parameters:
    grp (zarr.Group): group, e.g. a new ``tones`` or ``tones_init`` group.
    fres (array-like): tone frequencies in Hz.
    qres (array-like): span factors.
    ares (array-like or float): tone powers in dBm.
    res_idxs (array-like int): resonator indices (negative for calibration
        tones).
    tone_set (array-like int or None): tone set of each tone. None
        (default): all in set 0.
    tone_module (array-like int or None): module of each tone (from the
        tone-set plan). None (default): not saved.
    source (dict or None): where each array came from, saved as
        ``attrs['source']`` (e.g. {'fres': 'tune0/fq_sweep/fres_opt'}).
        Default None.

    Raises:
    ValueError: if the lengths differ.
    """
    fres = np.asarray(fres, dtype = np.float64)
    n = len(fres)
    arrays = dict(
        fres = fres,
        qres = np.broadcast_to(np.asarray(qres, dtype = np.float64),
                               (n,)),
        ares = np.broadcast_to(np.asarray(ares, dtype = np.float64), (n,)),
        res_idxs = np.asarray(res_idxs, dtype = np.int32),
        tone_set = np.zeros(n, dtype = np.int32) if tone_set is None
        else np.asarray(tone_set, dtype = np.int32))
    if tone_module is not None:
        arrays['tone_module'] = np.asarray(tone_module, dtype = np.int32)
    if any(v.shape != (n,) for v in arrays.values()):
        raise ValueError('fres, qres, ares, res_idxs, tone_set and '
                         'tone_module must have the same length')
    for name, value in arrays.items():
        write_array(grp, name, value)
    if source is not None:
        grp.attrs['source'] = to_jsonable(source)

def load_tones(grp):
    """
    Load a tone list saved by ``save_tones``.

    Parameters:
    grp (zarr.Group): group holding the tones (e.g. ``tune0/tones``).

    Returns:
    tones (dict): ``TONE_ARRAYS`` by name; ``tone_set`` is all zeros if the
        group has none, and ``tone_module`` is left out if it has none.
    """
    out = {name: grp[name][:] for name in TONE_ARRAYS if name in grp}
    if 'tone_set' not in out:
        out['tone_set'] = np.zeros(len(out['fres']), dtype = np.int32)
    return out

################################################################################
################################### Streams ####################################
################################################################################
def store_root(grp, mode = 'r'):
    """
    Open the root of the store a group is in.

    Parameters:
    grp (zarr.Group): any group of the store.
    mode (str): 'r' (default) or 'r+' (writable).

    Returns:
    root (zarr.Group): the store root.
    """
    return zarr.open_group(grp.store, path = '', mode = mode)

def calibrations(tune):
    """
    Return a tuning's saved calibration sweeps (``cal{j}``, made by
    ``crs.procedures.calibrate_tone_set``), oldest first.

    Parameters:
    tune (zarr.Group): the tuning.

    Returns:
    cals (list of dict): per sweep: ``name`` ('cal{j}'), ``path`` (in the
        store), ``index`` (j), ``tone_set``, ``created``, ``complete``,
        ``ntones``, ``npoints_fine`` (0 without a fine sweep), ``nsamps``,
        and ``nco_ids`` (the NCO settings it was taken under, see
        ``crs.procedures.calibration_in_frame``).
    """
    out = []
    for j in list_indices(tune, CAL_PREFIX):
        grp = tune[set_name(j, CAL_PREFIX)]
        ntones = int(grp['fres'].shape[0]) if 'fres' in grp else 0
        npoints = int(grp['fine_sweep/z'].shape[1]) \
            if 'fine_sweep' in grp and 'z' in grp['fine_sweep'] else 0
        out.append(dict(
            name = grp.basename, path = grp.path, index = j,
            tone_set = grp.attrs.get('tone_set'),
            created = grp.attrs.get('created'),
            complete = bool(grp.attrs.get('complete')), ntones = ntones,
            npoints_fine = npoints, nsamps = grp.attrs.get('nsamps'),
            nco_ids = grp.attrs.get('nco_ids')))
    return out

STREAMS_GROUP = 'streams'

def register_stream_run(tune, record):
    """
    Add a timestream run to a tuning's run list (``tune{k}/streams``,
    attr ``runs``), as the next ``ts{n}``.

    Parameters:
    tune (zarr.Group): the tuning, writable.
    record (dict): the run: ``name``, ``notes``, ``dec_stage``,
        ``duration_s``, ``tone_set``, ``calibration`` (the ``cal{j}``
        its streams are under), ``paths`` (its stream groups, filled in
        as they are taken), ``complete``, and anything else to keep.

    Returns:
    n (int): the run's index.
    """
    grp = require_group(tune, STREAMS_GROUP, 'raw')
    runs = list(grp.attrs.get('runs', []))
    n = len(runs)
    runs.append(to_jsonable(dict(record, run = n, created = _now())))
    grp.attrs['runs'] = runs
    return n

def update_stream_run(tune, n, **changes):
    """
    Change a run's record in the tuning's run list.

    Parameters:
    tune (zarr.Group): the tuning, writable.
    n (int): the run's index.
    **changes: fields to set, e.g. ``paths``, ``complete``.
    """
    grp = tune[STREAMS_GROUP]
    runs = list(grp.attrs.get('runs', []))
    runs[n] = to_jsonable(dict(runs[n], **changes))
    grp.attrs['runs'] = runs

def _stream_size(root, run):
    """
    Keep a run's finished streams and add its sample count and chunk
    layout from the first of them, if any.

    Parameters:
    root (zarr.Group): the store root.
    run (dict): the run, with ``paths``; changed in place.
    """
    paths = [p for p in run['paths']
             if p in root and 'dt' in root[p] and 'z' in root[p]]
    run['paths'] = paths
    if paths:
        st = root[paths[0]]
        nsamp, chunk = int(st['z'].shape[2]), int(st['z'].chunks[2])
        dt = float(st['dt'][...])
        run.update(nsamples = nsamp, chunk_samples = chunk,
                   n_chunks = -(-nsamp // chunk), chunk_s = chunk * dt)

def stream_runs(root, tuning = None):
    """
    Return the store's timestream runs, with each run's chunk layout.

    A run is one named timestream of a tone set, taken by
    ``crs.procedures.stream_tone_sets`` on its calibration sweep: its
    streams are ``tune{k}/cal{j}/write{w}/ts{n}`` (one per write; a tone
    set streams in several writes at fast decimation stages), and the
    tuning's run list (``tune{k}/streams``) records it. Runs saved by
    earlier versions as ``ts{n}`` at the store root, with the sweep
    copied into them, are listed too.

    Parameters:
    root (zarr.Group): the store root.
    tuning (str or None): only the runs of this tuning (e.g. 'tune0').
        None (default): all.

    Returns:
    runs (list of dict): per run, in order: its record (``name``,
        ``notes``, ``dec_stage``, ``duration_s``, ``tone_set``,
        ``calibration``, ``tuning``, ``complete``, ``created``) plus
        ``run`` (n), ``group`` ('ts{n}', its label; the group of a root
        run), ``paths`` (its finished streams, e.g.
        'tune0/cal0/write0/ts3'; one still being written is left out),
        and, if it has streams, ``nsamples``, ``chunk_samples`` (zarr
        chunk length along time), ``n_chunks``, and ``chunk_s`` (chunk
        length in s).
    """
    out = []
    # Runs at the store root (earlier layout)
    for n in list_indices(root, TS_PREFIX):
        grp = root[set_name(n, TS_PREFIX)]
        run = dict(grp.attrs, run = n, group = grp.basename)
        if tuning is not None and run.get('tuning') != tuning:
            continue
        paths = []
        for s in list_indices(grp, 'set'):
            sg = grp[set_name(s)]
            for w in list_indices(sg, 'write'):
                wg = sg[set_name(w, 'write')]
                if STREAM_NAME in wg:
                    paths.append(f'{grp.basename}/{set_name(s)}/'
                                 f'{set_name(w, "write")}/{STREAM_NAME}')
        run['paths'] = paths
        _stream_size(root, run)
        out.append(run)
    # Runs on their calibrations
    for k in list_indices(root, 'tune'):
        tune = root[set_name(k, 'tune')]
        if tuning is not None and tune.basename != tuning:
            continue
        if STREAMS_GROUP not in tune:
            continue
        for rec in tune[STREAMS_GROUP].attrs.get('runs', []):
            run = dict(rec, tuning = tune.basename,
                       group = set_name(int(rec['run']), TS_PREFIX))
            run['paths'] = list(rec.get('paths', []))
            _stream_size(root, run)
            out.append(run)
    return out

def describe_stream_runs(root, tuning = None):
    """
    Describe the store's timestream runs, one line each, for choosing what
    to calibrate.

    Parameters:
    root (zarr.Group): the store root.
    tuning (str or None): only this tuning's runs. Default None (all).

    Returns:
    text (str): the description.
    """
    lines = ['Timestream runs (ts{n}: name, tuning, calibration, stage, '
             'length, tone set(s), zarr chunks, notes):']
    for r in stream_runs(root, tuning):
        chunks = (f"{r['n_chunks']} chunks of {r['chunk_s']:.3g} s"
                  if 'n_chunks' in r else 'no streams')
        done = '' if r.get('complete') else ', incomplete'
        sets = r['tone_set'] if 'tone_set' in r else r.get('tone_sets')
        lines.append(
            f"  {r['group']}: {r.get('name')}, {r.get('tuning')}, "
            f"{r.get('calibration', 'copied')}, stage {r.get('dec_stage')}, "
            f"{r.get('duration_s', 0):g} s, set {sets}, {chunks}{done}"
            + (f", {r['notes']}" if r.get('notes') else ''))
    if len(lines) == 1:
        lines.append('  none')
    return '\n'.join(lines)

def select_streams(
    root, name = None, run = None, chunks = None, tone_sets = None,
    tuning = None
):
    """
    Turn one choice of streams into sources for
    ``pipeline.stream_sources``: one per calibration sweep of the run
    (with all its writes), with a window of zarr chunks.

    Parameters:
    root (zarr.Group): the store root.
    name (str or None): run name: the latest run with this name, unless
        ``run`` is given.
    run (int or None): run index n (``ts{n}``).
    chunks (int, tuple, or None): window along time, in zarr chunks: None
        (default) for all of it, n for the first n chunks, or (first,
        stop) for chunks first, ..., stop - 1.
    tone_sets (list of int or None): tone sets to keep. None (default):
        all.
    tuning (str or None): only this tuning's runs (``ts{n}`` counts
        per tuning). Default None: the latest tuning's for ``run``.

    Returns:
    sources (list of dict): ``group`` (the calibration sweep),
        ``writes``, ``start``, ``stop``, ``stream`` (the stream's name
        under each write).
    run (dict): the run (``stream_runs``), plus ``start`` and ``stop``.

    Raises:
    KeyError: if no run matches.
    ValueError: if the chunk window is empty, or the run's streams have
        different lengths.
    """
    runs = stream_runs(root, tuning)
    if run is not None:
        match = [r for r in runs if r['run'] == int(run)]
        if not match:
            raise KeyError(f'the store has no ts{run}')
        r = match[-1]
        if name is not None and r.get('name') != name:
            raise KeyError(f"ts{run} is {r.get('name')!r}, not {name!r}")
    else:
        named = [r for r in runs if r.get('name') == name]
        if not named:
            raise KeyError(f'the store has no run named {name!r}')
        r = named[-1]
    if 'n_chunks' not in r:
        raise KeyError(f"{r['group']} has no streams")
    if chunks is None:
        first, stop = 0, r['n_chunks']
    elif isinstance(chunks, (int, np.integer)):
        first, stop = 0, int(chunks)
    else:
        first, stop = (int(c) for c in chunks)
    start = first * r['chunk_samples']
    end = min(stop * r['chunk_samples'], r['nsamples'])
    if end <= start:
        raise ValueError(f"chunks {chunks} select no samples of "
                         f"{r['group']} ({r['n_chunks']} chunks)")
    sources = []
    for path in r['paths']:
        sweep, write, stream = path.rsplit('/', 2)
        if root[path]['z'].shape[2] != r['nsamples']:
            raise ValueError(f'{path} has a different length')
        if tone_sets is not None and \
                int(root[sweep].attrs['tone_set']) not in tone_sets:
            continue
        if sources and sources[-1]['group'] == sweep:
            sources[-1]['writes'].append(write)
        else:
            sources.append(dict(group = sweep, writes = [write],
                                start = int(start), stop = int(end),
                                stream = stream))
    return sources, dict(r, start = int(start), stop = int(end))

def migrate_root_runs(path, verbose = True):
    """
    Move a store's root-level timestream runs, the layout of earlier
    versions (``ts{n}`` at the root, with the calibration sweep copied
    into it and the streams in ``ts{n}/set{s}/write{w}/stream``), under
    the calibrations they were taken with
    (``tune{k}/cal{j}/write{w}/ts{n}``, see
    ``crs.procedures.stream_tone_sets``), and list them in their tunings'
    run lists; the copied sweeps are deleted. A run whose sets record no
    calibration (attr ``calibration``), or whose calibration or tuning
    is gone, is left where it is.

    The store must be a local directory store: its groups are moved on
    disk. Nothing else should have it open.

    Parameters:
    path (str or os.PathLike): the store's path.
    verbose (bool): If True (default), say what is moved.

    Returns:
    moved (list of str): the runs moved, e.g. ['ts0', 'ts1'].
    """
    import shutil
    path = os.path.expanduser(str(path))
    root = open_store(path, mode = 'a')
    moved = []
    for n in list_indices(root, TS_PREFIX):
        name = set_name(n, TS_PREFIX)
        grp = root[name]
        attrs = dict(grp.attrs)
        tuning = attrs.get('tuning')
        sets = list_indices(grp, 'set')
        cals = {}
        for s in sets:
            cal = grp[set_name(s)].attrs.get('calibration') or \
                (attrs.get('calibrations') or {}).get(str(s))
            if cal and cal in root:
                cals[s] = cal
        if tuning not in root or len(cals) != len(sets) or not sets:
            if verbose:
                print(f'{name}: left as it is (no tuning or calibration '
                      'to move it to)')
            continue
        tune = root[tuning]
        record = dict(
            name = attrs.get('name'), notes = attrs.get('notes', ''),
            dec_stage = attrs.get('dec_stage'),
            duration_s = attrs.get('duration_s'), tuning = tuning,
            tones = attrs.get('tones'), resonators = attrs.get('resonators'),
            complete = bool(attrs.get('complete')), paths = [],
            moved_from = name)
        if len(sets) == 1:
            record.update(tone_set = sets[0], calibration = cals[sets[0]])
        else:
            record.update(tone_sets = sets,
                          calibrations = {str(s): c for s, c in cals.items()})
        new_n = register_stream_run(tune, record)
        if attrs.get('created'):
            update_stream_run(tune, new_n, created = attrs['created'])
        ts_name = set_name(new_n, TS_PREFIX)
        paths = []
        for s in sets:
            sg = grp[set_name(s)]
            cal = root[cals[s]]
            for w in list_indices(sg, 'write'):
                wname = set_name(w, 'write')
                new_w = set_name(next_index(cal, 'write'), 'write')
                src = os.path.join(path, name, set_name(s), wname)
                dst = os.path.join(path, *cal.path.split('/'), new_w)
                shutil.move(src, dst)
                stream_dir = os.path.join(dst, STREAM_NAME)
                if os.path.isdir(stream_dir):
                    os.rename(stream_dir, os.path.join(dst, ts_name))
                    ts = root[f'{cal.path}/{new_w}/{ts_name}']
                    ts.attrs.update(dict(
                        name = record['name'], notes = record['notes'],
                        dec_stage = record['dec_stage'],
                        duration_s = record['duration_s'], run = new_n))
                    paths.append(ts.path)
                if verbose:
                    print(f'{name}/{set_name(s)}/{wname} -> {cal.path}/{new_w}')
        update_stream_run(tune, new_n, paths = paths)
        shutil.rmtree(os.path.join(path, name))
        moved.append(name)
        if verbose:
            print(f'{name}: now {tuning}/streams run {new_n} ({len(paths)} '
                  'streams); the copied sweep deleted')
    return moved

################################################################################
################################## Porting #####################################
################################################################################
def port_tuning(src, tune, dst, cooldown = None, setup = None):
    """
    Copy a tuning to another store, without the data used to make it.

    Copies the tuning's ``tones/`` and ``crs_config/`` to the next free
    ``tune{k}`` of ``dst``, with the resonator list it uses: an identical
    list already in ``dst`` is reused, else it is copied to the next free
    ``resonators{j}``. Raw data and analysis are not
    copied, and ``src`` is not modified (see
    ``documents/procedures/tuning_timestreams.pdf``).

    Parameters:
    src (zarr.Group, str, or os.PathLike): source store (may be zipped).
    tune (int): tuning index in ``src``.
    dst (str or os.PathLike): destination store path; created if needed.
        An existing store keeps its own cooldown and setup.
    cooldown (str or None): ``dst``'s cooldown if it is created. None
        (default): the same as ``src``. Ignored if ``dst`` exists.
    setup (dict or None): ``dst``'s setup if it is created. None (default):
        a copy of ``src``'s. Ignored if ``dst`` exists.

    Returns:
    k (int): index of the new tuning in ``dst``.

    Raises:
    ValueError: if ``dst`` belongs to another device.
    KeyError: if the tuning has no ``tones`` or records no resonator
        list.
    """
    src_root = open_store(src)
    src_path = str(src) if not isinstance(src, zarr.Group) else \
        str(src.store)
    name = set_name(int(tune), 'tune')
    src_tune = src_root[name]
    if 'tones' not in src_tune:
        raise KeyError(f"{name} has no 'tones' to port")
    if 'resonators' not in src_tune.attrs:
        raise KeyError(f'{name} records no resonator list')
    src_res = src_tune.attrs['resonators']
    dst = os.path.expanduser(str(dst))
    if os.path.exists(os.path.join(dst, 'zarr.json')):
        # An existing store keeps its own cooldown and setup
        dst_root = open_store(dst, mode = 'a')
        if dst_root.attrs['device'] != src_root.attrs['device']:
            raise ValueError(
                f"store {dst} is for device {dst_root.attrs['device']!r}, "
                f"not {src_root.attrs['device']!r}")
    else:
        dst_root = create_store(
            dst, src_root.attrs['device'],
            src_root.attrs['cooldown'] if cooldown is None else cooldown,
            setup = dict(src_root.attrs.get('setup', {})) if setup is None
            else setup)
    res = load_resonators(src_root, src_res)
    dst_res = None
    for j in list_indices(dst_root, RESONATORS_PREFIX):
        old = load_resonators(dst_root, j)
        if all(np.array_equal(old[key], res[key], equal_nan = key not in
                              ('res_idxs_all', 'source_all'))
               for key in RESONATOR_ARRAYS):
            dst_res = set_name(j, RESONATORS_PREFIX)
            break
    if dst_res is None:
        dst_res = set_name(next_index(dst_root, RESONATORS_PREFIX),
                           RESONATORS_PREFIX)
        grp = create_group(dst_root, dst_res, 'decision', attrs = dict(
            source = src_root[src_res].attrs.get('source'),
            ported_from = dict(store = src_path, group = src_res,
                               time = _now())))
        for key in RESONATOR_ARRAYS:
            write_array(grp, key, res[key])
    ported = dict(store = src_path, group = name, time = _now(),
                  citkid_version = citkid_version)
    new_tune, k = new_tuning(dst_root, dst_res,
                             attrs = dict(ported_from = ported))
    tones = load_tones(src_tune['tones'])
    grp = create_group(new_tune, 'tones', 'decision', attrs = dict(
        ported_from = dict(ported, group = f'{name}/tones',
                           source = src_tune['tones'].attrs.get('source'))))
    save_tones(grp, **tones)
    if 'crs_config' in src_tune:
        write_tune_config(new_tune, src_tune['crs_config'].attrs['config'])
        new_tune['crs_config'].attrs['ported_from'] = to_jsonable(
            dict(ported, group = f'{name}/crs_config'))
    return k

################################################################################
################################## Archiving ###################################
################################################################################
def archive_store(store_path, zip_path = None, verify = True):
    """
    Zip a finished store into one file, readable without unpacking.

    The zip is uncompressed (zarr has already compressed the data) and
    read-only when opened with ``open_store``.

    Parameters:
    store_path (str or os.PathLike): the ``.zarr`` directory.
    zip_path (str, os.PathLike, or None): output path. None (default):
        ``store_path`` + '.zip'.
    verify (bool): If True (default), check that the zip opens and holds
        every file of the store.

    Returns:
    zip_path (str): the archive's path.

    Raises:
    FileExistsError: if ``zip_path`` exists.
    RuntimeError: if verification fails.
    """
    store_path = os.path.expanduser(str(store_path)).rstrip('/\\')
    zip_path = store_path + '.zip' if zip_path is None else \
        os.path.expanduser(str(zip_path))
    if os.path.exists(zip_path):
        raise FileExistsError(f'{zip_path} exists')
    names = []
    with zipfile.ZipFile(zip_path, 'w', compression = zipfile.ZIP_STORED,
                         allowZip64 = True) as zf:
        for dirpath, _, files in os.walk(store_path):
            for fname in sorted(files):
                full = os.path.join(dirpath, fname)
                arc = os.path.relpath(full, store_path).replace(os.sep, '/')
                zf.write(full, arc)
                names.append(arc)
    if verify:
        with zipfile.ZipFile(zip_path) as zf:
            missing = set(names) - set(zf.namelist())
            bad = zf.testzip()
        if missing or bad is not None:
            raise RuntimeError(f'archive check failed: missing {len(missing)}'
                               f' files, first bad file {bad!r}')
        open_store(zip_path)
    return zip_path

def tree(root, max_depth = None):
    """
    Return a text tree of a store's groups and their kinds.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.
    max_depth (int or None): deepest level shown. None (default): all.

    Returns:
    text (str): one line per group, indented by depth.
    """
    root = open_store(root)
    lines = []

    def walk(grp, depth):
        """
        Add a group's children to ``lines``.

        Parameters:
        grp (zarr.Group): group.
        depth (int): its depth.
        """
        if max_depth is not None and depth >= max_depth:
            return
        for name in sorted(grp.group_keys(), key = _natural):
            child = grp[name]
            kind = child.attrs.get('kind', '')
            lines.append(f"{'  ' * depth}{name}/" + (f'  [{kind}]'
                                                      if kind else ''))
            walk(child, depth + 1)

    walk(root, 0)
    return '\n'.join(lines)

def _natural(name):
    """
    Sort key that orders 'sweep2' before 'sweep10'.

    Parameters:
    name (str): group name.

    Returns:
    key (list): text and integer parts.
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]
