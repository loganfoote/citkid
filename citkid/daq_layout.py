"""
Layout of a device's DAQ store: one zarr store per device per cooldown,
organized as a tree that follows the measurement procedure (see
``documents/daq_layout.md``).

```
<device>.zarr/
├── vna/sweep{k}/          raw VNA sweeps (stitched over bands)
├── resonators/            resonator identities (res_idxs_all, ...)
└── tune{k}/               tunings: crs_config/, tones_init/, tones/, ...
    └── sweep{j}/write{w}/ts{s}/
```

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
TONE_ARRAYS = ('fres', 'qres', 'ares', 'res_idxs', 'tone_set')
RESONATOR_ARRAYS = ('res_idxs_all', 'fres_all', 'qres_all', 'Qr_all',
                    'source_all')
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
    name (str): group name (may be a path, e.g. 'vna/sweep0').
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
    Open a layout group, creating it if needed (e.g. the ``vna`` container
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
    saved config (e.g. the tone-set plan) are ignored.

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
    sweep_grp (zarr.Group): the raw ``vna/sweep{k}`` group, empty.
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
    vna = root['vna']
    k = latest_index(vna, 'sweep') if k is None else int(k)
    grp = vna[set_name(k, 'sweep')]
    f, z = grp['f'][:], grp['z'][:]
    if not bands:
        return f, z
    return f, z, {name: grp[name][:] for name in VNA_BAND_ARRAYS}

################################################################################
################################# Resonators ###################################
################################################################################
def add_resonators(root, fres, qres, source, Qr = None):
    """
    Append resonators to the store's identity table (``resonators/``),
    creating it if needed.

    New resonators get indices max + 1, max + 2, ...; existing indices are
    never changed, so ``res_idxs`` mean the same thing in every tuning.

    Parameters:
    root (zarr.Group): store root, writable.
    fres (array-like): new resonance frequencies in Hz.
    qres (array-like or float): span factors for cutting them out of gain
        fits, one per resonator or one for all.
    source (str): where they were found, e.g. 'vna/sweep0'.
    Qr (array-like or None): fitted total quality factors (NaN where the
        fit failed), used for linewidths (e.g. ``tone_plan.select_isolated``).
        None (default): all NaN.

    Returns:
    res_idxs (np.array, int32): indices of the new resonators.
    """
    fres = np.atleast_1d(np.asarray(fres, dtype = np.float64))
    qres = np.broadcast_to(np.asarray(qres, dtype = np.float64),
                           fres.shape).copy()
    Qr = np.full(fres.shape, np.nan) if Qr is None else \
        np.broadcast_to(np.asarray(Qr, dtype = np.float64), fres.shape).copy()
    if fres.ndim != 1:
        raise ValueError('fres must be 1D')
    old = load_resonators(root) if 'resonators' in root else None
    start = int(old['res_idxs_all'].max()) + 1 \
        if old is not None and len(old['res_idxs_all']) else 0
    new = dict(
        res_idxs_all = np.arange(start, start + len(fres), dtype = np.int32),
        fres_all = fres, qres_all = qres, Qr_all = Qr,
        source_all = np.full(len(fres), str(source)))
    if old is not None:
        merged = {k: np.concatenate((old[k], new[k]))
                  for k in RESONATOR_ARRAYS}
        attrs = dict(root['resonators'].attrs)
        del root['resonators']
        grp = root.create_group('resonators')
        grp.attrs.update(attrs)
    else:
        merged = new
        grp = create_group(root, 'resonators', 'decision')
    for name in RESONATOR_ARRAYS:
        write_array(grp, name, merged[name])
    return new['res_idxs_all']

def load_resonators(root):
    """
    Load the resonator identity table.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.

    Returns:
    resonators (dict): ``RESONATOR_ARRAYS`` by name (``source_all`` as
        str; ``Qr_all`` NaN if the table has none).
    """
    grp = open_store(root)['resonators']
    out = {name: grp[name][:] for name in RESONATOR_ARRAYS if name in grp}
    if 'Qr_all' not in out:
        out['Qr_all'] = np.full(len(out['fres_all']), np.nan)
    out['source_all'] = np.asarray(out['source_all'], dtype = str)
    return out

################################################################################
################################### Tones ######################################
################################################################################
def save_tones(
    grp, fres, qres, ares, res_idxs, tone_set = None, source = None
):
    """
    Save a tone list (``fres``, ``qres``, ``ares``, ``res_idxs``,
    ``tone_set``) in a group.

    Parameters:
    grp (zarr.Group): group, e.g. a new ``tones`` or ``tones_init`` group.
    fres (array-like): tone frequencies in Hz.
    qres (array-like): span factors.
    ares (array-like or float): tone powers in dBm.
    res_idxs (array-like int): resonator indices (negative for calibration
        tones).
    tone_set (array-like int or None): tone set of each tone. None
        (default): all in set 0.
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
    if any(v.shape != (n,) for v in arrays.values()):
        raise ValueError('fres, qres, ares, res_idxs and tone_set must '
                         'have the same length')
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
        group has none.
    """
    out = {name: grp[name][:] for name in TONE_ARRAYS if name in grp}
    if 'tone_set' not in out:
        out['tone_set'] = np.zeros(len(out['fres']), dtype = np.int32)
    return out

################################################################################
################################## Porting #####################################
################################################################################
def port_tuning(src, tune, dst, cooldown = None, setup = None):
    """
    Copy a tuning to another store, without the data used to make it.

    Copies ``resonators/``, the tuning's ``tones/`` and ``crs_config/`` to
    the next free ``tune{k}`` of ``dst``. Raw data and analysis are not
    copied, and ``src`` is not modified. See ``documents/daq_layout.md``,
    section 7.2.

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
    ValueError: if ``dst`` belongs to another device, or its resonators
        differ from ``src``'s.
    KeyError: if the tuning has no ``tones``.
    """
    src_root = open_store(src)
    src_path = str(src) if not isinstance(src, zarr.Group) else \
        str(src.store)
    name = set_name(int(tune), 'tune')
    src_tune = src_root[name]
    if 'tones' not in src_tune:
        raise KeyError(f"{name} has no 'tones' to port")
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
    res = load_resonators(src_root)
    if 'resonators' in dst_root:
        old = load_resonators(dst_root)
        same = (np.array_equal(old['res_idxs_all'], res['res_idxs_all'])
                and np.array_equal(old['fres_all'], res['fres_all']))
        if not same:
            raise ValueError(
                "dst's resonators differ from src's: the res_idxs would "
                "mean different things")
    else:
        grp = create_group(dst_root, 'resonators', 'decision',
                           attrs = dict(ported_from = dict(
                               store = src_path, group = 'resonators',
                               time = _now())))
        for key in RESONATOR_ARRAYS:
            write_array(grp, key, res[key])
    ported = dict(store = src_path, group = name, time = _now(),
                  citkid_version = citkid_version)
    new_tune, k = new_indexed_group(dst_root, 'tune', 'decision',
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
