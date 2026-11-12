"""
Load the timestreams of one tone set from several runs, one after another,
into one DataSet, for the 'ts' (and 'ts_offres') calibrations of a
``citkid.daq_layout`` store; then apply the calibration to whole runs.

A source is one calibration sweep (``tune{k}/cal{j}``, one tone set, see
``crs.procedures.stream_tone_sets``) with one run's streams under it:
every write of the run (``write{w}``; at fast decimation stages a tone
set streams in several writes, each with some of its tones), the stream's
name under them (``ts{n}``), and a window of samples
(``daq_layout.select_streams`` makes them). Every source must hold the
same tones (the same ``res_idxs``: one tone set of one tuning). The DataSet's rows are those
tones; each row's sweep is the first source's, and its timestream is its
windows from every source, one after another in source order (the global
``source_bounds`` gives each source's samples in it). So the calibration
is done once, on everything supplied, and ``apply_stream_calibration``
then applies it to each whole run. The per-row ``write_idx`` says which
tones were streamed together in the first source (e.g. for common-mode
removal).

The sources may be from different decimation stages: ``dt`` is their
common sample period, or NaN if they differ. The calibration takes no
spectra (the 'ts_cal' path): a run's come from its applied x
(``apply_stream_calibration``) and its own ``dt``. Paths are inside the
store (same-store mode), so a DataSet keeps working when the store moves
or is zipped.
"""
import warnings
import numpy as np
import zarr

from .. import daq_layout as dl
from .dataset import DataSet
from .framework import plStep

# Which tones a DataSet holds
ROWS = ('resonators', 'calibration', 'all')

def stream_sources_steps(
    store, sources, fres_all, qres_all, rows = 'resonators', f_avg = 1
):
    """
    Return the calibration steps that load one tone set's streams from
    several sources, concatenated.

    The steps are load_global_data (``fres_all``, ``qres_all``, ``dt``,
    ``nrows``, ``source_bounds``), load_global_res_data (``fres``,
    ``qres``, ``ares``, ``res_idxs``, ``write_idx``), load_sweep_f and
    load_sweep_g (each row's sweep, from the first source), and load_ft
    and load_zt (each row's tone frequency, and its windows from every
    source one after another). The data is opened on first use.

    Parameters:
    store (zarr.Group): root of the store.
    sources (list of dict): one per source, in the order their windows
        are concatenated: ``group`` (the calibration sweep, e.g.
        'ts3/set0'), ``writes`` (write names under it, e.g. ['write0',
        'write1']), ``start`` and ``stop`` (sample window of the streams),
        and optionally ``stream`` (stream name, default 'stream'). Every
        source must hold the same tones (``res_idxs``).
    fres_all, qres_all (str): keys of every resonance's frequency and span
        factor in the store (e.g. 'resonators0/fres_all'), for cutting
        neighbors out of the fits.
    rows (str): tones to load: 'resonators' (default, res_idxs >= 0),
        'calibration' (res_idxs < 0, for the 'ts_offres' calibration), or
        'all'.
    f_avg (int): adjacent fine sweep points to average. Default 1.

    Returns:
    steps (list of plStep): the calibration steps.

    Raises:
    ValueError: if the parameters are invalid.
    """
    if rows not in ROWS:
        raise ValueError(f'rows must be one of {ROWS}')
    if not len(sources):
        raise ValueError('give at least one source')
    for s in sources:
        if not {'group', 'writes', 'start', 'stop'} <= set(s):
            raise ValueError('each source needs group, writes, start, stop')
        if int(s['stop']) <= int(s['start']):
            raise ValueError('each source needs stop > start')
    if int(f_avg) != f_avg or f_avg < 1:
        raise ValueError('f_avg must be a positive integer')
    f_avg = int(f_avg)
    cache = {}

    def table():
        """
        Map the rows to their sweep rows and, per source, to their
        writes (once).

        Returns:
        table (dict): ``sweep_row`` (row in the sweeps, per row);
            ``write`` and ``ts_row`` (per source and row: index into the
            source's writes, and row in that write); ``dt`` (the common
            sample period, NaN if the sources differ); ``source_bounds``
            ((n_sources, 2) start and stop of each source's samples in
            the concatenated streams); ``nsamples`` (their total).

        Raises:
        ValueError: if the sources hold different tones, a source's
            writes have different sample periods, or a row is in no
            write of a source.
        """
        if 'rows' in cache:
            return cache['rows']
        sweep0 = store[sources[0]['group']]
        res_idxs0 = np.asarray(sweep0['res_idxs'][...])
        keep = np.ones(len(res_idxs0), bool) if rows == 'all' else \
            (res_idxs0 >= 0 if rows == 'resonators' else res_idxs0 < 0)
        sweep_rows = np.flatnonzero(keep)
        write = np.full((len(sources), len(sweep_rows)), -1, dtype = np.int64)
        ts_row = np.full_like(write, -1)
        dts, bounds, offset, ft0 = [], [], 0, None
        for i, s in enumerate(sources):
            sweep = store[s['group']]
            res_idxs = np.asarray(sweep['res_idxs'][...])
            if not np.array_equal(res_idxs, res_idxs0):
                raise ValueError(
                    f"{s['group']} holds other tones than "
                    f"{sources[0]['group']}: every source must be the same "
                    'tone set of the same tuning')
            cal0 = sweep0.attrs.get('calibration', sources[0]['group'])
            cal = sweep.attrs.get('calibration', s['group'])
            if cal != cal0:
                warnings.warn(
                    f"{s['group']}'s streams were taken with calibration "
                    f"{cal}, {sources[0]['group']}'s with {cal0}: the first "
                    "source's sweep calibrates every stream")
            where = np.full((len(res_idxs), 2), -1)
            ft = np.full(len(res_idxs), np.nan)
            stream = s.get('stream', 'stream')
            src_dts = set()
            for w, name in enumerate(s['writes']):
                wg = sweep[name]
                held = np.asarray(wg['sweep_rows'][...]) \
                    if 'sweep_rows' in wg else np.arange(len(res_idxs))
                where[held, 0] = w
                where[held, 1] = np.arange(len(held))
                ft[held] = np.asarray(wg['fres'][...])
                src_dts.add(round(float(wg[f'{stream}/dt'][...]), 15))
            if len(src_dts) != 1:
                raise ValueError(f"{s['group']}: its writes have different "
                                 f'sample periods {sorted(src_dts)}')
            dts.append(src_dts.pop())
            missing = sweep_rows[where[sweep_rows, 0] < 0]
            if len(missing):
                raise ValueError(f"{s['group']}: tones {missing.tolist()} "
                                 f"are in none of the writes {s['writes']}")
            write[i] = where[sweep_rows, 0]
            ts_row[i] = where[sweep_rows, 1]
            if ft0 is None:
                ft0 = ft[sweep_rows]
            elif not np.allclose(ft[sweep_rows], ft0, atol = 1., rtol = 0):
                warnings.warn(f"{s['group']} streamed other tone "
                              f"frequencies than {sources[0]['group']}: "
                              "the first source's are used")
            n = int(s['stop']) - int(s['start'])
            bounds.append((offset, offset + n))
            offset += n
        cache['rows'] = dict(
            sweep_row = sweep_rows, write = write, ts_row = ts_row,
            dt = dts[0] if len(set(dts)) == 1 else float('nan'),
            source_bounds = np.asarray(bounds, dtype = np.int64),
            nsamples = offset)
        return cache['rows']

    def sweep_of(row):
        """
        Return a row's sweep group (the first source's) and its row in it.

        Parameters:
        row (int): DataSet row.

        Returns:
        sweep (zarr.Group): the calibration sweep.
        r (int): the tone's row in it.
        """
        t = table()
        return store[sources[0]['group']], int(t['sweep_row'][row])

    def read_all(key):
        """
        Read a store array as floats, sorted by frequency.

        Parameters:
        key (str): the array's key.

        Returns:
        f, q (np.ndarray): see ``load_global_data``.
        """
        return np.asarray(store[key][...], dtype = float)

    def load_global_data():
        """
        Load the band's resonances, the sample period, nrows, and the
        sources' bounds.

        Returns:
        fres_all (np.ndarray): every resonance frequency.
        qres_all (np.ndarray): their span factors.
        dt (float): timestream sample period (NaN if the sources differ).
        nrows (int): number of rows.
        source_bounds (np.ndarray): (n_sources, 2) start and stop of each
            source's samples in the concatenated streams.
        """
        t = table()
        f, q = read_all(fres_all), read_all(qres_all)
        # Gain cuts need the resonances in frequency order
        order = np.argsort(f, kind = 'stable')
        return (f[order], q[order], t['dt'], len(t['sweep_row']),
                t['source_bounds'].copy())

    def load_global_res_data():
        """
        Load every row's tone data (from the first source's sweep).

        Returns:
        fres, qres, ares, res_idxs (np.ndarray): tone frequencies, span
            factors, powers and resonator indices.
        write_idx (np.ndarray): each row's write in the first source
            (index into its writes).
        """
        t = table()
        sweep = store[sources[0]['group']]
        return (*(np.asarray(sweep[k][...])[t['sweep_row']]
                  for k in ('fres', 'qres', 'ares', 'res_idxs')),
                t['write'][0].copy())

    def load_sweep_f(data_idx):
        """
        Load one row's fine sweep, averaged by ``f_avg`` and sorted.

        Parameters:
        data_idx (int): row.

        Returns:
        ff (np.ndarray): frequencies, ascending.
        zf (np.ndarray): complex S21.
        """
        sweep, r = sweep_of(int(data_idx))
        ff = np.asarray(sweep['fine_sweep/f'][r])
        zf = np.asarray(sweep['fine_sweep/z'][r])
        if f_avg > 1:
            n = len(ff) // f_avg * f_avg
            ff = ff[:n].reshape(-1, f_avg).mean(axis = 1)
            zf = zf[:n].reshape(-1, f_avg).mean(axis = 1)
        idx = np.argsort(ff)
        return ff[idx], zf[idx]

    def load_sweep_g(data_idx):
        """
        Load one row's gain sweep, sorted by frequency.

        Parameters:
        data_idx (int): row.

        Returns:
        fg (np.ndarray): frequencies, ascending.
        zg (np.ndarray): complex S21.
        """
        sweep, r = sweep_of(int(data_idx))
        fg = np.asarray(sweep['gain_sweep/f'][r])
        zg = np.asarray(sweep['gain_sweep/z'][r])
        idx = np.argsort(fg)
        return fg[idx], zg[idx]

    def by_write(data_idx, i):
        """
        Group rows by the write that streamed them in a source.

        Parameters:
        data_idx (np.ndarray): rows.
        i (int): the source.

        Yields:
        positions (np.ndarray): positions in ``data_idx``.
        write (zarr.Group): the write group.
        ts_rows (np.ndarray): the rows in the write.
        """
        t = table()
        s = sources[i]
        writes = t['write'][i][data_idx]
        for w in np.unique(writes):
            pos = np.flatnonzero(writes == w)
            write = store[s['group']][s['writes'][int(w)]]
            yield pos, write, t['ts_row'][i][data_idx][pos]

    def load_ft(data_idx):
        """
        Load the rows' tone frequencies as streamed (in the first source).

        Parameters:
        data_idx (np.ndarray): rows.

        Returns:
        ft (np.ndarray): tone frequencies.
        """
        idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int64))
        ft = np.empty(len(idx))
        for pos, write, ts_rows in by_write(idx, 0):
            ft[pos] = np.asarray(write['fres'][...])[ts_rows]
        return ft if np.ndim(data_idx) else ft[0]

    def load_zt(data_idx):
        """
        Load the rows' timestreams: their windows from every source, one
        after another, as S21.

        Parameters:
        data_idx (np.ndarray): rows.

        Returns:
        zt (np.ndarray): complex S21, shape (len(data_idx), nsamples).
        """
        t = table()
        idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int64))
        zt = np.empty((len(idx), t['nsamples']), dtype = np.complex128)
        for i, s in enumerate(sources):
            a, b = t['source_bounds'][i]
            for pos, write, ts_rows in by_write(idx, i):
                stream = write[s.get('stream', 'stream')]
                o = np.argsort(ts_rows)
                z = stream['z'][:, ts_rows[o], int(s['start']):int(s['stop'])]
                scale = np.asarray(stream['counts_to_s21'][...])[ts_rows[o]]
                zt[pos[o], a:b] = (z[0] + 1j * z[1]) * scale[:, None]
        return zt if np.ndim(data_idx) else zt[0]

    return [
        plStep('load_global_data', load_global_data, [],
               ['fres_all', 'qres_all', 'dt', 'nrows', 'source_bounds'],
               'global'),
        plStep('load_global_res_data', load_global_res_data, [],
               ['fres', 'qres', 'ares', 'res_idxs', 'write_idx'],
               'global-res'),
        plStep('load_sweep_f', load_sweep_f, ['data_idx'], ['ff', 'zf'],
               'per-row'),
        plStep('load_sweep_g', load_sweep_g, ['data_idx'], ['fg', 'zg'],
               'per-row'),
        plStep('load_ft', load_ft, ['data_idx'], ['ft'], 'vectorized'),
        plStep('load_zt', load_zt, ['data_idx'], ['zt'], 'vectorized'),
    ]

def stream_sources_source(sources, fres_all, qres_all, rows = 'resonators',
                          f_avg = 1):
    """
    Return source code defining ``custom_cal_steps`` with
    ``stream_sources_steps``, for ``DataSet(custom_source=...)`` in
    same-store mode (``dataset_store`` is the store the DataSet lives in).

    Parameters:
    sources, fres_all, qres_all, rows, f_avg: see ``stream_sources_steps``.

    Returns:
    source (str): the source code.

    Raises:
    TypeError: if a source holds something other than strings, numbers,
        or lists of strings.
    """
    clean = []
    for s in sources:
        c = {}
        for k, v in s.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, (list, tuple)):
                v = [str(x) for x in v]
            elif not isinstance(v, (str, int, float)):
                raise TypeError(f'source {k} must be a string, number, or '
                                'list of strings')
            c[k] = v
        clean.append(c)
    return '\n'.join([
        'from citkid.pipeline.stream_sources import stream_sources_steps',
        '',
        '# dataset_store is the root of the store this DataSet lives in',
        'custom_cal_steps = stream_sources_steps(',
        '    store = dataset_store,',
        f'    sources = {clean!r},',
        f'    fres_all = {fres_all!r},',
        f'    qres_all = {qres_all!r},',
        f'    rows = {rows!r},',
        f'    f_avg = {int(f_avg)!r},',
        ')',
        '',
    ])

def open_stream_dataset(
    out_grp, sources, fres_all, qres_all, rows = 'resonators', f_avg = 1,
    cal_yaml_path = None, write_buffer = False
):
    """
    Open (or reopen) one DataSet of several sources' streams, stored in a
    group of the same store.

    Parameters:
    out_grp (zarr.Group): group for the DataSet, in the store holding the
        sources.
    sources, fres_all, qres_all, rows, f_avg: see ``stream_sources_steps``.
    cal_yaml_path (str or None): calibration YAML path or alias. None
        (default): 'ts_offres' for calibration tones, else 'ts_cal' (the
        'ts' calibration without its spectra, which the concatenated
        streams don't have; a run's come from
        ``apply_stream_calibration`` and its own ``dt``).
    write_buffer (bool): passed to DataSet. Default False.

    Returns:
    DS (DataSet): the DataSet.
    """
    if cal_yaml_path is None:
        cal_yaml_path = 'ts_offres' if rows == 'calibration' else 'ts_cal'
    return DataSet(
        zarr_path = out_grp, cal_yaml_path = cal_yaml_path,
        custom_source = stream_sources_source(sources, fres_all, qres_all,
                                              rows = rows, f_avg = f_avg),
        write_buffer = write_buffer)

def apply_stream_calibration(
    DS, store, source, out_grp, outputs = ('xt',), overwrite = False,
    verbose = True
):
    """
    Apply a calibrated DataSet to a whole stream of its tones, chunk by
    chunk: every row's calibration outputs (e.g. ``xt``) over the
    source's window, with the source's sample period.

    Parameters:
    DS (DataSet): the tone set's DataSet (``open_stream_dataset``), with
        its analysis done for the rows (``AnalysisRunner`` on the 'ts'
        path: gain, circle and x calibration).
    store (zarr.Group): the store root.
    source (dict): the run's tone set, as ``daq_layout.select_streams``
        makes it (``group``, ``writes``, ``start``, ``stop``, ``stream``),
        e.g. ``select_streams(root, name)[0][0]`` for a whole run. Its
        tones must be the DataSet's.
    out_grp (zarr.Group): group to save in: per output an array (nrows,
        nsamples) chunked along time as the stream, plus ``dt``,
        ``res_idxs`` (per row), and attrs ``source``, ``outputs``,
        ``dataset`` (the DataSet's path) and ``complete``.
    outputs (sequence of str): calibration outputs, one value per sample
        (``xt``, ``thetat``, ``At``, ...). Default ('xt',).
    overwrite (bool): If True, redo a group already complete. Default
        False: it is returned as is.
    verbose (bool): If True (default), show a progress bar.

    Returns:
    out_grp (zarr.Group): the group.

    Raises:
    ValueError: if the source holds other tones than the DataSet, or an
        output isn't one value per sample.
    """
    from tqdm.auto import tqdm
    outputs = [str(o) for o in outputs]
    if out_grp.attrs.get('complete') and not overwrite:
        return out_grp
    for name in list(out_grp.keys()):
        del out_grp[name]
    sweep = store[source['group']]
    res_idxs = np.asarray(sweep['res_idxs'][...])
    ds_res = np.asarray(DS.res_idxs[:])
    pos = {int(r): i for i, r in enumerate(res_idxs)}
    try:
        sweep_rows = np.array([pos[int(r)] for r in ds_res], dtype = np.int64)
    except KeyError as e:
        raise ValueError(f"{source['group']} has no tone {e}: it must hold "
                         "the DataSet's tones") from None
    start, stop = int(source['start']), int(source['stop'])
    nsamples, stream_name = stop - start, source.get('stream', 'stream')
    arrays, dt = {}, None
    writes = []
    for name in source['writes']:
        wg = sweep[name]
        held = np.asarray(wg['sweep_rows'][...]) if 'sweep_rows' in wg \
            else np.arange(len(res_idxs))
        where = {int(r): k for k, r in enumerate(held)}
        ds_rows = np.array([i for i, r in enumerate(sweep_rows)
                            if int(r) in where], dtype = np.int64)
        if not len(ds_rows):
            continue
        ts_rows = np.array([where[int(sweep_rows[i])] for i in ds_rows])
        stream = wg[stream_name]
        this_dt = float(stream['dt'][...])
        if dt is None:
            dt = this_dt
            chunk = int(stream['z'].chunks[2])
            for o in outputs:
                arrays[o] = out_grp.create_array(
                    o, shape = (len(ds_res), nsamples), dtype = np.float64,
                    chunks = (len(ds_res), chunk), fill_value = np.nan)
        writes.append((wg, stream, ds_rows, ts_rows))
    if dt is None:
        raise ValueError(f"{source['group']} has no stream of the DataSet's "
                         'tones')
    steps = [(wg, stream, ds_rows, ts_rows, a, min(a + chunk, stop))
             for wg, stream, ds_rows, ts_rows in writes
             for a in range(start, stop, chunk)]
    for wg, stream, ds_rows, ts_rows, a, b in tqdm(
            steps, desc = 'Applying', leave = False, disable = not verbose):
        o = np.argsort(ts_rows)
        z = stream['z'][:, ts_rows[o], a:b]
        scale = np.asarray(stream['counts_to_s21'][...])[ts_rows[o]]
        zt = np.empty((len(ds_rows), b - a), dtype = np.complex128)
        zt[o] = (z[0] + 1j * z[1]) * scale[:, None]
        ft = np.asarray(wg['fres'][...])[ts_rows]
        out = DS.apply_cal(ds_rows, outputs, replacements = dict(
            zt = zt, ft = ft, dt = dt))
        for name in outputs:
            value = np.atleast_2d(np.asarray(out[name], dtype = np.float64))
            if value.shape != (len(ds_rows), b - a):
                raise ValueError(f'{name} is not one value per sample '
                                 f'(shape {value.shape})')
            arrays[name][ds_rows, a - start:b - start] = value
    out_grp.create_array('dt', data = np.array(dt))
    out_grp.create_array('res_idxs', data = ds_res)
    out_grp.attrs.update(dict(
        source = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                  for k, v in source.items()},
        outputs = outputs, dataset = DS.root.path, complete = True))
    return out_grp

def plan_stream_datasets(root, selection):
    """
    Resolve a choice of streams and group them into DataSets: one per tone
    set (and resonator list), with the chosen runs' windows of that set
    one after another, in the order of ``selection``.

    Parameters:
    root (zarr.Group): the store root.
    selection (list of dict): choices of streams, each the keyword
        arguments of ``daq_layout.select_streams`` (e.g. 05's
        ``CALIBRATE``).

    Returns:
    groups (dict): DataSet name (e.g. 'set0'): dict(``sources``,
        ``entries``, ``resonators``, ``tone_set``), in order of first use.
        ``entries`` has one dict per source: ``entry`` (its index in
        ``selection``), ``run`` ('ts{n}'), ``name``, ``start``, ``stop``
        (samples), ``source`` (its calibration sweep's group).

    Raises:
    ValueError: if an entry selects no tone sets.
    """
    groups = {}
    for i, entry in enumerate(selection):
        sources, run = dl.select_streams(root, **entry)
        if not sources:
            raise ValueError(f'entry {i} ({entry}) selects no tone sets')
        res = run.get('resonators')
        for src in sources:
            s = int(root[src['group']].attrs['tone_set'])
            name = f'set{s}'
            if any(g['resonators'] != res for g in groups.values()) or \
                    name in groups and groups[name]['resonators'] != res:
                name = f'{name}_{res}'
            g = groups.setdefault(name, dict(sources = [], entries = [],
                                             resonators = res, tone_set = s))
            g['sources'].append(src)
            g['entries'].append(dict(
                entry = i, run = run['group'], name = run.get('name'),
                start = run['start'], stop = run['stop'],
                source = src['group']))
    return groups

def open_calibration(root, cal):
    """
    Reopen the DataSets of a stream calibration (``ts_cal{c}``, made by
    05_calibrate_streams.py), e.g. to apply them to whole runs
    (``apply_stream_calibration``).

    Parameters:
    root (zarr.Group): the store root.
    cal (zarr.Group or str): the calibration group, or its name.

    Returns:
    datasets (dict): DataSet name: dict(``DS`` (the DataSet), ``entries``,
        ``resonators``, ``tone_set`` (``plan_stream_datasets``)).
        Calibration-tone DataSets are not included.

    Raises:
    KeyError: if the group records no selection of streams.
    """
    cal = root[cal] if isinstance(cal, str) else cal
    if 'selection' not in cal.attrs:
        raise KeyError(f'{cal.path} records no selection of streams')
    out = {}
    for name, g in plan_stream_datasets(root, cal.attrs['selection']).items():
        res = g['resonators']
        grp = cal.require_group(name)
        DS = open_stream_dataset(
            grp, g['sources'], f'{res}/fres_all', f'{res}/qres_all',
            f_avg = int(grp.attrs.get('f_avg', 1)))
        out[name] = dict(DS = DS, entries = g['entries'], resonators = res,
                         tone_set = g['tone_set'])
    return out
