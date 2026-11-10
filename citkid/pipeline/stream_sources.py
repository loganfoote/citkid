"""
Load timestreams from several sources, one after another, into one
DataSet, for the 'ts' (and 'ts_offres') calibrations of a
``citkid.daq_layout`` store.

A source is one calibration sweep of a tuning (``tune{k}/sweep{j}``, one
tone set of one run, see ``crs.procedures.stream_tone_sets``) with its
streams: every write under it (``write{w}/ts0``; at fast decimation stages
a tone set streams in several writes, each with some of its tones) and a
window of samples. The DataSet's rows are the sources' tones in order:
source 0's tones, then source 1's, and so on. Each row loads its sweep
from its source and its timestream from the write that streamed it, and
the per-row ``source_idx`` and ``write_idx`` say which (tones streamed in
the same write were taken together, e.g. for common-mode removal).

Every source of a DataSet must have the same sample period and window
length. Paths are inside the store (same-store mode), so a DataSet keeps
working when the store moves or is zipped.
"""
import numpy as np

from .dataset import DataSet
from .framework import plStep

# Which tones a DataSet holds
ROWS = ('resonators', 'calibration', 'all')

def stream_sources_steps(
    store, sources, fres_all, qres_all, rows = 'resonators', f_avg = 1
):
    """
    Return the calibration steps that load the tones of several sources.

    The steps are load_global_data (``fres_all``, ``qres_all``, ``dt``,
    ``nrows``), load_global_res_data (``fres``, ``qres``, ``ares``,
    ``res_idxs``, ``source_idx``, ``write_idx``), load_sweep_f and
    load_sweep_g (each row's sweep, from its source), and load_ft and
    load_zt (each row's tone frequency and timestream, from the write that
    streamed it). The data is opened on first use.

    Parameters:
    store (zarr.Group): root of the store.
    sources (list of dict): one per source: ``group`` (the calibration
        sweep, e.g. 'tune0/sweep3'), ``writes`` (write names under it, e.g.
        ['write0', 'write1']), ``start`` and ``stop`` (sample window of the
        streams), and optionally ``stream`` (stream name, default 'ts0').
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
        Map the rows to their sources and writes (once).

        Returns:
        table (dict): per row: ``source``, ``sweep_row``, ``write`` (index
            into the source's writes), ``ts_row`` (row in that write); and
            the sample period ``dt``.

        Raises:
        ValueError: if the sources' sample periods or window lengths
            differ, or a row is in no write.
        """
        if 'rows' in cache:
            return cache['rows']
        out = {k: [] for k in ('source', 'sweep_row', 'write', 'ts_row')}
        dts, lengths = set(), set()
        for i, s in enumerate(sources):
            sweep = store[s['group']]
            res_idxs = np.asarray(sweep['res_idxs'][...])
            keep = np.ones(len(res_idxs), bool) if rows == 'all' else \
                (res_idxs >= 0 if rows == 'resonators' else res_idxs < 0)
            where = np.full((len(res_idxs), 2), -1)
            stream = s.get('stream', 'ts0')
            for w, name in enumerate(s['writes']):
                write = sweep[name]
                held = np.asarray(write['sweep_rows'][...]) \
                    if 'sweep_rows' in write else np.arange(len(res_idxs))
                where[held, 0] = w
                where[held, 1] = np.arange(len(held))
                dts.add(round(float(write[f'{stream}/dt'][...]), 15))
            lengths.add(int(s['stop']) - int(s['start']))
            for r in np.flatnonzero(keep):
                if where[r, 0] < 0:
                    raise ValueError(f"{s['group']}: tone {r} is in none of "
                                     f"the writes {s['writes']}")
                out['source'].append(i)
                out['sweep_row'].append(r)
                out['write'].append(where[r, 0])
                out['ts_row'].append(where[r, 1])
        if len(dts) != 1:
            raise ValueError(f'the sources have different sample periods '
                             f'{sorted(dts)}: use one DataSet per '
                             'decimation stage')
        if len(lengths) != 1:
            raise ValueError(f'the sources have different window lengths '
                             f'{sorted(lengths)}')
        cache['rows'] = {k: np.asarray(v, dtype = np.int64)
                         for k, v in out.items()}
        cache['rows']['dt'] = dts.pop()
        return cache['rows']

    def sweep_of(row):
        """
        Return a row's sweep group and its row in it.

        Parameters:
        row (int): DataSet row.

        Returns:
        sweep (zarr.Group): the calibration sweep.
        r (int): the tone's row in it.
        """
        t = table()
        return store[sources[t['source'][row]]['group']], \
            int(t['sweep_row'][row])

    def load_global_data():
        """
        Load the band's resonances, the sample period, and nrows.

        Returns:
        fres_all (np.ndarray): every resonance frequency.
        qres_all (np.ndarray): their span factors.
        dt (float): timestream sample period.
        nrows (int): number of rows.
        """
        t = table()
        f = np.asarray(store[fres_all][...], dtype = float)
        q = np.asarray(store[qres_all][...], dtype = float)
        # Gain cuts need the resonances in frequency order
        order = np.argsort(f, kind = 'stable')
        return f[order], q[order], t['dt'], len(t['source'])

    def load_global_res_data():
        """
        Load every row's tone data.

        Returns:
        fres, qres, ares, res_idxs (np.ndarray): tone frequencies, span
            factors, powers and resonator indices.
        source_idx, write_idx (np.ndarray): each row's source (index into
            the sources) and write (index into its source's writes).
        """
        t = table()
        cols = {k: [] for k in ('fres', 'qres', 'ares', 'res_idxs')}
        for i, s in enumerate(sources):
            sel = t['source'] == i
            sweep = store[s['group']]
            for k in cols:
                cols[k].append(np.asarray(sweep[k][...])[t['sweep_row'][sel]])
        return (*(np.concatenate(cols[k]) for k in cols), t['source'].copy(),
                t['write'].copy())

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

    def by_write(data_idx):
        """
        Group rows by the write that streamed them.

        Parameters:
        data_idx (np.ndarray): rows.

        Yields:
        positions (np.ndarray): positions in ``data_idx``.
        write (zarr.Group): the write group.
        ts_rows (np.ndarray): the rows in the write.
        source (dict): the source.
        """
        t = table()
        key = t['source'][data_idx] * 100000 + t['write'][data_idx]
        for k in np.unique(key):
            pos = np.flatnonzero(key == k)
            s = sources[int(k // 100000)]
            write = store[s['group']][s['writes'][int(k % 100000)]]
            yield pos, write, t['ts_row'][data_idx][pos], s

    def load_ft(data_idx):
        """
        Load the rows' tone frequencies as streamed.

        Parameters:
        data_idx (np.ndarray): rows.

        Returns:
        ft (np.ndarray): tone frequencies.
        """
        idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int64))
        ft = np.empty(len(idx))
        for pos, write, ts_rows, _ in by_write(idx):
            ft[pos] = np.asarray(write['fres'][...])[ts_rows]
        return ft if np.ndim(data_idx) else ft[0]

    def load_zt(data_idx):
        """
        Load the rows' timestreams in their sources' windows, as S21.

        Parameters:
        data_idx (np.ndarray): rows.

        Returns:
        zt (np.ndarray): complex S21, shape (len(data_idx), window).
        """
        idx = np.atleast_1d(np.asarray(data_idx, dtype = np.int64))
        s0 = sources[0]
        zt = np.empty((len(idx), int(s0['stop']) - int(s0['start'])),
                      dtype = np.complex128)
        for pos, write, ts_rows, s in by_write(idx):
            stream = write[s.get('stream', 'ts0')]
            o = np.argsort(ts_rows)
            z = stream['z'][:, ts_rows[o], int(s['start']):int(s['stop'])]
            scale = np.asarray(stream['counts_to_s21'][...])[ts_rows[o]]
            zt[pos[o]] = (z[0] + 1j * z[1]) * scale[:, None]
        return zt if np.ndim(data_idx) else zt[0]

    return [
        plStep('load_global_data', load_global_data, [],
               ['fres_all', 'qres_all', 'dt', 'nrows'], 'global'),
        plStep('load_global_res_data', load_global_res_data, [],
               ['fres', 'qres', 'ares', 'res_idxs', 'source_idx',
                'write_idx'], 'global-res'),
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
        (default): 'ts_offres' for calibration tones, else 'ts'.
    write_buffer (bool): passed to DataSet. Default False.

    Returns:
    DS (DataSet): the DataSet.
    """
    if cal_yaml_path is None:
        cal_yaml_path = 'ts_offres' if rows == 'calibration' else 'ts'
    return DataSet(
        zarr_path = out_grp, cal_yaml_path = cal_yaml_path,
        custom_source = stream_sources_source(sources, fres_all, qres_all,
                                              rows = rows, f_avg = f_avg),
        write_buffer = write_buffer)
