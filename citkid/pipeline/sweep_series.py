"""
Load and analyze a series of target sweeps, such as a power sweep or a
blackbody sweep.

Each series point is one ``citkid.crs.procedures.target_sweep`` saved to its
own group, named ``set{idx}`` (``citkid.zarr_util.set_name``), with
``fine_sweep`` and ``gain_sweep`` subgroups and optionally a timestream
group (e.g. ``ts_00``). ``citkid.crs.procedures.power_sweep`` writes this
format. The timestream group is written by ``CRS.capture_ts``, or by
``CRS.write_tones_for_stream`` with the stream in a subgroup of it
(``ts_stream``), named the same in every set.

Loading
-------
``target_sweep_steps`` returns the calibration steps that load one set.
``target_sweep_source`` returns the same steps as a few lines of source code,
which a DataSet embeds (``DataSet(custom_source=...)``), so the DataSet can be
reopened later without a custom steps file. ``open_series_datasets`` opens
one such DataSet per set, for ``run_iq_series(datasets=...)`` or
``run_series_analysis``.

The data is found either through a directory and store path (``main_dir``
mode, for data saved in separate stores) or in the same store as the
DataSets (same-store mode, for the DAQ layout of ``citkid.daq_layout``,
where the analysis lives next to the data): there, every path is a group
inside that store, so nothing needs updating when the store moves.

Fits and tuning
---------------
``a_vs_power_fit`` and ``responsivity_fit`` are ready-made ``SeriesXYFit``
fits for ``run_iq_series``. ``pick_optimal_ares`` picks each resonator's
optimal tone power from the fitted nonlinearity parameters, and ``save_popt``
saves the result.
"""

import os
import re

import numpy as np
import zarr
from tqdm.auto import tqdm

from .analysis import AnalysisRunner
from .dataset import DataSet
from .framework import plStep
from .series_xy_fit import SeriesXYFit
from ..zarr_util import list_sets, set_name

# Index of the nonlinearity parameter a in iq_popt
_A_IDX = 4
# An array given as '<store>.zarr/<key>'
_ARRAY_SPEC = re.compile(r'^(.*?\.zarr)[/\\](.+)$')
# Stands in for the store when same-store parameters are only validated
_SAME_STORE_CHECK = object()

################################################################################
################################### Loading ####################################
################################################################################

def target_sweep_steps(
    data_path = None, group = '', main_dir = None, fres_all = None,
    qres_all = None, ts = None, ts_stream = None, ts_nsamps = None, f_avg = 1,
    store = None,
):
    """
    Return the calibration steps that load one target sweep.

    The steps are load_global_data (fres_all, qres_all, nrows, and dt if
    ``ts`` is given), load_global_res_data (fres, qres, ares, res_idxs),
    load_sweep_f (ff, zf) and load_sweep_g (fg, zg), sorted by frequency, and
    load_ft and load_zt if ``ts`` is given. The data is opened on first use.

    Parameters:
    data_path (str or None): path to the zarr store holding the target
        sweep. Relative paths are relative to ``main_dir``. Give either
        ``data_path`` or ``store``.
    group (str): group holding the target sweep inside the store, e.g.
        'set3' or 'tune0/power_sweep/set3'. '' (default) for the store
        root.
    main_dir (str or None): directory that relative paths are relative to.
        None uses the current directory.
    fres_all (str or None): array of every resonance frequency in the band,
        used to cut adjacent resonances from the fits. Either
        '<store>.zarr/<key>' (e.g. 'daq_prep.zarr/fres_all'; the store is
        relative to ``main_dir``) or '<key>' for an array in the root of
        ``data_path``. None (default) uses the swept tones' fres.
    qres_all (str, float, or None): span factor of each resonance in
        ``fres_all``: an array given like ``fres_all``, or one number for
        every resonance. None (default) uses the swept tones' qres, and is
        only allowed if ``fres_all`` is None.
    ts (str or None): name of the timestream group inside ``group`` (e.g.
        'ts_00'), for the 'ts' and 'ts_offres' calibrations. It holds the
        tones ('fres'). None (default) loads no timestream.
    ts_stream (str or None): subgroup of ``ts`` holding the stream ('z',
        'counts_to_s21', 'dt'), e.g. 'ts_0' when several streams were taken
        with the same tones (``CRS.write_tones_for_stream`` then
        ``CRS.stream`` into subgroups). None (default) uses ``ts`` itself,
        as written by ``CRS.capture_ts``. Only allowed if ``ts`` is given.
    ts_nsamps (int or None): number of timestream samples to load, from the
        start. None (default) loads all of them.
    f_avg (int): number of adjacent fine sweep points to average together.
        Leftover points at the end are dropped. Default 1 (no averaging).
    store (zarr.Group or None): root group of the store holding the target
        sweep, instead of ``data_path`` (same-store mode). ``fres_all`` and
        ``qres_all`` are then keys inside this store. Default None.

    Returns:
    steps (list of plStep): the calibration steps.

    Raises:
    ValueError: If the parameters are invalid.
    """
    if (data_path is None) == (store is None):
        raise ValueError('give exactly one of data_path and store')
    if store is not None:
        if main_dir is not None:
            raise ValueError('main_dir is only used with data_path')
        for spec in (fres_all, qres_all):
            if isinstance(spec, str) and _ARRAY_SPEC.match(spec):
                raise ValueError(
                    'with store, fres_all and qres_all must be keys inside '
                    "the store (e.g. 'resonators/fres_all')")
    if fres_all is not None and qres_all is None:
        raise ValueError('qres_all is required if fres_all is given')
    if int(f_avg) != f_avg or f_avg < 1:
        raise ValueError('f_avg must be a positive integer')
    if ts_nsamps is not None and (int(ts_nsamps) != ts_nsamps or ts_nsamps < 1):
        raise ValueError('ts_nsamps must be a positive integer or None')
    if ts_stream is not None and ts is None:
        raise ValueError('ts_stream is only allowed if ts is given')
    f_avg = int(f_avg)
    stream = ts if ts_stream is None else f'{ts}/{ts_stream}'
    ts_slice = slice(None, None if ts_nsamps is None else int(ts_nsamps))
    if data_path is not None:
        data_path = _resolve_path(data_path, main_dir)
    opened = {}

    def grp():
        """
        Open the target sweep group on first use.
        """
        if 'grp' not in opened:
            root = store if store is not None else \
                zarr.open_group(data_path, mode = 'r')
            opened['root'] = root
            opened['grp'] = root[group] if group else root
        return opened['grp']

    def read_array(spec):
        """
        Read an array given as '<store>.zarr/<key>' or '<key>'.
        """
        match = _ARRAY_SPEC.match(spec)
        if match is None:
            grp()
            return np.asarray(opened['root'][spec][...])
        store = zarr.open_group(_resolve_path(match.group(1), main_dir),
                                mode = 'r')
        return np.asarray(store[match.group(2)][...])

    def load_global_data():
        """
        Load the band's resonances, nrows, and dt if ``ts`` is given.

        Returns:
        fres_all (np.ndarray): every resonance frequency in the band.
        qres_all (np.ndarray): span factor of each resonance in ``fres_all``.
        dt (float): timestream sample period. Only returned if ``ts`` is
            given.
        nrows (int): number of tones.
        """
        g = grp()
        nrows = g['fres'].shape[0]
        if fres_all is None:
            fres_all_ = np.asarray(g['fres'][...], dtype = float)
            qres_all_ = np.asarray(g['qres'][...], dtype = float)
        else:
            fres_all_ = np.asarray(read_array(fres_all), dtype = float)
            if isinstance(qres_all, str):
                qres_all_ = np.asarray(read_array(qres_all), dtype = float)
            else:
                qres_all_ = np.full_like(fres_all_, float(qres_all))
        if ts is None:
            return fres_all_, qres_all_, nrows
        dt = g[f'{stream}/dt'][...]
        return fres_all_, qres_all_, dt, nrows

    def load_global_res_data():
        """
        Load the per-tone resonance data of every tone at once.

        Returns:
        fres, qres, ares, res_idxs (np.ndarray): tone frequencies, span
            factors, powers and resonator indices.
        """
        g = grp()
        fres = g['fres'][...]
        qres = g['qres'][...]
        ares = g['ares'][...]
        res_idxs = g['res_idxs'][...]
        return fres, qres, ares, res_idxs

    def load_sweep_f(data_idx):
        """
        Load one tone's fine sweep, averaged by ``f_avg`` and sorted.

        Parameters:
        data_idx (int): tone index.

        Returns:
        ff (np.ndarray): fine sweep frequencies, ascending.
        zf (np.ndarray): complex fine sweep S21.
        """
        g = grp()
        ff = np.asarray(g['fine_sweep/f'][data_idx])
        zf = np.asarray(g['fine_sweep/z'][data_idx])
        if f_avg > 1:
            n = len(ff) // f_avg * f_avg
            ff = ff[:n].reshape(-1, f_avg).mean(axis = 1)
            zf = zf[:n].reshape(-1, f_avg).mean(axis = 1)
        idx = np.argsort(ff)
        return ff[idx], zf[idx]

    def load_sweep_g(data_idx):
        """
        Load one tone's gain sweep, sorted by frequency.

        Parameters:
        data_idx (int): tone index.

        Returns:
        fg (np.ndarray): gain sweep frequencies, ascending.
        zg (np.ndarray): complex gain sweep S21.
        """
        g = grp()
        fg = np.asarray(g['gain_sweep/f'][data_idx])
        zg = np.asarray(g['gain_sweep/z'][data_idx])
        idx = np.argsort(fg)
        return fg[idx], zg[idx]

    def load_ft(data_idx):
        """
        Load the timestream tone frequencies.

        Parameters:
        data_idx (np.ndarray): tone indices.

        Returns:
        ft (np.ndarray): tone frequencies.
        """
        return grp()[f'{ts}/fres'][data_idx]

    def load_zt(data_idx):
        """
        Load the timestreams, converted from counts to S21.

        Parameters:
        data_idx (np.ndarray): tone indices.

        Returns:
        zt (np.ndarray): complex S21 timestreams, shape
            (len(data_idx), nsamps).
        """
        g = grp()
        zt = g[f'{stream}/z'][:, data_idx, ts_slice]
        zt = zt[0] + 1j * zt[1]
        zt *= np.asarray(
            g[f'{stream}/counts_to_s21'][data_idx])[:, np.newaxis]
        return zt

    global_names = ['fres_all', 'qres_all', 'nrows']
    if ts is not None:
        global_names.insert(2, 'dt')
    steps = [
        plStep('load_global_data', load_global_data, [], global_names,
               'global'),
        plStep('load_global_res_data', load_global_res_data, [],
               ['fres', 'qres', 'ares', 'res_idxs'], 'global-res'),
        plStep('load_sweep_f', load_sweep_f, ['data_idx'], ['ff', 'zf'],
               'per-row'),
        plStep('load_sweep_g', load_sweep_g, ['data_idx'], ['fg', 'zg'],
               'per-row'),
    ]
    if ts is not None:
        steps += [
            plStep('load_ft', load_ft, ['data_idx'], ['ft'], 'vectorized'),
            plStep('load_zt', load_zt, ['data_idx'], ['zt'], 'vectorized'),
        ]
    return steps

def target_sweep_source(
    main_dir = None, data_path = None, group = '', fres_all = None,
    qres_all = None, ts = None, ts_stream = None, ts_nsamps = None, f_avg = 1,
    same_store = False,
):
    """
    Return source code defining ``custom_cal_steps`` with
    ``target_sweep_steps``, for ``DataSet(custom_source=...)``.

    The DataSet embeds the source, so it can be reopened with only its zarr
    path. In ``main_dir`` mode, ``main_dir`` is a top-level assignment in
    the source, so it can be replaced with ``custom_main_dir_overwrite``
    after the data moves. In same-store mode, the source reads the data
    from ``dataset_store``, the root of the store the DataSet lives in
    (provided by the DataSet when it runs the source), so nothing needs
    replacing.

    Parameters:
    main_dir (str or None): directory the paths are relative to. None in
        same-store mode.
    data_path (str or None): store path (see ``target_sweep_steps``). None
        in same-store mode.
    group, fres_all, qres_all, ts, ts_stream, ts_nsamps, f_avg:
        see ``target_sweep_steps``. In same-store mode, ``group``,
        ``fres_all`` and ``qres_all`` are paths inside the store.
    same_store (bool): If True, use same-store mode. Default False.

    Returns:
    source (str): the source code.

    Raises:
    TypeError: If a parameter isn't a string, number, or None.
    ValueError: If the parameters are invalid (see target_sweep_steps).
    """
    if same_store and (main_dir is not None or data_path is not None):
        raise ValueError('main_dir and data_path must be None in '
                         'same-store mode')
    if not same_store and (main_dir is None or data_path is None):
        raise ValueError('main_dir and data_path are required unless '
                         'same_store is True')
    params = {
        'data_path': data_path, 'group': group, 'fres_all': fres_all,
        'qres_all': qres_all, 'ts': ts, 'ts_nsamps': ts_nsamps,
        'f_avg': f_avg,
    }
    if same_store:
        del params['data_path']
    # Only written if given, so existing embedded sources stay unchanged
    if ts_stream is not None:
        params['ts_stream'] = ts_stream
    for name, value in params.items():
        if isinstance(value, np.generic):
            params[name] = value = value.item()
        if value is not None and not isinstance(value, (str, int, float)):
            raise TypeError(f'{name} must be a string, number, or None')
    # Build the steps once so invalid parameters fail here, not on load
    # (the data is only opened when a step runs)
    if same_store:
        target_sweep_steps(store = _SAME_STORE_CHECK, **params)
        return '\n'.join([
            'from citkid.pipeline.sweep_series import target_sweep_steps',
            '',
            '# Same-store mode: dataset_store is the root of the store this',
            '# DataSet lives in, provided by the DataSet when it runs this',
            '# source',
            'custom_cal_steps = target_sweep_steps(',
            '    store = dataset_store,',
            *(f'    {name} = {value!r},' for name, value in params.items()),
            ')',
            '',
        ])
    target_sweep_steps(main_dir = main_dir, **params)
    lines = [
        'from citkid.pipeline.sweep_series import target_sweep_steps',
        '',
        f'main_dir = {str(main_dir)!r}',
        '',
        'custom_cal_steps = target_sweep_steps(',
        '    main_dir = main_dir,',
        *(f'    {name} = {value!r},' for name, value in params.items()),
        ')',
        '',
    ]
    return '\n'.join(lines)

def open_series_datasets(
    main_dir = None, data_path = None, out_root = None, set_idxs = None,
    cal_yaml_path = 'iq', prefix = 'set', write_buffer = False, group = '',
    **loader_kwargs,
):
    """
    Open one DataSet per set of a target sweep series.

    The DataSet of set i is stored in ``out_root/set{i}``, and embeds its
    loading steps (see ``target_sweep_source``). Reopening with the same
    parameters loads the existing results.

    Two modes:
        ``main_dir`` mode (``main_dir`` and ``data_path`` given): the sets
            are in the store ``data_path``, relative to ``main_dir``.
            ``main_dir`` may differ from the first time, e.g. after the raw
            data moves.
        Same-store mode (``main_dir`` and ``data_path`` None): the sets are
            in the same store as ``out_root`` (e.g. ``tune0/power_sweep``
            and ``tune0/power_fits`` in a ``citkid.daq_layout`` store).
            Every path is inside that store, so nothing needs updating when
            it moves.

    Parameters:
    main_dir (str or None): directory the data paths are relative to. None
        for same-store mode.
    data_path (str or None): path of the zarr store holding the sets,
        relative to ``main_dir``. None for same-store mode.
    out_root (zarr.Group or str): group (or path of a zarr store) to save
        the DataSets in. In ``main_dir`` mode, keeping it separate from the
        raw data store is recommended; in same-store mode it must be a
        group of the store holding the sets.
    set_idxs (list of int or None): indices of the sets to open, in series
        order. None (default) opens every set in the store, in index order.
    cal_yaml_path (str): calibration YAML path or alias. Default 'iq'.
    prefix (str): set group name prefix. Default 'set'. Use e.g. 'amp' for
        data saved before the 'set' convention.
    write_buffer (bool): passed to DataSet. Default False.
    group (str): group holding the sets inside the data store, e.g.
        'tune0/power_sweep'. '' (default) for the store root.
    **loader_kwargs: fres_all, qres_all, ts, ts_stream, ts_nsamps, f_avg.
        See ``target_sweep_steps``. ``ts`` and ``ts_stream`` are the same
        for every set. In same-store mode, ``fres_all`` and ``qres_all``
        are keys inside the store (e.g. 'resonators/fres_all').

    Returns:
    datasets (list of DataSet): one DataSet per set.

    Raises:
    ValueError: If no sets are found, or the sets' res_idxs differ (rows
        must be the same resonators in every set).
    """
    if out_root is None:
        raise ValueError('out_root is required')
    same_store = main_dir is None and data_path is None
    if not same_store and (main_dir is None or data_path is None):
        raise ValueError('give both main_dir and data_path, or neither '
                         '(same-store mode)')
    if isinstance(out_root, str):
        out_root = zarr.open_group(out_root, mode = 'a')
    if set_idxs is None:
        if same_store:
            data_root = zarr.open_group(out_root.store, path = '',
                                        mode = 'r')
        else:
            data_root = zarr.open_group(_resolve_path(data_path, main_dir),
                                        mode = 'r')
        series = data_root[group] if group else data_root
        set_idxs = list_sets(series, prefix = prefix)
    if not len(set_idxs):
        where = group or data_path or 'the store root'
        raise ValueError(f"no '{prefix}' groups found in {where}")
    datasets = []
    for set_idx in set_idxs:
        name = set_name(set_idx, prefix = prefix)
        set_group = f'{group}/{name}' if group else name
        source = target_sweep_source(main_dir, data_path, set_group,
                                     same_store = same_store,
                                     **loader_kwargs)
        datasets.append(DataSet(
            zarr_path = out_root.require_group(name),
            cal_yaml_path = cal_yaml_path,
            custom_source = source,
            write_buffer = write_buffer,
        ))
    res_idxs = np.asarray(datasets[0].res_idxs[:])
    for set_idx, DS in zip(set_idxs, datasets):
        if not np.array_equal(np.asarray(DS.res_idxs[:]), res_idxs):
            raise ValueError(
                f"res_idxs of {set_name(set_idx, prefix = prefix)} differ "
                f"from {set_name(set_idxs[0], prefix = prefix)}")
    return datasets

def run_series_analysis(
    datasets, analysis_yaml_path = 'iq', data_idxs = None, verbose = True,
    **execute_kwargs,
):
    """
    Run the full analysis path on every DataSet, saving the results.

    Parameters:
    datasets (list of DataSet): one DataSet per series point.
    analysis_yaml_path (str): analysis YAML path or alias. Default 'iq'.
    data_idxs (array-like or None): rows to analyze, e.g.
        ``np.where(datasets[0].res_idxs[:] >= 0)[0]`` for on-resonance tones
        only. None (default) analyzes every row.
    verbose (bool): If True, shows progress bars.
    **execute_kwargs: other ``AnalysisRunner.execute_path`` parameters, e.g.
        path_per_row.

    Returns:
    runners (list of AnalysisRunner): the runners, one per DataSet.
    """
    runners = []
    for DS in tqdm(datasets, desc = 'Series analysis', leave = False,
                   disable = not verbose):
        AR = AnalysisRunner(DS, analysis_yaml_path = analysis_yaml_path)
        AR.execute_path(data_idx = data_idxs, verbose = verbose,
                        **execute_kwargs)
        runners.append(AR)
    return runners

################################################################################
################################# Series fits ##################################
################################################################################

def a_vs_power_fit(
    a_max = 0.8, a_min = 0, name = 'a_vs_power', group = None
):
    """
    Return a fit of the nonlinearity parameter a versus tone power, linear in
    tone amplitude: a = a_slope * 10 ** (P / 20) + a_offset, with P in dBm.

    Points from the first one with a > a_max on are left out, since the IQ
    fit is unreliable past bifurcation. Points before the first one with
    a >= a_min are left out too (e.g. low powers where a is too small to
    measure). Points are in order of increasing power.

    Parameters:
    a_max (float): highest a to include in the fit. Default 0.8.
    a_min (float): the fit starts at the first point with a >= a_min.
        Default 0 keeps the low-power points (a is never below 0).
    name (str): fit name. Default 'a_vs_power'.
    group (zarr.Group or None): group to save the fits to. None (default)
        uses ``root/xy_fit`` in ``run_iq_series``.

    Returns:
    xy_fit (SeriesXYFit): fit for ``run_iq_series(x_param_name = 'ares',
        y_param_name = 'a', xy_fit = ...)``. Parameters [a_slope, a_offset];
        NaN if fewer than 2 points are left. Its inverse is ``ares_for_a``,
        so with ``y_target`` the window marks the power at which each
        resonator's fit reaches the target a (see ``run_iq_series``).

    Raises:
    ValueError: If ``a_min`` is not below ``a_max``.
    """
    if a_min >= a_max:
        raise ValueError(f'a_min ({a_min}) must be below a_max ({a_max})')

    def fit(x, y):
        """
        Fit a line to a vs tone amplitude over the range a_min to a_max.

        Parameters:
        x (np.ndarray): tone powers in dBm, ascending.
        y (np.ndarray): nonlinearity parameter a at each power.

        Returns:
        popt (np.ndarray or list): [a_slope, a_offset], NaN if fewer than 2
            points are in range.
        """
        above = np.where(y > a_max)[0]
        end = above[0] if len(above) else len(y)
        past_min = np.where(y >= a_min)[0]
        start = past_min[0] if len(past_min) else len(y)
        x, y = x[start:end], y[start:end]
        if len(x) < 2:
            return [np.nan, np.nan]
        return np.polyfit(10 ** (x / 20), y, 1)

    def model(x, a_slope, a_offset):
        """
        Evaluate the linear a vs tone amplitude model.

        Parameters:
        x (np.ndarray): tone powers in dBm.
        a_slope, a_offset (float): fit parameters.

        Returns:
        a (np.ndarray): model nonlinearity parameter.
        """
        return a_slope * 10 ** (x / 20) + a_offset

    return SeriesXYFit(fit = fit, param_names = ['a_slope', 'a_offset'],
                       model = model, name = name, group = group,
                       inverse = ares_for_a)

def ares_for_a(popt, a_target):
    """
    Return the tone power at which an ``a_vs_power_fit`` reaches a target
    nonlinearity parameter.

    Parameters:
    popt (array-like): fitted parameters, shape (2,) or (nrows, 2), e.g.
        ``root['xy_fit/popt'][...]``.
    a_target (float): target a.

    Returns:
    ares (float or np.ndarray): tone power in dBm, shape popt.shape[:-1].
        NaN where the fit is NaN or never reaches a_target.
    """
    popt = np.asarray(popt, dtype = float)
    a_slope, a_offset = popt[..., 0], popt[..., 1]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        amplitude = (a_target - a_offset) / a_slope
        ares = np.where(amplitude > 0, 20 * np.log10(amplitude), np.nan)
    return ares[()] if ares.ndim == 0 else ares

def responsivity_fit(name = 'x_vs_power', group = None):
    """
    Return a fit of resonance frequency versus optical power to the
    integrated responsivity (citkid.responsivity.funcs.responsivity_int_x0):
    fr = f1 * (1 + x(P; R0, P0, x0)), where f1 is fr at the lowest power.

    The fit uses guesses, which can be edited in the series window.

    Parameters:
    name (str): fit name. Default 'x_vs_power'.
    group (zarr.Group or None): group to save the fits to. None (default)
        uses ``root/xy_fit`` in ``run_iq_series``.

    Returns:
    xy_fit (SeriesXYFit): fit for ``run_iq_series(x_values = P,
        y_param_name = 'fr', xy_fit = ...)``, with P in W. Parameters
        [resp_R0, resp_P0, resp_x0, resp_f1].
    """
    from ..responsivity.fitter import fit_responsivity_int_x0
    from ..responsivity.funcs import responsivity_int_x0

    def fit(x, y, p0):
        """
        Fit the integrated responsivity to fr vs optical power.

        Parameters:
        x (np.ndarray): optical powers in W, ascending.
        y (np.ndarray): resonance frequencies.
        p0 (np.ndarray or None): guess [R0, P0, x0, f1] (f1 is ignored), or
            None to make one.

        Returns:
        popt (list): fitted [R0, P0, x0, f1].
        p0 (list): guess used [R0, P0, x0, f1].
        """
        y0 = y[0]
        guess = None if p0 is None else p0[:3]
        p0_used, popt, perr, f0, f0err = fit_responsivity_int_x0(
            x, (y - y0) / y0, y0, guess = guess
        )
        return [*popt, y0], [*p0_used, y0]

    def model(P, R0, P0, x0, f1):
        """
        Evaluate fr at optical power P from the integrated responsivity.

        Parameters:
        P (np.ndarray): optical powers in W.
        R0, P0, x0, f1 (float): fit parameters.

        Returns:
        fr (np.ndarray): model resonance frequencies.
        """
        return f1 * (1 + responsivity_int_x0(P, R0, P0, x0))

    return SeriesXYFit(
        fit = fit, param_names = ['resp_R0', 'resp_P0', 'resp_x0', 'resp_f1'],
        model = model, name = name, group = group, uses_guess = True,
    )

################################################################################
################################ Power tuning ##################################
################################################################################

def pick_optimal_ares(datasets, a_max = 0.5):
    """
    Pick each resonator's optimal tone power from a power sweep's IQ fits.

    For each resonator, the series points are ordered by power, and the
    first point with a >= a_max is taken as bifurcated. Of the points below
    it, the one with the largest a is picked. Points without a saved fit, or
    marked bad (NaN), are skipped. Resonators with no usable point get NaN:
    replace those (and adjust any others) before using the powers.

    Parameters:
    datasets (list of DataSet): one DataSet per power, with saved iq_popt
        (e.g. from run_iq_series). Powers can differ between resonators.
    a_max (float): bifurcation threshold on a. Default 0.5.

    Returns:
    ares_opt (np.ndarray): optimal tone power of each resonator in dBm, NaN
        if there is no usable point.
    opt_idxs (np.ndarray): index into ``datasets`` of each resonator's
        optimal point, -1 if there is none.
    """
    aress, a = _series_ares_and_a(datasets)
    nrows = aress.shape[1]
    ares_opt = np.full(nrows, np.nan)
    opt_idxs = np.full(nrows, -1, dtype = int)
    for di in range(nrows):
        order = np.argsort(aress[:, di], kind = 'stable')
        a_di = a[order, di]
        bifurcated = np.where(a_di >= a_max)[0]
        cutoff = bifurcated[0] if len(bifurcated) else len(a_di)
        valid = np.isfinite(a_di) & (np.arange(len(a_di)) < cutoff)
        if not np.any(valid):
            continue
        opt_idx = order[np.where(valid, a_di, -np.inf).argmax()]
        ares_opt[di] = aress[opt_idx, di]
        opt_idxs[di] = opt_idx
    return ares_opt, opt_idxs

def save_popt(group, datasets, ares_opt, opt_idxs):
    """
    Save the optimal powers and the fine sweeps at the optimal points.

    Saves res_idxs, ares_opt, opt_idxs, and ff_opt and zf_opt (each
    resonator's calibrated fine sweep at its optimal point; NaN where
    opt_idxs is -1). Existing arrays are overwritten.

    Parameters:
    group (zarr.Group): group to save to.
    datasets (list of DataSet): the DataSets passed to pick_optimal_ares.
    ares_opt (array-like): optimal powers, possibly adjusted after
        pick_optimal_ares.
    opt_idxs (array-like): optimal points from pick_optimal_ares.
    """
    ares_opt = np.asarray(ares_opt, dtype = float)
    opt_idxs = np.asarray(opt_idxs, dtype = int)
    ff_opt, zf_opt = None, None
    for di, opt_idx in enumerate(opt_idxs):
        if opt_idx < 0:
            continue
        ff = np.asarray(datasets[opt_idx].ff[di])
        zf = np.asarray(datasets[opt_idx].zf[di])
        if ff_opt is None:
            ff_opt = np.full((len(opt_idxs), len(ff)), np.nan)
            zf_opt = np.full((len(opt_idxs), len(ff)), np.nan,
                             dtype = complex)
        ff_opt[di], zf_opt[di] = ff, zf
    arrays = {
        'res_idxs': np.asarray(datasets[0].res_idxs[:]),
        'ares_opt': ares_opt,
        'opt_idxs': opt_idxs,
    }
    if ff_opt is not None:
        arrays.update(ff_opt = ff_opt, zf_opt = zf_opt)
    for name, data in arrays.items():
        group.create_array(name, data = data, overwrite = True)

################################################################################
################################### Helpers ####################################
################################################################################

def _resolve_path(path, main_dir):
    """
    Join a relative path to main_dir.

    Parameters:
    path (str): path.
    main_dir (str or None): base directory, or None to leave path as is.

    Returns:
    path (str): the resolved path.
    """
    if main_dir is None or os.path.isabs(path):
        return path
    return os.path.join(main_dir, path)

def _series_ares_and_a(datasets):
    """
    Read the tone powers and saved nonlinearity parameters of a series.

    Parameters:
    datasets (list of DataSet): one DataSet per series point, all with the
        same number of rows.

    Returns:
    aress (np.ndarray): tone powers, shape (n_series, nrows).
    a (np.ndarray): fitted a, shape (n_series, nrows); NaN without a saved
        fit.
    """
    nrows = int(datasets[0].nrows)
    aress = np.full((len(datasets), nrows), np.nan)
    a = np.full((len(datasets), nrows), np.nan)
    for si, DS in enumerate(datasets):
        if int(DS.nrows) != nrows:
            raise ValueError('all datasets must have the same number of rows')
        aress[si] = np.asarray(DS.ares[:], dtype = float)
        rows = np.where(DS.saved_row_mask('iq_popt'))[0]
        if len(rows):
            a[si, rows] = np.asarray(DS.iq_popt[rows], dtype = float)[:, _A_IDX]
    return aress, a
