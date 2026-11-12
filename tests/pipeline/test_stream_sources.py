"""
Tests for ``citkid.pipeline.stream_sources``: loading one tone set's
streams from several runs, concatenated, into one DataSet, calibrating
them, and applying the calibration to whole runs.
"""
import asyncio
import numpy as np
import pytest
from citkid import daq_layout as dl
from citkid.crs.procedures import stream_tone_sets
from citkid.pipeline.analysis import AnalysisRunner
from citkid.pipeline.stream_sources import (
    apply_stream_calibration, open_stream_dataset, stream_sources_steps
)
from ..crs.test_streaming import make_crs, make_tuning


@pytest.fixture(scope = 'module')
def streamed(tmp_path_factory):
    """
    A simulated tuning with a stage-6 run and a chunked stage-0 run of
    both tone sets (each set calibrated anew as the runs switch to it),
    then a stage-6 run of tone set 1 alone, in the same frame (and with
    the same calibration) as the stage-0 run's.

    Returns:
    root (zarr.Group): store root.
    tune (zarr.Group): the tuning.
    """
    tmp = tmp_path_factory.mktemp('streams')
    crs = make_crs()

    async def take():
        """
        Take the runs.

        Returns:
        root, tune: see the fixture.
        """
        await crs.attach()
        root, tune = make_tuning(tmp, crs)
        kw = dict(npoints_fine = 200, npoints_gain = 20, verbose = False)
        await stream_tone_sets(crs, tune, 'slow', 2., 6, **kw)
        await stream_tone_sets(crs, tune, 'fast', 0.05, 0, **kw)
        await stream_tone_sets(crs, tune, 'slow2', 2., 6, set_idxs = [1],
                               calibrations = {1: 'tune0/cal3'}, **kw)
        assert dl.stream_runs(root)[2]['calibration'] == 'tune0/cal3'
        return root, tune

    return asyncio.run(take())


def source_of(root, run, tone_set, start, stop):
    """
    Make the source of one tone set of one run: its sweep with all its
    writes and a sample window.

    Parameters:
    root (zarr.Group): the store root.
    run (int): run index (ts{run}).
    tone_set (int): the tone set.
    start, stop (int): sample window.

    Returns:
    source (dict): the source.
    """
    out = None
    for path in dl.stream_runs(root)[run]['paths']:
        sweep, write, stream = path.rsplit('/', 2)
        if int(root[sweep].attrs['tone_set']) != tone_set:
            continue
        if out is None:
            out = dict(group = sweep, writes = [], start = start,
                       stop = stop, stream = stream)
        out['writes'].append(write)
    return out


def test_rows_follow_sources_and_writes(streamed):
    root, tune = streamed
    tone_set = tune['tones/tone_set'][:]
    n1 = int(np.sum(tone_set == 1))
    # Stage 0: tone set 1 streamed in two writes (one per module)
    src = [source_of(root, 1, 1, 10, 30)]
    assert len(src[0]['writes']) == 2
    steps = {s.name: s for s in stream_sources_steps(
        root, src, 'resonators0/fres_all', 'resonators0/qres_all',
        rows = 'all')}
    fres_all, qres_all, dt, nrows, bounds = steps['load_global_data'].func()
    assert nrows == n1 and np.isclose(dt, 1 / 38147.0, rtol = 1e-3)
    np.testing.assert_array_equal(bounds, [[0, 20]])
    assert np.all(np.diff(fres_all) >= 0)
    # An out-of-order saved list is sorted on load, qres along with it
    root.create_array('fres_bad', data = np.array([2e9, 1e9, 3e9]))
    root.create_array('qres_bad', data = np.array([2e3, 1e3, 3e3]))
    steps = {s.name: s for s in stream_sources_steps(
        root, src, 'fres_bad', 'qres_bad', rows = 'all')}
    f, q, _, _, _ = steps['load_global_data'].func()
    np.testing.assert_array_equal(f, [1e9, 2e9, 3e9])
    np.testing.assert_array_equal(q, [1e3, 2e3, 3e3])
    fres, qres, ares, res_idxs, write_idx = \
        steps['load_global_res_data'].func()
    np.testing.assert_array_equal(res_idxs, np.flatnonzero(tone_set == 1))
    # Each row's write holds it: one per module
    assert set(write_idx.tolist()) == {0, 1}
    mods = tune['tones/tone_module'][:][tone_set == 1]
    assert all(len(set(mods[write_idx == w].tolist())) == 1 for w in (0, 1))
    zt = steps['load_zt'].func(np.arange(n1))
    assert zt.shape == (n1, 20) and np.all(np.isfinite(zt))
    ft = steps['load_ft'].func(np.arange(n1))
    np.testing.assert_allclose(ft, fres, atol = 100)
    # Rows of different writes in one call, out of order
    rows = np.array([n1 - 1, 0, n1 // 2 + 1, n1 // 2])
    np.testing.assert_array_equal(steps['load_zt'].func(rows), zt[rows])
    ff, zf = steps['load_sweep_f'].func(n1 - 1)
    assert len(ff) == 200 and np.all(np.diff(ff) > 0)


def test_sources_concatenate(streamed):
    """
    Several sources of the same tones give one row per tone, with their
    windows one after another; stages may differ (dt is then NaN).
    """
    root, tune = streamed
    n1 = int(np.sum(tune['tones/tone_set'][:] == 1))
    a, b = source_of(root, 0, 1, 0, 500), source_of(root, 0, 1, 700, 800)
    c = source_of(root, 1, 1, 10, 30)
    steps = {s.name: s for s in stream_sources_steps(
        root, [a, b], 'resonators0/fres_all', 'resonators0/qres_all')}
    _, _, dt, nrows, bounds = steps['load_global_data'].func()
    assert nrows == n1 and np.isclose(dt, 1 / 596.046, rtol = 1e-3)
    np.testing.assert_array_equal(bounds, [[0, 500], [500, 600]])
    zt = steps['load_zt'].func(np.array([0, 3]))
    assert zt.shape == (2, 600)
    one = {s.name: s for s in stream_sources_steps(
        root, [a], 'resonators0/fres_all', 'resonators0/qres_all')}
    np.testing.assert_array_equal(zt[:, :500],
                                  one['load_zt'].func(np.array([0, 3])))
    # The single-row form, and a source from another stage
    np.testing.assert_array_equal(steps['load_zt'].func(3), zt[1])
    mixed = {s.name: s for s in stream_sources_steps(
        root, [a, c], 'resonators0/fres_all', 'resonators0/qres_all')}
    # Taken under different calibrations (the stage-0 run calibrated set
    # 1 anew): allowed, with a warning
    with pytest.warns(UserWarning, match = 'calibration'):
        _, _, dt, _, bounds = mixed['load_global_data'].func()
    assert np.isnan(dt)
    np.testing.assert_array_equal(bounds, [[0, 500], [500, 520]])
    assert mixed['load_zt'].func(np.arange(n1)).shape == (n1, 520)


def test_other_tones_refused(streamed):
    root, tune = streamed
    src = [source_of(root, 0, 0, 0, 20), source_of(root, 0, 1, 0, 20)]
    steps = {s.name: s for s in stream_sources_steps(
        root, src, 'resonators0/fres_all', 'resonators0/qres_all')}
    with pytest.raises(ValueError, match = 'same tone set'):
        steps['load_global_data'].func()
    with pytest.raises(ValueError, match = 'stop > start'):
        stream_sources_steps(root, [dict(src[0], stop = 0)], 'a', 'b')


def test_dataset_calibrates_and_applies(streamed):
    root, tune = streamed
    n1 = int(np.sum(tune['tones/tone_set'][:] == 1))
    # The stage-6 and stage-0 runs taken with the same calibration
    src = [source_of(root, 2, 1, 0, 1000), source_of(root, 1, 1, 0, 1000)]
    out = dl.require_group(tune, 'ts_cal0', 'analysis')
    DS = open_stream_dataset(out.require_group('set1'), src,
                             'resonators0/fres_all', 'resonators0/qres_all')
    assert len(np.asarray(DS.res_idxs)) == n1
    np.testing.assert_array_equal(np.asarray(DS.source_bounds),
                                  [[0, 1000], [1000, 2000]])
    AR = AnalysisRunner(DS, analysis_yaml_path = 'ts')
    rows = [0, n1 - 1]
    AR.execute_path(data_idx = rows, verbose = False)
    for r in rows:
        x = np.asarray(AR.DS.xt[r])
        assert x.shape == (2000,) and np.isfinite(x).all()
        # Dark: x is noise around a small offset, more of it per sample
        # at stage 0 (the second source)
        assert np.std(x[:1000]) < 1e-5 and np.std(x[1000:]) < 1e-4
    # Reopens from the store alone
    DS2 = open_stream_dataset(out.require_group('set1'), src,
                              'resonators0/fres_all', 'resonators0/qres_all')
    np.testing.assert_allclose(np.asarray(DS2.xt[rows[0]]),
                               np.asarray(AR.DS.xt[rows[0]]))
    # Applied to the whole stage-6 run: x of every tone over all of it,
    # equal to the DataSet's on the window it loaded
    AR.execute_remaining(verbose = False)
    whole, run = dl.select_streams(root, run = 2, tone_sets = [1])
    g = dl.require_group(out, 'set1_apply', 'analysis')
    ap = apply_stream_calibration(DS, root, whole[0],
                                  dl.require_group(g, 'ts2', 'analysis'),
                                  outputs = ['xt', 'thetat'], verbose = False)
    n = run['nsamples']
    assert ap['xt'].shape == (n1, n) and ap['thetat'].shape == (n1, n)
    assert ap.attrs['complete'] and ap.attrs['outputs'] == ['xt', 'thetat']
    assert ap.attrs['dataset'] == DS.root.path
    assert np.isclose(float(ap['dt'][...]), 1 / 596.046, rtol = 1e-3)
    xt = ap['xt'][:]
    assert np.isfinite(xt).all()
    for r in rows:
        np.testing.assert_allclose(xt[r, :1000], np.asarray(DS.xt[r])[:1000])
    # The chunked stage-0 run (two writes) too; its own sample period
    whole, run = dl.select_streams(root, run = 1, tone_sets = [1])
    ap1 = apply_stream_calibration(DS, root, whole[0],
                                   dl.require_group(g, 'ts1', 'analysis'),
                                   verbose = False)
    assert ap1['xt'].shape == (n1, run['nsamples'])
    assert np.isfinite(ap1['xt'][:]).all()
    assert np.isclose(float(ap1['dt'][...]), 1 / 38147.0, rtol = 1e-3)
    # Done once unless overwritten; other tones are refused
    assert apply_stream_calibration(DS, root, whole[0], ap1,
                                    verbose = False) is ap1
    other = dl.select_streams(root, run = 0, tone_sets = [0])[0][0]
    with pytest.raises(ValueError, match = "DataSet's tones"):
        apply_stream_calibration(DS, root, other,
                                 dl.require_group(g, 'bad', 'analysis'),
                                 verbose = False)


def test_select_streams(streamed):
    root, tune = streamed
    runs = dl.stream_runs(root)
    assert [r['name'] for r in runs] == ['slow', 'fast', 'slow2']
    assert runs[0]['nsamples'] == round(2 * 596.046) and \
        runs[0]['n_chunks'] >= 1
    text = dl.describe_stream_runs(root)
    assert 'slow' in text and 'chunks of' in text
    src, run = dl.select_streams(root, 'fast')
    assert run['run'] == 1 and len(src) == 2 and src[1]['writes'] == \
        ['write0', 'write1']
    assert src[1]['group'] == 'tune0/cal3' and src[1]['stream'] == 'ts1'
    assert (src[0]['start'], src[0]['stop']) == (0, runs[1]['nsamples'])
    src, run = dl.select_streams(root, run = 0, chunks = 1, tone_sets = [1])
    assert len(src) == 1 and src[0]['stop'] == min(
        runs[0]['chunk_samples'], runs[0]['nsamples'])
    with pytest.raises(KeyError, match = 'no run named'):
        dl.select_streams(root, 'nope')
    with pytest.raises(KeyError, match = 'not'):
        dl.select_streams(root, 'fast', run = 0)
    with pytest.raises(ValueError, match = 'no samples'):
        dl.select_streams(root, 'slow', chunks = (5, 6))


def test_execute_remaining(streamed):
    """
    ``execute_remaining`` finishes only the rows not done, keeping the
    global step, and does nothing when all are done.
    """
    root, tune = streamed
    src = [source_of(root, 0, 1, 0, 500)]
    out = dl.require_group(tune, 'cal_rem', 'analysis')
    DS = open_stream_dataset(out.require_group('set1'), src,
                             'resonators0/fres_all', 'resonators0/qres_all')
    AR = AnalysisRunner(DS, analysis_yaml_path = 'ts')
    n = len(np.asarray(DS.res_idxs))
    AR.execute_path(data_idx = [0, 1], verbose = False)
    np.testing.assert_array_equal(AR.missing_rows(), np.arange(2, n))
    x0 = np.asarray(AR.DS.xt[0]).copy()
    np.testing.assert_array_equal(AR.execute_remaining(verbose = False),
                                  np.arange(2, n))
    assert len(AR.missing_rows()) == 0
    np.testing.assert_array_equal(np.asarray(AR.DS.xt[0]), x0)
    assert len(AR.execute_remaining(verbose = False)) == 0
