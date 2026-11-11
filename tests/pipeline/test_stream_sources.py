"""
Tests for ``citkid.pipeline.stream_sources``: loading several sources'
streams into one DataSet and calibrating them.
"""
import asyncio
import numpy as np
import pytest
from citkid import daq_layout as dl
from citkid.crs.procedures import stream_tone_sets
from citkid.pipeline.analysis import AnalysisRunner
from citkid.pipeline.stream_sources import (
    open_stream_dataset, stream_sources_steps
)
from ..crs.test_streaming import make_crs, make_tuning


@pytest.fixture(scope = 'module')
def streamed(tmp_path_factory):
    """
    A simulated tuning with a stage-6 run and a chunked stage-0 run.

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
        return root, tune

    return asyncio.run(take())


def sources_of(root, run, start, stop):
    """
    Make the sources of one run: each tone set's sweep with all its writes.

    Parameters:
    root (zarr.Group): the store root.
    run (int): run index (ts{run}).
    start, stop (int): sample window.

    Returns:
    sources (list of dict): the sources.
    """
    out = []
    for path in dl.stream_runs(root)[run]['paths']:
        sweep = path.rsplit('/', 2)[0]
        if out and out[-1]['group'] == sweep:
            out[-1]['writes'].append(path.split('/')[-2])
            continue
        out.append(dict(group = sweep, writes = [path.split('/')[-2]],
                        start = start, stop = stop, stream = 'stream'))
    return out


def test_rows_follow_sources_and_writes(streamed):
    root, tune = streamed
    n = tune['tones/fres'].shape[0]
    # Stage 0: tone set 1 streamed in two writes (one per module)
    src = sources_of(root, 1, 10, 30)
    assert [len(s['writes']) for s in src] == [1, 2]
    steps = {s.name: s for s in stream_sources_steps(
        root, src, 'resonators0/fres_all', 'resonators0/qres_all',
        rows = 'all')}
    fres_all, qres_all, dt, nrows = steps['load_global_data'].func()
    assert nrows == n and np.isclose(dt, 1 / 38147.0, rtol = 1e-3)
    assert np.all(np.diff(fres_all) >= 0)
    # An out-of-order saved list is sorted on load, qres along with it
    root.create_array('fres_bad', data = np.array([2e9, 1e9, 3e9]))
    root.create_array('qres_bad', data = np.array([2e3, 1e3, 3e3]))
    steps = {s.name: s for s in stream_sources_steps(
        root, src, 'fres_bad', 'qres_bad', rows = 'all')}
    f, q, _, _ = steps['load_global_data'].func()
    np.testing.assert_array_equal(f, [1e9, 2e9, 3e9])
    np.testing.assert_array_equal(q, [1e3, 2e3, 3e3])
    fres, qres, ares, res_idxs, source_idx, write_idx = \
        steps['load_global_res_data'].func()
    np.testing.assert_array_equal(res_idxs, np.arange(n))
    np.testing.assert_array_equal(np.bincount(source_idx),
                                  np.bincount(tune['tones/tone_set'][:]))
    zt = steps['load_zt'].func(np.arange(n))
    assert zt.shape == (n, 20) and np.all(np.isfinite(zt))
    ft = steps['load_ft'].func(np.arange(n))
    np.testing.assert_allclose(ft, fres, atol = 100)
    # Rows of different writes in one call, out of order
    rows = np.array([n - 1, 0, n // 2 + 1, n // 2])
    np.testing.assert_array_equal(steps['load_zt'].func(rows), zt[rows])
    # Each row's write holds it
    sel = source_idx == 1
    assert set(write_idx[sel].tolist()) == {0, 1}
    ff, zf = steps['load_sweep_f'].func(n - 1)
    assert len(ff) == 200 and np.all(np.diff(ff) > 0)


def test_mixed_stages_refused(streamed):
    root, tune = streamed
    src = sources_of(root, 0, 0, 20) + sources_of(root, 1, 0, 20)
    steps = {s.name: s for s in stream_sources_steps(
        root, src, 'resonators0/fres_all', 'resonators0/qres_all')}
    with pytest.raises(ValueError, match = 'decimation stage'):
        steps['load_global_data'].func()
    with pytest.raises(ValueError, match = 'stop > start'):
        stream_sources_steps(root, [dict(src[0], stop = 0)], 'a', 'b')


def test_dataset_calibrates(streamed):
    root, tune = streamed
    src = sources_of(root, 0, 0, 1000)
    out = dl.require_group(tune, 'ts_cal0', 'analysis')
    DS = open_stream_dataset(out.require_group('dec6'), src,
                             'resonators0/fres_all', 'resonators0/qres_all')
    assert len(np.asarray(DS.res_idxs)) == tune['tones/fres'].shape[0]
    np.testing.assert_array_equal(np.asarray(DS.source_idx),
                                  tune['tones/tone_set'][:])
    AR = AnalysisRunner(DS, analysis_yaml_path = 'ts')
    rows = [0, len(np.asarray(DS.res_idxs)) - 1]
    AR.execute_path(data_idx = rows, verbose = False)
    for r in rows:
        x = np.asarray(AR.DS.xt[r])
        assert x.shape == (1000,) and np.isfinite(x).all()
        # Dark: x is noise around a small offset
        assert np.std(x) < 1e-5
    # Reopens from the store alone
    DS2 = open_stream_dataset(out.require_group('dec6'), src,
                              'resonators0/fres_all', 'resonators0/qres_all')
    np.testing.assert_allclose(np.asarray(DS2.xt[rows[0]]),
                               np.asarray(AR.DS.xt[rows[0]]))


def test_select_streams(streamed):
    root, tune = streamed
    runs = dl.stream_runs(root)
    assert [r['name'] for r in runs] == ['slow', 'fast']
    assert runs[0]['nsamples'] == round(2 * 596.046) and \
        runs[0]['n_chunks'] >= 1
    text = dl.describe_stream_runs(root)
    assert 'slow' in text and 'chunks of' in text
    src, run = dl.select_streams(root, 'fast')
    assert run['run'] == 1 and len(src) == 2 and src[1]['writes'] == \
        ['write0', 'write1']
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
    src = sources_of(root, 0, 0, 500)
    out = dl.require_group(tune, 'cal_rem', 'analysis')
    DS = open_stream_dataset(out.require_group('dec6'), src,
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
