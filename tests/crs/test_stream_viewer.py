"""
Tests of the timestream viewer on a simulated store.
"""
import asyncio
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
from citkid import daq_layout as dl
from citkid.crs.stream_gui import load_run_view
from citkid.crs.stream_viewer import TimestreamViewer
from .test_streaming import make_crs, make_tuning


@pytest.fixture(scope = 'module')
def viewer_store(tmp_path_factory):
    """
    A tuning with a short stage-6 run of both tone sets and a chunked
    stage-0 run of set 0.

    Returns:
    path (pathlib.Path): the store.
    """
    from citkid.crs import util
    from citkid.crs.procedures import stream_tone_sets
    tmp = tmp_path_factory.mktemp('viewer')
    crs = make_crs()

    async def take():
        """
        Take the runs.
        """
        await crs.attach()
        _, tune = make_tuning(tmp, crs)
        kw = dict(npoints_fine = 60, npoints_gain = 10, verbose = False)
        await stream_tone_sets(crs, tune, 'slow', 1., 6, **kw)
        await stream_tone_sets(crs, tune, 'fast', 0.05, 0, set_idxs = [0],
                               **kw)

    # Small zarr chunks, so a short fast run has several
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(util, 'MIN_SHARD_BYTES', 8 * 1024)
        mp.setattr(util, 'MIN_CHUNK_SAMPLES', 256)
        asyncio.run(take())
    return tmp / 's.zarr'


def test_viewer_lists_runs_and_views_chunks(viewer_store):
    """
    The viewer lists the tunings with runs and their runs, offers the
    selected run's chunks and tones, reads the chosen chunks (from any
    chunk) and draws the tone in every mode with the loop and the PSD;
    the time axis starts at the chunk's time.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = TimestreamViewer(viewer_store, path = str(viewer_store))
    win.show()
    app.processEvents()
    assert win.tune_box.count() == 1 and win.tune.basename == 'tune0'
    assert 'run(s)' in win.header.text()
    names = [r['name'] for r in win.runs]
    assert names[:2] == ['slow', 'slow'] or 'slow' in names
    assert win.runs_list.count() == len(win.runs)
    # The last run (fast, chunked) is selected: its chunks and tones
    last = win.runs[-1]
    assert win.runs_list.currentRow() == len(win.runs) - 1
    n = int(last['n_chunks'])
    assert win.count_box.maximum() == n and win.start_box.maximum() == n - 1
    assert win.view_btn.isEnabled()
    set_rows = np.flatnonzero(win.tones['tone_set'] == last['tone_set'])
    assert win.tone_box.count() == len(set_rows)
    # View from the second chunk
    win.start_box.setValue(1)
    win.count_box.setValue(min(2, n))
    view = win.view_run()
    assert view is not None and view['start'] == 1
    assert view['t0'] == pytest.approx(last['chunk_samples'] * view['dt'])
    assert view['chunks'] == (min(2, n - 1), n)
    assert view['zt'].shape[1] == last['chunk_samples'] * min(2, n - 1)
    assert 'chunks 1 to' in win.status.text()
    assert win.ts_plot.titleLabel.text.endswith(
        f'tone {win.tone_box.currentData()}')
    t = win.ts_plot.listDataItems()[0].getData()[0]
    assert t[0] == pytest.approx(view['t0'])
    assert len(win.iq_plot.listDataItems()) == 3
    # Every mode, the offset, and the PSD
    for mode in (1, 2, 0):
        win.mode_box.setCurrentIndex(mode)
        assert len(win.ts_plot.listDataItems()) == (1 if mode == 0 else 2)
    win.offset_box.setChecked(True)
    y = win.ts_plot.listDataItems()[0].getData()[1]
    assert abs(np.nanmean(y)) < 1e-9 * (np.nanstd(y) + 1e-30)
    assert not win.psd_plot.isVisible()
    win.psd_box.setChecked(True)
    assert win.psd_plot.isVisible()
    assert len(win.psd_plot.listDataItems()) == 1
    # Another tone of the viewed set redraws; a slow run of set 1 lists
    # its tones and views the whole run
    win.tone_box.setCurrentIndex(1)
    assert win.ts_plot.titleLabel.text.endswith(
        f'tone {win.tone_box.currentData()}')
    slow = next(i for i, r in enumerate(win.runs) if r['name'] == 'slow')
    win.runs_list.setCurrentRow(slow)
    # A run of both sets lists every tone; the chosen tone picks the set
    assert win.tone_box.count() == len(win.tones['fres'])
    r1 = int(np.flatnonzero(win.tones['tone_set'] == 1)[0])
    win.tone_box.setCurrentIndex(win.tone_box.findData(r1))
    view = win.view_run()
    assert view['tone_set'] == 1 and view['start'] == 0
    assert r1 in view['rows'] and view['calibration'].startswith('tune0')
    win.close()


def test_load_run_view_start(viewer_store):
    """
    ``load_run_view`` reads the chunks from ``start``, clipped to the
    run, and says where they begin.
    """
    root = dl.open_store(viewer_store)
    run = [r for r in dl.stream_runs(root) if r['name'] == 'fast'][-1]
    tones = dl.load_tones(root['tune0/tones'])
    rows = np.flatnonzero(tones['tone_set'] == run['tone_set'])
    n = int(run['n_chunks'])
    v0 = load_run_view(root, run, rows, chunks = 1)
    v1 = load_run_view(root, run, rows, chunks = 1, start = 1)
    assert v0['start'] == 0 and v0['t0'] == 0 and v1['start'] == 1
    assert not np.array_equal(v0['zt'], v1['zt'])
    far = load_run_view(root, run, rows, chunks = 5, start = n + 10)
    assert far['start'] == n - 1 and far['chunks'] == (1, n)
