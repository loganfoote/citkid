"""
Tests for streaming a tuning: the simulated board's streams, chunk
planning, ``stream_tone_sets``, and the timestream window.
"""
import asyncio
import time
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from citkid import daq_layout as dl
from citkid.crs import util
from citkid.crs.procedures import (
    plan_stream_chunks, stream_tone_sets, write_stream_chunk
)
from citkid.crs.sim import SimulatedCRS
from citkid.crs.stream_gui import MODES, StreamWindow
from citkid.vna import simulate as sim

SMALL = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 12,
             pair_frac = 0.)


def make_crs(global_control = True):
    """
    Make a simulated board with a small device.

    Parameters:
    global_control (bool): session control. Default True.

    Returns:
    crs (SimulatedCRS): the board.
    """
    rng = np.random.default_rng(0)
    return SimulatedCRS(sim.draw_device(rng, **SMALL),
                        sim.draw_gain(rng, f_lo = 1e8, f_hi = 5e9),
                        module_idxs = (1, 2), global_control = global_control)


def make_tuning(tmp_path, crs):
    """
    Make a store with a tuning of the simulated device's resonators in two
    tone sets (module 1 in set 0, modules 1 and 2 in set 1).

    Parameters:
    tmp_path (pathlib.Path): directory.
    crs (SimulatedCRS): the board.

    Returns:
    root (zarr.Group): store root.
    tune (zarr.Group): the tuning.
    """
    root = dl.create_store(tmp_path / 's.zarr', 'd', 'c')
    fres = crs.truth.fr.values
    n = len(fres)
    dl.save_resonators(root, 0, fres, crs.truth.Qr.values / 5, 'test',
                       Qr = crs.truth.Qr.values)
    tune, _ = dl.new_tuning(root, 0)
    tone_set = (np.arange(n) >= n // 2).astype(int)
    tone_module = np.where((tone_set == 1) & (np.arange(n) % 2 == 0), 2, 1)
    ncos = [{'1': 1.05e9}, {'1': 1.05e9 + 1e3, '2': 1.05e9 - 1e3}]
    dl.write_tune_config(tune, dict(dl.tune_config(crs), tone_sets = ncos))
    dl.save_tones(dl.create_group(tune, 'tones', 'decision'), fres,
                  crs.truth.Qr.values / 5, -55., np.arange(n),
                  tone_set = tone_set, tone_module = tone_module)
    return root, tune


def test_plan_stream_chunks():
    tm = np.array([1] * 300 + [2] * 200 + [3] * 50)
    for dec in range(7):
        chunks = plan_stream_chunks(tm, dec)
        # Every tone once
        np.testing.assert_array_equal(np.sort(np.concatenate(chunks)),
                                      np.arange(len(tm)))
        for c in chunks:
            mods = sorted(set(tm[c].tolist()))
            per_module, total = util.max_stream_tones(dec, len(mods))
            assert len(c) <= total
            assert all(np.sum(tm[c] == m) <= per_module for m in mods)
    assert len(plan_stream_chunks(tm, 6)) == 1
    # Stage 0: one module at a time, 128 tones each
    assert [len(c) for c in plan_stream_chunks(tm, 0)] == \
        [128, 128, 44, 128, 72, 50]
    assert plan_stream_chunks([], 0) == []


@pytest.mark.asyncio
async def test_simulated_streams():
    crs = make_crs(global_control = False)
    fres = crs.truth.fr.values[:4]
    with pytest.raises(RuntimeError, match = 'write tones'):
        await crs.stream_to_memory(1., 6)
    await write_stream_chunk(crs, fres, -55., [1, 1, 2, 2],
                             {'1': 1.05e9, '2': 1.04e9})
    zt = await crs.stream_to_memory(2., 6)
    assert zt.shape == (4, round(2 * util.get_sample_freq(6)))
    # Without global control, only the board's stage
    with pytest.raises(RuntimeError, match = 'global control'):
        await crs.stream_to_memory(0.1, 0)
    # Per-sample noise: the 100-sample sweep noise times 10
    g = np.abs(sim.gain_s21(fres, crs.gain))
    sigma = np.std(zt.real, axis = 1) / g
    np.testing.assert_allclose(sigma, 10 ** (-crs.snr_db / 20) * 10,
                               rtol = 0.15)
    crs.global_control = True
    # Two modules exceed the frame rate at stage 0; one is fine
    with pytest.raises(ValueError, match = 'frames/s'):
        await crs.stream_to_memory(0.05, 0)
    await write_stream_chunk(crs, fres, -55., [1] * 4, {'1': 1.05e9})
    z0 = await crs.stream_to_memory(0.05, 0)
    assert crs.dec_stage == 0
    assert np.std(z0.real, axis = 1)[0] / g[0] > 7 * sigma[0]
    # Too many tones per module for stage 0
    await write_stream_chunk(crs, np.full(130, 1.05e9), -55., [1] * 130,
                             {'1': 1.05e9})
    with pytest.raises(ValueError):
        await crs.stream_to_memory(0.05, 0)


@pytest.mark.asyncio
async def test_stream_tone_sets(tmp_path):
    crs = make_crs()
    await crs.attach()
    root, tune = make_tuning(tmp_path, crs)
    n = len(crs.truth)
    said = []
    kw = dict(npoints_fine = 40, npoints_gain = 10, verbose = False,
              status = said.append)
    with pytest.raises(ValueError, match = 'name'):
        await stream_tone_sets(crs, tune, ' ', 1., 6, **kw)
    paths = await stream_tone_sets(crs, tune, 'dark', 0.5, 6,
                                   notes = 'lights off', **kw)
    # One calibration sweep and one stream per tone set
    assert paths == ['tune0/sweep0/write0/ts0', 'tune0/sweep1/write0/ts0']
    for j, s in enumerate((0, 1)):
        cal = tune[f'sweep{j}']
        assert cal.attrs['tone_set'] == s and cal.attrs['complete']
        rows = np.flatnonzero(tune['tones/tone_set'][:] == s)
        w = cal['write0']
        np.testing.assert_array_equal(w['sweep_rows'][:],
                                      np.arange(len(rows)))
        np.testing.assert_array_equal(w['res_idxs'][:], rows)
        ts = w['ts0']
        assert ts.attrs['name'] == 'dark' and ts.attrs['notes'] == \
            'lights off'
        assert ts['z'].shape == (2, len(rows),
                                 round(0.5 * util.get_sample_freq(6)))
        zt = (ts['z'][0] + 1j * ts['z'][1]) * ts['counts_to_s21'][:][:, None]
        assert np.all(np.isfinite(zt))
    runs = tune['streams'].attrs['runs']
    assert len(runs) == 1 and runs[0]['paths'] == paths
    assert any('calibration' in m for m in said)
    # Stage 0 needs chunks: one module at a time in set 1
    paths = await stream_tone_sets(crs, tune, 'fast', 0.02, 0, **kw)
    assert len(paths) == 3 and paths[0] == 'tune0/sweep2/write0/ts0'
    w = [root[p.rsplit('/', 1)[0]] for p in paths[1:]]
    mods = tune['tones/tone_module'][:][tune['tones/tone_set'][:] == 1]
    for g in w:
        assert len(set(mods[g['sweep_rows'][:]].tolist())) == 1
    assert len(tune['streams'].attrs['runs']) == 2
    assert n == tune['tones/fres'].shape[0]


@pytest.fixture(scope = 'module')
def qapp():
    """
    Return the QApplication (offscreen).

    Returns:
    app (QApplication): the application.
    """
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(cond, timeout = 60.):
    """
    Process Qt events until a condition holds.

    Parameters:
    cond (callable): the condition.
    timeout (float): seconds before failing. Default 60.
    """
    t0 = time.time()
    while not cond():
        QtWidgets.QApplication.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError('condition not met')
        time.sleep(0.005)


def test_stream_window(qapp, tmp_path):
    crs = make_crs()
    asyncio.run(crs.attach())
    root, tune = make_tuning(tmp_path, crs)

    async def connect():
        """
        Return the simulated board.

        Returns:
        crs (SimulatedCRS): the board.
        """
        return crs

    win = StreamWindow(
        connect, tune, preview_s = 1., presets = [
            dict(name = 'dark_dec6', dec_stage = 6, duration_s = 0.5,
                 notes = 'dark')],
        sweep_kwargs = dict(npoints_fine = 60, npoints_gain = 10))
    wait_for(lambda: win.crs is not None)
    assert win.name_edit.text() == 'dark_dec6' and win.run_btn.isEnabled()
    assert not win.capture_btn.isEnabled()
    win.set_box.setCurrentIndex(1)
    win.calibrate()
    wait_for(lambda: win.cal is not None and not win.worker.busy)
    assert win.cal['set'] == 1 and win.capture_btn.isEnabled()
    win.tone_box.setCurrentIndex(2)
    win.preview_dec.setValue(0)
    win.preview_s.setValue(0.05)
    win.capture()
    wait_for(lambda: win.preview is not None and not win.worker.busy)
    # Stage 0: only the selected tone's module is captured
    r = win.tone_box.currentData()
    mods = tune['tones/tone_module'][:]
    assert r in win.preview['rows']
    assert set(mods[win.preview['rows']].tolist()) == {mods[r]}
    for i, n_ts in zip(range(len(MODES)), (1, 2, 2)):
        win.mode_box.setCurrentIndex(i)
        assert len(win.ts_plot.listDataItems()) == n_ts
        # Loop, stream points, and the tone's point (not for I and Q)
        assert len(win.iq_plot.listDataItems()) == (2 if i == 2 else 3)
    # The x preview is near zero (no photons in the simulation)
    win.mode_box.setCurrentIndex(0)
    x = win.ts_plot.listDataItems()[0].yData
    assert abs(np.mean(x)) < 1e-6
    win.run()
    wait_for(lambda: len(win.runs) == 1 and not win.worker.busy)
    assert win.runs_list.count() == 1
    assert tune['streams'].attrs['runs'][0]['name'] == 'dark_dec6'
    worker = win.worker
    win.close()
    win.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None,
                                             QtCore.QEvent.DeferredDelete)
    assert not worker.running


@pytest.mark.asyncio
async def test_chunked_streams_load_in_sweep_order(tmp_path):
    """
    A chunked write holds only some tones; the pipeline's loaders map the
    calibration sweep's rows to it (``sweep_rows``), with NaN for tones
    not streamed in that write. Unchunked writes load as before.
    """
    from citkid.pipeline.sweep_series import target_sweep_steps
    crs = make_crs()
    await crs.attach()
    root, tune = make_tuning(tmp_path, crs)
    kw = dict(npoints_fine = 20, npoints_gain = 10, verbose = False)
    await stream_tone_sets(crs, tune, 'slow', 0.2, 6, **kw)
    await stream_tone_sets(crs, tune, 'fast', 0.02, 0, **kw)
    # Tone set 1 at stage 0: sweep3, two writes (one per module)
    cal = tune['sweep3']
    n = cal['fres'].shape[0]
    rows = np.arange(n)
    for w, sweep_store in (('write0', 'sweep1'), ('write0', 'sweep3'),
                           ('write1', 'sweep3')):
        steps = {s.name: s for s in target_sweep_steps(
            store = root, group = f'tune0/{sweep_store}', ts = w,
            ts_stream = 'ts0')}
        ft = steps['load_ft'].func(rows)
        zt = steps['load_zt'].func(rows)
        grp = tune[f'{sweep_store}/{w}']
        if 'sweep_rows' in grp and sweep_store == 'sweep3':
            streamed = grp['sweep_rows'][:]
        else:
            streamed = rows
        ok = np.isin(rows, streamed)
        assert np.all(np.isfinite(ft[ok])) and np.all(np.isnan(ft[~ok]))
        assert np.all(np.isfinite(zt[ok])) and np.all(np.isnan(zt[~ok]))
        np.testing.assert_allclose(ft[ok], cal['fres'][:][ok], atol = 100)
        # One row on its own
        assert np.isfinite(steps['load_ft'].func(streamed[:1])).all()
    # The two writes of tone set 1 cover it once
    a = tune['sweep3/write0/sweep_rows'][:]
    b = tune['sweep3/write1/sweep_rows'][:]
    np.testing.assert_array_equal(np.sort(np.concatenate((a, b))), rows)


@pytest.mark.asyncio
async def test_series_fits_on_stitched_tone_sets(tmp_path):
    """
    The pipeline's series loading and IQ fits work on a power sweep
    stitched over two tone sets, in the same store (as in 03_tune.py).
    """
    from citkid.crs.procedures import stitched_power_sweep
    from citkid.pipeline.sweep_series import (
        open_series_datasets, run_series_analysis
    )
    crs = make_crs()
    await crs.attach()
    root, tune = make_tuning(tmp_path, crs)
    t = dl.load_tones(tune['tones'])
    ps = dl.require_group(tune, 'power_sweep', 'raw')
    await stitched_power_sweep(
        crs, t['fres'], [-60., -50.], t['qres'], t['res_idxs'],
        t['tone_set'], t['tone_module'],
        tune['crs_config'].attrs['config']['tone_sets'], ps, verbose = False,
        npoints_fine = 200, npoints_gain = 20, npoints_rough = None)
    out = dl.require_group(tune, 'power_fits', 'analysis')
    datasets = open_series_datasets(
        out_root = out, group = 'tune0/power_sweep',
        fres_all = 'resonators0/fres_all', qres_all = 'resonators0/qres_all')
    rows = [0, len(t['fres']) - 1]    # one tone from each set
    assert t['tone_set'][rows[0]] != t['tone_set'][rows[1]]
    runners = run_series_analysis(datasets, 'iq', data_idxs = rows,
                                  verbose = False)
    for ar in runners:
        for r in rows:
            popt = np.asarray(ar.DS.iq_popt[r], dtype = float)
            # The fitted fr is the tone's resonance
            assert abs(popt[0] - crs.truth.fr.values[r]) < \
                0.3 * crs.truth.fr.values[r] / crs.truth.Qr.values[r]
