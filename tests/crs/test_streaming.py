"""
Tests for streaming a tuning: the simulated board's streams, chunk
planning, ``stream_tone_sets``, and the timestream window.
"""
import asyncio
import time
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from citkid import daq_layout as dl
from citkid.crs import util
from citkid.crs.procedures import (
    calibrate_tone_set, calibration_in_frame, plan_stream_chunks,
    stream_tone_sets, tones_written, write_set_tones, write_stream_chunk
)
from citkid.crs.sim import DEFAULT_PULSES, SimulatedCRS
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
async def test_simulated_pulses():
    """
    With ``pulses``, streams hold photon pulses on each tone and cosmic rays
    on every tone at once, at the logged samples; without, none are logged.
    """
    crs = make_crs(global_control = True)
    crs.snr_db = 120.
    fres = crs.truth.fr.values[:4]
    await write_stream_chunk(crs, fres, -55., [1] * 4, {'1': 1.05e9})
    await crs.stream_to_memory(0.05, 0)
    assert crs.pulse_log == []
    crs.pulses = dict(DEFAULT_PULSES, rate = 100., cosmic_rate = 50.)
    zt = await crs.stream_to_memory(0.2, 0)
    log = crs.pulse_log[-1]
    assert log['n'] == zt.shape[1] and list(log['rows']) == [0, 1, 2, 3]
    assert len(log['cosmic']) > 0
    assert log['cosmic_x'].shape == (4, len(log['cosmic']))
    for k in range(4):
        dev = np.abs(zt[k] - np.median(zt[k]))
        jump = np.diff(dev)
        noise = np.median(np.abs(jump))
        # Pulses rise in one sample: |z - z0| jumps up at every logged
        # pulse start (photons of this tone, cosmic rays of all tones)
        true = np.union1d(log['photons'][k], log['cosmic'])
        true = true[true > 0]
        assert np.all(jump[true - 1] > 20 * noise)
        # and almost nowhere else
        assert np.sum(jump > 20 * noise) < 1.1 * len(true) + 2
    crs.pulses = None


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
    reports = []
    paths = await stream_tone_sets(crs, tune, 'dark', 0.5, 6,
                                   notes = 'lights off',
                                   progress = lambda d, t: reports.append(
                                       (d, t)), **kw)
    # The run is ts0 of the tuning: a calibration sweep per tone set (taken
    # and saved as cal0 and cal1, since none was given), each with a write
    # of its tones holding the stream
    assert paths == ['tune0/cal0/write0/ts0', 'tune0/cal1/write0/ts0']
    assert [c['name'] for c in dl.calibrations(tune)] == ['cal0', 'cal1']
    assert [c['tone_set'] for c in dl.calibrations(tune)] == [0, 1]
    runs = dl.stream_runs(root)
    assert len(runs) == 1 and runs[0]['paths'] == paths
    run = runs[0]
    assert run['name'] == 'dark' and run['notes'] == 'lights off'
    assert run['tuning'] == 'tune0' and run['group'] == 'ts0'
    assert run['tone_sets'] == [0, 1] and run['complete']
    assert run['calibrations'] == {'0': 'tune0/cal0', '1': 'tune0/cal1'}
    assert run['nsamples'] == round(0.5 * util.get_sample_freq(6))
    for s in (0, 1):
        cal = tune[f'cal{s}']
        assert cal.attrs['tone_set'] == s and cal.attrs['complete']
        rows = np.flatnonzero(tune['tones/tone_set'][:] == s)
        w = cal['write0']
        assert w.attrs['tones_reused'] is False
        np.testing.assert_array_equal(w['sweep_rows'][:],
                                      np.arange(len(rows)))
        np.testing.assert_array_equal(w['res_idxs'][:], rows)
        ts = w['ts0']
        assert ts.attrs['name'] == 'dark' and ts.attrs['run'] == 0
        assert ts['z'].shape == (2, len(rows),
                                 round(0.5 * util.get_sample_freq(6)))
        zt = (ts['z'][0] + 1j * ts['z'][1]) * ts['counts_to_s21'][:][:, None]
        assert np.all(np.isfinite(zt))
    assert dl.stream_runs(root, 'tune1') == []
    assert any('calibration' in m for m in said)
    # Progress: sweeps in points, streams in samples, each ending complete
    assert reports and all(d <= t for d, t in reports)
    n0 = int(np.sum(tune['tones/tone_set'][:] == 0))
    assert (n0 * 50, n0 * 50) in reports
    ns = round(0.5 * util.get_sample_freq(6))
    assert (ns, ns) in reports and reports[-1] == (ns, ns)
    # Stage 0 needs chunks: one module at a time in set 1. Set 1's NCOs
    # were set last, so cal1 is in frame and used; cal0 (set 0, whose
    # module 1 was set again for set 1) is out of frame and set 0 is
    # calibrated again (cal2), which puts set 1 out of frame in turn
    # (cal3): one tone set at a time
    cals = {0: 'tune0/cal0', 1: tune['cal1']}
    assert not calibration_in_frame(crs, tune['cal0'])
    assert calibration_in_frame(crs, tune['cal1'])
    said.clear()
    paths = await stream_tone_sets(crs, tune, 'fast', 0.02, 0,
                                   calibrations = cals, **kw)
    assert paths == ['tune0/cal2/write0/ts1', 'tune0/cal3/write0/ts1',
                     'tune0/cal3/write1/ts1']
    assert [c['name'] for c in dl.calibrations(tune)] == \
        ['cal0', 'cal1', 'cal2', 'cal3']
    assert dl.stream_runs(root)[1]['calibrations'] == {'0': 'tune0/cal2',
                                                       '1': 'tune0/cal3'}
    assert sum('out of frame' in m for m in said) == 2
    w = [root[p.rsplit('/', 1)[0]] for p in paths[1:]]
    mods = tune['tones/tone_module'][:][tune['tones/tone_set'][:] == 1]
    for g in w:
        assert len(set(mods[g['sweep_rows'][:]].tolist())) == 1
    assert [r['name'] for r in dl.stream_runs(root)] == ['dark', 'fast']
    assert n == tune['tones/fres'].shape[0]
    # Tones left on the board are not written again by the next run of
    # the same tones from the same calibration (set 1 is current): the
    # next stream goes under the same write
    n1 = int(np.sum(tune['tones/tone_set'][:] == 1))
    cals = {1: 'tune0/cal3'}
    written = {}
    paths = await stream_tone_sets(crs, tune, 'a', 0.1, 6, set_idxs = [1],
                                   calibrations = cals, written = written,
                                   keep_tones = True, **kw)
    assert paths == ['tune0/cal3/write2/ts2']
    assert written['tone_set'] == 1 and written['calibration'] == \
        'tune0/cal3' and len(written['rows']) == n1
    assert written['write'] == 'tune0/cal3/write2'
    assert root['tune0/cal3/write2'].attrs['tones_reused'] is False
    assert len(dl.calibrations(tune)) == 4
    said.clear()
    paths = await stream_tone_sets(crs, tune, 'b', 0.1, 6, set_idxs = [1],
                                   calibrations = cals, written = written,
                                   keep_tones = True, **kw)
    assert paths == ['tune0/cal3/write2/ts3']
    assert any('already written' in m for m in said)
    assert sorted(k for k in root['tune0/cal3/write2'].keys()
                  if k.startswith('ts')) == ['ts2', 'ts3']
    assert dl.stream_runs(root)[3]['calibration'] == 'tune0/cal3'
    # Another set means a calibration and a write; not keeping the tones
    # clears them
    paths = await stream_tone_sets(crs, tune, 'c', 0.1, 6, set_idxs = [0],
                                   written = written, **kw)
    assert paths == ['tune0/cal4/write0/ts4']
    assert dl.stream_runs(root)[4]['calibration'] == 'tune0/cal4'
    assert written == {}
    # A stream still being written (no 'dt' yet) is left out of the runs
    g = dl.create_group(root['tune0/cal4/write0'], 'ts9', 'raw')
    g.create_array('z', data = np.zeros((2, 1, 10), dtype = np.int32))
    dl.update_stream_run(tune, 4, paths = paths + ['tune0/cal4/write0/ts9'])
    assert dl.stream_runs(root)[-1]['paths'] == ['tune0/cal4/write0/ts4']
    text = dl.describe_stream_runs(root)
    assert 'ts4: c, tune0, tune0/cal4' in text

@pytest.mark.asyncio
async def test_calibrate_and_write_set_tones(tmp_path):
    crs = make_crs()
    await crs.attach()
    root, tune = make_tuning(tmp_path, crs)
    kw = dict(npoints_fine = 40, npoints_gain = 10, verbose = False)
    with pytest.raises(ValueError, match = 'tone set 5'):
        await calibrate_tone_set(crs, tune, 5, **kw)
    said = []
    cal = await calibrate_tone_set(crs, tune, 1, status = said.append, **kw)
    assert cal.path == 'tune0/cal0' and cal.attrs['tone_set'] == 1
    assert cal.attrs['tones'] == 'tune0/tones' and cal.attrs['complete']
    n1 = int(np.sum(tune['tones/tone_set'][:] == 1))
    assert cal['fine_sweep/z'].shape == (n1, 40)
    assert any('cal0' in m for m in said)
    c = dl.calibrations(tune)
    assert len(c) == 1 and c[0]['ntones'] == n1 and \
        c[0]['npoints_fine'] == 40 and c[0]['nsamps'] == 100
    # write_set_tones writes once, then reuses; a write group gets the
    # tones either way
    written = {}
    rows = np.arange(n1)
    assert not tones_written(written, 1, rows, cal.path)
    fres = await write_set_tones(crs, cal, rows, written)
    assert tones_written(written, 1, rows, cal.path)
    assert not tones_written(written, 1, rows[:2], cal.path)
    assert not tones_written(written, 0, rows, cal.path)
    assert not tones_written(written, 1, rows, 'tune0/cal9')
    g = dl.create_group(tune, 'w', 'raw')
    fres2 = await write_set_tones(crs, cal, rows, written, grp = g)
    np.testing.assert_array_equal(fres, fres2)
    assert g.attrs['tones_reused'] is True
    np.testing.assert_array_equal(g['fres'][:], fres)
    np.testing.assert_array_equal(g['fres_requested'][:], cal['fres'][:])
    assert 'crs_config' in g
    # A failed sweep leaves no calibration behind
    with pytest.raises(Exception):
        await calibrate_tone_set(crs, tune, 1, npoints_fine = 0, **{
            k: v for k, v in kw.items() if k != 'npoints_fine'})
    assert [c['name'] for c in dl.calibrations(tune)] == ['cal0']
    # The calibration is in frame while its modules' NCOs stay: setting
    # them again to the same values is a no-op; another value (another
    # tone set) starts a new frame, and setting them back doesn't restore
    # it
    assert cal.attrs['nco_ids'] == {str(m): crs.nco_ids[m]
                                    for m in (1, 2)}
    assert calibration_in_frame(crs, cal)
    ncos = tune['crs_config'].attrs['config']['tone_sets'][1]
    assert await crs.set_nco({int(m): f for m, f in ncos.items()}) == []
    assert calibration_in_frame(crs, cal)
    assert await crs.set_nco({1: ncos['1'] + 5e6}) == [1]
    assert not calibration_in_frame(crs, cal)
    with pytest.raises(ValueError, match = 'calibrate tone set 1 again'):
        await write_set_tones(crs, cal, rows, {})
    assert await crs.set_nco({1: ncos['1']}) == [1]
    assert not calibration_in_frame(crs, cal)
    assert await crs.set_nco({2: ncos['2']}, force = True) == [2]
    # A run with a stale calibration takes a new one
    said.clear()
    await stream_tone_sets(crs, tune, 'x', 0.05, 6, set_idxs = [1],
                           calibrations = {1: cal}, status = said.append,
                           **kw)
    assert any('out of frame' in m for m in said)
    assert dl.stream_runs(root)[0]['calibration'] == 'tune0/cal1'
    assert calibration_in_frame(crs, tune['cal1'])


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
    assert win.name_edit.text() == 'dark_dec6'
    # Nothing runs before the selected set has a usable calibration
    assert not win.run_btn.isEnabled() and not win.capture_btn.isEnabled()
    assert not win.write_btn.isEnabled()
    assert win.cal_box.count() == 0 and win.cal is None
    # The sweep settings start from sweep_kwargs and go to the sweep
    assert win.cal_params['npoints_fine'].value() == 60
    win.cal_params['npoints_fine'].setValue(50)
    assert win.cal_kwargs()['npoints_fine'] == 50 and \
        win.cal_kwargs()['npoints_gain'] == 10
    win.set_box.setCurrentIndex(1)
    win.calibrate()
    assert win.progress.isVisible() or win.worker.busy
    wait_for(lambda: win.cal is not None and not win.worker.busy)
    assert win.cal['set'] == 1 and win.cal['name'] == 'cal0'
    assert tune['cal0'].attrs['tone_set'] == 1
    assert tune['cal0/fine_sweep/z'].shape[1] == 50
    assert win.cal_box.count() == 1 and win.cal_box.currentData() == 'cal0'
    assert win.capture_btn.isEnabled() and win.write_btn.isEnabled()
    assert win.run_btn.isEnabled()
    assert not win.progress.isVisible() and win.written == {}
    win.tone_box.setCurrentIndex(2)
    win.preview_dec.setValue(0)
    win.preview_s.setValue(0.05)
    win.capture()
    wait_for(lambda: win.preview is not None and not win.worker.busy)
    # Stage 0: only the selected tone's module is captured, and its tones
    # stay on the board
    r = win.tone_box.currentData()
    mods = tune['tones/tone_module'][:]
    assert r in win.preview['rows']
    assert set(mods[win.preview['rows']].tolist()) == {mods[r]}
    assert win.written['tone_set'] == 1 and \
        win.written['calibration'] == 'tune0/cal0'
    assert 'Tones on the board: ' in win.written_label.text()
    for i, n_ts in zip(range(len(MODES)), (1, 2, 2)):
        win.mode_box.setCurrentIndex(i)
        assert len(win.ts_plot.listDataItems()) == n_ts
        # Loop, stream points, and the tone's point
        assert len(win.iq_plot.listDataItems()) == 3
    # The x preview is near zero (no photons in the simulation)
    win.mode_box.setCurrentIndex(0)
    x = win.ts_plot.listDataItems()[0].yData
    assert abs(np.mean(x)) < 1e-6
    # R (outside a text box) fits the plots to the data again
    vb = win.ts_plot.getViewBox()
    vb.enableAutoRange(y = False)
    vb.setYRange(-1., 1.)
    assert not vb.autoRangeEnabled()[1]
    win.name_edit.clearFocus()
    win.notes_edit.clearFocus()
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                      QtCore.Qt.Key_R, QtCore.Qt.NoModifier))
    assert vb.autoRangeEnabled()[1]
    # Run: the selected tone set (1) with its calibration; no name, no run
    win.name_edit.setText('')
    assert not win.run_btn.isEnabled()
    win.name_edit.setText('dark_dec6')
    assert win.run_btn.isEnabled()
    win.run()
    wait_for(lambda: len(win.runs) == 1 and not win.worker.busy)
    assert win.runs[0] == ['tune0/cal0/write0/ts0']
    assert win.runs_list.count() == 1
    runs = dl.stream_runs(root)
    assert runs[0]['name'] == 'dark_dec6' and runs[0]['tone_set'] == 1
    assert 'ts0: dark_dec6' in win.runs_list.item(0).text()
    # The run used the saved calibration (no new one), wrote the whole set
    # (the capture had written one module's chunk, saved nowhere), and
    # left the tones on
    assert runs[0]['calibration'] == 'tune0/cal0'
    assert [c['name'] for c in dl.calibrations(tune)] == ['cal0']
    assert root['tune0/cal0/write0'].attrs['tones_reused'] is False
    n1 = int(np.sum(tune['tones/tone_set'][:] == 1))
    assert len(win.written['rows']) == n1
    # The same run again streams without writing: under the same write
    win.run()
    wait_for(lambda: len(win.runs) == 2 and not win.worker.busy)
    assert win.runs[1] == ['tune0/cal0/write0/ts1']
    # Write tones writes the set again on purpose
    win.write_tones()
    wait_for(lambda: 'written;' in win.status.text() and not win.worker.busy)
    assert len(win.written['rows']) == n1
    # A second calibration of the set is listed and chosen; the first can
    # be picked again, and a run then uses it
    win.calibrate()
    wait_for(lambda: win.cal['name'] == 'cal1' and not win.worker.busy)
    assert win.cal_box.count() == 2 and win.written == {}
    win.cal_box.setCurrentIndex(0)
    assert win.cal['name'] == 'cal0' and win.chosen[1] == 'cal0'
    assert win.run_btn.isEnabled()
    win.run()
    wait_for(lambda: len(win.runs) == 3 and not win.worker.busy)
    assert dl.stream_runs(root)[2]['calibration'] == 'tune0/cal0'
    # Another tone set: nothing until it is calibrated; its calibration
    # sets its NCO, which puts set 1's calibrations out of frame
    win.set_box.setCurrentIndex(0)
    assert win.cal is None and win.cal_box.count() == 0
    assert not win.run_btn.isEnabled() and not win.capture_btn.isEnabled()
    win.calibrate()
    wait_for(lambda: win.cal is not None and not win.worker.busy)
    assert win.cal['name'] == 'cal2' and win.run_btn.isEnabled()
    win.run()
    wait_for(lambda: len(win.runs) == 4 and not win.worker.busy)
    assert dl.stream_runs(root)[3]['calibration'] == 'tune0/cal2'
    assert win.runs[3] == ['tune0/cal2/write0/ts3']
    assert [c['tone_set'] for c in dl.calibrations(tune)] == [1, 1, 0]
    assert [c['in_frame'] for c in win.cals] == [False, False, True]
    win.set_box.setCurrentIndex(1)
    assert win.cal is not None and not win.cal['in_frame']
    assert 'stale' in win.cal_box.currentText() and 'stale' in \
        win.status.text()
    assert not win.run_btn.isEnabled() and not win.capture_btn.isEnabled()
    assert not win.write_btn.isEnabled()
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
    # Tone set 1: calibrated as cal1 for the stage-6 run (one write),
    # and again as cal3 for the stage-0 run, in two writes (one per
    # module); the streams are ts0 and ts1 under the writes
    cal = root['tune0/cal3']
    n = cal['fres'].shape[0]
    rows = np.arange(n)
    for sweep_store, w, stream in (('tune0/cal1', 'write0', 'ts0'),
                                   ('tune0/cal3', 'write0', 'ts1'),
                                   ('tune0/cal3', 'write1', 'ts1')):
        steps = {s.name: s for s in target_sweep_steps(
            store = root, group = sweep_store, ts = w, ts_stream = stream)}
        ft = steps['load_ft'].func(rows)
        zt = steps['load_zt'].func(rows)
        streamed = root[f'{sweep_store}/{w}/sweep_rows'][:]
        ok = np.isin(rows, streamed)
        assert np.all(np.isfinite(ft[ok])) and np.all(np.isnan(ft[~ok]))
        assert np.all(np.isfinite(zt[ok])) and np.all(np.isnan(zt[~ok]))
        np.testing.assert_allclose(ft[ok], cal['fres'][:][ok], atol = 100)
        # One row on its own
        assert np.isfinite(steps['load_ft'].func(streamed[:1])).all()
    # The two writes of tone set 1 cover it once
    a = root['tune0/cal3/write0/sweep_rows'][:]
    b = root['tune0/cal3/write1/sweep_rows'][:]
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
