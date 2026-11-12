"""
Tests for the photon threshold window
(``citkid.pipeline.interactive.photon_thresholds``), offscreen.
"""
import dataclasses
import numpy as np
import pytest
from citkid.pipeline.photon_finder import Thresholds
from citkid.qt_compat import get_qapp
from citkid.signal import photons as ph
from ..signal.test_photons import DT, synth


@pytest.fixture
def qapp():
    """
    Return the QApplication.

    Returns:
    app (QApplication): the application.
    """
    return get_qapp()


@pytest.fixture(scope = 'module')
def tones():
    """
    Make two tones' detection timestreams.

    Returns:
    ds (dict): row: detection timestream.
    """
    rng = np.random.default_rng(0)
    n = int(3 / DT)
    out = {}
    for row in (5, 7):
        times = np.arange(300, n - 300, 350)
        out[row] = ph.detection_timestream(synth(rng, n, times), DT)
    return out


def make_window(tones, fail_row = None):
    """
    Make the window on two tones, recording the reruns.

    Parameters:
    tones (dict): row: detection timestream.
    fail_row (int or None): a row whose detection fails. Default None.

    Returns:
    win (PhotonThresholdWindow): the window.
    calls (list): the rows of each rerun.
    """
    from citkid.pipeline.interactive.photon_thresholds import \
        PhotonThresholdWindow
    ts = dict(rows = np.array([5, 7]), res_idxs = np.array([10, 11]),
              dt = DT)
    res_of = dict(zip([5, 7], [10, 11]))
    calls = []

    def rerun(rows, th):
        """
        Run the detection on some rows with the thresholds.

        Parameters:
        rows (list of int): rows.
        th (Thresholds): thresholds.

        Returns:
        results (dict): row: result or error.
        """
        calls.append(list(rows))
        out = {}
        for r in rows:
            if r == fail_row:
                out[r] = 'no events'
                continue
            t = th.for_tone(res_of[r])
            s = ph.DetectionSettings(of_iterations = 2,
                                     init_hmax = t['init_hmax'])
            out[r] = ph.run_detection(tones[r], DT, s, hmin = t['hmin'],
                                      hmax = t['hmax'])
        return out

    th = Thresholds()
    results = rerun([5, 7], th)
    calls.clear()
    return PhotonThresholdWindow(ts, results, th, rerun), calls


def test_window_thresholds(qapp, tones):
    win, calls = make_window(tones)
    assert win.tone_list.count() == 2
    assert 'res 10' in win.tone_list.item(0).text()
    assert win.hmin_box.value() == 0.5 and win.hmax_box.value() == 1.6
    assert len(win.h_curve.getData()[0]) > 0
    # Dragging a line sets its box, and the other way around
    win.hmax_line.setValue(1.8)
    assert win.hmax_box.value() == pytest.approx(1.8)
    win.hmin_box.setValue(0.45)
    assert win.hmin_line.value() == pytest.approx(0.45)
    # Global thresholds: every tone without its own reruns
    win.scope_all.setChecked(True)
    win.rerun()
    assert calls == [[5, 7]]
    assert (win.thresholds.hmin, win.thresholds.hmax) == \
        pytest.approx((0.45, 1.8))
    assert win.results[5]['hmax'] == pytest.approx(1.8)
    # One tone's own thresholds
    win.tone_list.setCurrentRow(1)
    win.hmin_box.setValue(0.7)
    win.scope_tone.setChecked(True)
    win.rerun()
    assert calls[-1] == [7]
    assert win.thresholds.overrides[11]['hmin'] == pytest.approx(0.7)
    assert win.results[7]['hmin'] == pytest.approx(0.7)
    assert win.tone_list.item(1).text().startswith('*')
    # Selecting a tone shows its thresholds
    win.tone_list.setCurrentRow(0)
    assert win.hmin_box.value() == pytest.approx(0.45)
    win.tone_list.setCurrentRow(1)
    assert win.hmin_box.value() == pytest.approx(0.7)
    # A global change now skips the tone with its own thresholds
    win.scope_all.setChecked(True)
    win.hmax_box.setValue(2.)
    win.rerun()
    assert calls[-1] == [5] and win.thresholds.overrides[11]['hmax'] == \
        pytest.approx(1.8)
    # Back to the global thresholds
    win.reset_tone()
    assert 11 not in win.thresholds.overrides and calls[-1] == [7]
    assert win.results[7]['hmax'] == pytest.approx(2.)
    # All tones plotted together: more peaks than one tone's
    one = win.h_curve.getData()[1].sum()
    win.all_tones.setChecked(True)
    assert win.h_curve.getData()[1].sum() > one
    assert len(win.pulse_curve.getData()[0] or []) == 0
    win.close()


def test_window_failed_tone(qapp, tones):
    win, calls = make_window(tones, fail_row = 7)
    assert 'failed' in win.tone_list.item(1).text()
    win.tone_list.setCurrentRow(1)
    assert 'failed' in win.status.text()
    win.close()


def test_thresholds_for_tone():
    th = Thresholds(hmin = 0.4, overrides = {3: dict(hmax = 2.)})
    assert th.for_tone(3) == dict(hmin = 0.4, hmax = 2., init_hmax = 3.)
    assert th.for_tone(4) == dict(hmin = 0.4, hmax = 1.6, init_hmax = 3.)
    assert th.to_dict()['overrides'] == {'3': dict(hmax = 2.)}
    copy = dataclasses.replace(th, overrides = dict(th.overrides))
    copy.overrides[5] = dict(hmin = 1.)
    assert 5 not in th.overrides
