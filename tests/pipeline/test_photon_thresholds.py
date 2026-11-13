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


def make_window(tones, fail_row = None, data = None):
    """
    Make the window on two tones, recording the reruns.

    Parameters:
    tones (dict): row: detection timestream.
    fail_row (int or None): a row whose detection fails. Default None.
    data (callable or None): the window's ``data``. Default None.

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
    return PhotonThresholdWindow(ts, results, th, rerun, data = data), calls


def test_window_thresholds(qapp, tones):
    win, calls = make_window(tones)
    assert win.tone_list.count() == 2
    assert 'res 10' in win.tone_list.item(0).text()
    assert win.hmin_box.value() == 0.5 and win.hmax_box.value() == 1.6
    # Rerun applies to the selected tone by default
    assert win.scope_tone.isChecked()
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
    # Back to the defaults (the window's starting thresholds): the global
    # ones differ now, so the tone keeps its own, equal to the defaults
    win.reset_tone()
    assert win.thresholds.overrides[11] == dict(hmin = 0.5, hmax = 1.6,
                                                init_hmax = 3.)
    assert calls[-1] == [7] and win.results[7]['hmax'] == pytest.approx(1.6)
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


def test_window_snippet_and_fit(qapp, tones):
    """
    With data, the window shows a snippet of the selected tone (raw and
    filtered, in photons) around its first pulse, marks the filter peaks
    by the thresholds, moves along the record, and fits the mean pulse's
    decay.
    """
    win, calls = make_window(tones, data = tones.get)
    res = win.results[5]
    snip = win.snippet
    # 0.2 s, or 400 samples if that is longer, around the first pulse
    length = max(int(round(0.2 / DT)), 400)
    assert snip['row'] == 5 and snip['stop'] - snip['start'] == length
    first = int(res['events'][0])
    assert snip['start'] <= first < snip['stop']
    t, y = win.y_curve.getData()
    assert len(y) == length and t[0] == pytest.approx(snip['start'] * DT)
    # The filter output in the snippet is the detection's: its value at
    # the event is the histogram's height
    assert y[first - snip['start']] == pytest.approx(
        res['coinc_h'][np.flatnonzero(res['coinc_idx'] == first)[0]],
        rel = 1e-3)
    assert len(win.raw_curve.getData()[1]) == length
    assert 'pulses in [0.5, 1.6]' in win.snip_label.text()
    n_in = len(win.event_marks.data)
    assert n_in >= 1
    # Raising hmin above the pulses turns them into peaks below hmin,
    # from the box or by dragging the horizontal line (synced with the
    # histogram's line)
    win.hmin_box.setValue(1.3)
    assert len(win.event_marks.data) < n_in
    assert len(win.low_marks.data) > 0
    assert win.hmin_hline.value() == pytest.approx(1.3)
    win.hmin_hline.setValue(0.4)
    assert win.hmin_box.value() == pytest.approx(0.4)
    assert win.hmin_line.value() == pytest.approx(0.4)
    assert len(win.event_marks.data) == n_in
    win.hmax_hline.setValue(0.9)
    assert win.hmax_box.value() == pytest.approx(0.9)
    assert len(win.high_marks.data) > 0
    win.hmax_box.setValue(1.6)
    # Next/Prev move to the neighbouring filter peaks; the slider and the
    # length move the window too
    center = (snip['start'] + snip['stop']) // 2
    win.jump(1)
    c2 = (win.snippet['start'] + win.snippet['stop']) // 2
    assert c2 > center
    win.jump(-1)
    assert abs((win.snippet['start'] + win.snippet['stop']) // 2 - center) \
        <= 1
    win.snip_pos.setValue(1000)
    assert win.snippet['stop'] == len(tones[5])
    win.snip_len.setValue(0.01)
    assert win.snippet['stop'] - win.snippet['start'] == round(0.01 / DT)
    # The mean pulse's decay fit is drawn with its decay time
    assert len(win.fit_curve.getData()[0]) > 4
    assert 'tau =' in win.pulse_plot.titleLabel.text
    # Another tone: its own snippet; a failed tone has none
    win.tone_list.setCurrentRow(1)
    assert win.snippet['row'] == 7
    win.close()
    win, _ = make_window(tones, fail_row = 7, data = tones.get)
    win.tone_list.setCurrentRow(1)
    assert win.snippet is None and 'No data' in win.snip_label.text()
    assert len(win.y_curve.getData()[0] or []) == 0
    win.close()
    # Without data: no snippet, the rest as before
    win, _ = make_window(tones)
    assert win.snippet is None and 'No data' in win.snip_label.text()
    win.close()


def test_window_units_and_rescale(qapp, tones):
    """
    Heights and thresholds are in kHz/GHz; the plots can be shown in photons
    (each tone by its own mean photon): the histogram, lines, mean pulse
    and snippet scale, the boxes stay in kHz/GHz; R and the button fit the
    plots again.
    """
    from pyqtgraph.Qt import QtCore, QtGui
    win, _ = make_window(tones, data = tones.get)
    res = win.results[5]
    k = 1 / (res['amp_ph'] * 1e6)
    assert 'mean photon:' in win.status.text()
    assert win.in_khz() and win._scale() == 1.
    assert win.h_plot.getAxis('bottom').labelText == 'Filter height (kHz/GHz)'
    h_khz = win.h_curve.getData()[0][-1]
    pulse_khz = win.pulse_curve.getData()[1].max()
    y_khz = win.y_curve.getData()[1].max()
    # The mean pulse is drawn in kHz/GHz: its maximum is the template
    # amplitude
    assert pulse_khz == pytest.approx(res['amp_ph'] * 1e6, rel = 1e-6)
    win.units_box.setCurrentIndex(1)
    assert not win.in_khz() and win._scale() == pytest.approx(k)
    assert win.h_plot.getAxis('bottom').labelText == \
        'Filter height (photons)'
    assert win.hmin_line.value() == pytest.approx(0.5 * k)
    assert win.hmax_hline.value() == pytest.approx(1.6 * k)
    assert win.hmin_box.value() == 0.5
    assert win.h_curve.getData()[0][-1] == pytest.approx(h_khz * k)
    assert win.pulse_curve.getData()[1].max() == pytest.approx(pulse_khz * k)
    assert win.y_curve.getData()[1].max() == pytest.approx(y_khz * k)
    assert win.snippet['y'].max() == pytest.approx(y_khz)
    # Dragging a line in photons sets the box in kHz/GHz
    win.hmin_hline.setValue(0.6 * k)
    assert win.hmin_box.value() == pytest.approx(0.6)
    assert win.hmin_line.value() == pytest.approx(0.6 * k)
    # All tones together: each by its own mean photon
    win.all_tones.setChecked(True)
    assert len(win.h_curve.getData()[0]) > 0
    win.all_tones.setChecked(False)
    win.units_box.setCurrentIndex(0)
    assert win.hmin_line.value() == pytest.approx(0.6)
    # Zooming turns auto-range off; R (outside a box) and the button fit
    # the data again
    vb = win.snippet_plot.getViewBox()
    vb.setXRange(0., 0.001, padding = 0)
    assert not vb.autoRangeEnabled()[0]
    win.hmin_box.clearFocus()
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                      QtCore.Qt.Key_R, QtCore.Qt.NoModifier))
    assert vb.autoRangeEnabled()[0]
    vb.setXRange(0., 0.001, padding = 0)
    win.rescale_btn.click()
    assert vb.autoRangeEnabled()[0]
    win.close()


def test_window_navigation_and_defaults(qapp, tones):
    """
    A and D (and the buttons) move between tones, applying thresholds
    changed but not rerun first; Revert to defaults puts the starting
    thresholds back on a tone; the snippet length follows the median
    pulse rate.
    """
    from pyqtgraph.Qt import QtCore, QtGui
    import citkid.pipeline.interactive.photon_thresholds as pt
    win, calls = make_window(tones, data = tones.get)
    assert win.current() == 0 and win.snippet['row'] == 5
    # Ten pulses at the median rate, but never shorter than the floor
    rates = [r['rate'] for r in win.results.values()]
    expect = max(0.2, 400 * DT, 10 / np.median(rates))
    assert win.snip_len.value() == pytest.approx(expect, abs = 1e-3)
    # D moves on; a changed threshold is applied to the tone left behind
    # (the default scope: that tone only) before moving
    win.hmax_box.setValue(1.9)
    assert win.pending(0) and not calls
    win.hmin_box.clearFocus()
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                      QtCore.Qt.Key_D, QtCore.Qt.NoModifier))
    assert win.current() == 1 and calls == [[5]]
    assert win.thresholds.overrides[10]['hmax'] == pytest.approx(1.9)
    assert win.results[5]['hmax'] == pytest.approx(1.9)
    assert win.hmax_box.value() == 1.6 and win.snippet['row'] == 7
    # Nothing pending: A moves back without a rerun; the end stops
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                      QtCore.Qt.Key_A, QtCore.Qt.NoModifier))
    assert win.current() == 0 and calls == [[5]]
    assert win.hmax_box.value() == pytest.approx(1.9)
    win.prev_tone_btn.click()
    assert win.current() == 0
    win.next_tone_btn.click()
    assert win.current() == 1
    # Revert to defaults (Ctrl+D's slot): the global thresholds are still
    # the defaults, so tone 0 just drops its own
    win.tone_list.setCurrentRow(0)
    win.reset_tone()
    assert 10 not in win.thresholds.overrides and calls[-1] == [5]
    assert win.results[5]['hmax'] == pytest.approx(1.6)
    assert win.hmax_box.value() == 1.6
    # ... and does nothing when there is nothing to revert
    n = len(calls)
    win.reset_tone()
    assert len(calls) == n
    # Pending global values are applied to all tones on moving when the
    # scope is all tones
    win.scope_all.setChecked(True)
    win.hmin_box.setValue(0.4)
    win.step_tone(1)
    assert calls[-1] == [5, 7] and win.thresholds.hmin == pytest.approx(0.4)
    assert win.current() == 1
    win.close()


class _FakeMouseEvent:
    """
    Stand-in for pyqtgraph's mouse events, in scene = data coordinates.
    """

    def __init__(self, x, y, x0 = None, y0 = None, shift = True,
                 finish = True):
        from pyqtgraph.Qt import QtCore
        self._pos = QtCore.QPointF(x, y)
        self._down = QtCore.QPointF(x if x0 is None else x0,
                                    y if y0 is None else y0)
        self._mod = QtCore.Qt.ShiftModifier if shift else \
            QtCore.Qt.NoModifier
        self._finish = finish
        self.accepted = False

    def button(self):
        from pyqtgraph.Qt import QtCore
        return QtCore.Qt.LeftButton

    def modifiers(self):
        return self._mod

    def scenePos(self):
        return self._pos

    def buttonDownScenePos(self):
        return self._down

    def isFinish(self):
        return self._finish

    def accept(self):
        self.accepted = True


def test_click_and_drag_set_thresholds(qapp, tones, monkeypatch):
    """
    A click on a histogram moves its nearer line and a drag carries it
    along; on the snippet a click moves the nearer of hmin and hmax to the
    click height, and a drag sets both from the drag's bottom and top.
    The plots have no pan, zoom, or menu: pyqtgraph's handlers never run.
    """
    import pyqtgraph as pg
    from citkid.pipeline.interactive.photon_thresholds import \
        ThresholdViewBox
    seen = []
    monkeypatch.setattr(pg.ViewBox, 'mouseClickEvent',
                        lambda self, ev: seen.append('click'))
    monkeypatch.setattr(pg.ViewBox, 'mouseDragEvent',
                        lambda self, ev, axis = None: seen.append('drag'))
    monkeypatch.setattr(pg.ViewBox, 'wheelEvent',
                        lambda self, ev, axis = None: seen.append('wheel'))
    win, calls = make_window(tones, data = tones.get)
    for plot in (win.init_plot, win.h_plot, win.pulse_plot,
                 win.snippet_plot):
        vb = plot.getViewBox()
        assert isinstance(vb, ThresholdViewBox)
        assert vb.mouseEnabled() == [False, False] and not vb.menuEnabled()
        # Data coordinates stand in for scene coordinates
        monkeypatch.setattr(vb, 'mapSceneToView', lambda p: p)
    # Height histogram: the nearer threshold moves to the click
    assert (win.hmin_box.value(), win.hmax_box.value()) == (0.5, 1.6)
    win.h_plot.getViewBox().mouseClickEvent(_FakeMouseEvent(0.7, 5.))
    assert win.hmin_box.value() == pytest.approx(0.7)
    assert win.hmax_box.value() == 1.6
    win.h_plot.getViewBox().mouseClickEvent(
        _FakeMouseEvent(1.9, 5., shift = False))
    assert win.hmax_box.value() == pytest.approx(1.9)
    assert win.hmin_line.value() == pytest.approx(0.7)
    assert win.hmax_hline.value() == pytest.approx(1.9)
    # A drag carries the line nearer to its start, even past the other
    vb = win.h_plot.getViewBox()
    vb.mouseDragEvent(_FakeMouseEvent(1.0, 5., x0 = 0.8, finish = False))
    assert win.hmin_box.value() == pytest.approx(1.0)
    vb.mouseDragEvent(_FakeMouseEvent(2.2, 5., x0 = 0.8, finish = False))
    assert win.hmin_box.value() == pytest.approx(2.2)
    assert win.hmax_box.value() == pytest.approx(1.9)
    vb.mouseDragEvent(_FakeMouseEvent(0.6, 5., x0 = 0.8))
    assert win.hmin_box.value() == pytest.approx(0.6)
    # ... and the next click chooses afresh
    vb.mouseClickEvent(_FakeMouseEvent(2.0, 5.))
    assert win.hmax_box.value() == pytest.approx(2.0)
    assert win.hmin_box.value() == pytest.approx(0.6)
    # First-pass histogram: the init_hmax line, by click or drag
    win.init_plot.getViewBox().mouseClickEvent(_FakeMouseEvent(2.5, 5.))
    assert win.init_hmax_box.value() == pytest.approx(2.5)
    assert win.init_line.value() == pytest.approx(2.5)
    win.init_plot.getViewBox().mouseDragEvent(
        _FakeMouseEvent(3.5, 5., x0 = 2.5))
    assert win.init_hmax_box.value() == pytest.approx(3.5)
    # Snippet: click moves the nearer line by height; drag sets both
    win.snippet_plot.getViewBox().mouseClickEvent(
        _FakeMouseEvent(0.01, 0.5))
    assert win.hmin_box.value() == pytest.approx(0.5)
    ev = _FakeMouseEvent(0.02, 1.4, x0 = 0.01, y0 = 0.6, finish = False)
    win.snippet_plot.getViewBox().mouseDragEvent(ev)
    assert ev.accepted
    assert (win.hmin_box.value(), win.hmax_box.value()) == \
        pytest.approx((0.6, 1.4))
    assert win.hmin_hline.value() == pytest.approx(0.6)
    # Upward or downward drags alike
    win.snippet_plot.getViewBox().mouseDragEvent(
        _FakeMouseEvent(0.02, 0.55, x0 = 0.01, y0 = 1.2))
    assert (win.hmin_box.value(), win.hmax_box.value()) == \
        pytest.approx((0.55, 1.2))
    # In photons the click heights are converted back to kHz/GHz
    win.units_box.setCurrentIndex(1)
    k = 1 / (win.results[5]['amp_ph'] * 1e6)
    win.h_plot.getViewBox().mouseClickEvent(_FakeMouseEvent(0.4 * k, 5.))
    assert win.hmin_box.value() == pytest.approx(0.4)
    win.units_box.setCurrentIndex(0)
    # Other buttons and the wheel do nothing, and pyqtgraph never pans,
    # zooms or opens its menu
    before = (win.hmin_box.value(), win.hmax_box.value())
    right = _FakeMouseEvent(1.0, 5.)
    right.button = lambda: pg.QtCore.Qt.RightButton
    win.h_plot.getViewBox().mouseClickEvent(right)
    win.h_plot.getViewBox().mouseDragEvent(right)
    win.h_plot.getViewBox().wheelEvent(right)
    assert right.accepted
    assert (win.hmin_box.value(), win.hmax_box.value()) == before
    assert seen == []
    # Nothing was rerun by any of this
    assert calls == []
    win.close()


def test_window_panel_width_and_filter_end(qapp, tones):
    """
    The left panel keeps its width while the status text changes (it
    used to shrink during a rerun), and the mean pulse plot marks where
    the filter's template ends.
    """
    win, calls = make_window(tones)
    win.show()
    qapp.processEvents()
    width = win._panel.width()
    assert width == 330
    win.status.setText('Running 2 tones... ' * 10)
    qapp.processEvents()
    assert win._panel.width() == width
    win.status.setText('')
    qapp.processEvents()
    assert win._panel.width() == width
    res = win.results[5]
    assert win.filter_end_line.isVisible()
    assert win.filter_end_line.value() == pytest.approx(
        res['filter_post'] * DT * 1e3)
    assert res['filter_post'] < res['post']
    win.all_tones.setChecked(True)
    assert not win.filter_end_line.isVisible()
    win.close()
