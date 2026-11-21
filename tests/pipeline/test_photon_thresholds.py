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


from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def _press(widget, key):
    """
    Send a key press to a widget, as the keyboard would.

    Parameters:
    widget (QWidget): the widget with the focus.
    key (Qt.Key): the key.
    """
    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier))


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


def make_window(tones, fail_row = None, data = None, not_run = ()):
    """
    Make the window on two tones, recording the reruns.

    Parameters:
    tones (dict): row: detection timestream.
    fail_row (int or None): a row whose detection fails. Default None.
    data (callable or None): the window's ``data``. Default None.
    not_run (tuple of int): rows left without a result. Default none.

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
    results = rerun([r for r in (5, 7) if r not in not_run], th)
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
    # Global thresholds: every tone without its own reruns, one at a
    # time (the progress bar shows while they run, then hides)
    shown = []
    win.progress.setVisible = lambda v: shown.append(v)
    win.scope_all.setChecked(True)
    win.rerun()
    assert calls == [[5], [7]]
    # The scope goes back to the selected tone afterwards
    assert win.scope_tone.isChecked() and not win.scope_all.isChecked()
    assert shown == [True] and win.progress.value() == 2
    assert win.progress.isHidden()
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
    # * means reviewed: tone 0 was looked at and left, tone 1 got its own
    # values; the earlier apply to all did not mark anything by itself
    assert win.thresholds.reviewed == {10, 11}
    assert win.thresholds.to_dict()['reviewed'] == [10, 11]
    assert win.tone_list.item(0).text().startswith('*')
    # Mark bad (B), and clear it again: no rerun either way
    n_calls = len(calls)
    win._ask_reason = lambda: 'weird pulse shape'
    win.toggle_bad()
    assert win.thresholds.bad == {11: 'weird pulse shape'}
    assert win.thresholds.is_bad(11)
    assert win.tone_list.item(1).text().startswith('x')
    assert win.status.text().startswith('MARKED BAD')
    assert win.bad_btn.text().startswith('Clear')
    assert win.thresholds.to_dict()['bad'] == {'11': 'weird pulse shape'}
    assert 'weird pulse shape' in win.tone_list.item(1).text()
    assert 'MARKED BAD: weird pulse shape' in win.status.text()
    win.toggle_bad()
    assert win.thresholds.bad == {} and not win.thresholds.is_bad(11)
    assert win.tone_list.item(1).text().startswith('*')
    assert not win.status.text().startswith('MARKED BAD')
    assert len(calls) == n_calls
    # The decay fit: the boxes show the tone's values; Apply fit gives
    # the tone its own, or every tone the same (dropping their own), with
    # no rerun; the plot title says what was fit
    assert win.tau_skip_box.value() == 2 and win.tau_fit_box.value() == 1.
    win.tau_skip_box.setValue(3)
    win.tau_fit_box.setValue(0.5)
    win.fit_scope_tone.setChecked(True)
    win.apply_fit()
    assert win.thresholds.for_tone(11)['tau_skip'] == 3
    assert win.thresholds.for_tone(11)['tau_fit_s'] == pytest.approx(5e-4)
    assert win.thresholds.for_tone(10)['tau_skip'] == 2
    assert 'from 3 samples' in win.pulse_plot.titleLabel.text
    tau1 = win.tau_of(1)[0]
    assert np.isfinite(tau1) and win.tau_of(1)[2] == round(0.5e-3 / DT)
    win.tau_fit_box.setValue(1.)
    win.fit_scope_all.setChecked(True)
    win.apply_fit()
    assert win.fit_scope_tone.isChecked()
    assert win.thresholds.tau_skip == 3 and win.thresholds.tau_fit_s == 1e-3
    assert 'tau_skip' not in win.thresholds.overrides.get(11, {})
    assert win.tau_of(0)[2] == round(1e-3 / DT) and len(calls) == n_calls
    # The overview: histograms over the tones, limits at the data's edges
    # at first, then where the lines go; the tones outside are listed and
    # can be the only ones in the tone list; a click selects a tone
    win.open_overview()
    ov = win.overview
    assert ov.isVisible() and set(win.thresholds.limits) == \
        {'height', 'tau', 'rate'}
    values = win.overview_values()
    assert np.isfinite(values['height']).all() and \
        np.isfinite(values['tau']).all()
    lo, hi = win.thresholds.limits['rate']
    assert lo == pytest.approx(values['rate'].min())
    assert hi == pytest.approx(values['rate'].max())
    assert win.outside_limits() == set() and ov.outside_list.count() == 0
    # Move a low limit between the two tones' values (the quantity that
    # differs most between them: the tones share one pulse train)
    key = max(('height', 'tau', 'rate'),
              key = lambda k: np.ptp(values[k]) / abs(values[k]).max())
    low_tone = int(np.argmin(values[key]))
    assert values[key].min() < values[key].max()
    k = 1e6 if key == 'tau' else 1.
    ov.lines[key][0].setValue(values[key].mean() * k)
    ov._on_line()
    assert win.outside_limits() == {low_tone}
    assert ov.outside_list.count() == 1 and key in \
        ov.outside_list.item(0).text()
    win.only_outside.setChecked(True)
    assert win._only_outside and win.tone_list.count() == 1
    assert win.current() == low_tone
    assert 'outside the limits: 1 of 2' in win.list_label.text()
    # A change that brings the selected tone inside the limits keeps it
    # listed and selected (applying does not jump to another tone)
    ov.lines[key][0].setValue(values[key].min() * k)
    ov._on_line()
    assert win.outside_limits() == set()
    assert win.tone_list.count() == 1 and win.current() == low_tone
    ov.lines[key][0].setValue(values[key].mean() * k)
    ov._on_line()
    other = 1 - low_tone
    win.select_tone(other)
    assert not win._only_outside and win.tone_list.count() == 2
    assert not win.only_outside.isChecked()
    assert win.current() == other
    ov._on_item(ov.outside_list.item(0))
    assert win.current() == low_tone
    # A tone marked bad is flagged in the overview list and left out of
    # the histograms
    assert ov.curves[key].getData()[1].sum() == 2
    win._ask_reason = lambda: 'high noise'
    win.toggle_bad()
    assert win.current() in win.outside_limits()
    # Leaving the bad tones out of the list keeps the selected one
    # until you move on; selecting a bad tone lists them again
    bad_tone = win.current()
    win.include_bad.setChecked(False)
    assert win.tone_list.count() == 2 and win.current() == bad_tone
    win.select_tone(1 - bad_tone)
    win._fill_list()
    assert win.tone_list.count() == 1 and win.current() == 1 - bad_tone
    win.select_tone(bad_tone)
    assert win.include_bad.isChecked() and win.tone_list.count() == 2
    assert any(ov.outside_list.item(i).text().startswith('x res')
               for i in range(ov.outside_list.count()))
    assert ov.curves[key].getData()[1].sum() == 1
    win.toggle_bad()
    assert ov.curves[key].getData()[1].sum() == 2
    ov.close()
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
    assert th.for_tone(3) == dict(hmin = 0.4, hmax = 2., init_hmax = 3.,
                                  tau_skip = 2, tau_fit_s = 1e-3)
    assert th.for_tone(4) == dict(hmin = 0.4, hmax = 1.6, init_hmax = 3.,
                                  tau_skip = 2, tau_fit_s = 1e-3)
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
    # A failed tone shows its raw timestream alone, to see why it failed
    assert win.snippet is None and 'detection failed' in \
        win.snip_label.text()
    assert len(win.y_curve.getData()[0] or []) == 0
    assert len(win.raw_curve.getData()[0]) == length
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
    assert calls[-2:] == [[5], [7]]
    assert win.thresholds.hmin == pytest.approx(0.4)
    assert win.current() == 1 and win.scope_tone.isChecked()
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

@pytest.mark.parametrize('answers, choice', [
    (['Continue'], 'continue'),
    (['Overwrite', 'Overwrite'], 'overwrite'),
    (['Overwrite', 'Cancel'], None),
    (['Cancel'], None),
    ([None], None),
])
def test_ask_existing_result(qapp, monkeypatch, answers, choice):
    """
    The existing-result popup offers Continue (default), Overwrite and
    Cancel; Overwrite asks once more, with Cancel as the default.
    """
    import citkid.pipeline.interactive.photon_thresholds as pt
    seen = []

    def fake_box(title, text, buttons, default, icon = None):
        seen.append((title, buttons, default))
        assert 'photons/x' in title and 'photons/x' in text
        return answers[len(seen) - 1]

    monkeypatch.setattr(pt, '_message_box', fake_box)
    assert pt.ask_existing_result('x', 'run tone set 0') == choice
    assert seen[0][1:] == (['Continue', 'Overwrite', 'Cancel'], 'Continue')
    if answers[0] == 'Overwrite':
        assert seen[1][1:] == (['Overwrite', 'Cancel'], 'Cancel')
    assert len(seen) == len(answers)


def test_window_tone_marks(qapp, tones):
    """
    With a store root, marking bad, clearing and adding a note write the
    store's tone marks, and the status shows them.
    """
    import zarr
    from citkid import daq_layout as dl
    win, _ = make_window(tones)
    root = zarr.group()
    win.root, win.stage = root, 'photons/x'
    win._ask_reason = lambda: 'double resonance'
    win._ask_note = lambda: 'keep an eye on it'
    win.tone_list.setCurrentRow(0)
    win.toggle_bad()
    # A bare reason from the dialog stand-in: the window's default
    # scope, local to this result
    assert dl.bad_tones(root, 'photons/x') == \
        {10: dl.tone_marks(root, 10)[0]}
    assert dl.bad_tones(root) == {}
    assert dl.tone_marks(root, 10)[0]['stage'] == 'photons/x'
    assert dl.tone_marks(root, 10)[0]['scope'] == 'local'
    win.add_note()
    assert [e['kind'] for e in dl.tone_marks(root, 10)] == ['bad', 'note']
    assert 'MARKED BAD: double resonance' in win.status.text()
    assert '1 note' in win.status.text()
    win.toggle_bad()
    assert dl.bad_tones(root, 'photons/x') == {}
    assert [e['kind'] for e in dl.tone_marks(root, 10)] == \
        ['bad', 'note', 'clear']
    assert '1 note' in win.status.text()
    # The dialog's (reason, scope) answer: a global mark
    win._ask_reason = lambda: ('high noise', 'global')
    win.toggle_bad()
    assert dl.tone_marks(root, 10)[-1]['scope'] == 'global'
    assert set(dl.bad_tones(root)) == {10}
    win.toggle_bad()
    assert dl.bad_tones(root) == {}
    # Cancelled dialogs write nothing
    win._ask_note = lambda: None
    win.add_note()
    assert len(dl.tone_marks(root, 10)) == 5
    win.close()


def test_keys_work_from_the_list(qapp, tones):
    """
    A and D (and the other letter shortcuts) move through the tones even
    while the tone list has the keyboard focus.
    """
    win, _ = make_window(tones)
    win.tone_list.setFocus()
    assert win.current() == 0
    _press(win.tone_list, QtCore.Qt.Key_D)
    assert win.current() == 1
    _press(win.tone_list, QtCore.Qt.Key_A)
    assert win.current() == 0
    win._ask_reason = lambda: 'high noise'
    _press(win.tone_list, QtCore.Qt.Key_B)
    assert win.thresholds.is_bad(10)
    win.close()


def test_tones_not_run(qapp, tones):
    """
    A tone without a result is listed "not run"; selecting it runs it
    with the thresholds as they stand, and applying to all tones runs
    the tones not run too.
    """
    win, calls = make_window(tones, not_run = (7,))
    assert win.tone_list.count() == 2
    assert win.tone_list.item(1).text() == '  res 11: not run'
    assert 7 not in win.results and calls == []
    win.tone_list.setCurrentRow(1)
    assert calls == [[7]] and isinstance(win.results[7], dict)
    assert 'pulses' in win.tone_list.item(1).text()
    assert len(win.h_curve.getData()[0]) > 0
    win2, calls2 = make_window(tones, not_run = (7,))
    win2.scope_all.setChecked(True)
    win2.rerun()
    assert calls2 == [[5], [7]] and isinstance(win2.results[7], dict)


def test_panel_scrolls(qapp, tones):
    """
    The left panel sits in a scroll area as wide as the panel, so a
    short screen scrolls it instead of squeezing the buttons together.
    """
    win, _ = make_window(tones)
    win.show()
    qapp.processEvents()
    assert win._scroll.widget() is win._panel
    assert win._scroll.width() > win._panel.width()
    assert win._panel.minimumSizeHint().height() > 300
    assert win._scroll.minimumHeight() == 300
