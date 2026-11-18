"""
Tests of the high-energy event (HEE) tail window
(citkid.pipeline.interactive.hee_tails).
"""
import numpy as np
import pytest
import zarr
from citkid import daq_layout as dl
from citkid.pipeline import hee_finder as cf
from citkid.qt_compat import get_qapp
from .test_hee_finder import SETTINGS
from ..signal.test_hee import rays
from ..signal.test_photons import DT, TAU


@pytest.fixture
def qapp():
    """
    Return the QApplication.

    Returns:
    app (QApplication): the application.
    """
    return get_qapp()


def make_window(rays, fail_row = None, data = False):
    """
    Make the window on the three tones, recording the refits.

    Parameters:
    rays (tuple): the ``rays`` fixture.
    fail_row (int or None): a row without a result. Default None.
    data (bool): If True, give the window the timestreams. Default
        False.

    Returns:
    win (HeeTailWindow): the window.
    calls (list): the rows of each refit.
    """
    from citkid.pipeline.interactive.hee_tails import HeeTailWindow
    ds, times, _ = rays
    ts = dict(rows = np.array([0, 1, 2]), res_idxs = np.array([10, 11, 12]),
              dt = DT, start = 0, stop = len(ds[0]))
    th = cf.TailThresholds(level = 3., fit_s = 1e-3)
    results = {}
    for r in range(3):
        if r == fail_row:
            results[r] = 'not finite'
            continue
        results[r] = cf.analyze_tone(ds[r], times, DT, SETTINGS,
                                     th.for_tone(10 + r))
        results[r]['res_idx'] = 10 + r
        results[r]['sign'] = -1
    calls = []

    def refit(rows, th):
        """
        Refit some rows (recorded).

        Parameters:
        rows (list of int): rows.
        th (TailThresholds): thresholds.

        Returns:
        results (dict): row: result.
        """
        calls.append(list(rows))
        return cf.refit_timestream(ts, win.results, th, rows)

    win = HeeTailWindow(ts, results, th, refit,
                           data = (lambda row: ds[row]) if data else None)
    return win, calls


def test_envelope():
    from citkid.pipeline.interactive.hee_tails import envelope
    d = np.arange(100.)
    idx, y = envelope(d, nbins = 10)
    assert len(idx) == 20 and len(y) == 20
    assert y[0] == 0 and y[1] == 9 and y[-1] == 99
    idx2, y2 = envelope(d[:15], nbins = 10)
    np.testing.assert_array_equal(y2, d[:15])


def test_window_values_and_apply(qapp, rays):
    """
    The window lists the tones with their fits, shows the selected
    tone's threshold, level and fit length (lines synced both ways),
    draws the timestream with the pulses and the aligned mean with its
    fit, and applies to the tone or to all.
    """
    win, calls = make_window(rays, data = True)
    assert win.tone_list.count() == 3
    assert 'res 10' in win.tone_list.item(0).text() and 'pulses' in \
        win.tone_list.item(0).text()
    assert win.level_box.value() == 3. and win.fit_box.value() == 1.
    assert win.hmin_box.value() == 0.
    assert win.scope_tone.isChecked()
    # The timestream envelope, the pulses marked, the aligned pulses,
    # the mean and the fit
    assert len(win.ts_curve.getData()[0]) > 100
    res = win.results[0]
    assert len(win.used_marks.data) == res['n_used'] > 0
    assert len(win.other_marks.data) == 0
    assert len(win.avg_curve.getData()[0]) == 170
    assert len(win.fit_curve.getData()[0]) == res['n_fit']
    assert 'tau =' in win.pulse_plot.titleLabel.text
    assert 'lifetime of the mean tail' in win.status.text()
    # The default y range stops near the 99th percentile of the pulses;
    # the mouse can zoom (no fixed view box); R puts the range back
    top = max(float(np.percentile(res['h'], 99)), 1.)
    assert win.ts_plot.vb.viewRange()[1][1] == pytest.approx(1.2 * top)
    assert win.ts_plot.vb.state['mouseEnabled'] == [True, True]
    win.ts_plot.vb.setYRange(0, 5)
    win.rescale()
    assert win.ts_plot.vb.viewRange()[1][1] == pytest.approx(1.2 * top)
    # The mean pulse shows the fit range and its own peak, not the single
    # pulses' peaks
    (x0, x1), (y0, y1) = win.pulse_plot.vb.viewRange()
    assert x1 == pytest.approx(1.1 * res['fit_s'] * 1e3)
    peak = float(np.nanmax(res['avg'])) * 1e6
    assert y1 == pytest.approx(1.2 * peak)
    assert y1 < np.nanmax(res['aligned']) * 1e6
    # Lines and boxes follow each other
    win.hmin_line.setValue(8.)
    assert win.hmin_box.value() == pytest.approx(8.)
    win.level_box.setValue(4.)
    assert win.level_line.value() == pytest.approx(4.)
    win.level_line.setValue(3.)
    assert win.level_box.value() == pytest.approx(3.)
    # Apply to this tone: its own values, one refit; pulses below the
    # threshold are left out of the average and marked red
    rows = win.apply()
    assert rows == [0] and calls == [[0]]
    assert win.thresholds.overrides[10] == dict(hmin = 8., level = 3.,
                                                fit_s = 1e-3)
    res = win.results[0]
    assert res['hmin'] == 8. and not res['in_window'].all()
    assert len(win.other_marks.data) == int((~res['used']).sum()) > 0
    assert win.tone_list.item(0).text().startswith('* res 10')
    # Another tone keeps the global values
    win.tone_list.setCurrentRow(1)
    assert win.hmin_box.value() == 0.
    # Apply to all: every tone refit, own values dropped, scope back to
    # this tone
    win.scope_all.setChecked(True)
    win.fit_box.setValue(0.5)
    win.rerun()
    assert calls[-1] == [0, 1, 2] and win.scope_tone.isChecked()
    assert win.thresholds.fit_s == 0.5e-3 and win.thresholds.overrides == {}
    assert all(win.results[r]['fit_s'] == 0.5e-3 for r in range(3))
    assert abs(win.results[2]['tau'] - TAU) < 0.1 * TAU
    # * means reviewed: tone 0 (own values, left), not tone 1 (shown)
    # nor tone 2 (never looked at)
    assert win.tone_list.item(0).text().startswith('*')
    assert not win.tone_list.item(2).text().startswith('*')
    assert win.thresholds.reviewed == {10}
    # D from the list moves on too; leaving a tone reviews it
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    win.tone_list.setFocus()
    QtWidgets.QApplication.sendEvent(win.tone_list, QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_D, QtCore.Qt.NoModifier))
    assert win.current() == 2 and win.thresholds.reviewed == {10, 11}
    win.close()
    assert win.thresholds.reviewed == {10, 11, 12}


def test_window_bad_overview_and_failed(qapp, rays):
    """
    Marking bad asks a reason and shows it; the overview histograms the
    tones' values with limits that filter the list; a failed tone shows
    no plots; without data there is no timestream.
    """
    win, _ = make_window(rays, fail_row = 2)
    assert len(win.ts_curve.getData()[0] or []) == 0
    win.tone_list.setCurrentRow(2)
    assert 'failed' in win.tone_list.item(2).text() and 'Failed' in \
        win.status.text()
    assert len(win.avg_curve.getData()[0] or []) == 0
    win.tone_list.setCurrentRow(1)
    win._ask_reason = lambda: 'low or no pulses'
    win.toggle_bad()
    assert win.thresholds.bad == {11: 'low or no pulses'}
    assert win.tone_list.item(1).text().startswith('x res 11: bad (low')
    assert win.status.text().startswith('MARKED BAD: low or no pulses')
    win.toggle_bad()
    assert win.thresholds.bad == {}
    win.open_overview()
    ov = win.overview
    assert ov.isVisible() and set(win.thresholds.limits) == \
        {'height', 'tau', 'rate'}
    values = win.overview_values()
    assert values['rate'][0] == pytest.approx(
        win.results[0]['n_used'] / (len(rays[0][0]) * DT))
    assert np.isfinite(values['tau'][:2]).all() and np.isnan(values['tau'][2])
    assert win.outside_limits() == {2}
    key = 'height'
    low = int(np.argmin(values[key][:2]))
    ov.lines[key][0].setValue(values[key][:2].mean())
    ov._on_line()
    assert win.outside_limits() == {low, 2}
    win.select_tone(1 - low)
    assert win.tone_list.count() == 3
    # The boxes above the list: only the flagged tones, without the
    # tones marked bad
    win.only_outside.setChecked(True)
    assert win.tone_list.count() == 3 and win.current() == 1 - low
    win.select_tone(low)
    win._fill_list()
    assert win.tone_list.count() == 2
    win._ask_reason = lambda: 'noisy'
    win.toggle_bad()
    win.include_bad.setChecked(False)
    assert win.tone_list.count() == 2 and win.current() == low
    win.select_tone(2)
    win._fill_list()
    assert win.tone_list.count() == 1
    win.include_bad.setChecked(True)
    assert win.tone_list.count() == 2
    win.only_outside.setChecked(False)
    assert win.tone_list.count() == 3
    ov.close()
    win.close()


def test_window_tone_marks(qapp, rays):
    """
    With a store root, marking bad and adding a note write the store's
    tone marks.
    """
    win, _ = make_window(rays)
    root = zarr.group()
    win.root, win.stage = root, 'hee/x'
    win._ask_reason = lambda: 'weird pulse shape'
    win._ask_note = lambda: 'note'
    win.tone_list.setCurrentRow(1)
    win.toggle_bad()
    win.add_note()
    assert set(dl.bad_tones(root, 'hee/x')) == {11}
    assert dl.bad_tones(root) == {}    # local by default here
    assert [e['kind'] for e in dl.tone_marks(root, 11)] == ['bad', 'note']
    assert dl.tone_marks(root, 11)[0]['stage'] == 'hee/x'
    win.toggle_bad()
    assert dl.bad_tones(root, 'hee/x') == {}
    win.close()
