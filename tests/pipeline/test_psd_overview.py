"""
Tests of the PSD overview window (citkid.pipeline.interactive.psd_overview).
"""
import numpy as np
import pytest
import zarr
from citkid import daq_layout as dl
from citkid.qt_compat import get_qapp


@pytest.fixture
def qapp():
    """
    Return the QApplication.

    Returns:
    app (QApplication): the application.
    """
    return get_qapp()


def make_result(n = 8, cn = True):
    """
    Make a synthetic timestream result: white spectra of different
    levels, one tone left out as bad, one without spectra.

    Parameters:
    n (int): tones.
    cn (bool): with common-mode removal.

    Returns:
    ts (dict): the timestream.
    result (dict): as ``psds.analyze_timestream``.
    f_reduced (np.array): the group's reduced frequencies.
    """
    f_reduced = np.array([1., 3., 10., 30., 100.])
    f_binned = np.logspace(0, 3, 40)
    levels = 10. ** (-16 + 0.2 * np.arange(n))
    sxx = np.outer(levels, np.ones(len(f_reduced)))
    sxx[:, 4] = np.nan              # not covered by this timestream
    sxx[2] = np.nan                 # no spectra
    sxx[5] = np.nan                 # marked bad by the analysis
    binned = np.outer(levels, np.ones(len(f_binned)))
    status = np.array([''] * n, dtype = object)
    status[2] = 'the calibrated timestream is not finite'
    status[5] = 'marked bad: double resonance (tune0/fq_sweep)'
    result = dict(valid_reduced = np.array([True] * 4 + [False]),
                  sxx_reduced = sxx, f_binned = f_binned,
                  sxx_binned = binned, status = status)
    if cn:
        result['sxx_reduced_cn'] = sxx / 2
        result['sxx_binned_cn'] = binned / 2
    ts = dict(rows = np.arange(n), res_idxs = 100 + np.arange(n),
              run = 'ts0', name = 'dark', tone_set = 0)
    return ts, result, f_reduced


def test_window(qapp):
    """
    The window histograms Sxx (1/Hz, log axis) at three frequencies of
    the grid
    (the covered ones), after common-mode removal when there is one;
    limits at the data's edges list only the tones without a value;
    dragging a limit lists the tones outside; marking bad asks a
    reason, records it, and clears again.
    """
    from citkid.pipeline.interactive.psd_overview import (
        PsdOverviewWindow, valid_frequencies
    )
    ts, result, f_reduced = make_result()
    assert valid_frequencies(result, f_reduced).tolist() == [0, 1, 2, 3]
    root = zarr.group()
    win = PsdOverviewWindow(ts, result, f_reduced, root = root,
                            stage = 'ts_psd0/timestream0')
    assert [c.currentText() for c in win.combos] == ['1 Hz', '10 Hz',
                                                     '30 Hz']
    assert win.cn_box.isChecked() and win.has_cn
    assert set(win.limits) == {'1 Hz cn', '10 Hz cn', '30 Hz cn'}
    v = win.values(0)
    assert np.isnan(v[2]) and np.isnan(v[5]) and np.isfinite(v[0])
    assert v[0] == pytest.approx(1e-16 / 2)
    # Plain log-axis labels, and the histogram's x axis is logarithmic
    axis = win.plots[0].getAxis('bottom')
    assert axis.logTickStrings([-16., -15.], 1, 1) == ['1e-16', '1e-15']
    # Only the decades get ticks
    assert axis.logTickValues(-16.4, -14.6, 300, []) == [(1.0, [-16., -15.])]
    assert axis.logTickValues(-15.8, -15.2, 300, []) == [(1.0, [-16., -15.])]
    assert win.spec_plot.getAxis('bottom').logTickStrings([0., 1., 2.],
                                                          1, 1) == \
        ['1', '10', '100']
    assert win.plots[0].ctrl.logXCheck.isChecked()
    # The analysis's bad tone is bad here too; the tones without a value
    # are outside; the histograms do not zoom
    assert win.bad == {105: 'double resonance (tune0/fq_sweep)'}
    assert win.outside_limits() == {2, 5}
    assert win.tone_list.count() == 2
    assert win.plots[0].vb.state['mouseEnabled'] == [False, False]
    assert len(win.curves[0].getData()[1]) > 0
    # Raising the low limit of the first histogram lists the tones below
    lo, hi = np.log10(win.limits['1 Hz cn'])
    win.lines[0][0].setValue(lo + 0.5 * (hi - lo))
    win._on_line()
    assert win.limits['1 Hz cn'][0] == pytest.approx(10 ** (lo + 0.5
                                                            * (hi - lo)))
    outside = win.outside_limits()
    assert {0, 1, 2, 5} <= outside and 7 not in outside
    assert win.tone_list.count() == len(outside)
    assert win.tone_list.item(0).text().startswith('* res 100')
    # Before common-mode removal the values double and the limits are
    # kept apart
    win.cn_box.setChecked(False)
    assert win.values(0)[0] == pytest.approx(1e-16)
    assert '1 Hz' in win.limits and win.limits['1 Hz'][0] == pytest.approx(
        1e-16)
    win.cn_box.setChecked(True)
    # Selecting a tone shows its spectrum; every tone can be listed
    win.only_outside.setChecked(False)
    assert win.tone_list.count() == 8
    win.tone_list.setCurrentRow(6)
    # D moves to the next tone, from the list too
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    win.tone_list.setFocus()
    QtWidgets.QApplication.sendEvent(win.tone_list, QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_D, QtCore.Qt.NoModifier))
    assert win.current() == 7
    assert len(win.spec_curve.getData()[0]) == 40
    assert len(win.spec_cn_curve.getData()[0]) == 40
    assert 'res 107: analyzed' in win.status.text()
    # Mark bad with a reason: the result's dict and the store's marks
    win._ask_reason = lambda: 'high noise'
    win.toggle_bad()
    assert win.bad[107] == 'high noise'
    assert set(dl.bad_tones(root)) == {107}
    assert dl.tone_marks(root, 107)[0]['stage'] == 'ts_psd0/timestream0'
    assert win.status.text().startswith('BAD: high noise')
    assert win.bad_btn.text().startswith('Clear')
    win._ask_note = lambda: 'checked'
    win.add_note()
    assert [e['kind'] for e in dl.tone_marks(root, 107)] == ['bad', 'note']
    win.toggle_bad()
    assert 107 not in win.bad and dl.bad_tones(root) == {}
    # A cancelled reason marks nothing
    win._ask_reason = lambda: None
    win.toggle_bad()
    assert 107 not in win.bad
    win.close()


def test_window_without_cn(qapp):
    from citkid.pipeline.interactive.psd_overview import PsdOverviewWindow
    ts, result, f_reduced = make_result(cn = False)
    win = PsdOverviewWindow(ts, result, f_reduced)
    assert not win.has_cn and not win.cn_box.isEnabled()
    assert set(win.limits) == {'1 Hz', '10 Hz', '30 Hz'}
    win.only_outside.setChecked(False)
    win.tone_list.setCurrentRow(0)
    assert len(win.spec_cn_curve.getData()[0] or []) == 0
    # No registry: marking bad still works in the result
    win._ask_reason = lambda: 'bad fit'
    win.toggle_bad()
    assert win.bad[100] == 'bad fit'
    win.close()
