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
    assert set(win.limits) == {'1 Hz cn', '10 Hz cn', '30 Hz cn',
                               'sfactor 10-100 Hz'}
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
    # Mark bad with a reason: the result's dict and the store's marks;
    # the tone leaves the histograms and is flagged in the list
    n_hist = win.curves[0].getData()[1].sum()
    win._ask_reason = lambda: 'high noise'
    win.toggle_bad()
    assert win.bad[107] == 'high noise'
    assert win.curves[0].getData()[1].sum() == n_hist - 1
    assert 7 in win.outside_limits()
    assert set(dl.bad_tones(root, 'ts_psd0/timestream0')) == {107}
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
    assert set(win.limits) == {'1 Hz', '10 Hz', '30 Hz',
                               'sfactor 10-100 Hz'}
    win.only_outside.setChecked(False)
    win.tone_list.setCurrentRow(0)
    assert len(win.spec_cn_curve.getData()[0] or []) == 0
    # No registry: marking bad still works in the result
    win._ask_reason = lambda: 'bad fit'
    win.toggle_bad()
    assert win.bad[100] == 'bad fit'
    win.close()


def test_ask_existing_psd(monkeypatch):
    """
    New and Load answer at once; Overwrite asks again; Cancel or closing
    a box gives None.
    """
    import citkid.pipeline.interactive.psd_overview as po
    answers = []
    monkeypatch.setattr(po, '_message_box',
                        lambda *a, **k: answers.pop(0))
    for answer, want in [('New', 'new'), ('Load', 'load'),
                         ('Cancel', None), (None, None)]:
        answers.append(answer)
        assert po.ask_existing_psd('ts_psd0', 'desc') == want
    answers += ['Overwrite', 'Cancel']
    assert po.ask_existing_psd('ts_psd0', 'desc') is None
    answers += ['Overwrite', 'Overwrite']
    assert po.ask_existing_psd('ts_psd0', 'desc') == 'overwrite'
    assert answers == []


def test_window_several_timestreams(qapp):
    """
    With other timestreams the histogram values are the geometric mean
    over the ones that have the tone and did not flag it, the s-factor
    histogram averages 10 to 100 Hz, the spectrum plot has a curve per
    timestream, and a bad mark goes to every timestream's stage.
    """
    from citkid.pipeline.interactive.psd_overview import (
        N_HIST, PsdOverviewWindow)
    ts, result, f_reduced = make_result()
    ts2, result2, _ = make_result()
    # The second timestream lacks the last tone, has every frequency,
    # has 10 x the noise and an s-factor; tone 5 is fine in it
    ts2['res_idxs'] = ts2['res_idxs'][:-1]
    ts2['rows'] = ts2['rows'][:-1]
    for key in ('sxx_reduced', 'sxx_binned', 'sxx_reduced_cn',
                'sxx_binned_cn'):
        result2[key] = result2[key][:-1] * 10.
    for key in ('sxx_reduced', 'sxx_reduced_cn'):
        result2[key][:, 4] = result2[key][:, 3]
        result2[key][5] = 1e-15
    result2['status'] = result2['status'][:-1]
    result2['status'][5] = ''
    result2['valid_reduced'] = np.array([True] * 5)
    result2['sfactor_reduced'] = np.outer(np.arange(7), np.ones(5)) + \
        np.array([0., 0., 1., 3., 100.])
    result2['sfactor_reduced'][2] = np.nan
    root = zarr.group()
    win = PsdOverviewWindow(ts, result, f_reduced, root = root,
                            stage = 'ts_psd0/timestream0', label = 'long',
                            others = [dict(ts = ts2, result = result2,
                                           stage = 'ts_psd0/timestream1',
                                           label = 'short')])
    assert win.res_idxs == list(100 + np.arange(8))
    assert [e['label'] for e in win.entries] == ['long', 'short']
    assert len(win.plots) == N_HIST + 1 and len(win.spec_curves) == 2
    # Tone 0 at 1 Hz (before the common modes): geometric mean of 1e-16
    # and 1e-15; the last tone has the first timestream only; tone 5 the
    # second only (flagged in the first); tone 2 none
    win.cn_box.setChecked(False)
    win.combos[0].setCurrentIndex(0)
    v = win.values(0)
    assert v[0] == pytest.approx(np.sqrt(1e-16 * 1e-15))
    assert v[7] == pytest.approx(10. ** (-16 + 0.2 * 7))
    assert v[5] == pytest.approx(1e-15) and np.isnan(v[2])
    # 100 Hz: only the second timestream has it
    win.combos[0].setCurrentIndex(list(win.f_idx).index(4))
    assert win.values(0)[0] == pytest.approx(
        result2['sxx_reduced'][0, 4])
    win.cn_box.setChecked(True)
    assert win.values(0)[0] == pytest.approx(
        result2['sxx_reduced_cn'][0, 4])
    # The s-factor: mean over 10, 30 and 100 Hz of the second (the
    # first has none): tone 0 -> (1 + 3 + 100) / 3
    sf = win.values(N_HIST)
    assert sf[0] == pytest.approx(104. / 3)
    assert np.isnan(sf[2]) and np.isnan(sf[7])
    assert win._key(N_HIST) == 'sfactor 10-100 Hz'
    assert 'sfactor 10-100 Hz' in win.limits
    # A bad mark goes to both stages
    win.only_outside.setChecked(False)
    win.tone_list.setCurrentRow(win._listed.index(0))
    win._ask_reason = lambda: 'high noise'
    win.toggle_bad()
    assert win.bad[100] == 'high noise'
    assert set(dl.bad_tones(root, 'ts_psd0/timestream0')) == {100}
    assert set(dl.bad_tones(root, 'ts_psd0/timestream1')) == {100}
    win.toggle_bad()
    assert 100 not in win.bad and dl.bad_tones(root, 'ts_psd0/timestream0') \
        == {}
    # The selected tone shows both curves and both statuses
    win.select(0)
    assert len(win.spec_curves[1].getData()[0]) == 40
    assert 'long: analyzed' in win.status.text()
    assert 'short: analyzed' in win.status.text()
    win.select(7)
    assert 'short: not in this timestream' in win.status.text()
    win.close()
