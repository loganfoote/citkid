"""
Tests of the NEP review window on a sweep without a store.
"""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
from .test_sweep import synthetic_sweep


def make_window(qt_app, **kwargs):
    """
    Build the review window on a synthetic sweep of three tones.

    Parameters:
    qt_app (QApplication): the application.
    **kwargs: ``NEPReviewWindow`` options.

    Returns:
    win (NEPReviewWindow): the window.
    sweep (PowerSweep): its sweep.
    """
    from citkid.nep.review import NEPReviewWindow
    sweep = synthetic_sweep()
    sweep.compute_all()
    win = NEPReviewWindow(sweep, **kwargs)
    win.show()
    qt_app.processEvents()
    return win, sweep


def test_review_window_lists_fits_and_pmin(qt_app):
    """
    The window lists every tone with its eta and NEPinc, shows the
    selected tone's fits, and p_min (box or dragged line) refits the
    tone, Apply to all every tone; a tone marked bad is listed so and
    the overview flags the tones outside the limits.
    """
    win, sweep = make_window(qt_app)
    assert win.tone_list.count() == 3
    assert win.tone_list.item(0).text().startswith('  res 0: eta 0.6')
    assert 'eta 0.600' in win.status.text() and 'R0 -' in win.status.text()
    assert len(win.x_points.getData()[0]) == 6
    assert len(win.nep_in.getData()[0]) == 6 and len(win.x_fit.getData()[0])
    # p_min from the box (in units of the highest power): fewer points
    pmax = sweep.power.max()
    win.pmin_box.setValue(0.5)
    assert sweep.fields['p_min'][0] == pytest.approx(0.5 * pmax)
    assert sweep.fields['n_fit'][0] == 3 and len(win.nep_out.getData()[0]) == 3
    assert 0 in win.reviewed
    # Dragging the line (log10 of the power)
    win.pmin_line.setValue(np.log10(0.3 * pmax))
    win.pmin_line.sigPositionChangeFinished.emit(win.pmin_line)
    assert sweep.fields['p_min'][0] == pytest.approx(0.3 * pmax, rel = 1e-6)
    assert sweep.fields['n_fit'][0] == 4
    # Apply to all
    win.apply_pmin_all()
    np.testing.assert_allclose(sweep.fields['p_min'], 0.3 * pmax, rtol = 1e-6)
    assert (sweep.fields['n_fit'] == 4).all()
    # Next tone, mark it bad with a stand-in reason dialog
    win.step_tone(1)
    assert win.current() == 1 and '*' in win.tone_list.item(0).text()
    from citkid.pipeline.interactive import marks
    win_ask = lambda *a, **k: ('noisy', 'local')
    old = marks.ask_reason
    marks.ask_reason = win_ask
    try:
        win.toggle_bad()
    finally:
        marks.ask_reason = old
    assert sweep.bad == {1: 'noisy'}
    assert win.tone_list.item(1).text().startswith('x res 1: bad (noisy)')
    assert 'MARKED BAD' in win.status.text()
    win.toggle_bad()
    assert sweep.bad == {}
    # The overview: limits flag tones, the list can show only those
    win.open_overview()
    assert win.overview.isVisible()
    sweep.limits['eta'] = [0.7, 1.]
    win.overview.refresh()
    assert win.overview.outside_list.count() == 3
    win.set_only_outside(True)
    assert win.tone_list.count() == 3
    sweep.limits['eta'] = [0.5, 0.7]
    win.set_only_outside(True)
    assert win.tone_list.count() == 1       # the selected tone stays listed
    win.close()
    assert not win.overview.isVisible()


def test_review_window_statuses_and_calibration_without_store(qt_app):
    """
    Tones without a fit show their status; the calibration button needs
    a calibrated row, and the final-pass message shows at first.
    """
    win, sweep = make_window(qt_app, message = 'Final pass done')
    assert win.status.text() == 'Final pass done'
    sweep.fr[2, 2:] = np.nan
    sweep.compute_all()
    win._fill_list()
    win.select_tone(2)
    assert 'points with an IQ fit' in win.tone_list.item(2).text()
    assert 'points with an IQ fit' in win.status.text()
    assert len(win.x_fit.getData()[0] or []) == 0
    sweep.rows[2, 0] = -1
    win.point_box.setCurrentIndex(0)
    win.open_calibration()
    assert 'not calibrated at p0' in win.status.text()
    assert win._cal_window is None
    win.close()


def test_calibration_window_recomputes_the_tone(tmp_path, qt_app):
    """
    Calibration... opens the calibration and IQ fit panels of the tone at
    the chosen point on the sweep's own DataSet; when that window closes
    the tone is recomputed there and the point is noted for the final
    pass; Done then runs the final pass through ``run_nep_review``.
    """
    from unittest.mock import patch
    from citkid import daq_layout as dl
    from citkid.nep import review as rv
    from citkid.nep import sweep as sw
    from citkid.pipeline.interactive.core import InteractiveAnalysisWindow
    from .test_sweep import power_store
    from ..procedures.test_scripts import load_script
    path = power_store(tmp_path)
    m = load_script('09_nep.py')
    m.NAME, m.CN_N_COMP = 'bb', 1
    m.POINTS = {f'P{j}': 1e-15 * (j + 1) for j in range(4)}
    m.FREQ_HZ, m.FMAX_DIV, m.MODES_RATE_HZ = 30., 1., 100.
    sweep = m.main(m.parse_args(['--store', str(path), '--no-gui']))
    win = rv.NEPReviewWindow(sweep)
    win.show()
    qt_app.processEvents()
    assert win.tone_list.count() == 5 and win.point_box.count() == 4
    win.tone_list.setCurrentRow(1)
    win.point_box.setCurrentIndex(2)
    win.open_calibration()
    cal = win._cal_window
    assert isinstance(cal, InteractiveAnalysisWindow)
    assert cal._data_idxs == [int(sweep.rows[1, 2])]
    assert [p.step_names[0] for p in cal.panels][-1] == 'fit_iq'
    for _ in range(30):
        qt_app.processEvents()
    sxx0 = sweep.sxx[1, 2]
    with patch.object(QtWidgets.QMessageBox, 'question',
                      return_value = QtWidgets.QMessageBox.Yes):
        cal.close()
    for _ in range(30):
        qt_app.processEvents()
    win._check_calibration(1, 2)       # what the timer does
    assert win._cal_window is None and sweep.changed == {2}
    assert np.isfinite(sweep.sxx[1, 2])
    assert sweep.sxx[1, 2] == pytest.approx(sxx0, rel = 0.5)
    assert 'recomputed' in win.status.text()
    win.close()
    # run_nep_review: the window, the final pass, the window again
    opened = []

    class FakeWindow(rv.NEPReviewWindow):
        """
        A review window that closes itself at once.
        """

        def show(self):
            """
            Record the opening and close.
            """
            opened.append(self._message)
            from pyqtgraph.Qt import QtCore
            QtCore.QTimer.singleShot(
                0, QtWidgets.QApplication.instance().quit)

    with patch.object(rv, 'NEPReviewWindow', FakeWindow), \
         patch.object(rv, 'fit_window_to_screen', lambda w: None):
        passes = rv.run_nep_review(sweep, verbose = False)
    assert passes == 1 and sweep.changed == set()
    assert opened[0] == '' and opened[1].startswith('Final pass done on 1')
    root = dl.open_store(path)
    assert sw.PowerSweep.load(root, root['nep/bb']).sxx.shape == (5, 4)
