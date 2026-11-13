"""
Tests for the VNA sweep view window (``citkid.vna.view``), on a store
made by the simulated board.
"""
import asyncio
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from citkid import daq_layout as dl
from citkid.crs.sim import simulate_vna_store
from citkid.vna.view import VnaViewWindow, load_vna_view, run_vna_view

SMALL = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1, n_per_bank = 6,
             pair_frac = 0.)


@pytest.fixture(scope = 'module')
def store(tmp_path_factory):
    """
    Make a simulated store with one sweep of zone 1 (6 bands, 6 passes).

    Returns:
    path (pathlib.Path): the store.
    """
    path = tmp_path_factory.mktemp('view') / 'sim.zarr'
    asyncio.run(simulate_vna_store(path, device_kwargs = SMALL, step = 5e-5,
                                   verbose = False))
    return path


@pytest.fixture(scope = 'module')
def qapp():
    """
    Return the QApplication (offscreen).

    Returns:
    app (QApplication): the application.
    """
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_load_vna_view(store):
    f, z, bands, info = load_vna_view(store)
    f0, z0 = dl.load_vna_sweep(store, 0)
    np.testing.assert_array_equal(f, f0)
    np.testing.assert_array_equal(z, z0)
    assert len(bands['chunk_edges']) == len(bands['chunk_passes']) == 6
    assert info.startswith('sim.zarr: vna0;') and f'{len(f):,} points' in info
    assert '-55 dBm' in info and '100 samples per point' in info
    assert 'df/f = 5e-05' in info and '6 bands in 6 passes' in info
    # The latest sweep by default, or by index; an open root works too
    root = dl.open_store(store)
    assert load_vna_view(root, 0)[3] == info
    with pytest.raises(KeyError):
        load_vna_view(store, 3)


def test_window(qapp, store):
    f, z, bands, info = load_vna_view(store)
    win = VnaViewWindow(f, z, bands = bands, info = info, title = 't')
    win.show()
    qapp.processEvents()
    assert win.windowTitle() == 't' and win.info.text() == info
    # |S21| in dB and the unwrapped phase against MHz, x axes linked
    x, y = win.mag_curve.getOriginalDataset()
    np.testing.assert_allclose(x, f / 1e6)
    np.testing.assert_allclose(y, 20 * np.log10(np.abs(z)))
    np.testing.assert_allclose(win.phase_curve.getOriginalDataset()[1],
                               np.unwrap(np.angle(z)))
    assert win.phase_plot.getViewBox().linkedView(0) is \
        win.mag_plot.getViewBox()
    assert len(win._band_lines) == 2 * (len(bands['chunk_edges']) - 1)
    # Wrapped phase on request
    win.unwrap.setChecked(False)
    np.testing.assert_allclose(win.phase_curve.getOriginalDataset()[1],
                               np.angle(z))
    # Zooming turns auto-range off; R fits the data again
    vb = win.mag_plot.getViewBox()
    vb.setXRange(1000., 1010., padding = 0)
    assert not vb.autoRangeEnabled()[0]
    lo, hi = vb.viewRange()[0]
    assert lo == pytest.approx(1000.) and hi == pytest.approx(1010.)
    # The phase plot followed
    assert win.phase_plot.getViewBox().viewRange()[0] == \
        pytest.approx([1000., 1010.])
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                      QtCore.Qt.Key_R, QtCore.Qt.NoModifier))
    assert all(vb.autoRangeEnabled())
    # The fit is applied with the next update (a paint, on screen)
    vb.updateAutoRange()
    lo, hi = vb.viewRange()[0]
    assert lo <= f[0] / 1e6 and hi >= f[-1] / 1e6
    # Other keys pass through
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                      QtCore.Qt.Key_A, QtCore.Qt.NoModifier))
    # New data replaces the curves and the band marks
    win.set_data(f[:100], z[:100])
    assert len(win.mag_curve.getOriginalDataset()[0]) == 100
    assert win._band_lines == []
    win.close()
    win.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None,
                                             QtCore.QEvent.DeferredDelete)


def test_run_vna_view(qapp, store):
    # Close the window as soon as the event loop runs
    QtCore.QTimer.singleShot(0, qapp.quit)
    assert run_vna_view(store) == 0
    QtCore.QTimer.singleShot(0, qapp.quit)
    assert run_vna_view(str(store), 0, title = 'x') == 0
    with pytest.raises(KeyError):
        run_vna_view(store, 5)
