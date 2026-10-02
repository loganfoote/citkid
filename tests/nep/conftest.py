"""Shared fixtures for nep tests (offscreen Qt for the interactive window)."""
import os

# Must be set before any Qt/pyqtgraph import occurs.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
import pyqtgraph as pg


@pytest.fixture(scope='session')
def qt_app():
    """Session-scoped QApplication for tests that instantiate Qt widgets."""
    return pg.mkQApp('citkid-nep-tests')


@pytest.fixture(autouse=True)
def _delete_qt_windows_on_main_thread():
    """
    Delete the test's leftover main windows on the main thread after each test.
    """
    yield
    import gc
    from pyqtgraph.Qt import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        if isinstance(widget, QtWidgets.QMainWindow):
            widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    gc.collect()
