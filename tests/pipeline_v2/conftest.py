"""Shared fixtures for pipeline_v2 tests.

Sets up offscreen Qt so that interactive-panel tests can run without a display.
Existing tests in this directory do not use Qt and are unaffected.
"""
import os

# Must be set before any Qt/pyqtgraph import occurs.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
import pyqtgraph as pg


@pytest.fixture(scope='session')
def qt_app():
    """Session-scoped QApplication for tests that instantiate Qt widgets."""
    return pg.mkQApp('citkid-pipeline_v2-tests')


@pytest.fixture(autouse=True)
def _delete_qt_windows_on_main_thread():
    """
    Delete the test's leftover windows on the main thread after each test.

    Closed windows are otherwise freed whenever Python's cyclic garbage
    collector next runs, which can be in a background thread (e.g. a
    sweep-fitter worker). Deleting Qt widgets off the GUI thread can abort
    the process. Only main windows are deleted: pyqtgraph keeps parentless
    context menus that it deletes itself.
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
