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
