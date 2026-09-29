"""
Tests for citkid.qt_compat.

The Qt constant resolution must work regardless of how pyqtgraph wraps its
binding.  We use lightweight stub classes (not MagicMock) to precisely control
which attribute names exist, avoiding MagicMock's auto-attribute generation.
"""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _flat_qtcore():
    """QtCore where all constants live directly on Qt (PyQt5-style)."""
    class _Qt:
        ShiftModifier = 0x02000000
        LeftButton    = 0x00000001
        DotLine       = 3
        Horizontal    = 1

    class _QtCore:
        Qt = _Qt

    return _QtCore


def _nested_qtcore():
    """QtCore where constants are only inside nested enum classes (PyQt6-style)."""
    class _KeyboardModifier:
        ShiftModifier = 0x02000000

    class _MouseButton:
        LeftButton = 0x00000001

    class _PenStyle:
        DotLine = 3

    class _Orientation:
        Horizontal = 1

    class _Qt:
        # No flat-namespace constants
        KeyboardModifier = _KeyboardModifier
        MouseButton      = _MouseButton
        PenStyle         = _PenStyle
        Orientation      = _Orientation

    class _QtCore:
        Qt = _Qt

    return _QtCore


def _empty_qtcore():
    """QtCore where Qt has no relevant constants at all."""
    class _Qt:
        pass

    class _QtCore:
        Qt = _Qt

    return _QtCore


# ---------------------------------------------------------------------------
# Tests: flat namespace (PyQt5-style)
# ---------------------------------------------------------------------------

class TestQtCompatFlatNamespace:
    """Constants available directly on QtCore.Qt."""

    def test_keyboard_modifier(self):
        with patch('citkid.qt_compat._QtCore', _flat_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.ShiftModifier == 0x02000000

    def test_mouse_button(self):
        with patch('citkid.qt_compat._QtCore', _flat_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.LeftButton == 0x00000001

    def test_pen_style(self):
        with patch('citkid.qt_compat._QtCore', _flat_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.DotLine == 3

    def test_orientation(self):
        with patch('citkid.qt_compat._QtCore', _flat_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.Horizontal == 1


# ---------------------------------------------------------------------------
# Tests: nested enum namespace (PyQt6 / PySide6-style)
# ---------------------------------------------------------------------------

class TestQtCompatNestedNamespace:
    """Constants only in nested enum groups."""

    def test_keyboard_modifier_nested(self):
        with patch('citkid.qt_compat._QtCore', _nested_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.ShiftModifier == 0x02000000

    def test_mouse_button_nested(self):
        with patch('citkid.qt_compat._QtCore', _nested_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.LeftButton == 0x00000001

    def test_pen_style_nested(self):
        with patch('citkid.qt_compat._QtCore', _nested_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.DotLine == 3

    def test_orientation_nested(self):
        with patch('citkid.qt_compat._QtCore', _nested_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert Qt.Horizontal == 1


# ---------------------------------------------------------------------------
# Tests: missing / error cases
# ---------------------------------------------------------------------------

class TestQtCompatMissing:
    """AttributeError raised for names that don't exist anywhere."""

    def test_unknown_name_raises(self):
        with patch('citkid.qt_compat._QtCore', _empty_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            with pytest.raises(AttributeError,
                               match="Qt has no member 'NonExistentConstant'"):
                _ = Qt.NonExistentConstant

    def test_repr(self):
        with patch('citkid.qt_compat._QtCore', _flat_qtcore()):
            from citkid import qt_compat
            Qt = qt_compat._QtCompat()
            assert repr(Qt) == "<citkid.qt_compat.Qt>"



# ---------------------------------------------------------------------------
# Tests: application and window sizing helpers
# ---------------------------------------------------------------------------

class _StubScreen:
    """Screen stub with a fixed logical DPI and device pixel ratio."""

    def __init__(self, dpi, dpr):
        self._dpi, self._dpr = dpi, dpr

    def logicalDotsPerInch(self):
        return self._dpi

    def devicePixelRatio(self):
        return self._dpr


class _StubApp:
    """Application stub exposing a list of stub screens."""

    def __init__(self, *screens):
        self._screens = list(screens)

    def screens(self):
        return self._screens


@pytest.fixture
def qapp():
    """Return a QApplication (offscreen, from tests/conftest.py)."""
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize('screens, expected', [
    ([_StubScreen(96, 1.0)], []),                          # 100% display
    ([_StubScreen(96, 2.5)], []),                          # scaled by Qt
    ([_StubScreen(240, 1.0)], [240]),                      # 250%, unscaled
    ([_StubScreen(96, 1.0), _StubScreen(144, 1.0)], [144]),
])
def test_unscaled_hidpi_screens(screens, expected):
    from citkid import qt_compat
    assert qt_compat._unscaled_hidpi_screens(_StubApp(*screens)) == expected


def test_get_qapp_creates_app_with_mkqapp():
    from citkid import qt_compat
    sentinel = object()
    with patch.object(qt_compat._QtWidgets.QApplication, 'instance',
                      return_value=None), \
         patch.object(qt_compat._pg, 'mkQApp', return_value=sentinel) as mk:
        assert qt_compat.get_qapp('My GUI') is sentinel
    mk.assert_called_once_with('My GUI')


@pytest.mark.parametrize('screen, warns', [
    (_StubScreen(240, 1.0), True),
    (_StubScreen(96, 2.5), False),
])
def test_get_qapp_warns_on_existing_unscaled_app(screen, warns):
    import warnings
    from citkid import qt_compat
    app = _StubApp(screen)
    with patch.object(qt_compat._QtWidgets.QApplication, 'instance',
                      return_value=app), \
         warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert qt_compat.get_qapp() is app
    assert any('high-DPI' in str(w.message) for w in caught) == warns


@pytest.mark.parametrize('size, expected', [
    (None, (800, 640)),           # frac of the screen
    ((600, 400), (600, 400)),     # preferred size fits
    ((1500, 850), (800, 640)),    # preferred size clamped to the screen
])
def test_fit_window_to_screen(qapp, size, expected):
    """Size the window within frac of the screen and centre it there."""
    from pyqtgraph.Qt import QtCore, QtWidgets
    from citkid import qt_compat
    win = QtWidgets.QWidget()
    with patch.object(qt_compat, 'available_screen_geometry',
                      return_value=QtCore.QRect(0, 0, 1000, 800)):
        result = qt_compat.fit_window_to_screen(win, frac=0.8, size=size)
    assert result == expected
    assert (win.width(), win.height()) == expected
    assert win.x() == (1000 - expected[0]) // 2
    assert win.y() == (800 - expected[1]) // 2
    win.close()
