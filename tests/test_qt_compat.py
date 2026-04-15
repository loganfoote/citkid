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

