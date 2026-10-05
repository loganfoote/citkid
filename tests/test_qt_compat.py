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
         patch.object(qt_compat._pg, 'mkQApp', return_value=sentinel) as mk, \
         patch.object(qt_compat, '_use_system_ui_font') as use_font:
        assert qt_compat.get_qapp('My GUI') is sentinel
    mk.assert_called_once_with('My GUI')
    use_font.assert_called_once_with(sentinel)


def test_delete_on_close_destroys_window_only_when_closed(qapp):
    """A closed window is destroyed by Qt; a cancelled close keeps it."""
    from pyqtgraph.Qt import QtCore, QtWidgets
    from citkid import qt_compat

    class _Refusing(QtWidgets.QMainWindow):
        def closeEvent(self, event):
            event.ignore()

    def flush_deletes():
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    win = QtWidgets.QMainWindow()
    win.result = [1, 2, 3]
    qt_compat.delete_on_close(win)
    win.show()
    win.close()
    flush_deletes()
    assert win.result == [1, 2, 3]          # plain attributes stay usable
    with pytest.raises(RuntimeError):
        win.isVisible()                     # the Qt object is gone

    refusing = _Refusing()
    qt_compat.delete_on_close(refusing)
    refusing.show()
    refusing.close()
    flush_deletes()
    assert refusing.isVisible()             # close was cancelled: not deleted
    refusing.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
    refusing.hide()


def test_use_system_ui_font_sets_app_font(qapp):
    """A known system UI font replaces the app default font."""
    from citkid import qt_compat
    original = qapp.font()
    try:
        with patch.object(qt_compat, '_windows_message_font', return_value=('Arial', 9.0)):
            qt_compat._use_system_ui_font(qapp)
        assert qapp.font().family() == 'Arial'
        assert qapp.font().pointSizeF() == 9.0

        with patch.object(qt_compat, '_windows_message_font', return_value=(None, None)):
            qt_compat._use_system_ui_font(qapp)
        assert qapp.font().family() == 'Arial'   # unknown: left unchanged
    finally:
        qapp.setFont(original)


def test_windows_message_font_is_plausible():
    """On Windows the message font is a real family at a normal UI size."""
    import sys
    from citkid import qt_compat
    family, point_size = qt_compat._windows_message_font()
    if sys.platform != 'win32':
        assert (family, point_size) == (None, None)
        return
    assert family
    assert 6 <= point_size <= 16


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


@pytest.mark.parametrize('size, expected', [
    ((600, 700), (600, 700)),     # tall content fits within the full height
    ((600, 900), (600, 760)),     # capped at screen height minus the margin
])
def test_fit_window_to_screen_height_margin(qapp, size, expected):
    """With height_margin, the height may use the whole screen minus it."""
    from pyqtgraph.Qt import QtCore, QtWidgets
    from citkid import qt_compat
    win = QtWidgets.QWidget()
    with patch.object(qt_compat, 'available_screen_geometry',
                      return_value=QtCore.QRect(0, 0, 1000, 800)):
        result = qt_compat.fit_window_to_screen(win, frac=0.8, size=size, height_margin=40)
    assert result == expected
    # The frame (title bar included) is centred inside the screen.
    assert win.y() == (800 - expected[1] - 40) // 2
    win.close()


def test_scroll_area_content_height(qapp):
    """The needed height is the content's minimum height plus the frame."""
    from pyqtgraph.Qt import QtWidgets
    from citkid import qt_compat
    scroll = QtWidgets.QScrollArea()
    assert qt_compat.scroll_area_content_height(scroll) == 0

    content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(content)
    for height in (150, 250):
        child = QtWidgets.QWidget()
        child.setMinimumHeight(height)
        layout.addWidget(child)
    scroll.setWidget(content)

    expected = content.minimumSizeHint().height() + 2 * scroll.frameWidth()
    assert qt_compat.scroll_area_content_height(scroll) == expected
    assert expected >= 400
    scroll.close()


# ---------------------------------------------------------------------------
# run_responsive
# ---------------------------------------------------------------------------

def test_run_responsive_keeps_processing_events(qapp):
    """Timers (repaints, OS pings) keep running while the work runs off-thread."""
    import threading
    import time
    from citkid import qt_compat
    from pyqtgraph.Qt import QtCore

    ticks, threads = [], []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(qt_compat.responsive_busy()))
    timer.start(10)

    def work(a, b=0):
        threads.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.3)
        return a + b

    try:
        assert qt_compat.run_responsive(work, 1, b=2) == 3
    finally:
        timer.stop()
    assert threads == [False]                 # ran off the GUI thread
    assert len(ticks) >= 5 and all(ticks)     # events processed meanwhile
    assert not qt_compat.responsive_busy()


def test_run_responsive_defers_user_input(qapp, monkeypatch):
    """
    Events are processed with ExcludeUserInputEvents, so real mouse and
    keyboard input waits until the work is done (no re-entrant clicks).
    Posted test events aren't spontaneous input, so the flag is checked.
    """
    import time
    from citkid import qt_compat
    from pyqtgraph.Qt import QtCore

    flags = []
    original = qapp.processEvents
    monkeypatch.setattr(qapp, 'processEvents', lambda *a: (flags.append(a), original(*a))[1])

    qt_compat.run_responsive(time.sleep, 0.1)

    expected = QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
    assert flags and all(a == (expected,) for a in flags)


def test_run_responsive_reraises_and_runs_directly_off_main_thread(qapp):
    import threading
    from citkid import qt_compat

    with pytest.raises(ValueError, match='boom'):
        qt_compat.run_responsive(lambda: (_ for _ in ()).throw(ValueError('boom')))
    assert not qt_compat.responsive_busy()

    seen = []
    worker = threading.Thread(target=lambda: seen.append(
        qt_compat.run_responsive(lambda: threading.current_thread().name)))
    worker.start()
    worker.join()
    assert seen == [worker.name]              # nested in a thread: no new thread


# ---------------------------------------------------------------------------
# Layout sizing helpers
# ---------------------------------------------------------------------------

def test_scroll_area_min_width_adds_scrollbar_and_frame(qapp):
    """
    Check the width is the content's minimum plus scrollbar and frame, and
    0 without content.
    """
    from pyqtgraph.Qt import QtWidgets
    from citkid.qt_compat import scroll_area_min_width

    scroll = QtWidgets.QScrollArea()
    assert scroll_area_min_width(scroll) == 0
    content = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(content).addWidget(QtWidgets.QLabel('x' * 40))
    scroll.setWidget(content)
    extra = (scroll.verticalScrollBar().sizeHint().width()
             + 2 * scroll.frameWidth())
    assert scroll_area_min_width(scroll) == (
        content.minimumSizeHint().width() + extra)


def test_vbox_height_for_width_wraps_text_and_uses_overrides(qapp):
    """
    Check that wrapped labels get taller at narrow widths and overrides
    replace measured heights.
    """
    from pyqtgraph.Qt import QtWidgets
    from citkid.qt_compat import vbox_height_for_width

    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    label = QtWidgets.QLabel(' '.join(['word'] * 60))
    label.setWordWrap(True)
    fixed = QtWidgets.QWidget()
    fixed.setFixedHeight(40)
    layout.addWidget(label)
    layout.addWidget(fixed)

    narrow = vbox_height_for_width(layout, 150)
    wide = vbox_height_for_width(layout, 2000)
    assert narrow > wide
    overridden = vbox_height_for_width(layout, 2000, overrides={fixed: 140})
    assert overridden == wide + 100
