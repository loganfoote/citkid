"""
Compatibility shim for Qt enum constants across pyqtgraph backends.

pyqtgraph 0.13+ wraps whichever Qt binding it finds (PyQt5, PyQt6, PySide2,
PySide6).  In some combinations the traditional flat namespace
(``QtCore.Qt.ShiftModifier``, ``QtCore.Qt.LeftButton``, etc.) is not
available — constants live inside nested enum classes instead
(``QtCore.Qt.KeyboardModifier.ShiftModifier``).

This module exposes every constant used by citkid under a single ``Qt``
namespace that works regardless of binding, so call sites can write::

    from citkid.qt_compat import Qt
    if ev.modifiers() & Qt.ShiftModifier: ...
    pen = pg.mkPen(color, style=Qt.DotLine)

It also provides window helpers shared by all citkid GUIs:

* ``get_qapp`` creates the QApplication with high-DPI scaling enabled, so
  Qt handles display scaling (e.g. 250% on Windows) and sizes are in
  logical pixels.
* ``available_screen_geometry`` and ``fit_window_to_screen`` size and
  center windows so they never open larger than the screen.
* ``scroll_area_content_height`` and ``vbox_height_for_width`` give the
  height a scroll area and a vertical layout need to show their contents
  without scrolling, for sizing windows to their content.
"""

import warnings as _warnings

import pyqtgraph as _pg
from pyqtgraph.Qt import QtCore as _QtCore, QtGui as _QtGui, QtWidgets as _QtWidgets

# Enum groups to probe for members.  Order matters: the first group that
# contains the name wins.
_ENUM_GROUPS = (
    'KeyboardModifier',   # ShiftModifier, ControlModifier, NoModifier …
    'MouseButton',        # LeftButton, RightButton …
    'Key',                # Key_A … Key_Z, Key_Left, Key_Right …
    'Orientation',        # Horizontal, Vertical
    'PenStyle',           # SolidLine, DashLine, DotLine …
    'CursorShape',        # ArrowCursor, WaitCursor …
    'TextFormat',         # RichText, PlainText …
    'AlignmentFlag',      # AlignTop, AlignCenter …
    'Modifier',           # (PySide6 uses this instead of KeyboardModifier)
)


class _QtCompat:
    """
    Proxy that resolves attribute lookups against ``QtCore.Qt``, falling
    back to nested enum classes when the flat namespace is unavailable.
    """

    def __getattr__(self, name: str):
        Qt = _QtCore.Qt
        # Fast path: flat namespace (native PyQt5 without pyqtgraph wrapping)
        val = getattr(Qt, name, None)
        if val is not None:
            return val
        # Slow path: check each enum group
        for grp in _ENUM_GROUPS:
            g = getattr(Qt, grp, None)
            if g is not None:
                val = getattr(g, name, None)
                if val is not None:
                    return val
        raise AttributeError(
            f"Qt has no member '{name}' (checked flat namespace and "
            f"enum groups {_ENUM_GROUPS})"
        )

    def __repr__(self):
        return "<citkid.qt_compat.Qt>"


Qt = _QtCompat()


################################################################################
# Application and window sizing helpers
################################################################################

# Logical DPI above which a screen at devicePixelRatio 1 means Qt is not
# handling display scaling (fonts scale with DPI, but geometry does not).
_UNSCALED_DPI_THRESHOLD = 110


def _unscaled_hidpi_screens(app):
    """
    Find high-DPI screens whose scaling Qt is not handling.

    Parameters:
    app (QApplication): running application.

    Returns:
    dpis (list): logical DPI of each screen that has DPI above
        ``_UNSCALED_DPI_THRESHOLD`` but a device pixel ratio of 1. Empty if
        Qt handles scaling on every screen.
    """
    return [
        round(s.logicalDotsPerInch()) for s in app.screens()
        if s.logicalDotsPerInch() > _UNSCALED_DPI_THRESHOLD
        and s.devicePixelRatio() == 1.0
    ]


def _windows_message_font():
    """
    Return the current Windows message font (the standard UI font).

    Qt 5 builds its default font from the ``DEFAULT_GUI_FONT`` stock
    object, which Windows creates once per sign-in at the display scaling
    in effect then. After changing scaling or switching displays it can be
    far too large (e.g. 20 pt instead of 9 pt) until the user signs out. The
    message font from ``SystemParametersInfo`` is kept current.

    Returns:
    family, point_size (str, float or None): Font family and size in points,
        or None, None if not on Windows or the query fails.
    """
    import sys
    if sys.platform != 'win32':
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        class LOGFONTW(ctypes.Structure):
            _fields_ = [
                ('lfHeight', wintypes.LONG), ('lfWidth', wintypes.LONG),
                ('lfEscapement', wintypes.LONG), ('lfOrientation', wintypes.LONG),
                ('lfWeight', wintypes.LONG), ('lfItalic', wintypes.BYTE),
                ('lfUnderline', wintypes.BYTE), ('lfStrikeOut', wintypes.BYTE),
                ('lfCharSet', wintypes.BYTE), ('lfOutPrecision', wintypes.BYTE),
                ('lfClipPrecision', wintypes.BYTE), ('lfQuality', wintypes.BYTE),
                ('lfPitchAndFamily', wintypes.BYTE), ('lfFaceName', wintypes.WCHAR * 32),
            ]

        class NONCLIENTMETRICSW(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.UINT), ('iBorderWidth', ctypes.c_int),
                ('iScrollWidth', ctypes.c_int), ('iScrollHeight', ctypes.c_int),
                ('iCaptionWidth', ctypes.c_int), ('iCaptionHeight', ctypes.c_int),
                ('lfCaptionFont', LOGFONTW), ('iSmCaptionWidth', ctypes.c_int),
                ('iSmCaptionHeight', ctypes.c_int), ('lfSmCaptionFont', LOGFONTW),
                ('iMenuWidth', ctypes.c_int), ('iMenuHeight', ctypes.c_int),
                ('lfMenuFont', LOGFONTW), ('lfStatusFont', LOGFONTW),
                ('lfMessageFont', LOGFONTW), ('iPaddedBorderWidth', ctypes.c_int),
            ]

        user32 = ctypes.windll.user32
        metrics = NONCLIENTMETRICSW()
        metrics.cbSize = ctypes.sizeof(metrics)
        SPI_GETNONCLIENTMETRICS = 0x29
        if not user32.SystemParametersInfoW(
                SPI_GETNONCLIENTMETRICS, metrics.cbSize, ctypes.byref(metrics), 0):
            return None, None
        dpi = user32.GetDpiForSystem() if hasattr(user32, 'GetDpiForSystem') else 96
        font = metrics.lfMessageFont
        if not font.lfHeight or not dpi:
            return None, None
        return font.lfFaceName, abs(font.lfHeight) * 72.0 / dpi
    except Exception:
        return None, None


def _use_system_ui_font(app):
    """
    Set the application font to the current system UI font, if known.

    Parameters:
    app (QApplication): Newly created application.
    """
    family, point_size = _windows_message_font()
    if family is None:
        return
    font = _QtGui.QFont(family)
    font.setPointSizeF(point_size)
    app.setFont(font)


def get_qapp(name=None):
    """
    Return the QApplication, creating it with high-DPI scaling if needed.

    Only one QApplication exists per process, and high-DPI scaling can only
    be enabled before it is created. A new application is created with
    ``pyqtgraph.mkQApp``, which enables it. Its default font is then set to
    the current system UI font, since Qt 5 on Windows can pick up a stale,
    much larger one (see ``_windows_message_font``). If an application
    already exists without high-DPI scaling (for example, created by other
    Qt code earlier in the same Jupyter kernel), a warning is issued, since
    windows and fonts may then appear too large.

    Parameters:
    name (str or None): application name for a new application. None
        (default) uses pyqtgraph's default.

    Returns:
    app (QApplication): the running application.
    """
    app = _QtWidgets.QApplication.instance()
    if app is None:
        app = _pg.mkQApp(name)
        _use_system_ui_font(app)
        return app
    dpis = _unscaled_hidpi_screens(app)
    if dpis:
        _warnings.warn(
            f"A Qt application already exists without high-DPI scaling "
            f"(screen at {dpis[0]} DPI), so windows and fonts may be too "
            f"large. Restart the Python kernel and open a citkid GUI (or run "
            f"pyqtgraph.mkQApp()) before any other Qt code.",
            RuntimeWarning,
            stacklevel=2,
        )
    return app


def available_screen_geometry():
    """
    Get the available geometry of the screen under the mouse cursor.

    Falls back to the primary screen if no screen is under the cursor.

    Returns:
    geom (QRect): available screen area (excluding taskbars) in logical
        pixels.
    """
    app = _QtWidgets.QApplication.instance() or get_qapp()
    screen = app.screenAt(_QtGui.QCursor.pos()) if hasattr(app, 'screenAt') else None
    if screen is None:
        screen = app.primaryScreen()
    return screen.availableGeometry()


# Logical pixels reserved for a window's title bar and frame when a window
# may use the full screen height (Windows 11 title bars are ~31 px).
TITLE_BAR_MARGIN = 40


def fit_window_to_screen(win, frac=0.8, size=None, height_margin=None):
    """
    Resize a window to fit on the screen and center it there.

    The window goes on the screen under the mouse cursor. Its size is
    clamped to ``frac`` of the available screen area, so it never opens
    larger than the screen unless its layout's minimum size requires it.

    Parameters:
    win (QWidget): top-level window.
    frac (float): maximum fraction of the available screen width (and
        height, unless ``height_margin`` is given) to use. Default 0.8.
    size (tuple or None): preferred (width, height) in logical pixels,
        clamped to the maximum size. None (default) uses the maximum size.
    height_margin (int or None): if given, the height may use the full
        available screen height minus this many logical pixels, instead of
        ``frac`` of it. The margin leaves room for the title bar, which is
        not part of the window's height. Use this for windows sized to
        their content. None (default) uses ``frac`` for the height too.

    Returns:
    width, height (int): the size the window was given.
    """
    geom = available_screen_geometry()
    max_w = round(geom.width() * frac)
    if height_margin is None:
        max_h = round(geom.height() * frac)
        frame_extra = 0
    else:
        max_h = geom.height() - int(height_margin)
        frame_extra = int(height_margin)
    if size is None:
        w, h = max_w, max_h
    else:
        w, h = min(int(size[0]), max_w), min(int(size[1]), max_h)
    win.resize(w, h)
    # Center using the actual size, which a layout minimum may have enlarged.
    # move() places the frame, so leave room for the title bar when asked.
    x = geom.x() + max(0, (geom.width() - win.width()) // 2)
    y = geom.y() + max(0, (geom.height() - win.height() - frame_extra) // 2)
    win.move(x, y)
    return win.width(), win.height()


def scroll_area_min_width(scroll):
    """
    Return the width a scroll area needs to show its widget without a
    horizontal scrollbar (with room for a vertical scrollbar).

    Parameters:
    scroll (QScrollArea): Scroll area whose widget holds the content.

    Returns:
    width (int): The widget's minimum size-hint width plus the vertical
        scrollbar and frame, in logical pixels. 0 if there is no widget.
    """
    widget = scroll.widget()
    if widget is None:
        return 0
    return (widget.minimumSizeHint().width()
            + scroll.verticalScrollBar().sizeHint().width() + 2 * scroll.frameWidth())


def scroll_area_content_height(scroll, width=None):
    """
    Return the height a scroll area needs to show its widget without scrolling.

    Use it to size a window to its content: replace the scroll area's own
    size hint with this height in the window's size hint, then pass the
    result to ``fit_window_to_screen``, which caps it at the screen height.

    A resizable scroll area (``setWidgetResizable(True)``) only shows a
    vertical scrollbar when its viewport is shorter than the widget's
    minimum size hint, so that is the height used. (Plots set their minimum
    heights, e.g. from ``plot_scale``, and grow into any extra space.)

    Parameters:
    scroll (QScrollArea): Scroll area whose widget holds the content.
    width (int or None): Width the scroll area will have. If it is too narrow
        for the content, a horizontal scrollbar appears, and its height is
        added. None (default) assumes it is wide enough.

    Returns:
    height (int): The widget's minimum size-hint height plus the scroll
        area's frame (and horizontal scrollbar, if needed), in logical
        pixels. 0 if the scroll area has no widget.
    """
    widget = scroll.widget()
    if widget is None:
        return 0
    height = widget.minimumSizeHint().height() + 2 * scroll.frameWidth()
    content_width = widget.minimumSizeHint().width() + 2 * scroll.frameWidth()
    if width is not None and width < content_width:
        height += scroll.horizontalScrollBar().sizeHint().height()
    return height


def vbox_height_for_width(layout, width, overrides=None):
    """
    Return the height a vertical box layout needs at a given width.

    Items that wrap text (height-for-width, e.g. word-wrapped labels) are
    measured at ``width``, which a plain size hint doesn't do: a toolbar
    whose labels wrap onto extra lines is taller than its size hint.

    Parameters:
    layout (QVBoxLayout): Layout to measure.
    width (int): Width of the layout's widget, in logical pixels.
    overrides (dict or None): Heights to use for specific widgets instead
        of measuring them, e.g. ``{scroll: scroll_area_content_height(scroll)}``.
        None (default) measures every item.

    Returns:
    height (int): Total height including margins and spacing, in logical
        pixels.
    """
    overrides = overrides or {}
    margins = layout.contentsMargins()
    inner_width = width - margins.left() - margins.right()
    heights = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        # Before a window is shown every child reports isHidden(); skip only
        # widgets that were hidden explicitly.
        if (widget is not None and widget.isHidden()
                and widget.testAttribute(_QtCore.Qt.WidgetAttribute.WA_WState_ExplicitShowHide)):
            continue
        if widget is not None and widget in overrides:
            height = overrides[widget]
        elif item.hasHeightForWidth():
            height = item.heightForWidth(inner_width)
        else:
            height = item.sizeHint().height()
        heights.append(max(height, item.minimumSize().height()))
    spacing = max(layout.spacing(), 0) * max(len(heights) - 1, 0)
    return margins.top() + margins.bottom() + sum(heights) + spacing
