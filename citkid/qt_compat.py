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


def get_qapp(name=None):
    """
    Return the QApplication, creating it with high-DPI scaling if needed.

    Only one QApplication exists per process, and high-DPI scaling can only
    be enabled before it is created. A new application is created with
    ``pyqtgraph.mkQApp``, which enables it. If an application already
    exists without it (for example, created by other Qt code earlier in the
    same Jupyter kernel), a warning is issued, since windows and fonts may
    then appear too large.

    Parameters:
    name (str or None): application name for a new application. None
        (default) uses pyqtgraph's default.

    Returns:
    app (QApplication): the running application.
    """
    app = _QtWidgets.QApplication.instance()
    if app is None:
        return _pg.mkQApp(name)
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


def fit_window_to_screen(win, frac=0.8, size=None):
    """
    Resize a window to fit on the screen and center it there.

    The window goes on the screen under the mouse cursor. Its size is
    clamped to ``frac`` of the available screen area, so it never opens
    larger than the screen unless its layout's minimum size requires it.

    Parameters:
    win (QWidget): top-level window.
    frac (float): maximum fraction of the available screen width and height
        to use. Default 0.8.
    size (tuple or None): preferred (width, height) in logical pixels,
        clamped to ``frac`` of the screen. None (default) uses the full
        ``frac`` of the screen.

    Returns:
    width, height (int): the size the window was given.
    """
    geom = available_screen_geometry()
    max_w = round(geom.width() * frac)
    max_h = round(geom.height() * frac)
    if size is None:
        w, h = max_w, max_h
    else:
        w, h = min(int(size[0]), max_w), min(int(size[1]), max_h)
    win.resize(w, h)
    # Center using the actual size, which a layout minimum may have enlarged
    x = geom.x() + max(0, (geom.width() - win.width()) // 2)
    y = geom.y() + max(0, (geom.height() - win.height()) // 2)
    win.move(x, y)
    return win.width(), win.height()
