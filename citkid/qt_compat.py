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
"""

from pyqtgraph.Qt import QtCore as _QtCore

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
