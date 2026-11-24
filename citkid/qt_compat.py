"""
Compatibility shim for Qt enum constants across pyqtgraph backends.

pyqtgraph 0.13+ wraps whichever Qt binding it finds (PyQt5, PyQt6, PySide2,
PySide6).  In some combinations the traditional flat namespace
(``QtCore.Qt.ShiftModifier``, ``QtCore.Qt.LeftButton``, etc.) is not
available — constants live inside nested enum classes instead
(``QtCore.Qt.KeyboardModifier.ShiftModifier``).

citkid itself always uses PyQt5: importing citkid sets
``PYQTGRAPH_QT_LIB=PyQt5``, and importing this module raises ImportError if
pyqtgraph is on another binding (imported before citkid). Loading a second
Qt major version into the process (PyQt6, which rfmux's periscope GUI
imports) crashes when the QApplication is created, so importing citkid
also sets ``CRS_EMBEDDED`` to keep rfmux from importing periscope, and
``get_qapp`` refuses to create the application if another binding is
loaded anyway.

This module exposes every constant used by citkid under a single ``Qt``
namespace that works regardless of binding, so call sites can write:

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
* ``run_responsive`` runs slow work (fits, zarr writes) in a worker thread
  while the GUI keeps processing events, so the operating system doesn't
  report the window as not responding.
"""

import math as _math
import os as _os
import sys as _sys
import threading as _threading
import warnings as _warnings
from html import escape as _escape

import pyqtgraph as _pg

# citkid sets PYQTGRAPH_QT_LIB=PyQt5 on import (see citkid/__init__.py).
if _pg.Qt.QT_LIB != 'PyQt5':
    raise ImportError(
        f"citkid's Qt code needs PyQt5, but pyqtgraph is using "
        f'{_pg.Qt.QT_LIB}. pyqtgraph was probably imported before citkid '
        '(e.g. by rfmux): restart Python and import citkid first.'
    )

from pyqtgraph.Qt import (
    QtCore as _QtCore, QtGui as _QtGui, QtWidgets as _QtWidgets
)

# Qt bindings that must not be loaded alongside PyQt5 (``get_qapp``)
_OTHER_BINDINGS = ('PyQt6', 'PySide2', 'PySide6')

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
        """
        Look up a Qt constant by name.

        Parameters:
        name (str): name of the constant, e.g. 'ShiftModifier'.

        Returns:
        val: the Qt constant.

        Raises:
        AttributeError: if no Qt constant has this name.
        """
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
        """
        Return the representation of the ``Qt`` proxy.

        Returns:
        text (str): '<citkid.qt_compat.Qt>'.
        """
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
            """
            Windows LOGFONTW structure (font attributes).
            """
            _fields_ = [
                ('lfHeight', wintypes.LONG), ('lfWidth', wintypes.LONG),
                ('lfEscapement', wintypes.LONG),
                ('lfOrientation', wintypes.LONG),
                ('lfWeight', wintypes.LONG), ('lfItalic', wintypes.BYTE),
                ('lfUnderline', wintypes.BYTE),
                ('lfStrikeOut', wintypes.BYTE),
                ('lfCharSet', wintypes.BYTE),
                ('lfOutPrecision', wintypes.BYTE),
                ('lfClipPrecision', wintypes.BYTE),
                ('lfQuality', wintypes.BYTE),
                ('lfPitchAndFamily', wintypes.BYTE),
                ('lfFaceName', wintypes.WCHAR * 32),
            ]

        class NONCLIENTMETRICSW(ctypes.Structure):
            """
            Windows NONCLIENTMETRICSW structure (non-client area metrics).
            """
            _fields_ = [
                ('cbSize', wintypes.UINT), ('iBorderWidth', ctypes.c_int),
                ('iScrollWidth', ctypes.c_int),
                ('iScrollHeight', ctypes.c_int),
                ('iCaptionWidth', ctypes.c_int),
                ('iCaptionHeight', ctypes.c_int),
                ('lfCaptionFont', LOGFONTW),
                ('iSmCaptionWidth', ctypes.c_int),
                ('iSmCaptionHeight', ctypes.c_int),
                ('lfSmCaptionFont', LOGFONTW),
                ('iMenuWidth', ctypes.c_int), ('iMenuHeight', ctypes.c_int),
                ('lfMenuFont', LOGFONTW), ('lfStatusFont', LOGFONTW),
                ('lfMessageFont', LOGFONTW),
                ('iPaddedBorderWidth', ctypes.c_int),
            ]

        user32 = ctypes.windll.user32
        metrics = NONCLIENTMETRICSW()
        metrics.cbSize = ctypes.sizeof(metrics)
        SPI_GETNONCLIENTMETRICS = 0x29
        if not user32.SystemParametersInfoW(
                SPI_GETNONCLIENTMETRICS, metrics.cbSize,
                ctypes.byref(metrics), 0):
            return None, None
        dpi = (user32.GetDpiForSystem() if hasattr(user32, 'GetDpiForSystem')
               else 96)
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


_LIGHT = False

def set_theme(light = False):
    """
    Choose the plot theme of every citkid window opened afterwards:
    pyqtgraph's background and foreground, and the colour ``fg``
    returns for lines drawn in the foreground colour.

    Parameters:
    light (bool): If True, white plots with black axes and text, for
        screens where the dark theme has too little contrast. Default
        False: black plots.
    """
    global _LIGHT
    _LIGHT = bool(light)
    _pg.setConfigOptions(background = 'w' if _LIGHT else 'k',
                         foreground = 'k' if _LIGHT else 'w')
    _install_light_colors(_LIGHT)

# pyqtgraph's own mkColor, kept while the light theme replaces it
_MKCOLOR = _pg.functions.mkColor

def light_color(color):
    """
    Return a plot colour as drawn with the light theme: a colour too
    pale to see on white is darkened and saturated (a pale blue
    becomes a full blue, yellow an olive, light grey a dark grey);
    white, black, and colours already dark are unchanged, and the
    alpha is kept.

    Parameters:
    color (QColor): the colour.

    Returns:
    color (QColor): the colour to draw.
    """
    lum = (0.2126 * color.red() + 0.7152 * color.green()
           + 0.0722 * color.blue()) / 255.
    if lum < 0.55 or lum > 0.97:
        return color
    h, s, v, a = color.getHsvF()
    if s < 0.15:
        s, v = 0., min(v, 0.45)
    else:
        s, v = max(s, 0.85), min(v, 0.72)
    return _QtGui.QColor.fromHsvF(max(h, 0.), s, v, a)

def _mk_color_light(*args, **kwargs):
    """
    Make a colour as pyqtgraph does, then ``light_color`` it: stands in
    for ``pyqtgraph.functions.mkColor`` (which ``mkPen`` and
    ``mkBrush`` use) with the light theme, so every pen and brush of
    every plot is readable on white without each call site changing.

    Parameters:
    *args, **kwargs: as ``pyqtgraph.mkColor``.

    Returns:
    color (QColor): the colour.
    """
    return light_color(_MKCOLOR(*args, **kwargs))

def _install_light_colors(on):
    """
    Put ``_mk_color_light`` in place of pyqtgraph's ``mkColor`` (in
    ``pyqtgraph`` and ``pyqtgraph.functions``), or pyqtgraph's own
    back.

    Parameters:
    on (bool): True for the light theme's colours.
    """
    f = _mk_color_light if on else _MKCOLOR
    _pg.functions.mkColor = f
    _pg.mkColor = f

def is_light():
    """
    Say whether the light theme is on (``set_theme``).

    Returns:
    light (bool): True if so.
    """
    return _LIGHT

def fg():
    """
    Return the plot foreground colour of the theme (``set_theme``), for
    pens that must stand out from the background.

    Returns:
    color (str): 'k' with the light theme, else 'w'.
    """
    return 'k' if _LIGHT else 'w'

def log_box_style():
    """
    Return the stylesheet of a monospace log box in the theme's colours.

    Returns:
    style (str): the stylesheet.
    """
    colors = 'background-color: #f4f4f4; color: #000000; ' if _LIGHT \
        else 'background-color: #2b2b2b; color: #ffffff; '
    return colors + 'font-family: monospace;'

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

    Raises:
    RuntimeError: if another Qt binding (PyQt6, PySide2, PySide6) is loaded
        when the application has to be created. Two Qt major versions in
        one process segfault in QApplication, so this fails clearly first.
    """
    app = _QtWidgets.QApplication.instance()
    if app is None:
        others = [lib for lib in _OTHER_BINDINGS if lib in _sys.modules]
        if others:
            raise RuntimeError(
                f"citkid's Qt code uses PyQt5, but {', '.join(others)} "
                f'{"is" if len(others) == 1 else "are"} also loaded, and '
                'two Qt versions in one process crash when the '
                'QApplication is created. rfmux.tools.periscope imports '
                'PyQt6: import citkid before rfmux (it then sets '
                'CRS_EMBEDDED so rfmux skips periscope), and do not import '
                'periscope or other Qt6 code in the same process.'
            )
        app = _pg.mkQApp(name)
        _use_system_ui_font(app)
        set_app_icon(app)
        return app
    # An application made elsewhere (a notebook, an IDE): the icon too
    icon = getattr(app, 'windowIcon', None)
    if callable(icon) and icon().isNull():
        set_app_icon(app)
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


# The icon of every citkid window: replace the file to change it
ICON_PATH = _os.path.join(_os.path.dirname(__file__), 'icon.png')

def draw_icon(size = 256):
    """
    Draw the built-in icon: the IQ loop of a resonance (a ring of sweep
    points, dense near resonance) with "citkid" in the middle, in
    white on a Caltech-orange rounded square.

    Parameters:
    size (int): pixels on a side. Default 256.

    Returns:
    pixmap (QPixmap): the icon.
    """
    pix = _QtGui.QPixmap(size, size)
    pix.fill(_QtCore.Qt.transparent)
    p = _QtGui.QPainter(pix)
    p.setRenderHint(_QtGui.QPainter.Antialiasing)
    p.setPen(_QtCore.Qt.NoPen)
    p.setBrush(_QtGui.QColor(255, 108, 12))     # Caltech orange
    r = size * 0.18
    p.drawRoundedRect(0, 0, size, size, r, r)
    # The IQ loop: the sweep's points, z = 1 - 1 / (1 + 2j Q x), lie
    # on a circle, dense far from resonance and sparse through the
    # dip (x from -3 to 3 in Q x)
    cx, cy, rad = size * 0.5, size * 0.5, size * 0.36
    n = 28
    zs = [1 - 1 / (1 + 2j * (-3. + 6. * i / (n - 1))) for i in range(n)]
    # The ring runs under the points only: from the first off-resonance
    # point round through the resonance to the last one, leaving the
    # sweep's gap open on the off-resonance side (circle of diameter 1
    # at 0.5; Qt angles counter-clockwise from 3 o'clock, in 1/16 deg)
    angle = lambda z: _math.degrees(_math.atan2(z.imag, z.real - 0.5))
    start, end = angle(zs[-1]), angle(zs[0])
    ring = _QtGui.QPen(_QtGui.QColor(255, 255, 255, 110), size * 0.045)
    ring.setCapStyle(_QtCore.Qt.RoundCap)
    p.setPen(ring)
    p.setBrush(_QtCore.Qt.NoBrush)
    p.drawArc(_QtCore.QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad),
              int(round(start * 16)),
              int(round(((end - start) % 360.) * 16)))
    p.setPen(_QtCore.Qt.NoPen)
    p.setBrush(_QtGui.QColor(255, 255, 255))
    for z in zs:
        pt = _QtCore.QPointF(cx + 2 * rad * (z.real - 0.5),
                             cy - 2 * rad * z.imag)
        p.drawEllipse(pt, size * 0.028, size * 0.028)
    # The name, as large as fits in the loop
    font = _QtGui.QFont('Arial')
    font.setBold(True)
    font.setPixelSize(int(size * 0.19))
    p.setFont(font)
    p.setPen(_QtGui.QColor(255, 255, 255))
    p.drawText(_QtCore.QRectF(0, 0, size, size), _QtCore.Qt.AlignCenter,
               'citkid')
    p.end()
    return pix

def app_icon():
    """
    Return the icon of the citkid windows: ``ICON_PATH`` (icon.png next
    to this module) if it exists, else the built-in drawing
    (``draw_icon``).

    Returns:
    icon (QIcon): the icon.
    """
    if _os.path.exists(ICON_PATH):
        icon = _QtGui.QIcon(ICON_PATH)
        if not icon.isNull():
            return icon
    return _QtGui.QIcon(draw_icon())

APP_ID = 'citkid'
DESKTOP_ENTRY = APP_ID + '.desktop'

def install_desktop_entry(home = None):
    """
    Write the desktop entry that lets a Linux desktop (GNOME on
    Ubuntu, for one) show the citkid icon for its windows:
    ``~/.local/share/applications/citkid.desktop``, hidden from the
    application menu, naming the icon file and the window class.
    The desktop matches windows to it by their application id
    (Wayland) or window class (X11), both set by ``set_app_icon``.
    Nothing is written when the entry is already up to date.

    Parameters:
    home (str or None): the home folder. Default None: the user's.

    Returns:
    path (str): the entry's path.
    written (bool): True if it was written or changed.
    """
    home = _os.path.expanduser('~') if home is None else str(home)
    icon = ICON_PATH
    if not _os.path.exists(icon):
        # No icon file shipped: save the built-in drawing
        icon = _os.path.join(home, '.local', 'share', 'icons', 'hicolor',
                             '256x256', 'apps', APP_ID + '.png')
        _os.makedirs(_os.path.dirname(icon), exist_ok = True)
        draw_icon(256).save(icon, 'PNG')
    text = ('[Desktop Entry]\nType=Application\nName=citkid\n'
            'Comment=citkid windows\nExec=python3\nTerminal=false\n'
            f'NoDisplay=true\nIcon={icon}\nStartupWMClass={APP_ID}\n')
    apps = _os.path.join(home, '.local', 'share', 'applications')
    path = _os.path.join(apps, DESKTOP_ENTRY)
    if _os.path.exists(path):
        with open(path, encoding = 'utf-8') as fh:
            if fh.read() == text:
                return path, False
    _os.makedirs(apps, exist_ok = True)
    with open(path, 'w', encoding = 'utf-8') as fh:
        fh.write(text)
    return path, True

def set_app_icon(app):
    """
    Give every window of an application the citkid icon
    (``app_icon``). On Windows the process is also given its own
    application id, so the taskbar shows the icon instead of
    Python's. On Linux the application is named ``citkid`` (its
    window class on X11 and application id on Wayland) and the
    desktop entry that carries the icon is written
    (``install_desktop_entry``), since the desktop shows the entry's
    icon, not the window's.

    Parameters:
    app (QApplication): the application.
    """
    if _sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                APP_ID)
        except Exception:
            pass
    elif _sys.platform.startswith('linux'):
        app.setApplicationName(APP_ID)
        if hasattr(app, 'setDesktopFileName'):
            app.setDesktopFileName(APP_ID)
        try:
            install_desktop_entry()
        except OSError:
            pass
    app.setWindowIcon(app_icon())

def available_screen_geometry():
    """
    Get the available geometry of the screen under the mouse cursor.

    Falls back to the primary screen if no screen is under the cursor.

    Returns:
    geom (QRect): available screen area (excluding taskbars) in logical
        pixels.
    """
    app = _QtWidgets.QApplication.instance() or get_qapp()
    screen = (app.screenAt(_QtGui.QCursor.pos()) if hasattr(app, 'screenAt')
              else None)
    if screen is None:
        screen = app.primaryScreen()
    return screen.availableGeometry()


def delete_on_close(win):
    """
    Make Qt delete a top-level window as soon as it is closed.

    Without this, a closed window lives until Python's garbage collector
    frees it, which can happen in any thread (including background threads
    of zarr or the GUIs). Destroying Qt widgets outside the GUI thread can
    crash the process, e.g. a Jupyter kernel. With this set, Qt destroys the
    window on the GUI thread right after it closes. The Python object stays
    usable for plain attributes (e.g. results), but Qt methods on it then
    raise ``RuntimeError``. A close that is cancelled (e.g. by a popup)
    deletes nothing.

    Parameters:
    win (QWidget): Top-level window.
    """
    win.setAttribute(_QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)


class HoldPanKey(_QtCore.QObject):
    """
    Single-key shortcut that pans by a step when tapped and scrolls smoothly
    while held.

    A tap calls ``pan(step)`` once. If the key is still down after
    ``hold_delay_ms``, the view scrolls continuously at ``rate`` view widths
    per second (in the direction of ``step``) until the key is released or
    the application loses focus. OS key auto-repeat is ignored, so the speed
    does not depend on the keyboard repeat settings.
    """

    def __init__(self, key, window, step, pan, rate=0.5, hold_delay_ms=350,
                 interval_ms=30):
        """
        Create the shortcut on ``window``.

        Parameters:
        key (str): single letter key, e.g. 'Z'.
        window (QWidget): window the shortcut is active in; also the Qt
            parent of this object.
        step (float): pan per tap as a signed fraction of the view width.
        pan (callable): called with a signed fraction of the view width to
            shift the view by. It should redraw immediately, since a held key
            pans every ``interval_ms``.
        rate (float): scroll speed while held, in view widths per second.
            Default 0.5.
        hold_delay_ms (int): how long the key must be held before smooth
            scrolling starts. Default 350.
        interval_ms (int): time between smooth-scroll updates. Default 30.
        """
        super().__init__(window)
        self.step = float(step)
        self.rate = float(rate)
        self._pan = pan
        self._key = getattr(_QtCore.Qt.Key, 'Key_' + key.upper())
        self._held = False

        self._shortcut = _QtGui.QShortcut(_QtGui.QKeySequence(key), window)
        self._shortcut.setAutoRepeat(False)
        self._shortcut.activated.connect(self._on_press)

        self._delay_timer = _QtCore.QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.setInterval(int(hold_delay_ms))
        self._delay_timer.timeout.connect(self._start_scroll)
        self._scroll_timer = _QtCore.QTimer(self)
        self._scroll_timer.setInterval(int(interval_ms))
        self._scroll_timer.timeout.connect(self._scroll)
        self._clock = _QtCore.QElapsedTimer()

    def _on_press(self):
        """
        Pan one step and start waiting to see whether the key is held.
        """
        if self._held:
            return
        self._held = True
        # Watch the whole application for the release, since it goes to
        # whichever widget has focus. Only installed while the key is down.
        _QtWidgets.QApplication.instance().installEventFilter(self)
        self._pan(self.step)
        self._delay_timer.start()

    def _start_scroll(self):
        """
        Begin smooth scrolling after the hold delay.
        """
        self._clock.start()
        self._scroll_timer.start()

    def _scroll(self):
        """
        Pan by the distance covered since the last update.
        """
        dt = self._clock.restart() / 1000.0
        direction = 1.0 if self.step >= 0 else -1.0
        self._pan(direction * self.rate * dt)

    def stop(self):
        """
        Stop scrolling and stop watching for the key release.
        """
        if not self._held:
            return
        self._held = False
        self._delay_timer.stop()
        self._scroll_timer.stop()
        app = _QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    def eventFilter(self, obj, event):
        """
        Stop on release of the key or when the application loses focus.

        Parameters:
        obj (QObject): object that received the event.
        event (QEvent): Qt event.

        Returns:
        handled (bool): always False, so the event is processed normally.
        """
        etype = event.type()
        if etype == _QtCore.QEvent.Type.KeyRelease:
            if event.key() == self._key and not event.isAutoRepeat():
                self.stop()
        elif etype == _QtCore.QEvent.Type.ApplicationDeactivate:
            self.stop()
        return False


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
            + scroll.verticalScrollBar().sizeHint().width()
            + 2 * scroll.frameWidth())


def scroll_area_content_height(scroll, width=None):
    """
    Return the height a scroll area needs to show its widget without
    scrolling.

    Use it to size a window to its content: replace the scroll area's own
    size hint with this height in the window's size hint, then pass the
    result to ``fit_window_to_screen``, which caps it at the screen height.

    A resizable scroll area (``setWidgetResizable(True)``) only shows a
    vertical scrollbar when its viewport is shorter than the widget's
    minimum size hint, so that is the height used. (Plots set their minimum
    heights, e.g. from ``plot_scale``, and grow into any extra space.)

    Parameters:
    scroll (QScrollArea): Scroll area whose widget holds the content.
    width (int or None): Width the scroll area will have. If it is too
        narrow for the content, a horizontal scrollbar appears, and its
        height is added. None (default) assumes it is wide enough.

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
        of measuring them, e.g.
        ``{scroll: scroll_area_content_height(scroll)}``. None (default)
        measures every item.

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
                and widget.testAttribute(
                    _QtCore.Qt.WidgetAttribute.WA_WState_ExplicitShowHide)):
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


# Number of run_responsive calls in progress on the GUI thread.
_responsive_depth = 0


def _exclude_user_input_flag():
    """
    Return the QEventLoop flag that defers mouse and keyboard events.

    Returns:
    flag (QEventLoop.ProcessEventsFlag): ``ExcludeUserInputEvents``.
    """
    loop = _QtCore.QEventLoop
    flags = getattr(loop, 'ProcessEventsFlag', None)
    if flags is not None and hasattr(flags, 'ExcludeUserInputEvents'):
        return flags.ExcludeUserInputEvents
    return loop.ExcludeUserInputEvents


def run_responsive(func, *args, poll_s=0.03, **kwargs):
    """
    Run a function in a worker thread while the GUI keeps processing events.

    Windows and Linux report a window as not responding when its GUI thread
    stops processing events for a few seconds, e.g. during a long fit or a
    zarr write. Running the work in a thread while this (GUI) thread keeps
    processing events lets the window repaint (including busy dialogs) and
    answer the operating system. Mouse and keyboard events are deferred
    until the work finishes, so the user can't start other work meanwhile;
    windows should still refuse to close while ``responsive_busy()`` (window
    close requests are not deferred).

    ``func`` must not touch Qt widgets: collect widget values before, and
    update widgets after.

    Parameters:
    func (callable): the work.
    *args, **kwargs: arguments for ``func``.
    poll_s (float): how often to process events while waiting (s).

    Returns:
    result: what ``func`` returned.

    Raises:
    Exception: whatever ``func`` raised, re-raised on this thread.

    Notes:
    Without a QApplication, or when called from a thread other than the main
    thread, ``func`` runs directly.
    """
    global _responsive_depth
    app = _QtWidgets.QApplication.instance()
    if app is None or \
            _threading.current_thread() is not _threading.main_thread():
        return func(*args, **kwargs)
    outcome = {}

    def target():
        """
        Run the work, keeping its result or exception.
        """
        try:
            outcome['result'] = func(*args, **kwargs)
        except BaseException as exc:  # re-raised on the GUI thread
            outcome['error'] = exc

    worker = _threading.Thread(target=target, name='citkid-responsive',
                               daemon=True)
    flag = _exclude_user_input_flag()
    _responsive_depth += 1
    try:
        worker.start()
        while worker.is_alive():
            worker.join(poll_s)
            app.processEvents(flag)
    finally:
        _responsive_depth -= 1
    if 'error' in outcome:
        raise outcome['error']
    return outcome.get('result')


def responsive_busy():
    """
    Check whether ``run_responsive`` work is in progress on the GUI thread.

    Returns:
    busy (bool): True while a ``run_responsive`` call is waiting for its
        work.
    """
    return _responsive_depth > 0


def describe_display(print_report=True):
    """
    Describe the Qt display setup, to debug window sizes and fonts.

    Reports the Qt binding pyqtgraph uses, the Qt platform and style,
    scaling environment variables, each screen's size, logical and physical
    DPI and device pixel ratio, the default font, and the palette colors
    used for muted text.

    Parameters:
    print_report (bool): If True (default), print the report.

    Returns:
    report (str): the report.
    """
    import os
    import sys
    app = _QtWidgets.QApplication.instance() or get_qapp()
    lines = [
        f'python {sys.version.split()[0]}, Qt {_QtCore.QT_VERSION_STR}, '
        f'pyqtgraph {_pg.__version__}, platform {sys.platform}',
        f'Qt binding used by pyqtgraph: {_pg.Qt.QT_LIB}',
        f'Qt platform plugin: {app.platformName()}, '
        f'style: {app.style().objectName()}',
        'high-DPI scaling: ' + str(app.testAttribute(
            _QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling)),
    ]
    env_names = ('QT_SCALE_FACTOR', 'QT_AUTO_SCREEN_SCALE_FACTOR',
                 'QT_SCREEN_SCALE_FACTORS', 'QT_ENABLE_HIGHDPI_SCALING',
                 'QT_FONT_DPI', 'QT_QPA_PLATFORM', 'QT_STYLE_OVERRIDE',
                 'XDG_SESSION_TYPE', 'XDG_CURRENT_DESKTOP', 'GDK_SCALE',
                 'GDK_DPI_SCALE')
    env = {name: os.environ[name] for name in env_names if name in os.environ}
    lines.append(f'environment: {env or "none set"}')
    for screen in app.screens():
        geom, avail = screen.geometry(), screen.availableGeometry()
        lines.append(
            f'screen {screen.name()!r}: {geom.width()}x{geom.height()} '
            f'logical px (available {avail.width()}x{avail.height()}), '
            f'logical DPI {screen.logicalDotsPerInch():.0f}, physical DPI '
            f'{screen.physicalDotsPerInch():.0f}, device pixel ratio '
            f'{screen.devicePixelRatio():g}')
    font = app.font()
    metrics = _QtGui.QFontMetrics(font)
    lines.append(f'default font: {font.family()!r} {font.pointSizeF():g} pt '
                 f'({metrics.height()} px line height)')
    ten_pt = _QtGui.QFont(font)
    ten_pt.setPointSizeF(10)
    ten_pt_px = _QtGui.QFontMetrics(ten_pt).height()
    lines.append(f'10 pt text is {ten_pt_px} px tall (about 16 px at 96 DPI)')
    palette = app.palette()
    roles = _QtGui.QPalette.ColorRole
    colors = {name: palette.color(getattr(roles, name)).name()
              for name in ('Window', 'WindowText', 'Mid', 'Text', 'Base')}
    lines.append(f'palette: {colors}')
    report = '\n'.join(lines)
    if print_report:
        print(report)
    return report


################################################################################
# Fonts and help dialogs
################################################################################

# Base font sizes in points at ui_scale = 1. Fonts are always set explicitly,
# so text size depends only on ui_scale, not on the system default font.
WIDGET_FONT_PT = 10
TICK_FONT_PT = 9
LABEL_FONT_PT = 11


def font_px(point_size, ui_scale=1.0, minimum=8):
    """
    Convert a font size in points to logical pixels.

    Fonts are set in logical pixels, like every window and widget size, so
    text keeps the same size relative to the window on every system. In
    points, text grows with the system's text scaling (e.g. a larger
    Xft.dpi on Linux) while the windows don't. On Windows (96 logical DPI)
    the result is the same size as the point size.

    Parameters:
    point_size (float): size in points at ui_scale = 1.
    ui_scale (float): font size multiplier. Default 1.0.
    minimum (int): smallest size returned, in pixels. Default 8.

    Returns:
    px (int): size in logical pixels (point_size * ui_scale * 96 / 72).
    """
    return max(int(minimum), round(point_size * ui_scale * 96 / 72))


def widget_font_stylesheet(ui_scale):
    """
    Return a stylesheet that sets the font size of every widget.

    Parameters:
    ui_scale (float): Font size multiplier.

    Returns:
    stylesheet (str): ``'* { font-size: Npx; }'`` with
        N = ``font_px(WIDGET_FONT_PT, ui_scale)``.
    """
    return f"* {{ font-size: {font_px(WIDGET_FONT_PT, ui_scale)}px; }}"


def hint_label(text):
    """
    Return a label for keyboard-shortcut hints.

    The text word-wraps, so put the label on a row of its own (full width);
    beside other widgets it can be squeezed to nothing. Its color is a mix
    of the theme's text and background colors, so it is muted but readable
    with any theme (``palette(mid)`` can match the background on some Linux
    themes).

    Parameters:
    text (str): the hints.

    Returns:
    label (QLabel): the label.
    """
    label = _QtWidgets.QLabel(text)
    label.setWordWrap(True)
    palette = label.palette()
    fg = palette.color(_QtGui.QPalette.ColorRole.WindowText)
    bg = palette.color(_QtGui.QPalette.ColorRole.Window)
    pairs = ((fg.red(), bg.red()), (fg.green(), bg.green()),
             (fg.blue(), bg.blue()))
    r, g, b = [round(0.65 * f + 0.35 * b) for f, b in pairs]
    label.setStyleSheet(f"color: rgb({r}, {g}, {b}); font-style: italic;")
    return label


def help_html(title, sections):
    """
    Format a help dialog's text as HTML.

    Parameters:
    title (str): heading.
    sections (list): ``(heading, items)`` pairs, where items is a list of
        ``(keys, description)`` string pairs; an empty ``keys`` shows the
        description alone. Plain text: ``<``, ``>`` and ``&`` are escaped.

    Returns:
    html (str): the help text.
    """
    parts = [f'<h3>{_escape(title)}</h3>']
    for heading, items in sections:
        parts.append(f'<p><b>{_escape(heading)}</b></p><ul>')
        parts.extend(f'<li><b>{_escape(keys)}:</b> {_escape(text)}</li>'
                     if keys else f'<li>{_escape(text)}</li>'
                     for keys, text in items)
        parts.append('</ul>')
    return ''.join(parts)


def add_help(window, title, sections):
    """
    Give a window a help dialog on the H key (``toggle_help_dialog``).

    Parameters:
    window (QWidget): the window.
    title (str): the dialog's heading.
    sections (list): ``(heading, [(keys, description), ...])`` pairs
        (``help_html``).
    """
    _QtWidgets.QShortcut(_QtGui.QKeySequence('H'), window,
                         lambda: toggle_help_dialog(window, title, sections))

def toggle_help_dialog(window, title, sections):
    """
    Show or hide a window's help dialog (the H key), building it on first
    use.

    The dialog lists the window's keyboard shortcuts, buttons and mouse
    controls. It scrolls if it is taller than the screen, and H (or Esc)
    closes it again.

    Parameters:
    window (QWidget): window the dialog belongs to; the dialog is kept in
        ``window._help_dialog`` and centered on the window.
    title (str): window name, e.g. 'IQ Series'.
    sections (list): ``(heading, items)`` pairs (see ``help_html``).

    Returns:
    dlg (QDialog): the dialog.
    """
    dlg = getattr(window, '_help_dialog', None)
    if dlg is None:
        dlg = _QtWidgets.QDialog(window)
        dlg.setWindowTitle(f'{title} Help  (H to close)')
        label = _QtWidgets.QLabel(help_html(f'{title} Controls', sections))
        label.setTextFormat(_QtCore.Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setMargin(8)
        scroll = _QtWidgets.QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(_QtWidgets.QFrame.Shape.NoFrame)
        layout = _QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        _QtGui.QShortcut(_QtGui.QKeySequence('H'), dlg, activated=dlg.hide)
        geom = available_screen_geometry()
        width = min(620, geom.width() - 40)
        label.setFixedWidth(width - 30)
        height = min(label.heightForWidth(width - 30) + 20,
                     geom.height() - TITLE_BAR_MARGIN - 40)
        dlg.resize(width, height)
        window._help_dialog = dlg
    if dlg.isVisible():
        dlg.hide()
        return dlg
    dlg.show()
    frame = dlg.frameGeometry()
    frame.moveCenter(window.frameGeometry().center())
    dlg.move(frame.topLeft())
    dlg.raise_()
    return dlg


def scale_plot_fonts(ui_scale, *plot_items):
    """
    Set tick, axis-label and title fonts of pyqtgraph plots from ui_scale.

    Plot text isn't affected by widget stylesheets, so it is set here.

    Parameters:
    ui_scale (float): Font size multiplier.
    *plot_items (pg.PlotItem): Plots to style.
    """
    tick_font = _QtGui.QFont()
    tick_font.setPixelSize(font_px(TICK_FONT_PT, ui_scale))
    label_font = _QtGui.QFont()
    label_font.setPixelSize(font_px(LABEL_FONT_PT, ui_scale, minimum=9))
    for plot in plot_items:
        for ax_name in ('bottom', 'left', 'top', 'right'):
            ax = plot.getAxis(ax_name)
            ax.setStyle(tickFont=tick_font)
            ax.label.setFont(label_font)
        plot.titleLabel.item.setFont(label_font)


class PlainAxisItem(_pg.AxisItem):
    """
    Axis whose ticks are plain numbers: no SI prefix scaling of the label
    (pyqtgraph's "(x0.001)"), fixed notation while the ticks are between
    1e-4 and 1e4 in size (0.2, 0.4, ...), and scientific notation
    (1.2e5, 3e-5) beyond that. Log mode is pyqtgraph's.

    Use it for an axis of a quantity without units, e.g.
    ``addPlot(axisItems = {'left': PlainAxisItem('left')})``.
    """
    # Tick sizes (largest |tick|) formatted in fixed notation
    FIXED_MIN = 1e-4
    FIXED_MAX = 1e4

    def __init__(self, orientation, **kwargs):
        """
        Make the axis.

        Parameters:
        orientation (str): 'left', 'right', 'top', or 'bottom'.
        **kwargs: ``pg.AxisItem`` arguments.
        """
        super().__init__(orientation, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        """
        Format the tick values (pyqtgraph calls this).

        Parameters:
        values (list of float): tick values.
        scale (float): pyqtgraph's scale factor (1 without SI prefixes).
        spacing (float): spacing between the ticks.

        Returns:
        strings (list of str): one per value.
        """
        if self.logMode or not values:
            return super().tickStrings(values, scale, spacing)
        vals = [v * scale for v in values]
        spacing = spacing * scale
        if not _math.isfinite(spacing) or spacing <= 0:
            return [f'{v:g}' for v in vals]
        size = max(abs(v) for v in vals)
        if size and (size >= self.FIXED_MAX or size < self.FIXED_MIN):
            # Mantissa decimals down to the spacing's decade, so that
            # neighbouring ticks differ (1.0e4, 1.2e4; 1e5, 2e5)
            digits = max(0, _decade(size) - _decade(spacing))
            return [format_sci(v, digits) for v in vals]
        places = max(0, int(_math.ceil(-_math.log10(spacing) - 1e-9)))
        return [f'{v:.{places}f}' for v in vals]


def format_sci(value, digits):
    """
    Format a number in short scientific notation: 1.2e5, 3e-5, 0.

    Parameters:
    value (float): the number.
    digits (int): decimals of the mantissa.

    Returns:
    text (str): the number.
    """
    if value == 0:
        return '0'
    mantissa, exponent = f'{value:.{digits}e}'.split('e')
    return f'{mantissa}e{int(exponent)}'


def _decade(value):
    """
    Return the decade of a positive number: floor(log10(value)).

    Parameters:
    value (float): the number, > 0.

    Returns:
    decade (int): e.g. 5 for 2e5, -5 for 2e-5.
    """
    return int(_math.floor(_math.log10(value) + 1e-9))

def freeze_heights(layout, skip = ()):
    """
    Fix every widget of a layout at its preferred height, so the panel
    does not move when text changes or a widget shows or hides. Call it
    once the panel is built. Nested layouts are frozen too.

    Parameters:
    layout (QLayout): the layout.
    skip (sequence of QWidget): widgets left free, e.g. a list that
        should take the stretch.
    """
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget()
        if w is not None:
            if w in skip:
                continue
            w.setFixedHeight(w.sizeHint().height())
            policy = w.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            w.setSizePolicy(policy)
        elif item.layout() is not None:
            freeze_heights(item.layout(), skip)


class _KeyForwarder(_QtCore.QObject):
    """
    Event filter that gives a widget's key presses to a window's
    ``keyPressEvent`` first (a list widget would otherwise swallow the
    window's letter shortcuts while it has the focus).
    """

    def __init__(self, window):
        """
        Make the filter.

        Parameters:
        window (QWidget): whose ``keyPressEvent`` gets the keys.
        """
        super().__init__(window)
        self._window = window

    def eventFilter(self, obj, event):
        """
        Hand a key press to the window; it is consumed if the window
        accepts it.

        Parameters:
        obj (QObject): the watched widget.
        event (QEvent): the event.

        Returns:
        handled (bool): True if the window took the key.
        """
        if event.type() == _QtCore.QEvent.KeyPress:
            event.ignore()
            self._window.keyPressEvent(event)
            if event.isAccepted():
                return True
        return super().eventFilter(obj, event)


def forward_keys(widget, window):
    """
    Let a window's letter shortcuts work while a child widget (a tone
    list, say) has the keyboard focus: the widget's key presses go to
    ``window.keyPressEvent`` first, and the widget only gets the keys the
    window does not accept.

    Parameters:
    widget (QWidget): the widget that takes the focus.
    window (QWidget): the window with the shortcuts.

    Returns:
    filter (QObject): the installed filter (kept alive by ``window``).
    """
    f = _KeyForwarder(window)
    widget.installEventFilter(f)
    return f
