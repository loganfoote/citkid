"""
Interactive view of a saved VNA sweep.

``run_vna_view`` opens a window with |S21| in dB and the phase of S21
versus frequency (linked x axes: zoom and pan one, the other follows; the
R key fits both to the data again) for ``vna{k}`` of a device store
(``citkid.daq_layout``), with the band edges marked and the sweep's
settings in a header. Use it after ``01_vna_sweep.py`` (which can open it
when the sweep finishes) and before ``02_find_resonances.py``, to check a
sweep before finding its resonances. Nothing is saved.
``procedures/vna_view.py`` is the script.
"""
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from .. import daq_layout as dl
from ..qt_compat import (
    Qt, delete_on_close, fit_window_to_screen, get_qapp, hint_label,
    widget_font_stylesheet
)

################################################################################
################################### Helpers ####################################
################################################################################
def load_vna_view(root, k = None):
    """
    Load a VNA sweep and describe it for the window.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.
    k (int or None): sweep index. None (default): the latest.

    Returns:
    f (np.array): frequencies in Hz, ascending.
    z (np.array): complex S21.
    bands (dict): the band arrays (``daq_layout.VNA_BAND_ARRAYS``).
    info (str): one line naming the store, the sweep, and its settings.

    Raises:
    KeyError: if the store has no such sweep.
    """
    root = dl.open_store(root)
    k = dl.latest_index(root, 'vna') if k is None else int(k)
    f, z, bands = dl.load_vna_sweep(root, k, bands = True)
    attrs = root[f'vna{k}'].attrs
    store = getattr(getattr(root, 'store', None), 'root', None)
    name = os.path.basename(os.path.normpath(str(store))) if store \
        else root.attrs.get('device', 'store')
    parts = [f'{name}: vna{k}', f'{len(f):,} points',
             f'{f[0] / 1e6:.1f}-{f[-1] / 1e6:.1f} MHz']
    if 'amplitude' in attrs:
        parts.append(f"{attrs['amplitude']:g} dBm")
    if 'nsamps' in attrs:
        parts.append(f"{attrs['nsamps']} samples per point")
    if 'step' in attrs:
        parts.append(f"df/f = {attrs['step']:.2g}")
    npass = int(bands['chunk_passes'].max()) + 1 if \
        len(bands['chunk_passes']) else 0
    parts.append(f"{len(bands['chunk_ncos'])} bands in {npass} passes, "
                 f"modules {sorted(set(bands['chunk_modules'].tolist()))}")
    if attrs.get('notes'):
        parts.append(f"notes: {str(attrs['notes']).strip()}")
    return f, z, bands, '; '.join(parts)

################################################################################
#################################### Window ####################################
################################################################################
class VnaViewWindow(QtWidgets.QMainWindow):
    """
    Window showing a VNA sweep: |S21| (dB) on top, the phase of S21
    (rad) below, against frequency (MHz), with linked x axes.

    Attributes:
    mag_plot, phase_plot (pg.PlotItem): the plots.
    mag_curve, phase_curve (pg.PlotDataItem): the curves.
    unwrap (QCheckBox): unwrap the phase (checked by default).
    """

    def __init__(
        self, f, z, bands = None, info = '', title = 'VNA sweep',
        ui_scale = 1.0
    ):
        """
        Build the window and plot the sweep.

        Parameters:
        f (np.array): frequencies in Hz, ascending.
        z (np.array): complex S21.
        bands (dict or None): band arrays (``chunk_edges`` marks the band
            edges on both plots). Default None: no marks.
        info (str): header text (the sweep and its settings). Default ''.
        title (str): window title. Default 'VNA sweep'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(title)
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.f = np.asarray(f, dtype = float)
        self.z = np.asarray(z, dtype = complex)
        self._band_lines = []
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        self.info = QtWidgets.QLabel(info)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.info)
        self._build_plots()
        lay.addWidget(self.plots, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.unwrap = QtWidgets.QCheckBox('Unwrap phase')
        self.unwrap.setChecked(True)
        self.unwrap.toggled.connect(self._plot_phase)
        row.addWidget(self.unwrap)
        self.rescale_btn = QtWidgets.QPushButton('Rescale (R)')
        self.rescale_btn.clicked.connect(self.rescale)
        row.addWidget(self.rescale_btn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(hint_label(
            'Drag to pan, scroll to zoom (both plots follow in frequency), '
            'right-drag to zoom one axis, R to fit the data again.'))
        self.setCentralWidget(central)
        self.set_data(self.f, self.z, bands)

    def _build_plots(self):
        """
        Build the |S21| and phase plots with linked x axes.
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.mag_plot = self.plots.addPlot(row = 0, col = 0)
        self.phase_plot = self.plots.addPlot(row = 1, col = 0)
        self.phase_plot.setXLink(self.mag_plot)
        self.mag_plot.setLabel('left', '|S21| (dB)')
        self.phase_plot.setLabel('left', 'Phase (rad)')
        self.phase_plot.setLabel('bottom', 'Frequency (MHz)')
        for plot in (self.mag_plot, self.phase_plot):
            plot.showGrid(x = True, y = True, alpha = 0.3)
        self.mag_curve = self.mag_plot.plot(pen = pg.mkPen(width = 1))
        self.phase_curve = self.phase_plot.plot(pen = pg.mkPen(width = 1))
        for item in (self.mag_curve, self.phase_curve):
            item.setDownsampling(auto = True, method = 'peak')
            item.setClipToView(True)

    def set_data(self, f, z, bands = None):
        """
        Show a sweep (replacing the current one) and fit the plots to it.

        Parameters:
        f (np.array): frequencies in Hz, ascending.
        z (np.array): complex S21.
        bands (dict or None): band arrays, for the band edges. Default
            None: no marks.
        """
        self.f = np.asarray(f, dtype = float)
        self.z = np.asarray(z, dtype = complex)
        with np.errstate(divide = 'ignore'):
            db = 20 * np.log10(np.abs(self.z))
        self.mag_curve.setData(self.f / 1e6, db)
        self._plot_phase()
        for plot, line in self._band_lines:
            plot.removeItem(line)
        self._band_lines = []
        if bands is not None and len(bands['chunk_edges']) > 1:
            pen = pg.mkPen((150, 150, 150), style = Qt.DotLine)
            for lo in np.asarray(bands['chunk_edges'])[1:, 0] / 1e6:
                for plot in (self.mag_plot, self.phase_plot):
                    line = pg.InfiniteLine(lo, pen = pen)
                    plot.addItem(line)
                    self._band_lines.append((plot, line))
        self.rescale()

    def phase(self):
        """
        Return the phase shown.

        Returns:
        phase (np.array): the phase of S21 in rad, unwrapped if the box
            is checked.
        """
        phase = np.angle(self.z)
        return np.unwrap(phase) if self.unwrap.isChecked() else phase

    def _plot_phase(self):
        """
        Draw the phase (wrapped or unwrapped) and fit its y axis.
        """
        self.phase_curve.setData(self.f / 1e6, self.phase())
        self.phase_plot.enableAutoRange(axis = 'y')

    def rescale(self):
        """
        Fit both plots to their data again (the R key), e.g. after zooming
        or panning.
        """
        # The curves are clipped to the view, so auto-range alone would fit
        # the part of the data in the zoomed view: first show the whole
        # frequency range (the phase plot follows, linked)
        if len(self.f):
            self.mag_plot.setXRange(self.f[0] / 1e6, self.f[-1] / 1e6,
                                    padding = 0.02)
        # Enabling auto-range fits the view at once (autoRange() would
        # set a fixed range and turn auto-range off again)
        for plot in (self.mag_plot, self.phase_plot):
            plot.enableAutoRange()

    def keyPressEvent(self, event):
        """
        R: rescale the plots.

        Parameters:
        event (QKeyEvent): the key event.
        """
        if event.key() == QtCore.Qt.Key_R and \
                not isinstance(self.focusWidget(), (QtWidgets.QLineEdit,
                                                    QtWidgets.QPlainTextEdit)):
            self.rescale()
            event.accept()
            return
        super().keyPressEvent(event)

################################################################################
################################### Launcher ###################################
################################################################################
def run_vna_view(root, k = None, title = None, ui_scale = 1.0):
    """
    Open the view of a saved VNA sweep and wait until the window is
    closed.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.
    k (int or None): sweep index. None (default): the latest.
    title (str or None): window title. None (default): 'VNA sweep:
        <store>: vna<k>'.
    ui_scale (float): font size multiplier. Default 1.0.

    Returns:
    k (int): the index of the sweep shown.

    Raises:
    KeyError: if the store has no such sweep.
    """
    f, z, bands, info = load_vna_view(root, k)
    k = dl.latest_index(dl.open_store(root), 'vna') if k is None else int(k)
    if title is None:
        title = f"VNA sweep: {info.split(';')[0]}"
    app = get_qapp(title)
    win = VnaViewWindow(f, z, bands = bands, info = info, title = title,
                        ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    # exec() is the modern name (PyQt6+); exec_() is kept for PyQt5.
    (getattr(app, 'exec', None) or app.exec_)()
    return k
