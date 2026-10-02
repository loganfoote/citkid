"""
Automatic resonance finder with adjustable parameters for VNA sweep data.

This module provides an interactive interface for automatically finding
resonances in VNA data using adjustable filtering and peak detection
parameters. The output can be used as initial guesses for the manual
resonance finder.
"""

import numpy as np
import zarr
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from scipy.signal import find_peaks
import os
from .s21_filt import highpass_filter, polynomial_baseline
from ..qt_compat import (
    Qt as _Qt,
    delete_on_close,
    fit_window_to_screen,
    get_qapp,
    responsive_busy,
    run_responsive,
)

def run_res_finder_auto(f, z, zarr_grp):
    """
    Run the automatic res finder.

    If data already exists (a file at the ``zarr_grp`` path, or 'fres_auto'
    in the group), a popup asks whether to overwrite it or cancel. Existing
    results are never loaded.

    Parameters:
    f (np.ndarray): Frequency data in Hz.
    z (np.ndarray): Complex S21 data.
    zarr_grp (zarr.Group or str): Zarr group or path to save results.
        The resonances are saved as 'fres_auto' in the group.

    Returns:
    fres (np.ndarray): Found resonant frequencies.

    Raises:
    RuntimeError: if the user cancels the overwrite popup.
    """
    finder = AutoResFinder(f, z, zarr_grp)
    finder.run()
    return np.array(finder.fres)


def _confirm_overwrite(message):
    """
    Ask whether to overwrite existing data or cancel.

    Parameters:
    message (str): description of the existing data, shown in the popup.

    Returns:
    overwrite (bool): True if the user chose Overwrite, False if they chose
        Cancel or closed the popup.
    """
    get_qapp("Auto Res Finder")
    box = QtWidgets.QMessageBox()
    box.setWindowTitle('Existing Data Found')
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setText(message + '\n\nOverwrite it, or cancel?')
    overwrite_btn = box.addButton(
        'Overwrite', QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = box.addButton(
        'Cancel', QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel_btn)
    (getattr(box, 'exec', None) or box.exec_)()
    return box.clickedButton() is overwrite_btn


class SpinBoxEventFilter(QtCore.QObject):
    """
    Event filter to select all text when spinbox gains focus.
    """
    def eventFilter(self, obj, event):
        """
        Select all spinbox text when the watched widget gains focus.

        Parameters:
        obj (QObject): watched object that received the event.
        event (QEvent): event delivered to ``obj``.

        Returns:
        handled (bool): always False, so the event is passed on for normal
            processing.
        """
        if event.type() == QtCore.QEvent.Type.FocusIn:
            if hasattr(obj, 'lineEdit'):
                QtCore.QTimer.singleShot(0, obj.lineEdit().selectAll)
        return False


class AutoResFinderWindow(QtWidgets.QMainWindow):
    """
    Custom QMainWindow that saves data when window is closed.
    """
    def __init__(self, finder=None, *args, **kwargs):
        """
        Initialize the main window.

        Parameters:
        finder (AutoResFinder or None): finder whose ``save_data`` is called
            when the window closes. If None (default), nothing is saved.
        *args (tuple): positional arguments passed to ``QMainWindow``.
        **kwargs (dict): keyword arguments passed to ``QMainWindow``.
        """
        super().__init__(*args, **kwargs)
        delete_on_close(self)  # destroy on the GUI thread when closed
        self.finder = finder
    
    def closeEvent(self, event):
        """
        Handle window close event by saving data.

        Parameters:
        event (QCloseEvent): Qt close event.
        """
        if responsive_busy():
            event.ignore()  # work in progress; close when it has finished
            return
        if self.finder is not None:
            self.finder.save_data()
        super().closeEvent(event)


class AutoResFinder:
    """
    Interactive GUI for automatically finding resonances in VNA sweep data.

    The magnitude data is smoothed (highpass, polynomial baseline, or none),
    and dips are found with ``scipy.signal.find_peaks`` using parameters
    adjustable in the control panel. Results are saved as 'fres_auto' in a
    zarr group.
    """
    def __init__(self, f, z, zarr_grp):
        """
        Initialize the automatic res finder for VNA sweep data.

        If a regular file already exists at the ``zarr_grp`` path, or the
        group already contains 'fres_auto', a popup asks whether to overwrite
        it or cancel. Overwriting deletes the file or 'fres_auto' right away;
        existing results are never loaded.

        Parameters:
        f (np.ndarray): Frequency data in Hz (1D array).
        z (np.ndarray): Complex S21 data (1D array).
        zarr_grp (zarr.Group or str): Zarr group or path to save results.
            The resonances are saved as 'fres_auto' in the group. A path
            must end in '.h5' and its parent directory must exist.

        Raises:
        ValueError: if ``zarr_grp`` is a path without an '.h5' extension.
        FileNotFoundError: if the parent directory of the path does not
            exist.
        RuntimeError: if the user cancels the overwrite popup.
        """
        self.f = np.asarray(f, dtype = np.float64)
        self.z = np.asarray(z, dtype = np.complex128)

        # Resolve zarr_grp to a zarr group
        if isinstance(zarr_grp, (str, os.PathLike)):
            zarr_path = os.path.expanduser(str(zarr_grp))
            
            # Check for .h5 extension
            if not zarr_path.endswith('.h5'):
                raise ValueError("Output path must have .h5 extension")
            
            # Check if parent directory exists
            parent_dir = os.path.dirname(zarr_path)
            if parent_dir and not os.path.isdir(parent_dir):
                raise FileNotFoundError("Output directory does not exist")
            
            # A regular file at the path blocks the zarr directory store
            if os.path.isfile(zarr_path):
                if not _confirm_overwrite(f"A file already exists at\n{zarr_path}"):
                    raise RuntimeError("User cancelled operation")
                os.remove(zarr_path)

            self.zarr_group = zarr.open_group(zarr_path, mode = 'a')
        else:
            self.zarr_group = zarr_grp

        # Existing results are overwritten (never loaded) if the user agrees
        if 'fres_auto' in self.zarr_group:
            if not _confirm_overwrite(
                    "The zarr group already contains 'fres_auto'."):
                raise RuntimeError("User cancelled operation")
            del self.zarr_group['fres_auto']
        
        # Compute magnitude
        self.mag_db = 20 * np.log10(np.abs(self.z))
        
        # Default parameters
        self.params = {
            'f_min': float(np.min(self.f)),
            'f_max': float(np.max(self.f)),
            'highpass_mhz': 10.0,  # HP cutoff (MHz)
            'poly_order': 3,  # Polynomial baseline order
            'height': -5.0,  # dB below baseline
            'width': 10,  # Min width (kHz/GHz)
            'distance': 10,  # Min spacing (kHz/GHz)
            'smoothing': 'highpass',  # smoothing method
        }
        
        self.margin_factor = 0.15
        
        self.fres = []
        self.filtered_mag = self.mag_db.copy()

        # Event filter for select-all behavior
        self.event_filter = SpinBoxEventFilter()

        # Debounce timers — created lazily on first use
        self._range_timer = None
        self._param_timer = None

        # Setup the application
        self.app = get_qapp("Auto Res Finder")
        self.setup_ui()
        self.setup_plot()

        # Initial view will be set in setup_plot after plot_filtered is created

        try:
            self.update_peaks()
        except Exception as e:
            self.win.close()
            raise
        
    def setup_ui(self):
        """
        Set up the main window and layout, and show the window.
        """
        self.win = AutoResFinderWindow(finder=self)
        self.win.setWindowTitle('Auto Resonance Finder - Press H for help')
        # Preferred size, shrunk to fit smaller screens, centred
        fit_window_to_screen(self.win, frac=0.9, size=(1400, 800))
        
        # Central widget with splitter
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        
        splitter = QtWidgets.QSplitter(_Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: plot
        plot_widget = pg.GraphicsLayoutWidget()
        self.plot_layout = plot_widget
        splitter.addWidget(plot_widget)
        
        # Right side: controls
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        splitter.addWidget(control_widget)
        
        # Set splitter sizes (80% plot, 20% controls)
        splitter.setSizes([800, 200])
        
        # Add controls
        self.setup_controls(control_layout)
        
        self.win.show()
        
    def setup_controls(self, layout):
        """
        Set up parameter control widgets.

        Parameters:
        layout (QVBoxLayout): Layout to add controls to.
        """
        # Title
        title = QtWidgets.QLabel('<b>Peak Finding Parameters</b>')
        layout.addWidget(title)
        
        # Frequency range
        freq_group = QtWidgets.QGroupBox('Frequency Range')
        freq_layout = QtWidgets.QFormLayout()
        
        self.f_min_spin = QtWidgets.QDoubleSpinBox()
        self.f_min_spin.setRange(np.min(self.f) / 1e6, np.max(self.f) / 1e6)
        self.f_min_spin.setValue(self.params['f_min'] / 1e6)
        self.f_min_spin.setDecimals(3)
        self.f_min_spin.setSingleStep(1.0)
        self.f_min_spin.valueChanged.connect(self.on_param_changed)
        self.f_min_spin.lineEdit().selectAll()
        self.f_min_spin.installEventFilter(self.event_filter)
        
        self.f_max_spin = QtWidgets.QDoubleSpinBox()
        self.f_max_spin.setRange(np.min(self.f) / 1e6, np.max(self.f) / 1e6)
        self.f_max_spin.setValue(self.params['f_max'] / 1e6)
        self.f_max_spin.setDecimals(3)
        self.f_max_spin.setSingleStep(1.0)
        self.f_max_spin.valueChanged.connect(self.on_param_changed)
        self.f_max_spin.lineEdit().selectAll()
        self.f_max_spin.installEventFilter(self.event_filter)
        
        freq_layout.addRow('Min Freq (MHz):', self.f_min_spin)
        freq_layout.addRow('Max Freq (MHz):', self.f_max_spin)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        # Smoothing
        smooth_group = QtWidgets.QGroupBox('Smoothing')
        smooth_layout = QtWidgets.QFormLayout()
        
        self.smooth_combo = QtWidgets.QComboBox()
        self.smooth_combo.addItems(
            ['highpass', 'polynomial', 'none']
        )
        self.smooth_combo.setCurrentText(self.params['smoothing'])
        self.smooth_combo.currentTextChanged.connect(self.on_smoothing_changed)
        
        # Highpass parameters
        self.highpass_label = QtWidgets.QLabel('HP Cutoff (MHz):')
        self.highpass_spin = QtWidgets.QDoubleSpinBox()
        self.highpass_spin.setRange(0.1, 1000)
        self.highpass_spin.setValue(self.params['highpass_mhz'])
        self.highpass_spin.setDecimals(1)
        self.highpass_spin.setSingleStep(1.0)
        self.highpass_spin.valueChanged.connect(self.on_param_changed)
        self.highpass_spin.lineEdit().selectAll()
        self.highpass_spin.installEventFilter(self.event_filter)
        
        # Polynomial baseline parameters
        self.poly_order_label = QtWidgets.QLabel('Poly Order:')
        self.poly_order_spin = QtWidgets.QSpinBox()
        self.poly_order_spin.setRange(1, 10)
        self.poly_order_spin.setValue(self.params['poly_order'])
        self.poly_order_spin.valueChanged.connect(self.on_param_changed)
        self.poly_order_spin.lineEdit().selectAll()
        self.poly_order_spin.installEventFilter(self.event_filter)
        
        smooth_layout.addRow('Method:', self.smooth_combo)
        smooth_layout.addRow(self.highpass_label, self.highpass_spin)
        smooth_layout.addRow(self.poly_order_label, self.poly_order_spin)
        smooth_group.setLayout(smooth_layout)
        layout.addWidget(smooth_group)
        
        # Initially show/hide appropriate controls
        self.update_smoothing_controls()
        
        # Peak detection
        peak_group = QtWidgets.QGroupBox('Peak Detection')
        peak_layout = QtWidgets.QFormLayout()
        
        self.height_spin = QtWidgets.QDoubleSpinBox()
        self.height_spin.setRange(0.1, 50)
        self.height_spin.setValue(-self.params['height'])
        self.height_spin.setDecimals(1)
        self.height_spin.setSingleStep(0.5)
        self.height_spin.valueChanged.connect(self.on_param_changed)
        self.height_spin.lineEdit().selectAll()
        self.height_spin.installEventFilter(self.event_filter)
        
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(self.params['width'])
        self.width_spin.setSingleStep(10)
        self.width_spin.valueChanged.connect(self.on_param_changed)
        self.width_spin.lineEdit().selectAll()
        self.width_spin.installEventFilter(self.event_filter)
        
        self.distance_spin = QtWidgets.QSpinBox()
        self.distance_spin.setRange(1, 5000)
        self.distance_spin.setValue(self.params['distance'])
        self.distance_spin.setSingleStep(10)
        self.distance_spin.valueChanged.connect(self.on_param_changed)
        self.distance_spin.lineEdit().selectAll()
        self.distance_spin.installEventFilter(self.event_filter)
        
        peak_layout.addRow('Height (dB):', self.height_spin)
        peak_layout.addRow('Width (kHz/GHz):', self.width_spin)
        peak_layout.addRow('Distance (kHz/GHz):', self.distance_spin)
        peak_group.setLayout(peak_layout)
        layout.addWidget(peak_group)
        
        # Status and info labels
        self.status_label = QtWidgets.QLabel(
            '<span style="color: green;">Ready</span>'
        )
        layout.addWidget(self.status_label)
        
        self.info_label = QtWidgets.QLabel('Peaks found: 0')
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        save_button = QtWidgets.QPushButton('Save (Ctrl+S)')
        save_button.clicked.connect(self.save_data)
        button_layout.addWidget(save_button)
        
        quit_button = QtWidgets.QPushButton('Save && Quit')
        quit_button.clicked.connect(self.quit_and_save)
        button_layout.addWidget(quit_button)
        
        layout.addLayout(button_layout)
        
        # Spacer
        layout.addStretch()
        
        # Setup keyboard shortcuts (use string sequences for portability)
        self.save_action = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self.win)
        self.save_action.activated.connect(self.save_data)

        self.help_action = QtGui.QShortcut(QtGui.QKeySequence("H"), self.win)
        self.help_action.activated.connect(self.show_help)
        
    def update_smoothing_controls(self):
        """
        Show/hide smoothing controls based on selected method.
        """
        method = self.smooth_combo.currentText()
        
        # Hide all controls first
        self.highpass_label.setVisible(False)
        self.highpass_spin.setVisible(False)
        self.poly_order_label.setVisible(False)
        self.poly_order_spin.setVisible(False)
        
        # Show relevant controls
        if method == 'highpass':
            self.highpass_label.setVisible(True)
            self.highpass_spin.setVisible(True)
        elif method == 'polynomial':
            self.poly_order_label.setVisible(True)
            self.poly_order_spin.setVisible(True)
        # 'none' shows no extra controls
        
    def on_smoothing_changed(self):
        """
        Update the smoothing controls and parameters when the smoothing
        method changes.
        """
        self.update_smoothing_controls()
        self.on_param_changed()
        
    def setup_plot(self):
        """
        Set up the magnitude plots and keyboard shortcuts.
        """
        # Add title
        title = ('<span style="color: #FFF; font-size: 10pt;">'
                 '<b>Controls:</b> Z/X: Pan 20% | A/S: Pan 80% | Ctrl+S: Save | H: Help'
                 '</span>')
        self.plot_layout.addLabel(title, col = 0)
        self.plot_layout.nextRow()
        
        # Filtered magnitude plot (top)
        self.plot_filtered = self.plot_layout.addPlot(row = 1, col = 0)
        self.plot_filtered.setLabel('left', 'Filtered |S21| (dB)')
        self.plot_filtered.showGrid(x = True, y = True, alpha = 0.3)
        self.plot_filtered.getAxis('bottom').setStyle(showValues = False)
        
        # Plot filtered data — initialised empty; _update_curves pushes
        # only the visible slice on every range-change, so pyqtgraph never
        # processes the full 1e6-point array during interactive use.
        self.filtered_curve = self.plot_filtered.plot(
            [], [],
            pen = pg.mkPen(color = (100, 200, 255), width = 1.5),
            name = 'Filtered'
        )
        
        # Peak markers for filtered plot
        self.peak_markers_filtered = []
        
        # Connect range change — single debounced handler covers both plots
        # (plot_original shares the x-axis via setXLink).
        self.plot_filtered.sigRangeChanged.connect(self.on_range_changed)
        
        self.plot_layout.nextRow()
        
        # Original magnitude plot (bottom)
        self.plot_original = self.plot_layout.addPlot(row = 2, col = 0)
        self.plot_original.setLabel('left', 'Original |S21| (dB)')
        self.plot_original.setLabel('bottom', 'Frequency', units = 'Hz')
        self.plot_original.showGrid(x = True, y = True, alpha = 0.3)
        
        # Plot original data — initialised empty; populated by _update_curves
        self.mag_curve = self.plot_original.plot(
            [], [],
            pen = pg.mkPen(color = (150, 150, 150), width = 1),
            name = 'Original'
        )
        
        # Peak markers for original plot
        self.peak_markers_original = []
        
        # No separate sigRangeChanged needed — plots share the x-axis
        # (setXLink) so the single debounced handler on plot_filtered covers both.
        
        # Link x-axes
        self.plot_original.setXLink(self.plot_filtered)
        
        # Set initial view so _update_curves can populate the curves on
        # the first update_peaks call.
        self.plot_filtered.setXRange(
            float(self.f[0]), float(self.f[-1]), padding = 0.02
        )
        
        # Setup keyboard shortcuts (after plots are created)
        self.setup_shortcuts()
        
    def auto_scale_y(self, plot, data):
        """
        Auto-scale y-axis based on visible x-range.

        Parameters:
        plot (PlotItem): The plot to scale.
        data (np.ndarray): The data to scale based on, same length as
            ``self.f``.
        """
        x_min, x_max = plot.viewRange()[0]

        # Binary search — O(log n) regardless of array length
        lo = int(np.searchsorted(self.f, x_min, side = 'left'))
        hi = int(np.searchsorted(self.f, x_max, side = 'right'))
        if lo >= hi:
            return

        visible_data = data[lo:hi]
        data_min = float(visible_data.min())
        data_max = float(visible_data.max())

        # Add margin
        data_range = data_max - data_min
        margin = data_range * self.margin_factor

        plot.setYRange(
            data_min - margin,
            data_max + margin,
            padding = 0
        )

    def _visible_slice(self, x_min, x_max):
        """
        Return a slice of the sorted frequency array covering [x_min, x_max].

        Uses binary search — O(log n) regardless of array length.

        Parameters:
        x_min (float): Left edge of range in Hz.
        x_max (float): Right edge of range in Hz.

        Returns:
        sl (slice): Slice into self.f / self.mag_db / self.filtered_mag.
        """
        lo = int(np.searchsorted(self.f, x_min, side = 'left'))
        hi = int(np.searchsorted(self.f, x_max, side = 'right'))
        return slice(lo, hi)

    def _update_curves(self):
        """
        Push only the visible data slice to the filtered and original curves.

        Using a 50% pad on each side prevents blank edges during fast
        panning.  pyqtgraph then only renders the visible ~1,000 points
        instead of the full 1e6-point dataset.
        """
        if not hasattr(self, 'plot_filtered'):
            return
        x_min, x_max = self.plot_filtered.viewRange()[0]
        span = x_max - x_min
        sl = self._visible_slice(x_min - 0.5 * span, x_max + 0.5 * span)
        if sl.start >= sl.stop:
            return
        self.filtered_curve.setData(self.f[sl], self.filtered_mag[sl])
        self.mag_curve.setData(self.f[sl], self.mag_db[sl])

    def on_range_changed(self):
        """
        Schedule a plot update when the view range changes (pan/zoom).

        Debounced: coalesces rapid-fire events into a single update every
        50 ms so the UI stays responsive during continuous mouse drag.
        """
        if self._range_timer is None:
            self._range_timer = QtCore.QTimer()
            self._range_timer.setSingleShot(True)
            self._range_timer.timeout.connect(self._do_range_update)
        self._range_timer.start(50)

    def _do_range_update(self):
        """
        Redraw the curves and rescale y after the range debounce delay.
        """
        self._update_curves()
        self.auto_scale_y(self.plot_filtered, self.filtered_mag)
        self.auto_scale_y(self.plot_original, self.mag_db)
        
    def on_param_changed(self):
        """
        Read the parameter controls and schedule a peak-detection update.

        Debounced: the peak detection runs 300 ms after the last change.
        """
        # Capture the latest spinbox values immediately so rapid changes
        # are not lost while the debounce timer is pending.
        self.params['f_min'] = self.f_min_spin.value() * 1e6
        self.params['f_max'] = self.f_max_spin.value() * 1e6
        self.params['highpass_mhz'] = self.highpass_spin.value()
        self.params['poly_order'] = self.poly_order_spin.value()
        self.params['height'] = -self.height_spin.value()
        self.params['width'] = self.width_spin.value()
        self.params['distance'] = self.distance_spin.value()
        self.params['smoothing'] = self.smooth_combo.currentText()

        self.status_label.setText(
            '<span style="color: orange;">Updating...</span>'
        )

        # Debounce the expensive smoothing + peak-detection work (300 ms).
        if self._param_timer is None:
            self._param_timer = QtCore.QTimer()
            self._param_timer.setSingleShot(True)
            self._param_timer.timeout.connect(self._do_param_update)
        self._param_timer.start(300)

    def _do_param_update(self):
        """
        Rerun peak detection after the parameter debounce delay.
        """
        self.win.setCursor(_Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        self.update_peaks()
        self.status_label.setText(
            '<span style="color: green;">Ready</span>'
        )
        self.win.setCursor(_Qt.ArrowCursor)
        
    def apply_smoothing(self):
        """
        Apply smoothing to magnitude data and store it in
        ``self.filtered_mag``.

        The method is set by ``self.params['smoothing']``: 'highpass',
        'polynomial', or 'none' (copy of the unsmoothed data).
        """
        self.filtered_mag = self._smoothed(self.params)

    def _smoothed(self, params):
        """
        Return the smoothed magnitude data (no Qt calls).

        Parameters:
        params (dict): parameters, as ``self.params``.

        Returns:
        filtered_mag (np.ndarray): smoothed magnitude (dB).
        """
        if params['smoothing'] == 'highpass':
            return highpass_filter(self.f, self.mag_db, params['highpass_mhz'])
        if params['smoothing'] == 'polynomial':
            return polynomial_baseline(self.f, self.mag_db, params['poly_order'])
        return self.mag_db.copy()  # 'none'

    def _compute_peaks(self, params):
        """
        Smooth the data and find the resonance dips (no Qt calls, so it can
        run in a worker thread).

        Parameters:
        params (dict): parameters, as ``self.params``.

        Returns:
        filtered_mag (np.ndarray): smoothed magnitude (dB).
        fres (list of float or None): resonance frequencies (Hz), or None if
            no data is in the frequency range.
        """
        filtered_mag = self._smoothed(params)

        # Mask for frequency range
        mask = (self.f >= params['f_min']) & (self.f <= params['f_max'])
        f_masked = self.f[mask]
        mag_masked = filtered_mag[mask]
        if len(f_masked) == 0:
            return filtered_mag, None

        # Calculate width and distance in samples
        # Width and distance are in kHz/GHz
        # Convert: (kHz/GHz) * (GHz) = kHz, then * 1e3 = Hz
        f_center = np.mean([params['f_min'], params['f_max']])
        f_center_ghz = f_center / 1e9
        width_hz = params['width'] * f_center_ghz * 1e3
        distance_hz = params['distance'] * f_center_ghz * 1e3

        # Convert to samples
        df = np.median(np.diff(f_masked))
        width_samples = max(1, int(width_hz / df))
        distance_samples = max(1, int(distance_hz / df))

        # Find peaks (looking for dips, so invert)
        peaks, properties = find_peaks(
            -mag_masked,
            height = -params['height'],
            width = width_samples,
            distance = distance_samples
        )
        return filtered_mag, f_masked[peaks].tolist()

    def update_peaks(self):
        """
        Update peak detection and plot.

        Results are stored in ``self.fres``. The smoothing and peak finding
        run in a worker thread (``run_responsive``), so the window keeps
        responding on long sweeps; the plots are updated afterwards.
        """
        self.filtered_mag, fres = run_responsive(self._compute_peaks, dict(self.params))

        # Update curves with visible slice only
        self._update_curves()

        # Trigger auto-scaling
        self.auto_scale_y(self.plot_filtered, self.filtered_mag)
        self.auto_scale_y(self.plot_original, self.mag_db)

        if fres is None:  # no data in the frequency range
            self.fres = []
            self.update_markers()
            return
        self.fres = fres

        # Update markers
        self.update_markers()

        # Update info
        self.info_label.setText(f'Peaks found: {len(self.fres)}')

    def update_markers(self):
        """
        Update res markers on plots.
        """
        # Remove old markers from filtered plot
        for marker in self.peak_markers_filtered:
            self.plot_filtered.removeItem(marker)
        self.peak_markers_filtered = []
        
        # Remove old markers from original plot
        for marker in self.peak_markers_original:
            self.plot_original.removeItem(marker)
        self.peak_markers_original = []
        
        # Cycle through 6 distinct (color, line-style) combinations so
        # adjacent markers are always visually distinct.
        _styles = [
            ((255, 255,   0, 180), _Qt.SolidLine),  # yellow solid
            ((  0, 255, 255, 180), _Qt.DashLine),   # cyan   dash
            ((255, 100, 255, 180), _Qt.DotLine),    # magenta dot
            ((255, 255,   0, 180), _Qt.DashLine),   # yellow dash
            ((  0, 255, 255, 180), _Qt.DotLine),    # cyan   dot
            ((255, 100, 255, 180), _Qt.SolidLine),  # magenta solid
        ]

        # Add new markers to both plots
        for i, freq in enumerate(self.fres):
            color, line_style = _styles[i % len(_styles)]

            # Marker for filtered plot
            marker_filtered = pg.InfiniteLine(
                pos = freq,
                angle = 90,
                pen = pg.mkPen(
                    color = color,
                    width = 2,
                    style = line_style
                )
            )
            self.plot_filtered.addItem(marker_filtered)
            self.peak_markers_filtered.append(marker_filtered)
            
            # Marker for original plot
            marker_original = pg.InfiniteLine(
                pos = freq,
                angle = 90,
                pen = pg.mkPen(
                    color = color,
                    width = 2,
                    style = line_style
                )
            )
            self.plot_original.addItem(marker_original)
            self.peak_markers_original.append(marker_original)
            
    def save_data(self):
        """
        Save resonances to zarr group.

        The resonances are written to 'fres_auto' (overwriting any existing
        array), and ``self.params`` is saved as group attributes.
        """
        fres_array = np.array(self.fres, dtype = np.float64)
        params = dict(self.params)

        def write():
            """
            Write the resonances and parameters (no Qt calls).
            """
            self.zarr_group.create_array(
                'fres_auto',
                data = fres_array,
                overwrite = True
                )
            # Save parameters as group attributes, in one metadata write
            self.zarr_group.attrs.update(params)

        # In a worker thread, so the window keeps responding on slow drives
        run_responsive(write)
        print(f"Saved {len(self.fres)} resonances to zarr group")

    def quit_and_save(self):
        """
        Close the window, which saves the data and ends the event loop.

        This does exactly what the window's X button does: ``closeEvent``
        saves (once), and closing the last window ends ``app.exec()``.
        """
        self.win.close()
        
    def show_help(self):
        """
        Toggle the help panel on/off.
        """
        if not hasattr(self, '_help_dlg'):
            self._help_dlg = self._build_help_dialog()
        if self._help_dlg.isVisible():
            self._help_dlg.hide()
        else:
            dlg = self._help_dlg
            dlg.adjustSize()
            dlg.show()
            # Centre within the main window
            _parent_center = self.win.frameGeometry().center()
            _dlg_frame = dlg.frameGeometry()
            _dlg_frame.moveCenter(_parent_center)
            dlg.move(_dlg_frame.topLeft())

    def _build_help_dialog(self):
        """
        Create the floating help dialog (created lazily on first use).

        Returns:
        dlg (QDialog): help dialog, not yet shown.
        """
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle("Auto Resonance Finder Help (H to close)")
        layout = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel(
            "<h3>Auto Resonance Finder Controls</h3>"
            "<p><b>Controls:</b></p>"
            "<ul>"
            "<li>Adjust parameters in the right panel to tune resonance detection</li>"
            "<li><b>Z/X:</b> Pan left/right by 20%</li>"
            "<li><b>A/S:</b> Pan left/right by 80%</li>"
            "<li><b>Mouse Wheel:</b> Zoom in/out</li>"
            "<li><b>Ctrl+S:</b> Save results</li>"
            "<li><b>H:</b> Toggle this help panel</li>"
            "</ul>"
            "<p><b>Parameters:</b></p>"
            "<ul>"
            "<li><b>Frequency Range:</b> Limit peak search to this range</li>"
            "<li><b>Smoothing Method:</b></li>"
            "<ul>"
            "<li><b>highpass:</b> Remove slow baseline drift (cutoff in MHz)</li>"
            "<li><b>polynomial:</b> Subtract polynomial baseline (order)</li>"
            "<li><b>none:</b> No smoothing</li>"
            "</ul>"
            "<li><b>Height:</b> Minimum peak depth (dB below baseline)</li>"
            "<li><b>Width:</b> Minimum peak width (kHz/GHz)</li>"
            "<li><b>Distance:</b> Minimum spacing between peaks (kHz/GHz)</li>"
            "</ul>"
        )
        lbl.setTextFormat(_Qt.RichText)
        lbl.setWordWrap(False)
        layout.addWidget(lbl)
        close_btn = QtWidgets.QPushButton("Close (H)")
        close_btn.clicked.connect(dlg.hide)
        layout.addWidget(close_btn)
        _sc = QtGui.QShortcut(QtGui.QKeySequence("H"), dlg)
        _sc.activated.connect(dlg.hide)
        return dlg
    
    def setup_shortcuts(self):
        """
        Set up keyboard shortcuts for panning (Z/X: 20%, A/S: 80%).
        """
        # Pan shortcuts (20% pan by Z/X, 80% pan by A/S)
        self.pan_left_action = QtGui.QShortcut(QtGui.QKeySequence("Z"), self.win)
        self.pan_left_action.activated.connect(self.pan_left)
        self.pan_right_action = QtGui.QShortcut(QtGui.QKeySequence("X"), self.win)
        self.pan_right_action.activated.connect(self.pan_right)
        self.fast_pan_left_action = QtGui.QShortcut(QtGui.QKeySequence("A"), self.win)
        self.fast_pan_left_action.activated.connect(self.fast_pan_left)
        self.fast_pan_right_action = QtGui.QShortcut(QtGui.QKeySequence("S"), self.win)
        self.fast_pan_right_action.activated.connect(self.fast_pan_right)
    
    def _pan(self, fraction):
        """
        Shift the x-axis by *fraction* of the current view width.
        
        Parameters:
        fraction (float): Fraction of current view width to pan by
            (positive = right, negative = left).
        """
        x0, x1 = self.plot_filtered.viewRange()[0]
        shift = fraction * (x1 - x0)
        self.plot_filtered.setXRange(x0 + shift, x1 + shift, padding=0)
    
    def pan_left(self):
        """
        Pan the view to the left by 20% of current width (Z key).
        """
        self._pan(-0.2)
    
    def pan_right(self):
        """
        Pan the view to the right by 20% of current width (X key).
        """
        self._pan(0.2)
    
    def fast_pan_left(self):
        """
        Pan the view to the left by 80% of current width (A key).
        """
        self._pan(-0.8)
    
    def fast_pan_right(self):
        """
        Pan the view to the right by 80% of current width (S key).
        """
        self._pan(0.8)
        
    def run(self):
        """
        Start the application event loop.
        """
        print("Starting Auto Resonance Finder...")
        print(f"Output: {self.zarr_group.store}")
        print("Press 'H' for help")
        
        self.app.exec()

