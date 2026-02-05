"""
Automatic peak finder with adjustable parameters for VNA sweep data.

This module provides an interactive interface for automatically finding
resonances in VNA data using adjustable filtering and peak detection
parameters. The output can be used as initial guesses for the manual
peak finder.
"""

import numpy as np
import h5py
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from scipy.signal import butter, filtfilt, find_peaks
import os


def run_auto_peak_finder(f, z, outpath, overwrite = False):
    """
    Run the automatic peak finder.
    
    Parameters:
    f (np.ndarray): Frequency data in Hz.
    z (np.ndarray): Complex S21 data.
    outpath (str): Path to save results (.h5 file).
    overwrite (bool): If False, raise error if output file already
        exists. Default is True.
        
    Returns:
    fres (np.ndarray): Found resonant frequencies.
    """
    finder = AutoPeakFinder(f, z, outpath, overwrite)
    finder.run()
    return np.array(finder.fres)


class SpinBoxEventFilter(QtCore.QObject):
    """
    Event filter to select all text when spinbox gains focus.
    """
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.FocusIn:
            if hasattr(obj, 'lineEdit'):
                QtCore.QTimer.singleShot(0, obj.lineEdit().selectAll)
        return False


class AutoPeakFinder:
    def __init__(self, f, z, outpath, overwrite = True):
        """
        Automatic peak finder for VNA sweep data.
        
        Parameters:
        f (np.ndarray): Frequency data in Hz (1D array).
        z (np.ndarray): Complex S21 data (1D array).
        outpath (str): Path to save results (.h5 file).
        overwrite (bool): If False, raise error if output file already
            exists. Default is True.

        Returns:
        None
        """
        self.f = np.asarray(f, dtype = np.float64)
        self.z = np.asarray(z, dtype = np.complex128)
        self.outpath = os.path.normpath(os.path.expanduser(outpath))
        
        # Check if file exists
        if os.path.exists(self.outpath):
            if not overwrite:
                msg = (f'Output file {self.outpath} already exists. '
                       f'Set overwrite=True to overwrite.')
                raise FileExistsError(msg)
            else:
                msg = (f'Warning: {self.outpath} already exists and '
                       f'will be overwritten on save.')
                print(msg)
        
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
        
        # Setup the application
        self.app = pg.mkQApp("Auto Peak Finder")
        self.setup_ui()
        self.setup_plot()
        
        try:
            self.update_peaks()
        except Exception as e:
            self.win.close()
            raise
        
    def setup_ui(self):
        """
        Setup the main window and layout.

        Parameters:
        None

        Returns:
        None
        """
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('Auto Peak Finder - Press H for help')
        self.win.resize(1400, 800)
        
        # Central widget with splitter
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
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
        Setup parameter control widgets.

        Parameters:
        layout (QVBoxLayout): Layout to add controls to.

        Returns:
        None
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
        
        save_button = QtWidgets.QPushButton('Save (S)')
        save_button.clicked.connect(self.save_data)
        button_layout.addWidget(save_button)
        
        quit_button = QtWidgets.QPushButton('Save && Quit')
        quit_button.clicked.connect(self.quit_and_save)
        button_layout.addWidget(quit_button)
        
        layout.addLayout(button_layout)
        
        # Spacer
        layout.addStretch()
        
        # Setup keyboard shortcuts
        self.save_action = QtGui.QShortcut(QtCore.Qt.Key_S, self.win)
        self.save_action.activated.connect(self.save_data)
        
        self.help_action = QtGui.QShortcut(QtCore.Qt.Key_H, self.win)
        self.help_action.activated.connect(self.show_help)
        
    def update_smoothing_controls(self):
        """
        Show/hide smoothing controls based on selected method.

        Parameters:
        None

        Returns:
        None
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
        Called when smoothing method is changed.

        Parameters:
        None

        Returns:
        None
        """
        self.update_smoothing_controls()
        self.on_param_changed()
        
    def setup_plot(self):
        """
        Setup the magnitude plots.

        Parameters:
        None

        Returns:
        None
        """
        # Add title
        title = ('<span style="color: #FFF; font-size: 10pt;">'
                 '<b>Controls:</b> Z/X: Pan | S: Save | H: Help'
                 '</span>')
        self.plot_layout.addLabel(title, col = 0)
        self.plot_layout.nextRow()
        
        # Filtered magnitude plot (top)
        self.plot_filtered = self.plot_layout.addPlot(row = 1, col = 0)
        self.plot_filtered.setLabel('left', 'Filtered |S21| (dB)')
        self.plot_filtered.showGrid(x = True, y = True, alpha = 0.3)
        self.plot_filtered.getAxis('bottom').setStyle(showValues = False)
        
        # Plot filtered data
        self.filtered_curve = self.plot_filtered.plot(
            self.f, self.filtered_mag,
            pen = pg.mkPen(color = (100, 200, 255), width = 1.5),
            name = 'Filtered'
        )
        
        # Peak markers for filtered plot
        self.peak_markers_filtered = []
        
        # Connect range change for auto-scaling
        self.plot_filtered.sigRangeChanged.connect(
            lambda: self.auto_scale_y(self.plot_filtered, self.filtered_mag)
        )
        
        self.plot_layout.nextRow()
        
        # Original magnitude plot (bottom)
        self.plot_original = self.plot_layout.addPlot(row = 2, col = 0)
        self.plot_original.setLabel('left', 'Original |S21| (dB)')
        self.plot_original.setLabel('bottom', 'Frequency', units = 'Hz')
        self.plot_original.showGrid(x = True, y = True, alpha = 0.3)
        
        # Plot original data
        self.mag_curve = self.plot_original.plot(
            self.f, self.mag_db,
            pen = pg.mkPen(color = (150, 150, 150), width = 1),
            name = 'Original'
        )
        
        # Peak markers for original plot
        self.peak_markers_original = []
        
        # Connect range change for auto-scaling
        self.plot_original.sigRangeChanged.connect(
            lambda: self.auto_scale_y(self.plot_original, self.mag_db)
        )
        
        # Link x-axes
        self.plot_original.setXLink(self.plot_filtered)
        
        # Setup keyboard shortcuts for panning
        self.pan_left_action = QtGui.QShortcut(
            QtCore.Qt.Key_Z, self.win
        )
        self.pan_left_action.activated.connect(self.pan_left)
        
        self.pan_right_action = QtGui.QShortcut(
            QtCore.Qt.Key_X, self.win
        )
        self.pan_right_action.activated.connect(self.pan_right)
        
    def auto_scale_y(self, plot, data):
        """
        Auto-scale y-axis based on visible x-range.

        Parameters:
        plot (PlotItem): The plot to scale.
        data (np.ndarray): The data to scale based on.

        Returns:
        None
        """
        x_range = plot.viewRange()[0]
        x_min, x_max = x_range
        
        # Find data within visible range
        mask = (self.f >= x_min) & (self.f <= x_max)
        if not np.any(mask):
            return
        
        visible_data = data[mask]
        data_min = np.min(visible_data)
        data_max = np.max(visible_data)
        
        # Add margin
        data_range = data_max - data_min
        margin = data_range * self.margin_factor
        
        plot.setYRange(
            data_min - margin,
            data_max + margin,
            padding = 0
        )
        
    def on_param_changed(self):
        """
        Called when any parameter is changed.

        Parameters:
        None

        Returns:
        None
        """
        # Show updating status
        self.status_label.setText(
            '<span style="color: orange;">Updating...</span>'
        )
        self.win.setCursor(QtCore.Qt.WaitCursor)
        
        # Process events to update UI
        QtWidgets.QApplication.processEvents()
        
        # Update parameters
        self.params['f_min'] = self.f_min_spin.value() * 1e6
        self.params['f_max'] = self.f_max_spin.value() * 1e6
        self.params['highpass_mhz'] = self.highpass_spin.value()
        self.params['poly_order'] = self.poly_order_spin.value()
        self.params['height'] = -self.height_spin.value()
        self.params['width'] = self.width_spin.value()
        self.params['distance'] = self.distance_spin.value()
        self.params['smoothing'] = self.smooth_combo.currentText()
        
        # Update peaks
        self.update_peaks()
        
        # Restore status
        self.status_label.setText(
            '<span style="color: green;">Ready</span>'
        )
        self.win.setCursor(QtCore.Qt.ArrowCursor)
        
    def apply_smoothing(self):
        """
        Apply smoothing to magnitude data.

        Parameters:
        None

        Returns:
        None
        """
        if self.params['smoothing'] == 'highpass':
            # High-pass filter with cutoff in MHz
            # Removes variations slower than cutoff across frequency sweep
            cutoff_hz = self.params['highpass_mhz'] * 1e6
            
            # Frequency spacing between samples
            df = np.median(np.diff(self.f))
            
            # Convert cutoff to normalized frequency
            # A feature of scale cutoff_hz spans (cutoff_hz/df) samples
            # Its frequency is 1/(cutoff_hz/df) = df/cutoff_hz cycles/sample
            # Nyquist is 0.5 cycles/sample, so normalized cutoff is:
            cutoff_norm = (df / cutoff_hz) / 0.5
            
            # Clamp to valid range
            cutoff_norm = np.clip(cutoff_norm, 0.0001, 0.99)
            
            b, a = butter(3, cutoff_norm, btype = 'high')
            self.filtered_mag = filtfilt(b, a, self.mag_db)
            
        elif self.params['smoothing'] == 'polynomial':
            # Polynomial baseline subtraction
            order = self.params['poly_order']
            
            # Fit polynomial to data
            coeffs = np.polyfit(self.f, self.mag_db, order)
            baseline = np.polyval(coeffs, self.f)
            
            # Subtract baseline
            self.filtered_mag = self.mag_db - baseline
            
        else:  # 'none'
            self.filtered_mag = self.mag_db.copy()
            
    def update_peaks(self):
        """
        Update peak detection and plot.

        Parameters:
        None

        Returns:
        None
        """
        # Apply smoothing
        self.apply_smoothing()
        
        # Update filtered curve
        self.filtered_curve.setData(self.f, self.filtered_mag)
        
        # Force plot update
        self.plot_filtered.update()
        
        # Trigger auto-scaling
        self.auto_scale_y(self.plot_filtered, self.filtered_mag)
        self.auto_scale_y(self.plot_original, self.mag_db)
        
        # Mask for frequency range
        mask = (self.f >= self.params['f_min']) & \
               (self.f <= self.params['f_max'])
        
        f_masked = self.f[mask]
        mag_masked = self.filtered_mag[mask]
        
        if len(f_masked) == 0:
            self.fres = []
            self.update_markers()
            return
        
        # Calculate width and distance in samples
        # Width and distance are in kHz/GHz
        # Convert: (kHz/GHz) * (GHz) = kHz, then * 1e3 = Hz
        f_center = np.mean([self.params['f_min'], self.params['f_max']])
        f_center_ghz = f_center / 1e9
        width_hz = self.params['width'] * f_center_ghz * 1e3
        distance_hz = self.params['distance'] * f_center_ghz * 1e3
        
        # Convert to samples
        df = np.median(np.diff(f_masked))
        width_samples = max(1, int(width_hz / df))
        distance_samples = max(1, int(distance_hz / df))
        
        # Find peaks (looking for dips, so invert)
        peaks, properties = find_peaks(
            -mag_masked,
            height = -self.params['height'],
            width = width_samples,
            distance = distance_samples
        )
        
        # Convert to frequencies
        self.fres = f_masked[peaks].tolist()
        
        # Update markers
        self.update_markers()
        
        # Update info
        self.info_label.setText(f'Peaks found: {len(self.fres)}')
        
    def update_markers(self):
        """
        Update peak markers on plots.

        Parameters:
        None

        Returns:
        None
        """
        # Remove old markers from filtered plot
        for marker in self.peak_markers_filtered:
            self.plot_filtered.removeItem(marker)
        self.peak_markers_filtered = []
        
        # Remove old markers from original plot
        for marker in self.peak_markers_original:
            self.plot_original.removeItem(marker)
        self.peak_markers_original = []
        
        # Add new markers to both plots
        for i, freq in enumerate(self.fres):
            if i % 2 == 0:
                color = (255, 0, 0, 200)
            else:
                color = (255, 255, 0, 200)
            
            # Marker for filtered plot
            marker_filtered = pg.InfiniteLine(
                pos = freq,
                angle = 90,
                pen = pg.mkPen(
                    color = color,
                    width = 2,
                    style = QtCore.Qt.DashLine
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
                    style = QtCore.Qt.DashLine
                )
            )
            self.plot_original.addItem(marker_original)
            self.peak_markers_original.append(marker_original)
            
    def pan_left(self):
        """
        Pan the view to the left by 20% of current width.

        Parameters:
        None

        Returns:
        None
        """
        x_range = self.plot_filtered.viewRange()[0]
        width = x_range[1] - x_range[0]
        shift = -0.2 * width
        self.plot_filtered.setXRange(
            x_range[0] + shift,
            x_range[1] + shift,
            padding = 0
        )
        
    def pan_right(self):
        """
        Pan the view to the right by 20% of current width.

        Parameters:
        None

        Returns:
        None
        """
        x_range = self.plot_filtered.viewRange()[0]
        width = x_range[1] - x_range[0]
        shift = 0.2 * width
        self.plot_filtered.setXRange(
            x_range[0] + shift,
            x_range[1] + shift,
            padding = 0
        )
        
    def save_data(self):
        """
        Save resonances and parameters to HDF5 file.

        Parameters:
        None

        Returns:
        None
        """
        fres_array = np.array(self.fres, dtype = np.float64)
        
        with h5py.File(self.outpath, 'w') as hf:
            hf.create_dataset('fres', data = fres_array)
            # Save parameters
            for key, val in self.params.items():
                hf.attrs[key] = val
        
        print(f"Saved {len(self.fres)} resonances to {self.outpath}")
        
    def quit_and_save(self):
        """
        Save data and close the application.

        Parameters:
        None

        Returns:
        None
        """
        self.save_data()
        self.win.close()
        self.app.quit()
        
    def show_help(self):
        """
        Display help dialog.

        Parameters:
        None

        Returns:
        None
        """
        help_text = """
        <h3>Auto Peak Finder Help</h3>
        <p><b>Controls:</b></p>
        <ul>
        <li>Adjust parameters in right panel to tune peak detection</li>
        <li><b>Z/X:</b> Pan left/right</li>
        <li><b>Mouse Wheel:</b> Zoom</li>
        <li><b>S:</b> Save results</li>
        <li><b>H:</b> Show this help</li>
        </ul>
        <p><b>Parameters:</b></p>
        <ul>
        <li><b>Frequency Range:</b> Limit search to this range</li>
        <li><b>Smoothing Method:</b></li>
        <ul>
        <li><b>highpass:</b> Remove slow baseline drift (cutoff in MHz)</li>
        <li><b>polynomial:</b> Subtract polynomial baseline (order)</li>
        <li><b>none:</b> No smoothing</li>
        </ul>
        <li><b>Height:</b> Minimum peak depth (dB below baseline)</li>
        <li><b>Width:</b> Min peak width (kHz/GHz)</li>
        <li><b>Distance:</b> Min spacing (kHz/GHz)</li>
        </ul>
        <p><b>Status:</b> {0} peaks found</p>
        <p><b>Output:</b> {1}</p>
        """.format(len(self.fres), self.outpath)
        
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Auto Peak Finder Help")
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setText(help_text)
        msg.exec_()
        
    def run(self):
        """
        Start the application event loop.

        Parameters:
        None

        Returns:
        None
        """
        print("Starting Auto Peak Finder...")
        print(f"Output file: {self.outpath}")
        print("Press 'H' for help")
        
        self.app.exec_()

