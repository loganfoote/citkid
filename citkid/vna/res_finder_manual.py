"""
Interactive manual resonance finder for VNA sweep data.

This module provides a fast, responsive interface for manually identifying
resonances in VNA sweep data with automatic y-axis scaling, phase
visualization, and undo support.
"""

import numpy as np
import h5py
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import os


def run_res_finder_manual(f, z, fres_initial, outpath, margin_factor = 0.15,
                    overwrite = False):
    """
    Run the interactive manual resonance finder.
    
    Parameters:
    f (np.ndarray): Frequency data in Hz.
    z (np.ndarray): Complex S21 data.
    fres_initial (array-like or str): Initial resonant frequency
        guesses in Hz, or path to .h5 file containing 'fres' dataset.
    outpath (str): Path to save the resonance list (.h5 file).
    margin_factor (float): Y-axis margin factor for auto-scaling.
        Default is 0.15.
    overwrite (bool): If False, raise error if output file already
        exists. Default is False.
        
    Returns:
    fres (np.ndarray): Final list of resonant frequencies.
    """
    # Handle loading fres from file if string is provided
    if isinstance(fres_initial, str):
        with h5py.File(fres_initial, 'r') as hf:
            fres_initial = hf['fres'][:]
    
    finder = ResFinder(
        f, z, fres_initial, outpath, margin_factor, overwrite
    )
    finder.run()
    return np.array(finder.fres)


class ResFinder:
    def __init__(self, f, z, fres_initial, outpath, margin_factor = 0.15,
                 overwrite = False):
        """
        Interactive resonance finder for VNA sweep data.
        
        Parameters:
        f (np.ndarray): Frequency data in Hz (1D array).
        z (np.ndarray): Complex S21 data (1D array).
        fres_initial (array-like): Initial resonant frequency guesses
            in Hz.
        outpath (str): Path to save the resonance list (.h5 file).
        margin_factor (float): Fraction of data range to add as margin
            when auto-scaling y-axis. Default is 0.15 (15% margin).
        overwrite (bool): If False, raise error if output file already
            exists. Default is False.

        Returns:
        None
        """
        self.f = np.asarray(f, dtype = np.float64)
        self.z = np.asarray(z, dtype = np.complex128)
        self.fres = list(np.asarray(fres_initial, dtype = np.float64))
        self.outpath = os.path.normpath(os.path.expanduser(outpath))
        self.margin_factor = margin_factor

        # Check file extension
        if not self.outpath.lower().endswith('.h5'):
            raise ValueError(
                f'Output path must have a .h5 extension, got: {self.outpath}'
            )

        # Check output directory exists
        outdir = os.path.dirname(self.outpath)
        if outdir and not os.path.isdir(outdir):
            raise FileNotFoundError(
                f'Output directory does not exist: {outdir}'
            )

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

        # Unwrap phase and remove 3rd order polynomial trend
        unwrapped_phase = np.unwrap(np.angle(self.z))
        # Fit 1st order polynomial to remove trend
        p = np.polyfit(self.f, unwrapped_phase, 1)
        phase_trend = np.polyval(p, self.f)
        self.z *= np.exp(-1j * phase_trend)
        
        # Compute magnitude and phase
        self.mag_db = 20 * np.log10(np.abs(self.z))
        self.phase = np.unwrap(np.angle(self.z))
        
        # Undo history
        self.undo_stack = []

        # IQ plot state (initialise before setup_ui so signal handlers are safe)
        self.iq_visible = False
        self._iq_f_sel = np.array([])
        self._iq_z_sel = np.array([], dtype = np.complex128)

        # Setup the application
        self.app = pg.mkQApp("Resonance Finder")
        self.setup_ui()
        
    def setup_ui(self):
        """
        Setup the main window and layout.

        Parameters:
        None

        Returns:
        None
        """
        self.win = pg.GraphicsLayoutWidget(
            show = True,
            title = "Interactive Resonance Finder"
        )
        self.win.resize(1400, 800)
        self.win.setWindowTitle('Resonance Finder - Press H for help')
        
        # Add title with instructions
        title_text = (
            '<span style="color: #FFF; font-size: 10pt;">'
            '<b>Controls:</b> '
            'Click: Add resonance | '
            'Shift+Click: Remove nearest | '
            'Ctrl+Z: Undo | '
            'Z/X: Pan left/right | '
            'S: Save | '
            'I: Toggle IQ plot'
            '</span>'
        )
        self.win.addLabel(title_text, col = 0)
        self.win.nextRow()
        
        # Add Quit and Save button
        quit_button_widget = QtWidgets.QPushButton('Quit and Save')
        quit_button_widget.clicked.connect(self.quit_and_save)
        quit_button_proxy = QtWidgets.QGraphicsProxyWidget()
        quit_button_proxy.setWidget(quit_button_widget)
        self.win.addItem(quit_button_proxy, row = 0, col = 1)
        self.win.ci.layout.setColumnStretchFactor(0, 10)
        self.win.ci.layout.setColumnStretchFactor(1, 1)
        
        # Add log/message area at the bottom
        self.win.nextRow()
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        style = ("background-color: #2b2b2b; color: #ffffff; "
                 "font-family: monospace;")
        self.log_text.setStyleSheet(style)
        log_proxy = QtWidgets.QGraphicsProxyWidget()
        log_proxy.setWidget(self.log_text)
        self.win.addItem(log_proxy, row = 3, col = 0, colspan = 2)
        
        # Setup plots after UI is ready
        self.setup_plots()
        
        # Initialize markers and auto-scale
        self.update_resonance_markers()
        self.auto_scale_y()
        
    def log(self, message):
        """
        Append a message to the log display.

        Parameters:
        message (str): Message to display in the log.

        Returns:
        None
        """
        if not hasattr(self, 'log_text'):
            return  # UI not initialized
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def setup_plots(self):
        """
        Setup the magnitude and phase plots stacked vertically.

        Parameters:
        None

        Returns:
        None
        """
        # Magnitude plot on top
        self.plot_mag = self.win.addPlot(row = 1, col = 0)
        self.plot_mag.setLabel('left', '|S21| (dB)')
        self.plot_mag.showGrid(x = True, y = True, alpha = 0.3)
        self.plot_mag.getAxis('bottom').setStyle(showValues = False)
        
        # Plot magnitude data with downsampling for performance
        self.mag_curve = self.plot_mag.plot(
            self.f, self.mag_db,
            pen = pg.mkPen(color = (100, 200, 255, 90), width = 1.5),
            autoDownsample = True,
            downsample = 10,
            downsampleMethod = 'subsample'
        )
        # Highlighted overlay (IQ window) — drawn on top, hidden by default
        self.mag_highlight = self.plot_mag.plot(
            [], [],
            pen = pg.mkPen(color = (255, 230, 0), width = 2.5)
        )
        
        # Move to next row for phase plot
        self.win.nextRow()
        
        # Phase plot on bottom
        self.plot_phase = self.win.addPlot(row = 2, col = 0)
        self.plot_phase.setLabel('left', 'Phase (rad)')
        self.plot_phase.setLabel('bottom', 'Frequency', units = 'Hz')
        self.plot_phase.showGrid(x = True, y = True, alpha = 0.3)
        
        # Plot phase data
        self.phase_curve = self.plot_phase.plot(
            self.f, self.phase,
            pen = pg.mkPen(color = (255, 150, 100, 90), width = 1.5),
            autoDownsample = True,
            downsample = 10,
            downsampleMethod = 'subsample'
        )
        # Highlighted overlay (IQ window) — drawn on top, hidden by default
        self.phase_highlight = self.plot_phase.plot(
            [], [],
            pen = pg.mkPen(color = (255, 230, 0), width = 2.5)
        )
        
        # Link x-axes so they pan together
        self.plot_phase.setXLink(self.plot_mag)

        # IQ (I vs Q) plot — right column, spans both mag and phase rows
        self.plot_iq = self.win.addPlot(row = 1, col = 1, rowspan = 2)
        self.plot_iq.setLabel('left', 'Q (Im)')
        self.plot_iq.setLabel('bottom', 'I (Re)')
        self.plot_iq.showGrid(x = True, y = True, alpha = 0.3)
        self.plot_iq.setAspectLocked(True)
        self.iq_curve = self.plot_iq.plot(
            [], [],
            pen = pg.mkPen(color = (255, 230, 0), width = 1.5)
        )
        # Fres markers on IQ plot
        self.iq_fres_scatter = pg.ScatterPlotItem(
            size = 14, pen = pg.mkPen('w', width = 1)
        )
        self.plot_iq.addItem(self.iq_fres_scatter)
        # Hidden until toggled on (iq_visible already set in __init__)
        self.plot_iq.hide()
        # Give col 0 most width; col 1 is square so needs less
        self.win.ci.layout.setColumnStretchFactor(0, 3)
        self.win.ci.layout.setColumnStretchFactor(1, 1)

        # Store resonance markers
        self.mag_markers = []
        self.phase_markers = []
        
        # Connect signals for both plots
        self.plot_mag.scene().sigMouseClicked.connect(self.on_click)
        self.plot_phase.scene().sigMouseClicked.connect(self.on_click)
        self.plot_iq.scene().sigMouseClicked.connect(self.on_iq_click)
        self.plot_mag.sigRangeChanged.connect(self.on_range_changed)
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """
        Setup keyboard shortcuts.

        Parameters:
        None

        Returns:
        None
        """
        # Save shortcut
        self.save_action = QtGui.QShortcut(QtCore.Qt.Key_S, self.win)
        self.save_action.activated.connect(self.save_data)
        
        # Undo shortcut
        self.undo_action = QtGui.QShortcut(
            QtCore.Qt.ControlModifier | QtCore.Qt.Key_Z, self.win
        )
        self.undo_action.activated.connect(self.undo)
        
        # Pan left shortcut
        self.pan_left_action = QtGui.QShortcut(QtCore.Qt.Key_Z, self.win)
        self.pan_left_action.activated.connect(self.pan_left)
        
        # Pan right shortcut
        self.pan_right_action = QtGui.QShortcut(QtCore.Qt.Key_X, self.win)
        self.pan_right_action.activated.connect(self.pan_right)
        
        # IQ plot toggle shortcut
        self.iq_action = QtGui.QShortcut(QtCore.Qt.Key_I, self.win)
        self.iq_action.activated.connect(self.toggle_iq)

        # Help shortcut
        self.help_action = QtGui.QShortcut(QtCore.Qt.Key_H, self.win)
        self.help_action.activated.connect(self.show_help)
        
    def on_click(self, event):
        """
        Handle mouse clicks for adding/removing resonances.

        Parameters:
        event (MouseEvent): Mouse click event.

        Returns:
        None
        """
        # Get the position in the magnitude plot
        pos = event.scenePos()
        
        # Check which plot was clicked
        freq = None
        if self.plot_mag.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_mag.vb.mapSceneToView(pos)
            freq = mouse_point.x()
        elif self.plot_phase.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_phase.vb.mapSceneToView(pos)
            freq = mouse_point.x()
        
        if freq is not None:
            modifiers = event.modifiers()
            
            if event.button() == QtCore.Qt.LeftButton:
                if modifiers == QtCore.Qt.ShiftModifier:
                    # Shift + Click: Remove nearest resonance
                    self.remove_nearest_resonance(freq)
                else:
                    # Regular Click: Add resonance
                    self.add_resonance(freq)

    def on_iq_click(self, event):
        """
        Handle mouse clicks on the IQ plot.

        Left click: add a resonance at the data frequency nearest to the
        clicked IQ point. Shift+Click: remove the nearest resonance.

        Parameters:
        event (MouseEvent): Mouse click event.

        Returns:
        None
        """
        if not self.iq_visible:
            return
        pos = event.scenePos()
        if not self.plot_iq.sceneBoundingRect().contains(pos):
            return
        if len(self._iq_f_sel) == 0:
            self.log("No data in IQ window.")
            return
        mouse_point = self.plot_iq.vb.mapSceneToView(pos)
        click_z = complex(mouse_point.x(), mouse_point.y())
        # Find the data point in the IQ window closest to where was clicked
        distances = np.abs(self._iq_z_sel - click_z)
        nearest_idx = int(np.argmin(distances))
        nearest_freq = float(self._iq_f_sel[nearest_idx])
        modifiers = event.modifiers()
        if event.button() == QtCore.Qt.LeftButton:
            if modifiers == QtCore.Qt.ShiftModifier:
                self.remove_nearest_resonance(nearest_freq)
            else:
                self.add_resonance(nearest_freq)

    def _local_df(self, freq):
        """
        Return the local data-point spacing in self.f nearest to freq.

        Parameters:
        freq (float): Frequency in Hz.

        Returns:
        df (float): Minimum spacing between adjacent samples near freq.
        """
        idx = int(np.searchsorted(self.f, freq))
        idx = int(np.clip(idx, 0, len(self.f) - 1))
        left = self.f[idx] - self.f[idx - 1] if idx > 0 else np.inf
        right = self.f[idx + 1] - self.f[idx] if idx < len(self.f) - 1 else np.inf
        return min(left, right)

    def _interpolate_z(self, freq):
        """
        Linearly interpolate self.z at freq between bracketing samples.

        Parameters:
        freq (float): Frequency in Hz.

        Returns:
        z_interp (complex): Interpolated complex S21 value.
        """
        idx = int(np.searchsorted(self.f, freq))
        if idx == 0:
            return self.z[0]
        if idx >= len(self.f):
            return self.z[-1]
        f0, f1 = self.f[idx - 1], self.f[idx]
        z0, z1 = self.z[idx - 1], self.z[idx]
        t = (freq - f0) / (f1 - f0)
        return z0 + t * (z1 - z0)

    def add_resonance(self, freq):
        """
        Add a resonance at the specified frequency.

        Refuses to add if a resonance already exists within one local
        data-point spacing of freq.

        Parameters:
        freq (float): Frequency in Hz to add as a resonance.

        Returns:
        None
        """
        if self.fres:
            distances = [abs(freq - f) for f in self.fres]
            min_dist = min(distances)
            if min_dist < self._local_df(freq):
                nearest = self.fres[int(np.argmin(distances))]
                msg = (f"Too close to existing resonance at "
                       f"{nearest / 1e6:.3f} MHz "
                       f"(within one sample spacing). Not added.")
                self.log(msg)
                return

        # Save state for undo
        self.undo_stack.append(('add', freq))
        
        self.fres.append(freq)
        self.fres.sort()
        self.update_resonance_markers()
        msg = (f"Added resonance at {freq / 1e6:.3f} MHz. "
               f"Total: {len(self.fres)}")
        self.log(msg)
        
    def remove_nearest_resonance(self, freq):
        """
        Remove the resonance nearest to the specified frequency.

        Parameters:
        freq (float): Frequency in Hz near the resonance to remove.

        Returns:
        None
        """
        if not self.fres:
            return
            
        # Find nearest resonance within the current view
        x_range = self.plot_mag.viewRange()[0]
        visible_fres = [f for f in self.fres if x_range[0] <= f <= x_range[1]]
        
        if not visible_fres:
            self.log("No resonances in current view.")
            return
            
        # Find nearest
        distances = [abs(freq - f) for f in visible_fres]
        min_distance = min(distances)
        nearest = visible_fres[np.argmin(distances)]
        
        # Only remove if click is within 10% of the current plot width
        view_width = x_range[1] - x_range[0]
        max_click_fraction = 0.10  # 10% of view width
        max_click_distance = max_click_fraction * view_width
        
        if min_distance > max_click_distance:
            # Show distance in same units as current scale
            msg = (f"Click too far from resonance. "
                   f"Clicked {min_distance / 1e6:.3f} MHz away, "
                   f"need to be within {max_click_distance / 1e6:.3f} "
                   f"MHz ({max_click_fraction * 100:.0f}% of view width)")
            self.log(msg)
            return
        
        # Remove it
        removed_freq = nearest
        
        # Save state for undo
        self.undo_stack.append(('remove', removed_freq))
        
        self.fres.remove(removed_freq)
        self.update_resonance_markers()
        msg = (f"Removed resonance at {removed_freq / 1e6:.3f} MHz. "
               f"Total: {len(self.fres)}")
        self.log(msg)        
    def undo(self):
        """
        Undo the last add or remove operation.

        Parameters:
        None

        Returns:
        None
        """
        if not self.undo_stack:
            self.log("Nothing to undo.")
            return
        
        action, freq = self.undo_stack.pop()
        
        if action == 'add':
            # Undo an add by removing
            if freq in self.fres:
                self.fres.remove(freq)
                self.update_resonance_markers()
                msg = (f"Undid add at {freq / 1e6:.3f} MHz. "
                       f"Total: {len(self.fres)}")
                self.log(msg)
        elif action == 'remove':
            # Undo a remove by adding back
            self.fres.append(freq)
            self.fres.sort()
            self.update_resonance_markers()
            msg = (f"Undid remove at {freq / 1e6:.3f} MHz. "
                   f"Total: {len(self.fres)}")
            self.log(msg)        
    def update_resonance_markers(self):
        """
        Update the vertical lines marking resonances.

        Parameters:
        None

        Returns:
        None
        """
        if not hasattr(self, 'mag_markers'):
            return  # UI not initialized
        # Remove old markers
        for marker in self.mag_markers:
            self.plot_mag.removeItem(marker)
        for marker in self.phase_markers:
            self.plot_phase.removeItem(marker)
            
        self.mag_markers = []
        self.phase_markers = []
        
        # Cycle through 6 distinct (color, line-style) combinations so
        # adjacent markers are always visually distinct.
        _styles = [
            ((255, 255,   0, 180), QtCore.Qt.SolidLine),  # yellow solid
            ((  0, 255, 255, 180), QtCore.Qt.DashLine),   # cyan   dash
            ((255, 100, 255, 180), QtCore.Qt.DotLine),    # magenta dot
            ((255, 255,   0, 180), QtCore.Qt.DashLine),   # yellow dash
            ((  0, 255, 255, 180), QtCore.Qt.DotLine),    # cyan   dot
            ((255, 100, 255, 180), QtCore.Qt.SolidLine),  # magenta solid
        ]

        # Add new markers
        for i, freq in enumerate(self.fres):
            color, line_style = _styles[i % len(_styles)]
            
            # Magnitude plot marker
            line_mag = pg.InfiniteLine(
                pos = freq,
                angle = 90,
                pen = pg.mkPen(
                    color = color,
                    width = 2,
                    style = line_style
                )
            )
            self.plot_mag.addItem(line_mag)
            self.mag_markers.append(line_mag)
            
            # Phase plot marker
            line_phase = pg.InfiniteLine(
                pos = freq,
                angle = 90,
                pen = pg.mkPen(
                    color = color,
                    width = 2,
                    style = line_style
                )
            )
            self.plot_phase.addItem(line_phase)
            self.phase_markers.append(line_phase)

        if getattr(self, 'iq_visible', False):
            self.update_iq_plot()

    def on_range_changed(self):
        """
        Called when the view range changes (pan/zoom).

        Parameters:
        None

        Returns:
        None
        """
        self.auto_scale_y()
        if self.iq_visible:
            self.update_iq_plot()

    def toggle_iq(self):
        """
        Toggle the IQ plot visibility.

        Parameters:
        None

        Returns:
        None
        """
        self.iq_visible = not self.iq_visible
        if self.iq_visible:
            self.plot_iq.show()
            self.update_iq_plot()
            self.log('IQ plot shown.')
        else:
            self.plot_iq.hide()
            self.mag_highlight.setData([], [])
            self.phase_highlight.setData([], [])
            self.iq_fres_scatter.setData([])
            self.log('IQ plot hidden.')

    def update_iq_plot(self):
        """
        Refresh the IQ plot using the center quarter of the current view span.

        Also highlights the corresponding data segment on the mag and phase
        plots in a contrasting colour, and shows interpolated fres positions
        as markers on the IQ plot.

        Parameters:
        None

        Returns:
        None
        """
        if not hasattr(self, 'plot_iq'):
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        span = x_max - x_min
        center = 0.5 * (x_min + x_max)
        iq_min = center - span / 8
        iq_max = center + span / 8
        mask = (self.f >= iq_min) & (self.f <= iq_max)
        if not np.any(mask):
            self.iq_curve.setData([], [])
            self.mag_highlight.setData([], [])
            self.phase_highlight.setData([], [])
            self.iq_fres_scatter.setData([])
            self._iq_f_sel = np.array([])
            self._iq_z_sel = np.array([], dtype = np.complex128)
            return
        f_sel = self.f[mask]
        z_sel = self.z[mask]
        self._iq_f_sel = f_sel
        self._iq_z_sel = z_sel
        self.iq_curve.setData(z_sel.real, z_sel.imag)
        self.mag_highlight.setData(f_sel, self.mag_db[mask])
        self.phase_highlight.setData(f_sel, self.phase[mask])
        # Fres markers: interpolate z at each fres within the IQ window
        _marker_colors = [
            (255, 255,   0, 230),
            (  0, 255, 255, 230),
            (255, 100, 255, 230),
            (255, 255,   0, 230),
            (  0, 255, 255, 230),
            (255, 100, 255, 230),
        ]
        spots = []
        for i, freq in enumerate(self.fres):
            if iq_min <= freq <= iq_max:
                z_interp = self._interpolate_z(freq)
                color = _marker_colors[i % len(_marker_colors)]
                spots.append({
                    'pos': (z_interp.real, z_interp.imag),
                    'brush': pg.mkBrush(color),
                    'size': 14,
                })
        self.iq_fres_scatter.setData(spots)
        
    def auto_scale_y(self):
        """
        Auto-scale y-axis based on the visible x-range.

        Parameters:
        None

        Returns:
        None
        """
        if not hasattr(self, 'plot_mag'):
            return  # UI not initialized
        x_range = self.plot_mag.viewRange()[0]
        
        # Find indices within the visible range
        mask = (self.f >= x_range[0]) & (self.f <= x_range[1])
        
        if not np.any(mask):
            return
            
        # Get visible data
        visible_mag = self.mag_db[mask]
        visible_phase = self.phase[mask]
        
        # Calculate ranges with margin
        mag_min, mag_max = visible_mag.min(), visible_mag.max()
        mag_range = mag_max - mag_min
        mag_margin = mag_range * self.margin_factor
        
        phase_min, phase_max = visible_phase.min(), visible_phase.max()
        phase_range = phase_max - phase_min
        phase_margin = phase_range * self.margin_factor
        
        # Set y-ranges
        self.plot_mag.setYRange(
            mag_min - mag_margin,
            mag_max + mag_margin,
            padding = 0
        )
        self.plot_phase.setYRange(
            phase_min - phase_margin,
            phase_max + phase_margin,
            padding = 0
        )
        
    def pan_left(self):
        """
        Pan the view to the left by 20% of current width.

        Parameters:
        None

        Returns:
        None
        """
        x_range = self.plot_mag.viewRange()[0]
        width = x_range[1] - x_range[0]
        shift = -0.2 * width
        self.plot_mag.setXRange(
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
        x_range = self.plot_mag.viewRange()[0]
        width = x_range[1] - x_range[0]
        shift = 0.2 * width
        self.plot_mag.setXRange(
            x_range[0] + shift,
            x_range[1] + shift,
            padding = 0
        )
        
    def save_data(self):
        """
        Save the current resonance list to HDF5 file.

        Parameters:
        None

        Returns:
        None
        """
        fres_array = np.array(self.fres, dtype = np.float64)
        
        with h5py.File(self.outpath, 'w') as hf:
            hf.create_dataset('fres', data = fres_array)
        
        self.log(f"Saved {len(self.fres)} resonances to {self.outpath}")
        
    def quit_and_save(self):
        """
        Save data and close the application.

        Parameters:
        None

        Returns:
        None
        """
        self.save_data()
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
        <h3>Resonance Finder Controls</h3>
        <p><b>Mouse:</b></p>
        <ul>
        <li><b>Left Click:</b> Add resonance at clicked frequency</li>
        <li><b>Shift + Click:</b> Remove nearest resonance in view</li>
        <li><b>Mouse Wheel:</b> Zoom in/out</li>
        <li><b>Right Click + Drag:</b> Zoom to rectangle</li>
        </ul>
        <p><b>Keyboard:</b></p>
        <ul>
        <li><b>Z:</b> Pan left by 20%</li>
        <li><b>X:</b> Pan right by 20%</li>
        <li><b>S:</b> Save resonances to file</li>
        <li><b>Ctrl+Z:</b> Undo last add/remove</li>
        <li><b>I:</b> Toggle IQ (I vs Q) plot</li>
        <li><b>H:</b> Show this help</li>
        </ul>
        <p><b>Button:</b></p>
        <ul>
        <li><b>Quit and Save:</b> Save and close application</li>
        </ul>
        <p><b>Current Status:</b></p>
        <ul>
        <li>Total resonances: """ + str(len(self.fres)) + """</li>
        <li>Output file: """ + self.outpath + """</li>
        </ul>
        """
        
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Resonance Finder Help")
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
        self.log("Starting Resonance Finder...")
        self.log(f"Initial resonances: {len(self.fres)}")
        self.log(f"Output file: {self.outpath}")
        self.log("Press 'H' for help")
        
        # Start the Qt event loop
        self.app.exec_()
