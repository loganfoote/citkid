"""
Interactive manual resonance finder for VNA sweep data.

This module provides a fast, responsive interface for manually identifying
resonances in VNA sweep data with automatic y-axis scaling, phase
visualization, and undo support.
"""

import numpy as np
import zarr
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import os

from ..qt_compat import Qt as _Qt


def run_res_finder_manual(
        f, z, fres_initial, zarr_grp, margin_factor = 0.15
):
    """
    Run the interactive manual resonance finder.
    
    Parameters:
    f (np.ndarray): Frequency data in Hz.
    z (np.ndarray): Complex S21 data.
    fres_initial (array-like or zarr.Group or str): Initial resonant
        frequency guesses in Hz, or a zarr group (or path) containing
        a 'fres_auto' dataset.
    zarr_grp (zarr.Group or str): Zarr group or path to save the resonance
        list.  The resonances are saved as 'fres_manual' in the group.
    margin_factor (float): Y-axis margin factor for auto-scaling.
        Default is 0.15.
        
    Returns:
    fres (np.ndarray): Final list of resonant frequencies.
    """
    # Handle loading fres from zarr group or path if provided
    if isinstance(fres_initial, (str, os.PathLike)):
        grp = zarr.open_group(str(fres_initial), mode = 'r')
        fres_initial = grp['fres_auto'][:]
    elif isinstance(fres_initial, zarr.Group):
        fres_initial = fres_initial['fres_auto'][:]
    
    # Resolve zarr_grp to a zarr group to check for existing data
    if isinstance(zarr_grp, (str, os.PathLike)):
        zarr_group = zarr.open_group(str(zarr_grp), mode = 'a')
    else:
        zarr_group = zarr_grp
    
    # Check if fres_manual already exists in zarr
    if 'fres_manual' in zarr_group:
        # Show dialog to ask user what to do
        choice = ResFinder._show_zarr_dialog()
        if choice == 'load':
            # Load from zarr, overriding fres_initial
            fres_initial = zarr_group['fres_manual'][:]
        elif choice == 'overwrite':
            # Use fres_initial to overwrite zarr (it will be saved on close)
            pass
        else:  # choice == 'cancel'
            return None
    
    finder = ResFinder(
        f, z, fres_initial, zarr_grp, margin_factor
    )
    finder.run()
    return np.array(finder.fres)



class ResFinderWindow(pg.GraphicsLayoutWidget):
    """
    Custom GraphicsLayoutWidget that saves data when window is closed.
    """
    def __init__(self, finder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finder = finder
    
    def closeEvent(self, event):
        """
        Handle window close event by saving data.
        
        Parameters:
        event: Qt close event
        
        Returns:
        None
        """
        if self.finder is not None:
            self.finder.save_data()
        super().closeEvent(event)


class ResFinder(QtCore.QObject):
    def __init__(
            self, f, z, fres_initial, zarr_grp, margin_factor = 0.15,
        ):
        """
        Interactive resonance finder for VNA sweep data.
        
        Parameters:
        f (np.ndarray): Frequency data in Hz (1D array).
        z (np.ndarray): Complex S21 data (1D array).
        fres_initial (array-like): Initial resonant frequency guesses
            in Hz.
        zarr_grp (zarr.Group or str): Zarr group or path to save the
            resonance list.  The resonances are saved as 'fres_manual' in
            the group.
        margin_factor (float): Fraction of data range to add as margin
            when auto-scaling y-axis. Default is 0.15 (15% margin).

        Returns:
        None
        """
        super().__init__()
        self.f = np.asarray(f, dtype = np.float64)
        self.z = np.asarray(z, dtype = np.complex128)
        self.fres = list(np.asarray(fres_initial, dtype = np.float64))
        self.margin_factor = margin_factor

        # Resolve zarr_grp to a zarr group
        if isinstance(zarr_grp, (str, os.PathLike)):
            self.zarr_group = zarr.open_group(str(zarr_grp), mode = 'a')
        else:
            self.zarr_group = zarr_grp

        # Unwrap phase and remove 3rd order polynomial trend
        unwrapped_phase = np.unwrap(np.angle(self.z))
        # Fit 1st order polynomial to remove trend
        p = np.polyfit(self.f, unwrapped_phase, 1)
        phase_trend = np.polyval(p, self.f)
        self.z *= np.exp(-1j * phase_trend)
        
        # Compute magnitude and phase
        self.mag_db = 20 * np.log10(np.abs(self.z))
        self.phase = np.unwrap(np.angle(self.z))

        # Compute baseline minimum y-range limits so auto-scaling doesn't
        # collapse the view when there are no strong features in the
        # current window. These are estimated from median per-bin ranges
        # across the full sweep.
        self._compute_min_y_ranges()
        
        # Undo history
        self.undo_stack = []

        # IQ plot state (initialise before setup_ui so signal handlers are safe)
        self.iq_visible = False
        self._iq_f_sel = np.array([])
        self._iq_z_sel = np.array([], dtype = np.complex128)

        # Drag selection state (for shift+drag to remove multiple resonances)
        self._drag_selection_active = False
        self._drag_start_freq = None
        self._drag_region_item = None

        # Debounce timer for range-change events (pan/zoom)
        self._range_timer = None

        # Overview navigator re-entrancy guard
        self._overview_updating = False

        # Setup the application
        self.app = pg.mkQApp("Resonance Finder")
        self.setup_ui()
    
    @staticmethod
    def _show_zarr_dialog():
        """
        Show dialog when zarr 'fres_manual' already exists.
        
        Returns:
        str: 'overwrite', 'load', or 'cancel'
        """
        # Create a simple Qt application if needed
        app = pg.mkQApp("Resonance Finder")
        
        dlg = QtWidgets.QDialog()
        dlg.setWindowTitle('Existing Data Found')
        layout = QtWidgets.QVBoxLayout(dlg)
        
        label = QtWidgets.QLabel(
            '<h3>Existing fres_manual Found</h3>'
            '<p>The zarr group already contains manual resonance data.</p>'
            '<p>Choose an option:</p>'
        )
        label.setTextFormat(_Qt.RichText)
        layout.addWidget(label)
        
        btn_layout = QtWidgets.QVBoxLayout()
        
        load_btn = QtWidgets.QPushButton('Load from Zarr')
        load_btn.setToolTip('Load existing resonance list and continue editing')
        
        overwrite_btn = QtWidgets.QPushButton('Overwrite with Initial')
        overwrite_btn.setToolTip('Replace zarr data with the provided fres_initial')
        
        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.setToolTip('Exit without doing anything')
        
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(overwrite_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        result = [None]
        
        def on_load():
            result[0] = 'load'
            dlg.accept()
        
        def on_overwrite():
            # Show confirmation dialog
            reply = QtWidgets.QMessageBox.question(
                dlg,
                'Confirm Overwrite',
                'Are you sure you want to overwrite the existing fres_manual?\n\n'
                'This will replace it with the provided fres_initial.',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No  # Default to No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                result[0] = 'overwrite'
                dlg.accept()
            # If No, dialog stays open
        
        def on_cancel():
            result[0] = 'cancel'
            dlg.reject()
        
        load_btn.clicked.connect(on_load)
        overwrite_btn.clicked.connect(on_overwrite)
        cancel_btn.clicked.connect(on_cancel)
        
        dlg.exec_()
        return result[0] if result[0] else 'cancel'
        
    def setup_ui(self):
        """
        Setup the main window and layout.

        Parameters:
        None

        Returns:
        None
        """
        self.win = ResFinderWindow(
            finder=self,
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
            'Shift+Drag: Remove range | '
            'Ctrl+Z: Undo | '
            'Z/X: Pan 20% | '
            'A/S: Pan 80% | '
            'Ctrl+S: Save | '
            'Q: Toggle IQ plot'
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
        self.win.addItem(log_proxy, row = 4, col = 0, colspan = 2)
        
        # Setup plots after UI is ready
        self.setup_plots()
        
        # Initialize markers and auto-scale
        self.update_resonance_markers()
        # Start zoomed to the first 10 MHz for a fast initial render;
        # the overview navigator shows the full range.
        _f_lo = float(self.f[0])
        _f_hi = _f_lo + 10e6
        self.plot_mag.setXRange(_f_lo, _f_hi, padding=0.02)
        self._update_curves()
        self.auto_scale_y()
        self._update_overview_region()
        
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
        # ---- Overview navigator (row 1) ------------------------------------
        self.plot_overview = self.win.addPlot(row = 1, col = 0)
        self.plot_overview.setMaximumHeight(80)
        self.plot_overview.showAxis('left')
        self.plot_overview.getAxis('left').setStyle(showValues = False)
        self.plot_overview.getAxis('bottom').setStyle(showValues = False)
        self.plot_overview.showGrid(x = False, y = False)
        self.plot_overview.setMouseEnabled(x = True, y = False)
        self.plot_overview.hideButtons()
        # Disable wheel zoom on overview plot
        self.plot_overview.vb.wheelScaleFactor = 0

        _stride = max(1, len(self.f) // 3000)
        self.plot_overview.plot(
            self.f[::_stride], self.mag_db[::_stride],
            pen = pg.mkPen(color = (100, 200, 255, 120), width = 1),
        )

        self._overview_region = pg.LinearRegionItem(
            values = [float(self.f[0]), float(self.f[-1])],
            brush = pg.mkBrush(255, 255, 255, 30),
            pen = pg.mkPen('w', width = 1),
            movable = True,
        )
        self.plot_overview.addItem(self._overview_region)
        self.plot_overview.setXRange(
            float(self.f[0]), float(self.f[-1]), padding = 0.01
        )
        self._overview_region.sigRegionChanged.connect(
            self._on_overview_region_changed
        )

        self.win.nextRow()

        # ---- Magnitude plot (row 2) ----------------------------------------
        self.plot_mag = self.win.addPlot(row = 2, col = 0)
        self.plot_mag.setLabel('left', '|S21| (dB)')
        self.plot_mag.showGrid(x = True, y = True, alpha = 0.3)
        self.plot_mag.getAxis('bottom').setStyle(showValues = False)
        
        # Plot magnitude data — initialised empty; _update_curves pushes
        # only the visible slice on each range change, so pyqtgraph never
        # processes the full 1e6-point array during startup or interaction.
        self.mag_curve = self.plot_mag.plot(
            [], [],
            pen = pg.mkPen(color = (100, 200, 255, 90), width = 1.5)
        )
        # Highlighted overlay (IQ window) — drawn on top, hidden by default
        self.mag_highlight = self.plot_mag.plot(
            [], [],
            pen = pg.mkPen(color = (255, 230, 0), width = 2.5)
        )
        
        # Move to next row for phase plot
        self.win.nextRow()
        
        # Phase plot on bottom (row 3)
        self.plot_phase = self.win.addPlot(row = 3, col = 0)
        self.plot_phase.setLabel('left', 'Phase (rad)')
        self.plot_phase.setLabel('bottom', 'Frequency', units = 'Hz')
        self.plot_phase.showGrid(x = True, y = True, alpha = 0.3)
        
        # Plot phase data — initialised empty; populated by _update_curves
        self.phase_curve = self.plot_phase.plot(
            [], [],
            pen = pg.mkPen(color = (255, 150, 100, 90), width = 1.5)
        )
        # Highlighted overlay (IQ window) — drawn on top, hidden by default
        self.phase_highlight = self.plot_phase.plot(
            [], [],
            pen = pg.mkPen(color = (255, 230, 0), width = 2.5)
        )
        
        # Link x-axes so they pan together
        self.plot_phase.setXLink(self.plot_mag)

        # IQ (I vs Q) plot — right column, spans both mag and phase rows
        self.plot_iq = self.win.addPlot(row = 2, col = 1, rowspan = 2)
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
        
        # Connect signals — all plots share the same scene so connect once
        self.plot_mag.scene().sigMouseClicked.connect(self.on_click)
        self.plot_iq.scene().sigMouseClicked.connect(self.on_iq_click)
        self.plot_mag.sigRangeChanged.connect(self.on_range_changed)
        self.plot_overview.scene().sigMouseClicked.connect(self.on_overview_click)
        
        # Connect mouse events for drag-to-remove functionality
        self.plot_mag.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        # Install event filter on the scene to intercept mouse events before ViewBox
        self.plot_mag.scene().installEventFilter(self)
        
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
        # Save shortcut (Ctrl+S, not S which is used for pan 80% right)
        self.save_action = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self.win)
        self.save_action.activated.connect(self.save_data)

        # Undo shortcut (Ctrl+Z)
        self.undo_action = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self.win)
        self.undo_action.activated.connect(self.undo)

        # Pan left/right shortcuts
        self.pan_left_action = QtGui.QShortcut(QtGui.QKeySequence("Z"), self.win)
        self.pan_left_action.activated.connect(self.pan_left)
        self.pan_right_action = QtGui.QShortcut(QtGui.QKeySequence("X"), self.win)
        self.pan_right_action.activated.connect(self.pan_right)

        # IQ plot toggle shortcut
        self.iq_action = QtGui.QShortcut(QtGui.QKeySequence("Q"), self.win)
        self.iq_action.activated.connect(self.toggle_iq)

        # Fast pan shortcuts (A/S = 80% jump, one resonance width)
        self.fast_pan_left_action = QtGui.QShortcut(QtGui.QKeySequence("A"), self.win)
        self.fast_pan_left_action.activated.connect(self.fast_pan_left)
        self.fast_pan_right_action = QtGui.QShortcut(QtGui.QKeySequence("S"), self.win)
        self.fast_pan_right_action.activated.connect(self.fast_pan_right)

        # Help shortcut
        self.help_action = QtGui.QShortcut(QtGui.QKeySequence("H"), self.win)
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
            
            if event.button() == _Qt.LeftButton:
                if modifiers == _Qt.ShiftModifier:
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
        if event.button() == _Qt.LeftButton:
            if modifiers == _Qt.ShiftModifier:
                self.remove_nearest_resonance(nearest_freq)
            else:
                self.add_resonance(nearest_freq)

    def eventFilter(self, obj, event):
        """
        Event filter to intercept mouse events on the scene for drag-to-remove.
        
        Allows Shift+Click+Drag to remove all resonances in a frequency range.
        Consumes mouse events during drag to prevent ViewBox panning.
        
        Parameters:
        obj: Object that received the event (the GraphicsScene)
        event: Qt event
        
        Returns:
        bool: True if event was handled, False otherwise
        """
        # Check if this is a mouse event during a shift+drag operation
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        is_shift_held = modifiers & _Qt.ShiftModifier
        
        if event.type() == QtCore.QEvent.Type.GraphicsSceneMousePress:
            if is_shift_held and event.button() == _Qt.LeftButton:
                # Check if click is on the magnitude plot
                if self.plot_mag.sceneBoundingRect().contains(event.scenePos()):
                    # Start drag selection
                    self._drag_selection_active = True
                    mouse_point = self.plot_mag.vb.mapSceneToView(event.scenePos())
                    self._drag_start_freq = mouse_point.x()
                    return True
        elif event.type() == QtCore.QEvent.Type.GraphicsSceneMouseMove:
            if self._drag_selection_active and self._drag_start_freq is not None:
                # Update visual during drag
                pos = event.scenePos()
                if self.plot_mag.sceneBoundingRect().contains(pos):
                    self.on_mouse_moved(pos)
                return True
        elif event.type() == QtCore.QEvent.Type.GraphicsSceneMouseRelease:
            if self._drag_selection_active and self._drag_start_freq is not None:
                self._drag_selection_active = False
                
                # Get the end position
                mouse_point = self.plot_mag.vb.mapSceneToView(event.scenePos())
                end_freq = mouse_point.x()
                
                # Calculate range
                f_min = min(self._drag_start_freq, end_freq)
                f_max = max(self._drag_start_freq, end_freq)
                
                # Remove the visual indicator
                if self._drag_region_item is not None:
                    self.plot_mag.removeItem(self._drag_region_item)
                    self._drag_region_item = None
                
                # Remove all resonances in range
                if f_max > f_min:  # Only if drag had extent
                    self.remove_resonances_in_range(f_min, f_max)
                
                self._drag_start_freq = None
                return True
        
        return False
    
    def on_mouse_moved(self, pos):
        """
        Handle mouse move event. Update drag selection visual if active.
        
        Parameters:
        pos: Scene position
        
        Returns:
        None
        """
        if not self._drag_selection_active or self._drag_start_freq is None:
            return
        
        # Only update if mouse is over the plot
        if not self.plot_mag.sceneBoundingRect().contains(pos):
            return
        
        mouse_point = self.plot_mag.vb.mapSceneToView(pos)
        current_freq = mouse_point.x()
        
        # Update or create the drag region item
        f_min = min(self._drag_start_freq, current_freq)
        f_max = max(self._drag_start_freq, current_freq)
        
        if self._drag_region_item is None:
            # Create a new region item
            self._drag_region_item = pg.LinearRegionItem(
                values=[f_min, f_max],
                brush=pg.mkBrush(255, 100, 100, 50),
                pen=pg.mkPen('r', width=2),
                movable=False
            )
            self.plot_mag.addItem(self._drag_region_item)
        else:
            # Update existing region
            self._drag_region_item.setRegion([f_min, f_max])

    def remove_resonances_in_range(self, f_min, f_max):
        """
        Remove all resonances with frequencies in the given range.
        
        Parameters:
        f_min (float): Minimum frequency in Hz
        f_max (float): Maximum frequency in Hz
        
        Returns:
        None
        """
        if not self.fres:
            self.log("No resonances to remove.")
            return
        
        # Save state for undo
        self.undo_stack.append(('remove_range', f_min, f_max, list(self.fres)))
        
        # Remove resonances in range
        removed_count = 0
        self.fres = [f for f in self.fres if not (f_min <= f <= f_max)]
        removed_count = len(self.undo_stack[-1][3]) - len(self.fres)
        
        if removed_count > 0:
            self.update_resonance_markers()
            self.log(f"Removed {removed_count} resonance(s) in range [{f_min/1e9:.3f}, {f_max/1e9:.3f}] GHz")
        else:
            self.log(f"No resonances found in range [{f_min/1e9:.3f}, {f_max/1e9:.3f}] GHz")

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
            ((255, 255,   0, 180), _Qt.SolidLine),  # yellow solid
            ((  0, 255, 255, 180), _Qt.DashLine),   # cyan   dash
            ((255, 100, 255, 180), _Qt.DotLine),    # magenta dot
            ((255, 255,   0, 180), _Qt.DashLine),   # yellow dash
            ((  0, 255, 255, 180), _Qt.DotLine),    # cyan   dot
            ((255, 100, 255, 180), _Qt.SolidLine),  # magenta solid
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

        Debounced: coalesces rapid-fire events (e.g. continuous mouse drag)
        into a single update every 50 ms so the UI stays responsive.

        Parameters:
        None

        Returns:
        None
        """
        if self._range_timer is None:
            self._range_timer = QtCore.QTimer()
            self._range_timer.setSingleShot(True)
            self._range_timer.timeout.connect(self._do_range_update)
        self._range_timer.start(50)  # ms

    def _do_range_update(self):
        """Actual work triggered by the debounce timer."""
        self._update_curves()
        self.auto_scale_y()
        self._update_overview_region()
        if self.iq_visible:
            self.update_iq_plot()

    def _on_overview_region_changed(self):
        """Pan/zoom the main plot to match the dragged overview region."""
        if self._overview_updating:
            return
        lo, hi = self._overview_region.getRegion()
        self._overview_updating = True
        self.plot_mag.setXRange(lo, hi, padding = 0)
        self._overview_updating = False

    def _update_overview_region(self):
        """Move the overview region to reflect the current main plot view."""
        if self._overview_updating:
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        self._overview_updating = True
        self._overview_region.setRegion([x_min, x_max])
        self._overview_updating = False

    def _update_curves(self):
        """
        Push only the visible data slice to the mag and phase curves.

        Using a 50% pad on each side prevents blank edges during fast panning.
        pyqtgraph then only has to render the visible ~1,000 points instead
        of the full 300,000-point dataset.

        Parameters:
        None

        Returns:
        None
        """
        if not hasattr(self, 'plot_mag'):
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        span = x_max - x_min
        sl = self._visible_slice(x_min - 0.5 * span, x_max + 0.5 * span)
        if sl.start >= sl.stop:
            return
        self.mag_curve.setData(self.f[sl], self.mag_db[sl])
        self.phase_curve.setData(self.f[sl], self.phase[sl])

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
        sl = self._visible_slice(iq_min, iq_max)
        if sl.start >= sl.stop:
            self.iq_curve.setData([], [])
            self.mag_highlight.setData([], [])
            self.phase_highlight.setData([], [])
            self.iq_fres_scatter.setData([])
            self._iq_f_sel = np.array([])
            self._iq_z_sel = np.array([], dtype = np.complex128)
            return
        f_sel = self.f[sl]
        z_sel = self.z[sl]
        self._iq_f_sel = f_sel
        self._iq_z_sel = z_sel
        self.iq_curve.setData(z_sel.real, z_sel.imag)
        self.mag_highlight.setData(f_sel, self.mag_db[sl])
        self.phase_highlight.setData(f_sel, self.phase[sl])
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
        
    def _visible_slice(self, x_min, x_max):
        """
        Return a slice of the (sorted) frequency array covering [x_min, x_max].

        Uses binary search — O(log n) regardless of array length.

        Parameters:
        x_min (float): Left edge of visible range in Hz.
        x_max (float): Right edge of visible range in Hz.

        Returns:
        sl (slice): Slice into self.f / self.mag_db / self.phase.
        """
        lo = int(np.searchsorted(self.f, x_min, side='left'))
        hi = int(np.searchsorted(self.f, x_max, side='right'))
        return slice(lo, hi)

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
        x_min, x_max = self.plot_mag.viewRange()[0]
        sl = self._visible_slice(x_min, x_max)

        if sl.start >= sl.stop:
            return

        # Get visible data
        visible_mag = self.mag_db[sl]
        visible_phase = self.phase[sl]
        
        # Calculate ranges with margin
        mag_min, mag_max = float(visible_mag.min()), float(visible_mag.max())
        mag_margin = (mag_max - mag_min) * self.margin_factor

        phase_min, phase_max = float(visible_phase.min()), float(visible_phase.max())
        phase_margin = (phase_max - phase_min) * self.margin_factor

        # Determine ranges and enforce minimum sensible ranges computed
        # at initialization so we don't zoom in too far when a window is
        # featureless.
        raw_mag_range = mag_max - mag_min
        desired_mag_range = max(raw_mag_range + 2 * mag_margin,
                                getattr(self, 'min_mag_range', 1.0))
        mag_center = 0.5 * (mag_max + mag_min)
        mag_half = 0.5 * desired_mag_range

        raw_phase_range = phase_max - phase_min
        desired_phase_range = max(raw_phase_range + 2 * phase_margin,
                                  getattr(self, 'min_phase_range', 0.1))
        phase_center = 0.5 * (phase_max + phase_min)
        phase_half = 0.5 * desired_phase_range

        self.plot_mag.setYRange(mag_center - mag_half,
                                mag_center + mag_half,
                                padding = 0)
        self.plot_phase.setYRange(phase_center - phase_half,
                                  phase_center + phase_half,
                                  padding = 0)

    def _compute_min_y_ranges(self, bin_count=100, min_bin_width=1e6,
                              min_mag=1.0, min_phase=0.1):
        """
        Estimate minimum y-range for magnitude (dB) and phase (rad) by
        computing the per-bin dynamic range across the full frequency
        sweep and taking the median (then halving it). This prevents
        auto-scaling from collapsing when the visible window lacks strong
        features.

        Parameters:
        bin_count (int): Number of bins to divide the sweep into.
        min_bin_width (float): Minimum bin width in Hz.
        min_mag (float): Absolute floor for magnitude range in dB.
        min_phase (float): Absolute floor for phase range in radians.
        """
        f0, f1 = float(self.f[0]), float(self.f[-1])
        span = f1 - f0
        if span <= 0 or len(self.f) < 2:
            self.min_mag_range = min_mag
            self.min_phase_range = min_phase
            return

        bin_width = max(span / float(bin_count), float(min_bin_width))
        mag_ranges = []
        phase_ranges = []
        for i in range(int(np.ceil(span / bin_width))):
            left = f0 + i * bin_width
            right = left + bin_width
            lo = int(np.searchsorted(self.f, left, side='left'))
            hi = int(np.searchsorted(self.f, right, side='right'))
            if hi - lo < 2:
                continue
            seg_mag = self.mag_db[lo:hi]
            seg_phase = self.phase[lo:hi]
            mag_ranges.append(float(np.max(seg_mag) - np.min(seg_mag)))
            phase_ranges.append(float(np.max(seg_phase) - np.min(seg_phase)))

        if mag_ranges:
            med_mag = float(np.median(np.asarray(mag_ranges)))
        else:
            med_mag = float(np.ptp(self.mag_db)) if len(self.mag_db) > 0 else 0.0

        if phase_ranges:
            med_phase = float(np.median(np.asarray(phase_ranges)))
        else:
            med_phase = float(np.ptp(self.phase)) if len(self.phase) > 0 else 0.0

        # Use half the median per-bin range as a conservative minimum, but
        # don't go below absolute floors.
        self.min_mag_range = max(med_mag * 0.5, float(min_mag))
        self.min_phase_range = max(med_phase * 0.5, float(min_phase))
        
    def _pan(self, fraction):
        """Shift the x-axis by *fraction* of the current view width."""
        x0, x1 = self.plot_mag.viewRange()[0]
        shift = fraction * (x1 - x0)
        self.plot_mag.setXRange(x0 + shift, x1 + shift, padding=0)

    def pan_left(self):
        """
        Pan the view to the left by 20% of current width (Z key).

        Parameters:
        None

        Returns:
        None
        """
        self._pan(-0.2)

    def pan_right(self):
        """
        Pan the view to the right by 20% of current width (X key).

        Parameters:
        None

        Returns:
        None
        """
        self._pan(0.2)

    def fast_pan_left(self):
        """
        Pan the view to the left by 80% of current width (A key).

        Parameters:
        None

        Returns:
        None
        """
        self._pan(-0.8)

    def fast_pan_right(self):
        """
        Pan the view to the right by 80% of current width (S key).

        Parameters:
        None

        Returns:
        None
        """
        self._pan(0.8)
    
    def _get_fractional_zoom(self, x_min, x_max):
        """
        Calculate fractional zoom level from a frequency range.
        
        Fractional zoom is defined as (x_max - x_min) / center_freq.
        This allows maintaining the "zoom" when moving to a new center.
        
        Parameters:
        x_min (float): Left edge of x-range in Hz.
        x_max (float): Right edge of x-range in Hz.
        
        Returns:
        frac_zoom (float): Fractional zoom level.
        """
        center = (x_min + x_max) / 2.0
        if center == 0:
            return 1.0
        return (x_max - x_min) / center
    
    def _set_centered_range(self, center_freq, frac_zoom):
        """
        Set x-range of main plot centered at center_freq with given fractional zoom.
        
        Parameters:
        center_freq (float): Frequency to center on in Hz.
        frac_zoom (float): Fractional zoom level to apply.
        
        Returns:
        None
        """
        half_span = center_freq * frac_zoom / 2.0
        new_x_min = center_freq - half_span
        new_x_max = center_freq + half_span
        self.plot_mag.setXRange(new_x_min, new_x_max, padding=0)
    
    def on_overview_click(self, event):
        """
        Handle clicks on the overview plot to center main plot at clicked frequency.
        
        Parameters:
        event (MouseEvent): Mouse click event.
        
        Returns:
        None
        """
        pos = event.scenePos()
        
        # Check if click was on the overview plot
        if not self.plot_overview.sceneBoundingRect().contains(pos):
            return
        
        # Map scene position to view coordinates
        mouse_point = self.plot_overview.vb.mapSceneToView(pos)
        clicked_freq = mouse_point.x()
        
        # Get current x-range from the main magnitude plot
        x_min, x_max = self.plot_mag.viewRange()[0]
        
        # Calculate fractional zoom and apply it centered on clicked frequency
        frac_zoom = self._get_fractional_zoom(x_min, x_max)
        self._set_centered_range(clicked_freq, frac_zoom)
        
    def save_data(self):
        """
        Save the current resonance list to zarr group.

        Parameters:
        None

        Returns:
        None
        """
        fres_array = np.array(self.fres, dtype = np.float64)

        if 'fres_manual' in self.zarr_group:
            del self.zarr_group['fres_manual']
        self.zarr_group.create_array('fres_manual', data = fres_array)

        self.log(f"Saved {len(self.fres)} resonances to zarr group")
        
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
        Toggle the help panel on/off.

        Parameters:
        None

        Returns:
        None
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
        """Create the floating help dialog (created lazily on first use)."""
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle("Resonance Finder Help (H to close)")
        layout = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel(
            "<h3>Resonance Finder Controls</h3>"
            "<p><b>Mouse:</b></p>"
            "<ul>"
            "<li><b>Left Click:</b> Add resonance at clicked frequency</li>"
            "<li><b>Shift+Click:</b> Remove nearest resonance in view</li>"
            "<li><b>Mouse Wheel:</b> Zoom in/out</li>"
            "<li><b>Right Click + Drag:</b> Zoom to rectangle</li>"
            "</ul>"
            "<p><b>Keyboard:</b></p>"
            "<ul>"
            "<li><b>Z:</b> Pan left by 20%</li>"
            "<li><b>X:</b> Pan right by 20%</li>"
            "<li><b>A:</b> Pan left by 80%</li>"
            "<li><b>S:</b> Pan right by 80%</li>"
            "<li><b>Ctrl+S:</b> Save resonances to file</li>"
            "<li><b>Ctrl+Z:</b> Undo last add/remove</li>"
            "<li><b>Q:</b> Toggle IQ (I vs Q) plot</li>"
            "<li><b>H:</b> Toggle this help panel</li>"
            "</ul>"
            "<p><b>Button:</b></p>"
            "<ul>"
            "<li><b>Quit and Save:</b> Save and close application</li>"
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
        self.log(f"Output: {self.zarr_group.store}")
        self.log("Press 'H' for help")
        
        # Start the Qt event loop
        self.app.exec()
