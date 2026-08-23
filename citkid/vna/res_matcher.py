"""
Interactive resonance matcher for pairing resonances between two VNA datasets.

Each resonance belongs to a MatchGroup that links zero or more resonances from
dataset 1 to zero or more resonances from dataset 2.  The user scrolls through
the sweep, edits groups by clicking and using keyboard shortcuts, and saves the
result to zarr.

Smart Re-Addition
-----------------
When adding a resonance in a window containing previously deleted resonances,
the user is prompted to choose from all removed res_idx values in that window
or create a new resonator index. This prevents accidental loss of resonator
identity when making corrections.

Automatic Re-Matching
---------------------
Shift+click sets a frequency threshold (yellow line). Press T to re-run the
matching algorithm (sorted or nearest) on all groups above that frequency,
while keeping manually corrected lower-frequency groups unchanged.
Output layout (zarr)
--------------------
fres1            float64[N1]  — final DS1 resonance frequencies (Hz)
res_idx1         int64[N1]    — resonator indices for DS1
group_ids1       int64[N1]    — match-group ID for each DS1 resonance

fres2            float64[N2]  — final DS2 resonance frequencies (Hz)
res_idx2         int64[N2]    — resonator indices for DS2
group_ids2       int64[N2]    — match-group ID for each DS2 resonance

ambiguous_groups int64[K]     — group IDs flagged as ambiguous

To query group g: fres1[group_ids1 == g] and fres2[group_ids2 == g].
"""

import copy
import dataclasses
import os
from typing import List, Optional, Tuple, Set

import numpy as np
import zarr
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from ..qt_compat import Qt as _Qt
from .s21_filt import highpass_filter, polynomial_baseline


# ---------------------------------------------------------------------------
# Colour palette — qualitative, readable on dark backgrounds
# ---------------------------------------------------------------------------
_GROUP_COLORS: List[Tuple[int, int, int]] = [
    (100, 180, 255),   # blue
    (255, 140,  60),   # orange
    (100, 220, 120),   # green
    (255, 100, 100),   # red
    (200, 130, 255),   # purple
    (255, 220,  60),   # yellow
    (100, 240, 210),   # teal
    (255, 120, 200),   # pink
    (180, 255,  80),   # lime
    (160, 160, 255),   # lavender
]


def _group_color(group_id: int, alpha: int = 220) -> Tuple:
    r, g, b = _GROUP_COLORS[group_id % len(_GROUP_COLORS)]
    return (r, g, b, alpha)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class MatchGroup:
    """
    One logical match group linking zero or more resonances from each dataset.

    Attributes:
    group_id (int): Unique integer identifier for this group.
    entries1 (list): List of (fres_hz, res_idx) tuples for dataset 1.
    entries2 (list): List of (fres_hz, res_idx) tuples for dataset 2.
    ambiguous (bool): True if this group has been flagged as ambiguous.
    """
    group_id: int
    entries1: List[Tuple[float, int]]   # [(fres_hz, res_idx), ...]
    entries2: List[Tuple[float, int]]   # [(fres_hz, res_idx), ...]
    ambiguous: bool = False

    def center_freq(self) -> float:
        freqs = [f for f, _ in self.entries1] + [f for f, _ in self.entries2]
        return float(np.mean(freqs)) if freqs else 0.0

    def mapping_str(self) -> str:
        return f"{len(self.entries1)}-{len(self.entries2)}"


# ---------------------------------------------------------------------------
# SpinBox event filter (select-all on focus)
# ---------------------------------------------------------------------------
class SpinBoxEventFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.FocusIn:
            if hasattr(obj, 'lineEdit'):
                QtCore.QTimer.singleShot(0, obj.lineEdit().selectAll)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_res_matcher(
    f1, z1, fres1, res_idx1,
    f2, z2, fres2, res_idx2,
    zarr_grp,
    new_res_start_idx: int = 2000,
    init_match: str = 'sorted',
    margin_factor: float = 0.15,
    apply_filter: bool = False,
):
    """
    Run the interactive resonance matcher.

    Parameters:
    f1 (array-like): Frequency array for dataset 1, Hz.
    z1 (array-like): Complex S21 for dataset 1.
    fres1 (array-like): Initial resonance frequencies for dataset 1, Hz.
    res_idx1 (array-like): Resonator indices for dataset 1 (locked to
        existing values).
    f2, z2, fres2, res_idx2 (array-like): Same for dataset 2.
    zarr_grp (zarr.Group or str): Zarr group or path to save results.
    new_res_start_idx (int): First res_idx value assigned to newly added
        resonances.  Increments by 1 for each addition.  Default is 2000.
        The actual starting index will be max(new_res_start_idx, max(existing_indices)+1)
        to avoid collisions with existing resonator indices.
    init_match (str): How to build the initial pairing.  'sorted' pairs by
        frequency order; 'nearest' uses median-offset-corrected
        nearest-neighbor.  Default is 'sorted'.
    margin_factor (float): Y-axis auto-scale margin fraction.  Default 0.15.
    apply_filter (bool): If False (default), both dataset filters start as
        'none' so the raw magnitude is shown and startup is fast.  Set True
        to start with the default highpass filter applied.

    Returns:
    groups (list[MatchGroup]): Final list of match groups.
    """
    matcher = ResMatcher(
        f1, z1, fres1, res_idx1,
        f2, z2, fres2, res_idx2,
        zarr_grp,
        new_res_start_idx=new_res_start_idx,
        init_match=init_match,
        margin_factor=margin_factor,
        apply_filter=apply_filter,
    )
    matcher.run()
    return matcher.groups


class CustomViewBox(pg.ViewBox):
    """ViewBox that suppresses context menu when Ctrl is held."""
    
    def raiseContextMenu(self, ev):
        """Override to check for Ctrl modifier before showing menu."""
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers & _Qt.ControlModifier:
            # Ctrl is held, don't show menu (let our custom handler process it)
            ev.accept()
            return
        # Otherwise, show normal PyQtGraph menu
        super().raiseContextMenu(ev)

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ResMatcher:

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        f1, z1, fres1, res_idx1,
        f2, z2, fres2, res_idx2,
        zarr_grp,
        new_res_start_idx: int = 2000,
        init_match: str = 'sorted',
        margin_factor: float = 0.15,
        overwrite: bool = False,
        apply_filter: bool = False,
    ):
        # ---- store raw arrays -----------------------------------------------
        self.f1 = np.asarray(f1, dtype=np.float64)
        self.z1 = np.asarray(z1, dtype=np.complex128)
        self.f2 = np.asarray(f2, dtype=np.float64)
        self.z2 = np.asarray(z2, dtype=np.complex128)
        self.fres1_init = np.asarray(fres1, dtype=np.float64)
        self.res_idx1_init = np.asarray(res_idx1, dtype=np.int64)
        self.fres2_init = np.asarray(fres2, dtype=np.float64)
        self.res_idx2_init = np.asarray(res_idx2, dtype=np.int64)

        # ---- Validate input res_idx arrays -----------------------------------
        if len(self.res_idx1_init) > 0:
            unique1 = np.unique(self.res_idx1_init)
            if len(unique1) != len(self.res_idx1_init):
                raise ValueError(
                    f"res_idx1 contains {len(self.res_idx1_init) - len(unique1)} "
                    f"duplicate values. Each resonator must have a unique index."
                )
        if len(self.res_idx2_init) > 0:
            unique2 = np.unique(self.res_idx2_init)
            if len(unique2) != len(self.res_idx2_init):
                raise ValueError(
                    f"res_idx2 contains {len(self.res_idx2_init) - len(unique2)} "
                    f"duplicate values. Each resonator must have a unique index."
                )

        self.margin_factor = margin_factor
        # Ensure new resonance indices don't collide with existing ones
        max_existing_idx = max(
            np.max(self.res_idx1_init) if len(self.res_idx1_init) > 0 else -1,
            np.max(self.res_idx2_init) if len(self.res_idx2_init) > 0 else -1,
        )
        self._next_new_idx = max(int(new_res_start_idx), int(max_existing_idx) + 1)
        self._next_group_id = 0    # set after initialisation

        # ---- zarr -----------------------------------------------------------
        if isinstance(zarr_grp, (str, os.PathLike)):
            self.zarr_group = zarr.open_group(str(zarr_grp), mode='a')
        else:
            self.zarr_group = zarr_grp

        # Check if zarr data exists and show dialog
        _output_keys = [
            'fres1', 'res_idx1', 'group_ids1',
            'fres2', 'res_idx2', 'group_ids2',
            'ambiguous_groups',
        ]
        _load_from_zarr = False
        _max_view_left_initial = None
        
        if all(key in self.zarr_group for key in _output_keys):
            # Data exists, show dialog
            choice = self._show_startup_dialog()
            if choice == 'cancel':
                raise RuntimeError("User cancelled operation")
            elif choice == 'load':
                _load_from_zarr = True
                # Load max_view_left if it exists
                if 'max_view_left' in self.zarr_group:
                    _max_view_left_initial = float(self.zarr_group['max_view_left'][()])
            elif choice == 'overwrite':
                # Delete existing data
                for key in _output_keys:
                    del self.zarr_group[key]
                if 'max_view_left' in self.zarr_group:
                    del self.zarr_group['max_view_left']

        # ---- phase (linear-trend removed per dataset) -----------------------
        self.mag_db1 = 20.0 * np.log10(np.abs(self.z1))
        self.mag_db2 = 20.0 * np.log10(np.abs(self.z2))

        _ph1_raw = np.unwrap(np.angle(self.z1))
        _p1 = np.polyfit(self.f1, _ph1_raw, 1)
        self.phase1 = np.unwrap(np.angle(self.z1 * np.exp(-1j * np.polyval(_p1, self.f1))))

        _ph2_raw = np.unwrap(np.angle(self.z2))
        _p2 = np.polyfit(self.f2, _ph2_raw, 1)
        self.phase2 = np.unwrap(np.angle(self.z2 * np.exp(-1j * np.polyval(_p2, self.f2))))

        # ---- filtered magnitude (updated when controls change) --------------
        self.filtered_mag1 = self.mag_db1.copy()
        self.filtered_mag2 = self.mag_db2.copy()

        # Default filter params
        _default_smoothing = 'highpass' if apply_filter else 'none'
        self.filter_params1 = {
            'smoothing': _default_smoothing,
            'highpass_mhz': 10.0,
            'poly_order': 3,
        }
        self.filter_params2 = {
            'smoothing': _default_smoothing,
            'highpass_mhz': 10.0,
            'poly_order': 3,
        }
        self._apply_filters()

        # ---- build initial groups -------------------------------------------
        self._init_match_method = init_match  # Store for re-matching
        if _load_from_zarr:
            self.groups, self._next_group_id = self._load_from_zarr()
        else:
            if init_match == 'sorted':
                self.groups, self._next_group_id = self._init_sorted()
            elif init_match == 'nearest':
                self.groups, self._next_group_id = self._init_nearest()
            else:
                raise ValueError(
                    f"init_match must be 'sorted' or 'nearest', got {init_match!r}"
                )

        # ---- interaction state ----------------------------------------------
        self.active_dataset: int = 1      # toggled with C
        # Multi-selection: set of (group_id, dataset, fres) tuples
        self._selected_resonances: Set[Tuple[int, int, float]] = set()
        self.undo_stack: list = []
        self._max_undo_stack = 50  # Limit undo stack to prevent memory issues
        self._scatter_just_clicked: bool = False
        self._rematch_freq_threshold: Optional[float] = None  # Auto-adjusting threshold
        self._last_edit_freq: Optional[float] = 0.0  # Track last edit (None = user-set, disables auto-adjust)
        self._threshold_pin_right: bool = True  # When True, threshold sticks to right edge of window
        
        # Track removed resonances: list of (fres, res_idx, dataset)
        self._removed_resonances: List[Tuple[float, int, int]] = []
        
        # Track highest left edge of window we've looked at
        self._max_view_left: float = _max_view_left_initial if _max_view_left_initial is not None else float(min(self.f1[0], self.f2[0]))

        # Debounce timers (created lazily)
        self._range_timer = None
        self._filter_timer1 = None
        self._filter_timer2 = None

        # ---- Qt setup -------------------------------------------------------
        self.app = pg.mkQApp("Resonance Matcher")
        self.event_filter = SpinBoxEventFilter()
        self.setup_ui()

    # ================================================================ init helpers

    def _init_sorted(self):
        """Pair by sorted frequency order; leftovers become unmatched groups."""
        idx1 = np.argsort(self.fres1_init)
        idx2 = np.argsort(self.fres2_init)
        fres1_s = self.fres1_init[idx1];  ridx1_s = self.res_idx1_init[idx1]
        fres2_s = self.fres2_init[idx2];  ridx2_s = self.res_idx2_init[idx2]

        n1, n2 = len(fres1_s), len(fres2_s)
        n_common = min(n1, n2)
        groups = []
        gid = 0

        for i in range(n_common):
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                entries2=[(float(fres2_s[i]), int(ridx2_s[i]))],
            ))
            gid += 1

        for i in range(n_common, n1):
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                entries2=[],
            ))
            gid += 1

        for i in range(n_common, n2):
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[],
                entries2=[(float(fres2_s[i]), int(ridx2_s[i]))],
            ))
            gid += 1

        groups.sort(key=lambda g: g.center_freq())
        return groups, gid

    def _init_nearest(self):
        """
        Pair by median-offset-corrected nearest-neighbor matching.

        A global median frequency offset between datasets is estimated first,
        then greedy nearest-neighbor matching is applied with a threshold of
        3× the median inter-resonance spacing.

        Parameters:
        None

        Returns:
        groups (list[MatchGroup]): Sorted list of initial match groups.
        gid (int): Next available group ID.
        """
        idx1 = np.argsort(self.fres1_init)
        idx2 = np.argsort(self.fres2_init)
        fres1_s = self.fres1_init[idx1];  ridx1_s = self.res_idx1_init[idx1]
        fres2_s = self.fres2_init[idx2];  ridx2_s = self.res_idx2_init[idx2]

        n1, n2 = len(fres1_s), len(fres2_s)
        n_common = min(n1, n2)

        offset = float(np.median(fres2_s[:n_common] - fres1_s[:n_common])) \
            if n_common > 0 else 0.0
        fres2_corr = fres2_s - offset

        if n1 > 1:
            threshold = 3.0 * float(np.median(np.diff(fres1_s)))
        elif n2 > 1:
            threshold = 3.0 * float(np.median(np.diff(fres2_s)))
        else:
            threshold = np.inf

        matched1: set = set()
        matched2: set = set()
        pairs = []

        for i, f1v in enumerate(fres1_s):
            if n2 == 0:
                break
            dists = np.abs(fres2_corr - f1v)
            j = int(np.argmin(dists))
            if dists[j] < threshold and j not in matched2:
                pairs.append((i, j))
                matched1.add(i)
                matched2.add(j)

        gid = 0
        groups = []
        for i, j in pairs:
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                entries2=[(float(fres2_s[j]), int(ridx2_s[j]))],
            ))
            gid += 1
        for i in range(n1):
            if i not in matched1:
                groups.append(MatchGroup(
                    group_id=gid,
                    entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                    entries2=[],
                ))
                gid += 1
        for j in range(n2):
            if j not in matched2:
                groups.append(MatchGroup(
                    group_id=gid,
                    entries1=[],
                    entries2=[(float(fres2_s[j]), int(ridx2_s[j]))],
                ))
                gid += 1

        groups.sort(key=lambda g: g.center_freq())
        return groups, gid

    def _show_startup_dialog(self):
        """
        Show dialog when zarr data already exists.
        
        Returns:
        str: 'overwrite', 'load', or 'cancel'
        """
        # Create a simple Qt application if needed
        app = pg.mkQApp("Resonance Matcher")
        
        dlg = QtWidgets.QDialog()
        dlg.setWindowTitle('Zarr Data Exists')
        layout = QtWidgets.QVBoxLayout(dlg)
        
        label = QtWidgets.QLabel(
            '<h3>Existing Data Found</h3>'
            '<p>The zarr group already contains matching data.</p>'
            '<p>Choose an option:</p>'
        )
        label.setTextFormat(_Qt.RichText)
        layout.addWidget(label)
        
        btn_layout = QtWidgets.QVBoxLayout()
        
        overwrite_btn = QtWidgets.QPushButton('Overwrite (Start Fresh with Auto-Grouping)')
        overwrite_btn.setToolTip('Delete existing data and start with automatic matching')
        
        load_btn = QtWidgets.QPushButton('Load from Zarr')
        load_btn.setToolTip('Load existing groups and continue editing')
        
        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.setToolTip('Exit without doing anything')
        
        btn_layout.addWidget(overwrite_btn)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        result = [None]
        
        def on_overwrite():
            # Show confirmation dialog
            reply = QtWidgets.QMessageBox.question(
                dlg,
                'Confirm Overwrite',
                'Are you sure you want to overwrite existing data?\n\n'
                'This will permanently delete all existing groups and matching data.',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No  # Default to No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                result[0] = 'overwrite'
                dlg.accept()
            # If No, dialog stays open
        
        def on_load():
            result[0] = 'load'
            dlg.accept()
        
        def on_cancel():
            result[0] = 'cancel'
            dlg.reject()
        
        overwrite_btn.clicked.connect(on_overwrite)
        load_btn.clicked.connect(on_load)
        cancel_btn.clicked.connect(on_cancel)
        
        dlg.exec_()
        return result[0] if result[0] else 'cancel'

    def _load_from_zarr(self):
        """
        Load existing groups from zarr.
        
        Returns:
        tuple: (groups, next_group_id)
        """
        fres1_out = np.array(self.zarr_group['fres1'])
        res_idx1_out = np.array(self.zarr_group['res_idx1'])
        group_ids1 = np.array(self.zarr_group['group_ids1'])
        
        fres2_out = np.array(self.zarr_group['fres2'])
        res_idx2_out = np.array(self.zarr_group['res_idx2'])
        group_ids2 = np.array(self.zarr_group['group_ids2'])
        
        ambiguous_groups = set(np.array(self.zarr_group['ambiguous_groups']))
        
        # Reconstruct groups
        group_dict = {}
        
        # Add DS1 entries
        for fres, ridx, gid in zip(fres1_out, res_idx1_out, group_ids1):
            if gid not in group_dict:
                group_dict[gid] = MatchGroup(
                    group_id=int(gid),
                    entries1=[],
                    entries2=[],
                    ambiguous=(gid in ambiguous_groups)
                )
            group_dict[gid].entries1.append((float(fres), int(ridx)))
        
        # Add DS2 entries
        for fres, ridx, gid in zip(fres2_out, res_idx2_out, group_ids2):
            if gid not in group_dict:
                group_dict[gid] = MatchGroup(
                    group_id=int(gid),
                    entries1=[],
                    entries2=[],
                    ambiguous=(gid in ambiguous_groups)
                )
            group_dict[gid].entries2.append((float(fres), int(ridx)))
        
        groups = list(group_dict.values())
        groups.sort(key=lambda g: g.center_freq())
        
        next_group_id = max(group_dict.keys()) + 1 if group_dict else 0
        
        return groups, next_group_id

    # ================================================================ filtering

    def _apply_filters(self):
        self.filtered_mag1 = self._apply_filter(
            self.f1, self.mag_db1, self.filter_params1
        )
        self.filtered_mag2 = self._apply_filter(
            self.f2, self.mag_db2, self.filter_params2
        )

    @staticmethod
    def _apply_filter(f, mag_db, params):
        method = params['smoothing']
        if method == 'highpass':
            return highpass_filter(f, mag_db, params['highpass_mhz'])
        elif method == 'polynomial':
            return polynomial_baseline(f, mag_db, params['poly_order'])
        else:
            return mag_db.copy()

    # ================================================================ UI setup

    def setup_ui(self):
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(
            'Resonance Matcher — E: delete  |  F/D: merge  |  W: unlink  |  Q: ambiguous  |  R: re-match  |  H: help'
        )
        self.win.resize(1500, 850)
        # Override close event to auto-save
        self.win.closeEvent = self._on_window_close

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(_Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        splitter.addWidget(self.plot_widget)

        # Right: controls
        ctrl_widget = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_widget)
        
        # ---- Toolbar at top ------------------------------------------------
        self.setup_toolbar(ctrl_layout)
        
        splitter.addWidget(ctrl_widget)
        splitter.setSizes([1150, 300])

        self.setup_controls(ctrl_layout)
        self.setup_plots()
        self.setup_shortcuts()

        # Initial state (filters already applied in __init__)
        f_lo = float(min(self.f1[0], self.f2[0]))
        
        # Use saved max_view_left if loading from zarr, otherwise start at beginning
        if self._max_view_left > f_lo:
            # Center on saved position with 10 MHz window
            f_center = self._max_view_left + 5e6
            f_start = f_center - 5e6
            f_end = f_center + 5e6
            f_hi = float(max(self.f1[-1], self.f2[-1]))
            # Make sure we don't go past the end
            if f_end > f_hi:
                f_end = f_hi
                f_start = max(f_lo, f_end - 10e6)
            self.plot_mag.setXRange(f_start, f_end, padding=0.02)
        else:
            # Start zoomed to first 10 MHz for fast initial render
            f_hi = f_lo + 10e6
            self.plot_mag.setXRange(f_lo, f_hi, padding=0.02)
        self._sync_mag2_geom()
        self._sync_phase2_geom()
        self._update_curves()
        self.update_markers()
        self.auto_scale_y()
        self._update_overview_region()
        
        # Initialize threshold at right edge of window
        self._update_threshold_position()

        self.win.show()

    # ---------------------------------------------------------------- controls

    def setup_controls(self, layout):
        # ---- Dataset 1 filter ----------------------------------------------
        ds1_grp = QtWidgets.QGroupBox('Dataset 1 Filter (blue)')
        ds1_form = QtWidgets.QFormLayout()

        self.smooth_combo1 = QtWidgets.QComboBox()
        self.smooth_combo1.addItems(['highpass', 'polynomial', 'none'])
        self.smooth_combo1.setCurrentText(self.filter_params1['smoothing'])
        self.smooth_combo1.currentTextChanged.connect(
            lambda: self._on_smoothing_changed(1)
        )

        self.hp_spin1 = QtWidgets.QDoubleSpinBox()
        self.hp_spin1.setRange(0.1, 10000.0)
        self.hp_spin1.setValue(self.filter_params1['highpass_mhz'])
        self.hp_spin1.setDecimals(1)
        self.hp_spin1.setSingleStep(1.0)
        self.hp_spin1.valueChanged.connect(lambda: self._on_filter_changed(1))
        self.hp_spin1.installEventFilter(self.event_filter)

        self.poly_spin1 = QtWidgets.QSpinBox()
        self.poly_spin1.setRange(1, 10)
        self.poly_spin1.setValue(self.filter_params1['poly_order'])
        self.poly_spin1.valueChanged.connect(lambda: self._on_filter_changed(1))
        self.poly_spin1.installEventFilter(self.event_filter)

        self.hp_label1 = QtWidgets.QLabel('HP Cutoff (MHz):')
        self.poly_label1 = QtWidgets.QLabel('Poly Order:')

        ds1_form.addRow('Method:', self.smooth_combo1)
        ds1_form.addRow(self.hp_label1, self.hp_spin1)
        ds1_form.addRow(self.poly_label1, self.poly_spin1)
        ds1_grp.setLayout(ds1_form)
        layout.addWidget(ds1_grp)

        # ---- Dataset 2 filter ----------------------------------------------
        ds2_grp = QtWidgets.QGroupBox('Dataset 2 Filter (orange)')
        ds2_form = QtWidgets.QFormLayout()

        self.smooth_combo2 = QtWidgets.QComboBox()
        self.smooth_combo2.addItems(['highpass', 'polynomial', 'none'])
        self.smooth_combo2.setCurrentText(self.filter_params2['smoothing'])
        self.smooth_combo2.currentTextChanged.connect(
            lambda: self._on_smoothing_changed(2)
        )

        self.hp_spin2 = QtWidgets.QDoubleSpinBox()
        self.hp_spin2.setRange(0.1, 10000.0)
        self.hp_spin2.setValue(self.filter_params2['highpass_mhz'])
        self.hp_spin2.setDecimals(1)
        self.hp_spin2.setSingleStep(1.0)
        self.hp_spin2.valueChanged.connect(lambda: self._on_filter_changed(2))
        self.hp_spin2.installEventFilter(self.event_filter)

        self.poly_spin2 = QtWidgets.QSpinBox()
        self.poly_spin2.setRange(1, 10)
        self.poly_spin2.setValue(self.filter_params2['poly_order'])
        self.poly_spin2.valueChanged.connect(lambda: self._on_filter_changed(2))
        self.poly_spin2.installEventFilter(self.event_filter)

        self.hp_label2 = QtWidgets.QLabel('HP Cutoff (MHz):')
        self.poly_label2 = QtWidgets.QLabel('Poly Order:')

        ds2_form.addRow('Method:', self.smooth_combo2)
        ds2_form.addRow(self.hp_label2, self.hp_spin2)
        ds2_form.addRow(self.poly_label2, self.poly_spin2)
        ds2_grp.setLayout(ds2_form)
        layout.addWidget(ds2_grp)

        self._update_filter_visibility(1)
        self._update_filter_visibility(2)

        # ---- Status --------------------------------------------------------
        self.active_ds_label = QtWidgets.QLabel()
        self._refresh_active_label()
        layout.addWidget(self.active_ds_label)

        self.sel_label = QtWidgets.QLabel('Selected: none')
        layout.addWidget(self.sel_label)

        self.status_label = QtWidgets.QLabel(
            '<span style="color:green;">Ready</span>'
        )
        layout.addWidget(self.status_label)

        # ---- Log -----------------------------------------------------------
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet(
            'background-color:#2b2b2b; color:#ffffff; font-family:monospace;'
        )
        layout.addWidget(self.log_text)

        # ---- Buttons -------------------------------------------------------
        btn_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton('Save (Ctrl+S)')
        save_btn.clicked.connect(self.save_data)
        quit_btn = QtWidgets.QPushButton('Save && Quit')
        quit_btn.clicked.connect(self.quit_and_save)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(quit_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

    def _update_filter_visibility(self, ds: int):
        if ds == 1:
            method = self.smooth_combo1.currentText()
            self.hp_label1.setVisible(method == 'highpass')
            self.hp_spin1.setVisible(method == 'highpass')
            self.poly_label1.setVisible(method == 'polynomial')
            self.poly_spin1.setVisible(method == 'polynomial')
        else:
            method = self.smooth_combo2.currentText()
            self.hp_label2.setVisible(method == 'highpass')
            self.hp_spin2.setVisible(method == 'highpass')
            self.poly_label2.setVisible(method == 'polynomial')
            self.poly_spin2.setVisible(method == 'polynomial')

    def _refresh_active_label(self):
        self.active_ds_label.setText(
            f'<b>Active dataset: DS{self.active_dataset}</b>'
        )

    # ------------------------------------------------------------------ plots

    def setup_plots(self):
        title = (
            '<span style="color:#FFF; font-size:9pt;">'
            'Z/X: pan 20%  |  A/S: pan 80%  |  E: delete  |  F/D: merge  |  W: unlink  |  Q: ambiguous  |  R: re-match  |  H: help'
            '</span>'
        )
        self.plot_widget.addLabel(title, col=0)
        self.plot_widget.nextRow()

        # ---- Overview navigator (row 1) ------------------------------------
        # Downsampled full-range view with a draggable region showing zoom.
        self.plot_overview = self.plot_widget.addPlot(row=1, col=0, viewBox=CustomViewBox())
        self.plot_overview.setMaximumHeight(80)
        self.plot_overview.setLabel('left', '')
        self.plot_overview.showAxis('left')
        self.plot_overview.getAxis('left').setStyle(showValues=False)
        self.plot_overview.getAxis('bottom').setStyle(showValues=False)
        self.plot_overview.showGrid(x=False, y=False)
        self.plot_overview.setMouseEnabled(x=True, y=False)
        self.plot_overview.hideButtons()

        _stride = max(1, len(self.f1) // 3000)
        self.plot_overview.plot(
            self.f1[::_stride], self.mag_db1[::_stride],
            pen=pg.mkPen(color=(100, 180, 255, 120), width=1),
        )
        _stride2 = max(1, len(self.f2) // 3000)
        self.plot_overview.plot(
            self.f2[::_stride2], self.mag_db2[::_stride2],
            pen=pg.mkPen(color=(255, 140, 60, 120), width=1),
        )

        f_lo = float(min(self.f1[0], self.f2[0]))
        f_hi = float(max(self.f1[-1], self.f2[-1]))
        self._overview_region = pg.LinearRegionItem(
            values=[f_lo, f_lo + 10e6],
            brush=pg.mkBrush(255, 255, 255, 30),
            pen=pg.mkPen('w', width=1),
            movable=True,
        )
        self.plot_overview.addItem(self._overview_region)
        self.plot_overview.setXRange(f_lo, f_hi, padding=0.01)

        # Two-way sync: region ↔ main plot
        self._overview_region.sigRegionChanged.connect(
            self._on_overview_region_changed
        )
        self._overview_updating = False   # re-entrancy guard

        self.plot_widget.nextRow()

        # ---- Magnitude plot (row 2) — dual Y axes -------------------------
        self.plot_mag = self.plot_widget.addPlot(row=2, col=0, viewBox=CustomViewBox())
        self.plot_mag.setLabel('left', '|S21| DS1 (dB)',
                               color=(100, 180, 255))
        self.plot_mag.showGrid(x=True, y=True, alpha=0.3)
        self.plot_mag.getAxis('bottom').setStyle(showValues=False)
        self.plot_mag.showAxis('right')
        self.plot_mag.setLabel('right', '|S21| DS2 (dB)',
                               color=(255, 140, 60))

        # Right-axis ViewBox for DS2 magnitude
        self._vb_mag2 = CustomViewBox()
        self.plot_mag.scene().addItem(self._vb_mag2)
        self.plot_mag.getAxis('right').linkToView(self._vb_mag2)
        self._vb_mag2.setXLink(self.plot_mag)
        self.plot_mag.vb.sigResized.connect(self._sync_mag2_geom)

        self.curve_ds1_mag = self.plot_mag.plot(
            [], [],
            pen=pg.mkPen(color=(100, 180, 255, 180), width=1.5),
        )
        self.curve_ds2_mag = pg.PlotCurveItem(
            [], [],
            pen=pg.mkPen(color=(255, 140, 60, 180), width=1.5),
        )
        self._vb_mag2.addItem(self.curve_ds2_mag)

        # Scatter: DS1 in main vb, DS2 in vb_mag2
        self.scatter_ds1_mag = pg.ScatterPlotItem(symbol='o', size=11)
        self.scatter_ds2_mag = pg.ScatterPlotItem(symbol='s', size=11)
        # Separate selection markers for DS1 (circles) and DS2 (squares)
        self.scatter_sel_ds1_mag = pg.ScatterPlotItem(
            symbol='o', size=18,
            pen=pg.mkPen('w', width=3), brush=pg.mkBrush(None),
        )
        self.scatter_sel_ds2_mag = pg.ScatterPlotItem(
            symbol='s', size=18,
            pen=pg.mkPen('w', width=3), brush=pg.mkBrush(None),
        )
        self.plot_mag.addItem(self.scatter_ds1_mag)
        self._vb_mag2.addItem(self.scatter_ds2_mag)
        self.plot_mag.addItem(self.scatter_sel_ds1_mag)
        self._vb_mag2.addItem(self.scatter_sel_ds2_mag)

        self.plot_widget.nextRow()

        # ---- Phase plot (row 3) — dual Y axes -----------------------------
        self.plot_phase = self.plot_widget.addPlot(row=3, col=0, viewBox=CustomViewBox())
        self.plot_phase.setLabel('left', 'Phase DS1 (rad)',
                                  color=(100, 180, 255))
        self.plot_phase.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.plot_phase.setXLink(self.plot_mag)
        self.plot_phase.showAxis('right')
        self.plot_phase.setLabel('right', 'Phase DS2 (rad)',
                                  color=(255, 140, 60))

        # Right-axis ViewBox for DS2 phase
        self._vb_phase2 = CustomViewBox()
        self.plot_phase.scene().addItem(self._vb_phase2)
        self.plot_phase.getAxis('right').linkToView(self._vb_phase2)
        self._vb_phase2.setXLink(self.plot_phase)
        self.plot_phase.vb.sigResized.connect(self._sync_phase2_geom)

        self.curve_ds1_phase = self.plot_phase.plot(
            [], [],
            pen=pg.mkPen(color=(100, 180, 255, 180), width=1.5),
        )
        self.curve_ds2_phase = pg.PlotCurveItem(
            [], [],
            pen=pg.mkPen(color=(255, 140, 60, 180), width=1.5),
        )
        self._vb_phase2.addItem(self.curve_ds2_phase)

        self.scatter_ds1_phase = pg.ScatterPlotItem(symbol='o', size=11)
        self.scatter_ds2_phase = pg.ScatterPlotItem(symbol='s', size=11)
        # Separate selection markers for DS1 (circles) and DS2 (squares)
        self.scatter_sel_ds1_phase = pg.ScatterPlotItem(
            symbol='o', size=18,
            pen=pg.mkPen('w', width=3), brush=pg.mkBrush(None),
        )
        self.scatter_sel_ds2_phase = pg.ScatterPlotItem(
            symbol='s', size=18,
            pen=pg.mkPen('w', width=3), brush=pg.mkBrush(None),
        )
        self.plot_phase.addItem(self.scatter_ds1_phase)
        self._vb_phase2.addItem(self.scatter_ds2_phase)
        self.plot_phase.addItem(self.scatter_sel_ds1_phase)
        self._vb_phase2.addItem(self.scatter_sel_ds2_phase)

        # ---- Re-match threshold line (hidden by default) -------------------
        self._rematch_line_mag = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen('y', width=2, style=_Qt.DashLine),
            movable=False
        )
        self._rematch_line_phase = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen('y', width=2, style=_Qt.DashLine),
            movable=False
        )
        self.plot_mag.addItem(self._rematch_line_mag)
        self.plot_phase.addItem(self._rematch_line_phase)
        self._rematch_line_mag.setVisible(False)
        self._rematch_line_phase.setVisible(False)

        # ---- Connect signals -----------------------------------------------
        self.plot_mag.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.plot_mag.sigRangeChanged.connect(self._on_range_changed)

        self.scatter_ds1_mag.sigClicked.connect(
            self._make_scatter_handler(ds=1)
        )
        self.scatter_ds2_mag.sigClicked.connect(
            self._make_scatter_handler(ds=2)
        )
        self.scatter_ds1_phase.sigClicked.connect(
            self._make_scatter_handler(ds=1)
        )
        self.scatter_ds2_phase.sigClicked.connect(
            self._make_scatter_handler(ds=2)
        )

    def _sync_mag2_geom(self):
        """Keep the DS2 magnitude ViewBox geometry in sync with the main one."""
        self._vb_mag2.setGeometry(self.plot_mag.vb.sceneBoundingRect())
        self._vb_mag2.linkedViewChanged(self.plot_mag.vb,
                                        self._vb_mag2.XAxis)

    def _sync_phase2_geom(self):
        """Keep the DS2 phase ViewBox geometry in sync with the main one."""
        self._vb_phase2.setGeometry(self.plot_phase.vb.sceneBoundingRect())
        self._vb_phase2.linkedViewChanged(self.plot_phase.vb,
                                          self._vb_phase2.XAxis)

    def _on_overview_region_changed(self):
        """Pan/zoom main plot to match the overview region."""
        if self._overview_updating:
            return
        lo, hi = self._overview_region.getRegion()
        self._overview_updating = True
        self.plot_mag.setXRange(lo, hi, padding=0)
        self._overview_updating = False

    def _update_overview_region(self):
        """Move the overview region to match the current main plot view."""
        if self._overview_updating:
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        self._overview_updating = True
        self._overview_region.setRegion([x_min, x_max])
        self._overview_updating = False

    def _make_scatter_handler(self, ds: int):
        """Return a slot that handles scatter clicks for the given dataset."""
        def handler(*args):
            # pyqtgraph ≥0.13: (scatter, points, ev)
            # pyqtgraph  <0.13: (points, ev)
            if len(args) == 3:
                _, pts, ev = args
            elif len(args) == 2:
                pts, ev = args
            else:
                return
            self._on_scatter_clicked(pts, ev, ds=ds)
        return handler

    # ---------------------------------------------------------------- shortcuts

    def setup_toolbar(self, layout):
        """Create toolbar with common operations."""
        toolbar_grp = QtWidgets.QGroupBox('Quick Actions')
        toolbar_layout = QtWidgets.QGridLayout()
        
        # Row 1: Edit operations
        delete_btn = QtWidgets.QPushButton('Delete (E)')
        delete_btn.clicked.connect(self.delete_selected)
        delete_btn.setToolTip('Delete selected resonances (E key)')
        
        unlink_btn = QtWidgets.QPushButton('Unlink (W)')
        unlink_btn.clicked.connect(self.unlink_selected)
        unlink_btn.setToolTip('Put selected resonances in separate groups (W key)')
        
        merge_btn = QtWidgets.QPushButton('Merge Groups (F)')
        merge_btn.clicked.connect(self.merge_groups)
        merge_btn.setToolTip('Merge all groups containing selected resonances (F key)')
        
        merge_sel_btn = QtWidgets.QPushButton('Merge Selected (D)')
        merge_sel_btn.clicked.connect(self.merge_selected_only)
        merge_sel_btn.setToolTip('Merge only selected resonances, unlink others (D key)')
        
        toolbar_layout.addWidget(delete_btn, 0, 0)
        toolbar_layout.addWidget(unlink_btn, 0, 1)
        toolbar_layout.addWidget(merge_btn, 1, 0)
        toolbar_layout.addWidget(merge_sel_btn, 1, 1)
        
        # Row 3: Ambiguous flag
        ambig_btn = QtWidgets.QPushButton('Ambiguous (Q)')
        ambig_btn.clicked.connect(self.toggle_ambiguous)
        ambig_btn.setToolTip('Toggle ambiguous flag (only for 1:1 matches, Q key)')
        
        toolbar_layout.addWidget(ambig_btn, 2, 0)
        
        # Row 4: Re-match and save
        rematch_btn = QtWidgets.QPushButton('Re-Match (R)')
        rematch_btn.clicked.connect(self.trigger_rematch)
        rematch_btn.setToolTip('Re-match all groups above threshold line (R key)')
        
        toolbar_layout.addWidget(rematch_btn, 3, 0)
        
        toolbar_grp.setLayout(toolbar_layout)
        layout.addWidget(toolbar_grp)

    def setup_shortcuts(self):
        _map = {
            'Z': self.pan_left,
            'X': self.pan_right,
            'A': self.fast_pan_left,
            'S': self.fast_pan_right,
            'H': self.show_help,
            'E': self.delete_selected,
            'R': self.trigger_rematch,
            'F': self.merge_groups,
            'D': self.merge_selected_only,
            'W': self.unlink_selected,
            'Q': self.toggle_ambiguous,
        }
        for key, slot in _map.items():
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.win)
            sc.activated.connect(slot)

        undo_sc = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Z'), self.win)
        undo_sc.activated.connect(self.undo)
        
        save_sc = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+S'), self.win)
        save_sc.activated.connect(self.save_data)

    # ================================================================ filter change

    def _on_smoothing_changed(self, ds: int):
        self._update_filter_visibility(ds)
        self._on_filter_changed(ds)

    def _on_filter_changed(self, ds: int):
        if ds == 1:
            self.filter_params1['smoothing'] = self.smooth_combo1.currentText()
            self.filter_params1['highpass_mhz'] = self.hp_spin1.value()
            self.filter_params1['poly_order'] = self.poly_spin1.value()
            timer_attr = '_filter_timer1'
        else:
            self.filter_params2['smoothing'] = self.smooth_combo2.currentText()
            self.filter_params2['highpass_mhz'] = self.hp_spin2.value()
            self.filter_params2['poly_order'] = self.poly_spin2.value()
            timer_attr = '_filter_timer2'

        self.status_label.setText(
            '<span style="color:orange;">Updating...</span>'
        )
        timer = getattr(self, timer_attr)
        if timer is None:
            timer = QtCore.QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda d=ds: self._do_filter_update(d))
            setattr(self, timer_attr, timer)
        timer.start(300)

    def _do_filter_update(self, ds: int):
        self.win.setCursor(_Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        if ds == 1:
            self.filtered_mag1 = self._apply_filter(
                self.f1, self.mag_db1, self.filter_params1
            )
        else:
            self.filtered_mag2 = self._apply_filter(
                self.f2, self.mag_db2, self.filter_params2
            )
        self._update_curves()
        self.update_markers()
        self.auto_scale_y()
        self.status_label.setText('<span style="color:green;">Ready</span>')
        self.win.setCursor(_Qt.ArrowCursor)

    # ================================================================ range / curves

    def _on_range_changed(self):
        if self._range_timer is None:
            self._range_timer = QtCore.QTimer()
            self._range_timer.setSingleShot(True)
            self._range_timer.timeout.connect(self._do_range_update)
        self._range_timer.start(50)

    def _do_range_update(self):
        self._update_curves()
        self.auto_scale_y()
        self.update_markers()
        self._update_overview_region()
        self._update_threshold_position()
        self._check_selection_visibility()
        
        # Track highest left edge of window we've looked at
        x_min, x_max = self.plot_mag.viewRange()[0]
        if x_min > self._max_view_left:
            self._max_view_left = x_min

    def _visible_slice(self, f_arr, x_min, x_max, pad: float = 0.5):
        span = x_max - x_min
        lo = int(np.searchsorted(f_arr, x_min - pad * span, side='left'))
        hi = int(np.searchsorted(f_arr, x_max + pad * span, side='right'))
        return slice(lo, hi)

    def _update_curves(self):
        if not hasattr(self, 'plot_mag'):
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        sl1 = self._visible_slice(self.f1, x_min, x_max)
        sl2 = self._visible_slice(self.f2, x_min, x_max)
        if sl1.start < sl1.stop:
            self.curve_ds1_mag.setData(self.f1[sl1], self.filtered_mag1[sl1])
            self.curve_ds1_phase.setData(self.f1[sl1], self.phase1[sl1])
        if sl2.start < sl2.stop:
            self.curve_ds2_mag.setData(self.f2[sl2], self.filtered_mag2[sl2])
            self.curve_ds2_phase.setData(self.f2[sl2], self.phase2[sl2])
        # Keep right-side ViewBoxes in geometry sync after data change
        self._sync_mag2_geom()
        self._sync_phase2_geom()

    def auto_scale_y(self):
        if not hasattr(self, 'plot_mag'):
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        sl1 = self._visible_slice(self.f1, x_min, x_max, pad=0.0)
        sl2 = self._visible_slice(self.f2, x_min, x_max, pad=0.0)

        def _yrange(data, sl, upper: bool):
            if sl.start >= sl.stop:
                return None
            seg = data[sl]
            mn, mx = float(seg.min()), float(seg.max())
            span = mx - mn
            if span == 0:
                span = max(abs(mn) * 0.01, 1.0)
            if upper:
                # Data sits in the upper third (66-83%) of the axis.
                # 100% padding above, 400% below → gradual scaling + visual offset.
                return mn - 4.0 * span, mx + span
            else:
                # Data sits in the lower third (17-33%) of the axis.
                return mn - span, mx + 4.0 * span

        r1_mag = _yrange(self.filtered_mag1, sl1, upper=True)
        r2_mag = _yrange(self.filtered_mag2, sl2, upper=False)
        r1_ph  = _yrange(self.phase1, sl1, upper=True)
        r2_ph  = _yrange(self.phase2, sl2, upper=False)

        if r1_mag:
            self.plot_mag.setYRange(*r1_mag, padding=0)
        if r2_mag:
            self._vb_mag2.setYRange(*r2_mag, padding=0)
        if r1_ph:
            self.plot_phase.setYRange(*r1_ph, padding=0)
        if r2_ph:
            self._vb_phase2.setYRange(*r2_ph, padding=0)

    # ================================================================ markers

    def _nearest_idx(self, f_arr, fres: float) -> int:
        idx = int(np.searchsorted(f_arr, fres))
        if idx == 0:
            return 0
        if idx >= len(f_arr):
            return len(f_arr) - 1
        # Return whichever neighbour is closer
        if abs(f_arr[idx] - fres) < abs(f_arr[idx - 1] - fres):
            return idx
        return idx - 1

    def update_markers(self):
        if not hasattr(self, 'plot_mag'):
            return
        x_min, x_max = self.plot_mag.viewRange()[0]
        pad = (x_max - x_min) * 0.3
        vis_min = x_min - pad
        vis_max = x_max + pad

        spots_ds1_mag, spots_ds1_phase = [], []
        spots_ds2_mag, spots_ds2_phase = [], []

        for g in self.groups:
            # Skip groups with no resonances in the visible window
            all_freqs = [f for f, _ in g.entries1 + g.entries2]
            if all_freqs and not any(vis_min <= f <= vis_max for f in all_freqs):
                continue
            color = _group_color(g.group_id)
            filled_brush = pg.mkBrush(color)
            # Ambiguous: dashed pen border on filled circles
            ds1_pen = pg.mkPen(
                color[:3], width=2,
                style=_Qt.DashLine if g.ambiguous else _Qt.SolidLine,
            )
            # DS2 open squares: pen carries the color; no fill
            ds2_pen = pg.mkPen(
                color, width=2.5 if g.ambiguous else 2,
                style=_Qt.DashLine if g.ambiguous else _Qt.SolidLine,
            )

            for (fres, ridx) in g.entries1:
                idx = self._nearest_idx(self.f1, fres)
                spot = {
                    'pos': (float(self.f1[idx]), 0.0),   # y filled below
                    'brush': filled_brush,
                    'pen': ds1_pen,
                    'size': 10,
                    'data': (g.group_id, 1, fres),
                }
                spots_ds1_mag.append({
                    **spot,
                    'pos': (float(self.f1[idx]), float(self.filtered_mag1[idx])),
                })
                spots_ds1_phase.append({
                    **spot,
                    'pos': (float(self.f1[idx]), float(self.phase1[idx])),
                })

            for (fres, ridx) in g.entries2:
                idx = self._nearest_idx(self.f2, fres)
                spot = {
                    'brush': pg.mkBrush(None),   # open square
                    'pen': ds2_pen,
                    'size': 10,
                    'data': (g.group_id, 2, fres),
                }
                spots_ds2_mag.append({
                    **spot,
                    'pos': (float(self.f2[idx]), float(self.filtered_mag2[idx])),
                })
                spots_ds2_phase.append({
                    **spot,
                    'pos': (float(self.f2[idx]), float(self.phase2[idx])),
                })

        self.scatter_ds1_mag.setData(spots_ds1_mag)
        self.scatter_ds2_mag.setData(spots_ds2_mag)
        self.scatter_ds1_phase.setData(spots_ds1_phase)
        self.scatter_ds2_phase.setData(spots_ds2_phase)
        self._sync_mag2_geom()
        self._sync_phase2_geom()
        self._update_selection_ring()

    def _update_selection_ring(self):
        """Draw white rings around all selected resonances."""
        if not self._selected_resonances:
            self.scatter_sel_ds1_mag.setData([])
            self.scatter_sel_ds2_mag.setData([])
            self.scatter_sel_ds1_phase.setData([])
            self.scatter_sel_ds2_phase.setData([])
            return

        sel_ds1_mag, sel_ds1_phase = [], []
        sel_ds2_mag, sel_ds2_phase = [], []
        
        for (group_id, dataset, fres) in self._selected_resonances:
            f_arr = self.f1 if dataset == 1 else self.f2
            mag_arr = self.filtered_mag1 if dataset == 1 else self.filtered_mag2
            ph_arr = self.phase1 if dataset == 1 else self.phase2
            
            idx = self._nearest_idx(f_arr, fres)
            mag_spot = {'pos': (float(f_arr[idx]), float(mag_arr[idx]))}
            phase_spot = {'pos': (float(f_arr[idx]), float(ph_arr[idx]))}
            
            if dataset == 1:
                sel_ds1_mag.append(mag_spot)
                sel_ds1_phase.append(phase_spot)
            else:
                sel_ds2_mag.append(mag_spot)
                sel_ds2_phase.append(phase_spot)

        self.scatter_sel_ds1_mag.setData(sel_ds1_mag)
        self.scatter_sel_ds2_mag.setData(sel_ds2_mag)
        self.scatter_sel_ds1_phase.setData(sel_ds1_phase)
        self.scatter_sel_ds2_phase.setData(sel_ds2_phase)
    
    def _update_selection_label(self):
        """Update the selection label to show selected resonances."""
        if not self._selected_resonances:
            self.sel_label.setText('Selected: (none)')
            return
        
        if len(self._selected_resonances) == 1:
            group_id, dataset, fres = next(iter(self._selected_resonances))
            g = self._find_group(group_id)
            mapping = g.mapping_str() if g else '?'
            ambig = '  [ambiguous]' if (g and g.ambiguous) else ''
            self.sel_label.setText(
                f'Selected: Group {group_id} ({mapping}){ambig}, '
                f'DS{dataset}, {fres / 1e6:.4f} MHz'
            )
        else:
            # Multiple selections
            group_ids = set(gid for gid, _, _ in self._selected_resonances)
            ds1_count = sum(1 for _, ds, _ in self._selected_resonances if ds == 1)
            ds2_count = sum(1 for _, ds, _ in self._selected_resonances if ds == 2)
            self.sel_label.setText(
                f'Selected: {len(self._selected_resonances)} resonances '
                f'(DS1: {ds1_count}, DS2: {ds2_count}) in {len(group_ids)} groups'
            )

    def _ask_reuse_res_idx(
        self, 
        new_freq: float, 
        ds: int, 
        removed_list: List[Tuple[float, int]]
    ):
        """
        Show a dialog asking user to choose from removed resonances or create new.
        
        Parameters:
        new_freq (float): Frequency of the resonance being added (Hz).
        ds (int): Dataset number (1 or 2).
        removed_list (list): List of (freq, res_idx) tuples for removed resonances
            in the visible window, sorted by distance from new_freq.
        
        Returns:
        int or str or None: res_idx to reuse (int), 'new', or None for cancel
        """
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle('Resonance Re-Addition')
        dlg.setModal(True)
        layout = QtWidgets.QVBoxLayout(dlg)
        
        msg = QtWidgets.QLabel(
            f'<b>Adding resonance in a window with removed resonances</b><br><br>'
            f'New frequency: <b>{new_freq / 1e6:.4f} MHz</b> (DS{ds})<br><br>'
            f'Choose a removed res_idx to reuse, or create a new one:'
        )
        msg.setTextFormat(_Qt.RichText)
        msg.setWordWrap(True)
        layout.addWidget(msg)
        
        # Radio button group for selecting removed resonances
        radio_group = QtWidgets.QButtonGroup(dlg)
        radio_layout = QtWidgets.QVBoxLayout()
        
        # Add radio button for each removed resonance
        for old_freq, old_idx in removed_list:
            freq_diff_khz = abs(old_freq - new_freq) / 1e3
            radio = QtWidgets.QRadioButton(
                f'Reuse res_idx {old_idx}  '
                f'(removed at {old_freq / 1e6:.4f} MHz, '
                f'Δ = {freq_diff_khz:.1f} kHz)'
            )
            radio.setProperty('res_idx', old_idx)
            radio_group.addButton(radio)
            radio_layout.addWidget(radio)
        
        # Set first option as default
        if radio_group.buttons():
            radio_group.buttons()[0].setChecked(True)
        
        layout.addLayout(radio_layout)
        layout.addSpacing(10)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton('OK')
        ok_btn.setDefault(True)
        new_btn = QtWidgets.QPushButton('Create New res_idx')
        cancel_btn = QtWidgets.QPushButton('Cancel')
        
        ok_btn.clicked.connect(lambda: dlg.done(1))
        new_btn.clicked.connect(lambda: dlg.done(2))
        cancel_btn.clicked.connect(lambda: dlg.done(0))
        
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(new_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        result = dlg.exec_()
        if result == 1:
            # OK - return selected res_idx
            checked = radio_group.checkedButton()
            if checked:
                return checked.property('res_idx')
            return 'new'  # Fallback if nothing checked
        elif result == 2:
            return 'new'
        else:
            return None  # Cancel

    # ================================================================ group helpers

    def _find_group(self, group_id: int) -> Optional[MatchGroup]:
        for g in self.groups:
            if g.group_id == group_id:
                return g
        return None

    def _save_undo_state(self):
        self.undo_stack.append(copy.deepcopy(self.groups))
        # Limit undo stack size
        if len(self.undo_stack) > self._max_undo_stack:
            self.undo_stack.pop(0)

    def _sort_groups(self):
        self.groups.sort(key=lambda g: g.center_freq())

    def _new_group_id(self) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        return gid

    def _new_res_idx(self) -> int:
        idx = self._next_new_idx
        self._next_new_idx += 1
        return idx

    def _remove_empty_groups(self):
        self.groups = [g for g in self.groups if g.entries1 or g.entries2]

    def _sorted_groups(self) -> List[MatchGroup]:
        return sorted(self.groups, key=lambda g: g.center_freq())

    # ================================================================ mouse / click

    def _on_scatter_clicked(self, pts, ev, ds: int):
        """
        Select clicked scatter point(s). Ctrl+click adds to selection.

        Parameters:
        pts (list): List of scatter points under the cursor.
        ev (MouseClickEvent): The mouse click event.
        ds (int): Dataset number (1 or 2).

        Returns:
        None
        """
        try:
            button = ev.button()
        except AttributeError:
            button = _Qt.LeftButton   # older pyqtgraph without ev
        if button != _Qt.LeftButton:
            return
        if len(pts) == 0:
            return

        pt = pts[0]
        data = pt.data()
        if data is None:
            return

        group_id, dataset, fres = data
        selection_tuple = (group_id, dataset, fres)
        
        # Check for Ctrl modifier
        try:
            modifiers = ev.modifiers()
        except AttributeError:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
        
        if modifiers & _Qt.ControlModifier:
            # Ctrl+click: toggle selection
            if selection_tuple in self._selected_resonances:
                self._selected_resonances.remove(selection_tuple)
            else:
                self._selected_resonances.add(selection_tuple)
        else:
            # Normal click: replace selection
            self._selected_resonances = {selection_tuple}
        
        self._update_selection_label()
        self.update_markers()
        # Suppress the scene's add-resonance handler for this click event
        self._scatter_just_clicked = True

    def _on_scene_clicked(self, ev):
        """
        Shift+Left-click on empty space → add resonance to dataset determined by Y-position proximity.
        Ctrl+Right-click → set re-match threshold at clicked frequency.
        Right-click → PyQtGraph context menu.
        Clicking on a scatter point → selection (handled by scatter signal;
        the _scatter_just_clicked flag suppresses add here).

        Parameters:
        ev (MouseClickEvent): The mouse click event from the scene.

        Returns:
        None
        """
        if self._scatter_just_clicked:
            self._scatter_just_clicked = False
            return

        pos = ev.scenePos()
        modifiers = ev.modifiers()

        # Ctrl+Right-click → manually set re-match threshold
        if ev.button() == _Qt.RightButton:
            if modifiers & _Qt.ControlModifier:
                result = self._scene_pos_to_freq_and_dataset(pos)
                if result is not None:
                    freq, _ = result
                    self._rematch_freq_threshold = freq
                    self._rematch_line_mag.setPos(freq)
                    self._rematch_line_phase.setPos(freq)
                    self._rematch_line_mag.setVisible(True)
                    self._rematch_line_phase.setVisible(True)
                    self._last_edit_freq = None  # Disable auto-adjustment
                    self._threshold_pin_right = False  # Manual mode, fixed at frequency
                    self.log(f'Re-match threshold manually set at {freq / 1e6:.4f} MHz.')
                ev.accept()  # Consume event to prevent PyQtGraph context menu
                return
            else:
                # Plain right-click → PyQtGraph context menu (do nothing, let it handle)
                return

        if ev.button() != _Qt.LeftButton:
            return
        
        # Shift+left-click on empty space → add resonance to dataset determined by Y-position
        if modifiers & _Qt.ShiftModifier:
            result = self._scene_pos_to_freq_and_dataset(pos)
            if result is not None:
                freq, ds = result
                self.add_resonance(freq, ds)
                return
        
        # Plain left-click → do nothing (click outside plots or no modifiers)

    def _scene_pos_to_freq_and_dataset(self, scene_pos):
        """
        Convert a scene position to a frequency and determine dataset based on Y-proximity.

        Checks if the click is closer to DS1 or DS2 data by comparing the click Y-coordinate
        to the actual data values at that frequency.

        Parameters:
        scene_pos (QPointF): Position in scene coordinates.

        Returns:
        tuple: (freq, dataset) where freq is float in Hz and dataset is 1 or 2,
               or None if click is outside both plots
        """
        # Check magnitude plot
        if self.plot_mag.sceneBoundingRect().contains(scene_pos):
            # Get position in DS1 (left axis) coordinates
            vb_pos_ds1 = self.plot_mag.vb.mapSceneToView(scene_pos)
            freq = float(vb_pos_ds1.x())
            click_y_ds1 = float(vb_pos_ds1.y())
            
            # Get position in DS2 (right axis) coordinates
            vb_pos_ds2 = self._vb_mag2.mapSceneToView(scene_pos)
            click_y_ds2 = float(vb_pos_ds2.y())
            
            # Get data values at this frequency
            idx1 = self._nearest_idx(self.f1, freq)
            idx2 = self._nearest_idx(self.f2, freq)
            data_y_ds1 = float(self.filtered_mag1[idx1])
            data_y_ds2 = float(self.filtered_mag2[idx2])
            
            # Determine which dataset is closer
            dist_ds1 = abs(click_y_ds1 - data_y_ds1)
            dist_ds2 = abs(click_y_ds2 - data_y_ds2)
            ds = 1 if dist_ds1 < dist_ds2 else 2
            
            return (freq, ds)
        
        # Check phase plot
        if self.plot_phase.sceneBoundingRect().contains(scene_pos):
            # Get position in DS1 (left axis) coordinates
            vb_pos_ds1 = self.plot_phase.vb.mapSceneToView(scene_pos)
            freq = float(vb_pos_ds1.x())
            click_y_ds1 = float(vb_pos_ds1.y())
            
            # Get position in DS2 (right axis) coordinates
            vb_pos_ds2 = self._vb_phase2.mapSceneToView(scene_pos)
            click_y_ds2 = float(vb_pos_ds2.y())
            
            # Get data values at this frequency
            idx1 = self._nearest_idx(self.f1, freq)
            idx2 = self._nearest_idx(self.f2, freq)
            data_y_ds1 = float(self.phase1[idx1])
            data_y_ds2 = float(self.phase2[idx2])
            
            # Determine which dataset is closer
            dist_ds1 = abs(click_y_ds1 - data_y_ds1)
            dist_ds2 = abs(click_y_ds2 - data_y_ds2)
            ds = 1 if dist_ds1 < dist_ds2 else 2
            
            return (freq, ds)
        
        return None

    def _show_context_menu(self, screen_pos, click_freq: float):
        """
        Build and show a right-click context menu near the clicked frequency.

        Parameters:
        screen_pos (QPointF): Screen position for the menu.
        click_freq (float): Frequency in Hz near the click position.

        Returns:
        None
        """
        x_range = self.plot_mag.viewRange()[0]
        visible = [
            g for g in self.groups
            if x_range[0] <= g.center_freq() <= x_range[1]
        ]
        candidates = visible if visible else self.groups
        if not candidates:
            return

        near = min(candidates, key=lambda g: abs(g.center_freq() - click_freq))

        sorted_gs = self._sorted_groups()
        try:
            pos_in_sorted = next(
                i for i, g in enumerate(sorted_gs)
                if g.group_id == near.group_id
            )
        except StopIteration:
            pos_in_sorted = None

        menu = QtWidgets.QMenu()
        menu.addSection(
            f'Group {near.group_id}  ({near.mapping_str()})'
            f'{"  [ambiguous]" if near.ambiguous else ""}'
        )

        act_del1 = menu.addAction('Delete nearest DS1 resonance')
        act_del2 = menu.addAction('Delete nearest DS2 resonance')
        menu.addSeparator()
        act_unlink1 = menu.addAction('Unlink nearest DS1 from group')
        act_unlink2 = menu.addAction('Unlink nearest DS2 from group')
        menu.addSeparator()
        act_merge_l = menu.addAction('Merge group ← left')
        act_merge_r = menu.addAction('Merge group → right')
        menu.addSeparator()
        act_ambig = menu.addAction(
            'Clear ambiguous flag' if near.ambiguous else 'Flag as ambiguous'
        )

        act_del1.setEnabled(bool(near.entries1))
        act_del2.setEnabled(bool(near.entries2))
        act_unlink1.setEnabled(bool(near.entries1))
        act_unlink2.setEnabled(bool(near.entries2))
        act_merge_l.setEnabled(
            pos_in_sorted is not None and pos_in_sorted > 0
        )
        act_merge_r.setEnabled(
            pos_in_sorted is not None
            and pos_in_sorted < len(sorted_gs) - 1
        )

        # Convert to QPoint for exec_
        if hasattr(screen_pos, 'toPoint'):
            qp = screen_pos.toPoint()
        else:
            qp = QtCore.QPoint(int(screen_pos.x()), int(screen_pos.y()))

        chosen = menu.exec_(qp)

        if chosen is None:
            return

        if chosen == act_del1 and near.entries1:
            self._save_undo_state()
            entry = min(near.entries1, key=lambda e: abs(e[0] - click_freq))
            near.entries1.remove(entry)
            # Track removed resonance for smart re-addition
            self._removed_resonances.append((entry[0], entry[1], 1))
            self._remove_empty_groups()
            self._clear_selection_if_gone()
            self.update_markers()
            self._last_edit_freq = entry[0]  # Track for auto-threshold
            self._update_threshold_after_edit(entry[0])
            self.log(
                f'Deleted DS1 resonance at {entry[0] / 1e6:.4f} MHz '
                f'(res_idx={entry[1]}) from group {near.group_id}'
            )

        elif chosen == act_del2 and near.entries2:
            self._save_undo_state()
            entry = min(near.entries2, key=lambda e: abs(e[0] - click_freq))
            near.entries2.remove(entry)
            # Track removed resonance for smart re-addition
            self._removed_resonances.append((entry[0], entry[1], 2))
            self._remove_empty_groups()
            self._clear_selection_if_gone()
            self.update_markers()
            self._last_edit_freq = entry[0]  # Track for auto-threshold
            self._update_threshold_after_edit(entry[0])
            self.log(
                f'Deleted DS2 resonance at {entry[0] / 1e6:.4f} MHz '
                f'(res_idx={entry[1]}) from group {near.group_id}'
            )

        elif chosen == act_unlink1 and near.entries1:
            entry = min(near.entries1, key=lambda e: abs(e[0] - click_freq))
            self._do_unlink(near, entry, ds=1)

        elif chosen == act_unlink2 and near.entries2:
            entry = min(near.entries2, key=lambda e: abs(e[0] - click_freq))
            self._do_unlink(near, entry, ds=2)

        elif chosen == act_merge_l and pos_in_sorted is not None and pos_in_sorted > 0:
            left_g = sorted_gs[pos_in_sorted - 1]
            self._do_merge(left_g.group_id, near.group_id)

        elif chosen == act_merge_r and pos_in_sorted is not None:
            right_g = sorted_gs[pos_in_sorted + 1]
            self._do_merge(near.group_id, right_g.group_id)

        elif chosen == act_ambig:
            self._save_undo_state()
            near.ambiguous = not near.ambiguous
            self.update_markers()
            self.log(f'Group {near.group_id} ambiguous = {near.ambiguous}')

    # ================================================================ resonance ops

    def add_resonance(self, freq: float, ds: int):
        """
        Add a new resonance at the given frequency.
        
        If there are removed resonances in the current visible window, prompts
        the user to choose between reusing an old res_idx or creating a new one.
        
        Parameters:
        freq (float): Frequency in Hz where resonance should be added.
        ds (int): Dataset number (1 or 2).
        
        Returns:
        None
        """
        # Get current visible window
        x_min, x_max = self.plot_mag.viewRange()[0]
        
        # Check for removed resonances in the visible window
        visible_removed = [
            (f, idx) for f, idx, dataset in self._removed_resonances
            if dataset == ds and x_min <= f <= x_max
        ]
        
        if visible_removed:
            # Sort by distance from clicked frequency
            visible_removed.sort(key=lambda x: abs(x[0] - freq))
            
            # Ask user to choose from removed resonances or create new
            choice = self._ask_reuse_res_idx(freq, ds, visible_removed)
            
            if choice is None:
                # User cancelled
                return
            elif choice == 'new':
                ridx = self._new_res_idx()
            else:
                # choice is the res_idx to reuse
                ridx = choice
                # Remove from removed list since we're re-adding it
                self._removed_resonances = [
                    (f, idx, dset) for f, idx, dset in self._removed_resonances
                    if not (dset == ds and idx == ridx)
                ]
        else:
            ridx = self._new_res_idx()
        
        self._save_undo_state()
        gid = self._new_group_id()
        entry = (float(freq), ridx)
        if ds == 1:
            g = MatchGroup(group_id=gid, entries1=[entry], entries2=[])
        else:
            g = MatchGroup(group_id=gid, entries1=[], entries2=[entry])
        self.groups.append(g)
        self._sort_groups()
        self.update_markers()
        self._last_edit_freq = freq  # Track for auto-threshold
        self._update_threshold_after_edit(freq)
        self.log(
            f'Added DS{ds} resonance at {freq / 1e6:.4f} MHz '
            f'(res_idx={ridx}, group={gid})'
        )

    def _do_unlink(self, g: MatchGroup, entry: tuple, ds: int):
        self._save_undo_state()
        if ds == 1:
            g.entries1.remove(entry)
        else:
            g.entries2.remove(entry)
        new_gid = self._new_group_id()
        if ds == 1:
            new_g = MatchGroup(group_id=new_gid, entries1=[entry], entries2=[])
        else:
            new_g = MatchGroup(group_id=new_gid, entries1=[], entries2=[entry])
        self.groups.append(new_g)
        self._remove_empty_groups()
        self._sort_groups()
        self._clear_selection_if_gone()
        self.update_markers()
        self._last_edit_freq = entry[0]  # Track for auto-threshold
        self._update_threshold_after_edit(entry[0])
        self.log(
            f'Unlinked DS{ds} at {entry[0] / 1e6:.4f} MHz '
            f'from group {g.group_id} → new group {new_gid}'
        )

    def _do_merge(self, gid_a: int, gid_b: int):
        """
        Absorb group gid_b into group gid_a.

        Parameters:
        gid_a (int): Group ID of the target group.
        gid_b (int): Group ID of the group to merge in.

        Returns:
        None
        """
        self._save_undo_state()
        ga = self._find_group(gid_a)
        gb = self._find_group(gid_b)
        if ga is None or gb is None:
            return
        ga.entries1.extend(gb.entries1)
        ga.entries2.extend(gb.entries2)
        if gb.ambiguous:
            ga.ambiguous = True
        self.groups.remove(gb)
        self._sort_groups()
        self._clear_selection_if_gone()
        self.update_markers()
        # Track center frequency for auto-threshold
        all_freqs = [f for f, _ in ga.entries1 + ga.entries2]
        if all_freqs:
            self._last_edit_freq = float(np.mean(all_freqs))
            self._update_threshold_after_edit(self._last_edit_freq)
        self.log(
            f'Merged group {gid_b} into {gid_a} → {ga.mapping_str()}'
        )

    def _clear_selection_if_gone(self):
        """Remove any selected resonances that no longer exist."""
        valid_selections = set()
        for (group_id, dataset, fres) in self._selected_resonances:
            g = self._find_group(group_id)
            if g is not None:
                entries = g.entries1 if dataset == 1 else g.entries2
                if any(e[0] == fres for e in entries):
                    valid_selections.add((group_id, dataset, fres))
        self._selected_resonances = valid_selections
        self._update_selection_label()

    def _clear_selection(self):
        self._selected_resonances = set()
        self._update_selection_label()
    
    def _check_selection_visibility(self):
        """
        Remove selected resonances that are out of view.
        
        Called when window pans to auto-deselect resonances that scroll off screen.
        """
        if not self._selected_resonances:
            return
        
        x_min, x_max = self.plot_mag.viewRange()[0]
        pad = (x_max - x_min) * 0.3  # Same padding as update_markers
        vis_min = x_min - pad
        vis_max = x_max + pad
        
        valid_selections = set()
        for (group_id, dataset, fres) in self._selected_resonances:
            if vis_min <= fres <= vis_max:
                valid_selections.add((group_id, dataset, fres))
        
        if len(valid_selections) != len(self._selected_resonances):
            deselected_count = len(self._selected_resonances) - len(valid_selections)
            self._selected_resonances = valid_selections
            self._update_selection_label()
            if deselected_count > 0:
                self.log(f'Deselected {deselected_count} out-of-view resonances')

    # ================================================================ keyboard actions

    def toggle_active_dataset(self):
        self.active_dataset = 2 if self.active_dataset == 1 else 1
        self._refresh_active_label()
        self.log(f'Active dataset: DS{self.active_dataset}')

    def merge_groups(self):
        """
        Merge all groups containing selected resonances (F key).
        
        Takes all resonances from the groups of selected resonances and
        combines them into a single group.
        
        Example: groups ((1, 2, 3), (1)) and ((4), (2))
        Select: DS1: 2, 4; DS2: 2
        Result: ((1,2,3,4), (1,2))
        
        Parameters:
        None
        
        Returns:
        None
        """
        if not self._selected_resonances:
            self.log('F: Select resonances first (Ctrl+click for multi-select).')
            return
        
        # Find all groups containing selected resonances
        group_ids = set(gid for gid, _, _ in self._selected_resonances)
        groups_to_merge = [self._find_group(gid) for gid in group_ids]
        groups_to_merge = [g for g in groups_to_merge if g is not None]
        
        if len(groups_to_merge) < 2:
            self.log('F: Selected resonances are already in the same group.')
            return
        
        self._save_undo_state()
        
        # Merge all into the first group
        target_group = groups_to_merge[0]
        for g in groups_to_merge[1:]:
            target_group.entries1.extend(g.entries1)
            target_group.entries2.extend(g.entries2)
            if g.ambiguous:
                target_group.ambiguous = True
            self.groups.remove(g)
        
        # Auto-set ambiguous if multi-res in either dataset
        if len(target_group.entries1) > 1 or len(target_group.entries2) > 1:
            target_group.ambiguous = True
        
        self._sort_groups()
        self.update_markers()
        
        # Track for auto-threshold
        all_freqs = [f for f, _ in target_group.entries1 + target_group.entries2]
        if all_freqs:
            self._last_edit_freq = float(np.mean(all_freqs))
            self._update_threshold_after_edit(self._last_edit_freq)
        
        self.log(f'Merged {len(groups_to_merge)} groups → Group {target_group.group_id} ({target_group.mapping_str()})')

    def merge_selected_only(self):
        """
        Merge only selected resonances into new group (W key).
        
        Unlinks selected resonances from their groups and merges them
        into a single new group. Other resonances from those groups
        stay in their original groups (or get their own if left alone).
        
        Example: groups ((1, 2, 3), (1)) and ((4), (2))
        Select: DS1: 2, 4; DS2: 2
        Result: ((2,4), (2)), ((1,3), (1))
        
        Parameters:
        None
        
        Returns:
        None
        """
        if not self._selected_resonances:
            self.log('W: Select resonances first (Ctrl+click for multi-select).')
            return
        
        self._save_undo_state()
        
        # Remove selected entries from their groups
        entries1_to_merge = []
        entries2_to_merge = []
        
        for (group_id, dataset, fres) in self._selected_resonances:
            g = self._find_group(group_id)
            if g is None:
                continue
            
            if dataset == 1:
                entry = next((e for e in g.entries1 if e[0] == fres), None)
                if entry:
                    g.entries1.remove(entry)
                    entries1_to_merge.append(entry)
            else:
                entry = next((e for e in g.entries2 if e[0] == fres), None)
                if entry:
                    g.entries2.remove(entry)
                    entries2_to_merge.append(entry)
        
        # Create new group with selected resonances
        new_gid = self._new_group_id()
        new_group = MatchGroup(
            group_id=new_gid,
            entries1=entries1_to_merge,
            entries2=entries2_to_merge,
            ambiguous=len(entries1_to_merge) > 1 or len(entries2_to_merge) > 1
        )
        self.groups.append(new_group)
        
        self._remove_empty_groups()
        self._sort_groups()
        self.update_markers()
        
        # Track for auto-threshold
        all_freqs = [f for f, _ in entries1_to_merge + entries2_to_merge]
        if all_freqs:
            self._last_edit_freq = float(np.mean(all_freqs))
            self._update_threshold_after_edit(self._last_edit_freq)
        
        self.log(f'Merged {len(self._selected_resonances)} selected resonances → Group {new_gid} ({new_group.mapping_str()})')

    def unlink_selected(self):
        """
        Put all selected resonances into their own separate groups (U key).
        
        Bound to the U key.

        Parameters:
        None

        Returns:
        None
        """
        if not self._selected_resonances:
            self.log('U: Select resonances first (Ctrl+click for multi-select).')
            return
        
        self._save_undo_state()
        
        # Unlink each selected resonance
        for (group_id, dataset, fres) in self._selected_resonances:
            g = self._find_group(group_id)
            if g is None:
                continue
            
            entries = g.entries1 if dataset == 1 else g.entries2
            entry = next((e for e in entries if e[0] == fres), None)
            if entry is None:
                continue
            
            # Remove from current group
            if dataset == 1:
                g.entries1.remove(entry)
            else:
                g.entries2.remove(entry)
            
            # Create new group
            new_gid = self._new_group_id()
            if dataset == 1:
                new_g = MatchGroup(group_id=new_gid, entries1=[entry], entries2=[], ambiguous=False)
            else:
                new_g = MatchGroup(group_id=new_gid, entries1=[], entries2=[entry], ambiguous=False)
            self.groups.append(new_g)
        
        self._remove_empty_groups()
        self._sort_groups()
        self._clear_selection()
        self.update_markers()
        
        self.log(f'Unlinked {len(self._selected_resonances)} resonances into separate groups')

    def toggle_ambiguous(self):
        """
        Toggle the ambiguous flag on selected group (B key).
        
        Only works for 1:1 matches. Groups with multiple resonances in
        either dataset are automatically marked ambiguous.

        Parameters:
        None

        Returns:
        None
        """
        if not self._selected_resonances:
            self.log('B: Select a resonance first.')
            return
        
        if len(self._selected_resonances) > 1:
            self.log('B: Can only toggle ambiguous on a single group. Select one resonance.')
            return
        
        group_id, _, _ = next(iter(self._selected_resonances))
        g = self._find_group(group_id)
        if g is None:
            return
        
        # Check if this is a 1:1 match
        if len(g.entries1) != 1 or len(g.entries2) != 1:
            self.log(f'B: Group {g.group_id} ({g.mapping_str()}) is not 1:1. Ambiguous flag is auto-managed for multi-resonance groups.')
            return
        
        self._save_undo_state()
        g.ambiguous = not g.ambiguous
        self.update_markers()
        self.log(f'Group {g.group_id} (1:1) ambiguous = {g.ambiguous}')

    def undo(self):
        if not self.undo_stack:
            self.log('Nothing to undo.')
            return
        self.groups = self.undo_stack.pop()
        # Clear removed resonances tracking after undo to avoid confusion
        self._removed_resonances = []
        self._clear_selection_if_gone()
        self.update_markers()
        self.log('Undo. (Removed resonance tracking cleared.)')

    def _update_threshold_position(self):
        """
        Auto-adjust threshold position based on window and edit history.
        
        Modes:
        1. Manual mode (_last_edit_freq is None): user set it, don't auto-adjust
        2. Pinned to right edge (_threshold_pin_right=True): stick to right edge as window pans
        3. Fixed frequency (_threshold_pin_right=False): stay at frequency until scrolled out
        
        On initialization, sets threshold to right edge of window (pinned mode).
        """
        x_min, x_max = self.plot_mag.viewRange()[0]
        
        if self._rematch_freq_threshold is None:
            # Initialize at right edge of window (pinned mode)
            self._rematch_freq_threshold = x_max
            self._rematch_line_mag.setPos(x_max)
            self._rematch_line_phase.setPos(x_max)
            self._rematch_line_mag.setVisible(True)
            self._rematch_line_phase.setVisible(True)
            self._threshold_pin_right = True
            return
        
        # If user manually set threshold, don't auto-adjust
        if self._last_edit_freq is None:
            return
        
        # If pinned to right edge, keep it there
        if self._threshold_pin_right:
            self._rematch_freq_threshold = x_max
            self._rematch_line_mag.setPos(x_max)
            self._rematch_line_phase.setPos(x_max)
            return
        
        # Fixed frequency mode: check if window scrolled past threshold
        if x_min > self._rematch_freq_threshold:
            # Threshold scrolled out of view, snap to right edge and pin
            self._rematch_freq_threshold = x_max
            self._rematch_line_mag.setPos(x_max)
            self._rematch_line_phase.setPos(x_max)
            self._threshold_pin_right = True

    def _update_threshold_after_edit(self, edit_freq: float):
        """
        Update threshold after user makes an edit.
        
        Moves threshold to the right of the edited frequency and pins it there
        (fixed frequency mode until scrolled out of view).
        
        Parameters:
        edit_freq (float): Frequency where edit was made (Hz).
        """
        if self._last_edit_freq is None:
            # User manually set threshold, don't auto-adjust
            return
        
        # Move threshold to right of edit and pin at this frequency
        new_threshold = edit_freq * 1.001  # Slightly to the right
        if new_threshold > self._rematch_freq_threshold:
            self._rematch_freq_threshold = new_threshold
            self._rematch_line_mag.setPos(new_threshold)
            self._rematch_line_phase.setPos(new_threshold)
            self._threshold_pin_right = False  # Pin to frequency, not right edge

    def delete_selected(self):
        """
        Delete all selected resonances (S key).
        
        Bound to the S key.
        
        Parameters:
        None
        
        Returns:
        None
        """
        if not self._selected_resonances:
            self.log('S: Select resonances first (Ctrl+click for multi-select).')
            return
        
        self._save_undo_state()
        
        # Delete all selected resonances
        for (group_id, dataset, fres) in self._selected_resonances:
            g = self._find_group(group_id)
            if g is None:
                continue
            
            entries = g.entries1 if dataset == 1 else g.entries2
            entry = next((e for e in entries if e[0] == fres), None)
            if entry is None:
                continue
            
            if dataset == 1:
                g.entries1.remove(entry)
            else:
                g.entries2.remove(entry)
            
            # Track removed resonance for smart re-addition
            self._removed_resonances.append((entry[0], entry[1], dataset))
        
        self._remove_empty_groups()
        
        # Track for auto-threshold (use mean of deleted frequencies)
        all_freqs = [fres for _, _, fres in self._selected_resonances]
        if all_freqs:
            self._last_edit_freq = float(np.mean(all_freqs))
            self._update_threshold_after_edit(self._last_edit_freq)
        
        self.log(f'Deleted {len(self._selected_resonances)} resonances')
        self._clear_selection()
        self.update_markers()

    def trigger_rematch(self):
        """
        Re-run automatic matching on all groups above the threshold frequency.
        
        Uses the current init_match method stored in self (defaults to 'sorted').
        Groups below the threshold are left unchanged. Groups above the threshold
        are dissolved and re-matched using the same algorithm as initialization.

        Bound to the T key.

        Parameters:
        None

        Returns:
        None
        """
        if self._rematch_freq_threshold is None:
            self.log('T: Set re-match threshold first (Shift+click).')
            return

        self._save_undo_state()
        self.win.setCursor(_Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

        thresh = self._rematch_freq_threshold
        
        # Separate groups into below-threshold (keep) and above-threshold (rematch)
        groups_keep = []
        entries1_rematch = []
        entries2_rematch = []
        
        for g in self.groups:
            cf = g.center_freq()
            if cf < thresh:
                groups_keep.append(g)
            else:
                entries1_rematch.extend(g.entries1)
                entries2_rematch.extend(g.entries2)
        
        n_rematch_groups = len(self.groups) - len(groups_keep)
        n_rematch_res = len(entries1_rematch) + len(entries2_rematch)
        
        if not entries1_rematch and not entries2_rematch:
            self.log(f'T: No groups above {thresh / 1e6:.4f} MHz to re-match.')
            self.win.setCursor(_Qt.ArrowCursor)
            return
        
        # Extract frequency and index arrays for re-matching
        if entries1_rematch:
            fres1_rematch = np.array([f for f, _ in entries1_rematch], dtype=np.float64)
            ridx1_rematch = np.array([i for _, i in entries1_rematch], dtype=np.int64)
        else:
            fres1_rematch = np.array([], dtype=np.float64)
            ridx1_rematch = np.array([], dtype=np.int64)
        
        if entries2_rematch:
            fres2_rematch = np.array([f for f, _ in entries2_rematch], dtype=np.float64)
            ridx2_rematch = np.array([i for _, i in entries2_rematch], dtype=np.int64)
        else:
            fres2_rematch = np.array([], dtype=np.float64)
            ridx2_rematch = np.array([], dtype=np.int64)
        
        # Run the matching algorithm
        # Use 'sorted' as default - you can expose this as a parameter if needed
        init_method = getattr(self, '_init_match_method', 'sorted')
        if init_method == 'sorted':
            new_groups, next_gid = self._init_sorted_from_arrays(
                fres1_rematch, ridx1_rematch, fres2_rematch, ridx2_rematch
            )
        else:
            new_groups, next_gid = self._init_nearest_from_arrays(
                fres1_rematch, ridx1_rematch, fres2_rematch, ridx2_rematch
            )
        
        # Renumber the new groups to avoid ID collision
        for g in new_groups:
            g.group_id = self._new_group_id()
        
        # Combine kept and re-matched groups
        self.groups = groups_keep + new_groups
        self._sort_groups()
        self._clear_selection_if_gone()
        self.update_markers()
        
        self.log(
            f'Re-matched {n_rematch_groups} groups ({n_rematch_res} resonances) '
            f'above {thresh / 1e6:.4f} MHz → {len(new_groups)} new groups.'
        )
        self.win.setCursor(_Qt.ArrowCursor)

    def _init_sorted_from_arrays(self, fres1, ridx1, fres2, ridx2):
        """
        Helper: perform sorted matching on provided arrays.
        
        Returns (groups, next_gid) but does NOT assign final group IDs.
        """
        idx1 = np.argsort(fres1)
        idx2 = np.argsort(fres2)
        fres1_s = fres1[idx1];  ridx1_s = ridx1[idx1]
        fres2_s = fres2[idx2];  ridx2_s = ridx2[idx2]

        n1, n2 = len(fres1_s), len(fres2_s)
        n_common = min(n1, n2)
        groups = []
        gid = 0

        for i in range(n_common):
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                entries2=[(float(fres2_s[i]), int(ridx2_s[i]))],
            ))
            gid += 1

        for i in range(n_common, n1):
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                entries2=[],
            ))
            gid += 1

        for i in range(n_common, n2):
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[],
                entries2=[(float(fres2_s[i]), int(ridx2_s[i]))],
            ))
            gid += 1

        groups.sort(key=lambda g: g.center_freq())
        return groups, gid

    def _init_nearest_from_arrays(self, fres1, ridx1, fres2, ridx2):
        """
        Helper: perform nearest-neighbor matching on provided arrays.
        
        Returns (groups, next_gid) but does NOT assign final group IDs.
        """
        idx1 = np.argsort(fres1)
        idx2 = np.argsort(fres2)
        fres1_s = fres1[idx1];  ridx1_s = ridx1[idx1]
        fres2_s = fres2[idx2];  ridx2_s = ridx2[idx2]

        n1, n2 = len(fres1_s), len(fres2_s)
        n_common = min(n1, n2)

        offset = float(np.median(fres2_s[:n_common] - fres1_s[:n_common])) \
            if n_common > 0 else 0.0
        fres2_corr = fres2_s - offset

        if n1 > 1:
            threshold = 3.0 * float(np.median(np.diff(fres1_s)))
        elif n2 > 1:
            threshold = 3.0 * float(np.median(np.diff(fres2_s)))
        else:
            threshold = np.inf

        matched1: set = set()
        matched2: set = set()
        pairs = []

        for i, f1v in enumerate(fres1_s):
            if n2 == 0:
                break
            dists = np.abs(fres2_corr - f1v)
            j = int(np.argmin(dists))
            if dists[j] < threshold and j not in matched2:
                pairs.append((i, j))
                matched1.add(i)
                matched2.add(j)

        gid = 0
        groups = []
        for i, j in pairs:
            groups.append(MatchGroup(
                group_id=gid,
                entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                entries2=[(float(fres2_s[j]), int(ridx2_s[j]))],
            ))
            gid += 1
        for i in range(n1):
            if i not in matched1:
                groups.append(MatchGroup(
                    group_id=gid,
                    entries1=[(float(fres1_s[i]), int(ridx1_s[i]))],
                    entries2=[],
                ))
                gid += 1
        for j in range(n2):
            if j not in matched2:
                groups.append(MatchGroup(
                    group_id=gid,
                    entries1=[],
                    entries2=[(float(fres2_s[j]), int(ridx2_s[j]))],
                ))
                gid += 1

        groups.sort(key=lambda g: g.center_freq())
        return groups, gid

    # ================================================================ navigation

    def _pan(self, fraction: float):
        x0, x1 = self.plot_mag.viewRange()[0]
        shift = fraction * (x1 - x0)
        self.plot_mag.setXRange(x0 + shift, x1 + shift, padding=0)

    def pan_left(self):        self._pan(-0.2)
    def pan_right(self):       self._pan(0.2)
    def fast_pan_left(self):   self._pan(-0.8)
    def fast_pan_right(self):  self._pan(0.8)

    # ================================================================ save / quit

    def save_data(self, compact_on_save: bool = True):
        """
        Save groups to the zarr group.

        If `compact_on_save` is True, group IDs written to disk are remapped
        to a dense 0..N-1 range based on the sorted groups order. This does
        not mutate `self.groups` in memory.
        """
        fres1_out, ridx1_out, gids1 = [], [], []
        fres2_out, ridx2_out, gids2 = [], [], []
        ambiguous_groups = []

        if compact_on_save:
            sorted_groups = self._sorted_groups()
            gid_map = {g.group_id: i for i, g in enumerate(sorted_groups)}
        else:
            gid_map = None

        for g in self.groups:
            out_gid = gid_map[g.group_id] if gid_map is not None else g.group_id
            if g.ambiguous:
                ambiguous_groups.append(out_gid)
            for fres, ridx in g.entries1:
                fres1_out.append(fres);  ridx1_out.append(ridx);  gids1.append(out_gid)
            for fres, ridx in g.entries2:
                fres2_out.append(fres);  ridx2_out.append(ridx);  gids2.append(out_gid)

        _to_save = {
            'fres1':            (np.float64, fres1_out),
            'res_idx1':         (np.int64,   ridx1_out),
            'group_ids1':       (np.int64,   gids1),
            'fres2':            (np.float64, fres2_out),
            'res_idx2':         (np.int64,   ridx2_out),
            'group_ids2':       (np.int64,   gids2),
            'ambiguous_groups': (np.int64,   ambiguous_groups),
            'max_view_left':    (np.float64, [self._max_view_left]),
        }
        for key, (dtype, data) in _to_save.items():
            if key in self.zarr_group:
                del self.zarr_group[key]
            self.zarr_group.create_array(key, data=np.array(data, dtype=dtype))

        self.log(
            f'Saved: {len(self.groups)} groups, '
            f'{len(fres1_out)} DS1 res, {len(fres2_out)} DS2 res '
            f'→ {self.zarr_group.store}'
        )

    def quit_and_save(self):
        self.save_data()
        self.win.close()
        self.app.quit()

    def _on_window_close(self, event):
        """
        Handle window close event - auto-save before closing.
        
        Parameters:
        event (QCloseEvent): The close event.
        
        Returns:
        None
        """
        self.save_data()
        event.accept()

    # ================================================================ logging

    def log(self, message: str):
        if not hasattr(self, 'log_text'):
            return
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    # ================================================================ help

    def show_help(self):
        if not hasattr(self, '_help_dlg'):
            self._help_dlg = self._build_help_dialog()
        if self._help_dlg.isVisible():
            self._help_dlg.hide()
        else:
            dlg = self._help_dlg
            dlg.adjustSize()
            dlg.show()
            center = self.win.frameGeometry().center()
            fr = dlg.frameGeometry()
            fr.moveCenter(center)
            dlg.move(fr.topLeft())

    def _build_help_dialog(self):
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle('Resonance Matcher Help  (H to close)')
        layout = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel(
            '<h3>Resonance Matcher Controls</h3>'
            '<p><b>Mouse:</b></p>'
            '<ul>'
            '<li><b>Left-click (on a marker):</b> Select that resonance (replaces selection)</li>'
            '<li><b>Ctrl+Left-click (on a marker):</b> Add resonance to selection (multi-select)</li>'
            '<li><b>Shift+Left-click (empty space):</b> Add resonance to DS1 or DS2 based on Y-position. '
            'The tool determines which dataset by comparing your click Y-coordinate to the actual data values.</li>'
            '<li><b>Ctrl+Right-click:</b> Manually set re-match threshold (yellow line)</li>'
            '<li><b>Right-click:</b> PyQtGraph menu (zoom, export, etc.)</li>'
            '<li><b>Scroll wheel:</b> Zoom in / out</li>'
            '</ul>'
            '<p><b>Keyboard (Left Hand):</b></p>'
            '<ul>'
            '<li><b>Z / X:</b> Pan left / right 20%</li>'
            '<li><b>A / S:</b> Pan left / right 80%</li>'
            '<li><b>E:</b> Delete all selected resonances</li>'
            '<li><b>F:</b> Merge groups (all groups containing selected resonances)</li>'
            '<li><b>D:</b> Merge selected only (unlinks selected from groups, merges into new group)</li>'
            '<li><b>W:</b> Unlink all selected resonances (each gets its own group)</li>'
            '<li><b>Q:</b> Toggle ambiguous flag (only for 1:1 matches)</li>'
            '<li><b>R:</b> Trigger automatic re-matching above threshold line</li>'
            '<li><b>H:</b> Toggle this help panel</li>'
            '<li><b>Ctrl+S:</b> Save to zarr</li>'
            '<li><b>Ctrl+Z:</b> Undo (max 50 steps)</li>'
            '</ul>'
            '<p><b>Multi-Selection Workflow:</b></p>'
            '<ul>'
            '<li>Click a resonance to select it (white ring)</li>'
            '<li>Ctrl+click other resonances to add to selection (multiple white rings)</li>'
            '<li>Use F/W/S/U keys to operate on all selected resonances at once</li>'
            '<li>Example: Select 3 resonances in DS1 and 2 in DS2, press F to merge into one group</li>'
            '</ul>'
            '<p><b>Merge Modes:</b></p>'
            '<ul>'
            '<li><b>F (Merge Groups):</b> Merges all groups containing selected resonances. '
            'Example: groups ((1,2,3),(1)) and ((4),(2)), select DS1:2,4 and DS2:2 → ((1,2,3,4),(1,2))</li>'
            '<li><b>D (Merge Selected Only):</b> Creates new group with only selected resonances. '
            'Same selection → ((2,4),(2)) and ((1,3),(1))</li>'
            '</ul>'
            '<p><b>Ambiguous Flag:</b></p>'
            '<ul>'
            '<li>Automatically set to True for groups with multiple resonances in either dataset</li>'
            '<li>B key only toggles flag for 1:1 matches (where you are uncertain about pairing)</li>'
            '<li>Groups with (1:many), (many:1), or (many:many) are always ambiguous</li>'
            '</ul>'
            '<p><b>Auto-Adjusting Threshold:</b></p>'
            '<ul>'
            '<li>Yellow dashed line shows re-match threshold</li>'
            '<li><b>On startup:</b> Pinned to right edge of window (moves with panning)</li>'
            '<li><b>After edits:</b> Moves to right of edit and stays at that frequency</li>'
            '<li><b>When panning:</b> If pinned-frequency scrolls out of view, snaps to right edge again</li>'
            '<li><b>Selected resonances:</b> Auto-deselected when panned out of view</li>'
            '<li>Press <b>R</b> to re-match all groups above line</li>'
            '<li><b>Shift+click</b> to manually set position (stays fixed at that frequency)</li>'
            '</ul>'
            '<p><b>Visual encoding:</b></p>'
            '<ul>'
            '<li>Filled circles = DS1 &nbsp;|&nbsp; Open squares = DS2</li>'
            '<li>Same color = same match group</li>'
            '<li>Dashed marker border = ambiguous group (auto-set for multi-res groups)</li>'
            '<li>White ring(s) = selected resonance(s)</li>'
            '<li>Yellow dashed line = re-match threshold (auto-adjusting)</li>'
            '</ul>'
            '<p><b>Output arrays (zarr):</b></p>'
            '<ul>'
            '<li>fres1, res_idx1, group_ids1</li>'
            '<li>fres2, res_idx2, group_ids2</li>'
            '<li>ambiguous_groups</li>'
            '</ul>'
            '<p>To query group g: <code>fres1[group_ids1 == g]</code></p>'
            '<p><b>Workflow:</b> Ctrl+click to multi-select → F/W/S/U → Close window to save</p>'
        )
        lbl.setTextFormat(_Qt.RichText)
        lbl.setWordWrap(True)
        
        # Put label in a scroll area to prevent off-screen content
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(600)  # Limit height to fit on screen
        layout.addWidget(scroll)
        
        close_btn = QtWidgets.QPushButton('Close (H)')
        close_btn.clicked.connect(dlg.hide)
        layout.addWidget(close_btn)
        sc = QtGui.QShortcut(QtGui.QKeySequence('H'), dlg)
        sc.activated.connect(dlg.hide)
        return dlg

    # ================================================================ run

    def run(self):
        n1 = sum(len(g.entries1) for g in self.groups)
        n2 = sum(len(g.entries2) for g in self.groups)
        self.log('Resonance Matcher ready.')
        self.log(
            f'Groups: {len(self.groups)}  |  DS1: {n1}  |  DS2: {n2}'
        )
        self.log(f'Active dataset: DS{self.active_dataset}  (C to toggle)')
        self.log("Press 'H' for help.")
        self.app.exec()
