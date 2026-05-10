"""
Interactive resonance matcher for pairing resonances between two VNA datasets.

Each resonance belongs to a MatchGroup that links zero or more resonances from
dataset 1 to zero or more resonances from dataset 2.  The user scrolls through
the sweep, edits groups by clicking and using keyboard shortcuts, and saves the
result to zarr.

Output layout (zarr)
--------------------
fres1_out        float64[N1]  — final DS1 resonance frequencies (Hz)
res_idx1_out     int64[N1]    — resonator indices for DS1
group_ids1       int64[N1]    — match-group ID for each DS1 resonance

fres2_out        float64[N2]  — final DS2 resonance frequencies (Hz)
res_idx2_out     int64[N2]    — resonator indices for DS2
group_ids2       int64[N2]    — match-group ID for each DS2 resonance

ambiguous_groups int64[K]     — group IDs flagged as ambiguous

To query group g: fres1_out[group_ids1 == g] and fres2_out[group_ids2 == g].
"""

import copy
import dataclasses
import os
from typing import List, Optional, Tuple

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
    new_res_start_idx: int = 1000,
    init_match: str = 'sorted',
    margin_factor: float = 0.15,
    overwrite: bool = False,
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
        resonances.  Increments by 1 for each addition.  Default is 1000.
    init_match (str): How to build the initial pairing.  'sorted' pairs by
        frequency order; 'nearest' uses median-offset-corrected
        nearest-neighbor.  Default is 'sorted'.
    margin_factor (float): Y-axis auto-scale margin fraction.  Default 0.15.
    overwrite (bool): Allow overwriting existing output keys in the zarr
        group.  Default is False.
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
        overwrite=overwrite,
        apply_filter=apply_filter,
    )
    matcher.run()
    return matcher.groups


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
        new_res_start_idx: int = 1000,
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

        self.margin_factor = margin_factor
        self._next_new_idx = int(new_res_start_idx)
        self._next_group_id = 0    # set after initialisation

        # ---- zarr -----------------------------------------------------------
        if isinstance(zarr_grp, (str, os.PathLike)):
            self.zarr_group = zarr.open_group(str(zarr_grp), mode='a')
        else:
            self.zarr_group = zarr_grp

        _output_keys = [
            'fres1_out', 'res_idx1_out', 'group_ids1',
            'fres2_out', 'res_idx2_out', 'group_ids2',
            'ambiguous_groups',
        ]
        for key in _output_keys:
            if key in self.zarr_group:
                if not overwrite:
                    raise FileExistsError(
                        f"'{key}' already exists in the zarr group. "
                        "Set overwrite=True to overwrite."
                    )

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
        self.sel_group_id: Optional[int] = None
        self.sel_dataset: Optional[int] = None
        self.sel_fres: Optional[float] = None
        self.undo_stack: list = []
        self._scatter_just_clicked: bool = False

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
            'Resonance Matcher — C: toggle dataset  |  H: help'
        )
        self.win.resize(1500, 850)

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
        splitter.addWidget(ctrl_widget)
        splitter.setSizes([1150, 300])

        self.setup_controls(ctrl_layout)
        self.setup_plots()
        self.setup_shortcuts()

        # Initial state (filters already applied in __init__)
        f_lo = float(min(self.f1[0], self.f2[0]))
        f_hi = f_lo + 10e6   # start zoomed to first 10 MHz for fast initial render
        self.plot_mag.setXRange(f_lo, f_hi, padding=0.02)
        self._sync_mag2_geom()
        self._sync_phase2_geom()
        self._update_curves()
        self.update_markers()
        self.auto_scale_y()
        self._update_overview_region()

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
        save_btn = QtWidgets.QPushButton('Save (S)')
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
            'Z/X: pan 20%  |  A/D: pan 80%  |  E/R: prev/next group  |  '
            'C: toggle DS  |  M: merge  |  U: unlink  |  B: ambiguous  |  H: help'
            '</span>'
        )
        self.plot_widget.addLabel(title, col=0)
        self.plot_widget.nextRow()

        # ---- Overview navigator (row 1) ------------------------------------
        # Downsampled full-range view with a draggable region showing zoom.
        self.plot_overview = self.plot_widget.addPlot(row=1, col=0)
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
        self.plot_mag = self.plot_widget.addPlot(row=2, col=0)
        self.plot_mag.setLabel('left', '|S21| DS1 (dB)',
                               color=(100, 180, 255))
        self.plot_mag.showGrid(x=True, y=True, alpha=0.3)
        self.plot_mag.getAxis('bottom').setStyle(showValues=False)
        self.plot_mag.showAxis('right')
        self.plot_mag.setLabel('right', '|S21| DS2 (dB)',
                               color=(255, 140, 60))

        # Right-axis ViewBox for DS2 magnitude
        self._vb_mag2 = pg.ViewBox()
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
        self.scatter_sel_mag = pg.ScatterPlotItem(
            symbol='o', size=20,
            pen=pg.mkPen('w', width=2.5), brush=pg.mkBrush(None),
        )
        self.plot_mag.addItem(self.scatter_ds1_mag)
        self._vb_mag2.addItem(self.scatter_ds2_mag)
        self.plot_mag.addItem(self.scatter_sel_mag)

        self.plot_widget.nextRow()

        # ---- Phase plot (row 3) — dual Y axes -----------------------------
        self.plot_phase = self.plot_widget.addPlot(row=3, col=0)
        self.plot_phase.setLabel('left', 'Phase DS1 (rad)',
                                  color=(100, 180, 255))
        self.plot_phase.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.plot_phase.setXLink(self.plot_mag)
        self.plot_phase.showAxis('right')
        self.plot_phase.setLabel('right', 'Phase DS2 (rad)',
                                  color=(255, 140, 60))

        # Right-axis ViewBox for DS2 phase
        self._vb_phase2 = pg.ViewBox()
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
        self.scatter_sel_phase = pg.ScatterPlotItem(
            symbol='o', size=20,
            pen=pg.mkPen('w', width=2.5), brush=pg.mkBrush(None),
        )
        self.plot_phase.addItem(self.scatter_ds1_phase)
        self._vb_phase2.addItem(self.scatter_ds2_phase)
        self.plot_phase.addItem(self.scatter_sel_phase)

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

    def setup_shortcuts(self):
        _map = {
            'Z': self.pan_left,
            'X': self.pan_right,
            'A': self.fast_pan_left,
            'D': self.fast_pan_right,
            'E': self.jump_prev_group,
            'R': self.jump_next_group,
            'S': self.save_data,
            'H': self.show_help,
            'C': self.toggle_active_dataset,
            'M': self.merge_selected,
            'U': self.unlink_selected,
            'B': self.toggle_ambiguous,
        }
        for key, slot in _map.items():
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.win)
            sc.activated.connect(slot)

        undo_sc = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Z'), self.win)
        undo_sc.activated.connect(self.undo)

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
                selected = (
                    self.sel_group_id == g.group_id
                    and self.sel_dataset == 1
                    and self.sel_fres == fres
                )
                spot = {
                    'pos': (float(self.f1[idx]), 0.0),   # y filled below
                    'brush': filled_brush,
                    'pen': ds1_pen,
                    'size': 13 if selected else 10,
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
                selected = (
                    self.sel_group_id == g.group_id
                    and self.sel_dataset == 2
                    and self.sel_fres == fres
                )
                spot = {
                    'brush': pg.mkBrush(None),   # open square
                    'pen': ds2_pen,
                    'size': 13 if selected else 10,
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
        if self.sel_group_id is None:
            self.scatter_sel_mag.setData([])
            self.scatter_sel_phase.setData([])
            return

        g = self._find_group(self.sel_group_id)
        if g is None:
            self.scatter_sel_mag.setData([])
            self.scatter_sel_phase.setData([])
            return

        entries = g.entries1 if self.sel_dataset == 1 else g.entries2
        f_arr = self.f1 if self.sel_dataset == 1 else self.f2
        mag_arr = self.filtered_mag1 if self.sel_dataset == 1 else self.filtered_mag2
        ph_arr = self.phase1 if self.sel_dataset == 1 else self.phase2

        sel_mag, sel_phase = [], []
        for (fres, _) in entries:
            if fres == self.sel_fres:
                idx = self._nearest_idx(f_arr, fres)
                sel_mag.append({'pos': (float(f_arr[idx]), float(mag_arr[idx]))})
                sel_phase.append({'pos': (float(f_arr[idx]), float(ph_arr[idx]))})

        self.scatter_sel_mag.setData(sel_mag)
        self.scatter_sel_phase.setData(sel_phase)

    # ================================================================ group helpers

    def _find_group(self, group_id: int) -> Optional[MatchGroup]:
        for g in self.groups:
            if g.group_id == group_id:
                return g
        return None

    def _save_undo_state(self):
        self.undo_stack.append(copy.deepcopy(self.groups))

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
        Select a clicked scatter point (left-click only).

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
        if not pts:
            return

        pt = pts[0]
        data = pt.data()
        if data is None:
            return

        group_id, dataset, fres = data
        self.sel_group_id = group_id
        self.sel_dataset = dataset
        self.sel_fres = fres

        g = self._find_group(group_id)
        mapping = g.mapping_str() if g else '?'
        ambig = '  [ambiguous]' if (g and g.ambiguous) else ''
        self.sel_label.setText(
            f'Selected: Group {group_id} ({mapping}){ambig}, '
            f'DS{dataset}, {fres / 1e6:.4f} MHz'
        )
        self.update_markers()
        # Suppress the scene's add-resonance handler for this click event
        self._scatter_just_clicked = True

    def _on_scene_clicked(self, ev):
        """
        Left-click on empty space → add resonance to the active dataset.
        Right-click → context menu near the click position.
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

        if ev.button() == _Qt.RightButton:
            freq = self._scene_pos_to_freq(pos)
            if freq is not None:
                self._show_context_menu(ev.screenPos(), freq)
            return

        if ev.button() != _Qt.LeftButton:
            return

        modifiers = ev.modifiers()
        # Ctrl+click → always DS2
        ds = 2 if (modifiers & _Qt.ControlModifier) else self.active_dataset

        freq = self._scene_pos_to_freq(pos)
        if freq is None:
            return

        self.add_resonance(freq, ds)

    def _scene_pos_to_freq(self, scene_pos) -> Optional[float]:
        """
        Convert a scene position to a frequency (Hz), checking both plots.

        Parameters:
        scene_pos (QPointF): Position in scene coordinates.

        Returns:
        freq (float or None): Frequency in Hz, or None if outside both plots.
        """
        for plot in (self.plot_mag, self.plot_phase):
            if plot.sceneBoundingRect().contains(scene_pos):
                vb_pos = plot.vb.mapSceneToView(scene_pos)
                return float(vb_pos.x())
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
            self._remove_empty_groups()
            self._clear_selection_if_gone()
            self.update_markers()
            self.log(
                f'Deleted DS1 resonance at {entry[0] / 1e6:.4f} MHz '
                f'from group {near.group_id}'
            )

        elif chosen == act_del2 and near.entries2:
            self._save_undo_state()
            entry = min(near.entries2, key=lambda e: abs(e[0] - click_freq))
            near.entries2.remove(entry)
            self._remove_empty_groups()
            self._clear_selection_if_gone()
            self.update_markers()
            self.log(
                f'Deleted DS2 resonance at {entry[0] / 1e6:.4f} MHz '
                f'from group {near.group_id}'
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
        self._save_undo_state()
        ridx = self._new_res_idx()
        gid = self._new_group_id()
        entry = (float(freq), ridx)
        if ds == 1:
            g = MatchGroup(group_id=gid, entries1=[entry], entries2=[])
        else:
            g = MatchGroup(group_id=gid, entries1=[], entries2=[entry])
        self.groups.append(g)
        self._sort_groups()
        self.update_markers()
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
        self.log(
            f'Merged group {gid_b} into {gid_a} → {ga.mapping_str()}'
        )

    def _clear_selection_if_gone(self):
        if self.sel_group_id is None:
            return
        g = self._find_group(self.sel_group_id)
        if g is None:
            self._clear_selection()
            return
        entries = g.entries1 if self.sel_dataset == 1 else g.entries2
        if not any(e[0] == self.sel_fres for e in entries):
            self._clear_selection()

    def _clear_selection(self):
        self.sel_group_id = None
        self.sel_dataset = None
        self.sel_fres = None
        self.sel_label.setText('Selected: none')

    # ================================================================ keyboard actions

    def toggle_active_dataset(self):
        self.active_dataset = 2 if self.active_dataset == 1 else 1
        self._refresh_active_label()
        self.log(f'Active dataset: DS{self.active_dataset}')

    def merge_selected(self):
        """
        Merge the selected group with its nearest sorted neighbor.

        Prefers the right neighbor; falls back to left.  Bound to the M key.

        Parameters:
        None

        Returns:
        None
        """
        if self.sel_group_id is None:
            self.log('M: Click a marker to select it first.')
            return
        sorted_gs = self._sorted_groups()
        try:
            pos = next(
                i for i, g in enumerate(sorted_gs)
                if g.group_id == self.sel_group_id
            )
        except StopIteration:
            return
        if len(sorted_gs) < 2:
            self.log('M: Only one group; nothing to merge.')
            return
        if pos < len(sorted_gs) - 1:
            target = sorted_gs[pos + 1]
        else:
            target = sorted_gs[pos - 1]
        self._do_merge(sorted_gs[pos].group_id, target.group_id)

    def unlink_selected(self):
        """
        Move the selected entry into its own standalone group.

        Bound to the U key.

        Parameters:
        None

        Returns:
        None
        """
        if self.sel_group_id is None:
            self.log('U: Click a marker to select it first.')
            return
        g = self._find_group(self.sel_group_id)
        if g is None:
            return
        entries = g.entries1 if self.sel_dataset == 1 else g.entries2
        entry = next((e for e in entries if e[0] == self.sel_fres), None)
        if entry is None:
            return
        self._do_unlink(g, entry, self.sel_dataset)

    def toggle_ambiguous(self):
        """
        Toggle the ambiguous flag on the selected group.

        Bound to the B key.

        Parameters:
        None

        Returns:
        None
        """
        if self.sel_group_id is None:
            self.log('B: Click a marker to select it first.')
            return
        g = self._find_group(self.sel_group_id)
        if g is None:
            return
        self._save_undo_state()
        g.ambiguous = not g.ambiguous
        self.update_markers()
        self.log(f'Group {g.group_id} ambiguous = {g.ambiguous}')

    def undo(self):
        if not self.undo_stack:
            self.log('Nothing to undo.')
            return
        self.groups = self.undo_stack.pop()
        self._clear_selection_if_gone()
        self.update_markers()
        self.log('Undo.')

    # ================================================================ navigation

    def _pan(self, fraction: float):
        x0, x1 = self.plot_mag.viewRange()[0]
        shift = fraction * (x1 - x0)
        self.plot_mag.setXRange(x0 + shift, x1 + shift, padding=0)

    def pan_left(self):        self._pan(-0.2)
    def pan_right(self):       self._pan(0.2)
    def fast_pan_left(self):   self._pan(-0.8)
    def fast_pan_right(self):  self._pan(0.8)

    def jump_prev_group(self):
        x0, x1 = self.plot_mag.viewRange()[0]
        center = 0.5 * (x0 + x1)
        width = x1 - x0
        candidates = [g for g in self.groups if g.center_freq() < center - 1.0]
        if not candidates:
            self.log('No group to the left.')
            return
        target = max(candidates, key=lambda g: g.center_freq())
        fc = target.center_freq()
        self.plot_mag.setXRange(fc - width / 2, fc + width / 2, padding=0)

    def jump_next_group(self):
        x0, x1 = self.plot_mag.viewRange()[0]
        center = 0.5 * (x0 + x1)
        width = x1 - x0
        candidates = [g for g in self.groups if g.center_freq() > center + 1.0]
        if not candidates:
            self.log('No group to the right.')
            return
        target = min(candidates, key=lambda g: g.center_freq())
        fc = target.center_freq()
        self.plot_mag.setXRange(fc - width / 2, fc + width / 2, padding=0)

    # ================================================================ save / quit

    def save_data(self):
        fres1_out, ridx1_out, gids1 = [], [], []
        fres2_out, ridx2_out, gids2 = [], [], []
        ambiguous_groups = []

        for g in self.groups:
            if g.ambiguous:
                ambiguous_groups.append(g.group_id)
            for fres, ridx in g.entries1:
                fres1_out.append(fres);  ridx1_out.append(ridx);  gids1.append(g.group_id)
            for fres, ridx in g.entries2:
                fres2_out.append(fres);  ridx2_out.append(ridx);  gids2.append(g.group_id)

        _to_save = {
            'fres1_out':        (np.float64, fres1_out),
            'res_idx1_out':     (np.int64,   ridx1_out),
            'group_ids1':       (np.int64,   gids1),
            'fres2_out':        (np.float64, fres2_out),
            'res_idx2_out':     (np.int64,   ridx2_out),
            'group_ids2':       (np.int64,   gids2),
            'ambiguous_groups': (np.int64,   ambiguous_groups),
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
            '<li><b>Left-click (empty space):</b> Add resonance to the active dataset</li>'
            '<li><b>Ctrl+Left-click:</b> Add to DS2 regardless of active dataset</li>'
            '<li><b>Left-click (on a marker):</b> Select that marker</li>'
            '<li><b>Right-click:</b> Context menu — delete / unlink / merge / flag</li>'
            '<li><b>Scroll wheel:</b> Zoom in / out</li>'
            '</ul>'
            '<p><b>Keyboard:</b></p>'
            '<ul>'
            '<li><b>C:</b> Toggle active dataset (DS1 ↔ DS2)</li>'
            '<li><b>Z / X:</b> Pan left / right 20%</li>'
            '<li><b>A / D:</b> Pan left / right 80%</li>'
            '<li><b>E / R:</b> Jump to previous / next group center</li>'
            '<li><b>M:</b> Merge selected group with nearest neighbor</li>'
            '<li><b>U:</b> Unlink selected marker from its group</li>'
            '<li><b>B:</b> Toggle ambiguous flag on selected group</li>'
            '<li><b>S:</b> Save to zarr</li>'
            '<li><b>Ctrl+Z:</b> Undo</li>'
            '<li><b>H:</b> Toggle this help panel</li>'
            '</ul>'
            '<p><b>Visual encoding:</b></p>'
            '<ul>'
            '<li>Filled circles = DS1 &nbsp;|&nbsp; Open squares = DS2</li>'
            '<li>Same color = same match group</li>'
            '<li>Dashed marker border = ambiguous group</li>'
            '<li>White ring = currently selected marker</li>'
            '</ul>'
            '<p><b>Output arrays (zarr):</b></p>'
            '<ul>'
            '<li>fres1_out, res_idx1_out, group_ids1</li>'
            '<li>fres2_out, res_idx2_out, group_ids2</li>'
            '<li>ambiguous_groups</li>'
            '</ul>'
            '<p>To query group g: <code>fres1_out[group_ids1 == g]</code></p>'
        )
        lbl.setTextFormat(_Qt.RichText)
        lbl.setWordWrap(False)
        layout.addWidget(lbl)
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
