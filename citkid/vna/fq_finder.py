"""
Interactive resonance frequency and Q-factor finder.

Displays, for each resonator index *i*:

    Left  plot — 20·log10(|z[i]|) vs f[i]  (dB amplitude)
    Right plot — z[i].imag vs z[i].real       (IQ loop)

Overlaid markers
----------------
* Vertical dashed line on the left plot at ``fres[i]``.
* Shaded band of width ``fres[i]/qres[i]`` centred on ``fres[i]``.
* An **X** on the right plot at the IQ point closest to ``fres[i]``.

Interaction
-----------
* **Shift + left-click** on either plot  → move ``fres`` to the clicked
  frequency (left) or the nearest sample in IQ space (right).
* **Shift + scroll wheel** on either plot → adjust ``qres`` (fine) by
  ±1 step per click.
* **Ctrl  + scroll wheel** on either plot → adjust ``qres`` (coarse) by
  ±10 steps per click.
* **Right arrow / D**  → save current values and advance to next resonator.
* **Left  arrow / A**  → go back one resonator (re-opens with saved values).
* **Closing the window** → saves current values and exits.

Calibration tones
-----------------
Resonators whose ``res_idxs[i] < 1`` are treated as calibration tones: the
input ``fres`` and ``qres`` values are written directly to the output zarr
arrays and the interactive step is skipped.

Output
------
Two 1-D zarr arrays are written into ``zarr_group``:
    ``fres_opt``  — optimised resonant frequency (Hz)
    ``qres_opt``  — optimised Q-factor
"""

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..qt_compat import Qt as _Qt
from ..multitone.fres import update_fres as _update_fres

# QEvent.KeyPress moved to QEvent.Type.KeyPress in PyQt6 / some PySide6 builds.
_QEVENT_KEY_PRESS = getattr(
    QtCore.QEvent, 'KeyPress',
    getattr(getattr(QtCore.QEvent, 'Type', None), 'KeyPress', None),
)


# ---------------------------------------------------------------------------
# Font-scaling helper (mirrors StepPanel._scale_plot_fonts)
# ---------------------------------------------------------------------------

def _scale_plot_fonts(ui_scale: float, *plot_items) -> None:
    """Apply *ui_scale* to tick labels, axis labels, and titles."""
    tick_pt  = max(6, round(9  * ui_scale))
    label_pt = max(7, round(11 * ui_scale))
    tick_font  = QtGui.QFont()
    tick_font.setPointSize(tick_pt)
    label_font = QtGui.QFont()
    label_font.setPointSize(label_pt)
    for plot in plot_items:
        for ax_name in ('bottom', 'left', 'top', 'right'):
            ax = plot.getAxis(ax_name)
            ax.setStyle(tickFont=tick_font)
            ax.label.setFont(label_font)
        plot.titleLabel.item.setFont(label_font)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z))


def _ensure_zarr_arrays(zarr_group, n: int, overwrite: bool) -> bool:
    """Ensure ``fres_opt``, ``qres_opt``, and ``reject_reason`` arrays exist
    in *zarr_group*.

    Returns True if the float arrays already existed (resume mode), False if
    they were freshly created.

    Behaviour:
    * Both float arrays absent  → create all three arrays fresh.
    * Both float arrays present, overwrite=True  → resume; create
      ``reject_reason`` if missing (legacy groups).
    * Both float arrays present, overwrite=False → raise ``FileExistsError``.
    * Only one float array present  → raise ``RuntimeError``.
    """
    existing = [name for name in ("fres_opt", "qres_opt") if name in zarr_group]
    if len(existing) == 1:
        raise RuntimeError(
            f"zarr_group contains {existing} but not the other; "
            "the group is in an inconsistent state."
        )
    if len(existing) == 2:
        if not overwrite:
            raise FileExistsError(
                "zarr_group already contains fres_opt and qres_opt. "
                "Pass overwrite=True to resume / overwrite."
            )
        # Resume mode: ensure reject_reason exists (may be absent in legacy groups)
        if "reject_reason" not in zarr_group:
            zarr_group.create_dataset(
                "reject_reason",
                shape=(n,),
                dtype=str,
                fill_value="",
            )
        return True
    # Create fresh arrays
    for name in ("fres_opt", "qres_opt"):
        zarr_group.create_dataset(
            name,
            shape=(n,),
            dtype=np.float64,
            fill_value=np.nan,
        )
    zarr_group.create_dataset(
        "reject_reason",
        shape=(n,),
        dtype=str,
        fill_value="",
    )
    return False


def _rmv_gain_simple(f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Simple gain / cable-delay removal for (M, N) sweep data.

    Steps
    -----
    1. **Amplitude**: divide each row of *z* by the median |z| of the
       first and last ``N = max(1, N_pts // 100)`` samples.
    2. **Phase offset**: rotate each row so the mean of the off-resonance
       edge samples (first/last N) lies on the positive real axis.
    """
    N = max(1, z.shape[1] // 100)

    # Off-resonance edge samples
    offres = np.concatenate([z[:, :N], z[:, -N:]], axis=1)  # (M, 2N)

    # 1. Amplitude normalisation
    amp_ref = np.median(np.abs(offres), axis=1, keepdims=True)  # (M, 1)
    amp_ref = np.where(amp_ref == 0, 1.0, amp_ref)
    z = z / amp_ref

    # 2. Phase-offset removal: rotate off-resonance mean onto real axis
    offres = np.concatenate([z[:, :N], z[:, -N:]], axis=1)
    phase_ref = np.angle(offres.mean(axis=1, keepdims=True))  # (M, 1)
    z = z * np.exp(-1j * phase_ref)

    return z



class _InteractiveViewBox(pg.ViewBox):
    """
    ViewBox that emits:

    ``sig_shift_click(x, y)``  — Shift + left-click inside the view.
    ``sig_scroll_qres(delta)`` — Shift/Ctrl + scroll wheel (delta = ±1, coarse ±10).
    """

    sig_shift_click = QtCore.pyqtSignal(float, float)
    sig_scroll_qres = QtCore.pyqtSignal(int)

    def mousePressEvent(self, ev):
        if (ev.button() == _Qt.LeftButton
                and ev.modifiers() & _Qt.ShiftModifier):
            ev.accept()
            pos = self.mapToView(ev.pos())
            self.sig_shift_click.emit(pos.x(), pos.y())
        else:
            super().mousePressEvent(ev)

    def wheelEvent(self, ev, axis=None):
        mods = ev.modifiers()
        if mods & _Qt.ShiftModifier or mods & _Qt.ControlModifier:
            ev.accept()
            # angleDelta().y() is ±120 per notch on most wheels
            raw = getattr(ev, 'angleDelta', lambda: None)()
            if raw is None:
                delta_y = getattr(ev, 'delta', lambda: 0)()
            else:
                delta_y = raw.y()
            step = 10 if bool(mods & _Qt.ControlModifier) else 1
            direction = 1 if delta_y > 0 else -1
            self.sig_scroll_qres.emit(direction * step)
        else:
            super().wheelEvent(ev, axis=axis)


class _SpanRegionItem(pg.LinearRegionItem):
    """LinearRegionItem whose body is not interactive (only the edge lines are)."""
    def mouseDragEvent(self, ev):
        ev.ignore()

    def mousePressEvent(self, ev):
        ev.ignore()

    def hoverEvent(self, ev):
        # Suppress body hover highlight; edge InfiniteLines handle their own hover
        pass


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class FqFinderWindow(QtWidgets.QMainWindow):
    """
    Stand-alone QMainWindow that hosts the interactive fres/qres editor.

    Parameters
    ----------
    f : np.ndarray, shape (M, N)
        Frequency arrays in Hz.
    z : np.ndarray, shape (M, N)
        Complex IQ data.
    fres : array-like, length M
        Initial resonant frequencies in Hz.
    qres : array-like, length M
        Initial Q-factors.
    res_idxs : array-like, length M
        Resonator indices.  Values < 1 indicate calibration tones.
    zarr_group : zarr.Group
        Output group; ``fres_opt`` and ``qres_opt`` 1-D arrays are
        created here (length M).
    title : str
        Window title.
    """

    # Fractional qres step size: each scroll notch changes qres by this
    # fraction of its current value (×step magnitude).
    _QRES_FRAC = 0.04
    _REJECT_REASONS = ["tone off resonance", "overlapping resonance", "bifurcated", "other"]

    def __init__(
        self,
        f: np.ndarray,
        z: np.ndarray,
        fres,
        qres,
        res_idxs,
        zarr_group,
        title: str = "FQ Finder",
        ui_scale: float = None,   # deprecated — ignored; scale is auto-computed
        plot_scale: float = None,  # deprecated — ignored; scale is auto-computed
        overwrite: bool = False,
        fres_update_method: str = "none",
        start_idx: int = 0,
        rmv_gain_simple: bool = False,
    ):
        super().__init__()
        self.setWindowTitle(title)

        self._f = np.asarray(f, dtype=np.float64)           # (M, N)
        self._z = np.asarray(z, dtype=complex)              # (M, N)
        # Sort along axis 1 so frequencies are always ascending
        _sort_idx = np.argsort(self._f, axis=1)
        self._f = np.take_along_axis(self._f, _sort_idx, axis=1)
        self._z = np.take_along_axis(self._z, _sort_idx, axis=1)
        if rmv_gain_simple:
            self._z = _rmv_gain_simple(self._f, self._z)
        fres = np.asarray(fres, dtype=np.float64)
        qres = np.asarray(qres, dtype=np.float64)
        res_idxs = np.asarray(res_idxs)
        # Apply automatic fres update before starting interactive session
        fres = _update_fres(
            self._f, self._z, fres, qres, res_idxs,
            method=fres_update_method,
        )
        self._fres_work = fres.copy()
        self._qres_work = qres.copy()
        self._fres_init = fres.copy()   # original values for Z-reset
        self._qres_init = qres.copy()
        self._res_idxs = res_idxs
        self._zg = zarr_group
        self._M = self._f.shape[0]

        # Auto-compute scale from the screen that the cursor is on.
        # ui_scale / plot_scale are accepted for backward compatibility but ignored.
        _app = QtWidgets.QApplication.instance()
        _screen = (_app.screenAt(QtGui.QCursor.pos())
                   if hasattr(_app, 'screenAt') else None)
        if _screen is None:
            _screen = _app.primaryScreen()
        _geom = _screen.availableGeometry()
        self._win_w = round(_geom.width() * 0.8)
        self._win_h = round(_geom.height() * 0.8)
        self._screen_geom = _geom
        # Scale relative to the baseline 1100×520 window
        self._ui_scale = max(0.7, min(self._win_w / 1100, self._win_h / 520))

        self._reject_reasons: dict = {}
        self._pre_reject: dict = {}   # fres/qres saved before rejection

        _ensure_zarr_arrays(self._zg, self._M, overwrite)

        # Pre-save calibration tones immediately (they are never interactive)
        for i in range(self._M):
            if self._res_idxs[i] < 1:
                self._zg["fres_opt"][i] = self._fres_work[i]
                self._zg["qres_opt"][i] = self._qres_work[i]

        # Build the list of interactive indices (non-cal tones)
        self._interactive_indices = [
            i for i in range(self._M) if self._res_idxs[i] >= 1
        ]
        if not self._interactive_indices:
            # Nothing to do
            self.close()
            return

        # Find the cursor position: first interactive index >= start_idx
        start_idx = max(0, int(start_idx))
        self._cursor = next(
            (pos for pos, i in enumerate(self._interactive_indices) if i >= start_idx),
            0,
        )

        self._build_ui()
        # Size and centre on the target screen
        self.resize(self._win_w, self._win_h)
        _g = self._screen_geom
        self.move(
            _g.x() + (_g.width() - self._win_w) // 2,
            _g.y() + (_g.height() - self._win_h) // 2,
        )
        self._load_resonator()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(6, 4, 6, 6)
        vbox.setSpacing(4)

        # ---- top control bar (row 1): navigation + fres/qres + status ----
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(10)

        self._prev_btn = QtWidgets.QPushButton("◀  Back (A/←)")
        self._prev_btn.setFixedWidth(round(120 * self._ui_scale))
        self._prev_btn.clicked.connect(self._go_back)
        ctrl.addWidget(self._prev_btn)

        self._next_btn = QtWidgets.QPushButton("Next (D/→)  ►")
        self._next_btn.setFixedWidth(round(120 * self._ui_scale))
        self._next_btn.clicked.connect(self._go_next)
        ctrl.addWidget(self._next_btn)

        ctrl.addSpacing(20)

        ctrl.addWidget(QtWidgets.QLabel("fres (MHz):"))
        self._fres_spin = QtWidgets.QDoubleSpinBox()
        self._fres_spin.setDecimals(4)
        self._fres_spin.setRange(0.0, 1e6)
        self._fres_spin.setSingleStep(0.001)
        self._fres_spin.setFixedWidth(round(130 * self._ui_scale))
        self._fres_spin.setKeyboardTracking(False)
        self._fres_spin.valueChanged.connect(self._on_spinbox_fres)
        ctrl.addWidget(self._fres_spin)

        ctrl.addSpacing(10)

        ctrl.addWidget(QtWidgets.QLabel("qres:"))
        self._qres_spin = QtWidgets.QDoubleSpinBox()
        self._qres_spin.setDecimals(0)
        self._qres_spin.setRange(1.0, 1e8)
        self._qres_spin.setSingleStep(1000.0)
        self._qres_spin.setFixedWidth(round(110 * self._ui_scale))
        self._qres_spin.setKeyboardTracking(False)
        self._qres_spin.valueChanged.connect(self._on_spinbox_qres)
        ctrl.addWidget(self._qres_spin)

        ctrl.addStretch()

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setMinimumWidth(round(220 * self._ui_scale))
        ctrl.addWidget(self._status_label)

        vbox.addLayout(ctrl)

        # ---- control bar row 2: rejection ----
        reject_row = QtWidgets.QHBoxLayout()
        reject_row.setSpacing(10)

        self._reason_combo = QtWidgets.QComboBox()
        self._reason_combo.addItem("not rejected")
        for _r in self._REJECT_REASONS:
            self._reason_combo.addItem(_r)
        self._reason_combo.setFixedWidth(round(200 * self._ui_scale))
        self._reason_combo.currentTextChanged.connect(self._on_reason_combo_changed)
        reject_row.addWidget(self._reason_combo)

        self._reason_edit = QtWidgets.QLineEdit()
        self._reason_edit.setPlaceholderText("custom reason…")
        self._reason_edit.setFixedWidth(round(150 * self._ui_scale))
        self._reason_edit.editingFinished.connect(self._on_reason_edit_finished)
        reject_row.addWidget(self._reason_edit)

        self._reject_label = QtWidgets.QLabel("")
        self._reject_label.setMinimumWidth(round(120 * self._ui_scale))
        reject_row.addWidget(self._reject_label)

        reject_row.addStretch()
        vbox.addLayout(reject_row)

        # Widget font — built once here and applied to the hint bar and
        # the central widget so all child widgets inherit the same size.
        widget_pt = max(7, round(9 * self._ui_scale))
        widget_font = QtGui.QFont()
        widget_font.setPointSize(widget_pt)
        central.setFont(widget_font)

        # ---- hint bar ----
        hint = QtWidgets.QLabel(
            "Shift+click → set fres  |  "
            "Shift+scroll → fine Δqres  |  "
            "Ctrl+scroll → coarse Δqres  |  "
            "←/A → back  |  →/D → next / save  |  "
            "R → rescale  |  Z → reset to initial  |  H → help"
        )
        hint.setStyleSheet("color: #aaa;")
        hint.setFont(widget_font)
        vbox.addWidget(hint)

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(300)
        vbox.addWidget(self._gw)

        # Left: amplitude plot
        vb_amp = _InteractiveViewBox()
        vb_amp.sig_shift_click.connect(self._on_shift_click_amp)
        vb_amp.sig_scroll_qres.connect(self._on_scroll_qres)
        self._plot_amp = self._gw.addPlot(row=0, col=0, title="Amplitude",
                                           viewBox=vb_amp)
        self._plot_amp.setLabel('left', '|S₂₁| (dB)')
        self._plot_amp.setLabel('bottom', 'f (MHz)')
        self._plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self._plot_amp.setDownsampling(auto=True, mode='peak')

        # Right: IQ loop
        vb_iq = _InteractiveViewBox()
        vb_iq.sig_shift_click.connect(self._on_shift_click_iq)
        vb_iq.sig_scroll_qres.connect(self._on_scroll_qres)
        self._plot_iq = self._gw.addPlot(row=0, col=1, title="IQ Loop",
                                          viewBox=vb_iq)
        self._plot_iq.setLabel('left', 'Q (Im)')
        self._plot_iq.setLabel('bottom', 'I (Re)')
        self._plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self._plot_iq.setAspectLocked(True, ratio=1)

        # Static data curves (set once per resonator)
        _data_pen = pg.mkPen(color=(100, 160, 255), width=1)
        _data_brush = pg.mkBrush(100, 160, 255, 160)
        self._amp_curve = self._plot_amp.plot(pen=_data_pen, name='|S₂₁|')
        self._iq_scatter = self._plot_iq.plot(
            pen=None, symbol='o', symbolSize=4,
            symbolBrush=_data_brush, symbolPen=None,
            name='IQ',
        )

        # Span-highlighted IQ points (within fres±span/2)
        _span_brush = pg.mkBrush(255, 200, 50, 220)
        self._iq_span = self._plot_iq.plot(
            pen=None, symbol='o', symbolSize=5,
            symbolBrush=_span_brush, symbolPen=None,
            name='span',
        )

        # fres vertical line (amplitude plot) — bright red, movable so the
        # user can drag it to adjust fres directly
        self._fres_vline = pg.InfiniteLine(
            angle=90, pen=pg.mkPen((255, 80, 80), width=2, style=_Qt.DotLine),
            movable=True,
            hoverPen=pg.mkPen((255, 80, 80), width=14),
        )
        self._fres_vline.sigPositionChanged.connect(self._on_fres_vline_moved)
        self._fres_updating = False  # re-entry guard
        self._plot_amp.addItem(self._fres_vline)
        self._fres_vline.setZValue(10)  # above the span region

        # span shaded region (amplitude plot) — edges draggable to adjust
        # qres; body drag is disabled (use the fres vline to move fres)
        self._span_region = _SpanRegionItem(
            brush=pg.mkBrush(255, 200, 50, 50),
            pen=pg.mkPen(color=(255, 200, 50), width=1),
            movable=True,
        )
        self._span_region.sigRegionChanged.connect(self._on_span_region_changed)
        self._plot_amp.addItem(self._span_region)
        self._span_updating = False  # re-entry guard
        # Widen the clickable area of the edge lines
        for _line in self._span_region.lines:
            _line.setHoverPen(pg.mkPen((255, 200, 50), width=14))

        # fres X marker (IQ plot) — same red as the amplitude-plot vline
        _fres_pen = pg.mkPen((255, 80, 80), width=2)
        self._fres_x = self._plot_iq.plot(
            pen=None, symbol='x', symbolSize=12,
            symbolBrush=pg.mkBrush(255, 80, 80, 255),
            symbolPen=_fres_pen,
            name='fres',
        )

        # Legends
        # Amp plot: lower-right corner
        # Use empty-data items added to the plot so pyqtgraph renders the
        # correct colours in the legend swatch.
        _fres_swatch = pg.PlotDataItem(
            x=[], y=[],
            pen=pg.mkPen((255, 80, 80), width=2, style=_Qt.DotLine),
        )
        _span_swatch = pg.PlotDataItem(
            x=[], y=[],
            pen=pg.mkPen((255, 200, 50), width=3),
        )
        self._plot_amp.addItem(_fres_swatch)
        self._plot_amp.addItem(_span_swatch)
        _leg_amp = self._plot_amp.addLegend()
        _leg_amp.addItem(self._amp_curve, '|S₂₁|')
        _leg_amp.addItem(_fres_swatch, 'fres')
        _leg_amp.addItem(_span_swatch, 'span')
        _leg_amp.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -10))

        # IQ plot: centre of frame; use actual items so colours match exactly
        _leg_iq = self._plot_iq.addLegend()
        _leg_iq.addItem(self._iq_scatter, 'IQ')
        _leg_iq.addItem(self._iq_span, 'span')
        _leg_iq.addItem(self._fres_x, 'fres')
        _leg_iq.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))

        _scale_plot_fonts(self._ui_scale, self._plot_amp, self._plot_iq)

        # R shortcut: auto-scale both plots
        _sc_r = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        _sc_r.activated.connect(self._autoscale_plots)

        # Z shortcut: reset fres/qres to initial values for current resonator
        _sc_z = QtGui.QShortcut(QtGui.QKeySequence("Z"), self)
        _sc_z.activated.connect(self._reset_current)

        # H shortcut: toggle help panel
        _sc_h = QtGui.QShortcut(QtGui.QKeySequence("H"), self)
        _sc_h.activated.connect(self._toggle_help)

        self.installEventFilter(self)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    @property
    def _ri(self) -> int:
        """Current resonator array index."""
        return self._interactive_indices[self._cursor]

    def _load_resonator(self):
        """Populate the plots and spinboxes for the current resonator."""
        ri = self._ri
        f = self._f[ri]         # (N,)
        z = self._z[ri]         # (N,)
        fres = self._fres_work[ri]
        qres = self._qres_work[ri]
        res_idx = self._res_idxs[ri]

        # Update status
        n_tot = len(self._interactive_indices)
        self._status_label.setText(
            f"Resonator {int(res_idx)}   "
            f"({self._cursor + 1}/{n_tot})"
        )

        # Block spinbox signals while updating programmatically
        self._fres_spin.blockSignals(True)
        self._qres_spin.blockSignals(True)
        if np.isnan(fres):
            self._fres_spin.setValue(0.0)
        else:
            self._fres_spin.setValue(fres * 1e-6)
        if np.isnan(qres):
            self._qres_spin.setValue(0.0)
        else:
            self._qres_spin.setValue(round(qres))
        self._fres_spin.blockSignals(False)
        self._qres_spin.blockSignals(False)

        # Static amplitude curve — x axis in kHz offset from initial fres
        self._f0_hz = self._fres_init[ri]  # fixed for this resonator
        f0_mhz = self._f0_hz * 1e-6
        f_khz = (f - self._f0_hz) * 1e-3
        db = _db(z)
        self._amp_curve.setData(f_khz, db)
        self._plot_amp.setLabel(
            'bottom',
            f'(f − {f0_mhz:.3f} MHz) (kHz)',
        )

        # Static IQ scatter
        self._iq_scatter.setData(z.real, z.imag)

        # Overlay markers
        self._update_overlay(f, z, fres, qres)

        # Restore rejection-reason widget state for this resonator
        stored_reason = self._reject_reasons.get(ri, None)
        self._reason_combo.blockSignals(True)
        if stored_reason is None:
            self._reason_combo.setCurrentText("not rejected")
            self._reason_edit.setEnabled(False)
            self._reason_edit.clear()
        elif stored_reason in self._REJECT_REASONS[:-1]:
            self._reason_combo.setCurrentText(stored_reason)
            self._reason_edit.setEnabled(False)
        else:
            self._reason_combo.setCurrentText("other")
            self._reason_edit.setText(
                "" if stored_reason == "other" else stored_reason
            )
            self._reason_edit.setEnabled(True)
        self._reason_combo.blockSignals(False)
        self._update_reject_label()

        # Autoscale after all items are updated, always deferred so pyqtgraph
        # has processed the new data before autoRange runs
        QtCore.QTimer.singleShot(0, self._autoscale_plots)

        # Navigation buttons
        self._prev_btn.setEnabled(self._cursor > 0)
        self._next_btn.setEnabled(True)

    def _update_overlay(self, f, z, fres, qres):
        """Redraw the fres vline, span region, and IQ X marker.
        If fres or qres is NaN, hide all overlay items."""
        if np.isnan(fres) or np.isnan(qres):
            self._fres_vline.setVisible(False)
            self._span_region.setVisible(False)
            self._iq_span.setData([], [])
            self._fres_x.setData([], [])
            return
        self._fres_vline.setVisible(True)
        self._span_region.setVisible(True)
        f0_hz = self._f0_hz
        span_hz = fres / qres
        fres_khz = (fres - f0_hz) * 1e-3
        fmin_khz = (fres - span_hz / 2 - f0_hz) * 1e-3
        fmax_khz = (fres + span_hz / 2 - f0_hz) * 1e-3

        self._fres_updating = True
        self._fres_vline.setValue(fres_khz)
        self._fres_updating = False
        self._span_updating = True
        self._span_region.setRegion([fmin_khz, fmax_khz])
        self._span_updating = False

        # Highlight IQ points within span
        ix_span = (f >= fres - span_hz / 2) & (f <= fres + span_hz / 2)
        self._iq_span.setData(z.real[ix_span], z.imag[ix_span])

        # X at the interpolated IQ position at fres
        x_fres = np.interp(fres, f, z.real)
        y_fres = np.interp(fres, f, z.imag)
        self._fres_x.setData([x_fres], [y_fres])

    def _autoscale_plots(self):
        """Auto-range both plots."""
        self._plot_amp.autoRange()
        self._plot_iq.autoRange()

    def _reject_current(self):
        """Reject the current resonator using the current combo selection.

        If the combo is on 'not rejected', switches to the first predefined
        rejection reason. This method is kept for backward compatibility
        (e.g. the test suite calls it directly).
        """
        if self._reason_combo.currentText() == "not rejected":
            self._reason_combo.setCurrentText(self._REJECT_REASONS[0])
        else:
            # Re-apply (idempotent); picks up any edit-field changes for "other"
            self._on_reason_combo_changed(self._reason_combo.currentText())

    def _on_reason_combo_changed(self, text: str):
        """Apply or remove rejection based on the combo selection."""
        self._reason_edit.setEnabled(text == "other")
        ri = self._ri
        if text == "not rejected":
            # Un-reject: restore pre-rejection values (or initial values)
            self._reject_reasons.pop(ri, None)
            pre = self._pre_reject.get(ri)
            if pre is not None:
                fres, qres = pre
            else:
                fres, qres = self._fres_init[ri], self._qres_init[ri]
            self._fres_work[ri] = fres
            self._qres_work[ri] = qres
            self._fres_spin.blockSignals(True)
            self._qres_spin.blockSignals(True)
            self._fres_spin.setValue(fres * 1e-6)
            self._qres_spin.setValue(round(qres))
            self._fres_spin.blockSignals(False)
            self._qres_spin.blockSignals(False)
            self._update_overlay(self._f[ri], self._z[ri], fres, qres)
        else:
            # Reject: save current fres/qres to pre-reject cache, then NaN them
            reason = text if text != "other" else (
                self._reason_edit.text().strip() or "other"
            )
            if not np.isnan(self._fres_work[ri]):
                self._pre_reject[ri] = (self._fres_work[ri], self._qres_work[ri])
            self._reject_reasons[ri] = reason
            self._fres_work[ri] = np.nan
            self._qres_work[ri] = np.nan
            self._fres_spin.blockSignals(True)
            self._qres_spin.blockSignals(True)
            self._fres_spin.setValue(0.0)
            self._qres_spin.setValue(0.0)
            self._fres_spin.blockSignals(False)
            self._qres_spin.blockSignals(False)
            self._update_overlay(self._f[ri], self._z[ri], np.nan, np.nan)
        self._update_reject_label()
        self.setFocus()

    def _on_reason_edit_finished(self):
        """Update the stored rejection reason when the custom text field changes."""
        ri = self._ri
        if self._reason_combo.currentText() == "other" and ri in self._reject_reasons:
            custom = self._reason_edit.text().strip() or "other"
            self._reject_reasons[ri] = custom
            self._update_reject_label()

    def _update_reject_label(self):
        """Refresh the rejection status indicator label."""
        ri = self._ri
        reason = self._reject_reasons.get(ri, None)
        pt = max(7, round(9 * self._ui_scale))
        base_style = f"font-size: {pt}pt; font-weight: bold;"
        if reason is None:
            self._reject_label.setText("✓ Not rejected")
            self._reject_label.setStyleSheet(f"color: #4f4; {base_style}")
        else:
            self._reject_label.setText("✗ Rejected")
            self._reject_label.setStyleSheet(f"color: #f44; {base_style}")

    def _reset_current(self):
        """Reset fres/qres for the current resonator to their initial values."""
        ri = self._ri
        self._reject_reasons.pop(ri, None)
        self._pre_reject.pop(ri, None)
        self._fres_work[ri] = self._fres_init[ri]
        self._qres_work[ri] = self._qres_init[ri]
        self._reason_combo.blockSignals(True)
        self._reason_combo.setCurrentText("not rejected")
        self._reason_edit.setEnabled(False)
        self._reason_edit.clear()
        self._reason_combo.blockSignals(False)
        self._fres_spin.blockSignals(True)
        self._qres_spin.blockSignals(True)
        self._fres_spin.setValue(self._fres_work[ri] * 1e-6)
        self._qres_spin.setValue(round(self._qres_work[ri]))
        self._fres_spin.blockSignals(False)
        self._qres_spin.blockSignals(False)
        self._update_overlay(
            self._f[ri], self._z[ri],
            self._fres_work[ri], self._qres_work[ri],
        )
        self._update_reject_label()

    def _toggle_help(self):
        """Toggle the floating help panel on/off."""
        if not hasattr(self, '_help_dlg'):
            self._help_dlg = self._build_help_dialog()
        if self._help_dlg.isVisible():
            self._help_dlg.hide()
        else:
            dlg = self._help_dlg
            dlg.adjustSize()
            # Centre within the main window using the stored screen geometry
            # (avoids platform-specific inaccuracies from frameGeometry())
            _sc = self._screen_geom
            _cx = _sc.x() + _sc.width()  // 2
            _cy = _sc.y() + _sc.height() // 2
            dlg.move(_cx - dlg.width() // 2, _cy - dlg.height() // 2)
            dlg.show()

    def _build_help_dialog(self) -> QtWidgets.QDialog:
        """Create the floating help dialog (created lazily on first use)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("FQ Finder \u2014 Help (H to close)")
        layout = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel(
            "<h3>FQ Finder Controls</h3>"
            "<p><b>Amplitude plot</b></p>"
            "<ul>"
            "<li><b>Shift+Click:</b> Move fres to clicked frequency</li>"
            "<li><b>Drag fres line:</b> Adjust fres directly</li>"
            "<li><b>Drag span edge:</b> Adjust qres symmetrically</li>"
            "<li><b>Shift+Scroll:</b> Fine qres adjust (\u00b14% per notch)</li>"
            "<li><b>Ctrl+Scroll:</b> Coarse qres adjust (\u00b140% per notch)</li>"
            "</ul>"
            "<p><b>IQ plot</b></p>"
            "<ul>"
            "<li><b>Shift+Click:</b> Snap fres to nearest IQ sample</li>"
            "<li><b>Shift/Ctrl+Scroll:</b> Adjust qres (same as amp plot)</li>"
            "</ul>"
            "<p><b>Keyboard</b></p>"
            "<ul>"
            "<li><b>\u2192 / D:</b> Save current resonator and go to next</li>"
            "<li><b>\u2190 / A:</b> Go back to previous resonator</li>"
            "<li><b>R:</b> Auto-scale both plots</li>"
            "<li><b>Z:</b> Reset fres/qres to initial values for this resonator</li>"
            "<li><b>H:</b> Toggle this help panel</li>"
            "</ul>"
            "<p><b>Rejection</b></p>"
            "<ul>"
            "<li>Select a rejection reason from the dropdown to reject a resonator.</li>"
            "<li>Selecting <i>not rejected</i> restores the last fres/qres values.</li>"
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

    def _save_current(self):
        """Write working fres/qres and rejection reason for the current
        resonator to zarr."""
        ri = self._ri
        self._zg["fres_opt"][ri] = self._fres_work[ri]
        self._zg["qres_opt"][ri] = self._qres_work[ri]
        self._zg["reject_reason"][ri] = self._reject_reasons.get(ri, "")

    # ------------------------------------------------------------------
    # Public navigation (also called by key/button handlers)
    # ------------------------------------------------------------------

    def _go_next(self):
        self._save_current()
        if self._cursor < len(self._interactive_indices) - 1:
            self._cursor += 1
            self._load_resonator()
        else:
            self._status_label.setText("All resonators complete — close to exit.")
            self._next_btn.setEnabled(False)

    def _go_back(self):
        if self._cursor > 0:
            self._save_current()
            self._cursor -= 1
            self._load_resonator()

    # ------------------------------------------------------------------
    # fres / qres change helpers
    # ------------------------------------------------------------------

    def _set_fres(self, fres: float):
        ri = self._ri
        self._fres_work[ri] = fres
        self._fres_spin.blockSignals(True)
        self._fres_spin.setValue(fres * 1e-6)
        self._fres_spin.blockSignals(False)
        self._update_overlay(self._f[ri], self._z[ri], fres, self._qres_work[ri])

    def _set_qres(self, qres: float):
        ri = self._ri
        qres = max(1.0, qres)
        self._qres_work[ri] = qres
        self._qres_spin.blockSignals(True)
        self._qres_spin.setValue(round(qres))
        self._qres_spin.blockSignals(False)
        self._update_overlay(self._f[ri], self._z[ri], self._fres_work[ri], qres)

    # ------------------------------------------------------------------
    # Spinbox callbacks
    # ------------------------------------------------------------------

    def _on_fres_vline_moved(self):
        """User dragged the fres vline → update fres."""
        if self._fres_updating:
            return
        fres_hz = self._fres_vline.value() * 1e3 + self._f0_hz
        self._set_fres(fres_hz)

    def _on_span_region_changed(self):
        """Drag a span edge → adjust qres symmetrically around fres."""
        if self._span_updating:
            return
        ri = self._ri
        if np.isnan(self._fres_work[ri]):
            return
        fmin_khz, fmax_khz = self._span_region.getRegion()
        fres_khz = (self._fres_work[ri] - self._f0_hz) * 1e-3
        half_left = fres_khz - fmin_khz
        half_right = fmax_khz - fres_khz
        new_half_hz = max((half_left + half_right) / 2.0 * 1e3, 1e-3)
        span_hz = new_half_hz * 2.0
        qres = self._fres_work[ri] / span_hz
        self._set_qres(qres)

    def _on_spinbox_fres(self, value: float):
        ri = self._ri
        fres_hz = value * 1e6
        self._fres_work[ri] = fres_hz
        if np.isnan(self._qres_work[ri]):
            return
        self._update_overlay(self._f[ri], self._z[ri], fres_hz, self._qres_work[ri])

    def _on_spinbox_qres(self, value: float):
        ri = self._ri
        self._qres_work[ri] = max(1.0, value)
        self._update_overlay(
            self._f[ri], self._z[ri], self._fres_work[ri], self._qres_work[ri]
        )

    # ------------------------------------------------------------------
    # Interactive-ViewBox callbacks
    # ------------------------------------------------------------------

    def _on_shift_click_amp(self, x_khz: float, _y: float):
        """Shift+click on amplitude plot → move fres to clicked position."""
        fres = x_khz * 1e3 + self._f0_hz
        ri = self._ri
        if np.isnan(self._qres_work[ri]):
            self._qres_work[ri] = self._qres_init[ri]
        self._set_fres(fres)

    def _on_shift_click_iq(self, x_re: float, y_im: float):
        """Shift+click on IQ plot → snap fres to nearest sample."""
        ri = self._ri
        z = self._z[ri]
        f = self._f[ri]
        dist = (z.real - x_re) ** 2 + (z.imag - y_im) ** 2
        ix = int(np.argmin(dist))
        if np.isnan(self._qres_work[ri]):
            self._qres_work[ri] = self._qres_init[ri]
        self._set_fres(float(f[ix]))

    def _on_scroll_qres(self, steps: int):
        """Shift/Ctrl + scroll → multiplicative qres adjustment."""
        ri = self._ri
        qres = self._qres_work[ri]
        if np.isnan(qres):
            return
        qres = qres * (1.0 + steps * self._QRES_FRAC)
        self._set_qres(qres)

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if event.type() == _QEVENT_KEY_PRESS:
            key = event.key()
            mods = event.modifiers()
            # Skip if Shift or Ctrl is held (reserved for plot interaction)
            if mods & (_Qt.ShiftModifier | _Qt.ControlModifier):
                return False
            if key in (_Qt.Key_Right, _Qt.Key_D):
                self._go_next()
                return True
            if key in (_Qt.Key_Left, _Qt.Key_A):
                self._go_back()
                return True
            if key == _Qt.Key_R:
                self._autoscale_plots()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._save_current()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fqfinder(
    f,
    z,
    fres,
    qres,
    res_idxs,
    zarr_group,
    title: str = "FQ Finder",
    ui_scale: float = None,    # deprecated — ignored; scale is auto-computed
    plot_scale: float = None,  # deprecated — ignored; scale is auto-computed
    overwrite: bool = False,
    fres_update_method: str = "none",
    start_idx: int = 0,
    rmv_gain_simple: bool = False,
) -> None:
    """
    Launch the interactive resonance frequency and Q-factor finder.

    Calibration tones (``res_idxs[i] < 1``) are saved immediately with
    their input values and skipped in the interactive loop.

    Parameters
    ----------
    f : array-like, shape (M, N)
        Frequency data in Hz.
    z : array-like, shape (M, N)
        Complex IQ data.
    fres : array-like, length M
        Initial resonant frequencies in Hz.
    qres : array-like, length M
        Initial Q-factors.
    res_idxs : array-like, length M
        Resonator indices.  Values < 1 mark calibration tones.
    zarr_group : zarr.Group
        Output group.  Three arrays of length M are written here:
        ``fres_opt`` (float64), ``qres_opt`` (float64), and
        ``reject_reason`` (str, empty string for non-rejected resonators).
    title : str, optional
        Window title.  Default ``'FQ Finder'``.
    ui_scale : float, optional
        Deprecated.  Accepted for backward compatibility but has no effect.
        The window is automatically sized to 80% of the screen and all fonts
        scale accordingly.
    plot_scale : float, optional
        Deprecated.  Accepted for backward compatibility but has no effect.
    overwrite : bool, optional
        If False (default) and ``fres_opt`` or ``qres_opt`` already exist
        in *zarr_group*, raise ``FileExistsError``.  Set True to resume:
        existing arrays are kept and only rows written during this session
        are updated.
    fres_update_method : str, optional
        Algorithm used to automatically update ``fres`` before the
        interactive session begins.  Passed directly to
        :func:`citkid.multitone.fres.update_fres`.  'mins21' finds the
        minimum of |S21| after subtracting a linear baseline.  'spacing'
        finds the point of maximum adjacent IQ spacing.  'distance' finds
        the point furthest from the off-resonance IQ value.  'none'
        (default) skips automatic updating and uses the input ``fres``
        values as the starting point.
    start_idx : int, optional
        Index into the resonator arrays at which to begin the interactive
        session.  The cursor starts at the first interactive resonator
        whose array index is >= *start_idx*.  The user can still navigate
        backwards to earlier resonators.  Default 0.
    rmv_gain_simple : bool, optional
        If True, apply a simple gain correction to *z* before displaying.
        Divides each row by the median off-resonance amplitude, then rotates
        so the off-resonance mean lies on the positive real axis.
        Default False.

    Returns
    -------
    None
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = FqFinderWindow(
        f=f,
        z=z,
        fres=fres,
        qres=qres,
        res_idxs=res_idxs,
        zarr_group=zarr_group,
        title=title,
        overwrite=overwrite,
        fres_update_method=fres_update_method,
        start_idx=start_idx,
        rmv_gain_simple=rmv_gain_simple,
    )
    win.show()
    # exec() is the modern name (PyQt6+); exec_() is kept for PyQt5/PySide2.
    (getattr(app, 'exec', None) or app.exec_)()
