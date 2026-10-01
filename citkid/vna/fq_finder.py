"""
Interactive resonance frequency and Q-factor finder.

Displays, for each resonator index *i*:

    Left  plot — 20·log10(|z[i]|) vs f[i]  (dB amplitude)
    Right plot — z[i].imag vs z[i].real       (IQ loop)

Overlaid markers:
* Vertical dashed line on the left plot at ``fres[i]``.
* Shaded band of width ``fres[i]/qres[i]`` centred on ``fres[i]``.
* An **X** on the right plot at the IQ point closest to ``fres[i]``.

Interaction:
* **Shift + left-click** on either plot  → move ``fres`` to the clicked
  frequency (left) or the nearest sample in IQ space (right).
* **Shift + scroll wheel** on either plot → adjust ``qres`` (fine) by
  ±1 step per click.
* **Ctrl  + scroll wheel** on either plot → adjust ``qres`` (coarse) by
  ±10 steps per click.
* **Right arrow / D**  → save current values and advance to next resonator.
* **Left  arrow / A**  → go back one resonator (re-opens with saved values).
* **Closing the window** → saves current values and exits.

Calibration tones:
Resonators whose ``res_idxs[i] < 0`` are treated as calibration tones: the
input ``fres`` and ``qres`` values are written directly to the output zarr
arrays and the interactive step is skipped.

Output:
These zarr arrays are written into ``zarr_group``:
    ``fres_opt``           — optimised resonant frequency (Hz)
    ``qres_opt``           — optimised Q-factor
    ``reject_reason``      — rejection reason ('' if not rejected)
    ``fq_finder_data_idx`` — data index shown at the last save (-1 if none)

Existing data:
If ``zarr_group`` already has ``fres_opt`` and ``qres_opt``, a dialog asks
whether to load them, overwrite them (with a second confirmation), or
cancel. When loading, a second dialog asks whether to resume at the saved
data index or start from data index 0.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..qt_compat import (
    Qt as _Qt,
    delete_on_close,
    available_screen_geometry,
    fit_window_to_screen,
    get_qapp,
)
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
    """
    Apply *ui_scale* to tick labels, axis labels, and titles.

    Parameters:
    ui_scale (float): font size multiplier. Tick fonts are
        ``max(6, round(9 * ui_scale))`` pt and label/title fonts are
        ``max(7, round(11 * ui_scale))`` pt.
    *plot_items (tuple): pyqtgraph PlotItems whose fonts are updated.
    """
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
    """
    Convert complex data to amplitude in dB.

    Parameters:
    z (np.array): complex S21 data.

    Returns:
    db (np.array): ``20 * log10(|z|)``.
    """
    return 20.0 * np.log10(np.abs(z))


# Saved data index of the resonator on screen at the last save (-1 = never saved)
_DATA_IDX_KEY = "fq_finder_data_idx"
_OUTPUT_KEYS = ("fres_opt", "qres_opt", "reject_reason", _DATA_IDX_KEY)


def _ensure_data_idx_array(zarr_group) -> None:
    """
    Create the saved data index array if it is missing.

    Parameters:
    zarr_group (zarr.Group): output group.
    """
    if _DATA_IDX_KEY not in zarr_group:
        zarr_group.create_array(
            _DATA_IDX_KEY, shape=(1,), dtype=np.int64, fill_value=-1,
        )


def _ensure_zarr_arrays(zarr_group, n: int) -> bool:
    """
    Ensure the fq_finder output arrays exist in the zarr group.

    If ``fres_opt`` and ``qres_opt`` both exist, they are kept (resume) and
    any missing ``reject_reason`` or saved data index array is created
    (legacy groups). If neither exists, all output arrays are created fresh.

    Parameters:
    zarr_group (zarr.Group): output group.
    n (int): number of resonators (length of each per-resonator array).

    Returns:
    existed (bool): True if ``fres_opt`` and ``qres_opt`` already existed,
        False if they were freshly created.

    Raises:
    RuntimeError: if only one of ``fres_opt`` and ``qres_opt`` exists.
    ValueError: if the existing arrays do not have length ``n``.
    """
    existing = [name for name in ("fres_opt", "qres_opt") if name in zarr_group]
    if len(existing) == 1:
        raise RuntimeError(
            f"zarr_group contains {existing} but not the other; "
            "the group is in an inconsistent state."
        )
    if len(existing) == 2:
        for name in existing:
            if zarr_group[name].shape != (n,):
                raise ValueError(
                    f"zarr_group[{name!r}] has shape {zarr_group[name].shape}, "
                    f"expected ({n},) to match the input data."
                )
        # Resume mode: ensure reject_reason exists (may be absent in legacy groups)
        if "reject_reason" not in zarr_group:
            zarr_group.create_dataset(
                "reject_reason",
                shape=(n,),
                dtype=str,
                fill_value="",
            )
        _ensure_data_idx_array(zarr_group)
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
    _ensure_data_idx_array(zarr_group)
    return False


def _clear_zarr_outputs(zarr_group) -> None:
    """
    Delete all fq_finder output arrays from the zarr group.

    Parameters:
    zarr_group (zarr.Group): output group.
    """
    for key in _OUTPUT_KEYS:
        if key in zarr_group:
            del zarr_group[key]


def _read_saved_data_idx(zarr_group, res_idxs):
    """
    Read the saved data index, if it points at an interactive resonator.

    Parameters:
    zarr_group (zarr.Group): output group.
    res_idxs (np.array): resonator indices; values < 0 are calibration tones.

    Returns:
    data_idx (int or None): saved data index, or None if none was saved, it
        is out of range, or it points at a calibration tone.
    """
    if _DATA_IDX_KEY not in zarr_group:
        return None
    data_idx = int(np.asarray(zarr_group[_DATA_IDX_KEY][:]).ravel()[0])
    if not 0 <= data_idx < len(res_idxs) or res_idxs[data_idx] < 0:
        return None
    return data_idx


def _show_startup_dialog() -> str:
    """
    Ask what to do with existing fq_finder data in the zarr group.

    Choosing overwrite opens a second confirmation dialog; answering No
    returns to the first dialog.

    Returns:
    choice (str): 'load', 'overwrite', or 'cancel' (also returned if the
        dialog is closed).
    """
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle('Zarr Data Exists')
    layout = QtWidgets.QVBoxLayout(dlg)

    label = QtWidgets.QLabel(
        '<h3>Existing Data Found</h3>'
        '<p>The zarr group already contains fres_opt and qres_opt.</p>'
        '<p>Choose an option:</p>'
    )
    label.setTextFormat(_Qt.RichText)
    layout.addWidget(label)

    load_btn = QtWidgets.QPushButton('Load from Zarr')
    load_btn.setToolTip(
        'Load saved fres, qres, and rejections and continue editing')
    overwrite_btn = QtWidgets.QPushButton('Overwrite (Start Fresh)')
    overwrite_btn.setToolTip(
        'Delete saved data and start from the input fres and qres')
    cancel_btn = QtWidgets.QPushButton('Cancel')
    cancel_btn.setToolTip('Exit without doing anything')
    for btn in (load_btn, overwrite_btn, cancel_btn):
        layout.addWidget(btn)

    result = ['cancel']

    def on_load():
        """
        Record the 'load' choice and accept the dialog.
        """
        result[0] = 'load'
        dlg.accept()

    def on_overwrite():
        """
        Confirm the overwrite, then record 'overwrite' and accept the dialog.

        If the user answers No, the startup dialog stays open.
        """
        reply = QtWidgets.QMessageBox.question(
            dlg,
            'Confirm Overwrite',
            'Are you sure you want to overwrite existing data?\n\n'
            'This will permanently delete the saved fres_opt, qres_opt, '
            'and rejection reasons.',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,  # Default to No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            result[0] = 'overwrite'
            dlg.accept()
        # If No, dialog stays open

    def on_cancel():
        """
        Record the 'cancel' choice and reject the dialog.
        """
        result[0] = 'cancel'
        dlg.reject()

    load_btn.clicked.connect(on_load)
    overwrite_btn.clicked.connect(on_overwrite)
    cancel_btn.clicked.connect(on_cancel)

    (getattr(dlg, 'exec', None) or dlg.exec_)()
    return result[0]


def _ask_start_idx(saved_idx: int, res_idx) -> int:
    """
    Ask whether to resume at the saved data index or start from index 0.

    Parameters:
    saved_idx (int): data index saved in the zarr group.
    res_idx (int): resonator index at ``saved_idx``, shown for reference.

    Returns:
    start_idx (int): ``saved_idx`` to resume, or 0 to start over (also
        returned if the dialog is closed).
    """
    box = QtWidgets.QMessageBox()
    box.setWindowTitle('Resume Position')
    box.setText(
        f'The saved data index is {saved_idx} (resonator {int(res_idx)}).\n\n'
        'Where do you want to start?'
    )
    resume_btn = box.addButton(
        f'Resume at data index {saved_idx}',
        QtWidgets.QMessageBox.ButtonRole.AcceptRole,
    )
    box.addButton(
        'Start from data index 0',
        QtWidgets.QMessageBox.ButtonRole.RejectRole,
    )
    box.setDefaultButton(resume_btn)
    (getattr(box, 'exec', None) or box.exec_)()
    return saved_idx if box.clickedButton() is resume_btn else 0


def _rmv_gain_simple(z: np.ndarray) -> np.ndarray:
    """
    Remove the gain amplitude and phase offset from (M, N) sweep data.

    Parameters:
    z (np.array): complex S21 data, shape (M, N).

    Returns:
    z (np.array): corrected complex S21 data, shape (M, N).

    Notes:
    Uses the off-resonance edge samples of each row, the first and last
    ``n = max(1, N // 100)`` points:
    1. Amplitude: divide each row by the median |z| of the edge samples.
    2. Phase offset: rotate each row so the mean of the edge samples lies on
       the positive real axis.
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
        """
        Emit ``sig_shift_click`` on Shift + left-click, else pass the event on.

        Parameters:
        ev (MouseClickEvent): pyqtgraph mouse press event. The click
            position is mapped to view coordinates before emitting.
        """
        if (ev.button() == _Qt.LeftButton
                and ev.modifiers() & _Qt.ShiftModifier):
            ev.accept()
            pos = self.mapToView(ev.pos())
            self.sig_shift_click.emit(pos.x(), pos.y())
        else:
            super().mousePressEvent(ev)

    def wheelEvent(self, ev, axis=None):
        """
        Emit ``sig_scroll_qres`` on Shift/Ctrl + scroll, else zoom as usual.

        Parameters:
        ev (QGraphicsSceneWheelEvent): wheel event. Shift gives a step of
            +/-1 per notch and Ctrl a step of +/-10, signed by the scroll
            direction.
        axis (int or None): axis to zoom, passed to the base class when no
            modifier is held. None zooms both axes.
        """
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
    """
    LinearRegionItem whose body is not interactive (only the edge lines are).
    """
    def mouseDragEvent(self, ev):
        """
        Ignore drags on the region body so it cannot be moved.

        Parameters:
        ev (MouseDragEvent): pyqtgraph mouse drag event.
        """
        ev.ignore()

    def mousePressEvent(self, ev):
        """
        Ignore presses on the region body so they reach items beneath it.

        Parameters:
        ev (QGraphicsSceneMouseEvent): mouse press event.
        """
        ev.ignore()

    def hoverEvent(self, ev):
        """
        Suppress the body hover highlight.

        Parameters:
        ev (HoverEvent): pyqtgraph hover event. Not used.
        """
        # Suppress body hover highlight; edge InfiniteLines handle their own hover
        pass


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class FqFinderWindow(QtWidgets.QMainWindow):
    """
    Stand-alone QMainWindow that hosts the interactive fres/qres editor.

    Shows the amplitude and IQ loop of one resonator at a time, lets the
    user adjust fres and qres or reject the resonator, and writes the
    results to a zarr group.
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
        ui_scale: float = 1.0,
        fres_update_method: str = "none",
        start_idx: int = 0,
        rmv_gain_simple: bool = False,
        load_saved: bool = False,
    ):
        """
        Initialize the window, prepare the data and zarr arrays, and build the
        UI.

        Parameters:
        f (np.array): frequency data in Hz, shape (M, N).
        z (np.array): complex IQ data, shape (M, N).
        fres (array-like): initial resonant frequencies in Hz, length M.
        qres (array-like): initial Q-factors, length M.
        res_idxs (array-like): resonator indices, length M. Values < 0 mark
            calibration tones.
        zarr_group (zarr.Group): output group. ``fres_opt`` and ``qres_opt``
            1-D arrays of length M are created here; existing arrays are
            kept (resume).
        title (str): window title.
        ui_scale (float or None): font size multiplier. Default 1.0; None is
            treated as 1.0. The window is sized to 80% of the screen
            independently of this.
        fres_update_method (str): automatic fres update applied to the input
            ``fres`` before the session; see ``run_fqfinder``.
        start_idx (int): data index to start at. The cursor starts at the
            first interactive resonator whose index is >= ``start_idx`` (0
            if there is none).
        rmv_gain_simple (bool): If True, apply a simple gain correction to
            ``z`` before displaying.
        load_saved (bool): If True, start from the saved ``fres_opt``,
            ``qres_opt``, and ``reject_reason`` values in ``zarr_group``
            instead of the input values, for resonators that were saved. If
            False (default), existing arrays are kept but the input values
            are shown.
        """
        super().__init__()
        delete_on_close(self)  # destroy on the GUI thread when closed
        self.setWindowTitle(title)

        self._f = np.asarray(f, dtype=np.float64)           # (M, N)
        self._z = np.asarray(z, dtype=complex)              # (M, N)
        # Sort along axis 1 so frequencies are always ascending
        _sort_idx = np.argsort(self._f, axis=1)
        self._f = np.take_along_axis(self._f, _sort_idx, axis=1)
        self._z = np.take_along_axis(self._z, _sort_idx, axis=1)
        if rmv_gain_simple:
            self._z = _rmv_gain_simple(self._z)
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

        # Qt handles display scaling (see get_qapp), so fonts only use the
        # user multiplier; None is accepted for backward compatibility.
        self._ui_scale = 1.0 if ui_scale is None else float(ui_scale)
        self._screen_geom = available_screen_geometry()

        self._reject_reasons: dict = {}
        self._pre_reject: dict = {}   # fres/qres saved before rejection

        _ensure_zarr_arrays(self._zg, self._M)
        if load_saved:
            self._load_saved_state()

        # Pre-save calibration tones immediately (they are never interactive)
        for i in range(self._M):
            if self._res_idxs[i] < 0:
                self._zg["fres_opt"][i] = self._fres_work[i]
                self._zg["qres_opt"][i] = self._qres_work[i]

        # Build the list of interactive indices (non-cal tones)
        self._interactive_indices = [
            i for i in range(self._M) if self._res_idxs[i] >= 0
        ]
        if not self._interactive_indices:
            # Nothing to edit: the UI is not built and run_fqfinder does not
            # show the window
            return

        # Find the cursor position: first interactive index >= start_idx
        start_idx = max(0, int(start_idx))
        self._cursor = next(
            (pos for pos, i in enumerate(self._interactive_indices) if i >= start_idx),
            0,
        )

        self._build_ui()
        # Use 80% of the screen and centre on it
        fit_window_to_screen(self, frac=0.8)
        self._load_resonator()

    def _load_saved_state(self):
        """
        Load saved fres, qres, and rejection reasons from the zarr group.

        For each interactive resonator, a saved rejection reason marks it
        rejected; otherwise saved fres and qres replace the working values.
        Resonators that were never saved (NaN and no reason) keep the input
        values. The input values remain the reset (Z) targets.
        """
        saved_f = np.asarray(self._zg["fres_opt"][:], dtype=np.float64)
        saved_q = np.asarray(self._zg["qres_opt"][:], dtype=np.float64)
        reasons = np.asarray(self._zg["reject_reason"][:], dtype=object)
        for i in range(self._M):
            if self._res_idxs[i] < 0:
                continue  # calibration tones always use the input values
            reason = "" if reasons[i] is None else str(reasons[i])
            if reason:
                self._reject_reasons[i] = reason
                self._fres_work[i] = np.nan
                self._qres_work[i] = np.nan
            elif not (np.isnan(saved_f[i]) or np.isnan(saved_q[i])):
                self._fres_work[i] = saved_f[i]
                self._qres_work[i] = saved_q[i]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """
        Build the control bars, hint bar, plots, overlays, legends, and
        keyboard shortcuts.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(6, 4, 6, 6)
        vbox.setSpacing(4)

        # Widget font — set before creating widgets so the font-metric
        # widths below use it; all child widgets inherit it.
        widget_pt = max(7, round(9 * self._ui_scale))
        widget_font = QtGui.QFont()
        widget_font.setPointSize(widget_pt)
        central.setFont(widget_font)
        _fm = QtGui.QFontMetrics(widget_font)

        def _text_width(sample: str) -> int:
            """
            Measure the width in pixels of *sample* in the widget font.

            Parameters:
            sample (str): text to measure.

            Returns:
            width (int): horizontal advance of ``sample`` in pixels.
            """
            return _fm.horizontalAdvance(sample)

        # Widths come from each widget's own size hint (buttons, spinboxes,
        # combo) or from the font metrics, never fixed pixel counts, so text
        # always fits whatever the font size.

        # ---- top control bar (row 1): navigation + fres/qres + status ----
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(10)

        self._prev_btn = QtWidgets.QPushButton("◀  Back (A/←)")
        self._prev_btn.clicked.connect(self._go_back)
        ctrl.addWidget(self._prev_btn)

        self._next_btn = QtWidgets.QPushButton("Next (D/→)  ►")
        self._next_btn.clicked.connect(self._go_next)
        ctrl.addWidget(self._next_btn)

        ctrl.addSpacing(20)

        ctrl.addWidget(QtWidgets.QLabel("fres (MHz):"))
        self._fres_spin = QtWidgets.QDoubleSpinBox()
        self._fres_spin.setDecimals(4)
        self._fres_spin.setRange(0.0, 1e6)
        self._fres_spin.setSingleStep(0.001)
        self._fres_spin.setKeyboardTracking(False)
        self._fres_spin.valueChanged.connect(self._on_spinbox_fres)
        ctrl.addWidget(self._fres_spin)

        ctrl.addSpacing(10)

        ctrl.addWidget(QtWidgets.QLabel("qres:"))
        self._qres_spin = QtWidgets.QDoubleSpinBox()
        self._qres_spin.setDecimals(0)
        self._qres_spin.setRange(1.0, 1e8)
        self._qres_spin.setSingleStep(1000.0)
        self._qres_spin.setKeyboardTracking(False)
        self._qres_spin.valueChanged.connect(self._on_spinbox_qres)
        ctrl.addWidget(self._qres_spin)

        ctrl.addStretch()

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setMinimumWidth(
            _text_width("Resonator 00000   (0000/0000)"))
        ctrl.addWidget(self._status_label)

        vbox.addLayout(ctrl)

        # ---- control bar row 2: rejection ----
        reject_row = QtWidgets.QHBoxLayout()
        reject_row.setSpacing(10)

        self._reason_combo = QtWidgets.QComboBox()
        self._reason_combo.addItem("not rejected")
        for _r in self._REJECT_REASONS:
            self._reason_combo.addItem(_r)
        self._reason_combo.currentTextChanged.connect(self._on_reason_combo_changed)
        reject_row.addWidget(self._reason_combo)

        self._reason_edit = QtWidgets.QLineEdit()
        self._reason_edit.setPlaceholderText("custom reason…")
        self._reason_edit.setMinimumWidth(_text_width("custom reason…" + "x" * 6))
        self._reason_edit.editingFinished.connect(self._on_reason_edit_finished)
        reject_row.addWidget(self._reason_edit)

        self._reject_label = QtWidgets.QLabel("")
        # Label is bold (see _update_reject_label), so leave some margin
        self._reject_label.setMinimumWidth(_text_width("✓ Not rejected  xx"))
        reject_row.addWidget(self._reject_label)

        reject_row.addStretch()
        vbox.addLayout(reject_row)

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
        # Wrap rather than force the window wider than the screen
        hint.setWordWrap(True)
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
        """
        Return the current resonator array index.

        Returns:
        ri (int): data index of the resonator at the cursor.
        """
        return self._interactive_indices[self._cursor]

    def _load_resonator(self):
        """
        Populate the plots and spinboxes for the current resonator.
        """
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
        """
        Redraw the fres vline, span region, and IQ X marker.

        If fres or qres is NaN, hide all overlay items.

        Parameters:
        f (np.array): frequency data in Hz for the resonator, shape (N,).
        z (np.array): complex IQ data for the resonator, shape (N,).
        fres (float): resonant frequency in Hz, or NaN.
        qres (float): Q-factor, or NaN. The span is ``fres / qres``.
        """
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
        """
        Auto-range both plots.
        """
        self._plot_amp.autoRange()
        self._plot_iq.autoRange()

    def _reject_current(self):
        """
        Reject the current resonator using the current combo selection.

        If the combo is on 'not rejected', switch to the first predefined
        rejection reason. This method is kept for backward compatibility
        (e.g. the test suite calls it directly).
        """
        if self._reason_combo.currentText() == "not rejected":
            self._reason_combo.setCurrentText(self._REJECT_REASONS[0])
        else:
            # Re-apply (idempotent); picks up any edit-field changes for "other"
            self._on_reason_combo_changed(self._reason_combo.currentText())

    def _on_reason_combo_changed(self, text: str):
        """
        Apply or remove rejection based on the combo selection.

        Parameters:
        text (str): selected combo text. 'not rejected' restores the
            pre-rejection fres/qres (or the initial values); any other
            reason stores the reason and sets fres/qres to NaN. For 'other',
            the custom text field is used as the reason ('other' if empty).
        """
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
        """
        Update the stored rejection reason when the custom text field changes.
        """
        ri = self._ri
        if self._reason_combo.currentText() == "other" and ri in self._reject_reasons:
            custom = self._reason_edit.text().strip() or "other"
            self._reject_reasons[ri] = custom
            self._update_reject_label()

    def _update_reject_label(self):
        """
        Refresh the rejection status indicator label.
        """
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
        """
        Reset fres/qres for the current resonator to their initial values.

        Also clear any rejection of the current resonator.
        """
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
        """
        Toggle the floating help panel on/off.
        """
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
        """
        Create the floating help dialog (created lazily on first use).

        Returns:
        dlg (QtWidgets.QDialog): help dialog, hidden by its Close button or
            the H key.
        """
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
        """
        Write working fres/qres, rejection reason, and data index for the
        current resonator to zarr.
        """
        ri = self._ri
        self._zg["fres_opt"][ri] = self._fres_work[ri]
        self._zg["qres_opt"][ri] = self._qres_work[ri]
        self._zg[_DATA_IDX_KEY][0] = ri
        # Handle reject_reason: read entire array, modify, and write back
        reason = str(self._reject_reasons.get(ri, ""))
        reject_arr = self._zg["reject_reason"]
        reasons_array = np.array(reject_arr[:], dtype=object)
        reasons_array[ri] = reason
        # Write back the entire array
        self._zg["reject_reason"][:] = reasons_array

    # ------------------------------------------------------------------
    # Public navigation (also called by key/button handlers)
    # ------------------------------------------------------------------

    def _go_next(self):
        """
        Save the current resonator and advance to the next one.

        On the last resonator, show a completion message and disable the
        Next button instead.
        """
        self._save_current()
        if self._cursor < len(self._interactive_indices) - 1:
            self._cursor += 1
            self._load_resonator()
        else:
            self._status_label.setText("All resonators complete — close to exit.")
            self._next_btn.setEnabled(False)

    def _go_back(self):
        """
        Save the current resonator and go back one, unless at the first.
        """
        if self._cursor > 0:
            self._save_current()
            self._cursor -= 1
            self._load_resonator()

    # ------------------------------------------------------------------
    # fres / qres change helpers
    # ------------------------------------------------------------------

    def _unreject_for_edit(self):
        """
        Clear the rejection of the current resonator before an fres/qres edit.

        Editing fres or qres means the resonator is kept, so a rejected
        resonator is marked not rejected and whichever of fres and qres is
        NaN is restored from its pre-rejection value (or its initial value if
        there is none). Both spinboxes are updated. Does nothing if the
        resonator is not rejected.
        """
        ri = self._ri
        if ri not in self._reject_reasons:
            return
        self._reject_reasons.pop(ri)
        pre = self._pre_reject.pop(ri, None)
        fres0, qres0 = pre if pre is not None else (
            self._fres_init[ri], self._qres_init[ri])
        if np.isnan(self._fres_work[ri]):
            self._fres_work[ri] = fres0
        if np.isnan(self._qres_work[ri]):
            self._qres_work[ri] = qres0
        self._reason_combo.blockSignals(True)
        self._reason_combo.setCurrentText("not rejected")
        self._reason_combo.blockSignals(False)
        self._reason_edit.setEnabled(False)
        self._reason_edit.clear()
        self._fres_spin.blockSignals(True)
        self._qres_spin.blockSignals(True)
        self._fres_spin.setValue(self._fres_work[ri] * 1e-6)
        self._qres_spin.setValue(round(self._qres_work[ri]))
        self._fres_spin.blockSignals(False)
        self._qres_spin.blockSignals(False)
        self._update_reject_label()

    def _set_fres(self, fres: float):
        """
        Set fres for the current resonator and update the spinbox and overlay.

        A rejected resonator is un-rejected first (see
        ``_unreject_for_edit``).

        Parameters:
        fres (float): new resonant frequency in Hz.
        """
        self._unreject_for_edit()
        ri = self._ri
        self._fres_work[ri] = fres
        self._fres_spin.blockSignals(True)
        self._fres_spin.setValue(fres * 1e-6)
        self._fres_spin.blockSignals(False)
        self._update_overlay(self._f[ri], self._z[ri], fres, self._qres_work[ri])

    def _set_qres(self, qres: float):
        """
        Set qres for the current resonator and update the spinbox and overlay.

        A rejected resonator is un-rejected first (see
        ``_unreject_for_edit``).

        Parameters:
        qres (float): new Q-factor. Values below 1 are clipped to 1.
        """
        self._unreject_for_edit()
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
        """
        Update fres after the user drags the fres vline.
        """
        if self._fres_updating:
            return
        fres_hz = self._fres_vline.value() * 1e3 + self._f0_hz
        self._set_fres(fres_hz)

    def _on_span_region_changed(self):
        """
        Adjust qres symmetrically about fres after a span edge is dragged.
        """
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
        """
        Update fres from the fres spinbox.

        A rejected resonator is un-rejected first (see
        ``_unreject_for_edit``).

        Parameters:
        value (float): spinbox value in MHz.
        """
        self._set_fres(value * 1e6)

    def _on_spinbox_qres(self, value: float):
        """
        Update qres from the qres spinbox.

        A rejected resonator is un-rejected first (see
        ``_unreject_for_edit``).

        Parameters:
        value (float): spinbox value. Values below 1 are clipped to 1.
        """
        self._set_qres(value)

    # ------------------------------------------------------------------
    # Interactive-ViewBox callbacks
    # ------------------------------------------------------------------

    def _on_shift_click_amp(self, x_khz: float, _y: float):
        """
        Move fres to a Shift+clicked position on the amplitude plot.

        A rejected resonator is un-rejected first (see
        ``_unreject_for_edit``).

        Parameters:
        x_khz (float): clicked x position in kHz, offset from the initial
            fres.
        _y (float): clicked y position. Not used.
        """
        self._set_fres(x_khz * 1e3 + self._f0_hz)

    def _on_shift_click_iq(self, x_re: float, y_im: float):
        """
        Snap fres to the sample nearest a Shift+click on the IQ plot.

        A rejected resonator is un-rejected first (see
        ``_unreject_for_edit``).

        Parameters:
        x_re (float): clicked I (real) position.
        y_im (float): clicked Q (imaginary) position.
        """
        ri = self._ri
        z = self._z[ri]
        f = self._f[ri]
        dist = (z.real - x_re) ** 2 + (z.imag - y_im) ** 2
        ix = int(np.argmin(dist))
        self._set_fres(float(f[ix]))

    def _on_scroll_qres(self, steps: int):
        """
        Adjust qres multiplicatively on Shift/Ctrl + scroll.

        Parameters:
        steps (int): signed step count; qres is multiplied by
            ``1 + steps * _QRES_FRAC``. Nothing happens if qres is NaN.
        """
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
        """
        Handle navigation and rescale key presses.

        Right/D goes to the next resonator, Left/A goes back, and R
        auto-scales the plots. Key presses with Shift or Ctrl held are not
        handled.

        Parameters:
        obj (QtCore.QObject): object that received the event.
        event (QtCore.QEvent): event to filter.

        Returns:
        handled (bool): True if the key press was handled, False if Shift or
            Ctrl was held, else the base-class result.
        """
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
        """
        Save the current resonator and accept the close event.

        Nothing is saved if there are no interactive resonators.

        Parameters:
        event (QtGui.QCloseEvent): close event.
        """
        if self._interactive_indices:
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
    title = "FQ Finder",
    ui_scale = 1.0,
    fres_update_method = "none",
    rmv_gain_simple = False,
):
    """
    Launch the interactive resonance frequency and Q-factor finder.

    Calibration tones (``res_idxs[i] < 0``) are saved immediately with
    their input values and skipped in the interactive loop.

    If *zarr_group* already contains ``fres_opt`` and ``qres_opt``, a
    dialog asks whether to load the saved data, overwrite it (after a
    second confirmation), or cancel. When loading, the saved fres, qres,
    and rejection reasons are used for resonators that were saved, and a
    second dialog asks whether to resume at the saved data index or start
    from data index 0. The data index of the current resonator is saved
    whenever data is saved.

    Parameters:
    f (array-like): frequency data in Hz, shape (M, N).
    z (array-like): complex IQ data, shape (M, N).
    fres (array-like): initial resonant frequencies in Hz, length M.
    qres (array-like): initial Q-factors, length M.
    res_idxs (array-like): resonator indices, length M. Values < 0 mark
        calibration tones.
    zarr_group (zarr.Group): output group. Three arrays of length M are
        written here: ``fres_opt`` (float64), ``qres_opt`` (float64), and
        ``reject_reason`` (str, empty string for non-rejected resonators).
        ``fq_finder_data_idx`` (int64, length 1) holds the data index of
        the resonator shown at the last save (-1 if never saved).
    title (str): window title. Default 'FQ Finder'.
    ui_scale (float or None): font size multiplier. Qt already scales fonts
        for the display (high-DPI scaling is enabled when the application is
        created), so the default 1.0 gives normal-size text; use e.g. 1.2
        for larger text. None is treated as 1.0. The window opens at 80% of
        the screen regardless.
    fres_update_method (str): algorithm used to automatically update
        ``fres`` before the interactive session begins, passed to
        ``citkid.multitone.fres.update_fres``. Can be 'none' (default,
        uses the input ``fres`` as the starting point), 'mins21' (minimum
        of |S21| after subtracting a linear baseline), 'spacing' (point of
        maximum adjacent IQ spacing), or 'distance' (point furthest from the
        off-resonance IQ value). Saved values loaded from ``zarr_group`` are
        not updated.
    rmv_gain_simple (bool): If True, apply a simple gain correction to
        ``z`` before displaying: divide each row by the median
        off-resonance amplitude, then rotate so the off-resonance mean lies
        on the positive real axis. Default False.

    Raises:
    RuntimeError: if the user cancels the startup dialog.
    """
    app = get_qapp(title)

    load_saved = False
    start_idx = 0
    if "fres_opt" in zarr_group and "qres_opt" in zarr_group:
        choice = _show_startup_dialog()
        if choice == 'cancel':
            raise RuntimeError("User cancelled operation")
        if choice == 'overwrite':
            _clear_zarr_outputs(zarr_group)
        else:
            load_saved = True
            res_idxs_arr = np.asarray(res_idxs)
            saved_idx = _read_saved_data_idx(zarr_group, res_idxs_arr)
            if saved_idx is not None:
                start_idx = _ask_start_idx(saved_idx, res_idxs_arr[saved_idx])

    win = FqFinderWindow(
        f=f,
        z=z,
        fres=fres,
        qres=qres,
        res_idxs=res_idxs,
        zarr_group=zarr_group,
        title=title,
        ui_scale=ui_scale,
        fres_update_method=fres_update_method,
        start_idx=start_idx,
        rmv_gain_simple=rmv_gain_simple,
        load_saved=load_saved,
    )
    if not win._interactive_indices:
        print("All resonators are calibration tones; their input fres and "
              "qres were saved. Nothing to edit.")
        return
    win.show()
    # exec() is the modern name (PyQt6+); exec_() is kept for PyQt5/PySide2.
    (getattr(app, 'exec', None) or app.exec_)()
