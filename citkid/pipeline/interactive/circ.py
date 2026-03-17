"""
Interactive panel for the circle fitting steps.

Covers three linked analysis steps:

    3. fit_iq_circle   — fits an IQ circle to ``zf_rmv[circ_mask]``
    4. get_idx_t       — finds the index in ``ff`` closest to ``ft``
    5. get_theta_phase_offset — computes the phase offset from ``zt_rmv``

Because steps 4 and 5 depend on the result of step 3, the three steps are
always executed together as a single logical unit.

Plots
-----
Left : ``20*log10(|zf_rmv|)`` vs ``ff``, with a
       :class:`~pyqtgraph.LinearRegionItem` for the circle-fit mask.
       A green dot marks the ``idx_t`` frequency after a successful run.
Right: IQ loop — ``zf_rmv.imag`` vs ``zf_rmv.real`` — with the fitted
       circle overlaid in red and the ``idx_t`` point marked in green.

Mask interaction
----------------
Hold **Shift** and left-click-drag on the amplitude plot to draw the mask
window.  Press **Reset Mask** to restore the full frequency range.

Step cascade
------------
After a successful run the panel emits ``downstream_rerun`` so that
subsequent panels are automatically updated.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .core import register_panel, StepPanel, _density_subsample


class _MaskViewBox(pg.ViewBox):
    """ViewBox that emits ``sig_range_selected(lo, hi)`` on Shift + left-drag."""

    sig_range_selected = QtCore.pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_origin_x = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.ShiftModifier:
            ev.accept()
            if ev.isStart():
                self._drag_origin_x = self.mapToView(ev.buttonDownPos()).x()
            cur_x = self.mapToView(ev.pos()).x()
            if self._drag_origin_x is not None:
                lo = min(self._drag_origin_x, cur_x)
                hi = max(self._drag_origin_x, cur_x)
                self.sig_range_selected.emit(lo, hi)
            if ev.isFinish():
                self._drag_origin_x = None
        else:
            super().mouseDragEvent(ev, axis=axis)


@register_panel('fit_iq_circle', 'get_idx_t', 'get_theta_phase_offset')
class CircleFitPanel(StepPanel):
    """
    Panel for the *fit_iq_circle* + *get_idx_t* + *get_theta_phase_offset* steps.

    The circle-fit mask is controlled interactively via a
    :class:`~pyqtgraph.LinearRegionItem` on the amplitude vs frequency plot.
    Steps 4 and 5 do not accept user parameters; they run automatically after
    step 3.

    Parameters
    ----------
    AR : AnalysisRunner
    step_names : tuple of str
    data_idx : int or None
    parent : QWidget, optional
    """

    def setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        self._plot_cache: dict = {}  # populated by prefetch_plot_data
        # ---- control row ----
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)

        self._reset_btn = QtWidgets.QPushButton("Reset Mask")
        self._reset_btn.setFixedWidth(90)
        self._reset_btn.clicked.connect(self._reset_mask)
        ctrl.addWidget(self._reset_btn)

        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setFixedWidth(60)
        self._run_btn.clicked.connect(self._on_run_clicked)
        ctrl.addWidget(self._run_btn)

        self._run_through_btn = QtWidgets.QPushButton("Run+")
        self._run_through_btn.setFixedWidth(55)
        self._run_through_btn.setToolTip("Run this panel and all following panels")
        self._run_through_btn.clicked.connect(lambda: self.run_from_here.emit())
        ctrl.addWidget(self._run_through_btn)

        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setFixedWidth(60)
        self._save_btn.clicked.connect(self._on_save_clicked)
        ctrl.addWidget(self._save_btn)

        self._bad_btn = QtWidgets.QPushButton("Bad Data")
        self._bad_btn.setFixedWidth(75)
        self._bad_btn.setToolTip("Mark this index as bad: fill all outputs with NaN")
        self._bad_btn.clicked.connect(self._on_bad_data_clicked)
        ctrl.addWidget(self._bad_btn)

        self._status_label = QtWidgets.QLabel("\u2014")
        self._status_label.setMinimumWidth(180)
        ctrl.addWidget(self._status_label)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(round(340 * self.plot_scale))
        root.addWidget(self._gw)

        self._plot_amp = self._gw.addPlot(row=0, col=0, title="Circle Fit Amplitude",
                                           viewBox=_MaskViewBox())
        self._plot_amp.setLabel('left', '|S21| (dB)')
        self._plot_amp.setLabel('bottom', 'Frequency (Hz)')
        self._plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self._plot_amp.setDownsampling(auto=True, mode='peak')

        self._plot_iq = self._gw.addPlot(row=0, col=1, title="IQ Loop")
        self._plot_iq.setLabel('left', 'Q (Im)')
        self._plot_iq.setLabel('bottom', 'I (Re)')
        self._plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self._plot_iq.setAspectLocked(True)

        self._scale_plot_fonts(self._plot_amp, self._plot_iq)

        # Data scatter — included (blue) and excluded (orange)
        _inc_brush = pg.mkBrush(100, 180, 255, 210)
        _exc_brush = pg.mkBrush(255, 140, 50, 180)
        self._amp_data = self._plot_amp.plot(pen=None, symbolBrush=_inc_brush,
                                              symbolPen=None, symbolSize=4, name='data')
        self._amp_excl = self._plot_amp.plot(pen=None, symbolBrush=_exc_brush,
                                              symbolPen=None, symbolSize=4, name='excluded')
        self._iq_data  = self._plot_iq.plot(pen=None, symbolBrush=_inc_brush,
                                             symbolPen=None, symbolSize=4, name='data')
        self._iq_excl  = self._plot_iq.plot(pen=None, symbolBrush=_exc_brush,
                                             symbolPen=None, symbolSize=4, name='excluded')

        # Circle fit overlay (red line)
        self._circle_fit = self._plot_iq.plot(
            pen=pg.mkPen(color=(255, 80, 80), width=2), name='circle fit'
        )

        # idx_t markers (green dot on amplitude plot and IQ plot)
        _idx_brush = pg.mkBrush(80, 255, 80, 220)
        self._idx_t_amp = self._plot_amp.plot(
            pen=None, symbolBrush=_idx_brush, symbolPen=None, symbolSize=8, name='ft'
        )
        self._idx_t_iq = self._plot_iq.plot(
            pen=None, symbolBrush=_idx_brush, symbolPen=None, symbolSize=8, name='ft'
        )

        # zt_rmv timestream on IQ (light green, small dots)
        _ts_brush = pg.mkBrush(160, 220, 160, 80)
        self._zt_iq = self._plot_iq.plot(
            pen=None, symbolBrush=_ts_brush, symbolPen=None, symbolSize=2, name='zt_rmv'
        )

        # Mask region indicator
        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(255, 255, 100, 30),
            pen=pg.mkPen(color=(255, 255, 100), width=1),
            movable=False,
        )
        self._plot_amp.addItem(self._region)

        # Connect Shift+drag signal
        self._plot_amp.getViewBox().sig_range_selected.connect(self._on_range_selected)

        # State
        self._mask: np.ndarray | None = None
        self._ff_cache: np.ndarray | None = None
        self._f_center: float = 0.0
        self._region_initialized: bool = False

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

    def on_data_idx_changing(self):
        """Restore saved circ_mask for the incoming data index, or reset."""
        self._autorange_next = True
        saved = self._get_initial_user_param(
            'fit_iq_circle', 'circ_mask', self.data_idx, fallback=None
        )
        if saved is not None:
            self._mask = np.asarray(saved, dtype=bool)
        else:
            self._mask = None
        self._f_center = 0.0
        self._region_initialized = False

    def prepare_run(self):
        """Rebuild mask from the current region before running."""
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)

    def get_params_for_step(self, step) -> dict:
        """Only ``fit_iq_circle`` takes a user parameter (``circ_mask``)."""
        if step.name == 'fit_iq_circle':
            return {'circ_mask': self._mask}
        return {}

    def update_plots(self):
        di = self.data_idx
        DS = self.AR.DS

        # Use pre-computed numpy arrays from the prefetch cache when available.
        cache = self._plot_cache.pop(di, None)

        try:
            ff     = cache['ff']     if cache else np.asarray(DS.ff[di],     dtype=np.float64)
            zf_rmv = cache['zf_rmv'] if cache else np.asarray(DS.zf_rmv[di], dtype=np.complex128)
        except Exception as exc:
            self._status_label.setText(f"Data load error: {exc}")
            return

        self._ff_cache = ff
        f_center = cache['f_center'] if cache else float(ff.mean())
        self._f_center = f_center
        ff_plot  = cache['ff_plot']  if cache else (ff - f_center) * 1e-3
        amp_db   = cache['amp_db']   if cache else 20.0 * np.log10(np.abs(zf_rmv))
        center_MHz = round(f_center * 1e-6, 2)
        xlabel = f"(f \u2212 {center_MHz} MHz) (kHz)"
        self._plot_amp.setLabel('bottom', xlabel)

        # Initialise region to data range (or saved mask bounds) on first call
        if not self._region_initialized:
            if self._mask is not None and self._mask.any():
                self._region.setRegion((
                    float((ff[self._mask].min() - f_center) * 1e-3),
                    float((ff[self._mask].max() - f_center) * 1e-3),
                ))
            else:
                self._region.setRegion((
                    float((ff.min() - f_center) * 1e-3),
                    float((ff.max() - f_center) * 1e-3),
                ))
            self._region_initialized = True

        mask = self._build_mask(ff)
        amp_db = 20.0 * np.log10(np.abs(zf_rmv))

        # Amplitude plot — included vs excluded
        self._amp_data.setData(ff_plot[mask],  amp_db[mask])
        self._amp_excl.setData(ff_plot[~mask], amp_db[~mask])

        # IQ loop — included vs excluded
        self._iq_data.setData(zf_rmv[mask].real,  zf_rmv[mask].imag)
        self._iq_excl.setData(zf_rmv[~mask].real, zf_rmv[~mask].imag)

        # Circle fit overlay
        try:
            if cache and len(cache['xc']):
                xc, yc = cache['xc'], cache['yc']
            else:
                circ_origin = complex(DS.circ_origin[di])
                circ_radius = float(DS.circ_radius[di])
                if np.isfinite(circ_origin.real) and np.isfinite(circ_radius):
                    theta_c = np.linspace(0, 2 * np.pi, 360)
                    xc = circ_origin.real + circ_radius * np.cos(theta_c)
                    yc = circ_origin.imag + circ_radius * np.sin(theta_c)
                else:
                    xc, yc = [], []
            self._circle_fit.setData(xc, yc)
        except Exception:
            self._circle_fit.setData([], [])

        # idx_t marker
        try:
            idx_t = cache['idx_t'] if cache else int(DS.idx_t[di])
            if idx_t is not None:
                self._idx_t_amp.setData([ff_plot[idx_t]], [amp_db[idx_t]])
                self._idx_t_iq.setData([zf_rmv[idx_t].real], [zf_rmv[idx_t].imag])
            else:
                self._idx_t_amp.setData([], [])
                self._idx_t_iq.setData([], [])
        except Exception:
            self._idx_t_amp.setData([], [])
            self._idx_t_iq.setData([], [])

        # zt_rmv on IQ — density-preserving subsample keeps sparse tail intact
        try:
            if cache and cache['zt'] is not None:
                zt_sub = cache['zt']
            else:
                zt_sub = _density_subsample(
                    np.asarray(DS.zt_rmv[di], dtype=np.complex128)
                )
            self._zt_iq.setData(zt_sub.real, zt_sub.imag)
        except Exception:
            self._zt_iq.setData([], [])

        if self._autorange_next:
            self._autorange_next = False
            self._plot_amp.autoRange()
            self._plot_iq.autoRange()

    def autoscale_plots(self):
        self._plot_amp.autoRange()
        self._plot_iq.autoRange()

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_mask(self, ff: np.ndarray) -> np.ndarray:
        """Build a boolean mask from the current region (shifted-kHz coords)."""
        lo_kHz, hi_kHz = self._region.getRegion()
        lo = lo_kHz * 1e3 + self._f_center
        hi = hi_kHz * 1e3 + self._f_center
        mask = (ff >= lo) & (ff <= hi)
        self._mask = mask
        return mask

    def _reset_mask(self):
        if self._ff_cache is not None:
            lo = float((self._ff_cache.min() - self._f_center) * 1e-3)
            hi = float((self._ff_cache.max() - self._f_center) * 1e-3)
            self._region.setRegion((lo, hi))
        self._mask = None
        self._status_label.setText("Mask reset \u2014 click Run")

    def _on_range_selected(self, lo: float, hi: float):
        self._region.setRegion((lo, hi))
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)
        self._status_label.setText("Mask set \u2014 click Run")

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)
        self._status_label.setText("Running\u2026")
        QtWidgets.QApplication.processEvents()
        ok = self.run_steps()
        if ok:
            self._status_label.setText("Done \u2713")
            self.trigger_downstream()
        else:
            self._status_label.setText("Error \u2717")

    def _on_save_clicked(self):
        try:
            self.save_outputs()
            self._status_label.setText("Saved \u2713")
        except Exception as exc:
            self._status_label.setText("Save error \u2717")
            print(f"Save error: {exc}")

    def save_outputs(self):
        """Save step outputs AND the ``circ_mask`` user param to zarr."""
        circ_step = next(s for s in self.steps if s.name == 'fit_iq_circle')
        user_params = self.get_params_for_step(circ_step)
        if user_params:
            self.AR._add_user_params(
                user_params, circ_step.func_type,
                self.data_idx, save=True
            )
        super().save_outputs()

    def _on_step_error(self, step, exc):
        msg = f"'{step.name}' failed: {exc}"
        self._status_label.setText(msg)
        print(msg)

    def _on_bad_data_clicked(self):
        self._status_label.setText("Marking bad\u2026")
        QtWidgets.QApplication.processEvents()
        ok = self._write_nan_outputs()
        if ok:
            self._status_label.setText("Bad data marked \u2713")
            self.trigger_downstream()
        else:
            self._status_label.setText("Bad data failed \u2717")

    def prefetch_plot_data(self, di: int):
        """Pre-compute numpy arrays for *di* in the background prefetch thread."""
        DS = self.AR.DS
        try:
            ff     = np.asarray(DS.ff[di],     dtype=np.float64)
            zf_rmv = np.asarray(DS.zf_rmv[di], dtype=np.complex128)
        except Exception:
            return

        f_center = float(ff.mean())
        ff_plot  = (ff - f_center) * 1e-3
        amp_db   = 20.0 * np.log10(np.abs(zf_rmv))

        # circle overlay
        try:
            circ_origin = complex(DS.circ_origin[di])
            circ_radius = float(DS.circ_radius[di])
            if np.isfinite(circ_origin.real) and np.isfinite(circ_radius):
                theta_c = np.linspace(0, 2 * np.pi, 360)
                xc = circ_origin.real + circ_radius * np.cos(theta_c)
                yc = circ_origin.imag + circ_radius * np.sin(theta_c)
            else:
                xc, yc = np.array([]), np.array([])
        except Exception:
            xc, yc = np.array([]), np.array([])

        # idx_t
        try:
            idx_t = int(DS.idx_t[di])
        except Exception:
            idx_t = None

        # zt_rmv subsampled
        try:
            zt = _density_subsample(
                np.asarray(DS.zt_rmv[di], dtype=np.complex128)
            )
        except Exception:
            zt = None

        if not hasattr(self, '_plot_cache'):
            self._plot_cache = {}
        self._plot_cache[di] = dict(
            ff=ff, ff_plot=ff_plot, f_center=f_center,
            zf_rmv=zf_rmv, amp_db=amp_db,
            xc=xc, yc=yc, idx_t=idx_t, zt=zt,
        )

    def _nan_outputs(self) -> dict:
        return {
            'circ_origin':        np.complex128(np.nan + 1j * np.nan),
            'circ_radius':        np.nan,
            'idx_t':              np.int64(0),
            'theta_phase_offset': np.nan,
        }
