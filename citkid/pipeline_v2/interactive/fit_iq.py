"""
Interactive panel for the IQ resonance fitting step (fit_iq).

Plots
-----
Left : ``20*log10(|zf_rmv|)`` vs ``ff`` with a
       :class:`~pyqtgraph.LinearRegionItem` that defines the fit mask (only
       data inside the region is fitted).
Right: IQ loop — ``zf_rmv.imag`` vs ``zf_rmv.real`` — with the model
       curve from :func:`citkid.res.funcs.nonlinear_iq` overlaid in red
       after a successful fit.

Mask interaction
----------------
Hold **Shift** and left-click-drag on the amplitude plot to draw the mask
window.  The shaded region updates in real time as you drag.  Click
**Run** to apply the new mask and refit.

Press **Reset Mask** to restore the full frequency range (all samples
included).

Cascade behaviour
-----------------
After a successful run the panel emits ``downstream_rerun`` so that
subsequent panels are automatically updated.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .core import register_panel, StepPanel
from ...qt_compat import Qt as _Qt
from ...res.funcs import nonlinear_iq


class _MaskViewBox(pg.ViewBox):
    """ViewBox that emits ``sig_range_selected(lo, hi)`` on Shift + left-drag."""

    sig_range_selected = QtCore.pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_origin_x = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() & _Qt.ShiftModifier:
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


@register_panel('fit_iq')
class FitIQPanel(StepPanel):
    """
    Panel for the fit_iq step.

    The IQ mask is controlled interactively via a LinearRegionItem on the
    amplitude vs frequency plot.

    Parameters:
    AR (AnalysisRunner): The analysis runner.
    step_names (tuple of str): Step names handled by this panel.
    data_idx (int or None): Initial data index.
    parent (QWidget or None): Parent widget.
    """

    def setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        # ---- control row ----
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)

        self._reset_btn = QtWidgets.QPushButton("Reset Mask")
        self._reset_btn.setFixedWidth(90)
        self._reset_btn.clicked.connect(self._reset_mask)
        ctrl.addWidget(self._reset_btn)

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

        self._status_label = QtWidgets.QLabel("—")
        self._status_label.setMinimumWidth(180)
        ctrl.addWidget(self._status_label)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(round(340 * self.plot_scale))
        root.addWidget(self._gw)

        self._plot_amp = self._gw.addPlot(row=0, col=0, title="IQ Amplitude",
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

        # Data scatter plots — included (blue) and excluded (orange) for contrast
        _inc_brush = pg.mkBrush(100, 180, 255, 210)
        _exc_brush = pg.mkBrush(255, 140, 50, 180)
        self._amp_data     = self._plot_amp.plot(pen=None, symbolBrush=_inc_brush,
                                                  symbolPen=None, symbolSize=4, name='data')
        self._amp_excl     = self._plot_amp.plot(pen=None, symbolBrush=_exc_brush,
                                                  symbolPen=None, symbolSize=4, name='excluded')
        self._iq_data      = self._plot_iq.plot(pen=None, symbolBrush=_inc_brush,
                                                 symbolPen=None, symbolSize=4, name='data')
        self._iq_data_excl = self._plot_iq.plot(pen=None, symbolBrush=_exc_brush,
                                                  symbolPen=None, symbolSize=4, name='excluded')

        # Fit overlay curve on IQ loop (red line)
        _fit_pen = pg.mkPen(color=(255, 80, 80), width=2)
        self._iq_fit  = self._plot_iq.plot(pen=_fit_pen, name='fit')
        self._amp_fit = self._plot_amp.plot(pen=_fit_pen, name='fit')

        # Fit-parameter text overlay (top-right corner, ignored by autoRange)
        self._fit_text = pg.TextItem(
            text='', anchor=(1, 0), color=(220, 220, 220),
            fill=pg.mkBrush(0, 0, 0, 140),
        )
        self._fit_text.setZValue(20)
        self._plot_iq.getViewBox().addItem(self._fit_text, ignoreBounds=True)
        self._plot_iq.getViewBox().sigRangeChanged.connect(
            self._reposition_fit_text
        )

        # Mask region indicator (visual only — updated via Shift+drag)
        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(255, 255, 100, 30),
            pen=pg.mkPen(color=(255, 255, 100), width=1),
            movable=False,
        )
        self._plot_amp.addItem(self._region)

        # Connect Shift+drag signal from the custom ViewBox
        self._plot_amp.getViewBox().sig_range_selected.connect(self._on_range_selected)

        # Current mask state (None = full range, i.e. all True)
        self._mask: np.ndarray | None = None
        # Cache of current ff (Hz) so the mask can be rebuilt from region bounds
        self._ff_cache: np.ndarray | None = None
        # Center frequency (Hz) used for the shifted-kHz x-axis
        self._f_center: float = 0.0
        # Has the region been positioned to the data range yet?
        self._region_initialized: bool = False

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

    def on_data_idx_changing(self):
        """Restore saved mask from DS for the incoming data index, or reset."""
        self._autorange_next = True
        saved = self._get_initial_user_param(
            'fit_iq', 'iq_mask', self.data_idx, fallback=None
        )
        if saved is not None:
            self._mask = np.asarray(saved, dtype=bool)
        else:
            self._mask = None
        self._f_center = 0.0
        self._region_initialized = False

    def prepare_run(self):
        """Rebuild mask from the current region state before running."""
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)

    def get_params_for_step(self, step):
        if step.name == 'fit_iq':
            return {'iq_mask': self._mask}
        return {}

    def update_plots(self):
        di = self.data_idx
        DS = self.AR.DS

        try:
            ff      = np.asarray(DS.ff[di],      dtype=np.float64)
            zf_rmv  = np.asarray(DS.zf_rmv[di],  dtype=np.complex128)
        except Exception as exc:
            self._status_label.setText(f"Data load error: {exc}")
            return

        self._ff_cache = ff

        f_center = float(ff.mean())
        self._f_center = f_center
        ff_plot = (ff - f_center) * 1e-3
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

        # Build mask from current region bounds
        mask = self._build_mask(ff)
        amp_db = 20.0 * np.log10(np.abs(zf_rmv))

        # Amplitude plot — included vs excluded
        self._amp_data.setData(ff_plot[mask],  amp_db[mask])
        self._amp_excl.setData(ff_plot[~mask], amp_db[~mask])

        # IQ loop
        self._iq_data.setData(zf_rmv[mask].real, zf_rmv[mask].imag)
        self._iq_data_excl.setData(zf_rmv[~mask].real, zf_rmv[~mask].imag)

        # Model fit overlay
        try:
            iq_popt = np.asarray(DS.iq_popt[di], dtype=np.float64)
            if not np.all(np.isfinite(iq_popt)):
                raise ValueError("NaN popt")
            f_model = np.linspace(ff[mask].min(), ff[mask].max(), 1000)
            z_model = nonlinear_iq(f_model, *iq_popt, downward=True)
            self._iq_fit.setData(z_model.real, z_model.imag)
            self._amp_fit.setData(
                (f_model - f_center) * 1e-3,
                20.0 * np.log10(np.abs(z_model)),
            )
            Qr  = iq_popt[1]
            amp = iq_popt[2]
            Qc  = Qr / amp
            Qi  = 1.0 / (1.0 / Qr - 1.0 / Qc)
            phi = iq_popt[3]
            a_nl = iq_popt[4]
            self._fit_text.setText(
                f"Qc = {int(round(Qc)):,}\n"
                f"Qi = {int(round(Qi)):,}\n"
                f"\u03d5 = {phi / np.pi:.2f}\u03c0\n"
                f"a\u2099\u2097 = {a_nl:.2f}"
            )
            self._reposition_fit_text()
        except Exception:
            self._iq_fit.setData([], [])
            self._amp_fit.setData([], [])
            # Show NaN legend if iq_popt exists but is all NaN (bad data)
            try:
                iq_popt = np.asarray(DS.iq_popt[di], dtype=np.float64)
                if not np.any(np.isfinite(iq_popt)):
                    self._fit_text.setText(
                        "Qc = NaN\nQi = NaN\n\u03d5 = NaN\na\u2099\u2097 = NaN"
                    )
                    self._reposition_fit_text()
                else:
                    self._fit_text.setText('')
            except Exception:
                self._fit_text.setText('')

        if self._autorange_next:
            self._autorange_next = False
            self._plot_amp.autoRange()
            self._plot_iq.autoRange()

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def autoscale_plots(self):
        self._plot_amp.autoRange()
        self._plot_iq.autoRange()

    def clear_plots(self):
        for curve in (self._amp_data, self._amp_excl, self._amp_fit,
                      self._iq_data, self._iq_data_excl, self._iq_fit):
            curve.setData([], [])
        self._fit_text.setText('')
        self._status_label.setText("—")

    def _reposition_fit_text(self):
        """Keep the fit-parameter label in the top-right corner of the IQ view."""
        vb = self._plot_iq.getViewBox()
        xr, yr = vb.viewRange()
        self._fit_text.setPos(xr[1], yr[1])

    def _build_mask(self, ff: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask for *ff* (Hz) based on the current region bounds.

        The region is stored in shifted-kHz coordinates
        ``(f - f_center) * 1e-3``; convert back to Hz before comparing.
        """
        lo_kHz, hi_kHz = self._region.getRegion()
        lo = lo_kHz * 1e3 + self._f_center
        hi = hi_kHz * 1e3 + self._f_center
        mask = (ff >= lo) & (ff <= hi)
        self._mask = mask
        return mask

    def _reset_mask(self):
        """Reset the region to the full frequency range."""
        if self._ff_cache is not None:
            lo = float((self._ff_cache.min() - self._f_center) * 1e-3)
            hi = float((self._ff_cache.max() - self._f_center) * 1e-3)
            self._region.setRegion((lo, hi))
        self._mask = None
        self._status_label.setText("Mask reset — click Run")

    def _on_range_selected(self, lo: float, hi: float):
        """Handle a Shift+drag mask selection from the ViewBox."""
        self._region.setRegion((lo, hi))
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)
        self._status_label.setText("Mask set — click Run")

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def prepare_run(self):
        """Rebuild the mask from the current region before run_steps is called."""
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)

    def _on_save_clicked(self):
        try:
            self.save_outputs()
            self._status_label.setText("Saved ✓")
        except Exception as exc:
            self._status_label.setText("Save error ✗")
            print(f"Save error: {exc}")

    def _on_step_error(self, step, exc):
        msg = f"'{step.name}' failed: {exc}"
        self._status_label.setText(msg)
        print(msg)

    def _on_bad_data_clicked(self):
        self._status_label.setText("Marking bad…")
        QtWidgets.QApplication.processEvents()
        ok = self._write_nan_outputs()
        if ok:
            self._status_label.setText("Bad data marked ✓")
            self.trigger_downstream()
        else:
            self._status_label.setText("Bad data failed ✗")

    def _nan_outputs(self) -> list:
        """Return the list of output names to delete when marking data as bad.
        
        In pipeline_v2, we delete outputs instead of marking with NaNs.
        """
        return ['iq_p0', 'iq_popt', 'iq_nrmse']
