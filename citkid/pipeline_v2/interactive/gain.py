"""
Interactive panel for the gain fitting steps (make_fr_spans + fit_gain).

Plots
-----
Left : Gain amplitude ``20*log10(|zg|)`` vs ``fg``, with the polynomial
       amplitude fit overlaid in red.
Right: Unwrapped gain phase ``unwrap(angle(zg))`` vs ``fg``, with the
       polynomial phase fit overlaid in red.

Controlled parameter
--------------------
``span_mult`` (float, default 2.0) — multiplier applied to each resonance
    span before masking out resonances from the gain data.  Increasing
    this widens the exclusion zone around each resonance.

Cascade behaviour
-----------------
After a successful run the panel emits ``downstream_rerun`` so that
subsequent panels (e.g. FitIQPanel) are automatically updated.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .core import register_panel, StepPanel


@register_panel('fit_gain')
class GainFitPanel(StepPanel):
    """
    Panel for the fit_gain step.

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

        ctrl.addWidget(QtWidgets.QLabel("span_mult:"))
        self._span_spin = QtWidgets.QDoubleSpinBox()
        self._span_spin.setRange(0.1, 50.0)
        self._span_spin.setSingleStep(0.5)
        self._span_spin.setDecimals(2)
        initial_span = self._get_initial_user_param(
            'fit_gain', 'span_mult', self.data_idx, fallback=2.0
        )
        self._span_spin.setValue(float(initial_span))
        self._span_spin.setFixedWidth(80)
        ctrl.addWidget(self._span_spin)

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
        self._status_label.setMinimumWidth(120)
        ctrl.addWidget(self._status_label)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # Per-index cache of span_mult values set by the user this session.
        # Populated in run_steps(); used by on_data_idx_changing() so that
        # navigating back to a previously-visited index restores the value
        # the user actually used, not the YAML default.
        self._span_mult_cache: dict = {}

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(round(320 * self.plot_scale))
        root.addWidget(self._gw)

        self._plot_amp = self._gw.addPlot(row=0, col=0, title="Gain Amplitude")
        self._plot_amp.setLabel('left', '|S21| (dB)')
        self._plot_amp.setLabel('bottom', 'Frequency (Hz)')
        self._plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self._plot_amp.setDownsampling(auto=True, mode='peak')

        self._plot_phase = self._gw.addPlot(row=0, col=1, title="Gain Phase")
        self._plot_phase.setLabel('left', 'Phase (rad)')
        self._plot_phase.setLabel('bottom', 'Frequency (Hz)')
        self._plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self._plot_phase.setDownsampling(auto=True, mode='peak')

        self._scale_plot_fonts(self._plot_amp, self._plot_phase)

        # Included data points (bright blue scatter)
        _inc_brush = pg.mkBrush(100, 180, 255, 210)
        self._amp_data   = self._plot_amp.plot(pen=None, symbolBrush=_inc_brush,
                                               symbolPen=None, symbolSize=4, name='data')
        self._phase_data = self._plot_phase.plot(pen=None, symbolBrush=_inc_brush,
                                                 symbolPen=None, symbolSize=4, name='data')

        # Fit overlay curves (red lines)
        _fit_pen = pg.mkPen(color=(255, 80, 80), width=2)
        self._amp_fit   = self._plot_amp.plot(pen=_fit_pen, name='fit')
        self._phase_fit = self._plot_phase.plot(pen=_fit_pen, name='fit')

        # Masked-out (excluded) data points — orange for strong contrast
        _exc_brush = pg.mkBrush(255, 140, 50, 180)
        self._amp_masked   = self._plot_amp.plot(pen=None,
                                                  symbolBrush=_exc_brush,
                                                  symbolPen=None,
                                                  symbolSize=4,
                                                  name='masked')
        self._phase_masked = self._plot_phase.plot(pen=None,
                                                    symbolBrush=_exc_brush,
                                                    symbolPen=None,
                                                    symbolSize=4,
                                                    name='masked')

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

    def run_steps(self, save: bool = False) -> bool:
        """Cache the current span_mult for this data_idx before running."""
        self._span_mult_cache[self.data_idx] = self._span_spin.value()
        return super().run_steps(save=save)

    def on_data_idx_changing(self):
        """Restore span_mult for the new index: session cache → DS/zarr → keep current."""
        self._autorange_next = True
        new_di = self.data_idx
        if new_di in self._span_mult_cache:
            val = float(self._span_mult_cache[new_di])
        else:
            val = self._get_initial_user_param(
                'fit_gain', 'span_mult', new_di, fallback=None
            )
        if val is not None:
            self._span_spin.blockSignals(True)
            self._span_spin.setValue(val)
            self._span_spin.blockSignals(False)

    def get_params_for_step(self, step):
        if step.name == 'fit_gain':
            return {'span_mult': self._span_spin.value()}
        return {}

    def update_plots(self):
        di  = self.data_idx
        DS  = self.AR.DS

        try:
            fg = np.asarray(DS.fg[di], dtype=np.float64)
            zg = np.asarray(DS.zg[di], dtype=np.complex128)
        except Exception as exc:
            self._status_label.setText(f"Data load error: {exc}")
            return

        f_center   = float(fg.mean())
        fg_plot    = (fg - f_center) * 1e-3
        center_MHz = round(f_center * 1e-6, 2)
        xlabel     = f"(f \u2212 {center_MHz} MHz) (kHz)"
        self._plot_amp.setLabel('bottom', xlabel)
        self._plot_phase.setLabel('bottom', xlabel)

        amp_db = 20.0 * np.log10(np.abs(zg))
        phase  = np.unwrap(np.angle(zg))

        # Split data into included/excluded once fit results are available
        try:
            p_amp   = np.asarray(DS.p_amp[di],    dtype=np.float64)
            p_phase = np.asarray(DS.p_phase[di],  dtype=np.float64)
            mask    = np.asarray(DS.gain_mask[di], dtype=bool)

            self._amp_data.setData(fg_plot[mask],    amp_db[mask])
            self._phase_data.setData(fg_plot[mask],  phase[mask])
            self._amp_masked.setData(fg_plot[~mask],  amp_db[~mask])
            self._phase_masked.setData(fg_plot[~mask], phase[~mask])
            self._amp_fit.setData(fg_plot, np.polyval(p_amp, fg))
            self._phase_fit.setData(fg_plot, np.polyval(p_phase, fg))
        except Exception:
            # Fit results not yet available — show all data, clear overlays
            self._amp_data.setData(fg_plot, amp_db)
            self._phase_data.setData(fg_plot, phase)
            self._amp_masked.setData([], [])
            self._phase_masked.setData([], [])
            self._amp_fit.setData([], [])
            self._phase_fit.setData([], [])

        if self._autorange_next:
            self._autorange_next = False
            self._plot_amp.autoRange()
            self._plot_phase.autoRange()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def autoscale_plots(self):
        self._plot_amp.autoRange()
        self._plot_phase.autoRange()

    def clear_plots(self):
        for curve in (self._amp_data, self._phase_data,
                      self._amp_masked, self._phase_masked,
                      self._amp_fit, self._phase_fit):
            curve.setData([], [])
        self._status_label.setText("—")

    def _on_save_clicked(self):
        try:
            self.save_outputs()
            self._status_label.setText("Saved ✓")
        except Exception as exc:
            self._status_label.setText("Save error ✗")
            print(f"Save error: {exc}")

    def save_outputs(self):
        """Save step outputs AND the span_mult user param to zarr."""
        fit_gain_step = next(s for s in self.steps if s.name == 'fit_gain')
        user_params = self.get_params_for_step(fit_gain_step)
        if user_params:
            pipeline_scope, step_index = self.AR._resolve_step_scope(fit_gain_step)
            self.AR._add_user_params(
                fit_gain_step, user_params,
                self.data_idx, save=True,
                pipeline_scope=pipeline_scope, step_index=step_index
            )
        super().save_outputs()

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
        return ['p_amp', 'p_phase', 'gain_mask']
