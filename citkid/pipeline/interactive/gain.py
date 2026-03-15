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


@register_panel('make_fr_spans', 'fit_gain')
class GainFitPanel(StepPanel):
    """
    Panel for the *make_fr_spans* + *fit_gain* steps.

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

        # ---- control row ----
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)

        ctrl.addWidget(QtWidgets.QLabel("span_mult:"))
        self._span_spin = QtWidgets.QDoubleSpinBox()
        self._span_spin.setRange(0.1, 50.0)
        self._span_spin.setSingleStep(0.5)
        self._span_spin.setDecimals(2)
        self._span_spin.setValue(2.0)
        self._span_spin.setFixedWidth(80)
        ctrl.addWidget(self._span_spin)

        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setFixedWidth(60)
        self._run_btn.clicked.connect(self._on_run_clicked)
        ctrl.addWidget(self._run_btn)

        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setFixedWidth(60)
        self._save_btn.clicked.connect(self._on_save_clicked)
        ctrl.addWidget(self._save_btn)

        self._status_label = QtWidgets.QLabel("—")
        self._status_label.setMinimumWidth(120)
        ctrl.addWidget(self._status_label)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(320)
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

        # Included data points (bright blue scatter)
        _inc_brush = pg.mkBrush(100, 180, 255, 200)
        self._amp_data   = self._plot_amp.plot(pen=None, symbolBrush=_inc_brush,
                                               symbolPen=None, symbolSize=4, name='data')
        self._phase_data = self._plot_phase.plot(pen=None, symbolBrush=_inc_brush,
                                                 symbolPen=None, symbolSize=4, name='data')

        # Fit overlay curves (red lines)
        _fit_pen = pg.mkPen(color=(255, 80, 80), width=2)
        self._amp_fit   = self._plot_amp.plot(pen=_fit_pen, name='fit')
        self._phase_fit = self._plot_phase.plot(pen=_fit_pen, name='fit')

        # Masked-out (excluded) data points (dimmed grey scatter)
        self._amp_masked   = self._plot_amp.plot(pen=None,
                                                  symbolBrush=(180, 180, 180, 80),
                                                  symbolPen=None,
                                                  symbolSize=4,
                                                  name='masked')
        self._phase_masked = self._plot_phase.plot(pen=None,
                                                    symbolBrush=(180, 180, 180, 80),
                                                    symbolPen=None,
                                                    symbolSize=4,
                                                    name='masked')

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

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

        amp_db = 20.0 * np.log10(np.abs(zg))
        phase  = np.unwrap(np.angle(zg))

        # Split data into included/excluded once fit results are available
        try:
            p_amp   = np.asarray(DS.p_amp[di],    dtype=np.float64)
            p_phase = np.asarray(DS.p_phase[di],  dtype=np.float64)
            mask    = np.asarray(DS.gain_mask[di], dtype=bool)

            self._amp_data.setData(fg[mask],   amp_db[mask])
            self._phase_data.setData(fg[mask], phase[mask])
            self._amp_masked.setData(fg[~mask],   amp_db[~mask])
            self._phase_masked.setData(fg[~mask], phase[~mask])
            self._amp_fit.setData(fg, np.polyval(p_amp, fg))
            self._phase_fit.setData(fg, np.polyval(p_phase, fg))
        except Exception:
            # Fit results not yet available — show all data, clear overlays
            self._amp_data.setData(fg, amp_db)
            self._phase_data.setData(fg, phase)
            self._amp_masked.setData([], [])
            self._phase_masked.setData([], [])
            self._amp_fit.setData([], [])
            self._phase_fit.setData([], [])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        self._status_label.setText("Running…")
        QtWidgets.QApplication.processEvents()
        ok = self.run_steps()
        if ok:
            self._status_label.setText("Done ✓")
            self.trigger_downstream()
        else:
            self._status_label.setText("Error ✗")

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
