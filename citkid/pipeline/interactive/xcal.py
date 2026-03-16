"""
Interactive panel for the x-calibration fitting steps.

Covers two linked analysis steps:

    6. get_xcal_mask — computes the calibration mask from theta_f / theta_t
    7. fit_x_theta   — fits a polynomial x vs theta calibration curve

Plots
-----
Left   : ``20*log10(|zf_rmv|)`` vs ``ff``, with ``xcal_mask`` coloring
         (blue = included, orange = excluded) after running.
Middle : IQ loop — ``zf_rmv.imag`` vs ``zf_rmv.real`` — with ``xcal_mask``
         coloring.
Right  : x vs theta calibration — ``xf`` vs ``thetaf`` scatter (with
         ``xcal_mask`` coloring), ``xt`` vs ``thetat`` scatter (grey), and
         the polynomial fit curve (red) after a successful run.

User parameters
---------------
``xcal_idx0_offset``  int   — extra indices to extend the mask below (default 3)
``xcal_idx1_offset``  int   — extra indices to extend the mask above (default 9)
``xcal_std_cutoff``   float — sigma cutoff on theta_t before range detection (default 12.0)
``poly_x_deg``        int   — polynomial degree for the x vs theta fit (default 3)
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .core import register_panel, StepPanel


@register_panel('get_xcal_mask', 'fit_x_theta')
class XCalPanel(StepPanel):
    """
    Panel for the *get_xcal_mask* + *fit_x_theta* steps.

    The xcal mask is computed automatically from the user-adjustable offset and
    cutoff parameters.  No interactive mask drawing is needed.

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
        ctrl.setSpacing(6)

        # xcal_idx0_offset (int)
        ctrl.addWidget(QtWidgets.QLabel("idx0_off:"))
        self._idx0_spin = QtWidgets.QSpinBox()
        self._idx0_spin.setRange(0, 500)
        self._idx0_spin.setFixedWidth(60)
        self._idx0_spin.setValue(
            int(self._get_initial_user_param(
                'get_xcal_mask', 'xcal_idx0_offset', self.data_idx, fallback=3
            ))
        )
        ctrl.addWidget(self._idx0_spin)

        # xcal_idx1_offset (int)
        ctrl.addWidget(QtWidgets.QLabel("idx1_off:"))
        self._idx1_spin = QtWidgets.QSpinBox()
        self._idx1_spin.setRange(0, 500)
        self._idx1_spin.setFixedWidth(60)
        self._idx1_spin.setValue(
            int(self._get_initial_user_param(
                'get_xcal_mask', 'xcal_idx1_offset', self.data_idx, fallback=9
            ))
        )
        ctrl.addWidget(self._idx1_spin)

        # xcal_std_cutoff (float)
        ctrl.addWidget(QtWidgets.QLabel("std_cut:"))
        self._std_spin = QtWidgets.QDoubleSpinBox()
        self._std_spin.setRange(0.1, 1000.0)
        self._std_spin.setDecimals(1)
        self._std_spin.setSingleStep(1.0)
        self._std_spin.setFixedWidth(70)
        self._std_spin.setValue(
            float(self._get_initial_user_param(
                'get_xcal_mask', 'xcal_std_cutoff', self.data_idx, fallback=12.0
            ))
        )
        ctrl.addWidget(self._std_spin)

        # poly_x_deg (int)
        ctrl.addWidget(QtWidgets.QLabel("poly_deg:"))
        self._poly_spin = QtWidgets.QSpinBox()
        self._poly_spin.setRange(1, 20)
        self._poly_spin.setFixedWidth(55)
        self._poly_spin.setValue(
            int(self._get_initial_user_param(
                'fit_x_theta', 'poly_x_deg', self.data_idx, fallback=3
            ))
        )
        ctrl.addWidget(self._poly_spin)

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
        self._status_label.setMinimumWidth(160)
        ctrl.addWidget(self._status_label)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # Per-index cache of user-param values set this session.
        self._param_cache: dict = {}  # {data_idx: (idx0, idx1, std, poly_deg)}

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(round(340 * self.plot_scale))
        root.addWidget(self._gw)

        self._plot_amp = self._gw.addPlot(row=0, col=0, title="XCal Amplitude")
        self._plot_amp.setLabel('left', '|S21| (dB)')
        self._plot_amp.setLabel('bottom', 'Frequency (Hz)')
        self._plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self._plot_amp.setDownsampling(auto=True, mode='peak')

        self._plot_iq = self._gw.addPlot(row=0, col=1, title="IQ Loop")
        self._plot_iq.setLabel('left', 'Q (Im)')
        self._plot_iq.setLabel('bottom', 'I (Re)')
        self._plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self._plot_iq.setAspectLocked(True)

        self._plot_xcal = self._gw.addPlot(row=0, col=2, title="x vs \u03b8 Calibration")
        self._plot_xcal.setLabel('left', 'x (frac. freq. det.)')
        self._plot_xcal.setLabel('bottom', '\u03b8 (rad)')
        self._plot_xcal.showGrid(x=True, y=True, alpha=0.3)

        self._scale_plot_fonts(self._plot_amp, self._plot_iq, self._plot_xcal)

        # Colour brushes
        _inc_brush  = pg.mkBrush(100, 180, 255, 210)   # included (blue)
        _exc_brush  = pg.mkBrush(255, 140,  50, 180)   # excluded (orange)
        _ts_brush   = pg.mkBrush(160, 220, 160, 120)   # timestream (light green)

        # Amplitude plot scatter items
        self._amp_data = self._plot_amp.plot(pen=None, symbolBrush=_inc_brush,
                                              symbolPen=None, symbolSize=4, name='data')
        self._amp_excl = self._plot_amp.plot(pen=None, symbolBrush=_exc_brush,
                                              symbolPen=None, symbolSize=4, name='excluded')

        # IQ loop scatter items
        self._iq_data = self._plot_iq.plot(pen=None, symbolBrush=_inc_brush,
                                            symbolPen=None, symbolSize=4, name='data')
        self._iq_excl = self._plot_iq.plot(pen=None, symbolBrush=_exc_brush,
                                            symbolPen=None, symbolSize=4, name='excluded')
        self._zt_iq   = self._plot_iq.plot(pen=None, symbolBrush=_ts_brush,
                                            symbolPen=None, symbolSize=2, name='zt_rmv')

        # x vs theta scatter + fit items
        self._xcal_fit  = self._plot_xcal.plot(
            pen=pg.mkPen(color=(255, 80, 80), width=2), name='poly fit'
        )
        self._xcal_fit.setZValue(-10)
        self._xcal_inc  = self._plot_xcal.plot(pen=None, symbolBrush=_inc_brush,
                                                symbolPen=None, symbolSize=4, name='xf incl.')
        self._xcal_exc  = self._plot_xcal.plot(pen=None, symbolBrush=_exc_brush,
                                                symbolPen=None, symbolSize=4, name='xf excl.')
        self._xcal_ts   = self._plot_xcal.plot(pen=None, symbolBrush=_ts_brush,
                                                symbolPen=None, symbolSize=2, name='xt')

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

    def on_data_idx_changing(self):
        """Restore user params for the new index: session cache → DS/YAML → keep current."""
        self._autorange_next = True
        new_di = self.data_idx
        if new_di in self._param_cache:
            idx0, idx1, std, poly = self._param_cache[new_di]
        else:
            idx0 = self._get_initial_user_param(
                'get_xcal_mask', 'xcal_idx0_offset', new_di, fallback=None)
            idx1 = self._get_initial_user_param(
                'get_xcal_mask', 'xcal_idx1_offset', new_di, fallback=None)
            std  = self._get_initial_user_param(
                'get_xcal_mask', 'xcal_std_cutoff',  new_di, fallback=None)
            poly = self._get_initial_user_param(
                'fit_x_theta', 'poly_x_deg', new_di, fallback=None)

        def _set(spin, val):
            if val is not None:
                spin.blockSignals(True)
                spin.setValue(type(spin.value())(val))
                spin.blockSignals(False)

        _set(self._idx0_spin, idx0)
        _set(self._idx1_spin, idx1)
        _set(self._std_spin,  std)
        _set(self._poly_spin, poly)

    def run_steps(self, save: bool = False) -> bool:
        """Cache the current param values for this data_idx before running."""
        self._param_cache[self.data_idx] = (
            self._idx0_spin.value(),
            self._idx1_spin.value(),
            self._std_spin.value(),
            self._poly_spin.value(),
        )
        return super().run_steps(save=save)

    def get_params_for_step(self, step) -> dict:
        if step.name == 'get_xcal_mask':
            return {
                'xcal_idx0_offset': int(self._idx0_spin.value()),
                'xcal_idx1_offset': int(self._idx1_spin.value()),
                'xcal_std_cutoff':  float(self._std_spin.value()),
            }
        if step.name == 'fit_x_theta':
            return {'poly_x_deg': int(self._poly_spin.value())}
        return {}

    def update_plots(self):
        di = self.data_idx
        DS = self.AR.DS

        # Raw sweep data (always available if cal steps have run)
        try:
            ff     = np.asarray(DS.ff[di],     dtype=np.float64)
            zf_rmv = np.asarray(DS.zf_rmv[di], dtype=np.complex128)
        except Exception as exc:
            self._status_label.setText(f"Data load error: {exc}")
            return

        f_center   = float(ff.mean())
        ff_plot    = (ff - f_center) * 1e-3
        center_MHz = round(f_center * 1e-6, 2)
        xlabel     = f"(f \u2212 {center_MHz} MHz) (kHz)"
        self._plot_amp.setLabel('bottom', xlabel)

        amp_db = 20.0 * np.log10(np.abs(zf_rmv))

        # Load xcal_mask if available; fall back to all-True
        try:
            xcal_mask = np.asarray(DS.xcal_mask[di], dtype=bool)
        except Exception:
            xcal_mask = np.ones(len(ff), dtype=bool)
            # All-included before step has run — render without mask coloring
            self._amp_data.setData(ff_plot, amp_db)
            self._amp_excl.setData([], [])
            self._iq_data.setData(zf_rmv.real, zf_rmv.imag)
            self._iq_excl.setData([], [])
        else:
            self._amp_data.setData(ff_plot[xcal_mask],  amp_db[xcal_mask])
            self._amp_excl.setData(ff_plot[~xcal_mask], amp_db[~xcal_mask])
            self._iq_data.setData(zf_rmv[xcal_mask].real,  zf_rmv[xcal_mask].imag)
            self._iq_excl.setData(zf_rmv[~xcal_mask].real, zf_rmv[~xcal_mask].imag)

        # zt_rmv on IQ
        try:
            zt_rmv_ts = np.asarray(DS.zt_rmv[di], dtype=np.complex128)
            self._zt_iq.setData(zt_rmv_ts.real, zt_rmv_ts.imag)
        except Exception:
            self._zt_iq.setData([], [])

        # x vs theta calibration plot (needs cal-step outputs thetaf, xf)
        try:
            thetaf = np.asarray(DS.thetaf[di], dtype=np.float64)
            xf     = np.asarray(DS.xf[di],     dtype=np.float64)
        except Exception:
            # Cal outputs not yet available — leave cal plot empty
            self._xcal_inc.setData([], [])
            self._xcal_exc.setData([], [])
            self._xcal_ts.setData([], [])
            self._xcal_fit.setData([], [])
        else:
            self._xcal_inc.setData(thetaf[xcal_mask],  xf[xcal_mask])
            self._xcal_exc.setData(thetaf[~xcal_mask], xf[~xcal_mask])

            # Polynomial fit line + timestream overlay (needs poly_x from step 7)
            try:
                poly_x = np.asarray(DS.poly_x[di], dtype=np.float64)
                theta_grid = np.linspace(thetaf.min(), thetaf.max(), 500)
                self._xcal_fit.setData(theta_grid, np.polyval(poly_x, theta_grid))

                # Timestream xt = polyval(poly_x, thetat)
                try:
                    thetat = np.asarray(DS.thetat[di], dtype=np.float64)
                    xt     = np.polyval(poly_x, thetat)
                    self._xcal_ts.setData(thetat, xt)
                    self._plot_xcal.setDownsampling(auto=True, mode='subsample')
                except Exception:
                    self._xcal_ts.setData([], [])
            except Exception:
                self._xcal_fit.setData([], [])
                self._xcal_ts.setData([], [])

        if self._autorange_next:
            self._autorange_next = False
            self._plot_amp.autoRange()
            self._plot_iq.autoRange()
            self._range_xcal()

    def _range_xcal(self):
        """Set x vs theta plot range to the masked theta/x extent plus 5% padding."""
        di = self.data_idx
        DS = self.AR.DS
        try:
            thetaf    = np.asarray(DS.thetaf[di], dtype=np.float64)
            xf        = np.asarray(DS.xf[di],     dtype=np.float64)
            xcal_mask = np.asarray(DS.xcal_mask[di], dtype=bool)
        except Exception:
            self._plot_xcal.autoRange()
            return
        if not xcal_mask.any():
            self._plot_xcal.autoRange()
            return
        theta_m = thetaf[xcal_mask]
        x_m     = xf[xcal_mask]
        t_lo, t_hi = float(theta_m.min()), float(theta_m.max())
        x_lo, x_hi = float(x_m.min()),     float(x_m.max())
        t_pad = max((t_hi - t_lo) * 0.05, 1e-6)
        x_pad = max((x_hi - x_lo) * 0.05, 1e-9)
        self._plot_xcal.setXRange(t_lo - t_pad, t_hi + t_pad, padding=0)
        self._plot_xcal.setYRange(x_lo - x_pad, x_hi + x_pad, padding=0)

    def autoscale_plots(self):
        self._plot_amp.autoRange()
        self._plot_iq.autoRange()
        self._range_xcal()

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
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
        """Save step outputs AND user params (offsets, cutoff, poly_deg) to zarr."""
        for step in self.steps:
            user_params = self.get_params_for_step(step)
            if user_params:
                self.AR._add_user_params(
                    user_params, step.func_type,
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

    def _nan_outputs(self) -> dict:
        DS = self.AR.DS
        di = self.data_idx
        try:
            ff_len = len(np.asarray(DS.ff[di]))
        except Exception:
            ff_len = 1
        poly_deg = int(self._poly_spin.value())
        return {
            'xcal_mask': np.zeros(ff_len, dtype=bool),
            'poly_x':    np.full(poly_deg + 1, np.nan),
        }
