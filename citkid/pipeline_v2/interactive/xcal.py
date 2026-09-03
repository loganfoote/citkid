"""
Interactive panel for the x-calibration fitting steps.

Covers two linked analysis steps:

    6. get_xcal_mask — computes the calibration mask from theta_f / theta_t
    7. fit_x_theta   — fits a polynomial x vs theta calibration curve

Plots
-----
Left   : ``20*log10(|zf_rmv|)`` vs ``ff``, with ``xcal_mask`` coloring
         (blue = included, orange = excluded) after running.
Middle : IQ loop — ``zf_cent.imag`` vs ``zf_cent.real`` — with ``xcal_mask``
         coloring, and ``zt_cent`` overlaid (light green).
Right  : x vs theta calibration — ``xf`` vs ``thetaf`` scatter (with
         ``xcal_mask`` coloring), ``xt`` vs ``thetat`` scatter (grey), and
         the polynomial fit curve (red) after a successful run.

User parameters
---------------
``xcal_idx0_offset``  int   — extra indices to extend the mask below (default 3)
``xcal_idx1_offset``  int   — extra indices to extend the mask above (default 9)
``xcal_std_cutoff``   float — sigma cutoff on theta_t before range detection (default 12.0)
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .core import register_panel, StepPanel, _density_subsample
from citkid.xcal.xcal import get_xcal_mask as _compute_xcal_mask


@register_panel('get_xcal_mask', 'fit_x_theta')
class XCalPanel(StepPanel):
    """
    Panel for the get_xcal_mask + fit_x_theta steps.

    The xcal mask is computed automatically from the user-adjustable offset and
    cutoff parameters. No interactive mask drawing is needed.

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
        ctrl.setSpacing(6)

        # xcal_idx0_offset (int)
        ctrl.addWidget(QtWidgets.QLabel("idx0_off:"))
        self._idx0_spin = QtWidgets.QSpinBox()
        self._idx0_spin.setRange(-500, 500)
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
        self._idx1_spin.setRange(-500, 500)
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
        self._param_cache: dict = {}  # {data_idx: (idx0, idx1, std)}
        self._plot_cache:  dict = {}  # populated by prefetch_plot_data

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
        _inc_brush    = pg.mkBrush(100, 180, 255, 210)   # included (blue)
        _exc_brush    = pg.mkBrush(255, 140,  50, 180)   # excluded (orange)
        _ts_brush     = pg.mkBrush(160, 220, 160, 120)   # timestream bulk (light green)
        _ts_cut_brush = pg.mkBrush(220,  80, 180, 200)   # timestream tail (magenta)

        # Amplitude plot scatter items
        self._amp_data = self._plot_amp.plot(pen=None, symbolBrush=_inc_brush,
                                              symbolPen=None, symbolSize=4, name='data')
        self._amp_excl = self._plot_amp.plot(pen=None, symbolBrush=_exc_brush,
                                              symbolPen=None, symbolSize=4, name='excluded')

        # IQ loop scatter items
        self._iq_data    = self._plot_iq.plot(pen=None, symbolBrush=_inc_brush,
                                               symbolPen=None, symbolSize=4, name='data')
        self._iq_excl    = self._plot_iq.plot(pen=None, symbolBrush=_exc_brush,
                                               symbolPen=None, symbolSize=4, name='excluded')
        self._zt_iq      = self._plot_iq.plot(pen=None, symbolBrush=_ts_brush,
                                               symbolPen=None, symbolSize=2, name='zt bulk')
        self._zt_iq_cut  = self._plot_iq.plot(pen=None, symbolBrush=_ts_cut_brush,
                                               symbolPen=None, symbolSize=3, name='zt tail')

        # x vs theta scatter + fit items
        self._xcal_fit  = self._plot_xcal.plot(
            pen=pg.mkPen(color=(255, 80, 80), width=2), name='poly fit'
        )
        self._xcal_fit.setZValue(-10)
        self._xcal_inc    = self._plot_xcal.plot(pen=None, symbolBrush=_inc_brush,
                                                  symbolPen=None, symbolSize=4, name='xf incl.')
        self._xcal_exc    = self._plot_xcal.plot(pen=None, symbolBrush=_exc_brush,
                                                  symbolPen=None, symbolSize=4, name='xf excl.')
        self._xcal_ts     = self._plot_xcal.plot(pen=None, symbolBrush=_ts_brush,
                                                  symbolPen=None, symbolSize=4, name='xt bulk')
        self._xcal_ts_cut = self._plot_xcal.plot(pen=None, symbolBrush=_ts_cut_brush,
                                                  symbolPen=None, symbolSize=4, name='xt tail')

        # Draggable mask region on amplitude plot
        self._mask_region = pg.LinearRegionItem(
            movable=True,
            brush=pg.mkBrush(100, 180, 255, 30),
            pen=pg.mkPen(color=(100, 180, 255, 200), width=4),
        )
        self._mask_region.setZValue(-5)
        self._plot_amp.addItem(self._mask_region)
        self._mask_region.sigRegionChangeFinished.connect(self._on_mask_region_changed)

        # State for spinbox ↔ region sync
        self._idx0_base: int = 0
        self._idx1_base: int = 0
        self._ff_plot_arr: np.ndarray | None = None
        self._region_updating: bool = False

        # Sync region when offsets are manually edited
        self._idx0_spin.valueChanged.connect(self._update_region_from_spinboxes)
        self._idx1_spin.valueChanged.connect(self._update_region_from_spinboxes)

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

    def on_data_idx_changing(self):
        """Restore user params for the new index: session cache → DS/YAML → keep current."""
        self._autorange_next = True
        new_di = self.data_idx
        if new_di in self._param_cache:
            idx0, idx1, std = self._param_cache[new_di]
        else:
            idx0 = self._get_initial_user_param(
                'get_xcal_mask', 'xcal_idx0_offset', new_di, fallback=None)
            idx1 = self._get_initial_user_param(
                'get_xcal_mask', 'xcal_idx1_offset', new_di, fallback=None)
            std  = self._get_initial_user_param(
                'get_xcal_mask', 'xcal_std_cutoff',  new_di, fallback=None)

        def _set(spin, val):
            if val is not None:
                spin.blockSignals(True)
                spin.setValue(type(spin.value())(val))
                spin.blockSignals(False)

        _set(self._idx0_spin, idx0)
        _set(self._idx1_spin, idx1)
        _set(self._std_spin,  std)

    def run_steps(self, save: bool = False) -> bool:
        """Cache the current param values for this data_idx before running."""
        self._param_cache[self.data_idx] = (
            self._idx0_spin.value(),
            self._idx1_spin.value(),
            self._std_spin.value(),
        )
        ok = super().run_steps(save=save)
        if ok:
            self._range_xcal()
        return ok

    def get_params_for_step(self, step) -> dict:
        if step.name == 'get_xcal_mask':
            return {
                'xcal_idx0_offset': int(self._idx0_spin.value()),
                'xcal_idx1_offset': int(self._idx1_spin.value()),
                'xcal_std_cutoff':  float(self._std_spin.value()),
            }
        return {}

    def update_plots(self):
        di = self.data_idx
        DS = self.AR.DS

        # Use pre-computed numpy arrays from the prefetch cache when available.
        cache = self._plot_cache.pop(di, None)

        # Raw sweep data
        try:
            ff      = cache['ff']      if cache else np.asarray(DS.ff[di],      dtype=np.float64)
            zf_rmv  = cache['zf_rmv']  if cache else np.asarray(DS.zf_rmv[di],  dtype=np.complex128)
            zf_cent = cache['zf_cent'] if cache else np.asarray(DS.zf_cent[di], dtype=np.complex128)
        except Exception as exc:
            self._status_label.setText(f"Data load error: {exc}")
            return

        f_center   = cache['f_center'] if cache else float(ff.mean())
        ff_plot    = cache['ff_plot']  if cache else (ff - f_center) * 1e-3
        amp_db     = cache['amp_db']   if cache else 20.0 * np.log10(np.abs(zf_rmv))
        center_MHz = round(f_center * 1e-6, 2)
        xlabel     = f"(f \u2212 {center_MHz} MHz) (kHz)"
        self._plot_amp.setLabel('bottom', xlabel)

        # Store ff_plot for drag/spinbox sync; re-derive base mask indices.
        self._ff_plot_arr = ff_plot
        if cache and cache.get('idx0_base') is not None:
            self._idx0_base = cache['idx0_base']
            self._idx1_base = cache['idx1_base']
        else:
            try:
                _thetaf_b = np.asarray(DS.thetaf[di], dtype=np.float64)
                _thetat_b = np.asarray(DS.thetat[di], dtype=np.float64)
                _bm = _compute_xcal_mask(ff, _thetaf_b, _thetat_b,
                                         idx0_offset=0, idx1_offset=0,
                                         std_cutoff=float(self._std_spin.value()))
                _wb = np.where(_bm)[0]
                if len(_wb):
                    self._idx0_base = int(_wb[0])
                    self._idx1_base = int(_wb[-1])
            except Exception:
                pass
        self._update_region_from_spinboxes()

        # Load xcal_mask if available; fall back to all-True
        try:
            xcal_mask = cache['xcal_mask'] if cache else np.asarray(DS.xcal_mask[di], dtype=bool)
        except Exception:
            xcal_mask = np.ones(len(ff), dtype=bool)
            # All-included before step has run — render without mask coloring
            self._amp_data.setData(ff_plot, amp_db)
            self._amp_excl.setData([], [])
            self._iq_data.setData(zf_cent.real, zf_cent.imag)
            self._iq_excl.setData([], [])
        else:
            self._amp_data.setData(ff_plot[xcal_mask],  amp_db[xcal_mask])
            self._amp_excl.setData(ff_plot[~xcal_mask], amp_db[~xcal_mask])
            self._iq_data.setData(zf_cent[xcal_mask].real,  zf_cent[xcal_mask].imag)
            self._iq_excl.setData(zf_cent[~xcal_mask].real, zf_cent[~xcal_mask].imag)

        # zt_cent on IQ: split by bulk_mask so tail points show in magenta.
        # bulk_mask needs thetat + std_cut; load thetat here for reuse below.
        _thetat_early = None
        _bulk_mask_early = None
        try:
            if cache and cache.get('zt_cent_bulk') is not None:
                self._zt_iq.setData(cache['zt_cent_bulk'].real, cache['zt_cent_bulk'].imag)
                self._zt_iq_cut.setData(cache['zt_cent_tail'].real, cache['zt_cent_tail'].imag)
            else:
                _thetat_early = np.asarray(DS.thetat[di], dtype=np.float64)
                std_cut = float(self._std_spin.value())
                _bulk_mask_early = (
                    np.abs(_thetat_early - _thetat_early.mean())
                    < std_cut * _thetat_early.std()
                )
                zt_full = np.asarray(DS.zt_cent[di], dtype=np.complex128)
                zt_bulk = _density_subsample(zt_full[_bulk_mask_early])
                zt_tail = zt_full[~_bulk_mask_early]
                self._zt_iq.setData(zt_bulk.real, zt_bulk.imag)
                self._zt_iq_cut.setData(zt_tail.real, zt_tail.imag)
        except Exception:
            self._zt_iq.setData([], [])
            self._zt_iq_cut.setData([], [])

        # x vs theta calibration plot (needs cal-step outputs thetaf, xf)
        try:
            thetaf = cache['thetaf'] if cache else np.asarray(DS.thetaf[di], dtype=np.float64)
            xf     = cache['xf']     if cache else np.asarray(DS.xf[di],     dtype=np.float64)
        except Exception:
            # Cal outputs not yet available — leave cal plot empty
            self._xcal_inc.setData([], [])
            self._xcal_exc.setData([], [])
            self._xcal_ts.setData([], [])
            self._xcal_ts_cut.setData([], [])
            self._xcal_fit.setData([], [])
        else:
            self._xcal_inc.setData(thetaf[xcal_mask],  xf[xcal_mask])
            self._xcal_exc.setData(thetaf[~xcal_mask], xf[~xcal_mask])

            # Polynomial fit line + timestream overlay (needs poly_x from step 7)
            try:
                poly_x = cache['poly_x'] if cache else np.asarray(DS.poly_x[di], dtype=np.float64)
                theta_grid = cache['theta_grid'] if cache else np.linspace(thetaf.min(), thetaf.max(), 500)
                fit_y      = cache['fit_y']      if cache else np.polyval(poly_x, theta_grid)
                self._xcal_fit.setData(theta_grid, fit_y)

                # Timestream xt split into bulk/tail — reuse _bulk_mask_early when available
                try:
                    if cache and cache.get('t_bulk') is not None:
                        t_bulk    = cache['t_bulk']
                        x_bulk    = cache['x_bulk']
                        t_tail    = cache['t_tail']
                        x_tail    = cache['x_tail']
                    else:
                        if _thetat_early is not None:
                            thetat    = _thetat_early
                            bulk_mask = _bulk_mask_early
                        else:
                            thetat    = np.asarray(DS.thetat[di], dtype=np.float64)
                            std_cut   = float(self._std_spin.value())
                            bulk_mask = (
                                np.abs(thetat - thetat.mean())
                                < std_cut * thetat.std()
                            )
                        xt     = np.polyval(poly_x, thetat)
                        t_bulk = thetat[bulk_mask]
                        x_bulk = xt[bulk_mask]
                        t_tail = thetat[~bulk_mask]
                        x_tail = xt[~bulk_mask]
                        # Density-subsample bulk (4000) and tail (2000) independently
                        # so neither dominates the visual density.
                        if len(t_bulk) > 4000:
                            sub_b  = _density_subsample(
                                (t_bulk + 1j * x_bulk).astype(np.complex128), n_keep=4000
                            )
                            t_bulk = sub_b.real
                            x_bulk = sub_b.imag
                        if len(t_tail) > 2000:
                            sub_t  = _density_subsample(
                                (t_tail + 1j * x_tail).astype(np.complex128), n_keep=2000
                            )
                            t_tail = sub_t.real
                            x_tail = sub_t.imag
                    self._xcal_ts.setData(t_bulk, x_bulk)
                    self._xcal_ts_cut.setData(t_tail, x_tail)
                except Exception:
                    self._xcal_ts.setData([], [])
                    self._xcal_ts_cut.setData([], [])
            except Exception:
                self._xcal_fit.setData([], [])
                self._xcal_ts.setData([], [])
                self._xcal_ts_cut.setData([], [])

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

    def _update_region_from_spinboxes(self):
        """Reposition the draggable mask region to reflect the current spinbox offsets."""
        if self._ff_plot_arr is None or len(self._ff_plot_arr) == 0:
            return
        n = len(self._ff_plot_arr)
        idx0 = max(0, self._idx0_base - self._idx0_spin.value())
        idx1 = min(n - 1, self._idx1_base + self._idx1_spin.value())
        if idx1 < idx0:
            idx1 = idx0
        self._region_updating = True
        self._mask_region.setRegion((
            float(self._ff_plot_arr[idx0]),
            float(self._ff_plot_arr[idx1]),
        ))
        self._region_updating = False

    def _on_mask_region_changed(self):
        """Update idx0_off / idx1_off spinboxes from the dragged region boundaries."""
        if self._region_updating or self._ff_plot_arr is None:
            return
        lo, hi = self._mask_region.getRegion()
        ff_arr = self._ff_plot_arr
        new_idx0 = int(np.argmin(np.abs(ff_arr - lo)))
        new_idx1 = int(np.argmin(np.abs(ff_arr - hi)))
        new_off0 = self._idx0_base - new_idx0
        new_off1 = new_idx1 - self._idx1_base
        self._idx0_spin.blockSignals(True)
        self._idx1_spin.blockSignals(True)
        self._idx0_spin.setValue(new_off0)
        self._idx1_spin.setValue(new_off1)
        self._idx0_spin.blockSignals(False)
        self._idx1_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------


    def _on_save_clicked(self):
        try:
            self.save_outputs()
            self._status_label.setText("Saved \u2713")
        except Exception as exc:
            self._status_label.setText("Save error \u2717")
            print(f"Save error: {exc}")

    def save_outputs(self):
        """Save step outputs AND user params (offsets, cutoff) to zarr."""
        for step in self.steps:
            user_params = self.get_params_for_step(step)
            if user_params:
                pipeline_scope, step_index = self.AR._resolve_step_scope(step)
                self.AR._add_user_params(
                    step, user_params,
                    self.data_idx, save=True,
                    pipeline_scope=pipeline_scope, step_index=step_index
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
            ff      = np.asarray(DS.ff[di],      dtype=np.float64)
            zf_rmv  = np.asarray(DS.zf_rmv[di],  dtype=np.complex128)
            zf_cent = np.asarray(DS.zf_cent[di], dtype=np.complex128)
        except Exception:
            return

        f_center = float(ff.mean())
        ff_plot  = (ff - f_center) * 1e-3
        amp_db   = 20.0 * np.log10(np.abs(zf_rmv))

        try:
            xcal_mask = np.asarray(DS.xcal_mask[di], dtype=bool)
        except Exception:
            xcal_mask = None

        # zt_cent split into bulk (density-subsampled) and tail
        zt_cent_bulk = zt_cent_tail = None
        try:
            zt_full = np.asarray(DS.zt_cent[di], dtype=np.complex128)
            thetat  = np.asarray(DS.thetat[di],  dtype=np.float64)
            std_cut = float(self._std_spin.value())
            bulk_mask_zt = (
                np.abs(thetat - thetat.mean()) < std_cut * thetat.std()
            )
            zt_cent_bulk = _density_subsample(zt_full[bulk_mask_zt], n_keep=4000)
            zt_tail_raw  = zt_full[~bulk_mask_zt]
            zt_cent_tail = (_density_subsample(zt_tail_raw, n_keep=2000)
                            if len(zt_tail_raw) > 2000 else zt_tail_raw)
        except Exception:
            pass

        # xcal plot data
        thetaf = xf = poly_x = theta_grid = fit_y = None
        t_bulk = x_bulk = t_tail = x_tail = None
        try:
            thetaf = np.asarray(DS.thetaf[di], dtype=np.float64)
            xf     = np.asarray(DS.xf[di],     dtype=np.float64)
            poly_x = np.asarray(DS.poly_x[di], dtype=np.float64)
            theta_grid = np.linspace(thetaf.min(), thetaf.max(), 500)
            fit_y      = np.polyval(poly_x, theta_grid)
            thetat_pf   = np.asarray(DS.thetat[di], dtype=np.float64)
            xt          = np.polyval(poly_x, thetat_pf)
            std_cut     = float(self._std_spin.value())
            bulk_mask_pf = (
                np.abs(thetat_pf - thetat_pf.mean()) < std_cut * thetat_pf.std()
            )
            t_bulk = thetat_pf[bulk_mask_pf]
            x_bulk = xt[bulk_mask_pf]
            t_tail = thetat_pf[~bulk_mask_pf]
            x_tail = xt[~bulk_mask_pf]
            if len(t_bulk) > 4000:
                sub_b  = _density_subsample(
                    (t_bulk + 1j * x_bulk).astype(np.complex128), n_keep=4000
                )
                t_bulk = sub_b.real
                x_bulk = sub_b.imag
            if len(t_tail) > 2000:
                sub_t  = _density_subsample(
                    (t_tail + 1j * x_tail).astype(np.complex128), n_keep=2000
                )
                t_tail = sub_t.real
                x_tail = sub_t.imag
        except Exception:
            pass

        # Base mask indices (zero offsets) for the draggable region
        idx0_base = idx1_base = None
        try:
            _thetaf_b = np.asarray(DS.thetaf[di], dtype=np.float64)
            _thetat_b = np.asarray(DS.thetat[di], dtype=np.float64)
            _bm = _compute_xcal_mask(ff, _thetaf_b, _thetat_b,
                                     idx0_offset=0, idx1_offset=0,
                                     std_cutoff=float(self._std_spin.value()))
            _wb = np.where(_bm)[0]
            if len(_wb):
                idx0_base = int(_wb[0])
                idx1_base = int(_wb[-1])
        except Exception:
            pass

        self._plot_cache[di] = dict(
            ff=ff, ff_plot=ff_plot, f_center=f_center,
            zf_rmv=zf_rmv, zf_cent=zf_cent, amp_db=amp_db,
            xcal_mask=xcal_mask, zt_cent_bulk=zt_cent_bulk, zt_cent_tail=zt_cent_tail,
            thetaf=thetaf, xf=xf, poly_x=poly_x,
            theta_grid=theta_grid, fit_y=fit_y,
            t_bulk=t_bulk, x_bulk=x_bulk, t_tail=t_tail, x_tail=x_tail,
            idx0_base=idx0_base, idx1_base=idx1_base,
        )

    def _nan_outputs(self) -> list:
        """Return list of parameter names to delete (v2 deletion-based marking)."""
        return [
            'xcal_mask',
            'poly_x',
        ]
