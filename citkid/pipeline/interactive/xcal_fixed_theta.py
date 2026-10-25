"""
Interactive panel for the fixed-theta x-calibration steps ('ts-fixed_theta').

Covers two linked analysis steps:

    6. get_xcal_mask_fixed_theta — keeps the sweep points whose theta_f is at
       least a fixed offset above its minimum and below its maximum
    7. fit_x_theta               — fits a polynomial x vs theta calibration

Plots
-----
Left   : ``20*log10(|zf_rmv|)`` vs ``ff``, colored by the mask (blue =
         included, orange = excluded).
Middle : IQ loop — ``zf_cent`` colored by the mask, with the timestream
         ``zt_cent`` overlaid (light green).
Right  : x vs theta — ``xf`` vs ``thetaf`` colored by the mask, the timestream
         ``xt`` vs ``thetat`` (light green) and the polynomial fit (red), with
         two dashed lines at the edges of the theta band. Drag them to change
         the offsets.

The mask colors follow the offsets as you change them; click Run to apply.

User parameters
---------------
``xcal_theta_min_offset``  float — rad up from the minimum of thetaf (0.2)
``xcal_theta_max_offset``  float — rad down from the maximum of thetaf (0.2)
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .core import register_panel, StepPanel, _density_subsample
from ...xcal.xcal import get_xcal_mask_fixed_theta

_STEP = 'get_xcal_mask_fixed_theta'
_DEFAULT_OFFSET = 0.2


def _load_arrays(DS, di):
    """
    Load and prepare one row's plot data.

    Parameters:
    DS (DataSet): DataSet of the series point shown.
    di (int): Row.

    Returns:
    arrays (dict): ``ff``, ``ff_plot`` (kHz from ``f_center``),
        ``f_center``, ``amp_db``, ``zf_cent`` and, when available,
        ``thetaf``, ``xf``, ``zt_sub`` (subsampled ``zt_cent``), ``thetat``,
        ``poly_x`` (None otherwise).

    Raises:
    Exception: If the sweep (``ff``, ``zf_rmv``, ``zf_cent``) can't be loaded.
    """
    ff = np.asarray(DS.ff[di], dtype=np.float64)
    zf_rmv = np.asarray(DS.zf_rmv[di], dtype=np.complex128)
    zf_cent = np.asarray(DS.zf_cent[di], dtype=np.complex128)
    f_center = float(ff.mean())
    arrays = dict(ff=ff, ff_plot=(ff - f_center) * 1e-3, f_center=f_center,
                  amp_db=20.0 * np.log10(np.abs(zf_rmv)), zf_cent=zf_cent)
    for name, dtype in (('thetaf', np.float64), ('xf', np.float64),
                        ('thetat', np.float64), ('poly_x', np.float64)):
        try:
            arrays[name] = np.asarray(getattr(DS, name)[di], dtype=dtype)
        except Exception:
            arrays[name] = None
    try:
        zt = np.asarray(DS.zt_cent[di], dtype=np.complex128)
        arrays['zt_sub'] = _density_subsample(zt, n_keep=4000)
    except Exception:
        arrays['zt_sub'] = None
    return arrays


@register_panel(_STEP, 'fit_x_theta')
class XCalFixedThetaPanel(StepPanel):
    """
    Panel for the get_xcal_mask_fixed_theta + fit_x_theta steps.

    The mask is set by two theta offsets, from spin boxes or by dragging the
    band edges on the x vs theta plot.

    Parameters:
    AR (AnalysisRunner): The analysis runner.
    step_names (tuple of str): Step names handled by this panel.
    data_idx (int or None): Initial data index.
    parent (QWidget or None): Parent widget.
    """

    def setup_ui(self):
        """Build the offset controls, buttons, plots and band edges."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        # ---- control row ----
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(6)
        self._min_spin = self._offset_spin('xcal_theta_min_offset')
        self._max_spin = self._offset_spin('xcal_theta_max_offset')
        for label, spin, tip in (
            ('θ min off:', self._min_spin,
             'Offset (rad) up from the minimum of θf'),
            ('θ max off:', self._max_spin,
             'Offset (rad) down from the maximum of θf'),
        ):
            spin.setToolTip(tip)
            ctrl.addWidget(QtWidgets.QLabel(label))
            ctrl.addWidget(spin)

        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setFixedWidth(50)
        self._run_btn.clicked.connect(self._on_run_clicked)
        ctrl.addWidget(self._run_btn)

        self._run_through_btn = QtWidgets.QPushButton("Run+")
        self._run_through_btn.setFixedWidth(55)
        self._run_through_btn.setToolTip(
            "Run this panel and all following panels")
        self._run_through_btn.clicked.connect(lambda: self.run_from_here.emit())
        ctrl.addWidget(self._run_through_btn)

        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setFixedWidth(60)
        self._save_btn.clicked.connect(self._on_save_clicked)
        ctrl.addWidget(self._save_btn)

        self._bad_btn = QtWidgets.QPushButton("Bad Data")
        self._bad_btn.setFixedWidth(75)
        self._bad_btn.setToolTip(
            "Mark this index as bad: fill all outputs with NaN")
        self._bad_btn.clicked.connect(self._on_bad_data_clicked)
        ctrl.addWidget(self._bad_btn)

        self._status_label = QtWidgets.QLabel("—")
        self._status_label.setMinimumWidth(160)
        ctrl.addWidget(self._status_label)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # Offsets set this session, per row: {data_idx: (min_off, max_off)}
        self._param_cache: dict = {}
        # Row data from the background prefetch: {data_idx: arrays}
        self._plot_cache: dict = {}
        # The shown row's data (see _load_arrays), for recoloring
        self._arrays: dict | None = None
        self._lines_updating = False

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(round(340 * self.plot_scale))
        root.addWidget(self._gw)

        self._plot_amp = self._gw.addPlot(row=0, col=0, title="XCal Amplitude")
        self._plot_amp.setLabel('left', '|S21| (dB)')
        self._plot_amp.setLabel('bottom', 'Frequency (Hz)')
        self._plot_amp.showGrid(x=True, y=True, alpha=0.3)

        self._plot_iq = self._gw.addPlot(row=0, col=1, title="IQ Loop")
        self._plot_iq.setLabel('left', 'Q (Im)')
        self._plot_iq.setLabel('bottom', 'I (Re)')
        self._plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self._plot_iq.setAspectLocked(True)

        self._plot_xcal = self._gw.addPlot(
            row=0, col=2, title="x vs θ Calibration")
        self._plot_xcal.setLabel('left', 'x (frac. freq. det.)')
        self._plot_xcal.setLabel('bottom', 'θ (rad)')
        self._plot_xcal.showGrid(x=True, y=True, alpha=0.3)

        self._scale_plot_fonts(self._plot_amp, self._plot_iq, self._plot_xcal)

        inc = pg.mkBrush(100, 180, 255, 210)   # included (blue)
        exc = pg.mkBrush(255, 140, 50, 180)    # excluded (orange)
        ts = pg.mkBrush(160, 220, 160, 120)    # timestream (light green)

        def scatter(plot, brush, size, name):
            return plot.plot(pen=None, symbolBrush=brush, symbolPen=None,
                             symbolSize=size, name=name)

        self._amp_data = scatter(self._plot_amp, inc, 4, 'data')
        self._amp_excl = scatter(self._plot_amp, exc, 4, 'excluded')
        self._zt_iq = scatter(self._plot_iq, ts, 2, 'zt')
        self._iq_data = scatter(self._plot_iq, inc, 4, 'data')
        self._iq_excl = scatter(self._plot_iq, exc, 4, 'excluded')
        self._xcal_fit = self._plot_xcal.plot(
            pen=pg.mkPen(color=(255, 80, 80), width=2), name='poly fit')
        self._xcal_fit.setZValue(-10)
        self._xcal_ts = scatter(self._plot_xcal, ts, 4, 'xt')
        self._xcal_inc = scatter(self._plot_xcal, inc, 4, 'xf incl.')
        self._xcal_exc = scatter(self._plot_xcal, exc, 4, 'xf excl.')

        # Band edges on the x vs theta plot (theta on the horizontal axis)
        pen = pg.mkPen((100, 180, 255), width=2, style=QtCore.Qt.DashLine)
        self._lo_line = pg.InfiniteLine(angle=90, movable=True, pen=pen)
        self._hi_line = pg.InfiniteLine(angle=90, movable=True, pen=pen)
        for line in (self._lo_line, self._hi_line):
            line.setVisible(False)
            self._plot_xcal.addItem(line, ignoreBounds=True)
            line.sigPositionChangeFinished.connect(self._on_line_moved)

        self._min_spin.valueChanged.connect(self._on_offsets_edited)
        self._max_spin.valueChanged.connect(self._on_offsets_edited)

    def _offset_spin(self, param):
        """
        Build a spin box for a theta offset, starting at its saved value.

        Parameters:
        param (str): User parameter name.

        Returns:
        spin (QDoubleSpinBox): the spin box.
        """
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 10.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        spin.setSuffix(' rad')
        spin.setFixedWidth(90)
        spin.setValue(float(self._get_initial_user_param(
            _STEP, param, self.data_idx, fallback=_DEFAULT_OFFSET)))
        return spin

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

    def on_data_idx_changing(self):
        """Restore offsets for the new row: session cache, then DS/YAML."""
        self._autorange_next = True
        self._arrays = None
        di = self.data_idx
        if di in self._param_cache:
            values = self._param_cache[di]
        else:
            params = ('xcal_theta_min_offset', 'xcal_theta_max_offset')
            values = [
                self._get_initial_user_param(_STEP, param, di, fallback=None)
                for param in params
            ]
        for spin, value in zip((self._min_spin, self._max_spin), values):
            if value is not None:
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

    def run_steps(self, save: bool = False) -> bool:
        """Cache the offsets for this row, then run."""
        self._param_cache[self.data_idx] = self._offsets()
        ok = super().run_steps(save=save)
        if ok:
            self._range_xcal()
        return ok

    def get_params_for_step(self, step) -> dict:
        """Return the theta offsets for ``get_xcal_mask_fixed_theta``."""
        if step.name == _STEP:
            lo, hi = self._offsets()
            return {'xcal_theta_min_offset': lo, 'xcal_theta_max_offset': hi}
        return {}

    def update_plots(self):
        """Plot the sweep, timestream, fit and mask of the current row."""
        di = self.data_idx
        arrays = self._plot_cache.pop(di, None)
        if arrays is None:
            try:
                arrays = _load_arrays(self.AR.DS, di)
            except Exception as exc:
                self._status_label.setText(f"Data load error: {exc}")
                return
        self._arrays = arrays
        center_MHz = round(arrays['f_center'] * 1e-6, 2)
        self._plot_amp.setLabel('bottom', f"(f − {center_MHz} MHz) (kHz)")

        zt = arrays['zt_sub']
        if zt is not None:
            self._zt_iq.setData(zt.real, zt.imag)
        else:
            self._zt_iq.setData([], [])
        thetaf, thetat = arrays['thetaf'], arrays['thetat']
        poly_x = arrays['poly_x']
        if (thetaf is not None and poly_x is not None
                and np.all(np.isfinite(poly_x))):
            grid = np.linspace(np.nanmin(thetaf), np.nanmax(thetaf), 500)
            self._xcal_fit.setData(grid, np.polyval(poly_x, grid))
            if thetat is not None:
                xt = np.polyval(poly_x, thetat)
                points = (thetat + 1j * xt).astype(np.complex128)
                sub = _density_subsample(points, n_keep=4000)
                self._xcal_ts.setData(sub.real, sub.imag)
            else:
                self._xcal_ts.setData([], [])
        else:
            self._xcal_fit.setData([], [])
            self._xcal_ts.setData([], [])
        self._recolor()

        if self._autorange_next:
            self._autorange_next = False
            self._plot_amp.autoRange()
            self._plot_iq.autoRange()
            self._range_xcal()

    def prefetch_plot_data(self, di: int):
        """Load the plot data of row *di* in the background prefetch thread."""
        try:
            self._plot_cache[di] = _load_arrays(self.AR.DS, di)
        except Exception:
            pass

    def autoscale_plots(self):
        """Auto-range every plot (x vs theta to the masked points)."""
        self._plot_amp.autoRange()
        self._plot_iq.autoRange()
        self._range_xcal()

    def clear_plots(self):
        """
        Blank every data curve, hide the band edges and reset the status label.
        """
        for curve in (self._amp_data, self._amp_excl, self._iq_data,
                      self._iq_excl, self._zt_iq, self._xcal_fit,
                      self._xcal_inc, self._xcal_exc, self._xcal_ts):
            curve.setData([], [])
        for line in (self._lo_line, self._hi_line):
            line.setVisible(False)
        self._arrays = None
        self._status_label.setText("—")

    # ------------------------------------------------------------------
    # Mask and band edges
    # ------------------------------------------------------------------

    def _offsets(self):
        """Return the (min, max) theta offsets in the spin boxes, in rad."""
        return float(self._min_spin.value()), float(self._max_spin.value())

    def _preview_mask(self):
        """
        Return the mask for the current offsets.

        Returns:
        mask (np.ndarray of bool or None): ``get_xcal_mask_fixed_theta`` of
            the shown row (all False if the band is empty), or None if
            ``thetaf`` isn't available.
        """
        thetaf = None if self._arrays is None else self._arrays['thetaf']
        if thetaf is None:
            return None
        try:
            return get_xcal_mask_fixed_theta(thetaf, *self._offsets())
        except ValueError:
            return np.zeros(len(thetaf), dtype=bool)

    def _recolor(self):
        """Color the sweep by the offsets' mask and place the band edges."""
        a = self._arrays
        if a is None:
            return
        mask = self._preview_mask()
        if mask is None:
            mask = np.ones(len(a['ff']), dtype=bool)
        self._amp_data.setData(a['ff_plot'][mask], a['amp_db'][mask])
        self._amp_excl.setData(a['ff_plot'][~mask], a['amp_db'][~mask])
        z = a['zf_cent']
        self._iq_data.setData(z[mask].real, z[mask].imag)
        self._iq_excl.setData(z[~mask].real, z[~mask].imag)
        thetaf, xf = a['thetaf'], a['xf']
        if thetaf is None or xf is None:
            self._xcal_inc.setData([], [])
            self._xcal_exc.setData([], [])
            for line in (self._lo_line, self._hi_line):
                line.setVisible(False)
            return
        self._xcal_inc.setData(thetaf[mask], xf[mask])
        self._xcal_exc.setData(thetaf[~mask], xf[~mask])
        lo, hi = self._offsets()
        self._lines_updating = True
        self._lo_line.setValue(float(np.nanmin(thetaf)) + lo)
        self._hi_line.setValue(float(np.nanmax(thetaf)) - hi)
        self._lines_updating = False
        for line in (self._lo_line, self._hi_line):
            line.setVisible(True)
        if not mask.any():
            self._status_label.setText("Offsets leave no points")

    def _on_offsets_edited(self):
        """Recolor and move the band edges after a spin box edit."""
        if self._arrays is not None:
            self._status_label.setText("Offsets set — click Run")
        self._recolor()  # says so if the offsets leave no points

    def _on_line_moved(self, line):
        """
        Set the offsets from a dragged band edge.

        Parameters:
        line (pg.InfiniteLine): the dragged line.
        """
        if self._lines_updating or self._arrays is None:
            return
        thetaf = self._arrays['thetaf']
        if thetaf is None:
            return
        if line is self._lo_line:
            spin, value = self._min_spin, line.value() - np.nanmin(thetaf)
        else:
            spin, value = self._max_spin, np.nanmax(thetaf) - line.value()
        spin.setValue(max(0.0, float(value)))  # recolors via valueChanged

    def _range_xcal(self):
        """Set the x vs theta range to the masked points plus 5% padding."""
        a = self._arrays
        mask = self._preview_mask()
        if a is None or mask is None or a['xf'] is None or not mask.any():
            self._plot_xcal.autoRange()
            return
        theta_m, x_m = a['thetaf'][mask], a['xf'][mask]
        t_lo, t_hi = float(theta_m.min()), float(theta_m.max())
        x_lo, x_hi = float(x_m.min()), float(x_m.max())
        t_pad = max((t_hi - t_lo) * 0.05, 1e-6)
        x_pad = max((x_hi - x_lo) * 0.05, 1e-9)
        self._plot_xcal.setXRange(t_lo - t_pad, t_hi + t_pad, padding=0)
        self._plot_xcal.setYRange(x_lo - x_pad, x_hi + x_pad, padding=0)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _on_step_error(self, step, exc):
        """Show a step error in the status label and print it."""
        msg = f"'{step.name}' failed: {exc}"
        self._status_label.setText(msg)
        print(msg)

    def _nan_outputs(self) -> dict:
        """Return the bad-data outputs: an all-False mask and NaN poly_x."""
        try:
            ff_len = len(np.asarray(self.AR.DS.ff[self.data_idx]))
        except Exception:
            ff_len = 1
        return {
            'xcal_mask': np.zeros(ff_len, dtype=bool),
            'poly_x': np.full(4, np.nan),
        }
