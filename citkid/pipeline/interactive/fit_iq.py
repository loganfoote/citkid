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
The shaded region on the left plot defines which frequency samples are
included in the fit (``iq_mask = True`` inside the region).  Drag either
edge of the region to adjust the window.  Changes are debounced (300 ms)
to avoid running the fitter on every mouse-move event.

Press **Reset Mask** to restore the full frequency range.

Cascade behaviour
-----------------
After a successful run the panel emits ``downstream_rerun`` so that
subsequent panels are automatically updated.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .core import register_panel, StepPanel
from ...res.funcs import nonlinear_iq


@register_panel('fit_iq')
class FitIQPanel(StepPanel):
    """
    Panel for the *fit_iq* step.

    The IQ mask is controlled interactively via a
    :class:`~pyqtgraph.LinearRegionItem` on the amplitude vs frequency plot.

    Parameters
    ----------
    AR : AnalysisRunner
    step_names : tuple of str
    data_idx : int or None
    parent : QWidget, optional
    """

    # Debounce interval in ms before re-running the fit after the region moves
    _DEBOUNCE_MS = 300

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

        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setFixedWidth(60)
        self._run_btn.clicked.connect(self._on_run_clicked)
        ctrl.addWidget(self._run_btn)

        self._status_label = QtWidgets.QLabel("—")
        self._status_label.setMinimumWidth(180)
        ctrl.addWidget(self._status_label)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- plots ----
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumHeight(340)
        root.addWidget(self._gw)

        self._plot_amp = self._gw.addPlot(row=0, col=0, title="IQ Amplitude")
        self._plot_amp.setLabel('left', '|S21| (dB)')
        self._plot_amp.setLabel('bottom', 'Frequency (Hz)')
        self._plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self._plot_amp.setDownsampling(auto=True, mode='peak')

        self._plot_iq = self._gw.addPlot(row=0, col=1, title="IQ Loop")
        self._plot_iq.setLabel('left', 'Q (Im)')
        self._plot_iq.setLabel('bottom', 'I (Re)')
        self._plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self._plot_iq.setAspectLocked(True)

        # Data curves
        _data_pen         = pg.mkPen(color=(100, 180, 255, 180), width=1)
        _data_pen_dimmed  = pg.mkPen(color=(100, 180, 255, 60),  width=1)
        self._amp_data      = self._plot_amp.plot(pen=_data_pen,       name='data')
        self._amp_excl      = self._plot_amp.plot(pen=_data_pen_dimmed, name='excluded')
        self._iq_data       = self._plot_iq.plot(pen=_data_pen,        name='data')
        self._iq_data_excl  = self._plot_iq.plot(pen=_data_pen_dimmed, name='excluded')

        # Fit overlay curve on IQ loop
        _fit_pen = pg.mkPen(color=(255, 80, 80), width=2)
        self._iq_fit = self._plot_iq.plot(pen=_fit_pen, name='fit')

        # Linear region item for mask (shown on amplitude plot)
        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(255, 255, 100, 30),
            pen=pg.mkPen(color=(255, 255, 100), width=1)
        )
        self._plot_amp.addItem(self._region)
        self._region.sigRegionChanged.connect(self._on_region_changed)

        # Debounce timer
        self._debounce_timer = QtCore.QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._on_run_clicked)

        # Current mask state (None = full range, i.e. all True)
        self._mask: np.ndarray | None = None
        # Cache of current ff so the mask can be rebuilt from region bounds
        self._ff_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # StepPanel interface
    # ------------------------------------------------------------------

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

        # Build mask from current region bounds
        mask = self._build_mask(ff)
        amp_db = 20.0 * np.log10(np.abs(zf_rmv))

        # Amplitude plot — included vs excluded
        self._amp_data.setData(ff[mask], amp_db[mask])
        self._amp_excl.setData(ff[~mask], amp_db[~mask])

        # IQ loop — full data
        self._iq_data.setData(zf_rmv[mask].real, zf_rmv[mask].imag)
        self._iq_data_excl.setData(zf_rmv[~mask].real, zf_rmv[~mask].imag)

        # Initialise region to full data range on first call
        lo, hi = self._region.getRegion()
        full_lo, full_hi = float(ff.min()), float(ff.max())
        if lo == hi or (lo <= full_lo and hi >= full_hi):
            # Region spans the entire range — keep it that way
            self._region.blockSignals(True)
            self._region.setRegion((full_lo, full_hi))
            self._region.blockSignals(False)

        # Model fit overlay
        try:
            iq_popt = np.asarray(DS.iq_popt[di], dtype=np.float64)
            f_model = np.linspace(ff[mask].min(), ff[mask].max(), 1000)
            z_model = nonlinear_iq(f_model, *iq_popt, downward=True)
            self._iq_fit.setData(z_model.real, z_model.imag)
        except Exception:
            self._iq_fit.setData([], [])

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_mask(self, ff: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask for *ff* based on the current region bounds.

        All samples whose frequency falls within ``[lo, hi]`` (inclusive)
        are ``True``.
        """
        lo, hi = self._region.getRegion()
        mask = (ff >= lo) & (ff <= hi)
        self._mask = mask
        return mask

    def _reset_mask(self):
        """Reset the region to the full frequency range."""
        if self._ff_cache is not None:
            lo = float(self._ff_cache.min())
            hi = float(self._ff_cache.max())
            self._region.blockSignals(True)
            self._region.setRegion((lo, hi))
            self._region.blockSignals(False)
        self._mask = None
        ok = self.run_steps()
        if ok:
            self._status_label.setText("Mask reset ✓")
            self.trigger_downstream()

    def _on_region_changed(self):
        """Debounced handler: rebuild the mask and re-run the fit."""
        self._debounce_timer.start(self._DEBOUNCE_MS)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        self._debounce_timer.stop()
        # Rebuild mask from current region before running
        if self._ff_cache is not None:
            self._build_mask(self._ff_cache)
        self._status_label.setText("Running…")
        QtWidgets.QApplication.processEvents()
        ok = self.run_steps()
        if ok:
            self._status_label.setText("Done ✓")
            self.trigger_downstream()
        else:
            self._status_label.setText("Error ✗")

    def _on_step_error(self, step, exc):
        msg = f"'{step.name}' failed: {exc}"
        self._status_label.setText(msg)
        print(msg)
