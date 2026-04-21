"""
Minimal power optimizer - just load, fit, and plot for one resonator
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from ..qt_compat import Qt as _Qt
from citkid.xcal.gain import make_fr_spans, fit_gain, remove_gain
from citkid.res.fitter import fit_nonlinear_iq
import zarr


def plot_power_sweep(zarr_path, data_idx=0, fres_all=None, qres_all=None):
    """
    Load and fit power sweep data for a single resonator.
    
    Parameters
    ----------
    zarr_path : str
        Path to zarr file with structure:
        - amp_00/, amp_01/, etc. (power groups)
        - Each has: ares, gain_sweep/f, gain_sweep/z, fine_sweep/f, fine_sweep/z
    data_idx : int
        Which resonator to analyze (index into first dimension of arrays)
    fres_all : array_like, optional
        All resonance frequencies for gain removal
    qres_all : array_like, optional
        All Q factors for gain removal
    """
    
    # Create GUI first
    app = pg.mkQApp()
    win = QtWidgets.QWidget()
    win.setWindowTitle(f'Power Sweep - Resonator {data_idx}')
    layout = QtWidgets.QVBoxLayout()
    win.setLayout(layout)
    
    # Add status text area at top
    status_text = QtWidgets.QTextEdit()
    status_text.setReadOnly(True)
    status_text.setMaximumHeight(150)
    layout.addWidget(status_text)
    
    def log(msg):
        """Add message to status text"""
        status_text.append(msg)
        QtWidgets.QApplication.processEvents()
    
    # Show window
    win.show()
    QtWidgets.QApplication.processEvents()
    
    # Open zarr
    log("Opening zarr file...")
    store = zarr.open(zarr_path, mode='r')
    
    # Find power groups
    power_groups = sorted([k for k in store.keys() if k.startswith('amp')])
    n_powers = len(power_groups)
    log(f"Found {n_powers} power levels")
    
    # Load data for this resonator across all powers
    log(f"Loading data for resonator {data_idx}...")
    ares_list = []
    fg_list = []
    zg_list = []
    ff_list = []
    zf_list = []
    
    for pgrp_name in power_groups:
        pgrp = store[pgrp_name]
        ares_list.append(pgrp['ares'][data_idx])
        fg_list.append(pgrp['gain_sweep']['f'][data_idx, :])
        zg_list.append(pgrp['gain_sweep']['z'][data_idx, :])
        ff_list.append(pgrp['fine_sweep']['f'][data_idx, :])
        zf_list.append(pgrp['fine_sweep']['z'][data_idx, :])
    
    ares = np.array(ares_list)
    fg = np.array(fg_list)
    zg = np.array(zg_list)
    ff = np.array(ff_list)
    zf = np.array(zf_list)
    
    log(f"Loaded resonator {data_idx}, ares range: {ares.min():.1f} to {ares.max():.1f} dBm")
    
    # Default gain removal parameters if not provided
    if fres_all is None:
        fres_all = np.array([np.nanmean(ff)])
    if qres_all is None:
        qres_all = np.array([50000.0])
    
    # Fit all powers
    log("Fitting all power levels...")
    p_amps = []
    p_phases = []
    popts = []
    zf_rmvds = []
    
    for i in range(n_powers):
        log(f"  Fitting power {i+1}/{n_powers} ({ares[i]:.1f} dBm)")
        
        # Sort gain data by frequency
        idx_sort = np.argsort(fg[i])
        fg_sorted = fg[i][idx_sort]
        zg_sorted = zg[i][idx_sort]
        
        # Make frequency spans around resonances for gain fitting
        fr_spans = make_fr_spans(fres_all, qres_all)
        
        # Fit gain
        try:
            p_amp, p_phase, mask = fit_gain(fg_sorted, zg_sorted, fr_spans)
        except Exception as e:
            log(f"    Gain fit failed: {e}")
            p_amp = np.array([0, 0, 0])
            p_phase = np.array([0, 0])
        
        p_amps.append(p_amp)
        p_phases.append(p_phase)
        
        # Remove gain from fine sweep
        zf_rmvd = remove_gain(ff[i], zf[i], p_amp, p_phase)
        zf_rmvds.append(zf_rmvd)
        
        # Fit nonlinear IQ
        try:
            p0, popt, perr, nrmse, _ = fit_nonlinear_iq(ff[i], zf_rmvd, plotq=False)
            popts.append(popt)
        except Exception as e:
            log(f"    IQ fit failed: {e}")
            popts.append(np.full(8, np.nan))
    
    log("Fitting complete!")
    
    # Create plot grid
    plot_layout = pg.GraphicsLayoutWidget()
    layout.addWidget(plot_layout)
    
    # Row 1: Gain amplitude and phase
    p_gain_amp = plot_layout.addPlot(row=0, col=0, title='Gain Amplitude')
    p_gain_amp.setLabel('bottom', 'Frequency', units='Hz')
    p_gain_amp.setLabel('left', 'Amplitude', units='dB')
    
    p_gain_phase = plot_layout.addPlot(row=0, col=1, title='Gain Phase')
    p_gain_phase.setLabel('bottom', 'Frequency', units='Hz')
    p_gain_phase.setLabel('left', 'Phase', units='rad')
    
    # Row 2: Fine sweep amplitude and IQ
    p_fine_amp = plot_layout.addPlot(row=1, col=0, title='Fine Sweep Amplitude (Gain Removed)')
    p_fine_amp.setLabel('bottom', 'Frequency', units='Hz')
    p_fine_amp.setLabel('left', 'Amplitude', units='dB')
    
    p_fine_iq = plot_layout.addPlot(row=1, col=1, title='Fine Sweep IQ (Gain Removed)')
    p_fine_iq.setLabel('bottom', 'I')
    p_fine_iq.setLabel('left', 'Q')
    p_fine_iq.setAspectLocked(True)
    
    # Plot all power levels
    cmap = pg.colormap.get('viridis')
    
    for i in range(n_powers):
        color = cmap.map(i / max(1, n_powers - 1))
        brush = pg.mkBrush(color)
        pen_color = pg.mkPen(color, width=2, style=_Qt.DotLine)
        
        # Gain plots - circular markers with fits
        p_gain_amp.plot(fg[i], 20 * np.log10(np.abs(zg[i])), 
                       pen=None, symbol='o', symbolSize=4, symbolBrush=brush, symbolPen=None)
        p_gain_phase.plot(fg[i], np.angle(zg[i]), 
                         pen=None, symbol='o', symbolSize=4, symbolBrush=brush, symbolPen=None)
        
        # Plot gain fits
        fg_fit = np.linspace(fg[i].min(), fg[i].max(), 200)
        # Amplitude fit: a0 + a1*f + a2*f^2 (in dB)
        amp_fit = np.polyval(p_amps[i], fg_fit)
        p_gain_amp.plot(fg_fit, amp_fit, pen=pen_color)
        # Phase fit: b0 + b1*f (in radians)
        phase_fit = np.polyval(p_phases[i], fg_fit)
        p_gain_phase.plot(fg_fit, phase_fit, pen=pen_color)
        
        # Fine sweep plots - circular markers only
        zf_r = zf_rmvds[i]
        p_fine_amp.plot(ff[i], 20*np.log10(np.abs(zf_r)), 
                       pen=None, symbol='o', symbolSize=4, symbolBrush=brush, symbolPen=None)
        p_fine_iq.plot(zf_r.real, zf_r.imag, 
                      pen=None, symbol='o', symbolSize=4, symbolBrush=brush, symbolPen=None)
        
        # Plot fits - dotted line with same color
        if not np.all(np.isnan(popts[i])):
            from citkid.res.funcs import nonlinear_iq
            ff_fit = np.linspace(ff[i].min(), ff[i].max(), 200)
            # nonlinear_iq expects specific numba types - pass params as individual args
            popt = popts[i]
            zf_fit = nonlinear_iq(ff_fit, popt[0], popt[1], popt[2], popt[3], 
                                 popt[4], popt[5], popt[6], popt[7], True)
            # Plot on IQ
            p_fine_iq.plot(zf_fit.real, zf_fit.imag, pen=pen_color)
            # Plot amplitude on fine amp plot
            p_fine_amp.plot(ff_fit, 20*np.log10(np.abs(zf_fit)), pen=pen_color)
    
    # Add info label at bottom
    info_label = QtWidgets.QLabel(f"Power levels: {ares.min():.1f} to {ares.max():.1f} dBm ({n_powers} levels)")
    layout.addWidget(info_label)
    
    win.resize(1200, 900)
    
    return win, app


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m citkid.vna.popt_minimal <zarr_path> [data_idx]")
        print("\nExample:")
        print("  python -m citkid.vna.popt_minimal /path/to/data.zarr 0")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    data_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    win, app = plot_power_sweep(zarr_path, data_idx)
    
    sys.exit(app.exec())
