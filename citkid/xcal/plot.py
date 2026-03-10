import numpy as np 
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
# use rasterized = True for noise 
from ..signal.psd import bin_psd
from ..res.funcs import nonlinear_iq

def plot_gain_fit(f, z, mask, p_amp, p_phase):
    """
    Plots the fit to gain amplitude and phase data.

    Parameters:
    f (np.array, float64, (M,)): Full frequency data in Hz.
    z (np.array, complex128, (M,)): Full complex S21 data.
    mask (np.array, bool, (M,)): Resonance mask for fitted points.
    p_amp (np.array, float64): Amplitude fit parameters.
    p_phase (np.array, float64): Phase fit parameters.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plots.
    axs (np.ndarray, matplotlib.axes.Axes): Array of axes for amplitude and
        phase plots.
    """
    # Input validation 
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    mask = np.asarray(mask, dtype = bool)
    if not f.shape == z.shape == mask.shape:
        raise ValueError("f, z, and mask must have the same shape.")
    p_amp = np.asarray(p_amp, dtype = np.float64)
    p_phase = np.asarray(p_phase, dtype = np.float64)
    
    # Create cut arrays
    f_cut, z_cut =  f[mask], z[mask]
    dB = 20 * np.log10(np.abs(z))
    dB_cut = 20 * np.log10(np.abs(z_cut))
    phase_cut = np.angle(z_cut)
    fmean = np.mean(f)

    # Setup plot
    fig, axs = plt.subplots(1, 2, figsize = [8, 3], dpi = 72, layout = 'tight')
    xlbl = f'(f - {fmean / 1e6:.04f} MHz) (kHz)'
    axs[1].set(ylabel = 'Phase', xlabel = xlbl)
    axs[0].set(ylabel = '|S21| (dB)', xlabel = xlbl)

    # Plot amplitude data
    c0, c1 = plt.cm.viridis(0.33), plt.cm.viridis(0.67)
    axs[0].plot((f - fmean) * 1e-3, dB, '.', color = c1, 
                label = 'Raw data', aa = False)
    axs[0].plot((f_cut - fmean) * 1e-3, dB_cut, '.', color = c0, 
                label = 'Fitted data', aa = False)
    fsamp = np.linspace(np.min(f),np.max(f), 100)
    if ~np.any(np.isnan(p_amp)):
        axs[0].plot((fsamp - fmean) * 1e-3, np.polyval(p_amp, fsamp), '--k', 
                    label = 'Fit', aa = False)

    # Plot phase data
    axs[1].plot([], [], '.', color = c1, label = 'Raw data', aa = False)
    axs[1].plot((f_cut - fmean) * 1e-3, phase_cut, '.', color = c0, 
                label = 'Fitted data', aa = False)
    if ~np.any(np.isnan(p_phase)):
        axs[1].plot((fsamp - fmean) * 1e-3, np.polyval(p_phase, fsamp), '--k', 
                    label = 'Fit', aa = False)

    # Add legend and return figure and axes
    axs[1].legend(
        framealpha = 1, 
        loc = [1.01, 0.]
        )
    return fig, axs

def plot_s21(f, z, zt = None, fg = None, zg = None):
    """
    Plots complex S21 sweep and optional timestream data in IQ plane and as 
    |S21| vs frequency.
     
    Parameters:
    f (np.array, float64): Fine sweep frequency data in Hz.
    z (np.array, complex128): Fine sweep complex S21 data.
    fg (np.array, float64 or None): Gain sweep frequency data in Hz, or None
        to omit from plots.
    zg (np.array, complex128 or None): Gain sweep complex S21 data, or None
        to omit from plots.
    zt (np.array, complex128 or None): Timestream complex S21 data, or None
        to omit from plots.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plots.
    axs (np.ndarray, matplotlib.axes.Axes): axs[0] is the |S21| vs frequency
        plot and axs[1] is the IQ plot.
    """
    # Input validation
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    if f.shape != z.shape:
        raise ValueError('f and z must be the same shape')
    fmean = np.mean(f) 

    # Set up plots
    fig, axs = plt.subplots(1, 2, figsize = [8, 3], layout = 'tight', dpi = 72)
    xlbl = f'(f - {fmean / 1e6:.02f} MHz) (kHz)'
    axs[0].set(xlabel = xlbl, ylabel = r'$S_{21}$ (dB)')
    axs[1].set(xlabel = 'I', ylabel = 'Q', aspect = 'equal', 
               adjustable = 'datalim')

    # Plot gain sweep data
    if fg is not None:
        fg = np.asarray(fg, dtype = np.float64)
        zg = np.asarray(zg, dtype = np.complex128)
        if fg.shape != zg.shape:
            raise ValueError('fg and zg must be the same shape')
        axs[0].plot((fg - fmean) / 1e3, 20 * np.log10(np.abs(zg)), '.', 
                    color = plt.cm.viridis(0.33), aa = False)
        axs[1].plot(zg.real, zg.imag, '.', color = plt.cm.viridis(0.33), 
                    aa = False, label = 'gain sweep')
        
    # Plot fine sweep data
    axs[0].plot((f - fmean) / 1e3, 20 * np.log10(np.abs(z)), '.', 
                color = plt.cm.viridis(0.), aa = False)
    axs[1].plot(z.real, z.imag, '.', color = plt.cm.viridis(0.), aa = False,
                label = 'fine sweep')

    # Plot timestream data
    if zt is not None:
        zt = np.asarray(zt, dtype = np.complex128) 
        axs[1].plot(zt.real, zt.imag, '.', color = plt.cm.viridis(0.67), 
                    aa = False, rasterized = True, label = 'Noise')
    axs[1].legend(
        framealpha = 1, 
        loc = [1.01, 0.]
        )
    return fig, axs

def plot_circfit(z, origin, radius, zt = None, mask = None):
    """
    Plots IQ data with a circular fit.

    Parameters:
    z (np.array, complex128): Complex IQ data.
    origin (complex): Circle origin.
    radius (float): Circle radius.
    zt (np.array, complex128 or None): Timestream complex S21 data, or None to
        omit from plot.
    mask (np.array, bool or None): Mask of z used for fitting. If None, all
        data points are used.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plot.
    ax (matplotlib.axes.Axes): Axis for the IQ plot.
    """
    # Input validation
    z = np.asarray(z, dtype = np.complex128) 
    origin = complex(origin)
    radius = float(radius) 

    # Create cut array based on mask
    if mask is not None:
        mask = np.asarray(mask, dtype = np.bool_)
        z_cut = z[mask]
    else:
        z_cut = None
    
    # Setup plot
    fig, ax = plt.subplots(figsize = (5, 3), layout = 'tight', dpi = 72)
    ax.set(xlabel = 'I', ylabel = 'Q')
    ax.set(aspect = 'equal', adjustable = 'datalim')
    # Plot data
    ax.plot(np.real(z), np.imag(z), '.', color = plt.cm.viridis(0.), 
            aa = False, label = 'data')
    if z_cut is not None:
        ax.plot(np.real(z_cut), np.imag(z_cut), '.', 
                color = plt.cm.viridis(0.33), aa = False, label = 'fit data')
        
    # Plot circle fit
    cir = plt.Circle((origin.real, origin.imag), radius, color = 'k', 
                     fill = False, aa = False)
    ax.add_patch(cir)
    ax.plot([], [], '-k', label = 'fit')

    # Plot timestream data
    if zt is not None:
        zt = np.asarray(zt, dtype = np.complex128) 
        ax.plot(zt.real, zt.imag, '.', color = plt.cm.viridis(0.67), 
                aa = False, rasterized = True, label = 'Noise')
        
    # Add legend and return figure and axis
    ax.legend(
        framealpha = 1, 
        loc = [1.01, 0.]
        )
    return fig, ax

def plot_sparper(f, spar, sper, nbins, fmin):
    """
    Plots binned parallel and perpendicular noise PSDs vs frequency.

    Parameters:
    f (np.array, float64): Frequency data in Hz.
    spar (np.array, float64): Parallel PSD in dBc/Hz.
    sper (np.array, float64): Perpendicular PSD in dBc/Hz.
    nbins (int): Number of bins for the spar/sper histogram.
    fmin (float): Minimum frequency for binning in Hz.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plot.
    ax (matplotlib.axes.Axes): Axis for the PSD plot.
    """
    # Input validation
    f = np.asarray(f, dtype = np.float64)
    spar = np.asarray(spar, dtype = np.float64)
    sper = np.asarray(sper, dtype = np.float64)
    if not f.shape == spar.shape == sper.shape:
        raise ValueError("f, spar, and sper must be the same shape.") 
    if not all(f[1:] - f[:-1] >= 0):
        raise ValueError("f must be sorted in ascending order.")
    
    # Bin PSDs 
    f, spar, sper = bin_psd(f, [f, spar, sper], nbins = nbins, fmin = fmin)

    # Setup plot
    fig, ax = plt.subplots(figsize = [7, 3], layout = 'tight', dpi = 72) 
    ax.set(ylabel = 'S (dBc/Hz)', xlabel = 'Frequency (Hz)')
    ax.set(xscale = 'log')

    # Plot PSDs
    ax.plot(f, spar, color = plt.cm.viridis(0.33), aa = False, label = 'PAR')
    ax.plot(f, sper, color = plt.cm.viridis(0.67), aa = False, label = 'PER')

    # Add legend and return figure and axis
    ax.legend(
        framealpha = 1, 
        loc = [1.01, 0.]
        )
    return fig, ax

def plot_xcal(thetaf, xf, zf_cent, xcal_mask, poly_x, thetat = None, 
              zt_cent = None, std_cutoff = None):
    """
    Plots x vs theta calibration data and IQ data with fit overlayed. 

    Parameters:
    thetaf (np.array, float64): Fine sweep theta data.
    xf (np.array, float64): Fine sweep x data.
    zf_cent (np.array, complex128): Fine sweep centered IQ data.
    xcal_mask (np.array, bool): Mask of fine sweep data used for x calibration.
    poly_x (np.array, float64): Polynomial fit parameters for x vs theta.
    thetat (np.array, float64 or None): Timestream theta data, or None to omit
        from plot.
    zt_cent (np.array, complex128 or None): Timestream centered IQ data, or
        None to omit from plot.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plots.
    axs (np.ndarray, matplotlib.axes.Axes): Axes for x-vs-theta and IQ plots.
    """
    # Input validation
    thetaf = np.asarray(thetaf, dtype = np.float64)
    xf = np.asarray(xf, dtype = np.float64)
    xcal_mask = np.asarray(xcal_mask, dtype = np.bool_)
    if thetat is not None:
        thetat = np.asarray(thetat, dtype = np.float64)
        if not thetat.ndim == 1:
            raise ValueError("thetat must be 1-dimensional")
    poly_x = np.asarray(poly_x, dtype = np.float64)
    zf_cent = np.asarray(zf_cent, dtype = np.complex128)
    if zt_cent is not None:
        zt_cent = np.asarray(zt_cent, dtype = np.complex128)
    if thetaf.shape != xf.shape:
        raise ValueError("thetaf and xf must have the same shape.")
    if not np.all((xcal_mask >= 0) & (xcal_mask < len(thetaf))):
        raise ValueError("xcal_mask must be within the range of thetaf.")
    if poly_x.ndim != 1:
        raise ValueError("poly_x must be 1-dimensional.")
    if zf_cent.shape != thetaf.shape:
        raise ValueError("zf_cent must have the same shape as thetaf.")
   
    # Cut fine sweep data for fit
    thetaf_cut, xf_cut = thetaf[xcal_mask], xf[xcal_mask]
    zf_cut = zf_cent[xcal_mask]

    # Create std cutoff mask for timestream data
    if thetat is not None and std_cutoff is not None:
        # determine signal cutoff
        theta_t_std = np.std(thetat)
        theta_t_mean = np.mean(thetat)
        std_mask = np.abs(thetat - theta_t_mean) < std_cutoff * theta_t_std
    else:
        std_mask = np.ones_like(thetat, dtype = bool)

    # Setup plot
    fig, axs = plt.subplots(1, 2, figsize = [8, 3], layout = 'tight', dpi = 72)
    axs[0].set(xlabel = 'theta', ylabel = 'x (kHz / GHz)')
    axs[1].set(xlabel = 'I', ylabel = 'Q', 
               aspect = 'equal', adjustable = 'datalim')
    
    # Plot x vs theta cal data
    axs[0].plot(thetaf_cut, xf_cut * 1e6, '.', color = plt.cm.viridis(0.33), 
                aa = False)
    tsamp = np.linspace(min(thetaf_cut), max(thetaf_cut), 60)
    axs[0].plot(tsamp, np.polyval(poly_x, tsamp) * 1e6, '--k', aa = False)
    if thetat is not None:
        axs[0].plot(
            thetat[std_mask], 
            np.polyval(poly_x, thetat[std_mask]) * 1e6, 
            '.', 
            color = plt.cm.viridis(0.67), 
            aa = False, rasterized = True
            )
        axs[0].plot(
            thetat[~std_mask], 
            np.polyval(poly_x, thetat[~std_mask]) * 1e6, 
            '.', 
            color = plt.cm.viridis(1.), 
            aa = False, rasterized = True
            )

    # Plot IQ data
    axs[1].plot(zf_cent.real, zf_cent.imag, '.', color = plt.cm.viridis(0.), 
                aa = False, label = 'full fine sweep') 
    axs[1].plot(zf_cut.real, zf_cut.imag, '.', color = plt.cm.viridis(0.33), 
                aa = False, label = 'fit fine sweep') 
    if zt_cent is not None: 
        
        axs[1].plot(zt_cent[std_mask].real, zt_cent[std_mask].imag, '.', 
                    color = plt.cm.viridis(0.67), aa = False, 
                    rasterized = True, label = 'timestream <= cutoff')
        axs[1].plot(zt_cent[~std_mask].real, zt_cent[~std_mask].imag, '.', 
                    color = plt.cm.viridis(1.), aa = False, 
                    rasterized = True, label = 'timestream > cutoff')
    axs[1].plot([], [], '--k', label = 'fit')

    # Add legend and return figure and axis
    axs[1].legend(
        framealpha = 1,
        loc = [1.01, 0.]
        )
    return fig, axs

def plot_nonlinear_iq_fit(ff, zf_rmv, popt, mask):
    """
    Plots nonlinear IQ fit to fine sweep data, in IQ space and as |S21| vs
    frequency.
    
    Parameters:
    ff (np.array, float64): Fine sweep frequency data in Hz.
    zf_rmv (np.array, complex128): Fine sweep complex S21 data with gain removed.
    popt (list): Fit parameters.
    mask (np.array, bool): Mask of fine sweep data used for fitting.
    
    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plots.
    axs (np.ndarray, matplotlib.axes.Axes): Axes for |S21| vs frequency and IQ
        plots.
    """
    # Setup plot
    fig, axs = plt.subplots(1, 2, figsize = [9, 3], layout = 'tight', dpi = 72)
    axs[0].set(aspect = 'equal', adjustable = 'datalim')
    axs[0].set(ylabel = 'Q', xlabel = 'I') 
    axs[1].set(xlabel = f'(f - {round(popt[0] / 1e9, 4)} GHz) (kHz)', 
            ylabel = r'$S_{21}$ (dB)')

    # Plot data
    axs[0].plot(zf_rmv[mask].real, zf_rmv[mask].imag, '.', 
                color = plt.cm.viridis(0.), aa = False, label = 'data')
    axs[1].plot((ff - popt[0]) * 1e-3, 20 * np.log10(np.abs(zf_rmv[mask])), '.',
                color = plt.cm.viridis(0.), aa = False)

    # Plot fit
    fsamp = np.linspace(ff[0], ff[-1], 500) 
    zsamp = nonlinear_iq(fsamp, *popt, True) 
    axs[0].plot(zsamp.real, zsamp.imag, '--', label = 'fit',
                color = plt.cm.viridis(0.5), aa = False)
    axs[1].plot((fsamp - popt[0]) * 1e-3, 20 * np.log10(np.abs(zsamp)), '--',
                color = plt.cm.viridis(0.5), aa = False)

    # Add legend and return figure and axes
    axs[0].legend(
        framealpha = 1, 
        loc = [1.01, 0.]
        )
    return fig, axs

################################################################################
########################## Default Plotting Functions ##########################
################################################################################
default_plot_funcs = {
    'raw_data': (
        'Raw Data', plot_s21, ['ff', 'zf', 'zt', 'fg', 'zg'], {}
        ),
    'gain_fit': (
        'Gain Fit', plot_gain_fit, 
        ['fg', 'zg', 'gain_mask', 'p_amp', 'p_phase'], {}
        ),
    's21_rmv': (
        'Gain Removed Fine Sweep', plot_s21, 
        ['ff', 'zf_rmv', 'zt_rmv'], {}
        ),
    'circfit': (
        'Circle Fit', plot_circfit, 
        ['zf_rmv', 'circ_origin', 'circ_radius', 'zt_rmv', 'circ_mask'], {}
        ),
    'sparper': (
        'Par/Per Noise PSDs', plot_sparper, 
        ['sparper_freq', 'spar', 'sper'], {'nbins': 100, 'fmin': 0.1}
        ),
    'xcal': (
        r'$x$ Calibration', plot_xcal, 
        ['thetaf', 'xf', 'zf_cent', 'xcal_mask', 'poly_x', 
         'thetat', 'zt_cent', 'xcal_std_cutoff'], {}
        )
}