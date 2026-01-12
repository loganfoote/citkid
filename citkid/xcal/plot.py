import numpy as np 
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
# use rasterized = True for noise 
from ..noise.psd import bin_psd

def plot_gain_fit(f, z, mask, p_amp, p_phase):
    """
    Plots the fit to gain amplitude and phase data.

    Parameters:
    f (np.array, float64, (M,)): full frequency data.
    dB (np.array, float64, (M,)): full amplitude data.
    mask (np.array, bool, (M,)): resonance mask.
    p_amp (np.array, float64): amplitude fit parameters.
    p_phase (np.array, float64): phase fit parameters.

    Returns:
    fig, axs (pyplot figure and axis): data and fit plot
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
    fig, axs = plt.subplots(1, 2, figsize = [6, 3], dpi = 72, layout = 'tight')
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
    axs[1].legend(framealpha = 1, loc = 'lower left')
    return fig, axs

def plot_s21(f, z, zt = None, fg = None, zg = None):
    """
    Plots complex S21 sweep and optional timestream data in IQ plane and as 
    |S21| vs frequency.
     
    Parameters:
    f (np.array, float64): fine sweep frequency data in Hz.
    z (np.array, complex128): fine sweep complex S21 data.
    fg (np.array, float64 or None): gain sweep frequency data in Hz, or None to 
        omit from plots.
    zg (np.array, complex128 or None): gain sweep complex S21 data, or None to 
        omit from plots.
    zt (np.array, complex128 or None): timestream complex S21 data, or None to 
        omit from plots.

    Returns:
    fig (pyplot.figure): plot figure.
    axs (list, pyplot.axis): pyplot axes.axs[0] is the |S21| vs frequency plot 
        and axs[1] is the IQ plot.
    """
    # Input validation
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    if f.shape != z.shape:
        raise ValueError('f and z must be the same shape')
    fmean = np.mean(f) 

    # Set up plots
    fig, axs = plt.subplots(1, 2, figsize = [6, 3], layout = 'tight', dpi = 72)
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
        axs[0].plot((fg - fmean) / 1e3, 20 * np.log10(np.abs(zg)), 'o', 
                    color = plt.cm.viridis(0.33), aa = False)
        axs[1].plot(zg.real, zg.imag, 'o', color = plt.cm.viridis(0.33), 
                    aa = False, label = 'gain sweep')
        
    # Plot fine sweep data
    axs[0].plot((f - fmean) / 1e3, 20 * np.log10(np.abs(z)), 'o', 
                color = plt.cm.viridis(0.), aa = False)
    axs[1].plot(z.real, z.imag, 'o', color = plt.cm.viridis(0.), aa = False,
                label = 'fine sweep')

    # Plot timestream data
    if zt is not None:
        zt = np.asarray(zt, dtype = np.complex128) 
        axs[1].plot(zt.real, zt.imag, '.', color = plt.cm.viridis(0.67), 
                    aa = False, rasterized = True, label = 'Noise')
    axs[1].legend(loc = 'center', framealpha = 1)
    return fig, axs

def plot_circfit(z, origin, radius, zt = None, mask = None):
    """
    Plots IQ data with a circular fit.

    Parameters:
    z (np.array): complex IQ data.
    origin (complex): circle origin.
    radius (float): circle radius.
    zt (np.array, complex128 or None): timestream complex S21 data, or None to 
        omit from plot.
    mask (np.array, bool or None): mask of z used for fitting. If None, all
        data points were used.

    Returns:
    fig, ax (pyplot figure and axis): data and fit plot.
    """
    # Input validation
    z = np.asarray(z, dtype = np.complex128) 
    if not isinstance(origin, (complex, np.complexfloating, 
                               np.floating, np.integer, float, int)):
        raise ValueError("origin must be complex.")
    if not isinstance(radius, (float, np.floating)):
        raise ValueError("radius must be float.") 

    # Create cut array based on mask
    if mask is not None:
        mask = np.asarray(mask, dtype = np.bool_)
        z_cut = z[mask]
    else:
        z_cut = None
    
    # Setup plot
    fig, ax = plt.subplots(figsize = (3, 3), layout = 'tight', dpi = 72)
    ax.set(xlabel = 'I', ylabel = 'Q')
    ax.set(aspect = 'equal', adjustable = 'datalim')
    # Plot data
    ax.plot(np.real(z), np.imag(z), 'o', color = plt.cm.viridis(0.), 
            aa = False, label = 'data')
    if z_cut is not None:
        ax.plot(np.real(z_cut), np.imag(z_cut), 'o', 
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
    ax.legend(loc = 'center', framealpha = 1)
    return fig, ax

def plot_sparper(f, spar, sper, nbins, fmin):
    """
    Plots binned parallel and perpendicular noise PSDs vs frequency.

    Parameters:
    f (np.array, float64): frequency data in Hz.
    spar (np.array, float64): spar data in dB.
    sper (np.array, float64): sper data in dB.
    nbins (int): number of bins for sparper histogram.
    fmin (float): minimum frequency for sparper histogram in Hz.

    Returns:
    fig, ax (pyplot figure and axis): sparper plot.
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
    fig, ax = plt.subplots(figsize = [5, 3], layout = 'tight', dpi = 72) 
    ax.set(ylabel = 'S (dBc/Hz)', xlabel = 'Frequency (Hz)')
    ax.set(xscale = 'log')

    # Plot PSDs
    ax.plot(f, spar, color = plt.cm.viridis(0.33), aa = False, label = 'PAR')
    ax.plot(f, sper, color = plt.cm.viridis(0.67), aa = False, label = 'PER')

    # Add legend and return figure and axis
    ax.legend(framealpha = 1)
    return fig, ax

def plot_xcal(thetaf, xf, zf_cent, xcal_mask, poly_x, thetat = None, 
              zt_cent = None):
    """
    Plots x vs theta calibration data and IQ data with fit overlayed. 

    Parameters:
    thetaf (np.array, float64): fine sweep theta data.
    xf (np.array, float64): fine sweep x data.
    zf_cent (np.array, complex128): fine sweep centered IQ data.
    xcal_mask (np.array, bool): mask of fine sweep data used for x calibration.
    poly_x (np.array, float64): polynomial fit parameters for x vs theta.
    thetat (np.array, float64 or None): timestream theta data, or None to omit
        from plot.
    zt_cent (np.array, complex128 or None): timestream centered IQ data, or 
        None to omit from plot.

    Returns:
    fig, axs (pyplot figure and axis): x vs theta and IQ data plot.
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

    # Setup plot
    fig, axs = plt.subplots(1, 2, figsize = [6, 3], layout = 'tight', dpi = 72)
    axs[0].set(xlabel = 'theta', ylabel = 'x (kHz / GHz)')
    axs[1].set(xlabel = 'I', ylabel = 'Q', 
               aspect = 'equal', adjustable = 'datalim')
    
    # Plot x vs theta cal data
    axs[0].plot(thetaf_cut, xf_cut * 1e6, 'o', color = plt.cm.viridis(0.33), 
                aa = False)
    tsamp = np.linspace(min(thetaf_cut), max(thetaf_cut), 60)
    axs[0].plot(tsamp, np.polyval(poly_x, tsamp) * 1e6, '--k', aa = False)
    if thetat is not None:
        axs[0].plot(thetat, np.polyval(poly_x, thetat) * 1e6, '.', 
                color = plt.cm.viridis(0.67), aa = False, rasterized = True)

    # Plot IQ data
    axs[1].plot(zf_cent.real, zf_cent.imag, 'o', color = plt.cm.viridis(0.), 
                aa = False, label = 'full fine sweep') 
    axs[1].plot(zf_cut.real, zf_cut.imag, 'o', color = plt.cm.viridis(0.33), 
                aa = False, label = 'fit fine sweep') 
    if zt_cent is not None:
        axs[1].plot(zt_cent.real, zt_cent.imag, '.', 
                    color = plt.cm.viridis(0.67), aa = False, 
                    rasterized = True, label = 'timestream')
    axs[1].plot([], [], '--k', label = 'fit')

    # Add legend and return figure and axis
    axs[1].legend(framealpha = 1)
    return fig, axs