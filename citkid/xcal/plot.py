import numpy as np 
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
# use rasterized = True for noise 

def plot_circle(z, A, B, R):
    """
    Plots IQ data with a circular fit

    Parameters:
    z (np.array): complex IQ data
    A, B (float, float): circle origin
    R (float): circle radius

    Returns:
    fig, ax (pyplot figure and axis): data and fit plot
    """
    z = np.asarray(z, dtype = np.complex128) 
    fig, ax = plt.subplots(figsize = (3, 3), dpi = 72)
    ax.set(xlabel = 'I', ylabel = 'Q')
    ax.set_aspect('equal', adjustable = 'datalim')
    ax.plot(np.real(z), np.imag(z), '.', color = plt.cm.viridis(0.5), 
            aa = False)
    cir = plt.Circle((A, B), R, color = 'k', fill = False, aa = False)
    ax.add_patch(cir)
    return fig, ax

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
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    mask = np.asarray(mask, dtype = bool)
    assert f.shape == z.shape == mask.shape
    f_cut, z_cut =  f[mask], z[mask]
    dB = 20 * np.log10(np.abs(z))
    dB_cut = 20 * np.log10(np.abs(z_cut))
    phase_cut = np.angle(z_cut)
    p_amp = np.asarray(p_amp, dtype = np.float64)
    p_phase = np.asarray(p_phase, dtype = np.float64)

    fmean = np.mean(f)

    fig, axs = plt.subplots(1, 2, figsize = [6, 3], dpi = 72, layout = 'tight')
    xlbl = f'(f - {fmean / 1e6:.04f} MHz) (kHz)'
    axs[1].set(ylabel = 'Phase', xlabel = xlbl)
    axs[0].set(ylabel = '|S21| (dB)', xlabel = xlbl)

    c0, c1 = plt.cm.viridis(0.33), plt.cm.viridis(0.67)
    axs[0].plot((f - fmean) * 1e-3, dB, '.', color = c1, 
                label = 'Raw data', aa = False)
    axs[0].plot((f_cut - fmean) * 1e-3, dB_cut, '.', color = c0, 
                label = 'Fitted data', aa = False)
    fsamp = np.linspace(np.min(f),np.max(f), 100)
    if ~np.any(np.isnan(p_amp)):
        axs[0].plot((fsamp - fmean) * 1e-3, np.polyval(p_amp, fsamp), '--k', 
                    label = 'Fit', aa = False)

    axs[1].plot([], [], '.', color = c1, label = 'Raw data', aa = False)
    axs[1].plot((f_cut - fmean) * 1e-3, phase_cut, '.', color = c0, 
                label = 'Fitted data', aa = False)
    if ~np.any(np.isnan(p_phase)):
        axs[1].plot((fsamp - fmean) * 1e-3, np.polyval(p_phase, fsamp), '--k', 
                    label = 'Fit', aa = False)

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
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    assert f.shape == z.shape, 'f and z must be the same shape'
    fmean = np.mean(f) 

    # Set up plots
    fig, axs = plt.subplots(1, 2, figsize = [6, 4], layout = 'tight', dpi = 72)
    xlbl = f'(f - {fmean / 1e6:.02f} MHz) (kHz)'
    axs[0].set(xlabel = xlbl, ylabel = r'$S_{21}$ (dB)')
    axs[1].set(xlabel = 'I', ylabel = 'Q', aspect = 'equal')

    # Plot gain sweep data
    if fg is not None:
        fg = np.asarray(fg, dtype = np.float64)
        zg = np.asarray(zg, dtype = np.complex128)
        assert fg.shape == zg.shape, 'fg and zg must be the same shape'
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
