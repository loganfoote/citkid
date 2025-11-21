import matplotlib.pyplot as plt
import numpy as np
from ..res.plot import plot_circle
from ..util import combine_figures_horizontally
from ..res.gain import remove_gain

def plot_cal(ffine, zfine, origin, radius, v, theta_range, theta_fine, p_amp, p_phase,
             p_x):
    """
    Plot theta and x calibration. Left plot is the IQ loop with noise, and right
    plot is the theta to x calibration, if on-resonance noise is provided.

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data
    origin (complex): center of the IQ circle
    radius (float): radius of the IQ circle
    v (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    theta_range (tuple): [lower, upper] range of theta over which the fit was
        performed
    theta_fine (np.array): fine scan theta data
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    p_x (np.array): x vs theta polynomial fit parameters

    Returns:
    fig (plt.figure): calibration plot figure
    """
    zfine_rmvd = remove_gain(ffine, zfine, p_amp, p_phase)
    color0 = plt.cm.viridis(0)
    color1 = plt.cm.viridis(0.9)
    fig, ax = plot_circle(zfine_rmvd, np.real(origin), np.imag(origin), radius)
    ax.plot([], [], '.r', label = 'data')
    ax.plot([], [], '-k', label = 'fit')


    fig2, ax2 = plt.subplots(figsize = [4, 4], dpi = 200)
    ax2.set_ylabel('x (Hz / MHz)')
    ax2.set_xlabel('Phase')

    r1 = v * radius * np.exp(-1j * max(theta_range)) + origin
    r2 = v * radius * np.exp(-1j * min(theta_range)) + origin
    ax.plot([np.real(origin), np.real(r1)],
            [np.imag(origin), np.imag(r1)], '--', color = color0)
    ax.plot([np.real(origin), np.real(r2)],
            [np.imag(origin), np.imag(r2)], '--', color = color0)
    # theta -> x calibration
    ix = np.argsort(theta_fine)
    theta_fine, ffine = theta_fine[ix], ffine[ix]

    theta_samp = np.linspace(theta_range[0], theta_range[1], 200)
    ax2.plot(theta_samp,
             (1 - np.polyval(p_x, theta_samp) / np.mean(ffine)) * 1e6, '-k')
    ix = (theta_fine <= theta_range[1]) & (theta_fine >= theta_range[0])
    ax2.plot(theta_fine, (1 - ffine / np.mean(ffine)) * 1e6, '.', color = 'r')
    ax2.axvline(theta_range[0], linestyle = '--', color = color0)
    ax2.axvline(theta_range[1], linestyle = '--', color = color0)

    ax.plot([], [], '--k', label = 'cal range')
    ax.legend(loc = 'center')
    fig = combine_figures_horizontally(fig, fig2)
    return fig
