import matplotlib.pyplot as plt
import numpy as np
from ..res.plot import plot_circle
from .psd import bin_psd
from ..util import combine_figures_horizontally

import warnings 

warnings.warn(
    "citkid.noise.plot is deprecated and will be removed in version 1.0.0",
    DeprecationWarning,
    stacklevel=2,
)

def plot_cal(ffine, zfine, popt_circle, fnoise, znoise, znoise_offres,
             theta_range, theta_fine, theta, poly, ixs):
    """
    Plot theta and x calibration. Left plot is the IQ loop with noise, and right
    plot is the theta to x calibration, if on-resonance noise is provided.

    Parameters:
    ffine (array-like): fine sweep frequency data in Hz
    zfine (array-like): fine sweep complex S21 data, with gain removed
    popt_circle (np.array): circle fit parameters
    fnoise (float): on-resonance noise tone frequency in Hz
    znoise (array-like or None): on-resonance complex noise data
    fnoise_offres (float): off-resonance noise tone frequency in Hz
    znoise_offres (array-like or None): off-resonance complex noise data
    theta (np.array): theta noise timestream
    theta_range (tuple): [lower, upper] range of theta over which the fit was
        performed
    theta_fine (np.array): fine sweep theta data
    poly (np.array): x vs theta polynomial fit parameters

    Returns:
    fig (plt.figure): calibration plot figure
    """
     
    color0 = plt.cm.viridis(0)
    color1 = plt.cm.viridis(0.9)
    fig, ax = plot_circle(zfine, *popt_circle)
    origin = popt_circle[0] + 1j * popt_circle[1]
    ax.plot([], [], '.r', label = 'data')
    ax.plot([], [], '-k', label = 'fit')

    if znoise is not None:
        ix0, ix1 = ixs
        fig2, ax2 = plt.subplots(figsize = [4, 4], dpi = 200)
        ax2.set_ylabel('x (Hz / MHz)')
        ax2.set_xlabel('Phase')
        nevery = int(len(znoise) / 10000)
        if nevery == 0:
            nevery = 1
        ax.plot(np.real(znoise[::nevery]), np.imag(znoise[::nevery]), '.',
                color = color0, markersize = 1, label = 'on-res noise')
        zn_med = np.mean(znoise - origin)
        r1 = zn_med * np.exp(1j * max(theta_range)) + origin
        r2 = zn_med * np.exp(1j * min(theta_range)) + origin
        ax.plot([np.real(origin), np.real(r1)],
                [np.imag(origin), np.imag(r1)], '--', color = color0)
        ax.plot([np.real(origin), np.real(r2)],
                [np.imag(origin), np.imag(r2)], '--', color = color0)
        # theta -> x calibration
        # ix = np.argsort(theta_fine)
        # theta_fine, ffine = theta_fine[ix], ffine[ix]
        # ix0 = np.where(theta_fine > min(theta))[0]
        # if len(ix0):
        #     ix0 = min(ix0) - 3
        #     if ix0 < 0:
        #         ix0 = 0
        # else:
        #     ix0 = 0
        # ix1 = np.where(theta_fine < max(theta))[0]
        # if len(ix1):
        #     ix1 = max(ix1) + 3
        #     if ix1 > len(theta_fine):
        #         ix1 = len(theta_fine)
        # else:
        #     ix1 = len(theta_fine)
        theta_samp = np.linspace(
            min(theta_fine[ix0:ix1]),
            max(theta_fine[ix0:ix1]),
            200,
        )
        ax2.plot(
            theta_samp,
            (1 - np.polyval(poly, theta_samp) / fnoise) * 1e6,
            '-k',
        )
        ax2.plot(
            theta_fine[ix0:ix1],
            (1 - ffine[ix0:ix1] / fnoise) * 1e6,
            '.',
            color = 'r',
        )
        ax2.plot(
            theta[::nevery],
            (1 - np.polyval(poly, theta[::nevery]) / fnoise) * 1e6,
            '.',
            color = color0,
            markersize = 5,
        )
        ax2.axvline(theta_range[0], linestyle = '--', color = color0)
        ax2.axvline(theta_range[1], linestyle = '--', color = color0)

    if znoise_offres is not None:
        nevery = int(len(znoise_offres) / 10000)
        if nevery == 0:
            nevery = 1
        ax.plot(np.real(znoise_offres[::nevery]),
                np.imag(znoise_offres[::nevery]), '.', markersize = 1,
                color = color1, label = 'off-res noise')

    ax.plot([], [], '--k', label = 'cal range')
    ax.legend(loc = 'center')
    if znoise is not None:
        fig = combine_figures_horizontally(fig, fig2)
    return fig

def plot_timestream(dt, theta, theta_clean, dt_offres, theta_offres, x,
                    cr_indices):
    """
    Plots noise timestreams. If theta is None, plots only the off-resonance
    theta timestream. If theta_offres is None, plots only the on-resonance
    theta timestream and the on-resonance x timestream, with and without
    deglitching and cosmic ray removal. If neither are None, plots on- and off-
    resonance timestreams.

    Parameters:
    dt (float or None): on-resonance timestream sample time in s
    theta (np.array or None): on-resonance theta timestream data
    theta_clean (np.array or None): on-resonance theta timestream data with
        cosmic rays removed and deglitched
    dt_offres (float or None): off-resonance timestream sample time in s
    theta_offres (np.array or None): off-resonance theta timestream data
    x (np.array): on-resonance x timestream data
    cr_indices (np.array): cosmic ray indices

    Returns:
    fig (plt.figure): timestream plot
    """
    color0 = plt.cm.viridis(0)
    color1 = plt.cm.viridis(0.9)
    fig = None
    nplots = 0
    if theta is not None:
        nplots += 2
    if theta_offres is not None:
        nplots += 1
    fig, axs = plt.subplots(nplots, 1, figsize = [8, nplots * 2],
                            layout = 'tight', dpi = 200)
    if nplots == 1:
        axs = [axs]
    if theta is not None:
        axs[0].set_ylabel('x (Hz / MHz)\non-res')
        axs[1].set_ylabel('Phase\non-res')
        axs[1].set_xlabel('Time (s)')

        time = np.linspace(0, len(theta) * dt, len(theta))
        axs[1].plot(time, theta, color = color0, label = 'raw data')
        axs[1].plot(
            time,
            theta_clean,
            color = color1,
            label = 'deglitched data',
        )
        axs[0].plot(time, x * 1e6, color = color0)
        axs[1].plot(time[cr_indices], theta[cr_indices], color = 'r',
                    marker = 'x', linestyle = '', label = 'removed peaks')
        axs[1].legend(loc = 'lower left')
    if theta_offres is not None:
        i = nplots - 1
        axs[i].set_ylabel('Phase\noff-res')
        axs[i].set_xlabel('Time (s)')
        time = np.linspace(0, len(theta_offres) * dt_offres, len(theta_offres))
        axs[i].plot(time, theta_offres, color = color0)
    return fig

def plot_psd(f_psd, spar, sper, sxx, f_psd_offres, spar_offres, sper_offres):
    """
    Plots on- and off-resonance psds. Produces one plot of perpendicular and
    parallel noise, and one plot of Sxx, if provided

    Parameters:
    f_psd (np.array or None): on-resonant frequency data in Hz
    spar  (np.array or None): on-resonance parallel noise in dBc
    sper  (np.array or None): on-resonance perpendicular noise in dBc
    sxx   (np.array or None): on-resonance Sxx in 1 / Hz
    f_psd_offres (np.array or None): off-resonant frequency data in Hz
    spar_offres  (np.array or None): off-resonance parallel noise in dBc
    sper_offres  (np.array or None): off-resonance perpendicular noise in dBc

    Returns:
    fig (plt.figure): plot of the PSDs
    """
    parcolor = plt.cm.viridis(0)
    percolor = plt.cm.viridis(0.9)
    if sxx is None:
        fig, ax0 = plt.subplots(figsize = [6, 3], dpi = 200)
        axs = [ax0]
    else:
        fig, [ax1, ax0] = plt.subplots(2, 1, figsize = [6, 6], sharex = True,
                                       dpi = 200)
        ax1.set_ylabel(r'$S_{xx}$ (1 / Hz)')
        ax1.set_yscale('log')
        axs = [ax0, ax1]
    for ax in axs:
        ax.set_xscale('log')
        ax.grid()
    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel(r'PSD (dBc / Hz)')
    # Need to bin before plotting
    if spar_offres is not None:
        fbin, sparbin, sperbin = bin_psd(f_psd_offres, [f_psd_offres,
                                         spar_offres, sper_offres], nbins = 500,
                                         fmin = f_psd_offres[1] * 10,
                                         filter_pt_n = 55)
        ax0.plot(fbin, sparbin, '--', color = parcolor, label = 'par off')
        ax0.plot(fbin, sperbin, '--', color = percolor, label = 'per off')
    if spar is not None:
        fbin, sparbin, sperbin, sxxbin = bin_psd(f_psd, [f_psd, spar,
                                                        sper, sxx], nbins = 500,
                                                 fmin = f_psd[1] * 10,
                                                 filter_pt_n = 55)
        ax0.plot(fbin, sparbin, '-', color = parcolor, label = 'par on')
        ax0.plot(fbin, sperbin, '-', color = percolor, label = 'per on')
        ax1.plot(fbin, sxxbin, '-', color = parcolor)
    ax0.legend()
    return fig
