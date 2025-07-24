import matplotlib.pyplot as plt
import numpy as np
from .funcs import *
from matplotlib.ticker import ScalarFormatter

################################################################################
############### Resonance shift from thermal QP density and TLS ################
################################################################################
def plot_f_vs_T(T, f, f_err, popt, p0, gamma):
    """
    Plots the fit and initial guess to f_vs_T.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    f_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    f0 = popt[0]
    fig, ax = setup_f_vs_T(f0)
    ax.set_title('f vs T, thermal QP + TLS shift')

    if f_err is None:
        ax.plot(T * 1e3, (f - f0) * 1e-3, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(T * 1e3, (f - f0) * 1e-3, yerr = f_err * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')
        ax.set_xlim(min(T) * 0.96e3, max(T) * 1.04e3)

    xsamp = np.geomspace(min(T), max(T), 200)
    ysamp = f_vs_T(xsamp, *popt, gamma = gamma)
    ax.plot(xsamp * 1e3, (ysamp - f0) * 1e-3, '--r', label = 'Fit')
    ylim = ax.get_ylim()
    ysamp = f_vs_T(xsamp, *p0, gamma = gamma)
    ax.plot(xsamp * 1e3, (ysamp - f0) * 1e-3, ':k', label = 'Guess')
    ax.set_ylim(ylim)
    ax.legend(framealpha = 1)
    return fig, ax

def plot_Q_vs_T(T, Q, Q_err, popt, p0, gamma):
    """
    Plots the fit and initial guess to Q_vs_T.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    Q_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    f0 = popt[0]
    fig, ax = setup_Q_vs_T()
    ax.set_title('Q vs T, thermal QP + TLS shift')

    if Q_err is None:
        ax.plot(T * 1e3, Q * 1e-3, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(T * 1e3, Q * 1e-3, yerr = Q_err * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')
        ax.set_xlim(min(T) * 0.96e3, max(T) * 1.04e3)

    xsamp = np.geomspace(min(T), max(T), 200)
    ysamp = Q_vs_T(xsamp, *popt, gamma = gamma)
    ax.plot(xsamp * 1e3, ysamp * 1e-3, '--r', label = 'Fit')
    ylim = ax.get_ylim()
    ysamp = Q_vs_T(xsamp, *p0, gamma = gamma)
    ax.plot(xsamp * 1e3, ysamp * 1e-3, ':k', label = 'Guess')
    ax.set_ylim(ylim)
    ax.legend(framealpha = 1)
    return fig, ax

################################################################################
################### Resonance shift from thermal QP density ####################
################################################################################
def plot_f_vs_T_qp(T, f, f_err, popt, p0, gamma):
    """
    Plots the fit and initial guess to f_vs_T_qp.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    f_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    f0 = popt[0]
    fig, ax = setup_f_vs_T(f0)
    ax.set_title('f vs T, thermal QP shift')

    if f_err is None:
        ax.plot(T * 1e3, (f - f0) * 1e-3, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(T * 1e3, (f - f0) * 1e-3, yerr = f_err * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')
        ax.set_xlim(min(T) * 0.96e3, max(T) * 1.04e3)

    xsamp = np.geomspace(min(T), max(T), 200)
    ysamp = f_vs_T_qp(xsamp, *popt, gamma = gamma)
    ax.plot(xsamp * 1e3, (ysamp - f0) * 1e-3, '--r', label = 'Fit')
    ylim = ax.get_ylim()
    ysamp = f_vs_T_qp(xsamp, *p0, gamma = gamma)
    ax.plot(xsamp * 1e3, (ysamp - f0) * 1e-3, ':k', label = 'Guess')
    ax.set_ylim(ylim)
    ax.legend(framealpha = 1)
    return fig, ax

def plot_Q_vs_T_qp(T, Q, Q_err, popt, p0, gamma):
    """
    Plots the fit and initial guess to Q_vs_T_qp.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    Q_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    fig, ax = setup_Q_vs_T()
    ax.set_title('Q vs T, thermal QP shift')

    if Q_err is None:
        ax.plot(T * 1e3, Q * 1e-3, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(T * 1e3, Q * 1e-3, yerr = Q_err * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')
        ax.set_xlim(min(T) * 0.96e3, max(T) * 1.04e3)

    xsamp = np.geomspace(min(T), max(T), 200)
    ysamp = Q_vs_T_qp(xsamp, *popt, gamma = gamma)
    ax.plot(xsamp * 1e3, ysamp * 1e-3, '--r', label = 'Fit')
    ylim = ax.get_ylim()
    ysamp = Q_vs_T_qp(xsamp, *p0, gamma = gamma)
    ax.plot(xsamp * 1e3, ysamp * 1e-3, ':k', label = 'Guess')
    ax.set_ylim(ylim)
    ax.legend(framealpha = 1)
    return fig, ax

################################################################################
########################### Resonance shift from TLS ###########################
################################################################################
def plot_f_vs_T_tls(T, f, f_err, popt, p0):
    """
    Plots the fit and initial guess to f_vs_T_tls.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    f_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    f0 = popt[0]
    fig, ax = setup_f_vs_T(f0)
    ax.set_title('f vs T, TLS shift')

    if f_err is None:
        ax.plot(T * 1e3, (f - f0) * 1e-3, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(T * 1e3, (f - f0) * 1e-3, yerr = f_err * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')
        ax.set_xlim(min(T) * 0.96e3, max(T) * 1.04e3)

    xsamp = np.geomspace(min(T), max(T), 200)
    ysamp = f_vs_T_tls(xsamp, *popt)
    ax.plot(xsamp * 1e3, (ysamp - f0) * 1e-3, '--r', label = 'Fit')
    ylim = ax.get_ylim()
    ysamp = f_vs_T_tls(xsamp, *p0)
    ax.plot(xsamp * 1e3, (ysamp - f0) * 1e-3, ':k', label = 'Guess')
    ax.set_ylim(ylim)
    ax.legend(framealpha = 1)
    return fig, ax

def plot_Q_vs_T_tls(T, Q, Q_err, popt, p0):
    """
    Plots the fit and initial guess to Q_vs_T_tls.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    Q_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    fig, ax = setup_Q_vs_T()
    ax.set_title('Q vs T, TLS shift')

    if Q_err is None:
        ax.plot(T * 1e3, Q * 1e-3, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(T * 1e3, Q * 1e-3, yerr = Q_err * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')
        ax.set_xlim(min(T) * 0.96e3, max(T) * 1.04e3)

    xsamp = np.geomspace(min(T), max(T), 200)
    ysamp = Q_vs_T_tls(xsamp, *popt)
    ax.plot(xsamp * 1e3, ysamp * 1e-3, '--r', label = 'Fit')
    ylim = ax.get_ylim()
    ysamp = Q_vs_T_tls(xsamp, *p0)
    ax.plot(xsamp * 1e3, ysamp * 1e-3, ':k', label = 'Guess')
    ax.set_ylim(ylim)
    ax.legend(framealpha = 1)
    return fig, ax

################################################################################
################################### Utility ####################################
################################################################################
def setup_f_vs_T(f0):
    """
    Set up an f vs T plot.

    Parameters:
    f0 (float): frequency at T = 0 K in Hz

    Returns:
    fig, ax (pyplot figure and axis): formatted figure and axis
    """
    fig, ax = plt.subplots(figsize = [5, 4], dpi = 200, layout = 'tight')
    lbl = r'$(f_r - ' + f'{round(f0 * 1e-6, 3)}' + r'\;\mathrm{MHz})$ (kHz)'
    ax.set_ylabel(lbl)
    ax.set_xlabel(r'Temperature (mK)')
    ax.set_xscale('log')
    log_without_scientific(ax.xaxis)
    return fig, ax

def setup_Q_vs_T():
    """
    Set up a Q vs T plot.

    Returns:
    fig, ax (pyplot figure and axis): formatted figure and axis
    """
    fig, ax = plt.subplots(figsize = [5, 4], dpi = 200, layout = 'tight')
    ax.set_ylabel(r'$Q_r/1,000$')
    ax.set_xlabel(r'Temperature (mK)')
    ax.set_xscale('log')
    log_without_scientific(ax.xaxis)
    return fig, ax

def log_without_scientific(axis):
    """
    Turns off scientific notation for a log axis

    Parameters:
    axis (matplotlib.axis.Xaxis or matplotlib.axis.Yaxis): axis with scaling
    set to 'log'
    """
    axis.set_major_formatter(ScalarFormatter())
    axis.get_major_formatter().set_scientific(False)
    axis.set_minor_formatter(ScalarFormatter())
    axis.get_minor_formatter().set_scientific(False)
