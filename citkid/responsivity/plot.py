import matplotlib.pyplot as plt
import numpy as np
from .funcs import responsivity_int

def plot_responsivity_int(power, x, x_err, popt, p0):
    """
    Plots the fit and initial guess to responsivity_int

    Parameters:
    power (array-like): array of blackbody powers in W
    x (array-like): array of fractional frequency shifts in Hz / Hz
    x_err (None or array-like): error bars for the plot, or None to plot without
        error bars
    p0 (list): initial guess parameters [R0_guess, P0_guess, c_guess]
    popt (list): fit parameters [R0, P0, c]

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    fig, ax = plt.subplots(figsize = [3, 2.8], dpi = 200, layout = 'tight')
    ax.set_ylabel(r'$df / f$ (kHz / GHz)')
    ax.set_xlabel(r'Power (W)')
    ax.set_xscale('log')
    if x_err is None:
        ax.plot(power, x * 1e6, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(power, x * 1e6, yerr = x_err * 1e6, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')

    pow = np.sort(power)
    pow0 = pow[0] if pow[0] != 0 else pow[1]
    psamp = np.geomspace(pow0, max(power), 200)
    ysamp = responsivity_int(psamp, *popt)
    ax.plot(psamp, ysamp * 1e6, '--r', label = 'Fit')
    ysamp = responsivity_int(psamp, *p0)
    ax.plot(psamp, ysamp * 1e6, ':k', label = 'Guess')
    ax.legend()
    return fig, ax
