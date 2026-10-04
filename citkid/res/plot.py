import matplotlib.pyplot as plt
import numpy as np
from .funcs import nonlinear_iq

# Fixed figure geometry (inches). Axes are placed directly, without a layout
# engine, which takes most of matplotlib's time for small plots. The legend
# sits in the right margin.
FIG_W, FIG_H = 8.0, 3.2
_LEFT, _RIGHT, _BOTTOM, _TOP, _GAP = 0.8, 1.2, 0.55, 0.2, 0.85
DPI = 72
# Number of frequencies at which fit and guess curves are drawn
NSAMP = 500

def plot_nonlinear_iq(f, z, popt, p0, downward = True, plot_guess = False):
    """
    Plot IQ data and the nonlinear IQ fit, in IQ space and as |S21| vs
    frequency.

    Drawn for speed, like the ``pipeline_v2`` quick plots: the axes have a
    fixed layout (no layout engine), the figure is 72 dpi, lines are not
    antialiased, and the fit is drawn at 500 frequencies. Frequencies are
    centered on the fitted resonance frequency, or on the mean frequency if
    the fit failed. Curves with non-finite parameters are not drawn.

    Parameters:
    f (np.array): Frequency data in Hz.
    z (np.array): Complex IQ data.
    popt (list): Fit parameters (fr, Qr, amp, phi, a, i0, q0, tau).
    p0 (list): Initial guess parameters, in the same order as popt.
    downward (bool): If True, plots the equation for a downward sweep. If
        False, plots for an upward sweep.
    plot_guess (bool): If True, also plot the guess curve.

    Returns:
    fig (matplotlib.figure.Figure): the figure.
    axs (np.ndarray of matplotlib.axes.Axes): IQ axes and |S21| vs frequency
        axes.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    popt = np.asarray(popt, dtype = np.float64)
    p0 = np.asarray(p0, dtype = np.float64)
    f0 = popt[0] if np.isfinite(popt[0]) else np.mean(f)

    # Setup figure with two axes at fixed positions
    fig = plt.figure(figsize = (FIG_W, FIG_H), dpi = DPI)
    width = (FIG_W - _LEFT - _RIGHT - _GAP) / 2
    height = FIG_H - _BOTTOM - _TOP
    axs = np.array([
        fig.add_axes([(_LEFT + i * (width + _GAP)) / FIG_W, _BOTTOM / FIG_H,
                      width / FIG_W, height / FIG_H])
        for i in range(2)
    ], dtype = object)
    axs[0].set(aspect = 'equal', adjustable = 'datalim', xlabel = 'I',
               ylabel = 'Q')
    axs[1].set(xlabel = f'(f - {round(f0 / 1e9, 4)} GHz) (kHz)',
               ylabel = r'$S_{21}$ (dB)')

    # Plot data
    color = plt.cm.viridis(0.)
    axs[0].plot(z.real, z.imag, '.', color = color, ms = 3, aa = False,
                label = 'data')
    axs[1].plot((f - f0) * 1e-3, 20 * np.log10(np.abs(z)), '.',
                color = color, ms = 3, aa = False)

    # Plot fit, and guess if requested
    fsamp = np.linspace(np.min(f), np.max(f), NSAMP)
    curves = [(popt, 'fit', plt.cm.viridis(0.5))]
    if plot_guess:
        curves.append((p0, 'guess', 'k'))
    for params, label, curve_color in curves:
        if not np.all(np.isfinite(params)):
            continue
        zsamp = nonlinear_iq(fsamp, *params, downward)
        axs[0].plot(zsamp.real, zsamp.imag, '--', color = curve_color,
                    aa = False, label = label)
        axs[1].plot((fsamp - f0) * 1e-3, 20 * np.log10(np.abs(zsamp)), '--',
                    color = curve_color, aa = False)

    # Legend in the right margin
    handles, labels = axs[0].get_legend_handles_labels()
    axs[1].legend(handles, labels, framealpha = 1, loc = [1.02, 0.],
                  fontsize = 'small')
    return fig, axs
