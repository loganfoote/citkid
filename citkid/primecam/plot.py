import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

def plot_ares_opt(a_nls, fcal_indices):
    """
    Plot the current status of the power optimization procedure

    Parameters:
    a_nls (list): each value is an array of the values of the nonlinearity
        parameter for each resonator for the given iteration
    fcal_indices (array-like): indices that are calibration tones

    Returns:
    fig_hist (pyplot.fig): histogram of nonlinearity parameters for each
        iteration
    fig_opt (pyplot.fig): plot of the percent of the array that is optimized
        versus iteration number
    """
    fig_hist, ax_hist = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_hist.set(xlim = (0,1.5), xlabel = 'Nonlinearity parameter', ylabel = 'Number of KIDs')
    bins = np.linspace(0, 1.5, 20)
    for index, a_nl in enumerate(a_nls):
        [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        if index == len(a_nls):
            alpha = 0.8 
        else:
            alpha = 0.5
        ax_hist.hist(a_nl, bins, label = index, edgecolor = color, color = color, alpha = alpha)
    ax_hist.legend(title = 'Iteration', framealpha = 1)

    fig_opt, ax_opt = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_opt.set(xlabel = 'Iteration', ylabel = 'Percentage of array optimized')
    ax_opt.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_opt.grid()
    indices, percents = [], []
    for index, a_nl in enumerate(a_nls):
        a_nl = [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        a_opt = [a for a in a_nl if a <= 0.6 and a >= 0.4]
        percent = len(a_opt) / (len(a_nl)) * 100
        ax_opt.scatter(index, percent, marker = 's', color = color)
        indices.append(index)
        percents.append(percent)
    ax_opt.plot(indices, percents, '--k')
    return fig_hist, fig_opt

def plot_gain_fit(f0, dB0, f, dB, phase, p_amp, p_phase):
    """
    Plot the fit to gain amplitude and phase data.

    Parameters:
    f0 (np.array): Raw frequency data.
    dB0 (np.array): Raw amplitude data.
    f (np.array): Cut frequency data.
    dB (np.array): Cut amplitude data.
    phase (np.array): Cut phase data.
    p_amp (list): Amplitude fit parameters.
    p_phase (list): Phase fit parameters.

    Returns:
    fig, axs: Data and fit plot.
    """
    fmean = np.mean(f0)
    fig, axs = plt.subplots(
        1,
        2,
        figsize = [6, 2.8],
        dpi = 200,
        layout = 'tight',
    )
    axs[1].set_ylabel('Phase')
    axs[1].set_xlabel(f'(f - {round(fmean / 1e9, 4)} GHz) (kHz)')
    axs[0].set_ylabel('|S21| (dB)')
    axs[0].set_xlabel(f'(f - {round(fmean / 1e9, 4)} GHz) (kHz)')

    color = plt.cm.viridis(0.1)
    color0 = plt.cm.viridis(0.99)
    axs[0].plot(
        (f0 - fmean) * 1e-3,
        dB0,
        '.',
        color = color0,
        label = 'Raw data',
    )
    axs[0].plot((f - fmean) * 1e-3, dB, '.', color = color, label='Fitted data')
    fsamp = np.linspace(np.min(f0), np.max(f0), 100)
    if ~np.any(np.isnan(p_amp)):
        axs[0].plot(
            (fsamp - fmean) * 1e-3,
            np.polyval(p_amp, fsamp),
            '--r',
            label = 'Fit',
        )

    axs[1].plot([], [], '.', color = color0, label = 'Raw data')
    axs[1].plot(
        (f - fmean) * 1e-3,
        phase,
        '.',
        color = color,
        label = 'Fitted data',
    )
    if ~np.any(np.isnan(p_phase)):
        axs[1].plot(
            (fsamp - fmean) * 1e-3,
            np.polyval(p_phase, fsamp),
            '--r',
            label = 'Fit',
        )

    # axs[1].legend(framealpha=1)
    return fig, axs

def plot_circle(z, A, B, R):
    """
    Plot IQ data with a circular fit.

    Parameters:
    z (np.array): Complex IQ data.
    A, B (float, float): Circle origin.
    R (float): Circle radius.

    Returns:
    fig, ax: Data and fit plot.
    """
    fig, ax = plt.subplots(figsize = (4, 4), dpi = 200)
    ax.plot(np.real(z), np.imag(z), 'r.')
    ax.set_aspect('equal', adjustable='datalim')
    cir = plt.Circle((A, B), R, color='k', fill=False)
    ax.add_patch(cir)
    ax.set(xlabel = 'I', ylabel = 'Q')
    return fig, ax
