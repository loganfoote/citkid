import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from ..util import fix_path, save_fig
import os


def plot_ares_opt(a_nls, fcal_indices):
    """
    Plots the current status of the power optimization procedure

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

def plot_update_fres(fs, zs, fres, fcal_indices, res_indices, cable_delay, plot_directory):
    """
    Plots the results of update_fres in batches

    Parameters:
    fs (array-like): fine sweep frequency data in Hz for each resonator in fres
    zs (array-like): fine sweep complex S21 data for each resonator in fres
    fres (np.array or None): list of resonance frequencies in Hz
    fcal_indices (array-like): list of calibrations tone indices (index into
        fs, zs, fres, Qres). Calibration tone frequencies will not be updated
    res_indices (array-like): resonator indices for plotting
    plot_directory (str): directory to save plots
    """
    fs, zs, fres = np.array(fs), np.array(zs), np.array(fres)
    fcal_indices = np.array(fcal_indices)
    res_indices = np.array(res_indices)
    plot_directory = fix_path(plot_directory)
    os.makedirs(plot_directory, exist_ok = True)
    num_plots = len(fres)

    plots_per_fig = 100
    max_n_cols = 4
    num_figs = (num_plots - 1) // plots_per_fig + 1

    for fig_index in range(num_figs):
        ix0, ix1 = fig_index * plots_per_fig, (fig_index + 1) * plots_per_fig
        fs0, zs0, fres0 = fs[ix0:ix1], zs[ix0:ix1], fres[ix0:ix1]
        rs0 = res_indices[ix0:ix1]

        data_indices = np.arange(ix0, ix1, 1)

        naxs = len(fs0) 
        nrows = naxs // max_n_cols
        len_last_row = naxs % max_n_cols  
        if len_last_row > max_n_cols or len_last_row == 0:
            ncols = max_n_cols 
        else:
            ncols = len_last_row
        fig, axs = plt.subplots(nrows, ncols * 2, figsize = [6 * ncols, 2.5 * nrows],
                            layout = 'tight')
        ax_pairs = [[axs[row][2 * column], axs[row][2 * column + 1]] for row in range(len(axs)) for column in range(int(len(axs[0]) / 2))]
        for index, (ax1, ax0) in enumerate(ax_pairs):
            # ax1.set(ylabel = 'Q', xlabel = 'I') 
            ax1.set_yticks([])
            ax1.set_xticks([])
            pos1 = ax1.get_position()
            # ax1.set_position([pos1.x0 - 0.04, pos1.y0, pos1.width, pos1.height])
            ax0.set(ylabel = r'$|S_{21}| (dB)') 

            f, z, fr = fs0[index], zs0[index], fres0[index] 
            ri = rs0[index]
            ax1.set(title = f'Fn {ri}')
            data_ix = data_indices[index] 
            dB = 20 * np.log10(np.abs(z)) 

            f0 = np.mean(f) 
            ax0.set(xlabel = f'(f - {int(round(fr / 1e3, 0))}) kHz') 

            if data_ix in fcal_indices:
                color = plt.cm.viridis(0.667) 
            else:
                color = plt.cm.viridis(0.) 
            ax0.plot((f - f0) / 1e3, dB, color = color) 
            ax1.plot(np.real(z), np.imag(z), '.', color = color)
            ax0.axvline((fr - f0) / 1e3, linestyle = '--', color = 'k') 
            ix = np.argmin(np.abs(f - fr))
            ax1.plot(np.real(z[ix]), np.imag(z[ix]), 'xk', markersize = 20)
        save_fig(fig, f'fres_update_{fig_index}', plot_directory, ftype = 'png',
                            tight_layout = False, close_fig = True)
        


