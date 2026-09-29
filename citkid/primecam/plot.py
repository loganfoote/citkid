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
