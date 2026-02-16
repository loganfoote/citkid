import numpy as np 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter

check_peak_validity = lambda p, s, N: (p >= 0) & (p < len(s) - N)

def get_noise_idx(starts, idx0 = 90, idx1 = 30):
    """
    Get indices into x that estimate the noise PSD. Finds the region of 
    maximum spacing between starts. 

    Parameters:
    starts (np.array, (N,), int): peak locations.
    idx0 (int): offset to the right of the left index to ensure peak is not 
        captured in the output.
    idx1 (int): offset to the left of the right index to ensure peak is not 
        captured in the output.

    Returns:
    noise_idx (slice): indices to use for estimating the noise PSD.
    """
    starts = np.asarray(starts, dtype = np.int32)
    assert isinstance(idx0, (int, np.integer)), "idx0 must be an integer" 
    assert isinstance(idx1, (int, np.integer)), "idx1 must be an integer" 
    idx = np.argmax(np.diff(starts))
    p0, p1 = starts[idx] + idx0, starts[idx + 1] - idx1
    noise_idx = slice(p0, p1)
    return noise_idx

def build_template(s, starts, N, idxa = None, normalize = True):
    """
    Build template by averaging segments around starts. Subtracts the baseline 
    from the template and normalizes such that the peak occurs at 1. 

    Parameters:
    s (np.ndarray): timestream.
    starts (list or np.ndarray): indices of pulse peak start locations.
    N (int): length of template
    idxa (int or None): baseline length (baseline is x[p:p + N] for p in
        starts). If not normalize, idxa is disregarded. 
    normalize (bool): 

    Returns:
    template (np.ndarray): averaged template.
    """
    s = np.asarray(s, dtype = np.float64)
    starts = np.asarray(starts, dtype = np.int32) 
    assert isinstance(N, (int, np.integer)), "N must be an integer" 
    assert isinstance(idxa, (int, np.integer, type(None))), \
            "idxa must be an integer" 
    
    valid_starts = [p for p in starts if check_peak_validity(p, s, N)]
    if len(valid_starts) == 0:
        raise ValueError("No valid starts to build template")

    # Accumulate segments for averaging
    acc = np.zeros(N, dtype=np.float64)
    for p in valid_starts:
        acc += s[p:p + N]

    template = acc / len(valid_starts)
    if not normalize:
        return template

    # subtract baseline
    baseline = template[0 : idxa + 1]
    template -= np.mean(baseline)

    # normalize to peak = 1
    template /= np.max(template)

    return template

def get_threshold(x, sigma_threshold):
    """
    Given a timestream and sigma threshold, return the absolute threshold.

    Parameters:
    x (np.ndarray, np.float64): timestream.
    sigma_threshold (float or list[float]): sigma threshold or thresholds.

    Returns:
    threshold (float): absolute threshold, in units of x.
    """
    threshold = x.mean() + np.asarray(sigma_threshold) * x.std()
    return threshold

def plot_template(aas, dt, idxa):
    """
    Plot a set of templates (iterations) and mark the baseline cutoff.

    Parameters:
    aas (list of np.ndarray): list of template arrays (each of length M) for
        each iteration.
    dt (float): sample spacing in seconds between adjacent samples in the
        templates.
    idxa (int): index into the template that denotes the baseline cutoff.

    Returns:
    fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): figure and axes
        containing the plotted templates.
    """
    fig, ax = plt.subplots(figsize = [5, 4], layout = 'tight', dpi = 100)
    ax.set(ylabel = 'A', xlabel = 'time (ms)')
    M = len(aas[0])
    tt = np.linspace(0, dt * (M - 1), M)
    for i, a in enumerate(aas):
        c = plt.cm.viridis(i / (len(aas) - 1))
        ax.plot(tt * 1e3, a, color = c, aa = False, label = f"{i}")
    ax.axvline(tt[idxa] * 1e3, color = 'k', linestyle = '--', 
               label = 'baseline\ncutoff')
    ax.legend(framealpha = 1, ncol = 2, title = 'Iteration')
    return fig, ax

def plot_DT(DTs):
    """
    Plot the relative template differences per iteration.

    Parameters:
    DTs (array-like): sequence of relative differences between consecutive
        templates, typically norm(T_i - T_{i-1}) / norm(T_{i-1}).

    Returns:
    fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): figure and axes
        containing the plotted iteration differences.
    """

    fig, ax = plt.subplots(figsize = [5, 3], layout = 'tight', dpi = 100) 
    ax.set(
        ylabel = r'$\mathrm{norm}(T_i - T_{i - 1}) / \mathrm{norm}(T_{i - 1})$',
        xlabel = 'Iteration', xticks = range(0, len(DTs) + 1, 1)
           )
    
    ax.plot(range(1, len(DTs) + 1), DTs, '-sk')
    return fig, ax

def plot_xy(dt, x, y, starts, hs, L):
    """
    Plot x and y timestreams with highlighted detected signals.

    Parameters:
    dt (float): sample spacing in seconds.
    x (np.ndarray): auxiliary timestream plotted on the top axis.
    y (np.ndarray): primary timestream plotted on the bottom axis.
    starts (array-like of int): indices marking detected signal start locations.
    hs (sequence of two floats): threshold range (hmin, hmax) used for shading
        the detection region on the `y` axis. Values are in the same units as
        `y`.
    L (int): length in samples used to plot each detected segment following
        a start index.

    Returns:
    fig, axs (matplotlib.figure.Figure, numpy.ndarray of Axes): the figure and
        the two axes (top: `x`, bottom: `y`) with signals highlighted.
    """

    fig, axs = plt.subplots(2, 1, figsize = [8, 6], layout = 'tight', 
                            sharex = True, dpi = 75)


    axs[0].set(ylabel = 'x (kHz / GHz)')
    axs[1].set(ylabel = 'y (kHz / GHz)', xlabel = 'Time (s)')
    t = np.arange(len(x)) * dt

    # Remove outliers from x, y
    sig = hs[1] / np.std(y) 
    x, y = x.copy(), y.copy() 
    x[np.abs(x) > sig * np.std(x)] = np.median(x)
    y[np.abs(y) > sig * np.std(y)] = np.median(y)

    axs[0].plot(t, x * 1e6, color = plt.cm.cividis(0.), 
                aa = False, rasterized = True)
    for start in starts:
        idx = slice(start, start + L)
        axs[0].plot(t[idx], x[idx] * 1e6, color = plt.cm.cividis(1.), 
                aa = False, rasterized = True)
    axs[1].plot(t, y * 1e6, color = plt.cm.cividis(0.), 
                aa = False, rasterized = True, label = 'data')
    axs[1].plot(t[starts], y[starts] * 1e6, 'x', color = plt.cm.cividis(1.), 
                aa = False, rasterized = False, label = 'signal start')
    axs[1].plot([], [], '-', color = plt.cm.cividis(1.), label = 'signal')
    
    xx = list(axs[1].get_xlim())
    yy = [hs[0] * 1e6] * 2 
    zz = [hs[1] * 1e6] * 2
    axs[1].fill_between(xx, yy, zz, color = plt.cm.Greys(0.5), aa = False, 
                        label = 'threshold')
    axs[1].set(xlim = [min(t), max(t)])
    axs[1].legend(framealpha = 1, loc = 'upper left')
    return fig, axs

def plot_a(sigmas, h_minmax, y, L):
    """
    Plot amplitude histograms for detected peaks at different sigma
    thresholds and overlay the chosen threshold region.

    Parameters:
    sigmas (iterable of float): sigma multiples used to compute thresholds
        via `get_threshold(y, sigma)` for peak detection.
    h_minmax (tuple of (float, float)): (hmin, hmax) absolute amplitude
        bounds (same units as `y`) used to filter peak heights.
    y (np.ndarray): timestream from which peak amplitudes are extracted.
    L (int): minimum peak separation (in samples) passed to
        `scipy.signal.find_peaks` as `distance`.

    Returns:
    fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): figure and
        axes containing the overlaid histograms and secondary sigma axis.
    """

    hmin, hmax = h_minmax
    amps = []
    sigma = ['None'] + list(sigmas)
    t1 = hmax * 1e-6
    for sig in sigma:
        if sig == 'None':
            a = y
        else:
            t0 = get_threshold(y, sig)
            peaks_th, d = find_peaks(y, height = (t0, t1), distance = L // 2)
            a = d['peak_heights']
        amps.append(a)
        
    fig, ax = plt.subplots(figsize = [5, 3], layout = 'tight', dpi = 100)
    ax.set(ylabel = 'Density', xlabel = 'x (kHz / GHz)')

    b0s = [min(a) for a in amps[1:] if len(a)]
    b0 = min(b0s) if len(b0s) else 1
    b1 = max([max(a) for a in amps[1:] if len(a)])
    if b0 >= 0:
        b0 = -b1 / 8
    bins = np.linspace(b0, b1, 50)
    for i, (amp, sig) in enumerate(zip(amps, sigma)):  
        c = plt.cm.viridis(i / (len(amps) - 1))
        lbl = rf'{sig}$\sigma$' if sig != 'None' else sig
        h = ax.hist(amp * 1e6, bins = bins * 1e6, color = c, 
                    label = lbl, alpha = 0.7, density = 1) 
    ylim = ax.get_ylim()
    ax.fill_between(
        [hmin, hmax], 
        ylim[0], ylim[1],
        color = plt.cm.Greys(0.5), alpha = 0.5, 
        zorder = 0, 
        label = "threshold"
    )
    ax.set(ylim = ylim)
    ax.legend(framealpha = 1, loc = [1.01, 0], title = 'Threshold')

    ymean, ystd = np.mean(y), np.std(y)
    def y_to_sig(y):
        return (y - ymean) / ystd / 1e6

    def sig_to_y(u):
        return (u * ystd + ymean) / 1e6

    secax = ax.secondary_xaxis('top', functions=(y_to_sig, sig_to_y))
    secax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: rf"{val:.0f}$\sigma$")
    )   
    ax.grid()
    return fig, ax