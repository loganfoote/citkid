from scipy.stats import binned_statistic
import numpy as np
import pyfftw


def get_psd(x, dt, get_frequencies=False):
    """
    Calculates the unilateral power spectral density magnitude of a timestream.

    Parameters:
    x (np.array, float64): Timeseries data.
    dt (float): Sample time in seconds.
    get_frequencies (bool): If True, also returns the frequency array.

    Returns:
    f (np.array, float64): Frequency array in Hz if get_frequencies is True.
    psd (np.array, float64): Power spectral density.
    """
    x = np.asarray(x, dtype=np.float64)
    a = pyfftw.interfaces.numpy_fft.rfft(x)
    psd = 2 * np.abs(a) ** 2 * dt / len(x)
    if not get_frequencies:
        return psd
    f = np.fft.rfftfreq(len(x), d=dt)
    return f, psd


def get_csd(x1, x2, dt):
    """
    Calculates the unilateral cross spectral density magnitude of two
    timestreams.

    Parameters:
    x1 (np.array): First timeseries data.
    x2 (np.array): Second timeseries data.
    dt (float): Sample time in seconds.

    Returns:
    f (np.array, float64): Frequency array in Hz.
    cpsd (np.array, float64): Cross spectral density magnitude.
    """
    a1 = pyfftw.interfaces.numpy_fft.rfft(x1)
    a2 = pyfftw.interfaces.numpy_fft.rfft(x2)
    cpsd = 2 * np.conj(a1) * a2 * dt / len(x1)
    cpsd = np.abs(cpsd)
    f = np.fft.rfftfreq(len(x1), d=dt)
    return f, cpsd


################################################################################
############################ Binning and filtering #############################
################################################################################

def bin_psd(f, data, nbins=500, fmin=3, filter_pt_n=None,
            pt_frequency=1.39296, statistic='mean'):
    """
    Bins noise data logarithmically. Optionally filters pulse tubes before
    binning and leaves frequencies below fmin unbinned.

    Parameters:
    f (np.array): Frequency array in Hz.
    data (list): List of 1D arrays corresponding to f to bin.
    nbins (int): Number of bins.
    fmin (float): Minimum frequency to bin; values below fmin are left unbinned.
    filter_pt_n (int or None): Number of pulse tube harmonics to filter, or
        None to bypass pulse tube filtering.
    pt_frequency (float): Pulse tube frequency in Hz.
    statistic (str): Statistic used to compute bin values.

    Returns:
    binned_data (list): Binned arrays corresponding to each input array.
    """
    if not type(f) == np.ndarray:
        f = np.array(f)
    for i, x in enumerate(data):
        if not type(x) == np.ndarray:
            data[i] = np.array(x)
    ix = f > 0
    f, data = f[ix], [d[ix] for d in data]
    if filter_pt_n is not None:
        data = [data[0]] + [filter_pt(f, d, filter_pt_n, pt_frequency)
                            for d in data[1:]]
    ix = f < fmin
    f0, data0 = f[ix], [d[ix] for d in data]
    # Create logarithmically spaced bins, and remove bins that don't have data
    bins = np.geomspace(fmin, max(f), nbins)
    bin_counts, _, _ = binned_statistic(f, [], statistic='count', bins=bins)
    bin_counts = np.concatenate([bin_counts, [1]])
    bins = bins[bin_counts != 0]
    # Bin data, and append unbinned data
    binned_data = binned_statistic(f, data, bins=bins,
                                   statistic=statistic)[0]
    binned_data = [np.concatenate([data0[i], binned_data[i]])
                   for i in range(len(binned_data))]
    return binned_data


def filter_pt(f, y, n=20, pt_frequency=1.39296):
    """
    Filter pulse tube spikes out of noise data.

    Parameters:
    f (np.array): Frequency array in Hz.
    y (np.array): Data to filter.
    n (int): Number of pulse tube harmonics to filter.
    pt_frequency (float): Pulse tube frequency in Hz.

    Returns:
    y_filt (np.array): Data with pulse tube spikes removed.
    """
    f0s = [pt_frequency * i for i in range(1, n)]
    for f0 in f0s:
        # Get width from typical values
        d = np.interp(f0, [1, 21, 32.1], [0.1, 0.08, 0.01])
        ix = np.where(abs(f - f0) < d)[0]
        N = len(ix)
        if N:
            y[ix] = np.mean([y[ix.min() - N:ix.min()],
                             y[ix.max():ix.max() + N]], axis=0)
    return y
