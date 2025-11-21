import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks

def butter_highpass_filter(data, fcut, fs, order = 1, direction = 'forward'):
    """
    Highpass filter for VNA scans to remove gain fluctuations.

    Parameters:
    data (np.array): data to filter.
    fcut (float): highpass cutoff frequency in Hz.
    fs (float): sample frequency in Hz.
    order (int): filter order.
    direction (str): 'forward' for forward filter or 'both' for
        forward-and-backward filter.

    Returns:
    y (np.array): filtered data.
    """
    nyq = 0.5 * fs
    low = fcut / nyq
    sos = butter(order, low, analog=False, btype='high', output='sos')
    if direction == 'forward':
        y = sosfilt(sos, data)
    elif direction == 'both':
        y = sosfiltfilt(sos, data)
    return y

def butter_lowpass_filter(data, fcut, fs, order = 1, direction = 'forward'):
    """
    Lowpass filter for VNA scans to remove gain fluctuations.

    Parameters:
    data (np.array): data to filter.
    fcut (float): lowpass cutoff frequency in Hz.
    fs (float): sample frequency in Hz.
    order (int): filter order.
    direction (str): 'forward' for forward filter or 'both' for
        forward-and-backward filter.

    Returns:
    y (np.array): filtered data.
    """
    nyq = 0.5 * fs
    high = fcut / nyq
    sos = butter(order, high, analog=False, btype='low', output='sos')
    if direction == 'forward':
        y = sosfilt(sos, data)
    elif direction == 'both':
        y = sosfiltfilt(sos, data)
    return y

def find_peaks_highpass(f, z, fcut, height = 0.2, width = 10e3, distance = 50e3,
                        order = 1, fcut_lowpass = False):
    '''
    Highpasses a VNA scan to get rid of gain fluctuations, then finds resonances
    with scipy.signal.find_peaks.

    Parameters:
    f (np.array, float64): array of vna scan frequencies.
    z (np.array, complex128): complex S21 or absolute value |S21| data of
        vna scan.
    fcut (float): cutoff "wavelength" for highpassing the vna scan. Should be in
        the same frequency units as fvna.
    height (float): peak height for scipy.signal.find_peaks.
    width (float): peak width, in same frequency units as fvna.
    distance (float): distance between neighboring peaks, in same frequency
        units as fvna.
    order (int): filter order.
    fcut_lowpass (float or None): if not None, applies a lowpass filter at this
        frequency in Hz.

    Returns:
    peaks (np.array, int32): array of indices where peaks were found.
    '''
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    fdiff = np.mean(np.diff(f))
    dB = 20 * np.log10(abs(z))
    dB -= dB[0]
    dBfilt = butter_highpass_filter(dB, fcut = fdiff / fcut, fs = 1,
                                   order = order)
    if fcut_lowpass is not None:
        dBfilt = butter_lowpass_filter(dBfilt, fcut = fdiff / fcut_lowpass,
                                       fs = 1, order = order)
    width = int(width / fdiff)
    distance = int(distance / fdiff)
    peaks, _ = find_peaks(-dBfilt, height = height, width = width,
                          distance = distance)
    return peaks
