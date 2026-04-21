import warnings
warnings.warn(
    "citkid.vna.peak_finding_legacy is legacy code and may be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks

def butter_highpass_filter(data, fcut, fs, order = 1, direction = 'forward'):
    """
    Highpass filter for VNA sweeps to remove gain fluctuations.

    Parameters:
    data (np.array): Data to filter.
    fcut (float): Highpass cutoff frequency in Hz.
    fs (float): Sample frequency in Hz.
    order (int): Filter order.
    direction (str): 'forward' for forward filter or 'both' for forward and
        backward filtering.

    Returns:
    y (np.array): Filtered data.
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
    Lowpass filter for VNA sweeps to remove gain fluctuations.

    Parameters:
    data (np.array): Data to filter.
    fcut (float): Lowpass cutoff frequency in Hz.
    fs (float): Sample frequency in Hz.
    order (int): Filter order.
    direction (str): 'forward' for forward filter or 'both' for forward and
        backward filtering.

    Returns:
    y (np.array): Filtered data.
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
    """
    Highpass a VNA sweep to remove gain fluctuations, then find resonances
    with scipy.signal.find_peaks.

    Parameters:
    f (np.array, float64): Array of VNA sweep frequencies.
    z (np.array, complex128): Complex S21 or |S21| data of the sweep.
    fcut (float): Cutoff "wavelength" for highpassing the sweep, in the same
        frequency units as f.
    height (float): Peak height for scipy.signal.find_peaks.
    width (float): Peak width in the same frequency units as f.
    distance (float): Distance between neighboring peaks in the same units as f.
    order (int): Filter order.
    fcut_lowpass (float or None): If not None, applies a lowpass filter at this
        frequency in Hz.

    Returns:
    peaks (np.array, int32): Indices where peaks were found.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    fdiff = np.mean(np.diff(f))
    dB = 20 * np.log10(abs(z))
    dB -= dB[0]
    dBfilt = butter_highpass_filter(
        dB,
        fcut = fdiff / fcut,
        fs = 1,
        order = order,
    )
    if fcut_lowpass is not None:
        dBfilt = butter_lowpass_filter(
            dBfilt,
            fcut = fdiff / fcut_lowpass,
            fs = 1,
            order = order,
        )
    width = int(width / fdiff)
    distance = int(distance / fdiff)
    peaks, _ = find_peaks(
        -dBfilt,
        height = height,
        width = width,
        distance = distance,
    )
    return peaks
