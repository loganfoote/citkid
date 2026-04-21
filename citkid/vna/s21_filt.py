"""
S21 filtering utilities for VNA sweep data.

Provides standalone functions for removing baseline drift from |S21|
magnitude data, suitable for use before peak/resonance detection.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(f, mag_db, cutoff_mhz=10.0):
    """
    Apply a high-pass Butterworth filter to remove slow baseline drift.

    The cutoff is expressed in MHz of *feature scale* in the frequency
    sweep (i.e. features wider than cutoff_mhz will be attenuated).

    Parameters:
    f (np.ndarray): Frequency array in Hz (1D, uniformly sampled).
    mag_db (np.ndarray): |S21| magnitude in dB (1D, same length as f).
    cutoff_mhz (float): High-pass cutoff scale in MHz. Features varying
        more slowly than this scale are removed. Default is 10.0.

    Returns:
    filtered_mag (np.ndarray): Filtered magnitude in dB.
    """
    cutoff_hz = cutoff_mhz * 1e6

    # Frequency spacing between samples
    df = np.median(np.diff(f))

    # Convert cutoff to normalised frequency (Nyquist = 1.0)
    # A feature of scale cutoff_hz spans cutoff_hz/df samples
    # -> spatial frequency df/cutoff_hz cycles/sample
    # -> normalised: (df/cutoff_hz) / 0.5
    cutoff_norm = (df / cutoff_hz) / 0.5
    cutoff_norm = np.clip(cutoff_norm, 0.0001, 0.99)

    b, a = butter(3, cutoff_norm, btype='high')
    return filtfilt(b, a, mag_db)


def polynomial_baseline(f, mag_db, order=3):
    """
    Remove a polynomial baseline from |S21| magnitude data.

    Fits a polynomial of the given order to the data and subtracts it,
    leaving only variations around the baseline.

    Parameters:
    f (np.ndarray): Frequency array in Hz (1D).
    mag_db (np.ndarray): |S21| magnitude in dB (1D, same length as f).
    order (int): Polynomial order for the baseline fit. Default is 3.

    Returns:
    filtered_mag (np.ndarray): Baseline-subtracted magnitude in dB.
    """
    coeffs = np.polyfit(f, mag_db, order)
    baseline = np.polyval(coeffs, f)
    return mag_db - baseline
