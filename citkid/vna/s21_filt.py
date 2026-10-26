"""
S21 filtering utilities for VNA sweep data.

Provides standalone functions for removing baseline drift from |S21|
magnitude data, removing a linear phase trend, and finding resonance dips.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


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


def filter_mag(f, mag_db, method='none', highpass_mhz=10.0, poly_order=3):
    """
    Remove the baseline from |S21| magnitude data with the chosen method.

    Parameters:
    f (np.ndarray): Frequency array in Hz (1D).
    mag_db (np.ndarray): |S21| magnitude in dB (1D, same length as f).
    method (str): Can be 'highpass' (``highpass_filter`` with
        ``highpass_mhz``), 'polynomial' (``polynomial_baseline`` with
        ``poly_order``), or 'none' (default, a copy of ``mag_db``). Any
        other value is treated as 'none'.
    highpass_mhz (float): High-pass cutoff scale in MHz. Default is 10.0.
    poly_order (int): Polynomial baseline order. Default is 3.

    Returns:
    filtered_mag (np.ndarray): Filtered magnitude in dB.
    """
    if method == 'highpass':
        return highpass_filter(f, mag_db, highpass_mhz)
    if method == 'polynomial':
        return polynomial_baseline(f, mag_db, poly_order)
    return np.array(mag_db, copy=True)


def remove_linear_phase(f, z):
    """
    Remove a linear trend from the unwrapped phase of S21 data.

    Parameters:
    f (np.ndarray): Frequency array in Hz (1D).
    z (np.ndarray): Complex S21 data (1D, same length as f).

    Returns:
    z_rot (np.ndarray): ``z`` with the linear phase trend removed (a new
        array; ``z`` is not modified).
    phase (np.ndarray): Unwrapped phase of ``z_rot`` in radians.
    p (np.ndarray): 1st-order polynomial fit parameters to the unwrapped
        phase of ``z``.
    """
    p = np.polyfit(f, np.unwrap(np.angle(z)), 1)
    z_rot = z * np.exp(-1j * np.polyval(p, f))
    phase = np.unwrap(np.angle(z_rot))
    return z_rot, phase, p


def find_res_dips(f, mag_db, height, width, distance, f_min=None, f_max=None):
    """
    Find resonance dips in |S21| magnitude data with ``find_peaks``.

    Parameters:
    f (np.ndarray): Frequency array in Hz (1D, sorted ascending, uniformly
        sampled).
    mag_db (np.ndarray): (Filtered) |S21| magnitude in dB (1D, same length
        as f).
    height (float): Dips must go below this magnitude in dB.
    width (float): Minimum dip width in kHz/GHz, i.e. scaled by the center
        frequency of the search range.
    distance (float): Minimum distance between dips in kHz/GHz.
    f_min (float or None): Lower edge of the search range in Hz. None uses
        ``f[0]``.
    f_max (float or None): Upper edge of the search range in Hz. None uses
        ``f[-1]``.

    Returns:
    fres (np.ndarray or None): Frequencies of the dips in Hz, or None if no
        data is in the search range.

    Notes:
    ``width`` and ``distance`` are converted to samples using the center of
    [f_min, f_max] and the median sample spacing, truncated and floored at
    1.
    """
    f = np.asarray(f)
    mag_db = np.asarray(mag_db)
    f_min = f[0] if f_min is None else f_min
    f_max = f[-1] if f_max is None else f_max
    mask = (f >= f_min) & (f <= f_max)
    f_masked = f[mask]
    mag_masked = mag_db[mask]
    if len(f_masked) == 0:
        return None

    # (kHz/GHz) * GHz = kHz, then * 1e3 = Hz
    f_center_ghz = np.mean([f_min, f_max]) / 1e9
    width_hz = width * f_center_ghz * 1e3
    distance_hz = distance * f_center_ghz * 1e3

    # Convert to samples
    df = np.median(np.diff(f_masked))
    width_samples = max(1, int(width_hz / df))
    distance_samples = max(1, int(distance_hz / df))

    # Look for dips, so invert
    peaks, _ = find_peaks(
        -mag_masked,
        height=-height,
        width=width_samples,
        distance=distance_samples
    )
    return f_masked[peaks]
