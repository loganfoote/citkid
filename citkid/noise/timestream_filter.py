from scipy.signal import butter, filtfilt
import warnings

warnings.warn('The citkid.noise.timestream_filter module is deprecated and will be '\
              'removed in a future release. Please use the citkid.xcal.ts_filt '\
              'module for timestream processing functions.', 
              DeprecationWarning)

def bandpass_filter(x, dt, f_low, f_high, order=8):
    """
    Applies a sharp bandpass filter to a timestream.

    Parameters:
    x (np.ndarray): Input timestream data.
    dt (float): Sample time interval.
    f_low (float): Lower cutoff frequency (Hz).
    f_high (float): Upper cutoff frequency (Hz).
    order (int): Order of the filter (default is 8).

    Returns:
    np.ndarray: Filtered timestream.
    """
    # Nyquist frequency
    nyquist = 0.5 / dt

    # Normalize cutoff frequencies
    low = f_low / nyquist
    high = f_high / nyquist

    # Design bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply filter
    filtered_x = filtfilt(b, a, x)

    return filtered_x

def lowpass_filter(x, dt, f_cutoff, order=8):
    """
    Applies a lowpass filter to a timestream.

    Parameters:
    x (np.ndarray): Input timestream data.
    dt (float): Sample time interval.
    f_cutoff (float): Cutoff frequency (Hz).
    order (int): Order of the filter (default is 8).

    Returns:
    np.ndarray: Filtered timestream.
    """
    # Nyquist frequency
    nyquist = 0.5 / dt

    # Normalize cutoff frequency
    cutoff = f_cutoff / nyquist

    # Design lowpass filter
    b, a = butter(order, cutoff, btype='low')

    # Apply filter
    filtered_x = filtfilt(b, a, x)

    return filtered_x

def highpass_filter(x, dt, f_cutoff, order=8):
    """
    Applies a highpass filter to a timestream.

    Parameters:
    x (np.ndarray): Input timestream data.
    dt (float): Sample time interval.
    f_cutoff (float): Cutoff frequency (Hz).
    order (int): Order of the filter (default is 8).

    Returns:
    np.ndarray: Filtered timestream.
    """
    # Nyquist frequency
    nyquist = 0.5 / dt

    # Normalize cutoff frequency
    cutoff = f_cutoff / nyquist

    # Design highpass filter
    b, a = butter(order, cutoff, btype='high')

    # Apply filter
    filtered_x = filtfilt(b, a, x)

    return filtered_x
