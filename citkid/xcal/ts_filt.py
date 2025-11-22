from scipy.signal import butter, filtfilt

def bandpass_filter(x, dt, f0, f1, order = 8):
    """
    Applies a bandpass filter to a timestream.

    Parameters:
    x (np.ndarray, (N,)): timestream data.
    dt (float): sample time in s.
    f0 (float): lower cutoff frequency (Hz).
    f1 (float): upper cutoff frequency (Hz).
    order (int): order of the filter.

    Returns:
    filtered_x (np.ndarray, (N,)): Filtered timestream.
    """
    if f0 <= 0 or f1 <= 0 or f0 >= f1 or f1 >= 0.5 / dt:
        raise ValueError(f"f0 and f1 must be positive with f0 < f1 < "\
                         f"{0.5 / dt} Hz")
    if len(x) <= 3 * order:
        m = "Input data length must be greater than 3 times the filter order."
        raise ValueError(m)
    
    # Normalize cutoff frequencies
    low = get_cutoff(dt, f0)
    high = get_cutoff(dt, f1)

    # Design bandpass filter
    b, a = butter(order, [low, high], btype = 'band')

    # Apply filter
    filtered_x = filtfilt(b, a, x)

    return filtered_x

def lowpass_filter(x, dt, f_cutoff, order = 8):
    """
    Applies a lowpass filter to a timestream.

    Parameters:
    x (np.ndarray, (N,)): timestream data.
    dt (float): sample time in s.
    f_cutoff (float): cutoff frequency (Hz).
    order (int): order of the filter.

    Returns:
    filtered_x (np.ndarray, (N,)): Filtered timestream.
    """
    if f_cutoff <= 0 or f_cutoff >= 0.5 / dt:
        raise ValueError(f"f_cuttoff must be positive and < {0.5 / dt} Hz")
    if len(x) <= 3 * order:
        m = "Input data length must be greater than 3 times the filter order."
        raise ValueError(m)
    
    # Normalize cutoff frequency
    cutoff = get_cutoff(dt, f_cutoff)

    # Design lowpass filter
    b, a = butter(order, cutoff, btype='low')

    # Apply filter
    filtered_x = filtfilt(b, a, x)

    return filtered_x

def highpass_filter(x, dt, f_cutoff, order = 8):
    """
    Applies a highpass filter to a timestream.

    Parameters:
    x (np.ndarray, (N,)): timestream data.
    dt (float): sample time in s.
    f_cutoff (float): cutoff frequency (Hz).
    order (int): order of the filter.

    Returns:
    filtered_x (np.ndarray, (N,)): Filtered timestream.
    """
    if f_cutoff <= 0 or f_cutoff >= 0.5 / dt:
        raise ValueError(f"f_cuttoff must be positive and < {0.5 / dt} Hz")
    if len(x) <= 3 * order:
        m = "Input data length must be greater than 3 times the filter order."
        raise ValueError(m)
    
    # Normalize cutoff frequency
    cutoff = get_cutoff(dt, f_cutoff)   

    # Design highpass filter
    b, a = butter(order, cutoff, btype='high')

    # Apply filter
    filtered_x = filtfilt(b, a, x)
    return filtered_x

def get_cutoff(dt, f_cutoff):
    """
    Calculate normalized cutoff frequency.

    Parameters:
    dt (float): sample time in s.
    f (float): cutoff frequency in Hz.

    Returns:
    float: Normalized cutoff frequency.
    """
    nyquist = 0.5 / dt
    return f_cutoff / nyquist
