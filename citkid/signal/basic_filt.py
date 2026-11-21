import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt

def bandpass_filter(x, dt, f0, f1, order = 8):
    """
    Apply a bandpass filter to a timestream.

    Parameters:
    x (np.ndarray, (N,)): Timestream data.
    dt (float): Sample time in s.
    f0 (float): Lower cutoff frequency in Hz.
    f1 (float): Upper cutoff frequency in Hz.
    order (int): Order of the filter.

    Returns:
    filtered_x (np.ndarray, (N,)): Filtered timestream.
    """
    if f0 <= 0 or f1 <= 0 or f0 >= f1 or f1 >= 0.5 / dt:
        raise ValueError(f"f0 and f1 must be positive with f0 < f1 < "\
                         f"{0.5 / dt} Hz")
    if x.shape[-1] <= 3 * (2 * order + 1):
        m = "x.shape[-1] must be greater than padlen = 3 * (2 * order + 1)."
        raise ValueError(m)
    
    # Normalize cutoff frequencies
    low = get_cutoff(dt, f0)
    high = get_cutoff(dt, f1)

    # Design and apply filter
    if order > 8: # Use second-order sections for high-order filters
        sos = butter(order, [low, high], btype = 'band', output = 'sos')
        filtered_x = sosfiltfilt(sos, x)
    else: # Use standard filter design for lower-order filters
        b, a = butter(order, [low, high], btype = 'band')
        filtered_x = filtfilt(b, a, x)
    return filtered_x

def lowpass_filter(x, dt, f_cutoff, order = 8):
    """
    Apply a lowpass filter to a timestream.

    Parameters:
    x (np.ndarray, (N,)): Timestream data.
    dt (float): Sample time in s.
    f_cutoff (float): Cutoff frequency in Hz.
    order (int): Order of the filter.

    Returns:
    filtered_x (np.ndarray, (N,)): Filtered timestream.
    """
    if f_cutoff <= 0 or f_cutoff >= 0.5 / dt:
        raise ValueError(f"f_cuttoff must be positive and < {0.5 / dt} Hz")
    if x.shape[-1] <= 3 * (order + 1):
        m = "x.shape[-1] must be greater than padlen = 3 * (order + 1)."
        raise ValueError(m)
    
    # Normalize cutoff frequency
    cutoff = get_cutoff(dt, f_cutoff)

    # Design and apply filter
    if order > 8: # Use second-order sections for high-order filters
        sos = butter(order, cutoff, btype = 'low', output ='sos')
        filtered_x = sosfiltfilt(sos, x)
    else: # Use standard filter design for lower-order filters
        b, a = butter(order, cutoff, btype = 'low')
        filtered_x = filtfilt(b, a, x)
    return filtered_x

def highpass_filter(x, dt, f_cutoff, order = 8):
    """
    Apply a highpass filter to a timestream.

    Parameters:
    x (np.ndarray, (N,)): Timestream data.
    dt (float): Sample time in s.
    f_cutoff (float): Cutoff frequency in Hz.
    order (int): Order of the filter.

    Returns:
    filtered_x (np.ndarray, (N,)): Filtered timestream.
    """
    if f_cutoff <= 0 or f_cutoff >= 0.5 / dt:
        raise ValueError(f"f_cuttoff must be positive and < {0.5 / dt} Hz")
    if x.shape[-1] <= 3 * (order + 1):
        m = "x.shape[-1] must be greater than padlen = 3 * (order + 1)."
        raise ValueError(m)
    
    # Normalize cutoff frequency
    cutoff = get_cutoff(dt, f_cutoff)   

    # Design and apply filter
    if order > 8: # Use second-order sections for high-order filters
        sos = butter(order, cutoff, btype = 'high', output = 'sos')
        filtered_x = sosfiltfilt(sos, x)
    else: # Use standard filter design for lower-order filters
        b, a = butter(order, cutoff, btype = 'high')
        filtered_x = filtfilt(b, a, x)
    return filtered_x

def get_cutoff(dt, f_cutoff):
    """
    Calculate normalized cutoff frequency.

    Parameters:
    dt (float): Sample time in s.
    f_cutoff (float): Cutoff frequency in Hz.

    Returns:
    cutoff (float): Normalized cutoff frequency.
    """
    nyquist = 0.5 / dt
    return f_cutoff / nyquist


def decimate(x, dt, rate_hz):
    """
    Lower a timestream's sample rate to about ``rate_hz`` with an
    anti-aliasing FIR filter (``scipy.signal.decimate``, zero phase), in
    steps of at most 10, so a spectrum well below the new Nyquist
    frequency is unchanged.

    Parameters:
    x (np.array, (T,) or (N, T)): the timestream(s).
    dt (float): sample period, s.
    rate_hz (float): wanted sample rate, Hz. The result's rate is the
        closest at or above it that an integer factor gives; a rate at
        or above the current one leaves ``x`` as it is.

    Returns:
    y (np.array): the decimated timestream(s).
    dt_out (float): its sample period, s.
    """
    from scipy.signal import decimate as _decimate
    y = np.asarray(x, dtype = np.float64)
    q_total = int(np.floor(1. / (dt * rate_hz)))
    if q_total < 2:
        return y, float(dt)
    dt_out = float(dt)
    while q_total >= 2:
        q = min(10, q_total)
        y = _decimate(y, q, ftype = 'fir', zero_phase = True, axis = -1)
        dt_out *= q
        q_total //= q
    return y, dt_out
