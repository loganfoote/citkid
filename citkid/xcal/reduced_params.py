import numpy as np

def get_sxx_reduced(f, sxx, freq):
    """ 
    Calculate the mean value of the Sxx at frequencies within 20% of the 
    given freq. 

    Parameters:
    f (array-like, float): frequencies in Hz. 
    sxx (array-like, float): PSD of x. 
    freq (float): frequency in Hz to find the Sxx around.

    Returns:
    sxx_reduced (float): mean value of Sxx at frequencies within 20% of freq. 
    """
    # Input validation - check freq is scalar
    if isinstance(freq, (list, tuple, np.ndarray)):
        raise ValueError("freq must be a positive scalar.")
    
    f = np.asarray(f, dtype = np.float64) 
    sxx = np.asarray(sxx, dtype = np.float64)
    
    try:
        freq = float(np.asarray(freq))
    except (TypeError, ValueError):
        raise ValueError("freq must be a positive scalar.")
    
    if f.shape != sxx.shape:
        raise ValueError("f and sxx must have the same shape.")
    if freq <= 0:
        raise ValueError("freq must be a positive scalar.") 
    
    # Calculate mask
    fmin, fmax = freq * 0.8, freq * 1.2 
    if fmin < np.min(f[f > 0]) or fmax > np.max(f):
        raise ValueError(
            "Invalid freq: freq +- 20% must be within the range of f."
            ) 
    mask = (f > fmin) & (f < fmax)

    # Apply mask 
    sxx_reduced = np.mean(sxx[mask])
    return sxx_reduced

def get_sfactor(f, spar, sper, freq):
    """
    Get the difference spar - sper (log-scaled) within 20% of the given freq. 

    Parameters:
    f (array-like, float): frequencies in Hz. 
    spar (array-like, float): PSD of parallel normalized voltage in dB. 
    sper (array-like, float): PSD of perpendicular normalized voltage in dB. 
    freq (float): frequency in Hz to find the mean Sxx around.

    Returns:
    sfactor (float): mean value of spar - sper at frequencies within 20% of 
        freq. 
    """
    # Input validation - check freq is scalar
    if isinstance(freq, (list, tuple, np.ndarray)):
        raise ValueError("freq must be a positive scalar.")
    
    f = np.asarray(f, dtype = np.float64) 
    spar = np.asarray(spar, dtype = np.float64)
    sper = np.asarray(sper, dtype = np.float64)
    
    try:
        freq = float(np.asarray(freq))
    except (TypeError, ValueError):
        raise ValueError("freq must be a positive scalar.")
    
    if f.shape != spar.shape or f.shape != sper.shape:
        raise ValueError("f, spar, and sper must have the same shape.")
    if freq <= 0:
        raise ValueError("freq must be a positive scalar.") 
    
    # Calculate mask
    fmin, fmax = freq * 0.8, freq * 1.2 
    if fmin < np.min(f[f > 0]) or fmax > np.max(f):
        raise ValueError(
            "Invalid freq: freq +- 20% must be within the range of f."
            ) 
    mask = (f > fmin) & (f < fmax)

    # Apply mask 
    sfactor = np.mean(spar[mask] - sper[mask])
    return sfactor

################################################################################
################################ Default freqs #################################
################################################################################
_freqs = [0.1, 0.3, 1, 3, 10, 30, 100, 300]

def default_freq_keys(prefix):
    """
    Return the names of the reduced parameters at the default frequencies.

    Parameters:
    prefix (str): Name prefix, e.g. 'sxx' or 'sfactor'.

    Returns:
    keys (list of str): f'{prefix}_{freq}' for each default frequency in
        order, with '.' replaced by 'p' (e.g. 'sxx_0p1', 'sxx_10').
    """
    return [f'{prefix}_{freq}'.replace('.', 'p') for freq in _freqs]

def get_sxx_reduced_default_freqs(f, sxx):
    """
    Get the mean value of Sxx at frequencies within 20% of each of the 
    default frequencies: 0.1, 0.3, 1, 3, 10, 30, 100, and 300 Hz. 

    Parameters:
    f (array-like, float): frequencies in Hz. 
    sxx (array-like, float): PSD of x. 

    Returns:
    sxx_reduced (dict): mean value of Sxx within 20% of each default
        frequency, with keys from ``default_freq_keys('sxx')``: 'sxx_0p1',
        'sxx_0p3', 'sxx_1', etc.
    """
    def _compute(freq):
        """
        Compute the reduced Sxx at one frequency.

        Parameters:
        freq (float): frequency in Hz.

        Returns:
        sxx_freq (float): mean Sxx near freq, or NaN if there is no data.
        """
        try:
            return get_sxx_reduced(f, sxx, freq)
        except ValueError:
            return np.nan
    return dict(zip(default_freq_keys('sxx'), map(_compute, _freqs)))

def get_sfactor_reduced_default_freqs(f, spar, sper):
    """
    Get the mean value of spar - sper at frequencies within 20% of each of 
    the default frequencies: 0.1, 0.3, 1, 3, 10, 30, 100, and 300 Hz. 

    Parameters:
    f (array-like, float): frequencies in Hz. 
    spar (array-like, float): PSD of parallel normalized voltage in dB. 
    sper (array-like, float): PSD of perpendicular normalized voltage in dB. 

    Returns:
    sfactor_reduced (dict): mean value of spar - sper within 20% of each
        default frequency, with keys from ``default_freq_keys('sfactor')``:
        'sfactor_0p1', 'sfactor_0p3', 'sfactor_1', etc.
    """
    def _compute(freq):
        """
        Compute the reduced spar - sper at one frequency.

        Parameters:
        freq (float): frequency in Hz.

        Returns:
        sfactor_freq (float): mean spar - sper near freq, or NaN if there is
            no data.
        """
        try:
            return get_sfactor(f, spar, sper, freq)
        except ValueError:
            return np.nan
    return dict(zip(default_freq_keys('sfactor'), map(_compute, _freqs)))
