import numpy as np 
from .psd import get_psd

def cent_rot_s21(z, center, phase):
    """
    Apply centering and rotation to S21 data.

    Parameters:
    z (array-like, complex128): complex S21 data points.
    center (complex): IQ circle center for centering the data.
    phase (float): phase angle in radians for rotation.

    Returns:
    (array-like, complex128): centered and rotated S21 data points.
    """
    z = np.asarray(z, dtype = np.complex128)
    return (z - center) * np.exp(-1j * phase)

def convert_to_theta(z, unwrap = False):    
    """
    Convert complex S21 data points to phase angles (theta).

    Parameters:
    z (array-like, complex128): complex S21 data points.
    unwrap (bool): whether to unwrap the phase angles.

    Returns:
    (array-like, float64): phase angles in radians.
    """
    z = np.asarray(z, dtype = np.complex128)
    theta = np.angle(z)
    if unwrap:
        theta = np.unwrap(theta)
    return theta

def convert_to_A(z):
    """
    Convert complex S21 data points to amplitude (A).

    Parameters:
    z (array-like, complex128): complex S21 data points.

    Returns:
    (array-like, float64): amplitudes.
    """
    z = np.asarray(z, dtype = np.complex128)
    return np.abs(z)

def get_spar_sper(theta, A, radius, dt = 1, get_freqs = True):
    """
    Calculate the PSDs of parallel and perpendicular noise components.
    
    Parameters:
    theta (array-like, float64): noise phase angle in radians.
    A (array-like, float64): noise amplitude.
    radius (float): radius of the IQ circle.
    dt (float): sample time in s. 
    get_freqs (bool): whether to return the frequency array.
    
    Returns:    
    (tuple): tuple containing:
        - frequency (array-like, float64): PSD frequencies in Hz if get_freqs,
                                           else None.
        - spar (array-like, float64): PSD of parallel noise component in dBc.
        - sper (array-like, float64): PSD of perpendicular noise component in 
                                      dBc.
    """
    theta = np.asarray(theta, dtype = np.float64)
    A = np.asarray(A, dtype = np.float64)

    assert np.isfinite(radius), "radius must be a finite number"
    assert np.isfinite(dt) and dt > 0, "dt must be a positive finite number"
    assert np.all(np.isfinite(theta)), "theta contains non-finite values"
    assert np.all(np.isfinite(A)), "A contains non-finite values"
    assert theta.shape == A.shape, "theta and A must have the same shape"

    # compute PSDs
    spar = get_psd(theta * radius, dt, get_frequencies = False)
    if get_freqs:
        freq, sper = get_psd(A, dt, get_frequencies = True)
    else:
        sper = get_psd(A, dt, get_frequencies = False)
        freq = None
    
    # Convert to dBc
    spar = 10 * np.log10(spar)
    sper = 10 * np.log10(sper)
    
    return freq, spar, sper