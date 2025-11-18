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

def get_xcal_ix(ffine, tfine, tnoise, ix0_offset = 7, ix1_offset = 5,
                std_cutoff = 12):
    """
    Get incides of the fine s21 sweep over which x vs theta should be fit to 
    produce the x calibration. Chooses the indices where tfine (after glitch 
    removal) falls within the min and max values of tnoise, with optional 
    offsets.

    Parameters:
    ffine (array-like, float64): frequency values from the fine s21 sweep in Hz.
    tfine (array-like, float64): theta values from the fine s21 sweep.
    tnoise (array-like, float64): theta values from the noise measurement.
    ix0_offset (int): number of indices to offset below the start index.
        The data is sorted such that the start index corresponds to the lowest
        frequency and the end index corresponds to the highest frequency.
    ix1_offset (int): number of indices to offset above the end index.
    std_cutoff (float or None): number of standard deviations from the mean to 
            use as a cutoff to apply to tnoise before determining the max and 
            min values. If None, no cutoff is applied.

    Returns:
    ix (array-like, int64): indices of tfine.
    """
    # format and sort inputs
    ffine = np.asarray(ffine, dtype = np.float64)
    tfine = np.asarray(tfine, dtype = np.float64)
    tnoise = np.asarray(tnoise, dtype = np.float64)
    assert (type(ix0_offset) == int) and (type(ix1_offset) == int)
    ix = np.argsort(ffine)
    ffine, tfine = ffine[ix], tfine[ix] 

    # apply cutoff to tnoise
    if std_cutoff is not None:
        # determine signal cutoff
        tnoise_std = np.std(tnoise)
        tnoise_mean = np.mean(tnoise)
        tnoise = tnoise[np.abs(tnoise - tnoise_mean) < std_cutoff * tnoise_std]
    tmin, tmax = np.min(tnoise), np.max(tnoise)

    # determine indices
    ix0 = np.where(tfine - tmin >= 0)[0]
    if len(ix0):
        ix0 = ix0[0] - 1
    else:
        ix0 = 0
    ix1 = np.where(tfine - tmax > 0)[0]
    if len(ix1):
        ix1 = ix1[0]
    else:
        ix1 = len(tfine) - 1
    ix0 = max(0, ix0 - ix0_offset)
    ix1 = min(len(tfine) - 1, ix1 + ix1_offset)
    if ix1 < ix0:
        ix1 = ix0 - 1
    ix = np.arange(ix0, ix1 + 1, 1, dtype = np.int32)
    return ix
