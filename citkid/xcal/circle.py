import numpy as np 
from scipy import optimize
from numba import njit, float64
from ..noise.psd import get_psd

################################################################################
################################ Circle fitting ################################
################################################################################
@njit(float64(float64[:], float64[:], float64[:]), cache = True)
def circle_objective(params, x, y):
    """
    Objective for circle fitting.

    Parameters:
    params (A: float, B: float, R: float): circle fit parameters. (A, B) is the
        origin and R is the radius
    x (np.array, float64): x data
    y (np.array, float64): y data

    Returns:
    error (float): error for minimization
    """
    A, B, R = params
    error = sum(((x - A) ** 2 + (y - B) ** 2 - R ** 2) ** 2)
    return error

def fit_iq_circle(z, mask = None):
    """
    Fits an IQ loop to a circle. The function describing the circle is

       [Re(S21)-A]^2 + [Im(S21)-B]^2 = R^2

       where the origin is (A, B) and the radius is R.

    Parameters:
    z (np.array, complex128): complex S21 data.
    mask (np.array, bool or None): mask of z to use for fitting. If None,
        all data points are used.

    Returns:
    origin (complex): center of the fitted circle.
    radius (float): radius of the fitted circle.
    """
    # Input validation and masking
    z = np.asarray(z, dtype = np.complex128)
    if mask is not None:
        mask = np.asarray(mask, dtype = np.bool_)
        z = z[mask]
    if not np.all(np.isfinite(z)):
        raise ValueError("Input data contains non-finite values.")
    if not len(z) >= 3:
        raise ValueError("At least 3 data points are required for fitting.")
    
    # Get x0
    i, q = z.real, z.imag
    x0 = [(max(i) + min(i))/2, (max(q) + min(q))/2]
    x0.append((max(i) - min(i) + max(q) - min(q)) / 4)

    # Perform minimization
    popt = optimize.fmin(circle_objective, x0, (i, q), disp = 0)

    # Decompose popt
    origin = popt[0] + 1j * popt[1]
    radius = popt[2]
    return origin, radius

################################################################################
###################### Convert complex S21 to theta and A ######################
################################################################################
def cent_rot_s21(z, center, phase):
    """
    Apply centering and rotation to S21 data.

    Parameters:
    z (array-like, complex128): complex S21 data with gain removed.
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
    z (array-like, complex128): complex S21 data after centering and rotation. 
        Must be sorted in ascending or descending order of frequency.
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
    z (array-like, complex128): complex S21 data after centering and rotation.

    Returns:
    (array-like, float64): amplitudes.
    """
    z = np.asarray(z, dtype = np.complex128)
    return np.abs(z)

################################################################################
############################### Spar and Sper ##################################
################################################################################
def get_spar_sper(theta, A, radius, dt, get_freqs = True):
    """
    Calculate the PSDs of parallel and perpendicular noise components.
    
    Parameters:
    theta (array-like, float64): phase noise timestream in radians.
    A (array-like, float64): amplitude noise timestream in normalized units.
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
    # Input validation
    theta = np.asarray(theta, dtype = np.float64)
    A = np.asarray(A, dtype = np.float64)
    if not np.isfinite(radius): 
        raise ValueError("radius must be a finite number")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be a positive finite number")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains non-finite values")
    if theta.shape != A.shape:
        raise ValueError("theta and A must have the same shape")

    # Compute PSDs
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