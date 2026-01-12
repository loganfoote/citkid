import numpy as np
import numbers 
from numba import vectorize, float64, boolean
from scipy import signal
from scipy.interpolate import interp1d

def calc_qc_qi(qr, amp):
    """
    Calculates Qc and Qi from Qr and amp, where amp = Qr / Qc and
    1 / Qr = 1 / Qc + 1 / Qi.

    Parameters:
    qr (float): total quality factor.
    amp (float): Qr / Qc.

    Returns:
    qc (float): coupling quality factor.
    qi (float): internal quality factor.
    """
    # Input validation
    if not (np.isscalar(qr) and isinstance(qr, numbers.Real)):
        raise ValueError("qr must be a scalar")
    if not (np.isscalar(amp) and isinstance(amp, numbers.Real)):
        raise ValueError("amp must be a scalar") 
    if qr <= 0:
        raise ValueError("qr must be positive")
    if amp < 0 or amp > 1:
        raise ValueError("amp must be in [0, 1]")
    
    # Edge cases of amp = 0 or 1
    if amp == 0:
        return np.inf, qr 
    elif amp == 1:
        return qr, np.inf
    
    # General case
    qc = qr / amp
    qi = 1.0 / ((1.0 / qr) - (1.0 / qc))
    return qc, qi
calc_qc_qi = np.vectorize(calc_qc_qi)

def bounds_check(p0, bounds):
    """
    First flips bounds if they are reversed. Then, if p0 is not strictly within 
    the bounds, modifies bounds to be 10% lower or higher than p0.

    Parameters:
    p0 (np.array): initial guesses for all parameters.
    bounds (tuple): 2d tuple of the format (lower_bounds, upper_bounds), where 
        each is an array or list of the same length as p0.

    Returns:
    new_bounds (tuple): modified bounds.
    """
    # Input validation
    N = len(p0) 
    if not isinstance(bounds, tuple):
        raise ValueError("bounds must be a tuple")
    if len(bounds) != 2:
        raise ValueError("bounds must be a tuple of length 2")
    if len(bounds[0]) != N or len(bounds[1]) != N:
        raise ValueError("bounds must have the same length as p0")
    bounds = (bounds[0].copy(), bounds[1].copy()) # avoid modifying input
    
    # Flip bounds if they are reversed
    for i, b1, b2 in zip(range(len(bounds[0])), bounds[0], bounds[1]):
        if b1 > b2:
            bounds[0][i] = b2
            bounds[1][i] = b1

    # Make sure p0 is within bounds
    lower_bounds = []
    upper_bounds = []
    for p, lb, ub in zip(p0, bounds[0], bounds[1]):
        if p < lb:
            if p > 0:
                lower_bounds.append(p * 0.9)
            elif p == 0:
                lower_bounds.append(-1.0)
            else:
                lower_bounds.append(p * 1.1)
        else:
            lower_bounds.append(lb)
        if p > ub:
            if p > 0:
                upper_bounds.append(p * 1.1)
            elif p == 0:
                upper_bounds.append(1.0)
            else:
                upper_bounds.append(p * 0.9)
        else:
            upper_bounds.append(ub)
    return lower_bounds, upper_bounds

def calc_nrmse(z, z_fit):
    """
    Given complex S21 data and values of the fit at the same frequencies, 
    return the normalized root mean square error: 
        nrmse = ||z - z_fit||^2 / ||z||^2. 

    Parameters:
    z (np.array, complex): array of measured S21 data.
    z_fit (np.array, complex): array of fitted S21 data.

    Returns:
    nrmse (float): normalized root mean square error.
    """
    # Input validation
    z = np.asarray(z, dtype = np.complex128)
    z_fit = np.asarray(z_fit, dtype = np.complex128)
    if z.shape != z_fit.shape:
        raise ValueError("z and z_fit must have the same shape")
    
    # Calculate nrmse
    resid = z - z_fit     
    err = np.vdot(resid, resid).real
    norm = np.vdot(z, z).real
    return err / norm

@vectorize([float64(float64, float64, float64, float64, boolean)], 
           nopython = True, cache = True)
def cardan(a, b, c, d, largest = True):
    """
    Analyticaly calculates the largest or smallest real root of a 3rd-order
    polynomial using Cardan's method.

    Parameters:
    a, b, c, d (float): polynomial coefficients. 
        a x^3 + b x^2 + c x + d = 0
    largest (bool): If True, returns the largest root when applicable. 
        Otherwise, returns the smallest root.

    Returns:
    root (float): largest, smallest, or only real root.
    """
    # Calculate roots using Cardan's method
    J = np.exp(2j * np.pi / 3)
    Jc = 1 / J
    u = np.empty(2, np.complex128)
    z0 = b / 3 / a
    a2, b2 = a * a, b * b
    p = -b2 / 3 / a2 + c / a
    q = (b / 27 * (2 * b2 / a2 - 9 * c / a) + d) / a
    D = -4 * p * p * p - 27 * q * q
    r = np.sqrt(-D / 27 + 0j)
    one_third = 1 / 3.0
    u = ((-q - r) / 2) ** one_third
    v = ((-q + r) / 2) ** one_third
    w = u * v
    w0 = np.abs(w + p / 3)
    w1 = np.abs(w * J + p / 3)
    w2 = np.abs(w * Jc + p / 3)
    if w0 < w1:
        if w2 < w0:
            v *= Jc
    elif w2 < w1:
        v *= Jc
    else:
        v *= J
    roots = np.asarray((u + v - z0, u * J + v * Jc - z0, u * Jc + v * J - z0))
    # Select the appropriate root based on D and largest flag
    if D > -1e-10: # 3 real roots, D > 0 with numerical tolerance
        if largest:
            return np.max(roots.real)
        else:
            return np.min(roots.real)
    else: # 1 real root
        return roots[np.argsort(np.abs(roots.imag))][0].real

def get_peak_fwhm(x, y):
    """
    Gets the approximate index and fwhm of a peak in (x, y) data using 
    scipy.signal.find_peaks and scipy.signal.peak_widths. x data must be evenly
    sampled.

    Parameters:
    x (np.array): x data.
    y (np.array): y data.

    Returns:
    x_peak (float): x value of the peak.
    y_peak (float): y value of the peak.
    fwhm (float): width in x units.
    """
    # Input validation and sorting
    x, y = np.asarray(x, dtype = np.float64), np.asarray(y, dtype = np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 4:
        raise ValueError("x and y must not be empty")
    if not all(x[1:] >= x[:-1]):
        ix = np.argsort(x)
        x, y = x[ix], y[ix]

    # Interpolate data for higher resolution
    interp_factor = 10
    x_interp = np.linspace(min(x), max(x), len(x) * interp_factor)
    interp_func = interp1d(x, y, kind = 'cubic')
    y_interp = interp_func(x_interp)
    x, y = x_interp, y_interp

    # Find peak and its width
    peak_index, _ = signal.find_peaks(y, height = (max(y) + min(y)) / 8)
    if not len(peak_index):
        # If no peak was found, use the middle point and 1/8 the span
        peak_index = len(y) // 2
        width  = len(y) / 8
    else:
        # if peaks were found, use the highest peak
        ix = np.argsort(y[peak_index])
        peak_index = peak_index[ix[-1]]
        width = signal.peak_widths(y, [peak_index], rel_height = 0.5)[0][0]

    # Calculate FWHM and return results
    fwhm = np.median(x[1:] - x[:-1]) * width
    return x[peak_index], y[peak_index], fwhm
