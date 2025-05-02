import numpy as np
from numba import jit, vectorize
from scipy import stats
from scipy import signal
from scipy.interpolate import interp1d

def calc_qc_qi(qr, amp):
    """
    Calculates Qc and Qi from Qr and amp, where amp = Qr / Qc and
    1 / Qr = 1 / Qc + 1 / Qi

    Parameters:
    qr (float): total quality factor
    amp (float): Qr / Qc

    Returns:
    qc (float): coupling quality factor
    qi (float): internal quality factor
    """
    qc = qr / amp
    qi = 1.0 / ((1.0 / qr) - (1.0 / qc))
    return qc, qi

def bounds_check(p0, bounds):
    """
    Checks that p0 is strictly within the bounds. If not, modifies bounds to be
    10% lower or higher than p0.

    Parameters:
    p0(np.array): initial guesses for all parameters
    bounds (tuple): 2d tuple of low values bounds[0] the high values bounds[1]
        to bound the fitting problem

    Returns:
    new_bounds (tuple): modified bounds
    """
    # First, flip bounds if they are reversed
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
            else:
                lower_bounds.append(p * 1.1)
        else:
            lower_bounds.append(lb)
        if p > ub:
            if p > 0:
                upper_bounds.append(p * 1.1)
            else:
                upper_bounds.append(p * 0.9)
        else:
            upper_bounds.append(ub)
    return lower_bounds, upper_bounds

@jit(nopython=True)
def calculate_residuals(z, z_fit):
    """
    Given IQ data and fitted IQ data, return the chi squared value and p value

    Parameters:
    z (np.array, complex): array of measured S21 data
    z_fit (np.array, complex): array of fitted S21 data

    Returns:
    chi_sq (float): chi squared value
    """
    z = np.hstack((np.real(z), np.imag(z)))
    z_fit = np.hstack((np.real(z_fit), np.imag(z_fit)))
    res = np.sqrt(sum((z - z_fit) ** 2)) / len(z)
    return res

@vectorize(nopython=True)
def cardan(a, b, c, d, largest = True):
    """
    Analyticaly calculates the largest or smallest real root of a 3rd-order
    polynomial. Based on code from
        https://github.com/Wheeler1711/submm_python_routines

    Parameters:
    a, b, c, d (float): polynomial coefficients
    largest (bool): If True, returns the largest root. Otherwise, returns the
        smallest root

    Returns:
    root (float): largest or smallest real root
    """
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
    where_real = np.where(np.abs(np.imag(roots)) < 1e-15)
    if D > 0:
        # two stable real roots: return the max. The other solution is the min
        # max solution corresponds to a downward sweep. An upward sweep is more
        # complicated to calculate. It will stay at the min to a certain point,
        # then jump to the max. See Swenson et al. 2013 for details
        if largest:
            return np.max(np.real(roots))
        else:
            return np.min(np.real(roots))
    else:
        # one real root: return value with smallest imaginary component
        return np.real(roots[np.argsort(np.abs(np.imag(roots)))][0])

def get_peak_fwhm(x, y):
    """
    Gets the index and fwhm of a peak in (x, y) data. x data must be evenly
    sampled.

    Parameters:
    x (np.array): x data
    y (np.array): y data

    Returns:
    peak_index (int): index of the peak
    fwhm (float): width in x units
    """
    x, y = np.asarray(x), np.asarray(y)
    ix = np.argsort(x)
    x, y = x[ix], y[ix]
    interp_factor = 10
    x_interp = np.linspace(min(x), max(x), len(x) * interp_factor)
    interp_func = interp1d(x, y, kind = 'cubic')
    y_interp = interp_func(x_interp)
    x, y = x_interp, y_interp

    peak_index, _ = signal.find_peaks(y, height = (max(y) + min(y)) / 8)
    if not len(peak_index):
        peak_index = len(y) // 2
        width  = len(y) / 8 # Need to modify this later
    else:
        peak_index = peak_index[len(peak_index) // 2]
        width = signal.peak_widths(y, [peak_index], rel_height = 0.5)[0][0]

    fwhm = np.median(x[1:] - x[:-1]) * width
    return x[peak_index], y[peak_index], fwhm
