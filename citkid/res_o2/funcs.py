import numpy as np
from numba import jit, vectorize
from .util import real_only

# @jit(nopython=True)
def nonlinear_iq(f, fr, Qr, amp, phi, a, b, i0, q0, tau, downward = True):
    r"""
    Describes the transmission through a nonlinear resonator

                    (-j*2*pi*f*tau)    /                           (j phi)   \
        (i0+j*q0)*e^                * |1 -        Qr             e^           |
                                      |     --------------  X  ------------   |
                                       \     Qc * cos(phi)       (1+ 2jy)    /

        where the nonlinearity of y is described by
                   16y^5 + 8y^3 - 4ay^2 + y - a - b
            y0 =  ----------------------------------
                          16y^4 + 8y^2 + 1
        and y0 = Qr*x0, where x0 is the fractional frequency shift in the
        low-power, linear limit.

    Parameters:
    f (np.array): array of frequencies in Hz.
    fr (float): resonance frequency in Hz.
    Qr (float): total quality factor.
    amp (float): Qr / Qc, where Qc is the coupling quality factor.
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit.
    a (float): 1st-order nonlinearity parameter.
    b (float): 2nd-order nonlinearity parameter.
    i0 (float): I gain factor.
    q0 (float): Q gain factor.
        i0 + j * q0 describes the overall constant gain and phase offset
    tau(float): cable delay in seconds.
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    z (np.array): array of complex IQ data corresponding to f.
    """
    deltaf = f - fr
    y0 = Qr * deltaf / fr
    y = get_y(y0, a, b, downward)
    s21_readout = (i0 + 1.j * q0) * np.exp(-2.j * np.pi * deltaf * tau)
    s21_res = (1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y))
    z = s21_readout * s21_res
    return z

@jit(nopython=True)
def circle_objective(params, x, y):
    """
    Objective for circle fitting.

    Parameters:
    params (A: float, B: float, R: float): circle fit parameters. (A, B) is the
        origin and R is the radius.
    x (np.array): x data.
    y (np.array): y data.

    Returns:
    error (float): error for minimization.
    """
    A, B, R = params
    error = sum(((x - A) ** 2+(y - B) ** 2 - R ** 2) ** 2)
    return error

################################################################################
######################### Utility functions ####################################
################################################################################
# @vectorize(nopython=True)
def _get_y_scalar(y0, a, b, largest = True):
    """
    Calculates the largest or smallest real root of
               16y^5 + 8y^3 - 4ay^2 + y - a - b
        y0 =  ----------------------------------.
                      16y^4 + 8y^2 + 1

    Parameters:
    y0 (float or np.array): resonance shift in the low-power and linear limit.
    a (float): first-order nonlinearity parameter.
    b (float): second-order nonlinearity parameter.
    largest (bool): If True, returns the largest root. Otherwise, returns the
        smallest root.

    Returns:
    y (float or np.array): largest or smallest real root of the above equation.
    """
    p = np.polynomial.Polynomial([-(a + b + y0), 1, -4*(a + 2*y0),
                                  8, -16 * y0, 16])
    all_roots = p.roots()
    real_roots = real_only(all_roots)
    if not len(real_roots):
        return np.nan
    if largest:
        return max(real_roots)
    else:
        return min(real_roots)
get_y = np.vectorize(_get_y_scalar, excluded = ['a', 'b', 'largest'])

################################################################################
######################## Equations for fitter ##################################
################################################################################
# @jit(nopython=True)
def nonlinear_iq_for_fitter(f, fr, Qr, amp, phi, a, b, i0, q0, tau,
                            downward = True):
    """
    Same as nonlinear_iq, but returns stacked real and imaginary components
    for the fitter. The input data should be scaled as follows:
    fr X 100^6
    Qr X 10^-4
    tau * 1e6.
    """
    z = nonlinear_iq(f, fr / 100e-6, Qr / 1e-4, amp, phi, a, b, i0, q0,
                     tau / 1e6, downward)
    return np.hstack((np.real(z), np.imag(z)))
