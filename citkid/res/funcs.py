import numpy as np
from numba import njit, float64, complex128, boolean
from .util import cardan

@njit(float64[:](float64[:], float64, boolean), cache = True)
def get_y(y0, a, largest = True):
    """
    Calculates the largest or smallest real root of
        y0 = y - a / (1 + y^2).

    Parameters:
    y0 (np.array): resonance shift in the low-power and linear limit.
    a (float): nonlinearity parameter.
    largest (bool): If True, returns the largest root. Otherwise, returns the
        smallest root.

    Returns:
    y (float or np.array): largest or smallest real root of the above equation.
    """
    y = cardan(4.0, -4.0 * y0, 1.0, -(y0 + a), largest)
    # Built-in numpy function can do this, but it is not compilable with numba
    return y

@njit(complex128[:](float64[:], float64, float64, float64, float64,
                 float64, float64, float64, float64, boolean), cache = True)
def nonlinear_iq(f, fr, Qr, amp, phi, a, i0, q0, tau, downward = True):
    r"""
    Describes the transmission through a nonlinear resonator

                    (-j*2*pi*f*tau)    /                           (j phi)   \
        (i0+j*q0)*e^                * |1 -        Qr             e^           |
                                      |     --------------  X  ------------   |
                                       \     Qc * cos(phi)       (1+ 2jy)    /

        where the nonlinearity of y is described by
            y0 = y - a/(1+y^2)
        and y0 = Qr*x0, where x0 = (f - fr) / fr is the fractional frequency shift in 
        the low-power, linear limit.

    Parameters:
    f (np.array): array of frequencies in Hz.
    fr (float): resonant frequency in the zero uW power limit, in Hz.
    Qr (float): total quality factor.
    amp (float): Qr / Qc, where Qc is the coupling quality factor.
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit.
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl.
    i0 (float): I gain factor.
    q0 (float): Q gain factor.
        i0 + j * q0 describes the overall constant gain and phase offset.
    tau(float): cable delay in seconds
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    z (np.array): array of complex IQ data corresponding to f.
    """
    deltaf = f - fr
    y0 = Qr * deltaf / fr
    y = get_y(y0, a, downward)
    s21_readout = (i0 + 1.j * q0) * np.exp(-2.j * np.pi * deltaf * tau)
    s21_res = (1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y))
    z = s21_readout * s21_res
    return z

@njit(float64(float64[:], float64[:], float64[:]), cache = True)
def circle_objective(params, x, y):
    """
    Objective for circle fitting. Legacy code: use 
    citkid.xcal.circle.circle_objective.

    Parameters:
    params (A: float, B: float, R: float): circle fit parameters. (A, B) is the
        origin and R is the radius.
    x (np.array): x data.
    y (np.array): y data.

    Returns:
    error (float): error for minimization.
    """
    A, B, R = params
    error = sum(((x - A) ** 2 + (y - B) ** 2 - R ** 2) ** 2)
    return error

################################################################################
####################### nonlinear_iq for fitter ################################
################################################################################
@njit(float64[:](float64[:], float64, float64, float64, float64,
                 float64, float64, float64, float64, boolean), cache = True)
def nonlinear_iq_for_fitter(f, fr, Qr, amp, phi, a, i0, q0, tau,
                            downward = True):
    """
    Same as nonlinear_iq, but returns stacked real and imaginary components
    for the fitter. The input data should be scaled as follows
    fr X 100^6
    Qr X 10^-4
    tau * 1e6.
    """
    z = nonlinear_iq(f, fr / 100e-6, Qr / 1e-4, amp, phi, a, i0, q0, tau / 1e6,
                     downward)
    return np.hstack((np.real(z), np.imag(z)))
