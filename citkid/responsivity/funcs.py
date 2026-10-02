import numpy as np
from numba import njit, float64

def responsivity(P, R_0, P_0):
    """
    Calculate the responsivity dx / dP of a KID

    Parameters:
    P (float or array-like): power (W)
    R_0 (float): responsivity at P = 0 (1 / W)
    P_0 (float): power at which the responsivity is rolled off by sqrt(2) from
        its value at P = 0

    Returns:
    (float or array-like): responsivity corresponding to the value(s) in P
    """
    if P_0 == 0:
        return np.nan
    return - R_0 * (1 + P / P_0) ** (-0.5)
responsivity = np.vectorize(responsivity)

@njit(float64[:](float64[:], float64, float64, float64), cache = True)
def responsivity_int(P, R_0, P_0, c):
    """
    Calculate the integrated form of the responsivity. This function describes
    the behavior of x versus P.

    Parameters:
    P (np.array, float64): power (W).
    R_0 (float64): responsivity at P = 0 (1 / W).
    P_0 (float64): power at which the responsivity is rolled off by sqrt(2) from
        its value at P = 0.
    c (float64): f0 / f1, where f0 is the actual frequency at P = 0, and f1 is 
        the frequency at P = 0 that was assumed when creating the x array. If x 
        is calibrated perfectly, c = 1. In practice, there may be a small 
        offset.

    Returns:
    (np.array, float64): integrated responsivity corresponding to the value(s) 
        in P.
    """
    return 2 * c * R_0 * P_0 * ((1 + P / P_0) ** (0.5) - 1) + c - 1

@njit(float64[:](float64[:], float64, float64, float64), cache = True)
def responsivity_int_for_fitter(P, R_0, P_0, c):
    """
    Perform the same fucntion as responsivity_int, but with parameters rescaled
    for fitting.

    Parameter scaling factors:
    log(P) -> P
    1e-9 * R_0 -> R_0
    1e16 * P_0 -> P_0
    c -> c

    Return scaling factors:
    x -> 1e6 * x
    """
    return responsivity_int(np.exp(P), R_0 * 1e9, P_0 * 1e-16, c) * 1e6

@njit(float64[:](float64[:], float64, float64, float64), cache = True)
def responsivity_int_x0(P, R_0, P_0, x_0):
    """
    Calculate the integrated form of the responsivity with c = 1 and a free
    offset x_0. This function describes the behavior of x versus P.

    Parameters:
    P (np.array, float64): power (W).
    R_0 (float64): responsivity at P = 0 (1 / W).
    P_0 (float64): power at which the responsivity is rolled off by sqrt(2) from
        its value at P = 0.
    x_0 (float64): x at P = 0. If x is calibrated perfectly, x_0 = 0. The
        frequency at P = 0 is f1 * (1 + x_0), where f1 is the frequency at
        P = 0 that was assumed when creating the x array.

    Returns:
    (np.array, float64): integrated responsivity corresponding to the value(s)
        in P.
    """
    return 2 * R_0 * P_0 * ((1 + P / P_0) ** (0.5) - 1) + x_0

@njit(float64[:](float64[:], float64, float64, float64), cache = True)
def responsivity_int_x0_for_fitter(P, R_0, P_0, x_0):
    """
    Perform the same function as responsivity_int_x0, but with parameters
    rescaled for fitting.

    Parameter scaling factors:
    log(P) -> P
    1e-9 * R_0 -> R_0
    1e16 * P_0 -> P_0
    1e6 * x_0 -> x_0

    Return scaling factors:
    x -> 1e6 * x
    """
    return responsivity_int_x0(np.exp(P), R_0 * 1e9, P_0 * 1e-16,
                               x_0 * 1e-6) * 1e6
