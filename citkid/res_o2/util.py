import numpy as np
from numba import jit

@jit(nopython=True)
def real_only(a, tol = 1e-8):
    """
    Returns an array of the values in a that are real within the specified
    tolerance.

    Parameters:
    a (array-like, numeric): array of complex values

    Returns:
    (np.array, float): array of the real values in a.
    """
    a = np.asarray(a)
    mask = np.abs(a.imag) < tol
    return a[mask].real
