import numpy as np
from scipy.constants import h


def nep_photon(power, eta, nu):
    """
    Compute the photon-noise NEP referred to incident power at one frequency.

    Parameters:
    power (float or array-like): incident power (W).
    eta (float): optical efficiency (power absorbed / incident power).
    nu (float): photon frequency (Hz).

    Returns:
    nep (float or np.array): photon-noise NEP sqrt(2 h nu P / eta)
        (W / Hz^0.5), NaN where power < 0.
    """
    power = np.asarray(power, dtype=float)
    with np.errstate(invalid='ignore'):
        return np.sqrt(2 * h * nu * power / eta)
