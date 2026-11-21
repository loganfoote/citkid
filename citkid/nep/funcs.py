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


def nep_from_sxx(sxx, power, R0, P0):
    """
    Compute the NEP referred to incident power from the fractional
    frequency noise and the fitted responsivity: sqrt(Sxx) / |R(P)| with
    R(P) = R0 (1 + P / P0)^(-1/2) (``responsivity.funcs.responsivity``).

    Parameters:
    sxx (array-like): Sxx at the NEP frequency at each power (1/Hz).
    power (array-like): the incident powers (W).
    R0 (float): responsivity at P = 0 (1/W); negative for a KID. NaN, or
        zero or positive (x rising with power: not a KID response), gives
        NaN everywhere; check the sign with ``responsivity_ok``.
    P0 (float): roll-off power of the responsivity (W).

    Returns:
    nep (np.array): NEP at each power (W/Hz^0.5), NaN where Sxx or the
        responsivity is not finite.
    """
    sxx = np.asarray(sxx, dtype = float)
    power = np.asarray(power, dtype = float)
    if not responsivity_ok(R0, P0):
        return np.full(sxx.shape, np.nan)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        resp = R0 * (1 + power / P0) ** -0.5
        return np.sqrt(sxx) / np.abs(resp)

def responsivity_ok(R0, P0):
    """
    Say whether a responsivity fit describes a KID: finite, with R0 < 0
    (x falls as the power rises) and P0 > 0. The fitter does not force
    the sign of R0, so a tone whose x rises with power fits with R0 > 0
    and must be marked bad rather than used through |R0|.

    Parameters:
    R0 (float): responsivity at P = 0 (1/W).
    P0 (float): roll-off power (W).

    Returns:
    ok (bool): True if the fit is usable.
    """
    return bool(np.isfinite(R0) and np.isfinite(P0) and R0 < 0 and P0 > 0)

def nu_per_tone(nu, res_idxs):
    """
    Give every tone its photon frequency: one value for all, or a dict of
    resonator index to frequency with a default under the key 'default'
    (or None) for the rest.

    Parameters:
    nu (float or dict): the frequency (Hz), or dict(res_idx: Hz, ...)
        with the optional 'default' key.
    res_idxs (array-like): the tones' resonator indices.

    Returns:
    nus (np.array): the frequency of each tone (Hz), NaN for a tone the
        dict leaves out without a default.

    Raises:
    ValueError: if ``nu`` is neither a number nor a dict.
    """
    res_idxs = np.asarray(res_idxs, dtype = int)
    if isinstance(nu, dict):
        default = nu.get('default', nu.get(None, np.nan))
        return np.array([float(nu.get(int(r), default)) for r in res_idxs])
    try:
        return np.full(len(res_idxs), float(nu))
    except (TypeError, ValueError):
        raise ValueError('nu must be a frequency in Hz or a dict of '
                         'resonator index to frequency')
