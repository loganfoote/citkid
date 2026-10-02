import numpy as np
from scipy.constants import h


def nep_fit_mask(power, nep, p_min=None, nep_err=None):
    """
    Select the points used to fit the photon-noise NEP.

    Parameters:
    power (array-like): incident power (W).
    nep (array-like): NEP referred to incident power (W / Hz^0.5).
    p_min (float or None): minimum power of the photon-noise dominated
        regime; points with power < p_min are excluded. None uses every
        power.
    nep_err (array-like or None): NEP uncertainties. If given, points
        without a finite, positive uncertainty are excluded too.

    Returns:
    mask (np.array): boolean mask of points that are finite, have
        power > 0 and nep > 0, and have power >= p_min.
    """
    power = np.asarray(power, dtype=float)
    nep = np.asarray(nep, dtype=float)
    with np.errstate(invalid='ignore'):
        mask = np.isfinite(power) & np.isfinite(nep) & (power > 0) & (nep > 0)
        if p_min is not None:
            mask &= power >= p_min
        if nep_err is not None:
            nep_err = np.asarray(nep_err, dtype=float)
            mask &= np.isfinite(nep_err) & (nep_err > 0)
    return mask


def fit_nep_photon(power, nep, nu, p_min=None, nep_err=None):
    """
    Fit NEP versus incident power to the photon-noise NEP to get the optical
    efficiency.

    The model is nep = sqrt(2 h nu P / eta) (nep_photon), fit in log space:
    each point gives an estimate eta_i = 2 h nu P_i / nep_i^2, and eta is
    their (weighted) geometric mean. This is the least-squares solution if
    every point has the same fractional NEP uncertainty, as it does when Sxx
    and the responsivity each have a constant fractional uncertainty.

    Parameters:
    power (array-like): incident power (W). NaN points are ignored.
    nep (array-like): NEP referred to incident power (W / Hz^0.5). NaN points
        are ignored.
    nu (float): photon frequency (Hz).
    p_min (float or None): minimum power of the photon-noise dominated
        regime; only points with power >= p_min are fit. None fits every
        point.
    nep_err (array-like or None): NEP uncertainties (W / Hz^0.5). If given,
        each point is weighted by (nep / nep_err)^2 and eta_err comes from
        these uncertainties. If None (default), points are weighted equally
        and eta_err comes from the scatter of the points about the model.

    Returns:
    eta (float): optical efficiency, or NaN if no points are usable.
    eta_err (float): uncertainty in eta, or NaN if it can't be estimated
        (no usable points, or a single point without nep_err).
    mask (np.array): boolean mask of the points used in the fit.
    n_fit (int): number of points used in the fit.

    Notes:
    Why the fit is in log space with equal weights: NEP = sqrt(Sxx) / R, and
    Sxx (a PSD estimate with a fixed number of averages) and R each have
    roughly the same fractional uncertainty at every power. Then
    sigma_NEP / NEP = sqrt((sigma_S / 2 S)^2 + (sigma_R / R)^2) is the same
    for every point, i.e. sigma(log nep) is constant, so equally weighted
    least squares in log(nep) is the correct (maximum-likelihood) fit.
    Ordinary least squares on nep instead assumes a constant absolute
    uncertainty, which overweights the high-power (high-NEP) points. If the
    per-point uncertainties are known (e.g. propagated from Sxx and R), pass
    ``nep_err``: each point is then weighted by 1 / sigma(log eta_i)^2 =
    (nep / (2 nep_err))^2, which reduces to the equal-weight fit when the
    fractional uncertainties are equal.

    eta_err is statistical only. A common error in the responsivity (e.g.
    R at every power from one responsivity fit) scales every NEP together and
    shifts eta without adding scatter, so it is not included.
    """
    power = np.asarray(power, dtype=float)
    nep = np.asarray(nep, dtype=float)
    mask = nep_fit_mask(power, nep, p_min, nep_err)
    n_fit = int(np.sum(mask))
    eta, eta_err = np.nan, np.nan
    if n_fit:
        # log(eta_i) for each point; the model is log(eta) = constant.
        # Log space because the NEP uncertainties are fractional (see Notes).
        log_eta = np.log(2 * h * nu * power[mask]) - 2 * np.log(nep[mask])
        if nep_err is None:
            # Equal fractional NEP uncertainties: equal weights in log space.
            log_eta_fit = np.mean(log_eta)
            if n_fit > 1:
                log_eta_err = np.std(log_eta, ddof=1) / np.sqrt(n_fit)
            else:
                log_eta_err = np.nan
        else:
            frac = np.asarray(nep_err, dtype=float)[mask] / nep[mask]
            weights = 1 / (2 * frac) ** 2  # sigma(log eta_i) = 2 nep_err / nep
            log_eta_fit = np.sum(weights * log_eta) / np.sum(weights)
            log_eta_err = 1 / np.sqrt(np.sum(weights))
        eta = float(np.exp(log_eta_fit))
        eta_err = float(eta * log_eta_err)
    return eta, eta_err, mask, n_fit
