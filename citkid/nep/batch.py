import numpy as np
from tqdm.auto import tqdm
from .fitter import fit_nep_photon


def per_set_values(value, n_sets, name):
    """
    Expand a scalar to one value per set, or check a per-set list.

    Parameters:
    value (float, array-like, or None): one value for every set, or one per
        set.
    n_sets (int): number of sets.
    name (str): parameter name, for the error message.

    Returns:
    values (list): one value per set.

    Raises:
    ValueError: if a list doesn't have one value per set.
    """
    if value is None or np.ndim(value) == 0:
        return [value] * n_sets
    values = list(value)
    if len(values) != n_sets:
        raise ValueError(f'{name} must be a scalar or have one value per set '
                         f'({n_sets}); got {len(values)}')
    return values


def per_set_powers(powers, neps):
    """
    Return one power array per set, expanding powers shared by every set.

    Parameters:
    powers (array-like or list of array-like): one 1-D array of powers for
        every set, or one array per set.
    neps (list of array-like or np.ndarray): NEPs of each set (one per set,
        or a 2-D array with one row per set).

    Returns:
    powers (list of np.ndarray or array-like): one entry per set (the shared
        array repeated, or ``powers`` as given).
    """
    try:
        shared = np.asarray(powers, dtype=float)
    except (TypeError, ValueError):  # sets of different lengths
        return list(powers)
    if shared.ndim == 1:
        return [shared] * len(neps)
    return list(powers)


def check_sets(powers, neps, nep_errs=None):
    """
    Check that the power, NEP (and NEP uncertainty) lists match.

    Parameters:
    powers (list of array-like): incident powers of each set (W), as
        returned by ``per_set_powers``.
    neps (list of array-like): NEPs of each set (W / Hz^0.5).
    nep_errs (list of array-like or None): NEP uncertainties of each set, or
        None.

    Raises:
    ValueError: if the lists have different lengths, or a set's arrays have
        different shapes.
    """
    if len(powers) != len(neps):
        raise ValueError(f'powers and neps must have the same number of sets; '
                         f'got {len(powers)} and {len(neps)}')
    if nep_errs is not None and len(nep_errs) != len(neps):
        raise ValueError(f'nep_errs must have one entry per set ({len(neps)}); '
                         f'got {len(nep_errs)}')
    for i, (power, nep) in enumerate(zip(powers, neps)):
        shapes = {np.shape(power), np.shape(nep)}
        if nep_errs is not None:
            shapes.add(np.shape(nep_errs[i]))
        if len(shapes) != 1:
            raise ValueError(f'set {i}: power, nep (and nep_err) must have the '
                             f'same shape; got {sorted(shapes)}')


def fit_nep_photon_sets(powers, neps, nu, p_min=None, nep_errs=None, progress=True):
    """
    Fit the photon-noise NEP of each of several sets.

    Parameters:
    powers (array-like or list of array-like): incident powers (W): one
        1-D array shared by every set (e.g. powers of length M with neps
        of shape (N, M)), or one array per set.
    neps (list of array-like): NEPs of each set (W / Hz^0.5).
    nu (float): photon frequency (Hz).
    p_min (float, array-like, or None): minimum fitted power, one for every
        set or one per set. None fits every point.
    nep_errs (list of array-like or None): NEP uncertainties of each set, or
        None (default) to weight points equally.
    progress (bool): If True (default), show a progress bar.

    Returns:
    eta (np.array): optical efficiency of each set, NaN for sets without
        usable points.
    eta_err (np.array): uncertainty in eta of each set (see
        ``fit_nep_photon``).
    n_fit (np.array): number of points fit in each set.

    Raises:
    ValueError: if the inputs don't have one entry per set.
    """
    powers = per_set_powers(powers, neps)
    check_sets(powers, neps, nep_errs)
    n_sets = len(neps)
    p_mins = per_set_values(p_min, n_sets, 'p_min')
    eta = np.full(n_sets, np.nan)
    eta_err = np.full(n_sets, np.nan)
    n_fit = np.zeros(n_sets, dtype=int)
    for i in tqdm(range(n_sets), total=n_sets, leave=False, disable=not progress):
        err = None if nep_errs is None else nep_errs[i]
        eta[i], eta_err[i], _, n_fit[i] = fit_nep_photon(powers[i], neps[i], nu, p_mins[i], err)
    return eta, eta_err, n_fit
