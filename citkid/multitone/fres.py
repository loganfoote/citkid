import numpy as np

def update_fres(fs, zs, fres, qres, res_idxs, method='distance'):
    """
    Update resonant frequencies given fine sweep data.

    Calibration tones (``res_idxs[i] < 0``) are left at their input
    frequency and are not passed to the update algorithm.

    Parameters:
    fs (array-like): Fine sweep frequency data in Hz, shape (M, N).
    zs (array-like): Fine sweep complex S21 data, shape (M, N).
    fres (array-like): Resonant frequencies in Hz, length M.
    qres (array-like): Q-factors, length M.
    res_idxs (array-like): Resonator indices, length M.  Entries with
        values < 0 are treated as calibration tones.
    method (str): Algorithm used to locate the resonance.  'mins21' finds
        the minimum of |S21| after subtracting a linear baseline.  'spacing'
        finds the point of maximum adjacent IQ spacing.  'distance' finds
        the point furthest from the off-resonance IQ value.  'none' returns
        fres unchanged.

    Returns:
    fres_new (np.ndarray): Updated resonant frequencies in Hz, length M.
    """
    # Input validation 
    fres = np.asarray(fres,    dtype=float)
    if method == 'none': # Return fres unchanged
        return fres.copy()
    
    fs   = np.asarray(fs,      dtype=float)
    zs   = np.asarray(zs)
    qres = np.asarray(qres,    dtype=float)
    res_idxs = np.asarray(res_idxs)

    # Select update method
    if method == 'mins21':
        update = update_fr_minS21
    elif method == 'spacing':
        update = update_fr_spacing
    elif method == 'distance':
        update = update_fr_distance
    else:
        raise ValueError(
            "method must be 'mins21', 'distance', 'spacing', or 'none'."
            )

    # Apply update to on-resonance tones only
    res_mask = res_idxs >= 0                          # True for resonators
    fres_new = fres.copy()
    if res_mask.any():
        fres_new[res_mask] = update(fs[res_mask], zs[res_mask])
    return fres_new

def update_fr_minS21(f, z):
    """
    Return the resonance frequency estimated as the minimum of |S21| after
    subtracting a linear baseline.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    """
    f = np.asarray(f, dtype=float)
    z = np.asarray(z)
    single = f.ndim == 1
    if single:
        f = f[np.newaxis]
        z = z[np.newaxis]
    dB = 20.0 * np.log10(np.abs(z))                          # M x N
    # Vectorised linear detrend along axis=1
    f_c  = f - f.mean(axis=1, keepdims=True)
    denom = np.sum(f_c ** 2, axis=1, keepdims=True)
    slope = np.sum(f_c * dB,  axis=1, keepdims=True) / denom
    trend = slope * f_c + dB.mean(axis=1, keepdims=True)
    ix = np.argmin(dB - trend, axis=1)                       # M
    fr = f[np.arange(len(f)), ix]                            # M
    return float(fr[0]) if single else fr

def update_fr_spacing(f, z):
    """
    Return the resonance frequency estimated as the point at which the sum of
    adjacent IQ segment lengths is largest.  This quantity peaks where the IQ
    trajectory moves fastest, which corresponds to the resonance.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    """
    f = np.asarray(f, dtype=float)
    z = np.asarray(z)
    single = f.ndim == 1
    if single:
        f = f[np.newaxis]
        z = z[np.newaxis]
    d     = np.abs(np.diff(z, axis=1))              # M x (N-1)
    score = d[:, 1:] + d[:, :-1]                    # M x (N-2)
    score = np.pad(score, ((0, 0), (1, 1)))         # M x N, edges = 0
    ix = np.argmax(score, axis=1)                   # M
    fr = f[np.arange(len(f)), ix]                   # M
    return float(fr[0]) if single else fr

def update_fr_distance(f, z):
    """
    Return the resonance frequency estimated as the point of greatest distance
    from the off-resonance IQ value.  The off-resonance reference is the mean
    of the first and last ``min(10, N//4)`` samples of each sweep.  Performance
    improves when cable delay has been removed beforehand.

    Accepts a single resonator (1-D arrays of length N) or a batch of
    resonators (2-D arrays of shape (M, N)).

    Parameters:
    f (np.ndarray): Frequency data in Hz, shape (N,) or (M, N).
    z (np.ndarray): Complex S21 data, shape (N,) or (M, N).

    Returns:
    fr (float or np.ndarray): Updated resonance frequency in Hz.  Returns a
        scalar for 1-D input and an array of length M for 2-D input.
    """
    f = np.asarray(f, dtype=float)
    z = np.asarray(z)
    single = f.ndim == 1
    if single:
        f = f[np.newaxis]
        z = z[np.newaxis]
    N = z.shape[1]
    n_edge = max(1, min(10, N // 4))
    offres = np.concatenate(
        [z[:, :n_edge], z[:, -n_edge:]], axis=1
    ).mean(axis=1, keepdims=True)                   # M x 1
    ix = np.argmax(np.abs(z - offres), axis=1)      # M
    fr = f[np.arange(len(f)), ix]                   # M
    return float(fr[0]) if single else fr