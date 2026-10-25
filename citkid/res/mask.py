import numpy as np

def nearby_tone_mask(f, f_tone, f_others, min_frac = 0.25):
    """
    Mask a sweep to exclude the halves nearer to neighboring tones.

    On each side of ``f_tone``, the sweep is cut at the midpoint between
    ``f_tone`` and the nearest other tone on that side, so a neighboring
    resonance doesn't enter the fit. A side with no other tone isn't cut.
    If the window keeps fewer than ``min_frac`` of the points (close tones on
    both sides), both of its edges are moved outward by the same frequency
    until it does, so the sweep is never fully masked.

    Parameters:
    f (np.array): sweep frequencies in Hz, sorted ascending.
    f_tone (float): frequency of this sweep's tone in Hz.
    f_others (np.array): frequencies of the other tones in Hz. NaN entries
        and entries equal to ``f_tone`` are ignored.
    min_frac (float): fewest points to keep, as a fraction of the sweep.
        Default 0.25. At least 3 points (or every point, if fewer) are kept.

    Returns:
    mask (np.array, bool): True for the points to fit.
    """
    f = np.asarray(f, dtype = float)
    f_tone = float(f_tone)
    others = np.asarray(f_others, dtype = float).ravel()
    others = others[np.isfinite(others) & (others != f_tone)]
    below = others[others < f_tone]
    above = others[others > f_tone]
    lo = (f_tone + below.max()) / 2 if below.size else -np.inf
    hi = (f_tone + above.min()) / 2 if above.size else np.inf
    mask = (f >= lo) & (f <= hi)

    n_min = min(len(f), max(3, int(np.ceil(min_frac * len(f)))))
    if mask.sum() < n_min:
        # How far each point is outside the window; keep the n_min closest,
        # i.e. widen both edges equally
        outside = np.maximum(np.maximum(lo - f, f - hi), 0)
        outside[~np.isfinite(f)] = np.inf
        keep = np.argsort(outside, kind = 'stable')[:n_min]
        mask = np.zeros(len(f), dtype = bool)
        mask[keep] = True
    return mask
