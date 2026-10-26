"""
Pair resonances between two resonance lists, e.g. the same device measured
in two full VNA sweeps.

Each function returns index arrays into the input lists, so the caller can
build its own groups from them.
"""
import numpy as np


def match_sorted(fres1, fres2):
    """
    Pair resonances by their rank in frequency order.

    Parameters:
    fres1 (array-like, float64): resonance frequencies of list 1 in Hz.
    fres2 (array-like, float64): resonance frequencies of list 2 in Hz.

    Returns:
    pairs (np.array, int, (K, 2)): indices (i1, i2) into fres1 and fres2 of
        the paired resonances, in frequency order. K = min(len(fres1),
        len(fres2)).
    unmatched1 (np.array, int): indices into fres1 of the highest-frequency
        resonances left over, in frequency order.
    unmatched2 (np.array, int): same for fres2.
    """
    idx1 = np.argsort(np.asarray(fres1, dtype = np.float64))
    idx2 = np.argsort(np.asarray(fres2, dtype = np.float64))
    n_common = min(len(idx1), len(idx2))
    pairs = np.stack([idx1[:n_common], idx2[:n_common]], axis = 1)
    return pairs, idx1[n_common:], idx2[n_common:]


def match_nearest(fres1, fres2, offset = 0., threshold = None):
    """
    Pair resonances by greedy nearest-neighbor matching.

    In frequency order, each resonance of list 1 is paired with the nearest
    resonance of list 2 (after adding ``offset`` to list 2), if that one is
    closer than ``threshold`` and not already paired. Otherwise it is left
    unmatched.

    Parameters:
    fres1 (array-like, float64): resonance frequencies of list 1 in Hz.
    fres2 (array-like, float64): resonance frequencies of list 2 in Hz.
    offset (float): frequency offset in Hz added to fres2 before matching,
        i.e. the expected fres1 - fres2. Default is 0 (no offset).
    threshold (float or None): maximum distance in Hz between paired
        resonances. If None (default), 3 times the median spacing of fres1
        (of fres2 if fres1 has fewer than 2 resonances; no limit if both
        do).

    Returns:
    pairs (np.array, int, (K, 2)): indices (i1, i2) into fres1 and fres2 of
        the paired resonances, in order of fres1 frequency.
    unmatched1 (np.array, int): indices into fres1 of unpaired resonances,
        in frequency order.
    unmatched2 (np.array, int): same for fres2.
    """
    fres1 = np.asarray(fres1, dtype = np.float64)
    fres2 = np.asarray(fres2, dtype = np.float64)
    idx1 = np.argsort(fres1)
    idx2 = np.argsort(fres2)
    fres1_s = fres1[idx1]
    fres2_s = fres2[idx2] + offset

    if threshold is None:
        if len(fres1_s) > 1:
            threshold = 3.0 * float(np.median(np.diff(fres1_s)))
        elif len(fres2_s) > 1:
            threshold = 3.0 * float(np.median(np.diff(fres2_s)))
        else:
            threshold = np.inf

    matched1 = np.zeros(len(idx1), dtype = bool)
    matched2 = np.zeros(len(idx2), dtype = bool)
    pairs = []
    if len(fres2_s):
        for i, f1 in enumerate(fres1_s):
            dists = np.abs(fres2_s - f1)
            j = int(np.argmin(dists))
            if dists[j] < threshold and not matched2[j]:
                pairs.append((idx1[i], idx2[j]))
                matched1[i] = True
                matched2[j] = True
    pairs = np.array(pairs, dtype = np.int64).reshape(-1, 2)
    return pairs, idx1[~matched1], idx2[~matched2]
