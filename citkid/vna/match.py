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
    Pair resonances by an optimal one-to-one nearest-frequency matching.

    After adding ``offset`` to list 2, the matching minimizes the total
    distance of the pairs plus a penalty of ``threshold / 2`` for every
    unmatched resonance, so two resonances are paired only if they are
    closer than ``threshold``. Unlike a greedy pass, a resonance whose
    nearest partner is better used elsewhere can pair with its next nearest
    one; and unlike maximizing the number of pairs, a resonance missing from
    each list doesn't shift every pair between the two gaps by one neighbor.
    Frequencies lie on a line, so the optimal matching never crosses and is
    found exactly by dynamic programming (like sequence alignment).

    Parameters:
    fres1 (array-like, float64): resonance frequencies of list 1 in Hz.
    fres2 (array-like, float64): resonance frequencies of list 2 in Hz.
    offset (float): frequency offset in Hz added to fres2 before matching,
        i.e. the expected fres1 - fres2. Default is 0 (no offset).
    threshold (float or None): maximum distance in Hz between paired
        resonances. If None (default), 3 times the median spacing of fres1
        (of fres2 if fres1 has fewer than 2 resonances; no limit if both
        do). With no limit, as many resonances as possible are paired.

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
    n1, n2 = len(fres1_s), len(fres2_s)

    if threshold is None:
        if n1 > 1:
            threshold = 3.0 * float(np.median(np.diff(fres1_s)))
        elif n2 > 1:
            threshold = 3.0 * float(np.median(np.diff(fres2_s)))
        else:
            threshold = np.inf

    matched1 = np.zeros(n1, dtype = bool)
    matched2 = np.zeros(n2, dtype = bool)
    pairs = np.zeros((0, 2), dtype = np.int64)
    if not (n1 and n2):
        return pairs, idx1, idx2

    # Cost of leaving one resonance unmatched. Without a threshold, it exceeds
    # any total pair distance, so the most pairs are made.
    if np.isfinite(threshold):
        penalty = threshold / 2
    else:
        span = max(fres1_s[-1], fres2_s[-1]) - min(fres1_s[0], fres2_s[0])
        penalty = (span + 1.0) * (n1 + n2)

    # cost[i, j] of the best matching of fres1_s[:i] and fres2_s[:j]. Moves
    # into (i, j): 0 = pair i-1 with j-1, 1 = leave i-1, 2 = leave j-1.
    steps = np.arange(n2 + 1) * penalty
    cost_prev = steps.copy()
    move = np.empty((n1 + 1, n2 + 1), dtype = np.int8)
    move[0, :] = 2
    for i in range(1, n1 + 1):
        dist = np.abs(fres2_s - fres1_s[i - 1])
        pair = np.where(dist < threshold, cost_prev[:-1] + dist, np.inf)
        leave1 = cost_prev[1:] + penalty
        best = np.empty(n2 + 1)
        best[0] = i * penalty
        best[1:] = np.minimum(pair, leave1)
        move_i = np.where(pair <= leave1, 0, 1).astype(np.int8)
        # leaving fres2_s[j - 1] chains along the row: cummin of best - j * p.
        # Compare with a tolerance so round-off doesn't fake a cheaper chain.
        chain = steps + np.minimum.accumulate(best - steps)
        use_chain = chain < best - 1e-9 * penalty
        cost_cur = np.where(use_chain, chain, best)
        move[i, 0] = 1
        move[i, 1:] = np.where(use_chain[1:], 2, move_i)
        cost_prev = cost_cur

    # Trace the optimal moves back from (n1, n2)
    out = []
    i, j = n1, n2
    while i > 0 or j > 0:
        m = move[i, j]
        if m == 0:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif m == 1:
            i -= 1
        else:
            j -= 1
    if out:
        rows, cols = np.array(out[::-1]).T
        pairs = np.stack([idx1[rows], idx2[cols]], axis = 1).astype(np.int64)
        matched1[rows] = True
        matched2[cols] = True
    return pairs, idx1[~matched1], idx2[~matched2]
