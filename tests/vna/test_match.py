import numpy as np
import pytest
from citkid.vna.match import match_nearest, match_sorted

def _grid(n = 20, seed = 0):
    rng = np.random.default_rng(seed)
    return 4e9 + np.cumsum(rng.uniform(0.5e6, 1.5e6, n))

def test_match_sorted_pairs_by_rank():
    f1 = np.array([3., 1., 2.])
    f2 = np.array([20., 10.])
    pairs, un1, un2 = match_sorted(f1, f2)
    np.testing.assert_array_equal(pairs, [[1, 1], [2, 0]])
    np.testing.assert_array_equal(un1, [0])
    assert len(un2) == 0

def test_match_sorted_empty():
    pairs, un1, un2 = match_sorted([], [1., 2.])
    assert pairs.shape == (0, 2) and len(un1) == 0
    np.testing.assert_array_equal(un2, [0, 1])

def test_match_nearest_identical_lists():
    f = _grid()
    perm = np.random.default_rng(1).permutation(len(f))
    pairs, un1, un2 = match_nearest(f, f[perm])
    np.testing.assert_array_equal(f[pairs[:, 0]], f[perm][pairs[:, 1]])
    assert len(pairs) == len(f) and len(un1) == 0 and len(un2) == 0

@pytest.mark.parametrize("missing", [5, 30, 70])
def test_match_nearest_missing_resonance_does_not_shift_pairs(missing):
    # A missing resonance must not shift every later pair by one neighbor
    # (the old rank-based median offset did), or give its neighbor's
    # partner away (greedy matching did)
    f1 = _grid(100)
    f2 = np.delete(f1, missing) + 50e3
    pairs, un1, un2 = match_nearest(f1, f2, offset = -50e3)
    correct = np.isclose(f2[pairs[:, 1]] - f1[pairs[:, 0]], 50e3)
    assert correct.sum() == 99
    np.testing.assert_array_equal(un1, [missing])

def test_match_nearest_no_offset_by_default():
    f1 = _grid(10)
    f2 = f1 + 50e3  # small next to the ~1 MHz spacing
    pairs, _, _ = match_nearest(f1, f2)
    np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])

def test_match_nearest_threshold():
    f1 = np.array([1e9, 2e9])
    f2 = np.array([1e9 + 10., 2e9 + 1e3])
    pairs, un1, un2 = match_nearest(f1, f2, threshold = 100.)
    np.testing.assert_array_equal(pairs, [[0, 0]])
    np.testing.assert_array_equal(un1, [1])
    np.testing.assert_array_equal(un2, [1])

def test_match_nearest_one_partner_for_two_candidates():
    f1 = np.array([1.0e9, 1.0e9 + 10.])
    f2 = np.array([1.0e9 + 4.])
    pairs, un1, un2 = match_nearest(f1, f2, threshold = 1e3)
    np.testing.assert_array_equal(pairs, [[0, 0]])
    np.testing.assert_array_equal(un1, [1])

def test_match_nearest_uses_next_nearest_when_nearest_is_taken():
    """
    101 MHz's nearest partner (100.6) is better used by 100 MHz, so 101
    pairs with 102 instead of being left unmatched (greedy matching did).
    """
    f1 = np.array([100.0, 101.0]) * 1e6
    f2 = np.array([100.6, 102.0]) * 1e6
    pairs, un1, un2 = match_nearest(f1, f2, threshold = 3e6)
    np.testing.assert_array_equal(pairs, [[0, 0], [1, 1]])
    assert len(un1) == 0 and len(un2) == 0

def test_match_nearest_minimizes_total_distance():
    """
    With a shifted second list, every resonance pairs with its shifted
    copy, not with a nearer neighbor of another resonance.
    """
    f1 = np.array([100.0, 100.5, 101.0, 101.5]) * 1e6
    f2 = f1 + 0.2e6
    pairs, _, _ = match_nearest(f1, f2, threshold = 1e6)
    np.testing.assert_array_equal(pairs, [[0, 0], [1, 1], [2, 2], [3, 3]])

def test_match_nearest_empty():
    pairs, un1, un2 = match_nearest([1e9, 2e9], [])
    assert pairs.shape == (0, 2)
    np.testing.assert_array_equal(un1, [0, 1])

def test_match_nearest_gaps_in_both_lists_do_not_shift_pairs():
    """
    One resonance missing from each list at different places: the
    resonances between the gaps keep their true partners instead of each
    shifting by one neighbor to make one more pair.
    """
    f = _grid(10)
    f1 = np.delete(f, 3)
    f2 = np.delete(f, 7)
    pairs, un1, un2 = match_nearest(f1, f2)
    np.testing.assert_array_equal(f1[pairs[:, 0]], f2[pairs[:, 1]])
    assert len(pairs) == 8
    np.testing.assert_array_equal(f1[un1], [f[7]])
    np.testing.assert_array_equal(f2[un2], [f[3]])

def test_match_nearest_no_threshold_pairs_as_many_as_possible():
    f1 = np.array([1e9, 2e9, 3e9])
    f2 = np.array([5e9, 9e9])
    pairs, un1, un2 = match_nearest(f1, f2, threshold = np.inf)
    assert len(pairs) == 2 and len(un1) == 1 and len(un2) == 0

@pytest.mark.parametrize("seed", range(40))
def test_match_nearest_matches_brute_force(seed):
    """
    The dynamic program finds the minimum of total pair distance plus
    threshold / 2 per unmatched resonance (checked against all matchings).
    """
    import itertools
    rng = np.random.default_rng(seed)
    f1 = np.sort(rng.uniform(0, 10, 5))
    f2 = np.sort(rng.uniform(0, 10, 4))
    thr = 3.0
    pairs, un1, un2 = match_nearest(f1, f2, threshold = thr)
    got = (np.abs(f1[pairs[:, 0]] - f2[pairs[:, 1]]).sum()
           + thr / 2 * (len(un1) + len(un2)))

    best = np.inf
    for k in range(0, 5):
        for s1 in itertools.combinations(range(5), k):
            for s2 in itertools.permutations(range(4), k):
                d = np.abs(f1[list(s1)] - f2[list(s2)])
                if np.all(d < thr):
                    best = min(best, d.sum() + thr / 2 * (9 - 2 * k))
    assert got == pytest.approx(best)
