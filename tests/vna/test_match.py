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
    # (the old rank-based median offset did). Greedy matching can still
    # give the missing resonance's neighbor away, so allow 2 bad pairs.
    f1 = _grid(100)
    f2 = np.delete(f1, missing) + 50e3
    pairs, un1, un2 = match_nearest(f1, f2, offset = -50e3)
    correct = np.isclose(f2[pairs[:, 1]] - f1[pairs[:, 0]], 50e3)
    assert correct.sum() >= 97

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

def test_match_nearest_taken_match_leaves_unmatched():
    f1 = np.array([1.0e9, 1.0e9 + 10.])
    f2 = np.array([1.0e9 + 5.])
    pairs, un1, un2 = match_nearest(f1, f2, threshold = 1e3)
    np.testing.assert_array_equal(pairs, [[0, 0]])
    np.testing.assert_array_equal(un1, [1])

def test_match_nearest_empty():
    pairs, un1, un2 = match_nearest([1e9, 2e9], [])
    assert pairs.shape == (0, 2)
    np.testing.assert_array_equal(un1, [0, 1])
