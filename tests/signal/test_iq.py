import pytest
import numpy as np
from citkid.signal.iq import density_subsample


################################################################################
# density_subsample
################################################################################

def _make_iq(n, seed=0):
    """Generate a complex IQ array of length n."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
        np.complex128
    )


def test_short_array_returned_unchanged():
    """If len(z) <= n_keep, the input is returned unchanged."""
    z = _make_iq(100)
    out = density_subsample(z, n_keep=200)
    np.testing.assert_array_equal(out, z)


def test_exact_length_returned_unchanged():
    """If len(z) == n_keep, the input is returned unchanged."""
    z = _make_iq(500)
    out = density_subsample(z, n_keep=500)
    np.testing.assert_array_equal(out, z)


def test_output_length():
    """Output has exactly n_keep points when len(z) > n_keep."""
    z = _make_iq(10000)
    out = density_subsample(z, n_keep=500)
    assert len(out) == 500


def test_output_is_subset():
    """Every output point is an element of the input."""
    z = _make_iq(5000)
    out = density_subsample(z, n_keep=200)
    for pt in out:
        assert pt in z


def test_output_dtype():
    """Output dtype is complex128."""
    z = _make_iq(2000)
    out = density_subsample(z, n_keep=100)
    assert out.dtype == np.complex128


def test_no_duplicates():
    """No duplicate points in the output (sampling without replacement)."""
    z = _make_iq(5000)
    out = density_subsample(z, n_keep=500)
    assert len(np.unique(out)) == len(out)


def test_reproducible_with_seed():
    """Same seed produces identical output."""
    z = _make_iq(5000)
    out1 = density_subsample(z, n_keep=200, seed=42)
    out2 = density_subsample(z, n_keep=200, seed=42)
    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_differ():
    """Different seeds produce different outputs."""
    z = _make_iq(5000)
    out1 = density_subsample(z, n_keep=200, seed=0)
    out2 = density_subsample(z, n_keep=200, seed=1)
    assert not np.array_equal(out1, out2)


def test_list_input_accepted():
    """Accepts list input, not just numpy arrays."""
    z = list(_make_iq(2000))
    out = density_subsample(z, n_keep=100)
    assert len(out) == 100
    assert out.dtype == np.complex128


def test_n_bins_parameter():
    """Different n_bins values produce valid output of the correct length."""
    z = _make_iq(5000)
    for n_bins in [10, 50, 200]:
        out = density_subsample(z, n_keep=300, n_bins=n_bins)
        assert len(out) == 300


def test_tail_preservation():
    """
    Sparse-tail regions should be represented in the output.
    Create a distribution with 95 % of points clustered near zero and 5 % far
    out.  Even at aggressive subsampling the tail points should survive.
    """
    rng = np.random.default_rng(0)
    cluster = (rng.standard_normal(9500) + 1j * rng.standard_normal(9500)) * 0.1
    tail = (rng.standard_normal(500) + 1j * rng.standard_normal(500)) * 10.0
    z = np.concatenate([cluster, tail])

    out = density_subsample(z, n_keep=500, seed=0)
    # At least some tail points (|z| > 5) should appear in the output
    assert np.any(np.abs(out) > 5.0)


def test_n_keep_one():
    """Edge case: n_keep=1 returns a single point."""
    z = _make_iq(1000)
    out = density_subsample(z, n_keep=1)
    assert len(out) == 1
