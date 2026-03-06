"""
Tests for s21_filt.py — standalone S21 filtering utilities.

Tests cover:
- Output shape and dtype
- highpass_filter: slow baseline removed, fast features preserved
- highpass_filter: cutoff clamping for extreme values
- polynomial_baseline: exact polynomial removed
- polynomial_baseline: different orders
- Edge cases: constant signal, single-resonance data
"""

import numpy as np
import pytest

from citkid.vna.s21_filt import highpass_filter, polynomial_baseline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sweep(n=10000, f_start=4e9, f_stop=8e9):
    """Return a uniformly-spaced frequency array."""
    return np.linspace(f_start, f_stop, n)


# ---------------------------------------------------------------------------
# highpass_filter
# ---------------------------------------------------------------------------

class TestHighpassFilter:

    def test_output_shape(self):
        f = _make_sweep()
        mag = np.zeros(len(f))
        result = highpass_filter(f, mag)
        assert result.shape == mag.shape

    def test_output_dtype_float(self):
        f = _make_sweep()
        mag = np.ones(len(f))
        result = highpass_filter(f, mag)
        assert np.issubdtype(result.dtype, np.floating)

    def test_slow_baseline_removed(self):
        """A linear ramp (very slow variation) should be nearly zeroed out."""
        f = _make_sweep(n=50000)
        # Linear ramp varies on the scale of the entire 4 GHz sweep (>> cutoff)
        mag = np.linspace(-10.0, 10.0, len(f))
        result = highpass_filter(f, mag, cutoff_mhz=10.0)
        # After high-pass the mean and trend should be gone; rms should be tiny
        assert np.std(result) < 0.5  # well below original std of ~11.5

    def test_fast_features_preserved(self):
        """
        A sinusoidal wobble much faster than the cutoff should pass through.
        """
        f = _make_sweep(n=100000)
        df = f[1] - f[0]
        # Feature period = 0.5 MHz, cutoff = 10 MHz  →  well above cutoff
        feature_period_hz = 0.5e6
        mag = np.sin(2 * np.pi * f / feature_period_hz)
        result = highpass_filter(f, mag, cutoff_mhz=10.0)
        # Amplitude should be mostly preserved (within 10%)
        assert np.std(result) > 0.9 * np.std(mag)

    def test_constant_signal_zeroed(self):
        """A constant signal is pure DC and should be removed by the HP filter."""
        f = _make_sweep(n=20000)
        mag = np.full(len(f), -30.0)
        result = highpass_filter(f, mag, cutoff_mhz=10.0)
        assert np.max(np.abs(result)) < 1e-6

    def test_cutoff_clamping_low(self):
        """Extremely low cutoff should be clamped and not raise an error."""
        f = _make_sweep(n=10000)
        mag = np.random.randn(len(f))
        result = highpass_filter(f, mag, cutoff_mhz=1e-6)
        assert result.shape == mag.shape
        assert np.all(np.isfinite(result))

    def test_cutoff_clamping_high(self):
        """Very high cutoff (larger than sweep span) should clamp and not error."""
        f = _make_sweep(n=10000)
        mag = np.random.randn(len(f))
        result = highpass_filter(f, mag, cutoff_mhz=1e6)
        assert result.shape == mag.shape
        assert np.all(np.isfinite(result))

    def test_default_cutoff(self):
        """Calling without explicit cutoff uses 10.0 MHz default."""
        f = _make_sweep()
        mag = np.linspace(0, 5, len(f))
        result_default = highpass_filter(f, mag)
        result_explicit = highpass_filter(f, mag, cutoff_mhz=10.0)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_with_resonance_dips(self, synthetic_vna_data):
        """Filter should preserve resonance dips (narrow features)."""
        f = synthetic_vna_data['f']
        mag_db = 20 * np.log10(np.abs(synthetic_vna_data['z']))
        result = highpass_filter(f, mag_db, cutoff_mhz=10.0)
        # Resonance dips are narrow; filtered data should still have dips
        # Check that the dynamic range is non-trivial
        assert np.ptp(result) > 1.0  # at least 1 dB peak-to-peak


# ---------------------------------------------------------------------------
# polynomial_baseline
# ---------------------------------------------------------------------------

class TestPolynomialBaseline:

    def test_output_shape(self):
        f = _make_sweep()
        mag = np.zeros(len(f))
        result = polynomial_baseline(f, mag)
        assert result.shape == mag.shape

    def test_output_dtype_float(self):
        f = _make_sweep()
        mag = np.ones(len(f))
        result = polynomial_baseline(f, mag)
        assert np.issubdtype(result.dtype, np.floating)

    def test_exact_polynomial_removed(self):
        """When signal IS a polynomial of matching order, residual ~ 0."""
        f = _make_sweep(n=10000)
        # Cubic polynomial baseline
        mag = 2.0 - 3.0 * (f / 1e9) + 0.5 * (f / 1e9) ** 2 - 0.1 * (f / 1e9) ** 3
        result = polynomial_baseline(f, mag, order=3)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_higher_order_polynomial_removed(self):
        """Order-5 polynomial should be removed when order=5 is used."""
        f = _make_sweep(n=10000)
        coeffs = np.array([1.0, -2.0, 0.5, 0.1, -0.05, 0.01])
        mag = np.polyval(coeffs, f / 1e9)
        result = polynomial_baseline(f, mag, order=5)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_residual_preserved(self):
        """Signal added on top of polynomial baseline is preserved in residual."""
        f = _make_sweep(n=20000)
        baseline = 1.5 * (f / 1e9) - 10.0
        # Fast sinusoidal feature (resonance-like)
        feature = 2.0 * np.sin(2 * np.pi * f / 50e6)
        mag = baseline + feature
        result = polynomial_baseline(f, mag, order=1)
        # Residual should match the feature closely
        np.testing.assert_allclose(result, feature, atol=0.1)

    def test_constant_signal(self):
        """A constant signal is a degree-0 polynomial; order>=1 removes it."""
        f = _make_sweep(n=5000)
        mag = np.full(len(f), -20.0)
        result = polynomial_baseline(f, mag, order=1)
        np.testing.assert_allclose(result, 0.0, atol=1e-8)

    def test_default_order(self):
        """Calling without explicit order uses 3 (cubic) by default."""
        f = _make_sweep(n=5000)
        mag = np.random.randn(len(f))
        result_default = polynomial_baseline(f, mag)
        result_explicit = polynomial_baseline(f, mag, order=3)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_order_1(self):
        """Linear baseline (order=1) is correctly removed."""
        f = _make_sweep(n=10000)
        mag = 0.5 * (f / 1e9) - 3.0
        result = polynomial_baseline(f, mag, order=1)
        np.testing.assert_allclose(result, 0.0, atol=1e-8)

    def test_with_resonance_dips(self, synthetic_vna_data):
        """Baseline subtraction should preserve resonance dips."""
        f = synthetic_vna_data['f']
        mag_db = 20 * np.log10(np.abs(synthetic_vna_data['z']))
        result = polynomial_baseline(f, mag_db, order=3)
        # Dips should survive; dynamic range should remain non-trivial
        assert np.ptp(result) > 1.0
