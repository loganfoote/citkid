"""
Tests of the decimation helper.
"""
import numpy as np
import pytest
from citkid.signal.basic_filt import decimate
from citkid.signal.psd import get_psd
from citkid.xcal.reduced_params import get_sxx_reduced


def test_decimate_keeps_low_frequency_spectrum():
    """
    Decimating to a low rate keeps Sxx well below the new Nyquist
    frequency (here 10 Hz at a 400 Hz rate from 20 kHz), takes several
    steps of at most 10, and leaves a rate at or below the wanted one
    alone.
    """
    rng = np.random.default_rng(1)
    dt = 5e-5
    T = 2_000_000
    x = rng.normal(0, 1., T) + 5 * np.sin(2 * np.pi * 10. * np.arange(T) * dt)
    y, dt_out = decimate(x, dt, 400.)
    assert dt_out == pytest.approx(dt * 50) and len(y) == T // 50
    f, p = get_psd(x, dt, return_freqs = True)
    fy, py = get_psd(y, dt_out, return_freqs = True)
    assert get_sxx_reduced(fy, py, 10.) == pytest.approx(
        get_sxx_reduced(f, p, 10.), rel = 0.15)
    # Two timestreams at once, along the last axis
    y2, _ = decimate(np.stack([x, 2 * x]), dt, 400.)
    assert y2.shape == (2, T // 50)
    np.testing.assert_allclose(y2[1], 2 * y2[0])
    same, dt_same = decimate(x, dt, 1e5)
    assert dt_same == dt and np.array_equal(same, x)
