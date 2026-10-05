"""Tests for citkid.signal.deglitch."""

import numpy as np

from citkid.signal.deglitch import (
    find_glitch_idxs,
    get_binned_baseline,
    replace_glitches_with_gaussian_noise,
)


def _glitchy_timestream(seed=0):
    """
    Build a noisy, drifting timestream with three positive glitches.

    Parameters:
    seed (int): random seed.

    Returns:
    ts (np.array): timestream.
    dt (float): sample time in s.
    glitch_idxs (list of int): glitch sample indices.
    """
    rng = np.random.default_rng(seed)
    n, dt = 20_000, 1e-3
    t = np.arange(n) * dt
    ts = 0.5 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.01, n)
    glitch_idxs = [3000, 9000, 15000]
    for idx in glitch_idxs:
        ts[idx:idx + 20] += 1.0 * np.exp(-np.arange(20) / 4)
    return ts, dt, glitch_idxs


def test_get_binned_baseline_follows_slow_drift():
    """
    Check that the baseline tracks a slow sinusoid but not the noise.
    """
    ts, dt, _ = _glitchy_timestream()
    baseline = get_binned_baseline(ts, dt, 0.5)
    drift = 0.5 * np.sin(2 * np.pi * 0.05 * np.arange(len(ts)) * dt)
    assert baseline.shape == ts.shape
    assert np.max(np.abs(baseline - drift)[1000:-1000]) < 0.02


def test_find_glitch_idxs_finds_each_glitch():
    """
    Check that every injected glitch is found and nothing else is.
    """
    ts, dt, glitch_idxs = _glitchy_timestream()
    idxs, baseline = find_glitch_idxs(
        ts, dt, dtbin=0.5, nstd=8, distance=50, width=None, nrounds=2,
        i0=10, i1=40)
    assert baseline.shape == ts.shape
    assert sorted(idxs) == glitch_idxs


def test_replace_glitches_masks_windows_and_keeps_other_samples():
    """
    Check the masked windows and that untouched samples are only
    median-subtracted.
    """
    rng = np.random.default_rng(1)
    ts = 3.0 + rng.normal(0, 0.01, 20_000)
    glitch_idxs = [3000, 9000, 15000]
    for idx in glitch_idxs:
        ts[idx:idx + 20] += 1.0
    clean, masked = replace_glitches_with_gaussian_noise(
        ts, glitch_idxs, i0=5, i1=30)

    expected = np.unique(np.concatenate(
        [np.arange(i - 5, i + 30) for i in glitch_idxs]))
    np.testing.assert_array_equal(masked, expected)
    keep = np.setdiff1d(np.arange(len(ts)), masked)
    np.testing.assert_allclose(clean[keep], (ts - np.median(ts))[keep])
    assert np.max(np.abs(clean[masked])) < 0.1
    assert ts[glitch_idxs[0]] > 3.5  # input not modified


def test_replace_glitches_clips_windows_at_edges():
    """
    Check that windows at the start and end are clipped to the array.
    """
    ts = np.zeros(100)
    _, masked = replace_glitches_with_gaussian_noise(ts, [2, 98], 5, 5)
    np.testing.assert_array_equal(
        masked, np.concatenate([np.arange(0, 7), np.arange(93, 100)]))
