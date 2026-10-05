"""Tests for citkid.res.generator (S21 sweep and noise generation)."""

import numpy as np
import pytest

from citkid.res.generator import (
    block_mean,
    generate_noise,
    get_S21_from_xA,
    get_S21_noise_ts,
    get_S21_vs_freq,
    get_S21_vs_freq_dual,
)

FR = 1e8
P = np.array([FR, 2e4, 0.6, 0.1, 0.0])  # [fr, Qr, amp, phi, a]


def _psd(x, fs):
    """
    Compute the one-sided PSD of ``x`` with the normalization the generator
    is scaled for.

    Parameters:
    x (np.array): real timestream.
    fs (float): sampling frequency in Hz.

    Returns:
    f (np.array): frequencies in Hz.
    psd (np.array): one-sided PSD in 1 / Hz.
    """
    n = len(x)
    return np.fft.rfftfreq(n, 1 / fs), 2 * np.abs(np.fft.rfft(x)) ** 2 / fs / n


def test_generate_noise_shapes_and_zero_mean():
    """
    Check output lengths and that the DC bin is removed.
    """
    x, a = generate_noise(1000, 100.0, 1.0, 1.0, -1.0, -1.0, 1e-16, 1e-10)
    assert x.shape == (1000,) and a.shape == (1000,)
    assert abs(x.mean()) < 1e-12
    assert abs(a.mean()) < 1e-12


def test_generate_noise_recovers_white_levels():
    """
    Check that white-noise-only timestreams have the requested PSD levels.
    """
    sxx, saa, fs = 4e-17, 2.5e-11, 1e3
    x, a = generate_noise(2 ** 16, fs, 1.0, 0.0, -1.0, -1.0, sxx, saa)
    _, px = _psd(x, fs)
    _, pa = _psd(a, fs)
    assert np.mean(px[1:-1]) == pytest.approx(sxx, rel=0.05)
    assert np.mean(pa[1:-1]) == pytest.approx(saa, rel=0.05)


def test_generate_noise_rolloffs_reduce_high_frequency_power():
    """
    Check that the QP and ringdown rolloffs suppress power above the corners.
    """
    fs, n = 1e4, 2 ** 15
    tau = 1 / (2 * np.pi * 50.0)  # 50 Hz corner
    x, a = generate_noise(n, fs, 1.0, 0.0, tau, tau, 1e-16, 1e-10)
    f, px = _psd(x, fs)
    _, pa = _psd(a, fs)
    low, high = (f > 1) & (f < 10), f > 2000
    assert np.mean(px[high]) < 1e-3 * np.mean(px[low])
    assert np.mean(pa[high]) < 1e-2 * np.mean(pa[low])


def test_block_mean_of_constant_blocks_and_total_mean():
    """
    Check constant input stays constant and the overall mean is preserved.
    """
    nsamps, m = 4, 6
    np.testing.assert_allclose(
        block_mean(np.full(nsamps * m, 2.5), nsamps, m), np.full(m, 2.5))
    x = np.random.default_rng(0).normal(size=nsamps * m)
    out = block_mean(x, nsamps, m)
    assert out.shape == (m,)
    assert out.mean() == pytest.approx(x.mean())


def test_get_S21_from_xA_without_noise_matches_linear_model():
    """
    Check the noiseless linear resonance: dip at fr and unity far away.
    """
    f = np.array([FR, FR * 1.1])
    s21 = get_S21_from_xA(f, P, 0.0, 0.0)
    amp, phi = P[2], P[3]
    assert s21[0] == pytest.approx(1 - amp / np.cos(phi) * np.exp(1j * phi))
    assert abs(s21[1] - 1) < 1e-3


def test_get_S21_from_xA_frequency_noise_shifts_resonance():
    """
    Check that positive x moves the resonance up in frequency.
    """
    p = P.copy()
    p[3] = 0.0  # symmetric dip, so the |S21| minimum is at fr
    f = FR * (1 + np.linspace(-1e-4, 1e-4, 401))
    x = np.full(len(f), 2e-5)
    s21 = get_S21_from_xA(f, p, x, np.zeros(len(f)))
    f_min = f[np.argmin(np.abs(s21))]
    assert f_min == pytest.approx(FR * (1 + 2e-5), rel=1e-6)


def test_get_S21_vs_freq_without_noise_is_noiseless_model():
    """
    Check that zero noise levels reproduce the noiseless sweep.
    """
    f = FR * (1 + np.linspace(-1e-4, 1e-4, 50))
    s21 = get_S21_vs_freq(f, 1.0, 0.0, -1.0, 0.0, 0.0, 1e3, 4, P)
    np.testing.assert_allclose(s21, get_S21_from_xA(f, P, 0.0, 0.0))


def test_get_S21_vs_freq_dual_reduces_to_single_far_from_second():
    """
    Check that a distant second resonator barely changes the first.
    """
    f = FR * (1 + np.linspace(-1e-4, 1e-4, 50))
    p2 = P.copy()
    p2[0] = 2 * FR
    dual = get_S21_vs_freq_dual(
        f, 1.0, 0.0, -1.0, 0.0, 0.0, 1e3, 4, P, p2, 0.0)
    single = get_S21_from_xA(f, P, 0.0, 0.0)
    np.testing.assert_allclose(dual, single, atol=1e-3)


def test_get_S21_noise_ts_without_noise_is_constant():
    """
    Check the timestream length and that zero noise gives a constant S21.
    """
    s21 = get_S21_noise_ts(
        FR, 256, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1e3, P)
    assert s21.shape == (256,)
    expected = get_S21_from_xA(np.array([FR]), P, 0.0, 0.0)[0]
    np.testing.assert_allclose(s21, expected)
