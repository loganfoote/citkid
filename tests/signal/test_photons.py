"""
Tests for ``citkid.signal.photons`` on synthetic timestreams.
"""
import numpy as np
import pytest
from citkid.signal import photons as ph

DT = 1 / 38147.     # stage 0
TAU = 150e-6        # pulse decay time


def pulse_shape(n = 120):
    """
    Return a unit pulse: one-sample rise, exponential decay.

    Parameters:
    n (int): length in samples. Default 120.

    Returns:
    shape (np.array): the pulse (maximum 1 at index 0).
    """
    return np.exp(-np.arange(n) * DT / TAU)


def synth(rng, n, times, amp = 1e-6, sigma = 1e-7, sign = -1, height = None):
    """
    Make a timestream with pulses at given times in white noise.

    Parameters:
    rng (np.random.Generator): random generator.
    n (int): samples.
    times (np.array, int): pulse maxima.
    amp (float): pulse amplitude. Default 1e-6.
    sigma (float): noise sigma. Default 1e-7.
    sign (int): pulse sign. Default -1 (x of a photon).
    height (np.array or None): relative heights. Default all 1.

    Returns:
    x (np.array): the timestream.
    """
    x = sigma * rng.standard_normal(n)
    shape = pulse_shape()
    height = np.ones(len(times)) if height is None else height
    for t, h in zip(times, height):
        m = min(len(shape), n - t)
        x[t:t + m] += sign * amp * h * shape[:m]
    return x


def test_detection_rate_and_tau():
    rng = np.random.default_rng(0)
    n = int(20 / DT)
    times = np.sort(rng.choice(np.arange(200, n - 200, 300), 1000,
                               replace = False))
    x = synth(rng, n, times, height = 1 + 0.05 * rng.standard_normal(1000))
    # Slow drift that the detection timestream removes
    x += 3e-7 * np.sin(2 * np.pi * 0.3 * np.arange(n) * DT)
    d = ph.detection_timestream(x, DT)
    res = ph.run_detection(d, DT)
    found = res['events']
    # Nearly every pulse found, at the right sample, and few extra
    hit = np.isin(times, found) | np.isin(times, found - 1) | \
        np.isin(times, found + 1)
    assert hit.mean() > 0.98
    assert len(found) < 1.02 * len(times)
    assert res['rate'] == pytest.approx(len(times) / (n * DT), rel = 0.03)
    # Mean pulse amplitude ~ the true one; heights ~ 1 by construction
    assert res['amp_ph'] == pytest.approx(1e-6, rel = 0.05)
    assert np.median(res['cand_h'][np.isin(res['cand_idx'], found)]) == \
        pytest.approx(1, abs = 0.05)
    tau, popt = ph.fit_tau(res['template'], res['p'], DT)
    assert tau == pytest.approx(TAU, rel = 0.1)
    assert len(res['history']) == ph.DetectionSettings().of_iterations
    assert res['sign'] == -1


def test_auto_hmin_and_noise_only():
    rng = np.random.default_rng(1)
    n = int(10 / DT)
    times = np.arange(500, n - 500, 400)
    x = synth(rng, n, times, amp = 5e-7)   # SNR 5 per sample
    d = ph.detection_timestream(x, DT)
    s = ph.DetectionSettings(hmin = 0.2, hmin_false_frac = 1e-3,
                             hmin_auto_max = 0.8)
    res = ph.run_detection(d, DT, s)
    assert 0.2 <= res['hmin'] <= 0.8
    ff = ph.false_fraction(res['cand_h'], res['neg_h'], res['clean_frac'],
                           res['hmin'], res['hmax'])
    assert ff <= 1e-3 or res['hmin'] == 0.8
    # Pure noise: no events to build a template from
    with pytest.raises(ph.DetectionError):
        ph.run_detection(ph.detection_timestream(
            1e-7 * rng.standard_normal(n), DT), DT)


def test_coincidences_and_average():
    rng = np.random.default_rng(2)
    n = int(10 / DT)
    cosmic = np.arange(1000, n - 1000, 7919)
    ds, results = [], []
    for k in range(3):
        # Independent photons on each tone, at least 250 samples apart
        own = 500 + np.cumsum(rng.integers(250, 1000, 600))
        own = own[own < n - 500]
        own = own[~ph.near(own, cosmic, 300, 300)]
        # Cosmic rays: big pulses on every tone at once
        times = np.concatenate((own, cosmic + k))
        height = np.concatenate((np.ones(len(own)), np.full(len(cosmic), 3.)))
        o = np.argsort(times)
        x = synth(rng, n, times[o], height = height[o])
        ds.append(ph.detection_timestream(x, DT))
        results.append(ph.run_detection(ds[-1], DT))
    mult = ph.coincidence_multiplicity([r['cand_idx'] for r in results],
                                       win = 3)
    for k, (r, m) in enumerate(zip(results, mult)):
        at_cosmic = ph.near(r['cand_idx'], cosmic + k, 3, 4)
        # Cosmic rays are above the noise threshold and coincident on all
        # three tones; nearly all others are on one tone
        assert at_cosmic.sum() == len(cosmic)
        assert np.all(m[at_cosmic] == 3)
        # (accidental coincidences at 60 photons/s per tone: ~2%)
        assert np.mean(m[~at_cosmic] == 1) > 0.95
        avg, p, used, coinc = ph.average_pulse(ds[k], r, m)
        assert set(coinc.tolist()) >= set(r['cand_idx'][at_cosmic].tolist())
        assert not np.any(ph.near(used, cosmic + k, 3, 4))
        # The cosmic rays (height 3, outside hmax) don't bias the average
        assert avg[p] == pytest.approx(1e-6, rel = 0.05)
    t, mlt = ph.coincidence_clusters(
        [ph.triggers(d, 6., 30) for d in ds], win = 3)
    assert np.sum(mlt == 3) >= len(cosmic) - 1


def test_helpers():
    assert ph.mad_sigma([np.nan, np.nan]) != ph.mad_sigma([np.nan, np.nan])
    np.testing.assert_array_equal(
        ph.near([5, 20], [10], 6, 3), [True, False])
    assert not ph.near([1], None, 1, 1).any()
    m = ph.coincidence_multiplicity([[10, 100], [12], []], win = 3)
    np.testing.assert_array_equal(m[0], [2, 1])
    np.testing.assert_array_equal(m[1], [2])
    assert len(m[2]) == 0
    tau, popt = ph.fit_tau(np.ones(3), 0, DT)
    assert np.isnan(tau)
    with pytest.raises(ValueError, match = 'no events'):
        ph.build_template(np.zeros(10), [0], 5, 5, 1)
