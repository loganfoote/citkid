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


def test_init_hmax_keeps_large_pulses_out_of_first_template():
    """
    Large pulses (e.g. cosmic rays at 20 photons) are found in the first
    pass but cut by ``init_hmax``, and the PSD still avoids them.
    """
    rng = np.random.default_rng(3)
    n = int(10 / DT)
    times = np.arange(500, n - 500, 400)
    big = rng.random(len(times)) < 0.05
    x = synth(rng, n, times, height = np.where(big, 20., 1.))
    d = ph.detection_timestream(x, DT)
    res = ph.run_detection(d, DT, ph.DetectionSettings(of_iterations = 1))
    assert np.isfinite(res['init_hmax_abs'])
    at_big = ph.near(res['init_idx'], times[big], 2, 3)
    assert at_big.sum() == big.sum()
    assert np.all(res['init_h'][at_big] > res['init_hmax_abs'])
    assert res['amp_ph'] == pytest.approx(1e-6, rel = 0.05)
    # Without the cut, the big pulses get into the first template, which
    # one iteration can't fully undo
    loose = ph.run_detection(d, DT, ph.DetectionSettings(
        of_iterations = 1, init_hmax = None))
    assert np.isinf(loose['init_hmax_abs'])


def test_coincidence_candidates_and_cosmic_events():
    """
    Cosmic rays: large pulses on tones 0 and 1, a pulse below hmin (0.4,
    about 7 sigma of the filter output) on tone 2. The lower coincidence
    threshold still counts tone 2, so every cosmic ray has multiplicity 3
    and is masked from the averages, and ``cosmic_events`` returns each
    one with the three tones' heights.
    """
    rng = np.random.default_rng(4)
    n = int(10 / DT)
    cosmic = np.arange(1000, n - 1000, 7919)
    heights = (4., 3., 0.4)
    # Arrival offsets in samples (within the +-2 sample window)
    offset = (0, 1, 1)
    ds, results = [], []
    for k in range(3):
        own = 500 + np.cumsum(rng.integers(250, 1000, 600))
        own = own[own < n - 500]
        own = own[~ph.near(own, cosmic, 300, 300)]
        times = np.concatenate((own, cosmic + offset[k]))
        height = np.concatenate((np.ones(len(own)),
                                 np.full(len(cosmic), heights[k])))
        o = np.argsort(times)
        x = synth(rng, n, times[o], height = height[o])
        ds.append(ph.detection_timestream(x, DT))
        results.append(ph.run_detection(ds[-1], DT))
    for r in results:
        # Every pulse candidate is a coincidence candidate
        assert set(r['cand_idx'].tolist()) <= set(r['coinc_idx'].tolist())
    # Tone 2's cosmic-ray pulses are mostly below hmin (noise lifts a few
    # above it), but all are coincidence candidates
    r2, t2 = results[2], cosmic + offset[2]
    assert ph.near(r2['cand_idx'], t2, 2, 3).sum() < 0.2 * len(cosmic)
    assert ph.near(r2['coinc_idx'], t2, 2, 3).sum() == len(cosmic)
    mult = ph.candidate_multiplicity(results, win = 2)
    for k in (0, 1):
        r, m = results[k], mult[k]
        at = ph.near(r['cand_idx'], cosmic + offset[k], 2, 3)
        assert at.sum() == len(cosmic) and np.all(m[at] == 3)
        avg, p, used, coinc = ph.average_pulse(ds[k], r, m)
        assert not ph.near(used, cosmic + offset[k], 2, 3).any()
        assert avg[p] == pytest.approx(1e-6, rel = 0.05)
    events, hits = ph.cosmic_events(results, win = 2, min_mult = 3)
    assert len(events['t0']) == len(cosmic)
    assert np.all(events['mult'] == 3)
    assert np.all(np.abs(events['t0'] - cosmic) <= 2)
    assert len(hits['event']) == 3 * len(cosmic)
    for k, hk in enumerate(heights):
        sel = hits['tone'] == k
        assert np.median(hits['h'][sel]) == pytest.approx(hk, rel = 0.15)
        assert np.median(hits['x'][sel]) == pytest.approx(hk * 1e-6,
                                                          rel = 0.15)
        assert np.median(hits['raw'][sel]) == pytest.approx(hk * 1e-6,
                                                            rel = 0.3)


def fake_result(idx, h, amp = 1e-6):
    """
    Make a minimal ``run_detection`` result for the coincidence functions.

    Parameters:
    idx (list of int): coincidence candidate samples.
    h (list of float): their heights.
    amp (float): mean photon amplitude. Default 1e-6.

    Returns:
    res (dict): the result.
    """
    idx = np.asarray(idx, dtype = np.int64)
    return dict(coinc_idx = idx, coinc_h = np.asarray(h, float),
                coinc_raw = np.asarray(h, float) * amp, amp_ph = amp,
                cand_idx = idx)


def test_cosmic_event_grouping():
    """
    Events start at the earliest pulse and span 2 win samples; a tone's
    largest pulse is kept; events with too few tones are dropped.
    """
    results = [fake_result([100, 102, 500], [1., 2., 1.]),
               fake_result([104, 503], [3., 1.]),
               fake_result([105, 505], [1., 1.]),
               fake_result([900], [1.])]
    events, hits = ph.cosmic_events(results, win = 2, min_mult = 2)
    # 100..104: tones 0 (largest: 102) and 1; 105 starts a new event alone
    # (dropped); 500..504: tones 0 and 1; 505 alone; 900 alone
    np.testing.assert_array_equal(events['t0'], [100, 500])
    np.testing.assert_array_equal(events['mult'], [2, 2])
    np.testing.assert_array_equal(hits['event'], [0, 0, 1, 1])
    np.testing.assert_array_equal(hits['tone'], [0, 1, 0, 1])
    np.testing.assert_array_equal(hits['idx'], [102, 104, 500, 503])
    np.testing.assert_allclose(hits['h'], [2., 3., 1., 1.])
    np.testing.assert_allclose(hits['x'], [2e-6, 3e-6, 1e-6, 1e-6])
    events, hits = ph.cosmic_events(results, win = 3, min_mult = 3)
    np.testing.assert_array_equal(events['t0'], [100, 500])
    np.testing.assert_array_equal(events['mult'], [3, 3])
    events, hits = ph.cosmic_events([], win = 2)
    assert len(events['t0']) == 0 and len(hits['idx']) == 0
    # Multiplicity: +-win around each candidate, inclusive
    m = ph.candidate_multiplicity(
        [fake_result([10, 100], [1, 1]), fake_result([12], [1]),
         fake_result([8, 103], [1, 1])], win = 2)
    np.testing.assert_array_equal(m[0], [3, 1])
    np.testing.assert_array_equal(m[1], [2])
    np.testing.assert_array_equal(m[2], [2, 1])
    np.testing.assert_allclose(
        ph.raw_peaks(np.array([0., -1., -3., -2., 0.]), [2, 0], -1, 1),
        [3., 1.])


def test_detect_with_filter():
    """
    The optimal filter of one timestream finds the pulses of another
    timestream of the same tone without iterating, with heights in the
    first one's photon units; a different sample period is refused.
    """
    rng = np.random.default_rng(5)
    n = int(10 / DT)
    t1 = np.arange(500, n - 500, 400)
    ref = ph.run_detection(ph.detection_timestream(synth(rng, n, t1), DT),
                           DT)
    assert ref['dt'] == DT
    # Second timestream: other pulse times, 10% larger responsivity
    t2 = np.arange(700, n - 700, 450)
    d2 = ph.detection_timestream(synth(rng, n, t2, amp = 1.1e-6), DT)
    res = ph.detect_with_filter(d2, DT, ref)
    hit = np.isin(t2, res['events']) | np.isin(t2, res['events'] - 1) | \
        np.isin(t2, res['events'] + 1)
    assert hit.mean() > 0.98 and len(res['events']) < 1.02 * len(t2)
    assert res['amp_ph'] == ref['amp_ph']
    assert res['amp_ph_own'] == pytest.approx(1.1e-6, rel = 0.05)
    assert np.median(res['cand_h']) == pytest.approx(1.1, abs = 0.05)
    assert set(res['cand_idx'].tolist()) <= set(res['coinc_idx'].tolist())
    assert len(res['history']) == 0 and np.isnan(res['init_hmax_abs'])
    # Thresholds can be changed; another sample period is refused
    narrow = ph.detect_with_filter(d2, DT, ref, hmin = 1.2, hmax = 2.)
    assert len(narrow['events']) < 0.6 * len(res['events'])
    with pytest.raises(ValueError, match = 'dt'):
        ph.detect_with_filter(d2, 2 * DT, ref)
    with pytest.raises(ph.DetectionError):
        ph.detect_with_filter(d2, DT, ref, hmin = 5., hmax = 6.)
