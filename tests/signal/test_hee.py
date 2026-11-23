"""
Tests of citkid.signal.hee: high-energy event (HEE) events and tail
fits.
"""
import numpy as np
import pytest
from citkid.signal import hee as cr
from citkid.signal import photons as ph
from .test_photons import DT, TAU, synth

PRE, POST = 20, 150


@pytest.fixture(scope = 'module')
def rays():
    """
    Make three tones hit by the same HEEs with different heights
    (sharing one decay time), one tone with a pulse of its own, and the
    events' samples.

    Returns:
    ds (list of np.array): detection timestreams.
    times (np.array, int): the HEEs' samples.
    heights (np.array): (tones, rays) relative heights.
    """
    rng = np.random.default_rng(1)
    n = int(2 / DT)
    times = np.arange(500, n - 500, 700)
    heights = np.exp(0.5 * rng.standard_normal((3, len(times)))) * 10
    ds = []
    for k in range(3):
        extra = np.array([250]) if k == 0 else np.zeros(0, dtype = int)
        t = np.r_[extra, times]
        h = np.r_[np.full(len(extra), 8.), heights[k]]
        x = synth(rng, n, t, amp = 1e-6, sigma = 5e-8, height = h)
        ds.append(ph.detection_timestream(x, DT))
    return ds, times, heights


def test_find_events(rays):
    """
    The rays on every tone are found as events of multiplicity 3; the
    pulse on one tone alone is not.
    """
    ds, times, _ = rays
    t0, mult = cr.find_events(ds, 6., 2, min_mult = 3, sign = -1,
                              dead = POST)
    assert len(t0) == len(times) and np.all(mult == 3)
    assert np.all(np.abs(t0 - times) <= 2)
    t0, mult = cr.find_events(ds, 6., 2, min_mult = 1, sign = -1,
                              dead = POST)
    assert len(t0) == len(times) + 1 and mult[0] == 1


def test_cut_and_peaks(rays):
    """
    Snippets are sign-corrected, baseline-subtracted, and dropped at the
    edges; the peaks are at the events with the simulated heights.
    """
    ds, times, heights = rays
    t0 = np.r_[5, times, len(ds[1]) - 10]
    pulses, kept = cr.cut_pulses(ds[1], t0, PRE, POST)
    assert kept.tolist() == [False] + [True] * len(times) + [False]
    assert pulses.shape == (len(times), PRE + POST)
    np.testing.assert_allclose(np.median(pulses[:, :PRE - 3], axis = 1), 0,
                               atol = 1e-12)
    h, p = cr.peak_heights(pulses, PRE, 10)
    assert np.all(p == PRE)
    np.testing.assert_allclose(h, heights[1] * 1e-6, rtol = 0.05)
    empty, kept = cr.cut_pulses(ds[1], [], PRE, POST)
    assert empty.shape == (0, PRE + POST) and len(kept) == 0
    assert cr.peak_heights(empty, PRE, 10)[0].shape == (0,)


def test_tail_fits(rays):
    """
    Each pulse's tail is fitted from where it falls through the level:
    the decay times agree with the simulation, whatever the height, and
    a pulse that never reaches the level, or a level below the noise, is
    not fitted.
    """
    ds, times, heights = rays
    level = 3e-6
    for k in range(3):
        pulses, _ = cr.cut_pulses(ds[k], times, PRE, POST)
        h, p = cr.peak_heights(pulses, PRE, 10)
        tau, start, n_fit = cr.fit_tails(pulses, p, DT, level, 1e-3)
        assert np.isfinite(tau).all()
        # The tail starts later for the higher pulses
        expected = PRE + np.ceil(np.log(h / level) * TAU / DT)
        assert np.all(np.abs(start - expected) <= 1)
        assert np.all(n_fit == round(1e-3 / DT))
        np.testing.assert_allclose(tau, TAU, rtol = 0.1)
        med, spread, n = cr.tail_summary(tau)
        assert n == len(times) and abs(med - TAU) < 0.05 * TAU
        assert spread < 0.1 * TAU
    # One pulse: the level above its peak means no tail
    assert cr.tail_start(pulses[0], p[0], 1.) == p[0]
    tau1, popt, n1, s1 = cr.fit_tail(pulses[0], p[0], DT, -1., 1e-3)
    assert np.isnan(tau1) and n1 == 0 and s1 == -1 and np.isnan(popt).all()
    assert cr.tail_summary([np.nan, np.nan]) == (pytest.approx(np.nan,
                                                               nan_ok = True),
                                                 pytest.approx(np.nan,
                                                               nan_ok = True),
                                                 0)


def test_align_and_mean_fit(rays):
    """
    Pulses of different heights are lined up where each falls through
    the level; the mean of the aligned pulses decays with the simulated
    time, and its tail fit recovers it.
    """
    ds, times, heights = rays
    level = 3e-6
    pulses, _ = cr.cut_pulses(ds[2], times, PRE, POST)
    h, p = cr.peak_heights(pulses, PRE, 10)
    start = np.array([cr.tail_start(pulses[k], p[k], level)
                      for k in range(len(h))])
    assert (start >= 0).all() and np.ptp(start) > 2
    aligned = cr.align_tails(pulses, start, PRE)
    assert aligned.shape == pulses.shape
    # At the anchor every pulse is at or just below the level
    at = aligned[:, PRE]
    assert np.isfinite(at).all() and np.all(at <= level)
    assert np.all(at > 0.5 * level)
    # Shifted samples are NaN at the edges
    assert np.isnan(aligned[np.argmax(start), -1])
    avg, count = cr.mean_aligned(aligned)
    assert count[PRE] == len(h) and count.max() == len(h)
    assert np.isnan(avg[count == 0]).all()
    tau, popt, n_fit = cr.fit_mean_tail(avg, PRE, DT, 1e-3)
    assert n_fit == round(1e-3 / DT)
    assert abs(tau - TAU) < 0.05 * TAU
    assert abs(popt[0] - avg[PRE]) < 0.1 * avg[PRE]
    # Empty or too short
    assert cr.mean_aligned(np.zeros((0, 10)))[0].shape == (0,)
    short = np.full(170, np.nan)
    short[PRE:PRE + 2] = 1.
    assert np.isnan(cr.fit_mean_tail(short, PRE, DT, 1e-3)[0])


def test_overlap_alignment_segment_mean_and_fit_start():
    """
    Pulses with the same tail but noisy level crossings are lined up by
    their whole tails, the mean covers the fit range only, the fit
    begins after the latest peak the averaging leaves, and a given
    start is used as it is.
    """
    from citkid.signal.hee import (
        align_tails, fit_mean_tail, fit_start, mean_segment,
        refine_alignment)
    rng = np.random.default_rng(3)
    dt, L, tau = 1e-5, 400, 1e-3
    t = np.arange(L) * dt
    n = 12
    pulses = np.zeros((n, L))
    true_start = 60 + rng.integers(-6, 7, n)
    for k in range(n):
        s = true_start[k]
        pulses[k, s:] = 2. * np.exp(-(t[s:] - t[s]) / tau)
        pulses[k, s - 5:s] = np.linspace(0., 2., 5)      # the rise
        pulses[k] += rng.normal(0, 0.02, L)
    # Level crossings a few samples off, as noise makes them
    noisy = true_start + rng.integers(-3, 4, n)
    n_fit = int(round(tau / dt))
    refined = refine_alignment(pulses, noisy, n_fit)
    offsets = refined - true_start
    assert np.abs(offsets - np.median(offsets)).max() <= 1
    anchor = 100
    aligned = align_tails(pulses, refined, anchor)
    avg, count = mean_segment(aligned, anchor, n_fit)
    assert np.isnan(avg[:anchor]).all() and np.isnan(avg[anchor + n_fit:]).all()
    assert np.isfinite(avg[anchor:anchor + n_fit]).all()
    assert count[anchor] == n and count[anchor - 1] == 0
    # A bump after the alignment point: the fit begins after it
    bumped = avg.copy()
    bumped[anchor + 3:anchor + 6] += [0.5, 1., 0.5]
    assert fit_start(bumped, anchor, 20) == anchor + 4
    assert fit_start(avg, anchor, 20) == anchor
    tau_fit, popt, n = fit_mean_tail(avg, anchor, dt, tau,
                                     start = anchor + 4)
    assert abs(tau_fit - tau) / tau < 0.1 and n <= n_fit - 4
    assert fit_mean_tail(avg, anchor, dt, tau)[0] == pytest.approx(
        fit_mean_tail(avg, anchor, dt, tau, start = anchor)[0])
