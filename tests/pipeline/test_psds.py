"""
Tests for ``citkid.pipeline.psds``: the reduced-frequency grid, reduced
and binned spectra, common-mode removal, deglitching, and the timestream
selector.
"""
import numpy as np
import pytest
from citkid.pipeline import psds


@pytest.mark.parametrize('dt, nsamples, expected', [
    # 1.5 s at 38 kHz: 10 / 1.5 = 6.7 Hz up to 38147 / 40 = 954 Hz
    (1 / 38147., 57220, [10., 30., 100., 300.]),
    # 600 s at 596 Hz: 0.0167 Hz up to 14.9 Hz
    (1 / 596., 357600, [0.03, 0.1, 0.3, 1., 3., 10.]),
    # Too short for any
    (1., 20, []),
])
def test_reduced_freqs(dt, nsamples, expected):
    np.testing.assert_array_equal(psds.reduced_freqs(dt, nsamples), expected)
    # The grid is exact (no 0.30000000000000004)
    assert all(f == float(repr(f)) for f in psds.reduced_freqs(dt, nsamples))


def test_reduced_freqs_multipliers():
    # A wider margin at both ends
    f = psds.reduced_freqs(1 / 38147., 57220, lo_mult = 100., hi_div = 100.)
    np.testing.assert_array_equal(f, [])
    f = psds.reduced_freqs(1 / 38147., 572200, lo_mult = 1., hi_div = 1.)
    np.testing.assert_array_equal(f, [0.1, 0.3, 1, 3, 10, 30, 100, 300,
                                      1000, 3000])


def test_union_freqs():
    ts = [dict(dt = 1 / 38147., start = 0, stop = 57220),
          dict(dt = 1 / 596., start = 0, stop = 357600)]
    np.testing.assert_array_equal(
        psds.union_freqs(ts), [0.03, 0.1, 0.3, 1., 3., 10., 30., 100., 300.])
    assert len(psds.union_freqs([])) == 0


def test_psd_summary_white_noise():
    """
    White noise of sigma s has a one-sided PSD of 2 s^2 dt; the reduced
    values and the bins give it, a NaN row gives NaN, frequencies not
    marked valid are NaN.
    """
    rng = np.random.default_rng(0)
    dt, T = 1e-3, 100000
    x = rng.normal(0, 3e-6, size = (3, T))
    x[1, 10] = np.nan
    freqs = np.array([0.3, 1., 3., 10., 30.])
    valid = np.array([False, True, True, True, False])
    fb, binned, reduced = psds.psd_summary(x, dt, freqs, valid, nbins = 50,
                                           fmin = 0.1)
    level = 2 * (3e-6) ** 2 * dt
    assert binned.shape == (3, len(fb)) and reduced.shape == (3, 5)
    assert np.isnan(binned[1]).all() and np.isnan(reduced[1]).all()
    for i in (0, 2):
        # 40 periodogram points within 20% of 1 Hz: a 16% scatter
        np.testing.assert_allclose(reduced[i, 1:4], level, rtol = 0.4)
        assert np.isnan(reduced[i, [0, 4]]).all()
        # The bins above 10 Hz hold hundreds of points: close to the level
        np.testing.assert_allclose(binned[i][fb > 10], level, rtol = 0.3)
    # Below fmin the spectrum is unbinned: consecutive frequencies 1 / T dt
    low = fb[fb < 0.1]
    np.testing.assert_allclose(np.diff(low), 1 / (T * dt))
    assert fb.max() < 0.5 / dt and np.all(np.diff(fb) > 0)
    # No valid mask: every in-range frequency; out-of-range ones are NaN
    _, _, r = psds.psd_summary(x[0], dt, np.array([1e-3, 1., 450.]),
                               fmin = 0.1)
    assert np.isnan(r[0, 0]) and np.isnan(r[0, 2]) and np.isfinite(r[0, 1])
    # All NaN: empty bins
    fb, binned, reduced = psds.psd_summary(np.full((2, T), np.nan), dt,
                                           freqs)
    assert len(fb) == 0 and binned.shape == (2, 0)


def test_psd_summary_pulse_tube_filter():
    rng = np.random.default_rng(1)
    dt, T = 1e-3, 200000
    t = np.arange(T) * dt
    x = rng.normal(0, 1e-6, size = T) + 5e-6 * np.sin(2 * np.pi * 1.39296 * t)
    freqs = np.array([1.])
    _, _, r = psds.psd_summary(x, dt, freqs, fmin = 0.5)
    _, _, rf = psds.psd_summary(x, dt, freqs, fmin = 0.5, filter_pt_n = 3)
    # The reduced value is of the unfiltered spectrum either way
    assert r[0, 0] == rf[0, 0]
    fb, b, _ = psds.psd_summary(x, dt, freqs, fmin = 0.5)
    fbf, bf, _ = psds.psd_summary(x, dt, freqs, fmin = 0.5, filter_pt_n = 3)
    np.testing.assert_array_equal(fb, fbf)
    k = np.argmin(np.abs(fb - 1.39296))
    assert bf[0, k] < 0.01 * b[0, k]


def test_remove_common_modes():
    """
    Tones sharing a slow common mode on top of white noise: the mode is
    removed (the cleaned spectrum at the mode's frequencies is the white
    level), NaN rows are kept, and the component count is capped.
    """
    rng = np.random.default_rng(2)
    dt, T, n = 1e-3, 60000, 8
    t = np.arange(T) * dt
    mode = np.cumsum(rng.normal(size = T)) * 1e-7
    weights = rng.uniform(0.5, 2, size = n)
    x = weights[:, None] * mode + rng.normal(0, 1e-6, size = (n, T))
    x[3] = np.nan
    # One component (the real mode): the cleaned tones are white at 3 Hz,
    # where the mode was far above the white level
    cm = psds.remove_common_modes(x, dt, 1, 5, (50., 2), (0.1, 2))
    assert cm['n_comp'] == 1 and cm['y'].shape == x.shape
    np.testing.assert_array_equal(cm['fit_rows'], [0, 1, 2, 4, 5, 6, 7])
    assert np.isnan(cm['y'][3]).all() and np.isnan(cm['a'][3]).all()
    assert cm['a'].shape == (n, 1) and cm['A'].shape == (1, T)
    assert cm['sigma_iter'].shape == (5, n)
    assert np.isnan(cm['sigma_iter'][:, 3]).all()
    assert np.isfinite(cm['sigma_iter'][:, cm['fit_rows']]).all()
    freqs = np.array([3., 100.])
    _, _, before = psds.psd_summary(x, dt, freqs, fmin = 0.5)
    _, _, after = psds.psd_summary(cm['y'], dt, freqs, fmin = 0.5)
    level = 2 * 1e-12 * dt
    assert np.all(before[cm['fit_rows'], 0] > 5 * level)
    # The mode is gone. The removed mode is not filtered, so each tone
    # also loses the share of its own white noise that it contributes to
    # the mode (up to about a third for the tone with the largest weight)
    for col in (0, 1):
        a = after[cm['fit_rows'], col]
        assert np.all(a < 1.3 * level) and np.all(a > 0.5 * level)
    np.testing.assert_allclose(after[cm['fit_rows'], 1].mean(), level,
                               rtol = 0.25)
    # Weights follow the injected ones (up to a common scale)
    a0 = cm['a'][cm['fit_rows'], 0]
    ratio = a0 / weights[cm['fit_rows']]
    np.testing.assert_allclose(ratio, ratio[0], rtol = 0.1)
    # More components: shapes follow; capped at tones - 1; nothing removed
    # from one tone
    cm3 = psds.remove_common_modes(x, dt, 3, 2, (50., 2), (0.1, 2))
    assert cm3['n_comp'] == 3 and cm3['a'].shape == (n, 3)
    assert cm3['A'].shape == (3, T) and cm3['sigma_iter'].shape == (2, n)
    cm2 = psds.remove_common_modes(x[:3], dt, 3, 2, (50., 2), None)
    assert cm2['n_comp'] == 2 and cm2['a'].shape == (3, 2)
    cm1 = psds.remove_common_modes(x[:1], dt, 3, 2, (50., 2), None)
    assert cm1['n_comp'] == 0 and cm1['a'].shape == (1, 0)
    np.testing.assert_array_equal(cm1['y'], x[:1])
    assert cm1['A'].shape == (0, T)


def test_deglitch_peaks():
    rng = np.random.default_rng(3)
    dt, T = 1e-3, 50000
    x = rng.normal(0, 1e-6, size = (3, T))
    # Negative glitches on tone 0 and 2, decaying over a few samples
    for i, where in ((0, [1000, 20000, 30000]), (2, [5000])):
        for w in where:
            x[i, w:w + 5] -= 2e-5 * np.exp(-np.arange(5) / 2)
    x[1, :10] = np.nan
    clean, n, frac = psds.deglitch_peaks(x, dt, sign = -1, nstd = 6.,
                                         baseline_s = 1., distance = 10,
                                         pre = 2, post = 8)
    np.testing.assert_array_equal(n, [3, 0, 1])
    np.testing.assert_allclose(frac[[0, 2]], [30 / T, 10 / T])
    assert frac[1] == 0 and np.isnan(clean[1, 0])
    assert np.abs(clean[0]).max() < 6e-6 and np.abs(clean[2]).max() < 6e-6
    # The sign matters: nothing found looking for positive glitches
    _, n_pos, _ = psds.deglitch_peaks(x, dt, sign = 1, nstd = 6.,
                                      baseline_s = 1.)
    np.testing.assert_array_equal(n_pos, 0)


def test_select_timestreams():
    def ts(index, run_index, name, tone_set, cal_index):
        return dict(index = index, run_index = run_index, name = name,
                    tone_set = tone_set, cal_index = cal_index,
                    run = f'ts{run_index}', cal = f'ts_cal{cal_index}')

    lst = [ts(0, 0, 'dark', 0, 0), ts(1, 0, 'dark', 1, 0),
           ts(2, 1, 'dark', 0, 0), ts(3, 1, 'dark', 1, 0),
           ts(4, 1, 'dark', 0, 1), ts(5, 2, 'light', 0, 1)]
    pick = lambda c: [t['index'] for t in psds.select_timestreams(lst, c)]
    assert pick([3, 0]) == [3, 0]
    # A name: the latest run, every tone set, the latest calibration
    assert pick(['dark']) == [3, 4]
    assert pick(['light']) == [5]
    # Dicts: tone sets, calibration, run index
    assert pick([dict(name = 'dark', tone_sets = [0])]) == [4]
    assert pick([dict(name = 'dark', cal = 0)]) == [2, 3]
    assert pick([dict(run = 0)]) == [0, 1]
    assert pick([dict(run = 1, tone_sets = [1], cal = 0)]) == [3]
    for bad in (['nope'], [9], [dict(run = 1, cal = 2)],
                [dict(name = 'light', tone_sets = [1])]):
        with pytest.raises(KeyError):
            psds.select_timestreams(lst, bad)
