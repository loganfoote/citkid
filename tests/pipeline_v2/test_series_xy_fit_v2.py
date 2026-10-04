"""
Tests for citkid.pipeline_v2.series_xy_fit (SeriesXYFit and SeriesXYFitStore).
"""

import threading

import numpy as np
import pytest
import zarr

from citkid.pipeline_v2.series_xy_fit import SeriesXYFit, SeriesXYFitStore, usable_points


def _linear_fit(uses_guess=False, **kwargs):
    """
    Build a linear SeriesXYFit that records its inputs.

    Parameters:
    uses_guess (bool): If True, the fit takes a guess and returns
        (popt, p0); its own guess is (1, 0).
    **kwargs: Extra SeriesXYFit arguments.

    Returns:
    xy_fit (SeriesXYFit): The fit.
    calls (list): ``(x, y)`` (and the guess, if used) passed to each call.
    """
    calls = []

    if uses_guess:
        def fit(x, y, p0):
            calls.append((x.copy(), y.copy(), None if p0 is None else p0.copy()))
            guess = np.array([1.0, 0.0]) if p0 is None else p0
            return np.polyfit(x, y, 1), guess
    else:
        def fit(x, y):
            calls.append((x.copy(), y.copy()))
            return np.polyfit(x, y, 1)

    xy_fit = SeriesXYFit(
        fit=fit,
        param_names=['slope', 'intercept'],
        model=lambda xs, slope, intercept: slope * xs + intercept,
        name='line',
        uses_guess=uses_guess,
        **kwargs,
    )
    return xy_fit, calls


def _store(xy_fit, nrows=4, n_series=4, group=None):
    """
    Build a store on an in-memory zarr group.

    Parameters:
    xy_fit (SeriesXYFit): Fit definition.
    nrows (int): Number of resonators.
    n_series (int): Number of series indices.
    group (zarr.Group or None): Group to use, or None for a new one.

    Returns:
    store (SeriesXYFitStore): The store.
    """
    if group is None:
        group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    return SeriesXYFitStore(group, xy_fit, nrows, n_series)


def test_usable_points_drops_nan_and_sorts():
    xs, ys = usable_points([3.0, np.nan, 1.0, 2.0], [30.0, 5.0, 10.0, np.nan])

    np.testing.assert_array_equal(xs, [1.0, 3.0])
    np.testing.assert_array_equal(ys, [10.0, 30.0])


def test_run_passes_usable_sorted_points_and_returns_an_array():
    xy_fit, calls = _linear_fit()

    popt, p0 = xy_fit.run(np.array([3.0, np.nan, 1.0, 2.0]), np.array([7.0, 1.0, 3.0, 5.0]))

    np.testing.assert_array_equal(calls[0][0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(calls[0][1], [3.0, 5.0, 7.0])
    assert isinstance(popt, np.ndarray) and popt.shape == (2,)
    np.testing.assert_allclose(popt, [2.0, 1.0])
    assert p0 is None


def test_run_with_too_few_points_returns_none():
    xy_fit, calls = _linear_fit()

    assert xy_fit.run(np.array([1.0, np.nan]), np.array([1.0, 2.0])) == (None, None)
    assert calls == []


def test_run_accepts_a_single_value_and_checks_the_count():
    single = SeriesXYFit(fit=lambda x, y: y.mean(), param_names=['mean'],
                        model=lambda xs, m: np.full_like(xs, m))
    popt, _ = single.run(np.array([1.0, 2.0]), np.array([4.0, 6.0]))
    np.testing.assert_array_equal(popt, [5.0])

    wrong = SeriesXYFit(fit=lambda x, y: (1.0, 2.0, 3.0), param_names=['a', 'b'],
                       model=lambda xs, a, b: xs)
    with pytest.raises(ValueError, match='returned 3 parameter values'):
        wrong.run(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_run_with_guess():
    xy_fit, calls = _linear_fit(uses_guess=True)
    x, y = np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 7.0])

    popt, p0 = xy_fit.run(x, y)                          # the fit's own guess
    assert calls[-1][2] is None
    np.testing.assert_allclose(popt, [2.0, 1.0])
    np.testing.assert_array_equal(p0, [1.0, 0.0])

    popt, p0 = xy_fit.run(x, y, p0=[5, 6])               # the user's guess
    np.testing.assert_array_equal(calls[-1][2], [5.0, 6.0])
    np.testing.assert_array_equal(p0, [5.0, 6.0])


def test_guess_errors():
    no_guess, _ = _linear_fit()
    with pytest.raises(ValueError, match="doesn't use guesses"):
        no_guess.run(np.array([1.0, 2.0]), np.array([1.0, 2.0]), p0=[1, 0])

    forgot = SeriesXYFit(fit=lambda x, y, p0: (1.0, 2.0), param_names=['a', 'b'],
                         model=lambda xs, a, b: xs, uses_guess=True)
    with pytest.raises(ValueError, match=r'must return \(popt, p0\)'):
        forgot.run(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    short = SeriesXYFit(fit=lambda x, y, p0: ([1.0, 2.0], [1.0]), param_names=['a', 'b'],
                        model=lambda xs, a, b: xs, uses_guess=True)
    with pytest.raises(ValueError, match='returned 1 guess values'):
        short.run(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


@pytest.mark.parametrize('names', [[], ['a', 'a']])
def test_invalid_param_names_raise(names):
    with pytest.raises(ValueError, match='param_names'):
        SeriesXYFit(fit=lambda x, y: x, param_names=names, model=lambda xs, *p: xs)


def test_curve_spans_usable_x_range():
    xy_fit, _ = _linear_fit(n_samples=5)
    popt = np.array([2.0, 1.0])

    xs, ys = xy_fit.curve(np.array([np.nan, 1.0, 3.0]), popt)

    np.testing.assert_allclose(xs, [1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_allclose(ys, 2.0 * xs + 1.0)
    assert xy_fit.curve(np.array([1.0, 3.0]), None) == (None, None)
    assert xy_fit.curve(np.array([1.0, np.nan]), popt) == (None, None)
    assert xy_fit.curve(np.array([1.0, 3.0]), np.array([np.nan, 1.0])) == (None, None)


def test_curve_log_x_spaces_samples_geometrically():
    xy_fit, _ = _linear_fit(n_samples=4)
    popt = np.array([2.0, 1.0])

    xs, ys = xy_fit.curve(np.array([-1.0, 1.0, np.nan, 1000.0]), popt, log_x=True)

    np.testing.assert_allclose(xs, [1.0, 10.0, 100.0, 1000.0])  # x <= 0 is ignored
    np.testing.assert_allclose(ys, 2.0 * xs + 1.0)
    assert xy_fit.curve(np.array([-1.0, 5.0]), popt, log_x=True) == (None, None)


def test_curve_works_with_numba_float64_model():
    """Parameters reach the model as floats, as numba float64 signatures need."""
    from numba import float64, njit

    @njit(float64[:](float64[:], float64, float64))
    def line(x, slope, intercept):
        return slope * x + intercept

    xy_fit = SeriesXYFit(fit=lambda x, y: (2.0, 1.0), param_names=['slope', 'intercept'],
                         model=line, n_samples=3)

    xs, ys = xy_fit.curve(np.array([1.0, 3.0]), np.array([2.0, 1.0]))

    np.testing.assert_allclose(ys, [3.0, 5.0, 7.0])


def test_describe_formats_parameters():
    xy_fit, _ = _linear_fit()

    assert xy_fit.describe(np.array([0.0312, 12.0])) == 'slope = 0.0312, intercept = 12'
    assert xy_fit.describe(None) == ''


def test_store_saves_popt_as_one_array():
    xy_fit, _ = _linear_fit()
    store = _store(xy_fit)
    x = np.array([1.0, 2.0, 3.0, np.nan])
    y = np.array([3.0, 5.0, 7.0, np.nan])

    assert not store.has_fit(2)
    store.save(2, x, y, xy_fit.run(x, y)[0])

    saved = store.load(2)
    np.testing.assert_array_equal(saved.fit_x, x)
    np.testing.assert_array_equal(saved.fit_y, y)
    np.testing.assert_allclose(saved.popt, [2.0, 1.0])
    assert saved.p0 is None and saved.p0_user is False
    assert store.group['popt'].shape == (4, 2)
    assert 'slope' not in store.group and 'p0' not in store.group
    popt = store.popt()
    np.testing.assert_allclose(popt[2], [2.0, 1.0])
    assert np.isnan(popt[[0, 1, 3]]).all()
    assert store.has_fit(2)
    assert list(np.flatnonzero(store.fitted_rows())) == [2]
    assert store.is_current(2, x, y)
    assert not store.is_current(2, x, np.array([3.0, 5.0, 8.0, np.nan]))
    assert not store.is_current(1, x, y)


def test_store_saves_guesses():
    xy_fit, _ = _linear_fit(uses_guess=True)
    store = _store(xy_fit)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    popt, p0 = xy_fit.run(x, 2 * x, p0=[3.0, 1.0])

    store.save(1, x, 2 * x, popt, p0, p0_user=True)

    saved = store.load(1)
    np.testing.assert_array_equal(saved.p0, [3.0, 1.0])
    assert saved.p0_user is True
    assert store.group['p0'].shape == (4, 2)
    assert list(store.group['p0_user'][...]) == [False, True, False, False]


def test_store_failed_fit_is_saved_as_attempted():
    xy_fit, _ = _linear_fit()
    store = _store(xy_fit)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    store.save(0, x, 2 * x, xy_fit.run(x, 2 * x)[0])

    store.save(1, x, x, None)

    assert store.has_fit(1)
    assert store.load(1).popt is None
    assert np.isnan(store.group['popt'][1]).all()


def test_store_definition_mismatch_and_clear():
    xy_fit, _ = _linear_fit()
    group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    store = _store(xy_fit, group=group)
    assert store.matches_definition()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    store.save(0, x, x, xy_fit.run(x, x)[0])
    assert _store(xy_fit, group=group).matches_definition()

    other = SeriesXYFit(fit=lambda x, y: y.mean(), param_names=['mean'],
                       model=lambda xs, m: xs, name='mean')
    other_store = _store(other, group=group)
    assert not other_store.matches_definition()
    assert "'line'" in other_store.describe_existing()
    assert not _store(xy_fit, nrows=5, group=group).matches_definition()
    assert not _store(_linear_fit(uses_guess=True)[0], group=group).matches_definition()

    other_store.clear()
    assert other_store.matches_definition()
    assert not other_store.has_fit(0)
    assert 'popt' not in group


def test_store_concurrent_saves(tmp_path):
    xy_fit, _ = _linear_fit()
    group = zarr.open_group(str(tmp_path / 'xy.zarr'), mode='w')
    nrows = 40
    store = _store(xy_fit, nrows=nrows, group=group)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    errors = []

    def writer(rows):
        try:
            for di in rows:
                store.save(di, x, x * di, xy_fit.run(x, x * di)[0])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(range(k, nrows, 2),)) for k in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.fitted_rows().all()
    np.testing.assert_allclose(group['popt'][:, 0], np.arange(nrows), atol=1e-9)
