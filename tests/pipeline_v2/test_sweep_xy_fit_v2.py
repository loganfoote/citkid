"""
Tests for citkid.pipeline_v2.sweep_xy_fit (SweepXYFit and SweepXYFitStore).
"""

import threading

import numpy as np
import pytest
import zarr

from citkid.pipeline_v2.sweep_xy_fit import SweepXYFit, SweepXYFitStore, usable_points


def _linear_fit(**kwargs):
    """
    Build a linear SweepXYFit that records its inputs.

    Parameters:
    **kwargs: Extra SweepXYFit arguments.

    Returns:
    xy_fit (SweepXYFit): The fit.
    calls (list): ``(x, y)`` passed to each fit call.
    """
    calls = []

    def fit(x, y):
        calls.append((x.copy(), y.copy()))
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    xy_fit = SweepXYFit(
        fit=fit,
        output_names=['slope', 'intercept'],
        model=lambda xs, slope, intercept: slope * xs + intercept,
        name='line',
        **kwargs,
    )
    return xy_fit, calls


def _store(xy_fit, nrows=4, n_sweep=4, group=None):
    """
    Build a store on an in-memory zarr group.

    Parameters:
    xy_fit (SweepXYFit): Fit definition.
    nrows (int): Number of resonators.
    n_sweep (int): Number of sweep indices.
    group (zarr.Group or None): Group to use, or None for a new one.

    Returns:
    store (SweepXYFitStore): The store.
    """
    if group is None:
        group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    return SweepXYFitStore(group, xy_fit, nrows, n_sweep)


def test_usable_points_drops_nan_and_sorts():
    xs, ys = usable_points([3.0, np.nan, 1.0, 2.0], [30.0, 5.0, 10.0, np.nan])

    np.testing.assert_array_equal(xs, [1.0, 3.0])
    np.testing.assert_array_equal(ys, [10.0, 30.0])


def test_run_passes_usable_sorted_points():
    xy_fit, calls = _linear_fit()

    outputs = xy_fit.run(np.array([3.0, np.nan, 1.0, 2.0]), np.array([7.0, 1.0, 3.0, 5.0]))

    np.testing.assert_array_equal(calls[0][0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(calls[0][1], [3.0, 5.0, 7.0])
    np.testing.assert_allclose([float(o) for o in outputs], [2.0, 1.0])


def test_run_with_too_few_points_returns_none():
    xy_fit, calls = _linear_fit()

    assert xy_fit.run(np.array([1.0, np.nan]), np.array([1.0, 2.0])) is None
    assert calls == []


def test_run_wraps_single_output_and_checks_count():
    single = SweepXYFit(fit=lambda x, y: y.mean(), output_names=['mean'],
                        model=lambda xs, m: np.full_like(xs, m))
    (mean,) = single.run(np.array([1.0, 2.0]), np.array([4.0, 6.0]))
    assert float(mean) == 5.0

    wrong = SweepXYFit(fit=lambda x, y: (1.0, 2.0, 3.0), output_names=['a', 'b'],
                       model=lambda xs, a, b: xs)
    with pytest.raises(ValueError, match='returned 3 outputs'):
        wrong.run(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


@pytest.mark.parametrize('names', [[], ['a', 'a']])
def test_invalid_output_names_raise(names):
    with pytest.raises(ValueError, match='output_names'):
        SweepXYFit(fit=lambda x, y: x, output_names=names, model=lambda xs, *o: xs)


def test_curve_spans_usable_x_range():
    xy_fit, _ = _linear_fit(n_samples=5)
    outputs = [np.asarray(2.0), np.asarray(1.0)]

    xs, ys = xy_fit.curve(np.array([np.nan, 1.0, 3.0]), outputs)

    np.testing.assert_allclose(xs, [1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_allclose(ys, 2.0 * xs + 1.0)
    assert xy_fit.curve(np.array([1.0, 3.0]), None) == (None, None)
    assert xy_fit.curve(np.array([1.0, np.nan]), outputs) == (None, None)
    assert xy_fit.curve(np.array([1.0, 3.0]), [np.asarray(np.nan), np.asarray(1.0)]) == (None, None)


def test_describe_formats_scalar_outputs():
    xy_fit, _ = _linear_fit()

    assert xy_fit.describe([np.asarray(0.0312), np.asarray(12.0)]) == 'slope = 0.0312, intercept = 12'
    assert xy_fit.describe(None) == ''


def test_store_save_load_and_is_current():
    xy_fit, _ = _linear_fit()
    store = _store(xy_fit)
    x = np.array([1.0, 2.0, 3.0, np.nan])
    y = np.array([3.0, 5.0, 7.0, np.nan])

    assert not store.has_fit(2)
    store.save(2, x, y, xy_fit.run(x, y))

    fit_x, fit_y, outputs = store.load(2)
    np.testing.assert_array_equal(fit_x, x)
    np.testing.assert_array_equal(fit_y, y)
    np.testing.assert_allclose([float(o) for o in outputs], [2.0, 1.0])
    assert store.has_fit(2)
    assert list(np.flatnonzero(store.fitted_rows())) == [2]
    assert store.is_current(2, x, y)
    assert not store.is_current(2, x, np.array([3.0, 5.0, 8.0, np.nan]))
    assert not store.is_current(1, x, y)


def test_store_failed_fit_is_saved_as_attempted():
    xy_fit, _ = _linear_fit()
    store = _store(xy_fit)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    store.save(0, x, 2 * x, xy_fit.run(x, 2 * x))

    store.save(1, x, x, None)

    assert store.has_fit(1)
    assert store.load(1)[2] is None
    assert np.isnan(store.group['slope'][1])


def test_store_definition_mismatch_and_clear():
    xy_fit, _ = _linear_fit()
    group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    store = _store(xy_fit, group=group)
    assert store.matches_definition()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    store.save(0, x, x, xy_fit.run(x, x))
    assert _store(xy_fit, group=group).matches_definition()

    other = SweepXYFit(fit=lambda x, y: y.mean(), output_names=['mean'],
                       model=lambda xs, m: xs, name='mean')
    other_store = _store(other, group=group)
    assert not other_store.matches_definition()
    assert "'line'" in other_store.describe_existing()
    assert not _store(xy_fit, nrows=5, group=group).matches_definition()

    other_store.clear()
    assert other_store.matches_definition()
    assert not other_store.has_fit(0)
    assert 'slope' not in group


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
                store.save(di, x, x * di, xy_fit.run(x, x * di))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(range(k, nrows, 2),)) for k in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.fitted_rows().all()
    np.testing.assert_allclose(group['slope'][...], np.arange(nrows), atol=1e-9)
