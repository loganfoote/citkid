"""
Tests for citkid.pipeline_v2.interactive.ts_series and the series-window
features it relies on (selectable quantities, precomputed values).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

import citkid.pipeline_v2.interactive.iq_series as iseries
import citkid.pipeline_v2.interactive.ts_series as its
from citkid.pipeline_v2.framework import plStep
from citkid.pipeline_v2.interactive.core import StepPanel


class _Panel(StepPanel):
    """Minimal panel used instead of the real TS panels."""

    def setup_ui(self):
        """Build an empty panel."""
        QtWidgets.QVBoxLayout(self).addWidget(QtWidgets.QLabel('ok'))

    def update_plots(self):
        """Do nothing."""

    def clear_plots(self):
        """Do nothing."""

    def _outputs_exist(self):
        """Report outputs as present."""
        return True


def _make_window(
    qt_app, monkeypatch, quantities=None, x='p', y='a', nrows=4, n_series=3,
    **kwargs
):
    """
    Build a TSSeriesWindow over mock runners with deterministic quantities.

    By default the quantities are ``p`` (= series index + 1), ``a`` (= 10 *
    series index + data_idx) and ``b`` (= -series index), each counting calls.

    Parameters:
    qt_app (QApplication): Qt application fixture.
    monkeypatch (pytest.MonkeyPatch): Used to stub panels and timers.
    quantities (dict or None): Quantities, or None for the defaults above.
    x, y (str): Initial quantities.
    nrows (int): Rows per mock dataset.
    n_series (int): Number of series indices.
    **kwargs: Extra TSSeriesWindow arguments.

    Returns:
    win (TSSeriesWindow): The window.
    calls (dict): Call count per quantity label.
    """
    steps = [
        plStep(name, lambda x=None: x, ['x'], [f'{name}_out'], 'per-row')
        for group in its._TS_PANELS for name in group
    ]
    ars = []
    for _ in range(n_series):
        ar = MagicMock()
        ar.analysis_steps = steps
        ar.path = [{'task': step} for step in steps]
        ar.DS = MagicMock()
        ar.DS.nrows = nrows
        ars.append(ar)
    calls = {}
    if quantities is None:
        def counted(label, func):
            def getter(AR, di):
                calls[label] = calls.get(label, 0) + 1
                return func(ars.index(AR), di)
            return getter
        quantities = {
            'p': counted('p', lambda si, di: si + 1.0),
            'a': counted('a', lambda si, di: 10.0 * si + di),
            'b': counted('b', lambda si, di: -float(si)),
        }
    monkeypatch.setattr(iseries, 'get_panel_class', lambda _names: _Panel)
    monkeypatch.setattr(iseries.QtCore.QTimer, 'singleShot', lambda *_a, **_k: None)
    win = its.TSSeriesWindow(ars, x=x, y=y, quantities=quantities, start_idx=0, title='t',
                             **kwargs)
    return win, calls


def _close(win):
    """
    Close a window, accepting any confirmation prompt.

    Parameters:
    win (QMainWindow): Window to close.
    """
    with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
        win.close()


def test_default_quantities_include_reduced_psds():
    assert its.DEFAULT_TS_QUANTITIES[:2] == ['ares', 'fres']
    assert 'sxx_10' in its.DEFAULT_TS_QUANTITIES
    assert 'sxx_0p1' in its.DEFAULT_TS_QUANTITIES
    assert 'sfactor_300' in its.DEFAULT_TS_QUANTITIES
    getters = its.normalize_quantities()
    assert list(getters) == its.DEFAULT_TS_QUANTITIES


def test_normalize_quantities_accepts_names_and_callables():
    getters = its.normalize_quantities({'P': 'ares', 'double': lambda AR, di: 2 * di,
                                        'broken': lambda AR, di: 1 / 0})
    assert list(getters) == ['P', 'double', 'broken']
    assert getters['double'](None, 3) == 6.0
    assert getters['broken'](None, 3) is None

    with pytest.raises(TypeError, match='parameter name, a callable, or SeriesValues'):
        its.normalize_quantities({'bad': 3})
    with pytest.raises(ValueError, match='at least one'):
        its.normalize_quantities([])


def test_window_uses_ts_panels_and_initial_quantities(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch)

    assert [p.step_names for p in win.panels] == [tuple(names) for names in its._TS_PANELS]
    assert win._precompute_values is True
    assert (win._x_combo.currentText(), win._y_combo.currentText()) == ('p', 'a')
    assert [win._y_combo.itemText(i) for i in range(win._y_combo.count())] == ['p', 'a', 'b']

    np.testing.assert_array_equal(win._get_x_array(0), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(win._get_y_array(2), [2.0, 12.0, 22.0])
    _close(win)


@pytest.mark.parametrize('x, y', [('nope', 'a'), ('p', 'nope')])
def test_unknown_initial_quantity_raises(qt_app, monkeypatch, x, y):
    with pytest.raises(ValueError, match="'nope' is not one of the quantities"):
        _make_window(qt_app, monkeypatch, x=x, y=y)


def test_changing_dropdown_replots_and_reuses_cached_values(qt_app, monkeypatch):
    win, calls = _make_window(qt_app, monkeypatch)
    win._update_series_scatter()
    assert calls['a'] == 3

    win._y_combo.setCurrentText('b')

    assert (win._y_key, win._y_name) == ('b', 'b')
    assert win._plot_series.getAxis('left').labelText == 'b'
    np.testing.assert_array_equal(win._get_y_array(0), [0.0, -1.0, -2.0])
    assert calls['b'] == 3

    win.select_quantities(y='a')          # back to a: cached, no new calls
    np.testing.assert_array_equal(win._get_y_array(0), [0.0, 10.0, 20.0])
    assert calls['a'] == 3

    win.select_quantities(x='b')
    assert win._series_combo_label.text() == 'series index, b:'
    with pytest.raises(ValueError, match='not one of the quantities'):
        win.select_quantities(x='nope')
    _close(win)


def test_worker_precomputes_values_for_current_quantities(qt_app, monkeypatch):
    win, calls = _make_window(qt_app, monkeypatch)
    worker_ars = win._ARs
    win._runner_outputs_exist = lambda _ar, _di: True   # already pre-fitted

    # The current row (0) is handled by the UI, not the worker.
    assert win._remaining_rows() == [1, 2, 3]
    win._initialize_remaining_data_indices(worker_ars, [1, 2, 3, 0])

    assert win._remaining_rows() == []
    assert win._values[('a', 2, 3)] == 23.0
    assert win._values[('p', 0, 1)] == 1.0
    calls.clear()
    np.testing.assert_array_equal(win._get_y_array(3), [3.0, 13.0, 23.0])
    assert calls == {}                                   # served from the worker's values

    win.select_quantities(y='b')
    assert win._remaining_rows() == [1, 2, 3]            # b still to compute
    _close(win)


def test_forget_values_and_nan_caching_rules(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch)
    win._values[('a', 0, 1)] = 5.0
    win._values[('b', 1, 1)] = 6.0
    win._values[('a', 0, 2)] = 7.0

    win._forget_values(1, [0])
    assert ('a', 0, 1) not in win._values
    assert ('b', 1, 1) in win._values
    win._forget_values(1)
    assert ('b', 1, 1) not in win._values
    assert ('a', 0, 2) in win._values

    # NaN is only cached once the row is pre-fitted for every series point.
    win._cached_value('a', 0, 3, lambda: np.nan)
    assert ('a', 0, 3) not in win._values
    win._initialized_data_idxs.add(3)
    win._cached_value('a', 0, 3, lambda: np.nan)
    assert np.isnan(win._values[('a', 0, 3)])
    _close(win)


def test_dataset_quantity_reads_per_row_and_global_values():
    from citkid.pipeline_v2.framework import LazyAttr

    lazy = MagicMock(spec=LazyAttr)
    lazy.__getitem__.side_effect = lambda di: np.float64(di * 2)
    ar = MagicMock()
    ar.DS.per_row = lazy
    ar.DS.glob = np.asarray(3.5)
    ar.DS.vector = np.arange(3.0)
    ar.DS.cplx = np.asarray(1 + 2j)

    assert iseries.dataset_quantity('per_row')(ar, 4) == 8.0
    assert iseries.dataset_quantity('glob')(ar, 4) == 3.5
    assert iseries.dataset_quantity('vector')(ar, 4) is None
    assert iseries.dataset_quantity('cplx')(ar, 4) is None
    assert iseries.dataset_quantity('missing')(MagicMock(DS=object()), 0) is None


def _real_datasets(tmp_path, nrows_list):
    """
    Build real DataSets with a trivial calibration, one per entry.

    Parameters:
    tmp_path (pathlib.Path): Directory for the zarr store and YAML.
    nrows_list (list of int): Number of rows of each dataset.

    Returns:
    datasets (list of DataSet): The datasets, in groups ``s0``, ``s1``, ...
    """
    from citkid.pipeline_v2.dataset import DataSet
    import zarr

    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text("CAL_STEPS:\n  1:\n    task: load_n\n", encoding="utf-8")
    root = zarr.open_group(str(tmp_path / "series.zarr"), mode="w")
    datasets = []
    for i, nrows in enumerate(nrows_list):
        steps = [plStep("load_n", lambda n=nrows: n, [], ["nrows"], "global")]
        datasets.append(DataSet(zarr_path=root.require_group(f"s{i}"), cal_yaml_path=str(cal_yaml),
                                custom_cal_steps=steps))
    return datasets


def test_series_runners_from_datasets(tmp_path):
    datasets = _real_datasets(tmp_path, [3, 3])

    ARs = iseries.series_runners(datasets=datasets, analysis_yaml_path=None)

    assert [AR.DS for AR in ARs] == datasets


@pytest.mark.parametrize("kwargs, error, match", [
    ({}, ValueError, "exactly one of"),
    ({"make_custom_steps": lambda i: [], "datasets": []}, ValueError, "exactly one of"),
    ({"make_custom_steps": lambda i: []}, ValueError, "requires root and n_series"),
    ({"datasets": []}, ValueError, "at least one"),
    ({"datasets": ["not a dataset"]}, TypeError, "must be DataSet"),
])
def test_series_runners_rejects_bad_inputs(kwargs, error, match):
    with pytest.raises(error, match=match):
        iseries.series_runners(**kwargs)


def test_series_runners_checks_counts_and_rows(tmp_path):
    datasets = _real_datasets(tmp_path, [3, 4])
    with pytest.raises(ValueError, match="does not match len"):
        iseries.series_runners(datasets=datasets[:1], n_series=2, analysis_yaml_path=None)
    with pytest.raises(ValueError, match="same number of rows"):
        iseries.series_runners(datasets=datasets, analysis_yaml_path=None)


def test_window_buffers_writes_and_restores_setting(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch)
    for ar in win._ARs:
        assert ar.DS.write_buffer is True
    win._original_write_buffer = [False] * len(win._ARs)

    _close(win)

    assert [ar.DS.write_buffer for ar in win._ARs] == [False] * len(win._ARs)


def test_worker_runners_are_copies_of_the_datasets(qt_app, monkeypatch, tmp_path):
    datasets = _real_datasets(tmp_path, [3, 3])
    ARs = iseries.series_runners(datasets=datasets, analysis_yaml_path=None)
    monkeypatch.setattr(iseries, "get_panel_class", lambda _names: _Panel)
    monkeypatch.setattr(iseries.QtCore.QTimer, "singleShot", lambda *_a, **_k: None)
    win = iseries.IQSeriesWindow(
        ARs, x_param_name="nrows", x_name="n", y_func=lambda ar, di: None, y_name="y",
        panels=[], start_idx=0,
    )

    worker_ars = win._make_worker_ars()

    for worker, ar in zip(worker_ars, ARs):
        assert worker.DS is not ar.DS
        assert worker.DS.root.path == ar.DS.root.path
        assert worker.DS.write_buffer is True        # window buffers during the session
    _close(win)



def test_series_values_lookup_and_validation():
    values = iseries.SeriesValues([-30.0, np.nan, -20.0])

    assert len(values) == 3
    assert values.value(0) == -30.0
    assert np.isnan(values.value(1))
    with pytest.raises(TypeError, match="series index"):
        values(None, 0)
    for bad in ([], [[1.0, 2.0]]):
        with pytest.raises(ValueError, match="1-D"):
            iseries.SeriesValues(bad)


def test_ts_x_values_become_the_default_x_quantity(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch, x=None, x_values=[-30.0, -25.0, -20.0],
                          x_name='Power (dBm)')

    assert win._x_combo.currentText() == 'Power (dBm)'
    assert 'Power (dBm)' in [win._y_combo.itemText(i) for i in range(win._y_combo.count())]
    np.testing.assert_array_equal(win._get_x_array(2), [-30.0, -25.0, -20.0])
    np.testing.assert_array_equal(win._get_y_array(2), [2.0, 12.0, 22.0])

    win.select_quantities(x='p')                       # another quantity, then back
    np.testing.assert_array_equal(win._get_x_array(2), [1.0, 2.0, 3.0])
    win.select_quantities(x='Power (dBm)', y='Power (dBm)')
    np.testing.assert_array_equal(win._get_y_array(1), [-30.0, -25.0, -20.0])
    _close(win)


def test_ts_explicit_x_wins_over_x_values(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch, x='b', x_values=[1.0, 2.0, 3.0])

    assert win._x_combo.currentText() == 'b'
    assert win._quantities['x'].value(1) == 2.0         # default label 'x'
    _close(win)


@pytest.mark.parametrize("kwargs", [
    {"x_values": [1.0, 2.0]},
    {"quantities": {"p": iseries.SeriesValues([1.0, 2.0]), "a": "a"}, "x": "p"},
])
def test_series_values_must_match_number_of_series_points(qt_app, monkeypatch, kwargs):
    kwargs.setdefault("x", None)
    with pytest.raises(ValueError, match="3 series points"):
        _make_window(qt_app, monkeypatch, **kwargs)


def test_worker_uses_series_index_for_series_values(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch, x=None, x_values=[5.0, 6.0, 7.0])
    worker_ars = [MagicMock(), MagicMock(), MagicMock()]      # copies: not the UI runners

    x, _ = win._compute_row_values(worker_ars, 1, store=True)

    np.testing.assert_array_equal(x, [5.0, 6.0, 7.0])
    assert win._values[('x', 2, 1)] == 7.0
    _close(win)


def test_iq_window_x_values(qt_app, monkeypatch):
    monkeypatch.setattr(iseries, "get_panel_class", lambda _names: _Panel)
    monkeypatch.setattr(iseries.QtCore.QTimer, "singleShot", lambda *_a, **_k: None)
    ars = []
    for _ in range(3):
        ar = MagicMock()
        ar.DS.nrows = 4
        ars.append(ar)
    win = iseries.IQSeriesWindow(
        ars, x_param_name='ares', x_name='T (mK)', y_func=lambda ar, di: 1.0, y_name='y',
        panels=[], start_idx=0, x_values=[10.0, 20.0, 30.0],
    )

    np.testing.assert_array_equal(win._get_x_array(0), [10.0, 20.0, 30.0])
    win._update_series_combo_items(0)
    assert win._series_combo.itemText(1) == '2, 20'
    _close(win)

    with pytest.raises(ValueError, match="3 series points"):
        iseries.IQSeriesWindow(
            ars, x_param_name='ares', x_name='x', y_func=lambda ar, di: 1.0, y_name='y',
            panels=[], start_idx=0, x_values=[10.0],
        )


def test_normalize_quantities_keeps_series_values():
    values = iseries.SeriesValues([1.0, 2.0])

    getters = its.normalize_quantities({"P": values, "fres": "fres"})

    assert getters["P"] is values


def test_ts_window_has_no_mark_bad_above(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch)
    win._mark_bad_above = MagicMock()

    texts = [b.text() for b in win.findChildren(QtWidgets.QPushButton)]
    hints = [l.text() for l in win.findChildren(QtWidgets.QLabel) if 'mark all bad' in l.text()]
    handled = win._handle_modified_letter_shortcut(
        iseries.QtCore.Qt.Key.Key_B, iseries._Qt.ShiftModifier | iseries._Qt.ControlModifier)

    assert 'Mark Bad Above' not in texts
    assert hints and 'mark bad above' not in hints[0]
    assert handled is False
    win._mark_bad_above.assert_not_called()
    _close(win)


def test_ts_window_shift_b_marks_all_series_bad(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch)
    win._mark_all_series_bad = MagicMock()

    handled = win._handle_modified_letter_shortcut(
        iseries.QtCore.Qt.Key.Key_B, iseries._Qt.ShiftModifier)

    assert handled is True
    win._mark_all_series_bad.assert_called_once()
    _close(win)


def _drawn_points(win):
    """
    Return the drawn series-scatter points in plot coordinates.

    Parameters:
    win (IQSeriesWindow): Window.

    Returns:
    points (dict): Series index -> (x, y) of each drawn point.
    """
    return {i: (float(item.data['x'][0]), float(item.data['y'][0]))
            for i, item in enumerate(win._scatter_items) if len(item.data)}


def test_log_scales_place_points_in_log_coordinates(qt_app, monkeypatch):
    # data_idx 0: x = p = 1, 2, 3 and y = a = 0, 10, 20 (y = 0 can't be drawn on log)
    win, _ = _make_window(qt_app, monkeypatch, xscale='log', yscale='log')
    win._series_idx = 2
    win._update_series_scatter()

    ctrl = win._plot_series.ctrl
    assert ctrl.logXCheck.isChecked() and ctrl.logYCheck.isChecked()
    points = _drawn_points(win)
    assert sorted(points) == [1, 2]
    np.testing.assert_allclose(points[1], (np.log10(2.0), 1.0))
    np.testing.assert_allclose(points[2], (np.log10(3.0), np.log10(20.0)))
    np.testing.assert_allclose(
        (win._selected_marker.data['x'][0], win._selected_marker.data['y'][0]),
        (np.log10(3.0), np.log10(20.0)))
    _close(win)


def test_linear_scale_is_default_and_menu_toggle_redraws(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch)
    win._update_series_scatter()
    assert not win._plot_series.ctrl.logXCheck.isChecked()
    assert _drawn_points(win)[2] == (3.0, 20.0)

    win._plot_series.ctrl.logXCheck.setChecked(True)   # as from the right-click menu

    np.testing.assert_allclose(_drawn_points(win)[2], (np.log10(3.0), 20.0))
    _close(win)


def test_log_scale_survives_quantity_change(qt_app, monkeypatch):
    win, _ = _make_window(qt_app, monkeypatch, yscale='log')
    win._y_combo.setCurrentText('b')                   # b = 0, -1, -2: nothing drawable

    assert win._plot_series.ctrl.logYCheck.isChecked()
    assert _drawn_points(win) == {}
    _close(win)


@pytest.mark.parametrize('xscale, expected', [
    ('linear', [1.0, 2.0, 3.0]),
    ('log', [1.0, 3.0 ** 0.5, 3.0]),
])
def test_xy_fit_curve_follows_x_scale(qt_app, monkeypatch, xscale, expected):
    from citkid.pipeline_v2.series_xy_fit import SeriesXYFit

    xy_fit = SeriesXYFit(fit=lambda x, y: (1.0, 0.0), param_names=['slope', 'intercept'],
                         model=lambda xs, m, c: m * xs + c, n_samples=3)
    win, _ = _make_window(qt_app, monkeypatch, xy_fit=xy_fit, xscale=xscale)  # x = p = 1, 2, 3
    win._update_series_scatter()

    xs, _ = win._xy_fit_curve.getOriginalDataset()
    np.testing.assert_allclose(xs, expected)
    _close(win)


@pytest.mark.parametrize('kwargs', [{'xscale': 'symlog'}, {'yscale': 'Log'}])
def test_bad_scale_raises(qt_app, monkeypatch, kwargs):
    with pytest.raises(ValueError, match="'linear', 'log'"):
        _make_window(qt_app, monkeypatch, **kwargs)


@pytest.mark.parametrize('module, run, window', [
    (its, 'run_ts_series', 'TSSeriesWindow'),
    (iseries, 'run_iq_series', 'IQSeriesWindow'),
])
def test_run_functions_pass_scales(monkeypatch, module, run, window):
    created = MagicMock()
    monkeypatch.setattr(module, window, created)
    monkeypatch.setattr(module, 'series_runners', lambda *_a, **_k: [MagicMock()])
    monkeypatch.setattr(module, 'get_qapp', lambda _title: MagicMock())

    getattr(module, run)(datasets=[MagicMock()], xscale='log', yscale='linear')

    assert created.call_args.kwargs['xscale'] == 'log'
    assert created.call_args.kwargs['yscale'] == 'linear'


@pytest.mark.parametrize('refit', ['panel_run', 'apply_to_all', 'mark_bad'])
def test_refits_rescale_the_series_plot(qt_app, monkeypatch, refit):
    win, _ = _make_window(qt_app, monkeypatch)
    win._series_idx = 0
    win._update_series_scatter(rescale=True)
    view = win._plot_series.getViewBox()
    data_range = view.viewRange()
    view.setRange(xRange=(100, 200), yRange=(100, 200), padding=0)   # user zoom

    win._update_series_scatter()                     # plain redraw keeps the zoom
    assert view.viewRange()[0] == pytest.approx([100, 200])

    for panel in win.panels:
        panel._store_nan_outputs = lambda **_kw: None
    {'panel_run': lambda: win._update_series_point(0),
     'apply_to_all': win._apply_to_all,
     'mark_bad': lambda: win._mark_series_bad([1])}[refit]()
    win._wait_for_marks()

    assert view.viewRange()[0][1] < 100              # back on the data
    assert view.viewRange()[0] == pytest.approx(data_range[0], rel=0.5)
    _close(win)
