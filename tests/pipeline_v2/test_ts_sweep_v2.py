"""
Tests for citkid.pipeline_v2.interactive.ts_sweep and the sweep-window
features it relies on (selectable quantities, precomputed values).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

import citkid.pipeline_v2.interactive.sweep_fitter as isweep
import citkid.pipeline_v2.interactive.ts_sweep as its
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


def _make_window(qt_app, monkeypatch, quantities=None, x='p', y='a', nrows=4, n_sweep=3):
    """
    Build a TSSweepWindow over mock runners with deterministic quantities.

    By default the quantities are ``p`` (= sweep index + 1), ``a`` (= 10 *
    sweep index + data_idx) and ``b`` (= -sweep index), each counting calls.

    Parameters:
    qt_app (QApplication): Qt application fixture.
    monkeypatch (pytest.MonkeyPatch): Used to stub panels and timers.
    quantities (dict or None): Quantities, or None for the defaults above.
    x, y (str): Initial quantities.
    nrows (int): Rows per mock dataset.
    n_sweep (int): Number of sweep indices.

    Returns:
    win (TSSweepWindow): The window.
    calls (dict): Call count per quantity label.
    """
    steps = [
        plStep(name, lambda x=None: x, ['x'], [f'{name}_out'], 'per-row')
        for group in its._TS_PANELS for name in group
    ]
    ars = []
    for _ in range(n_sweep):
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
    monkeypatch.setattr(isweep, 'get_panel_class', lambda _names: _Panel)
    monkeypatch.setattr(isweep.QtCore.QTimer, 'singleShot', lambda *_a, **_k: None)
    win = its.TSSweepWindow(ars, x=x, y=y, quantities=quantities, start_idx=0, title='t')
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

    with pytest.raises(TypeError, match='parameter name or a callable'):
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
    win._update_sweep_scatter()
    assert calls['a'] == 3

    win._y_combo.setCurrentText('b')

    assert (win._y_key, win._y_name) == ('b', 'b')
    assert win._plot_sweep.getAxis('left').labelText == 'b'
    np.testing.assert_array_equal(win._get_y_array(0), [0.0, -1.0, -2.0])
    assert calls['b'] == 3

    win.select_quantities(y='a')          # back to a: cached, no new calls
    np.testing.assert_array_equal(win._get_y_array(0), [0.0, 10.0, 20.0])
    assert calls['a'] == 3

    win.select_quantities(x='b')
    assert win._sweep_combo_label.text() == 'sweep index, b:'
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

    # NaN is only cached once the row is pre-fitted for every sweep.
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

    assert isweep.dataset_quantity('per_row')(ar, 4) == 8.0
    assert isweep.dataset_quantity('glob')(ar, 4) == 3.5
    assert isweep.dataset_quantity('vector')(ar, 4) is None
    assert isweep.dataset_quantity('cplx')(ar, 4) is None
    assert isweep.dataset_quantity('missing')(MagicMock(DS=object()), 0) is None
