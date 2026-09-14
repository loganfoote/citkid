import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pyqtgraph.Qt import QtWidgets

import citkid.pipeline_v2.interactive.core as icore
from citkid.pipeline_v2.framework import plStep
from citkid.pipeline_v2.interactive.core import (
    DefaultStepPanel,
    InteractiveAnalysisWindow,
    StepPanel,
)


def _make_step(name, func_type='per-row', return_names=None):
    if return_names is None:
        return_names = ['y']
    return plStep(name, lambda x=None: x, ['x'], return_names, func_type)


def _make_ar(*step_names, func_type='per-row'):
    steps = [_make_step(name, func_type=func_type) for name in step_names]
    ar = MagicMock()
    ar.analysis_steps = steps
    ar._last_failures = {}
    ar.execute_step.return_value = None
    ar._resolve_step_scope.return_value = ('analysis', 1)
    ar.DS = MagicMock()
    ar.DS.nrows = 3
    return ar, steps


class _ParamPanel(StepPanel):
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self._status_label = QtWidgets.QLabel('—')
        layout.addWidget(self._status_label)

    def get_params_for_step(self, step):
        return {'offset': 3.5}


class _WindowPanel(StepPanel):
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self._status_label = QtWidgets.QLabel('ok')
        layout.addWidget(self._status_label)
        self.clear_calls = 0
        self.update_calls = 0

    def update_plots(self):
        self.update_calls += 1

    def clear_plots(self):
        self.clear_calls += 1

    def _outputs_exist(self):
        return True


class TestStepPanelV2:
    def test_default_panel_has_run_button(self, qt_app):
        ar, _ = _make_ar('step_a')
        panel = DefaultStepPanel(ar, ('step_a',))
        assert hasattr(panel, '_run_btn')

    def test_save_outputs_persists_user_params_before_outputs(self, qt_app):
        ar, steps = _make_ar('step_b')
        panel = _ParamPanel(ar, ('step_b',), data_idx=2)

        panel.save_outputs()

        ar._add_user_params.assert_called_once()
        _, kwargs = ar._add_user_params.call_args
        assert kwargs['data_idx'] == 2
        assert kwargs['save'] is True
        ar.save_step_outputs.assert_called_once_with(steps[0], data_idx=2)

    def test_write_nan_outputs_is_row_scoped(self, qt_app):
        ar, _ = _make_ar('step_c')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('step_c',), data_idx=1)
        panel._nan_outputs = lambda: {'y': np.nan}

        assert panel._write_nan_outputs() is True

        invalidate_args = ar.DS.invalidate_memory_params.call_args.kwargs
        delete_args = ar.DS.delete_saved_params.call_args.kwargs
        store_kwargs = ar.DS._store_param.call_args.kwargs
        np.testing.assert_array_equal(invalidate_args['data_idx'], np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(delete_args['data_idx'], np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(store_kwargs['data_idx'], np.array([1], dtype=np.int32))
        assert panel._dirty is True
        assert panel._needs_run is False


class TestInteractiveWindowV2:
    def _make_window(self, qt_app, monkeypatch):
        step1 = _make_step('step1')
        step2 = _make_step('step2')
        ar = MagicMock()
        ar.analysis_steps = [step1, step2]
        ar._last_failures = {}
        ar.execute_step.return_value = None
        ar.DS = MagicMock()
        ar.DS.nrows = 2
        monkeypatch.setattr(icore, 'get_panel_class', lambda _names: _WindowPanel)
        monkeypatch.setattr(icore.QtCore.QTimer, 'singleShot', lambda *_args, **_kwargs: None)
        return InteractiveAnalysisWindow(
            ar,
            panels=[('step1',), ('step2',)],
            start_idx=0,
            data_idxs=[0, 1],
            title='test',
        )

    def test_run_panel_marks_downstream_stale(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)

        win._run_panel_by_index(0)

        assert win.panels[0]._has_run is True
        assert win.panels[1]._needs_run is True
        assert win.panels[1].clear_calls == 1
        assert win.AR.execute_step.call_count == 1
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_navigation_cancel_keeps_current_index_when_stale(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win.panels[1]._needs_run = True

        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.No):
            win._advance(+1)

        assert win._nav_pos == 0
        assert win._idx_spin.value() == 0
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_panel_save_cancel_skips_persist_when_downstream_stale(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win.panels[1]._needs_run = True

        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.No):
            win.panels[0]._on_save_clicked()

        assert win.AR.save_step_outputs.call_count == 0
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()