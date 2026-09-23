import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pyqtgraph.Qt import QtWidgets

import citkid.pipeline_v2.interactive.core as icore
import citkid.pipeline_v2.interactive.sweep_fitter as isweep
from citkid.pipeline_v2.framework import plStep
from citkid.pipeline_v2.interactive.core import (
    DefaultStepPanel,
    InteractiveAnalysisWindow,
    StepPanel,
)
from citkid.pipeline_v2.interactive.sweep_fitter import SweepFitterWindow


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


class TestSweepFitterWindowV2:
    def _make_window(self, qt_app, monkeypatch):
        step1 = _make_step('fit_gain')
        step2 = _make_step('fit_iq')
        ars = []
        for _ in range(2):
            ar = MagicMock()
            ar.analysis_steps = [step1, step2]
            ar.path = [{'task': step1}, {'task': step2}]
            ar._last_failures = {}
            ar.execute_step.return_value = None
            ar.DS = MagicMock()
            ar.DS.nrows = 2
            ars.append(ar)
        monkeypatch.setattr(isweep, 'get_panel_class', lambda _names: _WindowPanel)
        monkeypatch.setattr(isweep.QtCore.QTimer, 'singleShot', lambda *_args, **_kwargs: None)
        return SweepFitterWindow(
            ars,
            x_param_name='x',
            x_name='X',
            y_func=lambda _ar, _di: None,
            y_name='Y',
            start_sweep_idx=0,
            start_data_idx=0,
            title='test',
        )

    def test_run_panel_marks_downstream_stale(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._sweep_idx = 0
        win._update_sweep_point = MagicMock()

        win._run_panel_by_index(0)

        assert win.panels[0]._has_run is True
        assert win.panels[1]._needs_run is True
        assert win.panels[1].clear_calls == 1
        win._update_sweep_point.assert_called_once_with(0)
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_data_idx_navigation_cancel_keeps_state_when_stale(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win.panels[1]._needs_run = True

        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.No):
            win._set_data_idx(1)

        assert win._data_idx == 0
        assert win._data_idx_spin.value() == 0
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_panel_save_cancel_skips_persist_when_downstream_stale(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._sweep_idx = 0
        win.panels[1]._needs_run = True

        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.No):
            win.panels[0]._on_save_clicked()

        assert win._ARs[0].save_step_outputs.call_count == 0
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_prefetch_next_does_not_execute_pipeline(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._sweep_idx = 0

        win._prefetch_next()
        if win._prefetch_thread is not None:
            win._prefetch_thread.join(timeout=2)

        assert all(ar.execute_path.call_count == 0 for ar in win._ARs)
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_batch_init_runs_global_prefix_once_then_suffix_per_index(self, qt_app, monkeypatch):
        global_step = _make_step('global_step', func_type='global', return_names=['g'])
        per_row_step = _make_step('fit_gain', return_names=['y'])
        final_step = _make_step('fit_iq', return_names=['z'])
        ar = MagicMock()
        ar.analysis_steps = [global_step, per_row_step, final_step]
        ar.path = [{'task': global_step}, {'task': per_row_step}, {'task': final_step}]
        ar._last_failures = {}
        ar.execute_step.return_value = None
        ar.execute_path.return_value = None
        ar.DS = MagicMock()
        ar.DS.nrows = 3

        ars = [ar]
        monkeypatch.setattr(isweep, 'get_panel_class', lambda _names: _WindowPanel)
        monkeypatch.setattr(isweep.QtCore.QTimer, 'singleShot', lambda *_args, **_kwargs: None)

        win = SweepFitterWindow(
            ars,
            x_param_name='x',
            x_name='X',
            y_func=lambda _ar, _di: None,
            y_name='Y',
            start_sweep_idx=0,
            start_data_idx=0,
            title='test',
        )

        global_ready = {'done': False}

        def fake_step_outputs_exist(_ar, step, data_idx):
            if step.name == 'global_step':
                return global_ready['done']
            return False

        def fake_execute_step(step, data_idx=None, save=False, **_kwargs):
            if step.name == 'global_step':
                global_ready['done'] = True

        ar.execute_step.side_effect = fake_execute_step
        win._step_outputs_exist = fake_step_outputs_exist
        win._runner_outputs_exist = lambda _ar, _di: False

        win._batch_init_all_sweeps()
        win._data_idx = 1
        win._batch_init_all_sweeps()

        assert ar.execute_step.call_count == 1
        ar.execute_step.assert_called_with(global_step, data_idx=None, save=True)
        assert ar.execute_path.call_count == 2
        assert ar.execute_path.call_args_list[0].kwargs['start_from_idx'] == 1
        assert ar.execute_path.call_args_list[0].kwargs['data_idx'] == 0
        assert ar.execute_path.call_args_list[1].kwargs['start_from_idx'] == 1
        assert ar.execute_path.call_args_list[1].kwargs['data_idx'] == 1
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_shift_b_shortcut_marks_all_sweeps_bad(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._mark_all_sweeps_bad = MagicMock()

        handled = win._handle_modified_letter_shortcut(
            isweep.QtCore.Qt.Key.Key_B,
            isweep._Qt.ShiftModifier,
        )

        assert handled is True
        win._mark_all_sweeps_bad.assert_called_once()
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_shift_a_shortcut_applies_to_all(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._apply_to_all = MagicMock()

        handled = win._handle_modified_letter_shortcut(
            isweep.QtCore.Qt.Key.Key_A,
            isweep._Qt.ShiftModifier,
        )

        assert handled is True
        win._apply_to_all.assert_called_once()
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()