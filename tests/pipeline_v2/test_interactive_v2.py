import threading

import numpy as np
import pytest
import zarr
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
    def _make_window(self, qt_app, monkeypatch, nrows=2, data_idxs=None, start_idx=0,
                     state_group=None):
        """
        Build a SweepFitterWindow over two mock sweep runners.

        Parameters:
        qt_app (QApplication): Qt application fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture used to stub panels/timers.
        nrows (int): Number of rows reported by each mock dataset.
        data_idxs (list of int or None): Subset passed to the window.
        start_idx (int or None): Starting position in ``data_idxs``; None
            resumes from ``state_group``.
        state_group (zarr.Group or None): Group for persistent state.

        Returns:
        win (SweepFitterWindow): The window.
        """
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
            ar.DS.nrows = nrows
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
            start_idx=start_idx,
            data_idxs=data_idxs,
            title='test',
            state_group=state_group,
        )

    @staticmethod
    def _stub_refresh(win):
        """
        Replace plot refresh and row initialization with no-op mocks.

        Parameters:
        win (SweepFitterWindow): Window to stub.
        """
        for name in (
            '_update_sweep_combo_items', '_update_sweep_scatter',
            '_update_waterfall', '_autoscale_all', '_ensure_data_idx_initialized',
        ):
            setattr(win, name, MagicMock())
        win._all_sweeps_outputs_exist = lambda _di: True

    @staticmethod
    def _close(win):
        """
        Close the window, accepting any confirmation prompt.

        Parameters:
        win (SweepFitterWindow): Window to close.
        """
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_default_data_idxs_covers_all_rows(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=3)

        assert win._data_idxs == [0, 1, 2]
        assert win._data_idx == 0
        assert win._res_label.text() == '1 / 3'
        self._close(win)

    def test_data_idxs_subset_navigation(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=6, data_idxs=[4, 1, 3], start_idx=1)
        self._stub_refresh(win)

        assert win._data_idx == 1
        assert win._res_label.text() == '2 / 3'
        assert (win._data_idx_spin.minimum(), win._data_idx_spin.maximum()) == (1, 4)

        win._advance_resonator(+1)
        assert (win._data_idx, win._nav_pos) == (3, 2)
        assert win._res_label.text() == '3 / 3'

        win._advance_resonator(+1)
        assert win._data_idx == 3

        win._advance_resonator(-5)
        assert (win._data_idx, win._nav_pos) == (4, 0)
        assert win._data_idx_spin.value() == 4
        self._close(win)

    @pytest.mark.parametrize(
        'spin_value, expected',
        [(2, 3), (4, 4), (0, 1), (5, 1)],
    )
    def test_spin_box_snaps_to_subset(self, qt_app, monkeypatch, spin_value, expected):
        win = self._make_window(qt_app, monkeypatch, nrows=6, data_idxs=[1, 3, 4])
        self._stub_refresh(win)
        win._data_idx_spin.setMaximum(5)
        win._data_idx_spin.setMinimum(0)

        win._data_idx_spin.setValue(spin_value)

        assert win._data_idx == expected
        assert win._data_idx_spin.value() == expected
        self._close(win)

    def test_prefetch_uses_next_subset_index(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=6, data_idxs=[5, 2])
        win._get_x_array = MagicMock()
        win._get_y_array = MagicMock()

        win._prefetch_next()
        win._prefetch_thread.join(timeout=2)

        win._get_x_array.assert_called_once_with(2)
        self._close(win)

    def test_background_init_only_visits_subset(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=6, data_idxs=[5, 2, 0])
        win._worker_root = MagicMock()
        win._all_sweeps_outputs_exist = lambda _di: False
        win._make_worker_ars = MagicMock(return_value=[])
        visited = []
        win._initialize_remaining_data_indices = lambda _ars, dis: visited.extend(dis)

        win._start_background_initialize_remaining()
        win._init_all_thread.join(timeout=2)

        assert visited == [2, 0]
        self._close(win)

    @pytest.mark.parametrize('data_idxs', [[], [0, 2], [-1]])
    def test_invalid_data_idxs_raise(self, qt_app, monkeypatch, data_idxs):
        with pytest.raises(ValueError, match='data_idxs'):
            self._make_window(qt_app, monkeypatch, nrows=2, data_idxs=data_idxs)

    def test_background_init_continues_after_failed_fit(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=3)
        worker_ars = [MagicMock(), MagicMock()]
        attempted = []

        def fake_init(ar, di):
            attempted.append((worker_ars.index(ar), di))
            if (worker_ars.index(ar), di) == (0, 1):
                raise PermissionError('locked')

        win._runner_outputs_exist = lambda _ar, _di: False
        win._initialize_runner_outputs = fake_init

        win._initialize_remaining_data_indices(worker_ars, [1, 2])

        assert attempted == [(0, 1), (1, 1), (0, 2), (1, 2)]
        assert {1, 2} <= win._initialized_data_idxs
        worker_ars[0].release_rows.assert_any_call(1)
        self._close(win)

    def test_background_init_stops_between_sweeps(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=3)
        worker_ars = [MagicMock(), MagicMock()]
        attempted = []

        def fake_init(ar, di):
            attempted.append((worker_ars.index(ar), di))
            win._init_all_stop.set()

        win._runner_outputs_exist = lambda _ar, _di: False
        win._initialize_runner_outputs = fake_init

        win._initialize_remaining_data_indices(worker_ars, [1, 2])

        assert attempted == [(0, 1)]
        assert 1 not in win._initialized_data_idxs
        self._close(win)

    def test_attempted_rows_are_not_refit(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=4)
        win._initialized_data_idxs = {0, 2}
        win._all_sweeps_outputs_exist = lambda _di: False
        win._initialize_all_sweeps_for_data_idx = MagicMock()

        win._ensure_data_idx_initialized(2)
        win._initialize_all_sweeps_for_data_idx.assert_not_called()

        win._worker_root = MagicMock()
        win._make_worker_ars = MagicMock(return_value=[])
        visited = []
        win._initialize_remaining_data_indices = lambda _ars, dis: visited.extend(dis)
        win._start_background_initialize_remaining()
        win._init_all_thread.join(timeout=2)

        assert visited == [1, 3]
        self._close(win)

    def test_background_order_starts_after_current_position(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=6, data_idxs=[5, 4, 3, 2, 1], start_idx=2)
        win._worker_root = MagicMock()
        win._make_worker_ars = MagicMock(return_value=[])
        visited = []
        win._initialize_remaining_data_indices = lambda _ars, dis: visited.extend(dis)

        win._start_background_initialize_remaining()
        win._init_all_thread.join(timeout=2)

        assert visited == [2, 1, 5, 4]
        self._close(win)

    def test_resume_starts_at_first_unviewed_index(self, qt_app, monkeypatch, tmp_path):
        state_group = zarr.open_group(str(tmp_path / 'state.zarr'), mode='w')

        # Session 1 reviews a subset and closes on data_idx 3.
        win = self._make_window(qt_app, monkeypatch, nrows=6, data_idxs=[0, 1, 3, 4, 5],
                                start_idx=None, state_group=state_group)
        self._stub_refresh(win)
        assert win._data_idx == 0
        win._advance_resonator(+1)
        win._advance_resonator(+1)
        assert win._data_idx == 3
        self._close(win)
        assert state_group.attrs['sweep_fitter']['viewed_data_idxs'] == [0, 1, 3]

        # Session 2 over all rows starts at 2, the first row never viewed.
        win = self._make_window(qt_app, monkeypatch, nrows=6, start_idx=None,
                                state_group=state_group)
        assert win._data_idx == 2
        assert win._res_label.text() == '3 / 6'
        self._close(win)

    def test_resume_with_everything_viewed_or_explicit_start(self, qt_app, monkeypatch, tmp_path):
        state_group = zarr.open_group(str(tmp_path / 'state.zarr'), mode='w')
        state_group.attrs['sweep_fitter'] = {'viewed_data_idxs': [0, 1, 2], 'prefit_attempted': [1]}

        win = self._make_window(qt_app, monkeypatch, nrows=3, start_idx=None, state_group=state_group)
        assert win._data_idx == 0
        assert win._initialized_data_idxs == {1}
        self._close(win)

        win = self._make_window(qt_app, monkeypatch, nrows=3, start_idx=2, state_group=state_group)
        assert win._data_idx == 2
        self._close(win)

    def test_attempted_rows_are_saved(self, qt_app, monkeypatch, tmp_path):
        state_group = zarr.open_group(str(tmp_path / 'state.zarr'), mode='w')
        win = self._make_window(qt_app, monkeypatch, nrows=4, state_group=state_group)
        win._runner_outputs_exist = lambda _ar, _di: False
        win._initialize_runner_outputs = lambda _ar, _di: None

        win._initialize_remaining_data_indices([MagicMock()], [2, 3])

        assert state_group.attrs['sweep_fitter']['prefit_attempted'] == [2, 3]
        self._close(win)

    def test_close_waits_for_worker_and_consolidates(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=3)
        release = threading.Event()
        win._init_all_thread = threading.Thread(target=release.wait, daemon=True)
        win._init_all_thread.start()
        order = []
        win._consolidate_storage = lambda: order.append(win._init_all_thread.is_alive())
        threading.Timer(0.2, release.set).start()

        self._close(win)

        assert win._init_all_stop.is_set()
        assert order == [False]

    def test_consolidate_storage_runs_for_every_sweep(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=3)

        self._close(win)

        for ar in win._ARs:
            ar.DS.consolidate_storage.assert_called_once()

    def test_worker_skips_rows_saved_in_one_mask_read(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=4)
        worker_ars = [MagicMock(), MagicMock()]
        for ar, done_rows in zip(worker_ars, ([1, 2], [2])):
            mask = np.zeros(4, dtype=bool)
            mask[done_rows] = True
            ar.path = [{'task': _make_step('fit_iq', return_names=['z'])}]
            ar.DS.saved_row_mask.return_value = mask
        win._runner_outputs_exist = MagicMock(side_effect=AssertionError('should use the mask'))
        fitted = []
        win._initialize_runner_outputs = lambda ar, di: fitted.append((worker_ars.index(ar), di))

        win._initialize_remaining_data_indices(worker_ars, [1, 2, 3])

        assert fitted == [(1, 1), (0, 3), (1, 3)]
        self._close(win)

    def test_worker_ars_are_built_once(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch, nrows=3)
        win._worker_root = MagicMock()
        win._make_worker_ars = MagicMock(return_value=['ar'])
        win._initialize_remaining_data_indices = lambda _ars, _dis: None

        for _ in range(2):
            win._start_background_initialize_remaining()
            win._init_all_thread.join(timeout=2)

        win._make_worker_ars.assert_called_once()
        self._close(win)

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
            start_idx=0,
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

    def test_initialize_remaining_data_indices_runs_all_remaining_rows(self, qt_app, monkeypatch):
        step1 = _make_step('fit_gain')
        step2 = _make_step('fit_iq')
        ars = []
        for _ in range(2):
            ar = MagicMock()
            ar.analysis_steps = [step1, step2]
            ar.path = [{'task': step1}, {'task': step2}]
            ar._last_failures = {}
            ar.execute_step.return_value = None
            ar.execute_path.return_value = None
            ar.DS = MagicMock()
            ar.DS.nrows = 3
            ars.append(ar)

        monkeypatch.setattr(isweep, 'get_panel_class', lambda _names: _WindowPanel)
        monkeypatch.setattr(isweep.QtCore.QTimer, 'singleShot', lambda *_args, **_kwargs: None)
        win = SweepFitterWindow(
            ars,
            x_param_name='x',
            x_name='X',
            y_func=lambda _ar, _di: None,
            y_name='Y',
            start_sweep_idx=0,
            start_idx=0,
            title='test',
        )

        worker_ars = [MagicMock(), MagicMock()]
        for worker_ar in worker_ars:
            worker_ar.execute_path.return_value = None

        initialized = []
        win._runner_outputs_exist = lambda _ar, di: False
        win._initialize_runner_outputs = lambda ar, di: initialized.append((ar, di))

        win._initialize_remaining_data_indices(worker_ars, [1, 2])

        assert initialized == [
            (worker_ars[0], 1),
            (worker_ars[1], 1),
            (worker_ars[0], 2),
            (worker_ars[1], 2),
        ]
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_ensure_data_idx_initialized_runs_sync_when_row_not_ready(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._initialized_data_idxs = {0}
        win._all_sweeps_outputs_exist = lambda _di: False
        win._start_background_initialize_remaining = MagicMock()
        win._initialize_all_sweeps_for_data_idx = MagicMock()

        class _DeadThread:
            def is_alive(self):
                return False

        win._init_all_thread = _DeadThread()

        win._ensure_data_idx_initialized(1)

        win._initialize_all_sweeps_for_data_idx.assert_called_once_with(1)
        assert 1 in win._initialized_data_idxs
        win._start_background_initialize_remaining.assert_called_once()
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_ensure_data_idx_initialized_skips_ready_rows(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._initialized_data_idxs = {0, 1}
        win._initialize_all_sweeps_for_data_idx = MagicMock()

        win._ensure_data_idx_initialized(1)

        win._initialize_all_sweeps_for_data_idx.assert_not_called()
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win.close()

    def test_set_data_idx_does_not_trust_prefetch_when_outputs_missing(self, qt_app, monkeypatch):
        win = self._make_window(qt_app, monkeypatch)
        win._prefetched_idx = 1
        win._all_sweeps_outputs_exist = lambda di: False if di == 1 else True
        win._ensure_data_idx_initialized = MagicMock()
        win._update_sweep_scatter = MagicMock()
        win._update_waterfall = MagicMock()
        win._autoscale_all = MagicMock()

        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            win._set_data_idx(1)

        win._ensure_data_idx_initialized.assert_called_once_with(1)
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