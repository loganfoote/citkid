"""
Tests for StepPanel and DefaultStepPanel.

Uses Qt in offscreen mode. All tests require the ``qt_app`` fixture from
conftest.py to ensure a QApplication exists before widget instantiation.
"""

import pytest
from unittest.mock import MagicMock, patch
from pyqtgraph.Qt import QtWidgets

from citkid.pipeline.framework import plStep
import citkid.pipeline.interactive.core as icore
from citkid.pipeline.interactive.core import (
    StepPanel,
    DefaultStepPanel,
    InteractiveAnalysisWindow,
)


################################################################################
# Helpers
################################################################################

def _make_step(name, func_type='per-row'):
    return plStep(name, lambda x: x, ['x'], ['y'], func_type)


def _make_ar(*step_names, func_type='per-row'):
    """Return a mock AnalysisRunner with the given steps."""
    steps = [_make_step(n, func_type) for n in step_names]
    ar = MagicMock()
    ar.analysis_steps = steps
    ar._last_failures = {}
    return ar, steps


################################################################################
# StepPanel.__init__
################################################################################

class TestStepPanelInit:
    def test_raises_on_unknown_step(self, qt_app):
        ar, _ = _make_ar('good_step')
        with pytest.raises(ValueError, match="not found"):
            StepPanel(ar, ('bad_step',))

    def test_stores_ar_and_step_names(self, qt_app):
        ar, _ = _make_ar('step_a')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('step_a',))
        assert panel.AR is ar
        assert panel.step_names == ('step_a',)

    def test_coerces_step_names_to_tuple(self, qt_app):
        ar, _ = _make_ar('step_b')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ['step_b'])
        assert isinstance(panel.step_names, tuple)

    def test_stores_data_idx_and_scales(self, qt_app):
        ar, _ = _make_ar('step_c')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('step_c',), data_idx=5, ui_scale=1.5, plot_scale=2.0)
        assert panel.data_idx == 5
        assert panel.ui_scale == 1.5
        assert panel.plot_scale == 2.0

    def test_steps_list_populated(self, qt_app):
        ar, steps = _make_ar('step_d1', 'step_d2')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('step_d1', 'step_d2'))
        assert len(panel.steps) == 2
        assert panel.steps[0].name == 'step_d1'
        assert panel.steps[1].name == 'step_d2'

    def test_has_run_starts_false(self, qt_app):
        ar, _ = _make_ar('step_e')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('step_e',))
        assert not panel._has_run

    def test_panel_index_starts_none(self, qt_app):
        ar, _ = _make_ar('step_f')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('step_f',))
        assert panel.panel_index is None


################################################################################
# StepPanel.run_steps
################################################################################

class TestStepPanelRunSteps:
    def _panel(self, *step_names, func_type='per-row'):
        ar, _ = _make_ar(*step_names, func_type=func_type)
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, step_names)
        return panel, ar

    def test_returns_true_on_success(self, qt_app):
        panel, ar = self._panel('s1')
        ar.execute_step.return_value = None
        assert panel.run_steps() is True

    def test_sets_has_run_on_success(self, qt_app):
        panel, ar = self._panel('s2')
        ar.execute_step.return_value = None
        panel.run_steps()
        assert panel._has_run

    def test_sets_dirty_on_success(self, qt_app):
        panel, ar = self._panel('s3')
        ar.execute_step.return_value = None
        panel.run_steps()
        assert panel._dirty

    def test_clears_last_error_on_success(self, qt_app):
        panel, ar = self._panel('s4')
        panel._last_error = ValueError("stale error")
        ar.execute_step.return_value = None
        panel.run_steps()
        assert panel._last_error is None

    def test_returns_false_on_exception(self, qt_app):
        panel, ar = self._panel('s5')
        ar.execute_step.side_effect = RuntimeError("boom")
        assert panel.run_steps() is False

    def test_has_run_stays_false_on_exception(self, qt_app):
        panel, ar = self._panel('s6')
        ar.execute_step.side_effect = RuntimeError("boom")
        panel.run_steps()
        assert not panel._has_run

    def test_stores_last_error_on_exception(self, qt_app):
        panel, ar = self._panel('s7')
        err = RuntimeError("detail")
        ar.execute_step.side_effect = err
        panel.run_steps()
        assert panel._last_error is err

    def test_returns_false_on_failures_dict(self, qt_app):
        panel, ar = self._panel('s8')
        ar.execute_step.return_value = None
        ar._last_failures = {0: "traceback text"}
        assert panel.run_steps() is False

    def test_global_step_receives_none_data_idx(self, qt_app):
        panel, ar = self._panel('g1', func_type='global')
        panel.data_idx = 5
        ar.execute_step.return_value = None
        panel.run_steps()
        call_kwargs = ar.execute_step.call_args.kwargs
        assert call_kwargs['data_idx'] is None

    def test_per_row_step_receives_data_idx(self, qt_app):
        panel, ar = self._panel('pr1', func_type='per-row')
        panel.data_idx = 7
        ar.execute_step.return_value = None
        panel.run_steps()
        call_kwargs = ar.execute_step.call_args.kwargs
        assert call_kwargs['data_idx'] == 7

    def test_calls_execute_step_for_each_step(self, qt_app):
        panel, ar = self._panel('m1', 'm2')
        ar.execute_step.return_value = None
        panel.run_steps()
        assert ar.execute_step.call_count == 2

    def test_stops_at_first_failure(self, qt_app):
        panel, ar = self._panel('f1', 'f2')
        ar.execute_step.side_effect = RuntimeError("first fails")
        panel.run_steps()
        assert ar.execute_step.call_count == 1


################################################################################
# StepPanel.trigger_downstream
################################################################################

class TestTriggerDownstream:
    def test_emits_downstream_rerun(self, qt_app):
        ar, _ = _make_ar('sig_step')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('sig_step',))
        received = []
        panel.downstream_rerun.connect(received.append)
        panel.trigger_downstream()
        assert len(received) == 1
        assert received[0] is panel

    def test_multiple_emissions(self, qt_app):
        ar, _ = _make_ar('sig_step2')
        with patch.object(StepPanel, 'setup_ui'):
            panel = StepPanel(ar, ('sig_step2',))
        received = []
        panel.downstream_rerun.connect(received.append)
        panel.trigger_downstream()
        panel.trigger_downstream()
        assert len(received) == 2


################################################################################
# DefaultStepPanel
################################################################################

class TestDefaultStepPanel:
    def test_instantiation(self, qt_app):
        ar, _ = _make_ar('dsp1')
        panel = DefaultStepPanel(ar, ('dsp1',))
        assert isinstance(panel, StepPanel)

    def test_has_run_through_button(self, qt_app):
        ar, _ = _make_ar('dsp2')
        panel = DefaultStepPanel(ar, ('dsp2',))
        assert hasattr(panel, '_run_through_btn')

    def test_has_save_button(self, qt_app):
        ar, _ = _make_ar('dsp3')
        panel = DefaultStepPanel(ar, ('dsp3',))
        assert hasattr(panel, '_save_btn')

    def test_has_status_label(self, qt_app):
        ar, _ = _make_ar('dsp4')
        panel = DefaultStepPanel(ar, ('dsp4',))
        assert hasattr(panel, '_status_label')

    def test_initial_status_label_text(self, qt_app):
        ar, _ = _make_ar('dsp5')
        panel = DefaultStepPanel(ar, ('dsp5',))
        assert panel._status_label.text() == '—'

    def test_get_params_for_step_returns_empty_dict(self, qt_app):
        ar, steps = _make_ar('dsp6')
        panel = DefaultStepPanel(ar, ('dsp6',))
        result = panel.get_params_for_step(steps[0])
        assert result == {}


################################################################################
# InteractiveAnalysisWindow regressions
################################################################################

class _WindowTestPanel(StepPanel):
    """Minimal StepPanel for window-level behavior tests."""

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self._status_label = QtWidgets.QLabel("ok")
        layout.addWidget(self._status_label)
        self.update_calls = 0
        self.prefetch_calls = []

    def _outputs_exist(self):
        return True

    def update_plots(self):
        self.update_calls += 1

    def prefetch_plot_data(self, di):
        self.prefetch_calls.append(di)


def _make_window_ar():
    step = plStep('win_step', lambda x: x, ['x'], ['y'], 'per-row')
    ar = MagicMock()
    ar.analysis_steps = [step]
    ar.path = [{'task': step}]
    ar._last_failures = {}
    ar.DS = MagicMock()
    ar.DS.nrows = 2
    return ar


class TestInteractiveWindowPrefetchRegressions:
    def test_prefetched_render_does_not_mark_panel_dirty(self, qt_app, monkeypatch):
        """Rendering already-prefetched data should not create unsaved outputs."""
        ar = _make_window_ar()

        monkeypatch.setattr(icore, 'get_panel_class', lambda _names: _WindowTestPanel)
        monkeypatch.setattr(icore.QtCore.QTimer, 'singleShot', lambda *_args, **_kwargs: None)

        win = InteractiveAnalysisWindow(
            ar,
            panels=[('win_step',)],
            start_idx=0,
            data_idxs=[0, 1],
            title='test',
        )
        panel = win.panels[0]

        win._prefetched_idx = 1
        assert panel._dirty is False
        win._on_data_idx_changed(1)

        assert panel._has_run is True
        assert panel.update_calls == 1
        assert panel._dirty is False
        win.close()

    def test_prefetch_next_does_not_execute_pipeline_path(self, qt_app, monkeypatch):
        """Prefetch should only build panel plot caches and never run execute_path."""
        ar = _make_window_ar()

        monkeypatch.setattr(icore, 'get_panel_class', lambda _names: _WindowTestPanel)
        monkeypatch.setattr(icore.QtCore.QTimer, 'singleShot', lambda *_args, **_kwargs: None)

        win = InteractiveAnalysisWindow(
            ar,
            panels=[('win_step',)],
            start_idx=0,
            data_idxs=[0, 1],
            title='test',
        )
        panel = win.panels[0]

        win._prefetch_next()
        if win._prefetch_thread is not None:
            win._prefetch_thread.join(timeout=2)

        assert ar.execute_path.call_count == 0
        assert panel.prefetch_calls == [1]
        assert win._prefetched_idx == 1
        win.close()
