"""
Tests for the panel registry in citkid.pipeline.interactive.core.

Pure registry logic — no Qt widget instantiation required.
"""

import pytest
from citkid.pipeline.interactive.core import (
    _PANEL_REGISTRY,
    register_panel,
    get_panel_class,
    DefaultStepPanel,
    StepPanel,
)


################################################################################
# Isolation fixture
################################################################################

@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot and restore _PANEL_REGISTRY around every test."""
    snapshot = dict(_PANEL_REGISTRY)
    yield
    _PANEL_REGISTRY.clear()
    _PANEL_REGISTRY.update(snapshot)


################################################################################
# register_panel
################################################################################

class TestRegisterPanel:
    def test_single_step_name(self):
        @register_panel('_test_reg_a')
        class PanelA(StepPanel):
            pass

        assert ('_test_reg_a',) in _PANEL_REGISTRY
        assert _PANEL_REGISTRY[('_test_reg_a',)] is PanelA

    def test_multiple_step_names(self):
        @register_panel('_test_reg_b', '_test_reg_c')
        class PanelBC(StepPanel):
            pass

        assert ('_test_reg_b', '_test_reg_c') in _PANEL_REGISTRY
        assert _PANEL_REGISTRY[('_test_reg_b', '_test_reg_c')] is PanelBC

    def test_decorator_returns_class_unchanged(self):
        @register_panel('_test_reg_d')
        class PanelD(StepPanel):
            def custom_method(self):
                return 42

        assert PanelD.custom_method is not None
        assert issubclass(PanelD, StepPanel)

    def test_overwrites_existing_registration(self):
        @register_panel('_test_reg_e')
        class PanelE1(StepPanel):
            pass

        @register_panel('_test_reg_e')
        class PanelE2(StepPanel):
            pass

        assert _PANEL_REGISTRY[('_test_reg_e',)] is PanelE2


################################################################################
# get_panel_class
################################################################################

class TestGetPanelClass:
    def test_exact_tuple_match(self):
        @register_panel('_get_f', '_get_g')
        class PanelFG(StepPanel):
            pass

        result = get_panel_class(('_get_f', '_get_g'))
        assert result is PanelFG

    def test_single_step_fallback(self):
        """When only (name0,) is registered, a multi-step query falls back to it."""
        @register_panel('_get_h')
        class PanelH(StepPanel):
            pass

        result = get_panel_class(('_get_h', '_unregistered_suffix'))
        assert result is PanelH

    def test_default_fallback_when_unregistered(self):
        result = get_panel_class(('_totally_unknown_step_xyz',))
        assert result is DefaultStepPanel

    def test_exact_match_wins_over_single_fallback(self):
        @register_panel('_get_i')
        class PanelISingle(StepPanel):
            pass

        @register_panel('_get_i', '_get_j')
        class PanelIExact(StepPanel):
            pass

        result = get_panel_class(('_get_i', '_get_j'))
        assert result is PanelIExact

    def test_single_step_exact_is_not_a_fallback(self):
        """A single-element query should use exact match, not fallback path."""
        @register_panel('_get_k')
        class PanelK(StepPanel):
            pass

        result = get_panel_class(('_get_k',))
        assert result is PanelK

    def test_default_step_panel_is_step_panel_subclass(self):
        assert issubclass(DefaultStepPanel, StepPanel)
