"""
Tests for citkid.pipeline.default_steps.

Checks the structure of default_cal_steps and default_analysis_steps:
names, param_names, return_names, and func_types. Does not execute any
functions.
"""

import pytest
from citkid.pipeline.default_steps import default_cal_steps, default_analysis_steps
from citkid.pipeline.framework import plStep


################################################################################
# Helpers
################################################################################

def _step_map(steps):
    return {s.name: s for s in steps}


################################################################################
# Basic structure
################################################################################

def test_default_cal_steps_are_plsteps():
    for s in default_cal_steps:
        assert isinstance(s, plStep)


def test_default_analysis_steps_are_plsteps():
    for s in default_analysis_steps:
        assert isinstance(s, plStep)


def test_default_cal_steps_names_unique():
    names = [s.name for s in default_cal_steps]
    assert len(names) == len(set(names))


def test_default_analysis_steps_names_unique():
    names = [s.name for s in default_analysis_steps]
    assert len(names) == len(set(names))


def test_all_steps_have_callable_func():
    for s in default_cal_steps + default_analysis_steps:
        assert callable(s.func), f"{s.name}.func is not callable"


def test_all_steps_have_nonempty_param_names():
    for s in default_cal_steps + default_analysis_steps:
        assert len(s.param_names) > 0, f"{s.name} has no param_names"


def test_all_steps_have_nonempty_return_names():
    for s in default_cal_steps + default_analysis_steps:
        assert len(s.return_names) > 0, f"{s.name} has no return_names"


def test_all_param_names_are_strings():
    for s in default_cal_steps + default_analysis_steps:
        for p in s.param_names:
            assert isinstance(p, str), f"{s.name}: param_name {p!r} is not a str"


def test_all_return_names_are_strings():
    for s in default_cal_steps + default_analysis_steps:
        for r in s.return_names:
            assert isinstance(r, str), f"{s.name}: return_name {r!r} is not a str"


################################################################################
# func_type
################################################################################

_VALID_FUNC_TYPES = {"per-row", "vectorized", "global", "global-res"}

def test_all_cal_steps_func_type_valid():
    for s in default_cal_steps:
        assert s.func_type in _VALID_FUNC_TYPES, \
            f"{s.name} has invalid func_type {s.func_type!r}"


def test_all_analysis_steps_func_type_valid():
    for s in default_analysis_steps:
        assert s.func_type in _VALID_FUNC_TYPES, \
            f"{s.name} has invalid func_type {s.func_type!r}"


def test_all_cal_steps_are_per_row():
    """All cal steps are expected to be per-row."""
    for s in default_cal_steps:
        assert s.func_type == "per-row", \
            f"{s.name} is {s.func_type!r}, expected 'per-row'"


################################################################################
# Spot-checks: specific step names and their signatures
################################################################################

class TestCalStepSpotChecks:
    def setup_method(self):
        self.m = _step_map(default_cal_steps)

    def test_rmv_gain_f_params(self):
        s = self.m['rmv_gain_f']
        assert 'ff' in s.param_names
        assert 'zf' in s.param_names
        assert 'zf_rmv' in s.return_names

    def test_rmv_gain_t_params(self):
        s = self.m['rmv_gain_t']
        assert 'ft' in s.param_names
        assert 'zt' in s.param_names
        assert 'zt_rmv' in s.return_names

    def test_center_f_params(self):
        s = self.m['center_f']
        assert 'zf_rmv' in s.param_names
        assert 'circ_origin' in s.param_names
        assert 'zf_cent' in s.return_names

    def test_get_thetaf_params(self):
        s = self.m['get_thetaf']
        assert 'zf_cent' in s.param_names
        assert 'idx_t' in s.param_names
        assert 'thetaf' in s.return_names

    def test_get_thetat_params(self):
        s = self.m['get_thetat']
        assert 'zt_cent' in s.param_names
        assert 'thetat' in s.return_names

    def test_get_xf_formula(self):
        """get_xf lambda computes 1 - ff/ft."""
        s = self.m['get_xf']
        import numpy as np
        ff = np.array([1.0, 2.0])
        ft = 2.0
        result = s.func(ff, ft)
        expected = 1 - ff / ft
        np.testing.assert_allclose(result, expected)

    def test_get_sparper_params(self):
        s = self.m['get_sparper']
        assert 'thetat' in s.param_names
        assert 'At' in s.param_names
        assert 'circ_radius' in s.param_names
        assert 'dt' in s.param_names
        assert 'spar' in s.return_names
        assert 'sper' in s.return_names


class TestAnalysisStepSpotChecks:
    def setup_method(self):
        self.m = _step_map(default_analysis_steps)

    def test_make_fr_spans_is_global(self):
        s = self.m['make_fr_spans']
        assert s.func_type == 'global'
        assert 'fres_all' in s.param_names
        assert 'qres_all' in s.param_names
        assert 'fr_spans' in s.return_names

    def test_fit_gain_params(self):
        s = self.m['fit_gain']
        assert 'fg' in s.param_names
        assert 'zg' in s.param_names
        assert 'fr_spans' in s.param_names
        assert 'span_mult' in s.param_names
        assert 'p_amp' in s.return_names
        assert 'p_phase' in s.return_names

    def test_fit_iq_circle_params(self):
        s = self.m['fit_iq_circle']
        assert 'zf_rmv' in s.param_names
        assert 'circ_mask' in s.param_names
        assert 'circ_origin' in s.return_names
        assert 'circ_radius' in s.return_names

    def test_get_idx_t_formula(self):
        """get_idx_t finds index of closest frequency."""
        s = self.m['get_idx_t']
        import numpy as np
        ff = np.array([1.0, 2.0, 3.0, 4.0])
        ft = 2.9
        result = s.func(ff, ft)
        assert result == 2  # index of 3.0

    def test_get_xcal_mask_params(self):
        s = self.m['get_xcal_mask']
        assert 'ff' in s.param_names
        assert 'thetaf' in s.param_names
        assert 'thetat' in s.param_names
        assert 'xcal_mask' in s.return_names

    def test_fit_x_theta_params(self):
        s = self.m['fit_x_theta']
        assert 'thetaf' in s.param_names
        assert 'xf' in s.param_names
        assert 'xcal_mask' in s.param_names
        assert 'poly_x' in s.return_names

    def test_fit_iq_params(self):
        s = self.m['fit_iq']
        assert 'ff' in s.param_names
        assert 'zf_rmv' in s.param_names
        assert 'iq_mask' in s.param_names
        assert 'iq_popt' in s.return_names
        assert 'iq_nrmse' in s.return_names
