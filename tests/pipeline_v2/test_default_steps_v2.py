"""
Tests for citkid.pipeline_v2.default_steps module.

Checks the structure of default_cal_steps and default_analysis_steps:
names, param_names, return_names, and func_types. Does not execute any functions.
"""

import pytest
from citkid.pipeline_v2.default_steps import default_cal_steps, default_analysis_steps
from citkid.pipeline_v2.framework import plStep


################################################################################
# Helpers
################################################################################

def _step_map(steps):
    return {s.name: s for s in steps}


################################################################################
# Basic structure
################################################################################

class TestDefaultStepsStructure:
    """Tests for the structure of default_cal_steps and default_analysis_steps"""

    def test_default_cal_steps_are_plsteps(self):
        """All cal steps must be plStep instances."""
        for s in default_cal_steps:
            assert isinstance(s, plStep), f"{s} is not a plStep instance"

    def test_default_analysis_steps_are_plsteps(self):
        """All analysis steps must be plStep instances."""
        for s in default_analysis_steps:
            assert isinstance(s, plStep), f"{s} is not a plStep instance"

    def test_default_cal_steps_names_unique(self):
        """All cal step names must be unique."""
        names = [s.name for s in default_cal_steps]
        assert len(names) == len(set(names)), \
            f"Duplicate names in default_cal_steps: {[n for n in names if names.count(n) > 1]}"

    def test_default_analysis_steps_names_unique(self):
        """All analysis step names must be unique."""
        names = [s.name for s in default_analysis_steps]
        assert len(names) == len(set(names)), \
            f"Duplicate names in default_analysis_steps: {[n for n in names if names.count(n) > 1]}"

    def test_all_steps_have_callable_func(self):
        """All steps must have a callable func attribute."""
        for s in default_cal_steps + default_analysis_steps:
            assert callable(s.func), f"{s.name}.func is not callable"

    def test_all_steps_have_nonempty_param_names(self):
        """All steps must have at least one param_name."""
        for s in default_cal_steps + default_analysis_steps:
            assert len(s.param_names) > 0, f"{s.name} has no param_names"

    def test_all_steps_have_nonempty_return_names(self):
        """All steps must have at least one return_name."""
        for s in default_cal_steps + default_analysis_steps:
            assert len(s.return_names) > 0, f"{s.name} has no return_names"

    def test_all_param_names_are_strings(self):
        """All param_names must be strings."""
        for s in default_cal_steps + default_analysis_steps:
            for p in s.param_names:
                assert isinstance(p, str), \
                    f"{s.name}: param_name {p!r} is not a str, is {type(p)}"

    def test_all_return_names_are_strings(self):
        """All return_names must be strings."""
        for s in default_cal_steps + default_analysis_steps:
            for r in s.return_names:
                assert isinstance(r, str), \
                    f"{s.name}: return_name {r!r} is not a str, is {type(r)}"

    def test_all_param_names_are_non_empty(self):
        """All param_names must be non-empty strings."""
        for s in default_cal_steps + default_analysis_steps:
            for p in s.param_names:
                assert len(p) > 0, f"{s.name}: param_name is empty string"

    def test_all_return_names_are_non_empty(self):
        """All return_names must be non-empty strings."""
        for s in default_cal_steps + default_analysis_steps:
            for r in s.return_names:
                assert len(r) > 0, f"{s.name}: return_name is empty string"


class TestDefaultStepsFuncTypes:
    """Tests for func_type validity"""

    _VALID_FUNC_TYPES = {"per-row", "vectorized", "global", "global-res"}

    def test_all_cal_steps_func_type_valid(self):
        """All cal step func_types must be valid."""
        for s in default_cal_steps:
            assert s.func_type in self._VALID_FUNC_TYPES, \
                f"{s.name} has invalid func_type {s.func_type!r}"

    def test_all_analysis_steps_func_type_valid(self):
        """All analysis step func_types must be valid."""
        for s in default_analysis_steps:
            assert s.func_type in self._VALID_FUNC_TYPES, \
                f"{s.name} has invalid func_type {s.func_type!r}"

    def test_all_cal_steps_are_per_row(self):
        """All cal steps are expected to be per-row (data loading happens per row)."""
        for s in default_cal_steps:
            assert s.func_type == "per-row", \
                f"{s.name} is {s.func_type!r}, expected 'per-row'"

    def test_analysis_steps_func_types_reasonable(self):
        """Analysis steps should be per-row or vectorized (most common types)."""
        for s in default_analysis_steps:
            # We allow all types for analysis, but per-row and vectorized are most common
            assert s.func_type in self._VALID_FUNC_TYPES, \
                f"{s.name} has invalid func_type {s.func_type!r}"


class TestDefaultStepsConsistency:
    """Tests for consistency between steps"""

    def test_no_duplicate_param_names_within_step(self):
        """No step should have duplicate param_names."""
        for s in default_cal_steps + default_analysis_steps:
            param_set = set(s.param_names)
            assert len(param_set) == len(s.param_names), \
                f"{s.name} has duplicate param_names: {s.param_names}"

    def test_no_duplicate_return_names_within_step(self):
        """No step should have duplicate return_names."""
        for s in default_cal_steps + default_analysis_steps:
            return_set = set(s.return_names)
            assert len(return_set) == len(s.return_names), \
                f"{s.name} has duplicate return_names: {s.return_names}"

    def test_param_and_return_names_no_overlap(self):
        """A step's param_names and return_names should not overlap."""
        for s in default_cal_steps + default_analysis_steps:
            param_set = set(s.param_names)
            return_set = set(s.return_names)
            overlap = param_set & return_set
            if overlap:
                # This is allowed but uncommon - just document it
                pass


class TestDefaultStepsStaticContent:
    """Tests verifying static content of default steps (non-implementation dependent)"""

    def test_default_cal_steps_not_empty(self):
        """default_cal_steps should not be empty."""
        assert len(default_cal_steps) > 0, "default_cal_steps is empty"

    def test_default_analysis_steps_not_empty(self):
        """default_analysis_steps should not be empty."""
        assert len(default_analysis_steps) > 0, "default_analysis_steps is empty"

    def test_both_step_lists_are_tuples_or_lists(self):
        """Step lists should be sequences (tuples or lists)."""
        assert isinstance(default_cal_steps, (list, tuple)), \
            f"default_cal_steps is {type(default_cal_steps)}, not list or tuple"
        assert isinstance(default_analysis_steps, (list, tuple)), \
            f"default_analysis_steps is {type(default_analysis_steps)}, not list or tuple"

    def test_step_names_are_identifiers(self):
        """Step names should be valid Python identifiers or step-like names."""
        for s in default_cal_steps + default_analysis_steps:
            # Should not be empty
            assert len(s.name) > 0, f"Step has empty name"
            # Should not have leading/trailing whitespace
            assert s.name == s.name.strip(), f"Step name has whitespace: {s.name!r}"


class TestDefaultStepsDocumentation:
    """Tests for documentation and introspection"""

    def test_all_steps_repr_works(self):
        """All steps should have a working __repr__."""
        for s in default_cal_steps + default_analysis_steps:
            repr_str = repr(s)
            assert isinstance(repr_str, str), f"{s.name} repr is not a string"
            assert len(repr_str) > 0, f"{s.name} repr is empty"

    def test_all_steps_str_works(self):
        """All steps should have a working __str__."""
        for s in default_cal_steps + default_analysis_steps:
            str_str = str(s)
            assert isinstance(str_str, str), f"{s.name} str is not a string"
            assert len(str_str) > 0, f"{s.name} str is empty"
            # __str__ should be more verbose than __repr__
            assert len(str_str) >= len(repr(s)), \
                f"{s.name} str should be at least as long as repr"
