"""
Tests for validation helper functions.

Tests _validate_sweep_input and other validation functions.
"""

import pytest
import numpy as np

from citkid.crs.instrument import _validate_sweep_input


################################################################################
####################### _validate_sweep_input ##################################
################################################################################

def test_validate_sweep_input_valid_inputs():
    """Test _validate_sweep_input with all valid inputs."""
    module_idx = 1
    nco_freqs = {1: 4.0e9, 2: 4.5e9}  # Can contain other modules
    frequencies_map = {
        1: [[3.9e9, 3.95e9, 4.0e9], [4.0e9, 4.05e9, 4.1e9]],
        2: [[4.4e9, 4.45e9, 4.5e9]]  # Can contain other modules
    }
    ares_map = {1: [-50, -51], 2: [-52]}  # Can contain other modules
    sweep_f = {}
    sweep_z = {}
    nsamps = 10
    verbose = True
    pbar_description = "Testing"
    
    # Should not raise
    _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                         sweep_f, sweep_z, nsamps, verbose, pbar_description)


def test_validate_sweep_input_numpy_arrays():
    """Test _validate_sweep_input accepts numpy arrays."""
    module_idx = 1
    nco_freqs = {1: 4.0e9}
    frequencies_map = {1: np.array([[3.9e9, 4.0e9], [4.0e9, 4.1e9]])}
    ares_map = {1: np.array([-50, -51])}
    sweep_f = {}
    sweep_z = {}
    nsamps = 10
    verbose = True
    pbar_description = "Testing"
    
    # Should not raise
    _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                         sweep_f, sweep_z, nsamps, verbose, pbar_description)


################################################################################
################ nco_freqs validation ##########################################
################################################################################

def test_validate_sweep_input_nco_freqs_not_dict():
    """Test that non-dict nco_freqs raises TypeError."""
    with pytest.raises(TypeError, match = 'nco_freqs must be a dictionary'):
        _validate_sweep_input(1, [4.0e9], {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_nco_freqs_missing_module_idx():
    """Test that nco_freqs missing module_idx raises ValueError."""
    with pytest.raises(ValueError, match = 
        'nco_freqs does not contain module index 1'):
        _validate_sweep_input(1, {2: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_nco_freqs_value_not_float():
    """Test that non-float nco_freqs value raises TypeError."""
    with pytest.raises(TypeError, match = 'nco_freqs\\[1\\] must be a float'):
        _validate_sweep_input(
            1, {1: 4000000000, 2: 4.5e9},
            {1: [[3.9e9]]}, {1: [-50]},
            {}, {}, 10, True, "Test"
        )


def test_validate_sweep_input_nco_freqs_accepts_numpy_float():
    """Test that numpy float nco_freqs value is accepted."""
    module_idx = 1
    nco_freqs = {1: np.float64(4.0e9)}
    frequencies_map = {1: [[3.9e9, 4.0e9]]}
    ares_map = {1: [-50]}
    
    # Should not raise
    _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                         {}, {}, 10, True, "Test")


################################################################################
################ frequencies_map validation ####################################
################################################################################

def test_validate_sweep_input_frequencies_map_not_dict():
    """Test that non-dict frequencies_map raises TypeError."""
    with pytest.raises(
        TypeError, match = 'frequencies_map must be a dictionary'
        ):
        _validate_sweep_input(1, {1: 4.0e9}, [[3.9e9]], {1: [-50]},
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_frequencies_map_missing_module_idx():
    """Test that frequencies_map missing module_idx raises ValueError."""
    with pytest.raises(ValueError, match = 
        'frequencies_map does not contain module index 1'):
        _validate_sweep_input(1, {1: 4.0e9}, {2: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_frequencies_map_value_not_2d():
    """Test that 1D frequencies_map value raises TypeError."""
    with pytest.raises(TypeError, match = 
        'frequencies_map\\[1\\] must be 2D array-like of floats'):
        _validate_sweep_input(
            1, {1: 4.0e9},
            {1: [3.9e9, 4.0e9], 2: [[4.5e9]]}, {1: [-50]},
            {}, {}, 10, True, "Test"
        )


def test_validate_sweep_input_frequencies_map_value_3d():
    """Test that 3D frequencies_map value raises TypeError."""
    with pytest.raises(TypeError, match = 
        'frequencies_map\\[1\\] must be 2D array-like of floats'):
        _validate_sweep_input(1, {1: 4.0e9}, 
                             {1: [[[3.9e9, 4.0e9]]], 2: [[4.5e9]]}, {1: [-50]},
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_frequencies_map_value_not_numeric():
    """Test that non-numeric frequencies_map value raises TypeError."""
    with pytest.raises(TypeError, match = 
        'frequencies_map\\[1\\] must be 2D array-like of floats'):
        _validate_sweep_input(1, {1: 4.0e9}, 
                              {1: [['not', 'numbers']], 2: [[4.5e9]]}, 
                              {1: [-50]}, {}, {}, 10, True, "Test")


################################################################################
################ ares_map validation ###########################################
################################################################################

def test_validate_sweep_input_ares_map_not_dict():
    """Test that non-dict ares_map raises TypeError."""
    with pytest.raises(TypeError, match = 'ares_map must be a dictionary'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, [-50],
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_ares_map_missing_module_idx():
    """Test that ares_map missing module_idx raises ValueError."""
    with pytest.raises(ValueError, match = 
        'ares_map does not contain module index 1'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {2: [-50]},
                             {}, {}, 10, True, "Test")


def test_validate_sweep_input_ares_map_value_not_1d():
    """Test that 2D ares_map value raises TypeError."""
    with pytest.raises(TypeError, match = 
        'ares_map\\[1\\] must be 1D array-like of floats'):
        _validate_sweep_input(
            1, {1: 4.0e9}, {1: [[3.9e9]]}, 
            {1: [[-50, -51]], 2: [-52]}, {}, {}, 
            10, True, "Test"
        )


def test_validate_sweep_input_ares_map_value_not_numeric():
    """Test that non-numeric ares_map value raises TypeError."""
    with pytest.raises(TypeError, match = 
        'ares_map\\[1\\] must be 1D array-like of floats'):
        _validate_sweep_input(
            1, {1: 4.0e9}, {1: [[3.9e9]]}, 
            {1: ['not', 'numbers'], 2: [-52]}, {}, {}, 
            10, True, "Test"
        )


################################################################################
################ sweep_f and sweep_z validation ################################
################################################################################

def test_validate_sweep_input_sweep_f_not_dict():
    """Test that non-dict sweep_f raises TypeError."""
    with pytest.raises(TypeError, match = 'sweep_f must be a dictionary'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             [], {}, 10, True, "Test")


def test_validate_sweep_input_sweep_f_contains_module_idx():
    """Test that sweep_f containing module_idx raises ValueError."""
    with pytest.raises(
        ValueError, match = 'sweep_f already contains module index 1'
        ):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {1: [[3.9e9]]}, {}, 10, True, "Test")


def test_validate_sweep_input_sweep_z_not_dict():
    """Test that non-dict sweep_z raises TypeError."""
    with pytest.raises(TypeError, match = 'sweep_z must be a dictionary'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, [], 10, True, "Test")


def test_validate_sweep_input_sweep_z_contains_module_idx():
    """Test that sweep_z containing module_idx raises ValueError."""
    with pytest.raises(
        ValueError, match = 'sweep_z already contains module index 1'
        ):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {1: [[1+1j]]}, 10, True, "Test")


def test_validate_sweep_input_sweep_dicts_can_contain_other_modules():
    """Test that sweep_f and sweep_z can contain other module indices."""
    module_idx = 1
    nco_freqs = {1: 4.0e9, 2: 4.5e9}
    frequencies_map = {1: [[3.9e9, 4.0e9]], 2: [[4.4e9, 4.5e9]]}
    ares_map = {1: [-50], 2: [-51]}
    sweep_f = {2: [[4.4e9, 4.5e9]]}  # Contains module 2, not 1
    sweep_z = {2: [[1+1j, 2+2j]]}    # Contains module 2, not 1
    
    # Should not raise
    _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                         sweep_f, sweep_z, 10, True, "Test")


################################################################################
################ nsamps validation #############################################
################################################################################

def test_validate_sweep_input_nsamps_not_int():
    """Test that non-int nsamps raises TypeError."""
    with pytest.raises(TypeError, match = 'nsamps must be an integer'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10.5, True, "Test")


def test_validate_sweep_input_nsamps_zero():
    """Test that nsamps = 0 raises ValueError."""
    with pytest.raises(ValueError, match = 'nsamps must be greater than 0'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 0, True, "Test")


def test_validate_sweep_input_nsamps_negative():
    """Test that negative nsamps raises ValueError."""
    with pytest.raises(ValueError, match = 'nsamps must be greater than 0'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, -1, True, "Test")


def test_validate_sweep_input_nsamps_accepts_numpy_int():
    """Test that numpy int nsamps is accepted."""
    module_idx = 1
    nco_freqs = {1: 4.0e9}
    frequencies_map = {1: [[3.9e9, 4.0e9]]}
    ares_map = {1: [-50]}
    
    # Should not raise
    _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                         {}, {}, np.int32(10), True, "Test")


################################################################################
################ verbose validation ############################################
################################################################################

def test_validate_sweep_input_verbose_not_bool():
    """Test that non-bool verbose raises TypeError."""
    with pytest.raises(TypeError, match = 'verbose must be a boolean'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10, 1, "Test")


def test_validate_sweep_input_verbose_string():
    """Test that string verbose raises TypeError."""
    with pytest.raises(TypeError, match = 'verbose must be a boolean'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10, "True", "Test")


################################################################################
################ pbar_description validation ###################################
################################################################################

def test_validate_sweep_input_pbar_description_not_string():
    """Test that non-string pbar_description raises TypeError."""
    with pytest.raises(TypeError, match = 'pbar_description must be a string'):
        _validate_sweep_input(1, {1: 4.0e9}, {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, 10, True, 123)


def test_validate_sweep_input_pbar_description_empty_string():
    """Test that empty string pbar_description is accepted."""
    module_idx = 1
    nco_freqs = {1: 4.0e9}
    frequencies_map = {1: [[3.9e9, 4.0e9]]}
    ares_map = {1: [-50]}
    
    # Should not raise
    _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                         {}, {}, 10, True, "")


################################################################################
################ Multiple invalid inputs #######################################
################################################################################

def test_validate_sweep_input_catches_first_error():
    """Test that validation catches the first error encountered."""
    # Multiple errors: nco_freqs not dict, nsamps not int, etc.
    # Should raise error for nco_freqs first
    with pytest.raises(TypeError, match = 'nco_freqs must be a dictionary'):
        _validate_sweep_input(1, "not_dict", {1: [[3.9e9]]}, {1: [-50]},
                             {}, {}, "not_int", "not_bool", 123)
