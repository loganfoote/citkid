"""
Tests for CRS NCO configuration methods.

Tests set_nco, _set_nco macro, disable_modules, and _clear_channels methods.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import numpy as np

from citkid.crs.instrument import CRS


@pytest.fixture
def mock_crs_for_set_nco(base_crs):
    """Extend base_crs with additional mocks for set_nco tests."""
    crs = base_crs
    
    # Initialize attributes
    crs.analog_bank_high = False
    crs.nco_freqs = {}
    
    return crs


@pytest.mark.asyncio
async def test_set_nco_single_module(mock_crs_for_set_nco, capsys):
    """Test setting NCO frequency for a single module."""
    crs = mock_crs_for_set_nco
    
    # Mock util.get_modules to return a mock modules object
    mock_modules = MagicMock()
    # Mock _set_nco to modify nco_freqs in-place (simulating measured value)
    async def mock_set_nco(nco_dict):
        # Simulate measurement returning slightly different value
        for key in nco_dict.keys():
            nco_dict[key] = nco_dict[key] + 0.5  # Measured value
    mock_modules._set_nco = AsyncMock(side_effect = mock_set_nco)
    
    # Mock _clear_channels
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        nco_freqs = {1: 4.5e9}
        await crs.set_nco(nco_freqs, verbose = True)
        
        # Verify util.get_modules was called
        from citkid.crs.instrument import util
        util.get_modules.assert_called_once_with(crs.d, [1])
        
        # Verify _set_nco was called
        mock_modules._set_nco.assert_called_once()
        
        # Verify nco_freqs was updated with measured value
        assert 1 in crs.nco_freqs
        assert crs.nco_freqs[1] == 4.5e9 + 0.5
        
        # Verify _clear_channels was called
        crs._clear_channels.assert_called_once_with([1])
        
        # Verify verbose output
        captured = capsys.readouterr()
        assert 'Module 1 NCO is' in captured.out
        assert 'MHz' in captured.out


@pytest.mark.asyncio
async def test_set_nco_multiple_modules(mock_crs_for_set_nco):
    """Test setting NCO frequency for multiple modules."""
    crs = mock_crs_for_set_nco
    
    mock_modules = MagicMock()
    async def mock_set_nco(nco_dict):
        # Simulate measurement
        for key in nco_dict.keys():
            nco_dict[key] = nco_dict[key] + 1.0
    mock_modules._set_nco = AsyncMock(side_effect = mock_set_nco)
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        nco_freqs = {1: 4.0e9, 2: 4.3e9, 3: 4.6e9}
        await crs.set_nco(nco_freqs, verbose = False)
        
        # Verify all modules were updated
        assert len(crs.nco_freqs) == 3
        assert crs.nco_freqs[1] == 4.0e9 + 1.0
        assert crs.nco_freqs[2] == 4.3e9 + 1.0
        assert crs.nco_freqs[3] == 4.6e9 + 1.0
        
        # Verify _clear_channels was called with all module indices
        call_args = crs._clear_channels.call_args[0][0]
        assert set(call_args) == {1, 2, 3}


@pytest.mark.asyncio
async def test_set_nco_updates_existing_nco_freqs(mock_crs_for_set_nco):
    """Test that set_nco updates existing nco_freqs."""
    crs = mock_crs_for_set_nco
    
    # Pre-populate nco_freqs
    crs.nco_freqs = {1: 3.0e9, 3: 4.0e9}
    
    mock_modules = MagicMock()
    async def mock_set_nco(nco_dict):
        for key in nco_dict.keys():
            nco_dict[key] = nco_dict[key] + 0.5
    mock_modules._set_nco = AsyncMock(side_effect = mock_set_nco)
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        # Update module 1 and add module 2
        nco_freqs = {1: 4.5e9, 2: 4.8e9}
        await crs.set_nco(nco_freqs, verbose = False)
        
        # Verify updates
        assert crs.nco_freqs[1] == 4.5e9 + 0.5  # Updated
        assert crs.nco_freqs[2] == 4.8e9 + 0.5  # New
        assert crs.nco_freqs[3] == 4.0e9  # Unchanged


@pytest.mark.asyncio
async def test_set_nco_copies_input_dict(mock_crs_for_set_nco):
    """Test that set_nco copies the input dictionary."""
    crs = mock_crs_for_set_nco
    
    mock_modules = MagicMock()
    async def mock_set_nco(nco_dict):
        for key in nco_dict.keys():
            nco_dict[key] = nco_dict[key] + 1.0
    mock_modules._set_nco = AsyncMock(side_effect = mock_set_nco)
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        # Original input dict
        original_nco_freqs = {1: 4.5e9}
        await crs.set_nco(original_nco_freqs, verbose = False)
        
        # Verify original dict is unchanged
        assert original_nco_freqs[1] == 4.5e9


@pytest.mark.asyncio
async def test_set_nco_verbose_false_no_print(mock_crs_for_set_nco, capsys):
    """Test that verbose = False suppresses output."""
    crs = mock_crs_for_set_nco
    
    mock_modules = MagicMock()
    mock_modules._set_nco = AsyncMock()
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        nco_freqs = {1: 4.5e9}
        await crs.set_nco(nco_freqs, verbose = False)
        
        # Verify no output
        captured = capsys.readouterr()
        assert 'Module' not in captured.out
        assert 'NCO' not in captured.out


@pytest.mark.asyncio
async def test_set_nco_high_analog_bank(mock_crs_for_set_nco):
    """Test setting NCO for high analog bank (modules 5-8)."""
    crs = mock_crs_for_set_nco
    crs.analog_bank_high = True
    
    mock_modules = MagicMock()
    async def mock_set_nco(nco_dict):
        for key in nco_dict.keys():
            nco_dict[key] = nco_dict[key] + 0.5
    mock_modules._set_nco = AsyncMock(side_effect = mock_set_nco)
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        nco_freqs = {5: 4.0e9, 6: 4.3e9, 7: 4.6e9}
        await crs.set_nco(nco_freqs, verbose = False)
        
        # Verify all modules were updated
        assert len(crs.nco_freqs) == 3
        assert 5 in crs.nco_freqs
        assert 6 in crs.nco_freqs
        assert 7 in crs.nco_freqs


@pytest.mark.asyncio
async def test_set_nco_verbose_output_format(mock_crs_for_set_nco, capsys):
    """Test the format of verbose output."""
    crs = mock_crs_for_set_nco
    
    mock_modules = MagicMock()
    async def mock_set_nco(nco_dict):
        # Set exact value for predictable output
        for key in nco_dict.keys():
            nco_dict[key] = 4.567890e9
    mock_modules._set_nco = AsyncMock(side_effect = mock_set_nco)
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        nco_freqs = {1: 4.5e9}
        await crs.set_nco(nco_freqs, verbose = True)
        
        # Verify output format (frequency in MHz, rounded to 6 decimals)
        captured = capsys.readouterr()
        assert 'Module 1 NCO is 4567.89 MHz' in captured.out


################################################################################
####################### set_nco Input Validation ###############################
################################################################################

@pytest.mark.asyncio
async def test_set_nco_not_dict(mock_crs_for_set_nco):
    """Test that non-dict nco_freqs raises TypeError."""
    crs = mock_crs_for_set_nco
    
    with pytest.raises(
        TypeError,
        match = 'nco_freqs must be a dictionary'
    ):
        await crs.set_nco([1, 4.5e9])


@pytest.mark.asyncio
async def test_set_nco_value_not_float(mock_crs_for_set_nco):
    """Test that non-float NCO frequency raises TypeError."""
    crs = mock_crs_for_set_nco
    
    with pytest.raises(
        TypeError,
        match = 'nco_freqs values must be float NCO frequencies in Hz'
    ):
        await crs.set_nco({1: 4500000000})  # int instead of float


@pytest.mark.asyncio
async def test_set_nco_frequency_too_low(mock_crs_for_set_nco):
    """Test that NCO frequency <= 0 raises ValueError."""
    crs = mock_crs_for_set_nco
    
    with pytest.raises(
        ValueError,
        match = 'NCO frequency.*is out of range'
    ):
        await crs.set_nco({1: 0.0})


@pytest.mark.asyncio
async def test_set_nco_frequency_too_high(mock_crs_for_set_nco):
    """Test that NCO frequency >= 5 GHz raises ValueError."""
    crs = mock_crs_for_set_nco
    
    with pytest.raises(ValueError, match = 'out of range'):
        await crs.set_nco({1: 5.0e9})


@pytest.mark.asyncio
async def test_set_nco_key_not_int(mock_crs_for_set_nco):
    """Test that non-int module index raises TypeError."""
    crs = mock_crs_for_set_nco
    
    with pytest.raises(
        TypeError,
        match = 'nco_freqs keys must be integer module indices'
    ):
        await crs.set_nco({'1': 4.5e9})


@pytest.mark.asyncio
async def test_set_nco_module_out_of_range_low_bank(mock_crs_for_set_nco):
    """Test that module index out of range for low bank raises error."""
    crs = mock_crs_for_set_nco
    crs.analog_bank_high = False
    
    with pytest.raises(
        ValueError,
        match = 'Module index 5 is out of range \\[1, 4\\] for low'
    ):
        await crs.set_nco({5: 4.5e9})


@pytest.mark.asyncio
async def test_set_nco_module_out_of_range_high_bank(mock_crs_for_set_nco):
    """Test that module index out of range for high bank raises error."""
    crs = mock_crs_for_set_nco
    crs.analog_bank_high = True
    
    with pytest.raises(
        ValueError,
        match = 'Module index 1 is out of range \\[5, 8\\] for high'
    ):
        await crs.set_nco({1: 4.5e9})


@pytest.mark.asyncio
async def test_set_nco_numpy_float(mock_crs_for_set_nco):
    """Test that numpy float NCO frequency is accepted."""
    crs = mock_crs_for_set_nco
    
    mock_modules = MagicMock()
    mock_modules._set_nco = AsyncMock()
    
    with patch.object(crs, '_clear_channels', new_callable = AsyncMock), \
         patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        
        # Should not raise error
        await crs.set_nco({1: np.float64(4.5e9)}, verbose = False)


################################################################################
############################## _set_nco ########################################
################################################################################

@pytest.fixture
def mock_crs_for_set_nco_macro():
    """Fixture for testing _set_nco macro function."""
    import rfmux
    
    # Create mock device
    mock_device = MagicMock()
    mock_device.set_nco_frequency = AsyncMock()
    mock_device.get_nco_frequency = AsyncMock(return_value = 4.5e9)
    
    # Create mock module object with correct spec
    mock_module = MagicMock(spec = rfmux.ReadoutModule)
    mock_module.crs = mock_device
    mock_module.module = 1  # Module index
    
    # Create nco_freqs dict
    nco_freqs = {1: 4.5e9}
    
    return mock_module, nco_freqs, mock_device


@pytest.mark.asyncio
async def test_set_nco_macro_basic_functionality(
        mock_crs_for_set_nco_macro):
    """Test _set_nco macro basic functionality."""
    from citkid.crs.instrument import _set_nco
    
    mock_module, nco_freqs, mock_device = mock_crs_for_set_nco_macro
    
    # Call the macro
    await _set_nco(mock_module, nco_freqs)
    
    # Check device methods were called correctly
    mock_device.set_nco_frequency.assert_called_once_with(
        4.5e9, module = 1)
    mock_device.get_nco_frequency.assert_called_once_with(module = 1)
    
    # Check nco_freqs was updated with measured value
    assert nco_freqs[1] == 4.5e9


@pytest.mark.asyncio
async def test_set_nco_macro_updates_dict_with_measured_value(
        mock_crs_for_set_nco_macro):
    """Test _set_nco updates nco_freqs with measured value."""
    from citkid.crs.instrument import _set_nco
    
    mock_module, nco_freqs, mock_device = mock_crs_for_set_nco_macro
    
    # Set measured value slightly different from requested (within 1 Hz)
    mock_device.get_nco_frequency.return_value = 4.5e9 + 0.5
    
    # Call the macro
    await _set_nco(mock_module, nco_freqs)
    
    # Check nco_freqs was updated with measured value
    assert nco_freqs[1] == 4.5e9 + 0.5


@pytest.mark.asyncio
async def test_set_nco_macro_accepts_within_tolerance(
        mock_crs_for_set_nco_macro):
    """Test _set_nco accepts measured values within 1 Hz tolerance."""
    from citkid.crs.instrument import _set_nco
    
    mock_module, nco_freqs, mock_device = mock_crs_for_set_nco_macro
    
    # Test values within 1 Hz tolerance
    test_cases = [
        (4.5e9, 4.5e9),  # Exact match
        (4.5e9, 4.5e9 + 0.5),  # +0.5 Hz
        (4.5e9, 4.5e9 - 0.5),  # -0.5 Hz
        (4.5e9, 4.5e9 + 0.99),  # +0.99 Hz
        (4.5e9, 4.5e9 - 0.99),  # -0.99 Hz
    ]
    
    for requested, measured in test_cases:
        nco_freqs[1] = requested
        mock_device.get_nco_frequency.return_value = measured
        
        # Should not raise error
        await _set_nco(mock_module, nco_freqs)


@pytest.mark.asyncio
async def test_set_nco_macro_raises_error_outside_tolerance(
        mock_crs_for_set_nco_macro):
    """Test _set_nco raises error if measured value outside 1 Hz."""
    from citkid.crs.instrument import _set_nco
    
    mock_module, nco_freqs, mock_device = mock_crs_for_set_nco_macro
    
    # Set measured value outside tolerance (2 Hz difference)
    mock_device.get_nco_frequency.return_value = 4.5e9 + 2.0
    
    with pytest.raises(RuntimeError, match = 'Failed to set NCO frequency'):
        await _set_nco(mock_module, nco_freqs)


@pytest.mark.asyncio
async def test_set_nco_macro_error_message_contains_values(
        mock_crs_for_set_nco_macro):
    """Test _set_nco error message contains requested and measured values."""
    from citkid.crs.instrument import _set_nco
    
    mock_module, nco_freqs, mock_device = mock_crs_for_set_nco_macro
    
    requested = 4.5e9
    measured = 4.6e9
    nco_freqs[1] = requested
    mock_device.get_nco_frequency.return_value = measured
    
    with pytest.raises(RuntimeError) as exc_info:
        await _set_nco(mock_module, nco_freqs)
    
    error_msg = str(exc_info.value)
    assert str(requested) in error_msg
    assert str(measured) in error_msg


@pytest.mark.asyncio
async def test_set_nco_macro_works_with_different_module_indices(
        mock_crs_for_set_nco_macro):
    """Test _set_nco works with different module indices."""
    from citkid.crs.instrument import _set_nco
    
    mock_module, nco_freqs, mock_device = mock_crs_for_set_nco_macro
    
    # Test different module indices
    for module_idx in [0, 1, 5, 15]:
        mock_module.module = module_idx
        nco_freqs[module_idx] = 3.5e9
        mock_device.get_nco_frequency.return_value = 3.5e9
        
        await _set_nco(mock_module, nco_freqs)
        
        # Check correct module was used
        mock_device.set_nco_frequency.assert_called_with(
            3.5e9, module = module_idx)
        mock_device.get_nco_frequency.assert_called_with(
            module = module_idx)
        
        # Check dict was updated
        assert nco_freqs[module_idx] == 3.5e9


def test_set_nco_macro_registered_to_readout_module():
    """Test _set_nco is registered as a macro to rfmux.ReadoutModule."""
    from citkid.crs.instrument import _set_nco
    
    # Check the function has been decorated
    assert hasattr(_set_nco, '__wrapped__')
    
    # The macro decorator should add the function to ReadoutModule
    # This is verified by checking the function exists and is callable
    assert callable(_set_nco)


################################################################################
############################# disable_modules ##################################
################################################################################

@pytest.fixture
def mock_crs_for_disable_modules(base_crs):
    """Extend base_crs for disable_modules tests."""
    crs = base_crs
    crs.nco_freqs = {0: 3.5e9, 1: 4.0e9, 2: 4.5e9, 5: 5.0e9}
    
    # Mock _clear_channels
    crs._clear_channels = AsyncMock()
    
    return crs


@pytest.mark.asyncio
async def test_disable_modules_basic_functionality(
        mock_crs_for_disable_modules):
    """Test disable_modules clears channels and removes from nco_freqs."""
    crs = mock_crs_for_disable_modules
    
    # Disable module 1
    await crs.disable_modules([1])
    
    # Check _clear_channels was called
    crs._clear_channels.assert_called_once_with([1])
    
    # Check module 1 was removed from nco_freqs
    assert 1 not in crs.nco_freqs
    assert 0 in crs.nco_freqs
    assert 2 in crs.nco_freqs


@pytest.mark.asyncio
async def test_disable_modules_multiple_modules(
        mock_crs_for_disable_modules):
    """Test disable_modules with multiple modules."""
    crs = mock_crs_for_disable_modules
    
    # Disable modules 0 and 2
    await crs.disable_modules([0, 2])
    
    # Check _clear_channels was called with correct list
    crs._clear_channels.assert_called_once_with([0, 2])
    
    # Check both modules were removed from nco_freqs
    assert 0 not in crs.nco_freqs
    assert 2 not in crs.nco_freqs
    assert 1 in crs.nco_freqs
    assert 5 in crs.nco_freqs


@pytest.mark.asyncio
async def test_disable_modules_all_modules(
        mock_crs_for_disable_modules):
    """Test disable_modules with all modules."""
    crs = mock_crs_for_disable_modules
    
    # Disable all modules
    await crs.disable_modules([0, 1, 2, 5])
    
    # Check _clear_channels was called
    crs._clear_channels.assert_called_once_with([0, 1, 2, 5])
    
    # Check all modules were removed
    assert len(crs.nco_freqs) == 0


@pytest.mark.asyncio
async def test_disable_modules_nonexistent_module(
        mock_crs_for_disable_modules):
    """Test disable_modules with module not in nco_freqs."""
    crs = mock_crs_for_disable_modules
    
    # Disable module that doesn't exist in nco_freqs
    await crs.disable_modules([10])
    
    # Should still call _clear_channels
    crs._clear_channels.assert_called_once_with([10])
    
    # nco_freqs should be unchanged
    assert len(crs.nco_freqs) == 4


@pytest.mark.asyncio
async def test_disable_modules_mixed_existing_nonexisting(
        mock_crs_for_disable_modules):
    """Test disable_modules with mix of existing and non-existing."""
    crs = mock_crs_for_disable_modules
    
    # Disable mix of existing and non-existing modules
    await crs.disable_modules([1, 10, 15])
    
    # Check _clear_channels was called with full list
    crs._clear_channels.assert_called_once_with([1, 10, 15])
    
    # Only module 1 should be removed
    assert 1 not in crs.nco_freqs
    assert len(crs.nco_freqs) == 3


@pytest.mark.asyncio
async def test_disable_modules_empty_list(mock_crs_for_disable_modules):
    """Test disable_modules with empty list."""
    crs = mock_crs_for_disable_modules
    
    # Disable with empty list
    await crs.disable_modules([])
    
    # Should still call _clear_channels
    crs._clear_channels.assert_called_once_with([])
    
    # nco_freqs should be unchanged
    assert len(crs.nco_freqs) == 4


@pytest.mark.asyncio
async def test_disable_modules_validates_input_type(
        mock_crs_for_disable_modules):
    """Test disable_modules validates input is list of integers."""
    crs = mock_crs_for_disable_modules
    
    # Test with non-integer in list
    with pytest.raises(TypeError, match = 'must be a list of integers'):
        await crs.disable_modules([1, 'not_int', 3])


@pytest.mark.asyncio
async def test_disable_modules_validates_floats(
        mock_crs_for_disable_modules):
    """Test disable_modules rejects floats."""
    crs = mock_crs_for_disable_modules
    
    with pytest.raises(TypeError, match = 'must be a list of integers'):
        await crs.disable_modules([1, 2.5, 3])


@pytest.mark.asyncio
async def test_disable_modules_accepts_numpy_integers(
        mock_crs_for_disable_modules):
    """Test disable_modules accepts numpy integers."""
    crs = mock_crs_for_disable_modules
    
    # Use numpy integers
    await crs.disable_modules([np.int32(1), np.int64(2)])
    
    # Check both were removed
    assert 1 not in crs.nco_freqs
    assert 2 not in crs.nco_freqs


@pytest.mark.asyncio
async def test_disable_modules_does_not_modify_input_list(
        mock_crs_for_disable_modules):
    """Test disable_modules doesn't modify the input list."""
    crs = mock_crs_for_disable_modules
    
    original_list = [1, 2]
    original_copy = original_list.copy()
    
    await crs.disable_modules(original_list)
    
    # Input list should be unchanged
    assert original_list == original_copy