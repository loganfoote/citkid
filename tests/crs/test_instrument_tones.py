"""
Tests for CRS tone writing and channel management.

Tests write_tones, _write_tones macro, and _clear_channels methods.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import numpy as np
import warnings

from citkid.crs.instrument import CRS


################################################################################
############################# _clear_channels ##################################
################################################################################

@pytest.fixture
def mock_crs_for_clear_channels(base_crs):
    """Extend base_crs for _clear_channels tests."""
    crs = base_crs
    
    # Mock device clear_channels method
    crs.d.clear_channels = AsyncMock()
    
    # Set up initial state with tones
    crs.fres_map = {
        0: np.array([3.5e9, 3.501e9, 3.502e9]),
        1: np.array([4.0e9, 4.001e9]),
        2: np.array([4.5e9, 4.501e9, 4.502e9, 4.503e9])
    }
    crs.ares_map = {
        0: np.array([-50, -51, -52]),
        1: np.array([-50, -51]),
        2: np.array([-50, -51, -52, -53])
    }
    crs.ch_map = {
        0: [0, 1, 2],
        1: [3, 4],
        2: [5, 6, 7, 8]
    }
    crs.ntones = 9  # Total of 3 + 2 + 4 channels
    
    return crs


@pytest.mark.asyncio
async def test_clear_channels_basic_functionality(
        mock_crs_for_clear_channels):
    """Test _clear_channels clears device and updates maps."""
    crs = mock_crs_for_clear_channels
    
    # Clear module 1
    await crs._clear_channels([1])
    
    # Check device method was called
    crs.d.clear_channels.assert_called_once_with(module=1)
    
    # Check module 1 was removed from all maps
    assert 1 not in crs.fres_map
    assert 1 not in crs.ares_map
    assert 1 not in crs.ch_map
    
    # Check other modules still exist
    assert 0 in crs.fres_map
    assert 2 in crs.fres_map
    
    # Check ntones was decremented correctly
    assert crs.ntones == 7  # 9 - 2


@pytest.mark.asyncio
async def test_clear_channels_updates_ntones_correctly(
        mock_crs_for_clear_channels):
    """Test _clear_channels decrements ntones by channel count."""
    crs = mock_crs_for_clear_channels
    
    # Clear module 0 (3 channels)
    await crs._clear_channels([0])
    assert crs.ntones == 6  # 9 - 3
    
    # Clear module 2 (4 channels)
    await crs._clear_channels([2])
    assert crs.ntones == 2  # 6 - 4


@pytest.mark.asyncio
async def test_clear_channels_multiple_modules(
        mock_crs_for_clear_channels):
    """Test _clear_channels with multiple modules."""
    crs = mock_crs_for_clear_channels
    
    # Clear modules 0 and 2
    await crs._clear_channels([0, 2])
    
    # Check device method was called twice
    assert crs.d.clear_channels.call_count == 2
    crs.d.clear_channels.assert_any_call(module=0)
    crs.d.clear_channels.assert_any_call(module=2)
    
    # Check both modules removed from maps
    assert 0 not in crs.fres_map
    assert 2 not in crs.fres_map
    assert 0 not in crs.ares_map
    assert 2 not in crs.ares_map
    assert 0 not in crs.ch_map
    assert 2 not in crs.ch_map
    
    # Check ntones updated (3 + 4 = 7 channels removed)
    assert crs.ntones == 2  # 9 - 7


@pytest.mark.asyncio
async def test_clear_channels_all_modules(mock_crs_for_clear_channels):
    """Test _clear_channels removes all modules."""
    crs = mock_crs_for_clear_channels
    
    # Clear all modules
    await crs._clear_channels([0, 1, 2])
    
    # Check all maps are empty
    assert len(crs.fres_map) == 0
    assert len(crs.ares_map) == 0
    assert len(crs.ch_map) == 0
    
    # Check ntones is 0
    assert crs.ntones == 0


@pytest.mark.asyncio
async def test_clear_channels_nonexistent_module(
        mock_crs_for_clear_channels):
    """Test _clear_channels with module not in maps."""
    crs = mock_crs_for_clear_channels
    
    # Clear module that doesn't exist in maps
    await crs._clear_channels([10])
    
    # Should still call device method
    crs.d.clear_channels.assert_called_once_with(module=10)
    
    # ntones should be unchanged (no channels to remove)
    assert crs.ntones == 9
    
    # Maps should be unchanged
    assert len(crs.fres_map) == 3


@pytest.mark.asyncio
async def test_clear_channels_mixed_existing_nonexisting(
        mock_crs_for_clear_channels):
    """Test _clear_channels with mix of existing and non-existing."""
    crs = mock_crs_for_clear_channels
    
    # Clear mix of existing and non-existing
    await crs._clear_channels([1, 10, 15])
    
    # Check device called for all
    assert crs.d.clear_channels.call_count == 3
    
    # Only module 1 should be removed from maps
    assert 1 not in crs.fres_map
    assert 0 in crs.fres_map
    assert 2 in crs.fres_map
    
    # ntones should only decrement for module 1
    assert crs.ntones == 7  # 9 - 2


@pytest.mark.asyncio
async def test_clear_channels_empty_list(mock_crs_for_clear_channels):
    """Test _clear_channels with empty list."""
    crs = mock_crs_for_clear_channels
    
    # Clear with empty list
    await crs._clear_channels([])
    
    # Device should not be called
    crs.d.clear_channels.assert_not_called()
    
    # Everything should be unchanged
    assert crs.ntones == 9
    assert len(crs.fres_map) == 3


@pytest.mark.asyncio
async def test_clear_channels_ntones_never_negative(
        mock_crs_for_clear_channels):
    """Test ntones doesn't go negative."""
    crs = mock_crs_for_clear_channels
    
    # Clear all modules
    await crs._clear_channels([0, 1, 2])
    assert crs.ntones == 0
    
    # Try to clear again (modules already cleared)
    await crs._clear_channels([0, 1, 2])
    
    # ntones should still be 0, not negative
    assert crs.ntones == 0


@pytest.mark.asyncio
async def test_clear_channels_validates_input_type(
        mock_crs_for_clear_channels):
    """Test _clear_channels validates input is list of integers."""
    crs = mock_crs_for_clear_channels
    
    # Test with non-integer in list
    with pytest.raises(TypeError, match='must be a list of integers'):
        await crs._clear_channels([1, 'not_int', 3])


@pytest.mark.asyncio
async def test_clear_channels_validates_floats(
        mock_crs_for_clear_channels):
    """Test _clear_channels rejects floats."""
    crs = mock_crs_for_clear_channels
    
    with pytest.raises(TypeError, match='must be a list of integers'):
        await crs._clear_channels([1, 2.5, 3])


@pytest.mark.asyncio
async def test_clear_channels_accepts_numpy_integers(
        mock_crs_for_clear_channels):
    """Test _clear_channels accepts numpy integers."""
    crs = mock_crs_for_clear_channels
    
    # Use numpy integers
    await crs._clear_channels([np.int32(1), np.int64(2)])
    
    # Check both were cleared
    assert 1 not in crs.fres_map
    assert 2 not in crs.fres_map


@pytest.mark.asyncio
async def test_clear_channels_removes_from_all_three_maps(
        mock_crs_for_clear_channels):
    """Test _clear_channels removes from fres_map, ares_map, ch_map."""
    crs = mock_crs_for_clear_channels
    
    module_idx = 1
    
    # Verify module exists in all maps before clearing
    assert module_idx in crs.fres_map
    assert module_idx in crs.ares_map
    assert module_idx in crs.ch_map
    
    # Clear the module
    await crs._clear_channels([module_idx])
    
    # Verify removed from all three maps
    assert module_idx not in crs.fres_map
    assert module_idx not in crs.ares_map
    assert module_idx not in crs.ch_map


@pytest.mark.asyncio
async def test_clear_channels_with_different_ch_map_sizes(
        mock_crs_for_clear_channels):
    """Test _clear_channels correctly counts channels of different sizes."""
    crs = mock_crs_for_clear_channels
    
    # Module 0 has 3 channels
    initial_ntones = crs.ntones
    await crs._clear_channels([0])
    assert crs.ntones == initial_ntones - 3
    
    # Reset and test module 1 with 2 channels
    crs.ntones = 9
    crs.fres_map[0] = np.array([3.5e9, 3.501e9, 3.502e9])
    crs.ares_map[0] = np.array([-50, -51, -52])
    crs.ch_map[0] = [0, 1, 2]
    
    await crs._clear_channels([1])
    assert crs.ntones == 7  # 9 - 2


@pytest.mark.asyncio
async def test_clear_channels_does_not_modify_input_list(
        mock_crs_for_clear_channels):
    """Test _clear_channels doesn't modify the input list."""
    crs = mock_crs_for_clear_channels
    
    original_list = [1, 2]
    original_copy = original_list.copy()
    
    await crs._clear_channels(original_list)
    
    # Input list should be unchanged
    assert original_list == original_copy


################################################################################
############################### write_tones ####################################
################################################################################

@pytest.fixture
def mock_crs_for_write_tones(base_crs):
    """Extend base_crs for write_tones tests."""
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create, \
         patch('citkid.crs.instrument.util.get_modules') as mock_get_mods, \
         patch('citkid.crs.instrument.take_netanal') as mock_netanal:
        
        crs = base_crs
        
        # Set up NCO frequencies and bandwidth
        crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}
        crs.bw = 500e6
        
        # Mock _clear_channels
        crs._clear_channels = AsyncMock()
        
        # Mock create_ch_map to return valid mapping for 4 tones by default
        mock_create.return_value = (
            {1: [0, 1], 2: [2, 3]},  # ch_map for 4 tones
            []  # missing_chs
        )
        
        # Mock _safe_concatenate_frequencies to return input unchanged
        mock_netanal._safe_concatenate_frequencies = MagicMock(
            side_effect=lambda f, nco: f
        )
        
        # Mock get_modules to return object with _write_tones
        mock_modules = MagicMock()
        mock_modules._write_tones = AsyncMock()
        mock_get_mods.return_value = mock_modules
        
        yield crs, mock_create, mock_get_mods, mock_modules, mock_netanal


@pytest.mark.asyncio
async def test_write_tones_basic_functionality(mock_crs_for_write_tones):
    """Test write_tones basic functionality."""
    crs, mock_create, mock_get_mods, mock_modules, _ = \
        mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53])
    
    await crs.write_tones(fres, ares)
    
    # Check _clear_channels was called
    crs._clear_channels.assert_called_once()
    
    # Check maps were updated
    assert 1 in crs.fres_map
    assert 2 in crs.fres_map
    assert 1 in crs.ares_map
    assert 2 in crs.ares_map
    assert 1 in crs.ch_map
    assert 2 in crs.ch_map
    
    # Check _write_tones was called
    mock_modules._write_tones.assert_called_once()
    
    # Check ntones was set
    assert crs.ntones == 4


@pytest.mark.asyncio
async def test_write_tones_validates_nco_freqs_set(
        mock_crs_for_write_tones):
    """Test write_tones raises error if NCO frequencies not set."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    crs.nco_freqs = {}
    
    with pytest.raises(RuntimeError, match='NCO frequencies are not set'):
        await crs.write_tones([4.0e9], [-50])


@pytest.mark.asyncio
async def test_write_tones_converts_to_numpy_arrays(
        mock_crs_for_write_tones):
    """Test write_tones converts inputs to numpy arrays."""
    crs, mock_create, _, _, _ = mock_crs_for_write_tones
    
    # Override for 2 tones
    mock_create.return_value = ({1: [0, 1]}, [])
    
    # Use Python lists
    fres = [3.9e9, 4.0e9]
    ares = [-50, -51]
    
    await crs.write_tones(fres, ares)
    
    # Check converted to numpy arrays in fres_map
    assert isinstance(crs.fres_map[1], np.ndarray)
    assert isinstance(crs.ares_map[1], np.ndarray)


@pytest.mark.asyncio
async def test_write_tones_validates_matching_shapes(
        mock_crs_for_write_tones):
    """Test write_tones validates fres and ares have same shape."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9, 4.1e9])
    ares = np.array([-50, -51])  # Different length
    
    with pytest.raises(ValueError, match='same shape'):
        await crs.write_tones(fres, ares)


@pytest.mark.asyncio
async def test_write_tones_handles_empty_arrays(mock_crs_for_write_tones):
    """Test write_tones returns early with empty arrays."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([])
    ares = np.array([])
    
    result = await crs.write_tones(fres, ares)
    
    # Should return 0 and not call other methods
    assert result == 0
    crs._clear_channels.assert_not_called()


@pytest.mark.asyncio
async def test_write_tones_with_provided_ch_map(mock_crs_for_write_tones):
    """Test write_tones with user-provided ch_map."""
    crs, mock_create, mock_get_mods, mock_modules, _ = \
        mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53])
    ch_map = {1: [0, 1, 2], 2: [3]}
    
    await crs.write_tones(fres, ares, ch_map=ch_map)
    
    # Should not call create_ch_map
    mock_create.assert_not_called()
    
    # Check ch_map was used
    assert crs.ch_map[1] == [0, 1, 2]
    assert crs.ch_map[2] == [3]


@pytest.mark.asyncio
async def test_write_tones_detects_missing_channels_with_ch_map(
        mock_crs_for_write_tones):
    """Test write_tones detects missing channels when ch_map provided."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53])
    # ch_map missing channel 2
    ch_map = {1: [0, 1, 3]}
    
    with pytest.raises(ValueError, match='Tones must be within'):
        await crs.write_tones(fres, ares, ch_map=ch_map)


@pytest.mark.asyncio
async def test_write_tones_missing_channels_with_allow_missing(
        mock_crs_for_write_tones):
    """Test write_tones warns about missing channels with allow_missing."""
    crs, mock_create, _, _, _ = mock_crs_for_write_tones
    
    # Mock create_ch_map to return missing channels
    mock_create.return_value = (
        {1: [0, 1]},  # ch_map missing channels 2, 3
        [2, 3]  # missing_chs
    )
    
    fres = np.array([3.9e9, 4.0e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await crs.write_tones(fres, ares, allow_missing=True)
        
        assert len(w) == 1
        assert 'Ignoring' in str(w[0].message)
        assert '2' in str(w[0].message)  # 2 tones


@pytest.mark.asyncio
async def test_write_tones_missing_channels_raises_without_allow(
        mock_crs_for_write_tones):
    """Test write_tones raises error for missing channels."""
    crs, mock_create, _, _, _ = mock_crs_for_write_tones
    
    # Mock create_ch_map to return missing channels
    mock_create.return_value = (
        {1: [0, 1]},
        [2, 3]  # missing_chs
    )
    
    fres = np.array([3.9e9, 4.0e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53])
    
    with pytest.raises(ValueError, match='Tones must be within'):
        await crs.write_tones(fres, ares, allow_missing=False)


@pytest.mark.asyncio
async def test_write_tones_clears_existing_channels(
        mock_crs_for_write_tones):
    """Test write_tones clears existing channels before writing."""
    crs, mock_create, _, _, _ = mock_crs_for_write_tones
    
    # Override for 2 tones
    mock_create.return_value = ({1: [0, 1]}, [])
    
    # Set up existing maps
    crs.fres_map = {1: np.array([3.9e9]), 2: np.array([4.5e9])}
    crs.ares_map = {1: np.array([-50]), 2: np.array([-51])}
    crs.ch_map = {1: [0], 2: [1]}
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-52, -53])
    
    await crs.write_tones(fres, ares)
    
    # Check _clear_channels was called with existing module indices
    crs._clear_channels.assert_called_once_with([1, 2])


@pytest.mark.asyncio
async def test_write_tones_updates_maps_correctly(
        mock_crs_for_write_tones):
    """Test write_tones updates fres_map, ares_map, ch_map."""
    crs, mock_create, _, _, _ = mock_crs_for_write_tones
    
    mock_create.return_value = (
        {1: [0, 1], 2: [2]},
        []
    )
    
    fres = np.array([3.9e9, 4.0e9, 4.5e9])
    ares = np.array([-50, -51, -52])
    
    await crs.write_tones(fres, ares)
    
    # Check fres_map
    np.testing.assert_array_equal(crs.fres_map[1], fres[[0, 1]])
    np.testing.assert_array_equal(crs.fres_map[2], fres[[2]])
    
    # Check ares_map
    np.testing.assert_array_equal(crs.ares_map[1], ares[[0, 1]])
    np.testing.assert_array_equal(crs.ares_map[2], ares[[2]])
    
    # Check ch_map
    assert crs.ch_map[1] == [0, 1]
    assert crs.ch_map[2] == [2]


@pytest.mark.asyncio
async def test_write_tones_calls_safe_concatenate_frequencies(
        mock_crs_for_write_tones):
    """Test write_tones calls dithering function for each module."""
    crs, _, _, _, mock_netanal = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53])
    
    await crs.write_tones(fres, ares)
    
    # Should be called once for each module with tones
    assert mock_netanal._safe_concatenate_frequencies.call_count == 2


@pytest.mark.asyncio
async def test_write_tones_skips_empty_modules_in_dithering(
        mock_crs_for_write_tones):
    """Test write_tones skips empty modules in dithering."""
    crs, mock_create, _, _, mock_netanal = mock_crs_for_write_tones
    
    # Mock ch_map with one empty module
    mock_create.return_value = (
        {1: [0, 1], 2: []},  # Module 2 is empty
        []
    )
    
    fres = np.array([3.9e9, 4.0e9])
    ares = np.array([-50, -51])
    
    await crs.write_tones(fres, ares)
    
    # Should only be called for module 1
    assert mock_netanal._safe_concatenate_frequencies.call_count == 1


@pytest.mark.asyncio
async def test_write_tones_calls_get_modules_with_correct_indices(
        mock_crs_for_write_tones):
    """Test write_tones calls get_modules with correct module indices."""
    crs, mock_create, mock_get_mods, _, _ = mock_crs_for_write_tones
    
    mock_create.return_value = (
        {1: [0], 2: [1], 5: [2]},
        []
    )
    
    fres = np.array([3.9e9, 4.5e9, 5.0e9])
    ares = np.array([-50, -51, -52])
    
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9, 5: 5.0e9}
    
    await crs.write_tones(fres, ares)
    
    # Check get_modules was called with list of module indices
    call_args = mock_get_mods.call_args[0]
    assert call_args[0] == crs.d
    assert set(call_args[1]) == {1, 2, 5}


@pytest.mark.asyncio
async def test_write_tones_calls_write_tones_macro(
        mock_crs_for_write_tones):
    """Test write_tones calls _write_tones macro with correct args."""
    crs, mock_create, _, mock_modules, _ = mock_crs_for_write_tones
    
    # Override for 2 tones
    mock_create.return_value = ({1: [0, 1]}, [])
    
    fres = np.array([3.9e9, 4.0e9])
    ares = np.array([-50, -51])
    
    await crs.write_tones(fres, ares)
    
    # Check _write_tones was called
    mock_modules._write_tones.assert_called_once()
    call_args = mock_modules._write_tones.call_args[0]
    
    # Check arguments: nco_freqs, fres_map, ares_map
    assert call_args[0] == crs.nco_freqs
    assert call_args[1] == crs.fres_map
    assert call_args[2] == crs.ares_map


@pytest.mark.asyncio
async def test_write_tones_sets_ntones_correctly(mock_crs_for_write_tones):
    """Test write_tones sets ntones to number of input tones."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9, 4.1e9, 4.4e9, 4.5e9])
    ares = np.array([-50, -51, -52, -53, -54])
    
    await crs.write_tones(fres, ares)
    
    assert crs.ntones == 5


@pytest.mark.asyncio
async def test_write_tones_validates_ch_map_format(
        mock_crs_for_write_tones):
    """Test write_tones validates ch_map format."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9])
    ares = np.array([-50, -51])
    
    # Test invalid ch_map (not a dict)
    with pytest.raises(TypeError, match='must be a dictionary'):
        await crs.write_tones(fres, ares, ch_map=[1, 2, 3])


@pytest.mark.asyncio
async def test_write_tones_validates_ch_map_keys(mock_crs_for_write_tones):
    """Test write_tones validates ch_map keys are integers."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9])
    ares = np.array([-50, -51])
    
    # Test ch_map with non-integer key
    with pytest.raises(TypeError, match='keys must be integers'):
        await crs.write_tones(fres, ares, ch_map={'1': [0, 1]})


@pytest.mark.asyncio
async def test_write_tones_validates_ch_map_values(
        mock_crs_for_write_tones):
    """Test write_tones validates ch_map values are lists of integers."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9])
    ares = np.array([-50, -51])
    
    # Test ch_map with non-integer in value list
    with pytest.raises(TypeError, match='values must be lists of integers'):
        await crs.write_tones(fres, ares, ch_map={1: [0, '1']})


@pytest.mark.asyncio
async def test_write_tones_preserves_original_ch_map(
        mock_crs_for_write_tones):
    """Test write_tones doesn't modify input ch_map."""
    crs, _, _, _, _ = mock_crs_for_write_tones
    
    fres = np.array([3.9e9, 4.0e9])
    ares = np.array([-50, -51])
    original_ch_map = {1: [0, 1]}
    original_copy = {k: v.copy() for k, v in original_ch_map.items()}
    
    await crs.write_tones(fres, ares, ch_map=original_ch_map)
    
    # Input ch_map should be unchanged
    assert original_ch_map == original_copy


################################################################################
############################## _write_tones ####################################
################################################################################

@pytest.fixture
def mock_module_for_write_tones():
    """Fixture for testing _write_tones macro."""
    import rfmux
    
    mock_module = MagicMock(spec=rfmux.ReadoutModule)
    mock_module.module = 1
    mock_crs = Mock()
    mock_module.crs = mock_crs
    
    # Set default values
    mock_crs.full_scale_dbm = 0.0
    
    # Mock clear_channels
    mock_crs.clear_channels = AsyncMock()
    
    # Mock tuber_context - return an async context manager
    class MockTuberContext:
        def __init__(self):
            self.set_frequency = Mock()
            self.set_amplitude = Mock()
            self._call_mock = AsyncMock()
            
        def __call__(self):
            return self._call_mock()
            
    mock_ctx = MockTuberContext()
    
    mock_crs.tuber_context = MagicMock()
    mock_crs.tuber_context.return_value.__aenter__ = AsyncMock(
        return_value=mock_ctx
    )
    mock_crs.tuber_context.return_value.__aexit__ = AsyncMock(
        return_value=None
    )
    
    yield mock_module, mock_crs, mock_ctx


@pytest.mark.asyncio
async def test_write_tones_macro_converts_to_numpy(
        mock_module_for_write_tones):
    """Test _write_tones converts inputs to numpy arrays."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, _ = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: [3.9e9, 4.1e9]}  # Python list
    ares_map = {1: [-50, -51]}  # Python list
    
    # Should not raise
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)


@pytest.mark.asyncio
async def test_write_tones_macro_extracts_module_data(
        mock_module_for_write_tones):
    """Test _write_tones extracts data for specific module."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, mock_ctx = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9, 2: 5.0e9}
    fres_map = {1: np.array([3.9e9, 4.1e9]), 2: np.array([5.1e9])}
    ares_map = {1: np.array([-50, -51]), 2: np.array([-52])}
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    # Check that only module 1 data was used (2 calls)
    assert mock_ctx.set_frequency.call_count == 2
    assert mock_ctx.set_amplitude.call_count == 2


@pytest.mark.asyncio
async def test_write_tones_macro_raises_if_nco_not_set(
        mock_module_for_write_tones):
    """Test _write_tones raises if NCO freq not set for module."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, _ = mock_module_for_write_tones
    
    nco_freqs = {2: 5.0e9}  # Missing module 1
    fres_map = {1: np.array([3.9e9])}
    ares_map = {1: np.array([-50])}
    
    with pytest.raises(Exception, match='NCO frequency has not been set'):
        await _write_tones(mock_module, nco_freqs, fres_map, ares_map)


@pytest.mark.asyncio
async def test_write_tones_macro_raises_if_ares_exceeds_full_scale(
        mock_module_for_write_tones):
    """Test _write_tones raises if ares exceeds full_scale_dbm."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, mock_crs, _ = mock_module_for_write_tones
    mock_crs.full_scale_dbm = -10.0
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9])}
    ares_map = {1: np.array([-5])}  # Exceeds -10 dBm
    
    with pytest.raises(ValueError, match='ares must not exceed'):
        await _write_tones(mock_module, nco_freqs, fres_map, ares_map)


@pytest.mark.asyncio
async def test_write_tones_macro_warns_low_power_few_tones(
        mock_module_for_write_tones):
    """Test _write_tones warns for low power with few tones."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, _ = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9, 4.0e9])}  # < 100 tones
    ares_map = {1: np.array([-65, -70])}  # < -60 dBm
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
        assert len(w) == 1
        assert 'digitization noise may occur' in str(w[0].message)


@pytest.mark.asyncio
async def test_write_tones_macro_no_warn_low_power_many_tones(
        mock_module_for_write_tones):
    """Test _write_tones does not warn for low power with many tones."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, _ = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9] * 100)}  # >= 100 tones
    ares_map = {1: np.array([-65] * 100)}  # < -60 dBm
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
        assert len(w) == 0


@pytest.mark.asyncio
async def test_write_tones_macro_converts_ares_to_amplitude(
        mock_module_for_write_tones):
    """Test _write_tones converts ares from dBm to amplitude."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, mock_crs, mock_ctx = mock_module_for_write_tones
    mock_crs.full_scale_dbm = 0.0
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9])}
    ares_map = {1: np.array([-6.0])}  # -6 dBm
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    # Check amplitude conversion: 10^((-6 - 0) / 20) = 10^(-0.3) ≈ 0.5012
    call_args = mock_ctx.set_amplitude.call_args[0]
    assert np.isclose(call_args[0], 0.5012, rtol=1e-3)


@pytest.mark.asyncio
async def test_write_tones_macro_clears_channels(
        mock_module_for_write_tones):
    """Test _write_tones clears channels before writing."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, mock_crs, _ = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9])}
    ares_map = {1: np.array([-50])}
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    mock_crs.clear_channels.assert_called_once_with(module=1)


@pytest.mark.asyncio
async def test_write_tones_macro_writes_frequencies_relative_to_nco(
        mock_module_for_write_tones):
    """Test _write_tones writes frequencies relative to NCO."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, mock_ctx = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9, 4.1e9])}
    ares_map = {1: np.array([-50, -51])}
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    # Check first call: 3.9e9 - 4.0e9 = -0.1e9
    call1 = mock_ctx.set_frequency.call_args_list[0]
    assert call1[0][0] == -0.1e9
    assert call1[1]['channel'] == 1
    assert call1[1]['module'] == 1
    
    # Check second call: 4.1e9 - 4.0e9 = 0.1e9
    call2 = mock_ctx.set_frequency.call_args_list[1]
    assert call2[0][0] == 0.1e9
    assert call2[1]['channel'] == 2
    assert call2[1]['module'] == 1


@pytest.mark.asyncio
async def test_write_tones_macro_writes_amplitudes_with_correct_channels(
        mock_module_for_write_tones):
    """Test _write_tones writes amplitudes with correct channels."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, mock_crs, mock_ctx = mock_module_for_write_tones
    mock_crs.full_scale_dbm = 0.0
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9, 4.1e9])}
    ares_map = {1: np.array([-6.0, -12.0])}
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    # Check first call
    call1 = mock_ctx.set_amplitude.call_args_list[0]
    assert call1[1]['channel'] == 1
    assert call1[1]['module'] == 1
    
    # Check second call
    call2 = mock_ctx.set_amplitude.call_args_list[1]
    assert call2[1]['channel'] == 2
    assert call2[1]['module'] == 1


@pytest.mark.asyncio
async def test_write_tones_macro_calls_tuber_context(
        mock_module_for_write_tones):
    """Test _write_tones calls tuber context and executes."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, mock_crs, mock_ctx = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres_map = {1: np.array([3.9e9])}
    ares_map = {1: np.array([-50])}
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    # Check tuber_context was used
    mock_crs.tuber_context.assert_called_once()
    # Check context was executed
    mock_ctx._call_mock.assert_called_once()


@pytest.mark.asyncio
async def test_write_tones_macro_handles_multiple_tones(
        mock_module_for_write_tones):
    """Test _write_tones handles multiple tones correctly."""
    from citkid.crs.instrument import _write_tones
    
    mock_module, _, mock_ctx = mock_module_for_write_tones
    
    nco_freqs = {1: 4.0e9}
    fres = np.array([3.9e9, 3.95e9, 4.05e9, 4.1e9])
    ares = np.array([-50, -51, -52, -53])
    fres_map = {1: fres}
    ares_map = {1: ares}
    
    await _write_tones(mock_module, nco_freqs, fres_map, ares_map)
    
    # Check 4 frequencies and 4 amplitudes were set
    assert mock_ctx.set_frequency.call_count == 4
    assert mock_ctx.set_amplitude.call_count == 4
    
    # Verify all channels are sequential
    for i in range(4):
        freq_call = mock_ctx.set_frequency.call_args_list[i]
        amp_call = mock_ctx.set_amplitude.call_args_list[i]
        assert freq_call[1]['channel'] == i + 1
        assert amp_call[1]['channel'] == i + 1
