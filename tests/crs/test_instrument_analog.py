"""
Tests for CRS analog configuration methods.

Tests set_analog_bank and set_extended_bw methods.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import warnings

from citkid.crs.instrument import CRS


@pytest.fixture
def mock_crs_for_set_analog_bank(base_crs):
    """Extend base_crs with additional mocks for set_analog_bank tests."""
    crs = base_crs
    
    # Mock async methods
    crs.d.set_analog_bank = AsyncMock()
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    crs.d.set_dac_scale = AsyncMock()
    crs.d.get_dac_scale = AsyncMock(return_value=7)
    
    # Mock UNITS
    crs.d.UNITS = MagicMock()
    crs.d.UNITS.DBM = 'DBM'
    
    # Initialize the maps with some data
    crs.nco_freqs = {1: 4.5e9, 2: 5.0e9, 5: 6.0e9, 6: 6.5e9}
    crs.fres_map = {1: [1e9], 2: [2e9], 5: [3e9], 6: [4e9]}
    crs.ares_map = {1: [-50], 2: [-50], 5: [-50], 6: [-50]}
    crs.ch_map = {1: [0], 2: [0], 5: [0], 6: [0]}
    
    return crs


@pytest.mark.asyncio
async def test_set_analog_bank_low_success(mock_crs_for_set_analog_bank):
    """Test setting analog bank to low (modules 1-4) successfully."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    crs.d.get_dac_scale = AsyncMock(return_value=5)
    
    await crs.set_analog_bank(analog_bank_high=False, full_scale_dbm=5)
    
    # Verify device methods were called
    crs.d.set_analog_bank.assert_called_once_with(high=False)
    crs.d.get_analog_bank.assert_called_once()
    
    # Verify analog_bank_high attribute was set
    assert crs.analog_bank_high is False
    
    # Verify modules 5-8 were removed from maps
    assert 5 not in crs.nco_freqs
    assert 6 not in crs.nco_freqs
    assert 5 not in crs.fres_map
    assert 6 not in crs.fres_map
    assert 5 not in crs.ares_map
    assert 6 not in crs.ares_map
    assert 5 not in crs.ch_map
    assert 6 not in crs.ch_map
    
    # Verify modules 1-2 were NOT removed
    assert 1 in crs.nco_freqs
    assert 2 in crs.nco_freqs
    
    # Verify DAC scale was set for modules 1-4
    assert crs.d.set_dac_scale.call_count == 4
    for module_idx in range(1, 5):
        crs.d.set_dac_scale.assert_any_call(5, 'DBM', module_idx)
    
    # Verify DAC scale was confirmed for modules 1-4
    assert crs.d.get_dac_scale.call_count == 4
    for module_idx in range(1, 5):
        crs.d.get_dac_scale.assert_any_call('DBM', module_idx)
    
    # Verify full_scale_dbm was stored in self.d
    assert crs.d.full_scale_dbm == 5


@pytest.mark.asyncio
async def test_set_analog_bank_high_success(mock_crs_for_set_analog_bank):
    """Test setting analog bank to high (modules 5-8) successfully."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=True)
    crs.d.get_dac_scale = AsyncMock(return_value=6)
    
    await crs.set_analog_bank(analog_bank_high=True, full_scale_dbm=6)
    
    # Verify device methods were called
    crs.d.set_analog_bank.assert_called_once_with(high=True)
    crs.d.get_analog_bank.assert_called_once()
    
    # Verify analog_bank_high attribute was set
    assert crs.analog_bank_high is True
    
    # Verify modules 1-4 were removed from maps
    assert 1 not in crs.nco_freqs
    assert 2 not in crs.nco_freqs
    assert 1 not in crs.fres_map
    assert 2 not in crs.fres_map
    assert 1 not in crs.ares_map
    assert 2 not in crs.ares_map
    assert 1 not in crs.ch_map
    assert 2 not in crs.ch_map
    
    # Verify modules 5-6 were NOT removed
    assert 5 in crs.nco_freqs
    assert 6 in crs.nco_freqs
    
    # Verify DAC scale was set for modules 5-8
    assert crs.d.set_dac_scale.call_count == 4
    for module_idx in range(5, 9):
        crs.d.set_dac_scale.assert_any_call(6, 'DBM', module_idx)
    
    # Verify full_scale_dbm was stored in self.d
    assert crs.d.full_scale_dbm == 6


@pytest.mark.asyncio
async def test_set_analog_bank_failed_to_set(mock_crs_for_set_analog_bank):
    """Test that RuntimeError is raised when analog bank fails to set."""
    crs = mock_crs_for_set_analog_bank
    # Request high but get low back
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    
    with pytest.raises(RuntimeError, match='Failed to set analog bank'):
        await crs.set_analog_bank(analog_bank_high=True,
                                  full_scale_dbm=5)


@pytest.mark.asyncio
async def test_set_analog_bank_dac_scale_mismatch(
    mock_crs_for_set_analog_bank
):
    """Test that RuntimeError is raised when DAC scale fails to set."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    # Return wrong DAC scale values
    crs.d.get_dac_scale = AsyncMock(return_value=3)
    
    with pytest.raises(RuntimeError, match='Failed to set DAC full scale'):
        await crs.set_analog_bank(analog_bank_high=False,
                                  full_scale_dbm=5)


@pytest.mark.asyncio
async def test_set_analog_bank_removes_only_correct_modules(
    mock_crs_for_set_analog_bank
):
    """Test that only the correct modules are removed from maps."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=True)
    crs.d.get_dac_scale = AsyncMock(return_value=5)
    
    # Add modules 3, 4, 7, 8 to verify boundary conditions
    crs.nco_freqs[3] = 7.0e9
    crs.nco_freqs[4] = 7.5e9
    crs.nco_freqs[7] = 8.0e9
    crs.nco_freqs[8] = 8.5e9
    
    await crs.set_analog_bank(analog_bank_high=True, full_scale_dbm=5)
    
    # Verify modules 1-4 were removed
    for idx in [1, 2, 3, 4]:
        assert idx not in crs.nco_freqs
    
    # Verify modules 5-8 were NOT removed
    for idx in [5, 6, 7, 8]:
        assert idx in crs.nco_freqs


@pytest.mark.asyncio
async def test_set_analog_bank_empty_maps(mock_crs_for_set_analog_bank):
    """Test setting analog bank when maps are already empty."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    crs.d.get_dac_scale = AsyncMock(return_value=7)
    
    # Clear all maps
    crs.nco_freqs = {}
    crs.fres_map = {}
    crs.ares_map = {}
    crs.ch_map = {}
    
    # Should not raise any errors
    await crs.set_analog_bank(analog_bank_high=False, full_scale_dbm=7)
    
    assert crs.analog_bank_high is False
    assert crs.d.full_scale_dbm == 7


@pytest.mark.asyncio
async def test_set_analog_bank_dac_scale_within_tolerance(
    mock_crs_for_set_analog_bank
):
    """Test that DAC scale within tolerance (0.1) is accepted."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    # Return values within 0.1 tolerance
    crs.d.get_dac_scale = AsyncMock(return_value=5.05)
    
    # Should not raise error
    await crs.set_analog_bank(analog_bank_high=False, full_scale_dbm=5)
    
    assert crs.d.full_scale_dbm == 5


################################################################################
################## set_analog_bank Input Validation ############################
################################################################################

@pytest.mark.asyncio
async def test_set_analog_bank_high_not_bool(mock_crs_for_set_analog_bank):
    """Test that non-bool analog_bank_high raises TypeError."""
    crs = mock_crs_for_set_analog_bank
    
    with pytest.raises(
        TypeError,
        match='analog_bank_high must be a boolean value'
    ):
        await crs.set_analog_bank(analog_bank_high=1, full_scale_dbm=5)


@pytest.mark.asyncio
async def test_set_analog_bank_high_none(mock_crs_for_set_analog_bank):
    """Test that None analog_bank_high raises TypeError."""
    crs = mock_crs_for_set_analog_bank
    
    with pytest.raises(TypeError):
        await crs.set_analog_bank(analog_bank_high=None,
                                  full_scale_dbm=5)


@pytest.mark.asyncio
async def test_set_analog_bank_full_scale_not_number(
    mock_crs_for_set_analog_bank
):
    """Test that non-numeric full_scale_dbm raises TypeError."""
    crs = mock_crs_for_set_analog_bank
    
    with pytest.raises(
        TypeError,
        match='full_scale_dbm must be a number'
    ):
        await crs.set_analog_bank(analog_bank_high=False,
                                  full_scale_dbm='5')


@pytest.mark.asyncio
async def test_set_analog_bank_full_scale_below_range(
    mock_crs_for_set_analog_bank
):
    """Test that full_scale_dbm below 0 raises ValueError."""
    crs = mock_crs_for_set_analog_bank
    
    with pytest.raises(
        ValueError,
        match='full_scale_dbm must be in \\[0, 7\\]'
    ):
        await crs.set_analog_bank(analog_bank_high=False,
                                  full_scale_dbm=-1)


@pytest.mark.asyncio
async def test_set_analog_bank_full_scale_above_range(
    mock_crs_for_set_analog_bank
):
    """Test that full_scale_dbm above 7 raises ValueError."""
    crs = mock_crs_for_set_analog_bank
    
    with pytest.raises(
        ValueError,
        match='full_scale_dbm must be in \\[0, 7\\]'
    ):
        await crs.set_analog_bank(analog_bank_high=False,
                                  full_scale_dbm=8)


@pytest.mark.asyncio
async def test_set_analog_bank_full_scale_float_valid(
    mock_crs_for_set_analog_bank
):
    """Test that float full_scale_dbm in valid range is accepted."""
    crs = mock_crs_for_set_analog_bank
    crs.d.get_analog_bank = AsyncMock(return_value=False)
    crs.d.get_dac_scale = AsyncMock(return_value=5.5)
    
    # Should not raise error
    await crs.set_analog_bank(analog_bank_high=False,
                              full_scale_dbm=5.5)
    
    assert crs.d.full_scale_dbm == 5.5


################################################################################
############################# set_extended_bw ##################################
################################################################################

@pytest.fixture
def mock_crs_for_set_extended_bw(base_crs):
    """Extend base_crs with additional mocks for set_extended_bw tests."""
    crs = base_crs
    
    # Mock async methods
    crs.d.set_extended_module_bandwidth = AsyncMock()
    crs.d.get_extended_module_bandwidth = AsyncMock(return_value=False)
    
    return crs


@pytest.mark.asyncio
async def test_set_extended_bw_false_standard(mock_crs_for_set_extended_bw):
    """Test setting extended bandwidth to False (standard 500 MHz)."""
    crs = mock_crs_for_set_extended_bw
    crs.d.get_extended_module_bandwidth = AsyncMock(return_value=False)
    
    await crs.set_extended_bw(False)
    
    # Verify device methods were called
    crs.d.set_extended_module_bandwidth.assert_called_once_with(False)
    crs.d.get_extended_module_bandwidth.assert_called_once()
    
    # Verify attributes were set
    assert crs.extended_bw is False
    assert crs.bw == 500e6


@pytest.mark.asyncio
async def test_set_extended_bw_true_extended(mock_crs_for_set_extended_bw):
    """Test setting extended bandwidth to True (600 MHz) with warning."""
    crs = mock_crs_for_set_extended_bw
    crs.d.get_extended_module_bandwidth = AsyncMock(return_value=True)
    
    with pytest.warns(
        UserWarning,
        match='Extended module bandwidth set'
    ):
        await crs.set_extended_bw(True)
    
    # Verify device methods were called
    crs.d.set_extended_module_bandwidth.assert_called_once_with(True)
    crs.d.get_extended_module_bandwidth.assert_called_once()
    
    # Verify attributes were set
    assert crs.extended_bw is True
    assert crs.bw == 600e6


@pytest.mark.asyncio
async def test_set_extended_bw_false_no_warning(mock_crs_for_set_extended_bw):
    """Test that False does not raise a warning."""
    crs = mock_crs_for_set_extended_bw
    crs.d.get_extended_module_bandwidth = AsyncMock(return_value=False)
    
    # Should not raise any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        # If a warning is raised, this will fail
        await crs.set_extended_bw(False)


@pytest.mark.asyncio
async def test_set_extended_bw_failed_to_set(mock_crs_for_set_extended_bw):
    """Test that RuntimeError is raised when bandwidth fails to set."""
    crs = mock_crs_for_set_extended_bw
    # Request True but get False back
    crs.d.get_extended_module_bandwidth = AsyncMock(return_value=False)
    
    with pytest.raises(
        RuntimeError,
        match='Failed to set extended module bandwidth'
    ):
        await crs.set_extended_bw(True)


@pytest.mark.asyncio
async def test_set_extended_bw_failed_to_unset(mock_crs_for_set_extended_bw):
    """Test that RuntimeError is raised when bandwidth fails to unset."""
    crs = mock_crs_for_set_extended_bw
    # Request False but get True back
    crs.d.get_extended_module_bandwidth = AsyncMock(return_value=True)
    
    with pytest.raises(
        RuntimeError,
        match='Failed to set extended module bandwidth'
    ):
        await crs.set_extended_bw(False)


################################################################################
#################### set_extended_bw Input Validation ##########################
################################################################################

@pytest.mark.asyncio
async def test_set_extended_bw_not_bool(mock_crs_for_set_extended_bw):
    """Test that non-bool extended raises TypeError."""
    crs = mock_crs_for_set_extended_bw
    
    with pytest.raises(
        TypeError,
        match='extended must be a boolean value'
    ):
        await crs.set_extended_bw(1)


@pytest.mark.asyncio
async def test_set_extended_bw_string(mock_crs_for_set_extended_bw):
    """Test that string extended raises TypeError."""
    crs = mock_crs_for_set_extended_bw
    
    with pytest.raises(TypeError):
        await crs.set_extended_bw('True')


@pytest.mark.asyncio
async def test_set_extended_bw_none(mock_crs_for_set_extended_bw):
    """Test that None extended raises TypeError."""
    crs = mock_crs_for_set_extended_bw
    
    with pytest.raises(TypeError):
        await crs.set_extended_bw(None)
