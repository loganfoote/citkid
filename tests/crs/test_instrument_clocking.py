"""
Tests for CRS.set_clock_source method.

Tests clock source configuration and validation.
"""

import pytest
from unittest.mock import AsyncMock

from citkid.crs.instrument import CRS


@pytest.fixture
def mock_crs_for_set_clock(base_crs):
    """Extend base_crs with additional mocks for set_clock_source tests."""
    crs = base_crs
    
    # Mock async methods
    crs.d.set_clock_source = AsyncMock()
    crs.d.get_clock_source = AsyncMock(return_value='VCXO')
    
    return crs


@pytest.mark.asyncio
async def test_set_clock_source_vcxo_success(mock_crs_for_set_clock,
                                             capsys):
    """Test setting clock source to VCXO successfully."""
    crs = mock_crs_for_set_clock
    crs.d.get_clock_source = AsyncMock(return_value='VCXO')
    
    await crs.set_clock_source('VCXO')
    
    # Verify device methods were called
    crs.d.set_clock_source.assert_called_once_with('VCXO')
    crs.d.get_clock_source.assert_called_once()
    
    # Verify clock_source attribute was set
    assert crs.clock_source == 'VCXO'
    
    # Verify verbose output
    captured = capsys.readouterr()
    assert 'Clock source set to VCXO' in captured.out


@pytest.mark.asyncio
async def test_set_clock_source_sma_success(mock_crs_for_set_clock,
                                            capsys):
    """Test setting clock source to SMA successfully."""
    crs = mock_crs_for_set_clock
    crs.d.get_clock_source = AsyncMock(return_value='SMA')
    
    await crs.set_clock_source('SMA')
    
    # Verify device methods were called
    crs.d.set_clock_source.assert_called_once_with('SMA')
    crs.d.get_clock_source.assert_called_once()
    
    # Verify clock_source attribute was set
    assert crs.clock_source == 'SMA'
    
    # Verify verbose output
    captured = capsys.readouterr()
    assert 'Clock source set to SMA' in captured.out


@pytest.mark.asyncio
async def test_set_clock_source_fallback_warning(mock_crs_for_set_clock):
    """Test warning when requested clock source is unavailable."""
    crs = mock_crs_for_set_clock
    # Request SMA but get VCXO back
    crs.d.get_clock_source = AsyncMock(return_value='VCXO')
    
    with pytest.warns(
        UserWarning,
        match='Requested clock source SMA unavailable.*Using VCXO instead'
    ):
        await crs.set_clock_source('SMA')
    
    # Verify clock_source attribute reflects actual value
    assert crs.clock_source == 'VCXO'


@pytest.mark.asyncio
async def test_set_clock_source_verbose_false_no_print(
    mock_crs_for_set_clock,
    capsys
):
    """Test that verbose=False suppresses output."""
    crs = mock_crs_for_set_clock
    crs.d.get_clock_source = AsyncMock(return_value='VCXO')
    
    await crs.set_clock_source('VCXO', verbose=False)
    
    # Verify no output
    captured = capsys.readouterr()
    assert 'Clock source set to' not in captured.out


@pytest.mark.asyncio
async def test_set_clock_source_verbose_true_with_warning(
    mock_crs_for_set_clock,
    capsys
):
    """Test that print occurs even when warning is raised."""
    crs = mock_crs_for_set_clock
    crs.d.get_clock_source = AsyncMock(return_value='VCXO')
    
    with pytest.warns(UserWarning):
        await crs.set_clock_source('SMA', verbose=True)
    
    # Verify output shows actual clock source
    captured = capsys.readouterr()
    assert 'Clock source set to VCXO' in captured.out


################################################################################
#################### set_clock_source Input Validation #########################
################################################################################

@pytest.mark.asyncio
async def test_set_clock_source_invalid_value(mock_crs_for_set_clock):
    """Test that invalid clock_source raises ValueError."""
    crs = mock_crs_for_set_clock
    
    with pytest.raises(
        ValueError,
        match="clock_source must be 'VCXO' or 'SMA'"
    ):
        await crs.set_clock_source('INVALID')


@pytest.mark.asyncio
async def test_set_clock_source_lowercase(mock_crs_for_set_clock):
    """Test that lowercase clock_source raises ValueError."""
    crs = mock_crs_for_set_clock
    
    with pytest.raises(ValueError):
        await crs.set_clock_source('vcxo')


@pytest.mark.asyncio
async def test_set_clock_source_not_string(mock_crs_for_set_clock):
    """Test that non-string clock_source raises ValueError."""
    crs = mock_crs_for_set_clock
    
    with pytest.raises(ValueError):
        await crs.set_clock_source(123)


@pytest.mark.asyncio
async def test_set_clock_source_none(mock_crs_for_set_clock):
    """Test that None clock_source raises ValueError."""
    crs = mock_crs_for_set_clock
    
    with pytest.raises(ValueError):
        await crs.set_clock_source(None)
