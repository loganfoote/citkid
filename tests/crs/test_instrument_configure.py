"""
Tests for CRS.configure_system method.

Tests system configuration, firmware validation, and parameter validation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from citkid.crs.instrument import CRS


@pytest.fixture
def mock_crs_for_configure(base_crs):
    """Extend base_crs with additional mocks for configure_system tests."""
    crs = base_crs
    
    # Mock async methods
    crs.d.resolve = AsyncMock()
    crs.d.set_timestamp_port = AsyncMock()
    
    # Mock firmware release
    firmware_release = MagicMock()
    firmware_release.version = '1.6.0rc3'
    crs.d.get_firmware_release = AsyncMock(return_value=firmware_release)
    
    # Mock timestamp port
    crs.d.get_timestamp_port = AsyncMock(return_value='TEST')
    crs.d.TIMESTAMP_PORT = MagicMock()
    crs.d.TIMESTAMP_PORT.TEST = 'TEST'
    
    return crs


@pytest.mark.asyncio
async def test_configure_system_default_parameters(mock_crs_for_configure,
                                                    capsys):
    """Test configure_system with default parameters."""
    crs = mock_crs_for_configure
    
    # Mock the method calls
    with patch.object(crs, 'set_clock_source',
                      new_callable=AsyncMock) as mock_set_clock, \
         patch.object(crs, 'set_extended_bw',
                      new_callable=AsyncMock) as mock_set_bw, \
         patch.object(crs, 'set_analog_bank',
                      new_callable=AsyncMock) as mock_set_bank, \
         patch.object(crs, 'set_decimation',
                      new_callable=AsyncMock) as mock_set_dec:
        
        await crs.configure_system()
        
        # Verify d.resolve was called
        crs.d.resolve.assert_called_once()
        
        # Verify firmware version was checked
        crs.d.get_firmware_release.assert_called_once()
        
        # Verify timestamp port was checked but not set (already 'TEST')
        crs.d.get_timestamp_port.assert_called_once()
        crs.d.set_timestamp_port.assert_not_called()
        
        # Verify method calls with correct parameters
        mock_set_clock.assert_called_once_with('VCXO', verbose=True)
        mock_set_bw.assert_called_once_with(False)
        mock_set_bank.assert_called_once_with(False, 7)
        mock_set_dec.assert_called_once_with(
            6, short=False, module_idxs=None, verbose=True
        )
        
        # Verify ntones was set
        assert crs.ntones == 0
        
        # Verify verbose output
        captured = capsys.readouterr()
        assert 'System configured' in captured.out


@pytest.mark.asyncio
async def test_configure_system_custom_parameters(mock_crs_for_configure):
    """Test configure_system with custom parameters."""
    crs = mock_crs_for_configure
    
    with patch.object(crs, 'set_clock_source',
                      new_callable=AsyncMock) as mock_set_clock, \
         patch.object(crs, 'set_extended_bw',
                      new_callable=AsyncMock) as mock_set_bw, \
         patch.object(crs, 'set_analog_bank',
                      new_callable=AsyncMock) as mock_set_bank, \
         patch.object(crs, 'set_decimation',
                      new_callable=AsyncMock) as mock_set_dec:
        
        await crs.configure_system(
            clock_source='SMA',
            full_scale_dbm=5,
            analog_bank_high=True,
            verbose=False
        )
        
        # Verify method calls with custom parameters
        mock_set_clock.assert_called_once_with('SMA', verbose=False)
        mock_set_bank.assert_called_once_with(True, 5)
        mock_set_dec.assert_called_once_with(
            6, short=False, module_idxs=None, verbose=False
        )


@pytest.mark.asyncio
async def test_configure_system_timestamp_port_not_set(
    mock_crs_for_configure
):
    """Test that timestamp port is set when just_booted is True."""
    crs = mock_crs_for_configure
    
    # Mock get_timestamp_port to return something other than 'TEST'
    crs.d.get_timestamp_port = AsyncMock(return_value='NOTTEST')
    
    with patch.object(crs, 'set_clock_source', new_callable=AsyncMock), \
         patch.object(crs, 'set_extended_bw', new_callable=AsyncMock), \
         patch.object(crs, 'set_analog_bank', new_callable=AsyncMock), \
         patch.object(crs, 'set_decimation', new_callable=AsyncMock):
        
        await crs.configure_system()
        
        # Verify timestamp port was set
        crs.d.set_timestamp_port.assert_called_once_with(
            crs.d.TIMESTAMP_PORT.TEST
        )


@pytest.mark.asyncio
async def test_configure_system_timestamp_port_already_set(
    mock_crs_for_configure
):
    """Test that timestamp port is not set when already 'TEST'."""
    crs = mock_crs_for_configure
    
    # get_timestamp_port already returns 'TEST' in fixture
    with patch.object(crs, 'set_clock_source',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_extended_bw',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_analog_bank',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_decimation',
                      new_callable=AsyncMock):
        
        await crs.configure_system()
        
        # Verify timestamp port was NOT set
        crs.d.set_timestamp_port.assert_not_called()


@pytest.mark.asyncio
async def test_configure_system_wrong_firmware_version(
    mock_crs_for_configure
):
    """Test that wrong firmware version raises RuntimeError."""
    crs = mock_crs_for_configure
    
    # Mock wrong firmware version
    firmware_release = MagicMock()
    firmware_release.version = '1.6.0rc2'  # Wrong version
    crs.d.get_firmware_release = AsyncMock(
        return_value=firmware_release
    )
    
    with pytest.raises(
        RuntimeError,
        match='CRS firmware must be version 1.6.0rc3'
    ):
        await crs.configure_system()


@pytest.mark.asyncio
async def test_configure_system_verbose_false_no_print(
    mock_crs_for_configure,
    capsys
):
    """Test that verbose=False suppresses output."""
    crs = mock_crs_for_configure
    
    with patch.object(crs, 'set_clock_source',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_extended_bw',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_analog_bank',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_decimation',
                      new_callable=AsyncMock):
        
        await crs.configure_system(verbose=False)
        
        # Verify no output
        captured = capsys.readouterr()
        assert 'System configured' not in captured.out


################################################################################
################## configure_system Input Validation Tests ####################
################################################################################

@pytest.mark.asyncio
async def test_configure_system_invalid_clock_source(
    mock_crs_for_configure
):
    """Test that invalid clock_source raises ValueError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(
        ValueError,
        match="clock_source must be 'VCXO' or 'SMA'"
    ):
        await crs.configure_system(clock_source='INVALID')


@pytest.mark.asyncio
async def test_configure_system_clock_source_not_string(
    mock_crs_for_configure
):
    """Test that non-string clock_source raises ValueError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(ValueError):
        await crs.configure_system(clock_source=123)


@pytest.mark.asyncio
async def test_configure_system_full_scale_dbm_not_number(
    mock_crs_for_configure
):
    """Test that non-numeric full_scale_dbm raises TypeError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(
        TypeError,
        match='full_scale_dbm must be a number'
    ):
        await crs.configure_system(full_scale_dbm='7')


@pytest.mark.asyncio
async def test_configure_system_full_scale_dbm_below_range(
    mock_crs_for_configure
):
    """Test that full_scale_dbm below 0 raises ValueError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(
        ValueError,
        match='full_scale_dbm must be in \\[0, 7\\]'
    ):
        await crs.configure_system(full_scale_dbm=-1)


@pytest.mark.asyncio
async def test_configure_system_full_scale_dbm_above_range(
    mock_crs_for_configure
):
    """Test that full_scale_dbm above 7 raises ValueError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(
        ValueError,
        match='full_scale_dbm must be in \\[0, 7\\]'
    ):
        await crs.configure_system(full_scale_dbm=8)


@pytest.mark.asyncio
async def test_configure_system_full_scale_dbm_float(
    mock_crs_for_configure
):
    """Test that float full_scale_dbm is accepted."""
    crs = mock_crs_for_configure
    
    with patch.object(crs, 'set_clock_source',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_extended_bw',
                      new_callable=AsyncMock), \
         patch.object(crs, 'set_analog_bank',
                      new_callable=AsyncMock) as mock_set_bank, \
         patch.object(crs, 'set_decimation',
                      new_callable=AsyncMock):
        
        await crs.configure_system(full_scale_dbm=5.5)
        
        # Verify it was passed correctly
        mock_set_bank.assert_called_once_with(False, 5.5)


@pytest.mark.asyncio
async def test_configure_system_analog_bank_high_not_bool(
    mock_crs_for_configure
):
    """Test that non-bool analog_bank_high raises TypeError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(
        TypeError,
        match='analog_bank_high must be a boolean value'
    ):
        await crs.configure_system(analog_bank_high=1)


@pytest.mark.asyncio
async def test_configure_system_verbose_not_bool(mock_crs_for_configure):
    """Test that non-bool verbose raises TypeError."""
    crs = mock_crs_for_configure
    
    with pytest.raises(
        TypeError,
        match='verbose must be a boolean value'
    ):
        await crs.configure_system(verbose=1)
