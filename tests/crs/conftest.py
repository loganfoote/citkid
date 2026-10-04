"""
Shared fixtures for CRS instrument tests.

This module provides common fixtures used across multiple test files to avoid
duplication and ensure consistency in test setup.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from citkid.crs.instrument import CRS

# Centralized rfmux version used by tests
RFMUX_VERSION = '1.4.1'


@pytest.fixture
def mock_rfmux_base():
    """
    Base rfmux mock with common setup.
    
    Provides patched rfmux module and a mock device with standard configuration.
    Most CRS fixtures should build on this base fixture.
    
    Yields:
        tuple: (mock_rfmux, mock_device) - Patched rfmux module and device
    """
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        mock_rfmux.CRS = Mock()
        
        # Session setup
        mock_session = MagicMock()
        mock_device = MagicMock()
        query_result = MagicMock()
        query_result.one.return_value = mock_device
        mock_session.query.return_value = query_result
        mock_rfmux.load_session.return_value = mock_session
        
        yield mock_rfmux, mock_device


@pytest.fixture
def base_crs(mock_rfmux_base):
    """
    Create a basic CRS instance with mocked rfmux.
    
    This provides a minimal CRS instance for tests that don't need specific
    device method mocking. Build on this for more complex test scenarios.
    
    Returns:
        CRS: CRS instance with mocked device
    """
    mock_rfmux, mock_device = mock_rfmux_base
    crs = CRS(serial_number=1, interface='eth0')
    crs.d = mock_device
    # Module settings normally set by configure_system/attach
    crs.module_cfg = {mi: default_module_cfg() for mi in range(1, 9)}
    return crs


def mock_module_settings(device, full_scale_dbm = 7):
    """
    Mock the device setters and getters used by ``CRS.configure_modules``.
    Getters return the ``configure_modules`` defaults.

    Parameters:
    device (MagicMock): mock rfmux CRS device.
    full_scale_dbm (float): value returned by ``get_dac_scale``.

    Returns:
    None
    """
    device.UNITS = MagicMock()
    device.UNITS.DBM = 'DBM'
    for name in ['set_dac_scale', 'set_adc_attenuator', 'set_nyquist_zone',
                 'set_cable_length', 'set_adc_calibration_mode',
                 'set_adc_autocal']:
        setattr(device, name, AsyncMock())
    device.get_dac_scale = AsyncMock(return_value = full_scale_dbm)
    device.get_adc_attenuator = AsyncMock(return_value = 0.0)
    device.get_nyquist_zone = AsyncMock(return_value = 1)
    device.get_cable_length = AsyncMock(return_value = 0.0)
    device.get_adc_calibration_mode = AsyncMock(return_value = 'AUTO')
    device.get_adc_autocal = AsyncMock(return_value = True)


def default_module_cfg(full_scale_dbm = 7.0):
    """
    Return the module settings stored by ``CRS.configure_modules`` with its
    default arguments.

    Parameters:
    full_scale_dbm (float): DAC full scale in dBm.

    Returns:
    dict: module settings.
    """
    return {
        'full_scale_dbm': float(full_scale_dbm),
        'adc_attenuation_db': 0.0,
        'nyquist_zone': 1,
        'cable_length_m': 0.0,
        'adc_calibration_mode': 'AUTO',
        'adc_autocal': True,
    }


@pytest.fixture
def mock_rfmux_device():
    """
    Create a standalone mock rfmux device object.
    
    This fixture provides a base mock device that can be extended in tests
    without full CRS initialization.
    
    Returns:
        MagicMock: Mock device object
    """
    device = MagicMock()
    return device


@pytest.fixture
def mock_rfmux_session(mock_rfmux_device):
    """
    Create a mock rfmux session that returns our mock device.
    
    Returns:
        MagicMock: Mock session with query result pointing to device
    """
    session = MagicMock()
    query_result = MagicMock()
    query_result.one.return_value = mock_rfmux_device
    session.query.return_value = query_result
    return session