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
    crs = CRS()
    crs.d = mock_device
    return crs


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