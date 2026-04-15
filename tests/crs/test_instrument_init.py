"""
Tests for CRS.__init__ method.

Tests CRS initialization, parameter validation, and device setup.
"""

import pytest
from unittest.mock import Mock, patch

from citkid.crs.instrument import CRS
from .conftest import RFMUX_VERSION


def test_crs_init_default_parameters(mock_rfmux_session):
    """Test CRS initialization with default parameters."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        mock_rfmux.load_session.return_value = mock_rfmux_session
        mock_rfmux.CRS = Mock()
        
        crs = CRS(serial_number=27, interface='enp2s0')
        
        # Check stored inputs
        assert crs.serial_number == 27
        assert crs.interface == 'enp2s0'
        
        # Check initialized attributes
        assert crs.nco_freqs == {}
        assert crs.bw == 500e6
        assert crs.fres_map == {}
        assert crs.ares_map == {}
        assert crs.ch_map == {}
        assert crs.dec_stage is None
        assert crs.dec_short is None
        assert crs.dec_module_idxs is None
        
        # Check device was initialized
        assert crs.d is not None
        
        # Verify session string was created correctly
        expected_str = '!HardwareMap [ !CRS { serial: "0027" } ]'
        mock_rfmux.load_session.assert_called_once_with(expected_str)


def test_crs_init_sets_versions(mock_rfmux_session):
    """Test CRS.__init__ stores rfmux and citkid version attributes."""
    import citkid

    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):

        mock_rfmux.__version__ = RFMUX_VERSION
        mock_rfmux.load_session.return_value = mock_rfmux_session
        mock_rfmux.CRS = Mock()

        crs = CRS(serial_number=27, interface='enp2s0')

        assert crs.rfmux_version == mock_rfmux.__version__
        assert crs.citkid_version == citkid.__version__


def test_crs_init_custom_parameters(mock_rfmux_session):
    """Test CRS initialization with custom parameters."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        mock_rfmux.load_session.return_value = mock_rfmux_session
        mock_rfmux.CRS = Mock()
        
        crs = CRS(serial_number = 42, interface = 'eth0')
        
        # Check stored inputs
        assert crs.serial_number == 42
        assert crs.interface == 'eth0'
        
        # Verify session string with custom serial number
        expected_str = '!HardwareMap [ !CRS { serial: "0042" } ]'
        mock_rfmux.load_session.assert_called_once_with(expected_str)


def test_crs_init_session_string_formatting(mock_rfmux_session):
    """
    Test that session string is formatted correctly for various
    serial numbers.
    """
    test_cases = [
        (1, '!HardwareMap [ !CRS { serial: "0001" } ]'),
        (99, '!HardwareMap [ !CRS { serial: "0099" } ]'),
        (1234, '!HardwareMap [ !CRS { serial: "1234" } ]'),
    ]
    
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        mock_rfmux.load_session.return_value = mock_rfmux_session
        mock_rfmux.CRS = Mock()
        
        for serial_number, expected_str in test_cases:
            mock_rfmux.load_session.reset_mock()
            crs = CRS(serial_number=serial_number, interface='enp2s0')
            mock_rfmux.load_session.assert_called_once_with(expected_str)


def test_crs_init_wrong_rfmux_version():
    """Test that wrong rfmux version raises RuntimeError."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = '0.0.0' # Wrong version
        
        with pytest.raises(
            RuntimeError,
            match = f'rfmux version {RFMUX_VERSION} is required'
        ):
            CRS(serial_number=27, interface='enp2s0')


def test_crs_init_serial_number_not_int():
    """Test that non-integer serial_number raises TypeError."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        
        with pytest.raises(
            TypeError,
            match = 'serial_number must be an integer'
        ):
            CRS(serial_number='27', interface='enp2s0')


def test_crs_init_serial_number_float():
    """Test that float serial_number raises TypeError."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        
        with pytest.raises(
            TypeError,
            match = 'serial_number must be an integer'
        ):
            CRS(serial_number=27.0, interface='enp2s0')


def test_crs_init_serial_number_none():
    """Test that None serial_number raises TypeError."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        
        with pytest.raises(
            TypeError,
            match = 'serial_number must be an integer'
        ):
            CRS(serial_number=None, interface='enp2s0')


def test_crs_init_interface_does_not_exist():
    """Test that non-existent interface raises ValueError."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = False):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        
        with pytest.raises(
            ValueError,
            match = 'interface fake_interface does not exist'
        ):
            CRS(serial_number=27, interface='fake_interface')


def test_crs_init_device_query_chain(mock_rfmux_session,
                                      mock_rfmux_device):
    """Test that the device query chain is called correctly."""
    with patch('citkid.crs.instrument.rfmux') as mock_rfmux, \
         patch('citkid.crs.instrument.util.interface_exists',
               return_value = True):
        
        mock_rfmux.__version__ = RFMUX_VERSION
        mock_rfmux.load_session.return_value = mock_rfmux_session
        mock_rfmux.CRS = Mock()
        
        crs = CRS(serial_number=27, interface='enp2s0')
        
        # Verify the query chain
        mock_rfmux_session.query.assert_called_once_with(
            mock_rfmux.CRS
        )
        query_result = mock_rfmux_session.query.return_value
        query_result.one.assert_called_once()
        
        # Verify device assignment
        assert crs.d == mock_rfmux_device