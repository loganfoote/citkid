"""Shared fixtures for VNA resonance finding tests."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, Mock
import os

# Prevent Qt windows from opening during tests
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@pytest.fixture(autouse=True)
def mock_qt_ui():
    """
    Automatically mock Qt UI setup for all VNA tests to prevent windows from opening.
    This fixture is autouse=True so it applies to all tests in this directory.
    
    Mocks the UI creation and provides a default mock for update_peaks.
    Tests can override update_peaks behavior with their own patches if needed.
    """
    # Create a mock that does nothing for update_peaks by default
    mock_update = Mock(return_value=None)
    
    with patch('citkid.vna.res_finder_auto.AutoResFinder.setup_ui'), \
         patch('citkid.vna.res_finder_auto.AutoResFinder.setup_plot'), \
         patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks', mock_update), \
         patch('citkid.vna.res_finder_manual.ResFinder.setup_ui'), \
         patch('citkid.vna.res_finder_manual.ResFinder.run'):
        yield


@pytest.fixture
def synthetic_vna_data():
    """
    Generate synthetic VNA sweep data with known resonances.
    
    Returns a dictionary with:
    - f: frequency array (Hz)
    - z: complex S21 data
    - fres_true: true resonance frequencies
    - Q: quality factors
    """
    # Frequency sweep from 4 to 8 GHz
    f = np.linspace(4e9, 8e9, 100000)
    
    # True resonance parameters
    fres_true = np.array([4.5e9, 5.2e9, 6.1e9, 7.3e9])
    Q = 30000
    
    # Generate complex S21 with Lorentzian dips
    z = np.ones(len(f), dtype=complex)
    
    for fr in fres_true:
        # Lorentzian resonance model
        delta_f = f - fr
        z *= 1 / (1 + 2j * Q * delta_f / fr)
    
    # Add small noise
    noise_level = 0.001
    z += noise_level * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
    
    # Add slow baseline variation (polynomial)
    baseline = 1 + 0.1 * (f - f.mean()) / (f.max() - f.min())
    z *= baseline
    
    return {
        'f': f,
        'z': z,
        'fres_true': fres_true,
        'Q': Q
    }


@pytest.fixture
def sparse_vna_data():
    """
    Generate VNA data with few points (for edge case testing).
    """
    f = np.linspace(4e9, 8e9, 100)
    z = np.ones(len(f), dtype=complex) * 0.9
    
    # Single resonance
    fres_true = np.array([6e9])
    Q = 10000
    
    for fr in fres_true:
        delta_f = f - fr
        z *= 1 / (1 + 2j * Q * delta_f / fr)
    
    return {
        'f': f,
        'z': z,
        'fres_true': fres_true,
        'Q': Q
    }


@pytest.fixture
def no_resonance_vna_data():
    """
    Generate VNA data with no resonances (flat response).
    """
    f = np.linspace(4e9, 8e9, 10000)
    z = np.ones(len(f), dtype=complex) * 0.95
    
    # Add small noise
    noise_level = 0.001
    z += noise_level * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
    
    return {
        'f': f,
        'z': z,
        'fres_true': np.array([]),
        'Q': None
    }


@pytest.fixture
def dense_resonances_vna_data():
    """
    Generate VNA data with closely spaced resonances.
    """
    f = np.linspace(5e9, 6e9, 50000)
    
    # Closely spaced resonances (50 MHz apart)
    fres_true = np.array([5.1e9, 5.15e9, 5.2e9, 5.25e9, 5.3e9])
    Q = 25000
    
    z = np.ones(len(f), dtype=complex)
    
    for fr in fres_true:
        delta_f = f - fr
        z *= 1 / (1 + 2j * Q * delta_f / fr)
    
    # Add small noise
    noise_level = 0.0005
    z += noise_level * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
    
    return {
        'f': f,
        'z': z,
        'fres_true': fres_true,
        'Q': Q
    }
