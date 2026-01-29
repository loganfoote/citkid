"""
Tests for sweep methods.

Tests CRS.sweep and related sweep wrapper methods. The _sweep macro is tested
indirectly through CRS.sweep, which provides the proper interface for calling
the macro on ReadoutModule objects.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call
import warnings


################################################################################
####################### CRS.sweep tests ########################################
################################################################################

@pytest.mark.asyncio
async def test_sweep_basic(base_crs):
    """Test basic sweep functionality."""
    crs = base_crs
    
    # Set NCO frequencies (required for sweep)
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    # Setup: 2 channels, 3 sweep points each
    frequencies = np.array([
        [3.9e9, 4.0e9, 4.1e9],
        [4.05e9, 4.1e9, 4.15e9]
    ])
    ares = np.array([-50.0, -51.0])
    nsamps = 10
    
    # Mock the necessary methods
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    # Mock get_modules to return a mock with _sweep
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
            # Assume all channels map to module 1
            mock_create_ch_map.return_value = ({1: [0, 1]}, [])
            
            # Mock the _sweep to populate output dicts
            async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
                # Populate sweep_f and sweep_z as _sweep would
                for mod_idx, freqs in fres_map.items():
                    sweep_f[mod_idx] = freqs
                    sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex) * (1.0 + 1.0j)
            
            mock_modules._sweep.side_effect = mock_sweep_impl
            
            f, z = await crs.sweep(frequencies, ares, nsamps, verbose=False)
            
            # Verify clear_channels was called
            crs._clear_channels.assert_called_once()
            
            # Verify decimation was set to 6
            crs.set_decimation.assert_called_once_with(6, verbose=False)
            
            # Verify _sweep was called
            mock_modules._sweep.assert_called_once()
            
            # Verify output shapes
            assert f.shape == frequencies.shape
            assert z.shape == frequencies.shape
            
            # Verify data is not all NaN
            assert not np.all(np.isnan(f))
            assert not np.all(np.isnan(z))


@pytest.mark.asyncio
async def test_sweep_with_ch_map(base_crs):
    """Test sweep with provided ch_map."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    ch_map = {1: [0]}
    nsamps = 10
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        f, z = await crs.sweep(frequencies, ares, nsamps, ch_map=ch_map, verbose=False)
        
        # Verify ch_map was used (not created)
        assert f.shape == frequencies.shape
        assert z.shape == frequencies.shape


@pytest.mark.asyncio
async def test_sweep_allow_missing_true(base_crs):
    """Test sweep with allow_missing=True inserts NaNs for missing channels."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([
        [3.9e9, 4.0e9],
        [4.05e9, 4.1e9]
    ])
    ares = np.array([-50.0, -51.0])
    nsamps = 10
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    # Mock create_ch_map to return missing channels
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [1])  # Channel 1 is missing
        
        async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                f, z = await crs.sweep(frequencies, ares, nsamps, 
                                      allow_missing=True, verbose=False)
                
                # Verify warning was issued
                assert len(w) == 1
                assert "missing channel" in str(w[0].message).lower()
            
            # Verify channel 1 has NaN values
            assert np.all(np.isnan(f[1, :]))
            assert np.all(np.isnan(z[1, :]))
            
            # Verify channel 0 has data
            assert not np.all(np.isnan(f[0, :]))


@pytest.mark.asyncio
async def test_sweep_allow_missing_false_raises(base_crs):
    """Test sweep with allow_missing=False raises error for missing channels."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9], [4.05e9, 4.1e9]])
    ares = np.array([-50.0, -51.0])
    nsamps = 10
    
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [1])  # Channel 1 is missing
        
        with pytest.raises(ValueError, match="Tones must be within"):
            await crs.sweep(frequencies, ares, nsamps, allow_missing=False, verbose=False)


@pytest.mark.asyncio
async def test_sweep_clears_channels_before_sweep(base_crs):
    """Test that sweep clears existing channels before starting."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    
    # Set up existing fres_map
    crs.fres_map = {1: np.array([[3.8e9]]), 2: np.array([[4.2e9]])}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
            await crs.sweep(frequencies, ares, nsamps, verbose=False)
            
            # Verify clear_channels was called with modules from fres_map
            crs._clear_channels.assert_called_once_with([1, 2])


@pytest.mark.asyncio
async def test_sweep_updates_fres_ares_maps(base_crs):
    """Test that sweep updates fres_map and ares_map."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    ch_map = {1: [0]}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    
    # Capture the fres_map and ares_map passed to _sweep
    captured_fres_map = {}
    captured_ares_map = {}
    
    async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
        captured_fres_map.update(fres_map)
        captured_ares_map.update(ares_map)
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        await crs.sweep(frequencies, ares, nsamps, ch_map=ch_map, verbose=False)
        
        # Verify fres_map and ares_map were updated
        assert 1 in captured_fres_map
        assert captured_fres_map[1].shape == (1, 2)  # 1 channel, 2 points
        assert 1 in captured_ares_map
        np.testing.assert_array_equal(captured_ares_map[1], ares[[0]])


@pytest.mark.asyncio
async def test_sweep_dithers_frequencies(base_crs):
    """Test that sweep dithers frequencies using _safe_concatenate_frequencies."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    ch_map = {1: [0]}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    # Capture dithered frequencies
    captured_fres_map = {}
    
    async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
        captured_fres_map.update({k: v.copy() for k, v in fres_map.items()})
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        with patch('citkid.crs.instrument.take_netanal._safe_concatenate_frequencies') as mock_dither:
            # Mock dithering to add small offset
            def dither_impl(freq, nco_freq):
                return freq + np.random.uniform(-1, 1, freq.shape)
            mock_dither.side_effect = dither_impl
            
            f, z = await crs.sweep(frequencies, ares, nsamps, ch_map=ch_map, verbose=False)
            
            # Verify _safe_concatenate_frequencies was called for each sweep point
            assert mock_dither.call_count >= 2  # At least once per sweep point
            
            # Verify dithered frequencies are within tolerance of input
            assert np.allclose(f, frequencies, atol=1.0)  # 1 Hz tolerance


@pytest.mark.asyncio
async def test_sweep_sets_decimation_to_6(base_crs):
    """Test that sweep sets decimation stage to 6."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
            await crs.sweep(frequencies, ares, nsamps, verbose=False)
            
            # Verify decimation set to 6
            crs.set_decimation.assert_called_once_with(6, verbose=False)


@pytest.mark.asyncio
async def test_sweep_clears_fres_ares_maps_after(base_crs):
    """Test that sweep clears fres_map and ares_map after completion."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    ch_map = {1: [0]}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    
    async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        await crs.sweep(frequencies, ares, nsamps, ch_map=ch_map, verbose=False)
        
        # Verify fres_map and ares_map are cleared for module 1
        assert 1 in crs.fres_map
        assert crs.fres_map[1].size == 0
        assert 1 in crs.ares_map
        assert crs.ares_map[1].size == 0


@pytest.mark.asyncio
async def test_sweep_concatenates_results_via_ch_map(base_crs):
    """Test that sweep concatenates results using ch_map."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    # 4 channels split across 2 modules
    frequencies = np.array([
        [3.9e9, 4.0e9],
        [3.95e9, 4.05e9],
        [4.45e9, 4.5e9],
        [4.5e9, 4.55e9]
    ])
    ares = np.array([-50.0, -51.0, -52.0, -53.0])
    nsamps = 10
    ch_map = {1: [0, 1], 2: [2, 3]}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    
    async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
        # Simulate _sweep returning data for each module
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            # Different values per module for testing
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex) * mod_idx
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        f, z = await crs.sweep(frequencies, ares, nsamps, ch_map=ch_map, verbose=False)
        
        # Verify results are concatenated correctly
        assert f.shape == frequencies.shape
        assert z.shape == frequencies.shape
        
        # The sweep method divides by 10**(ares/20), so we need to account for that
        # Expected: original_value / 10**(ares/20)
        # Module 1 (rows 0,1): 1.0 / 10**(-50/20) = 1.0 / 10**(-2.5) = 1.0 * 10**(2.5) = 316.227...
        # Module 2 (rows 2,3): 2.0 / 10**(-52/20) = 2.0 * 10**(2.6) = 795.775...
        expected_0 = 1.0 / 10**(-50.0/20)
        expected_1 = 1.0 / 10**(-51.0/20)
        expected_2 = 2.0 / 10**(-52.0/20)
        expected_3 = 2.0 / 10**(-53.0/20)
        
        assert np.allclose(z[0, :].real, expected_0)
        assert np.allclose(z[1, :].real, expected_1)
        assert np.allclose(z[2, :].real, expected_2)
        assert np.allclose(z[3, :].real, expected_3)


@pytest.mark.asyncio
async def test_sweep_converts_to_dbc(base_crs):
    """Test that sweep converts output to dBc using ares."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    ch_map = {1: [0]}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    
    async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            # Return known amplitude
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex) * 10.0
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
        f, z = await crs.sweep(frequencies, ares, nsamps, ch_map=ch_map, verbose=False)
        
        # Verify conversion to dBc
        # z should be divided by 10 ** (ares / 20)
        # For ares = -50.0: 10 ** (-50 / 20) = 10 ** -2.5 ≈ 0.003162
        # So z = 10.0 / 0.003162 ≈ 3162.3
        expected = 10.0 / (10 ** (-50.0 / 20))
        assert np.allclose(np.abs(z), expected, rtol=1e-5)


@pytest.mark.asyncio
async def test_sweep_validation_frequencies_ares_mismatch(base_crs):
    """Test sweep raises error when frequencies and ares shapes don't match."""
    crs = base_crs
    
    frequencies = np.array([[3.9e9, 4.0e9], [4.05e9, 4.1e9]])
    ares = np.array([-50.0])  # Only 1 element, but frequencies has 2 channels
    
    with pytest.raises(ValueError, match="must have the same length"):
        await crs.sweep(frequencies, ares, 10, verbose=False)


@pytest.mark.asyncio
async def test_sweep_validation_frequencies_not_2d(base_crs):
    """Test sweep raises error when frequencies is not 2D."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([3.9e9, 4.0e9])  # 1D array
    ares = np.array([-50.0, -51.0])  # Match the length

@pytest.mark.asyncio
async def test_sweep_validation_nsamps_not_positive(base_crs):
    """Test sweep raises error when nsamps is not positive."""
    crs = base_crs
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    
    with pytest.raises(ValueError, match="must be a positive integer"):
        await crs.sweep(frequencies, ares, 0, verbose=False)


@pytest.mark.asyncio
async def test_sweep_validation_nco_not_set(base_crs):
    """Test sweep raises error when NCO frequencies are not set."""
    crs = base_crs
    crs.nco_freqs = {}  # Empty NCO freqs
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    
    with pytest.raises(RuntimeError, match="NCO frequencies are not set"):
        await crs.sweep(frequencies, ares, 10, verbose=False)


@pytest.mark.asyncio
async def test_sweep_passes_nsamps_to_macro(base_crs):
    """Test that sweep passes nsamps parameter to _sweep macro."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 25
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
            assert kwargs['nsamps'] == nsamps
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
            await crs.sweep(frequencies, ares, nsamps, verbose=False)


@pytest.mark.asyncio
async def test_sweep_passes_verbose_and_description(base_crs):
    """Test that sweep passes verbose and pbar_description to _sweep."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    with patch('citkid.crs.instrument.util.create_ch_map') as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs):
            assert kwargs['verbose'] == True
            assert kwargs['pbar_description'] == "Custom Sweep"
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype=complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch('citkid.crs.instrument.util.get_modules', return_value=mock_modules):
            await crs.sweep(frequencies, ares, nsamps, verbose=True,
                          pbar_description="Custom Sweep")


################################################################################
####################### Placeholder for other sweep methods ####################
################################################################################

def test_crs_sweep_placeholder():
    """Placeholder for sweep tests."""
    pass


def test_crs_sweep_linear_placeholder():
    """Placeholder for sweep_linear tests."""
    pass


def test_crs_sweep_qres_placeholder():
    """Placeholder for sweep_qres tests."""
    pass


def test_crs_sweep_full_placeholder():
    """Placeholder for sweep_full tests."""
    pass


def test_sweep_macro_placeholder():
    """Placeholder for _sweep macro tests."""
    pass
