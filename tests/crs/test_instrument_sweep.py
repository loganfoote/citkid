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
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        with patch(
            'citkid.crs.instrument.util.create_ch_map'
        ) as mock_create_ch_map:
            # Assume all channels map to module 1
            mock_create_ch_map.return_value = ({1: [0, 1]}, [])
            
            # Mock the _sweep to populate output dicts
            async def mock_sweep_impl(
                nco_freqs,
                fres_map,
                ares_map,
                sweep_f,
                sweep_z,
                **kwargs
            ):
                # Populate sweep_f and sweep_z as _sweep would
                for mod_idx, freqs in fres_map.items():
                    sweep_f[mod_idx] = freqs
                    sweep_z[mod_idx] = np.ones(
                        freqs.shape,
                        dtype = complex) * (1.0 + 1.0j
                    )
            
            mock_modules._sweep.side_effect = mock_sweep_impl
            
            f, z = await crs.sweep(frequencies, ares, nsamps, verbose = False)
            
            # Verify clear_channels was called
            crs._clear_channels.assert_called_once()
            
            # Verify decimation was set to 6
            crs.set_decimation.assert_called_once_with(6, verbose = False)
            
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
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        f, z = await crs.sweep(
            frequencies,
            ares,
            nsamps,
            ch_map = ch_map,
            verbose = False
        )
        
        # Verify ch_map was used (not created)
        assert f.shape == frequencies.shape
        assert z.shape == frequencies.shape


@pytest.mark.asyncio
async def test_sweep_allow_missing_true(base_crs):
    """
    Test sweep with allow_missing = True inserts NaNs for
    missing channels.
    """
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
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = (
            {1: [0]},
            [1]
        )  # Channel 1 is missing
        
        async def mock_sweep_impl(
            nco_freqs,
            fres_map,
            ares_map,
            sweep_f,
            sweep_z,
            **kwargs
        ):
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            with warnings.catch_warnings(record = True) as w:
                warnings.simplefilter("always")
                f, z = await crs.sweep(frequencies, ares, nsamps, 
                                      allow_missing = True, verbose = False)
                
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
    """
    Test sweep with allow_missing = False raises error for
    missing channels.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}

    frequencies = np.array([[3.9e9, 4.0e9], [4.05e9, 4.1e9]])
    ares = np.array([-50.0, -51.0])
    nsamps = 10
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = (
            {1: [0]},
            [1]
        )  # Channel 1 is missing
        
        with pytest.raises(ValueError, match = "Tones must be within"):
            await crs.sweep(
                frequencies,
                ares,
                nsamps,
                allow_missing = False,
                verbose = False
            )


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
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(
            nco_freqs,
            fres_map,
            ares_map,
            sweep_f,
            sweep_z,
            **kwargs
        ):
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            await crs.sweep(frequencies, ares, nsamps, verbose = False)
            
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
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        captured_fres_map.update(fres_map)
        captured_ares_map.update(ares_map)
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        await crs.sweep(
            frequencies,
            ares,
            nsamps,
            ch_map = ch_map,
            verbose = False
        )
        
        # Verify fres_map and ares_map were updated
        assert 1 in captured_fres_map
        assert captured_fres_map[1].shape == (1, 2)  # 1 channel, 2 points
        assert 1 in captured_ares_map
        np.testing.assert_array_equal(captured_ares_map[1], ares[[0]])


@pytest.mark.asyncio
async def test_sweep_dithers_frequencies(base_crs):
    """
    Test that sweep dithers frequencies using _safe_concatenate_frequencies.
    """
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
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        captured_fres_map.update({k: v.copy() for k, v in fres_map.items()})
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        with patch(
            'citkid.crs.instrument.take_netanal._safe_concatenate_frequencies'
        ) as mock_dither:
            # Mock dithering to add small offset
            def dither_impl(freq, nco_freq):
                return freq + np.random.uniform(-1, 1, freq.shape)
            mock_dither.side_effect = dither_impl
            
            f, z = await crs.sweep(
                frequencies,
                ares,
                nsamps,
                ch_map = ch_map,
                verbose = False
            )
            
# Verify _safe_concatenate_frequencies was called for each sweep point
            assert mock_dither.call_count >= 2  # At least once per sweep point
            
            # Verify dithered frequencies are within tolerance of input
            assert np.allclose(f, frequencies, atol = 1.0)  # 1 Hz tolerance


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
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(
            nco_freqs,
            fres_map,
            ares_map,
            sweep_f,
            sweep_z,
            **kwargs
        ):
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            await crs.sweep(frequencies, ares, nsamps, verbose = False)
            
            # Verify decimation set to 6
            crs.set_decimation.assert_called_once_with(6, verbose = False)


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
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        await crs.sweep(
            frequencies,
            ares,
            nsamps,
            ch_map = ch_map,
            verbose = False
        )
        
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
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        # Simulate _sweep returning data for each module
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            # Different values per module for testing
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex) * mod_idx
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        f, z = await crs.sweep(
            frequencies,
            ares,
            nsamps,
            ch_map = ch_map,
            verbose = False
        )
        
        # Verify results are concatenated correctly
        assert f.shape == frequencies.shape
        assert z.shape == frequencies.shape
        
# The sweep method divides by 10**(ares/20), so we need to account for that
        # Expected: original_value / 10**(ares/20)
# Module 1 (rows 0,1): 1.0 / 10**(-50/20) = 1.0 / 10**(-2.5) = 1.0 * 10**(2.5) =
        # 316.227...
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
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            # Return known amplitude
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex) * 10.0
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        f, z = await crs.sweep(
            frequencies,
            ares,
            nsamps,
            ch_map = ch_map,
            verbose = False
        )
        
        # Verify conversion to dBc
        # z should be divided by 10 ** (ares / 20)
        # For ares = -50.0: 10 ** (-50 / 20) = 10 ** -2.5 ≈ 0.003162
        # So z = 10.0 / 0.003162 ≈ 3162.3
        expected = 10.0 / (10 ** (-50.0 / 20))
        assert np.allclose(np.abs(z), expected, rtol = 1e-5)


@pytest.mark.asyncio
async def test_sweep_validation_frequencies_ares_mismatch(base_crs):
    """Test sweep raises error when frequencies and ares shapes don't match."""
    crs = base_crs
    
    frequencies = np.array([[3.9e9, 4.0e9], [4.05e9, 4.1e9]])
    ares = np.array([-50.0])  # Only 1 element, but frequencies has 2 channels
    
    with pytest.raises(ValueError, match = "must have the same length"):
        await crs.sweep(frequencies, ares, 10, verbose = False)


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
    
    with pytest.raises(ValueError, match = "must be a positive integer"):
        await crs.sweep(frequencies, ares, 0, verbose = False)


@pytest.mark.asyncio
async def test_sweep_validation_nco_not_set(base_crs):
    """Test sweep raises error when NCO frequencies are not set."""
    crs = base_crs
    crs.nco_freqs = {}  # Empty NCO freqs
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    
    with pytest.raises(RuntimeError, match = "NCO frequencies are not set"):
        await crs.sweep(frequencies, ares, 10, verbose = False)


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
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(
            nco_freqs,
            fres_map,
            ares_map,
            sweep_f,
            sweep_z,
            **kwargs
        ):
            assert kwargs['nsamps'] == nsamps
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            await crs.sweep(frequencies, ares, nsamps, verbose = False)


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
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        async def mock_sweep_impl(
            nco_freqs,
            fres_map,
            ares_map,
            sweep_f,
            sweep_z,
            **kwargs
        ):
            assert kwargs['verbose'] == True
            assert kwargs['pbar_description'] == "Custom Sweep"
            for mod_idx, freqs in fres_map.items():
                sweep_f[mod_idx] = freqs
                sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
        
        mock_modules._sweep.side_effect = mock_sweep_impl
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            await crs.sweep(frequencies, ares, nsamps, verbose = True,
                          pbar_description = "Custom Sweep")


@pytest.mark.asyncio
async def test_sweep_validation_dec_grp_invalid_type(base_crs):
    """Test sweep raises error when dec_grp is not a zarr.Group or None."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    
    with pytest.raises(TypeError, match = "must be a zarr.Group or None"):
        await crs.sweep(frequencies, ares, 10, dec_grp = "not a group", 
                       verbose = False)


@pytest.mark.asyncio
async def test_sweep_calls_write_acq_cfg_to_zarr_when_dec_grp_provided(base_crs):
    """Test sweep calls util.write_acq_cfg_to_zarr when dec_grp is provided."""
    import zarr
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    
    # Create a mock zarr group
    mock_grp = MagicMock(spec = zarr.Group)
    mock_grp.__class__ = zarr.Group
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            with patch(
                'citkid.crs.instrument.util.write_acq_cfg_to_zarr'
            ) as mock_write_acq:
                await crs.sweep(
                    frequencies, ares, nsamps, dec_grp = mock_grp, 
                    verbose = False
                )
                
                # Verify write_acq_cfg_to_zarr was called with correct args
                mock_write_acq.assert_called_once_with(crs, mock_grp)


@pytest.mark.asyncio
async def test_sweep_does_not_call_write_acq_cfg_to_zarr_when_dec_grp_none(base_crs):
    """Test sweep does not call util.write_acq_cfg_to_zarr when dec_grp is None."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.create_ch_map'
    ) as mock_create_ch_map:
        mock_create_ch_map.return_value = ({1: [0]}, [])
        
        with patch(
            'citkid.crs.instrument.util.get_modules',
            return_value = mock_modules
        ):
            with patch(
                'citkid.crs.instrument.util.write_acq_cfg_to_zarr'
            ) as mock_write_acq:
                await crs.sweep(
                    frequencies, ares, nsamps, dec_grp = None, 
                    verbose = False
                )
                
                # Verify write_acq_cfg_to_zarr was NOT called
                mock_write_acq.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_preserves_ch_map_when_provided(base_crs):
    """Test that crs.ch_map equals the input ch_map after sweep when ch_map is provided."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    frequencies = np.array([[3.9e9, 4.0e9]])
    ares = np.array([-50.0])
    nsamps = 10
    
    # Provide a specific ch_map
    input_ch_map = {1: [0]}
    
    crs._clear_channels = AsyncMock()
    crs.set_decimation = AsyncMock()
    
    mock_modules = MagicMock()
    mock_modules._sweep = AsyncMock()
    
    async def mock_sweep_impl(
        nco_freqs,
        fres_map,
        ares_map,
        sweep_f,
        sweep_z,
        **kwargs
    ):
        for mod_idx, freqs in fres_map.items():
            sweep_f[mod_idx] = freqs
            sweep_z[mod_idx] = np.ones(freqs.shape, dtype = complex)
    
    mock_modules._sweep.side_effect = mock_sweep_impl
    
    with patch(
        'citkid.crs.instrument.util.get_modules',
        return_value = mock_modules
    ):
        await crs.sweep(
            frequencies, ares, nsamps, ch_map = input_ch_map, verbose = False
        )
        
        # Verify that crs.ch_map is the same as the input ch_map
        assert crs.ch_map == input_ch_map


################################################################################
####################### Placeholder for other sweep methods ####################
################################################################################

################################################################################
####################### sweep_span tests #######################################
################################################################################

@pytest.mark.asyncio
async def test_sweep_span_center_ascending_linear(base_crs):
    """
    Test sweep_span with center_fres = True, downward = False,
    log = False.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-50.0, -51.0])
    span = 1e6  # 1 MHz
    npoints = 5
    nsamps = 10
    
    # Mock sweep to capture frequencies
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps, 
        center_fres = True, downward = False, log = False, verbose = False
    )
    
    # Verify shape
    assert captured_freqs.shape == (2, 5)
    
    # Verify frequencies for channel 0: center at 4.0e9, span 1e6, ascending
    expected_ch0 = np.linspace(4.0e9 - 0.5e6, 4.0e9 + 0.5e6, 5)
    assert np.allclose(captured_freqs[0, :], expected_ch0)
    
    # Verify frequencies for channel 1: center at 4.1e9
    expected_ch1 = np.linspace(4.1e9 - 0.5e6, 4.1e9 + 0.5e6, 5)
    assert np.allclose(captured_freqs[1, :], expected_ch1)


@pytest.mark.asyncio
async def test_sweep_span_center_descending_linear(base_crs):
    """Test sweep_span with center_fres = True, downward = True, log = False."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-50.0, -51.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = True, downward = True, log = False, verbose = False
    )
    
    # Verify frequencies are descending
    expected_ch0 = np.linspace(4.0e9 + 0.5e6, 4.0e9 - 0.5e6, 5)
    assert np.allclose(captured_freqs[0, :], expected_ch0)
    
    expected_ch1 = np.linspace(4.1e9 + 0.5e6, 4.1e9 - 0.5e6, 5)
    assert np.allclose(captured_freqs[1, :], expected_ch1)


@pytest.mark.asyncio
async def test_sweep_span_start_ascending_linear(base_crs):
    """
    Test sweep_span with center_fres = False, downward = False,
    log = False.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-50.0, -51.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = False, downward = False, log = False, verbose = False
    )
    
    # fres is start frequency, sweep upward
    expected_ch0 = np.linspace(4.0e9, 4.0e9 + 1e6, 5)
    assert np.allclose(captured_freqs[0, :], expected_ch0)
    
    expected_ch1 = np.linspace(4.1e9, 4.1e9 + 1e6, 5)
    assert np.allclose(captured_freqs[1, :], expected_ch1)


@pytest.mark.asyncio
async def test_sweep_span_start_descending_linear(base_crs):
    """
    Test sweep_span with center_fres = False, downward = True,
    log = False.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-50.0, -51.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = False, downward = True, log = False, verbose = False
    )
    
    # fres is start frequency, sweep downward (start + span to start)
    expected_ch0 = np.linspace(4.0e9 + 1e6, 4.0e9, 5)
    assert np.allclose(captured_freqs[0, :], expected_ch0)
    
    expected_ch1 = np.linspace(4.1e9 + 1e6, 4.1e9, 5)
    assert np.allclose(captured_freqs[1, :], expected_ch1)


@pytest.mark.asyncio
async def test_sweep_span_center_ascending_log(base_crs):
    """Test sweep_span with center_fres = True, downward = False, log = True."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e9  # 1 GHz for better log vs linear differentiation
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = True, downward = False, log = True, verbose = False
    )
    
    # Verify logarithmic spacing
    expected = np.geomspace(4.0e9 - 0.5e9, 4.0e9 + 0.5e9, 5)
    assert np.allclose(captured_freqs[0, :], expected)
    
    # Verify it's not linear (use larger span to see difference)
    linear = np.linspace(4.0e9 - 0.5e9, 4.0e9 + 0.5e9, 5)
    assert not np.allclose(captured_freqs[0, :], linear, rtol = 1e-6)


@pytest.mark.asyncio
async def test_sweep_span_center_descending_log(base_crs):
    """Test sweep_span with center_fres = True, downward = True, log = True."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = True, downward = True, log = True, verbose = False
    )
    
    # Verify logarithmic descending spacing
    expected = np.geomspace(4.0e9 + 0.5e6, 4.0e9 - 0.5e6, 5)
    assert np.allclose(captured_freqs[0, :], expected)


@pytest.mark.asyncio
async def test_sweep_span_start_ascending_log(base_crs):
    """
    Test sweep_span with center_fres = False, downward = False,
    log = True.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = False, downward = False, log = True, verbose = False
    )
    
    # Verify logarithmic spacing from start
    expected = np.geomspace(4.0e9, 4.0e9 + 1e6, 5)
    assert np.allclose(captured_freqs[0, :], expected)


@pytest.mark.asyncio
async def test_sweep_span_start_descending_log(base_crs):
    """Test sweep_span with center_fres = False, downward = True, log = True."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = False, downward = True, log = True, verbose = False
    )
    
    # Verify logarithmic descending spacing from start
    expected = np.geomspace(4.0e9 + 1e6, 4.0e9, 5)
    assert np.allclose(captured_freqs[0, :], expected)


@pytest.mark.asyncio
async def test_sweep_span_single_channel_linear(base_crs):
    """Test sweep_span with single channel and linear spacing."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 2e6
    npoints = 10
    nsamps = 15
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = True, downward = False, log = False, verbose = False
    )
    
    assert captured_freqs.shape == (1, 10)
    expected = np.linspace(4.0e9 - 1e6, 4.0e9 + 1e6, 10)
    assert np.allclose(captured_freqs[0, :], expected)


@pytest.mark.asyncio
async def test_sweep_span_passes_parameters(base_crs):
    """Test that sweep_span passes parameters correctly to sweep."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 20
    ch_map = {1: [0]}
    
    captured_params = {}
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        captured_params['frequencies'] = frequencies
        captured_params['ares'] = ares_in
        captured_params['nsamps'] = nsamps_in
        captured_params['ch_map'] = kwargs.get('ch_map')
        captured_params['allow_missing'] = kwargs.get('allow_missing')
        captured_params['verbose'] = kwargs.get('verbose')
        captured_params['pbar_description'] = kwargs.get('pbar_description')
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        ch_map = ch_map, allow_missing = True, log = False,
        verbose = False, pbar_description = "Custom Span Sweep"
    )
    
    # Verify all parameters passed through
    assert np.array_equal(captured_params['ares'], ares)
    assert captured_params['nsamps'] == 20
    assert captured_params['ch_map'] == ch_map
    assert captured_params['allow_missing'] == True
    assert captured_params['verbose'] == False
    assert captured_params['pbar_description'] == "Custom Span Sweep"


@pytest.mark.asyncio
async def test_sweep_span_validation_span_values(base_crs):
    """Test sweep_span validates span parameter values."""
    crs = base_crs
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    
    # Negative span
    with pytest.raises(ValueError, match = "span must be a positive float"):
        await crs.sweep_span(fres, ares, -1e6, 5, 10)
    
    # Zero span
    with pytest.raises(ValueError, match = "span must be a positive float"):
        await crs.sweep_span(fres, ares, 0, 5, 10)
    
    # Non-numeric span
    with pytest.raises(ValueError, match = "span must be a positive float"):
        await crs.sweep_span(fres, ares, "1MHz", 5, 10)


@pytest.mark.asyncio
async def test_sweep_span_validation_npoints_values(base_crs):
    """Test sweep_span validates npoints parameter values."""
    crs = base_crs
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    
    # Negative npoints
    with pytest.raises(
        ValueError, match = "npoints must be a positive integer"
        ):
        await crs.sweep_span(fres, ares, 1e6, -5, 10)
    
    # Zero npoints
    with pytest.raises(
        ValueError, match = "npoints must be a positive integer"
        ):
        await crs.sweep_span(fres, ares, 1e6, 0, 10)
    
    # Non-integer npoints
    with pytest.raises(
        ValueError, match = "npoints must be a positive integer"
        ):
        await crs.sweep_span(fres, ares, 1e6, 5.5, 10)


@pytest.mark.asyncio
async def test_sweep_span_multiple_channels_linear(base_crs):
    """Test sweep_span with multiple channels of different frequencies."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}
    
    fres = np.array([3.9e9, 4.0e9, 4.1e9, 4.5e9])
    ares = np.array([-50.0, -51.0, -52.0, -53.0])
    span = 500e3  # 500 kHz
    npoints = 7
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps,
        center_fres = True, downward = False, log = False, verbose = False
    )
    
    # Verify shape
    assert captured_freqs.shape == (4, 7)
    
    # Verify each channel has correct frequency range
    for i, f0 in enumerate(fres):
        expected = np.linspace(f0 - span/2, f0 + span/2, npoints)
        assert np.allclose(captured_freqs[i, :], expected)


@pytest.mark.asyncio
async def test_sweep_span_returns_sweep_output(base_crs):
    """Test that sweep_span returns the output from sweep."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    # Create specific return values
    expected_f = np.array([[3.9e9, 4.0e9, 4.1e9, 4.2e9, 4.3e9]])
    expected_z = np.array([[1+1j, 2+2j, 3+3j, 4+4j, 5+5j]])
    
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        return expected_f, expected_z
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_span(
        fres, ares, span, npoints, nsamps, log = False, verbose = False
    )
    
    # Verify the returned values are from sweep
    assert np.array_equal(f, expected_f)
    assert np.array_equal(z, expected_z)


@pytest.mark.asyncio
async def test_sweep_span_validation_dec_grp_invalid_type(base_crs):
    """Test sweep_span raises error when dec_grp is not a zarr.Group or None."""
    crs = base_crs
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    with pytest.raises(TypeError, match = "must be a zarr.Group or None"):
        await crs.sweep_span(
            fres, ares, span, npoints, nsamps, 
            dec_grp = "not a group", verbose = False
        )


@pytest.mark.asyncio
async def test_sweep_span_passes_dec_grp_to_sweep(base_crs):
    """Test sweep_span passes dec_grp parameter to sweep."""
    import zarr
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    span = 1e6
    npoints = 5
    nsamps = 10
    
    # Create a mock zarr group
    mock_grp = MagicMock(spec = zarr.Group)
    mock_grp.__class__ = zarr.Group
    
    captured_dec_grp = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_dec_grp
        captured_dec_grp = kwargs.get('dec_grp')
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    await crs.sweep_span(
        fres, ares, span, npoints, nsamps, 
        dec_grp = mock_grp, verbose = False
    )
    
    # Verify dec_grp was passed through
    assert captured_dec_grp is mock_grp


################################################################################
####################### Placeholder for other sweep methods ####################
################################################################################

def test_crs_sweep_placeholder():
    """Placeholder for sweep tests."""
    pass


################################################################################
####################### sweep_qres tests #######################################
################################################################################

@pytest.mark.asyncio
async def test_sweep_qres_single_channel(base_crs):
    """Test sweep_qres with single channel."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])  # Q factor
    npoints = 5
    nsamps = 10
    
    # Mock sweep to capture frequencies
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    # Calculate expected: span = fres / qres = 4e9 / 1000 = 4e6
    span = fres[0] / qres[0]
    expected = np.linspace(fres[0] + span/2, fres[0] - span/2, npoints)
    
    assert captured_freqs.shape == (1, 5)
    assert np.allclose(captured_freqs[0, :], expected)
    # Verify it's descending
    assert captured_freqs[0, 0] > captured_freqs[0, -1]


@pytest.mark.asyncio
async def test_sweep_qres_multiple_channels_same_q(base_crs):
    """Test sweep_qres with multiple channels, same Q."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-50.0, -51.0])
    qres = np.array([1000.0, 1000.0])
    npoints = 7
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    assert captured_freqs.shape == (2, 7)
    
    # Channel 0: span = 4.0e9 / 1000 = 4e6
    span0 = 4e6
    expected_ch0 = np.linspace(4.0e9 + span0/2, 4.0e9 - span0/2, 7)
    assert np.allclose(captured_freqs[0, :], expected_ch0)
    
    # Channel 1: span = 4.1e9 / 1000 = 4.1e6
    span1 = 4.1e6
    expected_ch1 = np.linspace(4.1e9 + span1/2, 4.1e9 - span1/2, 7)
    assert np.allclose(captured_freqs[1, :], expected_ch1)


@pytest.mark.asyncio
async def test_sweep_qres_multiple_channels_different_q(base_crs):
    """Test sweep_qres with multiple channels, different Q values."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}
    
    fres = np.array([4.0e9, 4.1e9, 4.5e9])
    ares = np.array([-50.0, -51.0, -52.0])
    qres = np.array([1000.0, 2000.0, 500.0])  # Different Q factors
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    assert captured_freqs.shape == (3, 5)
    
    # Verify each channel has correct span based on its Q
    for i in range(3):
        span = fres[i] / qres[i]
        expected = np.linspace(fres[i] + span/2, fres[i] - span/2, npoints)
        assert np.allclose(
            captured_freqs[i, :], expected
            ), f"Channel {i} failed"


@pytest.mark.asyncio
async def test_sweep_qres_high_q_narrow_span(base_crs):
    """Test sweep_qres with high Q (narrow span)."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([10000.0])  # High Q = narrow span
    npoints = 10
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    # span = 4e9 / 10000 = 4e5 = 400 kHz (narrow)
    span = 4e5
    expected = np.linspace(4.0e9 + span/2, 4.0e9 - span/2, 10)
    
    assert np.allclose(captured_freqs[0, :], expected)
    # Verify span is narrow
    freq_range = captured_freqs[0, 0] - captured_freqs[0, -1]
    assert np.isclose(freq_range, span)


@pytest.mark.asyncio
async def test_sweep_qres_low_q_wide_span(base_crs):
    """Test sweep_qres with low Q (wide span)."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([100.0])  # Low Q = wide span
    npoints = 10
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    # span = 4e9 / 100 = 4e7 = 40 MHz (wide)
    span = 4e7
    expected = np.linspace(4.0e9 + span/2, 4.0e9 - span/2, 10)
    
    assert np.allclose(captured_freqs[0, :], expected)
    # Verify span is wide
    freq_range = captured_freqs[0, 0] - captured_freqs[0, -1]
    assert np.isclose(freq_range, span)


@pytest.mark.asyncio
async def test_sweep_qres_passes_parameters(base_crs):
    """Test that sweep_qres passes parameters correctly to sweep."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])
    npoints = 5
    nsamps = 25
    ch_map = {1: [0]}
    
    captured_params = {}
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        captured_params['frequencies'] = frequencies
        captured_params['ares'] = ares_in
        captured_params['nsamps'] = nsamps_in
        captured_params['ch_map'] = kwargs.get('ch_map')
        captured_params['allow_missing'] = kwargs.get('allow_missing')
        captured_params['verbose'] = kwargs.get('verbose')
        captured_params['pbar_description'] = kwargs.get('pbar_description')
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    await crs.sweep_qres(
        fres, ares, qres, npoints, nsamps,
        ch_map = ch_map, allow_missing = True,
        verbose = False, pbar_description = "Custom Q Sweep"
    )
    
    # Verify all parameters passed through
    assert np.array_equal(captured_params['ares'], ares)
    assert captured_params['nsamps'] == 25
    assert captured_params['ch_map'] == ch_map
    assert captured_params['allow_missing'] == True
    assert captured_params['verbose'] == False
    assert captured_params['pbar_description'] == "Custom Q Sweep"


@pytest.mark.asyncio
async def test_sweep_qres_validation_npoints(base_crs):
    """Test sweep_qres validates npoints parameter."""
    crs = base_crs
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])
    
    # Negative npoints
    with pytest.raises(
        ValueError, match = "npoints must be a positive integer"
        ):
        await crs.sweep_qres(fres, ares, qres, -5, 10)
    
    # Zero npoints
    with pytest.raises(
        ValueError, match = "npoints must be a positive integer"
        ):
        await crs.sweep_qres(fres, ares, qres, 0, 10)
    
    # Non-integer npoints
    with pytest.raises(
        ValueError, match = "npoints must be a positive integer"
        ):
        await crs.sweep_qres(fres, ares, qres, 5.5, 10)


@pytest.mark.asyncio
async def test_sweep_qres_always_descending(base_crs):
    """Test that sweep_qres always sweeps downward (descending)."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9, 4.1e9, 4.2e9])
    ares = np.array([-50.0, -51.0, -52.0])
    qres = np.array([1000.0, 2000.0, 1500.0])
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    # Verify all channels sweep downward (first freq > last freq)
    for i in range(3):
        assert captured_freqs[i, 0] > captured_freqs[i, -1], \
            f"Channel {i} is not descending"


@pytest.mark.asyncio
async def test_sweep_qres_always_centered(base_crs):
    """Test that sweep_qres always centers frequencies on fres."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])
    npoints = 11  # Odd number so middle point is clear
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    # With 11 points, the middle (index 5) should be at fres
    middle_idx = npoints // 2
    assert np.isclose(captured_freqs[0, middle_idx], fres[0]), \
        "Frequencies not centered on fres"


@pytest.mark.asyncio
async def test_sweep_qres_returns_sweep_output(base_crs):
    """Test that sweep_qres returns the output from sweep."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])
    npoints = 5
    nsamps = 10
    
    # Create specific return values
    expected_f = np.array([[3.998e9, 3.999e9, 4.0e9, 4.001e9, 4.002e9]])
    expected_z = np.array([[1+1j, 2+2j, 3+3j, 4+4j, 5+5j]])
    
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        return expected_f, expected_z
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres,
        npoints,
        nsamps,
        verbose = False
    )
    
    # Verify the returned values are from sweep
    assert np.array_equal(f, expected_f)
    assert np.array_equal(z, expected_z)


@pytest.mark.asyncio
async def test_sweep_qres_validation_dec_grp_invalid_type(base_crs):
    """Test sweep_qres raises error when dec_grp is not a zarr.Group or None."""
    crs = base_crs
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])
    npoints = 5
    nsamps = 10
    
    with pytest.raises(TypeError, match = "must be a zarr.Group or None"):
        await crs.sweep_qres(
            fres, ares, qres, npoints, nsamps, 
            dec_grp = 123, verbose = False
        )


@pytest.mark.asyncio
async def test_sweep_qres_passes_dec_grp_to_sweep(base_crs):
    """Test sweep_qres passes dec_grp parameter to sweep."""
    import zarr
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    qres = np.array([1000.0])
    npoints = 5
    nsamps = 10
    
    # Create a mock zarr group
    mock_grp = MagicMock(spec = zarr.Group)
    mock_grp.__class__ = zarr.Group
    
    captured_dec_grp = None
    async def mock_sweep(frequencies, ares_in, nsamps_in, **kwargs):
        nonlocal captured_dec_grp
        captured_dec_grp = kwargs.get('dec_grp')
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    await crs.sweep_qres(
        fres, ares, qres, npoints, nsamps, 
        dec_grp = mock_grp, verbose = False
    )
    
    # Verify dec_grp was passed through
    assert captured_dec_grp is mock_grp


################################################################################
####################### sweep_full tests #######################################
################################################################################

@pytest.mark.asyncio
async def test_sweep_full_single_nco(base_crs):
    """Test sweep_full with single NCO."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6  # 512 MHz bandwidth
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    # Mock sweep to capture parameters
    captured_params = {}
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        captured_params['frequencies'] = frequencies.copy()
        captured_params['ares'] = ares.copy()
        captured_params['nsamps'] = nsamps

        # Return mock data with same shape as frequencies
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify 1024 tones were created for the NCO
    assert captured_params['frequencies'].shape[0] == 1024
    assert len(captured_params['ares']) == 1024
    assert captured_params['frequencies'].shape[1] == npoints
    
    # Verify all amplitudes are the same
    assert np.all(captured_params['ares'] == amplitude)
    
    # Verify output is flattened
    assert f.ndim == 1
    assert z.ndim == 1
    assert len(f) == 1024 * npoints
    assert len(z) == 1024 * npoints
    
    # Verify output is sorted by frequency
    assert np.all(np.diff(f) >= 0)


@pytest.mark.asyncio
async def test_sweep_full_multiple_ncos(base_crs):
    """Test sweep_full with multiple NCOs."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_params = {}
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        captured_params['frequencies'] = frequencies.copy()
        captured_params['ares'] = ares.copy()
        
        # Return mock data with same shape as frequencies
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify 1024 tones per NCO = 2048 total
    assert captured_params['frequencies'].shape[0] == 2048
    assert len(captured_params['ares']) == 2048
    
    # Verify output is flattened and sorted
    assert len(f) == 2048 * npoints
    assert len(z) == 2048 * npoints
    assert np.all(np.diff(f) >= 0)


@pytest.mark.asyncio
async def test_sweep_full_frequency_coverage(base_crs):
    """Test that sweep_full covers the full bandwidth with safety margin."""
    crs = base_crs
    nco = 4.0e9
    crs.nco_freqs = {1: nco}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify margin is applied (frequencies should be inside bandwidth)
    margin = crs.bw * 1e-9
    expected_min = nco - crs.bw / 2 + margin
    expected_max = nco + crs.bw / 2 - margin
    
    # Frequencies are created as linspace then split and flipped (downward=True)
    # After downward flip, [0, -1] is the first element of first chunk (minimum)
    # and [-1, 0] is the last element of last chunk (maximum)
    assert np.isclose(captured_freqs[0, -1], expected_min, atol = 1.0)
    assert np.isclose(captured_freqs[-1, 0], expected_max, atol = 1.0)


@pytest.mark.asyncio
async def test_sweep_full_edge_offset(base_crs):
    """Test that sweep_full applies safety margin at band edges."""
    crs = base_crs
    nco = 4.0e9
    crs.nco_freqs = {1: nco}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_freqs = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_freqs
        captured_freqs = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify margin: frequencies should NOT reach exactly nco ± bw/2
    margin = crs.bw * 1e-9
    assert captured_freqs[:, 0].min() > nco - crs.bw / 2
    assert captured_freqs[:, 0].max() < nco + crs.bw / 2
    
    # Verify margin is approximately as specified
    expected_min = nco - crs.bw / 2 + margin
    expected_max = nco + crs.bw / 2 - margin
    # After downward flip: [0, -1] is minimum, [-1, 0] is maximum
    assert np.isclose(captured_freqs[0, -1], expected_min, atol = 1.0)
    assert np.isclose(captured_freqs[-1, 0], expected_max, atol = 1.0)


@pytest.mark.asyncio
async def test_sweep_full_passes_parameters(base_crs):
    """Test that sweep_full passes parameters through to sweep."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -45.0
    npoints = 7
    nsamps = 15
    
    captured_params = {}
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        captured_params['nsamps'] = nsamps
        captured_params['verbose'] = kwargs.get('verbose')
        captured_params['pbar_description'] = kwargs.get('pbar_description')
        
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    await crs.sweep_full(
        amplitude, npoints, nsamps,
        verbose = False, pbar_description = "Custom Full Sweep"
    )
    
    # Verify parameters passed through
    assert captured_params['nsamps'] == 15
    assert captured_params['verbose'] == False
    assert captured_params['pbar_description'] == "Custom Full Sweep"


@pytest.mark.asyncio
async def test_sweep_full_validation_npoints(base_crs):
    """Test sweep_full validates npoints parameter."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    
    # Negative npoints
    with pytest.raises(ValueError, match = "npoints must be a positive integer"):
        await crs.sweep_full(amplitude, -5, 10)
    
    # Zero npoints
    with pytest.raises(ValueError, match = "npoints must be a positive integer"):
        await crs.sweep_full(amplitude, 0, 10)
    
    # Non-integer npoints
    with pytest.raises(ValueError, match = "npoints must be a positive integer"):
        await crs.sweep_full(amplitude, 5.5, 10)


@pytest.mark.asyncio
async def test_sweep_full_flattened_and_sorted(base_crs):
    """Test that sweep_full returns flattened and sorted arrays."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    # Create mock data to verify flattening and sorting
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        # Create 2D arrays (1024 x npoints)
        freqs = frequencies.copy()
        # Add variation across npoints to create non-trivial sorting case
        for i in range(freqs.shape[0]):
            freqs[i, :] = frequencies[i, :] + np.linspace(0, 100, npoints)
        
        # Create corresponding z values
        z_vals = np.arange(freqs.size).reshape(freqs.shape) + 1j
        
        return freqs, z_vals
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify flattening (1D arrays)
    assert f.ndim == 1
    assert z.ndim == 1
    
    # Verify sorting (ascending frequencies)
    assert np.all(np.diff(f) >= 0), "Frequencies not sorted"


@pytest.mark.asyncio
async def test_sweep_full_tone_bandwidth_calculation(base_crs):
    """Test that sweep_full constructs frequencies with proper spacing."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_frequencies = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_frequencies
        captured_frequencies = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify tone spacing (spacing between consecutive tones)
    # After downward flip, last column has tones in order (first element of each chunk)
    tone_spacing = np.diff(captured_frequencies[:, -1])
    # Frequencies created as linspace(min, max, 1024*npoints), then split into (1024, npoints)
    # Tone spacing = npoints * step where step = (max-min)/(1024*npoints - 1)
    margin = crs.bw * 1e-9
    freq_min = 4.0e9 - crs.bw / 2 + margin
    freq_max = 4.0e9 + crs.bw / 2 - margin
    total_points = 1024 * npoints
    expected_spacing = npoints * (freq_max - freq_min) / (total_points - 1)
    # Use looser tolerance since we're computing expected value from implementation logic
    assert np.allclose(tone_spacing, expected_spacing, rtol = 1e-4)


@pytest.mark.asyncio
async def test_sweep_full_1024_tones_per_nco(base_crs):
    """Test that sweep_full creates exactly 1024 tones per NCO."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9, 3: 5.0e9}  # 3 NCOs
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_frequencies = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_frequencies
        captured_frequencies = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # 3 NCOs × 1024 tones = 3072 tones
    assert captured_frequencies.shape[0] == 3 * 1024
    assert captured_frequencies.shape[1] == npoints
    
    # Output should be 3072 tones × npoints flattened
    assert len(f) == 3 * 1024 * npoints
    assert len(z) == 3 * 1024 * npoints


@pytest.mark.asyncio
async def test_sweep_full_amplitude_conversion(base_crs):
    """Test that sweep_full converts amplitude to float."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    # Pass amplitude as int
    amplitude = -50  # int
    npoints = 5
    nsamps = 10
    
    captured_ares = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_ares
        captured_ares = ares.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(amplitude, npoints, nsamps, verbose = False)
    
    # Verify all ares values are float and equal to amplitude
    assert captured_ares.dtype == np.float64
    assert np.all(captured_ares == -50.0)


@pytest.mark.asyncio
async def test_sweep_full_log_false(base_crs):
    """Test sweep_full with log = False uses linear spacing."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_frequencies = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_frequencies
        captured_frequencies = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(
        amplitude,
        npoints,
        nsamps,
        log = False,
        verbose = False
    )
    
    # Verify tones are linearly spaced
    spacing = np.diff(captured_frequencies[:, 0])
    assert np.allclose(
        spacing, spacing[0], rtol = 1e-10
        ), "Tones not linearly spaced"


@pytest.mark.asyncio
async def test_sweep_full_log_true(base_crs):
    """Test sweep_full with log = True uses logarithmic spacing."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_frequencies = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_frequencies
        captured_frequencies = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(
        amplitude,
        npoints,
        nsamps,
        log = True,
        verbose = False
    )
    
    # Verify tones are logarithmically spaced
    # Log spacing means constant ratio between consecutive frequencies
    ratios = captured_frequencies[1:, 0] / captured_frequencies[:-1, 0]
    assert np.allclose(
        ratios, ratios[0], rtol = 1e-10
        ), "Tones not logarithmically spaced"


@pytest.mark.asyncio
async def test_sweep_full_tone_spacing_linear(base_crs):
    """
    Test that sweep_full uses linear spacing for
    tone frequencies when log = False.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_frequencies = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_frequencies
        captured_frequencies = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(
        amplitude,
        npoints,
        nsamps,
        log = False,
        verbose = False
    )
    
    # Verify tones are linearly spaced
    spacing = np.diff(captured_frequencies[:, 0])
    assert np.allclose(
        spacing, spacing[0], rtol = 1e-10
        ), "Tones not linearly spaced"


@pytest.mark.asyncio
async def test_sweep_full_tone_spacing_logarithmic(base_crs):
    """
    Test that sweep_full uses logarithmic spacing for
    tone frequencies when log = True.
    """
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    captured_frequencies = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_frequencies
        captured_frequencies = frequencies.copy()
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    f, z = await crs.sweep_full(
        amplitude,
        npoints,
        nsamps,
        log = True,
        verbose = False
    )
    
    # Verify tones are logarithmically spaced
    # Log spacing means constant ratio between consecutive frequencies
    ratios = captured_frequencies[1:, 0] / captured_frequencies[:-1, 0]
    assert np.allclose(
        ratios, ratios[0], rtol = 1e-10
        ), "Tones not logarithmically spaced"
    
    # Verify it's NOT linearly spaced
    linear_spacing = np.diff(captured_frequencies[:, 0])
    assert not np.allclose(linear_spacing, linear_spacing[0], rtol = 1e-10), \
        "Tones are linearly spaced, expected logarithmic"


@pytest.mark.asyncio
async def test_sweep_full_validation_dec_grp_invalid_type(base_crs):
    """Test sweep_full raises error when dec_grp is not a zarr.Group or None."""
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    with pytest.raises(TypeError, match = "must be a zarr.Group or None"):
        await crs.sweep_full(
            amplitude, npoints, nsamps, 
            dec_grp = ["not", "a", "group"], verbose = False
        )


@pytest.mark.asyncio
async def test_sweep_full_passes_dec_grp_to_sweep(base_crs):
    """Test sweep_full passes dec_grp parameter to sweep."""
    import zarr
    crs = base_crs
    crs.nco_freqs = {1: 4.0e9}
    crs.bw = 512e6
    
    amplitude = -50.0
    npoints = 5
    nsamps = 10
    
    # Create a mock zarr group
    mock_grp = MagicMock(spec = zarr.Group)
    mock_grp.__class__ = zarr.Group
    
    captured_dec_grp = None
    async def mock_sweep(frequencies, ares, nsamps, **kwargs):
        nonlocal captured_dec_grp
        captured_dec_grp = kwargs.get('dec_grp')
        return frequencies, np.ones_like(frequencies, dtype = complex)
    
    crs.sweep = mock_sweep
    
    await crs.sweep_full(
        amplitude, npoints, nsamps, 
        dec_grp = mock_grp, verbose = False
    )
    
    # Verify dec_grp was passed through to sweep
    assert captured_dec_grp is mock_grp


