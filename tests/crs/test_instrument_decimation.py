"""
Tests for CRS.set_decimation method.

Tests decimation configuration and parameter auto-detection.
"""

import pytest
from unittest.mock import AsyncMock, patch
import numpy as np

from citkid.crs.instrument import CRS


@pytest.fixture
def mock_crs_for_set_decimation(base_crs):
    """Extend base_crs with additional mocks for set_decimation tests."""
    with patch('citkid.crs.instrument.util.get_sample_freq',
               return_value = 596.0):
        crs = base_crs
        
        # Mock async methods
        crs.d.set_decimation = AsyncMock()
        
        # Initialize attributes
        crs.dec_stage = None
        crs.dec_short = None
        crs.dec_module_idxs = None
        
        return crs


@pytest.mark.asyncio
async def test_set_decimation_explicit_parameters(
    mock_crs_for_set_decimation,
    capsys
):
    """Test set_decimation with all parameters explicitly provided."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9, 2e9], 2: [3e9]}
    
    await crs.set_decimation(
        dec_stage = 6,
        short = True,
        module_idxs = [1, 2],
        verbose = True
    )
    
    # Verify device method was called
    crs.d.set_decimation.assert_called_once_with(
        6, short = True, module = [1, 2]
    )
    
    # Verify attributes were set
    assert crs.dec_stage == 6
    assert crs.dec_short is True
    assert crs.dec_module_idxs == [1, 2]
    assert crs.sample_freq is not None
    
    # Verify verbose output
    captured = capsys.readouterr()
    assert 'Decimation set: stage = 6' in captured.out
    assert 'short = True' in captured.out
    assert 'modules = [1, 2]' in captured.out


@pytest.mark.asyncio
async def test_set_decimation_auto_short_small(
    mock_crs_for_set_decimation
):
    """Test short is auto-set to True when max tones <= 128."""
    crs = mock_crs_for_set_decimation
    # 100 tones per module (< 128)
    crs.fres_map = {1: [1e9] * 100, 2: [2e9] * 50}
    
    await crs.set_decimation(dec_stage = 5, verbose = False)
    
    # Should auto-set short to True
    assert crs.dec_short is True
    crs.d.set_decimation.assert_called_once_with(
        5, short = True, module = [1, 2]
    )


@pytest.mark.asyncio
async def test_set_decimation_auto_short_large(
    mock_crs_for_set_decimation
):
    """Test short is auto-set to False when max tones > 128."""
    crs = mock_crs_for_set_decimation
    # 200 tones in one module (> 128)
    crs.fres_map = {1: [1e9] * 200, 2: [2e9] * 50}
    
    await crs.set_decimation(dec_stage = 4, verbose = False)
    
    # Should auto-set short to False
    assert crs.dec_short is False
    crs.d.set_decimation.assert_called_once_with(
        4, short = False, module = [1, 2]
    )


@pytest.mark.asyncio
async def test_set_decimation_auto_short_exactly_128(
    mock_crs_for_set_decimation
):
    """Test short is True when max tones is exactly 128."""
    crs = mock_crs_for_set_decimation
    # Exactly 128 tones
    crs.fres_map = {1: [1e9] * 128}
    
    await crs.set_decimation(dec_stage = 5, verbose = False)
    
    # Should auto-set short to True (128 <= 128)
    assert crs.dec_short is True


@pytest.mark.asyncio
async def test_set_decimation_auto_short_empty_fres_map(
    mock_crs_for_set_decimation
):
    """Test short defaults to False when fres_map is empty."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {}
    
    await crs.set_decimation(dec_stage = 5, verbose = False)
    
    # Should default short to False
    assert crs.dec_short is False


@pytest.mark.asyncio
async def test_set_decimation_auto_module_idxs_from_fres_map(
    mock_crs_for_set_decimation
):
    """Test module_idxs is auto-set from fres_map with non-empty values."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9], 2: [], 3: [3e9], 5: [5e9]}
    
    await crs.set_decimation(
        dec_stage = 5,
        short = True,
        verbose = False
    )
    
    # Should only include modules with non-empty values
    assert set(crs.dec_module_idxs) == {1, 3, 5}


@pytest.mark.asyncio
async def test_set_decimation_auto_module_idxs_empty_list(
    mock_crs_for_set_decimation
):
    """Test module_idxs uses fres_map when provided as empty list."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9], 2: [2e9]}
    
    await crs.set_decimation(
        dec_stage = 5,
        short = True,
        module_idxs = [],
        verbose = False
    )
    
    # Should use fres_map when module_idxs is empty
    assert set(crs.dec_module_idxs) == {1, 2}


@pytest.mark.asyncio
async def test_set_decimation_sleep_on_change(mock_crs_for_set_decimation):
    """Test that sleep occurs when parameters change."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9]}
    
    # Set initial values
    crs.dec_stage = 5
    crs.dec_short = True
    crs.dec_module_idxs = [1]
    
    with patch('citkid.crs.instrument.time.sleep') as mock_sleep:
        # Change dec_stage
        await crs.set_decimation(dec_stage = 6, short = True,
                                module_idxs = [1], verbose = False)
        
        # Should have called sleep
        mock_sleep.assert_called_once_with(0.1)


@pytest.mark.asyncio
async def test_set_decimation_no_sleep_when_unchanged(
    mock_crs_for_set_decimation
):
    """Test that no sleep occurs when parameters don't change."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9]}
    
    # Set initial values
    crs.dec_stage = 6
    crs.dec_short = True
    crs.dec_module_idxs = [1]
    
    with patch('citkid.crs.instrument.time.sleep') as mock_sleep:
        # Call with same values
        await crs.set_decimation(dec_stage = 6, short = True,
                                module_idxs = [1], verbose = False)
        
        # Should NOT have called sleep
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_set_decimation_verbose_false_no_print(
    mock_crs_for_set_decimation,
    capsys
):
    """Test that verbose=False suppresses output."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9]}
    
    await crs.set_decimation(dec_stage = 6, short = True,
                            module_idxs = [1], verbose = False)
    
    # Verify no output
    captured = capsys.readouterr()
    assert 'Set decimation' not in captured.out


@pytest.mark.asyncio
async def test_set_decimation_all_stages(mock_crs_for_set_decimation):
    """Test that all valid dec_stage values (0-6) work."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9]}
    
    for stage in range(7):
        await crs.set_decimation(dec_stage = stage, short = True,
                                module_idxs = [1], verbose = False)
        assert crs.dec_stage == stage


################################################################################
##################### set_decimation Input Validation ##########################
################################################################################

@pytest.mark.asyncio
async def test_set_decimation_dec_stage_not_int(
    mock_crs_for_set_decimation
):
    """Test that non-int dec_stage raises ValueError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(
        ValueError,
        match = 'dec_stage must be an integer between 0 and 6'
    ):
        await crs.set_decimation(dec_stage = 5.5)


@pytest.mark.asyncio
async def test_set_decimation_dec_stage_negative(
    mock_crs_for_set_decimation
):
    """Test that negative dec_stage raises ValueError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(ValueError):
        await crs.set_decimation(dec_stage = -1)


@pytest.mark.asyncio
async def test_set_decimation_dec_stage_too_large(
    mock_crs_for_set_decimation
):
    """Test that dec_stage > 6 raises ValueError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(ValueError):
        await crs.set_decimation(dec_stage = 7)


@pytest.mark.asyncio
async def test_set_decimation_short_not_bool_or_none(
    mock_crs_for_set_decimation
):
    """Test that non-bool/non-None short raises TypeError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(
        TypeError,
        match = 'short must be a boolean or None'
    ):
        await crs.set_decimation(dec_stage = 5, short = 1)


@pytest.mark.asyncio
async def test_set_decimation_short_string(mock_crs_for_set_decimation):
    """Test that string short raises TypeError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(TypeError):
        await crs.set_decimation(dec_stage = 5, short = 'True')


@pytest.mark.asyncio
async def test_set_decimation_module_idxs_not_list(
    mock_crs_for_set_decimation
):
    """Test that non-list module_idxs raises TypeError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(
        TypeError,
        match = 'module_idxs must be a list of integers'
    ):
        await crs.set_decimation(dec_stage = 5, module_idxs = 1)


@pytest.mark.asyncio
async def test_set_decimation_module_idxs_non_int_elements(
    mock_crs_for_set_decimation
):
    """Test that module_idxs with non-int elements raises TypeError."""
    crs = mock_crs_for_set_decimation
    
    with pytest.raises(TypeError):
        await crs.set_decimation(dec_stage = 5, module_idxs = [1, '2', 3])


@pytest.mark.asyncio
async def test_set_decimation_module_idxs_numpy_array_valid(
    mock_crs_for_set_decimation
):
    """Test that numpy array module_idxs is accepted."""
    crs = mock_crs_for_set_decimation
    crs.fres_map = {1: [1e9]}
    
    # Should not raise error
    await crs.set_decimation(
        dec_stage = 5,
        short = True,
        module_idxs = np.array([1, 2, 3]),
        verbose = False
    )