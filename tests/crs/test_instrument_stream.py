"""
Tests for CRS.capture_ts and stream input validation.

capture_ts writes tones with write_tones_for_stream, calls stream, and clears
the tones. The streaming methods themselves are tested in
test_stream_capture.py.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call
import time
import os
import zarr

from citkid.crs.instrument import _validate_stream_input


def create_mock_zarr_group():
    """Create a mock zarr.Group for testing."""
    from zarr.core.group import Group
    mock_grp = MagicMock(spec = Group)
    # Make isinstance check pass
    mock_grp.__class__ = Group
    cfg_grp = MagicMock(spec = Group)
    cfg_grp.__class__ = Group
    mock_grp.require_group.return_value = cfg_grp
    return mock_grp


@pytest.fixture(autouse = True)
def mock_write_system_cfg_to_zarr():
    with patch('citkid.crs.instrument.util.write_system_cfg_to_zarr') as mock_write:
        yield mock_write


@pytest.fixture(autouse = True)
def mock_write_acq_cfg_to_zarr():
    with patch('citkid.crs.instrument.util.write_acq_cfg_to_zarr') as mock_write:
        yield mock_write


################################################################################
####################### CRS.capture_ts tests ###################################
################################################################################

@pytest.mark.asyncio
async def test_capture_ts_basic(base_crs, mock_write_system_cfg_to_zarr):
    """Test basic capture_ts workflow."""
    crs = base_crs
    crs.analog_bank_high = False
    
    fres = np.array([4.0e9, 4.1e9])
    ares = np.array([-50.0, -51.0])
    ts_duration_s = 10.0
    dec_stage = 6
    grp = create_mock_zarr_group()
    
    # Mock methods
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep') as mock_sleep:
        await crs.capture_ts(
            fres, ares, ts_duration_s, dec_stage, grp, verbose = False
        )
    
    # Verify channels cleared before write_tones
    assert crs._clear_channels.call_count == 2
    assert crs._clear_channels.call_args_list[0] == call([1, 2, 3, 4])
    
    # Verify write_tones called with correct parameters
    crs.write_tones.assert_called_once()
    call_args = crs.write_tones.call_args
    assert np.array_equal(call_args[0][0], fres)
    assert np.array_equal(call_args[0][1], ares)
    assert call_args[1]['ch_map'] is None
    assert call_args[1]['allow_missing'] is False
    
    # Verify sleep called for transient
    mock_sleep.assert_called_once_with(0.5)
    
    # Verify stream called with correct parameters
    crs.stream.assert_called_once_with(
        ts_duration_s = ts_duration_s,
        dec_stage = dec_stage,
        grp = grp,
        tmp_directory = None,
        verbose = False
    )

    # Verify channels cleared after stream (in finally)
    assert crs._clear_channels.call_args_list[1] == call([1, 2, 3, 4])

    grp.require_group.assert_called_once_with('crs_config')
    mock_write_system_cfg_to_zarr.assert_called_once_with(
        crs, grp.require_group.return_value
    )


@pytest.mark.asyncio
async def test_capture_ts_with_ch_map(base_crs):
    """Test capture_ts passes ch_map to write_tones."""
    crs = base_crs
    crs.analog_bank_high = False
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    ch_map = {1: [0]}
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        await crs.capture_ts(
            fres, ares, 10.0, 6, create_mock_zarr_group(),
            ch_map = ch_map, verbose = False
        )
    
    # Verify ch_map passed to write_tones
    call_args = crs.write_tones.call_args
    assert call_args[1]['ch_map'] == ch_map


@pytest.mark.asyncio
async def test_capture_ts_with_allow_missing(base_crs):
    """Test capture_ts passes allow_missing to write_tones."""
    crs = base_crs
    crs.analog_bank_high = False
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        await crs.capture_ts(
            fres, ares, 10.0, 6, create_mock_zarr_group(),
            allow_missing = True, verbose = False
        )
    
    # Verify allow_missing passed to write_tones
    call_args = crs.write_tones.call_args
    assert call_args[1]['allow_missing'] is True


@pytest.mark.asyncio
async def test_capture_ts_stream_parameters(base_crs, tmp_path):
    """Test capture_ts passes all stream parameters correctly."""
    crs = base_crs
    crs.analog_bank_high = False
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    ts_duration_s = 20.0
    dec_stage = 5
    grp = create_mock_zarr_group()
    tmp_directory = str(tmp_path / 'fast')
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        await crs.capture_ts(
            fres, ares, ts_duration_s, dec_stage, grp,
            tmp_directory = tmp_directory,
            verbose = True
        )
    
    # Verify all parameters passed to stream
    crs.stream.assert_called_once_with(
        ts_duration_s = ts_duration_s,
        dec_stage = dec_stage,
        grp = grp,
        tmp_directory = tmp_directory,
        verbose = True
    )


@pytest.mark.asyncio
async def test_capture_ts_analog_bank_low(base_crs):
    """Test capture_ts clears correct modules for analog_bank_low."""
    crs = base_crs
    crs.analog_bank_high = False
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        await crs.capture_ts(
            np.array([4.0e9]), np.array([-50.0]),
            10.0, 6, create_mock_zarr_group(), verbose = False
        )
    
    # Should clear modules 1-4 (range(1, 5))
    assert crs._clear_channels.call_count == 2
    assert crs._clear_channels.call_args_list[0] == call([1, 2, 3, 4])
    assert crs._clear_channels.call_args_list[1] == call([1, 2, 3, 4])


@pytest.mark.asyncio
async def test_capture_ts_analog_bank_high(base_crs):
    """Test capture_ts clears correct modules for analog_bank_high."""
    crs = base_crs
    crs.analog_bank_high = True
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        await crs.capture_ts(
            np.array([4.0e9]), np.array([-50.0]),
            10.0, 6, create_mock_zarr_group(), verbose = False
        )
    
    # Should clear modules 5-8 (range(5, 9))
    assert crs._clear_channels.call_count == 2
    assert crs._clear_channels.call_args_list[0] == call([5, 6, 7, 8])
    assert crs._clear_channels.call_args_list[1] == call([5, 6, 7, 8])


@pytest.mark.asyncio
async def test_capture_ts_clears_before_write(base_crs):
    """Test that capture_ts clears channels before write_tones."""
    crs = base_crs
    crs.analog_bank_high = False
    
    call_order = []
    
    async def mock_clear(idx):
        call_order.append(('clear', idx))
    
    async def mock_write(*args, **kwargs):
        call_order.append('write')
    
    async def mock_stream(**kwargs):
        call_order.append('stream')
    
    crs._clear_channels = mock_clear
    crs.write_tones = mock_write
    crs.stream = mock_stream
    
    with patch('time.sleep'):
        await crs.capture_ts(
            np.array([4.0e9]), np.array([-50.0]),
            10.0, 6, create_mock_zarr_group(), verbose = False
        )
    
    # Verify order: clear, write, stream, clear
    assert call_order[0] == ('clear', [1, 2, 3, 4])
    assert call_order[1] == 'write'
    assert call_order[2] == 'stream'
    assert call_order[3] == ('clear', [1, 2, 3, 4])


@pytest.mark.asyncio
async def test_capture_ts_sleeps_after_write(base_crs):
    """Test that capture_ts sleeps after write_tones for transient."""
    crs = base_crs
    crs.analog_bank_high = False
    
    call_order = []
    
    async def mock_write(*args, **kwargs):
        call_order.append('write')
    
    def mock_sleep(duration):
        call_order.append(('sleep', duration))
    
    async def mock_stream(**kwargs):
        call_order.append('stream')
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = mock_write
    crs.stream = mock_stream
    
    with patch('time.sleep', side_effect = mock_sleep):
        await crs.capture_ts(
            np.array([4.0e9]), np.array([-50.0]),
            10.0, 6, create_mock_zarr_group(), verbose = False
        )
    
    # Verify sleep called between write and stream with 0.5 seconds
    write_idx = call_order.index('write')
    stream_idx = call_order.index('stream')
    assert call_order[write_idx + 1] == ('sleep', 0.5)
    assert write_idx + 2 == stream_idx


@pytest.mark.asyncio
async def test_capture_ts_clears_on_stream_failure(base_crs):
    """Test that capture_ts clears channels if stream fails."""
    crs = base_crs
    crs.analog_bank_high = False
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    
    # Make stream raise an exception
    crs.stream = AsyncMock(side_effect = RuntimeError("Stream failed"))
    
    with patch('time.sleep'):
        with pytest.raises(RuntimeError, match = "Stream failed"):
            await crs.capture_ts(
                np.array([4.0e9]), np.array([-50.0]),
                10.0, 6, create_mock_zarr_group(), verbose = False
            )
    
    # Verify channels cleared before and after (in finally block)
    assert crs._clear_channels.call_count == 2
    assert crs._clear_channels.call_args_list[0] == call([1, 2, 3, 4])
    assert crs._clear_channels.call_args_list[1] == call([1, 2, 3, 4])


@pytest.mark.asyncio
async def test_capture_ts_clears_on_write_failure(base_crs):
    """Test that capture_ts clears channels again if write_tones fails."""
    crs = base_crs
    crs.analog_bank_high = False
    
    crs._clear_channels = AsyncMock()
    
    # Make write_tones raise an exception
    crs.write_tones = AsyncMock(side_effect = ValueError("Invalid tones"))
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        with pytest.raises(ValueError, match = "Invalid tones"):
            await crs.capture_ts(
                np.array([4.0e9]), np.array([-50.0]),
                10.0, 6, create_mock_zarr_group(), verbose = False
            )
    
    # Cleared before write_tones and again after the failure, stream never
    # called
    assert crs._clear_channels.call_count == 2
    assert crs._clear_channels.call_args_list[0] == call([1, 2, 3, 4])
    crs.stream.assert_not_called()


@pytest.mark.asyncio
async def test_capture_ts_validates_early(base_crs):
    """Test that capture_ts validates stream inputs before write_tones."""
    crs = base_crs
    crs.analog_bank_high = False
    
    # Mock methods that should NOT be called if validation fails
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        # Pass invalid ts_duration_s - should fail validation
        with pytest.raises(ValueError, match = 'ts_duration_s must be'):
            await crs.capture_ts(
                np.array([4.0e9]), np.array([-50.0]),
                -100, 6, create_mock_zarr_group(),  # Negative duration
                verbose = False
            )
    
    # Should NOT call write_tones or stream (validation failed early)
    crs.write_tones.assert_not_called()
    crs.stream.assert_not_called()
    # Should NOT clear channels (validation failed before starting)
    crs._clear_channels.assert_not_called()


################################################################################
################## _validate_stream_input tests ################################
################################################################################

def make_grp(tmp_path):
    """Create an empty zarr group in tmp_path."""
    return zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')


def test_validate_stream_input_valid(tmp_path):
    """Test _validate_stream_input with valid inputs."""
    result = _validate_stream_input(10, np.int64(3), make_grp(tmp_path), None,
                                    1)
    assert result == (10.0, 3, None, True)
    assert isinstance(result[0], float) and isinstance(result[1], int)


@pytest.mark.parametrize("ts_duration_s", [0, -1.0])
def test_validate_stream_input_invalid_ts_duration(tmp_path, ts_duration_s):
    """Test ts_duration_s must be positive."""
    with pytest.raises(ValueError, match = 'ts_duration_s must be'):
        _validate_stream_input(ts_duration_s, 6, make_grp(tmp_path), None,
                               True)


def test_validate_stream_input_invalid_ts_duration_type(tmp_path):
    """Test ts_duration_s must be numeric."""
    with pytest.raises((TypeError, ValueError)):
        _validate_stream_input('ten', 6, make_grp(tmp_path), None, True)


@pytest.mark.parametrize("dec_stage", [-1, 7])
def test_validate_stream_input_dec_stage_range(tmp_path, dec_stage):
    """Test dec_stage must be in [0, 6]."""
    with pytest.raises(ValueError, match = 'dec_stage must be'):
        _validate_stream_input(1.0, dec_stage, make_grp(tmp_path), None,
                               True)


def test_validate_stream_input_invalid_grp_type():
    """Test grp must be a zarr group."""
    with pytest.raises(TypeError, match = 'grp must be'):
        _validate_stream_input(1.0, 6, {}, None, True)


@pytest.mark.parametrize("name", ['counts_to_s21', 'dt', 'z'])
def test_validate_stream_input_conflicting_grp_keys(tmp_path, name):
    """Test names written by stream must not already exist in grp."""
    grp = make_grp(tmp_path)
    grp.create_array(name = name, data = np.array([1.0]))
    with pytest.raises(ValueError, match = 'already contains required names'):
        _validate_stream_input(1.0, 6, grp, None, True)


def test_validate_stream_input_allows_crs_config(tmp_path):
    """Test crs_config may exist, since write_tones_for_stream creates it."""
    grp = make_grp(tmp_path)
    grp.require_group('crs_config')
    _validate_stream_input(1.0, 6, grp, None, True)


def test_validate_stream_input_creates_tmp_directory(tmp_path):
    """Test tmp_directory is normalized and created."""
    tmp_dir = os.path.join(str(tmp_path), 'a', '..', 'fast', '')
    result = _validate_stream_input(1.0, 6, make_grp(tmp_path), tmp_dir,
                                    True)
    assert result[2] == str(tmp_path / 'fast')
    assert (tmp_path / 'fast').is_dir()


def test_validate_stream_input_invalid_tmp_directory(tmp_path):
    """Test tmp_directory must be a path or None."""
    with pytest.raises(TypeError, match = 'tmp_directory must be'):
        _validate_stream_input(1.0, 6, make_grp(tmp_path), 5, True)


def test_validate_stream_input_coerces_verbose(tmp_path):
    """Test verbose is converted to a bool."""
    result = _validate_stream_input(1.0, 6, make_grp(tmp_path), None, 0)
    assert result[3] is False
