"""
Tests for streaming methods.

Tests CRS.capture_ts and CRS.stream methods. The capture_ts method is a
wrapper that writes tones, calls stream, and cleans up. The stream method
handles the actual data acquisition.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call
import time
import os


def create_mock_zarr_group():
    """Create a mock zarr.Group for testing."""
    from zarr.core.group import Group
    mock_grp = MagicMock(spec = Group)
    # Make isinstance check pass
    mock_grp.__class__ = Group
    return mock_grp


def mock_validate_stream_return(
    ts_duration_s,
    dec_stage,
    ch_map,
    allow_missing,
    tmp_directory,
    data_directory,
    batch_size_mb,
    chunk_size_mb,
    delete_parser_data,
    verbose,
):
    """Create a return tuple for mocked _validate_stream_input."""
    return (
        float(ts_duration_s),
        int(dec_stage),
        ch_map,
        bool(allow_missing),
        os.path.normpath(tmp_directory),
        os.path.normpath(data_directory),
        float(batch_size_mb),
        float(chunk_size_mb),
        bool(delete_parser_data),
        bool(verbose),
    )


################################################################################
####################### CRS.capture_ts tests ###################################
################################################################################

@pytest.mark.asyncio
async def test_capture_ts_basic(base_crs):
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
    assert crs._clear_channels.call_args_list[0] == call(range(1, 5))
    
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
        tmp_directory = 'tmp',
        batch_size_mb = 1000.0,
        chunk_size_mb = 128.0,
        delete_parser_data = True,
        verbose = False
    )
    
    # Verify channels cleared after stream (in finally)
    assert crs._clear_channels.call_args_list[1] == call(range(1, 5))


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
async def test_capture_ts_stream_parameters(base_crs):
    """Test capture_ts passes all stream parameters correctly."""
    crs = base_crs
    crs.analog_bank_high = False
    
    fres = np.array([4.0e9])
    ares = np.array([-50.0])
    ts_duration_s = 20.0
    dec_stage = 7
    grp = create_mock_zarr_group()
    tmp_directory = 'custom_tmp/'
    batch_size_mb = 500
    chunk_size_mb = 64
    delete_parser_data = False
    
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    crs.stream = AsyncMock()
    
    with patch('time.sleep'):
        await crs.capture_ts(
            fres, ares, ts_duration_s, dec_stage, grp,
            tmp_directory = tmp_directory,
            batch_size_mb = batch_size_mb,
            chunk_size_mb = chunk_size_mb,
            delete_parser_data = delete_parser_data,
            verbose = True
        )
    
    # Verify all parameters passed to stream
    crs.stream.assert_called_once_with(
        ts_duration_s = ts_duration_s,
        dec_stage = dec_stage,
        grp = grp,
        tmp_directory = os.path.normpath(tmp_directory),
        batch_size_mb = float(batch_size_mb),
        chunk_size_mb = float(chunk_size_mb),
        delete_parser_data = delete_parser_data,
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
    assert crs._clear_channels.call_args_list[0] == call(range(1, 5))
    assert crs._clear_channels.call_args_list[1] == call(range(1, 5))


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
    assert crs._clear_channels.call_args_list[0] == call(range(5, 9))
    assert crs._clear_channels.call_args_list[1] == call(range(5, 9))


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
    assert call_order[0] == ('clear', range(1, 5))
    assert call_order[1] == 'write'
    assert call_order[2] == 'stream'
    assert call_order[3] == ('clear', range(1, 5))


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
    assert crs._clear_channels.call_args_list[0] == call(range(1, 5))
    assert crs._clear_channels.call_args_list[1] == call(range(1, 5))


@pytest.mark.asyncio
async def test_capture_ts_clears_on_write_failure(base_crs):
    """Test that capture_ts doesn't clear again if write_tones fails."""
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
    
    # Only cleared once (before write_tones), stream never called
    assert crs._clear_channels.call_count == 1
    assert crs._clear_channels.call_args_list[0] == call(range(1, 5))
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

def test_validate_stream_input_valid():
    """Test _validate_stream_input with valid inputs."""
    from citkid.crs.instrument import _validate_stream_input
    import tempfile
    import shutil
    
    grp = create_mock_zarr_group()
    tmp_dir = tempfile.mkdtemp()
    
    try:
        (
            ts_duration_s,
            dec_stage,
            ch_map,
            allow_missing,
            tmp_directory,
            data_directory,
            batch_size_mb,
            chunk_size_mb,
            delete_parser_data,
            verbose,
        ) = _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            ch_map=None,
            allow_missing=False,
            tmp_directory=tmp_dir,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )
        
        # Should return normalized paths
        assert tmp_directory == tmp_dir
        assert 'parser_data_00' in data_directory
        assert ts_duration_s == 10.0
        assert dec_stage == 6
        assert ch_map is None
        assert allow_missing is False
        assert batch_size_mb == 1000.0
        assert chunk_size_mb == 128.0
        assert delete_parser_data is True
        assert verbose is False
    finally:
        shutil.rmtree(tmp_dir)


def test_validate_stream_input_invalid_ts_duration_negative():
    """Test _validate_stream_input rejects negative ts_duration_s."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(ValueError, match='ts_duration_s must be a positive'):
        _validate_stream_input(
            ts_duration_s=-10.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_invalid_ts_duration_zero():
    """Test _validate_stream_input rejects zero ts_duration_s."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(ValueError, match='ts_duration_s must be a positive'):
        _validate_stream_input(
            ts_duration_s=0.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_invalid_ts_duration_type():
    """Test _validate_stream_input rejects non-float ts_duration_s."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(ValueError):
        _validate_stream_input(
            ts_duration_s="not_a_float",
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_data_directory_exists():
    """Test _validate_stream_input rejects existing data_directory."""
    from citkid.crs.instrument import _validate_stream_input
    import tempfile
    import shutil
    import os
    
    tmp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(tmp_dir, 'parser_data_00')
    os.makedirs(data_dir)
    
    try:
        with pytest.raises(FileExistsError, match='already exists'):
            _validate_stream_input(
                ts_duration_s=10.0,
                dec_stage=6,
                tmp_directory=tmp_dir,
                grp=create_mock_zarr_group(),
                ch_map=None,
                allow_missing=False,
                batch_size_mb=1000,
                chunk_size_mb=128,
                delete_parser_data=True,
                verbose=False
            )
    finally:
        shutil.rmtree(tmp_dir)


def test_validate_stream_input_invalid_grp_type():
    """Test _validate_stream_input rejects non-zarr.Group grp."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(TypeError, match='grp must be a zarr.Group object'):
        _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp="not_a_group",
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_conflicting_grp_keys(tmp_path):
    """Test _validate_stream_input rejects groups with required names."""
    from citkid.crs.instrument import _validate_stream_input

    grp = create_mock_zarr_group()
    grp.keys.return_value = ["z"]

    with pytest.raises(
        ValueError, match=r"grp already contains required names:.*z"
        ):
        _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory=str(tmp_path),
            grp=grp,
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False,
        )


def test_validate_stream_input_invalid_batch_size_negative():
    """Test _validate_stream_input rejects negative batch_size_mb."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(ValueError, match='batch_size_mb must be a positive'):
        _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=-100,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_invalid_batch_size_zero():
    """Test _validate_stream_input rejects zero batch_size_mb."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(ValueError, match='batch_size_mb must be a positive'):
        _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=0,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_invalid_chunk_size_negative():
    """Test _validate_stream_input rejects negative chunk_size_mb."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(ValueError, match='chunk_size_mb must be a positive'):
        _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=-128,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_invalid_chunk_size_exceeds_batch():
    """Test _validate_stream_input rejects chunk_size > batch_size."""
    from citkid.crs.instrument import _validate_stream_input
    
    with pytest.raises(
        ValueError,
        match='chunk_size_mb must be <= batch_size_mb'
    ):
        _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory='tmp',
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=100,
            chunk_size_mb=200,
            delete_parser_data=True,
            verbose=False
        )


def test_validate_stream_input_coerces_delete_parser_data():
    """Test _validate_stream_input coerces delete_parser_data to bool."""
    from citkid.crs.instrument import _validate_stream_input
    
    result = _validate_stream_input(
        ts_duration_s=10.0,
        dec_stage=6,
        tmp_directory='tmp',
        grp=create_mock_zarr_group(),
        ch_map=None,
        allow_missing=False,
        batch_size_mb=1000,
        chunk_size_mb=128,
        delete_parser_data="yes",
        verbose=False
    )
    assert result[-2] is True


def test_validate_stream_input_coerces_verbose():
    """Test _validate_stream_input coerces verbose to bool."""
    from citkid.crs.instrument import _validate_stream_input
    
    result = _validate_stream_input(
        ts_duration_s=10.0,
        dec_stage=6,
        tmp_directory='tmp',
        grp=create_mock_zarr_group(),
        ch_map=None,
        allow_missing=False,
        batch_size_mb=1000,
        chunk_size_mb=128,
        delete_parser_data=True,
        verbose="yes"
    )
    assert result[-1] is True


def test_validate_stream_input_creates_tmp_directory():
    """Test _validate_stream_input creates tmp_directory if needed."""
    from citkid.crs.instrument import _validate_stream_input
    import tempfile
    import shutil
    import os
    
    parent_dir = tempfile.mkdtemp()
    tmp_dir = os.path.join(parent_dir, 'new_subdir')
    
    try:
        # tmp_dir doesn't exist yet
        assert not os.path.exists(tmp_dir)
        
        result = _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory=tmp_dir,
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )
        tmp_directory, data_directory = result[4], result[5]
        
        # Should create tmp_dir
        assert os.path.exists(tmp_dir)
        assert os.path.isdir(tmp_dir)
    finally:
        shutil.rmtree(parent_dir)


def test_validate_stream_input_normalizes_path():
    """Test _validate_stream_input normalizes tmp_directory path."""
    from citkid.crs.instrument import _validate_stream_input
    import tempfile
    import shutil
    import os
    
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Pass path with redundant separators
        messy_path = tmp_dir + os.sep + '.' + os.sep + 'subdir'
        
        result = _validate_stream_input(
            ts_duration_s=10.0,
            dec_stage=6,
            tmp_directory=messy_path,
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )
        tmp_directory, data_directory = result[4], result[5]
        
        # Should be normalized
        assert tmp_directory == os.path.normpath(messy_path)
    finally:
        shutil.rmtree(tmp_dir)


def test_validate_stream_input_numpy_float():
    """Test _validate_stream_input accepts numpy float types."""
    from citkid.crs.instrument import _validate_stream_input
    import tempfile
    import shutil
    
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Pass numpy float64
        result = _validate_stream_input(
            ts_duration_s=np.float64(10.0),
            dec_stage=6,
            tmp_directory=tmp_dir,
            grp=create_mock_zarr_group(),
            ch_map=None,
            allow_missing=False,
            batch_size_mb=1000,
            chunk_size_mb=128,
            delete_parser_data=True,
            verbose=False
        )
        
        assert result[4] == tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


################################################################################
####################### CRS.stream tests #######################################
################################################################################

@pytest.mark.asyncio
async def test_stream_basic_workflow(base_crs):
    """Test basic stream workflow without verbose."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9, 4.1e9], 2: [4.2e9]}
    crs.ares_map = {1: [-50.0, -51.0], 2: [-52.0]}
    crs.ch_map = {1: [0, 1], 2: [2]}
    crs.ntones = 3
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    
    # Mock methods
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr') as mock_p2z, \
         patch('citkid.crs.instrument.shutil.rmtree') as mock_rmtree:
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            False,
        )
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            verbose=False
        )
        
        # Verify validation called
        mock_validate.assert_called_once()
        
        # Verify set_decimation called
        crs.set_decimation.assert_called_once_with(6, verbose=False)
        
        # Verify parser.main called
        mock_parser.main.assert_called_once()
        
        # Verify parser_to_zarr called
        mock_p2z.assert_called_once()
        
        # Verify cleanup (default delete_parser_data=True)
        mock_rmtree.assert_called_once_with('/tmp/parser_data_00')


@pytest.mark.asyncio
async def test_stream_verbose_mode(base_crs):
    """Test stream uses run_with_time_bar when verbose=True."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9]}
    crs.ares_map = {1: [-50.0]}
    crs.ch_map = {1: [0]}
    crs.ntones = 1
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.instrument.run_with_time_bar', new_callable=AsyncMock) as mock_time_bar, \
         patch('citkid.crs.util.parser_to_zarr'), \
         patch('citkid.crs.instrument.shutil.rmtree'):
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            True,
        )
        
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            verbose=True
        )
        
        # Verify run_with_time_bar called instead of parser.main
        mock_time_bar.assert_called_once()
        mock_parser.main.assert_not_called()
        
        # Check run_with_time_bar arguments
        call_args = mock_time_bar.call_args
        assert call_args[0][0] == mock_parser.main
        assert call_args[0][1] == 10.1  # ts_duration_s + 0.1
        assert call_args[0][2] == 'Streaming'


@pytest.mark.asyncio
async def test_stream_parser_arguments(base_crs):
    """Test stream constructs parser arguments correctly."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9, 4.1e9, 4.2e9], 2: [4.3e9, 4.4e9]}
    crs.ares_map = {1: [-50.0, -51.0, -52.0], 2: [-53.0, -54.0]}
    crs.ch_map = {1: [0, 1, 2], 2: [3, 4]}
    crs.ntones = 5
    crs.sample_freq = 1e6
    crs.interface = 'eth1'
    crs.serial_number = 5678
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr'), \
         patch('citkid.crs.instrument.shutil.rmtree'):
        
        mock_validate.return_value = mock_validate_stream_return(
            20.0,
            7,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            False,
        )
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        await crs.stream(
            ts_duration_s=20.0,
            dec_stage=7,
            grp=grp,
            verbose=False
        )
        
        # Verify parser.main called with correct arguments
        args = mock_parser.main.call_args[0]
        
        # Max ntones = 3 (from module 1)
        assert args[0] == '-i'
        assert args[1] == 'eth1'
        assert args[2] == '-d'
        assert args[3] == '/tmp/parser_data_00'
        assert args[4] == '-c'
        assert args[5] == '1-3'  # max_ntones from fres_map
        assert args[6] == '-s'
        assert args[7] == '5678'
        assert args[8] == '-n'
        # nframes = sample_freq * (ts_duration_s + 0.1)
        assert args[9] == str(int(1e6 * 20.1))


@pytest.mark.asyncio
async def test_stream_parser_to_zarr_arguments(base_crs):
    """Test stream passes correct arguments to parser_to_zarr."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9, 4.1e9], 2: [4.2e9]}
    crs.ares_map = {1: [-50.0, -51.0], 2: [-52.0]}
    crs.ch_map = {1: [0, 1], 2: [2]}
    crs.ntones = 3
    crs.sample_freq = 2e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr') as mock_p2z, \
         patch('citkid.crs.instrument.shutil.rmtree'):
        
        mock_validate.return_value = mock_validate_stream_return(
            15.0,
            6,
            crs.ch_map,
            False,
            '/custom',
            '/custom/parser_data_00',
            500,
            64,
            True,
            False,
        )
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        await crs.stream(
            ts_duration_s=15.0,
            dec_stage=6,
            grp=grp,
            tmp_directory='/custom',
            batch_size_mb=500,
            chunk_size_mb=64,
            verbose=False
        )
        
        # Verify parser_to_zarr called with correct arguments
        mock_p2z.assert_called_once_with(
            '/custom/parser_data_00',
            grp,
            1234,  # serial_number
            3,     # ntones
            2,     # max_ntones (max from fres_map)
            {1: [0, 1], 2: [2]},  # ch_map
            {1: [-50.0, -51.0], 2: [-52.0]},  # ares_map
            1 / 2e6,  # 1 / sample_freq
            batch_size_mb=500,
            chunk_size_mb=64
        )


@pytest.mark.asyncio
async def test_stream_delete_parser_data_true(base_crs):
    """Test stream deletes parser data when delete_parser_data=True."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9]}
    crs.ares_map = {1: [-50.0]}
    crs.ch_map = {1: [0]}
    crs.ntones = 1
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr'), \
         patch('citkid.crs.instrument.shutil.rmtree') as mock_rmtree:
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            False,
        )
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            delete_parser_data=True,
            verbose=False
        )
        
        # Verify rmtree called
        mock_rmtree.assert_called_once_with('/tmp/parser_data_00')


@pytest.mark.asyncio
async def test_stream_delete_parser_data_false(base_crs):
    """Test stream keeps parser data when delete_parser_data=False."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9]}
    crs.ares_map = {1: [-50.0]}
    crs.ch_map = {1: [0]}
    crs.ntones = 1
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr'), \
         patch('citkid.crs.instrument.shutil.rmtree') as mock_rmtree:
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            False,
            False,
        )
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            delete_parser_data=False,
            verbose=False
        )
        
        # Verify rmtree NOT called
        mock_rmtree.assert_not_called()


@pytest.mark.asyncio
async def test_stream_catches_system_exit(base_crs):
    """Test stream catches SystemExit from parser.main."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9]}
    crs.ares_map = {1: [-50.0]}
    crs.ch_map = {1: [0]}
    crs.ntones = 1
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr') as mock_p2z, \
         patch('citkid.crs.instrument.shutil.rmtree'):
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            False,
        )
        # parser.main raises SystemExit
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        # Should not raise - SystemExit should be caught
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            verbose=False
        )
        
        # Verify parser_to_zarr still called after SystemExit
        mock_p2z.assert_called_once()


@pytest.mark.asyncio
async def test_stream_empty_fres_map(base_crs):
    """Test stream handles empty fres_map correctly."""
    crs = base_crs
    crs.fres_map = {}  # Empty
    crs.ares_map = {}
    crs.ch_map = {}
    crs.ntones = 0
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    crs.set_decimation = AsyncMock()
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr'), \
         patch('citkid.crs.instrument.shutil.rmtree'):
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            False,
        )
        mock_parser.main = MagicMock(side_effect=SystemExit(0))
        
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            verbose=False
        )
        
        # Verify parser arguments use max_ntones = 0
        args = mock_parser.main.call_args[0]
        assert args[5] == '1-0'  # chs = '1-0' when empty


@pytest.mark.asyncio
async def test_stream_set_decimation_called_first(base_crs):
    """Test stream calls set_decimation before parser."""
    crs = base_crs
    crs.fres_map = {1: [4.0e9]}
    crs.ares_map = {1: [-50.0]}
    crs.ch_map = {1: [0]}
    crs.ntones = 1
    crs.sample_freq = 1e6
    crs.interface = 'eth0'
    crs.serial_number = 1234
    
    grp = create_mock_zarr_group()
    call_order = []
    
    async def mock_set_dec(*args, **kwargs):
        call_order.append('set_decimation')
    
    crs.set_decimation = mock_set_dec
    
    with patch('citkid.crs.instrument._validate_stream_input') as mock_validate, \
         patch('citkid.crs.util.parser_to_zarr'), patch('rfmux.tools.parser', create=True) as mock_parser, \
         patch('citkid.crs.util.parser_to_zarr'), \
         patch('citkid.crs.instrument.shutil.rmtree'):
        
        mock_validate.return_value = mock_validate_stream_return(
            10.0,
            6,
            crs.ch_map,
            False,
            '/tmp',
            '/tmp/parser_data_00',
            1000,
            128,
            True,
            False,
        )
        
        def mock_parser_main(*args):
            call_order.append('parser')
            raise SystemExit(0)
        
        mock_parser.main = mock_parser_main
        
        await crs.stream(
            ts_duration_s=10.0,
            dec_stage=6,
            grp=grp,
            verbose=False
        )
        
        # Verify set_decimation called before parser
        assert call_order == ['set_decimation', 'parser']




