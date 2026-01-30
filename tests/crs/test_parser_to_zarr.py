"""
Tests for parser_to_zarr function.

This module contains both input validation tests and functional tests for the
parser_to_zarr function in citkid.crs.util.
"""

import pytest
import numpy as np
import zarr
import os
import tempfile
from citkid.crs import util


################################################################################
########################## Input Validation Tests ##############################
################################################################################

@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        # Invalid path - not a directory
        (
            dict(
                path = "/nonexistent/path",
                grp = None,  # Will create valid zarr group in test
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid grp - not a zarr.hierarchy.Group
        (
            dict(
                path = ".",  # Will use temp dir in test
                grp = "not_a_group",
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            TypeError
        ),
        # Invalid grp - already contains 'counts_to_dbc'
        (
            dict(
                path = ".",
                grp = "HAS_counts_to_dbc",  # Special marker for test
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid grp - already contains 'dt'
        (
            dict(
                path = ".",
                grp = "HAS_dt",  # Special marker for test
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid grp - already contains 'z'
        (
            dict(
                path = ".",
                grp = "HAS_z",  # Special marker for test
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid crs_sn - not an int
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = "not_an_int",
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ntones - negative
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = -1,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ntones - not an int
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10.5,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid max_ntones - zero
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 0,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid max_ntones - negative
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = -128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid max_ntones - not an int
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = "128",
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ch_map - not a dict
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = [(0, [0, 1])],
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            TypeError
        ),
        # Invalid ch_map - non-int keys
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {"0": np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ch_map - non-convertible values
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: ["bad"]},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ch_map - non-convertible array values
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array(["bad"], dtype = object)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ares_map - not a dict
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = [(0, [0.0, 1.0])],
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            TypeError
        ),
        # Invalid ares_map - non-int keys
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {"0": np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ares_map - non-convertible values
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: ["bad"]},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid ares_map - non-convertible array values
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array(["bad"], dtype = object)},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid dt - not a float
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = "bad",
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid dt - negative
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = -0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid dt - zero
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.0,
                batch_size_mb = 1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid batch_size_mb - zero
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 0,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid batch_size_mb - negative
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = -1000,
                chunk_size_mb = 128,
            ),
            ValueError
        ),
        # Invalid chunk_size_mb - zero
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = 0,
            ),
            ValueError
        ),
        # Invalid chunk_size_mb - negative
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 1000,
                chunk_size_mb = -500,
            ),
            ValueError
        ),
        # Invalid chunk_size_mb - larger than batch_size_mb
        (
            dict(
                path = ".",
                grp = None,
                crs_sn = 1,
                ntones = 10,
                max_ntones = 128,
                ch_map = {0: np.array([0, 1], dtype = np.int32)},
                ares_map = {0: np.array([0.0, 1.0])},
                dt = 0.01,
                batch_size_mb = 10,
                chunk_size_mb = 20,
            ),
            ValueError
        ),
    ],
)
def test_parser_to_zarr_invalid_input(kwargs, expected_error, tmp_path):
    """Test that parser_to_zarr raises appropriate errors for invalid inputs."""
    import zarr
    import os

    # Normalize module index 0 to 1 so invalid tests target their intended
    # validation (module indices are now restricted to 1-8).
    ch_map = kwargs.get("ch_map")
    if isinstance(ch_map, dict) and set(ch_map.keys()) == {0}:
        kwargs["ch_map"] = {1: ch_map[0]}
    ares_map = kwargs.get("ares_map")
    if isinstance(ares_map, dict) and set(ares_map.keys()) == {0}:
        kwargs["ares_map"] = {1: ares_map[0]}
    
    # Create a temporary directory for path if needed
    if kwargs["path"] == ".":
        kwargs["path"] = str(tmp_path)
    
    # Create a valid zarr group if grp is None or handle special markers
    if kwargs["grp"] is None:
        zarr_path = os.path.join(str(tmp_path), "test.zarr")
        root = zarr.open(zarr_path, mode = 'a')
        kwargs["grp"] = root.require_group("test_group")
    elif isinstance(kwargs["grp"], str) and kwargs["grp"].startswith("HAS_"):
        # Special marker: create group with conflicting name
        zarr_path = os.path.join(str(tmp_path), "test.zarr")
        root = zarr.open(zarr_path, mode = 'a')
        grp = root.require_group("test_group")
        # Add the conflicting name
        conflict_name = kwargs["grp"][4:]  # Remove "HAS_" prefix
        grp.create_array(conflict_name, shape=(1,), dtype = 'f8')
        kwargs["grp"] = grp
    
    with pytest.raises(expected_error):
        util.parser_to_zarr(**kwargs)


################################################################################
############################ Functional Tests ##################################
################################################################################

def _create_mock_parser_files(tmp_path, crs_sn, module_idxs, max_ntones, 
                               n_samples_per_file, dtype):
    """
    Create mock parser files with known data patterns.
    
    Returns:
    -------
    file_paths : list
        Paths to created files
    expected_data : dict
        Expected data for each module {module_idx: (real, imag)}
    """
    serial_dir = os.path.join(tmp_path, f'serial_{crs_sn:04d}')
    os.makedirs(serial_dir, exist_ok = True)
    
    file_paths = []
    expected_data = {}
    
    for module_idx in module_idxs:
        # Create filename using the same logic as parser_to_zarr
        file_num = module_idx - (module_idx // 5) * 4
        file_path = os.path.join(serial_dir, f'm0{file_num}_raw32')
        file_paths.append(file_path)
        
        n_samples = n_samples_per_file[module_idx]
        total_records = n_samples * max_ntones
        
        # Create data with distinct patterns for each module
        # Real: module_idx * 1000 + time_idx * 10 + channel_in_module
        # Imag: -(module_idx * 1000 + time_idx * 10 + channel_in_module)
        real_data = np.zeros(total_records, dtype = np.int32)
        imag_data = np.zeros(total_records, dtype = np.int32)
        
        for t in range(n_samples):
            for ch in range(max_ntones):
                idx = t * max_ntones + ch
                value = module_idx * 1000 + t * 10 + ch
                real_data[idx] = value
                imag_data[idx] = -value
        
        # Save to file
        structured_array = np.zeros(total_records, dtype = dtype)
        structured_array['i'] = real_data
        structured_array['q'] = imag_data
        structured_array.tofile(file_path)
        
        expected_data[module_idx] = (real_data, imag_data)
    
    return file_paths, expected_data


def test_parser_to_zarr_basic_functionality(tmp_path):
    """Test basic functionality with simple inputs."""
    # Setup
    crs_sn = 1
    ntones = 4
    max_ntones = 4
    n_samples = 100
    dt = 0.001
    module_idxs = [1, 2]
    
    ch_map = {
        1: np.array([0, 1], dtype = np.int32),
        2: np.array([2, 3], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0]),
        2: np.array([-52.0, -53.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    file_paths, expected_data = _create_mock_parser_files(
        parser_path, crs_sn, module_idxs, max_ntones, n_samples_per_file, dtype
    )
    
    # Create zarr group
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Run parser_to_zarr
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt,
        batch_size_mb = 1,
        chunk_size_mb = 1
    )
    
    # Verify outputs exist
    assert 'z' in grp
    assert 'counts_to_dbc' in grp
    assert 'dt' in grp
    
    # Verify dt
    assert grp['dt'][()] == dt
    
    # Verify shape
    z = grp['z']
    assert z.shape == (2, ntones, n_samples)
    
    # Verify chunking (2, ntones, chunk_N)
    assert z.chunks[0] == 2
    assert z.chunks[1] == ntones
    
    # Verify data mapping is correct
    for module_idx in module_idxs:
        ch_idxs = ch_map[module_idx]
        exp_real, exp_imag = expected_data[module_idx]
        
        for i, ch_idx in enumerate(ch_idxs):
            # Extract expected data for this channel
            exp_real_ch = exp_real[i::max_ntones][:n_samples]
            exp_imag_ch = exp_imag[i::max_ntones][:n_samples]
            
            # Compare with output
            np.testing.assert_array_equal(z[0, ch_idx, :], exp_real_ch)
            np.testing.assert_array_equal(z[1, ch_idx, :], exp_imag_ch)


def test_parser_to_zarr_counts_to_dbc_mapping(tmp_path):
    """Test that counts_to_dbc correctly maps ares to each channel."""
    import rfmux.core.transferfunctions
    
    crs_sn = 1
    ntones = 6
    max_ntones = 4
    n_samples = 10
    dt = 0.001
    
    ch_map = {
        1: np.array([0, 1, 3], dtype = np.int32),
        2: np.array([2, 4, 5], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0, -60.0]),
        2: np.array([-52.0, -54.0, -56.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    # Create zarr group
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Run parser_to_zarr
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt
    )
    
    # Calculate expected scale factors
    rfmux_scale = rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
    expected_scale = np.full(ntones, np.nan)
    
    for module_idx in ch_map.keys():
        ares = ares_map[module_idx]
        ch_idxs = ch_map[module_idx]
        pscale = 1 / 10 ** (ares / 20)
        expected_scale[ch_idxs] = rfmux_scale * pscale
    
    # Verify
    np.testing.assert_allclose(grp['counts_to_dbc'][:], expected_scale)


def test_parser_to_zarr_missing_channels(tmp_path):
    """Test that missing channels are filled with zeros."""
    crs_sn = 1
    ntones = 5
    max_ntones = 4
    n_samples = 20
    dt = 0.001
    
    # Channel 2 is missing
    ch_map = {
        1: np.array([0, 1], dtype = np.int32),
        2: np.array([3, 4], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0]),
        2: np.array([-52.0, -53.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    # Create zarr group
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Run parser_to_zarr
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt
    )
    
    z = grp['z']
    
    # Verify missing channel is all zeros
    np.testing.assert_array_equal(
        z[0, 2, :],
        np.zeros(n_samples, dtype = np.int32)
    )
    np.testing.assert_array_equal(
        z[1, 2, :],
        np.zeros(n_samples, dtype = np.int32)
    )
    
    # Verify non-missing channels have data (not all zeros)
    assert not np.all(z[0, 0, :] == 0)
    assert not np.all(z[0, 1, :] == 0)


def test_parser_to_zarr_different_file_lengths(tmp_path):
    """Test that data is read up to the shortest file."""
    crs_sn = 1
    ntones = 4
    max_ntones = 4
    dt = 0.001
    
    # Different file lengths
    n_samples_per_file = {1: 100, 2: 80}  # File 2 is shorter
    
    ch_map = {
        1: np.array([0, 1], dtype = np.int32),
        2: np.array([2, 3], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0]),
        2: np.array([-52.0, -53.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    # Create zarr group
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Run parser_to_zarr
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt
    )
    
    z = grp['z']
    
    # Verify length matches shortest file
    assert z.shape[2] == 80


def test_parser_to_zarr_batch_and_chunk_sizes(tmp_path):
    """Test various batch_size_mb and chunk_size_mb values."""
    crs_sn = 1
    ntones = 8
    max_ntones = 4
    n_samples = 1000
    dt = 0.001
    
    ch_map = {
        1: np.array([0, 1, 2, 3], dtype = np.int32),
        2: np.array([4, 5, 6, 7], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -50.0, -50.0, -50.0]),
        2: np.array([-50.0, -50.0, -50.0, -50.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Test different combinations
    test_cases = [
        (1.0, 1.0),    # Small batches and chunks
        (10, 5),       # Larger batch, smaller chunk
        (5, 5),        # Equal batch and chunk
        (2.5, 1.25),   # Float sizes
        (100, 100),    # Large batches and chunks
    ]
    
    for batch_mb, chunk_mb in test_cases:
        # Create fresh directory for each test
        test_dir = tmp_path / f'test_b{batch_mb}_c{chunk_mb}'
        test_dir.mkdir()
        
        parser_path = str(test_dir / 'parser')
        os.makedirs(parser_path, exist_ok = True)
        _create_mock_parser_files(
            parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
        )
        
        zarr_path = str(test_dir / 'test.zarr')
        root = zarr.open(zarr_path, mode = 'a')
        grp = root.require_group('test_group')
        
        # Run parser_to_zarr
        util.parser_to_zarr(
            path = parser_path,
            grp = grp,
            crs_sn = crs_sn,
            ntones = ntones,
            max_ntones = max_ntones,
            ch_map = ch_map,
            ares_map = ares_map,
            dt = dt,
            batch_size_mb = batch_mb,
            chunk_size_mb = chunk_mb
        )
        
        z = grp['z']
        
        # Verify output shape is correct regardless of batch/chunk sizes
        assert z.shape == (2, ntones, n_samples)
        
        # Verify no gaps in data by checking continuity
        for ch in range(ntones):
            # Data should follow expected pattern
            data = z[0, ch, :]
            # Check that data is not all zeros (unless it's a missing channel)
            if ch in np.concatenate([ch_map[1], ch_map[2]]):
                assert not np.all(data == 0)


def test_parser_to_zarr_chunk_size_accuracy(tmp_path):
    """Test that chunk_size_mb accurately determines chunk dimensions."""
    crs_sn = 1
    ntones = 16
    max_ntones = 8
    n_samples = 10000
    dt = 0.001
    
    ch_map = {
        1: np.array(list(range(0, 8)), dtype = np.int32),
        2: np.array(list(range(8, 16)), dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0] * 8),
        2: np.array([-50.0] * 8)
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    # Test with specific chunk size
    chunk_size_mb = 1  # 1 MB
    
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt,
        batch_size_mb = 10,
        chunk_size_mb = chunk_size_mb
    )
    
    z = grp['z']
    
    # Calculate expected chunk length
    chunk_size_bytes = chunk_size_mb * (1024 ** 2)
    expected_chunk_N = chunk_size_bytes // (2 * ntones * 4)  # 4 bytes per int32
    expected_chunk_N = min(expected_chunk_N, n_samples)
    
    # Verify chunk dimensions
    assert z.chunks[0] == 2
    assert z.chunks[1] == ntones
    assert z.chunks[2] == expected_chunk_N


def test_parser_to_zarr_small_dataset_chunking(tmp_path):
    """Test that small datasets don't get oversized chunks."""
    crs_sn = 1
    ntones = 4
    max_ntones = 4
    n_samples = 10  # Very small dataset
    dt = 0.001
    
    ch_map = {
        1: np.array([0, 1], dtype = np.int32),
        2: np.array([2, 3], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0]),
        2: np.array([-52.0, -53.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Use large chunk size
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt,
        batch_size_mb = 100,
        chunk_size_mb = 100
    )
    
    z = grp['z']
    
    # Chunk size should be capped to dataset size
    assert z.chunks[2] <= n_samples


def test_parser_to_zarr_ntones_not_equal_max_ntones(tmp_path):
    """Test case where ntones != max_ntones * n_modules."""
    crs_sn = 1
    ntones = 6  # Not equal to max_ntones * n_modules
    max_ntones = 4
    n_samples = 50
    dt = 0.001
    
    # Only using 6 channels out of possible 8 (2 modules * 4 max_ntones)
    ch_map = {
        1: np.array([0, 1, 2], dtype = np.int32),
        2: np.array([3, 4, 5], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0, -60.0]),
        2: np.array([-52.0, -54.0, -56.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Verify chunk size calculation uses ntones, not max_ntones
    chunk_size_mb = 1
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt,
        batch_size_mb = 10,
        chunk_size_mb = chunk_size_mb
    )
    
    z = grp['z']
    
    # Verify shape uses ntones
    assert z.shape[1] == ntones
    
    # Verify chunk calculation used ntones
    chunk_size_bytes = chunk_size_mb * (1024 ** 2)
    expected_chunk_N = chunk_size_bytes // (2 * ntones * 4)
    expected_chunk_N = min(expected_chunk_N, n_samples)
    assert z.chunks[1] == ntones
    assert z.chunks[2] == expected_chunk_N


def test_parser_to_zarr_single_module(tmp_path):
    """Test with only a single module."""
    crs_sn = 1
    ntones = 3
    max_ntones = 4
    n_samples = 30
    dt = 0.001
    
    ch_map = {
        1: np.array([0, 1, 2], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0, -60.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples}
    
    # Create mock parser file
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    _create_mock_parser_files(
        parser_path, crs_sn, [1], max_ntones, n_samples_per_file, dtype
    )
    
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt
    )
    
    z = grp['z']
    assert z.shape == (2, ntones, n_samples)


def test_parser_to_zarr_data_continuity(tmp_path):
    """Test that there are no gaps in data across batch boundaries."""
    crs_sn = 1
    ntones = 4
    max_ntones = 4
    n_samples = 500
    dt = 0.001
    
    ch_map = {
        1: np.array([0, 1], dtype = np.int32),
        2: np.array([2, 3], dtype = np.int32)
    }
    ares_map = {
        1: np.array([-50.0, -55.0]),
        2: np.array([-52.0, -53.0])
    }
    
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    n_samples_per_file = {1: n_samples, 2: n_samples}
    
    # Create mock parser files
    parser_path = str(tmp_path / 'parser')
    os.makedirs(parser_path, exist_ok = True)
    file_paths, expected_data = _create_mock_parser_files(
        parser_path, crs_sn, [1, 2], max_ntones, n_samples_per_file, dtype
    )
    
    zarr_path = str(tmp_path / 'test.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    grp = root.require_group('test_group')
    
    # Use small batch size to force multiple batches
    util.parser_to_zarr(
        path = parser_path,
        grp = grp,
        crs_sn = crs_sn,
        ntones = ntones,
        max_ntones = max_ntones,
        ch_map = ch_map,
        ares_map = ares_map,
        dt = dt,
        batch_size_mb = 1,  # Small batch to test multiple batches
        chunk_size_mb = 1
    )
    
    z = grp['z']
    
    # Verify complete data by checking against expected patterns
    for module_idx in [1, 2]:
        ch_idxs = ch_map[module_idx]
        exp_real, exp_imag = expected_data[module_idx]
        
        for i, ch_idx in enumerate(ch_idxs):
            exp_real_ch = exp_real[i::max_ntones][:n_samples]
            exp_imag_ch = exp_imag[i::max_ntones][:n_samples]
            
            # Verify no gaps - all data matches
            np.testing.assert_array_equal(z[0, ch_idx, :], exp_real_ch)
            np.testing.assert_array_equal(z[1, ch_idx, :], exp_imag_ch)