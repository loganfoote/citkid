"""Tests for zarr_util module functions."""

import numpy as np
import pytest
import zarr

from citkid.zarr_util import (
    create_zarr_param,
    write_zarr_row,
    write_zarr,
    write_single_array,
)


################################################################################
########################## create_zarr_param tests #############################
################################################################################

class TestCreateZarrParam:
    """Tests for create_zarr_param shape, chunk, and shard behaviour."""

    def test_1d_value_shape_chunks_shards(self):
        """1D value → shape (nres, n), chunks (1, n), shards (nres, n)."""
        root = zarr.group()
        nres = 10
        value = np.array([1.0, 2.0, 3.0, 4.0])   # shape (4,)

        create_zarr_param(root, 'arr', value, nres)

        arr = root['arr']
        assert arr.shape == (nres, 4)
        assert arr.chunks == (1, 4)
        assert arr.shards == (nres, 4), (
            f"expected shards (nres, 4)=({nres}, 4), got {arr.shards}"
        )

    def test_2d_value_shape_chunks_shards(self):
        """2D value → shape (nres, n, m), chunks (1, n, m), shards (nres, n, m)."""
        root = zarr.group()
        nres = 5
        value = np.zeros((3, 7))  # shape (3, 7)

        create_zarr_param(root, 'arr', value, nres)

        arr = root['arr']
        assert arr.shape == (nres, 3, 7)
        assert arr.chunks == (1, 3, 7)
        assert arr.shards == (nres, 3, 7)

    def test_scalar_value_shape_chunks_shards(self):
        """Scalar value → shape (nres,), chunks (1,), shards (nres,)."""
        root = zarr.group()
        nres = 8
        value = np.float64(3.14)

        create_zarr_param(root, 'arr', value, nres)

        arr = root['arr']
        assert arr.shape == (nres,)
        assert arr.chunks == (1,)
        assert arr.shards == (nres,)

    def test_shards_equal_shape(self):
        """shards must equal shape so the whole array lives in one file."""
        root = zarr.group()
        nres = 6
        value = np.ones(12)

        create_zarr_param(root, 'arr', value, nres)

        arr = root['arr']
        # shards == shape → single shard file for the entire array
        assert arr.shards == arr.shape

    def test_dtype_respected(self):
        """Explicit dtype parameter is applied to the zarr array."""
        root = zarr.group()
        value = np.array([1, 2, 3])  # int64 by default

        create_zarr_param(root, 'arr', value, 4, dtype=np.float32)

        assert root['arr'].dtype == np.float32

    def test_dtype_inherited_from_value(self):
        """When dtype is None the array inherits value.dtype."""
        root = zarr.group()
        value = np.array([1.0, 2.0], dtype=np.float64)

        create_zarr_param(root, 'arr', value, 3)

        assert root['arr'].dtype == np.float64

    def test_idempotent_on_existing_array(self):
        """Calling create_zarr_param for an existing name is a no-op."""
        root = zarr.group()
        value = np.array([1.0, 2.0])

        create_zarr_param(root, 'arr', value, 4)
        create_zarr_param(root, 'arr', value, 4)   # second call must not raise

        assert root['arr'].shape == (4, 2)


################################################################################
############################# write_zarr_row tests #############################
################################################################################

class TestWriteZarrRow:
    """Tests for write_zarr_row."""

    def test_writes_correct_row(self):
        """Value is placed at the specified index."""
        root = zarr.group()
        nres = 4
        value = np.array([10.0, 20.0])
        create_zarr_param(root, 'arr', value, nres)

        write_zarr_row(root, 'arr', 2, np.array([7.0, 8.0]))

        np.testing.assert_array_equal(root['arr'][2], [7.0, 8.0])

    def test_shape_mismatch_raises(self):
        """Shape mismatch between value and array raises ValueError."""
        root = zarr.group()
        nres = 4
        create_zarr_param(root, 'arr', np.array([1.0, 2.0]), nres)

        with pytest.raises(ValueError, match="Shape mismatch"):
            write_zarr_row(root, 'arr', 0, np.array([1.0, 2.0, 3.0]))


################################################################################
############################### write_zarr tests ################################
################################################################################

class TestWriteZarr:
    """Tests for write_zarr (create-or-reuse + write)."""

    def test_creates_and_writes(self):
        """write_zarr creates the array and writes the value."""
        root = zarr.group()
        nres = 5
        value = np.array([1.0, 2.0, 3.0])

        write_zarr(root, 'arr', 0, value, nres)

        arr = root['arr']
        assert arr.shape == (nres, 3)
        np.testing.assert_array_equal(arr[0], value)

    def test_second_write_reuses_array(self):
        """Second write_zarr call reuses the existing array (no overwrite)."""
        root = zarr.group()
        nres = 5
        v0 = np.array([1.0, 2.0])
        v1 = np.array([3.0, 4.0])

        write_zarr(root, 'arr', 0, v0, nres)
        write_zarr(root, 'arr', 1, v1, nres)

        np.testing.assert_array_equal(root['arr'][0], v0)
        np.testing.assert_array_equal(root['arr'][1], v1)

    def test_shards_and_chunks_correct(self):
        """Arrays created by write_zarr have correct chunks and shards."""
        root = zarr.group()
        nres = 6
        value = np.ones((4, 5))   # 2D row

        write_zarr(root, 'arr', 0, value, nres)

        arr = root['arr']
        assert arr.chunks == (1, 4, 5)
        assert arr.shards == (nres, 4, 5)


################################################################################
########################## write_single_array tests ############################
################################################################################

class TestWriteSingleArray:
    """Tests for write_single_array (un-chunked single array)."""

    def test_writes_array(self):
        root = zarr.group()
        value = np.array([1.0, 2.0, 3.0])
        write_single_array(root, 'arr', value)
        np.testing.assert_array_equal(root['arr'][...], value)

    def test_overwrites_existing(self):
        root = zarr.group()
        write_single_array(root, 'arr', np.array([1.0, 2.0]))
        write_single_array(root, 'arr', np.array([9.0, 8.0]))
        np.testing.assert_array_equal(root['arr'][...], [9.0, 8.0])

    def test_dtype_respected(self):
        root = zarr.group()
        write_single_array(root, 'arr', np.array([1, 2, 3]), dtype=np.float64)
        assert root['arr'].dtype == np.float64

    def test_no_sharding(self):
        """write_single_array stores without sharding (no metadata overhead)."""
        root = zarr.group()
        write_single_array(root, 'arr', np.array([1.0, 2.0, 3.0]))
        # write_single_array is intentionally un-chunked: shards should be None
        assert root['arr'].shards is None
