"""Tests for CRS procedures helpers."""

from unittest.mock import Mock, MagicMock
import numpy as np
import pytest
import zarr
from citkid import daq_layout as dl
from .fake_crs import FakeVnaCRS
from citkid.crs.procedures import (
    SweepCancelled,
    NYQUIST_FREQ, NYQUIST_GUARD, VNA_NCHANNELS, VNA_STEP,
    _validate_target_sweep_inputs,
    plan_vna_sweep, vna_range, vna_step, vna_sweep, vna_sweep_to_store
)


@pytest.fixture
def mock_crs():
    """Create a mock CRS instance for testing."""
    crs = Mock()
    crs.__class__.__name__ = 'CRS'
    return crs


@pytest.fixture
def mock_dummy_crs():
    """Create a mock DummyCRS instance for testing."""
    crs = Mock()
    crs.__class__.__name__ = 'DummyCRS'
    return crs

################################################################################
##################### _validate_target_sweep_inputs tests ######################
################################################################################
@pytest.fixture
def valid_inputs():
    """Provide valid default inputs for _validate_target_sweep_inputs."""
    return {
        'fres': np.array([1e9, 2e9, 3e9]),
        'ares': np.array([0.1, 0.2, 0.3]),
        'qres': np.array([1000.0, 2000.0, 3000.0]),
        'res_idxs': np.array([0, 1, 2]),
        'grp': zarr.group(),
        'gain_span_factor': 10.0,
        'npoints_fine': 500,
        'npoints_gain': 50,
        'npoints_rough': 100,
        'nsamps': 100,
        'fres_update_method': 'spacing',
        'verbose': True,
    }


# Tests for _validate_target_sweep_inputs function.


def test_validate_valid_inputs(mock_crs, valid_inputs):
    """Test that valid inputs pass validation and return correct types."""
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    
    # Check return types
    assert isinstance(fres, np.ndarray)
    assert isinstance(ares, np.ndarray)
    assert isinstance(qres, np.ndarray)
    assert isinstance(res_idxs, np.ndarray)
    
    # Check dtypes
    assert fres.dtype == np.float64
    assert ares.dtype == np.float64
    assert qres.dtype == np.float64
    assert res_idxs.dtype == np.int32
    
    # Check shapes
    assert fres.shape == (3,)
    assert ares.shape == (3,)
    assert qres.shape == (3,)
    assert res_idxs.shape == (3,)


def test_validate_dummy_crs_accepted(mock_dummy_crs, valid_inputs):
    """Test that DummyCRS instances are accepted."""
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_dummy_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_invalid_crs_type(valid_inputs):
    """Test that invalid CRS type raises TypeError."""
    invalid_crs = "not a CRS instance"
    with pytest.raises(
        TypeError, match = "crs must be an instance of CRS class"
        ):
        _validate_target_sweep_inputs(invalid_crs, **valid_inputs)


def test_validate_array_conversion_from_list(mock_crs, valid_inputs):
    """Test that list inputs are converted to numpy arrays."""
    valid_inputs['fres'] = [1e9, 2e9, 3e9]
    valid_inputs['ares'] = [0.1, 0.2, 0.3]
    valid_inputs['qres'] = [1000.0, 2000.0, 3000.0]
    valid_inputs['res_idxs'] = [0, 1, 2]
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    
    assert isinstance(fres, np.ndarray)
    assert isinstance(ares, np.ndarray)
    assert isinstance(qres, np.ndarray)
    assert isinstance(res_idxs, np.ndarray)


def test_validate_fres_not_1d(mock_crs, valid_inputs):
    """Test that 2D fres array raises ValueError."""
    valid_inputs['fres'] = np.array([[1e9, 2e9], [3e9, 4e9]])
    with pytest.raises(ValueError, match = "fres must be a 1D array"):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_array_shape_mismatch(mock_crs, valid_inputs):
    """Test that mismatched array shapes raise ValueError."""
    valid_inputs['ares'] = np.array([0.1, 0.2])  # Different length
    with pytest.raises(ValueError, match = "must have the same shape"):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_gain_span_factor_too_small(mock_crs, valid_inputs):
    """Test that gain_span_factor <= 1 raises ValueError."""
    valid_inputs['gain_span_factor'] = 1.0
    with pytest.raises(
        ValueError, match = "gain_span_factor must be a number > 1"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    valid_inputs['gain_span_factor'] = 0.5
    with pytest.raises(
        ValueError, match = "gain_span_factor must be a number > 1"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_gain_span_factor_int_accepted(mock_crs, valid_inputs):
    """Test that integer gain_span_factor is accepted."""
    valid_inputs['gain_span_factor'] = 10
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_npoints_fine_invalid(mock_crs, valid_inputs):
    """Test that invalid npoints_fine raises ValueError."""
    valid_inputs['npoints_fine'] = 0
    with pytest.raises(
        ValueError, match = "npoints_fine must be a positive integer"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    valid_inputs['npoints_fine'] = -10
    with pytest.raises(
        ValueError, match = "npoints_fine must be a positive integer"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_npoints_fine_none_accepted(mock_crs, valid_inputs):
    """Test that None for npoints_fine is accepted."""
    valid_inputs['npoints_fine'] = None
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_npoints_gain_invalid(mock_crs, valid_inputs):
    """Test that invalid npoints_gain raises ValueError."""
    valid_inputs['npoints_gain'] = 0
    with pytest.raises(
        ValueError, match = "npoints_gain must be a positive integer"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_npoints_gain_none_accepted(mock_crs, valid_inputs):
    """Test that None for npoints_gain is accepted."""
    valid_inputs['npoints_gain'] = None
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_npoints_rough_invalid(mock_crs, valid_inputs):
    """Test that invalid npoints_rough raises ValueError."""
    valid_inputs['npoints_rough'] = 0
    with pytest.raises(
        ValueError, match = "npoints_rough must be a positive integer"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_npoints_rough_none_accepted(mock_crs, valid_inputs):
    """Test that None for npoints_rough is accepted."""
    valid_inputs['npoints_rough'] = None
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_nsamps_invalid(mock_crs, valid_inputs):
    """Test that invalid nsamps raises ValueError."""
    valid_inputs['nsamps'] = 0
    with pytest.raises(
        ValueError, match = "nsamps must be a positive integer"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    valid_inputs['nsamps'] = -5
    with pytest.raises(
        ValueError, match = "nsamps must be a positive integer"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_fres_update_method_invalid(mock_crs, valid_inputs):
    """Test that invalid fres_update_method raises ValueError."""
    valid_inputs['fres_update_method'] = 'invalid_method'
    with pytest.raises(ValueError, match = "fres_update_method must be"):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_fres_update_method_valid_options(mock_crs, valid_inputs):
    """Test all valid fres_update_method options."""
    for method in ['distance', 'spacing', 'minS21']:
        valid_inputs['fres_update_method'] = method
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)


def test_validate_cable_delay_not_accepted(mock_crs, valid_inputs):
    """Test that the removed cable_delay parameter is rejected."""
    valid_inputs['cable_delay'] = 0.0
    with pytest.raises(TypeError, match = 'cable_delay'):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_verbose_not_bool(mock_crs, valid_inputs):
    """Test that non-boolean verbose raises TypeError."""
    valid_inputs['verbose'] = "True"
    with pytest.raises(TypeError, match = "verbose must be a boolean"):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_grp_not_zarr_group(mock_crs, valid_inputs):
    """Test that non-Zarr group raises TypeError."""
    valid_inputs['grp'] = {}  # Dict instead of Zarr group
    with pytest.raises(
        TypeError, match = "grp must be a zarr Group instance"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_zarr_group_with_existing_fres(mock_crs, valid_inputs):
    """Test that existing 'fres' dataset in group raises ValueError."""
    grp = zarr.group()
    grp.create_array(name = 'fres', data = np.array([1, 2, 3]))
    valid_inputs['grp'] = grp
    
    with pytest.raises(
        ValueError, match = "already contains dataset 'fres'"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_zarr_group_with_existing_rough_sweep_data(
    mock_crs, valid_inputs
):
    """Test that an existing rough sweep group raises ValueError."""
    grp = zarr.group()
    grp.create_group('rough_sweep')
    valid_inputs['grp'] = grp
    valid_inputs['npoints_rough'] = 100
    
    with pytest.raises(
        ValueError, match = "already contains dataset 'rough_sweep'"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_zarr_group_with_existing_gain_sweep_data(
    mock_crs, valid_inputs
):
    """Test that an existing gain sweep group raises ValueError."""
    grp = zarr.group()
    grp.create_group('gain_sweep')
    valid_inputs['grp'] = grp
    valid_inputs['npoints_gain'] = 50
    
    with pytest.raises(
        ValueError, match = "already contains dataset 'gain_sweep'"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_zarr_group_with_existing_fine_sweep_data(
    mock_crs, valid_inputs
):
    """Test that an existing fine sweep group raises ValueError."""
    grp = zarr.group()
    grp.create_group('fine_sweep')
    valid_inputs['grp'] = grp
    valid_inputs['npoints_fine'] = 500
    
    with pytest.raises(
        ValueError, match = "already contains dataset 'fine_sweep'"
        ):
        _validate_target_sweep_inputs(mock_crs, **valid_inputs)


def test_validate_zarr_group_conflicts_skipped_when_npoints_none(
    mock_crs, valid_inputs
):
    """
    Test that Zarr conflicts are ignored when sweep is disabled 
    (npoints = None).
    """
    grp = zarr.group()
    grp.create_array(name = 's21_rough_f', data = np.array([1, 2, 3]))
    grp.create_array(name = 's21_gain_z', data = np.array([1+1j]))
    grp.create_array(name = 's21_fine_f', data = np.array([1, 2]))
    valid_inputs['grp'] = grp
    
    # Set all sweeps to None - should not raise any errors
    valid_inputs['npoints_rough'] = None
    valid_inputs['npoints_gain'] = None
    valid_inputs['npoints_fine'] = None
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_numpy_integer_types_accepted(mock_crs, valid_inputs):
    """Test that numpy integer types are accepted for integer parameters."""
    valid_inputs['npoints_fine'] = np.int64(500)
    valid_inputs['npoints_gain'] = np.int32(50)
    valid_inputs['nsamps'] = np.int16(100)
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_numpy_float_types_accepted(mock_crs, valid_inputs):
    """Test that numpy float types are accepted for float parameters."""
    valid_inputs['gain_span_factor'] = np.float64(10.0)
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    assert isinstance(fres, np.ndarray)


def test_validate_dtype_conversion_from_int_to_float64(mock_crs, valid_inputs):
    """
    Test that integer arrays are converted to float64 for frequency arrays.
    """
    valid_inputs['fres'] = np.array([1000000000, 2000000000, 3000000000], 
                                    dtype = np.int64)
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    
    assert fres.dtype == np.float64


def test_validate_dtype_conversion_from_int32_to_int32(mock_crs, valid_inputs):
    """Test that res_idxs maintains int32 dtype."""
    valid_inputs['res_idxs'] = np.array([0, 1, 2], dtype = np.int64)
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    
    assert res_idxs.dtype == np.int32


def test_validate_negative_res_idxs_accepted(mock_crs, valid_inputs):
    """Test that negative res_idxs (calibration tones) are accepted."""
    valid_inputs['res_idxs'] = np.array([-1, 0, 1, 2])
    valid_inputs['fres'] = np.array([1e9, 2e9, 3e9, 4e9])
    valid_inputs['ares'] = np.array([0.1, 0.2, 0.3, 0.4])
    valid_inputs['qres'] = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        mock_crs, **valid_inputs
    )
    
    assert res_idxs[0] == -1
    assert isinstance(fres, np.ndarray)


################################################################################
########################## target_sweep tests ##################################
################################################################################

@pytest.fixture
def mock_crs_for_target_sweep():
    """
    Create a mock CRS instance with sweep_qres capability for target_sweep 
    tests.
    """
    from unittest.mock import AsyncMock
    
    crs = MagicMock()
    crs.__class__.__name__ = 'CRS'
    
    # Mock ch_map that will be assigned after first sweep
    crs.ch_map = {'mock': 'ch_map'}
    
    # Mock sweep_qres to return frequency and S21 data
    async def mock_sweep_qres(
        fres, ares, qres, npoints, nsamps, ch_map, acq_cfg_grp, verbose,
        pbar_description
    ):
        """Mock sweep that returns frequency and S21 arrays."""
        nres = len(fres)
        # Create frequency arrays for each resonator
        f = np.zeros((nres, npoints), dtype = np.float64)
        z = np.zeros((nres, npoints), dtype = np.complex128)
        
        for i in range(nres):
            # Create sweep around each resonance
            f[i, :] = np.linspace(fres[i] - fres[i]/qres[i]/2, 
                                   fres[i] + fres[i]/qres[i]/2, 
                                   npoints)
            # Create mock S21 data (simple resonance dip)
            z[i, :] = ares[i] * (1 - 0.5 / (1 + ((f[i, :] - fres[i]) / \
                      (fres[i]/qres[i]/2))**2)) * \
                      np.exp(1j * np.linspace(0, np.pi, npoints))
        
        return f, z
    
    # Wrap in AsyncMock so we can track calls
    crs.sweep_qres = AsyncMock(side_effect = mock_sweep_qres)
    return crs


@pytest.fixture  
def mock_util_write_system_cfg():
    """Mock the util.write_system_cfg_to_zarr function."""
    from unittest.mock import patch
    with patch('citkid.crs.procedures.util.write_system_cfg_to_zarr') as mock:
        yield mock


@pytest.fixture
def mock_update_fres():
    """Mock the update_fres function to return modified frequencies."""
    from unittest.mock import patch
    
    def mock_update(fs, zs, fres, qres, res_idxs, method = 'distance'):
        """Mock update that slightly shifts frequencies."""
        fres_updated = fres.copy()
        # Simulate frequency update by shifting slightly
        fres_updated = fres_updated * 1.0001
        return fres_updated

    # autospec makes calls fail if they don't match the real signature
    with patch(
        'citkid.crs.procedures.update_fres', autospec = True,
        side_effect = mock_update
        ) as mock:
        yield mock


@pytest.fixture
def target_sweep_inputs():
    """Provide valid inputs for target_sweep tests."""
    return {
        'fres': np.array([1e9, 2e9, 3e9]),
        'ares': np.array([0.1, 0.2, 0.3]),
        'qres': np.array([1000.0, 2000.0, 3000.0]),
        'res_idxs': np.array([0, 1, 2]),
        'grp': zarr.group(),
        'ch_map': None,
        'gain_span_factor': 10,
        'npoints_fine': 500,
        'npoints_gain': 50,
        'npoints_rough': 100,
        'nsamps': 100,
        'fres_update_method': 'spacing',
        'verbose': True,
    }


# Tests for target_sweep function.


@pytest.mark.asyncio
async def test_target_sweep_all_sweeps_enabled(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test target_sweep with all three sweeps enabled."""
    from citkid.crs.procedures import target_sweep
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check return types
    assert isinstance(fres_out, np.ndarray)
    assert isinstance(ch_map_out, dict)
    assert fres_out.dtype == np.float64
    
    # Check that fres was updated (by mock_update_fres)
    assert not np.allclose(fres_out, target_sweep_inputs['fres'])
    
    # Check ch_map was assigned
    assert ch_map_out == {'mock': 'ch_map'}
    
    # Check that sweep_qres was called 3 times (rough, gain, fine)
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 3
    
    # Check util.write_system_cfg_to_zarr was called
    mock_util_write_system_cfg.assert_called_once()
    
    # Check update_fres was called once (after rough sweep)
    mock_update_fres.assert_called_once()


@pytest.mark.asyncio
async def test_target_sweep_sweep_order(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that sweeps are executed in correct order: rough, gain, fine."""
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    # Get pbar_description from each call
    call_descs = [call.kwargs['pbar_description'] 
            for call in mock_crs_for_target_sweep.sweep_qres.call_args_list]
    
    assert call_descs == ['Rough sweep', 'Gain sweep', 'Fine sweep']


@pytest.mark.asyncio
async def test_target_sweep_rough_sweep_parameters(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that rough sweep receives correct parameters."""
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    # Get the first call (rough sweep)
    call_args = mock_crs_for_target_sweep.sweep_qres.call_args_list[0]
    args, kwargs = call_args
    
    # Check positional arguments
    np.testing.assert_array_almost_equal(
        args[0], target_sweep_inputs['fres'])
    np.testing.assert_array_almost_equal(
        args[1], target_sweep_inputs['ares'])
    np.testing.assert_array_almost_equal(
        args[2], target_sweep_inputs['qres'])
    
    # Check keyword arguments
    assert kwargs['npoints'] == target_sweep_inputs['npoints_rough']
    assert kwargs['nsamps'] == target_sweep_inputs['nsamps']
    assert kwargs['ch_map'] is None  # First sweep, no ch_map
    assert kwargs['verbose'] == target_sweep_inputs['verbose']
    assert kwargs['pbar_description'] == 'Rough sweep'


@pytest.mark.asyncio
async def test_target_sweep_gain_sweep_parameters(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """
    Test that gain sweep receives correct parameters, especially 
    qres/gain_span_factor.
    """
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    # Get the second call (gain sweep)
    call_args = mock_crs_for_target_sweep.sweep_qres.call_args_list[1]
    args, kwargs = call_args
    
    # Check that fres is updated (not original)
    assert not np.allclose(args[0], target_sweep_inputs['fres'])
    
    # Check ares is same
    np.testing.assert_array_almost_equal(args[1], target_sweep_inputs['ares'])
    
    # Check qres is divided by gain_span_factor
    expected_qres = target_sweep_inputs['qres'] / \
                    target_sweep_inputs['gain_span_factor']
    np.testing.assert_array_almost_equal(args[2], expected_qres)
    
    # Check keyword arguments
    assert kwargs['npoints'] == target_sweep_inputs['npoints_gain']
    assert kwargs['nsamps'] == target_sweep_inputs['nsamps']
    assert kwargs['ch_map'] == {'mock': 'ch_map'}  # assigned after rough
    assert kwargs['verbose'] == target_sweep_inputs['verbose']
    assert kwargs['pbar_description'] == 'Gain sweep'


@pytest.mark.asyncio
async def test_target_sweep_fine_sweep_parameters(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that fine sweep receives correct parameters."""
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    # Get the third call (fine sweep)
    call_args = mock_crs_for_target_sweep.sweep_qres.call_args_list[2]
    args, kwargs = call_args
    
    # Check that fres is updated (not original)
    assert not np.allclose(args[0], target_sweep_inputs['fres'])
    
    # Check ares is same
    np.testing.assert_array_almost_equal(args[1], 
                                         target_sweep_inputs['ares'])
    
    # Check qres is NOT divided (full span)
    np.testing.assert_array_almost_equal(args[2], 
                                         target_sweep_inputs['qres'])
    
    # Check keyword arguments
    assert kwargs['npoints'] == target_sweep_inputs['npoints_fine']
    assert kwargs['nsamps'] == target_sweep_inputs['nsamps']
    assert kwargs['ch_map'] == {'mock': 'ch_map'}
    assert kwargs['verbose'] == target_sweep_inputs['verbose']
    assert kwargs['pbar_description'] == 'Fine sweep'


@pytest.mark.asyncio
async def test_target_sweep_only_rough_sweep(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with only rough sweep enabled."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_gain'] = None
    target_sweep_inputs['npoints_fine'] = None
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check sweep_qres called only once
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 1
    
    # Check update_fres was called
    mock_update_fres.assert_called_once()
    
    # Check return values
    assert isinstance(fres_out, np.ndarray)
    assert isinstance(ch_map_out, dict)


@pytest.mark.asyncio
async def test_target_sweep_only_gain_sweep(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with only gain sweep enabled (no rough, no fine)."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_rough'] = None
    target_sweep_inputs['npoints_fine'] = None
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check sweep_qres called only once
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 1
    
    # Check update_fres was NOT called (no rough sweep)
    mock_update_fres.assert_not_called()
    
    # Check fres is unchanged (no update)
    np.testing.assert_array_almost_equal(
        fres_out, target_sweep_inputs['fres']
        )


@pytest.mark.asyncio
async def test_target_sweep_only_fine_sweep(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with only fine sweep enabled."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_rough'] = None
    target_sweep_inputs['npoints_gain'] = None
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check sweep_qres called only once
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 1
    
    # Check update_fres was NOT called
    mock_update_fres.assert_not_called()
    
    # Check fres is unchanged
    np.testing.assert_array_almost_equal(fres_out, target_sweep_inputs['fres'])

    # Check the channel map made by the fine sweep is returned
    assert ch_map_out == {'mock': 'ch_map'}


@pytest.mark.asyncio
async def test_target_sweep_rough_and_gain_only(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with rough and gain sweeps, but no fine sweep."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_fine'] = None
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check sweep_qres called twice
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 2
    
    # Verify order
    call_descs = [call.kwargs['pbar_description'] 
            for call in mock_crs_for_target_sweep.sweep_qres.call_args_list]
    assert call_descs == ['Rough sweep', 'Gain sweep']


@pytest.mark.asyncio
async def test_target_sweep_gain_and_fine_only(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with gain and fine sweeps, but no rough sweep."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_rough'] = None
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check sweep_qres called twice
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 2
    
    # Verify order
    call_descs = [call.kwargs['pbar_description'] 
            for call in mock_crs_for_target_sweep.sweep_qres.call_args_list]
    assert call_descs == ['Gain sweep', 'Fine sweep']
    
    # Verify gain sweep gets ch_map=None (first sweep)
    a = mock_crs_for_target_sweep.sweep_qres.call_args_list[0]
    assert a.kwargs['ch_map'] is None
    # Verify fine sweep gets ch_map assigned
    b = mock_crs_for_target_sweep.sweep_qres.call_args_list[1]
    assert b.kwargs['ch_map'] == {'mock': 'ch_map'}


@pytest.mark.asyncio
async def test_target_sweep_ch_map_provided(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """
    Test that provided ch_map is used in first sweep, then updated from 
    crs.ch_map.
    """
    from citkid.crs.procedures import target_sweep
    
    provided_ch_map = {'provided': 'map'}
    target_sweep_inputs['ch_map'] = provided_ch_map
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # First sweep (rough) should receive the provided ch_map
    first_call = mock_crs_for_target_sweep.sweep_qres.call_args_list[0]
    assert first_call.kwargs['ch_map'] == provided_ch_map
    
    # Subsequent sweeps receive crs.ch_map (after it gets updated)
    for call in mock_crs_for_target_sweep.sweep_qres.call_args_list[1:]:
        assert call.kwargs['ch_map'] == {'mock': 'ch_map'}
    
    # Returned ch_map should be from crs.ch_map
    assert ch_map_out == {'mock': 'ch_map'}


@pytest.mark.asyncio
async def test_target_sweep_output_data_shapes(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that sweep outputs have correct shapes."""
    from citkid.crs.procedures import target_sweep
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    grp = target_sweep_inputs['grp']
    nres = len(target_sweep_inputs['fres'])
    
    # Check rough sweep data
    grpr = grp['rough_sweep']
    assert grpr['f'].shape == (nres, target_sweep_inputs['npoints_rough'])
    assert grpr['z'].shape == (nres, target_sweep_inputs['npoints_rough'])
    assert grpr['f'].dtype == np.float64
    assert grpr['z'].dtype == np.complex128
    
    # Check gain sweep data
    grpg = grp['gain_sweep']
    assert grpg['f'].shape == (nres, target_sweep_inputs['npoints_gain'])
    assert grpg['z'].shape == (nres, target_sweep_inputs['npoints_gain'])
    assert grpg['f'].dtype == np.float64
    assert grpg['z'].dtype == np.complex128
    
    # Check fine sweep data
    grpf = grp['fine_sweep']
    assert grpf['f'].shape == (nres, target_sweep_inputs['npoints_fine'])
    assert grpf['z'].shape == (nres, target_sweep_inputs['npoints_fine'])
    assert grpf['f'].dtype == np.float64
    assert grpf['z'].dtype == np.complex128


@pytest.mark.asyncio
async def test_target_sweep_zarr_group_structure(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that zarr group has correct structure and saved data."""
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    grp = target_sweep_inputs['grp']
    
    # Check top-level arrays exist
    assert 'ares' in grp
    assert 'qres' in grp
    assert 'res_idxs' in grp
    assert 'fres' in grp
    
    # Check top-level attribute
    assert grp.attrs['nsamps'] == target_sweep_inputs['nsamps']
    
    # Check subgroups exist
    assert 'rough_sweep' in grp
    assert 'gain_sweep' in grp
    assert 'fine_sweep' in grp
    
    # Check rough_sweep subgroup
    grpr = grp['rough_sweep']
    assert 'f' in grpr
    assert 'z' in grpr
    assert 'fres' in grpr  # fres_rough saved here
    a = grpr.attrs['fres_update_method']
    assert a == target_sweep_inputs['fres_update_method']
    assert 'cable_delay' not in grpr.attrs
    
    # Check gain and fine sweep subgroups
    for sweep_name in ['gain_sweep', 'fine_sweep']:
        grps = grp[sweep_name]
        assert 'f' in grps
        assert 'z' in grps


@pytest.mark.asyncio
async def test_target_sweep_zarr_saved_values(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that correct values are saved to zarr."""
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    grp = target_sweep_inputs['grp']
    
    # Check saved input arrays match inputs
    np.testing.assert_array_almost_equal(
        grp['ares'][:], target_sweep_inputs['ares'])
    np.testing.assert_array_almost_equal(
        grp['qres'][:], target_sweep_inputs['qres'])
    np.testing.assert_array_equal(
        grp['res_idxs'][:], target_sweep_inputs['res_idxs'])
    
    # Check fres is updated (not original)
    assert not np.allclose(grp['fres'][:], target_sweep_inputs['fres'])
    
    # Check fres_rough matches original
    np.testing.assert_array_almost_equal(
        grp['rough_sweep']['fres'][:], 
        target_sweep_inputs['fres']
    )


@pytest.mark.asyncio
async def test_target_sweep_update_fres_called_with_correct_params(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that update_fres is called with correct parameters."""
    from citkid.crs.procedures import target_sweep
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    # Check update_fres was called
    mock_update_fres.assert_called_once()
    
    # Check call arguments
    call_args = mock_update_fres.call_args
    args, kwargs = call_args
    
    # Check that f and z have correct shapes
    assert args[0].shape[0] == len(target_sweep_inputs['fres'])  # f
    assert args[1].shape[0] == len(target_sweep_inputs['fres'])  # z
    
    # Check fres and qres match inputs
    np.testing.assert_array_almost_equal(
        args[2], target_sweep_inputs['fres'])
    np.testing.assert_array_almost_equal(
        args[3], target_sweep_inputs['qres'])
    
    # Check res_idxs is passed to identify calibration tones
    np.testing.assert_array_equal(
        args[4], target_sweep_inputs['res_idxs'])

    # Check the only keyword argument is the method
    assert kwargs == {
        'method': target_sweep_inputs['fres_update_method']}


@pytest.mark.asyncio
async def test_target_sweep_update_fres_with_calibration_tones(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """
    Test that res_idxs with calibration tones (negative indices) is
    passed to update_fres.
    """
    from citkid.crs.procedures import target_sweep
    
    # Add calibration tones (negative indices)
    target_sweep_inputs['res_idxs'] = np.array([-1, 0, -2, 1, 2])
    target_sweep_inputs['fres'] = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
    target_sweep_inputs['ares'] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    target_sweep_inputs['qres'] = np.array([1000.0, 2000.0, 3000.0, 
                                            4000.0, 5000.0])
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    # Check res_idxs, including the calibration tones, is passed
    args = mock_update_fres.call_args[0]
    np.testing.assert_array_equal(args[4], np.array([-1, 0, -2, 1, 2]))


@pytest.mark.asyncio
async def test_target_sweep_all_fres_update_methods(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test that all valid fres_update_method values work."""
    from citkid.crs.procedures import target_sweep
    
    for method in ['distance', 'spacing', 'minS21']:
        target_sweep_inputs['fres_update_method'] = method
        target_sweep_inputs['grp'] = zarr.group()  # Fresh group each time
        mock_update_fres.reset_mock()
        
        await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
        
        # Check method was passed correctly
        call_args = mock_update_fres.call_args
        assert call_args[1]['method'] == method


@pytest.mark.asyncio
async def test_target_sweep_no_sweeps_enabled(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test behavior when all sweeps are disabled."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_rough'] = None
    target_sweep_inputs['npoints_gain'] = None
    target_sweep_inputs['npoints_fine'] = None
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # No sweeps should be called
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 0
    
    # fres should be unchanged
    np.testing.assert_array_almost_equal(
        fres_out, target_sweep_inputs['fres'])
    
    # ch_map should still be None (no sweeps to generate it)
    assert ch_map_out is None


@pytest.mark.asyncio
async def test_target_sweep_different_npoints_values(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with different npoints values for each sweep."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['npoints_rough'] = 25
    target_sweep_inputs['npoints_gain'] = 75
    target_sweep_inputs['npoints_fine'] = 1000
    
    await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
    
    grp = target_sweep_inputs['grp']
    nres = len(target_sweep_inputs['fres'])
    
    # Check shapes match the npoints parameters
    assert grp['rough_sweep']['f'].shape == (nres, 25)
    assert grp['gain_sweep']['f'].shape == (nres, 75)
    assert grp['fine_sweep']['f'].shape == (nres, 1000)


@pytest.mark.asyncio
async def test_target_sweep_different_gain_span_factors(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with different gain_span_factor values."""
    from citkid.crs.procedures import target_sweep
    
    for gain_factor in [5, 10, 20, 2.5]:
        target_sweep_inputs['gain_span_factor'] = gain_factor
        target_sweep_inputs['grp'] = zarr.group()
        mock_crs_for_target_sweep.sweep_qres.reset_mock()
        
        await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
        
        # Check gain sweep received correct qres
        gain_call = mock_crs_for_target_sweep.sweep_qres.call_args_list[1]
        expected_qres = target_sweep_inputs['qres'] / gain_factor
        np.testing.assert_array_almost_equal(
            gain_call.args[2], expected_qres)


@pytest.mark.asyncio
async def test_target_sweep_single_resonator(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with a single resonator."""
    from citkid.crs.procedures import target_sweep
    
    target_sweep_inputs['fres'] = np.array([1e9])
    target_sweep_inputs['ares'] = np.array([0.1])
    target_sweep_inputs['qres'] = np.array([1000.0])
    target_sweep_inputs['res_idxs'] = np.array([0])
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check shapes
    grp = target_sweep_inputs['grp']
    assert grp['rough_sweep']['f'].shape == (
        1, target_sweep_inputs['npoints_rough']
        )
    assert grp['gain_sweep']['f'].shape == (
        1, target_sweep_inputs['npoints_gain']
        )
    assert grp['fine_sweep']['f'].shape == (
        1, target_sweep_inputs['npoints_fine']
        )


@pytest.mark.asyncio
async def test_target_sweep_many_resonators(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, mock_update_fres
):
    """Test with many resonators."""
    from citkid.crs.procedures import target_sweep
    
    n_res = 100
    target_sweep_inputs['fres'] = np.linspace(1e9, 2e9, n_res)
    target_sweep_inputs['ares'] = np.full(n_res, 0.1)
    target_sweep_inputs['qres'] = np.full(n_res, 1000.0)
    target_sweep_inputs['res_idxs'] = np.arange(n_res)
    
    fres_out, ch_map_out = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs
    )
    
    # Check shapes
    grp = target_sweep_inputs['grp']
    assert grp['rough_sweep']['f'].shape == (
        n_res, target_sweep_inputs['npoints_rough']
        )
    assert grp['gain_sweep']['f'].shape == (
        n_res, target_sweep_inputs['npoints_gain']
        )
    assert grp['fine_sweep']['f'].shape == (
        n_res, target_sweep_inputs['npoints_fine']
        )


################################################################################
######################## _save_sweep_data tests ################################
################################################################################

# Tests for _save_sweep_data function.


def test_save_sweep_data_save_without_prefix():
    """Test saving sweep data without prefix."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9, 1.2e9], [2e9, 2.1e9, 2.2e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j, 0.7+0.3j], 
                  [0.8+0.4j, 0.9+0.5j, 1.0+0.6j]])
    
    _save_sweep_data(grp, '', f, z)
    
    # Check arrays were created
    assert 'f' in grp
    assert 'z' in grp
    
    # Check data matches
    np.testing.assert_array_almost_equal(grp['f'][:], f)
    np.testing.assert_array_almost_equal(grp['z'][:], z)
    
    # Check dtypes
    assert grp['f'][:].dtype == np.float64
    assert grp['z'][:].dtype == np.complex128
    
    # Check shapes
    assert grp['f'].shape == (2, 3)
    assert grp['z'].shape == (2, 3)


def test_save_sweep_data_save_with_prefix():
    """Test saving sweep data with prefix."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9], [2e9, 2.1e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j], [0.8+0.4j, 0.9+0.5j]])
    
    _save_sweep_data(grp, 'test', f, z)
    
    # Check arrays were created with prefix
    assert 'test_f' in grp
    assert 'test_z' in grp
    
    # Check 'f' and 'z' without prefix don't exist
    assert 'f' not in grp
    assert 'z' not in grp
    
    # Check data matches
    np.testing.assert_array_almost_equal(grp['test_f'][:], f)
    np.testing.assert_array_almost_equal(grp['test_z'][:], z)


def test_save_sweep_data_dtype_conversion_from_list():
    """Test that lists are converted to correct dtypes."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = [[1e9, 1.1e9], [2e9, 2.1e9]]
    z = [[0.5+0.1j, 0.6+0.2j], [0.8+0.4j, 0.9+0.5j]]
    
    _save_sweep_data(grp, '', f, z)
    
    # Check dtypes after conversion
    assert grp['f'][:].dtype == np.float64
    assert grp['z'][:].dtype == np.complex128


def test_save_sweep_data_dtype_conversion_from_int():
    """Test that integer arrays are converted to float64."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1, 2, 3], [4, 5, 6]], dtype = np.int32)
    z = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
    
    _save_sweep_data(grp, '', f, z)
    
    # Check f was converted to float64
    assert grp['f'][:].dtype == np.float64
    np.testing.assert_array_equal(
        grp['f'][:], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )


def test_save_sweep_data_shape_mismatch_error():
    """Test that mismatched shapes raise ValueError."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9], [2e9, 2.1e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j]])  # Different shape
    
    with pytest.raises(
        ValueError, match = "f and z must have the same shape"
        ):
        _save_sweep_data(grp, '', f, z)


def test_save_sweep_data_invalid_grp_type():
    """Test that non-zarr Group raises TypeError."""
    from citkid.crs.procedures import _save_sweep_data
    
    f = np.array([[1e9, 1.1e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j]])
    
    # Try with dict instead of zarr group
    with pytest.raises(
        TypeError, match = "grp must be a zarr Group instance"
        ):
        _save_sweep_data({}, '', f, z)


def test_save_sweep_data_invalid_prefix_type():
    """Test that non-string prefix raises TypeError."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j]])
    
    # Try with int prefix
    with pytest.raises(TypeError, match = "prefix must be a string"):
        _save_sweep_data(grp, 123, f, z)


def test_save_sweep_data_chunking_strategy():
    """Test that data is chunked correctly (1, ncols)."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9, 1.2e9, 1.3e9], 
                  [2e9, 2.1e9, 2.2e9, 2.3e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j, 0.7+0.3j, 0.8+0.4j], 
                  [0.9+0.5j, 1.0+0.6j, 1.1+0.7j, 1.2+0.8j]])
    
    _save_sweep_data(grp, '', f, z)
    
    # Check chunks are (1, 4)
    assert grp['f'].chunks == (1, 4)
    assert grp['z'].chunks == (1, 4)


def test_save_sweep_data_1d_arrays():
    """Test with 1D arrays (single resonator case)."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([1e9, 1.1e9, 1.2e9])
    z = np.array([0.5+0.1j, 0.6+0.2j, 0.7+0.3j])
    
    # Should raise an error because shape[1] doesn't exist for 1D
    # Actually, let me check what happens
    with pytest.raises((IndexError, AttributeError)):
        _save_sweep_data(grp, '', f, z)


def test_save_sweep_data_multiple_prefixes_in_same_group():
    """
    Test saving multiple datasets with different prefixes to same group.
    """
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f1 = np.array([[1e9, 1.1e9]])
    z1 = np.array([[0.5+0.1j, 0.6+0.2j]])
    f2 = np.array([[2e9, 2.1e9]])
    z2 = np.array([[0.7+0.3j, 0.8+0.4j]])
    
    _save_sweep_data(grp, 'rough', f1, z1)
    _save_sweep_data(grp, 'fine', f2, z2)
    
    # Check all datasets exist
    assert 'rough_f' in grp
    assert 'rough_z' in grp
    assert 'fine_f' in grp
    assert 'fine_z' in grp
    
    # Check data is correct for each
    np.testing.assert_array_almost_equal(grp['rough_f'][:], f1)
    np.testing.assert_array_almost_equal(grp['rough_z'][:], z1)
    np.testing.assert_array_almost_equal(grp['fine_f'][:], f2)
    np.testing.assert_array_almost_equal(grp['fine_z'][:], z2)


def test_save_sweep_data_empty_prefix_string():
    """Test that empty string prefix doesn't add underscore."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j]])
    
    _save_sweep_data(grp, '', f, z)
    
    # Should create 'f' and 'z', not '_f' and '_z'
    assert 'f' in grp
    assert 'z' in grp
    assert '_f' not in grp
    assert '_z' not in grp


def test_save_sweep_data_large_dataset():
    """Test with larger dataset (many resonators, many points)."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    n_res = 100
    n_points = 1000
    f = np.random.random((n_res, n_points)) * 1e9
    z = np.random.random((n_res, n_points)) + \
        1j * np.random.random((n_res, n_points))
    
    _save_sweep_data(grp, 'large', f, z)
    
    # Check data was saved correctly
    assert grp['large_f'].shape == (n_res, n_points)
    assert grp['large_z'].shape == (n_res, n_points)
    np.testing.assert_array_almost_equal(grp['large_f'][:], f)
    np.testing.assert_array_almost_equal(grp['large_z'][:], z)


def test_save_sweep_data_single_resonator_single_point():
    """Test with minimum size: single resonator, single point."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9]])
    z = np.array([[0.5+0.1j]])
    
    _save_sweep_data(grp, '', f, z)
    
    assert grp['f'].shape == (1, 1)
    assert grp['z'].shape == (1, 1)
    assert grp['f'][0, 0] == 1e9
    assert grp['z'][0, 0] == 0.5+0.1j


def test_save_sweep_data_complex_dtypes_preserved():
    """Test that complex dtypes are converted to complex128."""
    from citkid.crs.procedures import _save_sweep_data
    
    grp = zarr.group()
    f = np.array([[1e9, 1.1e9]])
    z = np.array([[0.5+0.1j, 0.6+0.2j]], dtype = np.complex64)
    
    _save_sweep_data(grp, '', f, z)
    
    # Should be converted to complex128
    assert grp['z'][:].dtype == np.complex128


def test_save_sweep_data_shards_cover_full_shape():
    """
    Test that both f and z use shards equal to the full array shape so
    each array is stored in a single file.
    """
    from citkid.crs.procedures import _save_sweep_data

    grp = zarr.group()
    nres, npts = 5, 50
    rng = np.random.default_rng(0)
    f = rng.random((nres, npts))
    z = rng.random((nres, npts)) + 1j * rng.random((nres, npts))

    _save_sweep_data(grp, '', f, z)

    # shards must equal the full array shape (one shard = one file)
    assert grp['f'].shards == (nres, npts), (
        f"f shards {grp['f'].shards} != expected ({nres}, {npts})"
    )
    assert grp['z'].shards == (nres, npts), (
        f"z shards {grp['z'].shards} != expected ({nres}, {npts})"
    )

    # chunks must be row-wise: (1, npts)
    assert grp['f'].chunks == (1, npts)
    assert grp['z'].chunks == (1, npts)


def test_save_sweep_data_shards_with_prefix():
    """Test that shards are correct when a prefix is used."""
    from citkid.crs.procedures import _save_sweep_data

    grp = zarr.group()
    nres, npts = 3, 20
    f = np.ones((nres, npts))
    z = np.ones((nres, npts), dtype=np.complex128)

    _save_sweep_data(grp, 'fine', f, z)

    assert grp['fine_f'].shards == (nres, npts)
    assert grp['fine_z'].shards == (nres, npts)
    assert grp['fine_f'].chunks == (1, npts)
    assert grp['fine_z'].chunks == (1, npts)



################################################################################
########################### power_sweep tests ##################################
################################################################################

# Tests for power_sweep.


@pytest.mark.asyncio
async def test_power_sweep_one_set_per_power(
    mock_crs_for_target_sweep, mock_util_write_system_cfg,
    mock_update_fres
):
    """Test that each power is saved to set{i} with the right ares."""
    from citkid.crs.procedures import power_sweep

    root = zarr.group()
    fres = np.array([1e9, 2e9])
    powers = np.array([-50., -45., -40.])
    await power_sweep(
        mock_crs_for_target_sweep, fres, powers, [1e3, 1e3], [0, 1], root,
        npoints_rough = None, npoints_gain = 5, npoints_fine = 7,
        nsamps = 10,
    )
    assert sorted(root.group_keys()) == ['set0', 'set1', 'set2']
    for i, power in enumerate(powers):
        np.testing.assert_array_equal(root[f'set{i}/ares'][:], [power] * 2)
        assert root[f'set{i}/fine_sweep/f'].shape == (2, 7)
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 6


@pytest.mark.asyncio
async def test_power_sweep_ch_map_and_fres_carried_forward(
    mock_crs_for_target_sweep, mock_util_write_system_cfg,
    mock_update_fres
):
    """
    Test that later sets reuse the channel map and start from the
    previous set's updated fres.
    """
    from citkid.crs.procedures import power_sweep

    fres = np.array([1e9, 2e9])
    fres_out, ch_map = await power_sweep(
        mock_crs_for_target_sweep, fres, [-50., -40.], [1e3, 1e3], [0, 1],
        zarr.group(), npoints_rough = 5, npoints_gain = None,
        npoints_fine = 5,
    )
    calls = mock_crs_for_target_sweep.sweep_qres.call_args_list
    # calls: rough0, fine0, rough1, fine1
    assert calls[0].kwargs['ch_map'] is None
    assert calls[2].kwargs['ch_map'] == {'mock': 'ch_map'}
    np.testing.assert_allclose(calls[2].args[0], fres * 1.0001)
    np.testing.assert_allclose(fres_out, fres * 1.0001 ** 2)
    assert ch_map == {'mock': 'ch_map'}


@pytest.mark.asyncio
async def test_power_sweep_fine_only_ch_map_carried_forward(
    mock_crs_for_target_sweep, mock_util_write_system_cfg,
    mock_update_fres
):
    """
    Test that later sets reuse the first set's channel map when only the
    fine sweep is taken.
    """
    from citkid.crs.procedures import power_sweep

    fres = np.array([1e9, 2e9])
    _, ch_map = await power_sweep(
        mock_crs_for_target_sweep, fres, [-50., -40.], [1e3, 1e3], [0, 1],
        zarr.group(), npoints_rough = None, npoints_gain = None,
        npoints_fine = 5,
    )
    calls = mock_crs_for_target_sweep.sweep_qres.call_args_list
    # calls: fine0, fine1
    assert calls[0].kwargs['ch_map'] is None
    assert calls[1].kwargs['ch_map'] == {'mock': 'ch_map'}
    assert ch_map == {'mock': 'ch_map'}


@pytest.mark.asyncio
async def test_power_sweep_2d_aress_and_shuffle(
    mock_crs_for_target_sweep, mock_util_write_system_cfg,
    mock_update_fres
):
    """
    Test per-tone powers, and that shuffle reorders each tone's powers
    independently.
    """
    from citkid.crs.procedures import power_sweep

    nsets, ntones = 6, 4
    aress = np.repeat(np.arange(nsets, dtype = float)[:, None], ntones, 1)
    root = zarr.group()
    await power_sweep(
        mock_crs_for_target_sweep, np.linspace(1e9, 2e9, ntones), aress,
        np.full(ntones, 1e3), np.arange(ntones), root, shuffle = True,
        seed = 0, npoints_rough = None, npoints_gain = None,
        npoints_fine = 3,
    )
    saved = np.array([root[f'set{i}/ares'][:] for i in range(nsets)])
    np.testing.assert_array_equal(np.sort(saved, axis = 0), aress)
    assert not np.all(saved == saved[:, :1])  # tones were shuffled apart


@pytest.mark.asyncio
async def test_power_sweep_existing_set_rejected_before_sweeping(
    mock_crs_for_target_sweep, mock_util_write_system_cfg
):
    """Test that an existing set group raises before any data is taken."""
    from citkid.crs.procedures import power_sweep

    root = zarr.group()
    root.create_group('set1')
    with pytest.raises(ValueError, match = 'set1'):
        await power_sweep(
            mock_crs_for_target_sweep, [1e9], [-50., -40.], [1e3], [0],
            root,
        )
    assert mock_crs_for_target_sweep.sweep_qres.call_count == 0


@pytest.mark.asyncio
async def test_power_sweep_aress_shape_mismatch(mock_crs_for_target_sweep):
    """Test that aress with the wrong number of tones raises."""
    from citkid.crs.procedures import power_sweep

    with pytest.raises(ValueError, match = 'aress must have shape'):
        await power_sweep(
            mock_crs_for_target_sweep, [1e9, 2e9], np.zeros((3, 3)),
            [1e3, 1e3], [0, 1], zarr.group(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ['distance', 'spacing', 'minS21'])
async def test_target_sweep_rough_update_with_real_update_fres(
    mock_crs_for_target_sweep, target_sweep_inputs,
    mock_util_write_system_cfg, method
):
    """
    Test the rough sweep update with the real update_fres: calibration tones
    are unchanged and resonators move within the rough sweep span.
    """
    from citkid.crs.procedures import target_sweep

    target_sweep_inputs['res_idxs'] = np.array([-1, 0, 1])
    target_sweep_inputs['fres_update_method'] = method
    fres_in = target_sweep_inputs['fres'].copy()
    qres = target_sweep_inputs['qres']

    fres, _ = await target_sweep(
        mock_crs_for_target_sweep, **target_sweep_inputs)

    assert fres.dtype == np.float64
    assert fres[0] == fres_in[0]
    assert np.all(np.abs(fres - fres_in) <= fres_in / qres / 2)
    np.testing.assert_array_equal(target_sweep_inputs['grp']['fres'][:], fres)


################################################################################
################################# VNA sweeps ###################################
################################################################################
def test_vna_step_and_range():
    assert vna_step() == VNA_STEP == 2e-6
    assert vna_step(2e4, 10) == pytest.approx(5e-6)
    with pytest.raises(ValueError, match = 'positive'):
        vna_step(0)
    assert vna_range() == (100e6, 2.5e9)
    assert vna_range('2') == (2.5e9, 5e9)
    assert vna_range('both', fmax = 3e9) == (100e6, 3e9)
    with pytest.raises(ValueError, match = 'nyquist'):
        vna_range('3')


def _grid(freqs):
    """
    Return the sorted frequencies of a plan.

    Parameters:
    freqs (list of np.array): ``plan_vna_sweep`` band frequencies.

    Returns:
    f (np.array): all frequencies, ascending.
    """
    return np.sort(np.concatenate([fb.ravel() for fb in freqs]))


@pytest.mark.parametrize('nyquist', ['1', '2', 'both'])
@pytest.mark.parametrize('step', [2e-6, 2e-5])
@pytest.mark.parametrize('modules', [[1], [1, 2, 3, 4]])
@pytest.mark.parametrize('bw', [500e6, 600e6])
def test_plan_vna_sweep_geometric(nyquist, step, modules, bw):
    fmin, fmax = vna_range(nyquist)
    bands, freqs, npc = plan_vna_sweep(fmin, fmax, bw, modules, step)
    f = _grid(freqs)
    # Geometric in each zone, no finer than needed
    for part in (f[f < NYQUIST_FREQ], f[f >= NYQUIST_FREQ]):
        r = np.diff(np.log(part))
        if len(r):
            assert r.max() <= np.log1p(step) * (1 + 1e-9)
            assert r.min() >= np.log1p(step) * (1 - 2e-3)
    # Covering the range (fmax excluded), except the guard band around
    # 2.5 GHz, which is never swept
    guard_lo = NYQUIST_GUARD if fmin == NYQUIST_FREQ else 0.
    guard_hi = NYQUIST_GUARD if fmax == NYQUIST_FREQ else 0.
    assert fmin + guard_lo <= f[0] <= (fmin + guard_lo) * (1 + 1e-9)
    assert f[-1] < fmax - guard_hi
    assert fmax - guard_hi <= f[-1] * (1 + step) * (1 + 1e-9)
    assert not np.any(np.abs(f - NYQUIST_FREQ) < NYQUIST_GUARD)
    if fmin < NYQUIST_FREQ < fmax:
        below, above = f[f < NYQUIST_FREQ], f[f >= NYQUIST_FREQ]
        # The last point below is within a step of the guard band
        assert NYQUIST_FREQ - below[-1] <= NYQUIST_GUARD + \
            (NYQUIST_FREQ - NYQUIST_GUARD) * step
        assert above[0] == pytest.approx(NYQUIST_FREQ + NYQUIST_GUARD)
    nb = len(freqs)
    for i, fb in enumerate(freqs):
        assert fb.shape == (bands['chunk_nchannels'][i], npc)
        assert fb.shape[0] <= VNA_NCHANNELS
        # Rows ascend and continue each other
        assert np.all(np.diff(fb.ravel()) > 0)
        lo, hi = bands['chunk_edges'][i]
        assert (lo, hi) == (fb[0, 0], fb[-1, -1])
        assert hi - lo < bw
        assert np.all(np.abs(fb - bands['chunk_ncos'][i]) < bw / 2 - 100)
        zone = bands['chunk_zones'][i]
        assert hi < NYQUIST_FREQ if zone == 1 else lo >= NYQUIST_FREQ
    assert np.all(np.diff(bands['chunk_edges'][:, 0]) > 0)
    np.testing.assert_array_equal(bands['chunk_passes'],
                                  np.arange(nb) // len(modules))
    np.testing.assert_array_equal(
        bands['chunk_modules'], np.array(modules)[np.arange(nb) %
                                                  len(modules)])
    # Each band is as wide as the channels or the bandwidth allow: one
    # more channel would exceed one of them. Bands are packed up from the
    # bottom, or down from the top for the second zone from 2.5 GHz.
    zones = bands['chunk_zones']
    for i in range(nb - 1):
        if zones[i] != zones[i + 1]:
            continue
        down = zones[i] == 2
        k = i + 1 if down else i
        if down:
            more = bands['chunk_edges'][k, 1] - freqs[i][-1, 0]
        else:
            more = freqs[i + 1][0, -1] - bands['chunk_edges'][k, 0]
        assert bands['chunk_nchannels'][k] == VNA_NCHANNELS or \
            more > bw * (1 - 1e-6)


@pytest.mark.parametrize('fmin, fmax, lo, hi', [
    # Zone edges move in by the guard; the sweep's own edges don't
    (100e6, 2.5e9, 100e6, 2.49e9), (2.5e9, 5e9, 2.51e9, 5e9),
    (100e6, 5e9, 100e6, 5e9), (2.495e9, 3e9, 2.51e9, 3e9),
    (2e9, 2.505e9, 2e9, 2.49e9), (1e9, 2e9, 1e9, 2e9),
    (3e9, 4e9, 3e9, 4e9)])
def test_plan_vna_sweep_nyquist_guard(fmin, fmax, lo, hi):
    """
    Check that nothing within NYQUIST_GUARD of 2.5 GHz is swept, whatever
    the range, and that the sweep otherwise covers it.
    """
    step = 1e-5
    _, freqs, _ = plan_vna_sweep(fmin, fmax, 500e6, [1], step)
    f = _grid(freqs)
    assert f[0] == pytest.approx(lo) and f[-1] < hi
    assert hi <= f[-1] * (1 + step) * (1 + 1e-9)
    assert not np.any(np.abs(f - NYQUIST_FREQ) < NYQUIST_GUARD)
    # Within each zone the grid is continuous: the only gap is the guard
    both = fmin < NYQUIST_FREQ - NYQUIST_GUARD and \
        fmax > NYQUIST_FREQ + NYQUIST_GUARD
    r = np.diff(np.log(f))
    if both:
        i = np.argmax(r)
        assert f[i] < NYQUIST_FREQ < f[i + 1]
        r = np.delete(r, i)
    assert r.max() <= np.log1p(step) * (1 + 1e-9)
    # A wider guard moves the edges further; zero puts them at 2.5 GHz
    if both:
        _, freqs, _ = plan_vna_sweep(fmin, fmax, 500e6, [1], step,
                                     nyquist_guard = 50e6)
        f = _grid(freqs)
        assert not np.any(np.abs(f - NYQUIST_FREQ) < 50e6)
        _, freqs, _ = plan_vna_sweep(fmin, fmax, 500e6, [1], step,
                                     nyquist_guard = 0)
        f = _grid(freqs)
        assert f[f >= NYQUIST_FREQ][0] == pytest.approx(NYQUIST_FREQ)


def test_plan_vna_sweep_within_guard_is_invalid():
    for fmin, fmax in [(2.495e9, 2.505e9), (2.492e9, 2.5e9),
                       (2.5e9, 2.508e9)]:
        with pytest.raises(ValueError, match = 'Nyquist'):
            plan_vna_sweep(fmin, fmax, 500e6, [1], 1e-5)
    with pytest.raises(ValueError, match = 'nyquist_guard'):
        plan_vna_sweep(1e9, 2e9, 500e6, [1], 1e-5, nyquist_guard = -1)


def test_plan_vna_sweep_chooses_fast_npoints():
    """
    Check that the chosen points per channel give the lowest estimated
    time of all choices.
    """
    def cost(npc):
        """
        Return the estimated cost of a plan.

        Parameters:
        npc (int): points per channel.

        Returns:
        cost (float): passes x (npc + 50).
        """
        b, _, _ = plan_vna_sweep(100e6, 2.5e9, 500e6, [1], 1e-5,
                                 npoints_per_ch = npc)
        return (b['chunk_passes'].max() + 1) * (npc + 50)

    _, _, best = plan_vna_sweep(100e6, 2.5e9, 500e6, [1], 1e-5)
    assert all(cost(best) <= cost(n) for n in range(1, 400, 7))


def test_plan_vna_sweep_invalid():
    with pytest.raises(ValueError, match = 'fmin'):
        plan_vna_sweep(1e9, 1e9, 500e6, [1])
    with pytest.raises(ValueError, match = 'module_idxs'):
        plan_vna_sweep(1e9, 2e9, 500e6, [])
    with pytest.raises(ValueError, match = 'step'):
        plan_vna_sweep(1e9, 2e9, 500e6, [1], step = 0)
    with pytest.raises(ValueError, match = 'npoints_per_ch'):
        plan_vna_sweep(1e9, 2e9, 500e6, [1], npoints_per_ch = 0)
    # 1000 points at df/f 1e-3 span far more than 500 MHz
    with pytest.raises(ValueError, match = 'bandwidth'):
        plan_vna_sweep(1e9, 2e9, 500e6, [1], step = 1e-3,
                       npoints_per_ch = 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize('modules', [(1,), (1, 2), (1, 2, 3, 4)])
async def test_vna_sweep_passes_and_stitching(modules):
    crs = FakeVnaCRS(modules = modules, fres = (1.0e9, 1.9e9))
    f, z, bands = await vna_sweep(crs, 400e6, 2400e6, -50., 10,
                                  step = 2e-5, verbose = False)
    npass = bands['chunk_passes'].max() + 1
    assert len(crs.sweep_calls) == len(crs.nco_calls) == npass
    assert len(f) == bands['chunk_nchannels'].sum() * \
        crs.sweep_calls[0]['frequencies'].shape[1]
    # Geometric across the sweep up to the 50 Hz dithering
    assert np.all(np.diff(f) > 0) and f[0] > 400e6 - 50 and f[-1] < 2400e6
    ideal = 400e6 * np.exp(np.arange(len(f)) * np.log(2400 / 400) / len(f))
    assert np.abs(f - ideal).max() <= 50
    # The dips are where the resonances are
    for fr in (1.0e9, 1.9e9):
        near = np.flatnonzero(np.abs(f - fr) < 1e6)
        assert abs(f[near[np.argmin(np.abs(z[near]))]] - fr) < 1e5
    assert bands['chunk_times'].shape == (len(bands['chunk_ncos']),)
    assert np.all(np.diff(bands['chunk_times']) >= 0)
    call = crs.sweep_calls[0]
    assert np.all(call['ares'] == -50.) and call['nsamps'] == 10
    assert call['desc'] == f'VNA pass 1/{npass}'
    # Every channel sweeps downward
    for call in crs.sweep_calls:
        assert np.all(np.diff(call['frequencies'], axis = 1) < 0)
    # Zone 1 only: no Nyquist zone changes
    assert not [c for c in crs.calls if c[0] == 'configure_modules']


@pytest.mark.asyncio
@pytest.mark.parametrize('modules', [(1,), (1, 2)])
async def test_vna_sweep_both_zones(modules):
    crs = FakeVnaCRS(modules = modules, fres = (2.4e9, 2.6e9))
    f, z, bands = await vna_sweep(crs, 2e9, 3e9, -50., 10, step = 2e-5,
                                  verbose = False)
    zones = bands['chunk_zones']
    assert set(zones) == {1, 2}
    # The fake checks each module's tones are in its zone; here check the
    # zones it swept with match the plan
    for p, sweep_zones in enumerate(crs.sweep_zones):
        idx = bands['chunk_passes'] == p
        assert sweep_zones == {int(m): int(zn) for m, zn in
                               zip(bands['chunk_modules'][idx], zones[idx])}
    # Zones are restored, and the other module settings kept
    assert all(cfg['nyquist_zone'] == 1 for cfg in crs.module_cfg.values())
    for name, kw in crs.calls:
        assert name == 'configure_modules' and kw['full_scale_dbm'] == 7.
    # Geometric in each zone up to the dithering (each zone's step is
    # rounded separately, by much less than the dithering), with the guard
    # band around 2.5 GHz left out
    d = np.diff(np.log(f)) / np.log1p(2e-5)
    i = np.argmax(d)
    assert f[i] < NYQUIST_FREQ - NYQUIST_GUARD * 0.999
    assert f[i + 1] == pytest.approx(NYQUIST_FREQ + NYQUIST_GUARD, abs = 60)
    d = np.delete(d, i)
    assert d.max() < 1.01 and d.min() > 0.99
    for fr in (2.4e9, 2.6e9):
        near = np.flatnonzero(np.abs(f - fr) < 1e6)
        assert abs(f[near[np.argmin(np.abs(z[near]))]] - fr) < 2e5


@pytest.mark.asyncio
@pytest.mark.parametrize('modules', [(1,), (1, 2)])
async def test_vna_sweep_progress(modules):
    crs = FakeVnaCRS(modules = modules)
    reports = []
    f, _, bands = await vna_sweep(
        crs, 400e6, 2400e6, -50., 10, step = 2e-5, verbose = False,
        progress = lambda done, total: reports.append((done, total)))
    npass = bands['chunk_passes'].max() + 1
    assert npass > 1
    # Progress runs over the whole sweep, not per pass
    assert reports[0] == (0, len(f)) and reports[-1] == (len(f), len(f))
    assert all(total == len(f) for _, total in reports)
    done = [d for d, _ in reports]
    assert done == sorted(done)
    # Each pass starts where the last one ended (its first report repeats
    # the previous pass's last)
    repeats = sum(reports[i] == reports[i - 1]
                  for i in range(1, len(reports)))
    assert repeats == npass - 1
    with pytest.raises(TypeError, match = 'progress'):
        await vna_sweep(crs, 400e6, 2400e6, -50., 10, step = 2e-5,
                        verbose = False, progress = 1)


@pytest.mark.asyncio
async def test_vna_sweep_restores_zones_on_failure():
    crs = FakeVnaCRS(modules = (1,), fail_pass = 1)
    with pytest.raises(RuntimeError, match = 'sweep failed'):
        await vna_sweep(crs, 2.6e9, 3.6e9, -50., 10, step = 2e-5,
                        verbose = False)
    assert crs.module_cfg[1]['nyquist_zone'] == 1


@pytest.mark.asyncio
async def test_vna_sweep_cancelled_in_a_pass():
    """
    Cancelling from the progress callback stops the sweep at that point:
    the tones are cleared, the zones restored, and no later pass runs.
    """
    crs = FakeVnaCRS(modules = (1,))
    reports = []
    stop_at = 3

    def progress(done, total):
        """
        Record a report.

        Parameters:
        done, total (int): points taken and total.
        """
        reports.append(done)

    with pytest.raises(SweepCancelled):
        await vna_sweep(crs, 2.6e9, 3.6e9, -50., 10, step = 2e-5,
                        verbose = False, progress = progress,
                        cancelled = lambda: len(reports) > stop_at)
    # Stopped during the first pass's points (the fake reports before it
    # records the call), after the report that tripped the flag
    assert len(reports) == stop_at + 1 and crs.sweep_calls == []
    assert ('clear_tones', {}) in crs.calls
    assert crs.module_cfg[1]['nyquist_zone'] == 1
    with pytest.raises(TypeError, match = 'cancelled'):
        await vna_sweep(crs, 2.6e9, 3.6e9, -50., 10, step = 2e-5,
                        verbose = False, cancelled = 1)


@pytest.mark.asyncio
async def test_vna_sweep_cancelled_between_passes():
    """
    The flag is also checked before each pass (without a progress
    callback), and a sweep that is never cancelled is unaffected.
    """
    crs = FakeVnaCRS(modules = (1,))
    with pytest.raises(SweepCancelled):
        await vna_sweep(crs, 400e6, 2400e6, -50., 10, step = 2e-5,
                        verbose = False,
                        cancelled = lambda: len(crs.sweep_calls) >= 1)
    assert len(crs.sweep_calls) == 1
    assert crs.module_cfg[1]['nyquist_zone'] == 1
    crs = FakeVnaCRS(modules = (1,))
    f, _, bands = await vna_sweep(crs, 400e6, 2400e6, -50., 10, step = 2e-5,
                                  verbose = False, cancelled = lambda: False)
    assert len(crs.sweep_calls) == bands['chunk_passes'].max() + 1
    assert len(f) == sum(c['frequencies'].size for c in crs.sweep_calls)
    assert ('clear_tones', {}) not in crs.calls


@pytest.mark.asyncio
async def test_vna_sweep_extended_bw_and_npoints():
    crs = FakeVnaCRS(modules = (1,), bw = 600e6)
    _, _, bands = await vna_sweep(crs, 1e9, 2.1e9, -50., 5, step = 1e-4,
                                  npoints_per_ch = 20, verbose = False)
    assert np.diff(bands['chunk_edges']).max() > 500e6
    assert crs.sweep_calls[0]['frequencies'].shape[1] == 20


@pytest.mark.asyncio
async def test_vna_sweep_to_store(tmp_path):
    root = dl.create_store(tmp_path / 'dev.zarr', 'devA', 'cd1')
    crs = FakeVnaCRS(modules = (1, 2), fres = (1.2e9,))
    k, f, z = await vna_sweep_to_store(crs, root, 400e6, 2400e6, -50., 10,
                                       step = 2e-5, verbose = False)
    assert k == 0
    grp = root['vna0']
    assert grp.attrs['kind'] == 'raw'
    assert grp.attrs['amplitude'] == -50. and grp.attrs['nsamps'] == 10
    assert grp.attrs['step'] == 2e-5 and grp.attrs['downward'] is True
    assert grp.attrs['spacing'] == 'geometric'
    assert grp.attrs['npoints_per_ch'] == \
        crs.sweep_calls[0]['frequencies'].shape[1]
    assert grp.attrs['acquisition']['clock_source'] == 'VCXO'
    cfg = grp['crs_config'].attrs['config']
    assert cfg['modules'] == [1, 2] and 'ncos' not in cfg
    assert 'nyquist_zone' not in cfg['module_settings']['1']
    assert root.attrs['crs_serial_number'] == 46
    f2, z2, bands = dl.load_vna_sweep(root, bands = True)
    np.testing.assert_array_equal(f2, f)
    np.testing.assert_array_equal(z2, z)
    nb = len(bands['chunk_ncos'])
    np.testing.assert_array_equal(bands['chunk_modules'],
                                  [1, 2] * (nb // 2) + [1] * (nb % 2))
    np.testing.assert_array_equal(bands['chunk_zones'], [1] * nb)
    # A second sweep is sweep1; sweep0 is untouched
    k, _, _ = await vna_sweep_to_store(crs, root, 1e9, 1.4e9, -40., 10,
                                       step = 2e-5, verbose = False)
    assert k == 1 and root['vna0'].attrs['amplitude'] == -50.


@pytest.mark.asyncio
async def test_vna_sweep_to_store_failure_leaves_nothing(tmp_path):
    root = dl.create_store(tmp_path / 'dev.zarr', 'devA', 'cd1')
    crs = FakeVnaCRS(modules = (1,), fail_pass = 2)
    with pytest.raises(RuntimeError, match = 'sweep failed'):
        await vna_sweep_to_store(crs, root, 400e6, 2400e6, -50., 10,
                                 step = 2e-5, verbose = False)
    assert dl.list_indices(root, 'vna') == []
    # The next attempt reuses index 0
    crs.fail_pass = None
    k, _, _ = await vna_sweep_to_store(crs, root, 400e6, 900e6, -50., 10,
                                       step = 2e-5, verbose = False)
    assert k == 0


@pytest.mark.asyncio
async def test_vna_sweep_to_store_refuses_other_crs(tmp_path):
    root = dl.create_store(tmp_path / 'dev.zarr', 'devA', 'cd1')
    await vna_sweep_to_store(FakeVnaCRS(), root, 1e9, 1.2e9, -50., 10,
                             step = 2e-5, verbose = False)
    for attr, value in (('serial_number', 12), ('rfmux_version', '9.9')):
        other = FakeVnaCRS()
        setattr(other, attr, value)
        with pytest.raises(RuntimeError, match = 'new store'):
            await vna_sweep_to_store(other, root, 1e9, 1.2e9, -50., 10,
                                     step = 2e-5, verbose = False)
        assert other.sweep_calls == []
    assert dl.list_indices(root, 'vna') == [0]
