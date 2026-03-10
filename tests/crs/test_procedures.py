"""Tests for CRS procedures helpers."""

import numpy as np
import pytest
import zarr
from unittest.mock import Mock, MagicMock
from citkid.crs.procedures import _validate_target_sweep_inputs


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
        'cable_delay': 0.0,
        'verbose': True,
    }


class TestValidateTargetSweepInputs:
    """Tests for _validate_target_sweep_inputs function."""
    
    def test_valid_inputs(self, mock_crs, valid_inputs):
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
    
    def test_dummy_crs_accepted(self, mock_dummy_crs, valid_inputs):
        """Test that DummyCRS instances are accepted."""
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_dummy_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_invalid_crs_type(self, valid_inputs):
        """Test that invalid CRS type raises TypeError."""
        invalid_crs = "not a CRS instance"
        with pytest.raises(
            TypeError, match = "crs must be an instance of CRS class"
            ):
            _validate_target_sweep_inputs(invalid_crs, **valid_inputs)
    
    def test_array_conversion_from_list(self, mock_crs, valid_inputs):
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
    
    def test_fres_not_1d(self, mock_crs, valid_inputs):
        """Test that 2D fres array raises ValueError."""
        valid_inputs['fres'] = np.array([[1e9, 2e9], [3e9, 4e9]])
        with pytest.raises(ValueError, match = "fres must be a 1D array"):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_array_shape_mismatch(self, mock_crs, valid_inputs):
        """Test that mismatched array shapes raise ValueError."""
        valid_inputs['ares'] = np.array([0.1, 0.2])  # Different length
        with pytest.raises(ValueError, match = "must have the same shape"):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_gain_span_factor_too_small(self, mock_crs, valid_inputs):
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
    
    def test_gain_span_factor_int_accepted(self, mock_crs, valid_inputs):
        """Test that integer gain_span_factor is accepted."""
        valid_inputs['gain_span_factor'] = 10
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_npoints_fine_invalid(self, mock_crs, valid_inputs):
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
    
    def test_npoints_fine_none_accepted(self, mock_crs, valid_inputs):
        """Test that None for npoints_fine is accepted."""
        valid_inputs['npoints_fine'] = None
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_npoints_gain_invalid(self, mock_crs, valid_inputs):
        """Test that invalid npoints_gain raises ValueError."""
        valid_inputs['npoints_gain'] = 0
        with pytest.raises(
            ValueError, match = "npoints_gain must be a positive integer"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_npoints_gain_none_accepted(self, mock_crs, valid_inputs):
        """Test that None for npoints_gain is accepted."""
        valid_inputs['npoints_gain'] = None
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_npoints_rough_invalid(self, mock_crs, valid_inputs):
        """Test that invalid npoints_rough raises ValueError."""
        valid_inputs['npoints_rough'] = 0
        with pytest.raises(
            ValueError, match = "npoints_rough must be a positive integer"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_npoints_rough_none_accepted(self, mock_crs, valid_inputs):
        """Test that None for npoints_rough is accepted."""
        valid_inputs['npoints_rough'] = None
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_nsamps_invalid(self, mock_crs, valid_inputs):
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
    
    def test_fres_update_method_invalid(self, mock_crs, valid_inputs):
        """Test that invalid fres_update_method raises ValueError."""
        valid_inputs['fres_update_method'] = 'invalid_method'
        with pytest.raises(ValueError, match = "fres_update_method must be"):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_fres_update_method_valid_options(self, mock_crs, valid_inputs):
        """Test all valid fres_update_method options."""
        for method in ['distance', 'spacing', 'minS21']:
            valid_inputs['fres_update_method'] = method
            fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
                mock_crs, **valid_inputs
            )
            assert isinstance(fres, np.ndarray)
    
    def test_cable_delay_negative(self, mock_crs, valid_inputs):
        """Test that negative cable_delay raises ValueError."""
        valid_inputs['cable_delay'] = -1.0
        with pytest.raises(
            ValueError, match = "cable_delay must be a positive number"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_cable_delay_zero_accepted(self, mock_crs, valid_inputs):
        """Test that cable_delay = 0 is accepted."""
        valid_inputs['cable_delay'] = 0.0
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_verbose_not_bool(self, mock_crs, valid_inputs):
        """Test that non-boolean verbose raises TypeError."""
        valid_inputs['verbose'] = "True"
        with pytest.raises(TypeError, match = "verbose must be a boolean"):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_grp_not_zarr_group(self, mock_crs, valid_inputs):
        """Test that non-Zarr group raises TypeError."""
        valid_inputs['grp'] = {}  # Dict instead of Zarr group
        with pytest.raises(
            TypeError, match = "grp must be a zarr Group instance"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_zarr_group_with_existing_fres(self, mock_crs, valid_inputs):
        """Test that existing 'fres' dataset in group raises ValueError."""
        grp = zarr.group()
        grp.create_array(name = 'fres', data = np.array([1, 2, 3]))
        valid_inputs['grp'] = grp
        
        with pytest.raises(
            ValueError, match = "already contains dataset 'fres'"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_zarr_group_with_existing_rough_sweep_data(self, mock_crs, valid_inputs):
        """Test that existing rough sweep data raises ValueError."""
        grp = zarr.group()
        grp.create_array(name = 's21_rough_f', data = np.array([1, 2, 3]))
        valid_inputs['grp'] = grp
        valid_inputs['npoints_rough'] = 100
        
        with pytest.raises(
            ValueError, match = "already contains dataset 's21_rough_f'"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_zarr_group_with_existing_gain_sweep_data(self, mock_crs, valid_inputs):
        """Test that existing gain sweep data raises ValueError."""
        grp = zarr.group()
        grp.create_array(name = 's21_gain_z', data = np.array([1+1j, 2+2j]))
        valid_inputs['grp'] = grp
        valid_inputs['npoints_gain'] = 50
        
        with pytest.raises(
            ValueError, match = "already contains dataset 's21_gain_z'"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_zarr_group_with_existing_fine_sweep_data(self, mock_crs, 
                                                      valid_inputs):
        """Test that existing fine sweep data raises ValueError."""
        grp = zarr.group()
        grp.create_array(name = 's21_fine_f', data = np.array([1, 2, 3]))
        valid_inputs['grp'] = grp
        valid_inputs['npoints_fine'] = 500
        
        with pytest.raises(
            ValueError, match = "already contains dataset 's21_fine_f'"
            ):
            _validate_target_sweep_inputs(mock_crs, **valid_inputs)
    
    def test_zarr_group_conflicts_skipped_when_npoints_none(self, mock_crs, 
                                                            valid_inputs):
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
    
    def test_numpy_integer_types_accepted(self, mock_crs, valid_inputs):
        """Test that numpy integer types are accepted for integer parameters."""
        valid_inputs['npoints_fine'] = np.int64(500)
        valid_inputs['npoints_gain'] = np.int32(50)
        valid_inputs['nsamps'] = np.int16(100)
        
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_numpy_float_types_accepted(self, mock_crs, valid_inputs):
        """Test that numpy float types are accepted for float parameters."""
        valid_inputs['gain_span_factor'] = np.float64(10.0)
        valid_inputs['cable_delay'] = np.float32(0.5)
        
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        assert isinstance(fres, np.ndarray)
    
    def test_dtype_conversion_from_int_to_float64(self, mock_crs, valid_inputs):
        """
        Test that integer arrays are converted to float64 for frequency arrays.
        """
        valid_inputs['fres'] = np.array([1000000000, 2000000000, 3000000000], 
                                        dtype = np.int64)
        
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        
        assert fres.dtype == np.float64
    
    def test_dtype_conversion_from_int32_to_int32(self, mock_crs, valid_inputs):
        """Test that res_idxs maintains int32 dtype."""
        valid_inputs['res_idxs'] = np.array([0, 1, 2], dtype = np.int64)
        
        fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
            mock_crs, **valid_inputs
        )
        
        assert res_idxs.dtype == np.int32
    
    def test_negative_res_idxs_accepted(self, mock_crs, valid_inputs):
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
    async def mock_sweep_qres(fres, ares, qres, npoints, nsamps, ch_map, 
                               dec_grp, verbose, pbar_description):
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
    
    def mock_update(f, z, fres, qres, fcal_indices, method, cable_delay, plotq):
        """Mock update that slightly shifts frequencies."""
        fres_updated = fres.copy()
        # Simulate frequency update by shifting slightly
        fres_updated = fres_updated * 1.0001
        return fres_updated
    
    with patch(
        'citkid.crs.procedures.update_fres', side_effect = mock_update
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
        'cable_delay': 0.0,
        'verbose': True,
    }


@pytest.mark.asyncio
class TestTargetSweep:
    """Tests for target_sweep function."""
    
    async def test_all_sweeps_enabled(self, mock_crs_for_target_sweep, 
                                       target_sweep_inputs, 
                                       mock_util_write_system_cfg,
                                       mock_update_fres):
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
    
    async def test_sweep_order(self, mock_crs_for_target_sweep, 
                               target_sweep_inputs,
                               mock_util_write_system_cfg, mock_update_fres):
        """Test that sweeps are executed in correct order: rough, gain, fine."""
        from citkid.crs.procedures import target_sweep
        
        await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
        
        # Get pbar_description from each call
        call_descs = [call.kwargs['pbar_description'] 
                for call in mock_crs_for_target_sweep.sweep_qres.call_args_list]
        
        assert call_descs == ['Rough sweep', 'Gain sweep', 'Fine sweep']
    
    async def test_rough_sweep_parameters(self, mock_crs_for_target_sweep, 
                                          target_sweep_inputs, 
                                          mock_util_write_system_cfg,
                                          mock_update_fres):
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
    
    async def test_gain_sweep_parameters(self, mock_crs_for_target_sweep, 
                                         target_sweep_inputs, mock_util_write_system_cfg,
                                         mock_update_fres):
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
    
    async def test_fine_sweep_parameters(self, mock_crs_for_target_sweep, 
                                         target_sweep_inputs,
                                         mock_util_write_system_cfg,
                                         mock_update_fres):
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
    
    async def test_only_rough_sweep(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_only_gain_sweep(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_only_fine_sweep(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_rough_and_gain_only(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_gain_and_fine_only(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_ch_map_provided(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_output_data_shapes(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_zarr_group_structure(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
        assert grpr.attrs['cable_delay'] == target_sweep_inputs['cable_delay']
        
        # Check gain and fine sweep subgroups
        for sweep_name in ['gain_sweep', 'fine_sweep']:
            grps = grp[sweep_name]
            assert 'f' in grps
            assert 'z' in grps
    
    async def test_zarr_saved_values(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_update_fres_called_with_correct_params(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
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
        
        # Check fcal_indices (should be empty for res_idxs = [0, 1, 2])
        np.testing.assert_array_equal(kwargs['fcal_indices'], np.array([]))
        
        # Check other kwargs
        assert kwargs['method'] == target_sweep_inputs['fres_update_method']
        assert kwargs['cable_delay'] == target_sweep_inputs['cable_delay']
        assert kwargs['plotq'] is False
    
    async def test_update_fres_with_calibration_tones(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres
            ):
        """
        Test that fcal_indices correctly identifies calibration tones 
        (negative res_idxs).
        """
        from citkid.crs.procedures import target_sweep
        
        # Add calibration tones (negative indices)
        target_sweep_inputs['res_idxs'] = np.array([-1, 0, -2, 1, 2])
        target_sweep_inputs['fres'] = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
        target_sweep_inputs['ares'] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        target_sweep_inputs['qres'] = np.array([1000.0, 2000.0, 3000.0, 
                                                4000.0, 5000.0])
        
        await target_sweep(mock_crs_for_target_sweep, **target_sweep_inputs)
        
        # Check fcal_indices identifies indices 0 and 2
        call_args = mock_update_fres.call_args
        kwargs = call_args[1]
        np.testing.assert_array_equal(kwargs['fcal_indices'], np.array([0, 2]))
    
    async def test_all_fres_update_methods(self, mock_crs_for_target_sweep, 
                                           target_sweep_inputs,
                                           mock_util_write_system_cfg, 
                                           mock_update_fres):
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
    
    async def test_no_sweeps_enabled(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_different_npoints_values(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_different_gain_span_factors(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_single_resonator(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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
    
    async def test_many_resonators(
            self, mock_crs_for_target_sweep, target_sweep_inputs,
            mock_util_write_system_cfg, mock_update_fres):
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

class TestSaveSweepData:
    """Tests for _save_sweep_data function."""
    
    def test_save_without_prefix(self):
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
    
    def test_save_with_prefix(self):
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
    
    def test_dtype_conversion_from_list(self):
        """Test that lists are converted to correct dtypes."""
        from citkid.crs.procedures import _save_sweep_data
        
        grp = zarr.group()
        f = [[1e9, 1.1e9], [2e9, 2.1e9]]
        z = [[0.5+0.1j, 0.6+0.2j], [0.8+0.4j, 0.9+0.5j]]
        
        _save_sweep_data(grp, '', f, z)
        
        # Check dtypes after conversion
        assert grp['f'][:].dtype == np.float64
        assert grp['z'][:].dtype == np.complex128
    
    def test_dtype_conversion_from_int(self):
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
    
    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        from citkid.crs.procedures import _save_sweep_data
        
        grp = zarr.group()
        f = np.array([[1e9, 1.1e9], [2e9, 2.1e9]])
        z = np.array([[0.5+0.1j, 0.6+0.2j]])  # Different shape
        
        with pytest.raises(
            ValueError, match = "f and z must have the same shape"
            ):
            _save_sweep_data(grp, '', f, z)
    
    def test_invalid_grp_type(self):
        """Test that non-zarr Group raises TypeError."""
        from citkid.crs.procedures import _save_sweep_data
        
        f = np.array([[1e9, 1.1e9]])
        z = np.array([[0.5+0.1j, 0.6+0.2j]])
        
        # Try with dict instead of zarr group
        with pytest.raises(
            TypeError, match = "grp must be a zarr Group instance"
            ):
            _save_sweep_data({}, '', f, z)
    
    def test_invalid_prefix_type(self):
        """Test that non-string prefix raises TypeError."""
        from citkid.crs.procedures import _save_sweep_data
        
        grp = zarr.group()
        f = np.array([[1e9, 1.1e9]])
        z = np.array([[0.5+0.1j, 0.6+0.2j]])
        
        # Try with int prefix
        with pytest.raises(TypeError, match = "prefix must be a string"):
            _save_sweep_data(grp, 123, f, z)
    
    def test_chunking_strategy(self):
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
    
    def test_1d_arrays(self):
        """Test with 1D arrays (single resonator case)."""
        from citkid.crs.procedures import _save_sweep_data
        
        grp = zarr.group()
        f = np.array([1e9, 1.1e9, 1.2e9])
        z = np.array([0.5+0.1j, 0.6+0.2j, 0.7+0.3j])
        
        # Should raise an error because shape[1] doesn't exist for 1D
        # Actually, let me check what happens
        with pytest.raises((IndexError, AttributeError)):
            _save_sweep_data(grp, '', f, z)
    
    def test_multiple_prefixes_in_same_group(self):
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
    
    def test_empty_prefix_string(self):
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
    
    def test_large_dataset(self):
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
    
    def test_single_resonator_single_point(self):
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
    
    def test_complex_dtypes_preserved(self):
        """Test that complex dtypes are converted to complex128."""
        from citkid.crs.procedures import _save_sweep_data
        
        grp = zarr.group()
        f = np.array([[1e9, 1.1e9]])
        z = np.array([[0.5+0.1j, 0.6+0.2j]], dtype = np.complex64)
        
        _save_sweep_data(grp, '', f, z)
        
        # Should be converted to complex128
        assert grp['z'][:].dtype == np.complex128

    def test_shards_cover_full_shape(self):
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

    def test_shards_with_prefix(self):
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

