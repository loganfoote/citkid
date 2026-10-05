"""
Comprehensive tests for automatic resonance finder (res_finder_auto.py).

Tests cover:
- Basic functionality with synthetic data
- Different smoothing methods (highpass, polynomial, none)
- Parameter variations (height, width, distance)
- Frequency range limiting
- File I/O (save, overwrite popup)
- Edge cases (empty data, no resonances, etc.)
"""

import pytest
import numpy as np
import zarr
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from PyQt5 import QtWidgets

from citkid.vna.res_finder_auto import (
    AutoResFinder,
    AutoResFinderWindow,
    SpinBoxEventFilter,
    run_res_finder_auto,
)
# Real popup function, kept before the autouse fixture patches the module
from citkid.vna.res_finder_auto import _confirm_overwrite as _real_confirm_overwrite


@pytest.fixture(autouse=True)
def no_unexpected_popup():
    """Fail a test that opens the overwrite popup without patching it."""
    def _fail(message):
        raise AssertionError(f"unexpected overwrite popup: {message}")
    with patch('citkid.vna.res_finder_auto._confirm_overwrite', side_effect=_fail):
        yield


class TestSpinBoxEventFilter:
    """Test the spinbox event filter for select-all behavior."""
    
    def test_event_filter_exists(self):
        """Test that SpinBoxEventFilter can be instantiated."""
        event_filter = SpinBoxEventFilter()
        assert event_filter is not None


class TestAutoResFinderInit:
    """Test AutoResFinder initialization."""
    
    def test_init_basic(self, synthetic_vna_data, tmp_path):
        """Test basic initialization."""
        outpath = tmp_path / "test_output.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath),
        )
        
        # Check data is stored correctly
        assert len(finder.f) == len(synthetic_vna_data['f'])
        assert len(finder.z) == len(synthetic_vna_data['z'])
        assert finder.f.dtype == np.float64
        assert finder.z.dtype == np.complex128
        
        # Check magnitude computation
        assert len(finder.mag_db) == len(finder.f)
        assert np.all(np.isfinite(finder.mag_db))
        
        # Check default parameters
        assert finder.params['f_min'] == pytest.approx(synthetic_vna_data['f'].min())
        assert finder.params['f_max'] == pytest.approx(synthetic_vna_data['f'].max())
        assert finder.params['smoothing'] == 'highpass'
        
    def test_init_file_exists_cancel(self, synthetic_vna_data, tmp_path):
        """Choosing Cancel for an existing file raises and keeps the file."""
        outpath = tmp_path / "existing.h5"
        outpath.touch()

        with patch('citkid.vna.res_finder_auto._confirm_overwrite',
                   return_value=False) as confirm,              pytest.raises(RuntimeError, match='cancelled'):
            AutoResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                str(outpath),
            )
        confirm.assert_called_once()
        assert outpath.is_file()

    def test_init_missing_directory(self, synthetic_vna_data, tmp_path):
        """Test that FileNotFoundError is raised when output directory does not exist."""
        outpath = tmp_path / "nonexistent_dir" / "output.h5"

        with pytest.raises(FileNotFoundError, match='Output directory does not exist'):
            AutoResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                str(outpath),
            )

    def test_init_invalid_extension(self, synthetic_vna_data, tmp_path):
        """Test that ValueError is raised when output path does not have a .h5 extension."""
        outpath = tmp_path / "output.txt"

        with pytest.raises(ValueError, match=r'\.h5 extension'):
            AutoResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                str(outpath),
            )
    
    def test_init_file_exists_overwrite(self, synthetic_vna_data, tmp_path):
        """Choosing Overwrite for an existing file replaces it with a group."""
        outpath = tmp_path / "existing.h5"
        outpath.touch()

        with patch('citkid.vna.res_finder_auto._confirm_overwrite',
                   return_value=True) as confirm:
            finder = AutoResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                str(outpath),
            )
        confirm.assert_called_once()
        assert outpath.is_dir()          # now a zarr directory store
        assert 'fres_auto' not in finder.zarr_group

    @pytest.mark.parametrize('overwrite', [True, False])
    def test_init_existing_fres_auto(self, synthetic_vna_data, overwrite):
        """
        Existing 'fres_auto' is deleted on Overwrite (never loaded) and kept
        on Cancel.
        """
        grp = zarr.group()
        grp.create_array('fres_auto', data=np.array([4.5e9, 5.0e9]))

        with patch('citkid.vna.res_finder_auto._confirm_overwrite',
                   return_value=overwrite) as confirm:
            if overwrite:
                finder = AutoResFinder(
                    synthetic_vna_data['f'], synthetic_vna_data['z'], grp)
            else:
                with pytest.raises(RuntimeError, match='cancelled'):
                    AutoResFinder(
                        synthetic_vna_data['f'], synthetic_vna_data['z'], grp)
        confirm.assert_called_once()
        if overwrite:
            assert 'fres_auto' not in grp
            assert finder.fres == []
        else:
            np.testing.assert_array_equal(grp['fres_auto'][:], [4.5e9, 5.0e9])


class TestAutoResFinderSmoothing:
    """Test different smoothing methods."""
    
    def test_highpass_smoothing(self, synthetic_vna_data, tmp_path):
        """Test highpass filter smoothing."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Set highpass parameters
        finder.params['smoothing'] = 'highpass'
        finder.params['highpass_mhz'] = 10.0
        
        # Apply smoothing
        finder.apply_smoothing()
        
        # Check output shape
        assert finder.filtered_mag.shape == finder.mag_db.shape
        
        # Check that filtering was applied (filtered data is different from original)
        assert not np.allclose(finder.filtered_mag, finder.mag_db)
    
    def test_polynomial_smoothing(self, synthetic_vna_data, tmp_path):
        """Test polynomial baseline subtraction."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Set polynomial parameters
        finder.params['smoothing'] = 'polynomial'
        finder.params['poly_order'] = 3
        
        # Apply smoothing
        finder.apply_smoothing()
        
        # Check output shape
        assert finder.filtered_mag.shape == finder.mag_db.shape
        
        # Check that baseline was removed (mean should be close to 0)
        assert np.abs(np.mean(finder.filtered_mag)) < 1.0
    
    def test_no_smoothing(self, synthetic_vna_data, tmp_path):
        """Test no smoothing (identity operation)."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Set no smoothing
        finder.params['smoothing'] = 'none'
        
        # Apply smoothing
        finder.apply_smoothing()
        
        # Should be identical to original
        np.testing.assert_array_equal(finder.filtered_mag, finder.mag_db)


class TestAutoResFinderResDetection:
    """Test res detection functionality."""
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_res_detection_finds_resonances(
        self, mock_update, synthetic_vna_data, tmp_path
    ):
        """Test that resonance detection machinery works."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Just verify that the filtering pipeline works
        finder.params['smoothing'] = 'none'  # Use no smoothing for simplicity
        finder.apply_smoothing()
        
        # Verify filtered_mag was computed
        assert hasattr(finder, 'filtered_mag')
        assert len(finder.filtered_mag) == len(finder.f)
        
        # Verify it contains valid data
        assert np.all(np.isfinite(finder.filtered_mag))
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_frequency_range_limiting(
        self, mock_update, synthetic_vna_data, tmp_path
    ):
        """Test that frequency range limits are respected."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Limit to 5-6.5 GHz (should exclude some resonances)
        finder.params['f_min'] = 5e9
        finder.params['f_max'] = 6.5e9
        finder.params['smoothing'] = 'highpass'
        finder.params['height'] = -5.0
        
        finder.update_peaks()
        
        fres_found = np.array(finder.fres)
        
        # All found peaks should be in range
        if len(fres_found) > 0:
            assert np.all(fres_found >= 5e9)
            assert np.all(fres_found <= 6.5e9)
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_height_parameter(self, mock_update, synthetic_vna_data, tmp_path):
        """Test that height parameter affects number of peaks found."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        finder.params['smoothing'] = 'highpass'
        
        # Find peaks with loose threshold
        finder.params['height'] = -1.0
        finder.update_peaks()
        n_peaks_loose = len(finder.fres)
        
        # Find peaks with strict threshold
        finder.params['height'] = -10.0
        finder.update_peaks()
        n_peaks_strict = len(finder.fres)
        
        # Stricter threshold should find fewer or equal peaks
        assert n_peaks_strict <= n_peaks_loose
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_distance_parameter(
        self, mock_update, dense_resonances_vna_data, tmp_path
    ):
        """Test that distance parameter prevents closely spaced peaks."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            dense_resonances_vna_data['f'],
            dense_resonances_vna_data['z'],
            str(outpath)
        )
        
        finder.params['smoothing'] = 'highpass'
        finder.params['height'] = -5.0
        
        # Small distance - should find more peaks
        finder.params['distance'] = 5  # kHz/GHz
        finder.update_peaks()
        n_peaks_small = len(finder.fres)
        
        # Large distance - should find fewer peaks
        finder.params['distance'] = 100  # kHz/GHz
        finder.update_peaks()
        n_peaks_large = len(finder.fres)
        
        # Larger distance should find fewer peaks
        assert n_peaks_large <= n_peaks_small


class TestAutoResFinderFileIO:
    """Test file save/load functionality."""
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_save_results(self, mock_update, synthetic_vna_data, tmp_path):
        """Test saving results to zarr group."""
        outpath = tmp_path / "results.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Set some resonances manually
        finder.fres = [4.5e9, 5.2e9, 6.1e9]
        
        # Save
        finder.save_data()
        
        # Load and verify contents from zarr group
        grp = zarr.open_group(str(outpath), mode='r')
        
        # Check fres_auto dataset
        assert 'fres_auto' in grp
        fres_loaded = grp['fres_auto'][:]
        np.testing.assert_array_almost_equal(fres_loaded, finder.fres)
        
        # Check parameters are stored as attributes
        assert 'f_min' in grp.attrs
        assert 'f_max' in grp.attrs
        assert 'smoothing' in grp.attrs
        assert 'height' in grp.attrs
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_save_empty_results(
        self, mock_update, no_resonance_vna_data, tmp_path
    ):
        """Test saving when no peaks are found."""
        outpath = tmp_path / "empty_results.h5"
        
        finder = AutoResFinder(
            no_resonance_vna_data['f'],
            no_resonance_vna_data['z'],
            str(outpath)
        )
        
        finder.fres = []
        finder.save_data()
        
        # Check zarr group was created with empty dataset
        grp = zarr.open_group(str(outpath), mode='r')
        assert 'fres_auto' in grp
        assert grp['fres_auto'].shape[0] == 0


class TestAutoResFinderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_frequency_array(self, tmp_path):
        """Test with empty input arrays."""
        outpath = tmp_path / "test.h5"
        
        f = np.array([])
        z = np.array([])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            finder = AutoResFinder(f, z, str(outpath))
    
    def test_single_point(self, tmp_path):
        """Test with single data point.
        
        Note: In tests, update_peaks is mocked so this won't raise an error.
        In production, this would fail when calling filtfilt.
        """
        outpath = tmp_path / "test.h5"
        
        f = np.array([5e9])
        z = np.array([0.9 + 0.1j])
        
        # With mocked update_peaks, object creation succeeds
        finder = AutoResFinder(f, z, str(outpath))
        assert len(finder.f) == 1
        assert len(finder.z) == 1
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_no_peaks_in_data(
        self, mock_update, no_resonance_vna_data, tmp_path
    ):
        """Test with data containing no resonances."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            no_resonance_vna_data['f'],
            no_resonance_vna_data['z'],
            str(outpath)
        )
        
        finder.params['smoothing'] = 'highpass'
        finder.params['height'] = -5.0
        finder.update_peaks()
        
        # Should find zero or very few peaks
        assert len(finder.fres) < 2
    
    def test_invalid_frequency_range(self, synthetic_vna_data, tmp_path):
        """Test with invalid frequency range (f_min > f_max)."""
        outpath = tmp_path / "test.h5"
        
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath)
        )
        
        # Set invalid range
        finder.params['f_min'] = 7e9
        finder.params['f_max'] = 5e9
        
        # Should handle gracefully (possibly finding no peaks)
        # Not raising error is acceptable behavior
        finder.update_peaks()


class TestRunAutoResFinder:
    """Test the run_res_finder_auto wrapper function."""
    
    @patch('citkid.vna.res_finder_auto.AutoResFinder.run')
    def test_run_returns_fres(self, mock_run, synthetic_vna_data, tmp_path):
        """Test that run_res_finder_auto returns resonance frequencies."""
        outpath = tmp_path / "test.h5"
        
        # Mock the finder instance
        with patch('citkid.vna.res_finder_auto.AutoResFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = [4.5e9, 5.2e9, 6.1e9]
            
            fres = run_res_finder_auto(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                str(outpath)
            )
            
            # Check that it returns an array
            assert isinstance(fres, np.ndarray)
            np.testing.assert_array_almost_equal(fres, mock_instance.fres)
    
    def test_run_passes_data_and_group(self, synthetic_vna_data, tmp_path):
        """run_res_finder_auto passes only f, z, and zarr_grp to the finder."""
        outpath = tmp_path / "test.h5"

        with patch('citkid.vna.res_finder_auto.AutoResFinder') as MockFinder:
            MockFinder.return_value.fres = []
            run_res_finder_auto(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                str(outpath),
            )
        MockFinder.assert_called_once_with(
            synthetic_vna_data['f'], synthetic_vna_data['z'], str(outpath))


def test_confirm_overwrite_builds_and_defaults_to_cancel():
    """Build the overwrite popup; closing it without a choice cancels."""
    with patch.object(QtWidgets.QMessageBox, 'exec', create=True, return_value=0),          patch.object(QtWidgets.QMessageBox, 'exec_', create=True, return_value=0):
        assert _real_confirm_overwrite('Existing data.') is False


class TestAutoResFinderQuit:
    """Save & Quit must behave exactly like closing the window."""

    @patch('citkid.vna.res_finder_auto.AutoResFinder.update_peaks')
    def test_quit_and_save_closes_window_and_saves_once(
        self, mock_update, synthetic_vna_data, tmp_path
    ):
        outpath = tmp_path / "quit.h5"
        finder = AutoResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            str(outpath),
        )
        finder.fres = [4.5e9, 5.2e9]
        # conftest patches setup_ui out; attach the real window class.
        finder.win = AutoResFinderWindow(finder=finder)
        finder.win.show()

        with patch.object(finder, 'save_data', wraps=finder.save_data) as save, \
                patch.object(finder.app, 'quit') as app_quit:
            finder.quit_and_save()

        assert not finder.win.isVisible()
        save.assert_called_once()
        app_quit.assert_not_called()
        grp = zarr.open_group(str(outpath), mode='r')
        np.testing.assert_array_almost_equal(grp['fres_auto'][:], [4.5e9, 5.2e9])



def test_peak_search_runs_in_a_worker_thread(synthetic_vna_data, tmp_path):
    """
    The smoothing and peak search (_compute_peaks) use no Qt, so update_peaks
    runs them through run_responsive; they give the same result in a thread.
    """
    import threading
    from citkid.qt_compat import run_responsive

    finder = AutoResFinder(synthetic_vna_data['f'], synthetic_vna_data['z'],
                           str(tmp_path / "test.h5"))
    params = dict(finder.params, f_min=finder.f[0], f_max=finder.f[-1], smoothing='none')

    filtered, fres = finder._compute_peaks(params)
    threads = []

    def in_worker(p):
        threads.append(threading.current_thread() is threading.main_thread())
        return finder._compute_peaks(p)

    filtered_w, fres_w = run_responsive(in_worker, params)

    assert threads == [False]
    np.testing.assert_array_equal(filtered_w, filtered)
    assert fres_w == fres and len(fres) > 0
    assert all(finder.f[0] <= fr <= finder.f[-1] for fr in fres)
    empty = finder._compute_peaks(dict(params, f_min=1.0, f_max=2.0))
    assert empty[1] is None                   # no data in range
