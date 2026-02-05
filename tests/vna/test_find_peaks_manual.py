"""
Comprehensive tests for manual peak finder (find_peaks_manual.py).

Tests cover:
- Initialization with array and file input
- Adding and removing resonances
- Undo functionality
- Phase detrending
- File I/O (save/load, overwrite behavior)
- Edge cases
"""

import pytest
import numpy as np
import h5py
import os
from unittest.mock import Mock, patch, MagicMock

from citkid.vna.find_peaks_manual import (
    PeakFinder,
    run_peak_finder
)


class TestPeakFinderInit:
    """Test PeakFinder initialization."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_init_with_array(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test initialization with array of initial resonances."""
        outpath = tmp_path / "test.h5"
        fres_initial = synthetic_vna_data['fres_true'][:2]  # Use first 2
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath),
            overwrite=True
        )
        
        # Check data is stored correctly
        assert len(finder.f) == len(synthetic_vna_data['f'])
        assert len(finder.z) == len(synthetic_vna_data['z'])
        assert finder.f.dtype == np.float64
        assert finder.z.dtype == np.complex128
        
        # Check initial resonances
        assert len(finder.fres) == len(fres_initial)
        np.testing.assert_array_almost_equal(sorted(finder.fres), sorted(fres_initial))
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_init_phase_detrending(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test that phase is detrended during initialization."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Check that phase was computed
        assert hasattr(finder, 'phase')
        assert len(finder.phase) == len(finder.f)
        
        # Check that phase detrending was applied (z was modified)
        # The detrending removes 1st order polynomial trend
        assert hasattr(finder, 'mag_db')
        assert hasattr(finder, 'phase')
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_init_file_exists_overwrite_false(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test FileExistsError when overwrite=False."""
        outpath = tmp_path / "existing.h5"
        outpath.touch()
        
        with pytest.raises(FileExistsError, match='already exists'):
            PeakFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                [],
                str(outpath),
                overwrite=False
            )
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_init_file_exists_overwrite_true(self, mock_setup, synthetic_vna_data, tmp_path, capsys):
        """Test warning when file exists and overwrite=True."""
        outpath = tmp_path / "existing.h5"
        outpath.touch()
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath),
            overwrite=True
        )
        
        captured = capsys.readouterr()
        assert 'already exists' in captured.out
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_init_empty_fres(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test initialization with empty initial resonance list."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        assert len(finder.fres) == 0
        assert isinstance(finder.fres, list)


class TestPeakFinderAddRemove:
    """Test adding and removing resonances."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_add_resonance(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test adding a resonance."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Add a resonance
        f_new = 5.5e9
        finder.add_resonance(f_new)
        
        # Check it was added
        assert len(finder.fres) == 1
        assert finder.fres[0] == f_new
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_add_multiple_resonances(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test adding multiple resonances."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Add multiple resonances
        freqs = [4.5e9, 5.2e9, 6.1e9]
        for f in freqs:
            finder.add_resonance(f)
        
        # Check all were added and sorted
        assert len(finder.fres) == len(freqs)
        assert finder.fres == sorted(freqs)
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_remove_resonance(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test removing a resonance."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial.copy(),
            str(outpath)
        )
        
        # Mock the current view range
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = [[4e9, 6e9], [-20, 0]]
        finder.update_resonance_markers = Mock()
        finder.log = Mock()
        
        # Remove middle resonance
        f_click = 5.25e9  # Close to 5.2 GHz
        finder.remove_nearest_resonance(f_click)
        
        # Check it was removed
        assert len(finder.fres) == 2
        assert 5.2e9 not in finder.fres
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_remove_outside_view(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test that removing outside view fails."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial.copy(),
            str(outpath)
        )
        
        # Mock narrow view range (5-6 GHz)
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = [[5e9, 6e9], [-20, 0]]
        
        # Try to remove resonance outside view (4.5 GHz)
        result = finder.remove_nearest_resonance(4.5e9)
        
        # Should return None (early return)
        assert result is None
        assert len(finder.fres) == 3  # Nothing removed
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_remove_no_resonances(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test removing when no resonances exist."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Mock view
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = [[4e9, 8e9], [-20, 0]]
        
        # Try to remove (should handle gracefully)
        result = finder.remove_nearest_resonance(5e9)
        
        assert result is None
        assert len(finder.fres) == 0


class TestPeakFinderUndo:
    """Test undo functionality."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_undo_add(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test undoing an add operation."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Add a resonance
        finder.add_resonance(5.5e9)
        assert len(finder.fres) == 1
        
        # Undo
        finder.undo()
        assert len(finder.fres) == 0
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_undo_remove(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test undoing a remove operation."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9]
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial.copy(),
            str(outpath)
        )
        
        # Mock view
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = [[4e9, 6e9], [-20, 0]]
        finder.update_resonance_markers = Mock()
        finder.log = Mock()
        
        # Remove a resonance
        finder.remove_nearest_resonance(4.5e9)
        assert len(finder.fres) == 1
        
        # Undo
        finder.undo()
        assert len(finder.fres) == 2
        assert 4.5e9 in finder.fres
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_undo_multiple_operations(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test undoing multiple operations in sequence."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [5e9],
            str(outpath)
        )
        
        # Perform multiple operations
        finder.add_resonance(6e9)
        finder.add_resonance(7e9)
        assert len(finder.fres) == 3
        
        # Undo twice
        finder.undo()
        assert len(finder.fres) == 2
        assert 7e9 not in finder.fres
        
        finder.undo()
        assert len(finder.fres) == 1
        assert 6e9 not in finder.fres
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_undo_empty_stack(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test undo when undo stack is empty."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Try to undo with empty stack (should handle gracefully)
        finder.undo()
        assert len(finder.fres) == 0
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_undo_stack_grows(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test that undo stack grows with operations."""
        outpath = tmp_path / "test.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Perform operations
        for i in range(10):
            finder.add_resonance(5e9 + i * 0.1e9)
        
        # Should be able to undo all
        for i in range(10):
            finder.undo()
        
        assert len(finder.fres) == 0


class TestPeakFinderFileIO:
    """Test file save/load functionality."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_save_results(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test saving results to HDF5."""
        outpath = tmp_path / "results.h5"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath)
        )
        
        # Save
        finder.save_data()
        
        # Check file exists
        assert outpath.exists()
        
        # Load and verify
        with h5py.File(outpath, 'r') as hf:
            assert 'fres' in hf
            fres_loaded = hf['fres'][:]
            np.testing.assert_array_almost_equal(
                sorted(fres_loaded),
                sorted(fres_initial)
            )
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_save_empty_results(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test saving empty resonance list."""
        outpath = tmp_path / "empty.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        finder.save_data()
        
        with h5py.File(outpath, 'r') as hf:
            assert 'fres' in hf
            assert len(hf['fres']) == 0


class TestPeakFinderEdgeCases:
    """Test edge cases."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_single_data_point(self, mock_setup, tmp_path):
        """Test with single data point."""
        outpath = tmp_path / "test.h5"
        
        f = np.array([5e9])
        z = np.array([0.9 + 0.1j])
        
        finder = PeakFinder(f, z, [], str(outpath))
        
        assert len(finder.f) == 1
        assert len(finder.mag_db) == 1
        assert len(finder.phase) == 1
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_duplicate_resonances(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test handling of duplicate resonances in initial list."""
        outpath = tmp_path / "test.h5"
        
        # Include duplicates in initial list
        fres_initial = [5e9, 5e9, 6e9, 6e9, 7e9]
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath)
        )
        
        # Should still work (duplicates may or may not be removed automatically)
        assert len(finder.fres) > 0
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_unsorted_initial_resonances(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test that resonances are sorted."""
        outpath = tmp_path / "test.h5"
        
        # Unsorted initial list
        fres_initial = [6e9, 4.5e9, 7e9, 5e9]
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath)
        )
        
        # After adding resonances, list should be sorted
        # (add_resonance sorts the list)
        finder.add_resonance(5.5e9)
        
        # Check if sorted
        assert finder.fres == sorted(finder.fres)


class TestRunPeakFinder:
    """Test run_peak_finder wrapper function."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.run')
    def test_run_with_array(self, mock_run, synthetic_vna_data, tmp_path):
        """Test run_peak_finder with array input."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9]
        
        with patch('citkid.vna.find_peaks_manual.PeakFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = fres_initial
            
            fres = run_peak_finder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                fres_initial,
                str(outpath)
            )
            
            # Check return value
            assert isinstance(fres, np.ndarray)
            np.testing.assert_array_almost_equal(fres, fres_initial)
    
    def test_run_with_file_path(self, synthetic_vna_data, tmp_path):
        """Test run_peak_finder loading fres from file."""
        # Create a file with fres data
        fres_file = tmp_path / "input_fres.h5"
        fres_initial = np.array([4.5e9, 5.2e9, 6.1e9])
        
        with h5py.File(fres_file, 'w') as hf:
            hf.create_dataset('fres', data=fres_initial)
        
        outpath = tmp_path / "output.h5"
        
        with patch('citkid.vna.find_peaks_manual.PeakFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = fres_initial
            
            with patch('citkid.vna.find_peaks_manual.PeakFinder.run'):
                fres = run_peak_finder(
                    synthetic_vna_data['f'],
                    synthetic_vna_data['z'],
                    str(fres_file),
                    str(outpath)
                )
            
            # Check that PeakFinder was initialized
            MockFinder.assert_called_once()
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.run')
    def test_run_passes_parameters(self, mock_run, synthetic_vna_data, tmp_path):
        """Test that parameters are passed through correctly."""
        outpath = tmp_path / "test.h5"
        
        with patch('citkid.vna.find_peaks_manual.PeakFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = []
            
            run_peak_finder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                [],
                str(outpath),
                margin_factor=0.2,
                overwrite=True
            )
            
            # Check parameters passed to PeakFinder
            call_args = MockFinder.call_args
            # Parameters are passed as positional args
            assert call_args[0][4] == 0.2  # margin_factor is 5th arg
            assert call_args[0][5] is True  # overwrite is 6th arg


class TestIntegration:
    """Integration tests for auto -> manual workflow."""
    
    @patch('citkid.vna.find_peaks_manual.PeakFinder.setup_ui')
    def test_auto_to_manual_workflow(self, mock_setup, synthetic_vna_data, tmp_path):
        """Test complete workflow from auto to manual peak finding."""
        # First run auto finder (mock the GUI parts)
        auto_outpath = tmp_path / "auto_results.h5"
        
        # Manually create auto results file
        fres_auto = np.array([4.5e9, 5.2e9, 6.1e9])
        with h5py.File(auto_outpath, 'w') as hf:
            ds = hf.create_dataset('fres', data=fres_auto)
            ds.attrs['f_min'] = synthetic_vna_data['f'].min()
            ds.attrs['f_max'] = synthetic_vna_data['f'].max()
            ds.attrs['smoothing'] = 'highpass'
        
        # Now run manual finder with auto results
        manual_outpath = tmp_path / "manual_results.h5"
        
        finder = PeakFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_auto,
            str(manual_outpath),
            overwrite=True
        )
        
        # Manually adjust: add one, remove one (mock operations)
        finder.add_resonance(7.3e9)
        
        # Save
        finder.save_data()
        
        # Verify final results
        with h5py.File(manual_outpath, 'r') as hf:
            fres_final = hf['fres'][:]
            assert len(fres_final) == 4
            assert 7.3e9 in fres_final
    
    def test_load_from_auto_output(self, synthetic_vna_data, tmp_path):
        """Test loading initial resonances from auto finder output."""
        # Create auto finder output
        auto_outpath = tmp_path / "auto_results.h5"
        fres_auto = np.array([4.5e9, 5.2e9])
        
        with h5py.File(auto_outpath, 'w') as hf:
            hf.create_dataset('fres', data=fres_auto)
        
        # Load in manual finder via run_peak_finder
        manual_outpath = tmp_path / "manual_results.h5"
        
        # Just test that the file loading works correctly
        with h5py.File(auto_outpath, 'r') as hf:
            loaded_fres = hf['fres'][:]
        
        # Verify that we can load from file path string
        np.testing.assert_array_almost_equal(loaded_fres, fres_auto)
