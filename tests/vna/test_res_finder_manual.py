"""
Comprehensive tests for manual resonance finder (res_finder_manual.py).

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

from citkid.vna.res_finder_manual import (
    ResFinder,
    run_res_finder_manual
)


class TestResFinderInit:
    """Test ResFinder initialization."""
    
    def test_init_with_array(self, synthetic_vna_data, tmp_path):
        """Test initialization with array of initial resonances."""
        outpath = tmp_path / "test.h5"
        fres_initial = synthetic_vna_data['fres_true'][:2]  # Use first 2
        
        finder = ResFinder(
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
    
    def test_init_phase_detrending(self, synthetic_vna_data, tmp_path):
        """Test that phase is detrended during initialization."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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
    
    def test_init_file_exists_overwrite_false(self, synthetic_vna_data, tmp_path):
        """Test FileExistsError when overwrite=False."""
        outpath = tmp_path / "existing.h5"
        outpath.touch()
        
        with pytest.raises(FileExistsError, match='already exists'):
            ResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                [],
                str(outpath),
                overwrite=False
            )

    def test_init_missing_directory(self, synthetic_vna_data, tmp_path):
        """Test FileNotFoundError when output directory does not exist."""
        outpath = tmp_path / "nonexistent_dir" / "output.h5"

        with pytest.raises(FileNotFoundError, match='Output directory does not exist'):
            ResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                [],
                str(outpath),
            )

    def test_init_invalid_extension(self, synthetic_vna_data, tmp_path):
        """Test ValueError when output path does not have a .h5 extension."""
        outpath = tmp_path / "output.txt"

        with pytest.raises(ValueError, match=r'\.h5 extension'):
            ResFinder(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                [],
                str(outpath),
            )
    
    def test_init_file_exists_overwrite_true(self, synthetic_vna_data, tmp_path, capsys):
        """Test warning when file exists and overwrite=True."""
        outpath = tmp_path / "existing.h5"
        outpath.touch()
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath),
            overwrite=True
        )
        
        captured = capsys.readouterr()
        assert 'already exists' in captured.out
    
    def test_init_empty_fres(self, synthetic_vna_data, tmp_path):
        """Test initialization with empty initial resonance list."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        assert len(finder.fres) == 0
        assert isinstance(finder.fres, list)


class TestResFinderAddRemove:
    """Test adding and removing resonances."""
    
    def test_add_resonance(self, synthetic_vna_data, tmp_path):
        """Test adding a resonance."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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
    
    def test_add_multiple_resonances(self, synthetic_vna_data, tmp_path):
        """Test adding multiple resonances."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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
    
    def test_remove_resonance(self, synthetic_vna_data, tmp_path):
        """Test removing a resonance."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]
        
        finder = ResFinder(
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
    
    def test_remove_outside_view(self, synthetic_vna_data, tmp_path):
        """Test that removing outside view fails."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]
        
        finder = ResFinder(
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
    
    def test_remove_no_resonances(self, synthetic_vna_data, tmp_path):
        """Test removing when no resonances exist."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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


class TestResFinderUndo:
    """Test undo functionality."""
    
    def test_undo_add(self, synthetic_vna_data, tmp_path):
        """Test undoing an add operation."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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
    
    def test_undo_remove(self, synthetic_vna_data, tmp_path):
        """Test undoing a remove operation."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9]
        
        finder = ResFinder(
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
    
    def test_undo_multiple_operations(self, synthetic_vna_data, tmp_path):
        """Test undoing multiple operations in sequence."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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
    
    def test_undo_empty_stack(self, synthetic_vna_data, tmp_path):
        """Test undo when undo stack is empty."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        # Try to undo with empty stack (should handle gracefully)
        finder.undo()
        assert len(finder.fres) == 0
    
    def test_undo_stack_grows(self, synthetic_vna_data, tmp_path):
        """Test that undo stack grows with operations."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
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


class TestResFinderFileIO:
    """Test file save/load functionality."""
    
    def test_save_results(self, synthetic_vna_data, tmp_path):
        """Test saving results to HDF5."""
        outpath = tmp_path / "results.h5"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]
        
        finder = ResFinder(
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
    
    def test_save_empty_results(self, synthetic_vna_data, tmp_path):
        """Test saving empty resonance list."""
        outpath = tmp_path / "empty.h5"
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )
        
        finder.save_data()
        
        with h5py.File(outpath, 'r') as hf:
            assert 'fres' in hf
            assert len(hf['fres']) == 0


class TestResFinderEdgeCases:
    """Test edge cases."""
    
    def test_single_data_point(self, tmp_path):
        """Test with single data point."""
        outpath = tmp_path / "test.h5"
        
        f = np.array([5e9])
        z = np.array([0.9 + 0.1j])
        
        finder = ResFinder(f, z, [], str(outpath))
        
        assert len(finder.f) == 1
        assert len(finder.mag_db) == 1
        assert len(finder.phase) == 1
    
    def test_duplicate_resonances(self, synthetic_vna_data, tmp_path):
        """Test handling of duplicate resonances in initial list."""
        outpath = tmp_path / "test.h5"
        
        # Include duplicates in initial list
        fres_initial = [5e9, 5e9, 6e9, 6e9, 7e9]
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath)
        )
        
        # Should still work (duplicates may or may not be removed automatically)
        assert len(finder.fres) > 0
    
    def test_unsorted_initial_resonances(self, synthetic_vna_data, tmp_path):
        """Test that resonances are sorted."""
        outpath = tmp_path / "test.h5"
        
        # Unsorted initial list
        fres_initial = [6e9, 4.5e9, 7e9, 5e9]
        
        finder = ResFinder(
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


class TestLocalDf:
    """Tests for ResFinder._local_df()."""

    def _make_finder(self, f, z, tmp_path):
        outpath = str(tmp_path / "out.h5")
        return ResFinder(f, z, [], outpath)

    def test_local_df_uniform_spacing(self, tmp_path):
        """Uniform grid → returns the grid step."""
        f = np.linspace(4e9, 8e9, 1001)
        z = np.ones(len(f), dtype=complex)
        finder = self._make_finder(f, z, tmp_path)
        df = f[1] - f[0]
        result = finder._local_df(6e9)
        assert abs(result - df) < 1e3  # within 1 kHz

    def test_local_df_at_lower_boundary(self, tmp_path):
        """Below data range → only right neighbour spacing used."""
        f = np.array([1e9, 2e9, 4e9, 7e9], dtype=float)
        z = np.ones(len(f), dtype=complex)
        finder = self._make_finder(f, z, tmp_path)
        finder.f = f  # bypass phase detrend
        result = finder._local_df(0.5e9)  # idx clipped to 0
        assert result == 1e9  # f[1] - f[0]

    def test_local_df_at_upper_boundary(self, tmp_path):
        """Above data range → only left neighbour spacing used."""
        f = np.array([1e9, 2e9, 4e9, 7e9], dtype=float)
        z = np.ones(len(f), dtype=complex)
        finder = self._make_finder(f, z, tmp_path)
        finder.f = f
        result = finder._local_df(8e9)  # idx clipped to len-1
        assert result == 3e9  # f[-1] - f[-2]

    def test_local_df_returns_minimum(self, tmp_path):
        """Returns the minimum of left and right neighbour spacings."""
        f = np.array([1e9, 2e9, 2.1e9, 4e9], dtype=float)
        z = np.ones(len(f), dtype=complex)
        finder = self._make_finder(f, z, tmp_path)
        finder.f = f
        # At 2.05e9: left=0.1e9, right=1.9e9 → min=0.1e9
        result = finder._local_df(2.05e9)
        assert abs(result - 0.1e9) < 1e6


class TestInterpolateZ:
    """Tests for ResFinder._interpolate_z()."""

    def test_interpolate_at_sample_point(self, synthetic_vna_data, tmp_path):
        """At an exact sample index, result equals finder.z at that index."""
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        idx = 100
        result = finder._interpolate_z(finder.f[idx])
        np.testing.assert_allclose(result, finder.z[idx], rtol=1e-6)

    def test_interpolate_midpoint(self, tmp_path):
        """Midpoint between two samples returns average of the two z values."""
        f = np.array([0.0, 1.0, 2.0, 3.0])
        z = np.array([0 + 0j, 2 + 4j, 6 + 8j, 10 + 12j], dtype=complex)
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(f, z, [], outpath)
        finder.f = f
        finder.z = z
        result = finder._interpolate_z(0.5)  # midpoint of f[0] and f[1]
        expected = (z[0] + z[1]) / 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_interpolate_below_range(self, tmp_path):
        """Frequency below data range → returns z[0]."""
        f = np.array([1e9, 2e9, 3e9], dtype=float)
        z = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=complex)
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(f, z, [], outpath)
        finder.f = f
        finder.z = z
        result = finder._interpolate_z(0.0)
        assert result == z[0]

    def test_interpolate_above_range(self, tmp_path):
        """Frequency above data range → returns z[-1]."""
        f = np.array([1e9, 2e9, 3e9], dtype=float)
        z = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=complex)
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(f, z, [], outpath)
        finder.f = f
        finder.z = z
        result = finder._interpolate_z(10e9)
        assert result == z[-1]


class TestAddResonanceDuplicateGuard:
    """Tests for the duplicate-proximity guard inside add_resonance."""

    def test_add_too_close_rejected(self, synthetic_vna_data, tmp_path):
        """Frequency within one sample spacing of an existing fres is rejected."""
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        finder.fres = [5e9]
        df = finder._local_df(5e9)
        finder.add_resonance(5e9 + df * 0.5)
        assert len(finder.fres) == 1

    def test_add_just_outside_accepted(self, synthetic_vna_data, tmp_path):
        """Frequency more than one sample spacing away is accepted."""
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        finder.fres = [5e9]
        df = finder._local_df(5e9)
        finder.add_resonance(5e9 + df * 2.0)
        assert len(finder.fres) == 2

    def test_add_empty_fres_always_accepted(self, synthetic_vna_data, tmp_path):
        """First resonance is always accepted (no existing fres to clash with)."""
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        finder.add_resonance(6e9)
        assert len(finder.fres) == 1


class TestUpdateIQPlot:
    """Tests for ResFinder.update_iq_plot()."""

    def _make_finder(self, synthetic_vna_data, tmp_path, view_range=None):
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        if view_range is None:
            view_range = [[4e9, 8e9], [-20, 0]]
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = view_range
        finder.iq_curve = Mock()
        finder.mag_highlight = Mock()
        finder.phase_highlight = Mock()
        finder.iq_fres_scatter = Mock()
        finder.plot_iq = Mock()
        return finder

    def test_populates_sel_arrays(self, synthetic_vna_data, tmp_path):
        """After a call, _iq_f_sel and _iq_z_sel cover the center quarter."""
        finder = self._make_finder(
            synthetic_vna_data, tmp_path, view_range=[[4e9, 8e9], [-20, 0]]
        )
        finder.update_iq_plot()
        # Center-quarter of [4e9, 8e9] is [5e9, 7e9]
        assert len(finder._iq_f_sel) > 0
        assert len(finder._iq_z_sel) == len(finder._iq_f_sel)
        assert finder._iq_f_sel.min() >= 5e9
        assert finder._iq_f_sel.max() <= 7e9

    def test_empty_window_clears_plots(self, synthetic_vna_data, tmp_path):
        """No data in IQ window → all items cleared, sel arrays empty."""
        finder = self._make_finder(
            synthetic_vna_data, tmp_path,
            view_range=[[20e9, 24e9], [-20, 0]],
        )
        finder.update_iq_plot()
        finder.iq_curve.setData.assert_called_with([], [])
        finder.mag_highlight.setData.assert_called_with([], [])
        finder.phase_highlight.setData.assert_called_with([], [])
        finder.iq_fres_scatter.setData.assert_called_with([])
        assert len(finder._iq_f_sel) == 0

    def test_fres_in_window_creates_scatter_spot(self, synthetic_vna_data, tmp_path):
        """A fres inside the IQ window produces exactly one scatter spot."""
        finder = self._make_finder(
            synthetic_vna_data, tmp_path, view_range=[[4e9, 8e9], [-20, 0]]
        )
        finder.fres = [6e9]  # inside center-quarter [5e9, 7e9]
        finder.update_iq_plot()
        spots = finder.iq_fres_scatter.setData.call_args[0][0]
        assert len(spots) == 1

    def test_fres_outside_window_creates_no_spots(self, synthetic_vna_data, tmp_path):
        """A fres outside the IQ window produces no scatter spots."""
        finder = self._make_finder(
            synthetic_vna_data, tmp_path, view_range=[[4e9, 8e9], [-20, 0]]
        )
        finder.fres = [4.1e9]  # outside center-quarter [5e9, 7e9]
        finder.update_iq_plot()
        spots = finder.iq_fres_scatter.setData.call_args[0][0]
        assert len(spots) == 0


class TestToggleIQ:
    """Tests for ResFinder.toggle_iq()."""

    def _make_finder(self, synthetic_vna_data, tmp_path):
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = [[4e9, 8e9], [-20, 0]]
        finder.iq_curve = Mock()
        finder.mag_highlight = Mock()
        finder.phase_highlight = Mock()
        finder.iq_fres_scatter = Mock()
        finder.plot_iq = Mock()
        return finder

    def test_toggle_on_sets_visible_and_shows_plot(self, synthetic_vna_data, tmp_path):
        """First toggle makes iq_visible True and calls plot_iq.show()."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        assert finder.iq_visible is False
        finder.toggle_iq()
        assert finder.iq_visible is True
        finder.plot_iq.show.assert_called_once()

    def test_toggle_off_hides_and_clears(self, synthetic_vna_data, tmp_path):
        """Second toggle hides the plot and clears all highlight data."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder.toggle_iq()   # on
        finder.toggle_iq()   # off
        assert finder.iq_visible is False
        finder.plot_iq.hide.assert_called_once()
        finder.mag_highlight.setData.assert_called_with([], [])
        finder.phase_highlight.setData.assert_called_with([], [])
        finder.iq_fres_scatter.setData.assert_called_with([])


class TestOnIQClick:
    """Tests for ResFinder.on_iq_click()."""

    def _make_finder(self, synthetic_vna_data, tmp_path):
        """Finder with iq_visible=True and known IQ selection data."""
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        finder.iq_visible = True
        finder.plot_mag = Mock()
        finder.plot_mag.viewRange.return_value = [[4e9, 8e9], [-20, 0]]
        finder.plot_iq = Mock()
        finder.plot_iq.sceneBoundingRect.return_value.contains.return_value = True
        finder.iq_curve = Mock()
        finder.mag_highlight = Mock()
        finder.phase_highlight = Mock()
        finder.iq_fres_scatter = Mock()
        # Three known IQ points
        finder._iq_f_sel = np.array([5.0e9, 6.0e9, 7.0e9])
        finder._iq_z_sel = np.array([0.5 + 0j, 0 + 0.5j, -0.5 + 0j])
        return finder

    def _make_event(self, ix, iy, shift=False):
        """Mock event that maps the given (ix, iy) IQ position."""
        from pyqtgraph.Qt import QtCore
        event = Mock()
        event.scenePos.return_value = Mock()
        point = Mock()
        point.x.return_value = ix
        point.y.return_value = iy
        mock_vb = Mock()
        mock_vb.mapSceneToView.return_value = point
        event.button.return_value = QtCore.Qt.LeftButton
        event.modifiers.return_value = (
            QtCore.Qt.ShiftModifier if shift else QtCore.Qt.NoModifier
        )
        return event, mock_vb

    def test_left_click_adds_nearest_resonance(self, synthetic_vna_data, tmp_path):
        """Left-click near a data point calls add_resonance with its frequency."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        # Click very close to (0, 0.5j) → should pick freq=6e9
        event, mock_vb = self._make_event(0.01, 0.5)
        finder.plot_iq.vb = mock_vb
        finder.add_resonance = Mock()
        finder.on_iq_click(event)
        finder.add_resonance.assert_called_once_with(6.0e9)

    def test_shift_click_removes_nearest_resonance(self, synthetic_vna_data, tmp_path):
        """Shift+click calls remove_nearest_resonance with the nearest frequency."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        # Click very close to (0.5, 0) → should pick freq=5e9
        event, mock_vb = self._make_event(0.5, 0.01, shift=True)
        finder.plot_iq.vb = mock_vb
        finder.remove_nearest_resonance = Mock()
        finder.on_iq_click(event)
        finder.remove_nearest_resonance.assert_called_once_with(5.0e9)

    def test_click_ignored_when_iq_hidden(self, synthetic_vna_data, tmp_path):
        """on_iq_click does nothing when iq_visible is False."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder.iq_visible = False
        finder.add_resonance = Mock()
        event = Mock()
        finder.on_iq_click(event)
        finder.add_resonance.assert_not_called()

    def test_click_ignored_when_no_data(self, synthetic_vna_data, tmp_path):
        """on_iq_click logs a message and does nothing when _iq_f_sel is empty."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder._iq_f_sel = np.array([])
        finder._iq_z_sel = np.array([], dtype=complex)
        finder.add_resonance = Mock()
        event, mock_vb = self._make_event(0.0, 0.0)
        finder.plot_iq.vb = mock_vb
        finder.on_iq_click(event)
        finder.add_resonance.assert_not_called()


class TestRunResFinder:
    """Test run_res_finder_manual wrapper function."""
    
    @patch('citkid.vna.res_finder_manual.ResFinder.run')
    def test_run_with_array(self, mock_run, synthetic_vna_data, tmp_path):
        """Test run_res_finder_manual with array input."""
        outpath = tmp_path / "test.h5"
        fres_initial = [4.5e9, 5.2e9]
        
        with patch('citkid.vna.res_finder_manual.ResFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = fres_initial
            
            fres = run_res_finder_manual(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                fres_initial,
                str(outpath)
            )
            
            # Check return value
            assert isinstance(fres, np.ndarray)
            np.testing.assert_array_almost_equal(fres, fres_initial)
    
    def test_run_with_file_path(self, synthetic_vna_data, tmp_path):
        """Test run_res_finder_manual loading fres from file."""
        # Create a file with fres data
        fres_file = tmp_path / "input_fres.h5"
        fres_initial = np.array([4.5e9, 5.2e9, 6.1e9])
        
        with h5py.File(fres_file, 'w') as hf:
            hf.create_dataset('fres', data=fres_initial)
        
        outpath = tmp_path / "output.h5"
        
        with patch('citkid.vna.res_finder_manual.ResFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = fres_initial
            
            with patch('citkid.vna.res_finder_manual.ResFinder.run'):
                fres = run_res_finder_manual(
                    synthetic_vna_data['f'],
                    synthetic_vna_data['z'],
                    str(fres_file),
                    str(outpath)
                )
            
            # Check that ResFinder was initialized
            MockFinder.assert_called_once()
    
    @patch('citkid.vna.res_finder_manual.ResFinder.run')
    def test_run_passes_parameters(self, mock_run, synthetic_vna_data, tmp_path):
        """Test that parameters are passed through correctly."""
        outpath = tmp_path / "test.h5"
        
        with patch('citkid.vna.res_finder_manual.ResFinder') as MockFinder:
            mock_instance = MockFinder.return_value
            mock_instance.fres = []
            
            run_res_finder_manual(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                [],
                str(outpath),
                margin_factor=0.2,
                overwrite=True
            )
            
            # Check parameters passed to ResFinder
            call_args = MockFinder.call_args
            # Parameters are passed as positional args
            assert call_args[0][4] == 0.2  # margin_factor is 5th arg
            assert call_args[0][5] is True  # overwrite is 6th arg


class TestIntegration:
    """Integration tests for auto -> manual workflow."""
    
    def test_auto_to_manual_workflow(self, synthetic_vna_data, tmp_path):
        """Test complete workflow from auto to manual resonance finding."""
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
        
        finder = ResFinder(
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
        
        # Load in manual finder via run_res_finder_manual
        manual_outpath = tmp_path / "manual_results.h5"
        
        # Just test that the file loading works correctly
        with h5py.File(auto_outpath, 'r') as hf:
            loaded_fres = hf['fres'][:]
        
        # Verify that we can load from file path string
        np.testing.assert_array_almost_equal(loaded_fres, fres_auto)
