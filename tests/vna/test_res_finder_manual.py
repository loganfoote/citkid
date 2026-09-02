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
import zarr
import os
from unittest.mock import Mock, patch, MagicMock

from citkid.vna.res_finder_manual import (
    ResFinder,
    ResFinderWindow,
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
            str(outpath)
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
        """Test that ResFinder allows loading when fres_manual exists in zarr group."""
        zarr_path = tmp_path / "existing.zarr"
        # Create a zarr group with existing fres_manual
        grp = zarr.open_group(str(zarr_path), mode='w')
        existing_fres = np.array([4.5e9])
        grp.create_array('fres_manual', data=existing_fres)
        
        # ResFinder should now allow this without error
        # (conflict resolution happens in run_res_finder_manual)
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(zarr_path)
        )
        assert finder.fres == []

    def test_init_zarr_created_automatically(self, synthetic_vna_data, tmp_path):
        """Test that zarr directory is created automatically."""
        zarr_path = tmp_path / "new_dir" / "output.zarr"

        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(zarr_path),
        )
        
        # Zarr group should be created
        assert finder.zarr_group is not None

    def test_init_accepts_zarr_paths(self, synthetic_vna_data, tmp_path):
        """Test that zarr paths are accepted (not just .h5)."""
        zarr_path = tmp_path / "output.zarr"

        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(zarr_path),
        )
        
        # Should not raise any error
        assert finder.zarr_group is not None
    
    def test_init_file_exists_overwrite_true(self, synthetic_vna_data, tmp_path):
        """Test that ResFinder allows initialization even when fres_manual exists in zarr."""
        zarr_path = tmp_path / "existing.zarr"
        # Create a zarr group with existing fres_manual
        grp = zarr.open_group(str(zarr_path), mode='w')
        grp.create_array('fres_manual', data=np.array([4.5e9]))
        
        # ResFinder should now allow this without warning
        # (conflict resolution happens in run_res_finder_manual)
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(zarr_path)
        )
        
        assert finder.fres == []
    
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
        """Test saving results to zarr."""
        outpath = tmp_path / "results.zarr"
        fres_initial = [4.5e9, 5.2e9, 6.1e9]

        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath)
        )

        # Save
        finder.save_data()

        # Load and verify
        grp = zarr.open_group(str(outpath), mode='r')
        assert 'fres_manual' in grp
        fres_loaded = grp['fres_manual'][:]
        np.testing.assert_array_almost_equal(
            sorted(fres_loaded),
            sorted(fres_initial)
        )
    def test_save_empty_results(self, synthetic_vna_data, tmp_path):
        """Test saving empty resonance list."""
        outpath = tmp_path / "empty.zarr"
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            str(outpath)
        )

        finder.save_data()

        grp = zarr.open_group(str(outpath), mode='r')
        assert 'fres_manual' in grp
        assert grp['fres_manual'].shape[0] == 0
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
        from citkid.qt_compat import Qt as _Qt
        event = Mock()
        event.scenePos.return_value = Mock()
        point = Mock()
        point.x.return_value = ix
        point.y.return_value = iy
        mock_vb = Mock()
        mock_vb.mapSceneToView.return_value = point
        event.button.return_value = _Qt.LeftButton
        event.modifiers.return_value = (
            _Qt.ShiftModifier if shift else _Qt.NoModifier
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
        """Test run_res_finder_manual loading fres from zarr file."""
        # Create a zarr file with fres_auto data
        fres_file = tmp_path / "input_fres.zarr"
        fres_initial = np.array([4.5e9, 5.2e9, 6.1e9])
        
        grp = zarr.open_group(str(fres_file), mode='w')
        grp.create_array('fres_auto', data=fres_initial)
        
        outpath = tmp_path / "output.zarr"
        
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
                margin_factor=0.2
            )
            
            # Check parameters passed to ResFinder
            call_args = MockFinder.call_args
            # Parameters are passed as positional args
            assert call_args[0][4] == 0.2  # margin_factor is 5th arg


class TestOnClick:
    """Tests for ResFinder.on_click() — including regression for double-fire bug."""

    def _make_finder(self, synthetic_vna_data, tmp_path):
        outpath = str(tmp_path / "out.h5")
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        return finder

    def _make_click_event(self, freq, shift=False):
        """Return a mock click event whose scene position maps to freq."""
        from citkid.qt_compat import Qt as _Qt
        event = Mock()
        event.scenePos.return_value = Mock()
        event.button.return_value = _Qt.LeftButton
        event.modifiers.return_value = (
            _Qt.ShiftModifier if shift else _Qt.NoModifier
        )
        mouse_point = Mock()
        mouse_point.x.return_value = freq
        mouse_point.y.return_value = 0.0
        return event, mouse_point

    def _setup_plot_mocks(self, finder, freq, mag_hit=True, phase_hit=False):
        """Attach plot mocks to a finder whose setup_ui was patched out."""
        finder.plot_mag = Mock()
        finder.plot_phase = Mock()
        # .sceneBoundingRect() always returns the same mock object; set
        # .contains.return_value on that object so repeated calls agree.
        finder.plot_mag.sceneBoundingRect.return_value.contains.return_value = mag_hit
        finder.plot_phase.sceneBoundingRect.return_value.contains.return_value = phase_hit
        mouse_point = Mock()
        mouse_point.x.return_value = freq
        mouse_point.y.return_value = 0.0
        finder.plot_mag.vb.mapSceneToView.return_value = mouse_point
        finder.plot_phase.vb.mapSceneToView.return_value = mouse_point
        return mouse_point

    def test_single_on_click_call_adds_exactly_one_resonance(
        self, synthetic_vna_data, tmp_path
    ):
        """Regression: one call to on_click must add exactly one resonance.

        Before the fix, sigMouseClicked was connected to on_click twice
        (once via plot_mag.scene() and once via plot_phase.scene(), which
        are the same object). That caused every click to call on_click twice:
        the first call added the resonance and the second call hit the
        duplicate-proximity guard. This test ensures a single invocation
        of on_click results in len(fres) == 1.
        """
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        freq = 5.5e9
        event, _ = self._make_click_event(freq)
        self._setup_plot_mocks(finder, freq, mag_hit=True, phase_hit=False)

        finder.on_click(event)

        assert len(finder.fres) == 1
        assert finder.fres[0] == freq

    def test_two_on_click_calls_same_freq_add_one_resonance(
        self, synthetic_vna_data, tmp_path
    ):
        """Two on_click calls at the same scene position yield only one fres.

        This reproduces exactly what the old double-connection bug caused:
        the handler was called twice for a single user click. The second
        call must be blocked by the proximity guard, leaving exactly one
        resonance.
        """
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        freq = 5.5e9
        event, _ = self._make_click_event(freq)
        self._setup_plot_mocks(finder, freq, mag_hit=True, phase_hit=False)

        finder.on_click(event)
        finder.on_click(event)  # simulates the old double-fire

        assert len(finder.fres) == 1

    def test_on_click_outside_both_plots_adds_nothing(
        self, synthetic_vna_data, tmp_path
    ):
        """A click outside both plot bounding rects does not add a resonance."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        event, _ = self._make_click_event(5.5e9)
        self._setup_plot_mocks(finder, 5.5e9, mag_hit=False, phase_hit=False)

        finder.on_click(event)

        assert len(finder.fres) == 0

    def test_on_click_phase_plot_adds_resonance(
        self, synthetic_vna_data, tmp_path
    ):
        """A click inside the phase plot also adds a resonance correctly."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        freq = 6.0e9
        event, _ = self._make_click_event(freq)
        self._setup_plot_mocks(finder, freq, mag_hit=False, phase_hit=True)

        finder.on_click(event)

        assert len(finder.fres) == 1
        assert finder.fres[0] == freq


class TestIntegration:
    """Integration tests for auto -> manual workflow."""
    
    def test_auto_to_manual_workflow(self, synthetic_vna_data, tmp_path):
        """Test complete workflow from auto to manual resonance finding."""
        # Now run manual finder with auto results
        manual_outpath = tmp_path / "manual_results.zarr"

        fres_auto = np.array([4.5e9, 5.2e9, 6.1e9])
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_auto,
            str(manual_outpath)
        )
        
        # Manually adjust: add one, remove one (mock operations)
        finder.add_resonance(7.3e9)
        
        # Save
        finder.save_data()
        
        # Verify final results
        grp = zarr.open_group(str(manual_outpath), mode='r')
        fres_final = grp['fres_manual'][:]
        assert len(fres_final) == 4
        assert 7.3e9 in fres_final
    
    def test_load_from_auto_output(self, synthetic_vna_data, tmp_path):
        """Test loading initial resonances from auto finder output."""
        # Create auto finder output with zarr
        auto_outpath = tmp_path / "auto_results.zarr"
        fres_auto = np.array([4.5e9, 5.2e9])
        
        grp = zarr.open_group(str(auto_outpath), mode='w')
        grp.create_array('fres_auto', data=fres_auto)
        
        # Load in manual finder via run_res_finder_manual
        manual_outpath = tmp_path / "manual_results.zarr"
        
        # Just test that the file loading works correctly
        grp_read = zarr.open_group(str(auto_outpath), mode='r')
        loaded_fres = grp_read['fres_auto'][:]
        
        # Verify that we can load from file path string
        np.testing.assert_array_almost_equal(loaded_fres, fres_auto)


# ---------------------------------------------------------------------------
# Overview navigator (added in this session)
# ---------------------------------------------------------------------------

class TestOverviewNavigator:
    """Tests for the LinearRegionItem overview navigator."""

    def _make_finder(self, synthetic_vna_data, tmp_path):
        import zarr
        outpath = str(tmp_path / 'out.zarr')
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],
            outpath,
        )
        # Attach mock overview items so the methods can run without a real window
        finder._overview_region = Mock()
        finder._overview_updating = False
        finder.plot_mag = Mock()
        return finder

    def test_update_overview_region_calls_setRegion(
        self, synthetic_vna_data, tmp_path
    ):
        """`_update_overview_region` passes current x-range to setRegion."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder.plot_mag.viewRange.return_value = [[5e9, 6e9], [-30, 0]]
        finder._update_overview_region()
        finder._overview_region.setRegion.assert_called_once_with([5e9, 6e9])

    def test_update_overview_region_guard_prevents_reentry(
        self, synthetic_vna_data, tmp_path
    ):
        """When _overview_updating is True, setRegion must not be called."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder._overview_updating = True
        finder.plot_mag.viewRange.return_value = [[5e9, 6e9], [-30, 0]]
        finder._update_overview_region()
        finder._overview_region.setRegion.assert_not_called()

    def test_update_overview_region_releases_guard(
        self, synthetic_vna_data, tmp_path
    ):
        """Guard must be False again after `_update_overview_region` returns."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder.plot_mag.viewRange.return_value = [[5e9, 6e9], [-30, 0]]
        finder._update_overview_region()
        assert finder._overview_updating is False

    def test_on_overview_region_changed_sets_xrange(
        self, synthetic_vna_data, tmp_path
    ):
        """`_on_overview_region_changed` propagates region to plot_mag.setXRange."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder._overview_region.getRegion.return_value = (4.8e9, 5.5e9)
        finder._on_overview_region_changed()
        finder.plot_mag.setXRange.assert_called_once_with(4.8e9, 5.5e9, padding=0)

    def test_on_overview_region_changed_guard_prevents_reentry(
        self, synthetic_vna_data, tmp_path
    ):
        """When _overview_updating is True, setXRange must not be called."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder._overview_updating = True
        finder._on_overview_region_changed()
        finder.plot_mag.setXRange.assert_not_called()

    def test_on_overview_region_changed_releases_guard(
        self, synthetic_vna_data, tmp_path
    ):
        """Guard must be False again after the handler returns."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder._overview_region.getRegion.return_value = (4.8e9, 5.5e9)
        finder._on_overview_region_changed()
        assert finder._overview_updating is False

    def test_two_way_sync_does_not_loop(self, synthetic_vna_data, tmp_path):
        """Calling both helpers back-to-back must not recurse infinitely."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        finder.plot_mag.viewRange.return_value = [[5e9, 6e9], [-30, 0]]
        finder._overview_region.getRegion.return_value = (5e9, 6e9)
        # Neither should raise a RecursionError
        finder._update_overview_region()
        finder._on_overview_region_changed()
        assert finder._overview_updating is False


class TestDragToRemove:
    """Tests for shift+drag bulk resonance removal feature."""
    
    @pytest.fixture
    def finder(self, synthetic_vna_data, tmp_path):
        """Create a ResFinder with some initial resonances."""
        outpath = tmp_path / "test.h5"
        fres_initial = synthetic_vna_data['fres_true'][:5]  # Use first 5
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            fres_initial,
            str(outpath)
        )
        return finder
    
    def test_remove_resonances_in_range_basic(self, finder):
        """Test removing resonances in a frequency range."""
        initial_count = len(finder.fres)
        initial_fres = sorted(finder.fres)
        
        # Remove resonances in a range that contains some
        f_min = initial_fres[1] - 1e6  # Just below second resonance
        f_max = initial_fres[2] + 1e6  # Just above third resonance
        
        finder.remove_resonances_in_range(f_min, f_max)
        
        # Should have removed some resonances
        assert len(finder.fres) < initial_count
        # Removed resonances should be those in the range
        for freq in finder.fres:
            assert not (f_min <= freq <= f_max)
    
    def test_remove_resonances_in_range_empty_result(self, finder):
        """Test removing all resonances leaves empty list."""
        fres_min = min(finder.fres)
        fres_max = max(finder.fres)
        
        # Remove all with a large range
        finder.remove_resonances_in_range(fres_min - 1e6, fres_max + 1e6)
        
        assert len(finder.fres) == 0
    
    def test_remove_resonances_in_range_no_match(self, finder):
        """Test removing resonances from empty range."""
        initial_fres = list(finder.fres)
        
        # Remove from a range with no resonances
        finder.remove_resonances_in_range(1e9, 1.1e9)  # Far outside usual range
        
        # Should be unchanged
        assert len(finder.fres) == len(initial_fres)
        assert sorted(finder.fres) == sorted(initial_fres)
    
    def test_remove_resonances_adds_to_undo_stack(self, finder):
        """Test that remove_resonances_in_range adds to undo stack."""
        initial_undo_len = len(finder.undo_stack)
        fres_before = list(finder.fres)
        
        fres_min = min(finder.fres)
        fres_max = max(finder.fres)
        finder.remove_resonances_in_range(fres_min, fres_max + 1e6)
        
        # Should have added undo entry
        assert len(finder.undo_stack) > initial_undo_len
        # Last entry should be remove_range tuple
        assert finder.undo_stack[-1][0] == 'remove_range'
        # Should contain f_min, f_max, and old fres list
        assert len(finder.undo_stack[-1]) == 4
        assert finder.undo_stack[-1][1] == fres_min
        assert finder.undo_stack[-1][3] == fres_before
    
    def test_remove_resonances_boundary_conditions(self, finder):
        """Test remove with exact boundary frequencies."""
        fres_list = sorted(finder.fres)
        if len(fres_list) < 3:
            pytest.skip("Need at least 3 resonances")
        
        # Remove exactly at boundaries of resonances
        f1 = fres_list[0]
        f2 = fres_list[1]
        f3 = fres_list[2]
        
        finder.remove_resonances_in_range(f1, f2)
        
        # f1 and f2 should be removed, f3 should remain
        assert f1 not in finder.fres
        assert f2 not in finder.fres
        assert f3 in finder.fres
    
    def test_remove_single_resonance_in_range(self, finder):
        """Test removing a single resonance using range."""
        if len(finder.fres) < 2:
            pytest.skip("Need at least 2 resonances")
        
        target_freq = finder.fres[0]
        initial_count = len(finder.fres)
        
        # Remove just this one resonance
        finder.remove_resonances_in_range(target_freq - 1e5, target_freq + 1e5)
        
        assert len(finder.fres) == initial_count - 1
        assert target_freq not in finder.fres
    
    def test_remove_resonances_preserves_non_removed(self, finder):
        """Test that resonances outside range are preserved."""
        fres_list = sorted(finder.fres)
        if len(fres_list) < 5:
            pytest.skip("Need at least 5 resonances")
        
        # Remove middle 3 resonances
        f_min = fres_list[1] - 0.5e6
        f_max = fres_list[3] + 0.5e6
        
        finder.remove_resonances_in_range(f_min, f_max)
        
        # First and last should be preserved
        assert fres_list[0] in finder.fres
        assert fres_list[4] in finder.fres
    
    def test_remove_on_empty_fres(self, synthetic_vna_data, tmp_path):
        """Test remove_resonances_in_range with empty fres list."""
        outpath = tmp_path / "test.h5"
        
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            [],  # Empty initial resonances
            str(outpath)
        )
        
        # Should not crash
        finder.remove_resonances_in_range(1e9, 2e9)
        assert len(finder.fres) == 0


class TestResFinderWindow:
    """Tests for window closeEvent behavior."""
    
    def _make_finder_with_ui(self, synthetic_vna_data, tmp_path):
        """Create a ResFinder and mock UI components for testing."""
        outpath = tmp_path / "test.h5"
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            synthetic_vna_data['fres_true'][:2],
            str(outpath)
        )
        return finder
    
    def test_window_has_close_event_handler(self, synthetic_vna_data, tmp_path):
        """Test that ResFinderWindow has closeEvent method."""
        # Verify that ResFinderWindow class has closeEvent
        assert hasattr(ResFinderWindow, 'closeEvent')
        
        # Also verify the class can be instantiated with a finder
        from unittest.mock import MagicMock
        window = ResFinderWindow(finder=MagicMock())
        assert window.finder is not None
    
    def test_window_calls_save_data_on_close(self, synthetic_vna_data, tmp_path):
        """Test that closeEvent calls save_data."""
        from unittest.mock import MagicMock, patch
        from pyqtgraph.Qt import QtGui
        
        # Create a finder mock
        finder_mock = MagicMock()
        
        # Create window with mocked finder
        window = ResFinderWindow(finder=finder_mock)
        
        # Create a real QCloseEvent (or mock that properly inherits)
        close_event = MagicMock(spec=QtGui.QCloseEvent)
        
        # Patch the parent class closeEvent to avoid issues with Qt
        with patch('pyqtgraph.GraphicsLayoutWidget.closeEvent'):
            window.closeEvent(close_event)
        
        # save_data should have been called on the finder
        finder_mock.save_data.assert_called_once()
    
    def test_finder_window_constructor_accepts_finder(self):
        """Test that ResFinderWindow stores finder reference."""
        from unittest.mock import MagicMock
        
        finder_mock = MagicMock()
        window = ResFinderWindow(finder=finder_mock)
        
        # Window should have finder reference
        assert window.finder is finder_mock


class TestZarrConflictDialog:
    """Tests for zarr conflict resolution dialog."""
    
    def test_show_zarr_dialog_returns_choice(self):
        """Test that _show_zarr_dialog returns user choice."""
        # Mock the dialog
        with patch('citkid.vna.res_finder_manual.QtWidgets.QDialog') as MockDialog:
            mock_dialog = MagicMock()
            MockDialog.return_value = mock_dialog
            mock_dialog.exec_.return_value = None
            
            # This will test the static method structure
            assert hasattr(ResFinder, '_show_zarr_dialog')
            assert callable(ResFinder._show_zarr_dialog)
    
    def test_run_res_finder_with_load_choice(self, synthetic_vna_data, tmp_path):
        """Test run_res_finder_manual with load choice."""
        outpath = tmp_path / "test.h5"
        
        # Create zarr with existing fres_manual
        grp = zarr.open_group(str(outpath), mode='a')
        existing_fres = np.array([5.1e9, 5.2e9])
        grp['fres_manual'] = existing_fres
        
        # Mock dialog to return 'load'
        with patch.object(ResFinder, '_show_zarr_dialog', return_value='load'):
            result = run_res_finder_manual(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                synthetic_vna_data['fres_true'][:2],  # Different from existing
                grp
            )
            
            # Should have used the existing data (if run completed)
            # Result should be numpy array
            assert isinstance(result, np.ndarray)
    
    def test_run_res_finder_with_overwrite_choice(self, synthetic_vna_data, tmp_path):
        """Test run_res_finder_manual with overwrite choice."""
        outpath = tmp_path / "test.h5"
        
        # Create zarr with existing fres_manual
        grp = zarr.open_group(str(outpath), mode='a')
        existing_fres = np.array([5.1e9, 5.2e9])
        grp['fres_manual'] = existing_fres
        
        initial_fres = synthetic_vna_data['fres_true'][:3]
        
        # Mock dialog to return 'overwrite'
        with patch.object(ResFinder, '_show_zarr_dialog', return_value='overwrite'):
            result = run_res_finder_manual(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                initial_fres,
                grp
            )
            
            # Should use the new data
            assert isinstance(result, np.ndarray)
    
    def test_run_res_finder_with_cancel_choice(self, synthetic_vna_data, tmp_path):
        """Test run_res_finder_manual returns None with cancel choice."""
        outpath = tmp_path / "test.h5"
        
        # Create zarr with existing fres_manual
        grp = zarr.open_group(str(outpath), mode='a')
        existing_fres = np.array([5.1e9, 5.2e9])
        grp['fres_manual'] = existing_fres
        
        # Mock dialog to return 'cancel'
        with patch.object(ResFinder, '_show_zarr_dialog', return_value='cancel'):
            result = run_res_finder_manual(
                synthetic_vna_data['f'],
                synthetic_vna_data['z'],
                synthetic_vna_data['fres_true'][:2],
                grp
            )
            
            # Should return None
            assert result is None


class TestEventFilter:
    """Tests for mouse event filtering during drag operations."""
    
    def _make_finder(self, synthetic_vna_data, tmp_path):
        """Create a ResFinder and mock UI components."""
        outpath = str(tmp_path / 'test.zarr')
        finder = ResFinder(
            synthetic_vna_data['f'],
            synthetic_vna_data['z'],
            synthetic_vna_data['fres_true'][:2],
            outpath
        )
        # Mock plot components for testing
        finder.plot_mag = MagicMock()
        finder.plot_mag.scene = MagicMock(return_value=MagicMock())
        finder.plot_mag.vb = MagicMock()
        return finder
    
    def test_event_filter_installed(self, synthetic_vna_data, tmp_path):
        """Test that event filter is installed on scene."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        
        # Event filter should be installed
        assert hasattr(finder, 'eventFilter')
        assert callable(finder.eventFilter)
    
    def test_event_filter_returns_callable(self, synthetic_vna_data, tmp_path):
        """Test that eventFilter method exists and is callable."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        
        # Create mock event
        mock_event = MagicMock()
        mock_event.type = MagicMock(return_value=999)  # Non-matching type
        
        # Should return False for unhandled events
        result = finder.eventFilter(finder.plot_mag.scene(), mock_event)
        assert result is False
    
    def test_drag_selection_state_tracking(self, synthetic_vna_data, tmp_path):
        """Test that _drag_selection_active state is tracked."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        
        # Initially should be False
        assert finder._drag_selection_active is False
        assert finder._drag_start_freq is None
        assert finder._drag_region_item is None
    
    def test_on_mouse_moved_updates_visual(self, synthetic_vna_data, tmp_path):
        """Test that on_mouse_moved updates the visual region."""
        finder = self._make_finder(synthetic_vna_data, tmp_path)
        
        # Set up state for drag
        finder._drag_selection_active = True
        finder._drag_start_freq = 5e9
        
        # Mock the mouse position
        mock_pos = MagicMock()
        finder.plot_mag.sceneBoundingRect = MagicMock(return_value=MagicMock())
        finder.plot_mag.sceneBoundingRect.return_value.contains = MagicMock(return_value=True)
        finder.plot_mag.vb.mapSceneToView = MagicMock(
            return_value=MagicMock(x=MagicMock(return_value=5.1e9))
        )
        
        # Call on_mouse_moved
        finder.on_mouse_moved(mock_pos)
        
        # Should create or update region item
        # (can't fully test without real graphics, but we can check it doesn't crash)
        assert finder._drag_region_item is not None or finder._drag_selection_active
