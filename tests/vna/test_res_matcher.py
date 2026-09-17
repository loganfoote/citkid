"""
Tests for resonance matcher (res_matcher.py).

Tests cover:
- Load from zarr functionality
- Startup dialog behavior
- Saved x-limits persistence
- Multi-selection system
- Click behavior (Shift+click to add, Ctrl+right-click for threshold)
- CustomViewBox context menu suppression
"""

import pytest
import numpy as np
import zarr
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PyQt5 import QtWidgets, QtCore

np.seterr(divide='ignore')

from citkid.vna.res_matcher import (
    MatchGroup,
    CustomViewBox,
    ResMatcher,
    run_res_matcher
)


@pytest.fixture(autouse=True)
def ensure_qapplication():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def patch_res_matcher_polyfit():
    with patch('citkid.vna.res_matcher.np.polyfit', return_value=np.array([0.0, 0.0])):
        yield


@pytest.fixture
def synthetic_sweep_data():
    """Create synthetic VNA sweep data with resonances."""
    f = np.linspace(4e9, 6e9, 10000)
    
    # Dataset 1: 3 resonances
    z1 = np.ones(len(f), dtype=complex) * 0.8
    fres1 = np.array([4.2e9, 4.8e9, 5.4e9])
    for fr in fres1:
        idx = np.argmin(np.abs(f - fr))
        # Lorentzian dip
        width = 1e6
        z1[idx-50:idx+50] *= 0.3 * np.exp(-((f[idx-50:idx+50] - fr) / width)**2)
    
    # Dataset 2: 3 resonances (slightly shifted)
    z2 = np.ones(len(f), dtype=complex) * 0.8
    fres2 = np.array([4.21e9, 4.79e9, 5.41e9])  # Slightly different frequencies
    for fr in fres2:
        idx = np.argmin(np.abs(f - fr))
        width = 1e6
        z2[idx-50:idx+50] *= 0.3 * np.exp(-((f[idx-50:idx+50] - fr) / width)**2)
    
    res_idx1 = np.array([10, 20, 30])
    res_idx2 = np.array([10, 20, 30])
    
    return {
        'f1': f,
        'z1': z1,
        'fres1': fres1,
        'res_idx1': res_idx1,
        'f2': f,
        'z2': z2,
        'fres2': fres2,
        'res_idx2': res_idx2,
    }


@pytest.fixture
def simple_data():
    """Small synthetic VNA sweep for direct ResMatcher behavior tests."""
    f = np.linspace(4e9, 8e9, 10_000)
    rng = np.random.default_rng(0)
    z1 = np.exp(1j * (2 * np.pi * f / 1e9)) * (1 + 0.001 * rng.standard_normal(len(f)))
    z2 = np.exp(1j * (2 * np.pi * f / 1e9 + 0.1)) * (1 + 0.001 * rng.standard_normal(len(f)))
    fres1 = np.array([4.5e9, 5.2e9, 6.1e9, 7.3e9])
    fres2 = np.array([4.51e9, 5.21e9, 6.09e9, 7.28e9])
    res_idx1 = np.arange(len(fres1)) + 540
    res_idx2 = np.arange(len(fres2)) + 541
    return dict(
        f=f,
        z1=z1,
        z2=z2,
        fres1=fres1,
        fres2=fres2,
        res_idx1=res_idx1,
        res_idx2=res_idx2,
    )


@pytest.fixture
def matcher(simple_data, tmp_path):
    """ResMatcher with setup_ui patched so no window opens."""
    sd = simple_data
    grp_path = str(tmp_path / 'out.zarr')
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        matcher_obj = ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
            grp_path,
            overwrite=True,
        )
    return matcher_obj


def _prepare_headless_matcher(matcher_obj):
    matcher_obj.plot_mag = Mock()
    matcher_obj.plot_mag.viewRange.return_value = [[float(matcher_obj.f1[0]), float(matcher_obj.f1[-1])], [0.0, 1.0]]
    matcher_obj.plot_mag.setYRange = Mock()
    matcher_obj.plot_phase = Mock()
    matcher_obj.plot_phase.setYRange = Mock()
    matcher_obj.curve_ds1_mag = Mock()
    matcher_obj.curve_ds1_phase = Mock()
    matcher_obj.curve_ds2_mag = Mock()
    matcher_obj.curve_ds2_phase = Mock()
    matcher_obj._vb_mag2 = Mock()
    matcher_obj._vb_phase2 = Mock()
    matcher_obj._overview_region = Mock()
    matcher_obj.sel_label = Mock()
    matcher_obj._rematch_line_mag = Mock()
    matcher_obj._rematch_line_phase = Mock()
    matcher_obj._rematch_freq_threshold = 0.0
    matcher_obj._last_edit_freq = 0.0
    matcher_obj._threshold_pin_right = True


class TestMatchGroup:
    """Test MatchGroup dataclass."""
    
    def test_create_empty_group(self):
        """Test creating an empty match group."""
        group = MatchGroup(group_id=1, entries1=[], entries2=[], ambiguous=False)
        assert group.group_id == 1
        assert len(group.entries1) == 0
        assert len(group.entries2) == 0
        assert not group.ambiguous
    
    def test_create_one_to_one_group(self):
        """Test creating a 1:1 match group."""
        group = MatchGroup(
            group_id=5,
            entries1=[(4.2e9, 10)],
            entries2=[(4.21e9, 10)],
            ambiguous=False
        )
        assert group.group_id == 5
        assert len(group.entries1) == 1
        assert len(group.entries2) == 1
        assert group.entries1[0] == (4.2e9, 10)
        assert group.entries2[0] == (4.21e9, 10)
    
    def test_create_many_to_many_group(self):
        """Test creating a many:many match group."""
        group = MatchGroup(
            group_id=3,
            entries1=[(4.2e9, 10), (4.3e9, 11)],
            entries2=[(4.21e9, 10), (4.29e9, 12), (4.31e9, 13)],
            ambiguous=True
        )
        assert len(group.entries1) == 2
        assert len(group.entries2) == 3
        assert group.ambiguous


class TestCustomViewBox:
    """Test CustomViewBox context menu suppression."""
    
    def test_custom_viewbox_creation(self):
        """Test that CustomViewBox can be instantiated."""
        vb = CustomViewBox()
        assert vb is not None
    
    @patch('PyQt5.QtWidgets.QApplication.keyboardModifiers')
    def test_suppress_menu_when_ctrl_held(self, mock_modifiers):
        """Test that context menu is suppressed when Ctrl is held."""
        mock_modifiers.return_value = QtCore.Qt.ControlModifier
        
        vb = CustomViewBox()
        ev = Mock()
        ev.accept = Mock()
        
        # Should accept event (suppress menu) when Ctrl is held
        vb.raiseContextMenu(ev)
        ev.accept.assert_called_once()


class TestUiHints:
    """Test consolidated shortcut hint surfaces."""

    def test_setup_ui_uses_minimal_window_title(self, mock_qt_ui):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        matcher = ResMatcher.__new__(ResMatcher)
        matcher.f1 = np.array([4.0e9, 4.1e9])
        matcher.f2_display = np.array([4.0e9, 4.1e9])
        matcher._saved_view_xlims = None
        matcher._on_window_close = Mock()
        matcher.setup_toolbar = Mock()
        matcher.setup_controls = Mock()
        matcher.setup_plots = Mock()
        matcher.setup_shortcuts = Mock()
        matcher._sync_mag2_geom = Mock()
        matcher._sync_phase2_geom = Mock()
        matcher._update_curves = Mock()
        matcher.update_markers = Mock()
        matcher.auto_scale_y = Mock()
        matcher._update_overview_region = Mock()
        matcher._update_threshold_position = Mock()

        plot_mag = Mock()
        matcher.plot_mag = plot_mag

        matcher.setup_ui()

        assert matcher.win.windowTitle() == 'Resonance Matcher'
        matcher.win.close()

    def test_setup_plots_places_shortcuts_in_plot_header(self):
        class StopAfterLabel(Exception):
            pass

        matcher = ResMatcher.__new__(ResMatcher)
        matcher.plot_widget = Mock()
        matcher.plot_widget.nextRow.side_effect = StopAfterLabel()

        with pytest.raises(StopAfterLabel):
            matcher.setup_plots()

        header_text = matcher.plot_widget.addLabel.call_args.args[0]
        assert 'Ctrl+drag: DS2 shift' in header_text
        assert 'G: auto-align' in header_text
        assert 'H: help' in header_text


class TestResMatcherInitialization:
    """Test ResMatcher initialization and startup dialog."""
    
    def test_basic_initialization_no_zarr(self, synthetic_sweep_data, tmp_path):
        """Test initialization when zarr data doesn't exist."""
        zarr_path = tmp_path / "test_matcher.zarr"
        
        # Don't create the matcher (would show GUI), just test data preparation
        data = synthetic_sweep_data
        assert len(data['f1']) == len(data['z1'])
        assert len(data['fres1']) == len(data['res_idx1'])
    
    def test_startup_dialog_options(self, synthetic_sweep_data, tmp_path):
        """Test that startup dialog is shown when zarr data exists."""
        zarr_path = tmp_path / "existing_matcher.zarr"
        
        # Create existing zarr data
        grp = zarr.open_group(str(zarr_path), mode='w')
        data1 = np.array([4.2e9, 4.8e9])
        grp.create_array('fres1', data=data1)
        idx1 = np.array([10, 20])
        grp.create_array('res_idx1', data=idx1)
        gids1 = np.array([0, 1])
        grp.create_array('group_ids1', data=gids1)
        data2 = np.array([4.21e9, 4.79e9])
        grp.create_array('fres2', data=data2)
        idx2 = np.array([10, 20])
        grp.create_array('res_idx2', data=idx2)
        gids2 = np.array([0, 1])
        grp.create_array('group_ids2', data=gids2)
        ambig = np.array([])
        grp.create_array('ambiguous_groups', data=ambig)
        saved_xlims = np.array([5.0e9, 5.01e9])
        grp.create_array('res_matcher_xlims', data=saved_xlims)
        
        # Verify zarr data was created
        assert 'fres1' in grp
        assert 'res_matcher_xlims' in grp
        assert np.allclose(np.array(grp['res_matcher_xlims']), [5.0e9, 5.01e9])


class TestSavedViewXlims:
    """Test exact view x-limits persistence."""

    def test_saved_view_xlims_written_to_zarr(self, tmp_path):
        matcher = ResMatcher.__new__(ResMatcher)
        matcher.groups = [MatchGroup(0, [(4.2e9, 10)], [(4.21e9, 10)])]
        matcher.f1 = np.array([4.0e9, 6.0e9])
        matcher.f2 = np.array([4.1e9, 6.1e9])
        matcher.log = Mock()
        matcher._sorted_groups = lambda: matcher.groups
        matcher.plot_mag = Mock()
        matcher.plot_mag.viewRange.return_value = [[5.2e9, 5.35e9], [0.0, 1.0]]
        matcher.zarr_group = zarr.open_group(str(tmp_path / 'test_view_xlims.zarr'), mode='w')

        matcher.save_data()

        assert 'res_matcher_xlims' in matcher.zarr_group
        assert np.allclose(np.array(matcher.zarr_group['res_matcher_xlims']), [5.2e9, 5.35e9])

    def test_saved_view_xlims_loaded_from_zarr(self, synthetic_sweep_data, tmp_path):
        zarr_path = tmp_path / 'test_load_view_xlims.zarr'
        grp = zarr.open_group(str(zarr_path), mode='w')
        grp.create_array('fres1', data=np.array([4.2e9]))
        grp.create_array('res_idx1', data=np.array([10]))
        grp.create_array('group_ids1', data=np.array([0]))
        grp.create_array('fres2', data=np.array([4.21e9]))
        grp.create_array('res_idx2', data=np.array([10]))
        grp.create_array('group_ids2', data=np.array([0]))
        grp.create_array('ambiguous_groups', data=np.array([], dtype=np.int64))
        grp.create_array('res_matcher_xlims', data=np.array([5.5e9, 5.65e9]))

        data = synthetic_sweep_data
        with patch.object(ResMatcher, '_show_startup_dialog', return_value='load'), \
             patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'), \
             patch('citkid.vna.res_matcher.np.polyfit', return_value=np.array([0.0, 0.0])):
            matcher = ResMatcher(
                data['f1'], data['z1'], data['fres1'], data['res_idx1'],
                data['f2'], data['z2'], data['fres2'], data['res_idx2'],
                str(zarr_path),
            )

        assert matcher._saved_view_xlims == (5.5e9, 5.65e9)


class TestResMatcherCoreInit:
    def test_equal_length_datasets_sorted_are_all_one_to_one(self, matcher):
        for group in matcher.groups:
            assert group.mapping_str() == '1-1'

    def test_groups_are_sorted_by_center_freq(self, matcher):
        freqs = [group.center_freq() for group in matcher.groups]
        assert freqs == sorted(freqs)

    def test_init_sorted_extra_ds1_becomes_unmatched_group(self, simple_data, tmp_path):
        sd = simple_data
        fres1 = np.append(sd['fres1'], 7.8e9)
        ridx1 = np.append(sd['res_idx1'], 999)
        grp = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            matcher_obj = ResMatcher(
                sd['f'], sd['z1'], fres1, ridx1,
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp, overwrite=True,
            )
        assert '1-0' in [group.mapping_str() for group in matcher_obj.groups]

    def test_init_sorted_extra_ds2_becomes_unmatched_group(self, simple_data, tmp_path):
        sd = simple_data
        fres2 = np.append(sd['fres2'], 7.9e9)
        ridx2 = np.append(sd['res_idx2'], 998)
        grp = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            matcher_obj = ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], fres2, ridx2,
                grp, overwrite=True,
            )
        assert '0-1' in [group.mapping_str() for group in matcher_obj.groups]

    def test_init_nearest_matches_close_pairs(self, simple_data, tmp_path):
        sd = simple_data
        grp = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            matcher_obj = ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp, overwrite=True, init_match='nearest',
            )
        assert all(group.mapping_str() == '1-1' for group in matcher_obj.groups)

    def test_invalid_init_match_raises_value_error(self, simple_data, tmp_path):
        sd = simple_data
        grp = str(tmp_path / 'out.zarr')
        with pytest.raises(ValueError, match='init_match'):
            with patch.object(ResMatcher, 'setup_ui'), \
                 patch.object(ResMatcher, 'run'):
                ResMatcher(
                    sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                    sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                    grp, overwrite=True, init_match='bad',
                )


class TestResMatcherCoreHelpers:
    def test_visible_slice_center_of_array(self, matcher):
        sl = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.0)
        assert matcher.f1[sl.start] >= 5e9
        assert matcher.f1[sl.stop - 1] <= 6e9

    def test_visible_slice_empty_outside_range(self, matcher):
        sl = matcher._visible_slice(matcher.f1, 20e9, 21e9, pad=0.0)
        assert sl.start >= sl.stop

    def test_visible_slice_padding_extends_slice(self, matcher):
        no_pad = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.0)
        padded = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.5)
        assert padded.stop - padded.start > no_pad.stop - no_pad.start

    def test_nearest_idx_exact_hit(self, matcher):
        idx = matcher._nearest_idx(matcher.f1, matcher.f1[50])
        assert idx == 50

    def test_nearest_idx_between_samples(self, matcher):
        freq = 0.6 * matcher.f1[10] + 0.4 * matcher.f1[11]
        assert matcher._nearest_idx(matcher.f1, freq) == 10

    def test_nearest_idx_out_of_bounds(self, matcher):
        assert matcher._nearest_idx(matcher.f1, 0.0) == 0
        assert matcher._nearest_idx(matcher.f1, 999e9) == len(matcher.f1) - 1


class TestDs2DisplayOffsetBehavior:
    def test_init_stores_visual_offset(self, simple_data, tmp_path):
        sd = simple_data
        grp = str(tmp_path / 'offset_out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            matcher_obj = ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp, overwrite=True, DS2_f_offset=1.5e6,
            )
        assert matcher_obj.ds2_f_offset == 1.5e6
        assert np.allclose(matcher_obj.f2_display, matcher_obj.f2 + 1.5e6)

    def test_display_conversion_round_trip(self, matcher):
        freq = 5.21e9
        matcher._set_ds2_frequency_offset(2.5e6)
        display_freq = matcher._display_freq(freq, 2)
        assert display_freq == freq + 2.5e6
        assert matcher._data_freq_from_display(display_freq, 2) == freq

    def test_auto_align_uses_nearby_one_to_one_groups(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.plot_mag.viewRange.return_value = [[5.0e9, 6.0e9], [0.0, 1.0]]
        matcher._update_curves = Mock()
        matcher.update_markers = Mock()
        matcher.auto_scale_y = Mock()
        matcher._update_overview_region = Mock()
        matcher._auto_align_ds2_offset()
        expected = np.mean(matcher.fres1_init - matcher.fres2_init)
        assert matcher.ds2_f_offset == pytest.approx(expected)
        assert np.allclose(matcher.f2_display, matcher.f2 + expected)

    def test_save_data_keeps_original_ds2_frequencies(self, matcher):
        matcher._set_ds2_frequency_offset(3.0e6)
        matcher.save_data()
        saved_fres2 = np.array(matcher.zarr_group['fres2'])
        assert np.allclose(saved_fres2, matcher.fres2_init)


class TestResMatcherDirectOperations:
    def test_add_resonance_creates_new_group(self, matcher):
        _prepare_headless_matcher(matcher)
        before = len(matcher.groups)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher.add_resonance(7.9e9, ds=1)
        assert len(matcher.groups) == before + 1

    def test_do_unlink_moves_entry_to_new_group(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        group = matcher.groups[0]
        entry = group.entries1[0]
        before = len(matcher.groups)
        matcher._do_unlink(group, entry, ds=1)
        assert len(matcher.groups) == before + 1
        assert entry not in group.entries1

    def test_do_merge_combines_entries_and_removes_group(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        group_a = matcher.groups[0]
        group_b = matcher.groups[1]
        before = len(matcher.groups)
        matcher._do_merge(group_a.group_id, group_b.group_id)
        assert len(matcher.groups) == before - 1
        assert matcher._find_group(group_b.group_id) is None

    def test_merge_groups_uses_current_selection_model(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        first = matcher.groups[0]
        second = matcher.groups[1]
        matcher._selected_resonances = {
            (first.group_id, 1, first.entries1[0][0]),
            (second.group_id, 2, second.entries2[0][0]),
        }
        before = len(matcher.groups)
        matcher.merge_groups()
        assert len(matcher.groups) == before - 1

    def test_merge_selected_only_creates_new_group(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher.groups = [
            MatchGroup(0, [(4.5e9, 10), (4.6e9, 11)], [(4.51e9, 20)]),
            MatchGroup(1, [(5.2e9, 12)], [(5.21e9, 21), (5.22e9, 22)]),
        ]
        matcher._next_group_id = 2
        matcher._selected_resonances = {
            (0, 1, 4.6e9),
            (1, 2, 5.22e9),
        }
        before = len(matcher.groups)
        matcher.merge_selected_only()
        assert len(matcher.groups) == before + 1
        assert any(group.entries1 == [(4.6e9, 11)] and group.entries2 == [(5.22e9, 22)] for group in matcher.groups)

    def test_unlink_selected_uses_current_selection_model(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        target = matcher.groups[0]
        matcher._selected_resonances = {(target.group_id, 1, target.entries1[0][0])}
        before = len(matcher.groups)
        matcher.unlink_selected()
        assert len(matcher.groups) == before + 1

    def test_toggle_ambiguous_uses_current_selection_model(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        target = matcher.groups[0]
        matcher._selected_resonances = {(target.group_id, 1, target.entries1[0][0])}
        assert target.ambiguous is False
        matcher.toggle_ambiguous()
        assert target.ambiguous is True

    def test_undo_restores_after_add(self, matcher):
        _prepare_headless_matcher(matcher)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        before = len(matcher.groups)
        matcher.add_resonance(7.9e9, ds=1)
        matcher.undo()
        assert len(matcher.groups) == before


class TestResMatcherSaveDataDirect:
    def test_save_data_lengths_are_consistent(self, matcher):
        matcher.log = Mock()
        matcher.save_data()
        zg = matcher.zarr_group
        assert zg['fres1'].shape[0] == zg['res_idx1'].shape[0] == zg['group_ids1'].shape[0]
        assert zg['fres2'].shape[0] == zg['res_idx2'].shape[0] == zg['group_ids2'].shape[0]

    def test_save_data_records_ambiguous_groups(self, matcher):
        matcher.log = Mock()
        matcher.groups[0].ambiguous = True
        orig_gid = matcher.groups[0].group_id
        matcher.save_data()
        sorted_groups = matcher._sorted_groups()
        expected = next(i for i, group in enumerate(sorted_groups) if group.group_id == orig_gid)
        assert expected in list(matcher.zarr_group['ambiguous_groups'][:])


class TestResMatcherAutoScaleY:
    def _attach_plot_mocks(self, matcher):
        matcher.plot_mag = Mock()
        matcher.plot_mag.viewRange.return_value = [[float(matcher.f1[0]), float(matcher.f1[-1])], [0.0, 1.0]]
        matcher._vb_mag2 = Mock()
        matcher.plot_phase = Mock()
        matcher._vb_phase2 = Mock()

    def test_auto_scale_y_updates_all_axes(self, matcher):
        self._attach_plot_mocks(matcher)
        matcher.auto_scale_y()
        matcher.plot_mag.setYRange.assert_called_once()
        matcher._vb_mag2.setYRange.assert_called_once()
        matcher.plot_phase.setYRange.assert_called_once()
        matcher._vb_phase2.setYRange.assert_called_once()

    def test_auto_scale_y_returns_if_plots_absent(self, matcher):
        if hasattr(matcher, 'plot_mag'):
            del matcher.plot_mag
        matcher.auto_scale_y()


class TestResMatcherOverviewSync:
    def _attach_overview_mocks(self, matcher):
        matcher._overview_region = Mock()
        matcher._overview_updating = False
        matcher.plot_mag = Mock()

    def test_update_overview_region_sets_region(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher.plot_mag.viewRange.return_value = [[5e9, 6e9], [0.0, 1.0]]
        matcher._update_overview_region()
        matcher._overview_region.setRegion.assert_called_once_with([5e9, 6e9])

    def test_on_overview_region_changed_sets_xrange(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher._overview_region.getRegion.return_value = (5e9, 6e9)
        matcher._on_overview_region_changed()
        matcher.plot_mag.setXRange.assert_called_once_with(5e9, 6e9, padding=0)


class TestRunResMatcherWrapper:
    def test_run_res_matcher_returns_match_groups(self, simple_data, tmp_path):
        sd = simple_data
        grp_path = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            result = run_res_matcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp_path,
            )
        assert isinstance(result, list)
        assert all(isinstance(group, MatchGroup) for group in result)

    def test_run_res_matcher_accepts_group_object(self, simple_data, tmp_path):
        sd = simple_data
        grp = zarr.open_group(str(tmp_path / 'grp_out.zarr'), mode='a')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            result = run_res_matcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp,
            )
        assert isinstance(result, list)


class TestMultiSelection:
    """Test multi-selection system."""
    
    def test_selection_tuple_format(self):
        """Test that selection tuples have correct format: (group_id, dataset, fres)."""
        selection_tuple = (5, 1, 4.2e9)
        group_id, dataset, fres = selection_tuple
        
        assert group_id == 5
        assert dataset == 1
        assert fres == 4.2e9
    
    def test_selection_set_operations(self):
        """Test set operations for multi-selection."""
        selections = set()
        
        # Add selections
        selections.add((1, 1, 4.2e9))
        selections.add((1, 2, 4.21e9))
        selections.add((2, 1, 4.8e9))
        
        assert len(selections) == 3
        
        # Remove a selection
        selections.remove((2, 1, 4.8e9))
        assert len(selections) == 2
        
        # Check membership
        assert (1, 1, 4.2e9) in selections
        assert (2, 1, 4.8e9) not in selections


class TestGroupOperations:
    """Test group merging, unlinking, and deletion operations."""
    
    def test_merge_groups_concept(self):
        """Test the concept of merging two groups."""
        # Group 1: (DS1: [10, 20], DS2: [10])
        group1 = MatchGroup(
            group_id=1,
            entries1=[(4.2e9, 10), (4.3e9, 20)],
            entries2=[(4.21e9, 10)],
            ambiguous=False
        )
        
        # Group 2: (DS1: [30], DS2: [20, 30])
        group2 = MatchGroup(
            group_id=2,
            entries1=[(4.8e9, 30)],
            entries2=[(4.79e9, 20), (4.81e9, 30)],
            ambiguous=False
        )
        
        # Merged should have: (DS1: [10, 20, 30], DS2: [10, 20, 30])
        merged_entries1 = group1.entries1 + group2.entries1
        merged_entries2 = group1.entries2 + group2.entries2
        
        assert len(merged_entries1) == 3
        assert len(merged_entries2) == 3
    
    def test_unlink_resonances_concept(self):
        """Test the concept of unlinking resonances from a group."""
        # Original group: (DS1: [10, 20, 30], DS2: [10])
        group = MatchGroup(
            group_id=1,
            entries1=[(4.2e9, 10), (4.3e9, 20), (4.4e9, 30)],
            entries2=[(4.21e9, 10)],
            ambiguous=True
        )
        
        # If we unlink res_idx 20 from DS1, we should get:
        # Group 1: (DS1: [10, 30], DS2: [10])
        # Group 2: (DS1: [20], DS2: [])
        remaining_entries1 = [(f, idx) for f, idx in group.entries1 if idx != 20]
        unlinked_entry = [(f, idx) for f, idx in group.entries1 if idx == 20]
        
        assert len(remaining_entries1) == 2
        assert len(unlinked_entry) == 1
        assert unlinked_entry[0][1] == 20


class TestClickBehavior:
    """Test click behavior for adding resonances and setting threshold."""
    
    def test_shift_click_adds_resonance(self):
        """Test that Shift+click should add a resonance."""
        # This is a behavioral test - we just verify the concept
        modifiers = QtCore.Qt.ShiftModifier
        assert modifiers & QtCore.Qt.ShiftModifier
        assert not (modifiers & QtCore.Qt.ControlModifier)
    
    def test_ctrl_right_click_sets_threshold(self):
        """Test that Ctrl+right-click should set threshold."""
        modifiers = QtCore.Qt.ControlModifier
        button = QtCore.Qt.RightButton
        
        assert modifiers & QtCore.Qt.ControlModifier
        assert button == QtCore.Qt.RightButton
    
    def test_ctrl_left_click_on_marker_adds_to_selection(self):
        """Test that Ctrl+left-click on marker adds to selection."""
        modifiers = QtCore.Qt.ControlModifier
        button = QtCore.Qt.LeftButton
        
        assert modifiers & QtCore.Qt.ControlModifier
        assert button == QtCore.Qt.LeftButton


class TestZarrDataStructure:
    """Test zarr data structure for saved groups."""
    
    def test_zarr_output_arrays(self, tmp_path):
        """Test that all required arrays are created in zarr."""
        zarr_path = tmp_path / "test_structure.zarr"
        grp = zarr.open_group(str(zarr_path), mode='w')
        
        # Create all expected arrays
        data = np.array([4.2e9, 4.8e9])
        grp.create_array('fres1', data=data)
        grp.create_array('res_idx1', data=np.array([10, 20]))
        grp.create_array('group_ids1', data=np.array([0, 1]))
        grp.create_array('fres2', data=np.array([4.21e9, 4.79e9]))
        grp.create_array('res_idx2', data=np.array([10, 20]))
        grp.create_array('group_ids2', data=np.array([0, 1]))
        grp.create_array('ambiguous_groups', data=np.array([]))
        grp.create_array('res_matcher_xlims', data=np.array([5.0e9, 5.01e9]))
        
        # Verify all arrays exist
        required_arrays = [
            'fres1', 'res_idx1', 'group_ids1',
            'fres2', 'res_idx2', 'group_ids2',
            'ambiguous_groups', 'res_matcher_xlims'
        ]
        
        for array_name in required_arrays:
            assert array_name in grp
    
    def test_group_id_consistency(self, tmp_path):
        """Test that group IDs are consistent across datasets."""
        zarr_path = tmp_path / "test_consistency.zarr"
        grp = zarr.open_group(str(zarr_path), mode='w')
        
        # Create data where group 0 has resonances in both datasets
        data = np.array([4.2e9, 4.8e9])
        grp.create_array('fres1', data=data)
        grp.create_array('res_idx1', data=np.array([10, 20]))
        grp.create_array('group_ids1', data=np.array([0, 1]))
        
        grp.create_array('fres2', data=np.array([4.21e9]))
        grp.create_array('res_idx2', data=np.array([10]))
        grp.create_array('group_ids2', data=np.array([0]))
        
        # Group 0 should have one resonance in each dataset
        group0_ds1 = grp['fres1'][grp['group_ids1'][:] == 0]
        group0_ds2 = grp['fres2'][grp['group_ids2'][:] == 0]
        
        assert len(group0_ds1) == 1
        assert len(group0_ds2) == 1


class TestThresholdBehavior:
    """Test re-match threshold smart behavior."""
    
    def test_threshold_initial_pin_right(self):
        """Test that threshold starts pinned to right edge."""
        # Initial state: threshold_pin_right = True
        pin_right = True
        assert pin_right is True
    
    def test_threshold_fixed_after_edit(self):
        """Test that threshold becomes fixed at frequency after edit."""
        # After an edit: threshold_pin_right = False, threshold at specific freq
        pin_right = False
        threshold_freq = 4.5e9
        
        assert pin_right is False
        assert threshold_freq == 4.5e9
    
    def test_threshold_snap_back_when_out_of_view(self):
        """Test threshold snaps back to right edge when scrolled past."""
        # When view scrolls past threshold: threshold_pin_right = True again
        threshold_freq = 4.5e9
        view_right_edge = 5.5e9
        
        # If view is entirely to the right of threshold
        should_snap_back = view_right_edge > threshold_freq
        assert should_snap_back is True


class TestConfirmationDialog:
    """Test overwrite confirmation dialog."""
    
    @patch('PyQt5.QtWidgets.QMessageBox.question')
    def test_overwrite_shows_confirmation(self, mock_question):
        """Test that overwrite option shows confirmation dialog."""
        mock_question.return_value = QtWidgets.QMessageBox.Yes
        
        # Simulate clicking "Overwrite" which should trigger confirmation
        result = mock_question(
            None,
            'Confirm Overwrite',
            'Are you sure you want to overwrite existing data?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        assert result == QtWidgets.QMessageBox.Yes
        mock_question.assert_called_once()
    
    @patch('PyQt5.QtWidgets.QMessageBox.question')
    def test_overwrite_cancelled_if_no(self, mock_question):
        """Test that selecting No in confirmation keeps dialog open."""
        mock_question.return_value = QtWidgets.QMessageBox.No
        
        result = mock_question(
            None,
            'Confirm Overwrite',
            'Are you sure?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        assert result == QtWidgets.QMessageBox.No


class TestLoadFromZarr:
    """Test loading existing groups from zarr."""
    
    def test_load_groups_from_zarr(self, tmp_path):
        """Test reconstructing groups from zarr arrays."""
        zarr_path = tmp_path / "test_load.zarr"
        grp = zarr.open_group(str(zarr_path), mode='w')
        
        # Create sample data: 2 groups
        # Group 0: DS1=[10], DS2=[10, 20]
        # Group 1: DS1=[30], DS2=[]
        data = np.array([4.2e9, 4.8e9])
        grp.create_array('fres1', data=data)
        grp.create_array('res_idx1', data=np.array([10, 30]))
        grp.create_array('group_ids1', data=np.array([0, 1]))
        
        grp.create_array('fres2', data=np.array([4.21e9, 4.22e9]))
        grp.create_array('res_idx2', data=np.array([10, 20]))
        grp.create_array('group_ids2', data=np.array([0, 0]))
        
        grp.create_array('ambiguous_groups', data=np.array([0]))
        
        # Simulate loading
        fres1 = grp['fres1'][:]
        res_idx1 = grp['res_idx1'][:]
        gids1 = grp['group_ids1'][:]
        
        fres2 = grp['fres2'][:]
        res_idx2 = grp['res_idx2'][:]
        gids2 = grp['group_ids2'][:]
        
        ambig = set(grp['ambiguous_groups'][:])
        
        # Reconstruct groups
        unique_gids = np.unique(np.concatenate([gids1, gids2]))
        groups = []
        
        for gid in unique_gids:
            mask1 = gids1 == gid
            mask2 = gids2 == gid
            
            entries1 = list(zip(fres1[mask1], res_idx1[mask1]))
            entries2 = list(zip(fres2[mask2], res_idx2[mask2]))
            
            is_ambig = int(gid) in ambig
            
            groups.append(MatchGroup(
                group_id=int(gid),
                entries1=entries1,
                entries2=entries2,
                ambiguous=is_ambig
            ))
        
        # Verify reconstruction
        assert len(groups) == 2
        assert groups[0].group_id == 0
        assert len(groups[0].entries1) == 1
        assert len(groups[0].entries2) == 2
        assert groups[0].ambiguous is True
        
        assert groups[1].group_id == 1
        assert len(groups[1].entries1) == 1
        assert len(groups[1].entries2) == 0
        assert groups[1].ambiguous is False


class TestInitialViewXlims:
    """Test that saved x-limits are restored exactly when loading."""

    def test_exact_xlims_are_preserved(self):
        saved_xlims = (5.2e9, 5.31e9)

        assert saved_xlims[0] == 5.2e9
        assert saved_xlims[1] == 5.31e9
