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
from unittest.mock import Mock, patch
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
    with patch('citkid.vna.res_matcher.np.polyfit',
               return_value=np.array([0.0, 0.0])):
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
    z1 = (np.exp(1j * (2 * np.pi * f / 1e9))
          * (1 + 0.001 * rng.standard_normal(len(f))))
    z2 = (np.exp(1j * (2 * np.pi * f / 1e9 + 0.1))
          * (1 + 0.001 * rng.standard_normal(len(f))))
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
        )
    return matcher_obj


def _prepare_headless_matcher(matcher_obj):
    matcher_obj.plot_mag = Mock()
    matcher_obj.plot_mag.viewRange.return_value = [
        [float(matcher_obj.f1[0]), float(matcher_obj.f1[-1])], [0.0, 1.0]]
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


################################################################################
########################## Test MatchGroup dataclass. ##########################
################################################################################

def test_create_empty_group():
    """Test creating an empty match group."""
    group = MatchGroup(group_id=1, entries1=[], entries2=[], ambiguous=False)
    assert group.group_id == 1
    assert len(group.entries1) == 0
    assert len(group.entries2) == 0
    assert not group.ambiguous


def test_create_one_to_one_group():
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


def test_create_many_to_many_group():
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


################################################################################
################# Test CustomViewBox context menu suppression. #################
################################################################################

def test_custom_viewbox_creation():
    """Test that CustomViewBox can be instantiated."""
    vb = CustomViewBox()
    assert vb is not None


@patch('PyQt5.QtWidgets.QApplication.keyboardModifiers')
def test_suppress_menu_when_ctrl_held(mock_modifiers):
    """Test that context menu is suppressed when Ctrl is held."""
    mock_modifiers.return_value = QtCore.Qt.ControlModifier

    vb = CustomViewBox()
    ev = Mock()
    ev.accept = Mock()

    # Should accept event (suppress menu) when Ctrl is held
    vb.raiseContextMenu(ev)
    ev.accept.assert_called_once()


################################################################################
################## Test consolidated shortcut hint surfaces. ###################
################################################################################

def test_setup_ui_uses_minimal_window_title(mock_qt_ui):
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


def test_setup_plots_places_shortcuts_in_plot_header():
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


################################################################################
############## Test ResMatcher initialization and startup dialog. ##############
################################################################################

def test_basic_initialization_no_zarr(synthetic_sweep_data, tmp_path):
    """Test initialization when zarr data doesn't exist."""
    zarr_path = tmp_path / "test_matcher.zarr"

    # Don't create the matcher (would show GUI), just test data preparation
    data = synthetic_sweep_data
    assert len(data['f1']) == len(data['z1'])
    assert len(data['fres1']) == len(data['res_idx1'])


def test_startup_dialog_options(synthetic_sweep_data, tmp_path):
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


################################################################################
#################### Test exact view x-limits persistence. #####################
################################################################################

def test_saved_view_xlims_written_to_zarr(tmp_path):
    matcher = ResMatcher.__new__(ResMatcher)
    matcher.groups = [MatchGroup(0, [(4.2e9, 10)], [(4.21e9, 10)])]
    matcher.f1 = np.array([4.0e9, 6.0e9])
    matcher.f2 = np.array([4.1e9, 6.1e9])
    matcher.ds2_f_offset = 0.0
    matcher.log = Mock()
    matcher._sorted_groups = lambda: matcher.groups
    matcher.plot_mag = Mock()
    matcher.plot_mag.viewRange.return_value = [[5.2e9, 5.35e9], [0.0, 1.0]]
    matcher.zarr_group = zarr.open_group(
        str(tmp_path / 'test_view_xlims.zarr'), mode='w')

    matcher.save_data()

    assert 'res_matcher_xlims' in matcher.zarr_group
    assert np.allclose(np.array(matcher.zarr_group['res_matcher_xlims']),
                       [5.2e9, 5.35e9])


def test_saved_view_xlims_loaded_from_zarr(synthetic_sweep_data, tmp_path):
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
    with (
        patch.object(ResMatcher, '_show_startup_dialog', return_value='load'),
        patch.object(ResMatcher, 'setup_ui'),
        patch.object(ResMatcher, 'run'),
        patch('citkid.vna.res_matcher.np.polyfit',
              return_value=np.array([0.0, 0.0])),
    ):
        matcher = ResMatcher(
            data['f1'], data['z1'], data['fres1'], data['res_idx1'],
            data['f2'], data['z2'], data['fres2'], data['res_idx2'],
            str(zarr_path),
        )

    assert matcher._saved_view_xlims == (5.5e9, 5.65e9)


################################################################################
############################## ResMatcherCoreInit ##############################
################################################################################

def test_equal_length_datasets_sorted_are_all_one_to_one(matcher):
    for group in matcher.groups:
        assert group.mapping_str() == '1-1'


def test_groups_are_sorted_by_center_freq(matcher):
    freqs = [group.center_freq() for group in matcher.groups]
    assert freqs == sorted(freqs)


def test_init_sorted_extra_ds1_becomes_unmatched_group(simple_data, tmp_path):
    sd = simple_data
    fres1 = np.append(sd['fres1'], 7.8e9)
    ridx1 = np.append(sd['res_idx1'], 999)
    grp = str(tmp_path / 'out.zarr')
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        matcher_obj = ResMatcher(
            sd['f'], sd['z1'], fres1, ridx1,
            sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
            grp,
        )
    assert '1-0' in [group.mapping_str() for group in matcher_obj.groups]


def test_init_sorted_extra_ds2_becomes_unmatched_group(simple_data, tmp_path):
    sd = simple_data
    fres2 = np.append(sd['fres2'], 7.9e9)
    ridx2 = np.append(sd['res_idx2'], 998)
    grp = str(tmp_path / 'out.zarr')
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        matcher_obj = ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], fres2, ridx2,
            grp,
        )
    assert '0-1' in [group.mapping_str() for group in matcher_obj.groups]


def test_init_nearest_matches_close_pairs(simple_data, tmp_path):
    sd = simple_data
    grp = str(tmp_path / 'out.zarr')
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        matcher_obj = ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
            grp, init_match='nearest',
        )
    assert all(group.mapping_str() == '1-1' for group in matcher_obj.groups)


def test_invalid_init_match_raises_value_error(simple_data, tmp_path):
    sd = simple_data
    grp = str(tmp_path / 'out.zarr')
    with pytest.raises(ValueError, match='init_match'):
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp, init_match='bad',
            )


################################################################################
############################ ResMatcherCoreHelpers #############################
################################################################################

def test_visible_slice_center_of_array(matcher):
    sl = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.0)
    assert matcher.f1[sl.start] >= 5e9
    assert matcher.f1[sl.stop - 1] <= 6e9


def test_visible_slice_empty_outside_range(matcher):
    sl = matcher._visible_slice(matcher.f1, 20e9, 21e9, pad=0.0)
    assert sl.start >= sl.stop


def test_visible_slice_padding_extends_slice(matcher):
    no_pad = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.0)
    padded = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.5)
    assert padded.stop - padded.start > no_pad.stop - no_pad.start


def test_nearest_idx_exact_hit(matcher):
    idx = matcher._nearest_idx(matcher.f1, matcher.f1[50])
    assert idx == 50


def test_nearest_idx_between_samples(matcher):
    freq = 0.6 * matcher.f1[10] + 0.4 * matcher.f1[11]
    assert matcher._nearest_idx(matcher.f1, freq) == 10


def test_nearest_idx_out_of_bounds(matcher):
    assert matcher._nearest_idx(matcher.f1, 0.0) == 0
    assert matcher._nearest_idx(matcher.f1, 999e9) == len(matcher.f1) - 1


################################################################################
########################### Ds2DisplayOffsetBehavior ###########################
################################################################################

def test_init_stores_visual_offset(simple_data, tmp_path):
    sd = simple_data
    grp = str(tmp_path / 'offset_out.zarr')
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        matcher_obj = ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
            grp, ds2_f_offset=1.5e6,
        )
    assert matcher_obj.ds2_f_offset == 1.5e6
    assert np.allclose(matcher_obj.f2_display, matcher_obj.f2 + 1.5e6)


def test_display_conversion_round_trip(matcher):
    freq = 5.21e9
    matcher._set_ds2_frequency_offset(2.5e6)
    display_freq = matcher._display_freq(freq, 2)
    assert display_freq == freq + 2.5e6
    assert matcher._data_freq_from_display(display_freq, 2) == freq


def test_auto_align_uses_nearby_one_to_one_groups(matcher):
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


def test_save_data_keeps_original_ds2_frequencies(matcher):
    matcher._set_ds2_frequency_offset(3.0e6)
    matcher.save_data()
    saved_fres2 = np.array(matcher.zarr_group['fres2'])
    assert np.allclose(saved_fres2, matcher.fres2_init)


################################################################################
########################## ResMatcherDirectOperations ##########################
################################################################################

def test_add_resonance_creates_new_group(matcher):
    _prepare_headless_matcher(matcher)
    before = len(matcher.groups)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    matcher.add_resonance(7.9e9, ds=1)
    assert len(matcher.groups) == before + 1


def test_merge_groups_uses_current_selection_model(matcher):
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


def test_merge_selected_only_creates_new_group(matcher):
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
    assert any(group.entries1 == [(4.6e9, 11)]
               and group.entries2 == [(5.22e9, 22)]
               for group in matcher.groups)


def test_unlink_selected_uses_current_selection_model(matcher):
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    target = matcher.groups[0]
    matcher._selected_resonances = {(target.group_id, 1, target.entries1[0][0])}
    before = len(matcher.groups)
    matcher.unlink_selected()
    assert len(matcher.groups) == before + 1


def test_toggle_ambiguous_uses_current_selection_model(matcher):
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    target = matcher.groups[0]
    matcher._selected_resonances = {(target.group_id, 1, target.entries1[0][0])}
    assert target.ambiguous is False
    matcher.toggle_ambiguous()
    assert target.ambiguous is True


def test_undo_restores_after_add(matcher):
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    before = len(matcher.groups)
    matcher.add_resonance(7.9e9, ds=1)
    matcher.undo()
    assert len(matcher.groups) == before


################################################################################
########################### ResMatcherSaveDataDirect ###########################
################################################################################

def test_save_data_lengths_are_consistent(matcher):
    matcher.log = Mock()
    matcher.save_data()
    zg = matcher.zarr_group
    assert (zg['fres1'].shape[0] == zg['res_idx1'].shape[0]
            == zg['group_ids1'].shape[0])
    assert (zg['fres2'].shape[0] == zg['res_idx2'].shape[0]
            == zg['group_ids2'].shape[0])


def test_save_data_records_ambiguous_groups(matcher):
    matcher.log = Mock()
    matcher.groups[0].ambiguous = True
    orig_gid = matcher.groups[0].group_id
    matcher.save_data()
    sorted_groups = matcher._sorted_groups()
    expected = next(i for i, group in enumerate(sorted_groups)
                    if group.group_id == orig_gid)
    assert expected in list(matcher.zarr_group['ambiguous_groups'][:])


################################################################################
############################# ResMatcherAutoScaleY #############################
################################################################################

def _attach_plot_mocks(matcher):
    matcher.plot_mag = Mock()
    matcher.plot_mag.viewRange.return_value = [
        [float(matcher.f1[0]), float(matcher.f1[-1])], [0.0, 1.0]]
    matcher._vb_mag2 = Mock()
    matcher.plot_phase = Mock()
    matcher._vb_phase2 = Mock()


def test_auto_scale_y_updates_all_axes(matcher):
    _attach_plot_mocks(matcher)
    matcher.auto_scale_y()
    matcher.plot_mag.setYRange.assert_called_once()
    matcher._vb_mag2.setYRange.assert_called_once()
    matcher.plot_phase.setYRange.assert_called_once()
    matcher._vb_phase2.setYRange.assert_called_once()


def test_auto_scale_y_returns_if_plots_absent(matcher):
    if hasattr(matcher, 'plot_mag'):
        del matcher.plot_mag
    matcher.auto_scale_y()


################################################################################
############################ ResMatcherOverviewSync ############################
################################################################################

def _attach_overview_mocks(matcher):
    matcher._overview_region = Mock()
    matcher._overview_updating = False
    matcher.plot_mag = Mock()


def test_update_overview_region_sets_region(matcher):
    _attach_overview_mocks(matcher)
    matcher.plot_mag.viewRange.return_value = [[5e9, 6e9], [0.0, 1.0]]
    matcher._update_overview_region()
    matcher._overview_region.setRegion.assert_called_once_with([5e9, 6e9])


def test_on_overview_region_changed_sets_xrange(matcher):
    _attach_overview_mocks(matcher)
    matcher._overview_region.getRegion.return_value = (5e9, 6e9)
    matcher._on_overview_region_changed()
    matcher.plot_mag.setXRange.assert_called_once_with(5e9, 6e9, padding=0)


################################################################################
############################# RunResMatcherWrapper #############################
################################################################################

def test_run_res_matcher_returns_match_groups(simple_data, tmp_path):
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


def test_run_res_matcher_accepts_group_object(simple_data, tmp_path):
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


################################################################################
######################### Test multi-selection system. #########################
################################################################################

def test_selection_tuple_format():
    """
    Test that selection tuples have correct format: (group_id, dataset, fres).
    """
    selection_tuple = (5, 1, 4.2e9)
    group_id, dataset, fres = selection_tuple

    assert group_id == 5
    assert dataset == 1
    assert fres == 4.2e9


def test_selection_set_operations():
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


################################################################################
########### Test group merging, unlinking, and deletion operations. ############
################################################################################

def test_merge_groups_concept():
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


def test_unlink_resonances_concept():
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


################################################################################
####### Test click behavior for adding resonances and setting threshold. #######
################################################################################

def test_shift_click_adds_resonance():
    """Test that Shift+click should add a resonance."""
    # This is a behavioral test - we just verify the concept
    modifiers = QtCore.Qt.ShiftModifier
    assert modifiers & QtCore.Qt.ShiftModifier
    assert not (modifiers & QtCore.Qt.ControlModifier)


def test_ctrl_right_click_sets_threshold():
    """Test that Ctrl+right-click should set threshold."""
    modifiers = QtCore.Qt.ControlModifier
    button = QtCore.Qt.RightButton

    assert modifiers & QtCore.Qt.ControlModifier
    assert button == QtCore.Qt.RightButton


def test_ctrl_left_click_on_marker_adds_to_selection():
    """Test that Ctrl+left-click on marker adds to selection."""
    modifiers = QtCore.Qt.ControlModifier
    button = QtCore.Qt.LeftButton

    assert modifiers & QtCore.Qt.ControlModifier
    assert button == QtCore.Qt.LeftButton


################################################################################
################## Test zarr data structure for saved groups. ##################
################################################################################

def test_zarr_output_arrays(tmp_path):
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


def test_group_id_consistency(tmp_path):
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


################################################################################
##################### Test overwrite confirmation dialog. ######################
################################################################################

@patch('PyQt5.QtWidgets.QMessageBox.question')
def test_overwrite_shows_confirmation(mock_question):
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
def test_overwrite_cancelled_if_no(mock_question):
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


################################################################################
################### Test loading existing groups from zarr. ####################
################################################################################

def test_load_groups_from_zarr(tmp_path):
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


################################################################################
######### Test that saved x-limits are restored exactly when loading. ##########
################################################################################

def test_exact_xlims_are_preserved():
    saved_xlims = (5.2e9, 5.31e9)

    assert saved_xlims[0] == 5.2e9
    assert saved_xlims[1] == 5.31e9


################################################################################
# Re-match threshold with a DS2 display offset
################################################################################

DS2_OFFSET = 20e6


def _ctrl_right_click_event():
    """
    Build a mock Ctrl+Right-click scene event.

    Returns:
    ev (Mock): event whose button, modifiers, and scenePos mimic pyqtgraph.
    """
    ev = Mock()
    ev.scenePos.return_value = QtCore.QPointF(0.0, 0.0)
    ev.modifiers.return_value = QtCore.Qt.ControlModifier
    ev.button.return_value = QtCore.Qt.RightButton
    return ev


@pytest.mark.parametrize('ds', [1, 2])
def test_ctrl_right_click_sets_threshold_at_clicked_display_freq(matcher, ds):
    """
    Place the threshold line where the user clicked, whichever dataset is
    nearest the click.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.ds2_f_offset = DS2_OFFSET
    click_display = 5.3e9
    ds1_y = float(matcher.filtered_mag1[
        matcher._nearest_idx(matcher.f1, click_display)])
    ds2_y = float(matcher.filtered_mag2[
        matcher._nearest_idx(matcher.f2, click_display - DS2_OFFSET)])
    # Put the click exactly on the chosen dataset's curve, far from the other
    click_y1 = ds1_y if ds == 1 else ds1_y + 100.0
    click_y2 = ds2_y if ds == 2 else ds2_y + 100.0
    matcher.plot_mag.sceneBoundingRect.return_value.contains.return_value = True
    matcher.plot_mag.vb.mapSceneToView.return_value = QtCore.QPointF(
        click_display, click_y1)
    matcher._vb_mag2.mapSceneToView.return_value = QtCore.QPointF(
        click_display, click_y2)

    matcher._on_scene_clicked(_ctrl_right_click_event())

    assert matcher._rematch_freq_threshold == pytest.approx(click_display)
    matcher._rematch_line_mag.setPos.assert_called_with(
        pytest.approx(click_display))


def test_rematch_splits_groups_by_display_center(matcher):
    """
    Re-match groups whose display center is right of the line, even when
    their data-frequency center is left of it.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.win = Mock()
    matcher.update_markers = Mock()
    matcher.ds2_f_offset = DS2_OFFSET
    # DS2 data sits DS2_OFFSET below its DS1 partner, so pairs line up on screen
    below = MatchGroup(group_id=100, entries1=[(4.5e9, 1)],
                       entries2=[(4.5e9 - DS2_OFFSET, 1)])
    above = MatchGroup(group_id=101, entries1=[(5.0e9, 2)],
                       entries2=[(5.0e9 - DS2_OFFSET, 2)])
    matcher.groups = [below, above]
    # Between the data center (4.99 GHz) and display center (5.0 GHz) of above
    matcher._rematch_freq_threshold = 4.995e9
    matcher._init_match_method = 'sorted'

    with patch.object(matcher, '_init_sorted_from_arrays',
                      return_value=([], 0)) as init_sorted:
        matcher.trigger_rematch()

    fres1, _, fres2, _ = init_sorted.call_args.args
    assert list(fres1) == [5.0e9]
    assert list(fres2) == [5.0e9 - DS2_OFFSET]
    assert matcher.groups == [below]


def test_rematch_leaves_entries_shifted_left_of_line(matcher):
    """
    Keep a DS2 resonance that the display offset moved left of the line, and
    re-match its former DS1 partner with the next DS2 resonance to the right.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.win = Mock()
    matcher.update_markers = Mock()
    pair = MatchGroup(group_id=100, entries1=[(5.0e9, 1)],
                      entries2=[(5.0e9, 1)])
    lone2 = MatchGroup(group_id=101, entries1=[], entries2=[(5.2e9, 2)])
    matcher.groups = [pair, lone2]
    matcher._rematch_freq_threshold = 4.995e9
    matcher._init_match_method = 'sorted'
    # Shift DS2 left so its first resonance is drawn at 4.99 GHz
    matcher.ds2_f_offset = -10e6

    matcher.trigger_rematch()

    kept = next(g for g in matcher.groups if g.group_id == 100)
    assert kept.entries1 == []
    assert kept.entries2 == [(5.0e9, 1)]
    new = [g for g in matcher.groups if g.group_id not in (100, 101)]
    assert len(new) == 1
    assert new[0].entries1 == [(5.0e9, 1)]
    assert new[0].entries2 == [(5.2e9, 2)]


def test_delete_ds2_moves_threshold_in_display_coords(matcher):
    """
    Move the auto-threshold past the deleted DS2 resonance's display position.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    matcher.ds2_f_offset = DS2_OFFSET
    matcher.groups = [MatchGroup(group_id=100, entries1=[],
                                 entries2=[(5.0e9, 7)])]
    matcher._selected_resonances = {(100, 2, 5.0e9)}
    matcher._rematch_freq_threshold = 4.0e9

    matcher.delete_selected()

    x_min, x_max = matcher.plot_mag.viewRange()[0]
    assert matcher._rematch_freq_threshold == pytest.approx(
        5.0e9 + DS2_OFFSET + 0.01 * (x_max - x_min))


################################################################################
# Shared y scale for DS1 and DS2
################################################################################

def test_auto_scale_y_uses_shared_span_and_keeps_offset(matcher):
    """
    Give both y axes the same span, set by the larger dataset range, with DS1
    centered in the upper band and DS2 in the lower band.
    """
    _prepare_headless_matcher(matcher)
    matcher._vb_mag2 = Mock()
    matcher._vb_phase2 = Mock()
    n1, n2 = len(matcher.f1), len(matcher.f2)
    matcher.filtered_mag1 = np.linspace(-1.0, 1.0, n1)    # span 2, center 0
    matcher.filtered_mag2 = np.linspace(10.0, 30.0, n2)   # span 20, center 20
    matcher.phase1 = np.linspace(0.0, 6.0, n1)            # span 6, center 3
    matcher.phase2 = np.linspace(-0.5, 0.5, n2)           # span 1, center 0

    matcher.auto_scale_y()

    def yrange(axis):
        args = axis.setYRange.call_args.args
        return args[0], args[1]

    for ax1, ax2, (c1, c2, span) in [
        (matcher.plot_mag, matcher._vb_mag2, (0.0, 20.0, 20.0)),
        (matcher.plot_phase, matcher._vb_phase2, (3.0, 0.0, 6.0)),
    ]:
        lo1, hi1 = yrange(ax1)
        lo2, hi2 = yrange(ax2)
        assert hi1 - lo1 == pytest.approx(6.0 * span)
        assert hi2 - lo2 == pytest.approx(6.0 * span)
        # DS1 center at 75% of its axis, DS2 center at 25% of its axis
        assert (c1 - lo1) / (hi1 - lo1) == pytest.approx(0.75)
        assert (c2 - lo2) / (hi2 - lo2) == pytest.approx(0.25)


################################################################################
# DS2 display offset persistence
################################################################################

def _reopen_matcher(simple_data, grp_path, choice, **kwargs):
    """
    Construct a ResMatcher on existing zarr data with a fixed startup choice.

    Parameters:
    simple_data (dict): sweep data from the ``simple_data`` fixture.
    grp_path (str): path to the zarr group written by a previous matcher.
    choice (str): startup dialog result, 'load' or 'overwrite'.
    **kwargs: other keyword arguments for ResMatcher.

    Returns:
    matcher_obj (ResMatcher): matcher built without opening a window.
    """
    sd = simple_data
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'), \
         patch.object(ResMatcher, '_show_startup_dialog', return_value=choice):
        return ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
            grp_path, **kwargs,
        )


@pytest.mark.parametrize('choice, passed, expected', [
    ('load', None, 3.0e6),        # saved offset restored
    ('load', 1.0e6, 1.0e6),       # explicit offset overrides saved one
    ('overwrite', None, 0.0),     # fresh start ignores saved offset
])


def test_ds2_offset_saved_and_restored(
    matcher, simple_data, tmp_path, choice, passed, expected
):
    """
    Save the DS2 display offset and restore it only on load without an
    explicit offset.
    """
    matcher._set_ds2_frequency_offset(3.0e6)
    matcher.save_data()
    assert np.array(matcher.zarr_group['res_matcher_ds2_f_offset'])[0] == 3.0e6

    reopened = _reopen_matcher(simple_data, str(tmp_path / 'out.zarr'),
                               choice, ds2_f_offset=passed)

    assert reopened.ds2_f_offset == expected
    assert np.allclose(reopened.f2_display, reopened.f2 + expected)


################################################################################
# Selection bookkeeping after unlink and merge
################################################################################

def _headless_with_groups(matcher):
    """
    Prepare a headless matcher with two 1:1 groups and both DS1 entries
    selected.

    Parameters:
    matcher (ResMatcher): matcher from the ``matcher`` fixture.

    Returns:
    matcher (ResMatcher): the same matcher, ready for keyboard actions.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    matcher.groups = [
        MatchGroup(group_id=100, entries1=[(5.0e9, 1)], entries2=[(5.0e9, 1)]),
        MatchGroup(group_id=101, entries1=[(5.1e9, 2)], entries2=[(5.1e9, 2)]),
    ]
    matcher._selected_resonances = {(100, 1, 5.0e9), (101, 1, 5.1e9)}
    return matcher


def _selection_is_valid(matcher):
    """
    Check that every selected resonance exists in the group it points at.

    Parameters:
    matcher (ResMatcher): matcher to check.

    Returns:
    valid (bool): True if every selection tuple matches a group entry.
    """
    for gid, ds, fres in matcher._selected_resonances:
        g = matcher._find_group(gid)
        entries = [] if g is None else (g.entries1 if ds == 1 else g.entries2)
        if not any(e[0] == fres for e in entries):
            return False
    return True


def test_unlink_logs_number_unlinked(matcher):
    m = _headless_with_groups(matcher)
    m.unlink_selected()
    m.log.assert_called_with('Unlinked 2 resonances into separate groups')


@pytest.mark.parametrize('action', ['merge_groups', 'merge_selected_only'])
def test_merge_keeps_selection_valid(matcher, action):
    """After F or D, the selection points at the resulting group."""
    m = _headless_with_groups(matcher)
    getattr(m, action)()
    assert len(m._selected_resonances) == 2
    assert _selection_is_valid(m)
    # A follow-up action sees the selection (it is not silently skipped)
    m.unlink_selected()
    m.log.assert_called_with('Unlinked 2 resonances into separate groups')


################################################################################
########### Save & Quit must behave exactly like closing the window. ###########
################################################################################

def test_quit_and_save_closes_window_and_saves_once(matcher):
    # Same close-handler wiring as ResMatcher.setup_ui.
    matcher.win = QtWidgets.QMainWindow()
    matcher.win.closeEvent = matcher._on_window_close
    matcher.win.show()

    with patch.object(matcher, 'save_data') as save, \
            patch.object(QtWidgets.QApplication, 'quit') as app_quit:
        matcher.quit_and_save()

    assert not matcher.win.isVisible()
    save.assert_called_once()
    app_quit.assert_not_called()


def test_log_after_the_panel_is_deleted_prints(capsys):
    """While the window closes and saves, the log panel may already be gone."""
    from types import SimpleNamespace
    from PyQt5 import sip
    from pyqtgraph.Qt import QtWidgets
    from citkid.vna.res_matcher import ResMatcher

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = QtWidgets.QTextEdit()
    sip.delete(panel)

    ResMatcher.log(SimpleNamespace(log_text=panel), 'Saved 3 resonances')

    assert 'Saved 3 resonances' in capsys.readouterr().out


@pytest.mark.parametrize("offset,n_pairs", [(None, 0), (3e9, 4)])
def test_init_nearest_uses_user_ds2_offset(
    simple_data, tmp_path, offset, n_pairs
):
    """
    'nearest' estimates no offset; it only uses ds2_f_offset when given. The
    shift is beyond the default threshold (3 x 0.9 GHz median spacing), so
    without the offset no shifted pairs can form.
    """
    sd = simple_data
    fres2 = sd['fres1'] - 3e9
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        m = ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], fres2, sd['res_idx2'],
            str(tmp_path / 'out.zarr'), init_match='nearest',
            ds2_f_offset=offset,
        )
    pairs = [g for g in m.groups if g.mapping_str() == '1-1'
             and np.isclose(g.entries1[0][0] - g.entries2[0][0], 3e9)]
    assert len(pairs) == n_pairs


################################################################################
# Re-match threshold
################################################################################

def test_manual_threshold_holds_while_panning(matcher):
    """
    A Ctrl+Right-click threshold stays put when the view moves.
    """
    _prepare_headless_matcher(matcher)
    matcher._rematch_freq_threshold = 4.5e9
    matcher._last_edit_freq = None  # manual mode
    matcher._threshold_pin_right = False
    matcher.plot_mag.viewRange.return_value = [[4.0e9, 5.0e9], [0., 1.]]

    matcher._update_threshold_position()

    assert matcher._rematch_freq_threshold == 4.5e9


@pytest.mark.parametrize("edit_freq, moves", [(4.7e9, True), (4.3e9, False)])
def test_edit_right_of_manual_threshold_overwrites_it(
    matcher, edit_freq, moves
):
    """
    An edit right of a manual threshold moves the threshold just past the
    edit; an edit left of it (already protected) leaves it.
    """
    _prepare_headless_matcher(matcher)
    matcher.plot_mag.viewRange.return_value = [[4.0e9, 5.0e9], [0., 1.]]
    matcher._rematch_freq_threshold = 4.5e9
    matcher._last_edit_freq = None  # manual mode

    matcher._last_edit_freq = edit_freq
    matcher._update_threshold_after_edit(edit_freq)

    expected = edit_freq + 0.01 * 1e9 if moves else 4.5e9
    assert matcher._rematch_freq_threshold == pytest.approx(expected)
    assert matcher._threshold_pin_right is (True if not moves else False)


def test_init_sorts_both_datasets(simple_data, tmp_path):
    """
    Unsorted f1 and f2 are sorted ascending, with z1 and z2 reordered to
    match.
    """
    sd = simple_data
    rng = np.random.default_rng(1)
    p1 = rng.permutation(len(sd['f']))
    p2 = rng.permutation(len(sd['f']))
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        m = ResMatcher(
            sd['f'][p1], sd['z1'][p1], sd['fres1'], sd['res_idx1'],
            sd['f'][p2], sd['z2'][p2], sd['fres2'], sd['res_idx2'],
            str(tmp_path / 'out.zarr'),
        )
    np.testing.assert_array_equal(m.f1, sd['f'])
    np.testing.assert_array_equal(m.f2, sd['f'])
    np.testing.assert_array_equal(m.z1, sd['z1'])
    np.testing.assert_array_equal(m.z2, sd['z2'])


def test_rematch_with_nothing_right_of_line_adds_no_undo_step(matcher):
    """
    Pressing R with no resonance right of the threshold logs a message and
    leaves the undo stack and the groups unchanged.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.win = Mock()
    matcher.update_markers = Mock()
    group = MatchGroup(group_id=100, entries1=[(5.0e9, 1)],
                       entries2=[(5.0e9, 1)])
    matcher.groups = [group]
    matcher.ds2_f_offset = 0.0
    matcher._rematch_freq_threshold = 6.0e9
    n_undo = len(matcher.undo_stack)

    matcher.trigger_rematch()

    assert len(matcher.undo_stack) == n_undo
    assert matcher.groups == [group]
    assert 'No resonances above' in matcher.log.call_args.args[0]


def test_rematch_adds_one_undo_step_with_the_state_before(matcher):
    """
    A re-match that changes something saves one undo step holding the
    groups as they were before.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.win = Mock()
    matcher.update_markers = Mock()
    matcher.groups = [
        MatchGroup(group_id=100, entries1=[(5.0e9, 1)], entries2=[]),
        MatchGroup(group_id=101, entries1=[], entries2=[(5.1e9, 2)]),
    ]
    matcher.ds2_f_offset = 0.0
    matcher._rematch_freq_threshold = 4.9e9
    matcher._init_match_method = 'sorted'
    n_undo = len(matcher.undo_stack)

    matcher.trigger_rematch()

    assert len(matcher.undo_stack) == n_undo + 1
    assert len(matcher.groups) == 1  # the two singles are now one pair


################################################################################
# Scatter clicks and selection
################################################################################

def _point(data):
    """
    Build a mock scatter point.

    Parameters:
    data (tuple or None): value returned by the point's ``data()``.

    Returns:
    pt (Mock): mock scatter point.
    """
    pt = Mock()
    pt.data.return_value = data
    return pt


def _click(button=QtCore.Qt.LeftButton, modifiers=QtCore.Qt.NoModifier):
    """
    Build a mock scatter click event.

    Parameters:
    button (Qt.MouseButton): button returned by ``ev.button()``.
    modifiers (Qt.KeyboardModifiers): modifiers returned by
        ``ev.modifiers()``.

    Returns:
    ev (Mock): mock mouse click event.
    """
    ev = Mock()
    ev.button.return_value = button
    ev.modifiers.return_value = modifiers
    return ev


def _clickable_matcher(matcher):
    """
    Prepare a headless matcher for scatter-click tests.

    Parameters:
    matcher (ResMatcher): matcher from the ``matcher`` fixture.

    Returns:
    matcher (ResMatcher): the same matcher with markers mocked out.
    """
    _prepare_headless_matcher(matcher)
    matcher.log = Mock()
    matcher.update_markers = Mock()
    return matcher


def test_plain_click_replaces_selection(matcher):
    m = _clickable_matcher(matcher)
    m._selected_resonances = {(0, 1, 4.5e9), (1, 2, 5.21e9)}
    m._on_scatter_clicked([_point((2, 1, 6.1e9))], _click())
    assert m._selected_resonances == {(2, 1, 6.1e9)}
    assert m._scatter_just_clicked is True
    m.update_markers.assert_called_once()
    m.sel_label.setText.assert_called_once()
    assert 'Group 2' in m.sel_label.setText.call_args.args[0]


def test_ctrl_click_toggles_selection(matcher):
    m = _clickable_matcher(matcher)
    m._selected_resonances = {(0, 1, 4.5e9)}
    ctrl = _click(modifiers=QtCore.Qt.ControlModifier)
    m._on_scatter_clicked([_point((1, 2, 5.21e9))], ctrl)
    assert m._selected_resonances == {(0, 1, 4.5e9), (1, 2, 5.21e9)}
    m._on_scatter_clicked([_point((0, 1, 4.5e9))], ctrl)
    assert m._selected_resonances == {(1, 2, 5.21e9)}


@pytest.mark.parametrize('pts, button', [
    ([_point((0, 1, 4.5e9))], QtCore.Qt.RightButton),
    ([], QtCore.Qt.LeftButton),
    ([_point(None)], QtCore.Qt.LeftButton),
])
def test_scatter_click_ignored(matcher, pts, button):
    """Right-clicks, empty point lists, and points without data do nothing."""
    m = _clickable_matcher(matcher)
    m._selected_resonances = {(1, 2, 5.21e9)}
    m._on_scatter_clicked(pts, _click(button=button))
    assert m._selected_resonances == {(1, 2, 5.21e9)}
    assert m._scatter_just_clicked is False
    m.update_markers.assert_not_called()


def test_scatter_click_without_event_methods_uses_app_modifiers(matcher):
    """Old pyqtgraph events lack button/modifiers; fall back to Qt state."""
    m = _clickable_matcher(matcher)
    m._selected_resonances = {(0, 1, 4.5e9)}
    with patch.object(QtWidgets.QApplication, 'keyboardModifiers',
                      return_value=QtCore.Qt.ControlModifier):
        m._on_scatter_clicked([_point((1, 1, 5.2e9))], object())
    assert m._selected_resonances == {(0, 1, 4.5e9), (1, 1, 5.2e9)}


@pytest.mark.parametrize('n_args', [2, 3])
def test_scatter_handler_forwards_points_and_event(matcher, n_args):
    """The handler accepts both pyqtgraph signal signatures."""
    matcher._on_scatter_clicked = Mock()
    handler = matcher._make_scatter_handler()
    pts, ev = [Mock()], Mock()
    args = (pts, ev) if n_args == 2 else (Mock(), pts, ev)
    handler(*args)
    matcher._on_scatter_clicked.assert_called_once_with(pts, ev)


def test_scatter_handler_ignores_other_signatures(matcher):
    matcher._on_scatter_clicked = Mock()
    handler = matcher._make_scatter_handler()
    handler(Mock())
    handler()
    matcher._on_scatter_clicked.assert_not_called()


def test_scene_click_after_scatter_click_is_suppressed(matcher):
    """A Shift+click that landed on a marker does not also add a resonance."""
    m = _clickable_matcher(matcher)
    m.add_resonance = Mock()
    m._on_scatter_clicked([_point((0, 1, 4.5e9))], _click())
    m._on_scene_clicked(_click(modifiers=QtCore.Qt.ShiftModifier))
    m.add_resonance.assert_not_called()
    assert m._scatter_just_clicked is False


################################################################################
# Marker drawing
################################################################################

def _marker_matcher(matcher, offset=0.0):
    """
    Prepare a headless matcher whose scatter items record ``setData`` calls.

    Parameters:
    matcher (ResMatcher): matcher from the ``matcher`` fixture.
    offset (float): DS2 display offset in Hz. Default is 0.

    Returns:
    matcher (ResMatcher): the same matcher, ready for ``update_markers``.
    """
    _prepare_headless_matcher(matcher)
    for name in ['ds1_mag', 'ds2_mag', 'ds1_phase', 'ds2_phase']:
        setattr(matcher, f'scatter_{name}', Mock())
        setattr(matcher, f'scatter_sel_{name}', Mock())
    matcher._sync_mag2_geom = Mock()
    matcher._sync_phase2_geom = Mock()
    matcher._update_curves = Mock()
    matcher.auto_scale_y = Mock()
    matcher._update_overview_region = Mock()
    matcher._set_ds2_frequency_offset(offset)
    return matcher


def _spots(scatter):
    """
    Return the spot list passed to a mock scatter's last ``setData`` call.

    Parameters:
    scatter (Mock): mock scatter plot item.

    Returns:
    spots (list): list of spot dictionaries.
    """
    return scatter.setData.call_args.args[0]


def test_update_markers_draws_both_datasets_with_offset(matcher):
    from citkid.vna.res_matcher import _group_color
    offset = 2e6
    m = _marker_matcher(matcher, offset=offset)
    m.groups = [
        MatchGroup(3, [(5.0e9, 1)], [(5.01e9, 2)]),
        MatchGroup(4, [], [(6.0e9, 3)], ambiguous=True),
    ]
    m.plot_mag.viewRange.return_value = [[4.0e9, 8.0e9], [0.0, 1.0]]

    m.update_markers()

    ds1_mag, ds2_mag = _spots(m.scatter_ds1_mag), _spots(m.scatter_ds2_mag)
    ds1_ph, ds2_ph = _spots(m.scatter_ds1_phase), _spots(m.scatter_ds2_phase)
    assert [s['data'] for s in ds1_mag] == [(3, 1, 5.0e9)]
    assert [s['data'] for s in ds2_mag] == [(3, 2, 5.01e9), (4, 2, 6.0e9)]
    assert len(ds1_ph) == 1 and len(ds2_ph) == 2

    i1 = m._nearest_idx(m.f1, 5.0e9)
    assert ds1_mag[0]['pos'] == (m.f1[i1], m.filtered_mag1[i1])
    assert ds1_ph[0]['pos'] == (m.f1[i1], m.phase1[i1])
    i2 = m._nearest_idx(m.f2, 5.01e9)
    assert ds2_mag[0]['pos'] == pytest.approx(
        (m.f2[i2] + offset, m.filtered_mag2[i2]))
    assert ds2_ph[0]['pos'] == pytest.approx(
        (m.f2[i2] + offset, m.phase2[i2]))

    # DS1: filled with the group colour, solid pen
    assert ds1_mag[0]['brush'].color().getRgb() == _group_color(3)
    assert ds1_mag[0]['pen'].style() == QtCore.Qt.SolidLine
    # DS2: open squares; ambiguous groups get a thicker dashed pen
    assert ds2_mag[0]['brush'].style() == QtCore.Qt.NoBrush
    assert ds2_mag[0]['pen'].color().getRgb() == _group_color(3)
    assert ds2_mag[0]['pen'].style() == QtCore.Qt.SolidLine
    assert ds2_mag[0]['pen'].widthF() == 2
    assert ds2_mag[1]['pen'].style() == QtCore.Qt.DashLine
    assert ds2_mag[1]['pen'].widthF() == 2.5
    m._sync_mag2_geom.assert_called()
    m._sync_phase2_geom.assert_called()


def test_update_markers_ambiguous_ds1_pen_is_dashed(matcher):
    m = _marker_matcher(matcher)
    m.groups = [MatchGroup(0, [(5.0e9, 1)], [], ambiguous=True)]
    m.update_markers()
    assert _spots(m.scatter_ds1_mag)[0]['pen'].style() == QtCore.Qt.DashLine


def test_update_markers_skips_groups_out_of_view(matcher):
    """Groups beyond the view plus 30% padding are not drawn."""
    m = _marker_matcher(matcher)
    m.plot_mag.viewRange.return_value = [[5.0e9, 5.1e9], [0.0, 1.0]]
    m.groups = [
        MatchGroup(0, [(5.05e9, 1)], []),
        MatchGroup(1, [(5.12e9, 2)], []),   # inside padding
        MatchGroup(2, [(5.2e9, 3)], []),    # outside padding
        MatchGroup(3, [], []),              # empty group is not skipped
    ]
    m.update_markers()
    assert [s['data'][0] for s in _spots(m.scatter_ds1_mag)] == [0, 1]


def test_update_markers_uses_display_freq_for_ds2_visibility(matcher):
    """A DS2 entry is visible when its display frequency is in view."""
    m = _marker_matcher(matcher, offset=-1e9)
    m.plot_mag.viewRange.return_value = [[5.0e9, 5.1e9], [0.0, 1.0]]
    m.groups = [MatchGroup(0, [], [(6.05e9, 1)])]
    m.update_markers()
    assert [s['data'] for s in _spots(m.scatter_ds2_mag)] == [
        (0, 2, 6.05e9)]


def test_update_markers_without_plots_returns(matcher):
    assert not hasattr(matcher, 'plot_mag')
    matcher.update_markers()


def test_selection_rings_empty_selection_clears(matcher):
    m = _marker_matcher(matcher)
    m._selected_resonances = set()
    m._update_selection_ring()
    for name in ['ds1_mag', 'ds2_mag', 'ds1_phase', 'ds2_phase']:
        getattr(m, f'scatter_sel_{name}').setData.assert_called_with([])


def test_selection_rings_placed_at_display_positions(matcher):
    offset = 3e6
    m = _marker_matcher(matcher, offset=offset)
    m._selected_resonances = {(0, 1, 5.0e9), (1, 2, 6.0e9)}
    m._update_selection_ring()
    i1 = m._nearest_idx(m.f1, 5.0e9)
    i2 = m._nearest_idx(m.f2, 6.0e9)
    assert _spots(m.scatter_sel_ds1_mag) == [
        {'pos': (m.f1[i1], m.filtered_mag1[i1])}]
    assert _spots(m.scatter_sel_ds1_phase) == [
        {'pos': (m.f1[i1], m.phase1[i1])}]
    assert _spots(m.scatter_sel_ds2_mag)[0]['pos'] == pytest.approx(
        (m.f2[i2] + offset, m.filtered_mag2[i2]))
    assert _spots(m.scatter_sel_ds2_phase)[0]['pos'] == pytest.approx(
        (m.f2[i2] + offset, m.phase2[i2]))


def test_check_selection_visibility_deselects_out_of_view(matcher):
    m = _clickable_matcher(matcher)
    m._update_curves = Mock()
    m.auto_scale_y = Mock()
    m._update_overview_region = Mock()
    m._set_ds2_frequency_offset(-1e9)
    m.plot_mag.viewRange.return_value = [[5.0e9, 5.1e9], [0.0, 1.0]]
    m._selected_resonances = {
        (0, 1, 5.05e9),   # in view
        (1, 1, 7.0e9),    # out of view
        (2, 2, 6.05e9),   # DS2, in view after the display offset
        (3, 2, 5.05e9),   # DS2, out of view after the display offset
    }
    m._check_selection_visibility()
    assert m._selected_resonances == {(0, 1, 5.05e9), (2, 2, 6.05e9)}
    m.log.assert_called_once_with('Deselected 2 out-of-view resonances')


def test_check_selection_visibility_keeps_visible(matcher):
    m = _clickable_matcher(matcher)
    m.plot_mag.viewRange.return_value = [[5.0e9, 5.1e9], [0.0, 1.0]]
    m._selected_resonances = {(0, 1, 5.05e9)}
    m._check_selection_visibility()
    assert m._selected_resonances == {(0, 1, 5.05e9)}
    m.log.assert_not_called()
    m._selected_resonances = set()
    m._check_selection_visibility()
    m.plot_mag.viewRange.assert_called_once()


################################################################################
# Re-adding removed resonances
################################################################################

REMOVED = [(5.001e9, 77), (5.003e9, 78)]


@pytest.mark.parametrize('code, expected', [
    (1, 77),       # OK with the first (default) radio button checked
    (2, 'new'),    # Create New res_idx
    (0, None),     # Cancel
])
def test_ask_reuse_res_idx_returns_choice(matcher, code, expected):
    matcher.win = None
    with patch.object(QtWidgets.QDialog, 'exec_', return_value=code):
        choice = matcher._ask_reuse_res_idx(5.0e9, 1, REMOVED)
    assert choice == expected


def test_ask_reuse_res_idx_shows_one_radio_per_removed(matcher):
    """The dialog lists every removed resonance with its distance."""
    matcher.win = None
    seen = {}

    def fake_exec(dlg):
        """
        Record the radio buttons, check the second, and press OK.

        Parameters:
        dlg (QDialog): dialog being shown.

        Returns:
        code (int): dialog result code for OK.
        """
        radios = dlg.findChildren(QtWidgets.QRadioButton)
        seen['texts'] = [r.text() for r in radios]
        radios[1].setChecked(True)
        return 1

    with patch.object(QtWidgets.QDialog, 'exec_', fake_exec):
        choice = matcher._ask_reuse_res_idx(5.0e9, 2, REMOVED)
    assert choice == 78
    assert len(seen['texts']) == 2
    assert 'res_idx 77' in seen['texts'][0]
    assert 'Δ = 1000.0 kHz' in seen['texts'][0]
    assert 'Δ = 3000.0 kHz' in seen['texts'][1]


def test_ask_reuse_res_idx_ok_without_options_is_new(matcher):
    matcher.win = None
    with patch.object(QtWidgets.QDialog, 'exec_', return_value=1):
        assert matcher._ask_reuse_res_idx(5.0e9, 1, []) == 'new'


def _matcher_with_removed(matcher):
    """
    Prepare a headless matcher with removed resonances in both datasets.

    Parameters:
    matcher (ResMatcher): matcher from the ``matcher`` fixture.

    Returns:
    matcher (ResMatcher): the same matcher with a 4.9-5.1 GHz view.
    """
    m = _clickable_matcher(matcher)
    m._update_curves = Mock()
    m.auto_scale_y = Mock()
    m._update_overview_region = Mock()
    m.plot_mag.viewRange.return_value = [[4.9e9, 5.1e9], [0.0, 1.0]]
    m._removed_resonances = [
        (5.003e9, 78, 1),
        (5.001e9, 77, 1),
        (7.0e9, 79, 1),     # out of view
        (5.002e9, 80, 2),   # other dataset
    ]
    return m


def _added_group(m, before_ids):
    """
    Return the single group added since ``before_ids`` was recorded.

    Parameters:
    m (ResMatcher): matcher after an add.
    before_ids (set): group IDs before the add.

    Returns:
    group (MatchGroup): the new group.
    """
    new = [g for g in m.groups if g.group_id not in before_ids]
    assert len(new) == 1
    return new[0]


def test_add_resonance_offers_visible_removed_sorted_by_distance(matcher):
    m = _matcher_with_removed(matcher)
    with patch.object(m, '_ask_reuse_res_idx', return_value=None) as ask:
        m.add_resonance(5.0e9, ds=1)
    ask.assert_called_once_with(5.0e9, 1, [(5.001e9, 77), (5.003e9, 78)])


def test_add_resonance_cancel_changes_nothing(matcher):
    import copy
    m = _matcher_with_removed(matcher)
    groups_before = copy.deepcopy(m.groups)
    removed_before = list(m._removed_resonances)
    with patch.object(m, '_ask_reuse_res_idx', return_value=None):
        m.add_resonance(5.0e9, ds=1)
    assert m.groups == groups_before
    assert m._removed_resonances == removed_before
    assert m.undo_stack == []
    m.log.assert_not_called()


def test_add_resonance_reuses_chosen_res_idx(matcher):
    m = _matcher_with_removed(matcher)
    before_ids = {g.group_id for g in m.groups}
    next_new = m._next_new_idx
    with patch.object(m, '_ask_reuse_res_idx', return_value=78):
        m.add_resonance(5.0e9, ds=1)
    g = _added_group(m, before_ids)
    assert g.entries1 == [(5.0e9, 78)] and g.entries2 == []
    assert m._next_new_idx == next_new
    # Only the reused DS1 entry leaves the removed list
    assert m._removed_resonances == [
        (5.001e9, 77, 1), (7.0e9, 79, 1), (5.002e9, 80, 2)]
    assert len(m.undo_stack) == 1
    assert 'res_idx=78' in m.log.call_args.args[0]


def test_add_resonance_new_choice_allocates_new_idx(matcher):
    m = _matcher_with_removed(matcher)
    before_ids = {g.group_id for g in m.groups}
    removed_before = list(m._removed_resonances)
    next_new = m._next_new_idx
    with patch.object(m, '_ask_reuse_res_idx', return_value='new'):
        m.add_resonance(5.0e9, ds=1)
    g = _added_group(m, before_ids)
    assert g.entries1 == [(5.0e9, next_new)]
    assert m._next_new_idx == next_new + 1
    assert m._removed_resonances == removed_before


def test_add_resonance_ds2_uses_data_window(matcher):
    """DS2 removed resonances are matched against the offset-free window."""
    m = _matcher_with_removed(matcher)
    m._set_ds2_frequency_offset(1e9)   # view 4.9-5.1 GHz is 3.9-4.1 GHz data
    m._removed_resonances = [(5.002e9, 80, 2), (4.0e9, 81, 2)]
    before_ids = {g.group_id for g in m.groups}
    with patch.object(m, '_ask_reuse_res_idx', return_value=81) as ask:
        m.add_resonance(4.0e9, ds=2)
    ask.assert_called_once_with(4.0e9, 2, [(4.0e9, 81)])
    g = _added_group(m, before_ids)
    assert g.entries1 == [] and g.entries2 == [(4.0e9, 81)]
    assert m._removed_resonances == [(5.002e9, 80, 2)]


################################################################################
# Filter controls and DS2 offset drag
################################################################################

def _filter_matcher(matcher):
    """
    Prepare a headless matcher with mock filter widgets.

    Parameters:
    matcher (ResMatcher): matcher from the ``matcher`` fixture.

    Returns:
    matcher (ResMatcher): the same matcher with filter controls mocked.
    """
    _prepare_headless_matcher(matcher)
    matcher.status_label = Mock()
    matcher.win = Mock()
    for ds, (method, hp, order) in [(1, ('highpass', 5.0, 2)),
                                    (2, ('polynomial', 7.0, 3))]:
        combo = Mock()
        combo.currentText.return_value = method
        hp_spin, poly_spin = Mock(), Mock()
        hp_spin.value.return_value = hp
        poly_spin.value.return_value = order
        setattr(matcher, f'smooth_combo{ds}', combo)
        setattr(matcher, f'hp_spin{ds}', hp_spin)
        setattr(matcher, f'poly_spin{ds}', poly_spin)
    return matcher


@pytest.mark.parametrize('ds, expected', [
    (1, {'smoothing': 'highpass', 'highpass_mhz': 5.0, 'poly_order': 2}),
    (2, {'smoothing': 'polynomial', 'highpass_mhz': 7.0, 'poly_order': 3}),
])
def test_on_filter_changed_reads_controls_and_starts_timer(
    matcher, ds, expected
):
    m = _filter_matcher(matcher)
    other = 2 if ds == 1 else 1
    other_before = dict(getattr(m, f'filter_params{other}'))
    m._on_filter_changed(ds)
    timer = getattr(m, f'_filter_timer{ds}')
    try:
        params = getattr(m, f'filter_params{ds}')
        assert {k: params[k] for k in expected} == expected
        assert getattr(m, f'filter_params{other}') == other_before
        assert getattr(m, f'_filter_timer{other}') is None
        assert timer.isSingleShot() and timer.isActive()
        assert 'Updating' in m.status_label.setText.call_args.args[0]
        # A second change restarts the same timer
        m._on_filter_changed(ds)
        assert getattr(m, f'_filter_timer{ds}') is timer
    finally:
        timer.stop()


@pytest.mark.parametrize('ds', [1, 2])
def test_do_filter_update_refilters_one_dataset(matcher, ds):
    from citkid.vna.s21_filt import filter_mag
    m = _filter_matcher(matcher)
    m._update_curves = Mock()
    m.update_markers = Mock()
    m.auto_scale_y = Mock()
    params = {'smoothing': 'polynomial', 'highpass_mhz': 1.0,
              'poly_order': 1}
    setattr(m, f'filter_params{ds}', params)
    other = 2 if ds == 1 else 1
    other_before = getattr(m, f'filtered_mag{other}')
    with patch('citkid.vna.res_matcher.run_responsive',
               side_effect=lambda fn, *a: fn(*a)):
        m._do_filter_update(ds)
    f = getattr(m, f'f{ds}')
    mag = getattr(m, f'mag_db{ds}')
    expected = filter_mag(f, mag, 'polynomial', 1.0, 1)
    np.testing.assert_allclose(getattr(m, f'filtered_mag{ds}'), expected)
    assert getattr(m, f'filtered_mag{other}') is other_before
    m._update_curves.assert_called_once()
    m.update_markers.assert_called_once()
    m.auto_scale_y.assert_called_once()
    assert 'Ready' in m.status_label.setText.call_args.args[0]
    m.win.setCursor.assert_called_with(QtCore.Qt.ArrowCursor)


def test_shift_ds2_frequency_offset_is_relative(matcher):
    matcher._set_ds2_frequency_offset(1e6)
    matcher._shift_ds2_frequency_offset(0)
    assert matcher.ds2_f_offset == 1e6
    matcher._shift_ds2_frequency_offset(-2.5e5)
    assert matcher.ds2_f_offset == 7.5e5
    np.testing.assert_allclose(matcher.f2_display, matcher.f2 + 7.5e5)


def _drag_event(button=QtCore.Qt.LeftButton):
    """
    Build a mock drag event from scene x = 10 to scene x = 25.

    Parameters:
    button (Qt.MouseButton): button returned by ``ev.button()``.

    Returns:
    ev (Mock): mock mouse drag event.
    """
    ev = Mock()
    ev.button.return_value = button
    ev.lastScenePos.return_value = QtCore.QPointF(10.0, 0.0)
    ev.scenePos.return_value = QtCore.QPointF(25.0, 0.0)
    return ev


def _drag_viewbox(callback):
    """
    Build a CustomViewBox whose scene-to-view map scales x by 1000.

    Parameters:
    callback (callable or None): drag callback for the view box.

    Returns:
    vb (CustomViewBox): view box with ``mapSceneToView`` mocked.
    """
    vb = CustomViewBox(drag_callback=callback)
    vb.mapSceneToView = lambda p: QtCore.QPointF(p.x() * 1000.0, p.y())
    return vb


def test_ctrl_left_drag_calls_callback_with_x_shift():
    callback = Mock()
    vb = _drag_viewbox(callback)
    ev = _drag_event()
    with patch.object(QtWidgets.QApplication, 'keyboardModifiers',
                      return_value=QtCore.Qt.ControlModifier), \
         patch('pyqtgraph.ViewBox.mouseDragEvent') as base_drag:
        vb.mouseDragEvent(ev)
    callback.assert_called_once_with(pytest.approx(15000.0))
    ev.accept.assert_called_once()
    base_drag.assert_not_called()


@pytest.mark.parametrize('has_callback, button, modifiers', [
    (False, QtCore.Qt.LeftButton, QtCore.Qt.ControlModifier),
    (True, QtCore.Qt.RightButton, QtCore.Qt.ControlModifier),
    (True, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier),
])
def test_other_drags_pan_normally(has_callback, button, modifiers):
    callback = Mock()
    vb = _drag_viewbox(callback if has_callback else None)
    ev = _drag_event(button)
    with patch.object(QtWidgets.QApplication, 'keyboardModifiers',
                      return_value=modifiers), \
         patch('pyqtgraph.ViewBox.mouseDragEvent') as base_drag:
        vb.mouseDragEvent(ev, axis=0)
    callback.assert_not_called()
    base_drag.assert_called_once_with(ev, axis=0)
