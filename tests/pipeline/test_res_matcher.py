"""
Tests for the interactive resonance matcher (pipeline/interactive/res_matcher.py).

Tests cover:
- MatchGroup dataclass
- _init_sorted and _init_nearest initialisation
- _visible_slice and _nearest_idx helpers
- add_resonance, _do_unlink, _do_merge
- merge_selected, unlink_selected, toggle_ambiguous
- undo
- save_data (zarr output)
- auto_scale_y (independent dual-axis scaling)
- run_res_matcher wrapper
- _update_overview_region / _on_overview_region_changed
"""

import os
import pytest
import numpy as np
import zarr
from unittest.mock import Mock, patch, MagicMock

# Offscreen Qt must be set before any pyqtgraph import.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from citkid.vna.res_matcher import (
    MatchGroup,
    ResMatcher,
    run_res_matcher,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Small synthetic VNA sweep for two datasets."""
    f = np.linspace(4e9, 8e9, 10_000)
    # Flat magnitude with tiny noise so phase detrending is stable
    rng = np.random.default_rng(0)
    z1 = np.exp(1j * (2 * np.pi * f / 1e9)) * (1 + 0.001 * rng.standard_normal(len(f)))
    z2 = np.exp(1j * (2 * np.pi * f / 1e9 + 0.1)) * (1 + 0.001 * rng.standard_normal(len(f)))
    fres1 = np.array([4.5e9, 5.2e9, 6.1e9, 7.3e9])
    fres2 = np.array([4.51e9, 5.21e9, 6.09e9, 7.28e9])
    res_idx1 = np.arange(len(fres1)) + 540
    res_idx2 = np.arange(len(fres2)) + 541
    return dict(f=f, z1=z1, z2=z2,
                fres1=fres1, fres2=fres2,
                res_idx1=res_idx1, res_idx2=res_idx2)


@pytest.fixture
def matcher(simple_data, tmp_path, qt_app):
    """ResMatcher with setup_ui patched so no window opens."""
    sd = simple_data
    grp_path = str(tmp_path / 'out.zarr')
    with patch.object(ResMatcher, 'setup_ui'), \
         patch.object(ResMatcher, 'run'):
        m = ResMatcher(
            sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
            sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
            grp_path,
            overwrite=True,
        )
    return m


# ---------------------------------------------------------------------------
# MatchGroup
# ---------------------------------------------------------------------------

class TestMatchGroup:
    def test_center_freq_1_1(self):
        g = MatchGroup(0, [(4.5e9, 0)], [(4.6e9, 1)])
        assert abs(g.center_freq() - 4.55e9) < 1e3

    def test_center_freq_single_entry(self):
        g = MatchGroup(0, [(5e9, 0)], [])
        assert g.center_freq() == 5e9

    def test_center_freq_empty(self):
        g = MatchGroup(0, [], [])
        assert g.center_freq() == 0.0

    def test_mapping_str_1_1(self):
        g = MatchGroup(0, [(4e9, 0)], [(4.1e9, 1)])
        assert g.mapping_str() == '1-1'

    def test_mapping_str_1_0(self):
        g = MatchGroup(0, [(4e9, 0)], [])
        assert g.mapping_str() == '1-0'

    def test_mapping_str_2_1(self):
        g = MatchGroup(0, [(4e9, 0), (5e9, 1)], [(4.5e9, 2)])
        assert g.mapping_str() == '2-1'

    def test_ambiguous_default_false(self):
        g = MatchGroup(0, [], [])
        assert g.ambiguous is False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitSorted:
    def test_equal_length_datasets(self, matcher):
        """sorted init pairs every DS1 res with a DS2 res (all 1-1)."""
        for g in matcher.groups:
            assert g.mapping_str() == '1-1'

    def test_group_count(self, matcher, simple_data):
        n = len(simple_data['fres1'])
        assert len(matcher.groups) == n

    def test_groups_sorted_by_center_freq(self, matcher):
        freqs = [g.center_freq() for g in matcher.groups]
        assert freqs == sorted(freqs)

    def test_init_sorted_extra_ds1(self, simple_data, tmp_path, qt_app):
        """Extra DS1 resonances become 1-0 groups."""
        sd = simple_data
        fres1 = np.append(sd['fres1'], 7.8e9)
        ridx1 = np.append(sd['res_idx1'], 999)
        grp = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            m = ResMatcher(
                sd['f'], sd['z1'], fres1, ridx1,
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp, overwrite=True,
            )
        mappings = [g.mapping_str() for g in m.groups]
        assert '1-0' in mappings

    def test_init_sorted_extra_ds2(self, simple_data, tmp_path, qt_app):
        """Extra DS2 resonances become 0-1 groups."""
        sd = simple_data
        fres2 = np.append(sd['fres2'], 7.9e9)
        ridx2 = np.append(sd['res_idx2'], 998)
        grp = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            m = ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], fres2, ridx2,
                grp, overwrite=True,
            )
        mappings = [g.mapping_str() for g in m.groups]
        assert '0-1' in mappings


class TestInitNearest:
    def test_equal_length_all_matched(self, simple_data, tmp_path, qt_app):
        """Nearest matching on close-offset data produces all 1-1 groups."""
        sd = simple_data
        grp = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            m = ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp, overwrite=True, init_match='nearest',
            )
        for g in m.groups:
            assert g.mapping_str() == '1-1'

    def test_invalid_init_match(self, simple_data, tmp_path, qt_app):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestVisibleSlice:
    def test_center_of_array(self, matcher):
        sl = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.0)
        assert matcher.f1[sl.start] >= 5e9
        assert matcher.f1[sl.stop - 1] <= 6e9

    def test_empty_slice_outside_range(self, matcher):
        sl = matcher._visible_slice(matcher.f1, 20e9, 21e9, pad=0.0)
        assert sl.start >= sl.stop

    def test_padding_extends_slice(self, matcher):
        sl_no_pad = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.0)
        sl_padded = matcher._visible_slice(matcher.f1, 5e9, 6e9, pad=0.5)
        assert sl_padded.stop - sl_padded.start > sl_no_pad.stop - sl_no_pad.start


class TestNearestIdx:
    def test_exact_hit(self, matcher):
        idx = matcher._nearest_idx(matcher.f1, matcher.f1[50])
        assert idx == 50

    def test_between_samples_picks_closer(self, matcher):
        f = matcher.f1
        mid = 0.6 * f[10] + 0.4 * f[11]   # closer to f[10]
        idx = matcher._nearest_idx(f, mid)
        assert idx == 10

    def test_below_range_returns_zero(self, matcher):
        assert matcher._nearest_idx(matcher.f1, 0.0) == 0

    def test_above_range_returns_last(self, matcher):
        last = len(matcher.f1) - 1
        assert matcher._nearest_idx(matcher.f1, 999e9) == last


# ---------------------------------------------------------------------------
# Resonance operations
# ---------------------------------------------------------------------------

class TestAddResonance:
    def test_add_creates_new_group(self, matcher):
        before = len(matcher.groups)
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher.add_resonance(7.9e9, ds=1)
        assert len(matcher.groups) == before + 1

    def test_add_ds1_entry_in_entries1(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher.add_resonance(7.9e9, ds=1)
        new_g = max(matcher.groups, key=lambda g: g.center_freq())
        assert len(new_g.entries1) == 1
        assert len(new_g.entries2) == 0

    def test_add_ds2_entry_in_entries2(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher.add_resonance(7.9e9, ds=2)
        new_g = max(matcher.groups, key=lambda g: g.center_freq())
        assert len(new_g.entries1) == 0
        assert len(new_g.entries2) == 1

    def test_add_pushes_undo_state(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        before = len(matcher.undo_stack)
        matcher.add_resonance(7.9e9, ds=1)
        assert len(matcher.undo_stack) == before + 1

    def test_add_keeps_groups_sorted(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher.add_resonance(4.1e9, ds=1)
        freqs = [g.center_freq() for g in matcher.groups]
        assert freqs == sorted(freqs)


class TestDoUnlink:
    def test_unlink_moves_entry_to_new_group(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        g = matcher.groups[0]
        entry = g.entries1[0]
        n_before = len(matcher.groups)
        matcher._do_unlink(g, entry, ds=1)
        assert len(matcher.groups) == n_before + 1
        assert entry not in g.entries1

    def test_unlink_ds2_entry(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        g = matcher.groups[0]
        entry = g.entries2[0]
        matcher._do_unlink(g, entry, ds=2)
        assert entry not in g.entries2

    def test_unlink_creates_new_group_with_entry(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        g = matcher.groups[1]
        entry = g.entries1[0]
        matcher._do_unlink(g, entry, ds=1)
        new_g = next(
            ng for ng in matcher.groups
            if ng.group_id != g.group_id and entry in ng.entries1
        )
        assert new_g is not None


class TestDoMerge:
    def test_merge_combines_entries(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        ga = matcher.groups[0]
        gb = matcher.groups[1]
        gid_a, gid_b = ga.group_id, gb.group_id
        n1_a = len(ga.entries1)
        n1_b = len(gb.entries1)
        matcher._do_merge(gid_a, gid_b)
        assert len(ga.entries1) == n1_a + n1_b

    def test_merge_removes_absorbed_group(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        ga = matcher.groups[0]
        gb = matcher.groups[1]
        gid_a, gid_b = ga.group_id, gb.group_id
        n_before = len(matcher.groups)
        matcher._do_merge(gid_a, gid_b)
        assert len(matcher.groups) == n_before - 1
        assert matcher._find_group(gid_b) is None

    def test_merge_propagates_ambiguous(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        ga = matcher.groups[0]
        gb = matcher.groups[1]
        gb.ambiguous = True
        matcher._do_merge(ga.group_id, gb.group_id)
        assert ga.ambiguous is True

    def test_merge_missing_group_noop(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        n_before = len(matcher.groups)
        matcher._do_merge(matcher.groups[0].group_id, 99999)
        assert len(matcher.groups) == n_before


# ---------------------------------------------------------------------------
# Keyboard actions
# ---------------------------------------------------------------------------

class TestToggleAmbiguous:
    def test_toggle_on(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        g = matcher.groups[0]
        matcher.sel_group_id = g.group_id
        assert g.ambiguous is False
        matcher.toggle_ambiguous()
        assert g.ambiguous is True

    def test_toggle_off(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        g = matcher.groups[0]
        g.ambiguous = True
        matcher.sel_group_id = g.group_id
        matcher.toggle_ambiguous()
        assert g.ambiguous is False

    def test_noop_when_nothing_selected(self, matcher):
        matcher.log = Mock()
        matcher.sel_group_id = None
        g = matcher.groups[0]
        g.ambiguous = False
        matcher.toggle_ambiguous()
        assert g.ambiguous is False


class TestMergeSelected:
    def test_merges_with_right_neighbor(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        sorted_gs = matcher._sorted_groups()
        # Select the first group
        matcher.sel_group_id = sorted_gs[0].group_id
        n_before = len(matcher.groups)
        matcher.merge_selected()
        assert len(matcher.groups) == n_before - 1

    def test_noop_when_nothing_selected(self, matcher):
        matcher.log = Mock()
        n_before = len(matcher.groups)
        matcher.sel_group_id = None
        matcher.merge_selected()
        assert len(matcher.groups) == n_before


class TestUnlinkSelected:
    def test_unlinks_selected_ds1(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        g = matcher.groups[0]
        fres = g.entries1[0][0]
        matcher.sel_group_id = g.group_id
        matcher.sel_dataset = 1
        matcher.sel_fres = fres
        n_before = len(matcher.groups)
        matcher.unlink_selected()
        assert len(matcher.groups) == n_before + 1

    def test_noop_when_nothing_selected(self, matcher):
        matcher.log = Mock()
        matcher.sel_group_id = None
        n_before = len(matcher.groups)
        matcher.unlink_selected()
        assert len(matcher.groups) == n_before


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

class TestUndo:
    def test_undo_restores_previous_state(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        n_before = len(matcher.groups)
        matcher.add_resonance(7.9e9, ds=1)
        assert len(matcher.groups) == n_before + 1
        matcher.undo()
        assert len(matcher.groups) == n_before

    def test_undo_empty_stack_noop(self, matcher):
        matcher.log = Mock()
        matcher.undo_stack.clear()
        n_before = len(matcher.groups)
        matcher.undo()
        assert len(matcher.groups) == n_before

    def test_multiple_undos(self, matcher):
        matcher.log = Mock()
        matcher.update_markers = Mock()
        matcher._clear_selection_if_gone = Mock()
        n_before = len(matcher.groups)
        for freq in [7.81e9, 7.82e9, 7.83e9]:
            matcher.add_resonance(freq, ds=1)
        for _ in range(3):
            matcher.undo()
        assert len(matcher.groups) == n_before


# ---------------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------------

class TestSaveData:
    def test_saves_all_keys(self, matcher, tmp_path):
        matcher.log = Mock()
        matcher.save_data()
        expected = [
            'fres1', 'res_idx1', 'group_ids1',
            'fres2', 'res_idx2', 'group_ids2',
            'ambiguous_groups',
        ]
        for key in expected:
            assert key in matcher.zarr_group

    def test_lengths_consistent(self, matcher):
        matcher.log = Mock()
        matcher.save_data()
        zg = matcher.zarr_group
        assert zg['fres1'].shape[0] == zg['res_idx1'].shape[0] == zg['group_ids1'].shape[0]
        assert zg['fres2'].shape[0] == zg['res_idx2'].shape[0] == zg['group_ids2'].shape[0]

    def test_group_ids_round_trip(self, matcher):
        """Saved group IDs are compact (0..N-1) and dense."""
        matcher.log = Mock()
        matcher.save_data()
        zg = matcher.zarr_group
        all_ids = set(int(i) for i in zg['group_ids1'][:]) | set(int(i) for i in zg['group_ids2'][:])
        # IDs should be dense in the range 0..(n_groups-1)
        assert max(all_ids) < len(matcher.groups)
        assert len(all_ids) == len(matcher.groups)

    def test_ambiguous_groups_recorded(self, matcher):
        matcher.log = Mock()
        matcher.groups[0].ambiguous = True
        orig_gid = matcher.groups[0].group_id
        matcher.save_data()
        zg = matcher.zarr_group
        ambig = list(zg['ambiguous_groups'][:])
        # compact_on_save maps group ids based on sorted groups order
        sorted_gs = matcher._sorted_groups()
        expected_compact = next(i for i, g in enumerate(sorted_gs) if g.group_id == orig_gid)
        assert expected_compact in list(ambig)

    def test_overwrite_raises_when_false(self, simple_data, tmp_path, qt_app):
        sd = simple_data
        grp_path = str(tmp_path / 'existing.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            m = ResMatcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp_path, overwrite=True,
            )
        m.log = Mock()
        m.save_data()   # write once
        with pytest.raises(FileExistsError):
            with patch.object(ResMatcher, 'setup_ui'), \
                 patch.object(ResMatcher, 'run'):
                ResMatcher(
                    sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                    sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                    grp_path, overwrite=False,
                )


# ---------------------------------------------------------------------------
# auto_scale_y (independent dual-axis logic)
# ---------------------------------------------------------------------------

class TestAutoScaleY:
    """Verify each ViewBox is scaled independently."""

    def _attach_plot_mocks(self, matcher):
        """Give the matcher mock plot objects so auto_scale_y can call them."""
        matcher.plot_mag = Mock()
        matcher.plot_mag.viewRange.return_value = [
            [float(matcher.f1[0]), float(matcher.f1[-1])], [0, 1]
        ]
        matcher._vb_mag2 = Mock()
        matcher.plot_phase = Mock()
        matcher._vb_phase2 = Mock()

    def test_ds1_mag_uses_plot_setYRange(self, matcher):
        self._attach_plot_mocks(matcher)
        matcher.auto_scale_y()
        matcher.plot_mag.setYRange.assert_called_once()

    def test_ds2_mag_uses_vb_setYRange(self, matcher):
        self._attach_plot_mocks(matcher)
        matcher.auto_scale_y()
        matcher._vb_mag2.setYRange.assert_called_once()

    def test_ds1_and_ds2_mag_ranges_differ(self, matcher):
        """DS1 is placed upper, DS2 lower — their axis ranges must differ."""
        self._attach_plot_mocks(matcher)
        matcher.auto_scale_y()
        r1 = matcher.plot_mag.setYRange.call_args[0]   # (mn, mx)
        r2 = matcher._vb_mag2.setYRange.call_args[0]
        # They should not be identical
        assert r1 != r2

    def test_no_error_when_plots_absent(self, matcher):
        """auto_scale_y returns silently if plot_mag not yet created."""
        if hasattr(matcher, 'plot_mag'):
            del matcher.plot_mag
        matcher.auto_scale_y()   # must not raise


# ---------------------------------------------------------------------------
# Overview navigator
# ---------------------------------------------------------------------------

class TestOverviewNavigator:
    def _attach_overview_mocks(self, matcher):
        matcher._overview_region = Mock()
        matcher._overview_updating = False
        matcher.plot_mag = Mock()

    def test_update_overview_region_sets_region(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher.plot_mag.viewRange.return_value = [[5e9, 6e9], [0, 1]]
        matcher._update_overview_region()
        matcher._overview_region.setRegion.assert_called_once_with([5e9, 6e9])

    def test_update_overview_region_respects_guard(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher._overview_updating = True
        matcher.plot_mag.viewRange.return_value = [[5e9, 6e9], [0, 1]]
        matcher._update_overview_region()
        matcher._overview_region.setRegion.assert_not_called()

    def test_on_overview_region_changed_sets_xrange(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher._overview_region.getRegion.return_value = (5e9, 6e9)
        matcher._on_overview_region_changed()
        matcher.plot_mag.setXRange.assert_called_once_with(5e9, 6e9, padding=0)

    def test_on_overview_region_changed_respects_guard(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher._overview_updating = True
        matcher._on_overview_region_changed()
        matcher.plot_mag.setXRange.assert_not_called()

    def test_guard_released_after_update_overview(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher.plot_mag.viewRange.return_value = [[5e9, 6e9], [0, 1]]
        matcher._update_overview_region()
        assert matcher._overview_updating is False

    def test_guard_released_after_region_changed(self, matcher):
        self._attach_overview_mocks(matcher)
        matcher._overview_region.getRegion.return_value = (5e9, 6e9)
        matcher._on_overview_region_changed()
        assert matcher._overview_updating is False


# ---------------------------------------------------------------------------
# run_res_matcher wrapper
# ---------------------------------------------------------------------------

class TestRunResMatcher:
    def test_returns_list_of_match_groups(self, simple_data, tmp_path, qt_app):
        sd = simple_data
        grp_path = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            result = run_res_matcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp_path,
                overwrite=True,
            )
        assert isinstance(result, list)
        assert all(isinstance(g, MatchGroup) for g in result)

    def test_result_length_matches_groups(self, simple_data, tmp_path, qt_app):
        sd = simple_data
        grp_path = str(tmp_path / 'out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            result = run_res_matcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp_path,
                overwrite=True,
            )
        # 4 DS1 + 4 DS2 with sorted pairing → 4 groups
        assert len(result) == len(sd['fres1'])

    def test_zarr_path_string_accepted(self, simple_data, tmp_path, qt_app):
        sd = simple_data
        grp_path = str(tmp_path / 'str_out.zarr')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            result = run_res_matcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp_path,
                overwrite=True,
            )
        assert result is not None

    def test_zarr_group_object_accepted(self, simple_data, tmp_path, qt_app):
        sd = simple_data
        grp = zarr.open_group(str(tmp_path / 'grp_out.zarr'), mode='a')
        with patch.object(ResMatcher, 'setup_ui'), \
             patch.object(ResMatcher, 'run'):
            result = run_res_matcher(
                sd['f'], sd['z1'], sd['fres1'], sd['res_idx1'],
                sd['f'], sd['z2'], sd['fres2'], sd['res_idx2'],
                grp,
                overwrite=True,
            )
        assert isinstance(result, list)
