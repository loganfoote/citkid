"""
Comprehensive tests for citkid.vna.fq_finder.

Coverage
--------
_db                   -- dB conversion helper
_ensure_zarr_arrays   -- zarr array creation / resume logic
_rmv_gain_simple      -- amplitude + phase normalisation
_SpanRegionItem       -- non-interactive body
_InteractiveViewBox   -- shift-click / scroll signal emission
FqFinderWindow        -- all public and private methods:
    __init__, _build_ui, _load_resonator, _update_overlay,
    _reject_current, _on_reason_combo_changed, _reset_current,
    _save_current, _go_next, _go_back, _set_fres, _set_qres,
    _on_spinbox_fres, _on_spinbox_qres, _on_fres_vline_moved,
    _on_span_region_changed, _on_shift_click_amp, _on_shift_click_iq,
    _on_scroll_qres, eventFilter, closeEvent
"""

import numpy as np
import pytest
import zarr
from unittest.mock import MagicMock, patch

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from citkid.vna.fq_finder import (
    _db,
    _ensure_zarr_arrays,
    _rmv_gain_simple,
    _SpanRegionItem,
    _InteractiveViewBox,
    FqFinderWindow,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sweep(M=3, N=60, base_freq=4e9, spacing=5e6, reversed_f=False):
    """
    Return ``(f, z, fres, qres, res_idxs)`` arrays with *M* resonators,
    each with *N* frequency points spanning ±1 MHz around *fres*.

    Parameters
    ----------
    reversed_f : bool
        If True, return frequencies in descending order (tests axis-1 sort).
    """
    rng = np.random.default_rng(42)
    fres = np.array([base_freq + i * spacing for i in range(M)])
    qres = np.full(M, 1e4)
    f = np.vstack([np.linspace(fr - 1e6, fr + 1e6, N) for fr in fres])
    if reversed_f:
        f = f[:, ::-1]
    z = np.ones((M, N), dtype=complex)
    for i, (fr, q) in enumerate(zip(fres, qres)):
        df = f[i] - fr if not reversed_f else f[i, ::-1] - fr
        lz = 1.0 / (1 + 2j * q * (f[i] - fr) / fr)
        z[i] = 1 - lz
        z[i] += 0.005 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    res_idxs = np.arange(1, M + 1)
    return f, z, fres.copy(), qres.copy(), res_idxs


def _make_zarr():
    """Return an empty in-memory zarr group."""
    return zarr.group()


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole test session (offscreen)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def basic_win(qapp):
    """FqFinderWindow with 3 interactive resonators (fresh zarr group)."""
    f, z, fres, qres, res_idxs = _make_sweep()
    zg = _make_zarr()
    w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
    yield w
    w.close()


# ===========================================================================
# _db
# ===========================================================================

class TestDb:
    def test_scalar_like(self):
        z = np.array([10.0 + 0j])
        np.testing.assert_allclose(_db(z), [20.0])

    def test_shape_preserved(self):
        z = np.ones((4, 10), dtype=complex)
        assert _db(z).shape == (4, 10)

    def test_all_ones_gives_zero(self):
        z = np.ones(5, dtype=complex)
        np.testing.assert_allclose(_db(z), np.zeros(5))

    def test_magnitude_100(self):
        z = np.full(3, 100.0 + 0j)
        np.testing.assert_allclose(_db(z), np.full(3, 40.0))

    def test_zero_magnitude_gives_neg_inf(self):
        z = np.array([0.0 + 0j])
        result = _db(z)
        assert np.isneginf(result[0])

    def test_imaginary_only(self):
        z = np.array([0 + 1j])
        np.testing.assert_allclose(_db(z), [0.0])


# ===========================================================================
# _ensure_zarr_arrays
# ===========================================================================

class TestEnsureZarrArrays:
    def test_fresh_creates_all_three_arrays(self):
        zg = _make_zarr()
        result = _ensure_zarr_arrays(zg, 5, overwrite=False)
        assert result is False
        assert "fres_opt" in zg
        assert "qres_opt" in zg
        assert "reject_reason" in zg

    def test_fresh_arrays_have_correct_shape(self):
        zg = _make_zarr()
        _ensure_zarr_arrays(zg, 7, overwrite=False)
        assert zg["fres_opt"].shape == (7,)
        assert zg["qres_opt"].shape == (7,)
        assert zg["reject_reason"].shape == (7,)

    def test_fresh_float_arrays_filled_with_nan(self):
        zg = _make_zarr()
        _ensure_zarr_arrays(zg, 4, overwrite=False)
        assert np.all(np.isnan(zg["fres_opt"][:]))
        assert np.all(np.isnan(zg["qres_opt"][:]))

    def test_existing_overwrite_false_raises(self):
        zg = _make_zarr()
        _ensure_zarr_arrays(zg, 3, overwrite=False)
        with pytest.raises(FileExistsError):
            _ensure_zarr_arrays(zg, 3, overwrite=False)

    def test_existing_overwrite_true_returns_true(self):
        zg = _make_zarr()
        _ensure_zarr_arrays(zg, 3, overwrite=False)
        result = _ensure_zarr_arrays(zg, 3, overwrite=True)
        assert result is True

    def test_existing_overwrite_true_preserves_data(self):
        zg = _make_zarr()
        _ensure_zarr_arrays(zg, 3, overwrite=False)
        zg["fres_opt"][0] = 1.23e9
        _ensure_zarr_arrays(zg, 3, overwrite=True)
        assert zg["fres_opt"][0] == pytest.approx(1.23e9)

    def test_only_one_array_raises_runtime_error(self):
        zg = _make_zarr()
        zg.create_dataset("fres_opt", shape=(3,), dtype=np.float64, fill_value=np.nan)
        with pytest.raises(RuntimeError, match="inconsistent state"):
            _ensure_zarr_arrays(zg, 3, overwrite=False)

    def test_resume_without_reject_reason_creates_it(self):
        """Legacy zarr group missing reject_reason → created on resume."""
        zg = _make_zarr()
        zg.create_dataset("fres_opt", shape=(4,), dtype=np.float64, fill_value=np.nan)
        zg.create_dataset("qres_opt", shape=(4,), dtype=np.float64, fill_value=np.nan)
        assert "reject_reason" not in zg
        _ensure_zarr_arrays(zg, 4, overwrite=True)
        assert "reject_reason" in zg
        assert zg["reject_reason"].shape == (4,)


# ===========================================================================
# _rmv_gain_simple
# ===========================================================================

class TestRmvGainSimple:
    def _make_input(self, M=4, N=100):
        rng = np.random.default_rng(7)
        f = np.vstack([np.linspace(4e9 - 1e6, 4e9 + 1e6, N) for _ in range(M)])
        amp = rng.uniform(0.5, 2.0, (M, 1))
        phase = rng.uniform(-np.pi, np.pi, (M, 1))
        z = amp * np.exp(1j * phase) * np.ones((M, N))
        # Add a resonance dip in the middle
        for i in range(M):
            z[i, N // 4: 3 * N // 4] *= 0.5
        return f, z

    def test_output_shape_unchanged(self):
        f, z = self._make_input()
        out = _rmv_gain_simple(f, z)
        assert out.shape == z.shape

    def test_output_is_complex(self):
        f, z = self._make_input()
        out = _rmv_gain_simple(f, z)
        assert np.iscomplexobj(out)

    def test_off_resonance_amplitude_near_one(self):
        """After normalisation the off-resonance edge samples ≈ amplitude 1."""
        f, z = self._make_input(M=2, N=200)
        out = _rmv_gain_simple(f, z)
        N = max(1, out.shape[1] // 100)
        edge = np.concatenate([out[:, :N], out[:, -N:]], axis=1)
        amp = np.abs(edge)
        np.testing.assert_allclose(np.median(amp, axis=1), np.ones(2), rtol=0.05)

    def test_off_resonance_phase_near_zero(self):
        """After phase rotation the off-resonance mean should be real and positive."""
        f, z = self._make_input(M=2, N=200)
        out = _rmv_gain_simple(f, z)
        N = max(1, out.shape[1] // 100)
        edge = np.concatenate([out[:, :N], out[:, -N:]], axis=1)
        mean_phase = np.angle(edge.mean(axis=1))
        np.testing.assert_allclose(mean_phase, np.zeros(2), atol=1e-10)

    def test_zero_amplitude_no_exception(self):
        """Rows with all-zero amplitude should not raise (divided by 1.0)."""
        M, N = 2, 50
        f = np.vstack([np.linspace(4e9, 5e9, N)] * M)
        z = np.zeros((M, N), dtype=complex)
        out = _rmv_gain_simple(f, z)
        assert out.shape == z.shape
        assert np.all(np.isfinite(out))

    def test_n_equals_one(self):
        """N=1 edge: should not crash (N=max(1, 1//100)=1)."""
        f = np.array([[4e9]])
        z = np.array([[1.0 + 0.5j]])
        out = _rmv_gain_simple(f, z)
        assert out.shape == (1, 1)

    def test_2d_input(self):
        f, z = self._make_input(M=5, N=150)
        out = _rmv_gain_simple(f, z)
        assert out.shape == (5, 150)


# ===========================================================================
# _SpanRegionItem
# ===========================================================================

class TestSpanRegionItem:
    def test_instantiation(self, qapp):
        item = _SpanRegionItem()
        assert item is not None

    def test_mouseDragEvent_ignores(self, qapp):
        item = _SpanRegionItem()
        ev = MagicMock()
        item.mouseDragEvent(ev)
        ev.ignore.assert_called_once()

    def test_mousePressEvent_ignores(self, qapp):
        item = _SpanRegionItem()
        ev = MagicMock()
        item.mousePressEvent(ev)
        ev.ignore.assert_called_once()

    def test_hoverEvent_does_not_raise(self, qapp):
        item = _SpanRegionItem()
        ev = MagicMock()
        item.hoverEvent(ev)  # should be a no-op


# ===========================================================================
# _InteractiveViewBox
# ===========================================================================

class TestInteractiveViewBox:
    def test_instantiation(self, qapp):
        vb = _InteractiveViewBox()
        assert vb is not None

    def test_has_sig_shift_click(self, qapp):
        vb = _InteractiveViewBox()
        assert hasattr(vb, "sig_shift_click")

    def test_has_sig_scroll_qres(self, qapp):
        vb = _InteractiveViewBox()
        assert hasattr(vb, "sig_scroll_qres")

    def test_sig_shift_click_connectable(self, qapp):
        vb = _InteractiveViewBox()
        received = []
        vb.sig_shift_click.connect(lambda x, y: received.append((x, y)))
        vb.sig_shift_click.emit(1.0, 2.0)
        assert received == [(1.0, 2.0)]

    def test_sig_scroll_qres_connectable(self, qapp):
        vb = _InteractiveViewBox()
        received = []
        vb.sig_scroll_qres.connect(lambda d: received.append(d))
        vb.sig_scroll_qres.emit(5)
        assert received == [5]


# ===========================================================================
# FqFinderWindow — __init__ and basic structure
# ===========================================================================

class TestFqFinderWindowInit:
    def test_basic_creation(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert w is not None
        w.close()

    def test_zarr_arrays_created(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert "fres_opt" in zg
        assert "qres_opt" in zg
        assert "reject_reason" in zg
        w.close()

    def test_f_sorted_ascending(self, qapp):
        """Even if f is passed in reversed order, axis-1 should be ascending."""
        f, z, fres, qres, res_idxs = _make_sweep(M=2, reversed_f=True)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        for i in range(w._M):
            assert np.all(np.diff(w._f[i]) > 0), f"Row {i} not ascending"
        w.close()

    def test_cal_tones_pre_saved(self, qapp):
        """Resonators with res_idxs < 1 are written to zarr immediately."""
        f, z, fres, qres, _ = _make_sweep(M=3)
        res_idxs = np.array([-1, 1, 2])  # index 0 is calibration
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert zg["fres_opt"][0] == pytest.approx(w._fres_init[0])
        assert zg["qres_opt"][0] == pytest.approx(w._qres_init[0])
        w.close()

    def test_interactive_indices_excludes_cal_tones(self, qapp):
        f, z, fres, qres, _ = _make_sweep(M=3)
        res_idxs = np.array([0, 1, 2])  # 0 is cal
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert 0 not in w._interactive_indices
        assert 1 in w._interactive_indices
        assert 2 in w._interactive_indices
        w.close()

    def test_start_idx_sets_cursor(self, qapp):
        """start_idx positions cursor at first interactive index >= start_idx."""
        f, z, fres, qres, res_idxs = _make_sweep(M=4)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg, start_idx=2)
        # Interactive indices are [0,1,2,3]. First index >= 2 is at cursor pos 2.
        assert w._interactive_indices[w._cursor] >= 2
        w.close()

    def test_start_idx_zero_starts_at_beginning(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=3)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg, start_idx=0)
        assert w._cursor == 0
        w.close()

    def test_start_idx_beyond_end_wraps_to_zero(self, qapp):
        """If start_idx > all indices, cursor falls back to 0."""
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg, start_idx=999)
        assert w._cursor == 0
        w.close()

    def test_all_cal_tones_no_interactive(self, qapp):
        """All-cal input: interactive_indices is empty; window closes itself."""
        f, z, fres, qres, _ = _make_sweep(M=2)
        res_idxs = np.array([0, -1])  # both cal
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert w._interactive_indices == []
        # Window should have called close() – just verify no crash

    def test_rmv_gain_simple_applied(self, qapp):
        """With rmv_gain_simple=True the stored z should differ from input."""
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        # Add a non-trivial amplitude offset so normalisation changes z
        z_off = z * 5.0 * np.exp(1j * 0.7)
        zg = _make_zarr()
        w = FqFinderWindow(f, z_off, fres, qres, res_idxs, zg,
                           rmv_gain_simple=True)
        assert not np.allclose(w._z, z_off)
        w.close()

    def test_fres_init_equals_fres_work_at_start(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        np.testing.assert_array_equal(w._fres_init, w._fres_work)
        np.testing.assert_array_equal(w._qres_init, w._qres_work)
        w.close()

    def test_reject_reasons_empty_at_start(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert w._reject_reasons == {}
        w.close()

    def test_overwrite_false_raises_if_arrays_exist(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        w.close()
        with pytest.raises(FileExistsError):
            FqFinderWindow(f, z, fres, qres, res_idxs, zg, overwrite=False)

    def test_overwrite_true_resumes(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        _ensure_zarr_arrays(zg, 2, overwrite=False)
        zg["fres_opt"][0] = 1.23e9
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg, overwrite=True)
        assert zg["fres_opt"][0] == pytest.approx(1.23e9)
        w.close()

    def test_title_set(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=1)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg, title="My Test")
        assert w.windowTitle() == "My Test"
        w.close()

    def test_reject_reasons_class_constant(self):
        expected = ["tone off resonance", "overlapping resonance",
                    "bifurcated", "other"]
        assert FqFinderWindow._REJECT_REASONS == expected


# ===========================================================================
# FqFinderWindow — _reject_current / _on_reason_combo_changed
# ===========================================================================

class TestFqFinderReject:
    def test_reject_sets_fres_to_nan(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        assert np.isnan(w._fres_work[ri])

    def test_reject_sets_qres_to_nan(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        assert np.isnan(w._qres_work[ri])

    def test_reject_stores_preset_reason(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reason_combo.setCurrentText("bifurcated")
        w._reject_current()
        assert w._reject_reasons[ri] == "bifurcated"

    def test_reject_stores_custom_reason(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reason_combo.setCurrentText("other")
        w._reason_edit.setText("my custom reason")
        w._reject_current()
        assert w._reject_reasons[ri] == "my custom reason"

    def test_reject_other_with_empty_text_stores_other(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reason_combo.setCurrentText("other")
        w._reason_edit.clear()
        w._reject_current()
        assert w._reject_reasons[ri] == "other"

    def test_reject_hides_overlay(self, basic_win):
        w = basic_win
        w._reject_current()
        assert not w._fres_vline.isVisible()
        assert not w._span_region.isVisible()

    def test_reject_reason_combo_has_all_options(self, basic_win):
        w = basic_win
        items = [w._reason_combo.itemText(i)
                 for i in range(w._reason_combo.count())]
        assert items == FqFinderWindow._REJECT_REASONS

    def test_reason_edit_hidden_by_default(self, basic_win):
        assert basic_win._reason_edit.isHidden()

    def test_on_reason_combo_changed_shows_edit_for_other(self, basic_win):
        w = basic_win
        w._reason_combo.setCurrentText("other")
        assert not w._reason_edit.isHidden()

    def test_on_reason_combo_changed_hides_edit_for_preset(self, basic_win):
        w = basic_win
        w._reason_combo.setCurrentText("other")
        assert not w._reason_edit.isHidden()
        w._reason_combo.setCurrentText("bifurcated")
        assert w._reason_edit.isHidden()


# ===========================================================================
# FqFinderWindow — _reset_current
# ===========================================================================

class TestFqFinderReset:
    def test_reset_restores_fres(self, basic_win):
        w = basic_win
        ri = w._ri
        original = w._fres_init[ri]
        w._fres_work[ri] = original + 1e6
        w._reset_current()
        assert w._fres_work[ri] == pytest.approx(original)

    def test_reset_restores_qres(self, basic_win):
        w = basic_win
        ri = w._ri
        original = w._qres_init[ri]
        w._qres_work[ri] = original * 2
        w._reset_current()
        assert w._qres_work[ri] == pytest.approx(original)

    def test_reset_clears_reject_reason(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        assert ri in w._reject_reasons
        w._reset_current()
        assert ri not in w._reject_reasons

    def test_reset_after_nan_restores_valid_overlay(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        assert np.isnan(w._fres_work[ri])
        w._reset_current()
        assert not np.isnan(w._fres_work[ri])


# ===========================================================================
# FqFinderWindow — _save_current
# ===========================================================================

class TestFqFinderSaveCurrent:
    def test_save_writes_fres_to_zarr(self, basic_win):
        w = basic_win
        ri = w._ri
        new_fres = w._fres_init[ri] + 500e3
        w._fres_work[ri] = new_fres
        w._save_current()
        assert w._zg["fres_opt"][ri] == pytest.approx(new_fres)

    def test_save_writes_qres_to_zarr(self, basic_win):
        w = basic_win
        ri = w._ri
        w._qres_work[ri] = 12345.0
        w._save_current()
        assert w._zg["qres_opt"][ri] == pytest.approx(12345.0)

    def test_save_writes_empty_reason_for_accepted(self, basic_win):
        w = basic_win
        ri = w._ri
        w._save_current()
        assert str(w._zg["reject_reason"][ri]) == ""

    def test_save_writes_reason_for_rejected(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reason_combo.setCurrentText("overlapping resonance")
        w._reject_current()
        w._save_current()
        assert str(w._zg["reject_reason"][ri]) == "overlapping resonance"

    def test_save_nan_fres_stored(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        w._save_current()
        assert np.isnan(w._zg["fres_opt"][ri])
        assert np.isnan(w._zg["qres_opt"][ri])


# ===========================================================================
# FqFinderWindow — navigation (_go_next / _go_back / closeEvent)
# ===========================================================================

class TestFqFinderNavigation:
    def test_go_next_increments_cursor(self, basic_win):
        w = basic_win
        start = w._cursor
        w._go_next()
        assert w._cursor == start + 1

    def test_go_next_saves_fres_to_zarr(self, basic_win):
        w = basic_win
        ri = w._ri
        new_f = w._fres_init[ri] + 200e3
        w._fres_work[ri] = new_f
        w._go_next()
        assert w._zg["fres_opt"][ri] == pytest.approx(new_f)

    def test_go_next_at_last_resonator_does_not_overflow(self, qapp):
        f, z, fres, qres, _ = _make_sweep(M=1)
        res_idxs = np.array([1])
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        assert w._cursor == 0
        w._go_next()
        # cursor must not go past len-1
        assert w._cursor == 0
        w.close()

    def test_go_next_at_last_disables_next_button(self, qapp):
        f, z, fres, qres, _ = _make_sweep(M=1)
        res_idxs = np.array([1])
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        w._go_next()
        assert not w._next_btn.isEnabled()
        w.close()

    def test_go_back_decrements_cursor(self, basic_win):
        w = basic_win
        w._go_next()  # cursor = 1
        cursor_before = w._cursor
        w._go_back()
        assert w._cursor == cursor_before - 1

    def test_go_back_saves_current_first(self, basic_win):
        w = basic_win
        w._go_next()  # move to cursor=1
        ri = w._ri
        new_f = w._fres_init[ri] + 100e3
        w._fres_work[ri] = new_f
        w._go_back()
        assert w._zg["fres_opt"][ri] == pytest.approx(new_f)

    def test_go_back_at_first_does_nothing(self, basic_win):
        w = basic_win
        assert w._cursor == 0
        w._go_back()
        assert w._cursor == 0

    def test_close_event_saves_current(self, basic_win):
        w = basic_win
        ri = w._ri
        new_f = w._fres_init[ri] + 77e3
        w._fres_work[ri] = new_f
        w.close()
        assert w._zg["fres_opt"][ri] == pytest.approx(new_f)

    def test_prev_btn_disabled_at_start(self, basic_win):
        assert not basic_win._prev_btn.isEnabled()

    def test_prev_btn_enabled_after_go_next(self, basic_win):
        w = basic_win
        w._go_next()
        assert w._prev_btn.isEnabled()

    def test_ri_property(self, basic_win):
        w = basic_win
        assert w._ri == w._interactive_indices[w._cursor]


# ===========================================================================
# FqFinderWindow — _set_fres / _set_qres
# ===========================================================================

class TestFqFinderSetFresQres:
    def test_set_fres_updates_work_array(self, basic_win):
        w = basic_win
        ri = w._ri
        new_f = 4.123e9
        w._set_fres(new_f)
        assert w._fres_work[ri] == pytest.approx(new_f)

    def test_set_fres_updates_spinbox(self, basic_win):
        w = basic_win
        new_f = 4.123e9
        w._set_fres(new_f)
        assert w._fres_spin.value() == pytest.approx(new_f * 1e-6, rel=1e-4)

    def test_set_qres_updates_work_array(self, basic_win):
        w = basic_win
        ri = w._ri
        w._set_qres(5000.0)
        assert w._qres_work[ri] == pytest.approx(5000.0)

    def test_set_qres_updates_spinbox(self, basic_win):
        w = basic_win
        w._set_qres(5000.0)
        assert w._qres_spin.value() == pytest.approx(5000.0)

    def test_set_qres_enforces_minimum_one(self, basic_win):
        w = basic_win
        ri = w._ri
        w._set_qres(0.0)
        assert w._qres_work[ri] == pytest.approx(1.0)

    def test_set_qres_enforces_minimum_negative(self, basic_win):
        w = basic_win
        ri = w._ri
        w._set_qres(-500.0)
        assert w._qres_work[ri] == pytest.approx(1.0)


# ===========================================================================
# FqFinderWindow — spinbox callbacks
# ===========================================================================

class TestFqFinderSpinboxCallbacks:
    def test_on_spinbox_fres_converts_mhz_to_hz(self, basic_win):
        w = basic_win
        ri = w._ri
        w._on_spinbox_fres(4000.0)   # 4000 MHz → 4 GHz
        assert w._fres_work[ri] == pytest.approx(4e9)

    def test_on_spinbox_qres_clamps_to_one(self, basic_win):
        w = basic_win
        ri = w._ri
        w._on_spinbox_qres(0.0)
        assert w._qres_work[ri] == pytest.approx(1.0)

    def test_on_spinbox_qres_normal_value(self, basic_win):
        w = basic_win
        ri = w._ri
        w._on_spinbox_qres(8000.0)
        assert w._qres_work[ri] == pytest.approx(8000.0)

    def test_on_spinbox_fres_nan_qres_no_crash(self, basic_win):
        """Spinbox fres update should not crash when qres is NaN."""
        w = basic_win
        ri = w._ri
        w._reject_current()
        w._on_spinbox_fres(4000.0)  # should not raise


# ===========================================================================
# FqFinderWindow — vline / span / click / scroll callbacks
# ===========================================================================

class TestFqFinderInteractionCallbacks:
    def test_on_fres_vline_moved_updates_fres(self, basic_win):
        w = basic_win
        ri = w._ri
        f0 = w._f0_hz
        # Simulate moving vline to +1 kHz offset
        w._fres_vline.setValue(1.0)  # 1 kHz offset in plot coords
        # Manually call the handler (sigPositionChanged fires automatically,
        # but we call directly to avoid re-entry issues in test)
        w._fres_updating = False
        w._on_fres_vline_moved()
        expected = 1.0 * 1e3 + f0
        assert w._fres_work[ri] == pytest.approx(expected)

    def test_on_span_region_changed_updates_qres(self, basic_win):
        w = basic_win
        ri = w._ri
        fres = w._fres_work[ri]
        f0 = w._f0_hz
        fres_khz = (fres - f0) * 1e-3
        # Set span to ±0.5 kHz → span_hz = 1 kHz → qres = fres/1000
        w._span_region.setRegion([fres_khz - 0.5, fres_khz + 0.5])
        w._span_updating = False
        w._on_span_region_changed()
        expected_qres = fres / 1e3
        assert w._qres_work[ri] == pytest.approx(expected_qres, rel=0.01)

    def test_on_span_region_changed_nan_fres_no_crash(self, basic_win):
        w = basic_win
        w._reject_current()
        w._span_updating = False
        w._on_span_region_changed()  # should not raise

    def test_on_shift_click_amp_sets_fres(self, basic_win):
        w = basic_win
        ri = w._ri
        f0 = w._f0_hz
        # Click at +2 kHz offset
        w._on_shift_click_amp(2.0, 0.0)
        expected = 2.0 * 1e3 + f0
        assert w._fres_work[ri] == pytest.approx(expected)

    def test_on_shift_click_amp_restores_qres_if_nan(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        assert np.isnan(w._qres_work[ri])
        w._on_shift_click_amp(0.0, 0.0)
        assert not np.isnan(w._qres_work[ri])
        assert w._qres_work[ri] == pytest.approx(w._qres_init[ri])

    def test_on_shift_click_iq_snaps_to_nearest_sample(self, basic_win):
        w = basic_win
        ri = w._ri
        z = w._z[ri]
        f = w._f[ri]
        # Click exactly on the first sample
        x_re, y_im = z.real[0], z.imag[0]
        w._on_shift_click_iq(x_re, y_im)
        assert w._fres_work[ri] == pytest.approx(float(f[0]))

    def test_on_shift_click_iq_restores_qres_if_nan(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        z = w._z[ri]
        w._on_shift_click_iq(z.real[0], z.imag[0])
        assert not np.isnan(w._qres_work[ri])
        assert w._qres_work[ri] == pytest.approx(w._qres_init[ri])

    def test_on_scroll_qres_increases(self, basic_win):
        w = basic_win
        ri = w._ri
        qres_before = w._qres_work[ri]
        w._on_scroll_qres(1)
        assert w._qres_work[ri] > qres_before

    def test_on_scroll_qres_decreases(self, basic_win):
        w = basic_win
        ri = w._ri
        qres_before = w._qres_work[ri]
        w._on_scroll_qres(-1)
        assert w._qres_work[ri] < qres_before

    def test_on_scroll_qres_no_op_when_nan(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_current()
        # Should not raise and should not change the NaN state
        w._on_scroll_qres(5)
        assert np.isnan(w._qres_work[ri])

    def test_on_scroll_qres_multiplicative_factor(self, basic_win):
        w = basic_win
        ri = w._ri
        qres_before = w._qres_work[ri]
        w._on_scroll_qres(1)
        expected = qres_before * (1.0 + FqFinderWindow._QRES_FRAC)
        assert w._qres_work[ri] == pytest.approx(expected, rel=1e-6)

    def test_on_scroll_qres_coarse_ten_steps(self, basic_win):
        w = basic_win
        ri = w._ri
        qres_before = w._qres_work[ri]
        w._on_scroll_qres(10)
        expected = qres_before * (1.0 + 10 * FqFinderWindow._QRES_FRAC)
        assert w._qres_work[ri] == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# FqFinderWindow — _update_overlay
# ===========================================================================

class TestFqFinderUpdateOverlay:
    def test_nan_fres_hides_vline(self, basic_win):
        w = basic_win
        ri = w._ri
        w._update_overlay(w._f[ri], w._z[ri], np.nan, 1e4)
        assert not w._fres_vline.isVisible()

    def test_nan_qres_hides_vline(self, basic_win):
        w = basic_win
        ri = w._ri
        w._update_overlay(w._f[ri], w._z[ri], w._fres_work[ri], np.nan)
        assert not w._fres_vline.isVisible()

    def test_nan_hides_span_region(self, basic_win):
        w = basic_win
        ri = w._ri
        w._update_overlay(w._f[ri], w._z[ri], np.nan, np.nan)
        assert not w._span_region.isVisible()

    def test_valid_shows_vline(self, basic_win):
        w = basic_win
        ri = w._ri
        # First hide it
        w._update_overlay(w._f[ri], w._z[ri], np.nan, np.nan)
        # Then show it
        w._update_overlay(w._f[ri], w._z[ri],
                          w._fres_work[ri], w._qres_work[ri])
        assert w._fres_vline.isVisible()

    def test_valid_shows_span_region(self, basic_win):
        w = basic_win
        ri = w._ri
        w._update_overlay(w._f[ri], w._z[ri], np.nan, np.nan)
        w._update_overlay(w._f[ri], w._z[ri],
                          w._fres_work[ri], w._qres_work[ri])
        assert w._span_region.isVisible()

    def test_vline_position_khz_offset(self, basic_win):
        w = basic_win
        ri = w._ri
        fres = w._fres_work[ri]
        f0 = w._f0_hz
        w._update_overlay(w._f[ri], w._z[ri], fres, w._qres_work[ri])
        expected_khz = (fres - f0) * 1e-3
        assert w._fres_vline.value() == pytest.approx(expected_khz, rel=1e-6)

    def test_span_region_width_matches_qres(self, basic_win):
        w = basic_win
        ri = w._ri
        fres = w._fres_work[ri]
        qres = w._qres_work[ri]
        f0 = w._f0_hz
        w._update_overlay(w._f[ri], w._z[ri], fres, qres)
        lo_khz, hi_khz = w._span_region.getRegion()
        span_hz = (hi_khz - lo_khz) * 1e3
        expected_span_hz = fres / qres
        assert span_hz == pytest.approx(expected_span_hz, rel=1e-5)

    def test_fres_x_data_set_for_valid_input(self, basic_win):
        w = basic_win
        ri = w._ri
        # First clear it via NaN
        w._update_overlay(w._f[ri], w._z[ri], np.nan, np.nan)
        xs_nan, _ = w._fres_x.getData()
        assert xs_nan is None or len(xs_nan) == 0
        # Now provide valid values
        w._update_overlay(w._f[ri], w._z[ri],
                          w._fres_work[ri], w._qres_work[ri])
        xs, ys = w._fres_x.getData()
        assert xs is not None and len(xs) == 1

    def test_fres_x_position_is_interpolated(self, basic_win):
        w = basic_win
        ri = w._ri
        fres = w._fres_work[ri]
        f, z = w._f[ri], w._z[ri]
        w._update_overlay(f, z, fres, w._qres_work[ri])
        xs, ys = w._fres_x.getData()
        expected_x = np.interp(fres, f, z.real)
        expected_y = np.interp(fres, f, z.imag)
        assert xs[0] == pytest.approx(expected_x, rel=1e-6)
        assert ys[0] == pytest.approx(expected_y, rel=1e-6)


# ===========================================================================
# FqFinderWindow — _load_resonator
# ===========================================================================

class TestFqFinderLoadResonator:
    def test_f0_hz_set_to_fres_init(self, basic_win):
        w = basic_win
        ri = w._ri
        w._load_resonator()
        assert w._f0_hz == pytest.approx(w._fres_init[ri])

    def test_status_label_shows_resonator_index(self, basic_win):
        w = basic_win
        w._load_resonator()
        text = w._status_label.text()
        ri = w._ri
        res_idx_str = str(int(w._res_idxs[ri]))
        assert res_idx_str in text

    def test_status_label_shows_cursor_position(self, basic_win):
        w = basic_win
        w._load_resonator()
        text = w._status_label.text()
        n_tot = len(w._interactive_indices)
        assert f"1/{n_tot}" in text

    def test_spinbox_fres_reflects_fres_mhz(self, basic_win):
        w = basic_win
        ri = w._ri
        w._load_resonator()
        expected = w._fres_work[ri] * 1e-6
        assert w._fres_spin.value() == pytest.approx(expected, rel=1e-4)

    def test_spinbox_fres_zero_when_nan(self, basic_win):
        w = basic_win
        ri = w._ri
        w._fres_work[ri] = np.nan
        w._load_resonator()
        assert w._fres_spin.value() == pytest.approx(0.0)

    def test_spinbox_qres_clamped_to_min_when_nan(self, basic_win):
        """When qres is NaN, the spinbox is set to 0 but clamps to its minimum (1.0)."""
        w = basic_win
        ri = w._ri
        w._qres_work[ri] = np.nan
        w._load_resonator()
        assert w._qres_spin.value() == pytest.approx(1.0)

    def test_reason_combo_reset_to_first_when_no_stored_reason(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_reasons.pop(ri, None)
        # Set combo to something else first
        w._reason_combo.blockSignals(True)
        w._reason_combo.setCurrentText("bifurcated")
        w._reason_combo.blockSignals(False)
        w._load_resonator()
        assert w._reason_combo.currentIndex() == 0

    def test_reason_combo_restored_for_rejected_resonator(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_reasons[ri] = "bifurcated"
        w._load_resonator()
        assert w._reason_combo.currentText() == "bifurcated"

    def test_reason_edit_shown_for_other_reason(self, basic_win):
        w = basic_win
        ri = w._ri
        w._reject_reasons[ri] = "some custom text"
        w._load_resonator()
        assert w._reason_combo.currentText() == "other"
        assert not w._reason_edit.isHidden()
        assert w._reason_edit.text() == "some custom text"

    def test_reason_edit_empty_for_stored_other(self, basic_win):
        """If stored reason is literally "other", text field should be empty."""
        w = basic_win
        ri = w._ri
        w._reject_reasons[ri] = "other"
        w._load_resonator()
        assert w._reason_edit.text() == ""


# ===========================================================================
# FqFinderWindow — _ri property and cursor invariants
# ===========================================================================

class TestFqFinderCursorInvariants:
    def test_ri_is_interactive_index(self, basic_win):
        w = basic_win
        assert w._ri == w._interactive_indices[0]

    def test_ri_changes_after_go_next(self, basic_win):
        w = basic_win
        ri_before = w._ri
        w._go_next()
        assert w._ri != ri_before

    def test_cursor_bounds_respected_after_repeated_next(self, basic_win):
        w = basic_win
        n = len(w._interactive_indices)
        for _ in range(n + 5):
            w._go_next()
        assert w._cursor == n - 1

    def test_cursor_never_negative(self, basic_win):
        w = basic_win
        for _ in range(10):
            w._go_back()
        assert w._cursor >= 0


# ===========================================================================
# FqFinderWindow — reject → navigate → zarr round-trip
# ===========================================================================

class TestFqFinderRoundTrip:
    def test_reject_navigate_saves_reason(self, qapp):
        """Reject resonator 0, navigate to 1, come back → reason persisted."""
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        ri0 = w._ri
        w._reason_combo.setCurrentText("tone off resonance")
        w._reject_current()
        w._go_next()    # saves ri0 then moves to cursor=1
        assert str(zg["reject_reason"][ri0]) == "tone off resonance"
        assert np.isnan(zg["fres_opt"][ri0])
        w.close()

    def test_accept_navigate_saves_empty_reason(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        ri0 = w._ri
        w._go_next()
        assert str(zg["reject_reason"][ri0]) == ""
        assert not np.isnan(zg["fres_opt"][ri0])
        w.close()

    def test_reject_reset_navigate_clears_reason_in_zarr(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        ri0 = w._ri
        w._reason_combo.setCurrentText("bifurcated")
        w._reject_current()
        w._reset_current()      # un-reject
        w._go_next()            # saves → empty reason
        assert str(zg["reject_reason"][ri0]) == ""
        assert not np.isnan(zg["fres_opt"][ri0])
        w.close()

    def test_set_fres_go_next_then_go_back_preserves_value(self, qapp):
        f, z, fres, qres, res_idxs = _make_sweep(M=2)
        zg = _make_zarr()
        w = FqFinderWindow(f, z, fres, qres, res_idxs, zg)
        ri0 = w._ri
        new_f = w._fres_init[ri0] + 300e3
        w._set_fres(new_f)
        w._go_next()
        w._go_back()
        # After going back, the working value should still be the saved one
        assert zg["fres_opt"][ri0] == pytest.approx(new_f)
        w.close()
