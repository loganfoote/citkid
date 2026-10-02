import numpy as np
import pytest
import zarr
from unittest.mock import patch

import citkid.nep.interactive as inep
from citkid.nep.funcs import nep_photon

NU = 150e9
POWER = np.geomspace(1e-14, 1e-10, 9)


def _data():
    """
    Build three sets: detector noise below 1e-12 W in set 0, all NaN set 2.

    Returns:
    powers, neps (list of np.ndarray): per-set data.
    """
    nep0 = nep_photon(POWER, 0.5, NU)
    nep0[POWER < 1e-12] *= 5
    return ([POWER, POWER, np.full(9, np.nan)],
            [nep0, nep_photon(POWER, 0.3, NU), np.full(9, np.nan)])


def _window(qt_app, **kwargs):
    """
    Build an NEPFitWindow over ``_data``.

    Parameters:
    qt_app (QApplication): Qt application fixture.
    **kwargs: extra NEPFitWindow arguments.

    Returns:
    win (NEPFitWindow): the window (not shown).
    """
    powers, neps = _data()
    return inep.NEPFitWindow(powers, neps, NU, names=['a', 'b', 'c'], **kwargs)


def _drawn(item):
    """
    Return the number of points drawn by a PlotDataItem.
    """
    x, _ = item.getData()
    return 0 if x is None else len(x)


def test_initial_fits_and_all_nan_set(qt_app):
    win = _window(qt_app)

    p_min, eta, eta_err, n_fit, bad = win.results
    assert all(isinstance(a, np.ndarray) for a in (p_min, eta, eta_err, n_fit, bad))
    assert eta[1] == pytest.approx(0.3)
    assert eta[0] < 0.4                                 # p_min = 0 includes detector noise
    assert np.isnan(eta[2]) and n_fit[2] == 0
    assert 'set 1/3 (a)' in win._plot.titleLabel.text
    win.close()


def test_p_min_spin_and_line_refit(qt_app):
    win = _window(qt_app)

    win._p_min_spin.setValue(1e-12)
    win._p_min_spin.sigValueChanged.emit(win._p_min_spin)  # skip the spin box delay

    assert win.results[1][0] == pytest.approx(0.5)
    assert _drawn(win._fitted) == int(np.sum(POWER >= 1e-12))
    assert _drawn(win._excluded) == int(np.sum(POWER < 1e-12))
    assert win._line.value() == pytest.approx(-12)

    win._line.setValue(np.log10(1e-11))                 # as if dragged
    assert win._p_min[0] == pytest.approx(1e-11)
    assert win._p_min_spin.value() == pytest.approx(1e-11)
    assert win.results[3][0] == int(np.sum(POWER >= 1e-11 * (1 - 1e-12)))
    win.close()


def test_navigation_carries_p_min_to_unvisited_sets(qt_app):
    win = _window(qt_app)
    win._set_p_min(1e-12)

    win._go(1)

    assert win._idx == 1 and win._viewed[0]
    assert win._p_min[1] == pytest.approx(1e-12)
    win._set_p_min(0.0)
    win._go(-1)
    win._go(1)
    assert win._p_min[1] == 0.0                         # visited: kept
    win.close()


def test_per_set_p_min_is_not_carried_over(qt_app):
    win = _window(qt_app, p_min=[1e-12, 0.0, 0.0])
    win._go(1)
    assert win._p_min[1] == 0.0
    win.close()


def test_mark_bad_and_apply_to_all(qt_app):
    win = _window(qt_app)

    win._toggle_bad()
    assert np.isnan(win.results[1][0]) and win.results[4][0]
    assert _drawn(win._fitted) == 0 and _drawn(win._curve) == 0
    assert 'marked bad' in win._plot.titleLabel.text
    win._toggle_bad()
    assert np.isfinite(win.results[1][0])

    win._set_p_min(1e-12)
    win._apply_p_min_to_all()
    p_min, eta, _, _, _ = win.results
    np.testing.assert_allclose(p_min, 1e-12)
    np.testing.assert_allclose(eta[:2], [0.5, 0.3])
    win.close()


def test_all_nan_set_hides_line(qt_app):
    win = _window(qt_app, start_idx=2)
    assert not win._line.isVisible() and _drawn(win._fitted) == 0
    assert 'no usable points' in win._plot.titleLabel.text
    win.close()


def test_saves_and_resumes_from_group(qt_app):
    group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    win = _window(qt_app, group=group)
    win._set_p_min(1e-12)
    win._go(1)
    win._toggle_bad()
    win.close()                                         # set 1 open: viewed

    resumed = _window(qt_app, group=group)

    assert resumed._idx == 2                            # first set not yet viewed
    assert resumed._p_min[0] == pytest.approx(1e-12)
    assert resumed.results[1][0] == pytest.approx(0.5)
    assert resumed.results[4][1]
    np.testing.assert_allclose(np.asarray(group['eta'][...])[0], 0.5)
    resumed.close()


@pytest.mark.parametrize('overwrite', [True, False])
def test_different_saved_data_asks_before_overwriting(qt_app, overwrite):
    group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    win = _window(qt_app, group=group)
    win.close()
    powers, neps = _data()

    with patch.object(inep, '_confirm_overwrite_nep_fits', return_value=overwrite) as ask:
        if overwrite:
            win = inep.NEPFitWindow(powers, neps, 2 * NU, group=group)
            assert win._idx == 0
            win.close()
        else:
            with pytest.raises(RuntimeError, match='User cancelled operation'):
                inep.NEPFitWindow(powers, neps, 2 * NU, group=group)
    ask.assert_called_once()


def test_bad_inputs_raise(qt_app):
    with pytest.raises(ValueError, match='at least one set'):
        inep.NEPFitWindow([], [], NU)
    with pytest.raises(ValueError, match='same shape'):
        inep.NEPFitWindow([np.ones(3)], [np.ones(2)], NU)
