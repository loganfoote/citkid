"""
Tests for the legacy primecam gain, fit-row, and fitting functions:
``primecam.gain``, ``primecam.data_io.make_fit_row``/``separate_fit_row``,
and ``primecam.fitter.fit_nonlinear_iq_with_gain``.
"""

import numpy as np
import pandas as pd
import pytest

from citkid.res.funcs import nonlinear_iq
from citkid.xcal import gain as xcal_gain
from citkid.primecam.gain import fit_and_remove_gain_phase
from citkid.primecam.data_io import make_fit_row, separate_fit_row
from citkid.primecam.fitter import fit_nonlinear_iq_with_gain

FR, QR = 500e6, 20000
# fr, Qr, amp, phi, a, i0, q0, tau
PARAMS = [FR, QR, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0]
P_AMP = np.array([-1e-14, 2e-7 - 2 * -1e-14 * FR, 0.0])
P_AMP[2] = -3 - np.polyval([P_AMP[0], P_AMP[1], 0], FR)
P_PHASE = np.array([4e-6, 0.3 - 4e-6 * FR])


def gain(f):
    """
    Return the synthetic gain applied to the test data.

    Parameters:
    f (np.array): frequencies in Hz.

    Returns:
    np.array: complex gain.
    """
    return 10 ** (np.polyval(P_AMP, f) / 20) * np.exp(1j * np.polyval(P_PHASE, f))


def make_sweeps(downward_gain = False, n_gain = 400, n_fine = 300):
    """
    Create gain and fine sweeps of one resonator with a known gain.

    Parameters:
    downward_gain (bool): If True, the gain sweep frequencies decrease.
    n_gain (int): number of gain sweep points.
    n_fine (int): number of fine sweep points.

    Returns:
    fgain, zgain, ffine, zfine (np.array): gain and fine sweeps.
    """
    fgain = np.linspace(FR - 50 * FR / QR, FR + 50 * FR / QR, n_gain)
    if downward_gain:
        fgain = fgain[::-1]
    ffine = np.linspace(FR - 3 * FR / QR, FR + 3 * FR / QR, n_fine)
    zgain = gain(fgain) * nonlinear_iq(fgain, *PARAMS, True)
    zfine = gain(ffine) * nonlinear_iq(ffine, *PARAMS, True)
    return fgain, zgain, ffine, zfine


################################################################################
########################### fit_and_remove_gain_phase ##########################
################################################################################

def test_gain_matches_xcal_gain():
    """Test the wrapper gives the xcal.gain fit and removal."""
    fgain, zgain, ffine, zfine = make_sweeps()
    p_amp, p_phase, z_rmvd, (fig, axs) = fit_and_remove_gain_phase(
        fgain, zgain, ffine, zfine, [FR], [QR])

    p_amp_x, p_phase_x, _ = xcal_gain.fit_gain(fgain, zgain, [(FR, FR / QR)])
    np.testing.assert_allclose(p_amp, p_amp_x)
    np.testing.assert_allclose(p_phase, p_phase_x)
    np.testing.assert_allclose(
        z_rmvd, xcal_gain.remove_gain(ffine, zfine, p_amp, p_phase))
    assert fig is None and axs is None


def test_gain_removes_known_gain():
    """Test the removed data is close to the bare resonator response."""
    fgain, zgain, ffine, zfine = make_sweeps()
    _, _, z_rmvd, _ = fit_and_remove_gain_phase(
        fgain, zgain, ffine, zfine, [FR], [QR / 5])
    truth = nonlinear_iq(ffine, *PARAMS, True)
    assert np.max(np.abs(z_rmvd - truth)) < 0.05


def test_gain_independent_of_sweep_direction():
    """Test descending and ascending gain sweeps give the same result."""
    up = fit_and_remove_gain_phase(*make_sweeps(False), [FR], [QR])
    down = fit_and_remove_gain_phase(*make_sweeps(True), [FR], [QR])
    for a, b in zip(up[:3], down[:3]):
        np.testing.assert_allclose(a, b, rtol = 1e-9, atol = 1e-12)


def test_gain_unsorted_and_nan_resonances():
    """Test unsorted resonances are sorted and NaN ones ignored."""
    fgain, zgain, ffine, zfine = make_sweeps()
    fr2 = FR + 20 * FR / QR
    ref = fit_and_remove_gain_phase(
        fgain, zgain, ffine, zfine, [FR, fr2], [QR, QR])
    out = fit_and_remove_gain_phase(
        fgain, zgain, ffine, zfine, [fr2, np.nan, FR], [QR, QR, QR])
    for a, b in zip(ref[:3], out[:3]):
        np.testing.assert_allclose(a, b)


def test_gain_plot(monkeypatch):
    """Test plotq returns the gain fit figure."""
    import matplotlib.pyplot as plt
    fgain, zgain, ffine, zfine = make_sweeps()
    *_, (fig, axs) = fit_and_remove_gain_phase(
        fgain, zgain, ffine, zfine, [FR], [QR], plotq = True)
    assert fig is not None
    assert len(axs) == 2
    plt.close(fig)


@pytest.mark.parametrize("kwargs, match", [
    (dict(zgain = np.ones(3)), 'fgain and zgain'),
    (dict(frs = [FR, FR]), 'frs and Qrs'),
])
def test_gain_invalid_inputs(kwargs, match):
    """Test mismatched input shapes raise ValueError."""
    fgain, zgain, ffine, zfine = make_sweeps()
    args = dict(fgain = fgain, zgain = zgain, ffine = ffine, zfine = zfine,
                frs = [FR], Qrs = [QR])
    args.update(kwargs)
    with pytest.raises(ValueError, match = match):
        fit_and_remove_gain_phase(**args)


################################################################################
################################### fit rows ###################################
################################################################################

def make_row(**kwargs):
    """
    Make a fit row from fixed test values.

    Parameters:
    kwargs: extra arguments for make_fit_row.

    Returns:
    row (pd.Series): fit row.
    inputs (tuple): (p_amp, p_phase, p0, popt, perr, nrmse).
    """
    p0 = np.arange(1., 9.)
    popt = np.array([5e8, 2e4, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0])
    perr = popt * 1e-3
    inputs = (np.array([1., 2., 3.]), np.array([4., 5.]), p0, popt, perr, 0.01)
    return make_fit_row(*inputs, **kwargs), inputs


@pytest.mark.parametrize("downward", [True, False])
def test_fit_row_round_trip(downward):
    """Test separate_fit_row inverts make_fit_row."""
    row, inputs = make_row(downward = downward, plot_path = 'a.png')
    assert isinstance(row, pd.Series)
    assert row['iq_sweep_direction'] == ('downward' if downward else 'upward')
    assert 'iq_Qc' in row and 'iq_Qi' in row
    out = separate_fit_row(row)
    for a, b in zip(out[:5], inputs[:5]):
        np.testing.assert_allclose(a, b)
    assert out[5] == inputs[5]
    assert out[6] is downward
    assert out[7] == 'a.png'


def test_fit_row_floats_only_and_prefix():
    """Test floats_only drops string columns and prefix renames columns."""
    row, _ = make_row(downward = True, prefix = 'x', floats_only = True)
    assert 'x_fr' in row and 'iq_fr' not in row
    assert 'x_sweep_direction' not in row and 'x_plotpath' not in row
    assert not any(isinstance(v, str) for v in row)
    # Every value is a numeric scalar (Qc and Qi are 0-d arrays)
    assert all(np.ndim(v) == 0 and np.isreal(v) for v in row)


################################################################################
########################## fit_nonlinear_iq_with_gain ##########################
################################################################################

def test_fit_with_gain_recovers_parameters():
    """Test the fit recovers fr and Qr after removing the gain."""
    fgain, zgain, ffine, zfine = make_sweeps()
    p_amp, p_phase, p0, popt, perr, nrmse, fig = fit_nonlinear_iq_with_gain(
        fgain, zgain, ffine, zfine, [FR], [QR / 5])
    assert abs(popt[0] - FR) < FR / QR / 20
    assert abs(popt[1] - QR) / QR < 0.1
    assert len(p_amp) == 3 and len(p_phase) == 2
    assert fig is None


def test_fit_with_gain_dataframe():
    """Test return_dataframe returns a fit row that can be separated."""
    fgain, zgain, ffine, zfine = make_sweeps()
    row, fig = fit_nonlinear_iq_with_gain(
        fgain, zgain, ffine, zfine, [FR], [QR / 5], return_dataframe = True)
    assert isinstance(row, pd.Series)
    p_amp, p_phase, p0, popt, perr, nrmse, downward, plot_path = \
        separate_fit_row(row)
    assert downward is True
    assert abs(popt[0] - FR) < FR / QR / 20


def test_fit_with_gain_no_deprecation_warning():
    """Test the primecam version does not warn like the removed res one."""
    import warnings
    fgain, zgain, ffine, zfine = make_sweeps()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine, [FR], [QR / 5])
