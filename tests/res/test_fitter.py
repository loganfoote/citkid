"""
Tests for citkid.res.fitter — fit_nonlinear_iq, fit_util, and the
legacy-deprecation warnings.
"""

import pytest
import numpy as np

from citkid.res.funcs import nonlinear_iq
from citkid.res.fitter import fit_nonlinear_iq, fit_util


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

# Reference parameters: fr, Qr, amp, phi, a, i0, q0, tau
PARAMS_TYPICAL = [500e6, 20000, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0]
PARAMS_UPWARD = [500e6, 20000, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0]


def make_synthetic_data(
    params=None, n=500, span_factor=6, downward=True, noise_sigma=None, seed=0
):
    """Generate synthetic IQ data from known parameters."""
    if params is None:
        params = PARAMS_TYPICAL
    fr, Qr = params[0], params[1]
    f = np.linspace(fr - span_factor * fr / Qr / 2,
                    fr + span_factor * fr / Qr / 2, n)
    z = nonlinear_iq(f, *params, downward)
    if noise_sigma is not None:
        rng = np.random.default_rng(seed)
        z = (z + rng.normal(0, noise_sigma, n)
             + 1j * rng.normal(0, noise_sigma, n))
    return f, z, params


################################################################################
############################# test_fit_util ####################################
################################################################################

def _make_inputs(params=None, downward=True):
    f, z, params = make_synthetic_data(params=params, downward=downward)
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # Use true params scaled as fit_util expects (p0 in physical units)
    p0 = list(params)
    bounds = [
        [np.min(f), 1e3, 0.01, -np.pi/2, 0, -1e2, -1e2, -1e-6],
        [np.max(f), 1e7, 1-1e-6, np.pi/2, 1, 1e2, 1e2, 1e-6],
    ]
    return p0, bounds, f, z_stacked, z, downward


def test_fit_util_returns_three_items():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    result = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert len(result) == 3


def test_fit_util_popt_length():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert len(popt) == 8


def test_fit_util_perr_length():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert len(perr) == 8


def test_fit_util_nrmse_is_scalar():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert np.isscalar(nrmse)


def test_fit_util_nrmse_nonnegative():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert nrmse >= 0


def test_fit_util_nrmse_small_on_clean_data():
    """Starting exactly at the true params should yield near-zero NRMSE."""
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert nrmse < 1e-4


def test_fit_util_perr_nonnegative():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert all(e >= 0 for e in perr)


def test_fit_util_fit_tau_false_sets_perr_tau_zero():
    """When fit_tau=False, perr[7] (tau) should be 0."""
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, False, f, z_stacked, z, downward)
    assert perr[7] == 0.0


def test_fit_util_fit_tau_false_enforces_tau():
    """When fit_tau=False, popt[7] should equal the given tau."""
    params = list(PARAMS_TYPICAL)
    params[7] = 1e-7  # non-zero tau
    p0, bounds, f, z_stacked, z, downward = _make_inputs(params=params)
    popt, perr, nrmse = fit_util(p0, bounds, False, f, z_stacked, z, downward)
    assert popt[7] == pytest.approx(params[7])


def test_fit_util_recovers_fr_accurately():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert popt[0] == pytest.approx(PARAMS_TYPICAL[0], rel=1e-4)


def test_fit_util_recovers_qr_accurately():
    p0, bounds, f, z_stacked, z, downward = _make_inputs()
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert popt[1] == pytest.approx(PARAMS_TYPICAL[1], rel=1e-3)


@pytest.mark.parametrize('downward', [True, False])
def test_fit_util_downward_flag(downward):
    """fit_util should converge for both sweep directions."""
    p0, bounds, f, z_stacked, z, _ = _make_inputs(downward=downward)
    popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
    assert nrmse < 1e-3


################################################################################
######################### test_fit_nonlinear_iq ################################
################################################################################

def test_fit_nonlinear_iq_returns_five_items():
    f, z, _ = make_synthetic_data()
    result = fit_nonlinear_iq(f, z)
    assert len(result) == 5


def test_fit_nonlinear_iq_p0_length():
    f, z, _ = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
    assert len(p0) == 8


def test_fit_nonlinear_iq_popt_length():
    f, z, _ = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
    assert len(popt) == 8


def test_fit_nonlinear_iq_perr_length():
    f, z, _ = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
    assert len(perr) == 8


def test_fit_nonlinear_iq_nrmse_small_on_clean_data():
    f, z, _ = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
    assert nrmse < 1e-2


def test_fit_nonlinear_iq_recovers_fr():
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
    assert popt[0] == pytest.approx(params[0], rel=1e-4)


def test_fit_nonlinear_iq_recovers_qr():
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
    assert popt[1] == pytest.approx(params[1], rel=2e-2)


def test_fit_nonlinear_iq_fr_guess_override():
    """fr_guess should override automatic guess."""
    f, z, params = make_synthetic_data()
    fr_true = params[0]
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, fr_guess=fr_true)
    assert popt[0] == pytest.approx(fr_true, rel=1e-4)


def test_fit_nonlinear_iq_tau_guess_override():
    """tau_guess should set the initial tau value."""
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(
        f, z, tau_guess=1e-8, fit_tau=False
    )
    assert popt[7] == pytest.approx(1e-8)


def test_fit_nonlinear_iq_fit_tau_false_perr_tau_zero():
    f, z, _ = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, fit_tau=False)
    assert perr[7] == 0.0


def test_fit_nonlinear_iq_unsorted_input_sorted_internally():
    """Passing reversed-frequency data should give the same result."""
    f, z, params = make_synthetic_data()
    p0_fwd, popt_fwd, _, nrmse_fwd, _ = fit_nonlinear_iq(f, z)
    p0_rev, popt_rev, _, nrmse_rev, _ = fit_nonlinear_iq(f[::-1], z[::-1])
    assert popt_fwd[0] == pytest.approx(popt_rev[0], rel=1e-6)


@pytest.mark.parametrize('downward', [True, False])
def test_fit_nonlinear_iq_downward_flag(downward):
    f, z, params = make_synthetic_data(downward=downward)
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, downward=downward)
    assert nrmse < 1e-2


def test_fit_nonlinear_iq_figax_none_when_not_plotq():
    f, z, _ = make_synthetic_data()
    p0, popt, perr, nrmse, figax = fit_nonlinear_iq(f, z, plotq=False)
    assert figax == (None, None)


def test_fit_nonlinear_iq_custom_p0_accepted():
    f, z, params = make_synthetic_data()
    p0_in = list(params)
    p0_out, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, p0=p0_in)
    assert nrmse < 1e-2


######################## default start and bounds ##############################
def test_fit_nonlinear_iq_does_not_modify_p0():
    f, z, params = make_synthetic_data()
    p0 = list(params)
    p0[2] = 0.3
    before = list(p0)
    fit_nonlinear_iq(f, z, p0 = p0, fr_guess = params[0] + 1e3)
    assert p0 == before

@pytest.mark.parametrize("gain", [0.05 + 1j, -1 + 0.05j, 0.7 - 0.7j])
def test_fit_nonlinear_iq_gain_near_axis(gain):
    # Default bounds let i0 and q0 change sign from the guess
    params = [1e9, 3e4, 0.6, 0.3, 0.65, gain.real, gain.imag, 1e-8]
    f, z, _ = make_synthetic_data(params = params, noise_sigma = 1e-3)
    _, popt, _, _, _ = fit_nonlinear_iq(f, z)
    lw = params[0] / params[1]
    assert abs(popt[0] - params[0]) < 0.05 * lw
    assert abs(popt[4] - params[4]) < 0.05

def test_fit_nonlinear_iq_bifurcated():
    params = [1e9, 3e4, 0.6, 0.1, 0.95, 1.0, 0.2, 0.]
    f, z, _ = make_synthetic_data(params = params, noise_sigma = 1e-3)
    _, popt, _, _, _ = fit_nonlinear_iq(f, z)
    assert popt[4] > 4 * np.sqrt(3) / 9


################################################################################
###################### fit_nonlinear_iq retry paths ############################
################################################################################

def _fake_fit_util(monkeypatch, nrmses):
    """
    Replace ``fit_util`` in ``citkid.res.fitter`` with a fake that returns
    the given nrmse values in order and records each call.

    Parameters:
    monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.
    nrmses (list of float): nrmse value returned by each successive call.

    Returns:
    calls (list of dict): filled with copies of ``p0`` and ``bounds`` for
        each call, and the returned ``popt``.
    """
    import citkid.res.fitter as fitter_mod
    calls = []

    def fake(p0, bounds, fit_tau, f, z_stacked, z, downward = True):
        """
        Record the inputs and return a distinct popt per call.

        Parameters:
        p0, bounds, fit_tau, f, z_stacked, z, downward: as ``fit_util``.

        Returns:
        popt (list), perr (list), nrmse (float): fake fit results.
        """
        i = len(calls)
        popt = [float(p) for p in p0]
        popt[0] += 1. + i  # distinct, recognizable fr per call
        calls.append(dict(p0 = np.array(p0, dtype = float),
                          bounds = np.array(bounds, dtype = float),
                          popt = popt))
        return popt, [0.1 * (i + 1)] * 8, nrmses[i]
    monkeypatch.setattr(fitter_mod, 'fit_util', fake)
    return calls


def test_fit_nonlinear_iq_no_retry_when_nrmse_small(monkeypatch):
    calls = _fake_fit_util(monkeypatch, [1e-3])
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, p0 = params)
    assert len(calls) == 1
    assert nrmse == 1e-3
    assert popt == calls[0]['popt']


def test_fit_nonlinear_iq_retry_from_previous_popt(monkeypatch):
    # 1e-2 < nrmse < 1e-1: refit starting from the previous popt
    calls = _fake_fit_util(monkeypatch, [5e-2, 1e-3])
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, p0 = params)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1]['p0'], calls[0]['popt'])
    # bounds are unchanged on this path
    np.testing.assert_array_equal(calls[1]['bounds'], calls[0]['bounds'])
    assert popt == calls[1]['popt'] and nrmse == 1e-3
    assert perr == [0.2] * 8
    # the returned p0 is the start of the last fit
    np.testing.assert_array_equal(p0, calls[0]['popt'])


def test_fit_nonlinear_iq_retry_divides_amp(monkeypatch):
    # nrmse > 1e-1: amp guess and its bounds are divided by 10
    calls = _fake_fit_util(monkeypatch, [0.5, 1e-3])
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, p0 = params)
    assert len(calls) == 2
    p0_0, p0_1 = calls[0]['p0'], calls[1]['p0']
    b0, b1 = calls[0]['bounds'], calls[1]['bounds']
    assert p0_1[2] == pytest.approx(p0_0[2] / 10)
    np.testing.assert_allclose(b1[:, 2], b0[:, 2] / 10)
    # everything else is unchanged
    keep = [0, 1, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(p0_1[keep], p0_0[keep])
    np.testing.assert_array_equal(b1[:, keep], b0[:, keep])
    assert popt == calls[1]['popt'] and nrmse == 1e-3
    assert p0[2] == pytest.approx(params[2] / 10)


def test_fit_nonlinear_iq_retry_stops_after_three_fits(monkeypatch):
    # Bad fits are retried at most twice; the last result is returned
    calls = _fake_fit_util(monkeypatch, [0.5, 0.5, 0.5, 0.5])
    f, z, params = make_synthetic_data()
    p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, p0 = params)
    assert len(calls) == 3
    assert calls[2]['p0'][2] == pytest.approx(params[2] / 100)
    assert popt == calls[2]['popt'] and nrmse == 0.5


def test_fit_nonlinear_iq_retry_mixed_paths(monkeypatch):
    # Amp division, then a refit from the previous popt
    calls = _fake_fit_util(monkeypatch, [0.5, 5e-2, 1e-3])
    f, z, params = make_synthetic_data()
    _, popt, _, nrmse, _ = fit_nonlinear_iq(f, z, p0 = params)
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[2]['p0'], calls[1]['popt'])
    assert calls[1]['p0'][2] == pytest.approx(params[2] / 10)
    assert popt == calls[2]['popt'] and nrmse == 1e-3


def test_fit_nonlinear_iq_retry_real_fit_with_bad_amp_guess():
    # A too-high amp guess should still converge to the true resonance
    f, z, params = make_synthetic_data(params = [500e6, 20000, 0.05, 0.1,
                                                 0.05, 1.0, 0.0, 0.0])
    p0 = list(params)
    p0[2] = 0.9
    _, popt, _, nrmse, _ = fit_nonlinear_iq(f, z, p0 = p0)
    assert nrmse < 1e-2
    assert popt[0] == pytest.approx(500e6, rel = 1e-7)
    assert popt[2] == pytest.approx(0.05, rel = 1e-2)


def test_fit_nonlinear_iq_plotq_returns_figure():
    import matplotlib.pyplot as plt
    f, z, params = make_synthetic_data()
    _, popt, _, _, (fig, ax) = fit_nonlinear_iq(f, z, p0 = params,
                                                plotq = True)
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    plt.close(fig)
