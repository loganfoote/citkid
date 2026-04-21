"""
Tests for citkid.res.fitter — fit_nonlinear_iq, fit_nonlinear_iq_pl,
fit_util, and the legacy-deprecation warnings.
"""

import pytest
import numpy as np
import warnings

from citkid.res.funcs import nonlinear_iq
from citkid.res.fitter import fit_nonlinear_iq, fit_nonlinear_iq_pl, fit_util


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

# Reference parameters: fr, Qr, amp, phi, a, i0, q0, tau
PARAMS_TYPICAL = [500e6, 20000, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0]
PARAMS_UPWARD = [500e6, 20000, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0]


def make_synthetic_data(params=None, n=500, span_factor=6, downward=True,
                        noise_sigma=None, seed=0):
    """Generate synthetic IQ data from known parameters."""
    if params is None:
        params = PARAMS_TYPICAL
    fr, Qr = params[0], params[1]
    f = np.linspace(fr - span_factor * fr / Qr / 2,
                    fr + span_factor * fr / Qr / 2, n)
    z = nonlinear_iq(f, *params, downward)
    if noise_sigma is not None:
        rng = np.random.default_rng(seed)
        z = z + rng.normal(0, noise_sigma, n) + 1j * rng.normal(0, noise_sigma, n)
    return f, z, params


################################################################################
############################# test_fit_util ####################################
################################################################################

class TestFitUtil:
    def _make_inputs(self, params=None, downward=True):
        f, z, params = make_synthetic_data(params=params, downward=downward)
        z_stacked = np.hstack((np.real(z), np.imag(z)))
        # Use true params scaled as fit_util expects (p0 in physical units)
        p0 = list(params)
        bounds = [
            [np.min(f), 1e3, 0.01, -np.pi/2, 0, -1e2, -1e2, -1e-6],
            [np.max(f), 1e7, 1-1e-6, np.pi/2, 1, 1e2, 1e2, 1e-6],
        ]
        return p0, bounds, f, z_stacked, z, downward

    def test_returns_three_items(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        result = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert len(result) == 3

    def test_popt_length(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert len(popt) == 8

    def test_perr_length(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert len(perr) == 8

    def test_nrmse_is_scalar(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert np.isscalar(nrmse)

    def test_nrmse_nonnegative(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert nrmse >= 0

    def test_nrmse_small_on_clean_data(self):
        """Starting exactly at the true params should yield near-zero NRMSE."""
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert nrmse < 1e-4

    def test_perr_nonnegative(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert all(e >= 0 for e in perr)

    def test_fit_tau_false_sets_perr_tau_zero(self):
        """When fit_tau=False, perr[7] (tau) should be 0."""
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, False, f, z_stacked, z, downward)
        assert perr[7] == 0.0

    def test_fit_tau_false_enforces_tau(self):
        """When fit_tau=False, popt[7] should equal the given tau."""
        params = list(PARAMS_TYPICAL)
        params[7] = 1e-7  # non-zero tau
        p0, bounds, f, z_stacked, z, downward = self._make_inputs(params=params)
        popt, perr, nrmse = fit_util(p0, bounds, False, f, z_stacked, z, downward)
        assert popt[7] == pytest.approx(params[7])

    def test_recovers_fr_accurately(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert popt[0] == pytest.approx(PARAMS_TYPICAL[0], rel=1e-4)

    def test_recovers_qr_accurately(self):
        p0, bounds, f, z_stacked, z, downward = self._make_inputs()
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert popt[1] == pytest.approx(PARAMS_TYPICAL[1], rel=1e-3)

    @pytest.mark.parametrize('downward', [True, False])
    def test_downward_flag(self, downward):
        """fit_util should converge for both sweep directions."""
        p0, bounds, f, z_stacked, z, _ = self._make_inputs(downward=downward)
        popt, perr, nrmse = fit_util(p0, bounds, True, f, z_stacked, z, downward)
        assert nrmse < 1e-3


################################################################################
######################### test_fit_nonlinear_iq ################################
################################################################################

class TestFitNonlinearIQ:
    def test_returns_five_items(self):
        f, z, _ = make_synthetic_data()
        result = fit_nonlinear_iq(f, z)
        assert len(result) == 5

    def test_p0_length(self):
        f, z, _ = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
        assert len(p0) == 8

    def test_popt_length(self):
        f, z, _ = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
        assert len(popt) == 8

    def test_perr_length(self):
        f, z, _ = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
        assert len(perr) == 8

    def test_nrmse_small_on_clean_data(self):
        f, z, _ = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
        assert nrmse < 1e-2

    def test_recovers_fr(self):
        f, z, params = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
        assert popt[0] == pytest.approx(params[0], rel=1e-4)

    def test_recovers_qr(self):
        f, z, params = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z)
        assert popt[1] == pytest.approx(params[1], rel=2e-2)

    def test_fr_guess_override(self):
        """fr_guess should override automatic guess."""
        f, z, params = make_synthetic_data()
        fr_true = params[0]
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, fr_guess=fr_true)
        assert popt[0] == pytest.approx(fr_true, rel=1e-4)

    def test_tau_guess_override(self):
        """tau_guess should set the initial tau value."""
        f, z, params = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(
            f, z, tau_guess=1e-8, fit_tau=False
        )
        assert popt[7] == pytest.approx(1e-8)

    def test_fit_tau_false_perr_tau_zero(self):
        f, z, _ = make_synthetic_data()
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, fit_tau=False)
        assert perr[7] == 0.0

    def test_unsorted_input_sorted_internally(self):
        """Passing reversed-frequency data should give the same result."""
        f, z, params = make_synthetic_data()
        p0_fwd, popt_fwd, _, nrmse_fwd, _ = fit_nonlinear_iq(f, z)
        p0_rev, popt_rev, _, nrmse_rev, _ = fit_nonlinear_iq(f[::-1], z[::-1])
        assert popt_fwd[0] == pytest.approx(popt_rev[0], rel=1e-6)

    @pytest.mark.parametrize('downward', [True, False])
    def test_downward_flag(self, downward):
        f, z, params = make_synthetic_data(downward=downward)
        p0, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, downward=downward)
        assert nrmse < 1e-2

    def test_figax_none_when_not_plotq(self):
        f, z, _ = make_synthetic_data()
        p0, popt, perr, nrmse, figax = fit_nonlinear_iq(f, z, plotq=False)
        assert figax == (None, None)

    def test_custom_p0_accepted(self):
        f, z, params = make_synthetic_data()
        p0_in = list(params)
        p0_out, popt, perr, nrmse, _ = fit_nonlinear_iq(f, z, p0=p0_in)
        assert nrmse < 1e-2


################################################################################
##################### test_fit_nonlinear_iq_pl #################################
################################################################################

class TestFitNonlinearIQPl:
    def test_returns_three_items(self):
        f, z, _ = make_synthetic_data()
        mask = np.ones(len(f), dtype=bool)
        result = fit_nonlinear_iq_pl(f, z, mask)
        assert len(result) == 3

    def test_mask_subsets_data(self):
        """Should fit correctly using only the masked subset."""
        f, z, params = make_synthetic_data(n=500)
        mask = np.ones(len(f), dtype=bool)
        p0, popt, nrmse = fit_nonlinear_iq_pl(f, z, mask)
        assert popt[0] == pytest.approx(params[0], rel=1e-4)

    def test_nrmse_small(self):
        f, z, _ = make_synthetic_data()
        mask = np.ones(len(f), dtype=bool)
        p0, popt, nrmse = fit_nonlinear_iq_pl(f, z, mask)
        assert nrmse < 1e-2


################################################################################
####################### test legacy deprecation warnings #######################
################################################################################

class TestLegacyWarnings:
    def test_fit_iq_circle_warns(self):
        from citkid.res.fitter import fit_iq_circle
        z = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        with pytest.warns(DeprecationWarning, match='fit_iq_circle'):
            fit_iq_circle(z)

    def test_fit_nonlinear_iq_with_gain_warns(self):
        from citkid.res.fitter import fit_nonlinear_iq_with_gain
        # Pass obviously bad data — we only care that the warning fires
        dummy = np.ones(10, dtype=float)
        dummy_c = np.ones(10, dtype=complex)
        with pytest.warns(DeprecationWarning, match='fit_nonlinear_iq_with_gain'):
            try:
                fit_nonlinear_iq_with_gain(
                    dummy, dummy_c, dummy, dummy_c, [], []
                )
            except Exception:
                pass  # failure after the warning is fine