import numpy as np
import pytest
from citkid.responsivity.funcs import responsivity_int_x0
from citkid.responsivity.fitter import fit_responsivity_int_x0

def _data(sigma, seed = 0):
    power = np.geomspace(1e-17, 1e-13, 25)
    x = responsivity_int_x0(power, -5e8, 1e-14, 0.)
    x = x + np.random.default_rng(seed).normal(0, sigma, power.size)
    return power, x

@pytest.mark.parametrize("sigma", [1e-7, 1e-6])
def test_fit_without_errors_scales_perr_by_residuals(sigma):
    power, x = _data(sigma)
    _, _, perr_none, _, _ = fit_responsivity_int_x0(power, x, 5e8)
    _, _, perr_true, _, _ = fit_responsivity_int_x0(
        power, x, 5e8, x_err = np.full(power.size, sigma))
    # Without errors, perr comes from the residual scatter, so it matches
    # the fit with the true errors to within the scatter of the estimate
    np.testing.assert_allclose(perr_none, perr_true, rtol = 0.5)
