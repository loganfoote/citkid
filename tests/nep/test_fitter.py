import numpy as np
import pytest
from scipy.constants import h

from citkid.nep.fitter import fit_nep_photon, nep_fit_mask
from citkid.nep.funcs import nep_photon

NU = 150e9


def test_nep_photon_formula():
    power = np.array([1e-12, 4e-12])
    np.testing.assert_allclose(nep_photon(power, 0.5, NU), np.sqrt(2 * h * NU * power / 0.5))
    assert np.isnan(nep_photon(-1.0, 0.5, NU))


def test_fit_recovers_eta_from_noiseless_data():
    power = np.geomspace(1e-13, 1e-10, 12)
    eta, eta_err, mask, n_fit = fit_nep_photon(power, nep_photon(power, 0.42, NU), NU)

    assert eta == pytest.approx(0.42, rel=1e-12)
    assert eta_err == pytest.approx(0.0, abs=1e-12)
    assert mask.all() and n_fit == 12


def test_p_min_excludes_detector_noise_regime():
    power = np.geomspace(1e-14, 1e-10, 9)
    nep = nep_photon(power, 0.5, NU)
    nep[power < 1e-12] *= 5  # low power: detector noise dominates

    eta_all = fit_nep_photon(power, nep, NU)[0]
    eta, _, mask, n_fit = fit_nep_photon(power, nep, NU, p_min=1e-12)

    assert eta == pytest.approx(0.5, rel=1e-12)
    assert eta_all < 0.4
    np.testing.assert_array_equal(mask, power >= 1e-12)
    assert n_fit == int(np.sum(power >= 1e-12))


def test_nan_points_are_ignored():
    power = np.array([1e-12, np.nan, 2e-12, 4e-12, 8e-12])
    nep = nep_photon(np.nan_to_num(power, nan=1e-12), 0.3, NU)
    nep[3] = np.nan

    eta, _, mask, n_fit = fit_nep_photon(power, nep, NU)

    assert eta == pytest.approx(0.3)
    np.testing.assert_array_equal(mask, [True, False, True, False, True])
    assert n_fit == 3


@pytest.mark.parametrize('power, nep, p_min', [
    (np.full(4, np.nan), np.full(4, np.nan), None),        # all NaN
    (np.geomspace(1e-13, 1e-12, 4), np.ones(4), 1e-11),    # all below p_min
    (np.array([]), np.array([]), None),                    # empty
])
def test_no_usable_points_gives_nan(power, nep, p_min):
    eta, eta_err, mask, n_fit = fit_nep_photon(power, nep, NU, p_min=p_min)

    assert np.isnan(eta) and np.isnan(eta_err)
    assert n_fit == 0 and not mask.any()


def test_single_point_has_eta_but_no_scatter_error():
    eta, eta_err, _, n_fit = fit_nep_photon([1e-12], nep_photon([1e-12], 0.6, NU), NU)

    assert eta == pytest.approx(0.6) and np.isnan(eta_err) and n_fit == 1


def test_unweighted_error_is_scatter_in_log_space():
    power = np.geomspace(1e-12, 1e-10, 4)
    factors = np.array([1.1, 0.9, 1.05, 0.95])  # fractional NEP scatter
    nep = nep_photon(power, 0.5, NU) * factors

    eta, eta_err, _, _ = fit_nep_photon(power, nep, NU)

    log_eta = np.log(0.5) - 2 * np.log(factors)
    assert eta == pytest.approx(np.exp(log_eta.mean()))
    assert eta_err == pytest.approx(eta * log_eta.std(ddof=1) / 2)


def test_nep_err_weights_points_by_fractional_uncertainty():
    power = np.array([1e-12, 1e-11])
    nep = nep_photon(power, 0.5, NU) * np.array([1.0, 1.2])
    frac = np.array([0.01, 0.1])  # the first point is 10x more precise

    eta, eta_err, _, _ = fit_nep_photon(power, nep, NU, nep_err=frac * nep)

    log_eta = np.log(0.5) - 2 * np.log([1.0, 1.2])
    weights = 1 / (2 * frac) ** 2
    assert eta == pytest.approx(np.exp(np.sum(weights * log_eta) / weights.sum()))
    assert eta_err == pytest.approx(eta / np.sqrt(weights.sum()))
    # Equal fractional uncertainties give the unweighted estimate.
    assert fit_nep_photon(power, nep, NU, nep_err=0.05 * nep)[0] == pytest.approx(
        fit_nep_photon(power, nep, NU)[0])


def test_nep_fit_mask_drops_bad_uncertainties():
    mask = nep_fit_mask([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], nep_err=[0.1, np.nan, 0.0])
    np.testing.assert_array_equal(mask, [True, False, False])

