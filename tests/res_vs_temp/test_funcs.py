import numpy as np
import pytest
from citkid.res_vs_temp import funcs

F0 = 1e9
ALPHA = 0.5
TC = 1.2
FDELTA0 = 1e-5
T = np.linspace(0.15, 0.45, 20)

@pytest.mark.parametrize("gamma", [1, 1 / 2, 1 / 3])
def test_f_vs_T_uses_gamma(gamma):
    f = funcs.f_vs_T(T, F0, ALPHA, TC, FDELTA0, gamma = gamma)
    f_expected = (funcs.f_vs_T_tls(T, F0, FDELTA0)
                  + funcs.f_vs_T_qp(T, F0, ALPHA, TC, gamma = gamma) - F0)
    assert np.allclose(f, f_expected, rtol = 0, atol = 1e-6)

def test_f_vs_T_gamma_changes_qp_shift():
    f1 = funcs.f_vs_T(T, F0, ALPHA, TC, FDELTA0, gamma = 1)
    f3 = funcs.f_vs_T(T, F0, ALPHA, TC, FDELTA0, gamma = 1 / 3)
    f_tls = funcs.f_vs_T_tls(T, F0, FDELTA0)
    # the QP shift scales linearly with gamma
    assert np.allclose(f3 - f_tls, (f1 - f_tls) / 3, rtol = 1e-6)

def test_f_vs_T_qp_falls_with_T():
    f = funcs.f_vs_T_qp(T, F0, ALPHA, TC)
    assert np.all(f < F0)
    assert np.all(np.diff(f) < 0)

def test_Q_vs_T_qp_falls_with_T():
    Q = funcs.Q_vs_T_qp(T, F0, ALPHA, TC, delta_z = 1e-6)
    assert np.all(Q > 0)
    assert np.all(Q < 1e6)
    assert np.all(np.diff(Q) < 0)

@pytest.mark.parametrize("gamma", [1, 1 / 2])
def test_Q_vs_T_qp_loss_scales_with_gamma_alpha(gamma):
    delta = 1 / funcs.Q_vs_T_qp(T, F0, ALPHA, TC, delta_z = 0, gamma = gamma)
    delta_ref = 1 / funcs.Q_vs_T_qp(T, F0, 1, TC, delta_z = 0, gamma = 1)
    assert np.allclose(delta, gamma * ALPHA * delta_ref)

def test_Q_vs_T_falls_at_high_T():
    Q = funcs.Q_vs_T(T, F0, ALPHA, TC, FDELTA0, delta_z = 1e-6)
    assert np.all(Q > 0)
    assert Q[-1] < Q[len(T) // 2]

@pytest.mark.parametrize("sigma", [20., 200.])
def test_fit_f_vs_T_qp_without_errors_scales_perr_by_residuals(sigma):
    from citkid.res_vs_temp.fitter import fit_f_vs_T_qp
    T_fit = np.linspace(0.25, 0.5, 30)
    f = funcs.f_vs_T_qp(T_fit, F0, ALPHA, TC)
    f = f + np.random.default_rng(0).normal(0, sigma, T_fit.size)
    guess = [F0, ALPHA, TC]
    _, _, perr_none = fit_f_vs_T_qp(T_fit, f, guess = guess)[:3]
    _, _, perr_true = fit_f_vs_T_qp(
        T_fit, f, guess = guess, f_err = np.full(T_fit.size, sigma))[:3]
    np.testing.assert_allclose(perr_none, perr_true, rtol = 0.5)

def test_fit_Q_vs_T_qp_weighted_fit_starts_from_unweighted_prefit(
        monkeypatch):
    from citkid.res_vs_temp import fitter
    calls = []
    def recording_curve_fit(*args, **kwargs):
        popt, pcov = fitter_curve_fit(*args, **kwargs)
        calls.append((list(kwargs['p0']), popt))
        return popt, pcov
    fitter_curve_fit = fitter.curve_fit
    monkeypatch.setattr(fitter, 'curve_fit', recording_curve_fit)
    T_fit = np.linspace(0.25, 0.5, 30)
    Q = funcs.Q_vs_T_qp(T_fit, F0, ALPHA, TC, 1e-5)
    Q = Q * (1 + np.random.default_rng(0).normal(0, 1e-3, T_fit.size))
    fitter.fit_Q_vs_T_qp(T_fit, Q, F0, Q_err = Q * 1e-3)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], calls[0][1])
