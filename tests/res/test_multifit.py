import pytest
import numpy as np
from citkid.res import multifit
from citkid.res.funcs import product_s21

def _window(rng, fr, Qr, amp, phi, sigma = 0., a = None, n = 1500,
            span = 4e5, ext = None):
    """Simulate a window of S21 with a smooth gain (incl. cable delay)."""
    fc = np.mean(fr)
    f = np.linspace(fc - span / 2, fc + span / 2, n)
    x = (f - fc) / (span / 2)
    gain = (0.8 - 0.3j) * (1 + 0.05 * x - 0.02 * x ** 2) \
        * np.exp(-2j * np.pi * (f - fc) * 30e-9)
    z = gain * product_s21(f, fr, Qr, amp, phi, a)
    if ext is not None:
        z = z * product_s21(f, *ext)
    z = z + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return f, z

################################################################################
############################ grouping and windows ##############################
################################################################################
def test_group_resonances():
    fr = np.array([1e9, 1e9 + 1e5, 1e9 + 1e6, 1.002e9])
    Qr = np.full(4, 2e4)  # linewidth 5e4 Hz; 8 lw windows: +-4e5 Hz
    groups = multifit.group_resonances(fr, Qr, 8)
    assert [list(g) for g in groups] == [[0, 1], [2], [3]]

def test_window_mask_min_points():
    f = np.linspace(1e9, 1.001e9, 1001)
    m = multifit.window_mask(f, [1.0005e9], [1e9], 1, min_points = 20)
    assert m.sum() == 20 and m[500]

@pytest.mark.parametrize("span,deg", [(1e6, 2), (3e6, 3), (1e9, 5)])
def test_gain_degree(span, deg):
    assert multifit.gain_degree(span) == deg

################################################################################
################################## fit_poles ###################################
################################################################################
@pytest.mark.parametrize("sigma", [0., 1e-3])
def test_fit_poles_two_resonators(sigma):
    rng = np.random.default_rng(1)
    fr = np.array([1e9 - 4e4, 1e9 + 3e4])
    Qr, amp, phi = np.array([3e4, 5e4]), np.array([0.6, 0.4]), \
        np.array([0.2, -0.3])
    f, z = _window(rng, fr, Qr, amp, phi, sigma)
    res = multifit.fit_poles(f, z, fr + 5e3, gain_deg = 3)
    tol = 1e-6 if sigma == 0 else 1e-2
    np.testing.assert_allclose(res['fr'], fr, rtol = tol * 1e-3)
    np.testing.assert_allclose(res['Qr'], Qr, rtol = 10 * tol)
    np.testing.assert_allclose(res['amp'], amp, atol = 10 * tol)
    np.testing.assert_allclose(res['phi'], phi, atol = 10 * tol)
    assert res['success'] and not res['at_bound'].any()
    if sigma == 0:
        np.testing.assert_allclose(multifit.multifit_model(f, res), z,
                                   atol = 1e-4)
    else:
        assert abs(res['sigma'] / sigma - 1) < 0.1

def test_fit_poles_external_resonance():
    rng = np.random.default_rng(2)
    ext = ([1e9 + 2.6e5], [2e4], [0.8], [0.])
    f, z = _window(rng, [1e9], [3e4], [0.5], [0.1], 1e-4, ext = ext)
    with_ext = multifit.fit_poles(f, z, [1e9], gain_deg = 3, ext_fr = ext[0],
                                  ext_Qr = ext[1], ext_amp = ext[2],
                                  ext_phi = ext[3])
    without = multifit.fit_poles(f, z, [1e9], gain_deg = 3)
    assert with_ext['rss'] < 0.5 * without['rss']
    assert abs(with_ext['Qr'][0] / 3e4 - 1) < 0.01

################################################################################
################################### init_qr ####################################
################################################################################
@pytest.mark.parametrize("Qr", [5e3, 3e4, 2e5])
def test_init_qr(Qr):
    rng = np.random.default_rng(3)
    f, z = _window(rng, [1e9], [Qr], [0.5], [0.1], 1e-3, n = 20001,
                   span = 4e6)
    fr, Qr_fit, snr = multifit.init_qr(f, z, 1e9 + 0.3e9 / Qr / 1e3)
    assert abs(fr - 1e9) < 0.1 * 1e9 / Qr
    assert 0.7 < Qr_fit / Qr < 1.4
    assert snr > 50

################################################################################
################################ fit_resonators ################################
################################################################################
def test_fit_resonators_linear():
    rng = np.random.default_rng(4)
    fr = np.array([1e9 - 4e4, 1e9 + 3e4])
    Qr, amp, phi = np.array([3e4, 5e4]), np.array([0.6, 0.4]), \
        np.array([0.2, -0.3])
    f, z = _window(rng, fr, Qr, amp, phi, 1e-3)
    a = multifit.fit_poles(f, z, fr, gain_deg = 3)
    res = multifit.fit_resonators(f, z, a['fr'], a['Qr'], a['amp'], a['phi'],
                                  gain_deg = 3)
    for k, v in (('fr', fr), ('Qr', Qr), ('amp', amp), ('phi', phi)):
        assert np.all(np.abs(res[k] - v) < 5 * res[k + '_err'])
    assert np.all(res['a'] == 0) and res['success']

@pytest.mark.parametrize("a_true", [0.3, 0.7])
def test_fit_resonators_nonlinear(a_true):
    rng = np.random.default_rng(5)
    f, z = _window(rng, [1e9], [5e4], [0.6], [0.1], 1e-3, a = [a_true],
                   span = 2e5)
    p = multifit.fit_poles(f, z, [1e9], gain_deg = 3)
    res = multifit.fit_resonators(f, z, p['fr'], p['Qr'], p['amp'], p['phi'],
                                  gain_deg = 3, fit_nonlinear = True)
    assert abs(res['a'][0] - a_true) < 0.03
    assert abs(res['fr'][0] - 1e9) < 0.05 * 1e9 / 5e4

def test_fit_resonators_bifurcated():
    # Above bifurcation (~0.77) the fitted a is above it too
    rng = np.random.default_rng(6)
    f, z = _window(rng, [1e9], [5e4], [0.6], [0.1], 1e-3, a = [0.9],
                   span = 2e5)
    p = multifit.fit_poles(f, z, [1e9], gain_deg = 3)
    res = multifit.fit_resonators(f, z, p['fr'], p['Qr'], p['amp'], p['phi'],
                                  gain_deg = 3, fit_nonlinear = True)
    assert res['a'][0] > 4 * np.sqrt(3) / 9

@pytest.mark.parametrize("double", [False, True])
def test_compare_single_double(double):
    rng = np.random.default_rng(8)
    lw = 1e9 / 4e4
    if double:  # two resonances 0.8 linewidths apart
        fr, Qr, amp = 1e9 + np.array([-0.4, 0.4]) * lw, [4e4] * 2, [0.4] * 2
    else:
        fr, Qr, amp = [1e9], [4e4], [0.6]
    f, z = _window(rng, fr, Qr, amp, np.zeros(len(fr)), 1e-4, span = 16 * lw)
    r = multifit.compare_single_double(f, z, 1e9, 4e4)
    assert (r['dbic'] > 1000) == double

def test_add_and_remove_poles():
    rng = np.random.default_rng(9)
    fr = np.array([1e9 - 4e4, 1e9 + 4e4])
    f, z = _window(rng, fr, [3e4, 3e4], [0.6, 0.5], [0., 0.], 1e-3)
    a = multifit.fit_poles(f, z, [fr[0]], gain_deg = 3)
    a = multifit.add_missing_poles(f, z, a, gain_deg = 3)
    assert a['n_added'] == 1 and abs(a['fr'][-1] - fr[1]) < 1e3
    # A duplicate pole held next to the first one adds nothing
    fb = [[np.nan, np.nan], [fr[0] - 1e4, fr[0] + 1e4], [np.nan, np.nan]]
    dup = multifit.fit_poles(f, z, [fr[0], fr[0] + 300, fr[1]], gain_deg = 3,
                             fr_bounds = fb)
    r = multifit.remove_redundant_poles(f, z, dup, gain_deg = 3)
    assert len(r['fr']) == 2 and len(r['kept']) == 2

def test_fit_resonators_upward():
    rng = np.random.default_rng(7)
    fc = 1e9
    f = np.linspace(fc - 1e5, fc + 1e5, 1500)
    z = 0.8 * product_s21(f, [fc], [5e4], [0.6], [0.1], [0.9],
                          downward = False)
    z = z + 1e-3 * (rng.standard_normal(1500) + 1j * rng.standard_normal(1500))
    p = multifit.fit_poles(f, z, [fc], gain_deg = 2)
    up = multifit.fit_resonators(f, z, p['fr'], p['Qr'], p['amp'], p['phi'],
                                 fit_nonlinear = True, downward = False)
    down = multifit.fit_resonators(f, z, p['fr'], p['Qr'], p['amp'],
                                   p['phi'], fit_nonlinear = True)
    # Below bifurcation both directions give the same S21; above it they
    # differ
    assert abs(up['a'][0] - 0.9) < 0.03 and up['rss'] < 0.5 * down['rss']
    assert not up['downward'] and down['downward']
