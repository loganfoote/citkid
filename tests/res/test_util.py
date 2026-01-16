import pytest 
import numpy as np 
from citkid.res import util 

################################################################################
################################## calc_qc_qi ##################################
################################################################################
@pytest.mark.parametrize("qc,qi", [
    (1, 1),
    (1e4, 1e3),
    (5e5, 2e4),
    (12345.67, 8901.23),
    (1e6, 1e2),
    (1, 1e9), 
    (np.array([1e3, 1e2]), np.array([1e2, 1e3])),
])
def test_calc_qc_qi(qc, qi):
    qr = 1.0 / (1.0 / qc + 1.0 / qi)
    amp = qr / qc
    qc_calc, qi_calc = util.calc_qc_qi(qr, amp)
    assert pytest.approx(qc_calc) == qc
    assert pytest.approx(qi_calc) == qi

def test_calc_qc_qi_edge_cases():
    # amp = 0 case
    qr = 1e4
    amp = 0.0
    qc_calc, qi_calc = util.calc_qc_qi(qr, amp)
    assert qc_calc == np.inf
    assert qi_calc == qr
    # amp = 1 case
    qr = 1e4
    amp = 1.0
    qc_calc, qi_calc = util.calc_qc_qi(qr, amp)
    assert qc_calc == qr
    assert qi_calc == np.inf

def test_calc_qc_qi_array_input():
    qr = np.array([1e3, 2e4, 5e5])
    amp = np.array([0.1, 0.3, 0.8])
    qc_expected = qr / amp
    qi_expected = 1.0 / ((1.0 / qr) - (1.0 / qc_expected))
    for dtype in [int, float, np.float32, np.float64, np.int32]:
        qc_calc, qi_calc = util.calc_qc_qi(qr.astype(dtype), amp.astype(dtype))
        if dtype not in [int, np.int32]: # int conversion changes amp value
            assert np.allclose(qc_calc, qc_expected)
            assert np.allclose(qi_calc, qi_expected)

def test_calc_qc_qi_broadcast_scalar_amp():
    qr = np.array([1e3, 2e4, 5e5])
    amp = 0.2
    qc_expected = qr / amp
    qi_expected = 1.0 / ((1.0 / qr) - (1.0 / qc_expected))
    qc_calc, qi_calc = util.calc_qc_qi(qr, amp)
    assert np.allclose(qc_calc, qc_expected)
    assert np.allclose(qi_calc, qi_expected)

def test_calc_qc_qi_broadcast_scalar_qr():
    qr = 1e4
    amp = np.array([0.1, 0.5, 0.9])
    qc_expected = qr / amp
    qi_expected = 1.0 / ((1.0 / qr) - (1.0 / qc_expected))
    qc_calc, qi_calc = util.calc_qc_qi(qr, amp)
    assert np.allclose(qc_calc, qc_expected)
    assert np.allclose(qi_calc, qi_expected)

def test_calc_qc_qi_near_endpoints():
    qr = 1e4
    eps = 1e-9
    qc_small, qi_small = util.calc_qc_qi(qr, eps)
    assert np.isfinite(qc_small) and qc_small > qr
    assert np.isclose(qi_small, qr, rtol=1e-6)
    qc_near_one, qi_near_one = util.calc_qc_qi(qr, 1 - eps)
    assert np.isclose(qc_near_one, qr / (1 - eps), rtol=1e-6)
    assert np.isclose(qi_near_one, qr / eps, rtol=1e-6)

@pytest.mark.parametrize("qr,amp,qc_expected,qi_expected", [
    (np.nan, 0.5, np.nan, np.nan),
    (1e4, np.nan, np.nan, np.nan),
])
def test_calc_qc_qi_nonfinite(qr, amp, qc_expected, qi_expected):
    qc_calc, qi_calc = util.calc_qc_qi(qr, amp)
    if np.isnan(qc_expected):
        assert np.isnan(qc_calc)
    else:
        assert np.isinf(qc_calc)
    if np.isnan(qi_expected):
        assert np.isnan(qi_calc)
    else:
        assert np.isinf(qi_calc)

def test_calc_qc_qi_inf_raises():
    with pytest.raises(ZeroDivisionError):
        util.calc_qc_qi(np.inf, 0.5)

@pytest.mark.parametrize("qr,amp", [
    (1, -0.1),
    (1, 1.1),
    (0, 0.5),
    (-1, 0.5),
    ("1000", 0.5),
    (1000, "0.5"),
    (1 + 0j, 0.5),
    (1, 1 + 0j)
])
def test_calc_qc_qi_invalid(qr, amp):
    with pytest.raises((ValueError)):
        util.calc_qc_qi(qr, amp)

################################################################################
################################# bounds_check #################################
################################################################################
@pytest.mark.parametrize("p0,bounds,expected", [
    (np.array([5.0, 10.0]), (np.array([0.0, 0.0]), np.array([20.0, 20.0])),
     (np.array([0.0, 0.0]), np.array([20.0, 20.0]))),
    (np.array([5.0, 10.0]), (np.array([20.0, 0.0]), np.array([0.0, 20.0])),
     (np.array([0.0, 0.0]), np.array([20.0, 20.0]))),
    (np.array([-5.0, -10.0]), (np.array([-20.0, -20.0]), np.array([0.0, 0.0])),
     (np.array([-20.0, -20.0]), np.array([0.0, 0.0]))),
    (np.array([25.0, 10.0]), (np.array([0.0, 0.0]), np.array([20.0, 20.0])),
     (np.array([0.0, 0.0]), np.array([27.5, 20.0]))),
    (np.array([-25.0, -10.0]), (np.array([-20.0, -20.0]), np.array([0.0, 0.0])),
     (np.array([-27.5, -20.0]), np.array([0.0, 0.0]))),
    (np.array([0.0, 0.0]), (np.array([-0.01, 10.0]), np.array([-10.0, -10.0])),
     (np.array([-10.0, -10.0]), np.array([1.0, 10.0]))),
    (np.array([5.0, -15.0]), (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
     (np.array([-10.0, -16.5]), np.array([10.0, 10.0]))),
    (np.array([0.]), (np.array([1.0]), np.array([2.0])), 
     (np.array([-1.0]), np.array([2.0]))),
    (np.array([0.]), (np.array([-1.0]), np.array([-2.0])), 
     (np.array([-2.0]), np.array([1.0])))
]) 
def test_bounds_check(p0, bounds, expected):
    bounds0 = (bounds[0].copy(), bounds[1].copy())
    new_bounds = util.bounds_check(p0, bounds)
    assert np.allclose(bounds, bounds0) # bounds input is not modified 
    assert type(new_bounds) is tuple 
    assert len(new_bounds) == 2 
    assert len(new_bounds[0]) == len(p0)
    assert len(new_bounds[1]) == len(p0) 
    assert np.allclose(new_bounds[0], expected[0])
    assert np.allclose(new_bounds[1], expected[1])

@pytest.mark.parametrize("p0,bounds", [
    (np.array([5.0, 10.0]), (np.array([5.0, 0.0]), np.array([10.0, 20.0]))),
    (np.array([0.0, -5.0]), (np.array([0.0, -10.0]), np.array([10.0, -5.0]))),
])
def test_bounds_check_on_edge_no_adjust(p0, bounds):
    new_bounds = util.bounds_check(p0, (bounds[0].copy(), bounds[1].copy()))
    assert np.allclose(new_bounds[0], bounds[0])
    assert np.allclose(new_bounds[1], bounds[1])

@pytest.mark.parametrize("p0,bounds,expected", [
    (np.array([5.0]), (np.array([5.0]), np.array([5.0])), 
     (np.array([5.0]), np.array([5.0]))),
    (np.array([10.0]), (np.array([5.0]), np.array([5.0])), 
     (np.array([5.0]), np.array([11.0]))),
    (np.array([0.0]), (np.array([0.0]), np.array([0.0])), 
     (np.array([0.0]), np.array([0.0])))
])
def test_bounds_check_equal_limits(p0, bounds, expected):
    new_bounds = util.bounds_check(p0, (bounds[0].copy(), bounds[1].copy()))
    assert np.allclose(new_bounds[0], expected[0])
    assert np.allclose(new_bounds[1], expected[1])

def test_bounds_check_return_types():
    p0 = np.array([1.0, 2.0])
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 3.0]))
    new_bounds = util.bounds_check(p0, bounds)
    assert isinstance(new_bounds, tuple)
    assert isinstance(new_bounds[0], list)
    assert isinstance(new_bounds[1], list)

@pytest.mark.parametrize("p0,bounds", [
    # bounds length mismatch
    (np.array([1.0]), (np.array([2.0]), np.array([0.0, 1.0]))), 
    (np.array([1.0]), (np.array([2.0, 3.0]), np.array([0.0, 1.0]))),  
    # bounds not a tuple
    (np.array([1.0]), [np.array([0.0]), np.array([1.0])]),
    (np.array([1.0]), "invalid_bounds"),
    # bounds tuple length not 2
    (np.array([1.0]), (np.array([0.0]))),
    (np.array([1.0]), (np.array([0.0]), np.array([1.0]), np.array([2.0]))),
    # p0 not array-like
    ("invalid_p0", (np.array([0.0]), np.array([1.0]))),
    (None, (np.array([0.0]), np.array([1.0]))),
    (["a"], (np.array([0.0]), np.array([1.0]))),
])
def test_bounds_check_invalid_input(p0, bounds):
    with pytest.raises((TypeError, ValueError)):
        util.bounds_check(p0, bounds)

################################################################################
################################## calc_nrmse ##################################
################################################################################
@pytest.mark.parametrize("z,z_fit,expected", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.0),
    (np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]), 2 / 14),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.nan),
    (np.array([1.0 + 1.0j, 2.0 + 2.0j]), np.array([1.0 + 0.0j, 2.0 + 0.0j]), 
     0.5),
    (np.array([1.0, -1.0, 1.0, -1.0]), np.array([-1.0, 1.0, -1.0, 1.0]), 4.0),
])
def test_calc_nrmse(z, z_fit, expected):
    nrmse = util.calc_nrmse(z, z_fit)
    assert pytest.approx(nrmse, nan_ok = True) == expected

def test_calc_nrmse_singleton():
    z = np.array([1.0 + 1.0j])
    z_fit = np.array([1.0 + 1.0j])
    nrmse = util.calc_nrmse(z, z_fit)
    assert nrmse == 0.0

def test_calc_nrmse_with_nan():
    z = np.array([1.0, np.nan, 3.0])
    z_fit = np.array([1.0, 2.0, 3.0])
    nrmse = util.calc_nrmse(z, z_fit)
    assert np.isnan(nrmse)

@pytest.mark.parametrize("z,z_fit", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])), # shape mismatch
    (np.array([1.0, 2.0, 3.0]), "invalid_fit"),      # invalid z_fit type
    ("invalid_z", np.array([1.0, 2.0, 3.0])),        # invalid z type 
])
def test_calc_nrmse_invalid_input(z, z_fit):
    with pytest.raises((ValueError, TypeError)):
        util.calc_nrmse(z, z_fit)

################################################################################
#################################### cardan ####################################
################################################################################
@pytest.mark.parametrize("a,b,c,d,largest,expected", [
    (1.0, -6.0, 11.0, -6.0, True, 3.0),   # roots: 1, 2, 3
    (1.0, -6.0, 11.0, -6.0, False, 1.0),  # roots: 1, 2, 3
    (1.0, 0.0, -1.0, 0.0, True, 1.0),    # roots: -1, 0, 1
    (1.0, 0.0, -1.0, 0.0, False, -1.0),   # roots: -1, 0, 1
    # roots: 1, -0.5 + j*sqrt(3)/2, -0.5 - j*sqrt(3)/2
    (1.0, 0.0, 0.0, -1.0, True, 1.0),
    # roots: 1, -0.5 + j*sqrt(3)/2, -0.5 - j*sqrt(3)/2
    (1.0, 0.0, 0.0, -1.0, False, 1.0),   
    (1.0, -3.0, 3.0, -1.0, True, 1.0),    # roots: 1 (triple root)
    (1.0, -3.0, 3.0, -1.0, False, 1.0),   # roots: 1 (triple root)
    (2.0, -4.0, -22.0, 24.0, True, 4.0),  # roots: -3, 4, 1
    (2.0, -4.0, -22.0, 24.0, False, -3.0), # roots: -3, 4, 1
    (np.nan, 0., 0.,  0., True, np.nan), # nan input
    (np.inf, 0., 0., 0., False, 0.), # inf input
    ([1.0, 1.0], -6.0, 11.0, -6.0, True, [3.0, 3.0]), # array input
    ([1.0, 1.0, 2.0], [-6.0, 0.0, -4.0], [11.0, -1.0, -22.0], 
     [-6.0, 0.0, 24.0], False, [1.0, -1.0, -3.0]), # array input
     ([1.0, 1.0, 2.0], [-6.0, 0.0, -4.0], [11.0, -1.0, -22.0], 
     [-6.0, 0.0, 24.0], True, [3.0, 1.0, 4.0]), # array input 
     (4., 4., 1., 0., True, 0.),
     (4., 4., 1., 0., False, -0.5)
])
def test_cardan(a, b, c, d, largest, expected):
    root = util.cardan(a, b, c, d, largest)
    assert pytest.approx(root, nan_ok = True) == expected

def test_cardan_no_real_roots():
    # Polynomial with no real roots: x^3 + x + 1 = 0
    a, b, c, d = 1.0, 0.0, 1.0, 1.0
    root = util.cardan(a, b, c, d, True)
    # The only real root is approximately -0.6823
    assert pytest.approx(root, rel=1e-4) == -0.6823

def test_cardan_degenerate_case():
    # Degenerate case: all coefficients zero
    a, b, c, d = 0.0, 0.0, 0.0, 0.0
    assert np.isnan(util.cardan(a, b, c, d, True)) 

@pytest.mark.parametrize("a,b,c,d,largest,expected", [
    (1.0, -2.0, 1.0, 0.0, True, 1.0),  # roots: 0, 1 (double)
    (1.0, -2.0, 1.0, 0.0, False, 0.0),
])
def test_cardan_double_root(a, b, c, d, largest, expected):
    root = util.cardan(a, b, c, d, largest)
    assert pytest.approx(root) == expected

@pytest.mark.parametrize("a,b,c,d", [
    (0.0, 1.0, -2.0, 1.0),
    (0.0, 0.0, 1.0, -1.0),
])
def test_cardan_non_cubic_returns_nan(a, b, c, d):
    assert np.isnan(util.cardan(a, b, c, d, True))

@pytest.mark.parametrize("a,b,c,d,largest", [
    ("1.0", -6.0, 11.0, -6.0, True), # invalid a
    (1.0, None, 11.0, -6.0, False), # invalid b
    (1.0, -6.0, "11.0", -6.0, True), # invalid c
    (1.0, -6.0, [11.0, 11.0], [ -6.0, -6.0, -6.0], False), # different sizes
    (1.0, -6.0, 11.0, -6.0, "True"), # invalid largest
])
def test_cardan_invalid_input(a, b, c, d, largest):
    with pytest.raises((TypeError, ValueError)):
        util.cardan(a, b, c, d, largest)


################################################################################
################################ get_peak_fwhm #################################
################################################################################
# Testing general properties of the output, not exact values
x = np.linspace(0, 100, 100)
lorentz = lambda x, x0, g, A: A * g**2 / ((x - x0) ** 2 + g**2)
@pytest.mark.parametrize("x,y", [
    (np.array([1, 2, 3, 4, 5]), np.array([0, 1, 0, 2, 0])),
    (np.linspace(0, 4, 50), np.random.random(50)),
    ([1, 2, 3, 4], [1, 1, 2, 1]),
    (x, lorentz(x, 50, 10, 10)),
    (x, lorentz(x, 10, 5, 0.5)),
    (x, lorentz(x, 80, 5, -0.5))
])
def test_get_peak_fwhm(x, y):
    xpeak, ypeak, fwhm = util.get_peak_fwhm(x, y)
    assert ypeak <= max(y) * 2
    assert xpeak >= min(x)
    assert xpeak <= max(x) 
    assert fwhm < (max(x) - min(x))

def test_get_peak_fwhm_descending_x():
    x = np.linspace(10, 0, 100)
    y = np.sin(np.linspace(0, 10, 100)) + 1.0
    xpeak, ypeak, fwhm = util.get_peak_fwhm(x, y)
    assert xpeak >= min(x)
    assert xpeak <= max(x)
    assert fwhm > 0 and fwhm < (max(x) - min(x))

def test_get_peak_fwhm_multiple_peaks_highest_selected():
    x = np.linspace(0, 100, 500)
    lorentz = lambda x, x0, g, A: A * g**2 / ((x - x0) ** 2 + g**2)
    y = lorentz(x, 30, 5, 10) + lorentz(x, 70, 5, 20)
    xpeak, ypeak, fwhm = util.get_peak_fwhm(x, y)
    assert abs(xpeak - 70) < abs(xpeak - 30)

def test_get_peak_fwhm_constant_signal_fallback():
    x = np.linspace(0, 100, 200)
    y = np.ones_like(x) * 5.0
    xpeak, ypeak, fwhm = util.get_peak_fwhm(x, y)
    assert pytest.approx(ypeak) == 5.0
    assert fwhm > 0 and fwhm < (x.max() - x.min())

def test_get_peak_fwhm_monotonic_signal_fallback():
    x = np.linspace(0, 50, 100)
    y = x.copy()
    xpeak, ypeak, fwhm = util.get_peak_fwhm(x, y)
    assert fwhm > 0 and fwhm < (x.max() - x.min())

@pytest.mark.parametrize("x,y", [
    (np.array([1, 2, 3]), np.array([0, 1])), # shape mismatch
    ("invalid_x", np.array([0, 1, 0])),       # invalid x type
    (np.array([1, 2, 3]), "invalid_y"),       # invalid y type
    (np.array([]), np.array([])), # empty arrays
    ([1, 2, 3], [1, 1, 2]), # size < 4
])
def test_get_peak_fwhm_invalid_input(x, y):
    with pytest.raises((ValueError, TypeError)):
        util.get_peak_fwhm(x, y)