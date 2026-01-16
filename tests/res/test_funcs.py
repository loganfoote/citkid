import pytest 
import numpy as np 
from citkid.res import funcs 
# not testing circle_objective, because is is legacy code. 
# citkid.xcal.circle.circle_objective should be used instead.
################################################################################
#################################### get_y #####################################
################################################################################
@pytest.mark.parametrize("y0,a", [
    ([0.0], 0.0),
    ([1], 0.0),
    ([-1], 0.0),
    ([0.0], 0.5),
    ([1], 0.5),
    ([-1], 0.5),
    ([0.0], 1.0),
    ([1], 1.0),
    ([-1], 1.0),
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.5),
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.77), 
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.8),
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.01),
])
def test_get_y(y0, a):
    y0 = np.array(y0, dtype = np.float64)
    # compare to numy root finding
    for largest in [True, False]:
        y_exp = []
        for y0i in y0:
            coeffs = [4, -4 * y0i, 1, -(y0i + a)]
            roots = np.roots(coeffs)
            real_mask = np.isclose(roots.imag, 0, atol=1e-12)
            real_roots = roots[real_mask].real
            y_exp.append(np.max(real_roots) if largest else np.min(real_roots))
        y_exp = np.array(y_exp, dtype = np.float64)
        y = funcs.get_y(y0, a, largest = largest)
        assert np.allclose(y, y_exp)

@pytest.mark.parametrize("y0,a", [
    (["0.0"], 0.0), # invalid y0
    ([0.0], "0.0"), # invalid a
    ([0.0, 1.0], 0.0), # array y0 with invalid size
])
def test_get_y_invalid_input(y0, a):
    with pytest.raises((TypeError, ValueError)):
        funcs.get_y(y0, a)

################################################################################
################################# nonlinear_iq #################################
######################### and nonlinear_iq_for_fitter ##########################
################################################################################
@pytest.mark.parametrize("f,fr,Qr,amp,phi,a,i0,q0,tau,downward,expected", [
    ([1e9, 1.1e9, 1.2e9], 1.1e9, 100., 0.9, -0.01, 0.4, 0., 0., 1e-9, True, 
     [0., 0., 0.]),
     # i0 and j0
    ([0.9e9, 1e9], 1e9, 1., 0., 0., 0., 1., 0., 0., True, [1., 1.]),
    ([0.9e9, 1e9], 1e9, 1., 0., 0., 0., 0., 1., 0., True, [1j, 1j]),
    # tau 
    ([0.9e9, 1e9], 1e9, 1., 0., 0., 0., 1., 0., 0.5e-8, True, [-1., 1.]),
    # amp 
    ([1e9], 1e9, 1., 1.,  0., 0., 1., 0., 0., True, [0.]),
    ([1e9], 1e9, 1., 0.5, 0., 0., 1., 0., 0., True, [1 / 2]),
    # phi 
    ([1e9], 1e9, 1., 1.,  np.pi / 4, 0., 1., 0., 0., True, [-1j]),
    # Qr 
    ([1.1e9], 1e9, 10., 0.5, 0., 0., 1., 0., 0., True, [0.9 + 0.2j]),
])
def test_nonlinear_iq(f, fr, Qr, amp, phi, a, i0, q0, tau, downward, expected):
    # nonlinear_iq
    result = funcs.nonlinear_iq(np.array(f), fr, Qr, amp, phi, a, i0, q0, tau, 
                                downward)
    assert np.allclose(result, expected)
    # nonlinear_iq_for_fitter
    result = funcs.nonlinear_iq_for_fitter(np.array(f), fr * 100e-6, Qr * 1e-4, 
                                           amp, phi, a, i0, q0, tau * 1e6, 
                                           downward)
    assert np.allclose(result, np.hstack((np.real(expected), 
                                          np.imag(expected))))

def test_nonlinear_iq_anl():
    # Not testing the output of get_y 
    f = np.array([0.])
    # nonlinear_iq
    result0 = funcs.nonlinear_iq(f, 1e9, 1., 1., 0., 0, 1., 0., 0., True)
    result1 = funcs.nonlinear_iq(f, 1e9, 1., 1., 0., 1., 1., 0., 0., True)
    result2 = funcs.nonlinear_iq(f, 1e9, 1., 1., 0., 1., 1., 0., 0., False)
 
    assert not np.allclose(result0, result1)
    assert not np.allclose(result1, result2) 
    assert not np.allclose(result0, result2)

    # nonlinear_iq_for_fitter
    result0 = funcs.nonlinear_iq_for_fitter(f, 1e5, 1e-4, 1., 0., 
                                            0., 1., 0., 0., True)
    result1 = funcs.nonlinear_iq_for_fitter(f, 1e5, 1e-4, 1., 0., 
                                            1., 1., 0., 0., True)
    result2 = funcs.nonlinear_iq_for_fitter(f, 1e5, 1e-4, 1., 0., 
                                            1., 1., 0., 0., False)
 
    assert not np.allclose(result0, result1)
    assert not np.allclose(result1, result2) 
    assert not np.allclose(result0, result2)

@pytest.mark.parametrize("f,fr,Qr,amp,phi,a,i0,q0,tau,downward", [
    (['a'], 0., 0., 0., 0., 0., 0., 0., 0., True), # f not float
    (0., 0., 0., 0., 0., 0., 0., 0., 0., True), # f not array-like
    ([0.], 'a', 0., 0., 0., 0., 0., 0., 0., True), # fr not float
    ([0.], 0., 'a', 0., 0., 0., 0., 0., 0., True), # Qr not float
    ([0.], 0., 0., 'a', 0., 0., 0., 0., 0., True), # amp not float
    ([0.], 0., 0., 0., 'a', 0., 0., 0., 0., True), # phi not float
    ([0.], 0., 0., 0., 0., 'a', 0., 0., 0., True), # a not float
    ([0.], 0., 0., 0., 0., 0., 'a', 0., 0., True), # i0 not float
    ([0.], 0., 0., 0., 0., 0., 0., 'a', 0., True), # q0 not float
    ([0.], 0., 0., 0., 0., 0., 0., 0., 'a', True), # tau not float
    ([0.], 0., 0., 0., 0., 0., 0., 0., 0., 'a'), # downward not bool
    ([0], 0., 0., 0., 0., 0., 0., 0., 0., False), # f is int 
    ([0.], np.nan, 0., 0., 0., 0., 0., 0., 0., False) # fr is nan
])
def test_nonlinear_iq_invalid_input(f, fr, Qr, amp, phi, a, i0, q0, tau, 
                                    downward):
    # nonlinear_iq
    with pytest.raises((TypeError, ValueError)):
        funcs.nonlinear_iq(f, fr, Qr, amp, phi, a, i0, q0, tau, downward)
    # nonlinear_iq_for_fitter
    with pytest.raises((TypeError, ValueError)):
        funcs.nonlinear_iq_for_fitter(f, fr, Qr, amp, phi, a, i0, q0, tau, 
                                      downward)