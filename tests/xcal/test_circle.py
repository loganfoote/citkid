import pytest
import numpy as np
from citkid.xcal import circle
from citkid.noise.psd import get_psd
import matplotlib
matplotlib.use("Agg")

################################################################################
################################# fit_iq_circle ################################
################################################################################
@pytest.mark.parametrize("z,idx,popt_exp", [
    ([1, 1j, -1, -1j], None, [0, 0, 1]),
    ([2, 1 + 1j, 0, 1 - 1j], None, [1, 0, 1]),
    ([1 + 1j, 2j, -1 + 1j, 0], None, [0, 1, 1]),
    ([1e30, 1e30j, -1e30, -1e30j], None, [0, 0, 1e30]),
    ([1e-30, 1e-30j, -1e-30, -1e-30j], None, [0, 0, 1e-30]),
    ([1, 1j, -1, -1j, 10], [0, 1, 2, 3], [0, 0, 1]),
    ([10, 1, 1j, -1, -1j, 10], [1, 2, 3, 4], [0, 0, 1]),
    ([10, 1, 1j, -1, -1j], [1, 2, 3, 4], [0, 0, 1]),
    ([1, 10, 1j, -1, -1j, 10], [0, 2, 3, 4], [0, 0, 1]),
])
def test_fit_iq_circle_gain(z, idx, popt_exp):
    mask = np.ones(len(z), dtype = np.bool_)
    if idx is not None:
        mask[:] = False
        mask[idx] = True
    origin, radius = circle.fit_iq_circle(z, mask)
    assert np.allclose([origin.real, origin.imag, radius], popt_exp)
    
@pytest.mark.parametrize("z,mask", [
    ([], None),  # empty input
    ([np.nan, 1, -1, 1j], None),  # nan input
    (['a', 1, -1, 1j], None),  # non-numeric input
    ([1, 1j], None),  # less than 3 points
    ([1, 1j, -1, -1j], [True, True, False, False]),  # less than 3 points after 
                                                     # masking
    ([1, 1j, -1, -1j], [True] * 5),  # wrong length mask
    ([1, 1j, -1, -1j], [True] * 2),  # wrong length mask
])
def test_fit_iq_circle_invalid_input(z, mask):
    with pytest.raises(Exception):
        circle.fit_iq_circle(z, mask)

################################################################################
################################# cent_rot_s21 #################################
################################################################################
@pytest.mark.parametrize("z,center,phase,z_exp", [
    ([1], 0, 0, [1]),
    (1, 0, 0, 1),
    ([1], 1, 0, [0]),
    ([1], 1j, 0, [1 - 1j]),
    ([1], 0, 2 * np.pi, [1]),
    ([1], 0, np.pi / 2, [-1j]),
    ([2], 1, np.pi, [-1]),
    ([np.nan, 1, -1, 1j], 0, 0, [np.nan, 1, -1, 1j]),  # nan input
    ([1 + 1j, 2 + 2j, 3 + 3j], 1 + 1j, np.pi / 4, 
     [0, np.sqrt(2), 2 * np.sqrt(2)]),
    ([], 0, 0, []),  # empty input
])
def test_cent_rot_s21(z, center, phase, z_exp):
    z_rot = circle.cent_rot_s21(z, center, phase)
    if isinstance(z_rot, np.ndarray):
        assert z_rot.dtype == np.complex128
    else:
        assert type(z_rot) == np.complex128
    
    assert np.allclose(z_rot, np.array(z_exp, dtype = np.complex128), 
                       equal_nan = True)
    
    
@pytest.mark.parametrize("z,center,phase", [
    (['a', 1, -1, 1j], 0, 0),  # non-numeric input
    ([1], 0, 'a'),  # non-numeric phase
    ([1], 'a', 0),  # non-numeric center 
])
def test_cent_rot_s21_invalid_input(z, center, phase):
    with pytest.raises(Exception):
        circle.cent_rot_s21(z, center, phase)


################################################################################
############################### convert_to_theta ###############################
################################################################################
t0 = np.linspace(0, 6 * np.pi, 100)
@pytest.mark.parametrize("z,unwrap,theta_exp", [
    ([1], False, [0]),
    (1, False, 0),
    ([1j], False, [np.pi / 2]),
    ([-1], False, [np.pi]),
    ([-1j], False, [-np.pi / 2]),
    ([1, 1j, -1, -1j], False, [0, np.pi / 2, np.pi, - np.pi / 2]),
    ([np.nan, 1, -1, 1j], False, [np.nan, 0, np.pi, np.pi / 2]),  # nan input
    ([], False, []),  # empty input
    ([1], True, [0]),
    (np.exp(1j * t0), False, np.angle(np.exp(1j * t0))),
    (np.exp(1j * t0), True, t0),
    ([], True, []),  # empty input
])
def test_convert_to_theta(z, unwrap, theta_exp):
    theta = circle.convert_to_theta(z, unwrap)
    if isinstance(theta, np.ndarray):
        assert theta.dtype == np.float64
    else:
        assert type(theta) == np.float64
    assert np.allclose(theta, np.array(theta_exp, dtype = np.float64), 
                       equal_nan = True, rtol = 1e-7)
    
@pytest.mark.parametrize("z", [
    (['a', 1, -1, 1j]),  # non-numeric input
])
def test_convert_to_theta_invalid_input(z):
    with pytest.raises(Exception):
        circle.convert_to_theta(z, False)

################################################################################
################################ convert_to_A ##################################
################################################################################
@pytest.mark.parametrize("z,A_exp", [
    ([1], [1]),
    (1, 1),
    ([1j], [1]),
    ([-1], [1]),
    ([-1j], [1]),
    ([1, 1j, -1, -1j], [1, 1, 1, 1]),
    ([3 + 4j, 0 + 0j, -3 - 4j], [5, 0, 5]),
    ([np.nan, 3 + 4j], [np.nan, 5]),  # nan input
    ([], []),  # empty input
])
def test_convert_to_A(z, A_exp):
    A = circle.convert_to_A(z)
    if isinstance(A, np.ndarray):
        assert A.dtype == np.float64
    else:
        assert type(A) == np.float64
    assert np.allclose(A, np.array(A_exp, dtype = np.float64), 
                       equal_nan = True)
    
@pytest.mark.parametrize("z", [
    (['a', 1, -1, 1j]),  # non-numeric input
])
def test_convert_to_A_invalid_input(z):
    with pytest.raises(Exception):
        circle.convert_to_A(z)

################################################################################
############################### get_spar_sper ##################################
################################################################################
x = np.arange(0, 10, 0.1)
y = np.sin(x)
freq_exp, y_exp = get_psd(y, dt = 0.1, get_frequencies = True)
y_exp = 10 * np.log10(y_exp)

m = "theta,A,radius,dt,get_freqs,freq_exp,spar_exp,sper_exp"
@pytest.mark.parametrize(m, [
    ([1], [1], 1, 0.5, True, [0], [0], [0]),
    ([1], [1], 1, 0.5, False, None, [0], [0]),
    ([1], [10], 1, 0.5, True, [0], [0], [20]),
    ([10], [1], 1, 0.5, True, [0], [20], [0]),
    ([1], [1], 10, 0.5, True, [0], [20], [0]),
    (y, y, 1, 0.1, True, freq_exp, y_exp, y_exp),
])
def test_get_spar_sper(theta, A, radius, dt, get_freqs, 
                       freq_exp, spar_exp, sper_exp):    
    freq, spar, sper = circle.get_spar_sper(theta, A, radius, dt, get_freqs)

    assert isinstance(spar, np.ndarray)
    assert isinstance(sper, np.ndarray)
    
    assert spar.dtype == np.float64
    assert sper.dtype == np.float64
    assert spar.shape == sper.shape
    if get_freqs:
        assert isinstance(freq, np.ndarray)
        assert freq.dtype == np.float64
        assert freq.shape == spar.shape
    else:
        assert freq is None

    assert np.allclose(spar, np.array(spar_exp, dtype = np.float64), 
                       rtol = 1e-7, equal_nan = True)
    assert np.allclose(sper, np.array(sper_exp, dtype = np.float64), 
                       rtol = 1e-7, equal_nan = True)
    if get_freqs:
        assert np.allclose(freq, np.array(freq_exp, dtype = np.float64), 
                           rtol = 1e-7)
        
@pytest.mark.parametrize("theta,A,radius,dt,get_freqs", [
    (['a', 1, -1], [1], 1, 0.5, True),  # non-numeric theta
    ([1], ['a', 1, -1], 1, 0.5, True),  # non-numeric A
    ([1], [1], 'a', 0.5, True),  # non-numeric radius
    ([1], [1], 1, 'a', True),  # non-numeric dt
    ([1], [1], 1, np.nan, True),  # non-finite dt
    ([1], [1], np.nan, 0.5, True),  # non-finite radius
    ([1, 2], [1], 1, 0.5, True),  # mismatched theta and A lengths
    ([np.nan], [1], 1, 0.5, True),  # non-finite value in theta
    ([1], [np.nan], 1, 0.5, True),  # non-finite value in A
    ([1], [1], 1, -1, True),  # negative dt
])
def test_get_spar_sper_invalid_input(theta, A, radius, dt, get_freqs):
    with pytest.raises(Exception):
        circle.get_spar_sper(theta, A, radius, dt, get_freqs)