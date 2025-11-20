from citkid.noise import cal
import pytest
import numpy as np
from citkid.noise.psd import get_psd

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
    z_rot = cal.cent_rot_s21(z, center, phase)
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
        cal.cent_rot_s21(z, center, phase)


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
    theta = cal.convert_to_theta(z, unwrap)
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
        cal.convert_to_theta(z, False)

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
    A = cal.convert_to_A(z)
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
        cal.convert_to_A(z)

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
    freq, spar, sper = cal.get_spar_sper(theta, A, radius, dt, get_freqs)

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
        cal.get_spar_sper(theta, A, radius, dt, get_freqs)

################################################################################
################################# get_xcal_ix #################################
################################################################################
# Need to add empty list
ffine = np.arange(0, 100, 1, dtype = np.float64)
tfine = ffine.copy()
tnoise = np.random.permutation(np.linspace(40, 60, 100))
tnoise_glitch = np.concatenate([tnoise, [1e3]])
m = "ffine,tfine,tnoise,ix0_offset,ix1_offset,std_cutoff,ix_exp"
ix = np.random.permutation(np.arange(100))
tfine1 = tfine.copy() 
tfine1[10], tfine1[21] = tfine1[21], tfine1[10]
ffine2, tfine2 = np.flip(ffine), np.flip(tfine)
@pytest.mark.parametrize(m, [
    (ffine, tfine, [20.5], 1, 1, None, np.arange(19, 23, 1, dtype = np.int32)),
    (ffine, tfine, [20.5], 0, 0, None, np.arange(20, 22, 1, dtype = np.int32)),
    (ffine, tfine, [20.5], 1, -100, None, np.array([], dtype = np.int32)),
    (ffine, tfine, [20.5], -100, 0, None, np.array([], dtype = np.int32)),
    (ffine, tfine, [20.5], 100, 1, None, np.arange(0, 23, 1, dtype = np.int32)),
    (ffine, tfine, [20.5], 1, 100, None, np.arange(19, 100, dtype = np.int32)),
    (ffine, tfine, [20.5, 21.5], 0, 0, None, np.arange(20, 23, 1, 
                                                       dtype = np.int32)),
    (ffine, tfine, [20, 22], 0, 0, None, np.arange(19, 24, 1, 
                                                       dtype = np.int32)),
    (ffine, tfine, tnoise, 0, 0, None, np.arange(39, 62, 1, dtype = np.int32)),
    (ffine, tfine, tnoise_glitch, 0, 0, 3, np.arange(39, 62, 1, 
                                                     dtype = np.int32)),
    (ffine, tfine, tnoise_glitch, 0, 0, 11, np.arange(39, 100, 1, 
                                                      dtype = np.int32)),
    (ffine[ix], tfine[ix], [20.5], 1, 1, None, np.arange(19, 23, 1, dtype = np.int32)),
    (ffine, tfine1, [20.5], 0, 0, None, np.arange(9, 23, 1, dtype = np.int32)),
    (ffine2, tfine2, [20.5], 1, 1, None, np.arange(19, 23, 1, dtype = np.int32)),
    (ffine, tfine2, [20.5], 1, 1, None, np.arange(77, 81, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise, 0, 0, None, np.arange(38, 61, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise_glitch, 0, 0, None, np.arange(0, 61, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise_glitch, 0, 0, 3, np.arange(38, 61, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise_glitch, 0, 0, 11, np.arange(0, 61, 1, dtype = np.int32)),
    (ffine, tfine1, tnoise, 0, 0, None, np.arange(39, 62, 1, dtype = np.int32)),
    ([], [], [2.5], 0, 0, None, np.array([], dtype = np.int32)),
])
def test_get_xcal_ix(ffine, tfine, tnoise, ix0_offset, ix1_offset, 
                     std_cutoff, ix_exp):
    ix = cal.get_xcal_ix(ffine, tfine, tnoise, 
                         ix0_offset, ix1_offset, std_cutoff)
    assert isinstance(ix, np.ndarray)
    assert ix.dtype == np.int32
    assert np.allclose(ix, ix_exp)
    

@pytest.mark.parametrize("ffine,tfine,tnoise,ix0_offset,ix1_offset,std_cutoff", [
    (['a', 1, 2], [1, 2, 3], [2], 1, 1, None),  # non-numeric ffine
    ([1, 2, 3], ['a', 2, 3], [2], 1, 1, None),  # non-numeric tfine
    ([1, 2, 3], [1, 2, 3], ['a', 2], 1, 1, None),  # non-numeric tnoise
    ([1, 2, 3], [1, 2], [2], 1, 1, None),  # mismatched ffine and tfine lengths
    ([1, 2, 3], [1, 2, 3], [2], 1.5, 1, None),  # non-integer ix0_offset
    ([1, 2, 3], [1, 2, 3], [2], 1, 'a', None),  # non-integer ix1_offset
    ([1, 2, 3], [1, 2, 3], [2], 1, 1, -1),  # negative std_cutoff
])  
def test_get_xcal_ix_invalid_input(ffine, tfine, tnoise, ix0_offset, 
                                   ix1_offset, std_cutoff):
    with pytest.raises(Exception):
        cal.get_xcal_ix(ffine, tfine, tnoise, 
                        ix0_offset, ix1_offset, std_cutoff)
    