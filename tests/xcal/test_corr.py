import pytest
import warnings
import numpy as np
from citkid.xcal import corr 

################################################################################
################################### calc_cm ####################################
################################################################################
# This does not test the math in detail, but ensures that the functions run and
# return arrays of the correct shape. 
x = np.random.randn(5, 100)
y = np.random.randn(1000, 100)
m = "x,N_comp,N_iter,dt,lowpass_params,highpass_params,verbose"
@pytest.mark.parametrize(m, [
    # basic test
    (x, 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_comp = 0
    (x, 0, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_iter = 1
    (x, 2, 1, 0.1, (1.0, 4), (0.5, 2), False),
    # verbose = True
    (x, 2, 3, 0.1, (1.0, 4), (0.5, 2), True),
    # different x format
    (y, 2, 3, 0.1, (0.1, 4), (0.5, 4), False),
    # N_comp = N
    (x, x.shape[0], 2, 0.1, (1.0, 4), (0.5, 2), False)
])
def test_calc_cm(x, N_comp, N_iter, dt, lowpass_params,  highpass_params, 
                 verbose):
    a, A, sig_iter, a_full = corr.calc_cm(x, N_comp, N_iter, dt, lowpass_params, 
                                          highpass_params, verbose)
    # check shapes
    assert a.shape == (x.shape[0], N_comp)
    assert A.shape == (N_comp, x.shape[1])
    assert sig_iter.shape == (N_iter, x.shape[0], N_comp)
    assert a_full.shape == (x.shape[0], x.shape[0])

@pytest.mark.parametrize(m, [
    # invalid x (1D)    
    (np.random.randn(100), 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # empty x     
    ([[]], 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # non-numeric x    
    ([['A', 'B'], ['C', 'D']], 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # negative dt
    (x, 2, 3, -0.1, (1.0, 4), (0.5, 2), False),
    # non-numeric dt
    (x, 2, 3, 'a', (1.0, 4), (0.5, 2), False),
    # zero dt
    (x, 2, 3, 0.0, (1.0, 4), (0.5, 2), False),
    # NaN dt
    (x, 2, 3, np.nan, (1.0, 4), (0.5, 2), False),
    # invalid filter parameters
    (x, 2, 3, 0.1, (1.0,), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,), False), 
    (x, 2, 3, 0.1, (), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (), False), 
    (x, 2, 3, 0.1, (1.0,), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,), False), 
    (x, 2, 3, 0.1, (1.0,2.0,3.0), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,2.0,4.0), False), 
    # None highpass_params unsupported
    (x, 2, 3, 0.1, (1.0, 4), None, False),
    # negative N_comp   
    (x, -1, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_comp > N
    (x, 6, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # negative N_iter
    (x, 2, -1, 0.1, (1.0, 4), (0.5, 2), False),
])
def test_find_cm_invalid(x, N_comp, N_iter, dt, lowpass_params, highpass_params, 
                         verbose):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(Exception):
            corr.calc_cm(x, N_comp, N_iter, dt, lowpass_params, highpass_params, 
                         verbose)

################################################################################
################################## calc_sigma ##################################
################################################################################
@pytest.mark.parametrize("x,sig_exp", [
    ([[1, 1, 1], [2, 2, 2]], [1, np.sqrt(12) / np.sqrt(3)]),
    ([[0, 0, 0], [0, 0, 0]], [0, 0]),
    ([[1, -1, 1, -1], [2, -2, 2, -2]], [1, 2]),
    ([[]], [np.nan]), # empty timestreams
    ([[1, np.nan], [2, 2]], [np.nan, 2]), # NaN propagates
])  
def test_calc_sigma(x, sig_exp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x = np.array(x, dtype = np.float64)
        sig_exp = np.array(sig_exp, dtype = np.float64)[:, np.newaxis]
        sig = corr.calc_sig(x)
        np.testing.assert_allclose(sig, sig_exp, equal_nan = True)
        assert sig.shape == (x.shape[0], 1)

@pytest.mark.parametrize("x", [
    (1), # 0D input
    ([1, 1, 1]), # 1D input
    ([['A'], ['B']]), # non-numeric input
])
def test_calc_sigma_invalid(x):
    with pytest.raises(Exception):
        corr.calc_sig(x)

################################################################################
####################################### pca ####################################
################################################################################
# This does not check all the pca math, but ensures that shapes and basic 
# matrix properties are correct
x = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 4, 6, 8, 10, 12, 14]], 
             dtype = np.float64)
@pytest.mark.parametrize("x,sig,highpass_params", [    
    ([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], None, None),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], None, None),
    ([[1, 2], [3, 4], [5, 6], [7, 8]], None, None),
    (x, None, None),
    (x, corr.calc_sig(x), None),
    (x, None, (1.0, 0.1, 1)),
    (x, corr.calc_sig(x), (1.0, 0.1, 1)),
])
def test_pca(x, sig, highpass_params):
    x = np.array(x, dtype = np.float64)
    # N_comp = number of timestreams
    N_comp = x.shape[0]
    a, A = corr.pca(x, N_comp, sig = None, highpass_params = None)
    assert np.allclose(x, a @ A) # a @ A reconstructs x
    # N_comp = 1
    for N_comp in [x.shape[0], 0, 1]:
        a, A = corr.pca(x, N_comp, sig = sig, highpass_params = highpass_params)
        # A is orthogonal
        assert np.allclose(A @ A.T, np.diag(np.diag(A @ A.T))) 
        # check shapes
        assert A.shape[0] == a.shape[1] == N_comp
        assert A.shape[1] == x.shape[1]
        assert a.shape[0] == x.shape[0] 


@pytest.mark.parametrize("x,N_comp,sig,highpass_params", [
    (1, 1, None, None), # 0D x
    ([[1, 2, 3]], -1, None, None), # negative N_comp
    ([[1, 2, 3]], 2, None, None), # N_comp > N
    ([[1, 2, 3]], 1, [1, 2], None), # invalid sig shape
    ([[1, 2, 3]], 1, None, (1.0,)), # invalid highpass_params shape
])
def test_pca_invalid(x, N_comp, sig, highpass_params):
    with pytest.raises(Exception):
        corr.pca(x, N_comp, sig = sig, highpass_params = highpass_params)

################################################################################
#################################### calc_a ####################################
################################################################################
@pytest.mark.parametrize("x,A,a_exp", [
    ([[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], 
     [[1 / 3, 1 / 4], [2 / 3, 24 / 48]]),
    ([[1, 1, 1]], [[3, 3, 3], [4, 4, 4]], 
     [[1 / 3, 1 / 4]]),
    ([[1, 1, 1], [2, 2, 2]], [[3, 3, 3]], 
    [[1 / 3], [2 / 3]]),
    # Complex A with real denominator via real(A)**2
    ([[1, 1], [2, 2]], [[1 + 1j, 1 - 1j]], 
    [[1], [2]]),
])
def test_calc_a(x, A, a_exp):
    x = np.array(x, dtype = np.float64)
    # Preserve complex values for complex test case
    _A_arr = np.array(A)
    A = np.array(A, dtype = np.complex128 if np.iscomplexobj(_A_arr) else np.float64)
    a_exp = np.array(a_exp, dtype = np.float64)
    a = corr.calc_a(x, A)
    np.testing.assert_allclose(a, a_exp, equal_nan = True)

@pytest.mark.parametrize("x,A", [
    (1, [[3, 3, 3], [4, 4, 4]]), # 0D x
    ([[1, 1, 1], [2, 2, 2]], 1), # 0D A
    ([1, 1, 1], [[3, 3, 3], [4, 4, 4]]), # 1D x
    ([[1, 1, 1], [2, 2, 2]], [3, 3, 3]),  # 1D A
    ([], [[3, 3, 3], [4, 4, 4]]), # empty x
    ([[1, 1, 1], [2, 2, 2]], []), # empty A
    ([[1, 1, 1], [2, 2, 2]], [[3, 3], [4, 4]]), # mismatched time length
    ([['A'], ['B']], [[3, 3, 3], [4, 4, 4]]), # non-numeric x
    ([[1, 1, 1], [2, 2, 2]], [['A', 'B', 'C'], ['D', 'E', 'F']]), # non-numeric A
    ([[1, 1, np.nan], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]), # NaN in x
    ([[1, 1, 1], [2, 2, 2]], [[3, 3, np.nan], [4, 4, 4]]), # NaN in A
])
def test_calc_a_invalid(x, A):
    with pytest.raises(Exception):
        corr.calc_a(x, A)

################################################################################
################################ calc_cm_complex ###############################
################################################################################
x = np.random.randn(5, 100) + 1j * np.random.randn(5, 100)
y = np.random.randn(1000, 100) + 1j * np.random.randn(1000, 100)
m = "z,theta,N_comp,N_iter,dt,lowpass_params,highpass_params,verbose"
@pytest.mark.parametrize(m, [
    # basic test
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # custom theta 
    (x, np.zeros(x.shape[0]), 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_comp = 0
    (x, None, 0, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_iter = 1
    (x, None, 2, 1, 0.1, (1.0, 4), (0.5, 2), False),
    # verbose = True
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5, 2), True),
    # different x format
    (y, None, 2, 3, 0.1, (0.1, 4), (0.5, 4), False),
    # complex64 input promoted to complex128
    (np.array(x, dtype = np.complex64), None, 2, 3, 0.1, (1.0, 4), (0.5, 2), False)
])
def test_calc_cm_complex(z, theta, N_comp, N_iter, dt, lowpass_params, 
                         highpass_params, verbose):
    aI, aQ, AI, AQ, sigI_iter, sigQ_iter, aI_full, aQ_full, theta_out = \
        corr.calc_cm_complex(z, theta, N_comp, N_iter, dt, lowpass_params, 
                             highpass_params, verbose)
    # check shapes
    assert aI.shape == (z.shape[0], N_comp)
    assert AI.shape == (N_comp, z.shape[1])
    assert sigI_iter.shape == (N_iter, z.shape[0], N_comp)
    assert aI_full.shape == (z.shape[0], z.shape[0]) 
    assert aQ.shape == (z.shape[0], N_comp)
    assert AQ.shape == (N_comp, z.shape[1])
    assert sigQ_iter.shape == (N_iter, z.shape[0], N_comp)
    assert aQ_full.shape == (z.shape[0], z.shape[0])
    # check theta output
    if theta is None:
        theta_exp = np.angle(np.median(z, axis = 1))
        np.testing.assert_allclose(theta_out, theta_exp, equal_nan = True)
    else:
        np.testing.assert_allclose(theta_out, theta, equal_nan = True)

m = "z,theta,N_comp,N_iter,dt,lowpass_params,highpass_params,verbose"
@pytest.mark.parametrize(m, [
    # invalid x (1D)    
    (np.random.randn(100), None, 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # empty x     
    ([[]], None, 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # non-numeric x    
    ([['A', 'B'], ['C', 'D']], None, 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # negative dt
    (x, None, 2, 3, -0.1, (1.0, 4), (0.5, 2), False),
    # non-numeric dt
    (x, None, 2, 3, 'a', (1.0, 4), (0.5, 2), False),
    # zero dt
    (x, None, 2, 3, 0.0, (1.0, 4), (0.5, 2), False),
    # invalid filter parameters
    (x, None, 2, 3, 0.1, (1.0,), (0.5, 2), False),
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5,), False), 
    (x, None, 2, 3, 0.1, (), (0.5, 2), False),
    (x, None, 2, 3, 0.1, (1.0, 4), (), False), 
    (x, None, 2, 3, 0.1, (1.0,), (0.5, 2), False),
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5,), False), 
    (x, None, 2, 3, 0.1, (1.0,2.0,3.0), (0.5, 2), False),
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5,2.0,4.0), False), 
    # negative N_comp   
    (x, None, -1, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_comp > N
    (x, None, 6, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # negative N_iter
    (x, None, 2, -1, 0.1, (1.0, 4), (0.5, 2), False),
    # invalid theta shape  
    (x, [0, 1], 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    (x, np.array([[0, 1]]), 2, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # None highpass_params unsupported
    (x, None, 2, 3, 0.1, (1.0, 4), None, False),
])
def test_calc_cm_complex_invalid(z, theta, N_comp, N_iter, dt, lowpass_params, 
                                 highpass_params, verbose):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(Exception):
            corr.calc_cm_complex(z, theta, N_comp, N_iter, dt, lowpass_params, 
                                 highpass_params, verbose)
        
################################################################################
################################## remove_cm ###################################
################################################################################
x = np.array([10, 10, 10, 10, 10], dtype = np.float64)
a = np.array([[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], dtype = np.float64)
A = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype = np.float64)
@pytest.mark.parametrize("x,a,A,idx,y_exp", [
    (x, a, A, 0, [8, 7, 8, 7, 8]),
    (x, a, A, 1, [6, 5, 6, 5, 6]),
    ([x], a, A, [0], [[8, 7, 8, 7, 8]]),
    ([x, x], a, A, [0, 1], [[8, 7, 8, 7, 8], [6, 5, 6, 5, 6]]),
])
def test_remove_cm(x, a, A, idx, y_exp):
    x = np.array(x, dtype = np.float64)
    a = np.array(a, dtype = np.float64)
    A =  np.array(A, dtype = np.float64)
    y_exp = np.array(y_exp, dtype = np.float64)
    y = corr.remove_cm(x, a, A, idx)
    np.testing.assert_allclose(y, y_exp, equal_nan = True)

@pytest.mark.parametrize("x,a,A,idx", [
    (1, a, A, 0), # 0D x 
    ([[1, 2, 3]], a, A, 0), # x with length mismatch
    (x, 1, A, 0), # 0D a
    ([x], a, A, 0), # idx dimension doesn't match x
    (x, [[1]], A, 0), # a with shape mismatch
    (x, a, 1, 0), # 0D A    
    (x, a, [[1, 2, 3]], 0), # A with shape mismatch
    (x, a, A, [0, 5]), # invalid idx
    (x, a, A, -1), # negative idx
    (x, a, A, [0, -1])
])
def test_remove_cm_invalid(x, a, A, idx):
    with pytest.raises(Exception):
        corr.remove_cm(x, a, A, idx)

################################################################################
################################ remove_cm_complex #############################
################################################################################
z = np.array([10 + 1j * 5, 0 + 1j * 0, -10 + 1j * -5], dtype = np.complex128)
aI = np.array([[2, 3], [4, 5], [6, 7]], dtype = np.float64)
aQ = np.array([[1, 1], [1, 1], [1, 1]], dtype = np.float64)
AI = np.array([[1, 0, 1], [0, 1, 0]], dtype = np.float64)
AQ = np.array([[0, 1, 0], [1, 0, 1]], dtype = np.float64)
@pytest.mark.parametrize("z,aI,aQ,AI,AQ,idx,theta,y_exp", [
    # single timestream, 1D input
    (z, aI, aQ, AI, AQ, 0, None, [8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6]),
    # Single timestream, 2D input
    ([z], aI, aQ, AI, AQ, [0], None, 
     [[8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6]]),
    # Multiple timestreams
    ([z, z], aI, aQ, AI, AQ, [0, 1], None, 
     [[8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6],
      [6 + 1j * 4, -5 + 1j * -1, -14 + 1j * -6]]),
    # Single timestream, idx = 1
    (z, aI, aQ, AI, AQ, 1, None, [6 + 1j * 4, -5 + 1j * -1, -14 + 1j * -6]),
    # Custom theta
    (z, aI, aQ, AI, AQ, 0, np.zeros(z.shape[0]), 
     [8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6]),
    (z, aI, aQ, AI, AQ, 1, np.zeros(z.shape[0]), 
     [6 + 1j * 4, -5 + 1j * -1, -14 + 1j * -6]),
    
])  
def test_remove_cm_complex(z, aI, aQ, AI, AQ, idx, theta, y_exp):
    z = np.array(z, dtype = np.complex128)
    aI = np.array(aI, dtype = np.float64)
    aQ = np.array(aQ, dtype = np.float64)
    AI =  np.array(AI, dtype = np.float64)
    AQ =  np.array(AQ, dtype = np.float64)
    y_exp = np.array(y_exp, dtype = np.complex128)
    y, theta_out = corr.remove_cm_complex(z, aI, aQ, AI, AQ, idx, theta)
    np.testing.assert_allclose(y, y_exp, equal_nan = True)
    print(theta_out)
    if theta is None:
        if len(z.shape) == 1:
            assert isinstance(theta_out, float) or theta_out.shape == ()
        else:
            assert theta_out.shape == (z.shape[0],)
    else:
        np.testing.assert_allclose(theta, theta_out, equal_nan = True)

@pytest.mark.parametrize("z,aI,aQ,AI,AQ,idx,theta", [
    (1, aI, aQ, AI, AQ, 0, None), # 0D z 
    ([z], aI, aQ, AI, AQ, 0, None), # idx dimension doesn't match z
    ([[1 + 1j * 1]], aI, aQ, AI, AQ, 0, None), # z with length mismatch
    (z, 1, aQ, AI, AQ, 0, None), # 0D aI
    (z, [[1]], aQ, AI, AQ, 0, None), # aI with shape mismatch
    (z, aI, 1, AI, AQ, 0, None), # 0D aQ
    (z, aI, [[1]], AI, AQ, 0, None), # aQ with shape mismatch
    (z, aI, aQ, 1, AQ, 0, None), # 0D AI    
    (z, aI, aQ, [[1, 2, 3]], AQ, 0, None), # AI with shape mismatch
    (z, aI, aQ, AI, 1, 0, None), # 0D AQ    
    (z, aI, aQ, AI, [[1, 2, 3]], 0, None), # AQ with shape mismatch
    (z, aI, aQ, AI, AQ, [0, 5], None), # invalid idx
    (z, aI, aQ, AI, AQ, -1, None), # negative idx
    (z, aI, aQ, AI, AQ, [0, -1], None),
    (z, aI, aQ, AI, AQ, 0, [0, 1]), # invalid theta shape
    (z, aI, aQ, AI, AQ, 0, np.array([[0, 1]])),
    # empty idx unsupported
    ([z, z], aI, aQ, AI, AQ, [], None),
])  
def test_remove_cm_complex_invalid(z, aI, aQ, AI, AQ, idx, theta):
    with pytest.raises(Exception):
        corr.remove_cm_complex(z, aI, aQ, AI, AQ, idx, theta)
