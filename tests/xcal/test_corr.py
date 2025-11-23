import pytest
import numpy as np
from citkid.xcal import corr 

################################################################################
################################# test_corr.py #################################
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
    (y, 2, 3, 0.1, (0.1, 4), (0.5, 4), False)
])
def test_calc_common_modes(x, N_comp, N_iter, dt, lowpass_params, 
                           highpass_params, verbose):
    a, A, sig_iter, a_full = corr.calc_common_modes(x, N_comp, N_iter, dt, 
                                                    lowpass_params, 
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
    # invalid filter parameters
    (x, 2, 3, 0.1, (1.0,), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,), False), 
    (x, 2, 3, 0.1, (), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (), False), 
    (x, 2, 3, 0.1, (1.0,), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,), False), 
    (x, 2, 3, 0.1, (1.0,2.0,3.0), (0.5, 2), False),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,2.0,4.0), False), 
    # negative N_comp   
    (x, -1, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # N_comp > N
    (x, 6, 3, 0.1, (1.0, 4), (0.5, 2), False),
    # negative N_iter
    (x, 2, -1, 0.1, (1.0, 4), (0.5, 2), False),
])
def test_find_common_modes_invalid(x, N_comp, N_iter, dt, lowpass_params, 
                                   highpass_params, verbose):
    with pytest.raises(Exception):
        corr.calc_common_modes(
            x, N_comp, N_iter, dt, lowpass_params, highpass_params, verbose
        )


################################################################################
################################## calc_sigma ##################################
################################################################################
@pytest.mark.parametrize("x,sig_exp", [
    ([[1, 1, 1], [2, 2, 2]], [1, np.sqrt(12) / np.sqrt(3)]),
    ([[0, 0, 0], [0, 0, 0]], [0, 0]),
    ([[1, -1, 1, -1], [2, -2, 2, -2]], [1, 2]),
    ([[]], [np.nan]), # empty timestreams
])  
def test_calc_sigma(x, sig_exp):
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
x = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 4, 6, 8, 10, 12, 14]], dtype = np.float64)
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
])
def test_calc_a(x, A, a_exp):
    x = np.array(x, dtype = np.float64)
    A = np.array(A, dtype = np.float64)
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