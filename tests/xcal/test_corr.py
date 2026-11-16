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
m = "x,N_comp,N_iter,dt,lowpass_params,highpass_params"
@pytest.mark.parametrize(m, [
    # basic test
    (x, 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # N_comp = 0
    (x, 0, 3, 0.1, (1.0, 4), (0.5, 2)),
    # N_iter = 1
    (x, 2, 1, 0.1, (1.0, 4), (0.5, 2)),
    # different x format
    (y, 2, 3, 0.1, (0.1, 4), (0.5, 4)),
    # N_comp = N
    (x, x.shape[0], 2, 0.1, (1.0, 4), (0.5, 2))
])
def test_calc_cm(
    x, N_comp, N_iter, dt, lowpass_params, highpass_params
):
    a, A, sig_iter, a_full = corr.calc_cm(
        x, N_comp, N_iter, dt, lowpass_params, highpass_params
    )
    # check shapes
    assert a.shape == (x.shape[0], N_comp)
    assert A.shape == (N_comp, x.shape[1])
    assert sig_iter.shape == (N_iter, x.shape[0], N_comp)
    assert a_full.shape == (x.shape[0], x.shape[0])


def test_calc_cm_sig_iter_is_sigma_of_highpassed_residual():
    """
    Check that ``sig_iter`` stores sigma (RMS), not sigma squared: for a
    single iteration it equals ``calc_sig`` of the highpassed residual.
    """
    rng = np.random.default_rng(1)
    x = 3.0 * rng.normal(size = (4, 2000))
    dt, lp, hp = 0.01, (1.0, 4), (0.5, 2)
    a, A0, sig_iter, _ = corr.calc_cm(x, 1, 1, dt, lp, hp)

    xc = x - x.mean(axis = 1)[:, np.newaxis]
    A = corr.lowpass_filter(A0, dt, *lp)
    sig = corr.calc_sig(corr.highpass_filter(xc - a @ A, dt, *hp))
    np.testing.assert_allclose(sig_iter[0], np.repeat(sig, 1, axis = 1))
    assert 1.0 < np.median(sig_iter) < 4.0  # sigma ~ 3, not 9

def test_calc_cm_without_highpass():
    """
    Check that highpass_params = None runs (as documented) and skips the
    highpass: with a single iteration, sig_iter is the RMS of the
    unfiltered residual.
    """
    rng = np.random.default_rng(3)
    xs = rng.normal(size = (4, 1000))
    dt, lp = 0.01, (1.0, 4)
    a, A0, sig_iter, a_full = corr.calc_cm(xs, 1, 1, dt, lp, None)

    assert a.shape == (4, 1) and A0.shape == (1, 1000)
    assert a_full.shape == (4, 4)
    xc = xs - xs.mean(axis = 1)[:, np.newaxis]
    A = corr.lowpass_filter(A0, dt, *lp)
    np.testing.assert_allclose(sig_iter[0], corr.calc_sig(xc - a @ A))

@pytest.mark.parametrize(m, [
    # invalid x (1D)    
    (np.random.randn(100), 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # empty x     
    ([[]], 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # non-numeric x    
    ([['A', 'B'], ['C', 'D']], 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # negative dt
    (x, 2, 3, -0.1, (1.0, 4), (0.5, 2)),
    # non-numeric dt
    (x, 2, 3, 'a', (1.0, 4), (0.5, 2)),
    # zero dt
    (x, 2, 3, 0.0, (1.0, 4), (0.5, 2)),
    # NaN dt
    (x, 2, 3, np.nan, (1.0, 4), (0.5, 2)),
    # invalid filter parameters
    (x, 2, 3, 0.1, (1.0,), (0.5, 2)),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,)), 
    (x, 2, 3, 0.1, (), (0.5, 2)),
    (x, 2, 3, 0.1, (1.0, 4), ()), 
    (x, 2, 3, 0.1, (1.0,), (0.5, 2)),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,)), 
    (x, 2, 3, 0.1, (1.0,2.0,3.0), (0.5, 2)),
    (x, 2, 3, 0.1, (1.0, 4), (0.5,2.0,4.0)), 
    # negative N_comp   
    (x, -1, 3, 0.1, (1.0, 4), (0.5, 2)),
    # N_comp > N
    (x, 6, 3, 0.1, (1.0, 4), (0.5, 2)),
    # negative N_iter
    (x, 2, -1, 0.1, (1.0, 4), (0.5, 2)),
])
def test_find_cm_invalid(
    x, N_comp, N_iter, dt, lowpass_params, highpass_params
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(Exception):
            corr.calc_cm(x, N_comp, N_iter, dt, lowpass_params, highpass_params)

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
    # Orthogonal modes: least squares equals the per-mode projection
    ([[2, 0, 1], [0, 3, 0]], [[1, 0, 0], [0, 1, 0]], [[2, 0], [0, 3]]),
    # Single mode
    ([[1, 1, 1], [2, 2, 2]], [[3, 3, 3]], [[1 / 3], [2 / 3]]),
    # Non-orthogonal modes: x = 2 * A0 + 3 * A1 exactly
    ([[5, 3, 2]], [[1, 0, 1], [1, 1, 0]], [[2, 3]]),
    # Complex mode: least squares a = x A^H / (A A^H)
    ([[1, 1], [2, 2]], [[1 + 1j, 1 - 1j]], [[0.5], [1.0]]),
])
def test_calc_a(x, A, a_exp):
    """
    Check least-squares scaling factors, including non-orthogonal modes.
    """
    a = corr.calc_a(x, A)
    np.testing.assert_allclose(a, a_exp, atol = 1e-12)


def test_calc_a_recovers_scaling_of_correlated_modes():
    """
    Check that noiseless mixtures of strongly correlated modes are
    recovered exactly (per-mode projection gets these wrong).
    """
    rng = np.random.default_rng(2)
    A0 = rng.normal(size = 500)
    A = np.array([A0, A0 + 0.3 * rng.normal(size = 500)])
    a_true = np.array([[1.5, -0.5], [0.2, 2.0], [0.0, 1.0]])
    np.testing.assert_allclose(corr.calc_a(a_true @ A, A), a_true, atol = 1e-12)

@pytest.mark.parametrize("x,A", [
    (1, [[3, 3, 3], [4, 4, 4]]), # 0D x
    ([[1, 1, 1], [2, 2, 2]], 1), # 0D A
    ([1, 1, 1], [[3, 3, 3], [4, 4, 4]]), # 1D x
    ([[1, 1, 1], [2, 2, 2]], [3, 3, 3]),  # 1D A
    ([], [[3, 3, 3], [4, 4, 4]]), # empty x
    ([[1, 1, 1], [2, 2, 2]], []), # empty A
    ([[1, 1, 1], [2, 2, 2]], [[3, 3], [4, 4]]), # mismatched time length
    ([['A'], ['B']], [[3, 3, 3], [4, 4, 4]]), # non-numeric x
    # non-numeric A
    ([[1, 1, 1], [2, 2, 2]], [['A', 'B', 'C'], ['D', 'E', 'F']]),
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
m = "z,theta,N_comp,N_iter,dt,lowpass_params,highpass_params"
@pytest.mark.parametrize(m, [
    # basic test
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # custom theta 
    (x, np.zeros(x.shape[0]), 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # N_comp = 0
    (x, None, 0, 3, 0.1, (1.0, 4), (0.5, 2)),
    # N_iter = 1
    (x, None, 2, 1, 0.1, (1.0, 4), (0.5, 2)),
    # different x format
    (y, None, 2, 3, 0.1, (0.1, 4), (0.5, 4)),
    # complex64 input promoted to complex128
    (np.array(x, dtype = np.complex64), None, 2, 3, 0.1, (1.0, 4), (0.5, 2))
])
def test_calc_cm_complex(
    z, theta, N_comp, N_iter, dt, lowpass_params, highpass_params
):
    aI, aQ, AI, AQ, sigI_iter, sigQ_iter, aI_full, aQ_full, theta_out = \
        corr.calc_cm_complex(z, theta, N_comp, N_iter, dt, lowpass_params, 
                             highpass_params)
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

m = "z,theta,N_comp,N_iter,dt,lowpass_params,highpass_params"
@pytest.mark.parametrize(m, [
    # invalid x (1D)    
    (np.random.randn(100), None, 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # empty x     
    ([[]], None, 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # non-numeric x    
    ([['A', 'B'], ['C', 'D']], None, 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    # negative dt
    (x, None, 2, 3, -0.1, (1.0, 4), (0.5, 2)),
    # non-numeric dt
    (x, None, 2, 3, 'a', (1.0, 4), (0.5, 2)),
    # zero dt
    (x, None, 2, 3, 0.0, (1.0, 4), (0.5, 2)),
    # invalid filter parameters
    (x, None, 2, 3, 0.1, (1.0,), (0.5, 2)),
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5,)), 
    (x, None, 2, 3, 0.1, (), (0.5, 2)),
    (x, None, 2, 3, 0.1, (1.0, 4), ()), 
    (x, None, 2, 3, 0.1, (1.0,), (0.5, 2)),
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5,)), 
    (x, None, 2, 3, 0.1, (1.0,2.0,3.0), (0.5, 2)),
    (x, None, 2, 3, 0.1, (1.0, 4), (0.5,2.0,4.0)), 
    # negative N_comp   
    (x, None, -1, 3, 0.1, (1.0, 4), (0.5, 2)),
    # N_comp > N
    (x, None, 6, 3, 0.1, (1.0, 4), (0.5, 2)),
    # negative N_iter
    (x, None, 2, -1, 0.1, (1.0, 4), (0.5, 2)),
    # invalid theta shape  
    (x, [0, 1], 2, 3, 0.1, (1.0, 4), (0.5, 2)),
    (x, np.array([[0, 1]]), 2, 3, 0.1, (1.0, 4), (0.5, 2)),
])
def test_calc_cm_complex_invalid(
    z, theta, N_comp, N_iter, dt, lowpass_params, highpass_params
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(Exception):
            corr.calc_cm_complex(z, theta, N_comp, N_iter, dt, lowpass_params, 
                                 highpass_params)
        
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
    # Custom theta: a scalar for 1D z
    (z, aI, aQ, AI, AQ, 0, 0.0, 
     [8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6]),
    (z, aI, aQ, AI, AQ, 1, 0.0, 
     [6 + 1j * 4, -5 + 1j * -1, -14 + 1j * -6]),
    # Custom theta: one angle per row of 2D z, or a scalar for every row
    ([z, z], aI, aQ, AI, AQ, [0, 1], np.zeros(2), 
     [[8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6],
      [6 + 1j * 4, -5 + 1j * -1, -14 + 1j * -6]]),
    ([z, z], aI, aQ, AI, AQ, [0, 1], 0.0, 
     [[8 + 1j * 4, -3 + 1j * -1, -12 + 1j * -6],
      [6 + 1j * 4, -5 + 1j * -1, -14 + 1j * -6]]),
    
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
        theta_exp = np.broadcast_to(theta, np.shape(theta_out))
        np.testing.assert_allclose(theta_out, theta_exp, equal_nan = True)


@pytest.mark.parametrize("T", [50, 3])  # T = N: square z
def test_remove_cm_complex_rotates_each_row_by_its_theta(T):
    """
    Check that each row of 2D z is rotated by its own theta: removing the
    rotated common modes from rotated data gives the rotated residual. The
    square case (T == N) used to rotate by column without raising.
    """
    rng = np.random.default_rng(0)
    N, C = 3, 2
    aI2, aQ2 = rng.normal(size=(N, C)), rng.normal(size=(N, C))
    AI2, AQ2 = rng.normal(size=(C, T)), rng.normal(size=(C, T))
    resid = rng.normal(size=(N, T)) + 1j * rng.normal(size=(N, T))
    z_rot = aI2 @ AI2 + 1j * (aQ2 @ AQ2) + resid
    theta = np.array([0.3, -1.2, 2.0])
    z2 = z_rot * np.exp(1j * theta)[:, np.newaxis]

    y, theta_out = corr.remove_cm_complex(
        z2, aI2, aQ2, AI2, AQ2, np.arange(N), theta)

    np.testing.assert_allclose(
        y, resid * np.exp(1j * theta)[:, np.newaxis], atol = 1e-12)
    np.testing.assert_allclose(theta_out, theta)

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
    (z, aI, aQ, AI, AQ, 0, np.zeros(3)), # per-sample theta for 1D z
    ([z, z], aI, aQ, AI, AQ, [0, 1], np.zeros(3)), # wrong length for 2D z
    # empty idx unsupported
    ([z, z], aI, aQ, AI, AQ, [], None),
])  
def test_remove_cm_complex_invalid(z, aI, aQ, AI, AQ, idx, theta):
    with pytest.raises(Exception):
        corr.remove_cm_complex(z, aI, aQ, AI, AQ, idx, theta)


def _calc_cm_reference(x, N_comp, N_iter, dt, lowpass_params,
                       highpass_params):
    """
    The common-mode iteration written out plainly, with full copies at
    every step, to check the memory-trimmed ``calc_cm`` against.
    """
    x = x - np.mean(x, axis = 1)[:, np.newaxis]
    sig = np.sqrt(np.sum(x ** 2, axis = 1)[:, np.newaxis] / x.shape[1])
    for i in range(N_iter):
        xf = x if highpass_params is None else \
            corr.highpass_filter(x, dt, *highpass_params)
        x0 = xf / sig
        C = x0 @ x0.T / x.shape[1]
        eigval, eigvec = np.linalg.eig(C)
        eigval, eigvec = np.real(eigval), np.real(eigvec)
        order = np.flip(np.argsort(eigval))
        a_full = sig * eigvec[:, order]
        a = a_full[:, :N_comp]
        A0 = (a.T @ (x / sig ** 2))
        A = corr.lowpass_filter(A0, dt, *lowpass_params)
        y = x - a @ A
        if highpass_params is not None:
            y = corr.highpass_filter(y, dt, *highpass_params)
        sig = np.sqrt(np.sum(y ** 2, axis = 1)[:, np.newaxis] / x.shape[1])
    return a, A0, sig, a_full


@pytest.mark.parametrize('highpass_params', [(0.5, 2), None])
def test_calc_cm_matches_plain_algebra(highpass_params):
    """
    The trimmed calc_cm (filters row by row, the normalization folded
    into the covariance, the residual formed in place) gives the plainly
    written algebra's result, and leaves its input unchanged.
    """
    rng = np.random.default_rng(5)
    T, dt = 4000, 0.01
    modes = rng.normal(size = (2, T))
    x = rng.normal(size = (6, 2)) @ modes + 0.5 * rng.normal(size = (6, T))
    x += rng.normal(size = (6, 1))
    before = x.copy()
    a, A0, sig_iter, a_full = corr.calc_cm(x, 2, 3, dt, (1.0, 4),
                                           highpass_params)
    np.testing.assert_array_equal(x, before)
    a_r, A0_r, sig_r, a_full_r = _calc_cm_reference(x, 2, 3, dt, (1.0, 4),
                                                    highpass_params)
    # Eigenvectors are defined up to a sign
    sign = np.sign(np.sum(a * a_r, axis = 0))
    np.testing.assert_allclose(a * sign, a_r, rtol = 1e-8, atol = 1e-10)
    np.testing.assert_allclose(A0 * sign[:, None], A0_r, rtol = 1e-8,
                               atol = 1e-10)
    np.testing.assert_allclose(sig_iter[-1][:, :1], sig_r, rtol = 1e-10)
    assert a_full.shape == a_full_r.shape
    # The modes removed reproduce the plain subtraction
    y = corr.remove_cm(x, a, A0, np.arange(6))
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(y, x - a @ A0, rtol = 1e-12)
    assert y is not x


def test_pca_and_remove_cm_leave_inputs_alone():
    """
    pca and remove_cm never write into their inputs, with or without the
    highpass, and remove_cm handles a complex input.
    """
    rng = np.random.default_rng(6)
    x = rng.normal(size = (4, 500))
    before = x.copy()
    a, A = corr.pca(x, 2, highpass_params = (0.01, 1.0, 2))
    np.testing.assert_array_equal(x, before)
    a2, A2 = corr.pca(x, 2)
    np.testing.assert_array_equal(x, before)
    assert np.allclose(x, *[v @ w for v, w in [corr.pca(x, 4)]])
    z = x + 1j * rng.normal(size = x.shape)
    y = corr.remove_cm(z, a2, A2, np.arange(4))
    np.testing.assert_allclose(y, z - a2 @ A2)
    assert y.dtype == np.complex128
