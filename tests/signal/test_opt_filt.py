from citkid.signal import opt_filt
import pytest
import numpy as np

################################################################################
################################ create_opt_filt ###############################
################################################################################

@pytest.mark.parametrize("a,SJ,nfft", [
    (np.sin(np.linspace(0, np.pi / 2, 8)), np.ones(5), 8), # len(a) == nfft
    (np.sin(np.linspace(0, np.pi / 2, 4)), np.ones(5), 8), # len(a) < nfft
    (np.random.rand(10), np.linspace(1, 2, 6), 10), # random noise
    (np.random.rand(16), np.linspace(0.5, 1.5, 9), 16) # random noise
])
def test_create_opt_filt_valid_inputs(a, SJ, nfft):
    h_out = opt_filt.create_opt_filt(a, SJ, nfft)
    assert isinstance(h_out, np.ndarray), "Output h should be a numpy array"
    assert h_out.dtype == np.float64, "Output h should be of type float64"
    assert len(h_out) == nfft, f"Output h should have length {nfft}"

    # check against implementation-consistent FFT/scaling (pyfftw)
    a_padded = np.pad(a, (0, nfft - len(a)))
    A = __import__('pyfftw').interfaces.numpy_fft.rfft(a_padded, n = nfft)
    if nfft % 2 == 0:
        denom = (np.abs(A[0])**2 / SJ[0]
                 + np.abs(A[-1])**2 / SJ[-1]
                 + 2.0 * np.sum(np.abs(A[1:-1])**2 / SJ[1:-1]))
    else:
        denom = (np.abs(A[0])**2 / SJ[0]
                 + 2.0 * np.sum(np.abs(A[1:])**2 / SJ[1:]))
    H = nfft * A.conj() / SJ / denom
    h_expected = __import__('pyfftw').interfaces.numpy_fft.irfft(H, n = nfft)
    h_expected = __import__('pyfftw').interfaces.numpy_fft.ifftshift(h_expected)
    assert np.allclose(h_out, h_expected), "Output h does not match expected values"

@pytest.mark.parametrize("a, SJ, nfft", [
    (np.sin(np.linspace(0, np.pi / 2, 10)), np.ones(8), 8), # a longer than nfft
    (np.sin(np.linspace(0, np.pi / 2, 8)), np.ones(4), 8),  # SJ wrong length
    (np.sin(np.linspace(0, np.pi / 2, 8)), np.ones(6), 8),  # SJ wrong length
    (np.sin(np.linspace(0, np.pi / 2, 8)), np.ones(8), 8.5), # nfft not integer
    (np.sin(np.linspace(0, np.pi / 2, 8)), np.ones(8), -1), # negative nfft
    (np.sin(np.linspace(0, np.pi / 2, 8)), np.ones(8), 0), # zero nfft
])
def test_create_opt_filt_invalid_inputs(a, SJ, nfft):
    with pytest.raises(AssertionError):
        opt_filt.create_opt_filt(a, SJ, nfft)

################################################################################
################################ apply_opt_filt ################################
################################################################################
def test_apply_opt_filt():
    s = np.random.rand(100)
    h = np.random.rand(16)
    y = opt_filt.apply_opt_filt(s, h)
    assert isinstance(y, np.ndarray), "Output y should be a numpy array"
    assert y.dtype == np.float64, "Output y should be of type float64"
    assert len(y) == len(s), f"Output y should have length {len(s)}"

    # check against scipy convolve
    from scipy.signal import convolve
    y_expected = np.roll(convolve(s, h, mode = 'same'), -1)
    assert np.allclose(y, y_expected), "Output y does not match expected values"

    # check against known output
    s = [0, 0, 0, 0, 0, 2, 4, 2, 0, 0, 0, 0] 
    h = [0, 0, 1, 2, 1, 0, 0] 
    y = opt_filt.apply_opt_filt(s, h) 
    y_expected = [0, 0, 0, 2, 8, 12, 8, 2, 0, 0, 0, 0]
    assert np.allclose(y, y_expected), "Output y does not match expected values"

@pytest.mark.parametrize("s,h", [
    (np.random.rand(10), np.random.rand(12)), # h longer than s
    (["a", "b"], [1, 2, 3]), # non-numeric s
    ([1, 2, 3], ["a", "b"]), # non-numeric h
])
def test_apply_opt_filt_invalid_inputs(s, h):
    with pytest.raises(Exception):
        opt_filt.apply_opt_filt(s, h)

################################################################################
################################## create_nsd ##################################
################################################################################
@pytest.mark.parametrize("s,nfft", [
    (np.random.rand(1000), 256),
    (np.random.rand(5000), 512),
    (np.random.rand(2048), 1024),
    (np.random.rand(256), 256),
])
def test_create_nsd_valid_inputs(s, nfft):
    SJ = opt_filt.create_nsd(s, nfft)
    assert isinstance(SJ, np.ndarray), "Output SJ should be a numpy array"
    assert SJ.dtype == np.float64, "Output SJ should be of type float64"
    m = nfft // 2 + 1
    assert len(SJ) == m, f"Output SJ should have length {m}"

    # check that values are non-negative
    assert np.all(SJ >= 0), "All values in SJ should be non-negative"

def test_create_nsd_amplitude():
    s = np.random.randn(1000) * 3.0  # noise with std dev of 3.0
    nfft = 256
    SJ = opt_filt.create_nsd(s, nfft)
    # The mean of the PSD should be approximately equal to the variance of s
    variance_s = np.var(s)
    mean_SJ = np.mean(SJ)
    assert np.isclose(mean_SJ / 2, variance_s, rtol = 0.2), \
        "Mean of SJ should be close to variance of s * 2"

def test_create_nsd_zero_input():
    s = np.zeros(512)
    SJ = opt_filt.create_nsd(s, 256)
    assert np.allclose(SJ, 0), "Zero input should yield zero PSD"
    
@pytest.mark.parametrize("s,nfft", [
    (np.random.rand(100), 256),  # s shorter than nfft
    (['a', 'b', 'c'], 128), # s not numeric
    (np.random.rand(500), -128), # negative nfft
    (np.random.rand(500), 128.5) # nfft not integer
])
def test_create_nsd_invalid_inputs(s, nfft):
    with pytest.raises((AssertionError, ValueError)):
        opt_filt.create_nsd(s, nfft)    

################################################################################
################################## iterate_of ##################################
################################################################################
def test_iterate_of():
    # Just checking output types and shapes here
    # simple build_template and get_start_idx functions
    def build_template(s, start_idx):
        return np.mean([s[i:i+8] for i in start_idx], axis = 0)
    def get_start_idx(y):
        return np.array([5, 20, 35, 50, 65, 80], dtype = np.int32) 
    s = np.random.rand(100)
    j = s[:30]
    a, y, h, start_idx = \
        opt_filt.iterate_of(s, j = j, 
                            start_idx = np.array([5, 20, 35, 50, 65, 80]),
                            build_template = build_template,
                            get_start_idx = get_start_idx,
                            N_iter = 5,
                            verbose = False)
    assert isinstance(a, np.ndarray), "Output a should be a numpy array"
    assert a.dtype == np.float64, "Output a should be of type float64"
    assert isinstance(y, np.ndarray), "Output y should be a numpy array"
    assert y.dtype == np.float64, "Output y should be of type float64"
    assert isinstance(h, np.ndarray), "Output h should be a numpy array"
    assert h.dtype == np.float64, "Output h should be of type float64"
    assert isinstance(start_idx, np.ndarray), \
        "Output start_idx should be a numpy array"
    assert np.issubdtype(start_idx.dtype, np.integer), \
        "Output start_idx should be of integer type"
    
    assert a.shape[0] == 6, "Output a should have shape (N_iter + 1, M)"
    assert a.shape[1] == 8, "Output a should have shape (N_iter + 1, M)"
    assert len(y) == len(s), f"Output y should have length {len(s)}"
    assert len(h) <= len(s), \
        "Output h should have length less than or equal to len(s)"
    assert len(start_idx) == 6, \
        "Output start_idx should have length equal to number of instances"

def test_iterate_of_invalid_iter():
    def build_template(s, start_idx):
        return np.mean([s[i:i+4] for i in start_idx], axis = 0)
    def get_start_idx(y):
        return np.array([1, 5, 9], dtype = np.int32)
    s = np.random.rand(20)
    j = s[:12]
    with pytest.raises(Exception):
        opt_filt.iterate_of(s, j, start_idx = np.array([1, 5, 9]),
                            build_template = build_template,
                            get_start_idx = get_start_idx,
                            N_iter = 0,
                            verbose = False)



################################################################################
################################### get_nfft ###################################
################################################################################
# This is essentially just testing pyfftw.next_fast_len, but we include it for
# completeness.
@pytest.mark.parametrize("L", [100, 256, 500, 1024, 2048])
def test_get_nfft(L):
    nfft = opt_filt.get_nfft(L)
    assert isinstance(nfft, int), "Output nfft should be an integer"
    assert nfft > 0, "Output nfft should be positive"

@pytest.mark.parametrize("N", [0, -10, 128.5, 'a'])
def test_get_nfft_invalid(N):
    with pytest.raises(AssertionError):
        opt_filt.get_nfft(N)
