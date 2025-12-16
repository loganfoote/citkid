import numpy as np 
import pyfftw 
from scipy.signal import fftconvolve
from tqdm.auto import tqdm

# Conventions: lower case = time domain, upper case = frequency domain

################################################################################
######################## Create and Apply Optimal Filter #######################
################################################################################
def create_opt_filt(a, SJ, nfft):
    """
    Create an optimal filter timestream h from template timestream a and noise 
    timestream j.

    Parameters:
    a (np.ndarray, float64, (M,)): template timestream. M must be <= nfft.
    SJ (np.ndarray, float64, (nfft//2 + 1,)): spectral noise density. 
    nfft (int): FFT length for computation.

    Returns:
    h (np.ndarray, float64, (nfft,)): optimal filter timestream. 
    """
    # input validation
    a = np.asarray(a, dtype = np.float64)
    SJ = np.asarray(SJ, dtype = np.float64) 
    assert isinstance(nfft, (int, np.integer)), "nfft must be an integer"
    assert nfft > 0, "nfft must be positive"
    assert len(a) <= nfft, "Length of a must be less than or equal to nfft" 
    assert len(SJ) == nfft // 2 + 1, "Length of SJ must be nfft // 2 + 1" 
      
    # pad a
    a = np.pad(a, (0, nfft - len(a)))
    
    # compute FFT of a, scaling ensures amplitude consistency
    A = pyfftw.interfaces.numpy_fft.rfft(a, n = nfft)  * nfft / 2
    
    # compute H and h
    H = A.conj() / SJ / np.sum(A * np.conj(A) / SJ)
    h = pyfftw.interfaces.numpy_fft.irfft(H, n = nfft) 
    h = pyfftw.interfaces.numpy_fft.ifftshift(h)
    return h

def apply_opt_filt(s, h):
    """
    Applies the optimal filter h to the signal timestream s using 
    FFT convolution.

    Parameters:
    s (np.ndarray, float64, (L,)): signal timestream.
    h (np.ndarray, float64, (nfft,)): optimal filter timestream. nfft must be 
        less than or equal to L.

    Returns:
    y (np.ndarray, float64, (L,)): filtered timestream.
    """
    # input validation
    s = np.asarray(s, dtype = np.float64)
    h = np.asarray(h, dtype = np.float64) 
    assert len(h) <= len(s), \
        "Length of h must be less than or equal to length of s"
    
    # apply filter using fftconvolve
    y = np.roll(fftconvolve(s, h, mode = 'same'), -1)
    return y

def create_nsd(s, nfft):
    """
    Get the noise spectral density of a signal by averaging over 
    segments of length nfft.

    Parameters:
    s (np.array, float64, (P,)): input noise timestream which does not contain 
        signal instances.
    nfft (int): length of each segment for FFT.

    Returns:
    SJ (np.array, (nfft//2 + 1,), float): one-sided noise spectral density
        estimate.
    """
    s = np.asarray(s, np.float64)
    assert isinstance(nfft, (int, np.integer)), "nfft must be an integer"
    assert nfft > 0, "nfft must be positive"
    assert len(s) >= nfft, \
        "Length of s must be greater than or equal to nfft"
    
    step = nfft
    psd_accum = np.zeros(nfft // 2 + 1)
    count = 0

    for i in range(0, len(s) - nfft + 1, step):
        chunk = s[i:i+nfft]
        X = pyfftw.interfaces.numpy_fft.rfft(chunk)
        X = (np.abs(X)**2) / nfft
        X[1:-1] *= 2
        psd_accum += X
        count += 1

    if count == 0:
        raise ValueError("Not enough samples for PSD")
    
    SJ = psd_accum / count
    return SJ

################################################################################
############################## Iterative Procedure #############################
################################################################################
def iterate_of(s, j, start_idx, build_template, get_start_idx,
               N_iter = 10, verbose = True):
    """
    Minimal iterative procedure for creating an optimal filter template.

    Parameters:
    s (np.ndarray, float64, (L,)): signal timestream for signal extraction. 
    j (np.ndarray, float64, (P,)): noise timestream for noise estimation. 
    start_idx (np.ndarray, int64, (K0,)): initial start indices of signal 
        instances.
    build_template (function): function that builds a template from the signal 
        timestream and start indices. Takes in (s, start_idx) as arguments and 
        returns a template timestream of shape (M,). 
    get_start_idx (function): function that gets start indices from the filtered 
        timestream. Takes in (y) as argument and returns start indices of shape 
        (K1,). 
    N_iter (int): number of iterations to perform. Default is 10. 
    verbose (bool): whether to show a progress bar. Default is True.

    Returns: 
    a (np.ndarray, float64, (N_iter + 1, M)): array of template timestreams 
        from each iteration, where the first index corresponds to the initial 
        template produced from start_idx.
    y (np.ndarray, float64, (L,)): final filtered timestream.
    h (np.ndarray, float64, (nfft,)): final optimal filter timestream. 
    start_idx (np.ndarray, int64, (K_final,)): final start indices of signal 
        instances.
    """
    s = np.asarray(s, dtype = np.float64)
    start_idx = np.asarray(start_idx, dtype = np.int64)

    # create amplitude array and populate the first row with initial template
    ai = build_template(s, start_idx) # initial template
    a = np.empty((N_iter + 1, len(ai)), dtype = np.float64)
    a[0] = ai   

    # get nfft
    nfft = get_nfft(a.shape[1]) 

    # create SJ
    SJ = create_nsd(j, nfft)
    
    # iterate 
    pbar = range(N_iter) 
    if verbose:
        pbar = tqdm(pbar, total = N_iter, leave = False, desc = "OF Iteration")
    for iter_idx in pbar:
        # Create and apply optimal filter
        h = create_opt_filt(a[iter_idx, :], SJ, nfft)
        y = apply_opt_filt(s, h)

        # find new starts and construct new template 
        start_idx = get_start_idx(y)
        a[iter_idx + 1, :] = build_template(s, start_idx)

    return a, y, h, start_idx

################################################################################
#################################### Utility ###################################
################################################################################
def get_nfft(N):
    """
    Get optimal FFT length given the length of template a.

    Parameters:
    N (int): length of the template timestream. 

    Returns:
    nfft (int): optimal FFT length for fast computation.
    """
    assert isinstance(N, (int, np.integer)), "N must be an integer."
    assert N > 0, "N must be positive."
    return pyfftw.next_fast_len(2 * N - 1)