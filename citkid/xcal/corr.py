import numpy as np
from tqdm.auto import tqdm
from ..signal.basic_filt import lowpass_filter, highpass_filter

################################################################################
############################## PCA and utilities ###############################
################################################################################
def calc_sig(x):
    """
    Given timestreams x, calculates normalized variances for each timestream.

    Parameters:
    x (array-like, (N, T)): Timestream data with N timestreams of length T.

    Returns:
    sig (np.array, (N, 1)): Variance normalized by timestream length.
    """
    x = np.asarray(x)
    return np.sqrt(np.sum(x ** 2, axis = 1)[:, np.newaxis] / x.shape[1])

def pca(x, N_comp, sig = None, highpass_params = None):
    """
    Calculates common-mode components given timestreams and variances. Assumes
    the individual timestreams are white:
        x_kt = a_kc * A_ct + n_kt
    where a_kc are scaling factors, A_ct are the common components, and n_kt are
    white noise timestreams. k is the detector index (max N), t is the time
    index (max T), and c is the component index (max N_comp).

    Parameters:
    x (array-like, (N, T)): Timestream data with N timestreams of length T.
    N_comp (int): Number of common components to calculate.
    sig (array-like, (N, 1) or None): Normalized variance array. If None,
        calculates sigma using `calc_sig`.
    highpass_params (tuple[float, float, int] or None): (dt, frequency, order).
        If not None, performs a highpass filter on the timestreams before
        calculating a and sigma, but not in the calculation of A.
        dt: sample time in seconds.
        frequency: highpass filter frequency in Hz.
        order: highpass filter order.
        The highpass filter improves performance for 1/f noise by whitening
        low-frequency components while preserving correlated noise in A.

    Returns:
    a (array-like, (N, N_comp)): Scaling factors from A -> x.
    A (array-like, (N_comp, T)): Common modes.
    """
    # Input validation 
    x = np.asarray(x)
    N, T = x.shape
    if not (0 <= N_comp <= N):
        raise ValueError("N_comp must be between 0 and N")
    
    # Apply highpass filter if specified
    if highpass_params is not None:
        if len(highpass_params) != 3:
            raise ValueError("highpass_params must be (dt, frequency, order)")
        x_filt = highpass_filter(x, *highpass_params)
    else:
        x_filt = x.copy()

    # Calculate sigma if not provided
    if sig is None:
        sig = calc_sig(x_filt)
    else:
        if sig.shape != (x.shape[0], 1):
            raise ValueError("sig must have shape (N, 1)")
        
    # Perform PCA
    x0 = x_filt / sig
    C = x0 @ x0.T / T
    eigval, eigvec = np.linalg.eig(C)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    v = eigvec[:, np.flip(np.argsort(eigval))[:N_comp]]
    a = sig * v
    A = a.T @ (x / sig**2)
    return a, A

def calc_a(x, A):
    """
    Calculate the values a that minimize |x - a @ A|^2.

    Parameters:
    x (np.array, (N, T)): Array of N timestreams of length T.
    A (np.array, (C, T)): Array of C common modes of length T.

    Returns:
    a (np.array, (N, C)): Scaling factors for each of C common modes that
        minimize ||x - a @ A||^2.
    """
    cov = np.sum(x[:, :, np.newaxis] * A.T[np.newaxis, :, :], axis = 1)
    a = cov / np.sum(np.real(A) ** 2, axis = 1)
    return a

################################################################################
####################### Iterative common mode algorithm ########################
################################################################################
def calc_cm(x, N_comp, N_iter, dt, lowpass_params, highpass_params, 
            verbose = True):
    """
    Iteratively find common modes in multiple timestreams. In each iteration,
    performs a PCA normalized by the variance of the timestreams with the
    common signal removed from the previous iteration.

    Parameters:
    x (array-like, (N, T)): Timestream data with N timestreams of length T.
    N_comp (int): Number of components.
    N_iter (int): Number of iterations.
    dt (float): Sample time in seconds.
    lowpass_params (tuple[float, int]): (frequency, order) for the lowpass
        filter applied to the common modes for convergence.
        frequency: lowpass filter frequency in Hz.
        order: lowpass filter order.
    highpass_params (tuple[float, int] or None): (frequency, order) for a
        highpass filter applied to the timestreams before calculating a and
        sigma (but not in the calculation of A).
        frequency: highpass filter frequency in Hz.
        order: highpass filter order.
    verbose (bool): If True, shows a progress bar while iterating.

    Returns:
    a (array-like, (N, N_comp)): Scaling factors from A -> x.
    A (array-like, (N_comp, T)): Common modes.
    sig_iter (array-like, (N_iter, N, N_comp)): Sigma values per iteration.
    a_full (array-like, (N, N)): Scaling factors from all components A -> x.
    """
    # Input validation
    x = np.asarray(x)

    if len(x.shape) != 2:
        raise ValueError("x must be 2D array-like")
    if N_iter <= 0:
        raise ValueError("N_iter must be positive")
    if len(lowpass_params) != 2:
        raise ValueError("lowpass_params must be (frequency, order)")
    highpass_check = highpass_params is None or len(highpass_params) == 2
    if not highpass_check:
        raise ValueError("highpass_params must be (frequency, order)")

    # Remove the mean from each timestream
    x = x - np.mean(x, axis = 1)[:, np.newaxis]

    # Calculate initial sigma
    sig = calc_sig(x)
    sig_iter = np.empty((N_iter, x.shape[0], N_comp), dtype = float)

    # Run iterative algorithm
    pbar = range(N_iter)
    if verbose:
        pbar = tqdm(pbar, total = N_iter, leave = False)
    for i in pbar: 
        if i == N_iter - 1:
            # For last iteration, calculate a_full
            a_full, A0 = pca(x, x.shape[0], sig, (dt, highpass_params[0], 
                                                  highpass_params[1]))
            a = a_full[:, :N_comp]
            A0 = A0[:N_comp, :]
        else:
            a, A0 = pca(x, N_comp, sig, (dt, highpass_params[0], 
                                         highpass_params[1]))
        A = lowpass_filter(A0, dt, *lowpass_params)
        y = highpass_filter(x - a @ A, dt, *highpass_params)
        sig = calc_sig(y)
        sig_iter[i] = sig ** 2
    return a, A0, sig_iter, a_full

def calc_cm_complex(z, theta = None, *calc_cm_params):
    """
    Given complex timestreams, calculates the complex scaling factors and 
    common modes for independent real/imaginary components using the iterative 
    common mode algorithm.

    Parameters:
    z (array-like, (N, T)): Complex gain-removed timestream data with N
        timestreams of length T.
    theta (array-like, (N,) or None): Rotation angles to rotate each timestream
        to the real axis. If None, uses the angle of the median of each
        timestream.
    *calc_cm_params: Parameters forwarded to `calc_cm`.

    Returns:
    aI, aQ (array-like, (N, N_comp)): Real and imaginary scaling factors from
        A -> z.
    AI, AQ (array-like, (N_comp, T)): Real and imaginary common modes.
    sigI_iter, sigQ_iter (array-like, (N_iter, N, N_comp)): Real and imaginary
        sigma values for each iteration.
    aI_full, aQ_full (array-like, (N, N)): Real and imaginary scaling factors
        from all components A.
    theta (array-like, (N,)): Rotation angles used to rotate each timestream to
        the real axis.
    """
    # Input validation 
    z = np.asarray(z, dtype = np.complex128)

    # Rotate each timestream median to real axis
    if theta is None:
        theta = np.angle(np.median(z, axis = 1))
    z = z * np.exp(-1j * theta[:, np.newaxis]) 

    # Calculate real and imaginary common modes
    aI, AI, sigI_iter, aI_full =\
        calc_cm(z.real, *calc_cm_params)
    
    aQ, AQ, sigQ_iter, aQ_full =\
        calc_cm(z.imag, *calc_cm_params)
    
    return aI, aQ, AI, AQ, sigI_iter, sigQ_iter, aI_full, aQ_full, theta

################################################################################
############################# common mode removal ##############################
################################################################################
def remove_cm(x, a, A, idx):
    """
    Removes common modes from a timestream.
    
    Parameters:
    x (array-like, (T,) or (N, T)): Single timestream data of length T, or
        (N, T) for N timestreams.
    a (array-like, (N, C)): Scaling factors from A -> x.
    A (array-like, (C, T)): Common modes.
    idx (array-like, (M,) or int): Indices into a corresponding to x.
    
    Returns:
    y (array-like, (N, T)): Timestreams with common modes removed.
    """
    # Input validation
    x = np.asarray(x)
    a = np.asarray(a)
    A = np.asarray(A)
    if len(x.shape) not in (1, 2):
        raise ValueError("x must be 1D or 2D array-like")
    if not isinstance(idx, (int, np.integer)):
        idx = np.asarray(idx, dtype = np.int32)
        if not len(idx) <= x.shape[0]:
            raise ValueError("idx length exceeds x length")
        if not np.all((0 <= idx) & (idx < a.shape[0])):
            raise ValueError("idx out of bounds")
    else:
        if not (0 <= idx < a.shape[0]):
            raise ValueError("idx out of bounds")
        m = "x must be 1D or have a single row when idx is int"
        if len(x.shape) != 1:
            raise ValueError(m)
    if len(x.shape) == 1:
        N = 1
        T = x.shape[0]
    else:
        N, T = x.shape
    if a.shape[0] < N:
        raise ValueError("a must have at least N rows")
    if a.shape[1] != A.shape[0]:
        raise ValueError("a and A shape mismatch")
    if A.shape[1] != T:
        raise ValueError("A and x shape mismatch")
    
    # Remove common modes 
    y = x - a[idx, :] @ A
    return y

def remove_cm_complex(z, aI, aQ, AI, AQ, idx, theta = None):
    """
    Removes common modes from complex timestreams. Rotates z to the real axis,
    then undoes the rotation after removing the common modes.
    
    Parameters:
    z (array-like, (T,) or (N, T)): Single complex timestream data of length T,
        or (N, T) for N timestreams.
    aI, aQ (array-like, (N, C)): Real and imaginary scaling factors from A -> z.
    AI, AQ (array-like, (C, T)): Real and imaginary common modes.
    idx (array-like, (M,) or int): Indices into aI/aQ corresponding to z.
    theta (array-like, (N,) or None): Rotation angles to rotate each timestream
        to the real axis. If None, uses the angle of the median of each
        timestream.
    Returns:
    y (array-like, (N, T)): Complex timestreams with common modes removed.
    theta (array-like, (N,)): Rotation angles used to rotate each timestream to
        the real axis.
    """
    # Input validation
    z = np.asarray(z, dtype = np.complex128)

    # Rotate each timestream median to real axis
    if theta is None:
        if len(z.shape) == 1:
            theta = np.angle(np.median(z))
        else:
            theta = np.angle(np.median(z, axis = 1))[:, np.newaxis]
    z = z * np.exp(-1j * theta)

    # Calculate common modes
    y_real = remove_cm(z.real, aI, AI, idx)
    y_imag = remove_cm(z.imag, aQ, AQ, idx)
    y = y_real + 1j * y_imag 

    # Undo rotation
    y = y * np.exp(1j * theta)
    
    # Format theta for return
    if len(theta.shape) == 2:
        theta = theta[:, 0]
    return y, theta

