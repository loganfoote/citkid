import numpy as np
from tqdm.auto import tqdm
from .timestream_filter import lowpass_filter, highpass_filter

def find_common_modes(x, N_comp, N_iter, dt, lowpass_f, lowpass_order,
                      highpass_params):
    """
    Iteratively find common modes in multiple timestreams. In each iteration,
    performs a PCA normalized by the variance of the timestreams with the
    common signal removed from the previous iteration

    Parameters:
    x (array-like, N X T): timestream data with N timestreams of length T
    N_comp (int): number of components
    N_iter (int): number of iterations
    dt (float): sample time in s
    lowpass_f (float): lowpass filter frequency in Hz
    lowpass_order (int): lowpass filter order
    highpass_params: (frequency, order) -> (float, int): performs a highpass
        filter on the timestreams before calculating a and sigma, but not in the
        calulation of A. Parameters are
            frequency: highpass filter frequency in Hz
            order: highpass filter order
        The highpass filter improves the performance when dealing with
        uncorrelated noise that contains a 1/f component, but filtering out the
        low-frequency noise until it is approximately white. The 1/f correlated
        noise is preserved in A.

    Returns:
    a (array-like, N X N_comp): scaling factors from A -> x
    A (array-like, N_comp X T): common modes
    sig_iter (array-like, N_iter X N X N_comp): array of sigma values for each
        iteration
    """
    x = np.asarray(x).copy()
    x -= np.mean(x, axis = 1)[:, np.newaxis]
    sig = calculate_sigma(x)
    sig_iter = np.empty((N_iter, x.shape[0], N_comp), dtype = float)
    pbar = tqdm(range(N_iter), total = N_iter, leave = False)
    for i in pbar:
        a, A0 = pca(x, N_comp, sig, (dt, highpass_params[0], highpass_params[1]))
        A = lowpass_filter(A0, dt, lowpass_f, lowpass_order)
        y = highpass_filter(x - a @ A, dt, *highpass_params)
        sig = calculate_sigma(y)
        sig_iter[i] = sig ** 2
    return a, A0, sig_iter

def calculate_sigma(x):
    """
    Given timestreams x, calculates normalized variances for each timestream

    Parameters:
    x (array-like, N X T): timestream data with N timestreams of length T

    Returns:
    sig (np.array, N): variance normalized by the timestream length
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
    index (max T), and c is the component index (max N_comp)

    Parameters:
    x (array-like, N X T): timestream data with N timestreams of length T
    N_comp (int): number of common components to calculate
    sig (array-like, N) or None: normalized variance array. If None, calculates
        sigma using calculate_sigma
    highpass_params: (dt, frequency, order) -> (float, float, int) or None: if
        not None, performs a highpass filter on the timestreams before
        calculating a and sigma, but not in the calulation of A. Parameters are
            dt: timestream sample rate in seconds
            frequency: highpass filter frequency in Hz
            order: highpass filter order
        The highpass filter improves the performance when dealing with
        uncorrelated noise that contains a 1/f component, but filtering out the
        low-frequency noise until it is approximately white. The 1/f correlated
        noise is preserved in A.

    Returns:
    a (array-like, N X N_comp): scaling factors from A -> x
    A (array-like, N_comp X T): common modes
    """
    x = np.asarray(x)
    N, T = x.shape
    if highpass_params is not None:
        x_filt = highpass_filter(x, *highpass_params)
    else:
        x_filt = x.copy()
    if sig is None:
        sig = calculate_sigma(x_filt)
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
    Calculate the values a that minimize |x - a @ A|^2

    Parameters:
    x (np.array, N X T): array of N timestreams of length T
    A (np.array, C X T): array of C common modes of length T

    Returns:
    a (np.array, N X C): array of N scaling factors for each of C common modes
        that minimize |x - a @ A|^2
    """
    cov = np.sum(x[:, :, np.newaxis] * A.T[np.newaxis, :, :], axis = 1)
    a = cov / np.sum(np.real(A) ** 2, axis = 1)
    return a
