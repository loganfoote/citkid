import numpy as np
from tqdm.auto import tqdm
from .timestream_filter import lowpass_filter

def find_common_modes(x, N_comp, N_iter, dt, lowpass_f, lowpass_order):
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
        a, A = pca(x, sig, N_comp)
        A = lowpass_filter(A, dt, lowpass_f, lowpass_order)
        sig = calculate_sigma(x - a @ A) ** 2
        sig_iter[i] = sig
    return a, A, sig_iter

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

def pca(x, sig, N_comp):
    """
    Calculates common-mode components given timestreams and variances. Assumes
    the individual timestreams are white:
        x_kt = a_kc * A_ct + n_kt
    where a_kc are scaling factors, A_ct are the common components, and n_kt are
    white noise timestreams. k is the detector index (max N), t is the time
    index (max T), and c is the component index (max N_comp)

    Parameters:
    x (array-like, N X T): timestream data with N timestreams of length T
    sig (array-like, N): normalized variance array
    N_comp (int): number of common components to calculate

    Returns:
    a (array-like, N X N_comp): scaling factors from A -> x
    A (array-like, N_comp X T): common modes
    """
    x = np.asarray(x)
    N, T = x.shape
    x0 = x / sig
    C = x0 @ x0.T / T
    eigval, eigvec = np.linalg.eig(C)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    v = eigvec[:, np.flip(np.argsort(eigval))[:N_comp]]
    a = sig * v
    A = a.T @ (x / sig**2)
    return a, A
