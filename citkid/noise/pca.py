import numpy as np

import warnings 

warnings.warn(
    "citkid.noise.pca is deprecated and will be removed in version 1.0.0"
    "Please use citkid.xcal.corr module for correlated noise removal.",
    DeprecationWarning,
    stacklevel=2,
)

def pca(y, n_components = 5):
    """
    Removes the largest principal components from a noise dataset using single
    value decomposition

    Parameters:
    y (array-like): array of timestream datasets, where each value is an array
        of timestream data
    n_components (int): number of principal components to remove

    Returns:
    z (np.array): array of timestream datasets corresponding to y, with
        principal components removed
    U (np.array): SVD unitary array, scaled output of np.linalg.svd
    S (np.array): SVD vector with singular values, scaled output of
        np.linalg.svd
    Vh (np.array): SVD unitary array, scaled output of np.linalg.svd
    """
    # Normalize input data
    y = np.array(y).T
    mean = np.mean(y, axis=0)
    y_normalized = y - mean
    std_dev = np.std(y_normalized, axis=0)
    y_normalized /= std_dev
    y_normalized = y_normalized.T
    # Perform SVD
    U, S, Vh = np.linalg.svd(y_normalized, full_matrices = False)
    S_rmvd = S.copy()
    S_rmvd[0:n_components] = 0. # set the components to remove to zero
    # Reconstruct data with modes removed
    z_normalized = (U * S_rmvd) @ Vh
    # Remove normalization
    z = ((z_normalized.T * std_dev) + mean).T
    # un-normalize S
    S = S * std_dev + mean
    return z, (U, S, Vh)

def get_common_mode(U, S, Vh, resonator_index, mode_index):
    """
    Extract a single common mode timestream for the given resonator after
    performing a principal component analysis using the SVD method.

    Parameters:
    U (np.array): SVD unitary array, scaled output of np.linalg.svd
    S (np.array): SVD vector with singular values, scaled output of
        np.linalg.svd
    Vh (np.array): SVD unitary array, scaled output of np.linalg.svd
    resonator_index (int): resonator index to scale the common mode
    mode_index (int): index of the common mode to extract

    Returns:
    m (np.array): common mode timestream for given resonator and mode index
    """
    A = (U * S[mode_index])
    m = A[resonator_index, mode_index] * Vh[mode_index, :]
    return m
