"""
IQ-data utilities for citkid.

Functions
---------
density_subsample
    Subsample a complex IQ array while preserving density in sparse tail
    regions, using PCA-guided inverse-density weighting.
"""

import numpy as np


def density_subsample(z, n_keep=5000, n_bins=100, seed=0):
    """
    Subsample a complex IQ array to n_keep points, preserving sparse tail
    regions via inverse-density weighting along the principal variance axis.

    Unlike uniform random sampling, this retains roughly equal representation
    across the full signal range, so rare excursions (photon events, glitches,
    calibration tails) are kept even in very long timestreams. The projection
    axis is found by PCA, so the method works for raw gain-removed data,
    centred/rotated IQ streams, or any other complex-valued series without
    array-specific tuning.

    Parameters:
    z (array-like, complex128): Input IQ timestream or sweep data.
    n_keep (int): Maximum number of output points. If len(z) <= n_keep the
        input is returned unchanged. Default 5000.
    n_bins (int): Number of histogram bins for density estimation along the
        principal axis. Default 100.
    seed (int): Seed for the NumPy random-number generator. Default 0.

    Returns:
    z_sub (np.ndarray, complex128): Subsampled array of length
        min(len(z), n_keep).
    """
    z = np.asarray(z, dtype=np.complex128)
    if len(z) <= n_keep:
        return z

    # Project onto 1st principal component (direction of maximum variance).
    X  = np.column_stack([z.real, z.imag])
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc          # 2×2; normalisation not needed for eigenvector
    _, eigvecs = np.linalg.eigh(cov)   # eigenvalues in ascending order
    proj = Xc @ eigvecs[:, -1]          # project onto largest-variance axis

    # Inverse-density weights via 1-D histogram along the projection axis.
    counts, edges = np.histogram(proj, bins=n_bins)
    bin_idx = np.clip(np.digitize(proj, edges[:-1]) - 1, 0, n_bins - 1)
    weights = 1.0 / np.maximum(counts[bin_idx], 1).astype(np.float64)
    weights /= weights.sum()

    rng  = np.random.default_rng(seed)
    isub = rng.choice(len(z), size=n_keep, replace=False, p=weights)
    return z[isub]
