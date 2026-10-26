"""
Run the CNN resonance detector on a full VNA sweep: resample to a
constant fractional step, build the gain-insensitive input channels, run
the network in chunks, and turn heatmap peaks into candidates (frequency,
rough Qr, score).
"""
import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from ...res.multifit import block_average
from ..sweep_info import resample_fractional

# Chunking of long sweeps (see ``detect``); the network was trained on
# inputs prepared the same way
CHUNK_RATIO, CHUNK_OVERLAP, MIN_STEP = 1.1, 0.01, 1.5e-6

def detector_features(z, valid = None, noise_window = 257, compress = 3.):
    """
    Compute the gain-insensitive network input channels from a sweep.

    Uses the log-derivative d ln z = d ln|z| + j d arg z per sample: a
    smooth gain contributes little, while a resonance traces a fast loop.
    Each channel is divided by a running robust noise estimate (median of
    |d ln z| / sqrt(2 ln 2)) and compressed with asinh(x / compress).

    Parameters:
    z (np.array): complex data on a constant fractional grid.
    valid (np.array, bool, or None): False in gaps (zeroed). Default None
        (all valid).
    noise_window (int): running-median window in samples. Default 257.
    compress (float): asinh compression scale in noise units. Default 3.

    Returns:
    x (np.array): float32, shape (3, len(z)): normalized Re(d ln z),
        Im(d ln z), and the validity mask.
    """
    z = np.asarray(z, dtype = np.complex128)
    valid = np.ones(len(z), dtype = bool) if valid is None else valid
    absz = np.maximum(np.abs(z), 1e-300)
    dl = np.gradient(np.log(absz)) + 1j * np.gradient(np.unwrap(np.angle(z)))
    dl[~valid] = 0
    width = max(3, int(min(len(dl), noise_window)) | 1)
    # 'reflect' (not 'nearest'): repeating an edge value would let a single
    # edge sample dominate the median near the ends
    sig = median_filter(np.abs(dl), width, mode = 'reflect') \
        / np.sqrt(2 * np.log(2))
    w = dl / np.maximum(sig, 1e-12)
    x = np.stack((np.arcsinh(w.real / compress),
                  np.arcsinh(w.imag / compress), valid.astype(float)))
    return x.astype(np.float32)

def predict_maps(model, x, chunk = 65536, overlap = 4096, device = None):
    """
    Run the network over a long feature array in overlapping pieces.

    Parameters:
    model (UNet1D): trained network (eval mode is set here).
    x (np.array): features, shape (C, L).
    chunk (int): piece length (multiple of 2**depth). Default 65536.
    overlap (int): overlap on each side, discarded. Default 4096.
    device (str or None): torch device. Default None (the model's device).

    Returns:
    heat (np.array): heatmap probabilities, shape (L,).
    offset (np.array): sub-sample offsets, shape (L,).
    log_q (np.array): log10(Qr), shape (L,).
    """
    model.eval()
    device = device or next(model.parameters()).device
    m = 2 ** model.depth
    L = x.shape[1]
    outs = np.zeros((3, L), dtype = np.float32)
    start = 0
    with torch.no_grad():
        while start < L:
            a = max(0, start - overlap)
            b = min(L, start + chunk)
            seg = x[:, a:b]
            pad = (-seg.shape[1]) % m
            seg = np.pad(seg, ((0, 0), (0, pad)))
            h, o, q = model(torch.from_numpy(seg[None]).to(device))
            res = torch.stack((torch.sigmoid(h[0]), o[0], q[0])).cpu().numpy()
            outs[:, start:b] = res[:, start - a:b - a]
            start = b
    return outs[0], outs[1], outs[2]

def sweep_chunks(
    f, chunk_ratio = CHUNK_RATIO, overlap = CHUNK_OVERLAP, min_points = 64
):
    """
    Split a sweep into chunks for detection (see ``detect``).

    Parameters:
    f (np.array): frequencies in Hz, ascending.
    chunk_ratio (float or None): chunk span as a frequency ratio. None: one
        chunk. Default 1.1.
    overlap (float): fractional overlap of the chunks on each side.
        Default 0.01.
    min_points (int): chunks with fewer points are skipped (when split).
        Default 64.

    Returns:
    chunks (list of tuple): (lo, hi, last, mask) per chunk: the core range
        in Hz (candidates in [lo, hi), or [lo, hi] for the ``last`` chunk,
        belong to it; -inf, inf for a single chunk) and the boolean mask of
        the points used.
    """
    if chunk_ratio is None or f[-1] / f[0] <= chunk_ratio:
        return [(-np.inf, np.inf, True, np.ones(len(f), bool))]
    n = int(np.ceil(np.log(f[-1] / f[0]) / np.log(chunk_ratio)))
    edges = f[0] * (f[-1] / f[0]) ** (np.arange(n + 1) / n)
    out = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (f >= lo * (1 - overlap)) & (f <= hi * (1 + overlap))
        if m.sum() >= min_points:
            out.append((lo, hi, k == n - 1, m))
    return out

def detect(
    model, f, z, threshold = 0.3, min_dist = 2, chunk_ratio = CHUNK_RATIO,
    overlap = CHUNK_OVERLAP, min_step = MIN_STEP, device = None
):
    """
    Detect resonance candidates in a full VNA sweep with the network.

    Wide sweeps are processed in chunks spanning a factor ``chunk_ratio``
    in frequency, each resampled at its own fractional step (one global
    step would interpolate the low end of a wide uniform sweep, giving
    smoothed noise the network never saw in training). Chunks overlap by
    ``overlap`` (fractional) on each side to avoid edge effects. Data
    finer than ``min_step`` is block averaged down to about it first: the
    network was trained on 3-60 points per linewidth.

    Parameters:
    model (UNet1D): trained network (``model.load_model``).
    f (np.array): frequencies in Hz, ascending, no repeats.
    z (np.array): complex S21 data.
    threshold (float): heatmap peak threshold. Default 0.3.
    min_dist (int): minimum peak separation in samples. Default 2.
    chunk_ratio (float or None): chunk span as a frequency ratio. None:
        the sweep in one piece. Default 1.1.
    overlap (float): fractional overlap of the chunks. Default 0.01.
    min_step (float or None): fractional step below which data is block
        averaged. Default 1.5e-6 (30 points per linewidth at Qr 2e4).
        None disables.
    device (str or None): torch device. Default None (the model's).

    Returns:
    fr (np.array): candidate frequencies in Hz, ascending.
    Qr (np.array): rough candidate quality factors.
    score (np.array): heatmap peak values (0-1).
    """
    out = []
    for lo, hi, last, m in sweep_chunks(f, chunk_ratio, overlap):
        fc, zc = f[m], z[m]
        if min_step is not None and len(fc) > 2:
            k = int(min_step / np.median(np.diff(fc) / fc[:-1]))
            fc, zc = block_average(fc, zc, k)
        fg, zg, valid = resample_fractional(fc, zc)
        heat, offset, log_q = predict_maps(
            model, detector_features(zg, valid), device = device)
        idx, _ = find_peaks(heat, height = threshold, distance = min_dist)
        fr = fg[0] * (fg[1] / fg[0]) ** (idx + offset[idx])
        keep = (fr >= lo) & ((fr < hi) | last)
        out.append((fr[keep], 10 ** log_q[idx][keep], heat[idx][keep]))
    if not out:
        return np.array([]), np.array([]), np.array([])
    fr, q, s = (np.concatenate(v) for v in zip(*out))
    o = np.argsort(fr)
    return fr[o], q[o], s[o]
