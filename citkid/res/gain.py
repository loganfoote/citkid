import numpy as np
import warnings
from .plot import plot_gain_fit

def fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, frs = [], Qrs = [],
                              plotq = False):
    """
    Removes the gain-scan fit parameters from the fine scan data.

    Qrs should be no higher than 10 X Qr of the resonances.

    Parameters:
    fgain (np.array): gain sweep frequency data.
    zgain (np.array): gain sweep complex S21 data.
    ffine (np.array): fine sweep frequency data.
    zfine (np.array): fine sweep complex S21 data.
    frs (list of float): resonance frequencies to cut from the gain scan.
    Qrs (list of float): Qrs to cut from the gain scan.
    plotq (bool): If True, also plots fits to the gain scan
        and corrections to the fine-scan.

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB.
    p_phase (np.array): 1st-order polynomial fit parameters to phase.
    z_rmvd (np.array): zfine with gain amplitude and phase removed.
    fig, axs (pyplot figure and axes, or None): if plotq, returns a plot of the
        gain amplitude and phase fits. Otherwise, returns (None, None).
    """
    fgain, zgain = np.array(fgain), np.array(zgain)
    ffine, zfine = np.array(ffine), np.array(zfine)
    fr_spans = []
    for fr, Qr in zip(frs, Qrs):
        fr_spans.append((fr, fr / Qr))
    p_amp, p_phase, (fig_gain, axs_gain) = fit_gain(fgain, zgain, fr_spans, plotq)
    z_rmvd = remove_gain(ffine, zfine, p_amp, p_phase)
    return p_amp, p_phase, z_rmvd, (fig_gain, axs_gain)

def remove_gain(f, z, p_amp, p_phase):
    """
    Removes the gain amplitude and phase from complex S21 data, given the raw
    data

    Parameters:
    f (np.array, float64, (N,)): frequency data in Hz.
    z (np.array, complex128, (N,)): complex S21 data.
    p_amp (np.array, float64, (K,)): polynomial fit parameters to gain
        amplitude.
    p_phase (np.array, float64, (L,)): polynomial fit parameters to gain phase.

    Returns:
    z_rmvd (np.array, complex128, (N,)): complex S21 data with gain amplitude
        and phase removed.
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    p_amp = np.asarray(p_amp, dtype = np.float64)
    p_phase = np.asarray(p_phase, dtype = np.float64)

    z_rmvd = z / 10 ** (np.polyval(p_amp, f) / 20)
    z_rmvd = z_rmvd / np.exp(1j * np.polyval(p_phase, f))
    return z_rmvd

def fit_gain(f, z, fr_spans, plotq = False):
    """
    Fits the amplitude and phase of gain data. Amplitude is fit to a 2nd order.
    polynomial and phase is fit to a 1st order polynomial.

    Parameters:
    f (np.array, float64, (N,)): gain frequency array.
    z (np.array, complex128, (N,)): gain complex S21 array.
    fr_spans (list): values are tuples (float64, float64) where the first value.
        is the resonance frequency and the second is the span. These frequencies
        are removed from the gain data.
    plotq (bool): If True, plots the fits and returns the figure.

    Returns:
    p_amp (np.array, float64): 2nd-order polynomial fit parameters to gain
        amplitude.
    p_phase (np.array, float64): 1st-order polynomial fit parameters to gain
        phase.
    fig, axs (pyplot figure and axes, or None): if plotq, returns a plot of the
        gain amplitude and phase fits. Otherwise, returns (None, None).
    """
    ### Check parameters
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)

    shape_check = (f.shape == z.shape) 
    shape_check = shape_check or (f.shape == tuple()) or (z.shape == tuple())
    assert shape_check, 'f and z must be the same length'
    assert len(f) >= 4, 'len(f) must be at least 3'

    for r in fr_spans:
        if not(len(r) == 2):
            raise ValueError('Incorrect fr_spans format')
        if r[1] < 0:
            raise ValueError('Span must be positive')
    if plotq:
        f0, z0 = f.copy(), z.copy()

    ### Remove resonances from span data
    intervals = np.array([(c - s / 2, c + s / 2) for c, s in fr_spans])
    if intervals.shape[0]:
        intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [] # merge intervals
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    merged = np.array(merged)

    mask = np.zeros_like(f, dtype = bool)
    for start, end in merged:
        mask |= (f >= start) & (f <= end)
    f, z = f[~mask], z[~mask]
    # find indices in the center of False regions of the mask for phase fitting
    false_groups = np.flatnonzero(np.diff(np.r_[True, mask, True]))
    cut_ixs = false_groups.reshape(-1, 2)

    ### Convert to dB, phase
    dB = 20 * np.log10(np.abs(z))
    phase = np.unwrap(2 * np.angle(z)) / 2

    ### Fit
    try:
        p_amp = np.polyfit(f, dB, 2)
        # Fit to each cut portion of phase separately to avoid unwrapping issues
        pps, dlens = [], []
        for ix0, ix1 in cut_ixs:
            N = len(f[ix0:ix1])
            if N >= 4:
                pps.append(np.polyfit(f[ix0:ix1], phase[ix0:ix1], 1))
                dlens.append(N)
        if not len(pps):
            # Allow N = 2 if necessary
            for ix0, ix1 in cut_ixs:
                N = len(f[ix0:ix1])
                if N >= 2:
                    pps.append(np.polyfit(f[ix0:ix1], phase[ix0:ix1], 1))
                    dlens.append(N)
        # Choose phase fits with the highest number of points
        pps, dlens = np.asarray(pps), np.asarray(dlens)
        for i in range(101, 0, -10):
            pps0 = pps[dlens > i]
            if len(pps0):
                break
        pps0 = [p[np.isfinite(p)] for p in pps0]
        if len(pps0):
            p_phase = np.mean(pps0, axis = 0)
        else:
            p_phase = np.array([np.nan,np.nan])
            warnings.warn('No phase data to fit, returning NAN')
    except Exception as e:
        p_amp = np.array([np.nan,np.nan,np.nan])
        p_phase = np.array([np.nan,np.nan])
        warnings.warn('Gain fit failed, returning NAN')

    ### Plot
    if plotq:
        dB0 = 20 * np.log10(np.abs(z0))
        fig, axs = plot_gain_fit(f0, dB0, f, dB, phase, p_amp, p_phase)
    else:
        fig, axs = None, None
    return p_amp, p_phase, (fig, axs)
