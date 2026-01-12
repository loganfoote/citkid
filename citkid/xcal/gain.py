import numpy as np
import warnings

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
    # Input validation
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    p_amp = np.asarray(p_amp, dtype = np.float64)
    p_phase = np.asarray(p_phase, dtype = np.float64)

    # Remove amplitude and phase
    z_rmvd = z / 10 ** (np.polyval(p_amp, f) / 20)
    z_rmvd = z_rmvd / np.exp(1j * np.polyval(p_phase, f))
    return z_rmvd

def get_res_mask(fg, fr_spans):
    """
    Creates a mask for cutting resonances out of a gain sweep.

    Parameters:
    fg (np.array, float64): gain sweep frequency data in Hz. Must be sorted in 
        ascending order.
    fr_spans (list): values are tuples (float64, float64) where the first value.
        is the resonant frequency in Hz and the second is the span. These
        frequency ranges are removed from the gain data.

    Returns:
    mask (np.array, bool): mask for f where False values are resonances to cut
        from the data.
    """
    # Input validation
    fg = np.asarray(fg, dtype = np.float64)
    if not np.all(fg[:-1] <= fg[1:]):
        raise ValueError('fg must be sorted in ascending order.')
    if len(fr_spans) and \
        not all(a[0] <= b[0] for a, b in zip(fr_spans, fr_spans[1:])):
        m = 'fr_spans must be sorted in ascending order. Make sure '
        m += 'fres_all and qres_all are sorted.'
        raise ValueError(m)
    
    for c, s in fr_spans:
        if not isinstance(c, (int, float)): 
            raise ValueError('Resonant frequency must be numeric')
        if not isinstance(s, (int, float)):
            raise ValueError('Span must be numeric')
        if s < 0:
            raise ValueError('Span must be positive')
    
    # Cut fr_spans that are outside fg range and create intervals
    intervals = [] 
    for c, s in fr_spans: # assumes fr_spans is sorted by resonant frequency
        start, end = c - s / 2, c + s / 2 
        if end < fg[0] or start > fg[-1]: # assumes fg is sorted ascending
            continue
        intervals.append((start, end))

    # Merge overlapping intervals
    merged = [] # merge intervals
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    # Create mask for frequencies outside merged intervals
    mask = np.zeros_like(fg, dtype = bool)
    for start, end in merged:
        mask |= (fg >= start) & (fg <= end)
    mask = ~mask
    return mask

def fit_gain(f, z, fr_spans):
    """
    Fits the amplitude and phase of gain data. Amplitude is fit to a 2nd order.
    polynomial and phase is fit to a 1st order polynomial.

    Parameters:
    f (np.array, float64, (N,)): gain frequency array in Hz.
    z (np.array, complex128, (N,)): gain complex S21 array.
    fr_spans (list): values are tuples (float64, float64) where the first value.
        is the resonant frequency in Hz and the second is the span. These
        frequency ranges are removed from the gain data.

    Returns:
    p_amp (np.array, float64): 2nd-order polynomial fit parameters to gain
        amplitude.
    p_phase (np.array, float64): 1st-order polynomial fit parameters to gain
        phase.
    mask (np.array, bool): mask for cutting resonances from f and z.
    """
    # Input validation
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)

    shape_check = (f.shape == z.shape)
    shape_check = shape_check or (f.shape == tuple()) or (z.shape == tuple())
    if not shape_check:
        raise ValueError('f and z must be the same length')
    if len(f) < 4:
        raise ValueError('len(f) must be at least 4')

    for r in fr_spans:
        if not(len(r) == 2):
            raise ValueError('Incorrect fr_spans format')
        if r[1] < 0:
            raise ValueError('Span must be positive')

    # Cut out resonances
    mask = get_res_mask(f, fr_spans)
    f, z = f[mask], z[mask]
    # Find indices in the center of False regions of the mask for phase fitting
    false_groups = np.flatnonzero(np.diff(np.r_[True, ~mask, True]))
    cut_ixs = false_groups.reshape(-1, 2)

    # Convert to dB, phase
    dB = 20 * np.log10(np.abs(z))
    phase = np.unwrap(np.angle(z))

    # Fit
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
    return p_amp, p_phase, mask

def make_fr_spans(fres_all, qres_all):
    """
    Makes resonance frequency spans for cutting resonances out of gain data. 

    Parameters:
    fres_all (np.array, float64, (M,)): all resonant frequencies in Hz. Must be 
        sorted in ascending order for get_res_mask.
    qres_all (np.array, float64, (M,)): all resonator quality factors.

    Returns:
    fr_spans (list): values are tuples (float64, float64) where the first value.
        is the resonant frequency in Hz and the second is the span. These
        frequency ranges are removed from the gain data.
    """
    # Input validation
    fres_all = np.asarray(fres_all, dtype = np.float64)
    qres_all = np.asarray(qres_all, dtype = np.float64)
    if not np.all(qres_all > 0):
        raise ValueError('All quality factors must be positive') 
    if fres_all.shape != qres_all.shape:
        raise ValueError('fres_all and qres_all must be the same length')
    if any(fres_all[1:] < fres_all[:-1]):
        raise ValueError('fres_all must be sorted in ascending order')
    
    # Make resonance frequency spans
    fr_spans = [(fr, fr / qr) for fr, qr in zip(fres_all, qres_all)]
    return np.asarray(fr_spans, dtype = np.float64)
