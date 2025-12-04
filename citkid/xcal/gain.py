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
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    p_amp = np.asarray(p_amp, dtype = np.float64)
    p_phase = np.asarray(p_phase, dtype = np.float64)

    z_rmvd = z / 10 ** (np.polyval(p_amp, f) / 20)
    z_rmvd = z_rmvd / np.exp(1j * np.polyval(p_phase, f))
    return z_rmvd

def get_res_mask(f, fr_spans):
    """
    Creates a mask for cutting resonances out of a gain sweep. 

    Parameters:
    f (np.array, float64): gain sweep frequency data in Hz. 
    fr_spans (list): values are tuples (float64, float64) where the first value.
        is the resonant frequency in Hz and the second is the span. These 
        frequency ranges are removed from the gain data.

    Returns:
    mask (np.array, bool): mask for f where False values are resonances to cut
        from the data. 
    """
    f = np.asarray(f, dtype = np.float64)
    for c, s in fr_spans:
        assert isinstance(c, (int, float)), 'Resonant frequency must be numeric'
        assert isinstance(s, (int, float)), 'Span must be numeric'
        if s < 0:
            raise ValueError('Span must be positive')
        
    ### Calculate resonance mask
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
    
    # Cut out resonances
    mask = get_res_mask(f, fr_spans)
    f, z = f[mask], z[mask]
    # Find indices in the center of False regions of the mask for phase fitting
    false_groups = np.flatnonzero(np.diff(np.r_[True, ~mask, True]))
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
    return p_amp, p_phase, mask

def make_fr_spans(fres_all, qres_all, fg):
    """
    Makes resonance frequency spans for cutting resonances out of gain data.

    Parameters:
    fres_all (np.array, float64, (M,)): all resonant frequencies in Hz.
    qres_all (np.array, float64, (M,)): all resonator quality factors.
    fg (np.array, float64, (N,)): gain frequency data in Hz.

    Returns:
    fr_spans (list): values are tuples (float64, float64) where the first value.
        is the resonant frequency in Hz and the second is the span. These 
        frequency ranges are removed from the gain data.
    """
    fres_all = np.asarray(fres_all, dtype = np.float64)
    qres_all = np.asarray(qres_all, dtype = np.float64)
    m = 'fres_all and qres_all must be the same length'
    assert fres_all.shape == qres_all.shape, m
    fg = np.asarray(fg, dtype = np.float64)

    fr_spans = []
    for fr, qr in zip(fres_all, qres_all):
        if qr <= 0:
            continue
        span = fr / qr
        # ensure span is in fg range
        if (fr + span / 2 < fg[0]) or (fr - span / 2 > fg[-1]):
            continue
        fr_spans.append((fr, span))
    return fr_spans