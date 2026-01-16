import numpy as np
from .plot import plot_update_fres

# Need to update docstrings, imports
def update_fres(fs, zs, fres, qres, fcal_indices, method = 'distance',
                cable_delay = 0, plotq = False, res_indices = None, 
                plot_directory = ''):
    """
    Update resonant frequencies given fine sweep data

    Parameters:
    fs (array-like): fine sweep frequency data in Hz for each resonator in fres
    zs (array-like): fine sweep complex S21 data for each resonator in fres
    fcal_indices (array-like): list of calibrations tone indices (index into
        fs, zs, fres, Qres). Calibration tone frequencies will not be updated
    method (str): 'mins21' to update using the minimum of |S21|. 'spacing' to
        update using the maximum spacing between adjacent IQ points. 'distance'
        to update using the point of furthest distance from the off-resonance
        point. 'none' to return fres
    cable_delay (float): cable delay in seconds to remove before updating the 
        frequency 
    fres (np.array or None): list of resonant frequencies in Hz
    qres (np.array or None): list of quality factors to cut if
        cut_other_resonators, or None. Cuts spans of fres / Qres from each
        sweep
    plotq (bool): If True, plots all of the updated frequencies in batches
    res_indices (array-like): resonator indices for plotting
    plot_directory (str): directory to save plots

    Returns:
    fres_new (np.array): array of updated frequencies in Hz
    """
    fs, zs = np.asarray(fs), np.asarray(zs)
    zs *= np.exp(1j * 2 * np.pi * fs * cable_delay)
    fres, qres = np.asarray(fres), np.asarray(qres)
    fcal_indices = np.asarray(fcal_indices)
    if method == 'none':
        fres = [np.mean(f) for f in fres]
        return np.array(fres)
    elif method == 'mins21':
        update = update_fr_minS21
    elif method == 'spacing':
        update = update_fr_spacing
    elif method == 'distance':
        update = update_fr_distance
    else:
        raise ValueError("method must be 'mins21', 'distance', or 'spacing'")
    fres_new = []
    for index, (fi, zi) in enumerate(zip(fs, zs)):
        if index not in fcal_indices:
            spans = fres / qres / 1.5
            fi, zi = cut_fine_sweep(fi, zi, fres, spans)
            fres_new.append(update(fi, zi))
        else:
            fres_new.append(np.mean(fi))
    if plotq:
        plot_update_fres(fs, zs, fres, fcal_indices, res_indices, cable_delay, plot_directory)
    return np.array(fres_new)

def update_fr_minS21(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the minimum of |S21| with a linear fit subtracted

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    dB = 20 * np.log10(abs(z))
    dB0 = dB - np.polyval(np.polyfit(f, dB, 1), f)
    ix = np.argmin(dB0)
    fr = f[ix]
    return fr

def update_fr_spacing(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the max spacing between adjacent IQ points

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    spacing = np.abs(np.diff(z))
    spacing = spacing[1:] + spacing[:-1]
    spacing = np.concatenate([[0],spacing, [0]])
    ix = np.argmax(spacing)
    fr = f[ix]
    return fr

def update_fr_distance(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the furthest point from the off-resonance data. This
    function will perform better if the cable delay is first removed.

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    if len(f) < 1:
        return np.nan
    offres = np.mean(np.roll(z, 10)[:20])
    diff = np.abs(z - offres)
    ix = np.argmax(diff)
    if len(f) > ix + 1:
        fr = np.mean([f[ix], f[ix + 1]])
    else:
        fr = f[ix]
    return fr

def cut_fine_sweep(f, z, fres, spans):
    """
    Cuts resonant frequencies out of a single set of fine sweep data

    Parameters:
    f, z (np.array, np.array): fine sweep frequency in Hz and complex S21 data
    fres (np.array): array of frequencies to cut in Hz
    spans (np.array): array of frequency spans in Hz to cut
    """
    ix = np.argmin(np.abs(fres - np.mean(f)))
    fr_keep, sp_keep = fres[ix], spans[ix]
    ix = (fres <= np.max(f)) & (fres >= np.min(f)) & (np.abs(fres - fr_keep) > 1)
    fres, spans = fres[ix], spans[ix]
    for fr, sp in zip(fres, spans):
        ix = (np.abs(f - fr) > sp) | (np.abs(f - fr_keep) < sp_keep)
        f, z = f[ix], z[ix]
    # Needs to leave the current sweep
    return f, z

def trim_nearby_resonances(f, z, fres, ires):
    """
    Trims a fine scan to remove nearby resonances.
    
    Parameters:
        f: fine scan frequencies
        z: fine scan complex S21
        fres: list of resonant frequencies
        ires: index of resonant frequency we want to keep
    Returns:
        ftrim: trimmed fine scan frequencies
        ztrim: trimmed fine scan complex S21
    """
    f1, f2 = min(f), max(f)
    if ires > 0:
        f1 = max(f1, (fres[ires-1]+fres[ires])/2)
    if ires < len(fres)-1:
        f2 = min(f2, (fres[ires+1]+fres[ires])/2)
    mask = (f>f1)&(f<f2)
    ftrim, ztrim = f[mask], z[mask]
    return ftrim, ztrim