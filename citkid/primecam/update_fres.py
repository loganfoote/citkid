import numpy as np
def update_fres(f, z, npoints, fcal_indices = [], method = 'mins21',
                    cut_other_resonators = False, fres = None, Qres = None):
    """
    Give a multitone rough sweep dataset, return the updated resonance
    frequencies

    Parameters:
    f (np.array): frequency data in Hz
    z (np.array): complex S21 data
    npoints (int): number of points per tone. Each group of npoints in f, z
        is the s21 sweep for each tone
    fcal_indices (list): list of calibrations tone indices to not update
    method (str): 'mins21' to update using the minimum of |S21|. 'spacing' to
        update using the maximum spacing between adjacent IQ points. 'distance'
        to update using the point of furthest distance from the off-resonance
        point. 'none' to return the center of each sweep.
    cut_other_resonators (bool): if True, cuts other resonators out of each
        sweep before updating the tone. Other resonators are defined using
        fres and Qres
    fres (np.array or None): list of resonant frequencies in Hz if
        cut_other_resonators, or None
    Qres (np.array or None): list of quality factors to cut if
        cut_other_resonators, or None. Cuts spans of fres / Qres from each
        sweep

    Returns:
    fres_new (np.array): array of updated frequencies in Hz
    """
    if method == 'none':
        fres = []
        for i in range(len(f) // npoints):
            i0, i1 = npoints * i, npoints * (i + 1)
            fi = f[i0:i1]
            fres.append(np.mean(fi))
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
    for i in range(len(f) // npoints):
        i0, i1 = npoints * i, npoints * (i + 1)
        fi, zi = f[i0:i1], z[i0:i1]
        if i not in fcal_indices:
            if cut_other_resonators:
                spans = fres / Qres
                fi, zi = cut_fine_scan(fi, zi, fres, spans)
            fres_new.append(update(fi, zi))
        else:
            fres_new.append(np.mean(fi))
    return np.array(fres_new)

def update_fr_minS21(fi, zi):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the minimum of |S21| with a linear fit subtracted

    Parameters:
    fi (np.array): Single resonator frequency data
    zi (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    dB = 20 * np.log10(abs(zi))
    dB0 = dB - np.polyval(np.polyfit(fi, dB, 1), fi)
    ix = np.argmin(dB0)
    fr = fi[ix]
    return fr

def update_fr_spacing(fi, zi):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the max spacing between adjacent IQ points

    Parameters:
    fi (np.array): Single resonator frequency data
    zi (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    spacing = np.abs(zi[1:] - zi[:-1])
    spacing = spacing[1:] + spacing[:-1]
    spacing = np.concatenate([[0],spacing])
    spacing = np.concatenate([spacing, [0]])
    ix = np.argmax(spacing)
    fr = fi[ix]
    return fr

def update_fr_distance(fi, zi):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the furthest point from the off-resonance data

    Parameters:
    fi (np.array): Single resonator frequency data
    zi (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    offres = np.mean(list(zi[:10]) + list(zi[-10:]))
    diff = abs(zi - offres)
    ix = np.argmax(diff)
    fr = fi[ix]
    return fr

def cut_fine_scan(fi, zi, fres, spans):
    """
    Cuts resonant frequencies out of a single set of fine scan data

    Parameters:
    fi, zi (np.array, np.array): fine scan frequency in Hz and complex S21 data
    fres (np.array): array of frequencies to cut in Hz
    spans (np.array): array of frequency spans in Hz to cut
    """
    ix = (fres <= max(fi)) & (fres >= min(fi))
    fres, spans = fres[ix], spans[ix]
    for fr, sp in zip(fres, spans):
        if abs(fr - np.mean(fi)) > 1e3:
            ix = abs(fi - fr) > sp
            fi, zi = fi[ix], zi[ix]
    return fi, zi
