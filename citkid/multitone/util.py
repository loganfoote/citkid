import warnings
warnings.warn(
    "citkid.multitone.util is legacy code and may be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import numpy as np

def make_cal_tones(fres, ares, qres, max_n_tones = 1000,
                   res_indices = None, fcal_power = -55,
                   fres_all = None):
    '''
    Adds calibration tones to the given resonator list. Fills in largest spaces
    between resonators, up to max_n_tones.

    Parameters:
    fres (np.array): frequency array in Hz
    ares (np.array): amplitude array in Hz
    qres (np.array): span factor array
    max_n_tones (int): maximum number of tones after adding cal tones
    res_indices (np.array or None): resonator indices corresponding to
        fres
    fcal_power (float): calibration tone power, in the same units as ares
    fres_all (array-like or None): if given, uses this array to determine
        calibration tone locations instead of fres. 

    Returns:
    fres, ares, qres (np.array): frequency, amplitude, and span factor arrays
        with calibration tones added
    fcal_indices (np.array): calibration tone indices into fres, ares, qres
    new_res_indices (np.array): new resonator index list with the new
        calibration tones. Calibration tone resonator indices are negative
    '''
    fres = np.asarray(fres, dtype = float)
    ares = np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    ix = np.argsort(fres)
    fres, ares, qres = fres[ix], ares[ix], qres[ix]
    if res_indices is not None:
        res_indices = np.asarray(res_indices)
        res_indices = res_indices[ix]
    if res_indices is None:
        res_indices = np.asarray(range(len(fres)))
    new_res_indices = np.asarray(res_indices, dtype = int)

    if fres_all is None:
        fres_all = fres
    ix = np.flip(np.argsort(np.diff(fres_all) / fres_all[:1]))[:max_n_tones - len(fres)]
    fcal = np.sort([np.mean(fres_all[i:i+2]) for i in ix])
    fcal_indices = np.searchsorted(fres, fcal)
    fcal_indices += np.asarray(range(len(fcal_indices)), dtype = int)
    for fcal_index, fres_index in enumerate(fcal_indices):
        fres = np.insert(fres, fres_index, fcal[fcal_index])
        ares = np.insert(ares, fres_index, fcal_power)
        qres = np.insert(qres, fres_index, np.inf)
        new_index = -fcal_index - 1
        new_res_indices = np.insert(new_res_indices, fres_index,
                                          new_index)
    return fres, ares, qres, fcal_indices, new_res_indices