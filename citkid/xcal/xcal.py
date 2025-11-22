import numpy as np 

################################################################################
##################### Cut fine sweep for fitting x vs theta ####################
################################################################################
def get_xcal_ix(ffine, tfine, tnoise, ix0_offset = 7, ix1_offset = 5,
                std_cutoff = 12):
    """
    Get incides of the fine s21 sweep over which x vs theta should be fit to 
    produce the x calibration. Chooses the indices where tfine (after glitch 
    removal) falls within the min and max values of tnoise, with optional 
    offsets.

    Parameters:
    ffine (array-like, float64): frequency values from the fine s21 sweep in Hz.
    tfine (array-like, float64): theta values from the fine s21 sweep.
    tnoise (array-like, float64): theta values from the noise measurement.
    ix0_offset (int): number of indices to offset below the start index.
        The data is sorted such that the start index corresponds to the lowest
        frequency and the end index corresponds to the highest frequency.
    ix1_offset (int): number of indices to offset above the end index.
    std_cutoff (float or None): number of standard deviations from the mean to 
            use as a cutoff to apply to tnoise before determining the max and 
            min values. If None, no cutoff is applied.

    Returns:
    ix (array-like, int64): indices of tfine.
    """
    # format and sort inputs
    ffine = np.asarray(ffine, dtype = np.float64)
    tfine = np.asarray(tfine, dtype = np.float64)
    tnoise = np.asarray(tnoise, dtype = np.float64)
    assert (type(ix0_offset) == int) and (type(ix1_offset) == int), \
        "ix0_offset and ix1_offset must be integers."
    assert len(ffine) == len(tfine), \
        "ffine and tfine must have the same length."
    if len(ffine) == 0:
        return np.array([], dtype = np.int32)
    ix = np.argsort(ffine)
    ffine, tfine = ffine[ix], tfine[ix] 

    # apply cutoff to tnoise
    if std_cutoff is not None:
        # determine signal cutoff
        tnoise_std = np.std(tnoise)
        tnoise_mean = np.mean(tnoise)
        tnoise = tnoise[np.abs(tnoise - tnoise_mean) < std_cutoff * tnoise_std]
    tmin, tmax = np.min(tnoise), np.max(tnoise)

    # Get ix0, ix1
    ix = []
    ix.extend(np.where((tfine >= tmin) & (tfine <= tmax))[0])
    s = tfine - tmin
    ix.extend(np.where(s[:-1] * s[1:] <= 0)[0] + 1)
    t = tfine - tmax
    ix.extend(np.where(t[:-1] * t[1:] <= 0)[0] + 1)
    ix0, ix1 = min(ix) - 1, max(ix)

    # extend ix0, ix1 by offsets
    ix0 = max(0, ix0 - ix0_offset)
    ix1 = min(len(tfine) - 1, ix1 + ix1_offset)
    if ix1 < ix0:
        ix1 = ix0 - 1
    ix = np.arange(ix0, ix1 + 1, 1, dtype = np.int32)
    return ix
