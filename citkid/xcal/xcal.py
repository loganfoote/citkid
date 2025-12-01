import numpy as np 

################################################################################
##################### Cut fine sweep for fitting x vs theta ####################
################################################################################
def get_xcal_idx(ff, theta_f, theta_t, idx0_offset = 7, idx1_offset = 5,
                std_cutoff = 12):
    """
    Get incides of the fine s21 sweep over which x vs theta should be fit to 
    produce the x calibration. Chooses the indices where theta_f (after glitch 
    removal) falls within the min and max values of theta_t, with optional 
    offsets.

    Parameters:
    ff (array-like, float64): frequency values from the fine s21 sweep in Hz.
    theta_f (array-like, float64): theta values from the fine s21 sweep.
    theta_t (array-like, float64): theta values from the noise measurement.
    idx0_offset (int): number of indices to offset below the start index.
        The data is sorted such that the start index corresponds to the lowest
        frequency and the end index corresponds to the highest frequency.
    idx1_offset (int): number of indices to offset above the end index.
    std_cutoff (float or None): number of standard deviations from the mean to 
            use as a cutoff to apply to theta_t before determining the max and 
            min values. If None, no cutoff is applied.

    Returns:
    idx (array-like, int64): indices of theta_f.
    """
    # format and sort inputs
    ff = np.asarray(ff, dtype = np.float64)
    theta_f = np.asarray(theta_f, dtype = np.float64)
    theta_t = np.asarray(theta_t, dtype = np.float64)
    assert (type(idx0_offset) == int) and (type(idx1_offset) == int), \
        "idx0_offset and idx1_offset must be integers."
    assert len(ff) == len(theta_f), \
        "ff and theta_f must have the same length."
    if len(ff) == 0:
        return np.array([], dtype = np.int32)
    idx = np.argsort(ff)
    ff, theta_f = ff[idx], theta_f[idx] 

    # apply cutoff to theta_t
    if std_cutoff is not None:
        # determine signal cutoff
        theta_t_std = np.std(theta_t)
        theta_t_mean = np.mean(theta_t)
        theta_t = theta_t[np.abs(theta_t - theta_t_mean) < std_cutoff * theta_t_std]
    tmin, tmax = np.min(theta_t), np.max(theta_t)

    # Get idx0, idx1
    idx = []
    idx.extend(np.where((theta_f >= tmin) & (theta_f <= tmax))[0])
    s = theta_f - tmin
    idx.extend(np.where(s[:-1] * s[1:] <= 0)[0] + 1)
    t = theta_f - tmax
    idx.extend(np.where(t[:-1] * t[1:] <= 0)[0] + 1)
    idx0, idx1 = min(idx) - 1, max(idx)

    # extend idx0, idx1 by offsets
    idx0 = max(0, idx0 - idx0_offset)
    idx1 = min(len(theta_f) - 1, idx1 + idx1_offset)
    if idx1 < idx0:
        idx1 = idx0 - 1
    idx = np.arange(idx0, idx1 + 1, 1, dtype = np.int32)
    return idx
