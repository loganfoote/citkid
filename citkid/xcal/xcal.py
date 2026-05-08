import numpy as np 

################################################################################
##################### Cut fine sweep for fitting x vs theta ####################
################################################################################
def get_xcal_mask(ff, theta_f, theta_t, idx0_offset = 3, idx1_offset = 7,
                  std_cutoff = 12):
    """
    Get mask of the fine s21 sweep over which x vs theta should be fit to 
    produce the x calibration. Chooses the indices where theta_f (after glitch 
    removal) falls within the min and max values of theta_t, with optional 
    offsets.

    Parameters:
    ff (array-like, float64): Frequency values from the fine S21 sweep in Hz.
        Must be sorted in ascending order.
    theta_f (array-like, float64): Theta values from the fine S21 sweep.
    theta_t (array-like, float64): Theta values from the noise measurement.
    idx0_offset (int): Number of indices to offset below the start index.
        The data is sorted such that the start index corresponds to the lowest
        frequency and the end index corresponds to the highest frequency.
    idx1_offset (int): Number of indices to offset above the end index.
    std_cutoff (float or None): Number of standard deviations from the mean to
        use as a cutoff on theta_t before determining min and max. If None, no
        cutoff is applied.

    Returns:
    mask (np.array, bool): Mask of theta_f.
    """
    # Input validation
    ff = np.asarray(ff, dtype = np.float64)
    theta_f = np.asarray(theta_f, dtype = np.float64)
    theta_t = np.asarray(theta_t, dtype = np.float64)
    if not isinstance(idx0_offset, (int, np.integer)) or \
        not isinstance(idx1_offset, (int, np.integer)):
        raise ValueError("idx0_offset and idx1_offset must be integers.")
    if len(ff) != len(theta_f):
        raise ValueError("ff and theta_f must have the same length.")
    if len(ff) == 0:
        return np.array([], dtype = bool)
    if not all(ff[1:] >= ff[:-1]):
        raise ValueError("ff must be sorted in ascending order.")

    # Apply cutoff to theta_t
    if std_cutoff is not None:
        # determine signal cutoff
        theta_t_std = np.std(theta_t)
        theta_t_mean = np.mean(theta_t)
        theta_t = theta_t[
            np.abs(theta_t - theta_t_mean) < std_cutoff * theta_t_std
        ]
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
        
    # Form mask from idx0, idx1
    mask = np.ones(len(ff), dtype = bool) * False 
    mask[idx0:idx1 + 1] = True
    return mask

def fit_x_theta(thetat, xf, mask, poly_x_deg):
    """
    Fits a polynomial to x vs theta data for the fine s21 sweep, using only the
    points where mask is True. The resulting polynomial is used for x 
    calibration.

    Parameters:
    thetat (array-like, float64): Theta values from the noise measurement.
    xf (array-like, float64): x values from the fine S21 sweep.
    mask (array-like, bool): Mask to apply to the data before fitting.
    poly_x_deg (int): Degree of the polynomial to fit.

    Returns:
    poly (np.array, float64): Coefficients of the fitted polynomial.
    """
    poly = np.polyfit(thetat[mask], xf[mask], poly_x_deg) 
    return poly 


################################################################################
# Simplified xcal creation for DataSet #
################################################################################
def create_xcal(DS, data_idx, ft):
    """
    Placeholder, I haven't decided how to implement this. Maybe as method of 
    DataSet, though it does rely on the specific pipeline.
    
    Converts gain fit and circle fit data into simplified calibration parameters
    z1, z2, and xpoly, such that 
        zt_cent = zt * z1 + z2
        thetat = angle(zt_cent) 
        At = abs(zt_cent) 
        xt = polyval(xpoly, thetat)

    Parameters:
    DS (dataset.DataSet): DataSet instance from which calibration parameters are 
        built. 
    data_idx (int): data index. 
    ft (float): tone frequency corresponding to the dataset of interest.
    """
    # remove gain
    amp = 10 ** (np.polyval(DS.p_amp[data_idx], ft) / 20)
    phase = np.exp(1j * np.polyval(DS.p_phase[data_idx], ft))
    z1 = 1 / (amp * phase)
    # center and rotate circle 
    za = np.exp(-1j * DS.theta_phase_offset[data_idx])  
    z1 *= za 
    z2 = -DS.circ_origin[data_idx] * za
    if not np.isfinite(z1):
        z2 = np.nan + 1j * np.nan 
    if not np.isfinite(z2):
        z1 = np.nan + 1j * np.nan
    poly_x = DS.poly_x[data_idx] 
    return z1, z2, poly_x