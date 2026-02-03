from scipy.signal import find_peaks
import numpy as np
import warnings 

warnings.warn(
    "citkid.noise.cosmic_rays is deprecated and will be removed in version 1.0.0",
    DeprecationWarning,
    stacklevel=2,
)

def remove_cosmic_rays(theta, A, tsample, cr_nstd = 5, cr_width = 100e-6,
                       cr_peak_spacing = 100e-6, cr_removal_time = 1e-3):
    """
    Remove cosmic rays from a timestream using a peak finding algorithm.
    Flags the cosmic rays and sets the data in the timestream equal to the
    average before and after the cosmic ray.

    Parameters:
    theta (np.array): theta array with cosmic rays
    A (np.array): amplitude array with cosmic rays
    tsample (float): sample time of theta array
    cr_nstd (float): number of standard deviations above the mean for find_peaks
    cr_width (float): width of cosmic rays in seconds
    cr_peak_spacing (float): number of seconds spacing between cosmic rays
    cr_removal_time (float): number of seconds to remove around each peak

    Returns:
    cr_indices (np.array): array of indices at which cosmic rays were found
    theta_rmvd (np.array): theta array with cosmic rays removed
    A_rmvd (np.array): amplitude array with cosmic rays removed
    """
    height = np.mean(-theta) + cr_nstd * np.std(theta)
    distance = int(cr_peak_spacing / tsample)
    cr_width_points = int(cr_width / tsample)
    if distance < 1:
        distance = 1
    if cr_width_points < 1:
        cr_width_points = 1
    cr_indices, _ = find_peaks(-theta, width = cr_width,
                               distance = distance,
                               height = height)
    start_offset = int(200e-6 / tsample)
    istarts = cr_indices - start_offset
    i_removal = int(cr_removal_time / tsample)
    iends = istarts + i_removal
    iranges = [[istarts[i], iends[i]] for i in range(len(istarts))]
    # Merge overlapping iranges
    iranges = remove_overlaps(iranges)
    # fix iranges on edge of data
    for index, ir in enumerate(iranges):
        if ir[0] < 0:
            iranges[index][0] = 0
        if ir[1] >= len(theta):
            iranges[index][1] = len(theta) - 1
    # Average data before and after iranges
    theta_rmvd = theta.copy()
    A_rmvd = A.copy()
    for irange in iranges:
        # determine indices of data before and after the data to cut
        ilen = irange[1] - irange[0]
        ibef = [max(0, irange[0] - ilen - 1), max(0, irange[0] - 1)]
        iaft = [min(len(theta), irange[1]), min(len(theta), irange[1] + ilen)]
        nbef = ibef[1] - ibef[0]
        naft = iaft[1] - iaft[0]
        # Make theta_bef and theta_aft arrays
        theta_bef = theta[ibef[0]: ibef[1]] if nbef else [np.nan]
        theta_aft = theta[iaft[0]: iaft[1]] if naft else [np.nan]
        A_bef = A[ibef[0]: ibef[1]] if nbef else [np.nan]
        A_aft = A[iaft[0]: iaft[1]] if naft else [np.nan]
        length_diff = len(theta_aft) - len(theta_bef)
        if length_diff > 0: # data before is too short
            theta_bef = np.concatenate([theta_bef, theta_aft[:length_diff]])
            A_bef = np.concatenate([A_bef, A_aft[:length_diff]])
        elif length_diff < 0: # data before is too short
            theta_aft = np.concatenate([theta_aft, theta_bef[:-length_diff]])
            A_aft = np.concatenate([A_aft, A_bef[:-length_diff]])
        theta_rmvd[irange[0]: irange[1]] = np.nanmean([theta_bef, theta_aft],
                                                    axis = 0)
        A_rmvd[irange[0]: irange[1]] = np.nanmean([A_bef, A_aft], axis = 0)
    return cr_indices, theta_rmvd, A_rmvd

################################################################################
######################### Utility functions ####################################
################################################################################

def remove_overlaps(iranges):
    """
    Given a list of index ranges, return a list where overlapping ranges are
    concatenated. Overlaps are defined by clearances between ranges equal to
    the lengths of the ranges

    Parameter:
    iranges (list): values (list) are [lower, upper] values

    Returns:
    iranges_concat (list): iranges where overlapping ranges are concatenated
    """
    overlaps = find_overlaps(iranges)
    if not len(overlaps):
        return iranges
    # Remove overlaps
    iranges_concat = []
    i = 0
    while i < len(iranges):
        if i in overlaps:
            iranges_concat.append([min(iranges[i] + iranges[i + 1]),
                                   max(iranges[i] + iranges[i + 1])])
            ix = [overlap - 1 for overlap in overlaps]
            i += 2
        else:
            iranges_concat.append(iranges[i])
            i += 1
    # Repeat until done
    return remove_overlaps(iranges_concat)

def find_overlaps(iranges):
    """
    Find overlapping ranges, where overlaps are defined by a clearance between
    ranges equal to the max length of the two ranges

    Parameters:
    iranges (list): values (list) are [lower, upper] values

    Returns:
    overlaps (list): values (int) are indices of overlapping ranges, where the
        two ranges that overlap are iranges[overlaps[i]] and
        iranges[overlaps[i + 1]]
    """
    sorted_ranges = sorted(iranges, key = lambda x: x[0])
    range_lens = [0] + [ir[1] - ir[0] for ir in iranges] + [0]
    range_lens = [ir[1] - ir[0] for ir in iranges]
    required_clearances = list(np.max([range_lens[:-1], range_lens[1:]],
                               axis = 0))
    actual_clearances = []
    for i in range(len(iranges) - 1):
        actual_clearances.append(iranges[i + 1][0] - iranges[i][1])
    overlaps = []
    for i in range(len(actual_clearances)):
        if actual_clearances[i] <= required_clearances[i]:
            overlaps.append(i)
    return overlaps
