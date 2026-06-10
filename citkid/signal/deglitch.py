### Functions for finding and removing glitches from timestreams, 
### e.g. from cosmic ray impacts.

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import binned_statistic


def get_binned_baseline(ts, dt, dtbin):
    """
    Bins a timestream to lower sample rate, then interpolates
    back up to the original sample rate to get a measure of the
    baseline.
    
    Parameters:
    ts (float, array-like): Timestream data. Glitches should have positive amplitude.
    dt (float): Sample time of timestream.
    dtbin (float): Sample time to bin the data to for subtraction.
    
    Returns:
    ts_baseline (array): Baseline array
    """
    tarr = np.arange(len(ts))*dt
    nbins = int(len(ts) * dt/dtbin)
    result = binned_statistic(tarr, [tarr, ts], statistic='mean', bins=nbins)
    tbin, ts_bin = result.statistic
    ts_baseline = np.interp(tarr, tbin, ts_bin)
    return ts_baseline
    

def find_glitch_idxs(ts, dt, dtbin, nstd, distance, nrounds, i0, i1):
    """
    Find glitch peaks in a timestream.
    The steps are as follows:
        1) Create a copy of the timestream which is binned to lower sample rate,
           then linearly interpolate the binned data back to the sample rate of
           the original data and subtract it from the original data.
        2) Go through at least one, and possibly multiple rounds of peak finding.
           At each round, the peaks which were found are removed from the timestream.
    
    Parameters:
    ts (float, array-like): Timestream data. Glitches should have positive amplitude.
    dt (float): Sample time of timestream.
    dtbin (float): Sample time to bin the data to for subtraction.
    nstd (float): Minimum peak amplitude in units of number of timestream standard deviations.
    distance (int): Minimum number of sample points between peaks.
    nrounds (int): Number of rounds of peak finding to do. Must be >= 1.
    i0 (int): Number of sample points before each peak to remove.
    i1 (int): Number of sample points after each peak to remove.
    
    Returns:
    idxs (int, array-like): The sample points of peaks which were found.
    """
    ts_bin_interp = get_binned_baseline(ts, dt, dtbin)
    
    ts_subtr = ts - ts_bin_interp
    ts_subtr -= np.nanmedian(ts_subtr)
    ts_clean = ts_subtr
    idxs = np.array([], dtype=int)
    
    for _ in range(nrounds):
        this_idxs, _ = find_peaks(ts_clean, height=nstd*np.nanstd(ts_clean), distance=distance)
        idxs = np.append(idxs, this_idxs)
        for idx in this_idxs:
            thisi0 = max(0, idx-i0)
            thisi1 = min(len(ts), idx+i1)
            ts_clean[thisi0:thisi1] = 0
        
    return idxs

def replace_glitches_with_gaussian_noise(ts, idxs, i0, i1):
    """
    Given a timestream and a list of sample points,
    replaces the areas around each point with Gaussian noise.
    
    Parameters:
    ts (float, array-like): Timestream data.
    idxs (int, array-like): The sample points to remove.
    i0 (int): Number of points before each sample point to remove.
    i1 (int): Number of points after each sample point to remove.
    
    Returns:
    ts_clean (float, array-like): The timestream with points replaced by
        Gaussian noise.
    """
    idxs_masked = np.array([], dtype=int)
    ts_clean = np.copy(ts)
    ts_clean -= np.nanmedian(ts_clean)
    for idx in idxs:
        thisi0 = max(0, idx-i0)
        thisi1 = min(len(ts), idx+i1)
        idxs_masked = np.append(idxs_masked, np.arange(thisi0, thisi1))
    idxs_masked = np.sort(np.unique(idxs_masked))
    ts_clean[idxs_masked] = 0
    std = np.nanstd(ts_clean)
    noise = np.random.normal(loc=0, scale=std, size=len(idxs_masked))
    ts_clean[idxs_masked] = noise
    return ts_clean
    