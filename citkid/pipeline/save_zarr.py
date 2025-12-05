'''
Functions for saving raw data to .zarr files.
'''

import numpy as np
import zarr
import os

def save_sweeps_and_timestreams_to_zarr(ffines, zfines, fgains, zgains, ftimestreams, ztimestreams, 
                                        fcal_indices, fsamples, outpath, suffix='', metadata={}):
    '''
    Save fine and gain sweeps and noise timestreams to a .zarr file.
    The tree structure of the .zarr file is as follows:
    /
    |--finesweep
    |--gainsweep
    |--ftimestream
    |  |--0
    |  ...
    |--ztimestream
    |  |--0
    |  ...
    |--fsample

    Metadata will be stored in the .attrs attribute of the .zarr file.

    Parameters:
        ffine: Arrays of fine sweep frequencies. Shape = (# KIDs, # fine sweep points)
        zfine: Arrays of fine sweep complex S21. Shape = (# KIDs, # fine sweep points)
        fgain: Arrays of gain sweep frequencies. Shape = (# KIDs, # gain sweep points)
        zgain: Arrays of gain sweep complex S21. Shape = (# KIDs, # gain sweep points)
        ftimestreams: Arrays of readout frequencies at which timestreams were taken.
            Multiple sets of timestreams can be provided.
            Shape = (# timestream sets, # KIDs)
        ztimestreams: Arrays of complex S21 timestream data.
            Multiple sets of timestreams can be provided.
            In general, the number of timestream points can differ between
            sets, so the shape may be irregular.
            Each subarray has Shape = (# KIDs, # timestream points)
        fcal_indices: Array of indices for calibration tones
        fsamples: Array of sample rates for each timestream set.
            Shape = (# timestream sets)
        outpath: Directory to save the .zarr file in. It will save as
            outpath+f'sweeps_and_timestreams{suffix}.zarr'
        suffix: Suffix to add to the end of the .zarr file name.
        metadata (dictionary-like): Dictionary-like object containing name-value
            pairs to save as metadata. E.g. {'Blackbody temperature': 35}
    '''
    filepath = outpath+f'sweeps_and_timestreams{suffix}.zarr'
    if os.path.exists(filepath):
        raise Exception('Output .zarr file already exists!')

    root = zarr.open_group(filepath, mode='w')
    finesweep = root.create_array('finesweep', shape=(3, *ffines.shape), 
                                  chunks=(3, 1, ffines.shape[1]), dtype='float64')
    gainsweep = root.create_array('gainsweep', shape=(3, *fgains.shape), 
                                  chunks=(3, 1, fgains.shape[1]), dtype='float64')
    finesweep[:, :] = np.array([ffines, np.real(zfines), np.imag(zfines)])
    gainsweep[:, :] = np.array([fgains, np.real(zgains), np.imag(zgains)])

    root.create_array('fcal_indices', data=np.array(fcal_indices))
    root.create_array('fsample', data=np.array(fsamples))

    for ii in range(len(ftimestreams)):
        f = ftimestreams[ii]
        z = ztimestreams[ii]
        root.create_array(f'ftimestream/{ii}', data=f)
        chunks = (1, z.shape[1])
        this_ztimestream = root.create_array(f'ztimestream/{ii}', shape=z.shape,
                                             chunks=chunks, dtype='complex128')
        this_ztimestream[:, :] = z

    for key in metadata.keys():
        root.attrs[key] = metadata[key]

