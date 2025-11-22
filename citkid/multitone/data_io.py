import numpy as np
import os

# Need to update docstrings, imports
def import_iq_noise(directory, file_suffix, noise_index = 0, import_noiseq = True):
    """
    Imports data from primecam.procedures.take_iq_noise

    Parameters:
    directory (str): directory containing the saved data
    file_index (int): file index
    import_noiseq (bool): if False, doesn't import noise

    Returns:
    fres_initial (np.array): initial frequency array in Hz
    fres (np.array): noise frequency array in Hz
    ares (np.array): RFSoC amplitude array
    qres (np.array): resonance Q array for cutting data
    fres_all (np.array): full list of resonance frequencies in Hz 
    qres_all (np.array): full list of resonance Qs for cutting data 
    fcal_indices (np.array): calibration tone indices
    frough, zrough (np.array): rough sweep frequency and complex S21 data
    fgain, zgain (np.array): gain sweep frequency and complex S21 data
    ffine, zfine (np.array): fine sweep frequency and complex S21 data
    znoise (np.array): complex S21 noise timestream array
    noise_dt (float): noise sample time in s
    """
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    path = directory + f'fres_initial{file_suffix}.npy'
    if os.path.exists(path):
        fres_initial = np.load(path)
    else:
        fres_initial = None
    fres = np.load(directory + f'fres{file_suffix}.npy')
    ares = np.load(directory + f'ares{file_suffix}.npy')
    qres = np.load(directory + f'qres{file_suffix}.npy')
    fcal_indices = np.load(directory + f'fcal_indices{file_suffix}.npy')
    res_indices = np.load(directory + f'res_indices{file_suffix}.npy')
    path = directory + f'fres_all{file_suffix}.npy'
    if os.path.exists(path):
        fres_all = np.load(path)   
    else:
        fres_all = np.delete(fres, fcal_indices) 
    path = directory + f'qres_all{file_suffix}.npy'
    if os.path.exists(path):
        qres_all = np.load(path)   
    else:
        qres_all = np.delete(qres, fcal_indices)
    
    # sweeps
    path = directory + f's21_rough{file_suffix}.npy'
    if os.path.exists(path):
        frough, irough, qrough = np.load(path)
        zrough = irough + 1j * qrough
    else:
        frough, zrough = None, None
    path = directory + f's21_gain{file_suffix}.npy'
    fgain, igain, qgain = np.load(path)
    zgain = igain + 1j * qgain
    path = directory + f's21_fine{file_suffix}.npy'
    ffine, ifine, qfine = np.load(path)
    zfine = ifine + 1j * qfine
    if import_noiseq:
        path = directory + f'noise{file_suffix}_{noise_index:02d}.npy'
        noise_dt = float(np.load(directory + f'noise{file_suffix}_tsample_{noise_index:02d}.npy'))
        inoise, qnoise = np.load(path)
        znoises = inoise + 1j * qnoise
        fres_noise = np.load(directory + f'fres_noise{file_suffix}.npy')
    else:
        noise_dt = None
        znoises = None 
        fres_noise = None
    return fres_initial, fres, ares, qres, fcal_indices, fres_all, qres_all, frough, zrough,\
           fgain, zgain, ffine, zfine, znoises, noise_dt, res_indices, fres_noise
