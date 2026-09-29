import numpy as np
import os

def import_iq_noise(directory, file_suffix, import_noiseq = True):
    """
    Import data from primecam.procedures.take_iq_noise

    Parameters:
    directory (str): directory containing the saved data
    file_index (int): file index
    import_noiseq (bool): if False, doesn't import noise

    Returns:
    fres_initial (np.array): initial frequency array in Hz
    fres (np.array): noise frequency array in Hz
    ares (np.array): RFSoC amplitude array
    Qres (np.array): resonance Q array for cutting data
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
    Qres = np.load(directory + f'Qres{file_suffix}.npy')
    fcal_indices = np.load(directory + f'fcal_indices{file_suffix}.npy')
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
    path = directory + f'noise{file_suffix}.npy'
    if os.path.exists(path) and import_noiseq:
        inoise, qnoise = np.load(path)
        znoise = inoise + 1j * qnoise
        noise_dt = float(np.load(directory + f'noise{file_suffix}_tsample.npy' ))
    else:
        znoise, noise_dt = None, None
    return fres_initial, fres, ares, Qres, fcal_indices, frough, zrough,\
           fgain, zgain, ffine, zfine, znoise, noise_dt
