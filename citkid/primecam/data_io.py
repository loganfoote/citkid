import numpy as np
import os
import pandas as pd
from ..res.util import calc_qc_qi

nonlinear_iq_names = ['fr', 'Qr', 'amp', 'phi', 'a', 'i0', 'q0', 'tau']
nonlinear_iq_labels = [
    r'$f_r$',
    r'$Q_r$',
    r'$Q_r / Q_c$',
    r'$\phi$',
    r'$a$',
    r'$i_0$',
    r'$q_0$',
    r'$\tau$',
]

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

def make_fit_row(
    p_amp, p_phase, p0, popt, perr, nrmse, downward, plot_path = '',
    prefix = 'iq', floats_only = False,
):
    """
    Wrap the output of .fitter.fit_nonlinear_iq_with_gain into a pd.Series.

    Parameters:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB.
    p_phase (np.array): 1st-order polynomial fit parameters to phase.
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter.
    perr (np.array): standard errors on fit parameters.
    nrmse (float): fit normalized root mean square error.
    downward (bool): True corresponds to a downward sweep, False corresponds to
        an upward sweep.
    plot_path (str): Path to the saved plot, or empty string if missing.
    prefix (str): Prefix for the column names. Default is 'iq'.
    floats_only (bool): Set to True to only keep columns whose values
        can be represented as floats, i.e. don't store columns for
        sweep_direction or plotpath.

    Returns:
    row (pd.Series): pd.Series with all input data.
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(nonlinear_iq_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(nonlinear_iq_names, popt):
        row[prefix + key] = pi
    qc, qi = calc_qc_qi(popt[1], popt[2])
    row[prefix + 'Qc'] = qc
    row[prefix + 'Qi'] = qi
    for key, pi in zip(nonlinear_iq_names, perr):
        row[prefix + key + '_err'] = pi
    for i, pi in enumerate(p_amp):
        row[prefix + f'pamp_{i:02d}'] = pi
    for i, pi in enumerate(p_phase):
        row[prefix + f'pphase_{i:02d}'] = pi
    if not floats_only:
        if downward:
            row[prefix + 'sweep_direction'] = 'downward'
        else:
            row[prefix + 'sweep_direction'] = 'upward'
    row[prefix + 'nrmse'] = nrmse
    if not floats_only:
        row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row(row, prefix = 'iq'):
    """
    Perform the inverse function of make_fit_row.

    Parameters:
    row (pd.Series): pd.Series with the input data.
    prefix (str): Prefix for the column names. Default is 'iq'.

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB.
    p_phase (np.array): 1st-order polynomial fit parameters to phase.
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter.
    perr (np.array): standard errors on fit parameters.
    downward (bool): True corresponds to a downward sweep, False corresponds to
        an upward sweep.
    nrmse (float): fit normalized root mean square error.
    plot_path (str): Path to the saved plot, or empty string if missing.
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in nonlinear_iq_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in nonlinear_iq_names:
        popt.append(row[prefix + key])
    perr = []
    for key in nonlinear_iq_names:
        perr.append(row[prefix + key + '_err'])
    p_amp = []
    s = prefix + 'pamp' + '_'
    indices = [int(key.replace(s, '')) for key in row.keys() if s in key]
    for index in range(max(indices) + 1):
        key = s + f'{index:02d}'
        p_amp.append(row[key])
    p_phase = []
    s = prefix + 'pphase' + '_'
    indices = [int(key.replace(s, '')) for key in row.keys() if s in key]
    for index in range(max(indices) + 1):
        key = s + f'{index:02d}'
        p_phase.append(row[key])
    nrmse = row[prefix + 'nrmse']
    plot_path = row[prefix + 'plotpath']
    if row[prefix + 'sweep_direction'] == 'downward':
        downward = True
    else:
        downward = False
    p_amp, p_phase = np.array(p_amp), np.array(p_phase)
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    return p_amp, p_phase, p0, popt, perr, nrmse, downward, plot_path
