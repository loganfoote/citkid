import pandas as pd
import numpy as np
from .util import calc_qc_qi
nonlinear_iq_names = ['fr', 'Qr', 'amp', 'phi', 'a', 'i0', 'q0', 'tau']
nonlinear_iq_labels = [r'$f_r$', r'$Q_r$', r'$Q_r / Q_c$', r'$\phi$', r'$a$',
                         r'$i_0$', r'$q_0$', r'$\tau$']

def make_fit_row(p_amp, p_phase, p0, popt, perr, nrmse, downward, plot_path = '',
                    prefix = 'iq', floats_only=False):
    """
    Wraps the output of fit_nonlinear_iq_with_gain fitting into a pd.Series
    instance.

    Parameters:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB.
    p_phase (np.array): 1st-order polynomial fit parameters to phase
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter.
    perr (np.array): standard errors on fit parameters.
    nrmse (float): fit normalized root mean square error.
    downward (bool): True corresponds to a downward sweep, False corresponds to
        an upward sweep.
    plot_path (str): path to the saved plot, or empty string if it does not
        exists.
    prefix (str): prefix for the column names. default is 'iq'
    floats_only (bool): Set to True to only keep columns whose values
        can be represented as floats, i.e. don't store columns for
        sweep_direction or plotpath.

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data.
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
    Performs the inverse function of make_fit_row.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'iq'.

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB.
    p_phase (np.array): 1st-order polynomial fit parameters to phase.
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter.
    perr (np.array): standard errors on fit parameters.
    downward (bool): True corresponds to a downward sweep, False corresponds to
        an upward sweep.
    nrmse (float): fit normalized root mean square error.
    plot_path (str): path to the saved plot, or empty string if it does not
        exists.
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
