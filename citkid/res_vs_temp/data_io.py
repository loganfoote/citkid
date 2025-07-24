import pandas as pd
import numpy as np

f_vs_T_names = ['fr0', 'alpha', 'Tc', 'Fdelta0']
f_vs_T_labels = [r'$f_r^0$', r'$\alpha$', r'$T_c$', r'$F\delta_0$']
f_vs_T_qp_names = ['fr0', 'alpha', 'Tc']
f_vs_T_qp_labels = [r'$f_r^0$', r'$\alpha$', r'$T_c$']
f_vs_T_tls_names = ['fr0', 'Fdelta0']
f_vs_T_tls_labels = [r'$f_r^0$', r'$F\delta_0$']
Q_vs_T_names = ['fr0', 'alpha', 'Tc', 'Fdelta0', 'delta_z']
Q_vs_T_labels = [r'$f_r^0$', r'$\alpha$', r'$T_c$', r'$F\delta_0$',
                 r'$\delta_z$']
Q_vs_T_qp_names = ['fr0', 'alpha', 'Tc', 'delta_z']
Q_vs_T_qp_labels = [r'$f_r^0$', r'$\alpha$', r'$T_c$', r'$\delta_z$']
Q_vs_T_tls_names = ['fr0', 'Fdelta0', 'delta_z']
Q_vs_T_tls_labels = [r'$f_r^0$', r'$F\delta_0$', r'$\delta_z$']

################################################################################
############### Resonance shift from thermal QP density and TLS ################
################################################################################
def make_fit_row_f_vs_T(p0, popt, perr, gamma, plot_path = '',
                            prefix = 'f_vs_T'):
    """
    Wraps the output of fit_f_vs_T fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'f_vs_T'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(f_vs_T_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(f_vs_T_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(f_vs_T_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'gamma'] = gamma
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_f_vs_T(row, prefix = 'f_vs_T'):
    """
    Performs the inverse function of make_fit_row_f_vs_T.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'f_vs_T'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in f_vs_T_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in f_vs_T_names:
        popt.append(row[prefix + key])
    perr = []
    for key in f_vs_T_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    gamma = row[prefix + 'gamma']
    return p0, popt, perr, gamma, plot_path

def make_fit_row_Q_vs_T(p0, popt, perr, gamma, plot_path = '',
                            prefix = 'Q_vs_T'):
    """
    Wraps the output of fit_Q_vs_T fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'Q_vs_T'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(Q_vs_T_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(Q_vs_T_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(Q_vs_T_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'gamma'] = gamma
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_Q_vs_T(row, prefix = 'Q_vs_T'):
    """
    Performs the inverse function of make_fit_row_Q_vs_T.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'Q_vs_T'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in Q_vs_T_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in Q_vs_T_names:
        popt.append(row[prefix + key])
    perr = []
    for key in Q_vs_T_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    gamma = row[prefix + 'gamma']
    return p0, popt, perr, gamma, plot_path

################################################################################
################### Resonance shift from thermal QP density ####################
################################################################################
def make_fit_row_f_vs_T_qp(p0, popt, perr, gamma, plot_path = '',
                                  prefix = 'f_vs_T_qp'):
    """
    Wraps the output of fit_f_vs_T_qp fitting into a pd.Series instance.

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'f_vs_T_qp'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(f_vs_T_qp_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(f_vs_T_qp_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(f_vs_T_qp_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'gamma'] = gamma
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_f_vs_T_qp(row, prefix = 'f_vs_T_qp'):
    """
    Performs the inverse function of make_fit_row_f_vs_T_qp.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'f_vs_T_qp'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in f_vs_T_qp_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in f_vs_T_qp_names:
        popt.append(row[prefix + key])
    perr = []
    for key in f_vs_T_qp_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    gamma = row[prefix + 'gamma']
    return p0, popt, perr, gamma, plot_path

def make_fit_row_Q_vs_T_qp(p0, popt, perr, gamma, plot_path = '',
                                 prefix = 'Q_vs_T_qp'):
    """
    Wraps the output of fit_Q_vs_T_qp fitting into a pd.Series instance.

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'Q_vs_T_qp'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(Q_vs_T_qp_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(Q_vs_T_qp_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(Q_vs_T_qp_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'gamma'] = gamma
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_Q_vs_T_qp(row, prefix = 'Q_vs_T_qp'):
    """
    Performs the inverse function of make_fit_row_Q_vs_T_qp.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'Q_vs_T_qp'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in Q_vs_T_qp_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in Q_vs_T_qp_names:
        popt.append(row[prefix + key])
    perr = []
    for key in Q_vs_T_qp_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    gamma = row[prefix + 'gamma']
    return p0, popt, perr, gamma, plot_path

################################################################################
########################### Resonance shift from TLS ###########################
################################################################################
def make_fit_row_f_vs_T_tls(p0, popt, perr, plot_path = '',
                            prefix = 'f_vs_T_tls'):
    """
    Wraps the output of fit_f_vs_T_tls fitting into a pd.Series instance.

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'f_vs_T_tls'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(f_vs_T_tls_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(f_vs_T_tls_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(f_vs_T_tls_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_f_vs_T_tls(row, prefix = 'f_vs_T_tls'):
    """
    Performs the inverse function of make_fit_row_f_vs_T_tls.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'f_vs_T_tls'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in f_vs_T_tls_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in f_vs_T_tls_names:
        popt.append(row[prefix + key])
    perr = []
    for key in f_vs_T_tls_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    return p0, popt, perr, plot_path

def make_fit_row_Q_vs_T_tls(p0, popt, perr, plot_path = '',
                            prefix = 'Q_vs_T_tls'):
    """
    Wraps the output of fit_Q_vs_T_tls fitting into a pd.Series instance.

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'Q_vs_T_tls'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(Q_vs_T_tls_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(Q_vs_T_tls_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(Q_vs_T_tls_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_Q_vs_T_tls(row, prefix = 'Q_vs_T_tls'):
    """
    Performs the inverse function of make_fit_row_Q_vs_T_tls.

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'Q_vs_T_tls'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in Q_vs_T_tls_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in Q_vs_T_tls_names:
        popt.append(row[prefix + key])
    perr = []
    for key in Q_vs_T_tls_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    return p0, popt, perr, plot_path
