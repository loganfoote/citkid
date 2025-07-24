import numpy as np
from scipy.special import digamma
k_B = 1.380649e-23
h = 6.62607015e-34
################################################################################
############### Resonance shift from thermal QP density and TLS ################
################################################################################


################################################################################
################### Resonance shift from thermal QP density ####################
################################################################################
def guess_p0_f_vs_T_qp(T, fr, Tc_guess = 1.2, gamma = 1):
    """
    Calculates an initial guess for f_vs_T_qp. Tc_guess must be provided.

    Parameters:
    T (array-like): temperature data in K
    fr (array-like): resonant frequency data in Hz
    Tc_guess (float): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    # f0 guess
    fr0_guess = max(fr)
    # Alpha guess
    alpha_guess = 0.7 / gamma
    # p0, bounds
    p0 = [fr0_guess, alpha_guess, Tc_guess]
    bounds = get_bounds_f_vs_T_qp(p0)
    return p0, bounds

def get_bounds_f_vs_T_qp(p0):
    """
    Gets bounds for the fitter for f_vs_T_qp

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3,         p0[2] / 2],
              [p0[0] * (1 + 1e-5), max(p0[1] * 3, 1), p0[2] * 2]]
    return bounds

def guess_p0_Q_vs_T_qp(T, Q, fr0_guess, Tc_guess = 1.2, gamma = 1):
    """
    Calculates an initial guess for Q_vs_T_qp. Tc_guess and fr0_guess must be
    provided.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    fr0_guess (float): guess for fr0 in Hz
    Tc_guess (float or None): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess, delta_z_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    # delta_z guess
    delta_z_guess = 1 / min(Q)
    # alpha guess
    alpha_guess = 0.7 / gamma
    # p0, bounds
    p0 = [fr0_guess, alpha_guess, Tc_guess, delta_z_guess]
    bounds = get_bounds_Q_vs_T_qp(p0)
    return p0, bounds

def get_bounds_Q_vs_T_qp(p0):
    """
    Gets bounds for the fitter for Q_vs_T_qp.

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess, delta_z_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3,         p0[2] / 2, p0[3] / 10],
              [p0[0] * (1 + 1e-5), max(p0[1] * 3, 1), p0[2] * 2, p0[3] * 10]]
    return bounds

################################################################################
########################### Resonance shift from TLS ###########################
################################################################################
def guess_p0_f_vs_T_tls(T, f):
    """
    Calculates an initial guess for f_vs_T

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, D_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    fr0_guess = np.median(f)
    xi = h * fr0_guess / (2 * k_B * T)
    G = np.real(digamma(1/2 + 1j * xi / np.pi) - np.log(xi / np.pi)) / np.pi
    fr0_guess = np.polyfit(G, f, 1)[1]
    # fr0_guess = max(f)
    # Fdelta0 guess
    xi = h * fr0_guess / (2 * k_B * T)
    G = np.real(digamma(1/2 + 1j * xi / np.pi) - np.log(xi / np.pi)) / np.pi
    Fdelta0_guess = np.polyfit(G, (f - fr0_guess) / fr0_guess, 1)[0]
    Fdelta0_guess = max(Fdelta0_guess, 1e-6)
    # p0, bounds
    p0 = [fr0_guess, Fdelta0_guess]
    bounds = get_bounds_f_vs_T_tls(p0)
    return p0, bounds

def get_bounds_f_vs_T_tls(p0):
    """
    Gets bounds for the fitter for f_vs_T_tls.

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, Fdelta0_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3],
              [p0[0] * (1 + 1e-5), p0[1] * 3]]
    return bounds

def guess_p0_Q_vs_T_tls(T, Q, fr0_guess):
    """
    Calculates an initial guess for Q_vs_T. fr0_guess must be provided.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data in Hz

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, Fdelta0_guess, delta_z_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    xi = h * fr0_guess / (2 * k_B * T)
    x = np.tanh(xi)
    Qinv = 1 / Q
    m, b = np.polyfit(x, Qinv, 1)
    delta_z_guess = b
    Fdelta0_guess = m
    # p0, bounds
    p0 = [fr0_guess, Fdelta0_guess, delta_z_guess]
    bounds = get_bounds_Q_vs_T_tls(p0)
    return p0, bounds

def get_bounds_Q_vs_T_tls(p0):
    """
    Gets bounds for the fitter for Q_vs_T_tls.

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, Fdelta0_guess, delta_z_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 10, p0[2] / 10],
              [p0[0] * (1 + 1e-5), p0[1] * 10, p0[2] * 10]]
    return bounds
