import numpy as np
from scipy.special import digamma
k_B = 1.380649e-23
h = 6.62607015e-34
################################################################################
############### Resonance shift from thermal QP density and TLS ################
################################################################################
def guess_p0_f_vs_T(T, f, Tc_guess = 1.2, gamma = 1):
    """
    Calculates an initial guess for f_vs_T. Tc_guess must be provided.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    Tc_guess (float): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [f0_guess, alpha_guess, Tc_guess, Fdelta0_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    ix = min(len(T) // 2, (len(T) - 4))
    p0_qp, bounds_qp = guess_p0_f_vs_T_qp(T[ix:], f[ix:], Tc_guess = Tc_guess,
                                    gamma = gamma)
    ix = max(len(T) // 2, 4)
    p0_tls, bounds_tls = guess_p0_f_vs_T_tls(T[:ix], f[:ix])
    p0 = [np.min([p0_qp[0], p0_tls[0]]), p0_qp[1], p0_qp[2], p0_tls[1]]
          # ^ both tend to overestimate f0
    bounds = get_bounds_f_vs_T(bounds_qp, bounds_tls)
    return p0, bounds

def get_bounds_f_vs_T(bounds_qp, bounds_tls):
    """
    Gets bounds for the fitter for f_vs_T.

    Parameters:
    bounds_qp (list): bounds from the QP guessing function
    bounds_tls (list): bounds from the TLS guessing function

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter
    """
    bounds = [b[:] for b in bounds_qp]
    # choose widest bounds for f0
    bounds[0][0] = min(bounds[0][0], bounds_tls[0][0])
    bounds[1][0] = max(bounds[1][0], bounds_tls[1][0])
    # append Fdelta0 to the end of bounds_qp
    bounds[0].append(bounds_tls[0][1])
    bounds[1].append(bounds_tls[1][1])
    return bounds

def guess_p0_Q_vs_T(T, Q, f0_guess, Tc_guess = 1.2, gamma = 1):
    """
    Calculates an initial guess for Q_vs_T. Tc_guess and f0_guess must be
    provided.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    f0_guess (float): guess for f0 in Hz
    Tc_guess (float or None): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [f0_guess, alpha_guess, Tc_guess, Fdelta0_guess, delta_z_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    ix = min(len(T) // 2, (len(T) - 4))
    p0_qp, bounds_qp = guess_p0_Q_vs_T_qp(T[ix:], Q[ix:], f0_guess = f0_guess,
                                          Tc_guess = Tc_guess, gamma = gamma)
    ix = max(len(T) // 2, 4)
    p0_tls, bounds_tls = guess_p0_Q_vs_T_tls(T[:ix], Q[:ix],
                                             f0_guess = f0_guess)
    p0 = [p0_qp[0], p0_qp[1], p0_qp[2], p0_tls[1], p0_qp[3]]
    #     ^ f0 is the same for both               ^ QP does a better job at
    #                                                extracting delta_z
    bounds = get_bounds_Q_vs_T(bounds_qp, bounds_tls)
    return p0, bounds

def get_bounds_Q_vs_T(bounds_qp, bounds_tls):
    """
    Gets bounds for the fitter for Q_vs_T.

    Parameters:
    bounds_qp (list): bounds from the QP guessing function
    bounds_tls (list): bounds from the TLS guessing function

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter
    """
    bounds = [b[:] for b in bounds_qp]
    # Choose widest bounds for f0
    bounds[0][0] = min(bounds[0][0], bounds_tls[0][0])
    bounds[1][0] = max(bounds[1][0], bounds_tls[1][0])
    # Insert Fdelt0 into index 3
    bounds[0].insert(3, bounds_tls[0][1])
    bounds[1].insert(3, bounds_tls[1][1])
    # Choose widest bounds for delta_z
    bounds[0][4] = min(bounds[0][4], bounds_tls[0][2])
    bounds[1][4] = max(bounds[1][4], bounds_tls[1][2])
    return bounds

################################################################################
################### Resonance shift from thermal QP density ####################
################################################################################
def guess_p0_f_vs_T_qp(T, f, Tc_guess = 1.2, gamma = 1):
    """
    Calculates an initial guess for f_vs_T_qp. Tc_guess must be provided.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data
    Tc_guess (float): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [f0_guess, alpha_guess, Tc_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    # f0 guess
    f0_guess = max(f)
    # Alpha guess
    alpha_guess = 0.7 / gamma
    # p0, bounds
    p0 = [f0_guess, alpha_guess, Tc_guess]
    bounds = get_bounds_f_vs_T_qp(p0)
    return p0, bounds

def get_bounds_f_vs_T_qp(p0):
    """
    Gets bounds for the fitter for f_vs_T_qp.

    Parameters:
    p0 (list): initial guess parameters
        [f0_guess, alpha_guess, Tc_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3,         p0[2] / 2],
              [p0[0] * (1 + 1e-5), max(p0[1] * 3, 1), p0[2] * 2]]
    return bounds

def guess_p0_Q_vs_T_qp(T, Q, f0_guess, Tc_guess = 1.2, gamma = 1):
    """
    Calculates an initial guess for Q_vs_T_qp. Tc_guess and f0_guess must be
    provided.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    f0_guess (float): guess for f0 in Hz
    Tc_guess (float or None): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [f0_guess, alpha_guess, Tc_guess, delta_z_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    # delta_z guess
    delta_z_guess = 1 / min(Q)
    # alpha guess
    alpha_guess = 0.7 / gamma
    # p0, bounds
    p0 = [f0_guess, alpha_guess, Tc_guess, delta_z_guess]
    bounds = get_bounds_Q_vs_T_qp(p0)
    return p0, bounds

def get_bounds_Q_vs_T_qp(p0):
    """
    Gets bounds for the fitter for Q_vs_T_qp.

    Parameters:
    p0 (list): initial guess parameters
        [f0_guess, alpha_guess, Tc_guess, delta_z_guess]

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
    Calculates an initial guess for f_vs_T_tls.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz

    Returns:
    p0 (list): initial guess parameters
        [f0_guess, Fdelta0_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    f0_guess = np.median(f)
    xi = h * f0_guess / (2 * k_B * T)
    G = np.real(digamma(1/2 + 1j * xi / np.pi) - np.log(xi / np.pi)) / np.pi
    f0_guess = np.polyfit(G, f, 1)[1]
    # f0_guess = max(f)
    # Fdelta0 guess
    xi = h * f0_guess / (2 * k_B * T)
    G = np.real(digamma(1/2 + 1j * xi / np.pi) - np.log(xi / np.pi)) / np.pi
    Fdelta0_guess = np.polyfit(G, (f - f0_guess) / f0_guess, 1)[0]
    Fdelta0_guess = max(Fdelta0_guess, 1e-6)
    # p0, bounds
    p0 = [f0_guess, Fdelta0_guess]
    bounds = get_bounds_f_vs_T_tls(p0)
    return p0, bounds

def get_bounds_f_vs_T_tls(p0):
    """
    Gets bounds for the fitter for f_vs_T_tls.

    Parameters:
    p0 (list): initial guess parameters
        [f0_guess, Fdelta0_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3],
              [p0[0] * (1 + 1e-5), p0[1] * 3]]
    return bounds

def guess_p0_Q_vs_T_tls(T, Q, f0_guess):
    """
    Calculates an initial guess for Q_vs_T_tls. f0_guess must be provided.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    f0_guess (float): guess for f0 in Hz

    Returns:
    p0 (list): initial guess parameters
        [f0_guess, Fdelta0_guess, delta_z_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    xi = h * f0_guess / (2 * k_B * T)
    x = np.tanh(xi)
    Qinv = 1 / Q
    m, b = np.polyfit(x, Qinv, 1)
    delta_z_guess = b
    Fdelta0_guess = m
    # p0, bounds
    p0 = [f0_guess, Fdelta0_guess, delta_z_guess]
    bounds = get_bounds_Q_vs_T_tls(p0)
    return p0, bounds

def get_bounds_Q_vs_T_tls(p0):
    """
    Gets bounds for the fitter for Q_vs_T_tls.

    Parameters:
    p0 (list): initial guess parameters
        [f0_guess, Fdelta0_guess, delta_z_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 10, p0[2] / 10],
              [p0[0] * (1 + 1e-5), p0[1] * 10, p0[2] * 10]]
    return bounds
