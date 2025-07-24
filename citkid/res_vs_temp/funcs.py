import numpy as np
from scipy.special import digamma # complex digamma function
from scipy.special import iv as I_n # modified bessel function of the first kind
from scipy.special import kv as K_n # modified bessel function of the second kind

k_B = 1.380649e-23
h = 6.62607015e-34
hbar = h / (2 * np.pi)
# N0 is not actually used here, but I am leaving it for reference
N0_Al = 1.0737e47
N0_Nb = 6.135e48

################################################################################
############### Resonance shift from thermal QP density and TLS ################
################################################################################
def f_vs_T(T, f0, alpha, Tc, Fdelta0, gamma = 1):
    """
    Calcuates the resonant frequency shift due to the thermal QP density and
    TLSs as a function of temperature.

    Parameters:
    T (float or array-like): temperature in K
    f0 (float): resonant frequency at T = 0 K
    alpha (float): kinetic inductance fraction
    Tc (float): critical temperature in K
    Fdelta0 (float): filling factor times dielectric loss at T = 0 K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    f (float or array-like): resonant frequency(ies) at the given
        temperature(s)
    """
    f_tls = f_vs_T_tls(T, f0, Fdelta0)
    f_qp = f_vs_T_qp(T, f0, alpha, Tc, gamma = 1)
    return f_tls + f_qp - f0

def Q_vs_T(T, f0, alpha, Tc, Fdelta0, delta_z, gamma = 1):
    """
    Calcuates the quality factor shift due to the thermal QP density and TLSs as
    a function of temperature.

    Parameters:
    T (float or array-like): temperature in K
    f0 (float): resonant frequency at T = 0 K
    alpha (float): kinetic inductance fraction
    Tc (float): critical temperature in K
    Fdelta0 (float): filling factor times dielectric loss at T = 0 K
    delta_z (float): inverse quality factor at T = 0 K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    Q (float or array-like): quality factor(s) at the given temperature(s)
    """
    Q_tls = Q_vs_T_tls(T, f0, Fdelta0, delta_z = 0)
    Q_qp = Q_vs_T_qp(T, f0, alpha, Tc, delta_z = 0, gamma = gamma)
    return 1 / (1 / Q_tls + 1 / Q_qp + delta_z)

################################################################################
################### Resonance shift from thermal QP density ####################
################################################################################
def f_vs_T_qp(T, f0, alpha, Tc, gamma = 1):
    """
    Calculates the resonant frequency shift due to the thermal QP density as a
    function of temperature.

    Parameters:
    T (float or array-like): temperature in K
    f0 (float): resonant frequency at T = 0 K
    alpha (float): kinetic inductance fraction
    Tc (float): critical temperature in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    f (float or array-like): resonant frequency(ies) at the given
        temperature(s)
    """
    T = np.asarray(T)
    kT = k_B * T
    Delta0 = 1.762 * k_B * Tc
    hf0 = h * f0

    nth_N0 = nth_over_N0(kT, Delta0)
    A = - gamma * alpha / (4 * Delta0)
    x_qp = A * S2(kT, Delta0, hf0) * nth_N0
    return f0 * (1 + x_qp)

def Q_vs_T_qp(T, f0, alpha, Tc, delta_z, gamma = 1):
    """
    Calculates the quality factor shift due to the thermal QP density as a
    function of temperature.

    Parameters:
    T (float or array-like): temperature in K
    f0 (float): resonant frequency at T = 0 K
    alpha (float): kinetic inductance fraction
    Tc (float): critical temperature in K
    delta_z (float): inverse quality factor at T = 0 K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    Q (float or array-like): quality factor(s) at the given temperature(s)
    """
    T = np.asarray(T)
    kT = k_B * T
    Delta0 = 1.762 * k_B * Tc
    hf0 = h * f0

    nth_N0 = nth_over_N0(kT, Delta0)
    A = - gamma * alpha / (2 * Delta0)
    delta_qp = A * S1(kT, Delta0, hf0)  * nth_N0
    return 1 / (delta_qp + delta_z)

################################################################################
########################### Resonance shift from TLS ###########################
################################################################################
def f_vs_T_tls(T, f0, Fdelta0):
    """
    Calculates the resonant frequency due to TLSs as a function of temperature.

    Parameters:
    T (float or array-like): temperature in K
    f0 (float): resonant frequency at T = 0 K
    Fdelta0 (float): filling factor times dielectric loss at T = 0 K

    Returns:
    f (float or array-like): resonant frequency(ies) at the given
        temperature(s)
    """
    T = np.asarray(T)
    kT = k_B * T
    xi = h * f0 / (2 * kT)

    G = np.real(digamma(1/2 + 1j * xi / np.pi) - np.log(xi / np.pi)) / np.pi
    x_tls = Fdelta0 * G
    return f0 * (1 + x_tls)

def Q_vs_T_tls(T, f0, Fdelta0, delta_z):
    """
    Calculates the quality factor shift due to TLSs as a function of
    temperature under the assumption that P_uW << P_crit(T).

    Parameters:
    T (float or array-like): temperature in K
    f0 (float): resonant frequency at T = 0 K
    Fdelta0 (float): filling factor times dielectric loss at T = 0 K
    delta_z (float): inverse quality factor at T = 0 K

    Returns:
    Q (float or array-like): quality factor(s) at the given temperature(s)
    """
    T = np.asarray(T)
    kT = k_B * T
    xi = h * f0 / (2 * kT)

    delta = Fdelta0 * np.tanh(xi)
    return 1 / (delta + delta_z)

################################################################################
########################## Mattis-Bardeen Functions ############################
################################################################################
def S1(kT, Delta0, hf0):
    """
    Calculates S1(T), as defined in Foote thesis Section 2.4.

    Parameters:
    kT (float or array-like): Boltzmann constant times temperature in J
    Delta0 (float): gap energy in J
    hf0 (float): resonant frequency energy corresponding to f(T = 0 K) in J

    Returns:
    S1 (float): Mattis-Bardeen S1 function value
    """
    kT = np.asarray(kT)
    xi = hf0 / (2 * kT)
    A = (2 / np.pi) * np.sqrt(2 * Delta0 / (np.pi * kT))
    S1 = A * np.sinh(xi) * K_n(0, xi)
    return S1

def S2(kT, Delta0, hf0):
    """
    Calculates S2(T), as defined in Foote thesis Section 2.4.

    Parameters:
    kT (float or array-like): Boltzmann constant times temperature in J
    Delta0 (float): gap energy in J
    hf0 (float): resonator energy corresponding to f(T = 0 K) in J

    Returns:
    S2 (float): Mattis-Bardeen S2 function value
    """
    kT = np.asarray(kT)
    xi = hf0 / (2 * kT)
    S2 = 1 + np.sqrt(2 * Delta0 / (np.pi * kT)) * np.exp(-xi) * I_n(0, xi)
    return S2

def nth_over_N0(kT, Delta0):
    """
    Calculates
        nth / N0,
    where nth is the thermal QP density and N0 is the single-spin density of
    states at the Fermi level.

    Parameters:
    kT (float or array-like): Boltzmann constant times temperature in J
    Delta0 (float): gap energy in J

    Returns:
    nth_N0 (float or array-like): thermal QP density divided by the single-spin
        density of states at the Fermi level
    """
    kT = np.asarray(kT)
    nth_N0 = 2 * np.sqrt(2 * np.pi * kT * Delta0) * np.exp(-Delta0 / kT)
    return nth_N0
