import numpy as np
from numba import jit
# Need to add a few functions and figure out a good naming scheme

################################################################################
############################## white noise #####################################
################################################################################

@jit(nopython = True)
def rolloff(f, tau):
    """
    Noise rolloff

                         1
           y =  --------------------
                 1 + (2 pi f tau)^2

    Parameters:
    f (np.array) : frequencies in Hz
    tau (float)  : rolloff time in s

    Returns:
    noise (np.array): noise values corresponding to f
    """
    return 1 / (1 + (2 * np.pi * f * tau) ** 2)


@jit(nopython = True)
def white_rolloff(f, a, b, tau):
    """
    Noise with a flat profile and a single rolloff to a lower flat profile.

                         a
           y =  --------------------  +  b
                 1 + (2 pi f tau)^2

    Parameters:
    f (np.array) : frequencies in Hz
    a (float)    : higher white noise level
    b (float)    : lower white noise level
    tau (float)  : rolloff time in s

    Returns:
    noise (np.array): noise values corresponding to f
    """
    return a * rolloff(f, tau) + b

@jit(nopython = True)
def white_rolloff_rd_elect(f, a, b, tau_qp, tau_elect, tau_rd):
    """
    Noise with a flat profile and a single rolloff to a lower flat profile,
    plus an electrical rolloff and the resonator ringdown time rolloff

           y =  [a * R(tau_qp) * R(tau_rd) + b] * R(tau_elect)

    where R is the standard rolloff

    Parameters:
    f (np.array)      : frequencies in Hz
    a (float)         : higher white noise level
    b (float)         : lower white noise level
    tau_qp (float)    : quasiparticle rolloff time in s
    tau_elect (float) : electrical rolloff time in s
                        = 1 / (f_electrical * 2 * np.pi)
    tau_rd (float)    : resonator ringdown time Qr / (pi fr)

    Returns:
    noise (np.array): Noise values corresponding to f
    """
    a1 = a * rolloff(f, tau_qp) * rolloff(f, tau_rd)
    return (a1 + b) * rolloff(f, tau_elect)

################################################################################
############################# 1/f noise ########################################
################################################################################

@jit(nopython = True)
def one_over_f(f, a, alpha):
    """
    One over f noise profile

                   /  1  \  alpha
           y =  a | ----- |
                   \  f  /

    Parameters:
    f (np.array)  : frequencies in Hz
    a (float)     : noise level at a frequency of 1 Hz
    alpha (float) : 1/f power

    Returns:
    noise (np.array): Noise values corresponding to f
    """
    return (gamma / f) ** alpha

@jit(nopython = True)
def white_1f(f, a, b, alpha):
    """
    Noise profile of white noise (no rolloff) with 1/f noise

                   /  1  \  alpha
           y =  a | ----- |        +  b
                   \  f  /

    Parameters:
    f (np.array)  : frequencies in Hz
    a (float)     : noise level at a frequency of 1 Hz
    b (float)     : white noise value
    alpha (float) : 1/f power

    Returns:
    y (np.array): noise value
    """
    y = one_over_f(f, a, alpha) + b
    return y
