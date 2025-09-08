from numba import jit
import numpy as np
from citkid.res.funcs import get_y
'''Code to generate random resonances for simulations'''

@jit(nopython=True)
def make_random_resonance_data(get_noise = False, nnoise_points = 1000,
                                npoints_fine = 500, npoints_gain = 50):
    """
    Makes a dataset of a random resonator and returns the data with the actual
    resonator parameters

    Parameters:
        get_noise (bool): if True, also returns a noise timestream that is white
            in the frequency and dissipation directions
        nnoise_points (int): number of points in the noise timestream
    Returns:
        ffine (np.array): fine sweep frequency data in Hz
        zfine (np.array): fine sweep complex S21 data
        fgain (np.array): gain sweep frequecy data in Hz
        zgain (np.array): gain sweep complex S21 data
        p (np.array): resonator parameters [fr, Qr, amp, phi, a]
        f0 (float): frequency at which noise was taken
        znoise (np.array): complex S21 noise timestream
    """
    p = get_random_resonance_parameters()

    fr, Qr = p[2], p[3]
    span = np.random.uniform(2, 100) * fr / Qr
    # Rough sweep
    frough = np.linspace(fr - span / 2, fr + span / 2, npoints_fine)
    zrough = get_resonance_s21(frough, *p)
    f0 = update_fr_distance(frough, zrough)
    # Fine sweep
    ffine = np.linspace(f0 - span / 2, f0 + span / 2, npoints_fine)
    zfine = get_resonance_s21(ffine, *p)
    fgain = np.linspace(f0 - 10 * span / 2, f0 + 10 * span / 2, npoints_gain)
    zgain = get_resonance_s21(fgain, *p)
    f0 = update_fr_distance(fgain, zgain)
    znoise = np.zeros(nnoise_points, dtype = np.complex64)
    if get_noise:
        f = np.asarray([f0])
        for i in range(nnoise_points):
            znoise[i] = get_resonance_s21(f, *p)[0]
    p = np.asarray(p[2:7])
    return ffine, zfine, fgain, zgain, p, f0, znoise

@jit(nopython=True)
def get_resonance_s21(f, fr_nstd, amp_nstd, fr, Qr, amp, phi, a, p_amp0, p_amp1,
                      p_amp2, p_phase0, p_phase1):
    """
    Gets S21 of a resonance, with no added gain or phase terms.

    Parameters:
    f (np.array): array of frequencies in Hz
    fr (float): resonance frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter.

    Returns:
    z (np.array): complex S21 data
    """
    fr_with_noise = np.random.normal(fr, fr_nstd * fr, size = len(f))
    amp_noise = np.random.normal(0, amp_nstd, size = len(f))

    fr = fr_with_noise
    deltaf = f - fr
    y0 = Qr * deltaf / fr
    y = get_y(y0, a, True)
    z0 = 1 / (1. + 2.j * y)
    theta = np.angle(z0)
    amp_noise = amp_noise * np.exp(1j * theta + 1j * np.pi)
    z = (1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y) + amp_noise)

    z_system = 10 ** (polyval([p_amp0, p_amp1, p_amp2], f - fr) / 20) + 0j
    z_system *= np.exp(1j * polyval([p_phase0, p_phase1], f - fr))
    z *= z_system
    return z

@jit(nopython=True)
def get_random_resonance_parameters():
    """
    Creates random resonance parameters and noise parameters

    Returns:
    fr_noise_nstd (float): frequency noise number of standard deviations
    amp_noise_nstd (float): amplitude noise number of standard deviations
    p (list): nonlinear IQ model parameters
    """
    random = lambda low, high: np.random.uniform(low, high)
    random_log = lambda low, high: np.exp(np.random.uniform(np.log(low),
                                                            np.log(high)))

    fr_noise_nstd  = random(-10, -7.3979400086720375)
    fr_noise_nstd  = 10 ** fr_noise_nstd
    amp_noise_nstd = random(-4.5, -2.3)
    amp_noise_nstd = 10 ** amp_noise_nstd
    fr = random_log(10e6, 10e9)
    Qr = random_log(1e3, 1e6)
    amp = random(1e-3, 1 - 1e-5)
    phi = random(-np.pi / 2, np.pi / 2)
    a = random(0, 1)

    p_amp0 = random(-5e-21, 5e-21)
    p_amp1 = random(-8e-8, 8e-8)
    p_amp2 = random(-120, 20)
    p_phase0 = -random_log(1e-9, 1e-8)
    p_phase1 = random(-1, 1)
    return fr_noise_nstd, amp_noise_nstd, fr, Qr, amp, phi, a,\
           p_amp0, p_amp1, p_amp2, p_phase0, p_phase1

@jit(nopython=True)
def polyval(p, x):
    """
    Performs the same function as np.polyval, but with numba compatability
    """
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[i]
    return y

@jit(nopython=True)
def update_fr_distance(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the furthest point from the off-resonance data. This
    function will perform better if the cable delay is first removed.

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    offres = np.mean(np.roll(z, 10)[:20])
    diff = np.abs(z - offres)
    ix = np.argmax(diff)
    if len(f) > ix + 1:
        fr = (f[ix] + f[ix + 1]) / 2
    else:
        fr = f[ix]
    return fr
