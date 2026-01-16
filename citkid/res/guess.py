import numpy as np
from ..xcal import circle
from .util import get_peak_fwhm
from scipy.ndimage import gaussian_filter

def guess_p0_nonlinear_iq(f, z):
    """
    Get initial guess parameters for nonlinear_iq.
    Assumes gain and cable delay have been removed

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data

    Returns:
    p0 (list): initial guess fit parameters
        [fr, Qr, amp, phi, a, i0, q0, tau]
    """
    # guess i0 and q0
    z0_guess = np.mean(np.roll(z, 2)[:4])
    i0_guess = np.real(z0_guess)
    q0_guess = np.imag(z0_guess)
    # guess tau
    tau_guess = 0
    # guess phi and amp
    phi_guess, amp_guess = guess_phi_amp(z, z0_guess)
    if amp_guess > 1 - 1e-6:
        amp_guess = 1 - 1e-6
    # guess Qr
    Qr_guess = guess_Qr(f, z, z0_guess, phi_guess, amp_guess)
    # guess a
    a_guess = guess_a(f, z, z0_guess, phi_guess, amp_guess)
    # guess fr
    fr_guess = guess_fr(f, z, z0_guess, phi_guess, amp_guess, a_guess, Qr_guess)
    # Package and return p0
    p0 = [fr_guess, Qr_guess, amp_guess, phi_guess, a_guess,
          i0_guess, q0_guess, tau_guess]
    return p0

def guess_Qr(f, z, z0, phi, amp):
    """
    Guesses Qr given complex S21 data.

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance
    phi (float): impedance mismatch angle

    Returns:
    Qr_guess (float): Qr guess
    """
    # rotate data, then filter
    z_rot = (1 - z / z0) * np.exp(-1j * phi) * np.cos(phi) / amp
    z_rot = abs(z_rot)
    z_rot = gaussian_filter(z_rot, sigma = 10)
    # Ensure that the data is upright
    z_off = np.mean(np.roll(z_rot, 2)[:4])
    ix = len(z_rot) // 2 - 2
    z_on = np.mean(z_rot[ix: ix + 4])
    if z_off > z_on:
        z_rot = - z_rot
    # Find the peak
    xpeak, ypeak, fwhm = get_peak_fwhm(f, z_rot)
    fwhm /= xpeak
    poly = [-0.94123147,  1.80699622]
    Qr_guess = np.exp(np.polyval(poly, np.log(fwhm)))
    return Qr_guess

def guess_a(f, z, z0, phi, amp):
    """
    Guesses the nonlinearity parameter

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance
    phi (float): impedance mismatch angle
    amp (float): Qr / Qc

    Returns:
    a_guess (float): nonlinearity parameter guess
    """
    z_rot = (1 - z / z0) * np.exp(-1j * phi) * np.cos(phi) / amp
    f0 = np.mean(f)

    fdiff = np.mean([f[1:], f[:-1]], axis = 0)
    ydiff = abs(np.diff(z_rot))
    ix = np.argmax(ydiff)
    x = np.log10(ydiff[ix] / (fdiff[ix] / f0))

    abin = [0.1517482 , 0.20842793, 0.24274033, 0.2653973 , 0.28081431,
        0.28519286, 0.29832441, 0.44475322, 0.81195896]
    xbin = [-1.91936449, -1.71424342, -1.49362239, -1.26713098, -1.04040067,
            -0.8144114 , -0.58510798, -0.3757271 , -0.14204129]
    a_guess = np.interp(x, xbin, abin)
    if a_guess > 1:
        a_guess = 1
    return a_guess

def guess_fr(f, z, z0, phi, amp, a, Qr):
    """
    Guess the resonant frequency

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance
    phi (float): impedance mismatch angle
    a (float): nonlinearity parameter guess
    Qr (float): guess for the total quality factor

    Returns:
    fr_guess (float): guess for the resonant frequency
    """
    z_rot = (1 - z / z0) * np.exp(-1j * phi) * np.cos(phi) / amp
    fdiff = np.mean([f[1:], f[:-1]], axis = 0)
    zdiff = abs(z_rot[1:] - z_rot[:-1])
    ix = np.argmax(zdiff)
    fr_guess = fdiff[ix]
    # Modification from nonlinearity parameter and Q
    poly = [1.12475270, 5.78740698e-6]
    fr_guess *= np.polyval(poly, a / Qr) + 1
    return fr_guess

def guess_phi_amp(z, z0):
    """
    Guess the impedance mismatch rotation and amplitude.

    Parameters:
    z (np.array): array of complex S21 data
    z0 (complex): value of z off resonance

    Returns:
    phi_guess (float): phi guess
    amp_guess (float): amp guess
    """
    popt = circle.fit_iq_circle(z)
    zc, R = popt
    xc, yc = np.real(zc), np.imag(zc)
    # angle between center of circle and off resonance point
    x1, y1, = -np.real(z0), -np.imag(z0)
    x2, y2 = xc + x1, yc + y1
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    phi_guess = np.arctan2(det, dot)
    amp_guess = 2 * R * np.abs(np.cos(phi_guess)) / np.abs(z0)
    return phi_guess, amp_guess
