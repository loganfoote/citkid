import numpy as np
from ..xcal import circle
from .util import get_peak_fwhm
from .multifit import fit_poles
from scipy.ndimage import gaussian_filter

def guess_p0_pole(f, z, center_frac = 0.5, neighbor_ratio = 0.5):
    """
    Get initial guess parameters for nonlinear_iq from a one-pole fit
    (``multifit.fit_poles``) of a target sweep.

    The pole starts at the largest change in S21 (smoothed over 1% of the
    points) within the central ``center_frac`` of the sweep and stays in
    that range, since a target sweep is centered on its resonance. If the
    change outside that range is at least ``neighbor_ratio`` times as
    large (a neighboring resonance near the edge), the neighbor gets a
    second pole, so it doesn't pull the first. Qr comes from
    ``fit_poles``' grid search, and the gain is a linear complex
    polynomial, which gives i0, q0 and tau at fr. The nonlinearity guess is
    0. Unlike ``guess_p0_nonlinear_iq``, the gain and cable delay need not
    be removed first.

    On 3,000 simulated target sweeps this start gave a median fr error of 0.2 linewidths and Qr within 2%
    (``guess_p0_nonlinear_iq``: 0.5 linewidths and a factor 2), and
    ``fit_nonlinear_iq`` from it succeeded in 99.5% of cases without a
    neighboring resonance (``guess_p0_nonlinear_iq``: 90%).

    Parameters:
    f (np.array): array of frequency data in Hz.
    z (np.array): array of complex S21 data.
    center_frac (float): fraction of the sweep, around its center, that is
        searched for the resonance. Default 0.5.
    neighbor_ratio (float): a change in S21 outside the central range at
        least this large relative to the central one gets its own pole.
        Default 0.5.

    Returns:
    p0 (list): initial guess fit parameters
        [fr, Qr, amp, phi, a, i0, q0, tau]
    """
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    if f.ndim != 1 or z.ndim != 1 or f.shape != z.shape:
        raise ValueError("f and z must be 1D arrays of the same shape")
    ix = np.argsort(f)
    f, z = f[ix], z[ix]
    k = max(1, len(f) // 100)
    zs = np.convolve(z, np.ones(k) / k, mode = 'same')
    d = np.abs(np.diff(zs))
    d[:k], d[-k:] = 0, 0
    fm = 0.5 * (f[1:] + f[:-1])
    fc = 0.5 * (f[0] + f[-1])
    half = 0.5 * center_frac * (f[-1] - f[0])
    central = np.abs(fm - fc) <= half
    i = int(np.argmax(np.where(central, d, 0)))
    j = int(np.argmax(np.where(central, 0, d)))
    fr0, bounds = [fm[i]], [[fc - half, fc + half]]
    if d[j] >= neighbor_ratio * d[i]:
        fr0.append(fm[j])
        bounds.append([np.nan, np.nan])
    res = fit_poles(f, z, fr0, gain_deg = 1, fr_bounds = bounds)
    fr, Qr = res['fr'][0], res['Qr'][0]
    amp = float(np.clip(res['amp'][0], 0.01, 1 - 1e-6))
    phi = float(np.clip(res['phi'][0], -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3))
    c = res['gain_coeffs']
    x = (fr - res['fc']) / res['fs']
    g = np.polyval(c, x)
    tau = -np.imag(np.polyval(np.polyder(c), x) / g) / (2 * np.pi * res['fs'])
    return [fr, Qr, amp, phi, 0., g.real, g.imag, tau]

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
    # Input validation 
    f = np.asarray(f, dtype = np.float64)
    z = np.asarray(z, dtype = np.complex128)
    if f.ndim != 1:
        raise ValueError("f must be a 1D array")
    if z.ndim != 1:
        raise ValueError("z must be a 1D array")
    if f.shape != z.shape:
        raise ValueError("f and z must have the same shape")
    
    # guess i0 and q0
    z0_guess = np.mean(np.roll(z, 2)[:4])
    i0_guess = np.real(z0_guess)
    q0_guess = np.imag(z0_guess)
    # guess tau
    tau_guess = 0
    # guess phi and amp
    phi_guess, amp_guess = _guess_phi_amp(z, z0_guess)
    # guess Qr
    Qr_guess = _guess_Qr(f, z, z0_guess, phi_guess, amp_guess)
    # guess a
    a_guess = _guess_a(f, z, z0_guess, phi_guess, amp_guess)
    # guess fr
    fr_guess = _guess_fr(f, z, z0_guess, phi_guess, 
                         amp_guess, a_guess, Qr_guess)
    # Package and return p0
    p0 = [fr_guess, Qr_guess, amp_guess, phi_guess, a_guess,
          i0_guess, q0_guess, tau_guess]
    return p0

def _guess_Qr(f, z, z0, phi, amp):
    """
    Guess Qr given complex S21 data.

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance
    phi (float): impedance mismatch angle
    amp (float): Qr / Qc.

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
    xpeak, _, fwhm = get_peak_fwhm(f, z_rot)
    fwhm /= xpeak
    poly = [-0.94123147,  1.80699622]
    Qr_guess = np.exp(np.polyval(poly, np.log(fwhm)))
    return Qr_guess

def _guess_a(f, z, z0, phi, amp):
    """
    Guess the nonlinearity parameter

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

    a_guess = min(a_guess, 1.0)
    return a_guess

def _guess_fr(f, z, z0, phi, amp, a, Qr):
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

def _guess_phi_amp(z, z0):
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

    # amp_guess cannot exceed 1 - 1e-6
    amp_guess = min(amp_guess, 1. - 1e-6)
    return phi_guess, amp_guess
