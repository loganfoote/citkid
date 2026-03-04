from numba import jit
import numpy as np
from citkid.res.funcs import get_y
'''Code to generate KID data for simulations'''

def get_resonance_s21(
    f,
    x_signal,
    alpha,
    f_knee,
    tau_qp,
    sxx_white,
    sAA_white,
    dt,
    fr,
    Qr,
    amp,
    phi,
    a,
    p_amp0,
    p_amp1,
    p_amp2,
    p_phase0,
    p_phase1
):
    """
    Get S21 of a resonance without added gain or phase terms.

    Parameters:
    f (np.array): Array of frequencies in Hz.
    x_signal (np.array): Array of fractional frequency signal in Hz,
        corresponding to f.
    alpha (float): Exponent of 1/f noise in frequency noise.
    fr (float): Resonant frequency in Hz.
    Qr (float): Total quality factor.
    amp (float): Qr / Qc, where Qc is the coupling quality factor.
    phi (float): Rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): Nonlinearity parameter.

    Returns:
    z (np.array): Complex S21 data.
    """
    x_with_noise = noise_1f_white_rolloff(
        n = len(f),
        fs = 1 / dt,
        alpha = alpha,
        f_knee = f_knee,
        fc = 1 / tau_qp,
        white_level = np.sqrt(sxx_white)
    ) 
    if x_signal is not None:
        x_with_noise += x_signal
    fr_with_noise = x_with_noise * fr + fr
    amp_noise = noise_1f_white_rolloff(
        n = len(f),
        fs = 1 / (dt),
        alpha = 0.,
        f_knee = 1., # no 1/f in amp noise
        fc = 1 / tau_qp,
        white_level = np.sqrt(sAA_white)
    )

    fr = fr_with_noise
    deltaf = f - fr
    y0 = Qr * deltaf / fr
    y = get_y(y0, a, True)
    z0 = 1 / (1. + 2.j * y)
    theta = np.angle(z0)
    amp_noise = amp_noise * np.exp(1j * theta + 1j * np.pi)
    z = (
        1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y)
        + amp_noise
    )

    z_system = 10 ** (polyval([p_amp0, p_amp1, p_amp2], f - fr) / 20) + 0j
    z_system *= np.exp(1j * polyval([p_phase0, p_phase1], f - fr))
    z *= z_system
    xt_real = x_with_noise #+ 1 - f / fr
    return z, xt_real

@jit(nopython=True)
def polyval(p, x):
    """
    Perform the same function as np.polyval, but numba compatible.
    """
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[i]
    return y

def noise_1f_white_rolloff(
    n,
    fs,
    alpha = 1.0,
    f_knee = 1.0,
    fc = None,
    white_level = 1.0,
):
    """
    Generate Gaussian noise with:
        - white floor.
        - 1/f^alpha component.
        - optional single-pole rolloff.

    Parameters:
    n (int) : number of samples.
    fs (float) : sample rate.
    alpha (float) : exponent in 1/f^alpha.
    f_knee (float) : frequency where 1/f^alpha equals white level.
    fc (float or None) : single-pole rolloff frequency (Hz).
    white_level (float) : white noise PSD level (linear units).

    Returns:
    x (ndarray) : real-valued noise time series.
    """

    freqs = np.fft.rfftfreq(n, d=1/fs)

    # Random complex Gaussian spectrum
    spec = np.random.normal(size = freqs.size) + 1j*np.random.normal(
                                                            size = freqs.size)

    # Avoid f=0 divergence
    f = freqs.copy()
    f[0] = f[1] if len(f) > 1 else 1.0

    # PSD model
    psd = white_level * (1 + (f_knee / f)**alpha)

    if fc is not None:
        psd /= (1 + (f / fc)**2)  # single-pole low-pass

    # Shape spectrum (amplitude ∝ sqrt(PSD))
    spec *= np.sqrt(psd)

    # Remove DC
    spec[0] = 0

    # Back to time domain
    x = np.fft.irfft(spec, n)

    return x
