import numpy as np
from numba import njit, float64, int64, complex128
from numba.types import UniTuple
from citkid.res.funcs import get_y
"""
KID S21 sweep and noise timestream generation functions. 

KID noise is modelled as 
Sxx = sxx_white * (1 + (f_knee / f) ** alpha) 
        / (1 + (f / fqp) ** 2) / (1 + (f / frd) ** 2)
SAA = sAA_white / (1 + (f / frd) ** 2)
where fqp = 1 / (2 * pi * tau_qp),  frd = 1 / (2 * pi * tau_rd),
tau_qp is the QP lifetime, tau_rd is the resonator ringdown time,
f_knee is the knee frequency for 1/f noise, alpha is the exponent of 1/f noise,
sxx_white is the white level of fractional frequency noise in 1 / Hz, and 
sAA_white is the white level of amplitude noise in dBc / Hz.
"""
# Add to environment.yml: rocket_fft=0.3.1

@njit(
    UniTuple(float64[:], 2)(
        int64, float64, float64, float64, float64, float64, float64, float64
    ),
    cache=True
)
def generate_noise(
    n, fs, alpha, f_knee, tau_qp, tau_rd, sxx_white, sAA_white
):
    """
    Generate fractional frequency and amplitude noise timestreams, where the 
    noise PSD is given by 
        Sxx: 1/f + white + QP rolloff + ringdown rolloff
        SAA: white + ringdown rolloff
 
    Parameters:
    n (int): Number of samples.
    fs (float): Sampling frequency in Hz.
    alpha (float): Exponent of 1/f noise.
    f_knee (float): Knee frequency for 1/f noise in Hz.
    tau_qp (float): Quasiparticle lifetime in seconds. Set to negative value to
        disable.
    tau_rd (float): Ringdown time in seconds. Set to negative value to disable.
    sxx_white (float): fractional frequency noise psd white value in 1 / Hz.
    sAA_white (float): amplitude noise psd white value in dBc / Hz.
 
    Returns:
    x (np.ndarray): real-valued fractional frequency noise signal.
    a (np.ndarray): real-valued amplitude noise signal.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    m = freqs.size
 
    spec_half_x = np.random.normal(0.0,1.0,m) + 1j * np.random.normal(0.0,1.0,m)
    spec_half_A = np.random.normal(0.0,1.0,m) + 1j * np.random.normal(0.0,1.0,m)
 
    f = freqs.copy()
    if len(f) > 1:
        f[0] = f[1]
    else:
        f[0] = 1.0
 
    psdx = sxx_white * (1 + (f_knee / f) ** alpha)
    psda = sAA_white
    psdx_ro = np.ones(len(f), dtype=np.float64) * psdx
    psda_ro = np.ones(len(f), dtype=np.float64) * psda
 
    if tau_qp > 0:
        fqp = 1 / (2 * np.pi * tau_qp)
        psdx_ro = psdx_ro / (1 + (f / fqp) ** 2)
    if tau_rd > 0:
        frd = 1 / (2 * np.pi * tau_rd)
        psdx_ro = psdx_ro / (1 + (f / frd) ** 2)
        psda_ro = psda_ro / (1 + (f / frd) ** 2)
 
    # Scaling derived so that measuring psd = 2*|rfft(x)|**2*dt/n on the
    # output recovers the target psd_x / sAA on average:
    #   E[psd_est] = 4 * A^2 * dt / n  ==  target_psd
    #   => A = 0.5 * sqrt(target_psd * n * fs)
    amp_scale = 0.5 * np.sqrt(n * fs)
    spec_x = spec_half_x * amp_scale * np.sqrt(psdx_ro)
    spec_a = spec_half_A * amp_scale * np.sqrt(psda_ro)
    spec_x[0] = 0
    spec_a[0] = 0
 
    x = np.fft.irfft(spec_x, n)
    a = np.fft.irfft(spec_a, n)
 
    return x, a

@njit(float64[:](float64[:], int64, int64), cache=True)
def block_mean(x, nsamps, m):
    """
    Average a timestream in blocks of nsamps, returning an array of length m. 
    numba compatible version of np.mean(x.reshape(nsamps, m), axis=0).

    Parameters:
    x (np.array): input array of length nsamps * m.
    nsamps (int): number of samples to average over.
    m (int): number of blocks to average over.

    Returns:
    out (np.array): array of length m, where each element is the mean of a block 
        of nsamps samples.
    """
    out = np.zeros(m, dtype=np.float64)
    for i in range(nsamps):
        for j in range(m):
            out[j] += x[i * m + j]
    for j in range(m):
        out[j] /= nsamps
    return out

@njit([
    complex128[:](float64[:], float64[:], float64, float64), 
    complex128[:](float64[:], float64[:], float64[:], float64[:]),
    complex128[:](float64, float64[:], float64[:], float64[:])
    ],
    cache=True
)
def get_S21_from_xA(
    f,
    p,
    x,
    A
):
    """
    Calculate S21 from fractional frequency and amplitude noise timestreams.

    Parameters:
    f (np.array): array of frequencies in Hz.
    p (np.array): array of resonator parameters [fr, Qr, amp, phi, a].
    x (np.array): fractional frequency noise timestream.
    A (np.array): amplitude noise timestream.

    Returns:
    S21 (np.array): complex S21 parameter.
    """
    fr, Qr = p[0], p[1]
    amp, phi, a = p[2], p[3], p[4]
    fr_with_noise = fr  * (1 + x)
    amp_with_noise = amp * (1 + A) 

    y0 = Qr * (f - fr_with_noise) / fr_with_noise
    y = get_y(y0, a, True)
    
    S21 = (
        1. - (amp_with_noise / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y)
        )
    
    return S21

@njit(
    complex128[:](float64[:], float64, float64, float64, float64,
                float64, float64, int64, float64[:]), 
                cache = True
)
def get_S21_vs_freq(
    f,
    alpha,
    f_knee,
    tau,
    sxx_white,
    sAA_white,
    fs,
    nsamps,
    p
):
    """
    Generate complex S21 data vs frequency. Applies realistic noise to the 
    sweep data.

    Parameters:
    f (np.array): array of frequencies in Hz.
    alpha (float): Exponent of 1/f noise.
    f_knee (float): Knee frequency for 1/f noise in Hz.
    tau (float): Quasiparticle lifetime in seconds. Set to negative value to
        disable.
    sxx_white (float): fractional frequency noise psd white value in 1 / Hz.
    sAA_white (float): amplitude noise psd white value in 1 / Hz.
    fs (float): Sampling frequency in Hz.
    nsamps (int): Number of samples to average over per point.
    p (np.array): array of resonator parameters [fr, Qr, amp, phi, a].

    Returns:
    S21 (np.array): complex S21 array corresponding to f.
    """
    x, A = generate_noise(
        n = len(f) * nsamps,
        fs = fs,
        alpha = alpha,
        f_knee = f_knee,
        tau_qp = tau,
        tau_rd = -1.,
        sxx_white = sxx_white,
        sAA_white = sAA_white
    )
    x = block_mean(x, nsamps, len(f))
    A = block_mean(A, nsamps, len(f))

    S21 = get_S21_from_xA(f, p, x, A)
    return S21

@njit(
    complex128[:](float64[:], float64, float64, float64, float64,
                float64, float64, int64, float64[:], float64[:], float64), 
                cache = True
)
def get_S21_vs_freq_dual(
    f,
    alpha,
    f_knee,
    tau,
    sxx_white,
    sAA_white,
    fs,
    nsamps,
    p1,
    p2,
    fl_phase
):
    """
    Generate complex S21 data vs frequency for two resonators. Applies realistic
    noise to the sweep data. Assumes both resonators have the same noise 
    PSDs.

    Parameters:
    f (np.array): array of frequencies in Hz.
    alpha (float): Exponent of 1/f noise.
    f_knee (float): Knee frequency for 1/f noise in Hz.
    tau (float): Quasiparticle lifetime in seconds. Set to negative value to
        disable.
    sxx_white (float): fractional frequency noise psd white value in 1 / Hz.
    sAA_white (float): amplitude noise psd white value in 1 / Hz.
    fs (float): Sampling frequency in Hz.
    nsamps (int): Number of samples to average over per point.
    p1 (np.array): array of resonator parameters [fr, Qr, amp, phi, a] for
        resonator 1.
    p2 (np.array): array of resonator parameters [fr, Qr, amp, phi, a] for
        resonator 2.
    fl_phase (float): phase of the feedline between the two resonators.

    Returns:
    S21 (np.array): complex S21 array corresponding to f.
    """
    S21a = get_S21_vs_freq(
        f,
        alpha,
        f_knee,
        tau,
        sxx_white,
        sAA_white,
        fs,
        nsamps,
        p1
    )
    S21b = get_S21_vs_freq(
        f,
        alpha,
        f_knee,
        tau,
        sxx_white,
        sAA_white,
        fs,
        nsamps,
        p2
    )
    gamma = np.exp(-2j * fl_phase)
    S21 = S21a * S21b / (1 - (1 - S21a) * (1 - S21b) * gamma)
    return S21


@njit(
    complex128[:](float64, int64, float64, float64, float64,
                 float64, float64, float64, float64, float64[:]), cache = True
)
def get_S21_noise_ts(
    ft,
    npoints,
    alpha,
    f_knee,
    tau_qp,
    tau_rd,
    sxx_white,
    sAA_white,
    fs,
    p
):
    """
    Generate complex S21 noise timestream data.

    Parameters:
    ft (float64): tone frequency in Hz.
    npoints (int): number of points in the timestream.
    alpha (float): Exponent of 1/f noise.
    f_knee (float): Knee frequency for 1/f noise in Hz.
    tau_qp (float): Quasiparticle lifetime in seconds. Set to negative value to
        disable.
    tau_rd (float): Ringdown time in seconds. Set to negative value to disable.
    sxx_white (float): fractional frequency noise psd white value in 1 / Hz
    sAA_white (float): amplitude noise psd white value in 1 / Hz.
    fs (float): Sampling frequency in Hz.
    p (np.array): array of resonator parameters [fr, Qr, amp, phi, a].

    Returns:
    S21 (np.array): Complex S21 timestream data.
    """
    x, A = generate_noise(
        n = npoints,
        fs = fs,
        alpha = alpha,
        f_knee = f_knee,
        tau_qp = tau_qp,
        tau_rd = tau_rd,
        sxx_white = sxx_white,
        sAA_white = sAA_white
    )
    S21 = get_S21_from_xA(ft, p, x, A)
    return S21



 
 


