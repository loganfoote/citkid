### Generate random resonance data
import numpy as np
from numba import jit, vectorize
from scipy import signal

# @jit(nopython=True)
def generate_data(npoints_fine = 600, npoints_gain = 50, noise_factor = 1,
                  generate_noise = True):
    """
    Generates random resonance data. Assumes resonant frequencies were found
    from the point of max spacing. I can add other options if desired.

    Parameters:
    npoints_fine (int): number of points in the fine sweep
    npoints_gain (int): number of points in the gain sweep
    noise_factor (float): factor that scales the noise. Lower this for lower
        noise, or raise for higher noise

    Returns:
    ffine (np.array): fine sweep frequency data in Hz
    zfine (np.array): fine sweep complex S21 data
    fgain (np.array): gain sweep frequency data in Hz
    zgain (np.array): gain sweep complex S21 data
    """
    fr, Qr, amp, phi, a, p_amp, p_phase, fine_bw, f_noise_std, a_noise_std =\
                                                 generate_resonance_parameters()
    f_noise_std *= noise_factor
    a_noise_std *= noise_factor
    # Rough sweep
    f = np.linspace(fr - fine_bw / 2, fr + fine_bw / 2, 400)
    f_noisy = f + np.random.normal(0, f_noise_std, len(f))
    z = nonlinear_iq_simple(f_noisy, fr, Qr, amp, phi, a, p_amp, p_phase)
    z *= np.random.normal(1, a_noise_std, len(z))
    f0 = update_fr_spacing(f, z)
    # Fine sweep
    ffine = np.linspace(f0 - fine_bw / 2, f0 + fine_bw / 2, npoints_fine)
    f_noisy = ffine + np.random.normal(0, f_noise_std, len(ffine))
    zfine = nonlinear_iq_simple(f_noisy, fr, Qr, amp, phi, a, p_amp, p_phase)
    zfine *= np.random.normal(1, a_noise_std, len(zfine))
    # Gain sweep
    fgain = np.linspace(f0 - 5 * fine_bw, f0 + 5 * fine_bw, npoints_gain)
    f_noisy = fgain + np.random.normal(0, f_noise_std, len(fgain))
    zgain = nonlinear_iq_simple(f_noisy, fr, Qr, amp, phi, a, p_amp, p_phase)
    zgain *= np.random.normal(1, a_noise_std, len(zgain))
    # Noise data
    if generate_noise:
        znoise = generate_timestream(f0, fr, Qr, amp, phi, a, p_amp, p_phase)
    else:
        znoise = np.array([], dtype = complex)
    return ffine, zfine, fgain, zgain, f0, znoise

@jit(nopython=True)
def generate_resonance_parameters():
    """
    Generates random resonance model parameters

    Returns:
    f (array-like): array of frequencies in Hz
    fr (float): resonant frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl
    p_amp (array-like): gain polynomial coefficients
    p_phase (array-like): phase polynomial coefficients
    fine_bw (float): fine sweep bandwidth in Hz
    f_noise_std (float): frequency noise standard deviation
    a_noise_std (float): amplitude noise standard deviation
    """
    random = lambda low, high: np.random.uniform(low, high)
    random_log = lambda low, high: np.exp(np.random.uniform(np.log(low),
                                                            np.log(high)))
    fr  = random(0.4e9, 2.4e9)
    # Qr  = random(1e3, 500e3)
    Qr  = random(20e3, 60e3)
    # amp = random(1e-5, 1 - 1e-5)
    amp = random(0.8, 0.99)
    # phi = random(-np.pi / 2, np.pi / 2)
    phi = random(-np.pi / 16, np.pi / 16)
    # a   = random_log(1e-5, 0.8)
    a = random_log(1e-2, 0.6)
    # p_amp  = np.array([random(-5e-17, 5e-17), random(-8e-5, 8e-5),
                       # random(-150, 20)])
    p_amp  = np.array([random(-5e-21, 5e-21), random(-8e-8, 8e-8),
                       random(-120, 20)])
    # p_phase = np.array([-random_log(1e-9, 1e-6), random(-1e3, 1e3)])
    p_phase = np.array([-random_log(1e-9, 1e-8), random(-1, 1)])
    fwhm = fr / Qr
    fine_bw = random(fwhm * 3, fwhm * 6)
    # f_noise_std = random(1, 500)
    f_noise_std = random(1, 100)
    # a_noise_std = random(1e-5, 3e-3)
    a_noise_std = random(1e-5, 1e-3)
    return fr, Qr, amp, phi, a, p_amp, p_phase, fine_bw, f_noise_std, a_noise_std

@jit(nopython=True)
def get_noise(f, std_dev):
    """
    Generates z noise. It won't be Gaussian in real life but this is good
    enough for testing

    Parameters:
    f (np.array): frequency array
    std_dev (float): standard deviation of the noise

    Returns:
    z (np.array): random z noise
    """
    i = np.random.normal(0, std_dev, len(f))
    q = np.random.normal(0, std_dev, len(f))
    z = (i + 1j * q)
    return z
################################################################################
########################### Resonance model ####################################
################################################################################
@jit(nopython = True)
def nonlinear_iq_simple(f, fr, Qr, amp, phi, a, p_amp, p_phase):
    """
    Describes the transmission through a nonlinear resonator

                                    (j phi)
        1 -        Qr             e^
             --------------  X  ------------
              Qc * cos(phi)       (1+ 2jy)

        where the nonlinearity of y is described by
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr
    This equation is further multiplied by a real polynomial determined by the
    coefficients in p_amp and a phase shift polynomial determined by the
    coefficients in p_phase

    Parameters:
    f (array-like): array of frequencies in Hz
    fr (float): resonant frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl
    p_amp (array-like): gain polynomial coefficients
    p_phase (array-like): phase polynomial coefficients

    Returns:
    z (np.array): array of complex IQ data corresponding to f
    """
    deltaf = f - fr
    yg = Qr * deltaf / fr
    y = get_y(yg, a, True)
    s21_res = (1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y))
    f0 = np.mean(f)
    s21_readout = 10 ** (polyval(p_amp, f - f0) / 20) + 0j
    s21_readout *= np.exp(1j * polyval(p_phase, f - f0))
    z = s21_readout * s21_res
    return z

@jit(nopython = True)
def get_y(yg, a, largest = True):
    """
    Calculates the largest or smallest real root of
        yg = y + a / (1 + y^2)

    Parameters:
    yg (float or np.array): unmodified resonance shift
        yg = Qr * (f - fr) / fr
    a (float): nonlinearity parameter
    largest (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    y (float or np.array): largest or smallest real root of the above equation
    """
    y = cardan(4.0, -4.0 * yg, 1.0, -(yg + a), largest)
    return y

@vectorize(nopython = True)
def cardan(a, b, c, d, largest = True):
    """
    Analyticaly calculates the largest real root of a 3rd-order polynomial
    Based on code from https://github.com/Wheeler1711/submm_python_routines

    Parameters:
    a, b, c, d (float): polynomial coefficients
    largest(bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    root (float): largest or smallest real root
    """
    J = np.exp(2j * np.pi / 3)
    Jc = 1 / J
    u = np.empty(2, np.complex128)
    z0 = b / 3 / a
    a2, b2 = a * a, b * b
    p = -b2 / 3 / a2 + c / a
    q = (b / 27 * (2 * b2 / a2 - 9 * c / a) + d) / a
    D = -4 * p * p * p - 27 * q * q
    r = np.sqrt(-D / 27 + 0j)
    one_third = 1 / 3.0
    u = ((-q - r) / 2) ** one_third
    v = ((-q + r) / 2) ** one_third
    w = u * v
    w0 = np.abs(w + p / 3)
    w1 = np.abs(w * J + p / 3)
    w2 = np.abs(w * Jc + p / 3)
    if w0 < w1:
        if w2 < w0:
            v *= Jc
    elif w2 < w1:
        v *= Jc
    else:
        v *= J
    roots = np.asarray((u + v - z0, u * J + v * Jc - z0, u * Jc + v * J - z0))
    where_real = np.where(np.abs(np.imag(roots)) < 1e-15)
    if D > 0:
        # three real roots: return the max
        if largest:
            return np.max(np.real(roots))
        else:
            return np.min(np.real(roots))
    else:
        # one real root: return value with smalles imaginary component
        return np.real(roots[np.argsort(np.abs(np.imag(roots)))][0])

@jit(nopython=True)
def update_fr_spacing(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the max spacing between adjacent IQ points

    Parameters:
    f (np.array): frequency data in Hz
    z (np.array): complex S21 data

    Returns:
    f0_new (float): updated frequency in Hz
    """
    spacing = np.abs(z[1:] - z[:-1])
    ix = np.argmax(spacing)
    if ix == 0:
        f0_new = f[0]
    elif ix == len(f):
        f0_new = f[-1]
    else:
        f0_new = (f[ix] + f[ix + 1]) / 2
    return f0_new

@jit(nopython=True)
def polyval(p, x):
    """
    Performs the same function as np.polyval, but with numba compatability
    """
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[i]
    return y

################################################################################
########################### Noise timestream ###################################
################################################################################
def generate_timestream(fnoise, fr, Qr, amp, phi, a, p_amp, p_phase, tlen = 10):
    """
    Generates a raw IQ time stream, purely noise

    Parameters:
    fnoise (float): streaming frequency in Hz
    fr (float): resonant frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl
    p_amp (array-like): gain polynomial coefficients
    p_phase (array-like): phase polynomial coefficients
    tlen (float): timestream length in seconds

    Returns:
    z (np.array): IQ time series complex S21 data
    """
    # Sampling parameters
    fsample = 10_000 # sampling frequency [Hz]
    # Sxx white and 1/f noise parameters
    sxx_white = 2e-17 # white noise term [1/Hz]
    fknee = 1 # knee of 1/f component [Hz]

    # Roll-off tau
    tau_qp = 0.001 # qp lifetime [s]
    # Amplifier white and 1/f noise parameters
    spar_sper = 10 # Ratio of Spar/Sper
    fknee_amp = 1 # knee of amplifier noise term [1/Hz]
    #
    # Create white noise plus 1/f
    sig_white = np.sqrt(sxx_white*fsample/2)
    dx_noise = np.random.normal(0, sig_white, fsample*tlen)
    steps = np.random.choice([-1, 1], size=fsample*tlen)
    position = np.cumsum(steps)
    position = position*4*sig_white*fknee/fsample
    dx_noise += position
    #
    # Add some cosmic ray events
    cr_peak_dx_vec = -1*np.array([1e-4, 2e-5])
    cr_t0_vec = np.array([0.23, 0.58])*tlen
    dx_cr = np.zeros(fsample*tlen)
    for ii in range(len(cr_t0_vec)):
        t0 = cr_t0_vec[ii]
        cr_peak_dx = cr_peak_dx_vec[ii]
        tvec = np.linspace(0, fsample*tlen-1, fsample*tlen)/fsample
        xvec = np.heaviside(tvec-t0, 1)
        indx = np.where(xvec > 0)
        xvec[indx] = cr_peak_dx*np.exp(-1*(tvec[indx]-t0)/tau_qp)
        dx_cr += xvec
#    Qr_vec = 1/(1/Qr - dx_cr/20)
#    amp_vec = Qr_vec/Qr*amp
#    print(np.min(Qr_vec), np.max(Qr_vec))
#    print(np.min(amp_vec), np.max(amp_vec))
    dx_noise += dx_cr
    #
    # Multiply by a low-pass filter
    sos = signal.butter(1, 1/(6*tau_qp), 'lp', fs=fsample, output='sos')
    dx_noise = signal.sosfilt(sos, dx_noise)
    df_noise = fr*dx_noise
    # Convert df to S21
    f = fnoise - df_noise
    z = nonlinear_iq_simple(f, fr, Qr, amp, phi, a, p_amp, p_phase)
    #
    # Add white amplifier noise plus 1/f
    ftest = fnoise*(1 + np.array([0, sig_white]))
    ztest = nonlinear_iq_simple(ftest, fr, Qr, amp, phi, a, p_amp, p_phase)
    sigma_s21 = np.abs(ztest[1] - ztest[0])/np.sqrt(spar_sper)
    z1 = np.random.normal(0, sigma_s21, fsample*tlen)
    steps = np.random.choice([-1, 1], size=fsample*tlen)
    position = np.cumsum(steps)
    position = position*4*sigma_s21*fknee_amp/fsample
    z1 += position
    z2 = np.random.normal(0, sigma_s21, fsample*tlen)
    steps = np.random.choice([-1, 1], size=fsample*tlen)
    position = np.cumsum(steps)
    position = position*4*sigma_s21*fknee_amp/fsample
    z2 += position
    z += z1 + 1.j*z2
    return z
