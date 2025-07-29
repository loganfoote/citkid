import numpy as np
from ..res.fitter import fit_iq_circle
from .plot import plot_cal, plot_timestream, plot_psd
from .psd import *
from .cosmic_rays import remove_cosmic_rays

def compute_psd(ffine, zfine, fnoise, znoise, dt, fnoise_offres = None,
                znoise_offres = None, dt_offres = None, flag_crs = True,
                deglitch_nstd = 5, plot_calq = True, plot_psdq = True,
                plot_timestreamq = True, min_cal_points = 5,
                circfit_npoints = None, xcal_weight_sigma = None,
                xcal_weight_theta0 = 0.0, **cr_kwargs):
    """
    Computes parallel and perpendicular noise PSDs, as well as Sxx

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data, with gain removed
    fnoise (float): on-resonance noise tone frequency in Hz
    znoise (array-like or None): on-resonance complex noise data, with gain
        removed. For off-resonance noise only, set this to None
    dt (float): sample time of the on-resonance noise timestream in s
    fnoise_offres (float): off-resonance noise tone frequency in Hz
    znoise_offres (array-like or None): off-resonance complex noise data, with
        gain removed. For on-resonance noise only, set this to None
    dt_offres (float): sample time of the off-resonance noise timestream in s
    flag_crs (bool): If True, flags cosmic rays and returns a list of indices
        where they were found. If False, does not flag cosmic rays
    deglitch_nstd (float or None): threshold for removing glitched data points from
        the timestream, or None to bypass deglitching. Points more than
        deglitch_nstd times the standard deviations of the theta timestream are
        removed from the data.
    plot_calq (bool): If True, plots the calibration figure. Else, returns None
        for the figure
    plot_psdq (bool): If True, plots the PSD figure. Else, returns None for the
        figure
    plot_timestreamq (bool): If True, plots the timestream figure. Else, returns
        None for the figure
    xcal_weight_sigma (float): stdev of gaussian weight function for
        fitter in radians. Defaults to None for no weighting.
    xcal_weight_theta0 (float): center point in radians for gaussian
        weight function
    **cr_kwargs: kwargs for cosmic ray removal

    Returns:
    psd_onres (tuple): on-resonance psd data, or None
        f_psd (np.array): frequency array for PSDs
        spar  (np.array): PSD of noise parallel to IQ loop
        sper  (np.array): PSD of noise perpendicular to IQ loop
        sxx   (np.array): PSD of fractional frequency noise
    psd_offres (tuple): off-resonance psd data, or None
        f_psd_offres (np.array): frequency array for PSDs
        spar_offres  (np.array): PSD of noise parallel to IQ loop
        sper_offres  (np.array): PSD of noise perpendicular to IQ loop
    timestream_onres (tuple): on-resonance timestream data, or None
        theta (np.array): theta timestream data, with no cosmic ray removal or
            deglitching
        x (np.array): fractional frequency shift timestream with cosmic rays
            removed and deglitching
    timestream_offres (tuple): off-resonance timestream data, or None
        theta_offres (np.array): theta timestream data, with no cosmic ray
            removal or deglitching
    cr_indices (np.array): indices into theta where cosmic rays were found
    theta_range (list): [lower, upper] range of theta over which x vs theta was
        fit to calibrate x
    poly (np.array): x vs theta polynomial fit parameters
    xcal_data (tuple): x vs theta calibration data. Not cut to theta_range
        x (np.array): fractional frequency shift data
        theta (np.array): theta data
    figs (tuple): plots
        fig_cal (plt.figure): plot of the IQ loop with noise balls and
            calibration
        fig_psd (plt.figure): plot of the PSDs
        fig_timestream (plt.figure): plot of the timestream data
    """
    assert len(zfine), "No fine scan points provided to compute_psd"
    # Prepare data
    ffine, zfine= np.array(ffine), np.array(zfine)
    ix = np.argsort(ffine)
    ffine, zfine = ffine[ix], zfine[ix]
    # Fit circle
    if znoise is None:
        origin = 0
        radius = 1
        popt_circle = [np.nan, np.nan, np.nan]
    else:
        popt_circle, _ = fit_iq_circle(zfine, plotq = False)
        origin = popt_circle[0] + 1j * popt_circle[1]
        radius = popt_circle[2]
    # Extract theta and x
    if znoise_offres is not None:
        znoise_offres = np.array(znoise_offres)
        #theta_fine, theta_offres, theta_offres_clean, A_offres_clean, (ix0_off, ix1_off) =\
        theta_fine, theta_offres, theta_offres_clean, A_offres_clean =\
        calibrate_timestreams(origin, ffine, zfine, fnoise_offres,
                              znoise_offres, dt_offres, deglitch_nstd,
                              flag_crs = False, offres = True, min_cal_points = min_cal_points,
                              xcal_weight_theta0 = xcal_weight_theta0, xcal_weight_sigma = xcal_weight_sigma)

        spar_offres = 10 * np.log10(get_psd(radius * theta_offres_clean, dt_offres))
        sper_offres = 10 * np.log10(get_psd(A_offres_clean, dt_offres))
        f_psd_offres = np.fft.rfftfreq(len(theta_offres_clean), d = dt_offres)
    else:
        theta_offres = None
        f_psd_offres, spar_offres, sper_offres = None, None, None
        #ix0_off, ix1_off = None, None
    if znoise is not None:
        znoise = np.array(znoise)

        theta_fine, theta, theta_clean, A_clean, theta_range, poly, x, cr_indices, (ix0, ix1) =\
        calibrate_timestreams(origin, ffine, zfine, fnoise, znoise, dt,
                              deglitch_nstd, flag_crs = flag_crs, offres = False,
                              min_cal_points = min_cal_points,
                              xcal_weight_theta0 = xcal_weight_theta0, xcal_weight_sigma = xcal_weight_sigma,
                              **cr_kwargs)

        sxx = get_psd(x, dt)
        spar = 10 * np.log10(get_psd(radius * theta_clean, dt))
        sper = 10 * np.log10(get_psd(A_clean, dt))
        f_psd = np.fft.rfftfreq(len(theta_clean), d = dt)
    else:
        theta, poly, x, cr_indices, theta_range = None, None, None, None, None
        f_psd, spar, sper, sxx = None, None, None, None
        theta_clean = None
        ix0, ix1 = None, None
    # Plots
    if plot_calq:
        fig_cal = plot_cal(ffine, zfine, popt_circle, fnoise, znoise,
                           znoise_offres, theta_range, theta_fine, theta_clean,
                           poly, (ix0, ix1))
    else:
        fig_cal = None
    if plot_timestreamq:
        fig_timestream = plot_timestream(dt, theta, theta_clean, dt_offres,
                                         theta_offres, x, cr_indices)
    else:
        fig_timestream = None
    if plot_psdq:
        fig_psd = plot_psd(f_psd, spar, sper, sxx,
                           f_psd_offres, spar_offres, sper_offres)
    else:
        fig_psd = None
    psd_onres = [f_psd, spar, sper, sxx]
    psd_offres = [f_psd_offres, spar_offres, sper_offres]
    timestream_onres = [theta, x]
    timestream_offres = [theta_offres]
    if fnoise is not None:
        xcal_data = (1 - ffine / fnoise, theta_fine)
    else:
        xcal_data = [None]
    figs = (fig_cal, fig_timestream, fig_psd)
    return psd_onres, psd_offres, timestream_onres, timestream_offres,\
           cr_indices, theta_range, poly, xcal_data, figs

def compute_psd_simple(ffine, zfine, fnoise, znoise, dt, deglitch_nstd = 5,
                       offres = False):
    """
    Computes an approximation of parallel and perpendicular noise PSDs
    by rotating the noise data to 0, 0 and returning PSDs of I, Q

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data, with gain removed
    fnoise (float): on-resonance noise tone frequency in Hz
    znoise (array-like or None): on-resonance complex noise data, with gain
        removed. For off-resonance noise only, set this to None
    dt (float): sample time of the on-resonance noise timestream in s
    deglitch_nstd (float or None): threshold for removing glitched data points from
        the timestream, or None to bypass deglitching. Points more than
        deglitch_nstd times the standard deviations of the theta timestream are
        removed from the data
    offres (bool): If True, sets the circle origin to 0 for off-res noise. Else,
        fits for the circle origin

    Returns:
    f_psd (np.array): frequency array for PSDs
    spar  (np.array): PSD of Q noise
    sper  (np.array): PSD of I noise
    """
    # Prepare data
    ffine, zfine= np.asarray(ffine), np.asarray(zfine)
    ix = np.argsort(ffine)
    ffine, zfine = ffine[ix], zfine[ix]
    znoise = np.asarray(znoise).copy()
    # Fit circle
    if not offres:
        popt_circle, _ = fit_iq_circle(zfine, plotq = False)
        origin = popt_circle[0] + 1j * popt_circle[1]
        radius = popt_circle[2]
    else:
        origin = 0. + 0.j
        radius = 1

    # deglitch
    I, Q = np.real(znoise), np.imag(znoise)
    ix = np.abs(np.mean(I) - I) > deglitch_nstd * np.std(I)
    ix = ix | (np.abs(np.mean(Q) - Q) > deglitch_nstd * np.std(Q))
    znoise[ix] = np.mean(znoise)

    # Center data on origin of circle and rotate to center noise ball
    zfine -= origin
    znoise -= origin
    a = np.angle(np.mean(znoise))
    zfine *= np.exp(-1j * a)
    znoise *= np.exp(-1j * a)

    # Get PSDs
    spar = get_psd(np.imag(znoise), dt)
    sper = get_psd(np.real(znoise), dt)
    f_psd = np.fft.rfftfreq(len(znoise), d = dt)
    return f_psd, spar, sper, znoise

################################################################################
##################### Timestream analysis function #############################
################################################################################

def calibrate_timestreams(origin, ffine, zfine, fnoise, znoise, dt,
                          deglitch_nstd, flag_crs, offres = False,
                          min_cal_points = 5, xcal_weight_sigma = None,
                          xcal_weight_theta0 = 0.0, **cr_kwargs):
    """
    Calculates theta and x timestreams given complex IQ noise timestreams.
    1) calculate theta of the sweep data an noise timestream
    2) flag and remove cosmic rays
    3) deglitch data and perform polynomial fit to get x from theta, if not
       offres

    Parameters:
    origin (complex): origin of the iq loop
    ffine (np.array): frequency data in Hz
    zfine (np.array): complex IQ data
    fnoise (float): noise tone frequency in Hz
    znoise (np.array): complex noise timestream
    dt (float): timestream sample time
    deglitch_nstd (float or None): deglitching threshold, or None to bypass
        deglitching
    flag_crs (bool): If True, flags cosmic rays and returns a list of indices
        where they were found. If False, does not flag cosmic rays
    offres (bool): if True, only calibrates theta and bypasses x calibration
    xcal_weight_sigma (float): stdev of gaussian weight function for
        fitter in radians. Defaults to None for no weighting.
    xcal_weight_theta0 (float): center point in radians for gaussian
        weight function
    **cr_kwargs: kwargs for cosmic ray removal

    Returns:
    theta_fine (np.array): theta of the complex sweep data
    theta (np.array): theta timestream
    # If not offres:
    theta_range (np.array): [lower, upper] bound on theta used in the polynomial
        fit to theta_fine versus ffine
    poly (np.array): polynomial fit parameters to theta_fine versus ffine
    x (np.array): deglitched x timestream
    cr_indices (np.array): cosmic ray indices
    """
    assert len(zfine), "No fine scan points provided to calibrate_timestreams"
    theta_fine, theta, A = calculate_theta_A(zfine, znoise, origin)
    if offres:
        znoise_clean = deglitch_timestream(znoise, deglitch_nstd)
        theta_fine, theta_clean, A_clean = calculate_theta_A(zfine,
                                                           znoise_clean, origin)
        return theta_fine, theta, theta_clean, A_clean
    # Remove cosmic rays from theta timestream
    if flag_crs:
        cr_indices, theta_rmvd, A_clean = remove_cosmic_rays(theta, A, dt,
                                                             **cr_kwargs)
    else:
        cr_indices = np.array([], dtype = np.int64)
        theta_rmvd = theta.copy()
        A_clean = A.copy()
    # Deglitch
    if deglitch_nstd is not None:
        theta_clean = deglitch_timestream(theta_rmvd, deglitch_nstd)
        A_clean = deglitch_timestream(A_clean, deglitch_nstd)
    else:
        theta_clean = theta.copy()
    # Calibrate x
    poly, theta_range, (ix0, ix1) = \
        calibrate_x(ffine, theta_fine, theta_clean, min_cal_points = min_cal_points,
                    weight_sigma = xcal_weight_sigma, weight_theta0 = xcal_weight_theta0)

    fs_clean = np.polyval(poly, theta_clean)
    x = 1 - fs_clean / fnoise
    return theta_fine, theta, theta_clean, A_clean, theta_range, poly, x, cr_indices, (ix0, ix1)

def calibrate_x(ffine, theta_fine, theta_clean, poly_deg = 3,
                min_cal_points = 5, weight_sigma = None,
                weight_theta0 = 0.0):
    """
    Fit fine scan frequency to phase

    Parameters:
    ffine (np.array): fine scan frequency data in Hz
    theta_fine (np.array): fine scan theta data
    theta_clean (np.array): deglitched theta noise timestream data
    poly_deg (int): degree of the polynomial fit
    min_cal_points (int): minimum number of points for the polynomial fit
    weight_sigma (float): stdev of gaussian weight function for
        fitter in radians. Defaults to None for no weighting.
    weight_theta0 (float): center point in radians for gaussian
        weight function

    Returns:
    poly (np.array): polynomial fit parameters
    theta_range (list): [lower, upper] bounds on theta used for the fit
    """
    ix0 = np.argmin(abs(min(theta_clean) - theta_fine))
    ix1 = np.argmin(abs(max(theta_clean) - theta_fine))
    if ix1 < ix0:
        ix0, ix1 = ix1, ix0

    if theta_fine[ix0] > min(theta_clean):
        ix0 -= 1
    if theta_fine[ix1] < max(theta_clean):
        ix1 += 1
    ix1 += 1 # ix1 is not inclusive
    npoints_missing = min_cal_points - (ix1 - ix0)
    if npoints_missing > 0:
        half = int(np.ceil(npoints_missing / 2))
        ix0 -= half
        ix1 += half
    if len(theta_fine) < min_cal_points:
        raise Exception(f'theta_fine must be at least min_cal_points = {min_cal_points} points long')
    if ix0 < 0:
        ix1 += - ix0 # Increase ix1 by the amount below 0
        ix0 = 0
    if ix1 >= len(theta_fine):
        ix0 -= ix1 - len(theta_fine) # increase ix2 by the amount above len(theta_fine)
        ix1 = len(theta_fine)

    # Bias the fit toward an area we care more about
    weight = None
    if weight_sigma is not None:
        t = theta_fine[ix0:ix1] - weight_theta0
        weight = np.exp(-(t)**2 / (2 * weight_sigma**2))

    # Fit to data in range
    poly = np.polyfit(theta_fine[ix0:ix1], ffine[ix0:ix1],
                      deg = poly_deg, w = weight)
    theta_range = np.array([theta_fine[ix0], theta_fine[ix1 - 1]])

    return poly, theta_range, (ix0, ix1)

def deglitch_timestream(x, deglitch_nstd):
    """
    Replaces points above a certain threshold from data with the mean of the
    data

    Parameters:
    x (np.array): data from which glitches are removed
    deglitch_nstd (float or None): threshold for glitch removal. points above this
        threshold times the standard deviation of the data are removed
    Returns:
    x_deglitch (np.array): data with glitches removed
    """
    if deglitch_nstd is None:
        return x
    x_mean = np.mean(x)
    x_deglitch = x.copy()

    devs = np.abs(x - x_mean)
    x_std = np.std(devs)
    x_deglitch[devs > deglitch_nstd * x_std] = x_mean
    return x_deglitch

def calculate_theta_A(zfine, znoise, origin):
    """
    Convert an IQ loop and timestream to theta using the origin of the circle

    Parameters:
    zfine (np.array): fine scan complex S21 data
    znoise (np.array): complex S21 noise timestream
    origin (complex): center of the IQ loop

    Returns:
    theta_fine (np.array): values of theta corresponding to the fine scan data
    theta_noise (np.array): theta timestream corresponding the the noise data
    A_noise (np.array): amplitude timestream corresponding to the noise data
    """
    assert len(zfine), "No fine scan points provided to calculate_theta_A"
    zn_mean = np.mean(znoise)
    # Get x, y basic vectors, where x is the vector that passes through the
    # origin of the circle and the mean of the noise ball, and y is
    # perpendicular to x
    x_complex = (zn_mean - origin)
    x_complex /= np.abs(x_complex)
    y_complex = 1j * x_complex
    x_vec = np.array([np.real(x_complex), np.imag(x_complex)])
    y_vec = np.array([np.real(y_complex), np.imag(y_complex)])
    # Calculate theta of the fine scan
    zfine_vec = np.transpose(np.array([np.real(zfine - origin),
                                       np.imag(zfine - origin)]))
    zfine_z = np.dot(zfine_vec, x_vec) + 1j * np.dot(zfine_vec, y_vec)
    theta_fine = np.angle(zfine_z)
    extrapolate_mode = (min(theta_fine) > 0) or (max(theta_fine) < 0)
    # Unwrap fine scan theta
    #theta_fine = np.unwrap(2 * theta_fine) / 2
    theta_fine = np.unwrap(theta_fine)
    if not extrapolate_mode:
        while max(theta_fine) < 0:
            theta_fine += 2 * np.pi
        while min(theta_fine) > 0:
            theta_fine -= 2 * np.pi
    # Calculate theta of the noise
    noise_vec = np.transpose(np.array([np.real(znoise - origin),
                                       np.imag(znoise - origin)]))
    noise_z = np.dot(noise_vec, x_vec) + 1j * np.dot(noise_vec, y_vec)
    theta_noise = np.angle(noise_z)
    A_noise = np.abs(noise_z)
    # Make sure the range of theta matches the range of the fine scan
    theta_noise = np.where(theta_noise > 1, theta_noise - 2 * np.pi,
                           theta_noise)
    return theta_fine, theta_noise, A_noise
