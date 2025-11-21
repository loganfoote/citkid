import numpy as np
from .apply_cal import calculate_theta
from ..res.gain import fit_gain, remove_gain
from ..res.fitter import fit_iq_circle
from .plot import plot_cal

def make_x_cal(ffine, zfine, fgain, zgain, fs, Qs, plotq = False):
    """
    Given fine scan data, produce the calibration data to convert complex S21
    data to theta

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data
    fgain (array-like): gain scan frequency data in Hz
    zgain (array-like): gain scan complex S21 data
    fs (array-like): resonance frequencies to remove from gain scans when
        fitting gain data
    Qs (array-like): scaled resonance quality factors to remove from gain scans
        when fitting gain data. frequency ranges of width fs / Qs centered on fs
        are removed from the gain scan data
    plotq (bool): if True, creates plots of the calibration

    Returns:
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the IQ circle
    v (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    theta_fine (np.array): fine scan theta data corresponding to the values in
        zfine
    p_x (array-like): polynomial fit parameters to x versus theta
    fig_gain (pyplot.figure or None): gain scan calibration plot if plotq,
        else None
    fig_fine (pyplot.figure or None): fine scan calibration plot if plotq,
        else None
    fig_x (pyplot.figure or None): theta to x calibration plot if plotq,
        else None
    """
    p_amp, p_phase, origin, radius, v, theta_fine, (fig_gain, fig_fine) =\
                make_theta_cal(ffine, zfine, fgain, zgain, fs, Qs, plotq = plotq)
    theta_range = [-1, 1]
    ix = (theta_fine <= theta_range[1]) & (theta_fine >= theta_range[0])
    p_x = np.polyfit(theta_fine[ix], ffine[ix], deg = 3)
    if plotq:
        fig_cal = plot_cal(ffine, zfine, origin, radius, v, theta_range,
                           theta_fine, p_amp, p_phase, p_x)
    else:
        fig_cal = None
    return p_amp, p_phase, origin, v, theta_fine, p_x, \
           (fig_gain, fig_fine, fig_cal)

def make_theta_cal(ffine, zfine, fgain, zgain, fs, Qs, plotq = False):
    """
    Given fine scan data, produce the calibration data to convert complex S21
    data to theta

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data
    fgain (array-like): gain scan frequency data in Hz
    zgain (array-like): gain scan complex S21 data
    fs (array-like): resonance frequencies to remove from gain scans when
        fitting gain data
    Qs (array-like): scaled resonance quality factors to remove from gain scans
        when fitting gain data. frequency ranges of width fs / Qs centered on fs
        are removed from the gain scan data
    plotq (bool): if True, creates plots of the calibration

    Returns:
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the IQ circle
    radius (float): radius of the IQ circle
    v (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    theta_fine (np.array): fine scan theta data corresponding to the values in
        zfine
    fig_gain, fig_fine (pyplot.figure or None): gain and fine scan calibration
        plots if plotq, else (None, None)
    """
    # Fit gain
    fr_spans = [[fi, fi / Qi] for fi, Qi in zip(fs, Qs)]
    p_amp, p_phase, (fig_gain, ax_gain) = fit_gain(fgain, zgain, fr_spans,
                                                  plotq = plotq)
    zfine_rmvd = remove_gain(ffine, zfine, p_amp, p_phase)
    # Theta vector
    origin, radius, v, fig_fine = get_theta_vec(zfine_rmvd, plotq = plotq)
    # fine-scan theta
    theta_fine = calculate_theta(ffine, zfine, p_amp, p_phase, origin, v)
    return p_amp, p_phase, origin, radius, v, theta_fine, (fig_gain, fig_fine)

def get_theta_vec(zfine_rmvd, plotq = False):
    """
    Given an IQ loop with gain removed, return the origin of the resonance
    circle and the unit vector that points from the origin to theta = 0

    Parameters:
    zfine_rmvd (np.array): fine scan complex S21 data with gain removed

    Returns:
    origin (complex): origin of the IQ circle
    v (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    """
    popt_circle, fig = fit_iq_circle(zfine_rmvd, plotq = plotq)
    origin = popt_circle[0] + 1j * popt_circle[1]
    radius = popt_circle[2]
    z_offres = np.mean(np.concatenate([zfine_rmvd[:2], zfine_rmvd[-2:]]))
    ix = np.argmax(np.abs(zfine_rmvd - z_offres))
    z_onres = zfine_rmvd[ix]
    # Old method: set v to be the vector from the origin to the on-res point
    # # Get v, the direction between the origin of the circle and the on-res point
    # v = (z_onres - origin)
    # v /= np.abs(v)
    # Current method: set v such that the min theta point is at -pi
    # This method avoids issues where the off-res data jumps at -pi or pi 
    zmod = (zfine_rmvd - origin) 
    zmod_rot = zmod[:len(zmod) // 10] * np.exp(-1j * zmod[0])
    # Next, find the index of the smallest angle, and rotate it to -pi 
    ix = np.argmax(np.angle(zmod_rot))
    v = (zmod[ix]) * np.exp(-1j * np.pi)
    v /= np.abs(v)
    if plotq:
        ax = fig.axes[0]
        ax.plot([], [], 'or', label = 'data')
        ax.plot([], [], '-k', label = 'circle fit')
        ax.plot([np.real(origin), np.real(z_onres)],
                [np.imag(origin), np.imag(z_onres)],
                '--k', label = r'$\theta = 0$')
        ax.plot(np.real(origin), np.imag(origin), 'xk', label = 'origin')
        ax.legend(framealpha = 1, bbox_to_anchor = (0.5, 0.7), loc = 'center',
                  ncols = 2, fontsize = 8)
    return origin, radius, v, fig
