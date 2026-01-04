import numpy as np
from scipy import optimize
import warnings
from .funcs import nonlinear_iq_for_fitter, nonlinear_iq, circle_objective
from .util import bounds_check, cal_nrmse
from .gain import fit_and_remove_gain_phase
from .plot import plot_nonlinear_iq, plot_circle
from ..util import  combine_figures_vertically
from citkid.res import guess
from .data_io import make_fit_row

def fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine, frs, Qrs,
                               downward = True, plotq = False,
                               return_dataframe = False, floats_only=False,
                               **kwargs):
    """
    Fits IQ data with gain amplitudes and phase correction from a gain sweep.
    Cuts resonance frequencies from the gain sweep in spans of fr / Qr around fr,
    where fr is an item in frs and Qr is a corresponding quality factor in Qrs.

    The optimal fine sweep width is 6 * fr / Qr.
    The optimal gain sweep width is 100 * fr / Qr.

    Parameters:
    fgain (np.array): gain sweep frequency data.
    zgain (np.array): gain sweep complex S21 data.
    ffine (np.array): fine sweep frequency data.
    zfine (np.array): fine sweep complex S21 data.
    frs (list of float): resonance frequencies to cut from the gain sweep.
    Qrs (list of float): spans of frs / Qrs are cut from the gain sweep.
    downward (bool): If True, fits the equation for a downward sweep. If
        False, fits for an upward sweep.
    plotq (bool): If True, plots the fits.
    return_dataframe (bool): if True, returns the output of
        .data_io.make_fit_row instead of the separated data.
    floats_only (bool): Set to True to only keep columns in the 
        dataframe whose values can be represented as floats, 
        i.e. don't store columns for sweep_direction or plotpath.
    **kwargs: other arguments for fit_nonlinear_iq.

    Returns:
    if return_dataframe:
        row (pd.Series): fit data as a pandas series.
    else:
        p_amp (np.array): 2nd-order polynomial fit parameters to dB.
        p_phase (np.array): 1st-order polynomial fit parameters to phase.
        p0 (np.array): fit parameter guess.
        popt (np.array): fit parameters. See p0 parameter.
        perr (np.array): standard errors on fit parameters.
        nrmse (float): normalized root mean square error of the fit.
        fig (pyplot.figure or None): figure with gain fit and nonlinear IQ
            fit if plotq, or None.
    """
    # Remove gain
    p_amp, p_phase, zfine_rmvd, (fig_gain, axs_gain) = \
        fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, frs, Qrs,
                                  plotq = plotq)
    # Rotate data for better plots
    zoff = np.mean(np.roll(zfine_rmvd, 6)[:6])
    zfine_rmvd *= np.exp(-1j * np.angle(zoff))
    p_phase[1] += np.angle(zoff)
    # Fit IQ
    p0, popt, perr, nrmse, (fig_fit, axs_fit) = fit_nonlinear_iq(ffine,
                                            zfine_rmvd, plotq = plotq,
                                            downward = downward, **kwargs)
    if plotq:
        fig = combine_figures_vertically(fig_gain, fig_fit)
    else:
        fig = None
    if return_dataframe:
        row = make_fit_row(p_amp, p_phase, p0, popt, perr, nrmse,
                           downward = downward, plot_path = '', prefix = 'iq',
                           floats_only = floats_only)
        return row, fig
    return p_amp, p_phase, p0, popt, perr, nrmse, fig

def fit_nonlinear_iq(f, z, bounds = None, p0 = None, fr_guess = None,
                     fit_tau = True, tau_guess = None, downward = True,
                     plotq = False):
    """
    Fit a nonlinear IQ with from an S21 sweep. Uses scipy.optimize.curve_fit.
    It is assumed that the system gain and phase are removed from the data
    before fitting. i0, q0, and tau are fitted only for fine-tuning.

    The optimal span of the data is 6 * fr / Qr
    The optimal length of the data is 500, but down to 200 still works ok

    Parameters:
    f (numpy.array): frequencies Hz
    z (numpy.array): complex s21.
    bounds (tuple or None): 2d tuple of low values bounds[0] the high values
        bounds[1] to bound the fitting problem. If None, sets default bounds
    p0 (list or None): initial guesses for all parameters
        fr_guess  = p0[0]
        Qr_guess  = p0[1]
        amp_guess = p0[2]
        phi_guess = p0[3]
        a_guess   = p0[4]
        i0_guess  = p0[5]
        q0_guess  = p0[6]
        tau_guess = p0[7]
        If None, calls citkid.fit.guess.guess_nonlinear_iq to find p0.
    fr_guess (float or None): if float, overrides p0[0].
    fit_tau (bool): if False, tau is enforced from p0[7] to speed up fitting.
        If True, tau is fit.
    tau_guess (float or None): If float, overides p0[7]
    downward (bool): If True, fits the equation for a downward sweep. If
        False, fits for an upward sweep.
    plotq (bool): if True, plots the data with the fit

    Returns:
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter
    perr (np.array): standard errors on fit parameters
    nrmse (float): normalized root mean square error of the fit.
    fig, ax (pyplot figure and axes, or None): plot of data with fit if plotq,
        or None, None
    """
    # Sort f and z
    f, z = np.array(f), np.array(z)
    ix = np.argsort(f)
    f, z = f[ix], z[ix]
    if p0 is None: # default initial guess
        p0 = guess.guess_p0_nonlinear_iq(f, z)
    if bounds is None:
        # default bounds. Phi range is increased to avoid jumping at bounds
        #                 fr,  Qr, amp,              phi,    a,   i0,   q0,     tau
        bounds = ([np.min(f), 1e3, .01,        -np.pi / 2, 0, -1e2, -1e2, -1.0e-6],
                  [np.max(f), 1e7,   1 - 1e-6,  np.pi / 2, 1,  1e2,  1e2,  1.0e-6])
        for index in [1, 5, 6]:
            # These will be flipped in bounds_check if needed
            if p0[index] != 0:
                bounds[0][index] = p0[index] / 10
                bounds[1][index] = p0[index] * 10
    if fr_guess is not None:
        p0[0] = fr_guess
    if tau_guess is not None:
        p0[7] = tau_guess
    # Stack z data
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # Check bounds
    bounds = bounds_check(p0, bounds)
    # fit
    nrmse_acceptable = False
    niter = 0
    while not nrmse_acceptable:
        popt, perr, nrmse = fit_util(np.array(p0), np.array(bounds), fit_tau, f,
                                   z_stacked, z, downward)
        if nrmse < 1e-2 or niter > 1:
            nrmse_acceptable = True
        elif nrmse < 1e-1:
            # If 1e-2 < nrmse < 1e-1, the fit is close but not perfect
            p0 = np.array(popt)
            niter += 1
        else:
            # Usually, amp will be too high if the fit residuals are this high
            p0[2] /= 10
            bounds[0][2] /= 10
            bounds[1][2] /= 10
            niter += 1
    # plot
    if plotq:
        figax = plot_nonlinear_iq(f, z, popt, p0, downward = downward)
    else:
        figax = None, None
    p0 = np.array(p0)
    return p0, popt, perr, nrmse, figax

def fit_iq_circle(z, x0 = None, plotq = False):
    """
    Fits an IQ loop to a circle. The function describing the circle is

       [Re(S21)-A]^2 + [Im(S21)-B]^2 = R^2

       where the origin is (A, B) and the radius is R.

    Parameters:
    z (np.array): complex S21 data
    x0: Initial guess for the fit parameters (A, B, R).
        If x0 == None, this function will generate its own guess.
    plotq (bool): if True, plots the fit and data

    Returns:
    popt (list): fit parameters (A, B, R).
    fig, ax (pyplot figure and axis): fit figure and axis, or None if not plotq
    """
    warnings.warn("fit_iq_circle is deprecated. Use citkid.xcal.circle.fit_iq_circle instead.", 
                  DeprecationWarning)
    z = np.asarray(z, dtype = np.complex128)
    if not np.all(np.isfinite(z)):
        raise ValueError("Input data contains non-finite values.")
    i, q = z.real, z.imag

    if x0 is None:
        x0 = [(max(i) + min(i))/2, (max(q) + min(q))/2]
        x0.append((max(i) - min(i) + max(q) - min(q)) / 4)
    popt = optimize.fmin(circle_objective, x0, (i, q), disp = 0)

    if plotq:
        fig, _ = plot_circle(z, *popt)
    else:
        fig = None
    return popt, fig

################################################################################
######################### Utility functions ####################################
################################################################################
def fit_util(p0, bounds, fit_tau, f, z_stacked, z, downward = True):
    """
    Utility function for fitting IQ loops. Given data and initial fit parameters,
    fits the IQ loop and returns the fit parameters

    Parameters:
    p0 (list): fit guess parameters
    bounds (list): fit bounds
    fit_tau (bool): if False, uses given tau instead of fitting
    f (np.array): frequency data in Hz
    z_stacked (np.array): stacked complex S21 data
    z (np.array) complex S21 data
    downward (bool): If True, fits the equation for a downward sweep. If
        False, fits for an upward sweep.

    Returns:
    popt (np.array): fit parameters
    perr (np.array): fit parameter uncertainties
    nrmse (float): normalized root mean square error of the fit.
    """
    #             fr,   Qr, amp, phi, a, i0, q0, tau
    scaler = [100e-6, 1e-4,   1,   1, 1,  1,  1, 1e6]
    p0 = [p0i * s for p0i, s in zip(p0, scaler)]
    bounds[0] = [bi * s for bi, s in zip(bounds[0], scaler)]
    bounds[1] = [bi * s for bi, s in zip(bounds[1], scaler)]
    f = np.asarray(f, dtype = np.float64) 
    z_stacked = np.asarray(z_stacked, dtype = np.float64)
    if not fit_tau:
        # Fit with tau enforced from p0[7]
        tau = p0[7]
        bounds = np.array([bounds[0][:7], bounds[1][:7]])
        p0 = p0[:7]
        def fit_func(x_lamb, a, b, c, d, e, f, g):
            return nonlinear_iq_for_fitter(x_lamb, a, b, c, d, e, f, g, tau,
                                           downward)
        p0 = np.asarray(p0, dtype = np.float64) 
        bounds = np.asarray(bounds, dtype = np.float64)
        popt, pcov = optimize.curve_fit(fit_func, f, z_stacked, p0,
                                        bounds = bounds)
        popt = np.insert(popt, 7, tau)
        perr = np.sqrt(np.diag(pcov))
        perr = np.insert(perr, 7, 0)
    else:
        # Fit without enforcing tau
        def fit_func(x_lamb, a, b, c, d, e, f, g, h):
            return nonlinear_iq_for_fitter(x_lamb, a, b, c, d, e, f, g, h,
                                           downward)
        p0 = np.asarray(p0, dtype = np.float64) 
        bounds = np.asarray(bounds, dtype = np.float64)
        popt, pcov = optimize.curve_fit(fit_func, f, z_stacked, p0, bounds = bounds)

        perr = np.sqrt(np.diag(pcov))
    popt = [pi / s for pi, s in zip(popt, scaler)]
    perr = [pi / s for pi, s in zip(perr, scaler)]
    z_fit = nonlinear_iq(f, *popt, downward)
    nrmse = cal_nrmse(z, z_fit)
    return popt, perr, nrmse