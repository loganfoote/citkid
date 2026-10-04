import numpy as np
from scipy import optimize
from .funcs import nonlinear_iq_for_fitter, nonlinear_iq
from .util import bounds_check, calc_nrmse
from .plot import plot_nonlinear_iq
from citkid.res import guess

def fit_nonlinear_iq(
    f, z, bounds = None, p0 = None, fr_guess = None, fit_tau = True,
    tau_guess = None, downward = True, plotq = False
):
    """
    Fit a nonlinear IQ with from an S21 sweep. Uses scipy.optimize.curve_fit.
    It is assumed that the system gain and phase are removed from the data
    before fitting. i0, q0, and tau are fitted only for fine-tuning.

    The optimal span of the data is 6 * fr / Qr.
    The optimal length of the data is 500, but down to 200 still works ok.

    Parameters:
    f (numpy.array): frequencies Hz.
    z (numpy.array): complex s21.
    bounds (tuple or None): 2d tuple of low values bounds[0] the high values
        bounds[1] to bound the fitting problem. If None, sets default bounds.
    p0 (list or None): initial guesses for all parameters
        fr_guess  = p0[0]
        Qr_guess  = p0[1]
        amp_guess = p0[2]
        phi_guess = p0[3]
        a_guess   = p0[4]
        i0_guess  = p0[5]
        q0_guess  = p0[6]
        tau_guess = p0[7].
        If None, calls citkid.fit.guess.guess_nonlinear_iq to find p0.
    fr_guess (float or None): if float, overrides p0[0].
    fit_tau (bool): if False, tau is enforced from p0[7] to speed up fitting.
        If True, tau is fit.
    tau_guess (float or None): If float, overides p0[7].
    downward (bool): If True, fits the equation for a downward sweep. If
        False, fits for an upward sweep.
    plotq (bool): if True, plots the data with the fit.

    Returns:
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. 
    perr (np.array): standard errors on fit parameters.
    nrmse (float): normalized root mean square error of the fit.
    fig, ax (pyplot figure and axes, or None): plot of data with fit if plotq,
        or None, None.
    """
    # Sort f and z
    f, z = np.array(f), np.array(z)
    ix = np.argsort(f)
    f, z = f[ix], z[ix]
    if p0 is None: # default initial guess
        p0 = guess.guess_p0_nonlinear_iq(f, z)
    if bounds is None:
        # default bounds. Phi range is increased to avoid jumping at bounds
        #                 fr,  Qr, amp,       phi,    a,   i0,   q0,    tau
        bounds = (
            [np.min(f), 1e3, .01, -np.pi / 2, 0, -1e2, -1e2, -1.0e-6],
            [np.max(f), 1e7, 1 - 1e-6, np.pi / 2, 1, 1e2, 1e2, 1.0e-6],
        )
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
        popt, perr, nrmse = fit_util(
            np.array(p0),
            np.array(bounds),
            fit_tau,
            f,
            z_stacked,
            z,
            downward,
        )
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

def fit_nonlinear_iq_pl(f, z, mask):
    """
    Wrap fit_nonlinear_iq for use as a plStep. See fit_nonlinear_iq 
    for details.

    Parameters:
    f (numpy.array): frequencies Hz.    
    z (numpy.array): complex s21.
    mask (numpy.array of bool or None): only fits data where mask is True.
    """
    p0, popt, _, nrmse, _ = fit_nonlinear_iq(
        f[mask], z[mask], bounds = None, p0 = None, fr_guess = None, fit_tau = True, 
        tau_guess = None, downward = True, plotq = False
    ) 
    return p0, popt, nrmse


################################################################################
######################### Utility functions ####################################
################################################################################
def fit_util(p0, bounds, fit_tau, f, z_stacked, z, downward = True):
    """
    Fit an IQ loop given data and initial fit parameters, and return the fit
    parameters.

    Parameters:
    p0 (list): fit guess parameters.
    bounds (list): fit bounds.
    fit_tau (bool): if False, uses given tau instead of fitting.
    f (np.array): frequency data in Hz.
    z_stacked (np.array): stacked complex S21 data.
    z (np.array): complex S21 data.
    downward (bool): If True, fits the equation for a downward sweep. If
        False, fits for an upward sweep.

    Returns:
    popt (np.array): fit parameters.
    perr (np.array): fit parameter uncertainties.
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
        popt, pcov = optimize.curve_fit(
            fit_func,
            f,
            z_stacked,
            p0,
            bounds = bounds,
        )

        perr = np.sqrt(np.diag(pcov))
    popt = [pi / s for pi, s in zip(popt, scaler)]
    perr = [pi / s for pi, s in zip(perr, scaler)]
    z_fit = nonlinear_iq(f, *popt, downward)
    nrmse = calc_nrmse(z, z_fit)
    return popt, perr, nrmse
