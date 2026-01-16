import numpy as np
from scipy.optimize import curve_fit
from .funcs import responsivity_int_for_fitter
from .guess import guess_p0_responsivity_int, get_bounds_responsivity_int
from .plot import plot_responsivity_int
from .data_io import make_fit_row

def fit_responsivity_int(power, x, f1, x_err = None, guess = None,
                         guess_nfit = 3, return_dataframe = False,
                         plotq = False):
    """
    Fit x versus power data to the integrated responsivity equation.
    The fitter works best if there are at least three data points at P >> P_0.
    Otherwise, an alternative initial guess may be required.

    Parameters:
    power (array-like): Array of blackbody powers in W.
    x (array-like): Fractional frequency shifts in Hz / Hz. This must be
        scaled close to x(P = 0) = 0 for the initial guess to work well.
    x_err (array-like or None): Error on x used in the fitting. If None,
        points are weighted equally.
    f1 (float): Frequency at P = 0 used to calculate x.
    guess (list or None): If not None, overwrite the initial guess
        [R0_guess, P0_guess, c_guess].
    guess_nfit (int): Number of high-power (P >> P_0) points for the guess.
    return_dataframe (bool): If True, returns a pandas series of output data.
    plotq (bool): If True, plots the fit and initial guess.

    Returns:
    p0 (list): Initial guess parameters [R0_guess, P0_guess, c_guess].
    popt (list): Fit parameters [R0, P0, c].
    perr (list): Fit parameter uncertainties [R0_err, P0_err, c_err].
    f0 (float): Frequency at P = 0 determined by the fit.
    f0err (float): Uncertainty in f0.
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq.
    """
    power, x = np.array(power), np.array(x)
    ix = np.argsort(power)
    power, x = power[ix], x[ix]
    if x_err is not None:
        x_err = np.array(x_err)[ix]
    # Initial guess
    if guess is not None:
        p0 = guess
        bounds = get_bounds_responsivity_int(p0)
    else:
        p0, bounds = guess_p0_responsivity_int(
            power,
            x,
            guess_nfit = guess_nfit,
        )
    # Fit
    if x_err is None:
        sigma = None
        p00 = p0
    else:
        sigma = x_err * 1e6
        try:
            p00, _ = curve_fit(
                responsivity_int_for_fitter,
                np.log(power),
                x * 1e6,
                sigma = None,
                p0 = p0,
                bounds = bounds,
            )
            # To fit with sigma, the initial guess must be really good, so
            # update the initial guess with curve_fit without sigma.
        except:
            p00 = p0
    try:
        popt, pcov = curve_fit(
            responsivity_int_for_fitter,
            np.log(power),
            x * 1e6,
            sigma = sigma,
            p0 = p00,
            bounds = bounds,
            absolute_sigma = True,
        )
        perr = np.sqrt(np.diag(pcov))
        p0[0], popt[0], perr[0] = (
            p0[0] * 1e9,
            popt[0] * 1e9,
            perr[0] * 1e9,
        )
        p0[1], popt[1], perr[1] = (
            p0[1] * 1e-16,
            popt[1] * 1e-16,
            perr[1] * 1e-16,
        )
        # p0[2], popt[2], perr[2] = (
        #     p0[2] * 1e-8,
        #     popt[2] * 1e-8,
        #     perr[2] * 1e-8,
        # )
    except Exception as e:
        p0[0] = p0[0] * 1e9
        p0[1] = p0[1] * 1e-16
        popt = [np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan]
    # Plot
    if plotq:
        fig, ax = plot_responsivity_int(power, x, x_err, popt, p0)
    else:
        fig, ax = None, None
    # Determine f0
    f0 = f1 * popt[2]
    f0err = f1 * perr[2]
    if return_dataframe:
        row = make_fit_row(p0, popt, perr, f1, f0, f0err)
        return row, (fig, ax)
    return p0, popt, perr, f0, f0err, (fig, ax)
