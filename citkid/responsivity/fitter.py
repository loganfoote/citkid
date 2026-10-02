import numpy as np
from scipy.optimize import curve_fit
from .funcs import responsivity_int_for_fitter, responsivity_int_x0_for_fitter
from .guess import guess_p0_responsivity_int, get_bounds_responsivity_int
from .guess import guess_p0_responsivity_int_x0, get_bounds_responsivity_int_x0
from .guess import guess_p0_responsivity_int_x0_fixed_R0
from .guess import get_bounds_responsivity_int_x0_fixed_R0

def fit_responsivity_int(power, x, f1, x_err = None, guess = None,
                         guess_nfit = 3):
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

    Returns:
    p0 (np.array): Initial guess parameters [R0_guess, P0_guess, c_guess].
    popt (np.array): Fit parameters [R0, P0, c], NaN if the fit failed.
    perr (np.array): Fit parameter uncertainties [R0_err, P0_err, c_err],
        NaN if the fit failed.
    f0 (float): Frequency at P = 0 determined by the fit.
    f0err (float): Uncertainty in f0.
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
    p0, popt, perr = (np.asarray(p, dtype = float) for p in (p0, popt, perr))
    # Determine f0
    f0 = f1 * popt[2]
    f0err = f1 * perr[2]
    return p0, popt, perr, f0, f0err

def fit_responsivity_int_x0(
    power, x, f1, x_err = None, guess = None, guess_nfit = 3,
    ):
    """
    Fit x versus power data to the integrated responsivity equation with
    c = 1 and a free offset x0 (responsivity_int_x0).
    The fitter works best if there are at least three data points at P >> P_0
    and the lowest-power point is at P << P_0. Otherwise, an alternative
    initial guess may be required.

    Parameters:
    power (array-like): Array of blackbody powers in W.
    x (array-like): Fractional frequency shifts in Hz / Hz.
    f1 (float): Frequency at P = 0 used to calculate x.
    x_err (array-like or None): Error on x used in the fitting. If None,
        points are weighted equally.
    guess (list or None): If not None, overwrite the initial guess
        [R0_guess, P0_guess, x0_guess] (unscaled units: 1 / W, W, Hz / Hz).
    guess_nfit (int): Number of high-power (P >> P_0) points for the guess.

    Returns:
    p0 (np.array): Initial guess parameters [R0_guess, P0_guess, x0_guess].
    popt (np.array): Fit parameters [R0, P0, x0], NaN if the fit failed.
    perr (np.array): Fit parameter uncertainties [R0_err, P0_err, x0_err],
        NaN if the fit failed.
    f0 (float): Frequency at P = 0 determined by the fit, f1 * (1 + x0).
    f0err (float): Uncertainty in f0.
    """
    scale = np.array([1e-9, 1e16, 1e6])
    power, x = np.array(power, dtype = float), np.array(x, dtype = float)
    ix = np.argsort(power)
    power, x = power[ix], x[ix]
    if x_err is not None:
        x_err = np.array(x_err, dtype = float)[ix]
    # Initial guess, scaled for responsivity_int_x0_for_fitter
    if guess is not None:
        p0 = list(np.asarray(guess, dtype = float) * scale)
        bounds = get_bounds_responsivity_int_x0(p0, np.ptp(x) * 1e6)
    else:
        p0, bounds = guess_p0_responsivity_int_x0(
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
        # To fit with sigma, the initial guess must be really good, so
        # update the initial guess with curve_fit without sigma.
        try:
            p00, _ = curve_fit(
                responsivity_int_x0_for_fitter,
                np.log(power),
                x * 1e6,
                p0 = p0,
                bounds = bounds,
            )
        except Exception:
            p00 = p0
    try:
        popt, pcov = curve_fit(
            responsivity_int_x0_for_fitter,
            np.log(power),
            x * 1e6,
            sigma = sigma,
            p0 = p00,
            bounds = bounds,
            absolute_sigma = True,
        )
        popt = popt / scale
        perr = np.sqrt(np.diag(pcov)) / scale
    except Exception:
        popt = np.full(3, np.nan)
        perr = np.full(3, np.nan)
    p0 = np.asarray(p0) / scale
    # Determine f0
    f0 = f1 * (1 + popt[2])
    f0err = f1 * perr[2]
    return p0, popt, perr, f0, f0err

def fit_responsivity_int_x0_fixed_R0(power, x, f1, R0, x_err = None,
                                     guess = None, guess_nfit = 3):
    """
    Fit x versus power data to the integrated responsivity equation with
    c = 1 (responsivity_int_x0), holding R0 fixed and fitting only P0 and x0.
    The fitter works best if there are at least three data points at P >> P_0
    and the lowest-power point is at P << P_0. Otherwise, an alternative
    initial guess may be required.

    Parameters:
    power (array-like): Array of blackbody powers in W.
    x (array-like): Fractional frequency shifts in Hz / Hz.
    f1 (float): Frequency at P = 0 used to calculate x.
    R0 (float): Fixed responsivity at P = 0 (1 / W). In the sign convention of
        responsivity_int_x0, R0 < 0 for x decreasing with power.
    x_err (array-like or None): Error on x used in the fitting. If None,
        points are weighted equally.
    guess (list or None): If not None, overwrite the initial guess
        [P0_guess, x0_guess] (unscaled units: W, Hz / Hz).
    guess_nfit (int): Number of high-power (P >> P_0) points for the guess.

    Returns:
    p0 (np.array): Initial guess parameters [R0, P0_guess, x0_guess].
    popt (np.array): Fit parameters [R0, P0, x0]. R0 is the fixed input; P0
        and x0 are NaN if the fit failed.
    perr (np.array): Fit parameter uncertainties [0, P0_err, x0_err].
    f0 (float): Frequency at P = 0 determined by the fit, f1 * (1 + x0).
    f0err (float): Uncertainty in f0.
    """
    scale = np.array([1e16, 1e6])
    R0_scaled = R0 * 1e-9
    def model(log_power, P0, x0):
        return responsivity_int_x0_for_fitter(log_power, R0_scaled, P0, x0)
    power, x = np.array(power, dtype = float), np.array(x, dtype = float)
    ix = np.argsort(power)
    power, x = power[ix], x[ix]
    if x_err is not None:
        x_err = np.array(x_err, dtype = float)[ix]
    # Initial guess, scaled for responsivity_int_x0_for_fitter
    if guess is not None:
        p0 = list(np.asarray(guess, dtype = float) * scale)
        bounds = get_bounds_responsivity_int_x0_fixed_R0(p0, np.ptp(x) * 1e6)
    else:
        p0, bounds = guess_p0_responsivity_int_x0_fixed_R0(
            power,
            x,
            R0,
            guess_nfit = guess_nfit,
        )
    # Fit
    if x_err is None:
        sigma = None
        p00 = p0
    else:
        sigma = x_err * 1e6
        # To fit with sigma, the initial guess must be really good, so
        # update the initial guess with curve_fit without sigma.
        try:
            p00, _ = curve_fit(
                model,
                np.log(power),
                x * 1e6,
                p0 = p0,
                bounds = bounds,
            )
        except Exception:
            p00 = p0
    try:
        popt, pcov = curve_fit(
            model,
            np.log(power),
            x * 1e6,
            sigma = sigma,
            p0 = p00,
            bounds = bounds,
            absolute_sigma = True,
        )
        popt = popt / scale
        perr = np.sqrt(np.diag(pcov)) / scale
    except Exception:
        popt = np.full(2, np.nan)
        perr = np.full(2, np.nan)
    p0 = np.concatenate([[R0], np.asarray(p0) / scale])
    popt = np.concatenate([[R0], popt])
    perr = np.concatenate([[0.], perr])
    # Determine f0
    f0 = f1 * (1 + popt[2])
    f0err = f1 * perr[2]
    return p0, popt, perr, f0, f0err
