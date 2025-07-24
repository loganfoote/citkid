import numpy as np
from scipy.optimize import curve_fit
from .guess import *
from .funcs import *
from .plot import *
from .data_io import *

################################################################################
############### Resonance shift from thermal QP density and TLS ################
################################################################################
def fit_f_vs_T(T, f, gamma = 1, Tc_guess = 1.2, f_err = None, guess = None,
               bounds = None, return_dataframe = False, plotq = False,
               catch_exceptions = False):
    """
    Fits resonant frequency versus temperature data to f_vs_T.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    Tc_guess (float): critical temperature guess in K
    f_err (array-like or None): If not None, f_err is the error on f used in
        the fitting. If None, points are weighted equally
    guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess.
    bounds (list or None): custom bounds to overwrite the default option
    return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess
    catch_exceptions (bool): If True, catches exceptions in the fitter and
       returns nan values instead. Else, raises exceptions in the fitter

    Returns:
    if return_dataframe is False:
        p0 (list): initial guess parameters [f0_guess, alpha_guess,
                                             Tc_guess, Fdelta0_guess]
        popt (list): fit parameters [f0, alpha, Tc, Fdelta0]
        perr (list): fit parameter uncertainties [f0_err, alpha_err,
                                                  Tc_err, Fdelta0_err]
    else:
        row (pd.Series): contains p0, popt, and perr
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    T, f = np.array(T), np.array(f)
    ix = np.argsort(T)
    T, f = T[ix], f[ix]
    if f_err is not None:
       f_err = np.array(f_err)[ix]

    fit_func = lambda a, b, c, d, e: f_vs_T(a, b, c, d, e, gamma = gamma)
    # Initial guess
    if guess is not None:
       p0 = guess
       bounds_qp, bounds_tls = get_bounds_f_vs_T_qp(p0), get_bounds_f_vs_T_tls(p0)
       bounds0 = get_bounds_f_vs_T(bounds_qp, bounds_tls)
    else:
       p0, bounds0 = guess_p0_f_vs_T(T, f, Tc_guess, gamma)
    if bounds is None:
       bounds = bounds0
    # Fit
    if f_err is None:
       sigma = None
       p00 = p0
    else:
       sigma = f_err
       try:
          p00, _ = curve_fit(fit_func, T, f, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except Exception as e:
          if not catch_exceptions:
              raise e
          p00 = p0
    try:
       popt, pcov = curve_fit(fit_func, T, f, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
    except Exception as e:
       if not catch_exceptions:
           raise e
       popt = [np.nan, np.nan, np.nan, np.nan]
       perr = [np.nan, np.nan, np.nan, np.nan]
    # Plot
    if plotq:
       fig, ax = plot_f_vs_T(T, f, f_err, popt, p0, gamma)
    else:
       fig, ax = None, None

    if return_dataframe:
       row = make_fit_row_f_vs_T(p0, popt, perr, gamma)
       return row, (fig, ax)
    return p0, popt, perr, (fig, ax)

def fit_Q_vs_T(T, Q, f0_guess, gamma = 1, Tc_guess = 1.2, Q_err = None,
               guess = None, bounds = None, return_dataframe = False,
               plotq = False, catch_exceptions = False):
    """
    Fits quality factor versus temperature data to Q_vs_T.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    f0_guess (float): guess for the resonant frequency in Hz at T = 0 K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    Tc_guess (float): critical temperature guess in K
    Q_err (array-like or None): If not None, Q_err is the error on Q used in
        the fitting. If None, points are weighted equally
    guess (list or None): If not None, overwrites the initial guess. Also
        overwrites f0_guess.
    bounds (list or None): custom bounds to overwrite the default option
    return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess
    catch_exceptions (bool): If True, catches exceptions in the fitter and
       returns nan values instead. Else, raises exceptions in the fitter

    Returns:
    if return_dataframe is False:
        p0 (list): initial guess parameters [f0_guess, alpha_guess,
                                             Tc_guess, Fdelta0_guess,
                                             delta_z_guess]
        popt (list): fit parameters [f0, alpha, Tc, Fdelta0, delta_z]
        perr (list): fit parameter uncertainties [f0_err, alpha_err,
                                                  Tc_err, Fdelta0_err,
                                                  delta_z_err]
    else:
        row (pd.Series): contains p0, popt, and perr
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    T, Q = np.array(T), np.array(Q)
    ix = np.argsort(T)
    T, Q = T[ix], Q[ix]
    if Q_err is not None:
      Q_err = np.array(Q_err)[ix]

    fit_func = lambda a, b, c, d, e, f: Q_vs_T(a, b, c, d, e, f,
                                                  gamma = gamma)
    # Initial guess
    if guess is not None:
        p0 = guess
        bounds_qp, bounds_tls = get_bounds_Q_vs_T_qp(p0), get_bounds_Q_vs_T_tls(p0)
        bounds0 = get_bounds_Q_vs_T(bounds_qp, bounds_tls)
    else:
          p0, bounds0 = guess_p0_Q_vs_T(T, Q, f0_guess, Tc_guess, gamma)
    if bounds is None:
        bounds = bounds0
    # Fit
    if Q_err is None:
        sigma = None
        p00 = p0
    else:
      sigma = Q_err
      try:
         p00, _ = curve_fit(fit_func, T, Q, sigma = None,
                             p0 = p0, bounds = bounds)
         # To fit with sigma, the initial guess must be really good, so
         # update the initial guess with curve_fit without sigma
      except Exception as e:
         if not catch_exceptions:
             raise e
         p00 = p0
    try:
      popt, pcov = curve_fit(fit_func, T, Q, sigma = sigma,
                             p0 = p00, bounds = bounds, absolute_sigma = True)
      perr = np.sqrt(np.diag(pcov))
    except Exception as e:
      if not catch_exceptions:
          raise e
      popt = [np.nan, np.nan, np.nan, np.nan, np.nan]
      perr = [np.nan, np.nan, np.nan, np.nan, np.nan]
    # Plot
    if plotq:
      fig, ax = plot_Q_vs_T(T, Q, Q_err, popt, p0, gamma)
    else:
      fig, ax = None, None

    if return_dataframe:
      row = make_fit_row_Q_vs_T(p0, popt, perr, gamma)
      return row, (fig, ax)
    return p0, popt, perr, (fig, ax)

################################################################################
################### Resonance shift from thermal QP density ####################
################################################################################
def fit_f_vs_T_qp(T, f, gamma = 1, Tc_guess = 1.2, f_err = None, guess = None,
                  bounds = None, return_dataframe = False, plotq = False,
                  catch_exceptions = False):
    """
    Fits resonant frequency versus temperature data to f_vs_T_qp.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    Tc_guess (float): critical temperature guess in K
    f_err (array-like or None): If not None, f_err is the error on f used in
        the fitting. If None, points are weighted equally
    guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess.
    bounds (list or None): custom bounds to overwrite the default option
    return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess
    catch_exceptions (bool): If True, catches exceptions in the fitter and
       returns nan values instead. Else, raises exceptions in the fitter

    Returns:
    if return_dataframe is False:
        p0 (list): initial guess parameters [f0_guess, alpha_guess, Tc_guess]
        popt (list): fit parameters [f0, alpha, Tc]
        perr (list): fit parameter uncertainties [f0_err, alpha_err, Tc_err]
    else:
        row (pd.Series): contains p0, popt, and perr
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    T, f = np.array(T), np.array(f)
    ix = np.argsort(T)
    T, f = T[ix], f[ix]
    if f_err is not None:
       f_err = np.array(f_err)[ix]

    fit_func = lambda a, b, c, d: f_vs_T_qp(a, b, c, d, gamma = gamma)
    # Initial guess
    if guess is not None:
       p0 = guess
       bounds0 = get_bounds_f_vs_T_qp(p0)
    else:
       p0, bounds0 = guess_p0_f_vs_T_qp(T, f, Tc_guess, gamma)
    if bounds is None:
       bounds = bounds0
    # Fit
    if f_err is None:
       sigma = None
       p00 = p0
    else:
       sigma = f_err
       try:
          p00, _ = curve_fit(fit_func, T, f, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except Exception as e:
          if not catch_exceptions:
              raise e
          p00 = p0
    try:
       popt, pcov = curve_fit(fit_func, T, f, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
    except Exception as e:
       if not catch_exceptions:
           raise e
       popt = [np.nan, np.nan, np.nan]
       perr = [np.nan, np.nan, np.nan]
    # Plot
    if plotq:
       fig, ax = plot_f_vs_T_qp(T, f, f_err, popt, p0, gamma)
    else:
       fig, ax = None, None

    if return_dataframe:
       row = make_fit_row_f_vs_T_qp(p0, popt, perr, gamma)
       return row, (fig, ax)
    return p0, popt, perr, (fig, ax)

def fit_Q_vs_T_qp(T, Q, f0_guess, gamma = 1, Tc_guess = 1.2, Q_err = None,
                  guess = None, bounds = None, return_dataframe = False,
                  plotq = False, catch_exceptions = False):
    """
    Fits quality factor versus temperature data to Q_vs_T_qp.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    f0_guess (float): guess for the resonant frequency in Hz at T = 0 K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    Tc_guess (float): critical temperature guess in K
    Q_err (array-like or None): If not None, Q_err is the error on Q used in
        the fitting. If None, points are weighted equally
    guess (list or None): If not None, overwrites the initial guess. Also
        overwrites f0_guess.
    bounds (list or None): custom bounds to overwrite the default option
    return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess
    catch_exceptions (bool): If True, catches exceptions in the fitter and
       returns nan values instead. Else, raises exceptions in the fitter

    Returns:
    if return_dataframe is False:
        p0 (list): initial guess parameters [f0_guess, alpha_guess,
                                             Tc_guess, delta_z_guess]
        popt (list): fit parameters [f0, alpha, Tc, delta_z]
        perr (list): fit parameter uncertainties [f0_err, alpha_err,
                                                  Tc_err, delta_z_err]
    else:
        row (pd.Series): contains p0, popt, and perr
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    T, Q = np.array(T), np.array(Q)
    ix = np.argsort(T)
    T, Q = T[ix], Q[ix]
    if Q_err is not None:
        Q_err = np.array(Q_err)[ix]

    fit_func = lambda a, b, c, d, e: Q_vs_T_qp(a, b, c, d, e, gamma = gamma)
    # Initial guess
    if guess is not None:
        p0 = guess
        bounds0 = get_bounds_Q_vs_T_qp(p0)
    else:
        p0, bounds0 = guess_p0_Q_vs_T_qp(T, Q, f0_guess, Tc_guess, gamma)
    if bounds is None:
        bounds = bounds0
    # Fit
    if Q_err is None:
        sigma = None
        p00 = p0
    else:
        sigma = Q_err
        try:
            p00, _ = curve_fit(fit_func, T, Q, sigma = None,
                             p0 = p0, bounds = bounds)
            # To fit with sigma, the initial guess must be really good, so
            # update the initial guess with curve_fit without sigma
        except Exception as e:
            if not catch_exceptions:
                raise e
         p00 = p0
    try:
        popt, pcov = curve_fit(fit_func, T, Q, sigma = sigma,
                             p0 = p00, bounds = bounds, absolute_sigma = True)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        if not catch_exceptions:
            raise e
        popt = [np.nan, np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan, np.nan]
    # Plot
    if plotq:
        fig, ax = plot_Q_vs_T_qp(T, Q, Q_err, popt, p0, gamma)
    else:
        fig, ax = None, None

    if return_dataframe:
        row = make_fit_row_Q_vs_T_qp(p0, popt, perr, gamma)
        return row, (fig, ax)
    return p0, popt, perr, (fig, ax)

################################################################################
########################### Resonance shift from TLS ###########################
################################################################################
def fit_f_vs_T_tls(T, f, f_err = None, guess = None, bounds = None,
                   return_dataframe = False, plotq = False,
                   catch_exceptions = False):
    """
    Fits resonant frequency versus temperature data to f_vs_T_tls.

    Parameters:
    T (array-like): temperature data in K
    f (array-like): resonant frequency data in Hz
    f_err (array-like or None): If not None, f_err is the error on f used in
        the fitting. If None, points are weighted equally
    guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess.
    bounds (list or None): custom bounds to overwrite the default option
    return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess
    catch_exceptions (bool): If True, catches exceptions in the fitter and
       returns nan values instead. Else, raises exceptions in the fitter

    Returns:
    if return_dataframe is False:
        p0 (list): initial guess parameters [f0_guess, Fdelta0_guess]
        popt (list): fit parameters [f0, Fdelta0]
        perr (list): fit parameter uncertainties [f0_err, Fdelta0_err]
    else:
        row (pd.Series): contains p0, popt, and perr
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    T, f = np.array(T), np.array(f)
    ix = np.argsort(T)
    T, f = T[ix], f[ix]
    if f_err is not None:
       f_err = np.array(f_err)[ix]

    fit_func = lambda a, b, c: f_vs_T_tls(a, b, c)
    # Initial guess
    if guess is not None:
       p0 = guess
       bounds0 = get_bounds_f_vs_T_tls(p0)
    else:
       p0, bounds0 = guess_p0_f_vs_T_tls(T, f)
    if bounds is None:
        bounds = bounds0
    # Fit
    if f_err is None:
       sigma = None
       p00 = p0
    else:
       sigma = f_err
       try:
           p00, _ = curve_fit(fit_func, T, f, sigma = None, p0 = p0,
                              bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except Exception as e:
           if not catch_exceptions:
               raise e
           p00 = p0
    try:
       popt, pcov = curve_fit(fit_func, T, f, sigma = sigma, p0 = p00,
                              bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        if not catch_exceptions:
            raise e
        popt = [np.nan, np.nan]
        perr = [np.nan, np.nan]
    # Plot
    if plotq:
       fig, ax = plot_f_vs_T_tls(T, f, f_err, popt, p0)
    else:
       fig, ax = None, None

    if return_dataframe:
       row = make_fit_row_f_vs_T_tls(p0, popt, perr)
       return row, (fig, ax)
    return p0, popt, perr, (fig, ax)

def fit_Q_vs_T_tls(T, Q, f0_guess, Q_err = None,  guess = None, bounds = None,
                   return_dataframe = False, plotq = False,
                   catch_exceptions = False):
    """
    Fits resonant frequency versus temperature data to Q_vs_T_tls.

    Parameters:
    T (array-like): temperature data in K
    Q (array-like): quality factor data
    Q_err (array-like or None): If not None, fr_err is the error on Q used in
        the fitting. If None, points are weighted equally
    f0_guess (float): guess for the resonant frequency in Hz at T = 0 K
    guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess.
    bounds (list or None): custom bounds to overwrite the default option
    return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess
    catch_exceptions (bool): If True, catches exceptions in the fitter and
       returns nan values instead. Else, raises exceptions in the fitter

    Returns:
    if return_dataframe is False:
        p0 (list): initial guess parameters [f0_guess, Fdelta0_guess,
                                             delta_z_guess]
        popt (list): fit parameters [f0, Fdelta0, delta_z]
        perr (list): fit parameter uncertainties [f0_err, Fdelta0_err,
                                                  delta_z_err]
    else:
        row (pd.Series): contains p0, popt, and perr
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    T, Q = np.array(T), np.array(Q)
    ix = np.argsort(T)
    T, Q = T[ix], Q[ix]
    if Q_err is not None:
        Q_err = np.array(Q_err)[ix]

    fit_func = lambda a, b, c, d: Q_vs_T_tls(a, b, c, d)
    # Initial guess
    if guess is not None:
        p0 = guess
        bounds = get_bounds_Q_vs_T_tls(p0)
    else:
        p0, bounds = guess_p0_Q_vs_T_tls(T, Q, f0_guess)
    if bounds is None:
        bounds = bounds0
    # Fit
    if Q_err is None:
       sigma = None
       p00 = p0
    else:
        sigma = Q_err
        try:
            p00, _ = curve_fit(fit_func, T, Q, sigma = None, p0 = p0,
                              bounds = bounds)
            # To fit with sigma, the initial guess must be really good, so
            # update the initial guess with curve_fit without sigma
        except Exception as e:
            if not catch_exceptions:
               raise e
            p00 = p0
    try:
        popt, pcov = curve_fit(fit_func, T, Q, sigma = sigma, p0 = p00,
                              bounds = bounds, absolute_sigma = True)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        if not catch_exceptions:
            raise e
        popt = [np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan]
    # Plot
    if plotq:
       fig, ax = plot_Q_vs_T_tls(T, Q, Q_err, popt, p0)
    else:
       fig, ax = None, None

    if return_dataframe:
       row = make_fit_row_Q_vs_T_tls(p0, popt, perr)
       return row, (fig, ax)
    return p0, popt, perr, (fig, ax)
