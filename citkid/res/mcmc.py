import numpy as np
import emcee
import corner
from .funcs import nonlinear_iq_for_fitter
from citkid.res.data_io import nonlinear_iq_p_labels

def run_mcmc(f, z_stacked, popt, nsteps = 2000, nwalkers = 256,
             ndiscard = 200, nthin = 1, downward = True, plotq = True,
             verbose = True, **kwargs):
    """
    Run a Monte-Carlo Markov Chain analysis of nonlinear_iq data.

    Parameters:
    f (np.array): Frequency data.
    z_stacked (np.array): Horizontally stacked complex IQ data.
    popt (array-like): Optimal parameters from least-squares or other fitting.
    nsteps (int): Number of MCMC steps. Minimum is 1,000.
    nwalkers (int): Number of MCMC walkers. 256 is recommended.
    ndiscard (int): Number of samples to discard.
    nthin (int): Factor by which the samples are thinned.
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.
    plotq (bool): If True, returns a corner plot.
    verbose (bool): If True, tracks progress with a progress bar. If False,
        does not track progress.
    **kwargs: Arguments for corner.corner when plotting.

    Returns:
    params_err (np.array): Uncertainty on optimal parameters.
    flat_samples (np.array): Flattened MCMC samples.
    fig (pyplot.figure): Corner plot figure.
    """
    if verbose:
        if running_in_notebook():
            progress = 'notebook'
        else:
            progress = True
    else:
        progress = False
    log_probability, bounds = get_log_probability(f, popt, downward)
    sigma = 0.5
    ndim = len(popt)
    # Initialize MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(f, z_stacked, sigma))
    initial_pos = np.zeros((nwalkers, len(popt)))
    for i in range(len(popt)):
        initial_pos[:, i] = np.random.uniform(bounds[0][i], bounds[1][i],
                                              nwalkers)
    # Run MCMC
    sampler.run_mcmc(initial_pos, nsteps, progress = progress,
                     progress_kwargs = {'leave': False})
    # Extract samples
    samples = sampler.get_chain(discard = ndiscard, flat=True)
    # Calculate uncertainties
    params_err = np.std(samples, axis=0)

    if plotq:
        flat_samples = sampler.get_chain(discard = ndiscard, thin = nthin,
                                         flat = True)
        fig = corner.corner(flat_samples, labels=nonlinear_iq_p_labels,
                            truths=popt, histogram_bin_factor = nthin,
                            truth_color = 'r', layout = 'tight', **kwargs)
        fig.tight_layout()
    else:
        fig = None
    return params_err, flat_samples, fig

################################################################################
######################### Utility functions ####################################
################################################################################
def get_log_probability(f, popt, downward = True):
    """
    Creates the log probability function

    Parameters:
    f (np.array): frequency data
    popt (array-like): optimal parameters, found using least-squares or other
        fitting method
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    log_probability (func): log probability function
    bounds (array-like): parameter bounds
    """
    # These bounds work pretty well for the data I tested on, but that was a
    # limited set
    bounds0 = (
        [np.min(f), 1e3, .01, -np.pi * 1.5, 0, -1e2, -1e2, -1.0e-6],
        [np.max(f), 1e7,   1,  np.pi * 1.5, 2,  1e2,  1e2,  1.0e-6],
    )
    bound_diffs = [1e5, 12000, 0.5, np.pi / 2, 0.5, 10, 10, 1e-7]
    bounds = [[popt[i] - bound_diffs[i] for i in range(len(popt))],
              [popt[i] + bound_diffs[i] for i in range(len(popt))]]

    for i in range(len(bounds[0])):
        if bounds[0][i] < bounds0[0][i]:
            bounds[0][i] = bounds0[0][i]
        if bounds[1][i] > bounds0[1][i]:
            bounds[1][i] = bounds0[1][i]
    def log_prior(params):
        """
        Calculate the log prior of the parameters

        Parameters:
        params (array-like): model parameters

        Returns:
        (float): log prior probability value
        """
        in_bounds = [
            bounds[0][i] < params[i] < bounds[1][i]
            for i in range(len(params))
        ]
        if all(in_bounds):
            return 0.0
        return -np.inf

    def log_probability(params, x, y, sigma, downward):
        """
        Calculate the log of the posterior probability

        Parameters:
        params (array-like): model parameters
        x (array-like): model x data
        y (array-like): model y data
        sigma (float): standard deviation of the observational uncertainties
        downward (bool): True corresponds to a downward sweep, False corresponds
            to an upward sweep

        Returns:
        (float): log of the posterior probability
        """
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y, sigma, downward)
    return log_probability, bounds

def model(f, params, downward = True):
    """
    Wrapping the nonlinear_iq_for_fitter model.

    Parameters:
    f (np.array): frequency data
    params (list): list of parameter inputs to nonlinear_iq_for_fitter
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    z (np.array): horizontally stacked complex IQ data
    """
    return nonlinear_iq_for_fitter(f, *params, downward)

def log_likelihood(params, x, y, sigma, downward = True):
    """
    Calculate the log likelihood of the given parameters

    Parameters:
    params (array-like): model parameters
    x (array-like): model x data
    y (array-like): model y data
    sigma (float): standard deviation of the observational uncertainties
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.

    Returns:
    ll (float): log likelihood value
    """
    # Compute the log-likelihood given the model parameters
    y_model = model(x, params, downward)
    ll = -0.5 * np.sum(
        ((y - y_model) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2)
    )
    return ll

def running_in_notebook():
    """
    Checks if the code is currently running in a jupyter notebook

    Returns:
    (bool): if True, the current platform is a notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        else:
            return False # Other type (?)
    except NameError:
        return False # Probably standard Python interpreter
