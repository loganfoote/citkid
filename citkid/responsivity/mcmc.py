import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from .funcs import responsivity_int_for_fitter, responsivity, responsivity_int
from ..util import get_fit_bound_curves, format_str_scientific_with_err

def run_mcmc(power, x, popt, nsteps = 100000, nwalkers = 256,
             ndiscard = 10000, nthin = 100, bound_factors = [3, 10, 1.2],
             nstd_cut = 5, plot_cornerq = True, plot_respq = True,
             plot_resp_intq = True, verbose = True, **kwargs):
    """
    Run a Monte-Carlo Markov Chain analysis.

    Parameters:
    popt (array-like): Optimal parameters from least-squares or other fitting.
    nsteps (int): Number of MCMC steps. Minimum is 1,000.
    nwalkers (int): Number of MCMC walkers. 256 is recommended.
    ndiscard (int): Number of samples to discard.
    nthin (int): Factor by which samples are thinned.
    bound_factors (list): list of factors by which each value in popt is
        multiplied/divided to get the bounds
    nstd_cut (float): Number of standard deviations away from the mean for
        which data should be cut before plotting
    plot_cornerq (bool): If True, returns a corner plot.
    plot_respq (bool): If True, returns a plot of the responsivity versus power
        with the fit and error bars
    plot_resp_intq (bool): If True, returns a plot of integrated responsivity
        versus power with the fit and error bars.
    verbose (bool): If True, tracks progress with a progress bar. If False,
        does not track progress.
    **kwargs: Arguments for corner.corner when plotting.

    Returns:
    perr (np.array): Uncertainty on optimal parameters.
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
    p0 = popt.copy()
    p0[0] /= 1e9
    p0[1] /= 1e-16
    log_probability, bounds = get_log_probability(
        np.log(power),
        p0,
        bound_factors = bound_factors,
    )
    sigma = 0.5
    ndim = len(p0)
    # Initialize MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args = (np.log(power), x * 1e6, sigma),
    )
    initial_pos = np.zeros((nwalkers, len(p0)))
    for i in range(len(p0)):
        initial_pos[:, i] = np.random.uniform(bounds[0][i], bounds[1][i],
                                              nwalkers)
    # Run MCMC
    sampler.run_mcmc(initial_pos, nsteps, progress = progress,
                     progress_kwargs = {'leave': False})
    # Extract samples
    samples = sampler.get_chain(discard = ndiscard, flat=True)
    # Calculate uncertainties
    perr = np.std(samples, axis=0)
    flat_samples = sampler.get_chain(
        discard = ndiscard,
        thin = nthin,
        flat = True,
    )
    perr[0] *= 1e9
    perr[1] *= 1e-16
    flat_samples[:, 0] *= 1e9
    flat_samples[:, 1] *= 1e-16
    if plot_cornerq:
        fig_corner = plot_corner(
            flat_samples,
            popt,
            nthin,
            nstd_cut = nstd_cut,
            **kwargs,
        )
    else:
        fig_corner = None
    if plot_respq:
        fig_resp = plot_resp(power, popt, perr)
    else:
        fig_resp = None
    if plot_resp_intq:
        fig_resp_int = plot_resp_int(power, x, popt, perr)
    else:
        fig_resp_int = None
    bounds[0][0] *= 1e9
    bounds[1][0] *= 1e9
    bounds[0][1] *= 1e-16
    bounds[1][1] *= 1e-16
    return perr, flat_samples, (fig_corner, fig_resp, fig_resp_int), bounds

################################################################################
################################## Plotting ####################################
################################################################################
def plot_corner(flat_samples, popt, nthin, nstd_cut = 5, **kwargs):
    """
    Create a corner plot of the MCMC results.

    Parameters:
    flat_samples (np.array): MCMC samples.
    popt (np.array): Optimal fit parameters.
    nthin (int): Factor by which data is thinned for the plot.
    nstd_cut (float): Number of standard deviations away from the mean for
        which data should be cut before plotting
    **kwargs: Arguments for corner.corner when plotting.

    Returns:
    fig (pyplot.figure): Corner plot figure.
    """
    mean = np.mean(flat_samples, axis=0)
    std  = np.std(flat_samples, axis=0)
    ix = np.all(np.abs(flat_samples - mean) <= nstd_cut * std, axis=1)
    flat_samples_filt = flat_samples[ix]
    fig = corner.corner(
        flat_samples_filt,
        labels = [r'$R_0$', r'$P_0$', r'$c$'],
        truths = popt,
        histogram_bin_factor = nthin,
        truth_color = plt.cm.viridis(0.),
        layout = 'tight',
        **kwargs,
    )
    fig.tight_layout()
    return fig

def plot_resp(power, popt, perr):
    """
    Plots the responsivity versus power best fit and error bars

    Parameters:
    power (array-like): blackbody power data in W
    popt (np.array): fit parameters
    perr (np.array): fit parameter uncertainties

    Returns:
    fig (pyplot.figure): responsivity versus power plot figure
    """
    R0, P0, c = popt
    R0_err, P0_err, c_err = perr
    fig, ax = plt.subplots(figsize = [6, 4], layout = 'tight')
    ax.set(xscale = 'log', yscale = 'log')
    ax.set(xlabel = 'Power (W)', ylabel = '- responsivity (1 / W)')
    psamp = np.geomspace(min(power), max(power), 100)
    rsamp, rsamp_up, rsamp_down = get_fit_bound_curves(psamp, popt[:-1],
                                                perr[:-1], model = responsivity)
    ax.plot(psamp, rsamp, '--k', label = 'fit')
    ax.fill_between(psamp, rsamp_up, rsamp_down, color = 'gray',
                    label = 'fit uncertainty')
    lbl = r'$R_0$ = ' + format_str_scientific_with_err(R0, R0_err) + ' 1 / W\n'
    lbl += r'$P_0$ = ' + format_str_scientific_with_err(P0, P0_err) + ' aW\n'
    lbl += r'$c - 1$ = 1 + ' + format_str_scientific_with_err(c - 1, c_err)
    ax.legend(title = lbl)
    return fig

def plot_resp_int(power, x, popt, perr):
    """
    Plot integrated responsivity versus power with fit and error bars.

    Parameters:
    power (array-like): blackbody power data in W
    x (array-like): df/f data
    popt (np.array): fit parameters
    perr (np.array): fit parameter uncertainties

    Returns:
    fig (pyplot.figure): responsivity versus power plot figure
    """
    R0, P0, c = popt
    R0_err, P0_err, c_err = perr
    fig, ax = plt.subplots(figsize = [6, 4], layout = 'tight')
    ax.set_xscale('log')
    ax.set(ylabel = r'$x$ (kHz / GHz)', xlabel = 'incident power (W)')
    ax.plot(power, x * 1e6, 's', color = plt.cm.viridis(0.), label = 'data')
    psamp = np.geomspace(min(power), max(power), 100)
    xsamp, xsamp_up, xsamp_down = get_fit_bound_curves(psamp, popt, perr,
                                                       model = responsivity_int)
    ax.plot(psamp, xsamp * 1e6, '--', color = 'black', label = 'fit')
    ylim = ax.get_ylim()
    ax.fill_between(psamp, xsamp_up * 1e6, xsamp_down * 1e6, color = 'gray',
                    label = 'fit uncertainty')
    ax.set_ylim(ylim)
    lbl = r'$R_0$ = ' + format_str_scientific_with_err(R0, R0_err) + ' 1 / W\n'
    lbl += r'$P_0$ = ' + format_str_scientific_with_err(P0, P0_err) + ' aW\n'
    lbl += r'$c - 1$ = 1 + ' + format_str_scientific_with_err(c - 1, c_err)
    ax.legend(title = lbl)
    return fig

################################################################################
########################### MCMC utility functions #############################
################################################################################

def get_log_probability(power, popt, bound_factors):
    """
    Creates the log probability function

    Parameters:
    power (array-like): power array
    popt (np.array): fit parameters
    bound_factors (list): list of factors by which each value in popt is
        multiplied/divided to get the bounds

    Returns:
    log_probability (func): log probability function
    bounds (array-like): parameter bounds
    """
    bounds = [[popt[0] * bound_factors[0], popt[1] / bound_factors[1],
               popt[2] / bound_factors[2]],
              [popt[0] / bound_factors[0], popt[1] * bound_factors[1],
               popt[2] * bound_factors[2]]]
    # Flip bounds if they are reversed
    for i in range(len(bounds[0])):
        if bounds[0][i] > bounds[1][i]:
            bounds[0][i], bounds[1][i] = bounds[1][i], bounds[0][i]
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

    def log_probability(params, x, y, sigma):
        """
        Calculate the log of the posterior probability

        Parameters:
        params (array-like): model parameters
        x (array-like): model x data
        y (array-like): model y data
        sigma (float): standard deviation of the observational uncertainties

        Returns:
        (float): log of the posterior probability
        """
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y, sigma)
    return log_probability, bounds

def model(log_power, params):
    """
    Wraps the integrated responsivity function to take a list of parameters
    instead of individual parameters
    """
    return responsivity_int_for_fitter(log_power, *params)

def log_likelihood(params, x, y, sigma):
    """
    Calculate the log likelihood of the given parameters

    Parameters:
    params (array-like): model parameters
    x (array-like): model x data
    y (array-like): model y data
    sigma (float): standard deviation of the observational uncertainties

    Returns:
    ll (float): log likelihood value
    """
    # Compute the log-likelihood given the model parameters
    y_model = model(x, params)
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
