import numpy as np
from ..res.fitter import fit_nonlinear_iq
from ..util import combine_figures_vertically_legacy as combine_figures_vertically
from .gain import fit_and_remove_gain_phase
from .data_io import make_fit_row

def fit_nonlinear_iq_with_gain(
    fgain, zgain, ffine, zfine, frs, Qrs, downward = True, plotq = False,
    return_dataframe = False, floats_only=False, **kwargs
):
    """
    Fit IQ data with gain amplitudes and phase correction from a gain sweep.
    Cuts resonant frequencies from the gain sweep in spans of fr / Qr around fr,
    where fr is an item in frs and Qr is a corresponding quality factor in Qrs.

    The optimal fine sweep width is 6 * fr / Qr.
    The optimal gain sweep width is 100 * fr / Qr.

    Parameters:
    fgain (np.array): gain sweep frequency data.
    zgain (np.array): gain sweep complex S21 data.
    ffine (np.array): fine sweep frequency data.
    zfine (np.array): fine sweep complex S21 data.
    frs (list of float): resonant frequencies to cut from the gain sweep.
    Qrs (list of float): spans of frs / Qrs are cut from the gain sweep.
    downward (bool): If True, fits the equation for a downward sweep. If
        False, fits for an upward sweep.
    plotq (bool): If True, plots the fits.
    return_dataframe (bool): if True, returns the output of
        .data_io.make_fit_row instead of the separated data.
    floats_only (bool): Set to True to only keep columns in the
        dataframe whose values can be represented as floats,
        i.e. don't store columns for sweep_direction or plotpath.
    **kwargs: other arguments for citkid.res.fitter.fit_nonlinear_iq.

    Returns:
    if return_dataframe:
        row (pd.Series): fit data as a pandas series.
        fig (pyplot.figure or None): figure with gain fit and nonlinear IQ
            fit if plotq, or None.
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
