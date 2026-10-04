import numpy as np
from ..xcal import gain as xcal_gain
from .plot import plot_gain_fit

def fit_and_remove_gain_phase(
    fgain, zgain, ffine, zfine, frs = [], Qrs = [], plotq = False
):
    """
    Fit the gain sweep with ``citkid.xcal.gain.fit_gain`` and remove the gain
    amplitude and phase from the fine sweep data.

    Resonances are cut from the gain sweep in spans of fr / Qr around each fr.
    Qrs should be no higher than 10 X Qr of the resonances. The gain sweep and
    the resonances are sorted by frequency before fitting, so they can be in
    any order. Resonances with a NaN frequency or span cut nothing and are
    ignored.

    Parameters:
    fgain (np.array): gain sweep frequency data in Hz.
    zgain (np.array): gain sweep complex S21 data.
    ffine (np.array): fine sweep frequency data in Hz.
    zfine (np.array): fine sweep complex S21 data.
    frs (list of float): resonant frequencies to cut from the gain sweep, in
        Hz.
    Qrs (list of float): spans of frs / Qrs are cut from the gain sweep.
    plotq (bool): If True, also plots the fits to the gain sweep.

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB.
    p_phase (np.array): 1st-order polynomial fit parameters to phase.
    z_rmvd (np.array): zfine with gain amplitude and phase removed.
    fig, axs (pyplot figure and axes, or None): if plotq, a plot of the gain
        amplitude and phase fits. Otherwise, (None, None).
    """
    # Input validation
    fgain = np.asarray(fgain, dtype = np.float64)
    zgain = np.asarray(zgain, dtype = np.complex128)
    frs = np.atleast_1d(np.asarray(frs, dtype = np.float64))
    Qrs = np.atleast_1d(np.asarray(Qrs, dtype = np.float64))
    if fgain.shape != zgain.shape:
        raise ValueError('fgain and zgain must be the same shape')
    if frs.shape != Qrs.shape:
        raise ValueError('frs and Qrs must be the same length')

    # Sort the gain sweep and the resonance spans, which xcal.gain requires
    ix = np.argsort(fgain)
    fgain, zgain = fgain[ix], zgain[ix]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        spans = frs / Qrs
    keep = ~np.isnan(frs) & ~np.isnan(spans)
    order = np.argsort(frs[keep])
    fr_spans = [(float(fr), float(span)) for fr, span in
                zip(frs[keep][order], spans[keep][order])]

    # Fit and remove gain
    p_amp, p_phase, mask = xcal_gain.fit_gain(fgain, zgain, fr_spans)
    z_rmvd = xcal_gain.remove_gain(ffine, zfine, p_amp, p_phase)

    # Plot
    if plotq:
        fcut, zcut = fgain[mask], zgain[mask]
        fig, axs = plot_gain_fit(
            fgain, 20 * np.log10(np.abs(zgain)), fcut,
            20 * np.log10(np.abs(zcut)), np.unwrap(np.angle(zcut)), p_amp,
            p_phase
            )
    else:
        fig, axs = None, None
    return p_amp, p_phase, z_rmvd, (fig, axs)
