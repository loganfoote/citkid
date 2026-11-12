"""
Quick conversion of a timestream to resonance frequency shift, by
projecting it onto the IQ loop's tangent at the tone frequency.

This is a preview-level calibration (e.g. for checking a stream live): it
needs only the calibration sweep around the tone, no circle or gain fit.
The full calibration is ``xcal``.
"""
import numpy as np

def loop_tangent_shift(f, z, f_tone, zt, n_fit = 5):
    """
    Project a timestream onto the IQ loop's tangent at the tone frequency.

    The calibration sweep's slope s = dz/df at the tone frequency comes
    from a quadratic fit of z vs f over the ``n_fit`` sweep points on each
    side of the tone (a straight line would be biased by the loop's
    curvature). A timestream sample's offset from the loop at the tone,
    dz = zt - z0 (z0 from the fit at ``f_tone``), splits into
    the part along the loop, Re[dz conj(s)] / |s|^2, which is the tone
    frequency that sample would correspond to on the sweep (in Hz from
    ``f_tone``), and the part across it, Im[dz conj(s)] / |s|^2, in the
    same units (the dissipation direction near resonance). A resonance
    shift dfr moves the loop under a fixed tone like a tone shift of -dfr,
    so the shift is returned as dfr = -Re[dz conj(s)] / |s|^2. The gain and
    cable delay cancel, since the sweep and the stream pass through the
    same chain at the same tone.

    The stream must be in the sweep's phase frame: measured with the
    same NCO setting, since setting an NCO changes the frame (see
    ``crs.procedures.calibrate_tone_set``); off the frame, the projection
    is wrong and its sign can flip.

    Parameters:
    f (np.array): calibration sweep frequencies in Hz (any order).
    z (np.array): calibration sweep complex S21, same shape as ``f``.
    f_tone (float): tone frequency in Hz.
    zt (np.array): complex timestream of the tone, in the units of ``z``.
    n_fit (int): sweep points on each side of the tone in the fit. Default
        5.

    Returns:
    dfr (np.array): resonance frequency shift in Hz, per sample (positive
        when the resonance moves up). Divide by ``f_tone`` for x.
    dperp (np.array): offset across the loop, in Hz-equivalent units.
    z0 (complex): the sweep's S21 at the tone (on the fitted curve).
    slope (complex): dz/df at the tone, per Hz.

    Raises:
    ValueError: if the sweep has fewer than 3 points or the slope is 0.
    """
    f = np.asarray(f, dtype = np.float64).ravel()
    z = np.asarray(z, dtype = np.complex128).ravel()
    if f.shape != z.shape or len(f) < 3:
        raise ValueError('f and z need the same shape and 3 or more points')
    o = np.argsort(f)
    f, z = f[o], z[o]
    i = int(np.searchsorted(f, f_tone))
    n_fit = max(int(n_fit), 2)
    lo, hi = max(0, i - n_fit), min(len(f), i + n_fit)
    if hi - lo < 3:
        lo, hi = max(0, min(lo, len(f) - 3)), max(3, hi)
    x = f[lo:hi] - f_tone
    _, slope_re, z0_re = np.polyfit(x, z[lo:hi].real, 2)
    _, slope_im, z0_im = np.polyfit(x, z[lo:hi].imag, 2)
    slope = complex(slope_re, slope_im)
    z0 = complex(z0_re, z0_im)
    if abs(slope) * (f[hi - 1] - f[lo]) <= 1e-12 * np.abs(z[lo:hi]).max():
        raise ValueError('the sweep has zero slope at the tone')
    proj = (np.asarray(zt, dtype = np.complex128) - z0) * np.conj(slope) / \
        abs(slope) ** 2
    return -proj.real, proj.imag, z0, slope
