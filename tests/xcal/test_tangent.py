"""
Tests for ``citkid.xcal.tangent``.
"""
import numpy as np
import pytest
from citkid.res.funcs import linear_iq
from citkid.xcal.tangent import loop_tangent_shift

P = (0.8, 0.3, 1.2, -0.4, 50e-9)   # amp, phi, i0, q0, tau


@pytest.mark.parametrize('detune_lw', [0., 0.3, -0.5])
def test_loop_tangent_shift_recovers_small_shifts(detune_lw):
    fr, Qr = 1e9, 2e4
    lw = fr / Qr
    f = fr + np.linspace(-3 * lw, 3 * lw, 601)
    z = linear_iq(f, fr, Qr, *P)
    f_tone = fr + detune_lw * lw
    # The resonance moves by a small shift: S21 at the tone is the
    # original loop at f_tone - shift
    shifts = np.array([-200., 0., 150., 300.])
    zt = linear_iq(f_tone - shifts, fr, Qr, *P)
    dfr, dperp, z0, slope = loop_tangent_shift(f, z, f_tone, zt)
    # Within ~1 Hz at zero shift (cubic residual of the fit); the rest is
    # second order in the shift (300 Hz is 0.006 linewidths)
    np.testing.assert_allclose(dfr, shifts, atol = 3.)
    assert abs(dfr[1]) < 2. and np.abs(dperp).max() < 5.
    zl = linear_iq(np.array([f_tone]), fr, Qr, *P)
    assert abs(z0 - zl[0]) < 2e-4 * abs(zl[0])


def test_loop_tangent_shift_dissipation_and_order():
    f = np.linspace(-10., 10., 21)
    z = (1 + 2j) * f / 10      # slope (1 + 2j) / 10 per Hz
    zt = np.array([1j * (1 + 2j) / 10 * 3])   # 3 Hz across the line
    dfr, dperp, z0, slope = loop_tangent_shift(f[::-1], z[::-1], 0., zt)
    np.testing.assert_allclose(dfr, [0.], atol = 1e-9)
    np.testing.assert_allclose(dperp, [3.])
    assert abs(slope - (1 + 2j) / 10) < 1e-12 and abs(z0) < 1e-12
    with pytest.raises(ValueError, match = 'shape'):
        loop_tangent_shift([1., 2.], [1j, 2j], 1., zt)
    with pytest.raises(ValueError, match = 'zero slope'):
        loop_tangent_shift(f, np.ones(21, complex), 0., zt)
