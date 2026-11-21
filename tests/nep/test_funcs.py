"""
Tests of the NEP formulas.
"""
import numpy as np
import pytest
from scipy.constants import h
from citkid.nep.funcs import (
    nep_from_sxx, nep_photon, nu_per_tone, responsivity_ok)


def test_nep_from_sxx_and_sign():
    """
    NEP = sqrt(Sxx) / |R(P)| with the rolled-off responsivity; a fit with
    R0 >= 0 or a bad P0 is not usable and gives NaN, never |R0|.
    """
    power = np.array([0., 1e-15, 3e-15])
    sxx = np.array([1e-16, 4e-16, 9e-16])
    R0, P0 = -2e12, 1e-15
    nep = nep_from_sxx(sxx, power, R0, P0)
    resp = np.abs(R0) * (1 + power / P0) ** -0.5
    np.testing.assert_allclose(nep, np.sqrt(sxx) / resp)
    assert nep[0] == pytest.approx(1e-8 / 2e12)
    assert responsivity_ok(R0, P0)
    for bad in ((2e12, P0), (0., P0), (np.nan, P0), (R0, 0.), (R0, -1e-15),
                (R0, np.nan)):
        assert not responsivity_ok(*bad)
        assert np.isnan(nep_from_sxx(sxx, power, *bad)).all()
    # NaN Sxx stays NaN
    nep = nep_from_sxx([np.nan, 4e-16], [0., 1e-15], R0, P0)
    assert np.isnan(nep[0]) and np.isfinite(nep[1])


def test_nu_per_tone():
    """
    One frequency for every tone, or a dict per resonator with a default.
    """
    res = [3, 7, 9]
    np.testing.assert_allclose(nu_per_tone(150e9, res), 150e9)
    nus = nu_per_tone({3: 100e9, 'default': 150e9}, res)
    np.testing.assert_allclose(nus, [100e9, 150e9, 150e9])
    nus = nu_per_tone({7: 220e9}, res)
    assert np.isnan(nus[0]) and nus[1] == 220e9 and np.isnan(nus[2])
    with pytest.raises(ValueError, match = 'frequency in Hz'):
        nu_per_tone('x', res)
    # The photon-noise NEP at the tone's frequency
    assert nep_photon(1e-15, 0.5, 150e9) == pytest.approx(
        np.sqrt(2 * h * 150e9 * 1e-15 / 0.5))
