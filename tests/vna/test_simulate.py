"""
Tests for the simulated device (``citkid.vna.simulate``).
"""
import numpy as np
import pytest
from citkid.res.funcs import product_s21
from citkid.vna import simulate as sim


def test_cascade_limits():
    f = np.linspace(0.99e9, 1.01e9, 4001)
    one = sim.cascade_s21(f, [1e9], [2e4], [0.8], [0.2], [])
    np.testing.assert_allclose(one, product_s21(f, 1e9, 2e4, 0.8, 0.2))
    # Far apart (4000 linewidths), reflections are negligible
    fr = np.array([0.995e9, 1.005e9])
    args = (fr, [2e4, 2e4], [0.8, 0.6], [0.1, -0.2])
    pair = sim.cascade_s21(f, *args, [1.3])
    np.testing.assert_allclose(pair, product_s21(f, *args), atol = 2e-3)
    with pytest.raises(ValueError, match = 'theta'):
        sim.cascade_s21(f, *args, [])


def test_dressed_params_keep_the_zero():
    """
    Reflections between close resonators move Qr, amp and phi but not Qi.
    """
    fr = np.array([1e9, 1e9 + 2 * 1e9 / 2e4])
    Qr, amp, phi = np.array([2e4, 2.5e4]), np.array([0.8, 0.6]), \
        np.array([0.1, -0.2])
    fr_d, Qr_d, amp_d, phi_d = sim.dressed_params(fr, Qr, amp, phi, [1.])
    assert np.abs(Qr_d / Qr - 1).max() > 0.05
    Qi, Qi_d = Qr / (1 - amp), Qr_d / (1 - amp_d)
    np.testing.assert_allclose(Qi_d, Qi, rtol = 1e-4)
    # The dressed product model reproduces the cascade
    f = np.linspace(fr[0] - 5e5, fr[1] + 5e5, 2001)
    np.testing.assert_allclose(
        product_s21(f, fr_d, Qr_d, amp_d, phi_d),
        sim.cascade_s21(f, fr, Qr, amp, phi, [1.]), atol = 1e-6)


def test_draw_device():
    rng = np.random.default_rng(1)
    t = sim.draw_device(rng, n_banks = 3, n_per_bank = 50, yield_frac = 1.,
                        pair_frac = 0.1)
    assert np.all(np.diff(t.fr) >= 0)
    assert list(t.res_id) == list(range(len(t)))
    assert 150 < len(t) < 175 and t.paired.sum() >= 2
    np.testing.assert_allclose(1 / t.Qr, 1 / t.Qi + 1 / t.Qc)
    np.testing.assert_allclose(t.amp, t.Qr / t.Qc)
    assert set(t.bank) == {0, 1, 2}
    assert np.all(np.abs(t.phi) <= 1.3)
    t2 = sim.draw_device(np.random.default_rng(1), n_banks = 3,
                         n_per_bank = 50, yield_frac = 1., pair_frac = 0.1)
    np.testing.assert_array_equal(t.fr, t2.fr)


def test_device_s21_and_dressed_truth():
    rng = np.random.default_rng(2)
    t = sim.draw_device(rng, f_lo = 1e9, f_hi = 1.05e9, n_banks = 1,
                        n_per_bank = 20, pair_frac = 0.3)
    f = np.linspace(1e9, 1.05e9, 50001)
    full = sim.device_s21(f, t, window_lw = None)
    win = sim.device_s21(f, t)
    assert np.abs(full - win).max() < 5e-3
    # Shape and order of f are kept
    f2 = f[:-1][::-1].reshape(2, -1)
    s2 = sim.device_s21(f2, t)
    assert s2.shape == (2, 25000)
    np.testing.assert_allclose(s2.ravel(), win[:-1][::-1])
    d = sim.dressed_truth(t)
    iso = d.group_size == 1
    np.testing.assert_array_equal(d.Qr_d[iso], d.Qr[iso])
    grouped = d[(d.group_size > 1) & (d.amp_d > 0) & (d.amp_d < 1)]
    assert len(grouped)
    np.testing.assert_allclose(grouped.Qi_d, grouped.Qi, rtol = 1e-3)
    # Higher power: more nonlinearity changes the response near resonances
    hi = sim.device_s21(f, t, power_dbm = -45.)
    assert np.abs(hi - win).max() > 1e-2


def test_gain():
    rng = np.random.default_rng(3)
    p = sim.draw_gain(rng, f_lo = 1e8, f_hi = 2.5e9)
    f = np.linspace(1e8, 2.5e9, 10000)
    g = sim.gain_s21(f, p)
    assert np.all(np.isfinite(g)) and g.shape == f.shape
    assert sim.gain_s21(f.reshape(100, -1), p).shape == (100, 100)


def test_compare_to_truth():
    rng = np.random.default_rng(4)
    t = sim.dressed_truth(sim.draw_device(rng, n_banks = 1, n_per_bank = 30))
    fit = dict(fr = t.fr_d.values, Qr = t.Qr_d.values, Qi = t.Qi.values,
               Qc = t.Qc_d.values, phi = t.phi_d.values,
               valid_Qi = np.ones(len(t), bool))
    pairs, missed, extra = sim.compare_to_truth(fit, t)
    assert len(pairs) == len(t) and not len(missed) and not len(extra)
    assert np.abs(pairs[['dfr_lw', 'dQr', 'dQi', 'dQc', 'dphi']]
                  .values).max() < 1e-9
    assert pairs.valid_Qi.all()
    # Drop one fit and add a far-away one
    fit = {k: np.append(v[1:], v[0]) for k, v in fit.items()}
    fit['fr'][-1] = 3e9
    pairs, missed, extra = sim.compare_to_truth(fit, t)
    assert list(missed) == [0] and list(extra) == [len(t) - 1]
