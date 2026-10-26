import numpy as np
from citkid.res.funcs import product_s21
from citkid.vna.gain import resonator_model, fit_gain_spline

def test_resonator_model_matches_product(full_sweep):
    f, _, _, t = full_sweep
    exact = product_s21(f, t['fr'], t['Qr'], t['amp'], t['phi'])
    model = resonator_model(f, t['fr'], t['Qr'], t['amp'], t['phi'])
    # Far tails are interpolated: steps of ~amp / (2 n_lw) at the edges of
    # each resonance's exact window
    np.testing.assert_allclose(model, exact, atol = 3e-3)
    assert np.median(np.abs(model - exact)) < 1e-4

def test_resonator_model_empty():
    f = np.linspace(1e9, 1.01e9, 100)
    np.testing.assert_array_equal(resonator_model(f, [], [], [], []),
                                  np.ones(100))

def test_fit_gain_spline_recovers_gain(full_sweep):
    f, z, gain, t = full_sweep
    g, info = fit_gain_spline(f, z, t['fr'], t['Qr'], t['amp'], t['phi'])
    assert np.median(np.abs(g / gain - 1)) < 2e-3
    assert info['n_modeled'] == len(t['fr']) and info['n_masked'] == 0

def test_fit_gain_spline_masks_unfit(full_sweep):
    f, z, gain, t = full_sweep
    Qr = t['Qr'].copy()
    Qr[3] = np.nan  # no fit: its neighborhood is masked instead
    g, info = fit_gain_spline(f, z, t['fr'], Qr, t['amp'], t['phi'])
    assert info['n_modeled'] == len(Qr) - 1 and info['n_masked'] == 1
    assert np.median(np.abs(g / gain - 1)) < 5e-3
