import numpy as np
from citkid.res.funcs import product_s21
from citkid.vna.gain import resonator_model, fit_gain_spline, normalize_edges

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

def _edge_sweeps(M = 4, N = 100):
    rng = np.random.default_rng(7)
    amp = rng.uniform(0.5, 2.0, (M, 1))
    phase = rng.uniform(-np.pi, np.pi, (M, 1))
    z = amp * np.exp(1j * phase) * np.ones((M, N))
    z[:, N // 4: 3 * N // 4] *= 0.5  # resonance dip in the middle
    return z

def test_normalize_edges_shape_and_dtype():
    z = _edge_sweeps(M = 5, N = 150)
    out = normalize_edges(z)
    assert out.shape == z.shape
    assert np.iscomplexobj(out)

def test_normalize_edges_amplitude_and_phase():
    z = _edge_sweeps(M = 2, N = 200)
    out = normalize_edges(z)
    n = out.shape[1] // 100
    edge = np.concatenate([out[:, :n], out[:, -n:]], axis = 1)
    np.testing.assert_allclose(np.median(np.abs(edge), axis = 1), 1)
    np.testing.assert_allclose(np.angle(edge.mean(axis = 1)), 0, atol = 1e-10)
    # the dip is preserved relative to the edges
    np.testing.assert_allclose(np.abs(out[:, 100]), 0.5)

def test_normalize_edges_n_edge():
    z = _edge_sweeps(M = 1, N = 200)
    z[:, 2:5] *= np.exp(1j)  # only seen when the edge region is wider than 2
    np.testing.assert_array_equal(normalize_edges(z),
                                  normalize_edges(z, n_edge = 2))
    assert not np.allclose(normalize_edges(z), normalize_edges(z, n_edge = 5))

def test_normalize_edges_does_not_modify_input():
    z = _edge_sweeps()
    z0 = z.copy()
    normalize_edges(z)
    np.testing.assert_array_equal(z, z0)

def test_normalize_edges_zero_rows():
    out = normalize_edges(np.zeros((2, 50), dtype = complex))
    assert np.all(np.isfinite(out))

def test_normalize_edges_single_sample():
    out = normalize_edges(np.array([[1.0 + 0.5j]]))
    np.testing.assert_allclose(out, [[1.0]])
