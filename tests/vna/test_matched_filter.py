import numpy as np
from citkid.vna.matched_filter import lorentz_template, find_candidates_mf

def test_lorentz_template():
    t = lorentz_template(10.)
    np.testing.assert_allclose(np.linalg.norm(t), 1)
    # Orthogonal to the nuisance polynomial (constant, linear, quadratic)
    u = np.arange(len(t)) - len(t) // 2
    for k in range(3):
        assert abs(np.vdot((u / u.max()) ** k, t)) < 1e-10

def test_find_candidates_mf(full_sweep):
    f, z, _, t = full_sweep
    fr, Qr, score = find_candidates_mf(f, z)
    lw = t['fr'] / t['Qr']
    d = np.array([np.min(np.abs(fr - x)) for x in t['fr']]) / lw
    assert np.mean(d < 0.5) > 0.9
    assert np.all(np.diff(fr) > 0) and np.all(score >= 6)
