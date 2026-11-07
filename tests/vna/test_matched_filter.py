import numpy as np
import pytest
from citkid.res.funcs import product_s21
from citkid.vna.matched_filter import lorentz_template, find_candidates_mf

F0 = 1.01e9   # strong resonance
QR = 2e4      # Qr of every simulated resonance
LW = F0 / QR  # its linewidth in Hz

def _sweep(fr, amp, sigma = 1e-4, f = None, seed = 0):
    """
    Simulate a sweep of resonances with Qr = ``QR`` on a flat gain.

    Parameters:
    fr (list): resonance frequencies in Hz.
    amp (list): their amplitudes Qr / Qc.
    sigma (float): white noise per quadrature. Default 1e-4.
    f (np.array or None): frequencies in Hz. Default None (1-1.02 GHz in
        2 kHz steps).
    seed (int): random seed. Default 0.

    Returns:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21.
    """
    rng = np.random.default_rng(seed)
    if f is None:
        f = np.arange(1e9, 1.02e9, 2e3)
    z = product_s21(f, fr, [QR] * len(fr), amp, np.zeros(len(fr)))
    z = z + sigma * (rng.standard_normal(len(f))
                     + 1j * rng.standard_normal(len(f)))
    return f, z

def _near(fr, x, tol_lw):
    """
    Return the candidates within ``tol_lw`` linewidths of ``x``.

    Parameters:
    fr (np.array): candidate frequencies in Hz.
    x (float): frequency in Hz.
    tol_lw (float): tolerance in linewidths ``LW``.

    Returns:
    sel (np.array, bool): mask of ``fr``.
    """
    return np.abs(fr - x) < tol_lw * LW

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

def test_suppress_sidelobes_single_resonance():
    """
    Check that a strong, isolated resonance gives spurious sidelobe
    candidates that ``suppress_sidelobes`` removes.
    """
    f, z = _sweep([F0], [0.95])
    fr, Qr, score = find_candidates_mf(f, z)
    # Without suppression: the resonance plus junk well away from it
    assert _near(fr, F0, 0.5).sum() == 1 and len(fr) > 5
    assert np.all(np.abs(fr[~_near(fr, F0, 0.5)] - F0) > 20 * LW)
    assert np.all(score[~_near(fr, F0, 0.5)] < score[_near(fr, F0, 0.5)])
    fr, Qr, score = find_candidates_mf(f, z, suppress_sidelobes = True)
    assert len(fr) == 1 and _near(fr, F0, 0.5).all()

@pytest.mark.parametrize("d_lw,amp_weak,kept", [
    (12., 0.3, True),    # separate weak resonance: clearly above sidelobe
    (30., 0.05, True),   # far and weak: kept
    (12., 0.1, False),   # weak dip at the strong one's sidelobe level
])
def test_suppress_sidelobes_weak_neighbor(d_lw, amp_weak, kept):
    """
    Check which weak neighbors of a strong resonance survive sidelobe
    suppression (all are found without it).
    """
    f_weak = F0 + d_lw * LW
    f, z = _sweep([F0, f_weak], [0.95, amp_weak])
    fr, _, _ = find_candidates_mf(f, z)
    assert _near(fr, F0, 0.5).sum() == 1 and _near(fr, f_weak, 1).any()
    fr, _, _ = find_candidates_mf(f, z, suppress_sidelobes = True)
    assert _near(fr, F0, 0.5).sum() == 1
    assert _near(fr, f_weak, 1).any() == kept
    # Everything else (the strong resonance's sidelobes) is gone
    assert len(fr) == 1 + kept

def test_find_candidates_mf_no_usable_scale():
    """
    Check that no candidates are returned when every Qr in ``Qr_range`` is
    narrower than the sampling allows, while the default range finds the
    resonance.
    """
    f = np.arange(1e9, 1.002e9, 1e3)
    f, z = _sweep([1.001e9], [0.8], f = f)
    fr, Qr, score = find_candidates_mf(f, z, Qr_range = (1e7, 1e8))
    assert len(fr) == len(Qr) == len(score) == 0
    fr, _, _ = find_candidates_mf(f, z, suppress_sidelobes = True)
    assert np.any(np.abs(fr - 1.001e9) < 0.5 * 1.001e9 / QR)

def test_find_candidates_mf_too_few_points():
    """
    Check that a sweep shorter than one usable chunk (64 points) gives no
    candidates.
    """
    f, z = _sweep([1.00005e9], [0.8], f = np.arange(1e9, 1.0001e9, 2e3))
    assert len(f) < 64
    out = find_candidates_mf(f, z)
    assert all(len(v) == 0 for v in out)

def test_find_candidates_mf_chunks_with_gap():
    """
    Check a sweep spanning several chunks with an empty chunk in between:
    resonances on both sides are found once each.
    """
    f = np.r_[np.arange(1e9, 1.01e9, 2e3), np.arange(1.25e9, 1.26e9, 2e3)]
    frs = [1.005e9, 1.255e9]
    f, z = _sweep(frs, [0.8, 0.8], f = f)
    fr, Qr, score = find_candidates_mf(f, z, suppress_sidelobes = True)
    assert len(fr) == 2 and np.all(np.diff(fr) > 0)
    np.testing.assert_allclose(fr, frs, atol = 0.5 * 1.255e9 / QR)
    assert np.all(score >= 6) and np.all((Qr >= 1e3) & (Qr <= 3e5))
