import numpy as np
import pytest
from citkid.res.funcs import product_s21
from citkid.vna import dips

def _mag(rng, fr, Qr, amp, sigma = 2e-3):
    f = np.arange(1e9, 1.01e9, 1e3)
    mag = np.abs(product_s21(f, fr, Qr, amp, np.zeros(len(fr))))
    return f, mag + sigma * rng.standard_normal(len(f))

def test_dip_significance():
    rng = np.random.default_rng(0)
    f, mag = _mag(rng, [1.005e9], [2e4], [0.5])
    sig, depth = dips.dip_significance(f, mag, 1.005e9, 1.005e9 / 2e4)
    # Mean over +-0.5 lw, so shallower than the 6 dB minimum
    assert sig > 50 and 3 < depth < 6
    # Noise only: no significant dip
    flat = 1 + 2e-3 * rng.standard_normal(len(f))
    sig_noise, _ = dips.dip_significance(f, flat, 1.005e9, 1.005e9 / 2e4)
    assert abs(sig_noise) < 5

def test_find_dips():
    rng = np.random.default_rng(1)
    fr = np.array([1.002e9, 1.005e9, 1.008e9])
    f, mag = _mag(rng, fr, [2e4] * 3, [0.6] * 3)
    fd, prom, lw = dips.find_dips(f, mag)
    assert len(fd) == 3 and np.all(np.abs(fd - fr) < 2e4)
    assert 2e4 < lw < 1e5 and np.all(prom > 5)

def test_unexplained_dips():
    rng = np.random.default_rng(2)
    fr = np.array([1.002e9, 1.005e9, 1.008e9])
    f, mag = _mag(rng, fr, [2e4] * 3, [0.6] * 3)
    fd, _ = dips.unexplained_dips(f, mag, fr[:2], [2e4] * 2)
    assert len(fd) == 1 and abs(fd[0] - fr[2]) < 2e4

def test_window_means():
    """
    Check window means, and that data shorter than a window gives none.
    """
    np.testing.assert_allclose(dips._window_means(np.arange(6.), 2, 2),
                               [0.5, 2.5, 4.5])
    assert len(dips._window_means(np.arange(3.), 5, 1)) == 0

@pytest.mark.parametrize("case", ['no_core', 'few_windows', 'zero_dip'])
def test_dip_significance_unmeasurable(case):
    """
    Check that ``dip_significance`` gives NaN when the dip has no points,
    there are too few comparison windows, or the dip mean is zero.
    """
    rng = np.random.default_rng(3)
    f, mag = _mag(rng, [1.005e9], [2e4], [0.5])
    fr, lw, kw = 1.005e9, 1.005e9 / 2e4, {}
    if case == 'no_core':
        fr, lw = 1.005e9 + 500, 10.  # narrower than the 1 kHz step
    elif case == 'few_windows':
        kw = dict(min_windows = 1000)
    else:
        mag[np.abs(f - fr) < lw] = 0.
    sig, depth = dips.dip_significance(f, mag, fr, lw, **kw)
    assert np.isnan(sig) and np.isnan(depth)

def test_dip_significance_exclude():
    """
    Check that ``exclude`` removes other resonances from the comparison
    region but never the dip's own resonance.
    """
    rng = np.random.default_rng(4)
    lw = 1.005e9 / 2e4
    frs = [1.005e9, 1.005e9 + 8 * lw]
    f, mag = _mag(rng, frs, [2e4] * 2, [0.5] * 2)
    lws = [lw, lw]
    sig0, d0 = dips.dip_significance(f, mag, frs[0], lw, exclude_lw = 5.)
    # Its own entry in exclude is skipped (else +-5 lw would be cut)
    sig1, d1 = dips.dip_significance(f, mag, frs[0], lw, exclude_lw = 5.,
                                     exclude = ([frs[0]], [lw]))
    assert (sig1, d1) == (sig0, d0)
    # The neighbor's dip no longer pollutes the comparison windows
    sig2, d2 = dips.dip_significance(f, mag, frs[0], lw, exclude_lw = 5.,
                                     exclude = (frs, lws))
    assert sig2 > 2 * sig0 and d2 > d0

@pytest.mark.parametrize("return_widths", [False, True])
def test_find_dips_nothing(return_widths):
    """
    Check that ``find_dips`` returns empty results (and NaN linewidth) for
    too short a sweep and for one without clear dips.
    """
    rng = np.random.default_rng(5)
    f = np.arange(1e9, 1.01e9, 1e3)
    flat = 1 + 1e-3 * rng.standard_normal(len(f))
    for ff, mm in ((f[:5], flat[:5]), (f, flat)):
        out = dips.find_dips(ff, mm, return_widths = return_widths)
        assert len(out) == (4 if return_widths else 3)
        assert len(out[0]) == len(out[1]) == 0 and np.isnan(out[2])
        if return_widths:
            assert len(out[3]) == 0

@pytest.mark.parametrize("n_known", [0, 1])
def test_unexplained_dips_few_resonances(n_known):
    """
    Check ``unexplained_dips`` with no known resonances (all dips are
    unexplained; the linewidth is estimated) and with a single one.
    """
    rng = np.random.default_rng(2)
    fr = np.array([1.002e9, 1.005e9, 1.008e9])
    f, mag = _mag(rng, fr, [2e4] * 3, [0.6] * 3)
    fd, prom = dips.unexplained_dips(f, mag, fr[:n_known], [2e4] * n_known)
    assert len(fd) == 3 - n_known and np.all(prom > 3)
    np.testing.assert_allclose(fd, fr[n_known:], atol = 2e4)

def test_unexplained_dips_no_dips():
    """
    Check that a flat sweep has no unexplained dips, with ``return_info``.
    """
    rng = np.random.default_rng(6)
    f = np.arange(1e9, 1.01e9, 1e3)
    mag = 1 + 2e-3 * rng.standard_normal(len(f))
    fd, prom, info = dips.unexplained_dips(f, mag, [1.005e9], [2e4],
                                           return_info = True)
    assert len(fd) == len(prom) == 0
    assert info == dict(n_strong = 0, capped = False)

@pytest.mark.parametrize("strong_db,max_strong,n_kept,capped", [
    (None, None, 0, False),   # not significant enough: dropped
    (3., None, 2, False),     # prominent enough: kept anyway
    (3., 1, 0, True),         # more strong-only dips than allowed: none
    (3., 2, 2, False),
    (50., None, 0, False),    # not that prominent
])
def test_unexplained_dips_strong(strong_db, max_strong, n_kept, capped):
    """
    Check ``strong_db`` and ``max_strong`` on dips that fail the (here
    impossibly strict) significance test.
    """
    rng = np.random.default_rng(2)
    fr = np.array([1.002e9, 1.005e9, 1.008e9])
    f, mag = _mag(rng, fr, [2e4] * 3, [0.6] * 3)
    fd, prom, info = dips.unexplained_dips(
        f, mag, fr[:1], [2e4], min_sig = np.inf, strong_db = strong_db,
        max_strong = max_strong, return_info = True)
    assert len(fd) == n_kept and info['capped'] == capped
    assert info['n_strong'] == n_kept
    if n_kept:
        np.testing.assert_allclose(fd, fr[1:], atol = 2e4)
