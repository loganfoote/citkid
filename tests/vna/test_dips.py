import numpy as np
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
