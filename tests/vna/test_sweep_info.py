import numpy as np
import pandas as pd
import pytest
from citkid.res.funcs import product_s21
from citkid.vna import sweep_info as si

def test_prepare_full_sweep_sorts_and_averages():
    f = np.array([3., 1., 2., 1.]) * 1e9
    z = np.array([3, 1, 2, 3], dtype = complex)
    fp, zp, _ = si.prepare_full_sweep(f, z, decorrelate = False)
    np.testing.assert_array_equal(fp, [1e9, 2e9, 3e9])
    np.testing.assert_array_equal(zp, [2, 2, 3])

def test_noise_correlation_length():
    rng = np.random.default_rng(1)
    white = rng.standard_normal(20000) + 1j * rng.standard_normal(20000)
    corr = np.convolve(white, np.ones(8) / 8, mode = 'same')
    assert si.noise_correlation_length(white) == 1
    assert si.noise_correlation_length(corr) >= 4

@pytest.mark.parametrize("make,expected", [
    (lambda z: z, None),
    (lambda z: np.abs(z) + 0j, 'missing'),
])
def test_phase_problem(full_sweep, make, expected):
    _, z, _, _ = full_sweep
    assert si.phase_problem(make(z))[0] == expected

def test_sweep_metrics_and_notes(full_sweep):
    f, z, gain, t = full_sweep
    res = pd.DataFrame(dict(fr = t['fr'], Qr = t['Qr'], snr = 300.,
                            flags = ''))
    m = si.sweep_metrics(f, z, gain, res)
    # Noise 2e-5 on a gain of ~0.01: S/N ~ 54 dB
    assert 50 < m['noise_db'] < 57
    assert m['phase_jump_frac'] < 1e-3
    assert 10 < m['pts_per_lw'] < 30
    assert si.sweep_notes(m) == []
    notes = si.sweep_notes(dict(m, noise_db = 20., ripple_db = 25.))
    assert any('very noisy' in n for n in notes)
    assert any('ripple' in n for n in notes)

def test_narrow_core():
    f = np.linspace(1e9 - 2e5, 1e9 + 2e5, 401)
    mag = np.abs(product_s21(f, [1e9], [2e4], [0.5], [0.]))
    assert not si.narrow_core(f, mag, 1e9, 2e4, 0.5, 0.)
    mag[200] = 1e-3  # a one-point dip far below the model
    assert si.narrow_core(f, mag, 1e9, 2e4, 0.5, 0.)

def test_describe_range():
    fr = np.r_[np.linspace(1e9, 2e9, 200), 3e9, 3.01e9]
    text, info = si.describe_range(fr)
    assert text.startswith('1,000-2,000 MHz') and '+ 2' in text
    assert info['n_main'] == 200 and info['n_high'] == 2

def test_describe_banks():
    # 16 banks of 50 resonances, spaced evenly in log frequency
    rng = np.random.default_rng(2)
    starts = np.log10(1e9) + 0.035 * np.arange(16)
    fr = 10 ** np.concatenate([s + rng.uniform(0, 0.02, 50) for s in starts])
    text, info = si.describe_banks(fr)
    assert text.startswith('16 banks, separated')
    assert info['n_found'] == 16
    assert (info['label'] >= 0).mean() > 0.95 and len(info['banks']) == 16

def _bank_device(rng, shape):
    """
    16 banks of 50 resonances with Qi = 1e5 x a shape over the position in
    each bank, times 10% scatter.
    """
    starts = np.log10(1e9) + 0.035 * np.arange(16)
    u = rng.uniform(0, 1, (16, 50))
    fr = 10 ** (starts[:, None] + 0.02 * u).ravel()
    q = 1e5 * shape(u.ravel(), np.repeat(np.arange(16), 50))
    return fr, q * 10 ** (0.04 * rng.standard_normal(len(q)))

@pytest.mark.parametrize("shape,trend", [
    (lambda u, k: 1 + 0 * u, 'none'),
    (lambda u, k: 10 ** u, 'rises'),
    (lambda u, k: 10 ** (1 - 4 * (u - 0.5) ** 2), 'peaks'),
    (lambda u, k: 10 ** (k / 15.), 'frequency'),
])
def test_q_variation(shape, trend):
    fr, q = _bank_device(np.random.default_rng(3), shape)
    _, info = si.describe_banks(fr)
    label, banks = info['label'], info['banks']
    vf = si.q_vs_frequency(fr, q, label, banks)
    vp = si.q_vs_bank_position(fr, q, label, banks)
    if trend == 'none':
        assert vf is None and vp is None
    elif trend == 'frequency':
        # Varies across the band, the same everywhere within a bank
        assert vf['trend'] == 'rises' and vp is None
        assert 0.7e5 < vf['low'] < 1.5e5 and 7e5 < vf['high'] < 1.5e6
    else:
        assert vf is None and vp['trend'] == trend
        # End bins are averaged over the first and last 1/8 of the bank
        assert vp['high'] / vp['low'] > 3

def test_describe_full_sweep(full_sweep):
    f, z, gain, t = full_sweep
    n = len(t['fr'])
    res = pd.DataFrame(dict(
        fr = t['fr'], Qr = t['Qr'], Qi = t['Qr'] / (1 - t['amp']),
        Qc = t['Qr'] / t['amp'], snr = 300., fit_status = 'ok',
        flags = [''] * (n - 2) + ['overlap', 'poor fit']))
    m = si.sweep_metrics(f, z, gain, res)
    text, _ = si.describe_full_sweep(f, res, m, si.sweep_notes(m))
    lines = text.split('\n')
    assert lines[0].startswith(f'Full VNA sweep: {n} resonances, 1,0')
    assert 'Fits: 30 ok' in text and '1 overlap, 1 poor fit' in text
    assert lines[-1] == 'Notes: none'
