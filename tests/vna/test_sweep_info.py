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

################################################################################
############################## Sweep preparation ###############################
################################################################################
def test_prepare_full_sweep_decorrelates():
    """
    Check that noise correlated over several points is block averaged to
    the correlation length (and left alone with ``decorrelate = False`` or
    when the averaged step would be too coarse).
    """
    rng = np.random.default_rng(7)
    f = np.arange(1e9, 1.001e9, 100.)
    white = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    z = 1 + 1e-3 * np.convolve(white, np.ones(8) / 8, mode = 'same')
    fp, zp, k = si.prepare_full_sweep(f, z)
    assert k >= 4 and len(fp) == len(f) // k and len(zp) == len(fp)
    fp, zp, k2 = si.prepare_full_sweep(f, z, decorrelate = False)
    assert k2 == k and len(fp) == len(f)
    fp, _, _ = si.prepare_full_sweep(f, z, max_step = 100.)
    assert len(fp) == len(f)

def test_phase_problem_cases(full_sweep):
    """
    Check ``has_phase`` on empty data and ``phase_problem`` on a sweep with
    a long cable delay (usable, judged by winding alone), a fake phase
    (|S21| times a constant phase: lines, not circles), and a flat sweep
    with no dips (cannot judge: usable).
    """
    f, z, _, _ = full_sweep
    assert not si.has_phase(np.zeros(0, complex))
    delayed = z * np.exp(-2j * np.pi * (f - f[0]) * 1e-6)
    problem, info = si.phase_problem(delayed)
    assert problem is None and info['phase_span_rad'] > 20
    assert np.isnan(info['circularity'])
    problem, info = si.phase_problem(np.abs(z) * np.exp(0.3j))
    assert problem == 'fake' and info['n_dips'] >= 5
    assert info['circularity'] < 0.25
    flat = np.full(len(z), 0.01 * np.exp(0.3j))
    problem, info = si.phase_problem(flat)
    assert problem is None and info['n_dips'] == 0

################################################################################
############################ Sweep quality metrics #############################
################################################################################
def test_narrow_core_cannot_judge():
    """
    Check that ``narrow_core`` is False for non-finite or negative Qr and
    for resonances with too few points within a linewidth.
    """
    f = np.linspace(1e9 - 2e5, 1e9 + 2e5, 401)
    mag = np.abs(product_s21(f, [1e9], [2e4], [0.5], [0.]))
    mag[200] = 1e-3
    assert si.narrow_core(f, mag, 1e9, 2e4, 0.5, 0.)
    assert not si.narrow_core(f, mag, 1e9, np.nan, 0.5, 0.)
    assert not si.narrow_core(f, mag, 1e9, -2e4, 0.5, 0.)
    # Linewidth 1 kHz: a single point within +-1 linewidth
    assert not si.narrow_core(f, mag, 1e9, 1e6, 0.5, 0.)

@pytest.mark.parametrize("override,expected", [
    (dict(noise_db = 25.), 'noisy (S/N 25 dB)'),
    (dict(noise_db = 20.), 'very noisy (S/N 20 dB)'),
    (dict(ripple_db = 25.), 'large gain ripple (25.0 dB p-p)'),
    (dict(features_db = 15.), 'sharp gain features (15.0 dB p-p)'),
    (dict(gain_range_db = 60.), 'gain varies by 60 dB'),
    (dict(phase_jump_frac = 0.05),
     'phase jumps between points (5.0% of steps)'),
    (dict(corr_length = 4), 'oversampled: noise correlated over 4 points'),
    (dict(pts_per_lw = 3.), 'undersampled (3.0 points per linewidth)'),
    (dict(frac_poor_fit = 0.3), 'many poor fits (30%)'),
    (dict(frac_overlap = 0.6), 'very crowded (60% overlapping)'),
])
def test_sweep_notes_each(override, expected):
    """
    Check that each metric past its threshold gives exactly its note.
    """
    good = dict(noise_db = 45., ripple_db = 10., features_db = 5.,
                gain_range_db = 30., phase_jump_frac = 0., corr_length = 1,
                pts_per_lw = 20., frac_poor_fit = 0., frac_overlap = 0.)
    assert si.sweep_notes(good) == []
    assert si.sweep_notes(dict(good, **override)) == [expected]

def test_sweep_notes_narrow_and_thresholds():
    """
    Check the narrow-core note (at least max(5, 10%) of the resonances),
    threshold overrides, and missing or NaN metrics.
    """
    assert si.sweep_notes({}) == []
    assert si.sweep_notes(dict(noise_db = np.nan, pts_per_lw = np.nan)) == []
    notes = si.sweep_notes({}, n_narrow = 5, n_res = 30)
    assert notes == ['sweep distorted: 5 dips have a narrow core the '
                     'resonator model cannot fit']
    assert si.sweep_notes({}, n_narrow = 5, n_res = 100) == []
    assert si.sweep_notes({}, n_narrow = 4, n_res = 10) == []
    m = dict(noise_db = 30.)
    assert si.sweep_notes(m) == []
    assert si.sweep_notes(m, thresholds = dict(noise_db = 35.)) == [
        'noisy (S/N 30 dB)']

################################################################################
############################ Resonance arrangement #############################
################################################################################
@pytest.mark.parametrize("v,text", [
    (6e4, '60k'), (1.234e6, '1.2M'), (950., '950'), (12345., '12k'),
    (np.inf, 'inf'), (np.nan, 'n/a'),
])
def test_fmt_q(v, text):
    """
    Check the quality factor formatting.
    """
    assert si._fmt_q(v) == text

def test_describe_range_few():
    """
    Check ``describe_range`` with no and with few resonances.
    """
    text, info = si.describe_range([np.nan])
    assert text == 'none' and info['n_main'] == 0 and np.isnan(info['fmin'])
    text, info = si.describe_range([1e9, 1.5e9, 3e9])
    assert text == '1,000-3,000 MHz' and info['n_main'] == 3
    assert info['gaps'] == []

def _bands(*specs):
    """
    Build sorted resonance frequencies from evenly filled bands.

    Parameters:
    specs (tuple): (f_lo, f_hi, n) per band, Hz.

    Returns:
    fr (np.array): frequencies in Hz.
    """
    return np.sort(np.concatenate([np.linspace(a, b, n)
                                   for a, b, n in specs]))

@pytest.mark.parametrize("fr,text,counts,gaps", [
    # Two big bands far apart: one range with a gap
    (_bands((1e9, 1.5e9, 100), (3e9, 3.5e9, 100)),
     '1,000-3,500 MHz (gap 1,500-3,000 MHz)', (200, 0, 0, 0),
     [(1.5e9, 3e9)]),
    # A gap of 20 local spacings: too small to mention, bands merged
    (_bands((1e9, 1.5e9, 101), (1.6e9, 2.1e9, 101)),
     '1,000-2,100 MHz', (202, 0, 0, 0), []),
    # Three groups too small to be main bands: everything is reported
    (_bands((1e9, 1.003e9, 4), (2e9, 2.003e9, 4), (3e9, 3.003e9, 4)),
     '1,000-3,003 MHz (gap 1,003-2,000, 2,003-3,000 MHz)',
     (12, 0, 0, 0), [(1.003e9, 2e9), (2.003e9, 3e9)]),
    # Spread-out stragglers below the main band
    (_bands((2e9, 3e9, 200), (5e8, 1.1e9, 3)),
     '2,000-3,000 MHz + 3 low (500-1,100 MHz)', (200, 3, 0, 0), []),
    # A small group between two main bands
    (_bands((1e9, 1.5e9, 100), (2e9, 2.003e9, 3), (3e9, 3.5e9, 100)),
     '1,000-3,500 MHz (gap 1,500-3,000 MHz) + 3 around 2.0 GHz',
     (200, 0, 0, 3), [(1.5e9, 3e9)]),
])
def test_describe_range_cases(fr, text, counts, gaps):
    """
    Check gaps, merged bands, small groups and stragglers in
    ``describe_range``.
    """
    t, info = si.describe_range(fr)
    assert t == text
    assert (info['n_main'], info['n_low'], info['n_high'],
            info['n_between']) == counts
    np.testing.assert_allclose(np.reshape(info['gaps'], (-1, 2)),
                               np.reshape(gaps, (-1, 2)))

# Bank starts (log10 Hz) of a 16-bank device: 0.02 decades wide, period 0.035
S = np.log10(1e9) + 0.035 * np.arange(16)

def _banks(keep = range(16), extra = (), seed = 2):
    """
    Build a device of 50 resonances per bank (bank starts ``S``).

    Parameters:
    keep (iterable): indices of the banks present (of 16). Default all.
    extra (iterable): log10 frequencies of additional resonances. Default
        none.
    seed (int): random seed. Default 2.

    Returns:
    fr (np.array): frequencies in Hz.
    """
    rng = np.random.default_rng(seed)
    x = [S[k] + rng.uniform(0, 0.02, 50) for k in keep]
    return 10 ** np.concatenate(x + [np.asarray(extra, dtype = float)])

@pytest.mark.parametrize("keep,f_lo,f_hi,text,n_cut", [
    ([k for k in range(16) if k != 7], None, None,
     '15 banks, separated; 1 missing near 1.8 GHz', (0, 0)),
    (range(14), None, None, '14 banks, separated; 2 missing', (0, 0)),
    (range(14), 0.5e9, 10 ** (S[13] + 0.03),
     '14 banks, separated; 2 cut off at the high end', (0, 2)),
    (range(2, 16), 10 ** (S[2] - 0.01), 10 ** (S[15] + 0.1),
     '14 banks, separated; 2 cut off at the low end', (2, 0)),
    (range(1, 15), 10 ** (S[1] - 0.01), 10 ** (S[14] + 0.03),
     '14 banks, separated; 2 cut off', (0, 0)),
])
def test_describe_banks_missing(keep, f_lo, f_hi, text, n_cut):
    """
    Check banks missing inside the range, missing at the ends, and cut off
    by the swept range.
    """
    keep = list(keep)
    text_out, info = si.describe_banks(_banks(keep), f_lo, f_hi)
    assert text_out == text
    assert info['n_found'] == len(keep) == len(info['banks'])
    assert (info['n_cut_low'], info['n_cut_high']) == n_cut
    assert info['n_missing_inside'] == (0 if 7 in keep else 1)

def test_describe_banks_between():
    """
    Check that isolated resonances in the gaps between banks are counted as
    between banks.
    """
    fr = _banks(extra = S[:15] + 0.024)
    text, info = si.describe_banks(fr)
    assert text.startswith('16 banks, mostly separated (')
    assert info['n_between'] >= 15
    assert np.all(info['label'][-15:] == -1)

def test_describe_banks_overlap_and_small():
    """
    Check that resonances spread without periodic gaps give 'banks
    overlap' (NaN frequencies ignored), and that small devices get no
    description.
    """
    fr = 10 ** np.random.default_rng(0).uniform(9, 9.6, 800)
    fr[::100] = np.nan
    text, info = si.describe_banks(fr)
    assert text == 'banks overlap' and info['overlapping']
    assert info['label'] is None and info['banks'] is None
    text, info = si.describe_banks(_banks(range(8)))
    assert text == '' and info['n_found'] == 0

def test_find_banks_merges_and_degenerate():
    """
    Check that ``find_banks`` merges the least separated neighbors when it
    finds more banks than ``n_banks``, and returns no banks for too few
    resonances or without gaps.
    """
    fr = _banks()
    label, banks, info = si.find_banks(fr, n_banks = 8)
    assert len(banks) == 8 and info['slot'] == list(range(8))
    counts = np.array([b[2] for b in banks])
    # Each merged bank holds whole banks of 50 (the ends may lose a few)
    assert np.all(np.abs(counts / 50 - np.round(counts / 50)) < 0.2)
    assert counts.sum() > 0.95 * len(fr)
    assert np.all(np.diff([b[0] for b in banks]) > 0)
    for x in (fr[:20], np.geomspace(1e9, 2e9, 100)):
        label, banks, info = si.find_banks(x)
        assert banks == [] and np.all(label == -1)
        assert np.isnan(info['period'])

def _bank_res(qi_shape, qc_shape, flags = ''):
    """
    Build a fit table for a 16-bank device, with Qi and Qc following given
    shapes over the position in each bank and the bank index.

    Parameters:
    qi_shape, qc_shape (callable): (u, k) -> factor, with u the position
        in the bank (0-1) and k the bank index.
    flags (str): flags of every resonance. Default ''.

    Returns:
    f (np.array): the swept range [f_lo, f_hi] in Hz.
    res (pd.DataFrame): fit table.
    """
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 1, (16, 50))
    fr = 10 ** (S[:, None] + 0.02 * u).ravel()
    u, k = u.ravel(), np.repeat(np.arange(16), 50)
    scatter = lambda: 10 ** (0.04 * rng.standard_normal(len(fr)))
    qi = 1e5 * qi_shape(u, k) * scatter()
    qc = 3e4 * qc_shape(u, k) * scatter()
    res = pd.DataFrame(dict(fr = fr, Qi = qi, Qc = qc,
                            Qr = 1 / (1 / qi + 1 / qc), fit_status = 'ok',
                            flags = flags))
    return np.array([fr.min() * 0.999, fr.max() * 1.001]), res

FLAT = lambda u, k: 1 + 0 * u

@pytest.mark.parametrize("qc_shape,qc_text", [
    (lambda u, k: 10 ** u, 'within each bank rises from'),
    (lambda u, k: 10 ** -u, 'within each bank falls from'),
    (lambda u, k: 10 ** (1 - 4 * (u - 0.5) ** 2),
     'highest in the middle of each bank'),
    (lambda u, k: 10 ** (4 * (u - 0.5) ** 2), 'lowest in the middle of each'),
    (lambda u, k: 10 ** np.sin(4 * np.pi * u),
     'repeating pattern within each bank'),
])
def test_describe_full_sweep_bank_patterns(qc_shape, qc_text):
    """
    Check the bank line and the Q descriptions of ``describe_full_sweep``:
    Qi rising with frequency, Qc following a pattern within each bank.
    """
    f, res = _bank_res(lambda u, k: 10 ** (k / 15.), qc_shape)
    m = dict(noise_db = 40., gain_range_db = 10., ripple_db = 5.,
             features_db = 2., phase_jump_frac = 0.)
    text, info = si.describe_full_sweep(f, res, m, [])
    lines = text.split('\n')
    assert 'Banks: 16 banks, separated' in lines
    q_line = next(x for x in lines if x.startswith('Quality factors'))
    qi_text, rest = q_line.split('Qc~')
    assert 'rises with frequency' in qi_text
    assert qc_text in rest.split('Qr~')[0]
    assert info['q']['Qi']['frequency']['trend'] == 'rises'
    assert info['q']['Qc']['bank_position'] is not None
    # No pts_per_lw in the metrics: no sampling line
    assert not any(x.startswith('Sampling') for x in lines)
    assert info['banks']['n_found'] == 16 and 'label' not in info['banks']

def test_q_vs_bank_position_too_few_banks():
    """
    Check that a clear within-bank pattern is not reported when fewer than
    ``min_banks`` banks have enough reliable Q values.
    """
    _, res = _bank_res(FLAT, lambda u, k: 10 ** u)
    fr, qc = res.fr.values, res.Qc.values
    _, info = si.describe_banks(fr)
    label, banks = info['label'], info['banks']
    assert si.q_vs_bank_position(fr, qc, label, banks)['trend'] == 'rises'
    # Only 3 banks keep their Q values (the rest unreliable, or too few)
    q = np.where((label >= 0) & (label <= 2), qc, np.nan)
    i3 = np.flatnonzero(label == 3)[:5]
    q[i3] = qc[i3]
    assert si.q_vs_bank_position(fr, q, label, banks) is None
    q[label == 4] = qc[label == 4]
    assert si.q_vs_bank_position(fr, q, label, banks)['trend'] == 'rises'

def test_describe_full_sweep_varies_and_unreliable():
    """
    Check a Q that varies between banks without a trend, and the text when
    no fit is reliable.
    """
    f, res = _bank_res(lambda u, k: 10 ** (k % 2.), FLAT)
    text, info = si.describe_full_sweep(f, res, {}, [])
    assert '(varies across the band, ' in text.split('Qc~')[0]
    assert info['q']['Qi']['frequency']['trend'] is None
    assert info['q']['Qc'] == dict(frequency = None, bank_position = None)
    _, res = _bank_res(FLAT, FLAT, flags = 'poor fit')
    text, info = si.describe_full_sweep(f, res, {}, ['noisy (S/N 25 dB)'])
    assert ('Quality factors (reliable fits): Qi: none reliable, Qc: none '
            'reliable, Qr: none reliable') in text
    assert info['q'] == {}
    assert text.endswith('Notes: noisy (S/N 25 dB)')

################################################################################
############################ Bookkeeping functions #############################
################################################################################
def test_run_sweep_info(tmp_path, capsys):
    """
    Check that ``run_sweep_info`` prints the report of a saved fit (NaN
    metrics stored as None), and raises without one.
    """
    import zarr
    path = tmp_path / 'data.zarr'
    grp = zarr.open_group(str(path), mode = 'w')
    with pytest.raises(KeyError):
        si.run_sweep_info(grp)
    out = grp.create_group('full_sweep_fit')
    cols = dict(fr = [1.001e9, 1.002e9, 1.003e9], Qr = [1e4] * 3,
                Qi = [2e4] * 3, Qc = [2e4] * 3)
    for c, v in cols.items():
        out.create_array(c, data = np.asarray(v, dtype = np.float64))
    for c, v in dict(fit_status = ['ok', 'ok', 'failed'],
                     flags = ['', 'overlap', '']).items():
        arr = out.create_array(c, shape = (3,), dtype = str,
                               fill_value = '')
        arr[:] = np.array(v)
    out.create_group('gain').create_array(
        'f', data = np.linspace(1e9, 1.004e9, 11))
    out.attrs.update(dict(
        sweep_metrics = dict(noise_db = 40., gain_range_db = 3.,
                             ripple_db = 1., features_db = 0.5,
                             phase_jump_frac = 0., pts_per_lw = None),
        notes = ['gain varies by 60 dB']))
    report = si.run_sweep_info(str(path))
    assert capsys.readouterr().out.strip() == report
    lines = report.split('\n')
    assert lines[0] == 'Full VNA sweep: 3 resonances, 1,001-1,003 MHz'
    assert lines[1] == 'Fits: 2 ok, 1 failed (fr only)'
    assert lines[2] == 'Flags: 1 overlap'
    assert not any(x.startswith('Sampling') for x in lines)
    assert lines[-1] == 'Notes: gain varies by 60 dB'
