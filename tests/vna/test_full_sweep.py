import numpy as np
import pandas as pd
import pytest
import zarr
from citkid.vna import full_sweep as fs

def test_coupling_error_pair():
    k = fs.coupling_error([1e9, 1e9 + 1e6, np.nan], [2e4, 1e4, 1e4],
                          [0.8, 0.5, 0.5])
    np.testing.assert_allclose(k[0], 0.5 * (1e9 + 1e6) / 1e4 / 2e6)
    np.testing.assert_allclose(k[1], 0.8 * 1e9 / 2e4 / 2e6)
    assert np.isnan(k[2])

def test_reliability_flags():
    flags, misfit = fs.reliability_flags(
        coupling = [0.5, 0., 0., 0.], excess = [0., 100., 0., 0.],
        snr = [100., 100., 100., 100.],
        narrow = [False, False, True, False], a = [0., 0., 0., 0.9])
    assert list(flags) == ['overlap', 'poor fit', 'distorted', 'bifurcated']
    np.testing.assert_allclose(misfit[1], np.sqrt(200) / 100)

def test_fit_full_sweep(full_sweep):
    f, z, gain, t = full_sweep
    rng = np.random.default_rng(3)
    lw = t['fr'] / t['Qr']
    # Listed frequencies off by up to 0.3 linewidths, in reversed order,
    # plus one spot with no resonance
    fres = (t['fr'] + rng.uniform(-0.3, 0.3, len(lw)) * lw)[::-1]
    fres = np.r_[fres, 1.0195e9]
    res, g, info = fs.fit_full_sweep(f, z, fres)
    assert list(res.columns) == fs.FIT_COLUMNS
    np.testing.assert_array_equal(res.fres_input, fres)
    ok = res.fit_status.values[:-1] == 'ok'
    assert ok.mean() > 0.9
    fr_err = np.abs(res.fr.values[:-1] - t['fr'][::-1]) / lw[::-1]
    assert np.all(fr_err[ok] < 0.05)
    Qr_err = np.abs(res.Qr.values[:-1] / t['Qr'][::-1] - 1)
    assert np.median(Qr_err[ok]) < 0.01
    # The empty spot fails and keeps its listed frequency
    assert res.fit_status.iloc[-1].startswith('failed')
    assert res.fr.iloc[-1] == fres[-1] and np.isnan(res.Qr.iloc[-1])
    assert np.median(np.abs(g / gain - 1)) < 2e-3
    assert np.all(res.a.values[:-1][ok] == 0)

def test_fit_full_sweep_workers_identical(full_sweep):
    f, z, _, t = full_sweep
    a = fs.fit_full_sweep(f, z, t['fr'], n_workers = 1)[0]
    b = fs.fit_full_sweep(f, z, t['fr'], n_workers = 2)[0]
    for c in a.columns:
        if a[c].dtype.kind == 'f':
            np.testing.assert_array_equal(a[c].values, b[c].values)
        else:
            assert list(a[c]) == list(b[c])

def test_run_full_sweep_fit(full_sweep, tmp_path, capsys):
    f, z, gain, t = full_sweep
    path = str(tmp_path / 'sweep.zarr')
    grp = zarr.open_group(path, mode = 'a')
    with pytest.raises(KeyError):
        fs.run_full_sweep_fit(f, z, grp, progress = False)
    grp.create_array('fres_manual', data = t['fr'])
    res, report = fs.run_full_sweep_fit(f, z, grp, progress = False)
    assert report in capsys.readouterr().out
    out = grp['full_sweep_fit']
    for c in fs.FIT_COLUMNS:
        assert out[c].shape == (len(t['fr']),)
    assert out['fit_status'][0] == 'ok'
    assert out['gain/gain'].dtype == np.complex128
    assert out.attrs['report'] == report
    assert out.attrs['fit_nonlinear'] is False
    # The description can be reprinted from the saved results
    from citkid.vna.sweep_info import run_sweep_info
    assert run_sweep_info(grp) == report
    with pytest.raises(FileExistsError):
        fs.run_full_sweep_fit(f, z, grp, progress = False)
    fs.run_full_sweep_fit(f, z, path, overwrite = True, progress = False)

################################################################################
############################ Branch coverage tests #############################
################################################################################
FA = 1.0015e9  # resonance frequency used by the small sweeps below
LW = FA / 2e4  # its linewidth (Qr 2e4), ~50 kHz

def _sweep(fr, amp, Qr = 2e4, span = 4e6, noise = 2e-5, seed = 0):
    """
    Simulate a small VNA sweep (1 kHz steps from 1 GHz) with cable delay.

    Parameters:
    fr (list): resonance frequencies in Hz.
    amp (list): resonance amplitudes, same length as ``fr``.
    Qr (float): quality factor of every resonance. Default 2e4.
    span (float): sweep span in Hz. Default 4 MHz.
    noise (float): white noise per quadrature. Default 2e-5.
    seed (int): random seed. Default 0.

    Returns:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21.
    """
    from citkid.res.funcs import product_s21
    rng = np.random.default_rng(seed)
    f = np.arange(1e9, 1e9 + span, 1e3)
    n = len(fr)
    z = (0.01 * np.exp(-2j * np.pi * (f - 1e9) * 40e-9)
         * product_s21(f, np.array(fr, float), np.full(n, Qr),
                       np.array(amp, float), np.zeros(n)))
    z = z + noise * (rng.standard_normal(len(f))
                     + 1j * rng.standard_normal(len(f)))
    return f, z

def _state(fr, q = 2e4, amp = 0.6):
    """
    Build a ``_stage_a`` fit state for guesses sorted by frequency.

    Parameters:
    fr (list): guesses in Hz, ascending.
    q (float): window Qr of every guess. Default 2e4.
    amp (float): amp of every guess (used for external factors). Default
        0.6.

    Returns:
    st (dict): fit state.
    """
    n = len(fr)
    return dict(fr = np.array(fr, float), q_win = np.full(n, q),
                amp = np.full(n, amp), phi = np.zeros(n),
                active = np.ones(n, bool), bounds = [1e3, 1e7], n_lw = 8.,
                ext_lw = 30., max_pts_lw = 10., max_pts_lw_final = 20.)

_OPTS = dict(add_poles = False, fit_nonlinear = False, downward = True,
             dup_lw = 0.1)

def _joint_at_bound(monkeypatch):
    """
    Make every joint (multi-member) Stage A fit report a Qr at a bound.

    Parameters:
    monkeypatch (pytest.MonkeyPatch): fixture used to patch ``_stage_a``.

    Returns:
    calls (list): sizes of the groups passed to ``_stage_a``.
    """
    real, calls = fs._stage_a, []
    def stage_a(f, z, st, members, Qr_init, screen = False):
        """
        Run the real ``_stage_a``, flagging joint fits as at a bound.
        """
        calls.append(len(members))
        out = real(f, z, st, members, Qr_init, screen = screen)
        if len(members) > 1:
            res = dict(out[2], at_bound = np.ones(len(out[2]['fr']), bool))
            out = out[:2] + (res,) + out[3:]
        return out
    monkeypatch.setattr(fs, '_stage_a', stage_a)
    return calls

def _boom(*args, **kwargs):
    """
    Raise a RuntimeError (stands in for a failing fit stage).
    """
    raise RuntimeError('boom')

def _table(rows):
    """
    Build a minimal ``fit_from_guesses`` table for ``recover_drifted``.

    Parameters:
    rows (list of dict): rows; missing columns get defaults (a clean fit).

    Returns:
    df (pd.DataFrame): the table.
    """
    d = dict(group = 0, Qr = 2e4, amp = 0.6, phi = 0., duplicate_of = -1,
             error = '', recovered = False)
    return pd.DataFrame([dict(d, **r) for r in rows])

def _fake_overlap(monkeypatch, snr_mult):
    """
    Make the final fit of the second guess land on the first resonance
    (1.001 GHz), as if it had converged onto its neighbor.

    Parameters:
    monkeypatch (pytest.MonkeyPatch): fixture used to patch ``_final_unit``.
    snr_mult (float): factor applied to the moved fit's SNR.
    """
    real = fs._final_unit
    def final_unit(f, z, st, g, opts):
        """
        Run the real ``_final_unit``, moving the fit of sorted index 1.
        """
        out = real(f, z, st, g, opts)
        for rows, _ in out:
            for r in rows:
                if r['sorted_idx'] == 1:
                    r['fr'] = 1.001e9 + 1.
                    r['snr'] = r['snr'] * snr_mult
        return out
    monkeypatch.setattr(fs, '_final_unit', final_unit)

def test_coupling_error_single():
    """Check that a lone resonance has zero coupling error."""
    k = fs.coupling_error([1e9, np.nan], [1e4, 1e4], [0.5, 0.5])
    assert k[0] == 0 and np.isnan(k[1])

def test_residual_excess_short_window():
    """Check that a window with too few points gives NaN."""
    f = 1e9 + 1e3 * np.arange(10)
    ex = fs.residual_excess(f, np.ones(10, complex), np.array([1e9 + 5e3]),
                            np.array([1e3]), np.array([0.5]),
                            np.array([0.]), np.array([1e-3]))
    assert np.isnan(ex[0])

def test_final_unit_joint():
    """Check that _final_unit fits overlapping members jointly."""
    f, z = _sweep([FA, FA + 2 * LW], [0.6, 0.6])
    out = fs._final_unit(f, z, _state([FA, FA + 2 * LW]), np.array([0, 1]),
                         _OPTS)
    assert len(out) == 1
    rows, dropped = out[0]
    assert [r['sorted_idx'] for r in rows] == [0, 1] and dropped == []
    np.testing.assert_allclose([r['fr'] for r in rows],
                               [FA, FA + 2 * LW], atol = 0.02 * LW)

def test_final_unit_redone_per_member_at_bound(monkeypatch):
    """Check that a joint fit at a bound is redone one member at a time."""
    calls = _joint_at_bound(monkeypatch)
    # 6 linewidths apart: the +-8 linewidth windows overlap
    f, z = _sweep([FA, FA + 6 * LW], [0.6, 0.6])
    out = fs._final_unit(f, z, _state([FA, FA + 6 * LW]), np.array([0, 1]),
                         _OPTS)
    assert calls == [2, 1, 1]
    assert len(out) == 2
    for k, (rows, dropped) in enumerate(out):
        assert len(rows) == 1 and dropped == []
        r = rows[0]
        assert r['sorted_idx'] == k and r['error'] == ''
        assert abs(r['fr'] - (FA + 6 * k * LW)) < 0.05 * LW
        assert abs(r['Qr'] / 2e4 - 1) < 0.05

def test_final_unit_stage_a_raises(monkeypatch):
    """Check that failing fits give one error row per member."""
    monkeypatch.setattr(fs, '_stage_a', _boom)
    f, z = _sweep([FA, FA + 2 * LW], [0.6, 0.6])
    out = fs._final_unit(f, z, _state([FA, FA + 2 * LW]), np.array([0, 1]),
                         _OPTS)
    assert [[r['sorted_idx'] for r in rows] for rows, _ in out] == [[0], [1]]
    for rows, dropped in out:
        assert rows[0]['error'] == "RuntimeError('boom')" and dropped == []
        assert 'fr' not in rows[0]

def test_final_unit_stage_b_fallback(monkeypatch):
    """Check that a failed Stage B falls back to the Stage A values."""
    real = fs.fit_resonators
    monkeypatch.setattr(fs, 'fit_resonators', lambda *a, **k: dict(
        real(*a, **k), success = False))
    f, z = _sweep([FA], [0.6])
    (rows, dropped), = fs._final_unit(f, z, _state([FA]), np.array([0]),
                                      _OPTS)
    r, = rows
    assert not r['b_ok'] and dropped == []
    assert r['fr'] == r['a_fr'] and r['Qr'] == r['a_Qr']
    assert r['amp'] == r['a_amp']
    assert np.isnan(r['amp_err']) and np.isnan(r['phi_err'])
    assert r['nl_a'] == 0 and abs(r['fr'] - FA) < 0.05 * LW

@pytest.mark.parametrize('d', [1e3, 1e4])
def test_final_unit_add_poles_drops_duplicate_guess(d):
    """Check that a redundant guess is dropped, pointing to the kept one."""
    f, z = _sweep([FA], [0.6])
    opts = dict(_OPTS, add_poles = True)
    (rows, dropped), = fs._final_unit(f, z, _state([FA, FA + d]),
                                      np.array([0, 1]), opts)
    assert len(rows) == 1 and len(dropped) == 1
    kept = rows[0]['sorted_idx']
    assert dropped == [(1 - kept, kept)]
    assert abs(rows[0]['fr'] - FA) < 0.05 * LW

def test_final_unit_add_poles_finds_missing():
    """Check that add_poles adds an unlisted resonance (index -1)."""
    f, z = _sweep([FA, FA + 3 * LW], [0.6, 0.6])
    opts = dict(_OPTS, add_poles = True)
    (rows, dropped), = fs._final_unit(f, z, _state([FA]), np.array([0]),
                                      opts)
    assert [r['sorted_idx'] for r in rows] == [0, -1] and dropped == []
    assert abs(rows[1]['fr'] - (FA + 3 * LW)) < 0.05 * LW
    assert all(r['b_ok'] for r in rows)

def test_final_unit_add_poles_retry_without_added(monkeypatch):
    """Check that a failed Stage B with added poles is retried without."""
    real, n_poles = fs.fit_resonators, []
    def fit_resonators(f, z, fr0, *args, **kwargs):
        """
        Run the real ``fit_resonators``, failing with more than one pole.
        """
        n_poles.append(len(fr0))
        b = real(f, z, fr0, *args, **kwargs)
        return dict(b, success = len(fr0) == 1)
    monkeypatch.setattr(fs, 'fit_resonators', fit_resonators)
    f, z = _sweep([FA, FA + 3 * LW], [0.6, 0.6])
    opts = dict(_OPTS, add_poles = True)
    (rows, dropped), = fs._final_unit(f, z, _state([FA]), np.array([0]),
                                      opts)
    assert n_poles == [2, 1]
    assert [r['sorted_idx'] for r in rows] == [0]
    # Without the unlisted neighbor in the model, fr is pulled a little
    assert rows[0]['b_ok'] and abs(rows[0]['fr'] - FA) < 0.5 * LW

def test_refine_split_large_cluster():
    """Check that clusters above max_group are split into separate fits."""
    fr = [FA, FA + 2 * LW, FA + 4 * LW]
    f, z = _sweep(fr, [0.6] * 3)
    joint = fs.fit_from_guesses(f, z, fr)[0]
    assert joint.group.nunique() == 1
    df = fs.fit_from_guesses(f, z, fr, max_group = 1)[0]
    assert df.group.nunique() == 3
    assert (df.error == '').all() and df.b_ok.all()
    np.testing.assert_allclose(df.sort_values('guess_idx').fr, fr,
                               atol = 0.05 * LW)

def test_refine_joint_failure_fits_members(monkeypatch):
    """Check that failed joint Stage A fits are redone per member."""
    calls = _joint_at_bound(monkeypatch)
    fr = [FA, FA + 6 * LW]
    f, z = _sweep(fr, [0.6, 0.6])
    df = fs.fit_from_guesses(f, z, fr)[0]
    assert 2 in calls
    assert df.group.nunique() == 2 and (df.error == '').all()
    np.testing.assert_allclose(df.sort_values('guess_idx').fr, fr,
                               atol = 0.05 * LW)

def test_fit_full_sweep_stage_a_raises(monkeypatch):
    """Check that fits that raise give 'failed: <error>' rows."""
    monkeypatch.setattr(fs, '_stage_a', _boom)
    fr = [FA, FA + 2 * LW]
    f, z = _sweep(fr, [0.6, 0.6])
    res = fs.fit_full_sweep(f, z, fr)[0]
    assert list(res.fit_status) == ["failed: RuntimeError('boom')"] * 2
    np.testing.assert_array_equal(res.fr, fr)
    assert res.Qr.isna().all()

def test_refine_duplicate_guesses():
    """Check that two guesses of one resonance give one fit and a link."""
    fr = [1.001e9, 1.002e9, 1.002e9 + 5e3, 1.003e9]
    f, z = _sweep([1.001e9, 1.002e9, 1.003e9], [0.6] * 3)
    df = fs.fit_from_guesses(f, z, fr, max_group = 1)[0]
    dup = df[df.guess_idx == 2].iloc[0]
    assert dup.duplicate_of == 1 and np.isnan(dup.fr) and dup.error == ''
    res = fs.fit_full_sweep(f, z, fr, max_group = 1)[0]
    assert list(res.fit_status) == ['ok', 'ok',
                                    'failed: merged with a neighbor', 'ok']

def test_refine_add_poles_links_redundant_guess():
    """Check that a guess removed as redundant links to the kept one."""
    fr = [1.001e9, 1.002e9, 1.002e9 + 2e4, 1.003e9]
    f, z = _sweep([1.001e9, 1.002e9, 1.003e9], [0.6] * 3)
    df = fs.fit_from_guesses(f, z, fr, add_poles = True)[0]
    assert len(df) == 4 and (df.guess_idx >= 0).all()
    dup = df[df.duplicate_of >= 0]
    assert len(dup) == 1 and np.isnan(dup.fr.iloc[0])
    pair = {int(dup.guess_idx.iloc[0]), int(dup.duplicate_of.iloc[0])}
    assert pair == {1, 2}
    kept = df[df.guess_idx == dup.duplicate_of.iloc[0]].iloc[0]
    assert abs(kept.fr - 1.002e9) < 0.05 * LW and kept.b_ok

def test_fit_full_sweep_empty_spot_pinned():
    """Check that a listed spot with no resonance ends 'Qr at a bound'."""
    f, z = _sweep([1.001e9, 1.002e9, 1.003e9], [0.6] * 3)
    df = fs.fit_from_guesses(f, z, [1.001e9, 1.0025e9])[0]
    r = df[df.guess_idx == 1].iloc[0]
    assert r.error == 'Qr at a bound' and np.isnan(r.fr)
    res = fs.fit_full_sweep(f, z, [1.001e9, 1.0025e9])[0]
    assert list(res.fit_status) == ['ok', 'failed: Qr at a bound']
    assert res.fr.iloc[1] == 1.0025e9 and res.group.iloc[1] == -1

@pytest.mark.parametrize('snr_mult, loser', [(1e-3, 1), (1e3, 0)])
def test_refine_duplicate_fit_keeps_higher_snr(monkeypatch, snr_mult, loser):
    """Check that of two fits of one resonance the lower SNR is dropped."""
    _fake_overlap(monkeypatch, snr_mult)
    fr = [1.001e9, 1.002e9]
    f, z = _sweep(fr, [0.6, 0.6])
    df = fs.fit_from_guesses(f, z, fr)[0].set_index('guess_idx')
    assert df.error[loser] == 'duplicate fit'
    assert df.error[1 - loser] == ''

def test_fit_full_sweep_recovers_drifted(monkeypatch):
    """Check that a resonance whose fit drifted away is recovered."""
    _fake_overlap(monkeypatch, 1e-3)
    fr = [1.001e9, 1.002e9]
    f, z = _sweep(fr, [0.6, 0.6])
    res, _, info = fs.fit_full_sweep(f, z, fr)
    assert info['n_recovered'] == 1
    assert list(res.fit_status) == ['ok', 'stage_a']
    np.testing.assert_allclose(res.fr, fr, atol = 0.05 * LW)
    np.testing.assert_allclose(res.Qr, 2e4, rtol = 0.05)
    np.testing.assert_allclose(res.amp, 0.6, rtol = 0.05)
    assert res.group.iloc[0] != res.group.iloc[1]

def test_recover_drifted_onto_neighbor():
    """Check recovery of a weak resonance whose fit moved to a neighbor."""
    fb = FA + 1.5 * LW
    f, z = _sweep([FA, fb], [0.6, 0.3])
    df = _table([
        dict(guess_idx = 0, fr0 = FA, fr = FA),
        dict(guess_idx = 1, fr0 = fb, fr = FA + 100., group = 1,
             error = 'duplicate fit')])
    out = fs.recover_drifted(f, z, df, np.full(2, LW))
    assert len(out) == 3
    # The drifted fit is kept but no longer tied to the listed frequency
    assert out.guess_idx.iloc[1] == -1 and out.fr.iloc[1] == FA + 100.
    r = out[out.guess_idx == 1].iloc[0]
    assert r.recovered and not r.b_ok and r.group == 2 and r.error == ''
    assert abs(r.fr - fb) < 0.05 * LW
    assert abs(r.Qr / 2e4 - 1) < 0.1 and abs(r.amp - 0.3) < 0.03
    assert r.a_fr == r.fr and np.isfinite(r.snr) and r.snr > 100

def test_recover_drifted_merged_no_fit_inside():
    """Check recovery of a merged guess with no other fit in its window."""
    fb = FA + 20 * LW
    f, z = _sweep([FA, fb], [0.6, 0.6])
    df = _table([
        dict(guess_idx = 0, fr0 = FA, fr = FA),
        dict(guess_idx = 1, fr0 = fb, fr = np.nan, Qr = np.nan,
             amp = np.nan, phi = np.nan, group = np.nan, duplicate_of = 0)])
    out = fs.recover_drifted(f, z, df, np.full(2, LW))
    # The merged row is replaced by the recovered fit
    assert len(out) == 2 and list(out.guess_idx) == [0, 1]
    r = out.iloc[1]
    assert r.recovered and r.duplicate_of == -1 and r.group == 1
    assert abs(r.fr - fb) < 0.05 * LW and abs(r.Qr / 2e4 - 1) < 0.05

def test_recover_drifted_skips():
    """Check the cases where no recovery is attempted or kept."""
    f, z = _sweep([FA], [0.6])
    df = _table([
        dict(guess_idx = 0, fr0 = FA, fr = FA),
        # no resonance at the listed frequency: refit rejected
        dict(guess_idx = 1, fr0 = FA + 2 * LW, fr = FA + 50.,
             error = 'duplicate fit'),
        # no step-1 linewidth
        dict(guess_idx = 7, fr0 = FA + 4 * LW, fr = FA),
        # failed for another reason
        dict(guess_idx = 2, fr0 = FA + 4 * LW, fr = np.nan,
             error = 'Qr at a bound'),
        # did not move
        dict(guess_idx = 3, fr0 = FA + 0.2 * LW, fr = FA + 0.1 * LW,
             error = 'duplicate fit'),
        # moved, but another fit covers the listed frequency
        dict(guess_idx = 4, fr0 = FA + 0.3 * LW, fr = FA + 1e6),
        # window outside the sweep
        dict(guess_idx = 5, fr0 = 1.01e9, fr = FA, error = 'duplicate fit')])
    out = fs.recover_drifted(f, z, df, np.full(6, LW))
    assert out is df

def test_fit_full_sweep_status_mapping(monkeypatch):
    """Check how each kind of fit_from_guesses row maps to fit_status."""
    fr = 1.001e9
    f, z = _sweep([fr], [0.6])
    base = dict(group = 0, b_ok = True, a_at_bound = False, fr = fr,
                Qr = 2e4, amp = 0.6, phi = 0., fr_err = 1., Qr_err = 10.,
                amp_err = 0.01, phi_err = 0.01, nl_a = 0., nl_a_err = np.nan,
                nrmse = 0.01, snr = 300., sigma_noise = 2e-5,
                duplicate_of = -1, error = '', recovered = False)
    nan = dict(fr = np.nan, Qr = np.nan, amp = np.nan, group = np.nan)
    rows = [dict(base, **dict(nan, error = 'Qr at a bound')),
            dict(base, **dict(nan, duplicate_of = 0)),
            dict(base, **nan),
            dict(base, a_at_bound = True),
            dict(base, amp = 1.5),
            dict(base, fr = fr + 5 * LW),
            dict(base, b_ok = False),
            dict(base)]
    table = pd.DataFrame([dict(r, guess_idx = k) for k, r in
                          enumerate(rows)])
    fres = np.full(len(rows) + 1, fr)
    monkeypatch.setattr(fs, 'fit_from_guesses',
                        lambda *a, **k: (table, np.full(9, LW), (1e2, 1e7)))
    monkeypatch.setattr(fs, 'recover_drifted', lambda f, z, t, *a, **k: t)
    res = fs.fit_full_sweep(f, z, fres)[0]
    assert list(res.fit_status) == [
        'failed: Qr at a bound', 'failed: merged with a neighbor',
        'failed: not fit', 'failed: Qr at a bound', 'failed: unphysical amp',
        'failed: fit moved to another resonance', 'stage_a', 'ok',
        'failed: not fit']
    np.testing.assert_array_equal(res.fr, fres)
    assert res.Qr.isna().sum() == 7 and (res.Qr.iloc[6:8] == 2e4).all()

def test_fit_full_sweep_empty():
    """Check that an empty list gives an empty table and a gain."""
    f, z = _sweep([FA], [0.6])
    res, gain, info = fs.fit_full_sweep(f, z, [])
    assert len(res) == 0 and list(res.columns) == fs.FIT_COLUMNS
    assert gain.shape == f.shape and info['n_recovered'] == 0

@pytest.mark.xfail(strict = True, reason = (
    "res.multifit.init_qr stops its scale search early and returns Qr ~10x "
    "too high for this resonance; with one guess the Qr floor (0.2 x that) "
    "is above the true Qr, so it fails with 'Qr at a bound'. A fix was "
    "tried and rejected: it made real sweeps worse (see the init_qr "
    "docstring Notes)"))
def test_fit_full_sweep_single_deep_resonance():
    """Check that one clean, deep resonance listed alone is fit."""
    f, z = _sweep([1.002e9], [0.9])
    res = fs.fit_full_sweep(f, z, [1.002e9])[0]
    assert res.fit_status.iloc[0] == 'ok'
    assert abs(res.Qr.iloc[0] / 2e4 - 1) < 0.05

def test_sweep_snr_edge_cases():
    """
    Check rescale_snr_to_sweep with no usable rows and with no healthy
    group.
    """
    t = pd.DataFrame(dict(fr = [np.nan], sigma_noise = [np.nan],
                          snr = [1.], amp = [0.5], a_at_bound = [False],
                          group = [0]))
    out = fs.rescale_snr_to_sweep(t)
    assert np.isnan(out.sigma_sweep.iloc[0]) and out.snr.iloc[0] == 1.
    # Every group unhealthy (amp > 1.2): all groups are used anyway
    t = pd.DataFrame(dict(fr = [1., 2., 3.], sigma_noise = [1., 5., 3.],
                          snr = [10.] * 3, amp = [2.] * 3,
                          a_at_bound = [False] * 3, group = [0, 1, 2]))
    out = fs.rescale_snr_to_sweep(t)
    np.testing.assert_allclose(out.sigma_sweep, [1., 3., 3.])
    np.testing.assert_allclose(out.snr, [10., 50 / 3, 10.])

def test_run_full_sweep_fit_no_phase(tmp_path):
    """Check that a sweep with no phase is refused."""
    f, z = _sweep([FA], [0.6])
    grp = zarr.open_group(str(tmp_path / 'sweep.zarr'), mode = 'a')
    grp.create_array('fres_manual', data = np.array([FA]))
    with pytest.raises(ValueError, match = 'no usable phase'):
        fs.run_full_sweep_fit(f, np.abs(z), grp, progress = False)
    assert 'full_sweep_fit' not in grp
