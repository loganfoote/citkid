import importlib.util
import numpy as np
import pandas as pd
import pytest
from citkid.vna.res_finder_cnn import (
    accept_candidates, dip_only_resonances, find_resonances_cnn,
    merge_candidates, resolve_close_pairs
)

HAS_TORCH = importlib.util.find_spec('torch') is not None

@pytest.mark.skipif(HAS_TORCH, reason = 'PyTorch is installed')
def test_cnn_finder_without_torch(full_sweep):
    f, z, _, _ = full_sweep
    with pytest.raises(ImportError, match = 'environment-cnn.yml'):
        find_resonances_cnn(f, z)

def test_accept_candidates():
    f = np.arange(1e9, 1.001e9, 1e3)
    table = pd.DataFrame(dict(
        fr = 1.0005e9, Qr = [2e4, 2e4, 2e4, 2e4, 2e6, 2e4, 2e4],
        amp = [0.5, 0.01, 0.5, 0.5, 0.5, 0.999, 0.5],
        snr = [50, 50, 2, 50, 50, 50, 50], phi = [0, 0, 0, 0, 0, 1.5, 0],
        a_at_bound = [False, False, False, True, False, False, False],
        error = '', duplicate_of = [-1, -1, -1, -1, -1, -1, 0]))
    # ok; too shallow; too weak; Qr at a bound; narrower than a step;
    # pinned; duplicate
    assert list(accept_candidates(table, f)) == [True] + [False] * 6

def test_merge_candidates():
    fr_a, q_a = np.array([1e9, 1.001e9]), np.array([2e4, 2e4])
    fr_b = np.array([1e9 + 1e3, 1.0005e9])
    assert list(merge_candidates(fr_a, q_a, fr_b, [2e4, 2e4])) == [False,
                                                                   True]
    assert merge_candidates([], [], fr_b, [2e4, 2e4]).all()

def _fit_table(full_sweep):
    """Fit the true resonances of the fixture as a candidate table."""
    from citkid.vna.full_sweep import _refine, _sweep_snr
    f, z, _, t = full_sweep
    table = _sweep_snr(_refine(f, z, t['fr'], n_workers = 1)[0])
    table['accepted'] = accept_candidates(table, f)
    return f, z, t, table.reset_index(drop = True)

def test_resolve_close_pairs(full_sweep):
    f, z, t, table = _fit_table(full_sweep)
    # A duplicate of resonance 5, a tenth of a linewidth away
    dup = table.iloc[[5]].copy()
    dup['fr'] += 0.1 * dup.fr / dup.Qr
    table = pd.concat([table, dup], ignore_index = True)
    table = resolve_close_pairs(f, z, table)
    assert table.accepted.sum() == len(t['fr'])
    assert table.merged.sum() == 1

def test_dip_only_resonances(full_sweep):
    f, z, t, table = _fit_table(full_sweep)
    acc = table[table.accepted]
    # Leave resonance 7 out: its clear dip is reported
    keep = ~np.isclose(acc.fr.values, t['fr'][7], rtol = 1e-7, atol = 2e3)
    mag = np.abs(z / full_sweep[2])
    add, kind, capped = dip_only_resonances(
        f, mag, acc.fr.values[keep], acc.Qr.values[keep],
        table[np.r_[keep, np.zeros(len(table) - len(acc), bool)]])
    assert len(add) == 1 and abs(add[0] - t['fr'][7]) < 2e4
    assert kind[0] == 'dip' and not capped
