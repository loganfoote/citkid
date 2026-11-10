import numpy as np
import pytest
from citkid.tone_plan import plan_tone_sets, select_isolated

LW = 5e4  # linewidth of a 1 GHz resonator at Qr = 2e4

################################################################################
############################## select_isolated #################################
################################################################################
def test_select_isolated_drop_both():
    """
    Check that both resonators of a close pair are dropped by default, and
    the distances are in linewidths.
    """
    fr = np.array([1e9, 1e9 + 3 * LW, 1.1e9])
    keep, near = select_isolated(fr, np.full(3, LW))
    np.testing.assert_array_equal(keep, [False, False, True])
    np.testing.assert_allclose(near[:2], [3, 3])
    assert near[2] > 1000


def test_select_isolated_threshold_is_configurable():
    """
    Check that a smaller threshold keeps a pair 3 linewidths apart.
    """
    fr = np.array([1e9, 1e9 + 3 * LW, 1.1e9])
    keep, _ = select_isolated(fr, np.full(3, LW), min_sep_lw = 2.)
    assert keep.all()


def test_select_isolated_keep_stronger_chain():
    """
    Check 'keep_stronger' on a chain A-B-C, each 3 linewidths apart (A and
    C 6 apart): with B weakest, A and C are both kept.
    """
    fr = 1e9 + np.array([0, 3, 6]) * LW
    keep, _ = select_isolated(fr, np.full(3, LW), pairs = 'keep_stronger',
                              strength = [2., 0., 1.])
    np.testing.assert_array_equal(keep, [True, False, True])
    # With B strongest, only B is kept
    keep, _ = select_isolated(fr, np.full(3, LW), pairs = 'keep_stronger',
                              strength = [1., 2., np.nan])
    np.testing.assert_array_equal(keep, [False, True, False])


def test_select_isolated_candidates():
    """
    Check that non-candidates never get a tone but still make their
    neighbors close for 'drop_both', and don't for 'keep_stronger'.
    """
    fr = np.array([1e9, 1e9 + 2 * LW, 1.2e9])
    cand = [True, False, True]
    keep, _ = select_isolated(fr, np.full(3, LW), candidates = cand)
    np.testing.assert_array_equal(keep, [False, False, True])
    keep, _ = select_isolated(fr, np.full(3, LW), candidates = cand,
                              pairs = 'keep_stronger',
                              strength = [1., 5., 1.])
    np.testing.assert_array_equal(keep, [True, False, True])


def test_select_isolated_uses_larger_linewidth_and_fills_nan():
    """
    Check that closeness uses the larger of the two linewidths, and a NaN
    linewidth becomes the median.
    """
    fr = np.array([1e9, 1e9 + 20 * LW, 1.5e9])
    keep, near = select_isolated(fr, [LW, 10 * LW, LW])
    np.testing.assert_array_equal(keep, [False, False, True])
    np.testing.assert_allclose(near[:2], [2, 2])
    # The NaN linewidth becomes the median (4 LW), which makes the last two
    # close (10 LW apart = 2.5 linewidths)
    fr = np.array([1e9, 1.5e9, 1.5e9 + 10 * LW])
    keep, near = select_isolated(fr, [4 * LW, 4 * LW, np.nan])
    np.testing.assert_array_equal(keep, [True, False, False])
    np.testing.assert_allclose(near[1:], [2.5, 2.5])


def test_select_isolated_input_order_and_brute_force():
    """
    Check against a brute-force computation on random, unsorted resonators.
    """
    rng = np.random.default_rng(0)
    fr = rng.uniform(1e9, 1.05e9, 300)
    lw = rng.uniform(2e4, 1e5, 300)
    keep, near = select_isolated(fr, lw)
    d = np.abs(fr[:, None] - fr[None, :]) / np.maximum(lw[:, None],
                                                       lw[None, :])
    np.fill_diagonal(d, np.inf)
    np.testing.assert_allclose(near, d.min(axis = 1))
    np.testing.assert_array_equal(keep, d.min(axis = 1) >= 5)
    keep, _ = select_isolated(fr, lw, pairs = 'keep_stronger',
                              strength = rng.uniform(0, 1, 300))
    kept = np.flatnonzero(keep)
    assert np.all(d[np.ix_(kept, kept)] >= 5)
    # Every dropped resonator is close to a kept one
    dropped = np.flatnonzero(~keep)
    assert np.all((d[np.ix_(dropped, kept)] < 5).any(axis = 1))


@pytest.mark.parametrize('kwargs, match', [
    (dict(pairs = 'other'), 'pairs'),
    (dict(pairs = 'keep_stronger'), 'strength is required'),
    (dict(pairs = 'keep_stronger', strength = [1.]), 'strength must'),
    (dict(candidates = [True]), 'candidates'),
    (dict(min_sep_lw = -1), 'min_sep_lw'),
])
def test_select_isolated_invalid(kwargs, match):
    with pytest.raises(ValueError, match = match):
        select_isolated([1e9, 2e9], [LW, LW], **kwargs)


def test_select_isolated_edge_cases():
    keep, near = select_isolated([], [])
    assert keep.shape == near.shape == (0,)
    keep, near = select_isolated([1e9], [LW])
    assert keep.all() and np.isinf(near[0])
    with pytest.raises(ValueError, match = 'linewidth'):
        select_isolated([1e9, 2e9], [np.nan, np.nan])


################################################################################
############################### plan_tone_sets #################################
################################################################################
def _check_windows(fres, plan, half):
    """
    Check that every tone is within ``half`` of its set's NCO on its module.

    Parameters:
    fres (np.array): tone frequencies in Hz.
    plan (dict): ``plan_tone_sets`` output.
    half (float): largest allowed distance from the NCO in Hz.
    """
    for f, s, m in zip(fres, plan['tone_set'], plan['tone_module']):
        assert abs(f - plan['ncos'][s][m]) <= half * (1 + 1e-12)


def test_plan_one_window():
    fres = np.linspace(1.0e9, 1.4e9, 50)
    plan = plan_tone_sets(fres, [2])
    assert plan['extended_bw'] is False
    assert plan['ncos'] == [{2: 1.2e9}]
    assert np.all(plan['tone_set'] == 0) and np.all(plan['tone_module'] == 2)
    _check_windows(fres, plan, 240e6)


@pytest.mark.parametrize('extended_bw, nw, half', [
    (False, 5, 240e6), (True, 4, 287.5e6), ('auto', 4, 287.5e6)])
def test_plan_400_2400_one_module(extended_bw, nw, half):
    """
    Check 400-2400 MHz on one module: 5 standard windows (480 MHz each) or
    4 extended ones (575 MHz); 'auto' picks extended since it needs fewer.
    """
    fres = np.arange(400e6, 2400e6 + 1, 2e6)
    plan = plan_tone_sets(fres, [1], extended_bw = extended_bw)
    assert len(plan['window_ncos']) == nw == len(plan['ncos'])
    assert plan['extended_bw'] is (extended_bw is not False)
    np.testing.assert_array_equal(np.unique(plan['tone_set']), range(nw))
    _check_windows(fres, plan, half)


def test_plan_four_modules_sets():
    """
    Check that windows go to the modules in turn, one pass per tone set.
    """
    fres = np.arange(400e6, 2400e6 + 1, 2e6)
    plan = plan_tone_sets(fres, [1, 2, 3, 4], extended_bw = False)
    assert len(plan['ncos']) == 2
    assert sorted(plan['ncos'][0]) == [1, 2, 3, 4]
    assert list(plan['ncos'][1]) == [1]
    _check_windows(fres, plan, 240e6)


@pytest.mark.parametrize('span, use_ext', [(520e6, True), (590e6, False)])
def test_plan_auto_extended_only_within_575(span, use_ext):
    """
    Check 'auto': tones spanning 520 MHz fit one extended window (fewer than
    two standard ones); 590 MHz needs two either way, so standard.
    """
    fres = np.linspace(1e9, 1e9 + span, 200)
    plan = plan_tone_sets(fres, [1])
    assert plan['extended_bw'] is use_ext
    assert len(plan['window_ncos']) == (1 if use_ext else 2)


def test_plan_cuts_at_widest_gap():
    """
    Check that the cut between two windows falls in the wide gap between
    two clusters, not where a greedy fill would cut.
    """
    a = np.linspace(1.0e9, 1.3e9, 50)
    b = np.linspace(1.32e9, 1.7e9, 50)
    fres = np.concatenate((a, b))
    plan = plan_tone_sets(fres, [1], extended_bw = False)
    np.testing.assert_array_equal(plan['tone_set'], [0] * 50 + [1] * 50)
    np.testing.assert_allclose(plan['window_edges'],
                               [[1.0e9, 1.3e9], [1.32e9, 1.7e9]])
    _check_windows(fres, plan, 240e6)


def test_plan_max_tones_and_input_order():
    """
    Check the per-window tone limit, and that results follow the input
    order.
    """
    rng = np.random.default_rng(1)
    fres = np.sort(rng.uniform(1e9, 1.1e9, 3000))
    plan = plan_tone_sets(fres, [1, 2, 3])
    assert len(plan['window_ncos']) == 3
    assert np.bincount(plan['tone_set'] * 3 + plan['tone_module'] - 1).max() \
        <= 1024
    perm = rng.permutation(3000)
    shuffled = plan_tone_sets(fres[perm], [1, 2, 3])
    np.testing.assert_array_equal(shuffled['tone_set'], plan['tone_set'][perm])
    np.testing.assert_array_equal(shuffled['tone_module'],
                                  plan['tone_module'][perm])


def test_plan_edge_margin():
    """
    Check that a larger edge margin shrinks the windows.
    """
    fres = np.linspace(1e9, 1.47e9, 100)
    assert len(plan_tone_sets(fres, [1], extended_bw = False)[
        'window_ncos']) == 1
    assert len(plan_tone_sets(fres, [1], extended_bw = False,
                              edge_margin = 20e6)['window_ncos']) == 2


@pytest.mark.parametrize('args, kwargs, match', [
    (([1e9], []), {}, 'modules'),
    (([1e9], [1, 1]), {}, 'modules'),
    (([np.nan], [1]), {}, 'finite'),
    (([1e9], [1]), dict(extended_bw = 'yes'), 'extended_bw'),
    (([1e9], [1]), dict(edge_margin = 300e6), 'positive'),
    (([1e9], [1]), dict(max_tones = 0), 'max_tones'),
])
def test_plan_invalid(args, kwargs, match):
    with pytest.raises(ValueError, match = match):
        plan_tone_sets(*args, **kwargs)


def test_plan_no_tones():
    plan = plan_tone_sets([], [1])
    assert plan['ncos'] == [] and plan['window_edges'].shape == (0, 2)


################################################################################
########################### place_cal_tones ####################################
################################################################################
def test_place_cal_tones_widest_gaps():
    from citkid.tone_plan import place_cal_tones
    fr = np.array([1.00e9, 1.01e9, 1.05e9, 1.06e9])
    lw = np.full(4, 5e4)
    f = place_cal_tones(fr, lw, 0.99e9, 1.10e9, 2)
    # Widest gaps: 1.06-1.10 (edge) and 1.01-1.05
    np.testing.assert_allclose(f, [1.03e9, 1.08e9])
    assert len(place_cal_tones(fr, lw, 0.99e9, 1.10e9, 0)) == 0
    # Asking for more than there are gaps returns what there is
    assert len(place_cal_tones(fr, lw, 0.99e9, 1.10e9, 10)) == 5


def test_place_cal_tones_clearance():
    from citkid.tone_plan import place_cal_tones
    # A wide resonance next to a gap's middle rules it out
    fr = np.array([1.00e9, 1.02e9, 1.021e9])
    lw = np.array([5e4, 5e4, 2e6])     # the last is very wide
    f = place_cal_tones(fr, lw, 1.00e9, 1.022e9, 3, min_sep_lw = 10.)
    assert not np.any(np.isclose(f, 1.0205e9))
    for fc in f:
        assert np.all(np.abs(fc - fr) >= 10 * lw)
    # NaN linewidths use the median
    f2 = place_cal_tones(fr, [5e4, np.nan, 5e4], 1.0e9, 1.022e9, 1)
    assert len(f2) == 1
    with pytest.raises(ValueError):
        place_cal_tones(fr, lw[:2], 1e9, 2e9, 1)
    with pytest.raises(ValueError):
        place_cal_tones(fr, lw, 2e9, 1e9, 1)
