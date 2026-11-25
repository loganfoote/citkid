"""
Tests of citkid.pipeline.hee_finder: thresholds, per-tone analysis and
refits, saving (the whole path through a store is in
tests/procedures/test_scripts.py).
"""
import numpy as np
import pytest
import zarr
from citkid import daq_layout as dl
from citkid.pipeline import hee_finder as cf
from ..signal.test_hee import rays
from ..signal.test_photons import DT, TAU

SETTINGS = cf.TailSettings(pre_s = 20 * DT, post_s = 150 * DT,
                           search_s = 10 * DT)


def test_thresholds_round_trip():
    th = cf.TailThresholds(level = 3., fit_s = 1e-3)
    th.overrides[5] = dict(hmin = 2., fit_s = 0.5e-3)
    th.set_bad(7, 'high noise')
    th.limits['tau'] = [1e-4, 2e-4]
    th.mark_reviewed(5)
    assert th.for_tone(5) == dict(hmin = 2., hmax = 1000., level = 3.,
                                  fit_s = 0.5e-3, fit_start_s = None)
    assert th.for_tone(6)['fit_s'] == 1e-3
    assert th.has_own(5) and not th.has_own(6)
    assert th.has_own(5, ('hmin',)) and not th.has_own(5, ('level',))
    assert th.is_bad(7) and not th.is_bad(5)
    back = cf.TailThresholds.from_dict(th.to_dict())
    assert back == th
    d = th.to_dict()
    assert d['overrides'] == {'5': dict(hmin = 2., fit_s = 0.5e-3)}
    assert d['bad'] == {'7': 'high noise'} and d['reviewed'] == [5]
    # Setting a global value drops the tones' own; an emptied override
    # goes away
    th.set_all(fit_s = 2e-3)
    assert th.fit_s == 2e-3 and th.overrides == {5: dict(hmin = 2.)}
    th.set_all(hmin = 1.)
    assert th.overrides == {}
    th.clear_bad(7)
    assert th.bad == {}


def test_analyze_and_refit(rays):
    """
    A tone's pulses are cut at its write's events, lined up at the tail
    level, averaged and the mean fitted; a refit with other thresholds
    changes only what depends on them, without the data.
    """
    ds, times, heights = rays
    th = cf.TailThresholds(hmin = 0., hmax = 1000., level = 3.,
                           fit_s = 1e-3)
    res = cf.analyze_tone(ds[1], times, DT, SETTINGS, th.for_tone(0))
    # The same in a worker process (the pool of the per-tone fits)
    from citkid.parallel import map_in_pool
    pooled = map_in_pool(cf.analyze_tone, [
        (k, (ds[1], times, DT, SETTINGS, th.for_tone(0)))
        for k in range(2)], 2)
    for key in ('h', 'start', 'avg', 'tau'):
        np.testing.assert_allclose(pooled[1][key], res[key], equal_nan = True)
    assert res['pulses'].shape == (len(times), 170) and \
        res['pulses'].dtype == np.float32
    assert res['pre'] == 20 and res['post'] == 150
    np.testing.assert_allclose(res['h'], heights[1] * 1., rtol = 0.05)
    assert res['in_window'].all() and res['used'].all()
    assert res['n_used'] == len(times) and res['anchor'] == 20
    assert res['aligned'].shape == (len(times), 170)
    assert abs(res['tau'] - TAU) < 0.05 * TAU and res['n_fit'] > 4
    assert res['avg'].shape == (170,)
    # The mean is at the level at the anchor and decays from there
    assert 2e-6 < res['avg'][20] <= 3e-6 and res['avg'][30] < res['avg'][20]
    assert res['count'][20] == len(times)
    # A height window that keeps the pulses of middling height
    th2 = dict(th.for_tone(0), hmin = 8., hmax = 12.)
    res2 = cf.refit_tone(res, th2)
    inside = (res['h'] >= 8.) & (res['h'] <= 12.)
    assert res2['in_window'].tolist() == inside.tolist()
    assert res2['used'].sum() == inside.sum() == res2['n_used']
    assert np.all(res2['start'][~inside] == -1)
    # The overlap alignment is against the mean of the pulses kept, so a
    # start may move by a sample
    assert np.abs(res2['start'][inside] - res['start'][inside]).max() <= 1
    assert abs(res2['tau'] - TAU) < 0.1 * TAU
    assert res2['hmin'] == 8. and res['hmin'] == 0.
    # A level above every peak: the tails start at the peaks
    res3 = cf.refit_tone(res, dict(th.for_tone(0), level = 100.))
    # (then moved later, by at most the overlap alignment's reach: three
    # rounds of a tenth of the fit range)
    n_range = int(round(1e-3 / DT))
    moved = res3['start'] - res3['p']
    assert moved.min() >= 0 and moved.max() <= 3 * max(2, n_range // 10)
    # A level below the noise: nothing is averaged
    res4 = cf.refit_tone(res, dict(th.for_tone(0), level = -1.))
    assert res4['n_used'] == 0 and np.isnan(res4['tau']) and \
        np.isnan(res4['avg']).all() and res4['n_fit'] == 0


def test_save(rays, tmp_path):
    """
    A result saves per-tone arrays, the events and the pulse table, with
    failed tones marked.
    """
    ds, times, heights = rays
    th = cf.TailThresholds(level = 3., fit_s = 1e-3)
    th.set_bad(12, 'other')
    ts = dict(dataset = 'set0', run = 'ts0', name = 'dark', tone_set = 0,
              source = dict(group = 'tune0/cal0', writes = ['write0']),
              start = 0, stop = len(ds[0]), dt = DT,
              rows = np.array([0, 1, 2]), res_idxs = np.array([10, 11, 12]))
    results = {}
    for r in (0, 2):
        results[r] = cf.analyze_tone(ds[r], times, DT, SETTINGS,
                                     th.for_tone(10 + r))
        results[r]['res_idx'] = 10 + r
    results[1] = 'the calibrated timestream is not finite'
    events = dict(t0 = times, mult = np.full(len(times), 3),
                  write_idx = np.zeros(len(times), dtype = int))
    root = zarr.open_group(tmp_path / 's.zarr', mode = 'a')
    grp = dl.create_group(cf.results_group(root), 'dark', 'analysis')
    cf.save_timestream(grp, ts, results, events, th, SETTINGS, 'triggers')
    g = zarr.open_group(tmp_path / 's.zarr', mode = 'r')['hee/dark']
    assert g.attrs['events'] == 'triggers' and g.attrs['n_events'] == \
        len(times)
    assert g.attrs['thresholds']['bad'] == {'12': 'other'}
    assert g.attrs['settings']['post_s'] == pytest.approx(150 * DT)
    assert list(g['status'][:]) == ['', results[1], '']
    assert g['bad'][:].tolist() == [False, False, True]
    assert list(g['bad_reason'][:]) == ['', '', 'other']
    assert g['n_used'][:].tolist() == [len(times), 0, len(times)]
    assert np.isnan(g['tau'][:][1]) and np.isfinite(g['tau'][:][0])
    assert g.attrs['anchor'] == 20 and g['n_fit'][:][1] == 0
    np.testing.assert_allclose(g['rate'][:][0], len(times) / (len(ds[0])
                                                              * DT))
    assert g['avg_pulse'].shape == (3, 170)
    np.testing.assert_allclose(g['events/time_s'][:], times * DT)
    p = g['pulses']
    assert p['row'].shape == (2 * len(times),)
    assert set(p['res_idx'][:].tolist()) == {10, 12}
    assert p['used'][:].all() and 'tau' not in p
    th_back = cf.load_thresholds(g)
    assert th_back == th
    assert cf.list_results(root) == ['dark']
    assert 'dark: ts0 set 0' in cf.describe_results(root) and \
        '1 bad' in cf.describe_results(root)
    assert '3 tones' in cf.describe_result(g) and '1 marked bad' in \
        cf.describe_result(g)
    assert cf.describe_results(zarr.open_group(tmp_path / 'e.zarr',
                                               mode = 'a')) \
        .endswith('none')


def test_fit_start_per_tone(rays):
    """
    The fit of the mean tail begins after the latest peak within the
    search window by default, at a tone's own ``fit_start_s`` when set
    (saved per tone, and in the thresholds round trip), and the mean is
    averaged over the fit range only.
    """
    import numpy as np
    from citkid.pipeline import hee_finder as cf
    ds, times, _ = rays
    th = cf.TailThresholds(level = 3., fit_s = 1e-3)
    res = cf.analyze_tone(ds[0], times, DT, SETTINGS, th.for_tone(10))
    assert res['fit_start_s'] is None and res['fit_start'] >= res['anchor']
    assert np.isnan(res['avg'][:res['anchor']]).all()
    n_range = int(round(1e-3 / DT))
    assert np.isnan(res['avg'][res['anchor'] + n_range:]).all()
    assert np.isfinite(res['tau'])
    own = cf.refit_tone(res, dict(th.for_tone(10), fit_start_s = 2e-4))
    assert own['fit_start'] == res['anchor'] + int(round(2e-4 / DT))
    assert own['fit_start_s'] == 2e-4 and own['n_fit'] <= res['n_fit']
    th.overrides[10] = dict(fit_start_s = 2e-4)
    th.set_all(fit_start_s = 1e-4)
    assert th.for_tone(10)['fit_start_s'] == 1e-4 and 10 not in th.overrides
    again = cf.TailThresholds.from_dict(th.to_dict())
    assert again.fit_start_s == 1e-4
    th.set_all(fit_start_s = None)
    assert cf.TailThresholds.from_dict(th.to_dict()).fit_start_s is None
