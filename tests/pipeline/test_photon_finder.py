"""
Tests of the photon finder's bookkeeping.
"""
import numpy as np
from citkid.pipeline import photon_finder as pf
from citkid.signal.photons import DetectionSettings


def test_timestream_data_blocks(monkeypatch):
    """
    Tones are calibrated in blocks bounded by the memory limit and by
    ``BLOCK_BYTES``, so a long timestream's first tone is handed out
    before the rest are calibrated; the detection timestreams are kept
    within the memory limit and not computed again.
    """
    calls = []

    def fake_apply(DS, root, source, rows = None):
        """
        Stand in for the calibration: record the rows, give short x.

        Parameters:
        DS, root, source: unused.
        rows (list of int): the rows.

        Returns:
        out (dict): 'xt': one short array per row.
        """
        calls.append(list(rows))
        return {'xt': [np.full(64, float(r)) for r in rows]}

    monkeypatch.setattr(pf, 'apply_stream_calibration', fake_apply)
    rng = np.random.default_rng(0)
    # 50 M samples per tone: 600 MB of x and detection per tone, so
    # three tones per block under BLOCK_BYTES, not 20 under the limit
    ts = dict(dt = 1e-6, start = 0, stop = 50_000_000, source = 's',
              rows = np.arange(8))
    data = pf.TimestreamData(None, ts, None, DetectionSettings(),
                             max_bytes = 12e9)
    got = []
    for r, d in data.get(list(range(8))):
        got.append(r)
        if r == 0:
            assert calls == [[0, 1, 2]]
    assert got == list(range(8))
    assert calls == [[0, 1, 2], [3, 4, 5], [6, 7]]
    # Kept: asked again, nothing is calibrated anew
    assert [r for r, _ in data.get([1, 7])] == [1, 7] and len(calls) == 3
    # Short timestreams: as many tones per block as the limit allows
    ts = dict(ts, stop = 1000)
    data = pf.TimestreamData(None, ts, None, DetectionSettings(),
                             max_bytes = 1000 * 12 * 5)
    calls.clear()
    assert len(list(data.get(list(range(8))))) == 8
    assert calls == [[0, 1, 2, 3, 4], [5, 6, 7]]


def test_list_timestreams_keeps_each_tuning_apart(tmp_path, monkeypatch):
    """
    Runs are numbered per tuning (tune0/ts0, tune1/ts0, ...): the listing
    takes each run's streams from its own tuning, so a calibration of
    the first tuning's ts0 never gets the streams of a later tuning's
    ts0 (the mix-up that put the 140 K timestream under the 3.4 K name).
    """
    from types import SimpleNamespace
    from citkid import daq_layout as dl
    root = dl.create_store(tmp_path / 'a.zarr', 'devA', 'cd1')
    dl.save_resonators(root, 0, [1e9, 1.1e9], 2e4, 'x')
    for k, name in enumerate(('cold', 'hot')):
        tune, _ = dl.new_tuning(root, 0)
        dl.save_tones(dl.create_group(tune, 'tones', 'decision'),
                      [1e9, 1.1e9], 3e4, -50., [0, 1], tone_set = [0, 0])
        cal = dl.create_group(tune, 'cal0', 'raw', attrs = dict(
            tone_set = 0, complete = True, nsamps = 10))
        cal.create_array('fres', data = np.full(2, 1e9))
        cal.create_array('res_idxs', data = np.arange(2))
        w = dl.create_group(cal, 'write0', 'raw', attrs = dict(
            tones_reused = False))
        w.create_array('sweep_rows', data = np.arange(2))
        st = dl.create_group(w, 'ts0', 'raw', attrs = dict(
            name = name, dec_stage = 6, duration_s = 2., run = 0))
        st.create_array('z', data = np.zeros((2, 2, 8), dtype = np.int32),
                        chunks = (1, 1, 4))
        st.create_array('counts_to_s21', data = np.ones(2))
        st.create_array('dt', data = np.array(0.25))
        dl.register_stream_run(tune, dict(
            name = name, dec_stage = 6, duration_s = 2.,
            tuning = tune.basename, tones = f'{tune.basename}/tones',
            resonators = 'resonators0', complete = True, tone_set = 0,
            calibration = f'{tune.basename}/cal0', paths = [st.path]))
    root = dl.open_store(tmp_path / 'a.zarr')
    runs = dl.stream_runs(root)
    assert [(r['tuning'], r['group'], r['name']) for r in runs] == [
        ('tune0', 'ts0', 'cold'), ('tune1', 'ts0', 'hot')]
    monkeypatch.setattr(pf, 'calibrated_rows', lambda DS: np.arange(2))
    DS = SimpleNamespace(res_idxs = np.arange(2))
    for k, name in enumerate(('cold', 'hot')):
        datasets = {'set0': dict(DS = DS, calibration = f'tune{k}/cal0')}
        listed = pf.list_timestreams(root, datasets)
        assert len(listed) == 1
        ts = listed[0]
        assert ts['name'] == name and ts['tuning'] == f'tune{k}'
        assert ts['source']['group'] == f'tune{k}/cal0'
        assert ts['run'] == 'ts0'
        assert pf.result_name(ts, listed) == name
    assert 'tune1 ts0' in pf.describe_timestreams(listed)
