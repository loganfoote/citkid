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
