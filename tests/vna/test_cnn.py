import numpy as np
import pytest
import zarr

torch = pytest.importorskip('torch')
from citkid.vna.cnn import infer, model as cm  # noqa: E402
from citkid.vna.cnn.net import UNet1D  # noqa: E402
from citkid.vna.res_finder_cnn import (  # noqa: E402
    find_resonances_cnn, run_res_finder_cnn
)

SMALL = dict(width = 4, depth = 3, max_width = 8)

def _model_file(path, arch = 'UNet1D'):
    """Save a small random model in the citkid model-file format."""
    torch.manual_seed(0)
    net = UNet1D(**SMALL)
    torch.save(dict(arch = arch, name = 'test', model_kwargs = SMALL,
                    state_dict = net.state_dict(), threshold = 0.3,
                    chunk_ratio = 1.1, overlap = 0.01, min_step = 1.5e-6),
               path)
    return path

def test_unet_shapes():
    net = UNet1D(**SMALL)
    h, o, q = net(torch.zeros(2, 3, 64))
    assert h.shape == o.shape == q.shape == (2, 64)

def test_resample_fractional():
    f = np.r_[np.linspace(1e9, 1.01e9, 1000), np.linspace(1.02e9, 1.03e9,
                                                           1000)]
    z = np.exp(1j * f / 1e6)
    fg, zg, valid = infer.resample_fractional(f, z)
    np.testing.assert_allclose(np.diff(np.log(fg)), np.log(fg[1] / fg[0]))
    gap = (fg > 1.0101e9) & (fg < 1.0199e9)
    assert not valid[gap].any() and valid[fg < 1.009e9].all()

def test_detector_features():
    rng = np.random.default_rng(0)
    z = 1 + 0.01 * (rng.standard_normal(500) + 1j * rng.standard_normal(500))
    x = infer.detector_features(z)
    assert x.shape == (3, 500) and x.dtype == np.float32
    assert np.all(x[2] == 1)

def test_sweep_chunks():
    f = np.geomspace(1e9, 2e9, 10000)
    chunks = infer.sweep_chunks(f)
    assert len(chunks) == int(np.ceil(np.log(2) / np.log(1.1)))
    assert chunks[-1][2] and not chunks[0][2]
    one = infer.sweep_chunks(f[:100])
    assert len(one) == 1 and one[0][0] == -np.inf

def test_load_model(tmp_path):
    net = cm.load_model(path = _model_file(tmp_path / 'm.pt'),
                        device = 'cpu')
    assert net.depth == 3 and net.settings['threshold'] == 0.3
    with pytest.raises(ValueError):
        cm.load_model(path = _model_file(tmp_path / 'bad.pt', 'other'))

def test_model_path_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'CACHE_DIR', tmp_path)
    p = _model_file(tmp_path / 'released.pt')
    monkeypatch.setattr(cm, 'MODELS', {
        'x': ('https://example.invalid/released.pt', cm._sha256(p)),
        'unreleased': ('https://example.invalid/u.pt', None)})
    assert cm.model_path('x') == p  # cached copy, no download
    with pytest.raises(RuntimeError, match = 'not released'):
        cm.model_path('unreleased')
    with pytest.raises(KeyError):
        cm.model_path('unknown')

def test_detect_and_finder(full_sweep, tmp_path):
    f, z, _, _ = full_sweep
    net = cm.load_model(path = _model_file(tmp_path / 'm.pt'),
                        device = 'cpu')
    fr, q, s = infer.detect(net, f, z, threshold = 0.)
    assert len(fr) and np.all(np.diff(fr) > 0) and np.all(q > 0)
    fres, qres, cand = find_resonances_cnn(f, z, model = net,
                                           threshold = 0.,
                                           fit_candidates = False)
    assert len(fres) == len(cand) and 'score' in cand

def test_find_resonances_cnn_device(full_sweep, tmp_path):
    """
    Check that ``device`` loads the model there, or moves a loaded model.
    """
    f, z, _, _ = full_sweep
    path = str(_model_file(tmp_path / 'm.pt'))
    fres, _, _ = find_resonances_cnn(f, z, model_path = path, threshold = 0.,
                                     classical = False, fit_candidates = False,
                                     device = 'cpu')
    net = cm.load_model(path = path, device = 'cpu')
    fres2, _, _ = find_resonances_cnn(f, z, model = net, threshold = 0.,
                                      classical = False,
                                      fit_candidates = False, device = 'cpu')
    assert next(net.parameters()).device.type == 'cpu'
    np.testing.assert_array_equal(fres, fres2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no CUDA')
def test_predict_maps_moves_model_to_device(tmp_path):
    """
    Check that ``predict_maps(device = ...)`` runs a model loaded on another
    device, the same as on the CPU.
    """
    net = cm.load_model(path = _model_file(tmp_path / 'm.pt'),
                        device = 'cpu')
    x = infer.detector_features(np.exp(1j * np.linspace(0, 3, 300)))
    cpu = infer.predict_maps(net, x)
    gpu = infer.predict_maps(net, x, device = 'cuda')
    assert next(net.parameters()).device.type == 'cuda'
    for a, b in zip(cpu, gpu):
        np.testing.assert_allclose(a, b, atol = 1e-4)

def test_candidates_qr_hints(full_sweep, tmp_path, monkeypatch):
    """
    Check that every candidate is fit with its detector's Qr as the hint,
    the matched filter's included.
    """
    import citkid.vna.res_finder_cnn as rfc
    f, z, _, _ = full_sweep
    seen = {}

    def fake_fit(f, z, fr0, Qr0, **kw):
        """
        Record the guesses and stop.

        Parameters:
        f, z, fr0, Qr0, kw: as ``fit_from_guesses``.

        Raises:
        RuntimeError: always.
        """
        seen['fr0'], seen['Qr0'] = np.asarray(fr0), np.asarray(Qr0)
        raise RuntimeError('stop')

    monkeypatch.setattr(rfc, 'fit_from_guesses', fake_fit)
    path = str(_model_file(tmp_path / 'm.pt'))
    _, _, cand = find_resonances_cnn(f, z, model_path = path,
                                     threshold = 0.99, fit_candidates = False)
    with pytest.raises(RuntimeError, match = 'stop'):
        find_resonances_cnn(f, z, model_path = path, threshold = 0.99)
    assert (cand.source == 'classical').any()
    np.testing.assert_array_equal(seen['fr0'], cand.fr.values)
    np.testing.assert_array_equal(seen['Qr0'], cand.Qr.values)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no CUDA')
def test_find_resonances_cnn_cuda_matches_cpu(full_sweep, tmp_path):
    """
    Check that the network gives the same candidates on CUDA and the CPU.
    """
    f, z, _, _ = full_sweep
    path = str(_model_file(tmp_path / 'm.pt'))
    kw = dict(model_path = path, threshold = 0., classical = False,
              fit_candidates = False)
    fc, qc, _ = find_resonances_cnn(f, z, device = 'cpu', **kw)
    fg, qg, _ = find_resonances_cnn(f, z, device = 'cuda', **kw)
    assert len(fc) == len(fg)
    np.testing.assert_allclose(fg, fc, rtol = 1e-9)
    np.testing.assert_allclose(qg, qc, rtol = 1e-3)

def test_run_res_finder_cnn(full_sweep, tmp_path):
    f, z, _, t = full_sweep
    path = _model_file(tmp_path / 'm.pt')
    grp = zarr.open_group(str(tmp_path / 's.zarr'), mode = 'a')
    # A threshold above 1: the (untrained) network finds nothing, so with
    # the CNN alone, empty lists are saved
    fres = run_res_finder_cnn(f, z, grp, model_path = str(path),
                              threshold = 1.1, classical = False,
                              progress = False)
    assert len(fres) == 0 and grp['fres_auto'].shape == (0,)
    assert grp['res_finder_auto'].attrs['finder'] == 'cnn'
    with pytest.raises(FileExistsError):
        run_res_finder_cnn(f, z, grp, model_path = str(path),
                           threshold = 1.1)
    # The matched filter finds the resonances on its own; the fits keep
    # every true one
    fres = run_res_finder_cnn(f, z, grp, model_path = str(path),
                              threshold = 1.1, overwrite = True,
                              progress = False)
    lw = t['fr'] / t['Qr']
    d = np.array([np.min(np.abs(fres - x)) for x in t['fr']]) / lw
    assert np.all(d < 0.1) and len(fres) == len(t['fr'])
    assert grp['qres_auto'].shape == (len(fres),)
