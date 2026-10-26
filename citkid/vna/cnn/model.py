"""
Load a released CNN resonance detector.

Released models are files attached to a citkid GitHub release. The first
use downloads the file into a cache folder (``CACHE_DIR``) and checks its
SHA-256 checksum; later uses load the cached copy, so no network is
needed. A model file can also be loaded from a local path (``path``),
e.g. while developing a new model.

A model file holds a dict: 'arch' ('UNet1D'), 'name', 'model_kwargs',
'state_dict', and the inference settings it was trained for
('threshold', 'chunk_ratio', 'overlap', 'min_step').
"""
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
import torch
from .net import UNet1D

_RELEASES = 'https://github.com/loganfoote/citkid/releases/download'
# name: (url, sha256). A model is usable once its release is published and
# its checksum is filled in.
MODELS = {
    'det_v2': (f'{_RELEASES}/cnn-det_v2/citkid_cnn_det_v2.pt', None),
}
DEFAULT_MODEL = 'det_v2'
CACHE_DIR = Path(os.environ.get('CITKID_CACHE', '~/.citkid')).expanduser() \
    / 'models'

def _sha256(path):
    """
    Compute the SHA-256 checksum of a file.

    Parameters:
    path (Path): file.

    Returns:
    digest (str): hex digest.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def model_path(name = None):
    """
    Return the local path of a released model, downloading it if needed.

    Parameters:
    name (str or None): model name in ``MODELS``. None: ``DEFAULT_MODEL``.

    Returns:
    path (Path): cached model file.

    Raises:
    KeyError: if the name is unknown.
    RuntimeError: if the model is not released yet, or the download does
        not match the checksum.
    """
    name = name or DEFAULT_MODEL
    if name not in MODELS:
        raise KeyError(f'unknown model {name!r}; known: {list(MODELS)}')
    url, sha = MODELS[name]
    if sha is None:
        raise RuntimeError(f'model {name!r} is not released yet; load a '
                           'local file with path = ...')
    path = CACHE_DIR / Path(url).name
    if path.exists() and _sha256(path) == sha:
        return path
    CACHE_DIR.mkdir(parents = True, exist_ok = True)
    print(f'Downloading the CNN model {name!r} (one time) from {url}')
    fd, tmp = tempfile.mkstemp(dir = CACHE_DIR, suffix = '.part')
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as fh:
            shutil.copyfileobj(r, fh)
        if _sha256(tmp) != sha:
            raise RuntimeError(f'downloaded model {name!r} does not match '
                               'its checksum')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path

def load_model(name = None, path = None, device = None):
    """
    Load a CNN resonance detector.

    Parameters:
    name (str or None): released model name (``MODELS``). None:
        ``DEFAULT_MODEL``. Ignored if ``path`` is given.
    path (str, Path, or None): local model file. Default None (download
        or use the cached released model).
    device (str or None): torch device. Default None: CUDA if available,
        else CPU.

    Returns:
    model (UNet1D): network in eval mode, with ``model.settings`` (dict:
        'name', 'threshold', 'chunk_ratio', 'overlap', 'min_step').
    """
    path = Path(path) if path is not None else model_path(name)
    ck = torch.load(path, map_location = 'cpu', weights_only = True)
    if ck.get('arch') != 'UNet1D':
        raise ValueError(f'{path} is not a citkid CNN model file')
    model = UNet1D(**ck.get('model_kwargs', {}))
    model.load_state_dict({k: v.float() for k, v in ck['state_dict'].items()})
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    model.settings = {k: ck[k] for k in ('name', 'threshold', 'chunk_ratio',
                                         'overlap', 'min_step') if k in ck}
    return model
