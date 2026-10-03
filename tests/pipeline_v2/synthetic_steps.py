"""
Synthetic calibration loading steps ('ts' and 'iq' templates) for pipeline_v2 tests.
"""
import numpy as np

from citkid.pipeline_v2.framework import plStep
from citkid.res.funcs import nonlinear_iq

NROWS, NPTS, NT, DT, QR = 2, 200, 3000, 1e-3, 2e4
FRES = np.array([4.0e8, 4.5e8])


def s21(f, fr):
    """
    Synthetic linear resonator S21.

    Parameters:
    f (np.ndarray): frequencies (Hz).
    fr (float): resonance frequency (Hz).

    Returns:
    z (np.ndarray): complex S21.
    """
    return nonlinear_iq(np.asarray(f, float), fr, QR, 0.6, 0.05, 0.0, 1.0, 0.1, 1e-9, True)


def synthetic_data(seed=1):
    """
    Synthetic sweeps and timestreams for each row.

    Parameters:
    seed (int): random seed.

    Returns:
    data (dict): row -> dict of ff, zf, fg, zg, ft, zt.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for r, fr in enumerate(FRES):
        ff = np.linspace(fr - 6 * fr / QR, fr + 6 * fr / QR, NPTS)
        fg = np.linspace(fr - 60 * fr / QR, fr + 60 * fr / QR, NPTS)
        x = 2e-7 * rng.standard_normal(NT)
        zt = s21(fr * (1 + x), fr) + 2e-4 * (rng.standard_normal(NT) + 1j * rng.standard_normal(NT))
        data[r] = dict(ff=ff, zf=s21(ff, fr) + 1e-3 * rng.standard_normal(NPTS),
                       fg=fg, zg=s21(fg, fr), ft=fr, zt=zt)
    return data


def ts_steps(data):
    """
    Calibration loading steps for the 'ts' calibration.

    Parameters:
    data (dict): output of ``synthetic_data``.

    Returns:
    steps (list of plStep): loading steps.
    """
    return [
        plStep('load_global_data', lambda: (FRES, np.full(NROWS, QR), DT, NROWS), [],
               ['fres_all', 'qres_all', 'dt', 'nrows'], 'global'),
        plStep('load_global_res_data',
               lambda: (FRES, np.full(NROWS, QR), np.zeros(NROWS), np.arange(NROWS)), [],
               ['fres', 'qres', 'ares', 'res_idxs'], 'global-res'),
        plStep('load_ft', lambda data_idx: np.array([data[int(r)]['ft'] for r in np.atleast_1d(data_idx)]),
               ['data_idx'], ['ft'], 'vectorized'),
        plStep('load_zt', lambda data_idx: np.array([data[int(r)]['zt'] for r in np.atleast_1d(data_idx)]),
               ['data_idx'], ['zt'], 'vectorized'),
        plStep('load_data_f', lambda data_idx: (data[int(data_idx)]['ff'], data[int(data_idx)]['zf']),
               ['data_idx'], ['ff', 'zf'], 'per-row'),
        plStep('load_data_g', lambda data_idx: (data[int(data_idx)]['fg'], data[int(data_idx)]['zg']),
               ['data_idx'], ['fg', 'zg'], 'per-row'),
    ]


def iq_steps(data):
    """
    Calibration loading steps for the 'iq' calibration.

    Parameters:
    data (dict): output of ``synthetic_data``.

    Returns:
    steps (list of plStep): loading steps.
    """
    return [
        plStep('load_global_data', lambda: (FRES, np.full(NROWS, QR), NROWS), [],
               ['fres_all', 'qres_all', 'nrows'], 'global'),
        plStep('load_global_res_data',
               lambda: (FRES, np.full(NROWS, QR), np.zeros(NROWS), np.arange(NROWS)), [],
               ['fres', 'qres', 'ares', 'res_idxs'], 'global-res'),
        plStep('load_data_f', lambda data_idx: (data[int(data_idx)]['ff'], data[int(data_idx)]['zf']),
               ['data_idx'], ['ff', 'zf'], 'per-row'),
        plStep('load_data_g', lambda data_idx: (data[int(data_idx)]['fg'], data[int(data_idx)]['zg']),
               ['data_idx'], ['fg', 'zg'], 'per-row'),
    ]
