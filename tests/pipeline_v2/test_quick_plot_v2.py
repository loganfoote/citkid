"""
Tests for pipeline_v2 quick plots (DataSet.plot and DataSet.plot_full_cal).
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from citkid.pipeline_v2 import AnalysisRunner, DataSet
from citkid.pipeline_v2 import quick_plot
from citkid.pipeline_v2.framework import plStep
from citkid.res.funcs import nonlinear_iq

NROWS, NPTS, NT, DT, QR = 2, 200, 3000, 1e-3, 2e4
FRES = np.array([4.0e8, 4.5e8])


def _s21(f, fr):
    """
    Synthetic linear resonator S21.

    Parameters:
    f (np.ndarray): frequencies (Hz).
    fr (float): resonance frequency (Hz).

    Returns:
    z (np.ndarray): complex S21.
    """
    return nonlinear_iq(np.asarray(f, float), fr, QR, 0.6, 0.05, 0.0, 1.0, 0.1, 1e-9, True)


def _data(seed=1):
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
        zt = _s21(fr * (1 + x), fr) + 2e-4 * (rng.standard_normal(NT) + 1j * rng.standard_normal(NT))
        data[r] = dict(ff=ff, zf=_s21(ff, fr) + 1e-3 * rng.standard_normal(NPTS),
                       fg=fg, zg=_s21(fg, fr), ft=fr, zt=zt)
    return data


def _ts_steps(data):
    """
    Calibration loading steps for the 'ts' calibration.

    Parameters:
    data (dict): output of ``_data``.

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


def _iq_steps(data):
    """
    Calibration loading steps for the 'iq' calibration.

    Parameters:
    data (dict): output of ``_data``.

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


@pytest.fixture(scope='module')
def ts_dataset(tmp_path_factory):
    """
    A 'ts' DataSet with the analysis run for row 0 only.

    Returns:
    DS (DataSet): the dataset.
    """
    path = tmp_path_factory.mktemp('ts') / 'ts.zarr'
    DS = DataSet(zarr_path=str(path), cal_yaml_path='ts', custom_cal_steps=_ts_steps(_data()))
    AnalysisRunner(DS, analysis_yaml_path='ts').execute_path(data_idx=0, save=True, verbose=False)
    return DS


@pytest.fixture(scope='module')
def iq_dataset(tmp_path_factory):
    """
    An 'iq' DataSet with the analysis run for row 0.

    Returns:
    DS (DataSet): the dataset.
    """
    path = tmp_path_factory.mktemp('iq') / 'iq.zarr'
    DS = DataSet(zarr_path=str(path), cal_yaml_path='iq', custom_cal_steps=_iq_steps(_data()))
    AnalysisRunner(DS, analysis_yaml_path='iq').execute_path(data_idx=0, save=True, verbose=False)
    return DS


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Close every figure after each test.
    """
    yield
    plt.close('all')


def _titles(fig):
    """
    Return the texts placed on a figure (panel titles and the main title).
    """
    return [t.get_text() for t in fig.texts]


@pytest.mark.parametrize('plot_type, n_axes', [
    ('raw_data', 2), ('gain_fit', 2), ('s21_rmv', 2), ('circfit', 1), ('sparper', 1), ('xcal', 2),
])
def test_plot_each_ts_type(ts_dataset, plot_type, n_axes):
    fig, axs = ts_dataset.plot(0, plot_type)

    assert len(np.atleast_1d(axs)) == n_axes
    assert all(ax.figure is fig for ax in np.atleast_1d(axs))
    assert quick_plot.PLOT_TYPES[plot_type].title in _titles(fig)


def test_plot_title_and_errors(ts_dataset):
    fig, _ = ts_dataset.plot(0, 'gain_fit', title='row 0 gain')
    assert _titles(fig) == ['row 0 gain']
    with pytest.raises(ValueError, match="not recognized"):
        ts_dataset.plot(0, 'nope')
    with pytest.raises(ValueError, match="needs 'iq_popt'"):
        ts_dataset.plot(0, 'iq_fit')                 # not in the 'ts' analysis
    with pytest.raises(ValueError, match="not available for data_idx 1"):
        ts_dataset.plot(1, 'circfit')                # analysis not run for row 1


def test_plot_thins_the_timestream(ts_dataset):
    def noise_points(axs):
        return [len(line.get_xdata()) for line in axs[1].lines if line.get_label() == 'Noise']

    _, axs = ts_dataset.plot(0, 'raw_data')
    assert noise_points(axs) == [quick_plot.DEFAULT_MAX_POINTS if NT > 5000 else NT]
    _, axs = ts_dataset.plot(0, 'raw_data', max_points=500)
    assert noise_points(axs) == [500]
    _, axs = ts_dataset.plot(0, 'raw_data', max_points=None, nbins=7)   # nbins: ignored here
    assert noise_points(axs) == [NT]


def test_plot_options_reach_the_plot_function(ts_dataset):
    _, ax = ts_dataset.plot(0, 'sparper', nbins=10)
    assert len(ax.lines[0].get_xdata()) <= 10


def test_full_cal_ts_returns_png_and_skips_missing_types(ts_dataset):
    png = ts_dataset.plot_full_cal(0)
    assert png.getvalue()[:8] == b'\x89PNG\r\n\x1a\n'
    assert not plt.get_fignums()                     # the figure was closed

    fig = ts_dataset.plot_full_cal(0, as_figure=True)
    titles = _titles(fig)
    for plot_type in ('raw_data', 'gain_fit', 's21_rmv', 'circfit', 'sparper', 'xcal'):
        assert quick_plot.PLOT_TYPES[plot_type].title in titles
    assert 'data_idx 0  (no data for: iq_fit)' in titles
    assert len(fig.axes) == 2 + 2 + 2 + 1 + 1 + 2


def test_full_cal_iq_dataset(iq_dataset):
    fig = iq_dataset.plot_full_cal(0, as_figure=True)
    titles = _titles(fig)

    assert quick_plot.PLOT_TYPES['iq_fit'].title in titles
    assert 'data_idx 0  (no data for: circfit, sparper, xcal)' in titles
    with pytest.raises(ValueError, match='None of the plot types'):
        iq_dataset.plot_full_cal(0, plot_types=['xcal', 'sparper'])


def test_full_cal_shows_a_failing_panel_instead_of_raising(ts_dataset, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('bad row')

    spec = quick_plot.PLOT_TYPES['xcal']
    monkeypatch.setitem(quick_plot.PLOT_TYPES, 'xcal', spec._replace(func=broken))

    fig = ts_dataset.plot_full_cal(0, as_figure=True)

    assert any('xcal failed' in t.get_text() and 'bad row' in t.get_text()
               for ax in fig.axes for t in ax.texts)


def test_plot_types_are_reserved_names(ts_dataset):
    """``plot`` and ``plot_full_cal`` are methods, not pipeline parameters."""
    reserved = ts_dataset._get_reserved_attrs()
    assert {'plot', 'plot_full_cal'} <= reserved
