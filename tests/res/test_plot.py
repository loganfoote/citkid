"""
Tests for citkid.res.plot.
"""

import warnings
import numpy as np
import pytest
import matplotlib.pyplot as plt

from citkid.res.funcs import nonlinear_iq
from citkid.res.plot import plot_nonlinear_iq
from citkid.res.fitter import fit_nonlinear_iq

PARAMS = [500e6, 20000, 0.5, 0.1, 0.05, 1.0, 0.0, 0.0]


@pytest.fixture
def data():
    """Synthetic downward-sweep IQ data from PARAMS."""
    f = np.linspace(500e6 - 75e3, 500e6 + 75e3, 300)
    return f, nonlinear_iq(f, *PARAMS, True)


def test_module_has_no_legacy_warning():
    """Test importing citkid.res.plot does not warn."""
    import importlib
    import citkid.res.plot as module
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        importlib.reload(module)


@pytest.mark.parametrize("plot_guess, nlines", [(False, 2), (True, 3)])
def test_plot_nonlinear_iq_lines(data, plot_guess, nlines):
    """Test the data, fit, and optional guess are drawn on both axes."""
    f, z = data
    fig, axs = plot_nonlinear_iq(f, z, PARAMS, PARAMS, plot_guess = plot_guess)
    assert len(axs) == 2
    for ax in axs:
        assert len(ax.lines) == nlines
    labels = [t.get_text() for t in axs[1].get_legend().get_texts()]
    assert labels == ['data', 'fit', 'guess'][:nlines]
    np.testing.assert_allclose(axs[0].lines[0].get_xdata(), z.real)
    plt.close(fig)


def test_plot_nonlinear_iq_fast_settings(data):
    """Test the figure uses the fast settings: no layout engine, 72 dpi."""
    f, z = data
    fig, axs = plot_nonlinear_iq(f, z, PARAMS, PARAMS)
    assert fig.get_layout_engine() is None
    assert fig.dpi == 72
    assert all(not line.get_antialiased() for line in axs[0].lines)
    plt.close(fig)


def test_plot_nonlinear_iq_centers_on_fr(data):
    """Test the |S21| axis is centered on the fitted fr."""
    f, z = data
    popt = list(PARAMS)
    popt[0] = 500e6 + 10e3
    fig, axs = plot_nonlinear_iq(f, z, popt, PARAMS)
    x = axs[1].lines[0].get_xdata()
    np.testing.assert_allclose(x, (f - popt[0]) * 1e-3)
    assert '0.5 GHz' in axs[1].get_xlabel()
    plt.close(fig)


def test_plot_nonlinear_iq_failed_fit(data):
    """Test a failed (NaN) fit draws only the data, centered on mean f."""
    f, z = data
    nan = [np.nan] * 8
    fig, axs = plot_nonlinear_iq(f, z, nan, nan, plot_guess = True)
    assert len(axs[0].lines) == 1
    np.testing.assert_allclose(axs[1].lines[0].get_xdata(),
                               (f - np.mean(f)) * 1e-3)
    plt.close(fig)


@pytest.mark.parametrize("downward", [True, False])
def test_plot_nonlinear_iq_direction(downward):
    """Test the fit curve uses the requested sweep direction."""
    f = np.linspace(500e6 - 75e3, 500e6 + 75e3, 300)
    z = nonlinear_iq(f, *PARAMS, downward)
    fig, axs = plot_nonlinear_iq(f, z, PARAMS, PARAMS, downward = downward)
    fsamp = np.linspace(f.min(), f.max(), 500)
    expected = nonlinear_iq(fsamp, *PARAMS, downward)
    np.testing.assert_allclose(axs[0].lines[1].get_xdata(), expected.real)
    plt.close(fig)


def test_fit_nonlinear_iq_plotq(data):
    """Test fit_nonlinear_iq returns the plot when plotq is True."""
    f, z = data
    p0, popt, perr, nrmse, (fig, axs) = fit_nonlinear_iq(f, z, plotq = True)
    assert fig is not None and len(axs) == 2
    plt.close(fig)
