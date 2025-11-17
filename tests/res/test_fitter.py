from citkid.res import fitter
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

@pytest.mark.parametrize("z,plotq,popt_exp", [
    ([1, 1j, -1, -1j], False, [0, 0, 1]),
    ([2, 1 + 1j, 0, 1 - 1j], False, [1, 0, 1]),
    ([1 + 1j, 2j, -1 + 1j, 0], False, [0, 1, 1]),
    ([1, 1j, -1, -1j], True, [0, 0, 1]),
    ([1e30, 1e30j, -1e30, -1e30j], False, [0, 0, 1e30]),
    ([1e-30, 1e-30j, -1e-30, -1e-30j], False, [0, 0, 1e-30]),
])
def test_fit_iq_circle_gain(z, plotq, popt_exp):
    popt, fig = fitter.fit_iq_circle(z, plotq = plotq)
    assert np.allclose(popt, popt_exp)
    if plotq:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    else:
        assert fig is None
    
@pytest.mark.parametrize("z", [
    ([]),  # empty input
    ([np.nan, 1, -1, 1j]),  # nan input
    (['a', 1, -1, 1j]),  # non-numeric input
])
def test_fit_iq_circle_invalid_input(z):
    with pytest.raises(Exception):
        fitter.fit_iq_circle(z, plotq = False)