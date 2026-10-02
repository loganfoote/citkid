from citkid.xcal import plot
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Not testing the actual plots, just that they generate without error

# These tests are incomplete, need to finish building plot functions first

################################################################################
################################# plot_circle ##################################
################################################################################
@pytest.mark.parametrize("z,A,B,R", [
    (np.array([1+1j, 2+2j, 3+3j]), 0.0, 0.0, 5.0),
    (np.array([-1-1j, -2-2j, -3-3j]), 1.0, 1.0, 4.0),
])  
def test_plot_circle(z, A, B, R):
    fig, ax = plot.plot_circfit(z, A, B, R)
    assert fig is not None
    assert ax is not None   
    assert hasattr(fig, 'canvas')
    assert hasattr(ax, 'plot')  
    assert ax.get_xlabel() == 'I'
    assert ax.get_ylabel() == 'Q'
    plt.close(fig)

@pytest.mark.parametrize("z,A,B,R", [
    (np.array(['c']), 0.0, 0.0, 5.0),
    (np.array([1+1j, 2+2j]), 'a', 0.0, 5.0),
    (np.array([1+1j, 2+2j]), 0.0, 'b', 5.0),
    (np.array([1+1j, 2+2j]), 0.0, 0.0, 'c'),
])  
def test_plot_circle_invalid_input(z, A, B, R):
    with pytest.raises(Exception):
        plot.plot_circfit(z, A, B, R)

################################################################################
################################ plot_gain_fit #################################
################################################################################    
@pytest.mark.parametrize("f,z,mask,p_amp,p_phase", [
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     [0, 0, 0], [0, 1])
])  
def test_plot_gain_fit(f, z, mask, p_amp, p_phase):
    fig, axs = plot.plot_gain_fit(f, z, mask, p_amp, p_phase)
    assert fig is not None
    assert axs is not None   
    assert hasattr(fig, 'canvas')
    assert len(axs) == 2
    for ax in axs:
        assert hasattr(ax, 'plot')  
    assert axs[0].get_ylabel() == '|S21| (dB)'
    assert axs[1].get_ylabel() == 'Phase'
    plt.close(fig)

@pytest.mark.parametrize("f,z,mask,p_amp,p_phase", [
    ([1, 2, 3], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False],
     [0, 0, 0], [0, 1]),
    (1, [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], 1, [True, True, False, True], [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], 1, [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     1, [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     [0, 0, 0], 1),
    ('a', [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], 'a', [True, True, False, True], [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], 'a', [0, 0, 0], [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     'a', [0, 1]),
    ([1, 2, 3, 4], [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [True, True, False, True],
     [0, 0, 0], 'a'),
])  
def test_plot_gain_fit_invalid_input(f, z, mask, p_amp, p_phase):
    with pytest.raises(Exception):
        plot.plot_gain_fit(f, z, mask, p_amp, p_phase)


def test_plots_draw_into_given_axes():
    """Each plot can draw into existing axes (e.g. one combined figure)."""
    fig, axs = plt.subplots(2, 3)
    f = np.linspace(1e9, 1.001e9, 50)
    z = np.exp(1j * np.linspace(0, 3, 50))
    zt = np.exp(1j * np.random.default_rng(0).normal(0, 0.01, 20_000))
    out_fig, out_axs = plot.plot_s21(f, z, zt=zt, axs=axs[0, :2], max_points=300)
    assert out_fig is fig and list(out_axs) == list(axs[0, :2])
    noise = [l for l in axs[0, 1].lines if l.get_label() == 'Noise'][0]
    assert len(noise.get_xdata()) == 300
    out_fig, out_ax = plot.plot_circfit(z, 0, 1, zt=zt, ax=axs[0, 2], max_points=None)
    assert out_ax is axs[0, 2] and len(out_ax.lines[-1].get_xdata()) == 20_000
    thetat = np.angle(zt)
    out_fig, out_axs = plot.plot_xcal(np.angle(z), np.angle(z) * 1e-6, z, np.ones(50, bool),
                                      [1e-6, 0], thetat=thetat, zt_cent=zt, std_cutoff=2,
                                      axs=axs[1, :2], max_points=1000)
    # timestream points drawn in both plots: 1000 in total, split by the cutoff
    n_theta = sum(len(l.get_xdata()) for l in axs[1, 0].lines[2:])
    n_iq = sum(len(l.get_xdata()) for l in axs[1, 1].lines[2:4])
    assert n_theta == n_iq == 1000
    with pytest.raises(ValueError, match='expected 2 axes'):
        plot.plot_gain_fit(f, z, np.ones(50, bool), [0, 0, 0], [0, 0], axs=axs[1, 2])
    plt.close(fig)


def test_nonlinear_iq_fit_plot_with_failed_fit():
    """A failed fit (NaN parameters) still plots the data, centered on the sweep."""
    f = np.linspace(1e9, 1.001e9, 50)
    z = np.exp(1j * np.linspace(0, 3, 50))
    fig, axs = plot.plot_nonlinear_iq_fit(f, z, np.full(8, np.nan), None)
    x = axs[1].lines[0].get_xdata()
    assert np.all(np.isfinite(x)) and abs(np.mean(x)) < 1
    plt.close(fig)
