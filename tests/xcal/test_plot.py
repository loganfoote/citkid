from citkid.xcal import plot
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Not testing the actual plots, just that they generate without error

################################################################################
################################# plot_circle ##################################
################################################################################
@pytest.mark.parametrize("z,A,B,R", [
    (np.array([1+1j, 2+2j, 3+3j]), 0.0, 0.0, 5.0),
    (np.array([-1-1j, -2-2j, -3-3j]), 1.0, 1.0, 4.0),
])  
def test_plot_circle(z, A, B, R):
    fig, ax = plot.plot_circle(z, A, B, R)
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
        plot.plot_circle(z, A, B, R)

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