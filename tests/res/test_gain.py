from citkid.res import gain
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

################################################################################
################################# remove_gain ##################################
################################################################################
f0, z0 = np.linspace(1e6, 1.1e6, 100), np.ones(100, dtype = np.complex128)
f1 = np.array([0, 1, 2], dtype = np.float64)
z1 = np.ones(3, dtype = np.complex128)
f2 = np.array([0, 1, 2], dtype = np.float64)
z2 = np.ones(3, dtype = np.complex128)
z2[2] = np.nan
@pytest.mark.parametrize("f,z,p_amp,p_phase,expected", [
    (f0, z0, [0, 0, 0, 0], [0, 0, 0], z0),
    (1e9, z0, [0, 0, 0, 0], [0, 0, 0], z0),
    (f0, z0[0], [0, 0, 0, 0], [0, 0, 0], np.ones(f0.shape) * z0[0]),
    (f0, z0, [0], [0], z0),
    (f1, z1, [20], [0], z1 / 10),
    (f1, z1, [0], [np.pi], z1 * -1),
    (f1, z1, [0], [np.pi / 2], z1 * np.exp(-1j * np.pi / 2)),
    (f1, z1, [0], [np.pi, 0], [1, -1, 1]),
    ([], [], [], [], []),
    (f2, z2, [0], [0], z2), # single nan in z returns single nan
])
def test_remove_gain(f, z, p_amp, p_phase, expected):
    z_rmvd = gain.remove_gain(f, z, p_amp, p_phase)
    check_out_shape = (z_rmvd.shape == np.asarray(z).shape) 
    check_out_shape = check_out_shape or (z_rmvd.shape == np.asarray(f).shape)
    assert  check_out_shape
    assert z_rmvd.dtype == np.complex128
    np.testing.assert_allclose(z_rmvd, expected)

def test_remove_gain_numerical_stability():
    f = np.linspace(0, 1e9, 1000)
    z = np.exp(1j * 2 * np.pi * f * 1e-6)
    z_rmvd = gain.remove_gain(f, z, np.array([1e-12, 1]),
                              np.array([1e-9, 0]))
    assert np.all(np.isfinite(z_rmvd))

@pytest.mark.parametrize("f,z,p_amp,p_phase", [
    ([], [], 1, []),
    ([], [], [], 1),
    (['a'], [0], [0], [0]),
    ([0, 1], [0, 1, 2], [0], [0])
])
def test_remove_gain_invalid_input(f, z, p_amp, p_phase):
    with pytest.raises(Exception):
        gain.remove_gain(f, z, p_amp, p_phase)

################################################################################
################################### fit_gain ###################################
################################################################################
@pytest.mark.parametrize("f,z,fr_spans,plotq,p_amp_exp,p_phase_exp", [
    ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [], False, [0, 0, 0], [0, 0]),
    ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [], True, [0, 0, 0], [0, 0]),
    ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [(2, 1)], False, [0, 0, 0], [0, 0]),
    ([0, 1, 2, 3, 4], [1, 1, 5, 1, 1], [(2, 1)], False, [0, 0, 0], [0, 0]),
    ([0, 1, 2, 3, 4], [10, 10, 10, 10, 10], [(2, 1)], False, [0, 0, 20], [0, 0]),
    ([0, 1, 2, 3, 4], None, [(2, 1)], False, [0, 20, 0], [0, 0]),
    ([0, 1, 2, 3, 4], None, [(2, 1)], False, [0, 20, 0], [0, 1e-8]),
    ([0, 1, 2, 3, 4], None, [(2, 1)], False, [0, 20, 0], [1e-8, -1e-8]),
])
def test_fit_gain(f, z, fr_spans, plotq, p_amp_exp, p_phase_exp):
    if z is None:
        z = 10 ** (np.polyval(p_amp_exp, f) / 20)
        z = z * np.exp(1j * np.polyval(p_phase_exp, f))
    p_amp, p_phase, (fig, axs) = gain.fit_gain(f, z, fr_spans, plotq)
    if plotq:
        assert isinstance(fig, plt.Figure)
        assert len(axs) == 2
        for ax in axs:
            assert isinstance(ax, plt.Axes)
        plt.close(fig)
    else:
        assert fig is None
        assert axs is None
    assert p_amp.dtype == np.float64
    assert p_phase.dtype == np.float64
    assert p_amp.shape == (3,)
    assert p_phase.shape == (2,)
    np.testing.assert_allclose(p_amp, p_amp_exp, atol = 1e-12)
    np.testing.assert_allclose(p_phase, p_phase_exp, atol = 1e-12)

@pytest.mark.parametrize("f,z,fr_spans,plotq", [
    ([0, 1, 2, 3, 4], [10, 10, 10, 10, 10], [1], False),  # fr_spans not list of tuples
    ([0, 1, 2], [1, 1], [], False),                       # f and z different lengths
    (['a'], [0], [], False),                              # f not numeric
    ([0], ['a'], [], False),                              # z not numeric
])
def test_fit_gain_invalid_input(f, z, fr_spans, plotq):
    with pytest.raises(Exception):
        gain.fit_gain(f, z, fr_spans, plotq)