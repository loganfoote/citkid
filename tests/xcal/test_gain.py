from citkid.xcal import gain
import pytest
import numpy as np
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
@pytest.mark.parametrize("f,z,fr_spans,p_amp_exp,p_phase_exp,mask_exp", [
    ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [], [0, 0, 0], [0, 0], [True] * 5),
    ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [], [0, 0, 0], [0, 0], [True] * 5),
    ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [(2, 1)], [0, 0, 0], [0, 0],
     [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], [1, 1, 5, 1, 1], [(2, 1)], [0, 0, 0], [0, 0],
     [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], [10, 10, 10, 10, 10], [(2, 1)], [0, 0, 20], [0, 0],
     [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], None, [(2, 1)], [0, 20, 0], [0, 0],
     [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], None, [(2, 1)], [0, 20, 0], [0, 1e-8],
     [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], None, [(2, 1)], [0, 20, 0], [1e-8, -1e-8],
     [True, True, False, True, True]),
])
def test_fit_gain(f, z, fr_spans, p_amp_exp, p_phase_exp, mask_exp):
    if z is None:
        z = 10 ** (np.polyval(p_amp_exp, f) / 20)
        z = z * np.exp(1j * np.polyval(p_phase_exp, f))
    p_amp, p_phase, mask = gain.fit_gain(f, z, fr_spans)
    assert p_amp.dtype == np.float64
    assert p_phase.dtype == np.float64
    assert p_amp.shape == (3,)
    assert p_phase.shape == (2,)
    np.testing.assert_allclose(p_amp, p_amp_exp, atol = 1e-12)
    np.testing.assert_allclose(p_phase, p_phase_exp, atol = 1e-12)
    np.testing.assert_equal(mask, mask_exp)

@pytest.mark.parametrize("f,z,fr_spans", [
    ([0, 1, 2, 3, 4], [10, 10, 10, 10, 10], [1]),  # fr_spans not list of tuples
    ([0, 1, 2], [1, 1], []),                       # f and z different lengths
    (['a'], [0], []),                              # f not numeric
    ([0], ['a'], []),                              # z not numeric
])
def test_fit_gain_invalid_input(f, z, fr_spans):
    with pytest.raises(Exception):
        gain.fit_gain(f, z, fr_spans)

################################################################################
################################## get_res_mask ################################
################################################################################
@pytest.mark.parametrize("f,fr_spans,mask_exp", [
    ([0, 1, 2, 3, 4], [], [True] * 5),
    ([0, 1, 2, 3, 4], [(2, 1)], [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], [(2, 1)], [True, True, False, True, True]),
    ([0, 1, 2, 3, 4], [(1, 2)], [False, False, False, True, True]),
    ([0, 1, 2 + 1e-9, 3, 4], [(1, 2)], [False, False, True, True, True]),
    ([0, 1, 2, 3, 4], [(0, 5)], [False, False, False, True, True]),
    ([0, 1, 2, 3, 4], [(0, 10)], [False, False, False, False, False]),
    ([0, 1, 2, 3, 4], [(7, 1)], [True] * 5),
    ([0, 1, 2, 3, 4], [(1, 1), (3, 1)], [True, False, True, False, True]),
    ([0, 1, 2, 3, 4], [(1, 1), (2.5, 2)], [True, False, False, False, True]),
    ([0, 1, 2, 3, 4], [(1, 3), (2, 2)], [False, False, False, False, True]),
])
def test_get_res_mask(f, fr_spans, mask_exp):
    for m in [1e-12, 1, 1e6, 1e9, 1e12]:
        mask = gain.get_res_mask([fi * m for fi in f], 
                                [(f[0] * m, f[1] * m) for f in fr_spans])
        np.testing.assert_array_equal(mask, mask_exp)

@pytest.mark.parametrize("f,fr_spans", [
    ([0, 1, 2], [1]),  # fr_spans not list of tuples
    (['a'], []),       # f not numeric  
    ([0], ['a']),      # fr_spans not numeric
])
def test_get_res_mask_invalid_input(f, fr_spans):
    with pytest.raises(Exception):
        gain.get_res_mask(f, fr_spans)

################################################################################
################################ make_fr_spans #################################
################################################################################
# Neet to finish writing tests here
@pytest.mark.parametrize("fres_all,qres_all,fg,fr_spans_exp", [
    ([1e9, 2e9], [1e9, 1e9], [1e9, 1.1e9], [[1e9, 1]]), 
    ([1e9, 2e9, 3e9], [1e9, 2e9, 1e9], [1e9, 1.5e9, 2.5e9], [[1e9, 1], [2e9, 1]]), 
    ([1e9], [1e9], [0.5e9, 1.5e9], [[1e9, 1]]), 
    ([1e9], [1e9], [0.5e9, 0.9e9], []),
    ([], [], [1e9, 1.1e9], []),
    
])
def test_make_fr_spans(fres_all, qres_all, fg, fr_spans_exp):
    fr_spans = gain.make_fr_spans(fres_all, qres_all, fg)
    assert np.allclose(fr_spans, fr_spans_exp)

@pytest.mark.parametrize("fres_all,qres_all,fg", [
    ([1e9, 2e9], [1e9], [1e9, 1.1e9]),  # fres_all and qres_all different lengths
    (['a'], [1e9], [1e9, 1.1e9]),     # fres_all not numeric
    ([1e9], ['a'], [1e9, 1.1e9]),     # qres_all not numeric
    ([1e9], [1e9], ['a']),            # fg not numeric  
    ([2e9, 1e9], [1e9, 1e9], [1.1e9, 1e9]),  # fg not sorted
])
def test_make_fr_spans_invalid_input(fres_all, qres_all, fg):
    with pytest.raises(Exception):
        gain.make_fr_spans(fres_all, qres_all, fg)