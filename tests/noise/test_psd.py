from citkid.signal import psd
import pytest
import numpy as np

################################################################################
#################################### get_psd ###################################
################################################################################
x = np.arange(0, 100, 0.1)
y = np.sin(2 * np.pi * x) 
y_exp = np.zeros(len(y) // 2 + 1) 
y_exp[100] = 50

@pytest.mark.parametrize("x,dt,get_frequencies,psd_exp", [
    ([1, 1], 1/4, False, [1, 0]),
    ([1, 1], 1/4, True, [1, 0]),
    ([1, 1, 1], 1/4, True, [1.5, 0]),
    ([1, 1, 1, 1], 1/8, False, [1, 0, 0]), 
    (np.ones(64), 1/128, False, [1] + [0]*32),  
    (np.ones(64), 1/128, True, [1] + [0]*32),  
    (y, 0.1, False, y_exp),
    (y, 0.1, True, y_exp),
    ([np.nan, 1, 2], 1, False, [np.nan, np.nan]),  # nan input
    ([np.nan, 1, 2], 1, True, [np.nan, np.nan]),  # nan input
    ([0, 1, 2], np.nan, False, [np.nan, np.nan]),  # nan input
])
def test_get_psd(x, dt, get_frequencies, psd_exp):
    psd_exp = np.array(psd_exp, dtype = np.float64)
    if get_frequencies:
        f, y = psd.get_psd(x, dt, get_frequencies = True)
        assert f.shape == y.shape 
        assert len(y) == (len(x) // 2 + 1)
        assert 1 / (len(x) * f[1]) == dt
        assert (len(x) // 2) / (len(x) * dt) == f[-1]
    else:
        y = psd.get_psd(x, dt, get_frequencies = False)
    assert np.allclose(y, np.array(psd_exp, dtype = np.float64), 
                        rtol = 1e-7, atol = 1e-20, equal_nan = True)
    
    
@pytest.mark.parametrize("x,dt", [
    (['a'], 1),  # non-numeric x
    ([1, 2, 3], 'a'),  # non-numeric dt
    ([], 1),  # empty input
])
def test_get_psd_invalid_input(x, dt):
    with pytest.raises(Exception):
        psd.get_psd(x, dt, get_frequencies = False)

def test_get_psd_singleton():
    x = [2.0]
    dt = 0.5
    y = psd.get_psd(x, dt, get_frequencies = False)
    assert np.allclose(y, np.array([4.0]))

################################################################################
#################################### get_csd ##################################
################################################################################
def test_get_csd_matches_psd_identical_signals():
    x = np.sin(2 * np.pi * np.linspace(0, 1, 64))
    dt = 1 / 64
    f_psd, y_psd = psd.get_psd(x, dt, get_frequencies = True)
    f_csd, y_csd = psd.get_csd(x, x, dt)
    assert np.allclose(f_psd, f_csd)
    assert np.allclose(y_psd, y_csd)

def test_get_csd_zero_second_signal():
    x1 = np.random.randn(32)
    x2 = np.zeros_like(x1)
    f, y = psd.get_csd(x1, x2, 0.1)
    assert np.allclose(y, 0)
    assert len(f) == len(y)

################################################################################
#################################### bin_psd ##################################
################################################################################
def test_bin_psd_preserves_low_freqs():
    f = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    d0 = np.array([10, 20, 30, 40, 50, 60])
    d1 = np.array([1, 2, 3, 4, 5, 6])
    out = psd.bin_psd(f, [d0, d1], nbins = 3, fmin = 2.0)
    assert len(out) == 2
    # Values below fmin should be preserved at the front
    assert np.array_equal(out[0][:2], d0[:2])
    assert np.array_equal(out[1][:2], d1[:2])
    # Output length should be between len(f<fmin) and len(f)
    assert len(out[0]) >= 2 and len(out[0]) <= len(f)

################################################################################
################################### filter_pt #################################
################################################################################
def test_filter_pt_single_spike():
    f = np.array([0.5, 0.89, 1.0, 1.11, 1.5])
    y = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
    y_filt = psd.filter_pt(f, y.copy(), n = 2, pt_frequency = 1.0)
    # Only the 1.0 bin should be replaced by mean of neighbor and itself
    expected = y.copy()
    expected[2] = np.mean([y[1:2], y[2:3]], axis = 0)
    assert np.allclose(y_filt, expected)