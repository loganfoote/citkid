from citkid.noise import psd
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