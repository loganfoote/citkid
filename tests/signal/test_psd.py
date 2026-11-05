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

@pytest.mark.parametrize("x,dt,return_freqs,psd_exp", [
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
def test_get_psd(x, dt, return_freqs, psd_exp):
    psd_exp = np.array(psd_exp, dtype = np.float64)
    if return_freqs:
        f, y = psd.get_psd(x, dt, return_freqs = True)
        assert f.shape == y.shape 
        assert len(y) == (len(x) // 2 + 1)
        assert 1 / (len(x) * f[1]) == dt
        assert (len(x) // 2) / (len(x) * dt) == f[-1]
    else:
        y = psd.get_psd(x, dt, return_freqs = False)
    assert np.allclose(y, np.array(psd_exp, dtype = np.float64), 
                        rtol = 1e-7, atol = 1e-20, equal_nan = True)
    
    
@pytest.mark.parametrize("x,dt", [
    (['a'], 1),  # non-numeric x
    ([1, 2, 3], 'a'),  # non-numeric dt
    ([], 1),  # empty input
])
def test_get_psd_invalid_input(x, dt):
    with pytest.raises(Exception):
        psd.get_psd(x, dt, return_freqs = False)

def test_get_psd_singleton():
    x = [2.0]
    dt = 0.5
    y = psd.get_psd(x, dt, return_freqs = False)
    assert np.allclose(y, np.array([4.0]))

################################################################################
#################################### get_csd ##################################
################################################################################
def test_get_csd_matches_psd_identical_signals():
    x = np.sin(2 * np.pi * np.linspace(0, 1, 64))
    dt = 1 / 64
    f_psd, y_psd = psd.get_psd(x, dt, return_freqs = True)
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


def test_bin_psd_does_not_modify_data_list():
    """
    Check that list entries are not replaced by arrays in the caller's list.
    """
    f = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    data = [list(f), [1, 2, 3, 4, 5, 6]]
    psd.bin_psd(f, data, nbins = 3, fmin = 2.0)
    assert isinstance(data[0], list) and isinstance(data[1], list)


def test_bin_psd_filters_pulse_tubes_but_not_frequencies():
    """
    Check that pulse-tube filtering applies to every entry but the first
    (the frequency array).
    """
    f = np.linspace(0.01, 5, 500)
    y = np.ones_like(f)
    y[np.argmin(abs(f - 2.0))] = 1e3
    f_out, y_out = psd.bin_psd(
        f, [f, y], nbins = 20, fmin = 3, filter_pt_n = 3,
        pt_frequency = 1.0)
    low = f < 3  # unbinned, so the filtered values come back directly
    np.testing.assert_allclose(f_out[:low.sum()], f[low])
    assert np.max(y_out) < 2

################################################################################
################################### filter_pt #################################
################################################################################
def test_filter_pt_single_spike():
    f = np.array([0.5, 0.89, 1.0, 1.11, 1.5])
    y = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
    y_filt = psd.filter_pt(f, y, n = 2, pt_frequency = 1.0)
    # Only the 1.0 bin is replaced, by the mean of its two neighbors
    expected = y.copy()
    expected[2] = np.mean([y[1], y[3]])
    assert np.allclose(y_filt, expected)


def test_filter_pt_filters_n_harmonics_without_modifying_input():
    """
    Check that harmonics 1 through n are filtered (not n - 1) and that the
    input array is left unchanged.
    """
    f = np.linspace(0, 4, 401)
    y = np.ones_like(f)
    spikes = [np.argmin(abs(f - k)) for k in (1, 2, 3)]
    y[spikes] = 100.0
    y_in = y.copy()

    y_filt = psd.filter_pt(f, y, n = 2, pt_frequency = 1.0)

    np.testing.assert_array_equal(y, y_in)
    assert y_filt is not y
    np.testing.assert_allclose(y_filt[spikes[:2]], 1.0)
    assert y_filt[spikes[2]] == 100.0  # 3rd harmonic not requested


@pytest.mark.parametrize("f0_idx", [1, 98])
def test_filter_pt_near_array_edges_uses_available_side(f0_idx):
    """
    Check that a spike window too close to either end is filled from the
    side that has enough neighbors, instead of crashing.
    """
    f = np.linspace(0, 1, 100) + 1.0 - (f0_idx / 99)
    y = np.linspace(1, 2, 100)
    k = np.argmin(abs(f - 1.0))
    ix = np.where(abs(f - 1.0) < 0.1)[0]
    y[ix] = 100.0

    y_filt = psd.filter_pt(f, y, n = 1, pt_frequency = 1.0)

    assert np.all(y_filt[ix] < 3)
    assert y_filt[k] != 100.0