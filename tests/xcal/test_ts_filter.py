import pytest
import numpy as np
from citkid.xcal import ts_filt

################################################################################
# For the following functions, we will test that the output shape matches input 
# shape and trust the underlying scipy filter implementation for correctness.
################################################################################

################################################################################
################################# bandpass_filter ##############################
################################################################################
x = np.random.randn(1000)
y = np.array([0, 1, 2, 3, 4, 5, 6])
z = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("x,dt,f_low,f_high,order", [  
    (x, 0.1, 0.5, 2.0, 10),
    (x, 0.1, 0.5, 2.0, 4),
    (x, 0.2, 0.1, 1.0, 2), 
    (np.array([x, x]), 0.1, 0.5, 2.0, 10),
    (np.array([x, x]), 0.1, 0.5, 2.0, 4),
    (z, 0.1, 0.5, 2.0, 1), # data length = padlen + 1
    (np.array([z, z]), 0.1, 0.5, 2.0, 1), # data length = padlen + 1
])
def test_bandpass_filter(x, dt, f_low, f_high, order):
    filtered_x = ts_filt.bandpass_filter(x, dt, f_low, f_high, order)
    assert filtered_x.shape == x.shape

@pytest.mark.parametrize("x,dt,f_low,f_high,order", [  
    (np.array([0, 1, 0, -1, 0]), 'a', 0.2, 0.5, 1),
    (np.array([1, 2, 3, 4, 5]), 0.2, 'b', 1.0, 1),
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.1, 'c', 1),
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.1, 1.0, 'd'),
    (np.array([1, 2, 3, 4, 5]), 0.2, 1.0, 0.5, 2), # f_low >= f_high
    (np.array([1, 2, 3, 4, 5]), 0.2, -1, 0.5, 2), # negative f_low
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.1, -0.5, 2), # negative f_high
    (np.array([1, 2, 3, 4, 5]), 0.2, 0, 0.5, 2),  # zero f_low
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.1, 0, 2),   # zero f_high
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.1, 10, 2), # f_high > Nyquist
    (np.array([1, 2, 3, 4, 5]), 0.2, 10, 20, 2), # f_low > Nyquist
    (z[2:], 0.1, 0.5, 2.0, 2), # data length < padlen
    (z[1:], 0.1, 0.5, 2.0, 1), # data length == padlen
    (np.array([]), 0.2, 0.1, 0.5, 2), # empty data array, too short to filter
])
def test_bandpass_filter_invalid_input(x, dt, f_low, f_high, order):
    with pytest.raises(Exception):
        ts_filt.bandpass_filter(x, dt, f_low, f_high, order)
    
################################################################################
################################# lowpass_filter ###############################
################################################################################
@pytest.mark.parametrize("x,dt,f_cutoff,order", [
    (x, 0.1, 2.0, 10),
    (x, 0.1, 1.0, 4),
    (x, 0.2, 0.5, 2),
    (np.array([x, x]), 0.1, 2.0, 10),
    (np.array([x, x]), 0.1, 2.0, 4),
    (y, 0.2, 1.0, 1), # data length = padlen + 1
    (np.array([y, y]), 0.2, 1.0, 1), # data length = padlen + 1
    # Needs to work for both 1D and 2D inputs
])
def test_lowpass_filter(x, dt, f_cutoff, order):
    filtered_x = ts_filt.lowpass_filter(x, dt, f_cutoff, order)
    assert filtered_x.shape == x.shape

@pytest.mark.parametrize("x,dt,f_cutoff,order", [
    (np.array([0, 1, 0, -1, 0]), 'a', 1.0, 4),
    (np.array([1, 2, 3, 4, 5]), 0.2, 'b', 2),
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.5, 'c'),
    (np.array([1, 2, 3, 4, 5]), 0.2, 10, 2), # cutoff > Nyquist
    (np.array([1, 2, 3, 4, 5]), 0.2, -1, 2), # negative cutoff
    (np.array([1, 2, 3, 4, 5]), 0.2, 0, 2),  # zero cutoff
    (y[2:], 0.2, 0.1, 2), # data length < padlen
    (y[1:], 0.2, 0.1, 1), # data length == padlen
    (np.array([]), 0.2, 0.1, 2), # empty data array, too short to filter
])
def test_lowpass_filter_invalid_input(x, dt, f_cutoff, order):
    with pytest.raises(Exception):
        ts_filt.lowpass_filter(x, dt, f_cutoff, order)
    
################################################################################
################################# highpass_filter ##############################
################################################################################
@pytest.mark.parametrize("x,dt,f_cutoff,order", [
    (x, 0.1, 0.5, 10),
    (x, 0.1, 0.5, 4),
    (x, 0.2, 0.1, 2),  
    (np.array([x, x]), 0.1, 0.5, 10),
    (y, 0.2, 1.0, 1), # data length = padlen + 1
    (np.array([y, y]), 0.2, 1.0, 1), # data length = padlen + 1
])
def test_highpass_filter(x, dt, f_cutoff, order):
    filtered_x = ts_filt.highpass_filter(x, dt, f_cutoff, order)
    assert filtered_x.shape == x.shape

@pytest.mark.parametrize("x,dt,f_cutoff,order", [
    (np.array([0, 1, 0, -1, 0]), 'a', 0.5, 4),
    (np.array([1, 2, 3, 4, 5]), 0.2, 'b', 2),
    (np.array([1, 2, 3, 4, 5]), 0.2, 0.1, 'c'),
    (np.array([1, 2, 3, 4, 5]), 0.2, 10, 2), # cutoff > Nyquist
    (np.array([1, 2, 3, 4, 5]), 0.2, -1, 2), # negative cutoff
    (np.array([1, 2, 3, 4, 5]), 0.2, 0, 2),  # zero cutoff
    (y[2:], 0.2, 0.1, 2), # data length < padlen
    (y[1:], 0.2, 0.1, 1), # data length == padlen
    (np.array([]), 0.2, 0.1, 2), # empty data array, too short to filter
])
def test_highpass_filter_invalid_input(x, dt, f_cutoff, order):
    with pytest.raises(Exception):
        ts_filt.highpass_filter(x, dt, f_cutoff, order)

################################################################################
################################# get_cutoff ###################################
################################################################################
@pytest.mark.parametrize("dt,f_cutoff,expected", [
    (0.1, 1.0, 0.2),    
    (0.2, 0.5, 0.2),
])
def test_get_cutoff(dt, f_cutoff, expected):
    cutoff = ts_filt.get_cutoff(dt, f_cutoff)
    assert np.isclose(cutoff, expected)

@pytest.mark.parametrize("dt,f_cutoff", [
    ('a', 1.0),    
    (0.1, 'b'),
])
def test_get_cutoff_invalid_input(dt, f_cutoff):
    with pytest.raises(Exception):
        ts_filt.get_cutoff(dt, f_cutoff)