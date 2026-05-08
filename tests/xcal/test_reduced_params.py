from citkid.xcal.reduced_params import (
    get_sxx_reduced,
    get_sfactor,
    get_sxx_reduced_default_freqs,
    get_sfactor_reduced_default_freqs,
    _freqs,
)
import pytest
import numpy as np

# Wide frequency array covering all default frequencies with 20% margin
_f_wide = np.logspace(-2, 3, 10000)  # 0.01 to 1000 Hz
_sxx_flat = np.ones_like(_f_wide)
_spar_flat = np.full_like(_f_wide, 2.0)
_sper_flat = np.full_like(_f_wide, 1.0)

################################################################################
############################# get_sxx_reduced ##################################
################################################################################

@pytest.mark.parametrize("freq", [0.1, 0.3, 1, 3, 10, 30, 100, 300])
def test_get_sxx_reduced_flat(freq):
    # Flat PSD = 1 everywhere → mean over any band should equal 1
    result = get_sxx_reduced(_f_wide, _sxx_flat, freq)
    assert np.isclose(result, 1.0)

def test_get_sxx_reduced_returns_mean():
    # Verify the function computes mean of sxx where f is within 20% of freq
    f = np.linspace(1, 20, 1000)
    sxx = np.arange(1000, dtype=np.float64)
    freq = 10.0
    fmin, fmax = freq * 0.8, freq * 1.2
    mask = (f > fmin) & (f < fmax)
    expected = np.mean(sxx[mask])
    assert np.isclose(get_sxx_reduced(f, sxx, freq), expected)

def test_get_sxx_reduced_accepts_lists():
    result = get_sxx_reduced(list(_f_wide), list(_sxx_flat), 10.0)
    assert np.isclose(result, 1.0)

def test_get_sxx_reduced_returns_scalar():
    result = get_sxx_reduced(_f_wide, _sxx_flat, 10.0)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)

# Input validation
@pytest.mark.parametrize("f,sxx", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])),      # f longer than sxx
    (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),      # sxx longer than f
])
def test_get_sxx_reduced_shape_mismatch(f, sxx):
    with pytest.raises(ValueError, match="same shape"):
        get_sxx_reduced(f, sxx, 2.0)

@pytest.mark.parametrize("freq", [0, -1.0, -100.0])
def test_get_sxx_reduced_freq_nonpositive(freq):
    with pytest.raises(ValueError, match="positive scalar"):
        get_sxx_reduced(_f_wide, _sxx_flat, freq)

@pytest.mark.parametrize("freq", [np.array([10.0]), np.array([10.0, 20.0]), [10.0]])
def test_get_sxx_reduced_freq_not_scalar(freq):
    with pytest.raises(ValueError, match="positive scalar"):
        get_sxx_reduced(_f_wide, _sxx_flat, freq)

def test_get_sxx_reduced_freq_too_low():
    # freq * 0.8 = 4.0 < min(f) = 5.0 → invalid
    f = np.linspace(5.0, 50.0, 1000)
    sxx = np.ones_like(f)
    with pytest.raises(ValueError, match="Invalid freq"):
        get_sxx_reduced(f, sxx, 5.0)

def test_get_sxx_reduced_freq_too_high():
    # freq * 1.2 = 12.0 > max(f) = 10.0 → invalid
    f = np.linspace(1.0, 10.0, 1000)
    sxx = np.ones_like(f)
    with pytest.raises(ValueError, match="Invalid freq"):
        get_sxx_reduced(f, sxx, 10.0)

def test_get_sxx_reduced_freq_just_inside_range():
    # freq * 0.8 exactly equals min positive f → should NOT raise
    f = np.linspace(8.0, 100.0, 1000)
    sxx = np.ones_like(f)
    result = get_sxx_reduced(f, sxx, 10.0)
    assert np.isclose(result, 1.0)

################################################################################
############################### get_sfactor ####################################
################################################################################

@pytest.mark.parametrize("freq", [0.1, 0.3, 1, 3, 10, 30, 100, 300])
def test_get_sfactor_flat(freq):
    # spar - sper = 2 - 1 = 1 everywhere → sfactor = 1
    result = get_sfactor(_f_wide, _spar_flat, _sper_flat, freq)
    assert np.isclose(result, 1.0)

def test_get_sfactor_returns_mean_difference():
    f = np.linspace(1, 20, 1000)
    spar = np.arange(1000, dtype=np.float64) * 2.0
    sper = np.arange(1000, dtype=np.float64)
    freq = 10.0
    fmin, fmax = freq * 0.8, freq * 1.2
    mask = (f > fmin) & (f < fmax)
    expected = np.mean(spar[mask] - sper[mask])
    assert np.isclose(get_sfactor(f, spar, sper, freq), expected)

def test_get_sfactor_zero_difference():
    # spar == sper → sfactor = 0
    result = get_sfactor(_f_wide, _sxx_flat, _sxx_flat, 10.0)
    assert np.isclose(result, 0.0)

def test_get_sfactor_accepts_lists():
    result = get_sfactor(list(_f_wide), list(_spar_flat), list(_sper_flat), 10.0)
    assert np.isclose(result, 1.0)

def test_get_sfactor_returns_scalar():
    result = get_sfactor(_f_wide, _spar_flat, _sper_flat, 10.0)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)

# Input validation
@pytest.mark.parametrize("f,spar,sper", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),  # spar too short
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])),  # sper too short
    (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),  # f too short
])
def test_get_sfactor_shape_mismatch(f, spar, sper):
    with pytest.raises(ValueError, match="same shape"):
        get_sfactor(f, spar, sper, 2.0)

@pytest.mark.parametrize("freq", [0, -1.0, -100.0])
def test_get_sfactor_freq_nonpositive(freq):
    with pytest.raises(ValueError, match="positive scalar"):
        get_sfactor(_f_wide, _spar_flat, _sper_flat, freq)

@pytest.mark.parametrize("freq", [np.array([10.0]), np.array([10.0, 20.0]), [10.0]])
def test_get_sfactor_freq_not_scalar(freq):
    with pytest.raises(ValueError, match="positive scalar"):
        get_sfactor(_f_wide, _spar_flat, _sper_flat, freq)

def test_get_sfactor_freq_too_low():
    f = np.linspace(5.0, 50.0, 1000)
    spar, sper = np.ones_like(f), np.ones_like(f)
    with pytest.raises(ValueError, match="Invalid freq"):
        get_sfactor(f, spar, sper, 5.0)

def test_get_sfactor_freq_too_high():
    f = np.linspace(1.0, 10.0, 1000)
    spar, sper = np.ones_like(f), np.ones_like(f)
    with pytest.raises(ValueError, match="Invalid freq"):
        get_sfactor(f, spar, sper, 10.0)

def test_get_sfactor_freq_just_inside_range():
    f = np.linspace(8.0, 100.0, 1000)
    spar, sper = np.full_like(f, 3.0), np.full_like(f, 1.0)
    result = get_sfactor(f, spar, sper, 10.0)
    assert np.isclose(result, 2.0)

################################################################################
###################### get_sxx_reduced_default_freqs ##########################
################################################################################

def test_get_sxx_reduced_default_freqs_returns_dict():
    result = get_sxx_reduced_default_freqs(_f_wide, _sxx_flat)
    assert isinstance(result, dict)

def test_get_sxx_reduced_default_freqs_keys():
    result = get_sxx_reduced_default_freqs(_f_wide, _sxx_flat)
    expected_keys = {f'sxx_{freq}' for freq in _freqs}
    assert set(result.keys()) == expected_keys

def test_get_sxx_reduced_default_freqs_length():
    result = get_sxx_reduced_default_freqs(_f_wide, _sxx_flat)
    assert len(result) == len(_freqs)

def test_get_sxx_reduced_default_freqs_values_match_individual():
    result = get_sxx_reduced_default_freqs(_f_wide, _sxx_flat)
    for freq in _freqs:
        expected = get_sxx_reduced(_f_wide, _sxx_flat, freq)
        assert np.isclose(result[f'sxx_{freq}'], expected)

def test_get_sxx_reduced_default_freqs_flat():
    result = get_sxx_reduced_default_freqs(_f_wide, _sxx_flat)
    for key, val in result.items():
        assert np.isclose(val, 1.0), f"Expected 1.0 for {key}, got {val}"

################################################################################
######################### get_sfactor_default_freqs ############################
################################################################################

def test_get_sfactor_reduced_default_freqs_returns_dict():
    result = get_sfactor_reduced_default_freqs(_f_wide, _spar_flat, _sper_flat)
    assert isinstance(result, dict)

def test_get_sfactor_reduced_default_freqs_keys():
    result = get_sfactor_reduced_default_freqs(_f_wide, _spar_flat, _sper_flat)
    expected_keys = {f'sfactor_{freq}' for freq in _freqs}
    assert set(result.keys()) == expected_keys

def test_get_sfactor_reduced_default_freqs_length():
    result = get_sfactor_reduced_default_freqs(_f_wide, _spar_flat, _sper_flat)
    assert len(result) == len(_freqs)

def test_get_sfactor_reduced_default_freqs_values_match_individual():
    result = get_sfactor_reduced_default_freqs(_f_wide, _spar_flat, _sper_flat)
    for freq in _freqs:
        expected = get_sfactor(_f_wide, _spar_flat, _sper_flat, freq)
        assert np.isclose(result[f'sfactor_{freq}'], expected)

def test_get_sfactor_reduced_default_freqs_flat():
    result = get_sfactor_reduced_default_freqs(_f_wide, _spar_flat, _sper_flat)
    for key, val in result.items():
        assert np.isclose(val, 1.0), f"Expected 1.0 for {key}, got {val}"
