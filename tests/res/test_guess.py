import pytest 
from citkid.res import guess, funcs
import numpy as np 
from scipy.ndimage import gaussian_filter

################################################################################
############################# guess_p0_nonlinear_iq ############################
################################################################################
def test_guess_p0_nonlinear_iq_shape():
    f = np.linspace(1e9, 1.1e9, 5)
    z = np.ones_like(f, dtype = np.complex128)
    p0 = guess.guess_p0_nonlinear_iq(f, z)
    assert isinstance(p0, list)
    assert len(p0) == 8

@pytest.mark.parametrize("f,z", [
    (np.linspace(1e9, 1.1e9, 5), np.ones(5, dtype = np.complex128)),
    (np.linspace(2e9, 2.2e9, 10), np.ones(10, dtype = np.complex128))
])
def test_guess_p0_nonlinear_iq_values(f, z, monkeypatch):
    monkeypatch.setattr(guess, "_guess_phi_amp", lambda z, z0: (0.1, 0.5))
    monkeypatch.setattr(guess, "_guess_Qr", lambda f, z, z0, phi, amp: 10000.)
    monkeypatch.setattr(guess, "_guess_a", lambda f, z, z0, phi, amp: 0.2)
    monkeypatch.setattr(guess, "_guess_fr", 
                        lambda f, z, z0, phi, amp, a, Qr: 1.05e9)
    p0 = guess.guess_p0_nonlinear_iq(f, z)

    assert pytest.approx(
        np.mean(np.roll(z, 2)[:4]), abs = 1e-3
        ) == p0[5] + 1j * p0[6]
    assert p0[7] == 0  # tau
    assert p0[2] == 0.5  # amp
    assert p0[3] == 0.1  # phi
    assert p0[1] == 10000.  # Qr
    assert p0[4] == 0.2  # a
    assert p0[0] == 1.05e9  # fr

@pytest.mark.parametrize("f,z", [
    (np.array([[1e9, 1.1e9]]), np.ones((1,2), dtype = np.complex128)), # f not 1D
    (np.array([1e9, 1.1e9]), np.array([[1.+0.j, 1.+0.j]])), # z not 1D
    (np.array([1e9, 1.1e9]), np.array([1.+0.j])), # f and z different shapes
    ("not an array", np.ones(2, dtype = np.complex128)), # f not array-like
    (np.array([1e9, 1.1e9]), "not an array"), # z not array-like    
    (1, np.ones(2, dtype = np.complex128)), # f not array-like
    (np.array([1e9, 1.1e9]), 1), # z not array-like
])
def test_guess_p0_nonlinear_iq_invalid_inputs(f, z):
    with pytest.raises(ValueError):
        guess.guess_p0_nonlinear_iq(f, z)

################################################################################
################################# test helpers #################################
################################################################################
# No input validation here, since it is only called internally 
@pytest.mark.parametrize("f,z", [
    (np.linspace(1e9, 1.1e9, 5), np.random.rand(5) + 1j * np.random.rand(5)),
    (np.linspace(0.5e9, 1.5e9, 20), 
     gaussian_filter(np.abs(np.sin(np.linspace(0, 10, 20))), sigma = 1) + 0.1j),
    (np.linspace(2e9, 2.2e9, 10), 
     np.exp(1j * np.linspace(0, np.pi, 10))),
    (np.linspace(3e9, 3.3e9, 15), 
     np.cos(np.linspace(0, 5, 15)) + 1j * np.sin(np.linspace(0, 5, 15))),
    (np.linspace(4e9, 4.4e9, 8), funcs.nonlinear_iq(
        np.linspace(4e9, 4.4e9, 8), 4.2e9, 5000., 0.7, 0.2, 
        0.3, 1.0, 0.0, 2e-9, True)),
])
def test_guess_helper_shapes(f, z):
    z0 = np.mean(np.roll(z, 2)[:4])
    # _guess_phi_amp
    phi_guess, amp_guess = guess._guess_phi_amp(z, z0)
    assert isinstance(phi_guess, float)
    assert isinstance(amp_guess, float)
    assert amp_guess >= 0.0
    assert amp_guess <= 1.0
    assert phi_guess >= -np.pi
    assert phi_guess <= np.pi

    # _guess_Qr
    Qr_guess = guess._guess_Qr(f, z, z0, phi_guess, amp_guess)
    assert isinstance(Qr_guess, float)
    assert Qr_guess >= 0.0
    assert Qr_guess <= 1e9

    # _guess_a
    a_guess = guess._guess_a(f, z, z0, phi_guess, amp_guess)
    assert isinstance(a_guess, float)
    assert a_guess >= 0.0
    assert a_guess <= 1.0

    # _guess_fr
    fr_guess = guess._guess_fr(f, z, z0, phi_guess, amp_guess, 
                               a_guess, Qr_guess)
    assert isinstance(fr_guess, float)
    assert fr_guess >= min(f)
    assert fr_guess <= max(f)

def test_guess_phi_amp_clamps_amp(monkeypatch):
    z = np.ones(5, dtype = np.complex128)
    monkeypatch.setattr(guess.circle, "fit_iq_circle", 
                        lambda z: (1. + 0.j, 1.0))
    phi_guess, amp_guess = guess._guess_phi_amp(z, 1. + 0.j)
    assert min(abs(phi_guess), abs(abs(phi_guess) - np.pi)) <= 1e-12
    assert amp_guess == pytest.approx(1. - 1e-6)

def test_guess_a_clamps_to_one(monkeypatch):
    f = np.linspace(1e9, 1.1e9, 5)
    z = np.ones(5, dtype = np.complex128)
    monkeypatch.setattr(guess.np, "interp", 
                        lambda x, xbin, abin: 1.5)
    a_guess = guess._guess_a(f, z, 1. + 0.j, 0.0, 1.0)
    assert a_guess == 1.0

def test_guess_Qr_flips_when_off_res_greater(monkeypatch):
    f = np.linspace(1e9, 1.1e9, 10)
    z = np.ones(10, dtype = np.complex128)
    z[:2] = -1
    z[-2:] = -1
    z[3:7] = 1
    monkeypatch.setattr(guess, "gaussian_filter", 
                        lambda arr, sigma: arr)
    captured = {}

    def fake_get_peak_fwhm(f_in, z_in):
        captured["max"] = np.max(z_in)
        return f_in[len(f_in) // 2], None, 1.0

    monkeypatch.setattr(guess, "get_peak_fwhm", fake_get_peak_fwhm)
    Qr_guess = guess._guess_Qr(f, z, 1. + 0.j, 0.0, 1.0)
    assert captured["max"] <= 0.0
    assert isinstance(Qr_guess, float)

    