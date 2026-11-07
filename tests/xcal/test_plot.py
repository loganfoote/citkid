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
    out_fig, out_axs = plot.plot_s21(
        f, z, zt=zt, axs=axs[0, :2], max_points=300
    )
    assert out_fig is fig and list(out_axs) == list(axs[0, :2])
    noise = [l for l in axs[0, 1].lines if l.get_label() == 'Noise'][0]
    assert len(noise.get_xdata()) == 300
    out_fig, out_ax = plot.plot_circfit(
        z, 0, 1, zt=zt, ax=axs[0, 2], max_points=None
    )
    assert out_ax is axs[0, 2] and len(out_ax.lines[-1].get_xdata()) == 20_000
    thetat = np.angle(zt)
    out_fig, out_axs = plot.plot_xcal(
        np.angle(z), np.angle(z) * 1e-6, z, np.ones(50, bool), [1e-6, 0],
        thetat=thetat, zt_cent=zt, std_cutoff=2, axs=axs[1, :2],
        max_points=1000
    )
    # timestream points drawn in both plots: 1000 in total, split by the cutoff
    n_theta = sum(len(l.get_xdata()) for l in axs[1, 0].lines[2:])
    n_iq = sum(len(l.get_xdata()) for l in axs[1, 1].lines[2:4])
    assert n_theta == n_iq == 1000
    with pytest.raises(ValueError, match='expected 2 axes'):
        plot.plot_gain_fit(
            f, z, np.ones(50, bool), [0, 0, 0], [0, 0], axs=axs[1, 2]
        )
    plt.close(fig)


def test_nonlinear_iq_fit_plot_with_failed_fit():
    """
    A failed fit (NaN parameters) still plots the data, centered on the sweep.
    """
    f = np.linspace(1e9, 1.001e9, 50)
    z = np.exp(1j * np.linspace(0, 3, 50))
    fig, axs = plot.plot_nonlinear_iq_fit(f, z, np.full(8, np.nan), None)
    x = axs[1].lines[0].get_xdata()
    assert np.all(np.isfinite(x)) and abs(np.mean(x)) < 1
    plt.close(fig)


################################################################################
################################# plot_sparper #################################
################################################################################
def test_plot_sparper_draws_both_psds():
    """
    Check that both binned PSDs are drawn into the given axes.
    """
    f = np.linspace(0.1, 1000, 5000)
    spar = -90 - 10 * np.log10(f)
    sper = np.full_like(f, -100.0)
    fig0, ax0 = plt.subplots()
    fig, ax = plot.plot_sparper(f, spar, sper, nbins=50, fmin=0.5, ax=ax0)
    assert fig is fig0 and ax is ax0
    assert len(ax.get_lines()) >= 2
    plt.close(fig)


@pytest.mark.parametrize("f, spar, sper", [
    (np.array([1.0, 2.0]), np.zeros(3), np.zeros(2)),
    (np.array([2.0, 1.0]), np.zeros(2), np.zeros(2)),
])
def test_plot_sparper_rejects_bad_input(f, spar, sper):
    """
    Check shape mismatches and unsorted frequencies raise ValueError.
    """
    with pytest.raises(ValueError):
        plot.plot_sparper(f, spar, sper, nbins=5, fmin=0.1)


################################################################################
################################## plot_xcal ###################################
################################################################################
def _xcal_inputs(n = 50):
    """
    Make simple fine-sweep inputs for ``plot_xcal``.

    Parameters:
    n (int): number of fine sweep points.

    Returns:
    thetaf, xf, zf_cent, xcal_mask, poly_x: ``plot_xcal`` inputs.
    """
    thetaf = np.linspace(-1, 1, n)
    zf_cent = np.exp(1j * thetaf)
    mask = np.ones(n, bool)
    mask[:5] = False
    return thetaf, thetaf * 1e-6, zf_cent, mask, np.array([1e-6, 0])


@pytest.mark.parametrize("change, match", [
    (dict(thetat = np.zeros((2, 3))), 'thetat must be 1-dimensional'),
    (dict(xf = np.zeros(49)), 'same shape'),
    (dict(poly_x = np.zeros((2, 2))), 'poly_x must be 1-dimensional'),
    (dict(zf_cent = np.zeros(49, complex)), 'zf_cent must have the same'),
    (dict(xcal_mask = np.ones(49, bool)), 'xcal_mask must have the same'),
])
def test_plot_xcal_invalid_input(change, match):
    """
    Check malformed inputs raise ValueError.
    """
    thetaf, xf, zf_cent, mask, poly_x = _xcal_inputs()
    kw = dict(thetaf = thetaf, xf = xf, zf_cent = zf_cent, xcal_mask = mask,
              poly_x = poly_x)
    kw.update(change)
    with pytest.raises(ValueError, match = match):
        plot.plot_xcal(**kw)
    plt.close('all')


def test_plot_xcal_without_timestream():
    """
    Check only the fine sweep and fit are drawn when no timestream is given.
    """
    thetaf, xf, zf_cent, mask, poly_x = _xcal_inputs()
    fig, axs = plot.plot_xcal(thetaf, xf, zf_cent, mask, poly_x)
    assert len(axs) == 2
    assert len(axs[0].lines) == 2
    assert len(axs[0].lines[0].get_xdata()) == mask.sum()
    labels = [l.get_label() for l in axs[1].lines]
    assert labels == ['full fine sweep', 'fit fine sweep', 'fit']
    plt.close(fig)


def test_plot_xcal_thetat_only():
    """
    Check a theta timestream without IQ data is drawn in the x-vs-theta plot
    only, split by the std cutoff, and subsampled to max_points.
    """
    thetaf, xf, zf_cent, mask, poly_x = _xcal_inputs()
    thetat = np.random.default_rng(1).normal(0, 0.01, 5000)
    fig, axs = plot.plot_xcal(thetaf, xf, zf_cent, mask, poly_x,
                              thetat = thetat, std_cutoff = 1,
                              max_points = 400)
    inside, outside = axs[0].lines[2:4]
    assert len(inside.get_xdata()) + len(outside.get_xdata()) == 400
    std = np.std(thetat)
    assert np.all(np.abs(inside.get_xdata() - np.mean(thetat)) < std)
    assert np.all(np.abs(outside.get_xdata() - np.mean(thetat)) >= std)
    assert len(axs[1].lines) == 3  # no timestream in the IQ plot
    plt.close(fig)


def test_plot_xcal_thetat_without_cutoff_draws_all_alike():
    """
    Check every timestream point is drawn as within the cutoff when
    std_cutoff is None.
    """
    thetaf, xf, zf_cent, mask, poly_x = _xcal_inputs()
    thetat = np.random.default_rng(1).normal(0, 0.01, 300)
    fig, axs = plot.plot_xcal(thetaf, xf, zf_cent, mask, poly_x,
                              thetat = thetat)
    inside, outside = axs[0].lines[2:4]
    assert len(inside.get_xdata()) == 300 and len(outside.get_xdata()) == 0
    plt.close(fig)


def test_plot_xcal_zt_only():
    """
    Check an IQ timestream without theta data is drawn in the IQ plot only,
    all within the cutoff, and subsampled to max_points.
    """
    thetaf, xf, zf_cent, mask, poly_x = _xcal_inputs()
    zt = np.exp(1j * np.random.default_rng(2).normal(0, 0.01, 3000))
    fig, axs = plot.plot_xcal(thetaf, xf, zf_cent, mask, poly_x,
                              zt_cent = zt, std_cutoff = 1, max_points = 250)
    assert len(axs[0].lines) == 2
    inside, outside = axs[1].lines[2:4]
    assert len(inside.get_xdata()) == 250 and len(outside.get_xdata()) == 0
    plt.close(fig)


################################################################################
############################ plot_s21, plot_circfit ############################
################################################################################
def test_plot_s21_gain_sweep():
    """
    Check the gain sweep is drawn and its shape is validated.
    """
    f = np.linspace(1e9, 1.001e9, 50)
    z = np.exp(1j * np.linspace(0, 3, 50))
    fg = np.linspace(0.99e9, 1.01e9, 30)
    zg = np.full(30, 0.5 + 0j)
    fig, axs = plot.plot_s21(f, z, fg = fg, zg = zg)
    assert axs[1].lines[0].get_label() == 'gain sweep'
    np.testing.assert_allclose(axs[0].lines[0].get_ydata(),
                               20 * np.log10(0.5))
    plt.close(fig)
    with pytest.raises(ValueError, match = 'fg and zg'):
        plot.plot_s21(f, z, fg = fg, zg = zg[:-1])
    with pytest.raises(ValueError, match = 'f and z'):
        plot.plot_s21(f, z[:-1])
    plt.close('all')


def test_plot_circfit_with_mask():
    """
    Check the masked fit data are drawn separately.
    """
    z = np.exp(1j * np.linspace(0, 3, 50))
    mask = np.arange(50) < 20
    fig, ax = plot.plot_circfit(z, 0, 1, mask = mask)
    fit_data = [l for l in ax.lines if l.get_label() == 'fit data'][0]
    np.testing.assert_allclose(fit_data.get_xdata(), z[:20].real)
    plt.close(fig)


def test_nonlinear_iq_fit_plot_draws_fit():
    """
    Check a finite fit is drawn over the masked data.
    """
    popt = [1e9, 2e4, 0.5, 0.1, 0.0, 1.0, 0.0, 0.0]
    f = np.linspace(1e9 - 1.5e5, 1e9 + 1.5e5, 100)
    from citkid.res.funcs import nonlinear_iq
    z = nonlinear_iq(f, *popt, True)
    mask = np.arange(100) % 2 == 0
    fig, axs = plot.plot_nonlinear_iq_fit(f, z, popt, mask)
    assert len(axs[0].lines[0].get_xdata()) == 50
    fit = [l for l in axs[0].lines if l.get_label() == 'fit'][0]
    assert len(fit.get_xdata()) == 500
    plt.close(fig)
