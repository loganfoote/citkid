import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from .funcs import get_NEPinc, get_NEPinc_with_power_distribution

def fit_NEPinc(Pinc, NEPinc, nu, eta_pb, Delta0, p0=[0.1, 1], plot_fit=False, 
               plot_guess=False):
    '''
    Fits a series of NEP vs. Power data to a model including photon shot noise
    and a limiting NEP term at zero power.
    
    Parameters:
    Pinc (float or array-like): incident power in W.
    NEPinc (float): NEP referred to incident power. Units = W Hz^-1/2.
    nu (float): Center frequency of bandpass filter in Hz.
    eta_pb (float): Pair-breaking efficiency.
    Delta0 (float): Gap energy in J.
    p0 (list): guess parameters for NEP0_abs (aW Hz^-1/2) and eta_opt.
    plot_fit (bool): if True, return a plot of the fit.
    plot_guess (bool): if True, include the initial guess in the plot.

    Returns:
    popt (np.array): fit parameters:
        NEP0_abs: Limiting NEP at zero optical power. Referred to absorbed 
                    power. Units = aW Hz^-1/2.
        eta_opt: optical efficiency.
    pcov (np.array): fit variance from scipy.optimize.curve_fit.
    residual (float): Normalized fit residual.
    fig_fit (plt.figure): Plot of the fit, or None if plot_fit = False.
    '''
    fit_func = lambda Pinc, NEP0_abs, eta_opt: get_NEPinc(Pinc, nu, eta_pb, 
                                                          Delta0, NEP0_abs, 
                                                          eta_opt)
    popt, pcov = curve_fit(fit_func, Pinc, NEPinc * 1e18, p0 = p0)
    popt = np.array([abs(popt[0] / 1e18), popt[1]])
    NEP0_abs, eta_opt = popt
    
    NEPfit = get_NEPinc(Pinc, nu, eta_pb, Delta0, *popt)
    residual = np.sum(((NEPfit - NEPinc) / NEPinc) ** 2) / len(NEPinc)
    
    fig_fit = None
    if plot_fit:
        Pabs = Pinc * eta_opt
        Psamp_abs = np.geomspace(min(Pabs) * .5, max(Pabs) * 1.5, 100)
        Psamp_inc = Psamp_abs / eta_opt

        NEPshot = get_NEPinc(Psamp_abs, nu, eta_pb, Delta0, 0, 1) / 1e18
        NEPfit = get_NEPinc(Psamp_inc, nu, eta_pb, Delta0, 
                            NEP0_abs * 1e18, eta_opt) * eta_opt / 1e18
        NEPabs = NEPinc * eta_opt

        fig_fit, ax_fit = plt.subplots(figsize=(5, 4), dpi = 300)
        label = '$NEP_{0,abs}=$'+f'{NEP0_abs:.2e} W Hz'
        label += '$^{-1/2}$\n' + r'$\eta_{opt}=$' + f'{eta_opt:.2e}'
        ax_fit.plot(Psamp_abs, NEPfit, 'g--', lw = 1, label=label)
        ax_fit.plot(Psamp_abs, NEPshot, 'r-.', lw = 1)
        if plot_guess:
            NEPguess = get_NEPinc(Psamp_inc, nu, eta_pb, Delta0, 
                                  p0[0], p0[1]) * p0[1] / 1e18
            ax_fit.plot(Psamp_abs, NEPguess, color = 'k', ls = 'dotted', lw = 1)
        ax_fit.scatter(Pabs, NEPabs, s = 25, facecolors = 'none', 
                       edgecolors = 'k')
        ax_fit.set(xlabel = 'Power absorbed [W]', 
                   ylabel = 'NEP$_{abs}$ [W Hz$^{-1/2}$]', 
                   xscale = 'log', yscale = 'log', 
                   ylim = [0.5 * min(NEPabs), 1.5 * max(NEPabs)])
        ax_fit.legend(loc = 'upper left')
        ax_fit.grid()
        plt.tight_layout()
    
    return popt, pcov, residual, fig_fit


def fit_NEPinc_with_power_distribution(dPinc_dnu, NEPinc, nu,
                                       eta_pb, Delta0, p0=[0.1, 1], 
                                       plot_fit=False, plot_guess=False):
    '''
    Similar to fit_NEPinc, but this function allows for a distribution
    of incident power with frequency, which is determined by the 
    transmission of the filters between the optical source and the
    focal plane.
    
    Parameters:
    nu (array-like): Array of frequencies in Hz for the filter
        transmission profile. This array must be 1D.
    dPinc_dnu (array-like): Differential incident power in W per unit 
        frequency in Hz. This array must be 2D,
        and the last axis must be the axis along which frequency varies,
        with length = len(nu).
    All others: see fit_NEPinc

    Returns:
    popt (np.array): fit parameters:
        NEP0_abs: Limiting NEP at zero optical power. Referred to absorbed 
                    power. Units = aW Hz^-1/2.
        eta_opt: optical efficiency.
    pcov (np.array): fit variance from scipy.optimize.curve_fit.
    residual (float): Normalized fit residual.
    fig_fit (plt.figure): Plot of the fit, or None if plot_fit = False.
    '''
    fit_func = lambda dPinc_dnu, NEP0_abs, eta_opt: \
        get_NEPinc_with_power_distribution(dPinc_dnu, nu, eta_pb, 
                                           Delta0, NEP0_abs, eta_opt)
    popt, pcov = curve_fit(fit_func, dPinc_dnu, NEPinc * 1e18, p0 = p0)
    popt = np.array([abs(popt[0] / 1e18), popt[1]])
    NEP0_abs, eta_opt = popt
    
    NEPfit = get_NEPinc_with_power_distribution(dPinc_dnu, nu, 
                                                eta_pb, Delta0, *popt)
    residual = np.sum(((NEPfit - NEPinc) / NEPinc) ** 2) / len(NEPinc)
    
    fig_fit = None
    if plot_fit:
        dPabs_dnu = dPinc_dnu * eta_opt
                
        x = np.arange(dPabs_dnu.shape[0])
        y = nu
        interp = RegularGridInterpolator(
            points = (x, y),
            values = dPabs_dnu,
            method = 'linear'
        )
        
        xsamp = np.linspace(0, x[-1], 100)
        dPabs_dnu_samp = np.zeros((100, dPabs_dnu.shape[1]))
        for ii in range(len(xsamp)):
            pts = np.array([[xsamp[ii], this_nu] for this_nu in y])
            dPabs_dnu_samp[ii] = interp(pts)
        dPinc_dnu_samp = dPabs_dnu_samp / eta_opt
        Psamp_abs = np.trapezoid(dPabs_dnu_samp, nu, axis=-1)
        Pabs = np.trapezoid(dPabs_dnu, nu, axis=-1)
    

        NEPshot = get_NEPinc_with_power_distribution(dPabs_dnu_samp, nu,
                                                     eta_pb, Delta0, 0, 1) / 1e18
        NEPfit = get_NEPinc_with_power_distribution(dPinc_dnu_samp, nu,
                                                    eta_pb, Delta0, 
                                                    NEP0_abs * 1e18, eta_opt) * eta_opt / 1e18
        NEPabs = NEPinc * eta_opt

        fig_fit, ax_fit = plt.subplots(figsize=(5, 4), dpi = 300)
        label = '$NEP_{0,abs}=$'+f'{NEP0_abs:.2e} W Hz'
        label += '$^{-1/2}$\n' + r'$\eta_{opt}=$' + f'{eta_opt:.2e}'
        ax_fit.plot(Psamp_abs, NEPfit, 'g--', lw = 1, label=label)
        ax_fit.plot(Psamp_abs, NEPshot, 'r-.', lw = 1)
        if plot_guess:
            NEPguess = get_NEPinc_with_power_distribution(
                            dPinc_dnu_samp, nu, 
                            eta_pb, Delta0, 
                            p0[0], p0[1]
                        ) * p0[1] / 1e18
            ax_fit.plot(Psamp_abs, NEPguess, color = 'k', ls = 'dotted', lw = 1)
        ax_fit.scatter(Pabs, NEPabs, s = 25, facecolors = 'none', 
                       edgecolors = 'k')
        ax_fit.set(xlabel = 'Power absorbed [W]', 
                   ylabel = 'NEP$_{abs}$ [W Hz$^{-1/2}$]', 
                   xscale = 'log', yscale = 'log', 
                   ylim = [0.5 * min(NEPabs), 1.5 * max(NEPabs)])
        ax_fit.legend(loc = 'upper left')
        ax_fit.grid()
        plt.tight_layout()
    
    return popt, pcov, residual, fig_fit