import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .funcs import get_NEPinc

def fit_NEPinc(Pinc, NEPinc, nu, eta_pb, Delta0, p0=[0.1, 1], plot_fit=False, plot_guess=False):
    '''
    Fits a series of NEP vs. Power data to a model including photon shot noise
    and a limiting NEP term at zero power.
    
    Parameters:
        Pinc: incident power in W
        NEPinc: NEP referred to incident power. Units = W Hz^-1/2
        nu: Center frequency of bandpass filter in Hz
        eta_pb: Pair-breaking efficiency
        Delta0: Gap energy in J
        p0: guess parameters for NEP0_abs (aW Hz^-1/2) and eta_opt.
        plot_fit: if True, return a plot of the fit
        plot_guess: if True, include the initial guess in the plot
    Returns:
        popt: fit parameters
            NEP0_abs: Limiting NEP at zero optical power. Referred to absorbed power. Units = aW Hz^-1/2
            eta_opt: optical efficiency
        pcov: fit variance from scipy.optimize.curve_fit
        residual: Normalized fit residual
        fig_fit: Plot of the fit. Is None if plot_fit=False
    '''
    fit_func = lambda Pinc, NEP0_abs, eta_opt: get_NEPinc(Pinc, nu, eta_pb, Delta0, NEP0_abs, eta_opt)
    popt, pcov = curve_fit(fit_func, Pinc, NEPinc*1e18, p0=p0)
    popt = np.array([popt[0]/1e18, popt[1]])
    NEP0_abs, eta_opt = popt
    
    NEPfit = get_NEPinc(Pinc, nu, eta_pb, Delta0, *popt)
    residual = np.sum(((NEPfit - NEPinc)/NEPinc)**2) / len(NEPinc)
    
    fig_fit = None
    if plot_fit:
        Pabs = Pinc * eta_opt
        Psamp_abs = np.geomspace(min(Pabs)*.5, max(Pabs)*1.5, 100)
        Psamp_inc = Psamp_abs/eta_opt

        NEPshot = get_NEPinc(Psamp_abs, nu, eta_pb, Delta0, 0, 1)/1e18
        NEPfit = get_NEPinc(Psamp_inc, nu, eta_pb, Delta0, NEP0_abs*1e18, eta_opt)*eta_opt/1e18
        NEPabs = NEPinc * eta_opt

        fig_fit, ax_fit = plt.subplots(figsize=(5,4), dpi=300)
        label = '$NEP_{0,abs}=$'+f'{NEP0_abs:.2e} W Hz'+'$^{-1/2}$\n'+r'$\eta_{opt}=$'+f'{eta_opt:.2f}'
        ax_fit.plot(Psamp_abs, NEPfit, 'g--', lw=1, label=label)
        ax_fit.plot(Psamp_abs, NEPshot, 'r-.', lw=1)
        if plot_guess:
            NEPguess = get_NEPinc(Psamp_inc, nu, eta_pb, Delta0, 
                                  p0[0], p0[1])*p0[1]/1e18
            ax_fit.plot(Psamp_abs, NEPguess, color='k', ls='dotted', lw=1)
        ax_fit.scatter(Pabs, NEPabs, s=25, facecolors='none', edgecolors='k')
        ax_fit.set(xlabel='Power absorbed [W]', ylabel='NEP$_{abs}$ [W Hz$^{-1/2}$]', 
            xscale='log', yscale='log', ylim=[0.5*min(NEPabs), 1.5*max(NEPabs)])
        ax_fit.legend(loc='upper left')
        ax_fit.grid()
        plt.tight_layout()
    
    return popt, pcov, residual, fig_fit