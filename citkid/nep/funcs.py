import numpy as np
from scipy.constants import h, k
from scipy.integrate import trapezoid

def get_NEPinc(Pinc, nu, eta_pb, Delta0, NEP0_abs, eta_opt):
    '''
    Calculates the NEP referred to units of incident power.
    The incident power is assumed to be narrow-band about a center frequency nu.
    Wave noise term is assumed to be 0.
    Optical recombination noise is included.
    
    Parameters:
    Pinc (float or array-like): incident power in W. Absorbed power is defined 
        as Pabs = Pinc * eta_opt.
    nu (float): Center frequency of bandpass filter in Hz.
    eta_pb (float): Pair-breaking efficiency.
    Delta0 (float): Gap energy in J.
    NEP0_abs (float): Limiting NEP at zero optical power. Referred to absorbed 
        power. Units = aW Hz^-1/2
    eta_opt (float): optical efficiency.

    Returns:
    NEPinc (float or array-like): total NEP referred to incident power. 
        Units = aW Hz^-1/2.
    '''
    Pabs = Pinc * eta_opt
    NEP_optical2 = 2 * Pabs * (h * nu + 2 * Delta0 / eta_pb)
    NEPabs = ((NEP0_abs / 1e18) ** 2 + NEP_optical2) ** .5
    NEPinc = NEPabs / eta_opt
    return NEPinc * 1e18

def get_NEPinc_with_power_distribution(dPinc_dnu, nu, eta_pb, Delta0, 
                                       NEP0_abs, eta_opt):
    '''
    Similar to get_NEPinc, but this function allows for a distribution
    of incident power with frequency, which is determined by the 
    transmission of the filters between the optical source and the
    focal plane.

    Parameters:
    nu (array-like): Array of frequencies in Hz for the filter
        transmission profile. Must be 1D.
    dPinc_dnu (array-like): Differential incident power in W per unit 
        frequency in Hz. If 2D, the last axis must be the axis along 
        which frequency varies, with length = len(nu).
    All others: see get_NEPinc

    Returns:
    NEPinc (float or array-like): total NEP referred to incident power. 
        Units = aW Hz^-1/2.
    '''
    nu = np.array(nu)
    dPabs_dnu = dPinc_dnu * eta_opt
    integrand = 2 * dPabs_dnu * (h * nu + 2 * Delta0 / eta_pb)
    NEP_optical2 = trapezoid(integrand, nu, axis=-1)
    NEPabs = ((NEP0_abs / 1e18) ** 2 + NEP_optical2)**.5
    NEPinc = NEPabs / eta_opt
    return NEPinc * 1e18