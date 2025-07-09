import numpy as np
from scipy import constants
h = constants.h

def get_NEPinc(Pinc, nu, eta_pb, Delta0, NEP0_abs, eta_opt):
    '''
    Calculates the NEP referred to units of incident power.
    The incident power is assumed to be narrow-band about a center frequency nu.
    Wave noise term is assumed to be 0.
    Optical recombination noise is included.
    
    Parameters:
        Pinc: incident power in aW. Absorbed power is defined as Pabs = Pinc*eta_opt
        nu: Center frequency of bandpass filter in Hz
        eta_pb: Pair-breaking efficiency
        Delta0: Gap energy in J
        NEP0_abs: Limiting NEP at zero optical power. Referred to absorbed power. Units = aW Hz^-1/2
        eta_opt: optical efficiency
    Returns:
        NEPinc: total NEP referred to incident power. Units = aW Hz^-1/2
    '''
    Pabs = Pinc * eta_opt
    NEPabs = ((NEP0_abs/1e18)**2 + 2*Pabs*(h*nu + 2*Delta0/eta_pb))**.5
    NEPinc = NEPabs / eta_opt
    return NEPinc*1e18