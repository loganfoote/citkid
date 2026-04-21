import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def update_ares_pscale(f, a, a_nl, dbm_change_high = 2, dbm_change_low = 2,
                       a_target = 0.5, a_max = 1000):
    """
    Updates the amplitude of a tone to target a_nl  by scaling the output
    power of the RFSoC linearly with a_nl. If a_nl < a_target * 0.1 / 0.5 or
    a_nl > 0.77, shifts the output amplitude by a fixed value instead.

    Parameters:
    f (float): frequency in Hz
    a (float): amplitude
    a_nl (float): nonlinearity parameter
    dbm_change_high (float): number of dBm to decrease the power if a_nl > 0.8
    dbm_change_low (float) : number of dBm to increase the power if a_nl < 0.01
        We should explore if we can expand the range
    a_target (float): target value for a. Must be in (0, 0.77]
    a_max (float): maximum value of the amplitude

    Returns:
    a_new (float): updated value of a
    """
    if a_target > 0.77 or a_target <= 0:
        raise ValueError('a_target must be in (0, 0.77]')
    dbm = get_dbm(a, f)
    mW_power = 10 ** (dbm / 10)
    if a_nl > 0.77:
        new_dbm = dbm - dbm_change_high
    elif a_nl > a_target * 0.01 / 0.5:
        new_dbm = 10 * np.log10(mW_power * a_target / a_nl)
    else:
        new_dbm = dbm + dbm_change_low
    a_new = get_rfsoc_power(new_dbm, f)
    if a_new > a_max:
        a_new = a_max
    return a_new
update_ares_pscale = np.vectorize(update_ares_pscale)

def update_ares_addonly(f, a, a_nl, dbm_change_high = 1, dbm_change_low = 1,
                        a_target = 0.5, a_max = 1000):
    """
    Updates the amplitude of a tone to target 0.4 < a_nl < 0.6 by adding or
    subtracting a fixed power in dB.

    Parameters:
    f (float): frequency in Hz
    a (float): amplitude
    a_nl (float): nonlinearity parameter
    dbm_change_high (float): number of dBm to decrease the power if a_nl > 0.6
    dbm_change_low (float) : number of dBm to increase the power if a_nl < 0.4
        We should explore if we can expand the range
    a_target (float): target value for a. Must be in (0, 0.77]
    a_max (float): maximum value of the amplitude

    Returns:
    a_new (float): updated value of a
    """
    if a_target > 0.77 or a_target <= 0:
        raise ValueError('a_target must be in (0, 0.77]')
    dbm = get_dbm(a, f)
    mW_power = 10 ** (dbm / 10)
    if a_nl > a_target * 0.6 / 0.5:
        new_dbm = dbm - dbm_change_high
    elif a_nl < a_target * 0.4 / 0.5:
        new_dbm = dbm + dbm_change_low
    else:
        new_dbm = dbm
    a_new = get_rfsoc_power(new_dbm, f)
    if a_new > a_max:
        a_new = a_max
    return a_new
update_ares_addonly = np.vectorize(update_ares_addonly)

################################################################################
########################## power conversion functions ##########################
################################################################################
cal_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'cal_data',
)
ares = np.load(os.path.join(cal_directory, 'ares_values.npy'))
fres = np.load(os.path.join(cal_directory, 'fres_values.npy'))
dBm_powers = np.load(os.path.join(cal_directory, 'dbm_powers.npy'))
dBm_powers_aliased = np.load(os.path.join(cal_directory, 'dbm_powers_aliased.npy'))
output_freqs = np.load(os.path.join(cal_directory, 'output_freqs.npy'))
output_freqs_aliased = np.load(
    os.path.join(cal_directory, 'output_freqs_aliased.npy')
)

def get_dbm(a, f, aliased=False):
    """
    Converts RFSoC power units to dBm.

    Parameters:
    a (float): A power level in RFSoC units, i.e. what you supply for ares
    f (float): the tone frequency you write to the RFSoC
    aliased (bool): False if you want to calculate the power in the base tone, 
        True if you want to calculate the power in the aliased tone

    Returns:
    dbm (float): the predicted output power in dBm
    """
    if aliased:
        interp = RegularGridInterpolator((fres, ares), dBm_powers_aliased)
    else:
        interp = RegularGridInterpolator((fres, ares), dBm_powers)
    
    dbm = interp([f, a])
    return dbm

def get_rfsoc_power(dbm, f, aliased=False):
    '''
    Converts power in dBm units to RFSoC units.

    Parameters:
    dbm (float): power in dBm
    f (float): tone frequency in Hz
    aliased (bool): False if you want to calculate the RFSoC power for the base tone, 
        True if you want to calculate the RFSoC power for the aliased tone

    Returns:
    a (float): power in RFSoC units
    '''
    ii_freq = np.argmin(abs(fres - f))
    if aliased:
        a = ares[np.argmin(abs(dBm_powers_aliased[ii_freq,:] - dbm))]
    else:
        a = ares[np.argmin(abs(dBm_powers[ii_freq,:] - dbm))]
    return a
