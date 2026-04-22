import warnings
warnings.warn(
    "citkid.multitone.ares is legacy code and may be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import numpy as np

# Need to update docstrings, imports
def update_ares_pscale(frequency, power_dbm, a_nl, dbm_change_high = 2,
                       dbm_change_low = 2, a_target = 0.5, dbm_max = -50):
    """
    Updates the tone amplitude list to target the given value of a_nl by scaling
    the output power linearly with a_nl, or scales the power linearly if it
    is outside the scalable range (a_target * 0.001 / 0.5, 0.77).

    Parameters:
    frequency (float): frequency in Hz
    power_dbm (float): tone power in dBm
    a_nl (float): nonlinearity parameter
    dbm_change_high (float): number of dBm to decrease the power if
        a_nl > 0.77
    dbm_change_low (float) : number of dBm to increase the power if
        a_nl < 0.001 / 0.5
    a_target (float): target value for a_nl. Must be in (0, 0.77]
    dbm_max (float): maximum value of the tone amplitude in dBm

    Returns:
    power_dbm_updated (float): updated value of the tone amplitude in dBm
    """
    if a_target > 0.77 or a_target <= 0:
        raise ValueError('a_target must be in (0, 0.77]')
    power_mW = 10 ** (power_dbm / 10)
    if a_nl > 0.77:
        power_dbm_updated = power_dbm - dbm_change_high
    elif a_nl > a_target * 0.001 / 0.5:
        power_dbm_updated = 10 * np.log10(power_mW * a_target / a_nl)
    else:
        power_dbm_updated = power_dbm + dbm_change_low
    if power_dbm_updated > dbm_max:
        power_dbm_updated = dbm_max
    return power_dbm_updated
update_ares_pscale = np.vectorize(update_ares_pscale)

def update_ares_addonly(f, power_dbm, a_nl, dbm_change_high = 1,
                        dbm_change_low = 1, a_target = 0.5, dbm_max = -50,
                        threshold = 0.2):
    """
    Updates the amplitude of a tone to within 80% of the target by adding or
    subtracting a fixed power in dB.

    Parameters:
    frequency (float): frequency in Hz
    power_dbm (float): tone power in dBm
    a_nl (float): nonlinearity parameter
    dbm_change_high (float): number of dBm to decrease the power if
        a_nl > a_target * 1.2
    dbm_change_low (float) : number of dBm to increase the power if
        a_nl < a_target * 0.8
    a_target (float): target value for a_nl. Must be in (0, 0.77]
    dbm_max (float): maximum value of the tone amplitude in dBm
    threshold (float): optimization will occur within (1-threshold) and 
        (1+threshold) of the target 

    Returns:
    power_dbm_updated (float): updated value of the tone amplitude in dBm
    """
    if a_target > 0.77 or a_target <= 0:
        raise ValueError('a_target must be in (0, 0.77]')
    # power_mW = 10 ** (power_dbm / 20)
    if a_nl > a_target * (1 + threshold):
        power_dbm_updated = power_dbm - dbm_change_high
    elif a_nl < a_target * (1 - threshold):
        power_dbm_updated = power_dbm + dbm_change_low
    else:
        power_dbm_updated = power_dbm
    if power_dbm_updated > dbm_max:
        power_dbm_updated = dbm_max
    return power_dbm_updated
update_ares_addonly = np.vectorize(update_ares_addonly)
