import pandas as pd
import numpy as np

def create_dataset_df(ID, temperature, cooldown, readout_system, other = {}):
    """
    Creates per-dataset metadata DataFrame.

    Parameters:
    ID (str): dataset identifier.
    temperature (float): in K.
    cooldown (str): cooldown name (e.g. BFCD20250101).
    readout_system (str): readout system and serial number (CRS11, PRIMECAM2).
    other (dict): values (str) are metadata keys and values (tuple) are 
        (value, dtype).

    Returns:
    (pd.DataFrame): per-dataset metadata, formatted as a single-row DataFrame.
    """
    dtypes = {'id': pd.StringDtype(), 'temperature': pd.Float64Dtype(), 
              'cooldown': pd.StringDtype(), 'readout_system': pd.StringDtype()}
    d = {'id': ID, 'temperature': temperature, 'cooldown': cooldown, 
         'readout_system': readout_system}
    for key, value in other.items():
        d[key] = value[0]
        dtypes[key] = pd.api.types.pandas_dtype(value[1])
    return pd.DataFrame(d, index = [0]).astype(dtypes)

def create_tone_df(ID, res_idx, ares, atten, other = {}):
    """
    Creates per-tone metadata DataFrame. N is the number of tones. Arrays must 
    be sorted in the same order as the raw data.

    Parameters:
    ID (str): dataset identifier. 
    res_idx (np.array, int32, (N,)): resonator indices. Negative for 
        calibration tones.
    ares (np.array, float64, (N,)): readout system output power in dBm.
    atten ((np.array, float64, (N,)) or float64): Attenuation between the 
        readout system and the device in dB. Can be a single float if the 
        frequency-dependence is ignored.
    other (dict): keys (str) are metadata keys and values (tuple) are 
        (value, dtype), where value can be (np.array, (N,)) or a single value to
        apply to all columns.

    Returns:
    (pd.DataFrame): per-tone metadata, formatted as a N-row DataFrame.
    """
    ares = np.asarray(ares, dtype = np.float64)
    dtypes = {'id': pd.StringDtype(), 'res_idx': pd.Int32Dtype(),
              'out_power': pd.Float64Dtype(), 'atten': pd.Float64Dtype(),
              'power': pd.Float64Dtype()}
    d = {'id': ID, 'res_idx': res_idx, 'out_power': ares, 'atten': atten,
         'power': ares - atten}
    for key, value in other.items():
        d[key] = value[0]
        dtypes[key] = pd.api.types.pandas_dtype(value[1])
    return pd.DataFrame(d).astype(dtypes)

def create_gain_df(ID, res_idx, p_amp, p_phase):
    """
    Creates a DataFrame of gain fit parameters.

    Parameters:
    ID (str): dataset identifier. 
    res_idx (int): resonator index. 
    p_amp (np.array, float64, (M,)): gain |S21| (dB) vs frequency (Hz) 
        polynomial fit parameters. Default is M = 3.
    p_phase (np.array, float64, (P,)): gain phase vs frequency (Hz) 
        polynomial fit parameters. Default is P = 2.

    Returns:
    (pd.DataFrame): gain fit data, formatted as a single-row DataFrame.
    """
    dtypes = {'id': pd.StringDtype(), 'res_idx': pd.Int32Dtype()}
    d = {'id': ID, 'res_idx': res_idx}
    for i, p in enumerate(p_amp):
        d[f'pamp_{i}'] = p
        dtypes[f'pamp_{i}'] = pd.Float64Dtype()
    for i, p in enumerate(p_phase):
        d[f'pphase_{i}'] = p
        dtypes[f'pphase_{i}'] = pd.Float64Dtype()
    return pd.DataFrame(d, index = [0]).astype(dtypes)

def read_gain_df(row):
    """
    Reads a row from a gain fit DataFrame and extracts the gain fit parameters.

    Parameters:
    row (pd.Series): a row from a gain fit DataFrame.

    Returns:
    ID (str): dataset identifier.
    res_idx (int): resonator index.
    p_amp (np.array, float64, (M,)): gain amplitude fit parameters. 
        Default is M = 3.
    p_phase (np.array, float64, (P,)): gain phase fit parameters. 
        Default is P = 2.
    """
    ID = row.id
    res_idx = row.res_idx
    cols = [c for c in row.keys() if c.startswith('pamp_')] 
    p_amp = np.array([row[c] for c in cols], dtype = np.float64)
    cols = [c for c in row.keys() if c.startswith('pphase_')] 
    p_phase = np.array([row[c] for c in cols], dtype = np.float64)
    return ID, res_idx, p_amp, p_phase

def create_theta_df(ID, res_idx, origin, radius, theta0, sfactor = None):
    """
    Creates a DataFrame of resonator circle fit parameters.

    Parameters:
    ID (str): dataset identifier.
    res_idx (int): resonator index.
    origin (complex): center of the resonant circle.
    radius (float): radius of the resonant circle.
    theta0 (float): offset angle by which the data is rotated after subtracting
        the origin.
    sfactor (float, optional): scaling factor for the resonant circle. 
        Default is None.

    Returns:
    (pd.DataFrame): resonator circle fit data, formatted as a single-row 
        DataFrame.
    """
    dtypes = {'id': pd.StringDtype(), 'res_idx': pd.Int32Dtype(),
              'origin_real': pd.Float64Dtype(), 
              'origin_imag': pd.Float64Dtype(), 'radius': pd.Float64Dtype(), 
              'theta0': pd.Float64Dtype(), 'sfactor': pd.Float64Dtype()}
    d = {'id': ID, 'res_idx': res_idx, 'origin_real': origin.real, 
         'origin_imag': origin.imag, 'radius': radius, 'theta0': theta0, 
         'sfactor': sfactor}
    return pd.DataFrame(d, index = [0]).astype(dtypes)

def create_x_df(ID, res_idx, px):
    """
    Creates a DataFrame of theta -> x calibration fit parameters.

    Parameters:
    ID (str): dataset identifier.
    res_idx (int): resonator index.
    px (np.array, float64, (Q,)): x vs theta calibration polynomial fit 
        parameters. Default is Q = 3.

    Returns:
    (pd.DataFrame): theta -> x calibration fit data, formatted as a single-row 
        DataFrame.
    """
    dtypes = {'id': pd.StringDtype(), 'res_idx': pd.Int32Dtype()}
    d = {'id': ID, 'res_idx': res_idx}
    for i, p in enumerate(px):
        d[f'px_{i}'] = p
        dtypes[f'px_{i}'] = pd.Float64Dtype()
    return pd.DataFrame(d, index = [0]).astype(dtypes)