import numpy as np
import os
import rfmux
import subprocess
import signal
import sys
from time import sleep
from tqdm.auto import tqdm
import warnings

def convert_parser_to_z(path, crs_sn, module, ntones, max_ntones):
    """
    Import a parser file and convert the data to complex S21 in V

    Parameters:
    path (str): path to the parser folder
    crs_sn (int): CRS serial number
    module (int): module number
    ntones (int): number of tones

    Returns:
    z (np.array): complex S21 data in V
    """
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    parser_batch_file ='m0%d_raw32'%(module)
    with open(os.path.join(path, f'serial_{crs_sn:04d}', parser_batch_file),
              'rb') as f:
        parser_dat = np.fromfile(f, dtype = dtype)
        z = parser_dat['i'] + 1j * parser_dat['q']
        z = np.array([z[i::max_ntones] for i in range(ntones)])
        z = z * rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
    return z

def convert_parser_to_z_batch(path, outpath, crs_sn, module_indices, ntones,
                              max_ntones, return_dbc, ares, ch_ix_dict,
                              batch_size = 500):
    """
    Import a parser file in batchesand reformat it in order of the channels of
    interest. Saves each batch as int32 data, to later be scaled by the scaling
    factors saved by CRS.take_noise.

    Parameters:
    path (str): path to the parser folder.
    outpath (str): path to the output file. Must end in .npy. Suffices will be
        appended to the output files for each batch.
    crs_sn (int): CRS serial number.
    module (int): module number.
    ntones (int): number of tones.
    max_ntones (int): maximum number of tones, for parsing data.
    return_dbc (bool): if true, divides the output by the tone power.
    ares (np.ndarray or None): if return_dbc, uses ares as the tone powers.
    ch_ix_dict (dict): channel index dictionary. keys (int) are module indices.
        Values are lists where values (int) are indices into the data
        corresponding to the channels, in order.
    batch_size (int): batch size, in MB.
    """
    warnings.warn("convert_parser_to_z_batch will overwrite data",
                  UserWarning)

    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    record_size = dtype.itemsize

    target_bytes = batch_size * (1024 ** 2) # 500 MB
    batch_size = target_bytes // (record_size * len(module_indices))
    batch_size = (batch_size // max_ntones) * max_ntones
    # channel data is stored, sequentially, so batch_size must be a multiple of max_ntones
    file_paths = [
        os.path.join(path, f'serial_{crs_sn:04d}', 'm0%d_raw32'%(module))
        for module in module_indices
    ]

    files = [open(fp, 'rb') for fp in file_paths]
    try:
        batch_index = 0
        while True:
            batch_parts = [
            np.fromfile(f, dtype = dtype, count = batch_size)
            for f in files
            ]
            z_real = [[]] * ntones
            z_imag = [[]] * ntones
            for module_index, parser_dat  in zip(module_indices, batch_parts):
                zi_real = parser_dat['i'].astype(np.int32)
                zi_imag = parser_dat['q'].astype(np.int32)
                for index, ch_index in enumerate(ch_ix_dict[module_index]):
                    z_real[ch_index] = zi_real[index::max_ntones]
                    z_imag[ch_index] = zi_imag[index::max_ntones]
            data_len = min([len(zi) for zi in z_real])
            if data_len == 0:
                break
            z_real = np.array([zi[:data_len] for zi in z_real])
            z_imag = np.array([zi[:data_len] for zi in z_imag])
            if z_real.shape[1] == 0:
                break
            np.save(outpath.replace('.npy', f'_batch{batch_index:02d}.npy'),
                    [z_real, z_imag])
            batch_index += 1
    finally:
        for f in files:
            f.close()

def import_noise_data(data_path, scale_factor_path):
    """
    Imports noise data as saved by the batch processor.

    Parameters:
    data_path (str): path to the complex IQ data.
    scale_factor_path (str): path to the scale factor data.

    Returns:
    z (np.array): data, converted to complex128 and scaled by scale_factor.
    """
    i, q = np.load(data_path).astype(np.int32)
    scale_factor = np.load(scale_factor_path).astype(np.float64)
    z = np.empty(i.shape, dtype = np.complex128)
    np.multiply(i, scale_factor, out = z.real)
    np.multiply(q, scale_factor, out = z.imag)
    return z


def find_key_and_index(dictionary, j):
    """
    Finds the key in a dictionary where the numpy array (value) contains the
    integer 'j', and returns both the key and the index of 'j' in that array

    Parameters:
    dictionary (dict): A dictionary where keys are integers and values are numpy
        arrays of integers
    j (int): The integer to search for in the numpy arrays

    Returns:
    tuple: A tuple (key, index) where 'key' is the dictionary key and 'index'
        is the position of 'j' in the array If 'j' is not found, returns
        (None, None)
    """
    for key, value_array in dictionary.items():
        indices = np.where(value_array == j)[0]
        if indices.size > 0:  # Check if 'j' was found
            return key, indices[0]
    return None, None  # Return (None, None) if 'j' is not found in any array

def get_modules(crs, module_indices):
    """
    Gets the subset of crs ReadoutModule objects given the module indices

    Parameters:
    crs (rfmux.CRS): rfmux CRS system module
    module_indices (array-like, int): module indices

    Returns:
    modules (array-like, rfmux.ReadoutModule): crs system object readout modules
        corresponding to the module indices
    """
    modules_generic = rfmux.ReadoutModule.module.in_(module_indices)
    modules = crs.modules.filter(modules_generic)
    return modules

def run_for_duration(cmd, duration, verbose = True):
    """
    Run a command for a given duration, then shut it down safely,
    even if Python is closed.

    Parameters:
    cmd (str): command to run
    duration (int): duration in seconds
    verbose (bool): If True, displays a progress bar while running
    """
    if os.name == "nt":  # Windows
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:  # Linux/macOS
        process = subprocess.Popen(cmd, preexec_fn=os.setsid)

    try:
        pbar = list(range(int(duration)))
        if verbose:
            pbar = tqdm(pbar, leave = False)
        sleep(0.04)
        for i in pbar:
            sleep(1)
    finally:
        if os.name == "nt":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)])  # Windows force kill
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Unix kill entire process group
