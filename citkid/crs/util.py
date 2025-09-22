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
    record_size = dtype.itemsize
    batch_size = 100_000
    parser_batch_file ='m0%d_raw32'%(module)
    with open(os.path.join(path, f'serial_{crs_sn:04d}', parser_batch_file),
              'rb') as f:
      while True:
          data = np.fromfile(f, dtype = dtype, count = batch_size)
          if data.size == 0:
              break
          z = parser_dat['i'].astype(np.float64) + 1j * parser_dat['q'].astype(np.float64)
          z = np.array([z[i::max_ntones] for i in range(ntones)])
          z = z * rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
          ### save to file here
    return z

def convert_parser_to_z_batch(path, outpath, crs_sn, module, ntones,
                              max_ntones, return_dbc, ares, ch_ixs):
    """
    Import a parser file and convert the data to complex S21 in V

    Parameters:
    path (str): path to the parser folder
    outpath (str): path to the output file. Must end in .npy. Suffices will be
        appended to the output files for each batch.
    crs_sn (int): CRS serial number
    module (int): module number
    ntones (int): number of tones
    return_dbc (bool): if true, divides the output by the tone power
    ares (np.ndarray or None): if return_dbc, uses ares as the tone powers
    ch_ixs (list): values (int) are indices into the data corresponding to the
        channels, in order.

    Returns:
    z (np.array): complex S21 data in V
    """
    warnings.warn("convert_parser_to_z_batch will overwrite data",
                  warnings.UserWarning)

    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    record_size = dtype.itemsize

    target_bytes = 500 * (1024 ** 2) # 500 MB
    batch_size = target_bytes // (record_size * len(module_indices))

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
          z = [[]] * len(fres)
          for module_index, parser_dat  in zip(module_indices, batch_parts):
              zi = parser_dat['i'].astype(np.float64) + 1j * parser_dat['q'].astype(np.float64)
              for index, ch_index in enumerate(ch_ix_dict[module_index]):
                  z[ch_index] = zi[index]
          data_len = min([len(zi) for zi in z])
          z = np.array([zi[:data_len] for zi in z])
          if z.shape[1] == 0:
              break

          z = np.array([z[i::max_ntones] for i in range(ntones)])
          z = z * rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
          if return_dbc:
              z /= 10 ** (ares[:, np.newaxis] / 20)
          np.save(outpath.replace('.npy', f'_{batch_index:02d}.npy'))
          batch_index += 1
     finally:
          for f in files:
              f.close()


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
