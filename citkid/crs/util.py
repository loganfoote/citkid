import numpy as np
import os
import rfmux
import subprocess
import signal
from time import sleep
from tqdm.auto import tqdm
import warnings
import socket 

def convert_parser_to_z(path, crs_sn, module, ntones, max_ntones):
    """
    Import a parser file and convert the data to complex S21 in V.

    Parameters:
    path (str): path to the parser folder.
    crs_sn (int): CRS serial number.
    module (int): module number.
    ntones (int): number of tones.
    max_ntones (int): maximum number of tones per module.

    Returns:
    z (np.ndarray): complex S21 data in V.
    """
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    parser_batch_file ='m0%d_raw32'%(module)
    with open(os.path.join(path, f'serial_{crs_sn:04d}', parser_batch_file),
              'rb') as f:
        parser_dat = np.fromfile(f, dtype = dtype)
        z = (
            parser_dat['i'].astype(np.float64)
            + 1j * parser_dat['q'].astype(np.float64)
        )
        z = np.array([z[i::max_ntones] for i in range(ntones)])
        z = z * rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
    return z

def convert_parser_to_z_batch(path, outpath, crs_sn, module_indices, ntones,
                              max_ntones, return_dbc, ares, ch_ix_dict,
                              batch_size = 500):
    """
    Import a parser file in batches and reformat for channels of interest.

    Saves each batch as int32 data, to later be scaled by the factors saved by
    `CRS.take_noise`.

    Parameters:
    path (str): path to the parser folder.
    outpath (str): path to the output file. Must end in .npy. Suffices will be
        appended to the output files for each batch.
    crs_sn (int): CRS serial number.
    module_indices (array-like): module indices.
    ntones (int): number of tones.
    max_ntones (int): maximum number of tones per module.
    return_dbc (bool): if True, divide the output by the tone power.
    ares (np.ndarray or None): if return_dbc, uses ares as the tone powers.
    ch_ix_dict (dict): channel index dictionary. Keys (int) are module indices.
        Values are lists where values (int) are channel indices.
    batch_size (int): batch size, in MB.

    Returns:
    None
    """
    warnings.warn("convert_parser_to_z_batch will overwrite data",
                  UserWarning)

    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    record_size = dtype.itemsize

    target_bytes = batch_size * (1024 ** 2)  # 500 MB
    batch_size = target_bytes // (record_size * len(module_indices))
    batch_size = (batch_size // max_ntones) * max_ntones
    # Channel data is stored sequentially, so batch_size must be
    # a multiple of max_ntones
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
            N = min([b.shape[0] // (max_ntones) for b in batch_parts])
            z_real = np.empty((ntones, N), dtype = np.int32)
            z_imag = np.empty((ntones, N), dtype = np.int32)
            for module_index, parser_dat  in zip(module_indices, batch_parts):
                zi_real = parser_dat['i'].astype(np.int32)
                zi_imag = parser_dat['q'].astype(np.int32)
                for index, ch_index in enumerate(ch_ix_dict[module_index]):
                    z_real[ch_index] = zi_real[index:N * max_ntones:max_ntones]
                    z_imag[ch_index] = zi_imag[index:N * max_ntones:max_ntones]
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
    Import noise data as saved by the batch processor.

    Parameters:
    data_path (str): path to the complex IQ data.
    scale_factor_path (str): path to the scale factor data.

    Returns:
    z (np.ndarray): data scaled by scale_factor in complex128.
    """
    i, q = np.load(data_path).astype(np.int32)
    scale_factor = np.load(scale_factor_path).astype(np.float64)
    z = np.empty(i.shape, dtype = np.complex128)
    np.multiply(i, scale_factor, out = z.real)
    np.multiply(q, scale_factor, out = z.imag)
    return z

def find_key_and_index(dictionary, j):
    """
    Find the key in a dictionary whose array contains integer `j`.

    Parameters:
    dictionary (dict): Keys are integers and values are integer arrays.
    j (int): Integer to search for.

    Returns:
    tuple: (key, index) where key is the dictionary key and index is the
        position of `j` in the array. Returns (None, None) if not found.
    """
    for key, value_array in dictionary.items():
        indices = np.where(value_array == j)[0]
        if indices.size > 0:  # Check if 'j' was found
            return key, indices[0]
    return None, None  # Return (None, None) if 'j' is not found in any array

def get_modules(crs, module_indices):
    """
    Get ReadoutModule objects given module indices.

    Parameters:
    crs (rfmux.CRS): rfmux CRS system module.
    module_indices (array-like, int): module indices.

    Returns:
    modules (array-like): CRS readout modules for the provided indices.
    """
    modules_generic = rfmux.ReadoutModule.module.in_(module_indices)
    modules = crs.modules.filter(modules_generic)
    return modules

def run_for_duration(cmd, duration, verbose = True):
    """
    Run a command for a given duration, then shut it down safely.

    Parameters:
    cmd (str): command to run.
    duration (int): duration in seconds.
    verbose (bool): if True, display a progress bar while running.

    Returns:
    None
    """
    if os.name == "nt":  # Windows
        process = subprocess.Popen(
            cmd,
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:  # Linux/macOS
        process = subprocess.Popen(cmd, preexec_fn = os.setsid)

    try:
        pbar = list(range(int(duration)))
        if verbose:
            pbar = tqdm(pbar, leave = False)
        sleep(0.04)
        for i in pbar:
            sleep(1)
    finally:
        if os.name == "nt":
            # Windows force kill
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)]
            )
        else:
            # Unix kill entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

def get_sample_frequency(dec_stage):
    """
    Return the sample frequency in Hz given the decimation stage index.

    Parameters:
    dec_stage (int): decimation stage index.

    Returns:
    float: sample frequency in Hz.
    """
    return 625e6 / (256 * 64 * 2 ** dec_stage)

def estimate_timestream_data_size(dec_stage, noise_time, nmodules, max_ntones,
                                  ntones):
    """
    Estimate and print raw and processed timestream data sizes.

    Parameters:
    dec_stage (int): decimation stage.
    noise_time (float): timestream length in s.
    nmodules (int): number of active modules. If an NCO has been set, the
        modules will stream max_ntones channels whether or not tones are
        written.
    max_ntones (int): maximum number of tones on a module.
    ntones (int): total number of tones across the modules.

    Returns:
    None
    """
    # Type and range checks
    if type(dec_stage) != int or dec_stage < 0 or dec_stage > 6:
        raise ValueError('dec_stage must be an int in range [0, 6]')
    if noise_time < 0:
        raise ValueError('noise_time must be positive')
    if nmodules not in [1, 2, 3, 4]:
        raise ValueError('nmodules must be in [1, 2, 3, 4]')
    if type(max_ntones) != int or max_ntones < 0 or max_ntones > 1024:
        raise ValueError('max_ntones must be an int in range [0, 1024]')
    if type(ntones) != int or ntones < 0 or ntones > 1024 * 4:
        raise ValueError('ntones must be an int in range [0, 4 * 1024]')
    # Calculate file sizes
    sample_frequency = get_sample_frequency(dec_stage)
    size_perchannel = 4 * 2 * (noise_time * sample_frequency)
    size_raw = size_perchannel * nmodules * max_ntones + 103
    size_processed = size_perchannel * ntones
    size_processed += 8 * ntones  # scale factors

    # print files sizes
    size_raw /= 1e6
    unit = 'MB'
    s = f'{size_raw:.0f}'
    if size_raw // 1000:
        size_raw /= 1e3
        unit = 'GB'
        s = f'{size_raw:.1f}'
        
    print(f'Raw parser data size: {s} {unit}')
    size_processed /= 1e6
    s = f'{size_processed:.0f}'
    if size_processed // 1000:
        size_processed /= 1e3
        unit = 'GB'
        s = f'{size_processed:.1f}'
    print(f'Processed data size:  {s} {unit}') 


def interface_exists(iface):
    """
    Check if a network interface exists on the system.
    
    Parameters:
    iface (str): Name of the network interface to check.
    
    Returns:
    bool: True if the interface exists, False otherwise.
    """
    try:
        socket.if_nametoindex(iface)
        return True
    except OSError:
        return False
