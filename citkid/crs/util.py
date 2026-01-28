import numpy as np
import os
import rfmux
import warnings
import socket 
from .. import zarr_util

################################################################################
##################### tone frequency -> NCO frequency map ######################
################################################################################
def create_ch_map(nco_freqs, freqs, bw):
    """
    Create a mapping between NCO frequencies and tone frequencies. Each tone 
    frequency is assigned to a module NCO if it falls within the NCO bandwidth. 
    If multiple NCOs can accommodate a tone frequency, the NCO closest to the 
    median frequency of the tone(s) is selected. Tones that do not fall within 
    any NCO bandwidth are returned as missing channels.

    Parameters: 
    nco_freqs (dict): Keys are module indices (int), values are NCO frequencies
        (float).
    freqs (array-like(float) or array-like(array-like(float))): Each index 
        corresponds to a channel index and each value is either a single tone 
        frequency or a list of tone frequencies for that channel. 
    bw (float): bandwidth in Hz.

    Returns:
    ch_map (dict): Keys are module indices (int), values are arrays of channel 
        indices (int) that fall within the NCO bandwidth for that module.
    missing_chs (np.ndarray, int32): List of channel indices that fall outside 
        all NCO bandwidths. 
    """
    # Input validation
    if not isinstance(nco_freqs, dict):
        raise TypeError("nco_freqs must be a dictionary")
    if not all(isinstance(v, (float, np.floating)) for v in nco_freqs.values()):
        raise ValueError("All values in nco_freqs must be float")
    if not all(isinstance(k, (int, np.integer)) for k in nco_freqs.keys()):
        raise ValueError("All keys in nco_freqs must be int") 
    freqs = np.asarray(freqs, dtype = np.float64) 
    bw = float(bw)
    if bw <= 0:
        raise ValueError("bw must be positive") 
    if freqs.ndim > 2:
        raise ValueError("freqs must be a list or numpy array")
    if freqs.ndim == 0:
        raise ValueError("freqs values must be float or list of floats")

    # Reshape freqs to 2D array if necessary
    if freqs.ndim == 1:
        freqs = freqs[:, np.newaxis]

    # Create channel mapping
    ch_map = {module: [] for module in nco_freqs.keys()} 
    missing_chs = [] 
    for ch_idx, freq in enumerate(freqs):
        fmin, fmax = freq.min(), freq.max() 
        # Find candidate modules whose NCO bandwidth contains the tone(s)
        candidates = []
        for module, nco_freq in nco_freqs.items():
            if (fmin >= nco_freq - bw / 2) and (fmax <= nco_freq + bw / 2):
                candidates.append((module, nco_freq))
        # Select the best module based on median frequency
        if len(candidates) == 1:
            # 1 candidate found
            best_module = candidates[0][0]
            ch_map[best_module].append(ch_idx)
        elif candidates:
            # Multiple candidates found, choose closest to median frequency
            median_freq = np.median(freq)
            best_module, _ = min(candidates, 
                                 key=lambda item: abs(median_freq - item[1]))
            ch_map[best_module].append(ch_idx)
        else:
            # No candidates found, append to missing_chs
            missing_chs.append(ch_idx)

    # Convert channel lists to numpy arrays and return
    for module in ch_map.keys():
        ch_map[module] = np.array(ch_map[module], dtype = np.int32) 
    missing_chs = np.array(missing_chs, dtype = np.int32)
    return ch_map, missing_chs

################################################################################
############################### rfmux utilities ################################
################################################################################
def get_modules(d, module_idxs):
    """
    Get ReadoutModule objects given module indices.

    Parameters:
    d (rfmux.core.schema.CRS): rfmux CRS system module.
    module_idxs (array-like, int): module indices.

    Returns:
    modules (array-like): CRS readout modules for the provided indices.
    """
    # Input validation
    if not isinstance(d, rfmux.core.schema.CRS):
        raise TypeError("d must be an instance of rfmux.core.schema.CRS")
    if not hasattr(module_idxs, '__iter__'):
        raise TypeError("module_idxs must be an iterable of ints")
    if not all(isinstance(i, int) for i in module_idxs):
        raise ValueError("All module_idxs must be ints")
    
    # Get modules
    modules_generic = rfmux.ReadoutModule.module.in_(module_idxs)
    modules = d.modules.filter(modules_generic)
    return modules

def get_sample_freq(dec_stage):
    """
    Return the sample frequency in Hz given the decimation stage index.

    Parameters:
    dec_stage (int): decimation stage index.

    Returns:
    float: sample frequency in Hz.
    """
    # Input validation
    if not isinstance(dec_stage, int) or dec_stage < 0 or dec_stage > 6:
        raise ValueError("dec_stage must be an int in range [0, 6]")
    
    # Calculate sample frequency
    return 625e6 / (256 * 64 * 2 ** dec_stage)

################################################################################
############################## parser processing ###############################
################################################################################
def parser_to_zarr(path, grp, crs_sn, ntones, max_ntones, 
                   ch_map, ares_map, dt, batch_size_mb = 1_000):
    """
    Import a parser file in batches and reformat for channels of interest. Save 
    to a Zarr file.

    Saves each batch as int32 data, to later be scaled by the factors saved by
    `CRS.take_ts`.

    Parameters:
    path (str): path to the parser folder.
    grp (zarr.hierarchy.Group): Zarr group to save data.
    crs_sn (int): CRS serial number.
    ntones (int): number of tones.
    max_ntones (int): maximum number of tones per module.
    ch_map (dict): channel index dictionary. Keys (int) are module indices.
        Values are lists where values (int) are channel indices.
    ares_map (dict): power dictionary. Keys (int) are module indices. Values are
        arrays where values (float) are power in dBc. Used to create scaling 
        from CRS amplitude to dBc.
    dt (float): sample time in seconds.
    batch_size_mb (int): batch size, in MB.

    Returns:
    None
    """
    ### Write scale_factor and dt
    rfmux_scale = rfmux.core.transferfunctions.VOLTS_PER_ROC 
    rfmux_scale = rfmux_scale / 256 / np.sqrt(2)
    scale_factor = np.full(ntones, fill_value = np.nan) 

    for module_idx in ch_map.keys():
        ares = ares_map[module_idx]
        ch_idxs = ch_map[module_idx] 
        pscale = 1 / 10 ** (ares / 20)
        scale_factor[ch_idxs] = rfmux_scale * pscale

    # Save scale_factor and dt  
    zarr_util.write_single_array(
        grp, 'counts_to_dbc', scale_factor, dtype = np.float64
    )
    zarr_util.write_single_array(
        grp, 'dt', dt, dtype = np.float64
    )

    ### Batch process parser file
    # Open files 
    module_idxs = list(ch_map.keys())
    file_paths = [
        os.path.join(path, f'serial_{crs_sn:04d}', 'm0%d_raw32'%(module))
        for module in module_idxs
    ] 
    files = [open(fp, 'rb') for fp in file_paths]

    # Setup batches and initialize Zarr array 
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    record_size = dtype.itemsize
    target_bytes = batch_size_mb * (1024 ** 2)
    batch_size = int(target_bytes // (record_size * len(files)))
    batch_size = (batch_size // max_ntones) * max_ntones
    z_out = grp.create_array(
        name = 'z', 
        shape = (2, ntones, 0), 
        chunks = (2, ntones, batch_size), 
        dtype = np.int32
    )

    # Process batches
    try:
        batch_idx = 0
        max_batch_N = batch_size // max_ntones
        z_real_buf = np.empty((ntones, max_batch_N), dtype = np.int32)
        z_imag_buf = np.empty((ntones, max_batch_N), dtype = np.int32)

        while True:
            batch_parts = [
                np.fromfile(
                    f, 
                    dtype = dtype, 
                    count = batch_size
                )
                for f in files
            ]
            N = min([b.shape[0] // (max_ntones) 
                    for b in batch_parts]) 
            if N == 0:
                # End when one or more files ends
                break
            
            z_real = z_real_buf[:, :N]
            z_imag = z_imag_buf[:, :N]
            for module_idx,  parser_dat  in zip(
                module_idxs, batch_parts
            ):
                zi_real = parser_dat['i'][:N * max_ntones]
                zi_imag = parser_dat['q'][:N * max_ntones]
                ch_idxs = ch_map[module_idx]
                z_real[ch_idxs] = zi_real.reshape(N, max_ntones).T
                z_imag[ch_idxs] = zi_imag.reshape(N, max_ntones).T            
            
            t0 = z_out.shape[2] 
            t1 = t0 + N 
            z_out.resize((z_out.shape[0], z_out.shape[1], 
                        z_out.shape[2] + N))
            z_out[0, :, t0:t1] = z_real 
            z_out[1, :, t0:t1] = z_imag 
            
            batch_idx += 1
            batch_parts = None # free memory
    finally:
        for f in files:
            f.close()

def import_ts(data_path, scale_factor_path):
    """
    Import timestream data as saved by the batch processor.

    Parameters:
    data_path (str): path to the complex IQ data.
    scale_factor_path (str): path to the scale factor data.

    Returns:
    z (np.ndarray): data scaled by scale_factor in complex128.
    """
    raise NotImplementedError("import_ts is deprecated. Need to update convert_parser")
    i, q = np.load(data_path).astype(np.int32)
    scale_factor = np.load(scale_factor_path).astype(np.float64)
    z = np.empty(i.shape, dtype = np.complex128)
    np.multiply(i, scale_factor, out = z.real)
    np.multiply(q, scale_factor, out = z.imag)
    return z

def estimate_ts_data_size(dec_stage, total_time, nmodules, max_ntones, ntones):
    """
    Estimate and print raw and processed timestream data sizes.

    Parameters:
    dec_stage (int): decimation stage.
    total_time (float): timestream length in s.
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
    if total_time < 0:
        raise ValueError('total_time must be positive')
    if nmodules not in [1, 2, 3, 4]:
        raise ValueError('nmodules must be in [1, 2, 3, 4]')
    if type(max_ntones) != int or max_ntones < 0 or max_ntones > 1024:
        raise ValueError('max_ntones must be an int in range [0, 1024]')
    if type(ntones) != int or ntones < 0 or ntones > 1024 * 4:
        raise ValueError('ntones must be an int in range [0, 4 * 1024]')
    # Calculate file sizes
    sample_frequency = get_sample_freq(dec_stage)
    size_per_ch = 4 * 2 * (total_time * sample_frequency)
    size_raw = size_per_ch * nmodules * max_ntones + 103
    size_processed = size_per_ch * ntones
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

################################################################################
########################### network interface check ############################
################################################################################
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

