import numpy as np
import os
import rfmux
import socket 
import zarr 
from typing import TYPE_CHECKING
from .. import zarr_util

if TYPE_CHECKING:
    from .instrument import CRS

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

    # Short-circuit if no NCOs provided
    if len(nco_freqs) == 0:
        return {}, np.arange(freqs.shape[0], dtype = np.int32)

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
    if not all(isinstance(i, (int, np.integer)) for i in module_idxs):
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
    if not isinstance(dec_stage, (int, np.integer)) or dec_stage < 0 or dec_stage > 6:
        raise ValueError("dec_stage must be an int in range [0, 6]")
    
    # Calculate sample frequency
    return 625e6 / (256 * 64 * 2 ** dec_stage)

################################################################################
############################## parser processing ###############################
################################################################################
def parser_to_zarr(path, grp, crs_sn, ntones, max_ntones, 
                   ch_map, ares_map, dt, batch_size_mb = 1_000,
                   chunk_size_mb = 128):
    """
    Import parser file data in batches and reformat for channels of interest. 
    Save to a Zarr file.

    Saves each batch as int32 data, to later be scaled by the factors saved by
    `CRS.capture_ts`/`CRS.stream`.

    Parameters:
    path (str): path to the parser folder.
    grp (zarr.hierarchy.Group): Zarr group to save data.
    crs_sn (int): CRS serial number.
    ntones (int): number of tones.
    max_ntones (int): maximum number of tones per module.
    ch_map (dict): channel index dictionary. Keys (int) are module indices.
        Values are lists where values (int) are channel indices.
    ares_map (dict): power dictionary. Keys (int) are module indices. Values are
        arrays where values (float) are power in dBm. Used to create scaling 
        from CRS amplitude to dBc.
    dt (float): sample time in seconds.
    batch_size_mb (float): total input read size per batch, in MB.
    chunk_size_mb (float): target Zarr chunk size, in MB. If None,
        defaults to batch_size_mb. The chunk length is capped to the total
        available samples to avoid oversized final chunks.

    Returns:
    None
    """
    ### Input validation 
    path, crs_sn, ch_map, ares_map, dt, batch_size_mb, chunk_size_mb = \
    _validate_parser_to_zarr_inputs(
        path, grp, crs_sn, ntones, max_ntones, ch_map, ares_map, dt, 
        batch_size_mb, chunk_size_mb
    )
        
    ### Write scale_factor and dt
    rfmux_scale = rfmux.core.transferfunctions.VOLTS_PER_ROC 
    rfmux_scale = rfmux_scale / 256 / np.sqrt(2)
    scale_factor = np.full(ntones, fill_value = np.nan) 

    for module_idx in ch_map.keys():
        # Use ares to modify scale_factor from dBm to dBc 
        ares = ares_map[module_idx]
        ch_idxs = ch_map[module_idx] 
        pscale = 1 / 10 ** (ares / 20)
        scale_factor[ch_idxs] = rfmux_scale * pscale

    # Save scale_factor and dt  
    zarr_util.write_single_array(
        grp, 'counts_to_s21', scale_factor, dtype = np.float64
    )
    zarr_util.write_single_array(
        grp, 'dt', dt, dtype = np.float64
    )

    ### Batch process parser file
    # Open files 
    module_idxs = list([k for k in ch_map.keys() if len(ch_map[k]) > 0])
    file_paths = [
        # module idxs are 1 - 4 in parser data, regardless of analog_bank_high
        # might be changed in future release
        os.path.join(path, f'serial_{crs_sn:04d}', 
                     'm0%d_raw32'%(module - (module // 5) * 4))
        for module in module_idxs
    ] 
    files = [open(fp, 'rb') for fp in file_paths]

    # Setup batches and initialize Zarr array 
    dtype = np.dtype([('i', np.int32), ('q', np.int32)])
    batch_size_bytes = int(batch_size_mb * (1024 ** 2))
    chunk_size_bytes = int(
        chunk_size_mb * (1024 ** 2)
    )
    # Total bytes read per batch across all files is ~batch_size_mb
    # Each file stores max_ntones records per time sample
    read_count = max(
        max_ntones, batch_size_bytes // (len(files) * dtype.itemsize)
        )
    # read_count must be multiple of max_ntones
    read_count = (read_count // max_ntones) * max_ntones
    # Chunk length in time samples for output array (decoupled from read size)
    chunk_N = max(
        1,
        chunk_size_bytes // (2 * ntones * np.dtype(np.int32).itemsize)
    )
    # Cap chunk length to total available samples to avoid oversized tail chunk
    samples_per_file = [
        (os.path.getsize(fp) // dtype.itemsize) // max_ntones
        for fp in file_paths
    ]
    total_samples = int(min(samples_per_file)) if samples_per_file else 0
    if total_samples > 0:
        chunk_N = min(chunk_N, total_samples)

    # Compute shard: smallest multiple of chunk_N that covers all samples,
    # so all time data lands in one shard (one file).
    if total_samples > 0:
        shard_N = int(np.ceil(total_samples / chunk_N)) * chunk_N
        z_shards = (2, ntones, shard_N)
    else:
        z_shards = None

    # Initialize output Zarr array
    z_out = grp.create_array(
        name = 'z', 
        shape = (2, ntones, 0), 
        chunks = (1, 1, chunk_N), 
        shards = z_shards, 
        dtype = np.int32
    )

    # Process batches
    try:
        batch_idx = 0
        # Pre-allocate buffers (will be resized if N exceeds chunk_N)
        z_real_buf = np.zeros((ntones, chunk_N), dtype = np.int32)
        z_imag_buf = np.zeros((ntones, chunk_N), dtype = np.int32)

        while True:
            batch_parts = [
                np.fromfile(
                    f, 
                    dtype = dtype, 
                    count = read_count
                )
                for f in files
            ]
            N = min([b.shape[0] // (max_ntones) 
                     for b in batch_parts]) 
            if N == 0:
                # End when one or more files ends
                break
            
            # Resize buffers if needed
            if N > z_real_buf.shape[1]:
                z_real_buf = np.zeros((ntones, N), dtype = np.int32)
                z_imag_buf = np.zeros((ntones, N), dtype = np.int32)
            else:
                # Zero out the portion we'll use
                z_real_buf[:, :N] = 0
                z_imag_buf[:, :N] = 0
            
            z_real = z_real_buf[:, :N]
            z_imag = z_imag_buf[:, :N]
            for module_idx,  parser_dat  in zip(
                module_idxs, batch_parts
            ):
                zi_real = parser_dat['i'][:N * max_ntones]
                zi_imag = parser_dat['q'][:N * max_ntones]
                ch_idxs = ch_map[module_idx]
                # Extract only the channels we need from the parser data
                n_ch = len(ch_idxs)
                z_real[ch_idxs] = zi_real.reshape(N, max_ntones)[:, :n_ch].T
                z_imag[ch_idxs] = zi_imag.reshape(N, max_ntones)[:, :n_ch].T            
            
            t0 = z_out.shape[2] 
            t1 = t0 + N 
            z_out.resize(
                (z_out.shape[0], z_out.shape[1], z_out.shape[2] + N)
                )
            z_out[0, :, t0:t1] = z_real 
            z_out[1, :, t0:t1] = z_imag 
            
            batch_idx += 1
            batch_parts = None # free memory
    finally:
        for f in files:
            f.close()

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
    if not isinstance(dec_stage, int) or dec_stage < 0 or dec_stage > 6:
        raise ValueError('dec_stage must be an int in range [0, 6]')
    if total_time < 0:
        raise ValueError('total_time must be positive')
    if nmodules not in [1, 2, 3, 4]:
        raise ValueError('nmodules must be in [1, 2, 3, 4]')
    if not isinstance(max_ntones, int) or max_ntones < 0 or max_ntones > 1024:
        raise ValueError('max_ntones must be an int in range [0, 1024]')
    if not isinstance(ntones, int) or ntones < 0 or ntones > 1024 * 4:
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


################################################################################
# CRS config saving helpers
################################################################################
def write_acq_cfg_to_zarr(crs, grp):
    """
    Write CRS acquisition configuration to a Zarr group.
    
    This includes parameters that typically change between measurements:
    decimation settings, sample frequency, and channel mapping.

    Parameters:
    crs (CRS): initialized CRS instrument class.
    grp (zarr.Group): Zarr group to which configuration data is saved.

    Returns:
    None
    """
    # Input validation 
    if type(crs).__name__ != 'CRS' and type(crs).__name__ != 'DummyCRS':
        raise TypeError("crs must be an instance of CRS class.")
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.") 
    
    for name in ['dec_module_idxs', 'dec_short', 'dec_stage', 'ch_map', 'sample_freq']:
        if not hasattr(crs, name):
            raise ValueError(f"crs is missing attribute '{name}'.")
    
    # Check for attribute conflicts
    for name in ['dec_module_idxs', 'dec_short', 'dec_stage', 'sample_freq']:
        if name in grp.attrs.keys():
            raise ValueError(f"Zarr group already contains attribute '{name}'.")
    
    _validate_ch_map(crs.ch_map)
    
    # Check for array conflicts (ch_map)
    existing_arrays = set(grp.keys())
    required_arrays = set()
    for module_idx in crs.ch_map.keys():
        required_arrays.add(f'chs_module{module_idx:d}')
    
    array_conflicts = existing_arrays & required_arrays
    if array_conflicts:
        # Find first conflict for error message
        conflict_name = sorted(array_conflicts)[0]
        raise ValueError(
            f"Zarr group already contains dataset '{conflict_name}'."
        )

    # Save decimation parameters as attributes
    grp.attrs['dec_module_idxs'] = np.asarray(
        crs.dec_module_idxs, dtype=np.uint8
    ).tolist()
    grp.attrs['dec_short'] = bool(crs.dec_short)
    grp.attrs['dec_stage'] = int(np.uint8(crs.dec_stage))
    grp.attrs['sample_freq'] = float(crs.sample_freq)
    
    # Save ch_map as arrays (one per module) - can be large
    for module_idx, chs in crs.ch_map.items():
        grp.create_array(
            name=f'chs_module{module_idx:d}',
            data=np.asarray(chs, dtype=np.int32)
        )

def write_system_cfg_to_zarr(crs, grp):
    """
    Write CRS system configuration to a Zarr group as attributes.
    
    This includes static system parameters that don't change during a measurement
    procedure: NCO frequencies, firmware version, analog bank settings, etc.

    Parameters:
    crs (CRS): initialized CRS instrument class.
    grp (zarr.Group): Zarr group to which configuration data is saved.

    Returns:
    None
    """
    # Input validation 
    if type(crs).__name__ != 'CRS' and type(crs).__name__ != 'DummyCRS':
        raise TypeError("crs must be an instance of CRS class.")
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.") 
    
    for name in ['nco_freqs', 'firmware_release',
                 'analog_bank_high', 'bw', 'clock_source',
                 'extended_bw', 'serial_number',
                 'rfmux_version', 'citkid_version']:
        if not hasattr(crs, name):
            raise ValueError(f"crs is missing attribute '{name}'.")
        
    if not hasattr(crs.firmware_release, 'version') or \
        not isinstance(crs.firmware_release.version, str):
        raise ValueError("crs.firmware_release.version must be a string.")
    
    # Check for attribute conflicts
    existing_attrs = set(grp.attrs.keys())
    required_attrs = {
        'analog_bank_high', 'bw', 'clock_source', 'extended_bw',
        'serial_number', 'rfmux_version', 'citkid_version',
        'firmware_version'
    }
    # Add nco_freqs module-specific attributes
    for module_idx in crs.nco_freqs.keys():
        required_attrs.add(f'nco_module{module_idx:d}')
    
    conflicts = existing_attrs & required_attrs
    if conflicts:
        # Find first conflict for error message
        conflict_name = sorted(conflicts)[0]
        raise ValueError(
            f"Zarr group already contains attribute '{conflict_name}'."
            )
    
    # Save nco_freqs as attributes (one per module)
    for module_idx, nco in crs.nco_freqs.items():
        grp.attrs[f'nco_module{module_idx:d}'] = float(nco)
    
    # Save other configuration as attributes
    grp.attrs['firmware_version'] = str(crs.firmware_release.version)
    grp.attrs['analog_bank_high'] = bool(crs.analog_bank_high)
    grp.attrs['bw'] = float(crs.bw)
    grp.attrs['clock_source'] = str(crs.clock_source)
    grp.attrs['extended_bw'] = bool(crs.extended_bw)
    grp.attrs['serial_number'] = int(np.uint16(crs.serial_number))
    grp.attrs['rfmux_version'] = str(crs.rfmux_version)
    grp.attrs['citkid_version'] = str(crs.citkid_version)

################################################################################
########################### input validation ###################################
################################################################################ 
def _validate_parser_to_zarr_inputs(
    path, grp, crs_sn, ntones, max_ntones, ch_map, ares_map, dt, 
    batch_size_mb, chunk_size_mb
):
    """Validate inputs for parser_to_zarr function."""
    if not isinstance(path, str):
        raise TypeError('path must be a str')
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise ValueError(f'path {path} is not a valid directory')
     
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError('grp must be a zarr.core.group.Group object')
    # Check that required names don't already exist in the group
    existing_names = set(grp.keys())
    required_names = {'counts_to_s21', 'dt', 'z'}
    conflicts = existing_names & required_names
    if conflicts:
        msg = f'grp already contains required names: {sorted(conflicts)}'
        raise ValueError(msg) 
    
    crs_sn = int(crs_sn)
    if crs_sn < 0:
        raise ValueError('crs_sn must be a positive int')

    if not isinstance(ntones, (int, np.integer)) or ntones < 0:
        raise ValueError('ntones must be a positive int')
    
    if not isinstance(max_ntones, (int, np.integer)) or max_ntones <= 0:
        raise ValueError('max_ntones must be a positive int')
    
    ch_map = _validate_ch_map(ch_map)

    if not isinstance(ares_map, dict):
        raise TypeError('ares_map must be a dict')
    for k, v in ares_map.items():
        if k not in [1, 2, 3, 4, 5, 6, 7, 8]:
            raise ValueError('ares_map keys must be integer module indices')
        try:
            ares_map[k] = np.asarray(v, dtype = np.float64)
        except Exception as e:
            msg = f'ares_map values could not be converted to float arrays: {e}'
            raise ValueError(msg)
        
    dt = float(dt)
    if dt <= 0:
        raise ValueError('dt must be a positive float')
    
    batch_size_mb, chunk_size_mb = _validate_batch_chunk_sizes(
        batch_size_mb, chunk_size_mb
    )
    return path, crs_sn, ch_map, ares_map, dt, batch_size_mb, chunk_size_mb

def _validate_ch_map(ch_map):
    """
    Validate the ch_map dictionary format. Converts values to int32 arrays.

    Parameters:
    See docstring of write_tones for parameter description.

    Returns:
    None
    """
    if ch_map is not None:
        if not isinstance(ch_map, dict):
            raise TypeError('ch_map must be a dictionary') 
        ch_map = ch_map.copy() # to avoid modifying input
        for k, v in ch_map.items():
            if k not in [1, 2, 3, 4, 5, 6, 7, 8]:
                raise ValueError('ch_map keys must be integer module indices')
            try:
                ch_map[k] = np.asarray(v, dtype = np.int32)
            except Exception as e:
                raise ValueError(f'ch_map could not be converted to int32')
    return ch_map

def _validate_batch_chunk_sizes(batch_size_mb, chunk_size_mb):
    """
    Validate batch_size_mb and chunk_size_mb parameters.

    Parameters:
    batch_size_mb (float): batch size in MB.
    chunk_size_mb (float): chunk size in MB.

    Returns:
    tuple: validated (batch_size_mb, chunk_size_mb)
    """
    batch_size_mb = float(batch_size_mb)
    chunk_size_mb = float(chunk_size_mb)
    if batch_size_mb <= 0:
        raise ValueError('batch_size_mb must be a positive float')
    if chunk_size_mb <= 0:
        raise ValueError('chunk_size_mb must be a positive float')
    if chunk_size_mb > batch_size_mb:
        raise ValueError('chunk_size_mb must be <= batch_size_mb')
    return batch_size_mb, chunk_size_mb 

