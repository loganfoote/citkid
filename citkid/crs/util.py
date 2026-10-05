import numpy as np
import os
import rfmux
import psutil
import socket
import time
import zarr
from tqdm.auto import tqdm
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
def parser_to_zarr(
    path, grp, crs_sn, ntones, max_ntones, ch_map, ares_map, dt,
    batch_size_mb = 1_000, chunk_size_mb = 128
):
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
    _write_counts_to_s21_and_dt(grp, ntones, ch_map, ares_map, dt)

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
    # Total bytes read per batch across all files is ~batch_size_mb
    # Each file stores max_ntones records per time sample
    read_count = max(
        max_ntones, batch_size_bytes // (len(files) * dtype.itemsize)
        )
    # read_count must be multiple of max_ntones
    read_count = (read_count // max_ntones) * max_ntones
    samples_per_file = [
        (os.path.getsize(fp) // dtype.itemsize) // max_ntones
        for fp in file_paths
    ]
    total_samples = int(min(samples_per_file)) if samples_per_file else 0

    # Initialize output Zarr array with full known shape
    z_out, _ = _create_z_array(grp, ntones, total_samples, chunk_size_mb)

    # Process batches
    samples_per_batch = read_count // max_ntones
    n_batches = int(np.ceil(total_samples / samples_per_batch)) if total_samples > 0 else 0
    try:
        # Pre-allocate buffers
        z_real_buf = np.zeros((ntones, samples_per_batch), dtype = np.int32)
        z_imag_buf = np.zeros((ntones, samples_per_batch), dtype = np.int32)

        for batch_idx in tqdm(
            range(n_batches), 
            total = n_batches, 
            desc = 'Processing batches', 
            leave = False
            ):
            t0 = batch_idx * samples_per_batch
            N = min(samples_per_batch, total_samples - t0)
            count = N * max_ntones
            batch_parts = [
                np.fromfile(
                    f, 
                    dtype = dtype, 
                    count = count
                )
                for f in files
            ]
            
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
            for module_idx, parser_dat in zip(module_idxs, batch_parts):
                zi_real = parser_dat['i'][:N * max_ntones]
                zi_imag = parser_dat['q'][:N * max_ntones]
                ch_idxs = ch_map[module_idx]
                # Extract only the channels we need from the parser data
                n_ch = len(ch_idxs)
                z_real[ch_idxs] = zi_real.reshape(N, max_ntones)[:, :n_ch].T
                z_imag[ch_idxs] = zi_imag.reshape(N, max_ntones)[:, :n_ch].T

            t1 = t0 + N
            z_out[0, :, t0:t1] = z_real
            z_out[1, :, t0:t1] = z_imag

            batch_parts = None # free memory
    finally:
        for f in files:
            f.close()

def _write_counts_to_s21_and_dt(grp, ntones, ch_map, ares_map, dt):
    """
    Write the counts-to-S21 scale factors and the sample time to a Zarr group.

    Parameters:
    grp (zarr.Group): Zarr group to save data.
    ntones (int): number of tones.
    ch_map (dict): keys (int) are module indices and values (np.array int)
        are channel indices.
    ares_map (dict): keys (int) are module indices and values (np.array
        float) are tone powers in dBm. Used to scale from CRS counts to dBc.
    dt (float): sample time in seconds.

    Returns:
    None
    """
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

def _create_z_array(grp, ntones, total_samples, chunk_size_mb):
    """
    Create the int32 timestream array 'z' with shape (2, ntones,
    total_samples) in a Zarr group.

    Parameters:
    grp (zarr.Group): Zarr group to save data.
    ntones (int): number of tones.
    total_samples (int): number of time samples.
    chunk_size_mb (float): target size of each shard along the time axis, in
        MB. The chunk length is capped to total_samples.

    Returns:
    z_out (zarr.Array): the created array.
    chunk_N (int): chunk length along the time axis, in samples.
    """
    chunk_size_bytes = int(chunk_size_mb * (1024 ** 2))
    chunk_N = max(
        1,
        chunk_size_bytes // (2 * ntones * np.dtype(np.int32).itemsize)
    )
    # Cap chunk length to total available samples to avoid oversized tail chunk
    if total_samples > 0:
        chunk_N = min(chunk_N, total_samples)

    # Shard covers all channels but only one chunk along the time axis,
    # so each batch write lands in its own shard file without touching others.
    if total_samples > 0:
        z_shards = (2, ntones, chunk_N)
    else:
        z_shards = None

    z_out = grp.create_array(
        name = 'z',
        shape = (2, ntones, total_samples),
        chunks = (1, 1, chunk_N),
        shards = z_shards,
        dtype = np.int32
    )
    return z_out, chunk_N

################################################################################
############################## memory streaming ################################
################################################################################
# Maximum fraction of the currently available memory that a memory stream may
# use
MEMORY_STREAM_MAX_FRACTION = 0.5

def estimate_memory_stream_bytes(
    nmodules, max_ntones, ntones, nframes, chunk_size_mb
):
    """
    Estimate the peak memory used by a memory stream (``CRS.stream`` with
    ``method = 'memory'``): the receive buffers plus the blocks used to write
    to Zarr.

    Parameters:
    nmodules (int): number of streamed modules.
    max_ntones (int): maximum number of tones on a module.
    ntones (int): total number of tones.
    nframes (int): number of time samples per module.
    chunk_size_mb (float): Zarr chunk size along the time axis, in MB.

    Returns:
    int: estimated peak memory in bytes.
    """
    itemsize = 2 * np.dtype(np.int32).itemsize
    buffers = nmodules * nframes * max_ntones * itemsize
    # One block assembled in numpy, plus one being encoded by zarr
    block = min(int(chunk_size_mb * 1024 ** 2), ntones * nframes * itemsize)
    return int(buffers + 2 * block)

def check_memory_for_stream(
    nmodules, max_ntones, ntones, nframes, chunk_size_mb
):
    """
    Check that a memory stream fits in the available memory with headroom.

    The estimate from ``estimate_memory_stream_bytes`` must not exceed
    ``MEMORY_STREAM_MAX_FRACTION`` of the currently available memory.

    Parameters:
    nmodules (int): number of streamed modules.
    max_ntones (int): maximum number of tones on a module.
    ntones (int): total number of tones.
    nframes (int): number of time samples per module.
    chunk_size_mb (float): Zarr chunk size along the time axis, in MB.

    Returns:
    int: estimated peak memory in bytes.

    Raises:
    MemoryError: if the estimate exceeds the allowed fraction of available
        memory.
    """
    required = estimate_memory_stream_bytes(nmodules, max_ntones, ntones,
                                            nframes, chunk_size_mb)
    available = psutil.virtual_memory().available
    allowed = MEMORY_STREAM_MAX_FRACTION * available
    if required > allowed:
        raise MemoryError(
            f'The memory stream needs about {required / 1e9:.2f} GB, but only '
            f'{allowed / 1e9:.2f} GB ({MEMORY_STREAM_MAX_FRACTION:.0%} of the '
            f'{available / 1e9:.2f} GB available) may be used. Shorten the '
            "timestream, use fewer tones, or use method = 'parser'."
            )
    return required

def capture_to_memory(
    interface, serial_number, module_idxs, max_ntones, nframes, timeout_s = 10.0
):
    """
    Receive streamed readout packets into memory. Uses the same packet
    receiver as the rfmux parser, and stores the same raw int32 I/Q values
    that the parser writes to disk.

    Stops when every module has nframes samples. Missing packets are not
    filled in, so a module's samples are contiguous like the parser's.

    Parameters:
    interface (str): Ethernet interface identifier e.g., 'enp2s0'.
    serial_number (int): CRS serial number. Packets from other boards are
        ignored.
    module_idxs (array-like int): modules to capture (1-8). Other modules are
        ignored.
    max_ntones (int): number of channels to keep per module (channels
        1 to max_ntones).
    nframes (int): number of time samples to capture per module.
    timeout_s (float): raise an error if a module that is not finished
        receives no packets for this long, in seconds.

    Returns:
    buffers (dict): keys (int) are module indices and values (np.array int32)
        are arrays of shape (nframes, max_ntones, 2) with I and Q along the
        last axis.
    dropped (dict): keys (int) are module indices and values (int) are the
        number of packets missing from the sequence.

    Raises:
    RuntimeError: if a module stops receiving packets, or packets have fewer
        than max_ntones channels.
    """
    from rfmux import streamer
    from rfmux.tools.parser import resolve_interface

    # Input validation
    module_idxs = [int(mi) for mi in module_idxs]
    if not module_idxs or any(mi not in range(1, 9) for mi in module_idxs):
        raise ValueError('module_idxs must be a non-empty list in [1, 8]')
    if len(set((mi - 1) % 4 for mi in module_idxs)) != len(module_idxs):
        raise ValueError('module_idxs must not contain the same module in '
                         'both analog banks')
    max_ntones, nframes = int(max_ntones), int(nframes)
    if max_ntones <= 0 or nframes <= 0:
        raise ValueError('max_ntones and nframes must be positive')
    serial_number = int(serial_number)

    # Allocate buffers
    buffers = {mi: np.empty((nframes, max_ntones, 2), dtype = np.int32)
               for mi in module_idxs}
    counts = {mi: 0 for mi in module_idxs}
    dropped = {mi: 0 for mi in module_idxs}
    last_seq = {mi: None for mi in module_idxs}
    # Packets number modules 0-3 within the active analog bank
    packet_module_map = {(mi - 1) % 4: mi for mi in module_idxs}

    interface_ip = resolve_interface(interface)
    with streamer.get_multicast_socket(
        crs_hostname = None,
        port = streamer.STREAMER_PORT,
        interface = interface_ip,
        buffer_size = 67108864,
    ) as sock:
        receiver = streamer.ReadoutPacketReceiver(sock, reorder_window = 256)
        start = time.monotonic()
        last_packet = {mi: start for mi in module_idxs}
        while any(counts[mi] < nframes for mi in module_idxs):
            receiver.receive_batch(256, timeout_ms = 1000)
            now = time.monotonic()
            for serial, module, queue in receiver.get_all_queues():
                mi = packet_module_map.get(module) \
                    if serial == serial_number else None
                while (packet := queue.try_pop()):
                    # Packets from other boards and modules are discarded
                    if mi is None or counts[mi] >= nframes:
                        continue
                    pkt = packet.to_python()
                    if last_seq[mi] is not None:
                        dropped[mi] += (pkt.seq - last_seq[mi] - 1) \
                                       & 0xFFFFFFFF
                    last_seq[mi] = pkt.seq
                    raw = np.asarray(pkt.raw_samples)
                    if raw.size < 2 * max_ntones:
                        raise RuntimeError(
                            f'Packets from module {mi} have {raw.size // 2} '
                            f'channels, but {max_ntones} are needed. The CRS '
                            'is streaming short packets.'
                            )
                    buffers[mi][counts[mi]] = \
                        raw[:2 * max_ntones].reshape(max_ntones, 2)
                    counts[mi] += 1
                    last_packet[mi] = now
            stalled = [mi for mi in module_idxs
                       if counts[mi] < nframes and
                       now - last_packet[mi] > timeout_s]
            if stalled:
                raise RuntimeError(
                    f'No packets received from modules {stalled} for '
                    f'{timeout_s} s. Check that they are streaming (see '
                    'CRS.set_decimation).'
                    )
    return buffers, dropped

def memory_to_zarr(
    buffers, grp, ntones, max_ntones, ch_map, ares_map, dt, chunk_size_mb = 128
):
    """
    Save timestreams captured by ``capture_to_memory`` to a Zarr group, in the
    same format as ``parser_to_zarr``.

    Parameters:
    buffers (dict): keys (int) are module indices and values (np.array int32)
        are arrays of shape (nframes, max_ntones, 2) with I and Q along the
        last axis.
    grp (zarr.Group): Zarr group to save data.
    ntones (int): number of tones.
    max_ntones (int): maximum number of tones per module.
    ch_map (dict): channel index dictionary. Keys (int) are module indices.
        Values are lists where values (int) are channel indices.
    ares_map (dict): power dictionary. Keys (int) are module indices. Values
        are arrays where values (float) are power in dBm. Used to create
        scaling from CRS amplitude to dBc.
    dt (float): sample time in seconds.
    chunk_size_mb (float): target Zarr chunk size, in MB. The chunk length is
        capped to the total available samples.

    Returns:
    None
    """
    ### Input validation
    ch_map, ares_map, dt = _validate_zarr_output_inputs(
        grp, ntones, max_ntones, ch_map, ares_map, dt
        )
    chunk_size_mb = float(chunk_size_mb)
    if chunk_size_mb <= 0:
        raise ValueError('chunk_size_mb must be a positive float')
    module_idxs = [k for k in ch_map.keys() if len(ch_map[k]) > 0]
    for mi in module_idxs:
        if mi not in buffers:
            raise ValueError(f'buffers does not contain module {mi}')
        if buffers[mi].ndim != 3 or buffers[mi].shape[1:] != (max_ntones, 2):
            raise ValueError(f'buffers[{mi}] must have shape '
                             f'(nframes, {max_ntones}, 2)')

    ### Write scale_factor and dt
    _write_counts_to_s21_and_dt(grp, ntones, ch_map, ares_map, dt)

    ### Write timestreams one shard at a time
    total_samples = min(buffers[mi].shape[0] for mi in module_idxs) \
        if module_idxs else 0
    z_out, chunk_N = _create_z_array(grp, ntones, total_samples, chunk_size_mb)
    block_buf = np.zeros((2, ntones, chunk_N), dtype = np.int32)
    for t0 in range(0, total_samples, chunk_N):
        N = min(chunk_N, total_samples - t0)
        block = block_buf[:, :, :N]
        block[...] = 0
        for mi in module_idxs:
            ch_idxs = ch_map[mi]
            data = buffers[mi][t0:t0 + N, :len(ch_idxs)]
            block[0, ch_idxs] = data[:, :, 0].T
            block[1, ch_idxs] = data[:, :, 1].T
        z_out[:, :, t0:t0 + N] = block

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
    unit = 'MB'
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
    ``dec_module_idxs`` and ``dec_short`` are saved as None if they are
    unknown (sessions without global control).

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

    # Save decimation parameters as attributes. dec_module_idxs and dec_short
    # are None in sessions without global control, which can't read them
    if crs.dec_module_idxs is None:
        grp.attrs['dec_module_idxs'] = None
    else:
        grp.attrs['dec_module_idxs'] = np.asarray(
            crs.dec_module_idxs, dtype=np.uint8
        ).tolist()
    if crs.dec_short is None:
        grp.attrs['dec_short'] = None
    else:
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
    Per-module settings in ``crs.module_cfg`` are saved as
    ``<setting>_module<idx>`` attributes, e.g. ``adc_attenuation_db_module1``.

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
                 'rfmux_version', 'citkid_version', 'module_cfg']:
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
    # Add nco_freqs and module_cfg module-specific attributes
    for module_idx in crs.nco_freqs.keys():
        required_attrs.add(f'nco_module{module_idx:d}')
    for module_idx, cfg in crs.module_cfg.items():
        for key in cfg.keys():
            required_attrs.add(f'{key}_module{module_idx:d}')
    
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
    
    # Save per-module settings as attributes (one per module and setting)
    for module_idx, cfg in crs.module_cfg.items():
        for key, value in cfg.items():
            if isinstance(value, (bool, np.bool_)):
                value = bool(value)
            elif isinstance(value, (int, np.integer)):
                value = int(value)
            elif isinstance(value, (float, np.floating)):
                value = float(value)
            else:
                value = str(value)
            grp.attrs[f'{key}_module{module_idx:d}'] = value

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
############################# Sweep Helpers ####################################
################################################################################ 
def piecewise_geomspace(x0, x1, bw, npoints_per_ch, nchs = 1024):
    """
    Space values approximately geometrically between x0 and
    x1, while ensuring that every chunk of bw space has 
    exactly nchs * npoints_per_ch tones. 

    Parameters:
    x0 (float): start frequency in Hz.
    x1 (float): end frequency in Hz.
    bw (float): bandwidth in Hz.
    npoints_per_ch (int): number of points per channel.
    nchs (int): number of channels per bandwidth. 

    Returns:
    np.ndarray: concatenated array of geometrically spaced values.
    """
    n = npoints_per_ch * nchs

    # Linear bin edges
    edges = np.arange(x0, x1 + bw, bw)
    edges[-1] = x1   # ensure exact endpoint

    xs = []

    for a, b in zip(edges[:-1], edges[1:]):
        xs.append(np.geomspace(a, b, n, endpoint=False))

    return np.concatenate(xs)

################################################################################
########################### input validation ###################################
################################################################################ 
def _validate_parser_to_zarr_inputs(
    path, grp, crs_sn, ntones, max_ntones, ch_map, ares_map, dt, batch_size_mb,
    chunk_size_mb
):
    """Validate inputs for parser_to_zarr function."""
    if not isinstance(path, str):
        raise TypeError('path must be a str')
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise ValueError(f'path {path} is not a valid directory')

    crs_sn = int(crs_sn)
    if crs_sn < 0:
        raise ValueError('crs_sn must be a positive int')

    ch_map, ares_map, dt = _validate_zarr_output_inputs(
        grp, ntones, max_ntones, ch_map, ares_map, dt
        )

    batch_size_mb, chunk_size_mb = _validate_batch_chunk_sizes(
        batch_size_mb, chunk_size_mb
    )
    return path, crs_sn, ch_map, ares_map, dt, batch_size_mb, chunk_size_mb

def _validate_zarr_output_inputs(grp, ntones, max_ntones, ch_map, ares_map, dt):
    """
    Validate the inputs shared by ``parser_to_zarr`` and ``memory_to_zarr``.

    Parameters:
    See docstring of parser_to_zarr for parameter descriptions.

    Returns:
    ch_map (dict): validated ch_map with int32 array values.
    ares_map (dict): ares_map with float64 array values.
    dt (float): sample time in seconds.
    """
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError('grp must be a zarr.core.group.Group object')
    # Check that required names don't already exist in the group
    existing_names = set(grp.keys())
    required_names = {'counts_to_s21', 'dt', 'z'}
    conflicts = existing_names & required_names
    if conflicts:
        msg = f'grp already contains required names: {sorted(conflicts)}'
        raise ValueError(msg)

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
    return ch_map, ares_map, dt

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

