import contextlib
import multiprocessing
import numpy as np
import os
import queue as queue_module
import rfmux
import psutil
import shutil
import socket
import threading
import time
import traceback
import warnings
import zarr
from multiprocessing import shared_memory
from multiprocessing import spawn as mp_spawn
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

def dbm_to_vrms(p_dbm):
    """
    Return the RMS voltage of a tone with the given power across the rfmux
    termination (50 Ohms). Dividing a measured RMS voltage by this value gives
    S21 relative to the tone power, so that 20 * log10(|S21|) is in dBc.

    Parameters:
    p_dbm (float or array-like float): tone power in dBm.

    Returns:
    float or np.array: RMS voltage in V.
    """
    p_w = 1e-3 * 10 ** (np.asarray(p_dbm, dtype = np.float64) / 10)
    return np.sqrt(rfmux.core.transferfunctions.TERMINATION * p_w)

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
############################### timestream output ##############################
################################################################################
def get_counts_to_s21(ntones, ch_map, ares_map):
    """
    Return the factors that scale streamed int32 counts to S21, normalized so
    that 20 log10|S21| is in dBc.

    Parameters:
    ntones (int): number of tones.
    ch_map (dict): keys (int) are module indices and values (np.array int)
        are channel indices.
    ares_map (dict): keys (int) are module indices and values (np.array
        float) are tone powers in dBm.

    Returns:
    np.array: float64 scale factors indexed by tone. NaN for tones that are
        not in ch_map.
    """
    rfmux_scale = rfmux.core.transferfunctions.VOLTS_PER_ROC
    rfmux_scale = rfmux_scale / 256 / np.sqrt(2)
    scale_factor = np.full(ntones, fill_value = np.nan)

    for module_idx in ch_map.keys():
        # Use ares to convert from RMS voltage to dBc
        ares = ares_map[module_idx]
        ch_idxs = np.asarray(ch_map[module_idx], dtype = np.int32)
        scale_factor[ch_idxs] = rfmux_scale / dbm_to_vrms(ares)
    return scale_factor

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
    zarr_util.write_single_array(
        grp, 'counts_to_s21', get_counts_to_s21(ntones, ch_map, ares_map),
        dtype = np.float64
    )
    zarr_util.write_single_array(
        grp, 'dt', dt, dtype = np.float64
    )

# Zarr layout of streamed timestreams. Each chunk holds one tone (I or Q) for
# one time block, and each shard holds every tone for one time block.
# Minimum time samples per chunk (64 KiB). Writing costs a fixed overhead per
# chunk, so short chunks make streaming slow when there are many tones
MIN_CHUNK_SAMPLES = 16384
# Minimum size of a shard, in bytes. Lengthens the chunks when there are few
# tones
MIN_SHARD_BYTES = 128 * 1024 ** 2
# Maximum fraction of the currently available memory used by the streaming
# buffers
STREAM_BUFFER_MAX_FRACTION = 0.25
# Receive blocks per module while streaming. Packets are lost if the writer
# falls this many shards behind
STREAM_NSLOTS = 3

def get_stream_chunk_length(ntones, nmodules, max_ntones, nframes):
    """
    Return the number of time samples per chunk (and shard) for streaming to
    Zarr.

    The chunk length is the longer of MIN_CHUNK_SAMPLES and the length that
    makes a shard MIN_SHARD_BYTES, capped so that the streaming buffers use
    at most STREAM_BUFFER_MAX_FRACTION of the available memory, and capped to
    nframes. Warns if the memory cap makes chunks shorter than
    MIN_CHUNK_SAMPLES, since streaming many tones may then drop packets.

    Parameters:
    ntones (int): number of tones.
    nmodules (int): number of streamed modules.
    max_ntones (int): channels received per module.
    nframes (int): number of time samples.

    Returns:
    int: time samples per chunk.
    """
    itemsize = np.dtype(np.int32).itemsize
    shard_bytes_per_sample = 2 * ntones * itemsize
    chunk_N = max(MIN_SHARD_BYTES // shard_bytes_per_sample, MIN_CHUNK_SAMPLES)

    # Receive blocks, plus a shard being assembled and one being written
    buffer_bytes_per_sample = (
        nmodules * STREAM_NSLOTS * 2 * max_ntones * itemsize
        + 2 * shard_bytes_per_sample
        )
    allowed = STREAM_BUFFER_MAX_FRACTION * psutil.virtual_memory().available
    max_N = int(allowed // buffer_bytes_per_sample)
    if max_N < min(chunk_N, MIN_CHUNK_SAMPLES, nframes):
        warnings.warn(
            f'Not enough memory for {MIN_CHUNK_SAMPLES} samples per chunk; '
            f'using {max(max_N, 1)}. Streaming many tones may drop packets.',
            UserWarning
            )
    return int(max(1, min(chunk_N, max_N, nframes)))

def _create_z_array(grp, ntones, total_samples, chunk_length):
    """
    Create the int32 timestream array 'z' with shape (2, ntones,
    total_samples) in a Zarr group. Each chunk holds one tone (I or Q) for
    chunk_length samples, and each shard holds every tone for chunk_length
    samples.

    Parameters:
    grp (zarr.Group): Zarr group to save data.
    ntones (int): number of tones.
    total_samples (int): number of time samples.
    chunk_length (int): time samples per chunk and shard (see
        ``get_stream_chunk_length``). Capped to total_samples.

    Returns:
    z_out (zarr.Array): the created array.
    chunk_N (int): chunk length along the time axis, in samples.
    """
    chunk_N = max(1, int(chunk_length))
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

def copy_z_array(z_src, grp, verbose = True):
    """
    Copy a timestream array 'z' to a Zarr group one shard at a time, keeping
    its shape, chunks, shards, and compressors.

    Parameters:
    z_src (zarr.Array): array to copy.
    grp (zarr.Group): Zarr group to copy the array into, as 'z'.
    verbose (bool): If True, displays a progress bar.

    Returns:
    zarr.Array: the copied array.
    """
    z_out = grp.create_array(
        name = 'z',
        shape = z_src.shape,
        chunks = z_src.chunks,
        shards = z_src.shards,
        dtype = z_src.dtype,
        compressors = z_src.compressors
    )
    step = z_src.shards[2] if z_src.shards else z_src.chunks[2]
    nsamples = z_src.shape[2]
    for t0 in tqdm(range(0, nsamples, step), desc = 'Copying', leave = False,
                   disable = not verbose):
        z_out[:, :, t0:t0 + step] = z_src[:, :, t0:t0 + step]
    return z_out

def local_store_path(grp):
    """
    Return the directory of a Zarr group in a local (directory) store.

    Parameters:
    grp (zarr.Group): Zarr group.

    Returns:
    str or None: the group's directory, or None if the store is not local.
    """
    if not isinstance(grp.store, zarr.storage.LocalStore):
        return None
    return os.path.join(str(grp.store.root), grp.path)

def check_free_space(path, nbytes):
    """
    Check that the drive holding path has nbytes free.

    Parameters:
    path (str): an existing path on the drive. If path does not exist, its
        closest existing parent is checked.
    nbytes (int): required space in bytes.

    Returns:
    None

    Raises:
    OSError: if less than nbytes is free.
    """
    path = os.path.abspath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    free = shutil.disk_usage(path).free
    if nbytes > free:
        raise OSError(
            f'Streaming needs up to {nbytes / 1e9:.1f} GB (uncompressed) in '
            f'{path}, but only {free / 1e9:.1f} GB is free.'
            )

################################################################################
############################## streaming limits ################################
################################################################################
# Highest rates verified to stream reliably, using
# benchmarks/crs_rate_sweep.py. Measured 2026-10-16 with a Realtek r8169 1 GbE
# NIC, an AMD Ryzen 7 PRO 5850U, and a USB SSD (280 MiB/s). After verifying a
# faster setup with the benchmarks, raise these limits.
# Ethernet frames per second received from the board, summed over the
# streamed modules. Each module sends one packet per sample: a short packet
# (128 channels) is one frame and a long packet (1024 channels) is
# LONG_PACKET_FRAMES frames. 57,220 frames/s (3 modules at stage 1, or 4
# modules with long packets at stage 4) streamed without NIC drops for 20 s,
# and lost about 1 in 360,000 packets over 5 min, because the NIC receive ring
# (256 frames) overflowed. At 76,294 frames/s (2 modules at stage 0) the NIC
# stops transmitting, which also cuts off control of the board.
MAX_FRAME_RATE = 3 * 625e6 / (256 * 64 * 2)
LONG_PACKET_FRAMES = 6
# Bytes per second written to Zarr by CRS.stream (4096 tones at stage 4)
MAX_STREAM_WRITE_RATE = 4096 * 2 * 4 * 625e6 / (256 * 64 * 2 ** 4)

def check_frame_rate(dec_stage, nmodules, short):
    """
    Check that streaming nmodules at dec_stage is within MAX_FRAME_RATE.

    Parameters:
    dec_stage (int): decimation stage.
    nmodules (int): number of streamed modules.
    short (bool): True for short packets (128 channels), False for long
        packets (1024 channels).

    Returns:
    float: frame rate in frames per second.

    Raises:
    ValueError: if the frame rate exceeds MAX_FRAME_RATE.
    """
    frames = 1 if short else LONG_PACKET_FRAMES
    rate = get_sample_freq(dec_stage) * nmodules * frames
    if rate > MAX_FRAME_RATE * (1 + 1e-9):
        raise ValueError(
            f'Streaming {nmodules} modules at decimation stage {dec_stage} '
            f'with {"short" if short else "long"} packets sends {rate:,.0f} '
            f'frames/s, which exceeds the verified limit of '
            f'{MAX_FRAME_RATE:,.0f} frames/s (util.MAX_FRAME_RATE). Stream '
            'fewer modules or use a higher decimation stage.'
            )
    return rate

def check_stream_write_rate(dec_stage, ntones):
    """
    Check that streaming ntones to Zarr at dec_stage is within
    MAX_STREAM_WRITE_RATE.

    Parameters:
    dec_stage (int): decimation stage.
    ntones (int): number of tones.

    Returns:
    float: write rate in bytes per second.

    Raises:
    ValueError: if the write rate exceeds MAX_STREAM_WRITE_RATE.
    """
    rate = get_sample_freq(dec_stage) * ntones * 2 * np.dtype(np.int32).itemsize
    if rate > MAX_STREAM_WRITE_RATE * (1 + 1e-9):
        raise ValueError(
            f'Streaming {ntones} tones at decimation stage {dec_stage} writes '
            f'{rate / 1024 ** 2:.1f} MiB/s, which exceeds the verified limit '
            f'of {MAX_STREAM_WRITE_RATE / 1024 ** 2:.1f} MiB/s '
            '(util.MAX_STREAM_WRITE_RATE). Use fewer tones or a higher '
            'decimation stage.'
            )
    return rate

################################################################################
############################## packet receiving ################################
################################################################################
# Maximum fraction of the currently available memory that
# ``CRS.stream_to_memory`` may use
MEMORY_STREAM_MAX_FRACTION = 0.5

def estimate_memory_stream_bytes(nmodules, max_ntones, ntones, nframes):
    """
    Estimate the peak memory used by ``CRS.stream_to_memory``: the int32
    receive buffers, the complex128 output, and the complex128 buffer used to
    convert one module.

    Parameters:
    nmodules (int): number of streamed modules.
    max_ntones (int): maximum number of tones on a module.
    ntones (int): total number of tones.
    nframes (int): number of time samples per module.

    Returns:
    int: estimated peak memory in bytes.
    """
    buffers = nmodules * nframes * max_ntones * 2 * np.dtype(np.int32).itemsize
    itemsize = np.dtype(np.complex128).itemsize
    return int(buffers + (ntones + max_ntones) * nframes * itemsize)

def check_memory_for_stream(nmodules, max_ntones, ntones, nframes):
    """
    Check that ``CRS.stream_to_memory`` fits in the available memory with
    headroom.

    The estimate from ``estimate_memory_stream_bytes`` must not exceed
    ``MEMORY_STREAM_MAX_FRACTION`` of the currently available memory.

    Parameters:
    nmodules (int): number of streamed modules.
    max_ntones (int): maximum number of tones on a module.
    ntones (int): total number of tones.
    nframes (int): number of time samples per module.

    Returns:
    int: estimated peak memory in bytes.

    Raises:
    MemoryError: if the estimate exceeds the allowed fraction of available
        memory.
    """
    required = estimate_memory_stream_bytes(nmodules, max_ntones, ntones,
                                            nframes)
    available = psutil.virtual_memory().available
    allowed = MEMORY_STREAM_MAX_FRACTION * available
    if required > allowed:
        raise MemoryError(
            f'The memory stream needs about {required / 1e9:.2f} GB, but only '
            f'{allowed / 1e9:.2f} GB ({MEMORY_STREAM_MAX_FRACTION:.0%} of the '
            f'{available / 1e9:.2f} GB available) may be used. Shorten the '
            'timestream, use fewer tones, or use CRS.stream to save to disk.'
            )
    return required

def _validate_capture_inputs(module_idxs, max_ntones, nframes, serial_number):
    """
    Validate the inputs shared by ``capture_to_memory`` and
    ``capture_to_zarr``.

    Parameters:
    See docstring of capture_to_memory for parameter descriptions.

    Returns:
    module_idxs (list of int): module indices.
    max_ntones (int): channels to keep per module.
    nframes (int): time samples per module.
    serial_number (int): CRS serial number.
    """
    module_idxs = [int(mi) for mi in module_idxs]
    if not module_idxs or any(mi not in range(1, 9) for mi in module_idxs):
        raise ValueError('module_idxs must be a non-empty list in [1, 8]')
    if len(set((mi - 1) % 4 for mi in module_idxs)) != len(module_idxs):
        raise ValueError('module_idxs must not contain the same module in '
                         'both analog banks')
    max_ntones, nframes = int(max_ntones), int(nframes)
    if max_ntones <= 0 or nframes <= 0:
        raise ValueError('max_ntones and nframes must be positive')
    return module_idxs, max_ntones, nframes, int(serial_number)

def _receive_blocks(
    interface, serial_number, module_idxs, max_ntones, nframes, blocks,
    free_slot, on_block, timeout_s = 10.0, poll = None
):
    """
    Receive streamed readout packets into per-module blocks of time samples.
    Uses the same packet receiver as the rfmux parser, and stores the raw
    int32 I/Q values.

    Stops when every module has nframes samples. Missing packets are not
    filled in, so a module's samples are contiguous.

    Parameters:
    interface, serial_number, module_idxs, max_ntones, nframes: see
        ``capture_to_memory``. Inputs must already be validated.
    blocks (np.array int32): block buffers of shape (nmodules, nslots,
        block_length, 2 * max_ntones), with I and Q interleaved along the
        last axis. The first axis is the module's position in module_idxs.
    free_slot (callable): free_slot(m) returns a free slot index for module
        position m, or None if every slot is in use. Packets received while
        no slot is free are lost, and are counted as dropped.
    on_block (callable): on_block(m, slot, k, rows) is called when module
        position m has filled its k-th block, stored in slot, with rows
        samples. rows is block_length except for a module's last block.
    timeout_s (float): raise an error if a module that is not finished
        receives no packets for this long, in seconds.
    poll (callable or None): if not None, called after every batch of
        packets.

    Returns:
    dropped (dict): keys (int) are module indices and values (int) are the
        number of packets missing from the saved samples.

    Raises:
    RuntimeError: if a module stops receiving packets, or packets have fewer
        than max_ntones channels.
    """
    from rfmux import streamer
    from rfmux.tools.parser import resolve_interface

    nmodules, nslots, block_length = blocks.shape[:3]
    nt2 = 2 * max_ntones
    # Views per module and slot, to keep indexing out of the packet loop
    views = [[blocks[m, s] for s in range(nslots)] for m in range(nmodules)]
    counts, rows, block_idx = [0] * nmodules, [0] * nmodules, [0] * nmodules
    slots, last_seq = [None] * nmodules, [None] * nmodules
    dropped = [0] * nmodules
    # Packets number modules 0-3 within the active analog bank
    packet_module_map = {(mi - 1) % 4: m for m, mi in enumerate(module_idxs)}

    interface_ip = resolve_interface(interface)
    with streamer.get_multicast_socket(
        crs_hostname = None,
        port = streamer.STREAMER_PORT,
        interface = interface_ip,
        buffer_size = 67108864,
    ) as sock:
        receiver = streamer.ReadoutPacketReceiver(sock, reorder_window = 256)
        start = time.monotonic()
        last_packet = [start] * nmodules
        while min(counts) < nframes:
            receiver.receive_batch(256, timeout_ms = 1000)
            now = time.monotonic()
            for serial, module, queue in receiver.get_all_queues():
                m = packet_module_map.get(module) \
                    if serial == serial_number else None
                while (packet := queue.try_pop()):
                    # Packets from other boards and modules are discarded
                    if m is None or counts[m] >= nframes:
                        continue
                    pkt = packet.to_python()
                    if last_seq[m] is not None:
                        dropped[m] += (pkt.seq - last_seq[m] - 1) & 0xFFFFFFFF
                    last_seq[m] = pkt.seq
                    last_packet[m] = now
                    if slots[m] is None:
                        slots[m] = free_slot(m)
                        if slots[m] is None:
                            dropped[m] += 1
                            continue
                    raw = np.asarray(pkt.raw_samples)
                    if raw.size < nt2:
                        raise RuntimeError(
                            f'Packets from module {module_idxs[m]} have '
                            f'{raw.size // 2} channels, but {max_ntones} are '
                            'needed. The CRS is streaming short packets.'
                            )
                    views[m][slots[m]][rows[m]] = raw[:nt2]
                    rows[m] += 1
                    counts[m] += 1
                    if rows[m] == block_length or counts[m] == nframes:
                        on_block(m, slots[m], block_idx[m], rows[m])
                        slots[m] = None
                        rows[m] = 0
                        block_idx[m] += 1
            stalled = [module_idxs[m] for m in range(nmodules)
                       if counts[m] < nframes and
                       now - last_packet[m] > timeout_s]
            if stalled:
                raise RuntimeError(
                    f'No packets received from modules {stalled} for '
                    f'{timeout_s} s. Check that they are streaming (see '
                    'CRS.set_decimation).'
                    )
            if poll is not None:
                poll()
    return {mi: dropped[m] for m, mi in enumerate(module_idxs)}

def capture_to_memory(
    interface, serial_number, module_idxs, max_ntones, nframes, timeout_s = 10.0
):
    """
    Receive streamed readout packets into memory. Uses the same packet
    receiver as the rfmux parser, and stores the raw int32 I/Q values.

    Stops when every module has nframes samples. Missing packets are not
    filled in, so a module's samples are contiguous.

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
    module_idxs, max_ntones, nframes, serial_number = \
        _validate_capture_inputs(module_idxs, max_ntones, nframes,
                                 serial_number)

    # One block per module, holding the whole timestream
    blocks = np.empty((len(module_idxs), 1, nframes, 2 * max_ntones),
                      dtype = np.int32)
    dropped = _receive_blocks(
        interface, serial_number, module_idxs, max_ntones, nframes, blocks,
        free_slot = lambda m: 0, on_block = lambda m, slot, k, rows: None,
        timeout_s = timeout_s
        )
    buffers = {mi: blocks[m, 0].reshape(nframes, max_ntones, 2)
               for m, mi in enumerate(module_idxs)}
    return buffers, dropped

def capture_to_zarr(
    interface, serial_number, module_idxs, max_ntones, nframes, z_out, ch_map,
    writer = None, nslots = STREAM_NSLOTS, timeout_s = 10.0
):
    """
    Receive streamed readout packets and write them to a Zarr timestream
    array while streaming.

    Packets are received into blocks of one shard along the time axis per
    module. Once every module has filled its k-th block, a writer sorts the
    blocks into tone order (see ch_map) and writes shard k of z_out. The
    writer runs in a separate process for arrays in a local (directory)
    store, so that compressing and writing do not slow down receiving.

    Stops when every module has nframes samples. Missing packets are not
    filled in, so a module's samples are contiguous.

    Parameters:
    interface (str): Ethernet interface identifier e.g., 'enp2s0'.
    serial_number (int): CRS serial number. Packets from other boards are
        ignored.
    module_idxs (array-like int): modules to capture (1-8). Other modules are
        ignored.
    max_ntones (int): number of channels received per module (channels
        1 to max_ntones).
    nframes (int): number of time samples to capture per module.
    z_out (zarr.Array): int32 array of shape (2, ntones, nframes), created
        by ``_create_z_array``. Tones that are not in ch_map are left as 0.
    ch_map (dict): keys (int) are module indices and values (array-like int)
        are the tone indices in z_out of the module's channels 1 to
        len(ch_map[module]).
    writer (str or None): 'process' or 'thread'. If None (default), uses a
        process for arrays in a local store and a thread otherwise.
    nslots (int): blocks per module. Packets are lost (and counted as
        dropped) if the writer falls this many shards behind.
    timeout_s (float): raise an error if a module that is not finished
        receives no packets for this long, in seconds.

    Returns:
    dropped (dict): keys (int) are module indices and values (int) are the
        number of packets missing from the saved samples.

    Raises:
    RuntimeError: if a module stops receiving packets, packets have fewer
        than max_ntones channels, or the writer fails.
    """
    ### Input validation
    module_idxs, max_ntones, nframes, serial_number = \
        _validate_capture_inputs(module_idxs, max_ntones, nframes,
                                 serial_number)
    if z_out.ndim != 3 or z_out.shape[0] != 2 or z_out.shape[2] != nframes:
        raise ValueError(f'z_out must have shape (2, ntones, {nframes})')
    ch_idxs = [np.asarray(ch_map.get(mi, []), dtype = np.int32)
               for mi in module_idxs]
    if any(len(c) > max_ntones for c in ch_idxs):
        raise ValueError('ch_map has more channels than max_ntones')
    if writer is None:
        writer = 'process' if isinstance(z_out.store, zarr.storage.LocalStore) \
            else 'thread'
    if writer not in ('process', 'thread'):
        raise ValueError("writer must be 'process', 'thread', or None")
    nslots = int(nslots)
    if nslots < 1:
        raise ValueError('nslots must be a positive int')

    ### Set up the block buffers and writer
    block_length = z_out.shards[2] if z_out.shards else z_out.chunks[2]
    shape = (len(module_idxs), nslots, block_length, 2 * max_ntones)
    shm = None
    if writer == 'process':
        ctx = multiprocessing.get_context('spawn')
        shm = shared_memory.SharedMemory(
            create = True, size = int(np.prod(shape)) * 4)
        blocks = np.ndarray(shape, dtype = np.int32, buffer = shm.buf)
        work_q, status_q = ctx.Queue(), ctx.Queue()
        free_qs = [ctx.Queue() for _ in module_idxs]
        worker = ctx.Process(
            target = _zarr_writer, daemon = True,
            args = (z_out, ch_idxs, shape, shm.name, None, work_q, free_qs,
                    status_q))
    else:
        blocks = np.empty(shape, dtype = np.int32)
        work_q, status_q = queue_module.Queue(), queue_module.Queue()
        free_qs = [queue_module.Queue() for _ in module_idxs]
        worker = threading.Thread(
            target = _zarr_writer, daemon = True,
            args = (z_out, ch_idxs, shape, None, blocks, work_q, free_qs,
                    status_q))
    # Touch the pages now, so page faults don't slow down receiving
    blocks.fill(0)
    local_free = [list(range(nslots)) for _ in module_idxs]

    def free_slot(m):
        if local_free[m]:
            return local_free[m].pop()
        try:
            return free_qs[m].get_nowait()
        except queue_module.Empty:
            return None

    def on_block(m, slot, k, rows):
        work_q.put((m, slot, k, rows))

    def check_writer():
        if not status_q.empty():
            raise RuntimeError(f'The Zarr writer failed:\n{status_q.get()}')

    ### Stream
    try:
        with _spawn_without_main():
            worker.start()
        status = status_q.get(timeout = 120)
        if status != 'ready':
            raise RuntimeError(f'The Zarr writer failed to start:\n{status}')
        dropped = _receive_blocks(
            interface, serial_number, module_idxs, max_ntones, nframes,
            blocks, free_slot, on_block, timeout_s = timeout_s,
            poll = check_writer
            )
        work_q.put(None)
        status = status_q.get()
        if status != 'done':
            raise RuntimeError(f'The Zarr writer failed:\n{status}')
    finally:
        if worker.is_alive():
            work_q.put(None)
            worker.join(timeout = 60)
            if writer == 'process' and worker.is_alive():
                worker.terminate()
        blocks = None
        if shm is not None:
            shm.close()
            shm.unlink()
    return dropped

def _zarr_writer(
    z_out, ch_idxs, shape, shm_name, blocks, work_q, free_qs, status_q
):
    """
    Write the blocks received by ``capture_to_zarr`` to z_out. Runs in a
    separate process or thread.

    Receives (m, slot, k, rows) items on work_q (see ``_receive_blocks``).
    Once every module has delivered block k, the blocks are sorted into one
    shard, written, and their slots returned on free_qs. None on work_q ends
    the writer.

    Parameters:
    z_out (zarr.Array): output array of shape (2, ntones, nframes).
    ch_idxs (list of np.array int): tone indices in z_out of each module's
        channels, by module position.
    shape (tuple): shape of the block buffers (nmodules, nslots,
        block_length, 2 * max_ntones).
    shm_name (str or None): shared memory holding the block buffers, if
        running in a process.
    blocks (np.array int32 or None): the block buffers, if running in a
        thread.
    work_q (queue): full blocks, from the receiver.
    free_qs (list of queue): slots returned to the receiver, by module
        position.
    status_q (queue): receives 'ready' once started, then 'done' when
        finished, or a traceback if an error occurs.

    Returns:
    None
    """
    shm = None
    try:
        if blocks is None:
            shm = shared_memory.SharedMemory(name = shm_name)
            blocks = np.ndarray(shape, dtype = np.int32, buffer = shm.buf)
        nmodules, _, block_length, _ = shape
        # Tones that are not in ch_map stay 0
        shard = np.zeros((2, z_out.shape[1], block_length), dtype = np.int32)
        pending = {}
        status_q.put('ready')
        while (item := work_q.get()) is not None:
            m, slot, k, rows = item
            pending.setdefault(k, {})[m] = (slot, rows)
            if len(pending[k]) < nmodules:
                continue
            parts = pending.pop(k)
            N = min(rows for _, rows in parts.values())
            for m, (slot, _) in parts.items():
                n = len(ch_idxs[m])
                data = blocks[m, slot, :N, :2 * n]
                shard[0, ch_idxs[m], :N] = data[:, 0::2].T
                shard[1, ch_idxs[m], :N] = data[:, 1::2].T
                free_qs[m].put(slot)
            data = None
            t0 = k * block_length
            z_out[:, :, t0:t0 + N] = shard[:, :, :N]
        status_q.put('done')
    except BaseException:
        status_q.put(traceback.format_exc())
    finally:
        blocks = None
        if shm is not None:
            shm.close()

@contextlib.contextmanager
def _spawn_without_main():
    """
    Stop processes started with the 'spawn' method from importing the
    parent's __main__ module. Otherwise, a script without an
    ``if __name__ == '__main__'`` guard would run again in the writer
    process (connecting to the CRS, setting the decimation, ...). The writer
    only needs citkid.

    Returns:
    None
    """
    original = mp_spawn.get_preparation_data

    def get_preparation_data(name):
        data = original(name)
        data.pop('init_main_from_name', None)
        data.pop('init_main_from_path', None)
        return data

    mp_spawn.get_preparation_data = get_preparation_data
    try:
        yield
    finally:
        mp_spawn.get_preparation_data = original

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
        
    print(f'Raw streamed data size: {s} {unit}')
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
