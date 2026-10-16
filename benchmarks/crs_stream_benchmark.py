#!/usr/bin/env python3
"""
Standalone CRS streaming benchmark. Only the memory subcommand imports
citkid, so nothing in citkid limits which modules or decimation stages can be
tested.

Subcommands
-----------
disk
    Writes timestream data in the citkid zarr layout (int32, shape
    (2, ntones, nsamples), one shard per time block) to a directory as fast
    as possible. No CRS is needed. Use a large --gb to find out whether the
    drive slows down once its write cache is full.

stream
    Receives readout packets with the rfmux C++ receiver and a lean Python
    loop, optionally passing full blocks to a zarr writer thread or process.
    Reports where packets were lost:
        nic rx_missed/fifo/over   NIC ring buffer overflow (hardware/driver)
        softnet dropped           kernel backlog overflow (netdev_max_backlog)
        udp RcvbufErrors          socket buffer overflow (receive loop too
                                  slow or paused)
        rfmux queue drops         rfmux C++ packet queue overflow
        writer stall drops        no free block buffer (writer too slow)
        sequence gaps             packets missing for any of the above
                                  reasons, or lost before reaching this
                                  computer

parser
    Runs the rfmux parser the way citkid's 'parser' method does (plus
    --drop-stats) and reports its drops and the counters above.

memory
    Runs citkid's capture_to_memory (the 'memory' method) and reports its
    drops and the counters above. The only subcommand that imports citkid.

    By default the stream, parser and memory subcommands only listen. It changes the board's
    decimation only with --set-decimation, which affects every session
    streaming from the board. The NIC and kernel counters are system-wide,
    so stop other streaming scripts first.

Examples
--------
    python crs_stream_benchmark.py disk /media/elijah/LF2 --gb 20
    python crs_stream_benchmark.py stream -i enp5s0 -s 27 -m 1,2 --dec 0 -t 60
    python crs_stream_benchmark.py stream -i enp5s0 -s 27 -m 1,2 --dec 0 \\
        -t 60 --set-decimation --writer process --out /media/elijah/LF2
"""
import argparse
import datetime
import gc
import os
import queue
import shutil
import socket
import sys
import threading
import time
import traceback

import numpy as np

MiB = 1024 ** 2


################################################################################
################################### shared #####################################
################################################################################
def sample_freq(dec_stage):
    """Return the sample frequency in Hz for a decimation stage."""
    return 625e6 / (256 * 64 * 2 ** dec_stage)


def chunk_length(ntones, chunk_mb, min_samples = 16384):
    """Return the time samples per zarr shard, as in citkid's
    get_stream_chunk_length (without its memory cap): the longer of
    min_samples and the length that makes a chunk_mb shard."""
    return max(int(chunk_mb * MiB) // (2 * ntones * 4), int(min_samples))


def create_z(path, ntones, nsamples, chunk_N, compressor):
    """Create the citkid timestream array 'z' in a new zarr group."""
    import zarr
    kwargs = {} if compressor == 'zstd' else {'compressors': None}
    grp = zarr.open_group(path, mode = 'w')
    return grp.create_array(
        name = 'z', shape = (2, ntones, nsamples), chunks = (1, 1, chunk_N),
        shards = (2, ntones, chunk_N), dtype = np.int32, **kwargs
        )


def timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def needed_rates():
    """Return the data rates (MiB/s) of 1 and 2 modules x 128 channels at
    dec 0. Every rung of the 128-channel ladder (2 x dec 1, 4 x dec 2, ...)
    matches the 1-module rate."""
    one = sample_freq(0) * 128 * 8 / MiB
    return one, 2 * one


def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


################################################################################
#################################### disk ######################################
################################################################################
def cmd_disk(args):
    ntones = args.ntones
    chunk_N = chunk_length(ntones, args.chunk_mb, args.min_chunk_samples)
    block_bytes = 2 * ntones * chunk_N * 4
    nblocks = max(1, int(np.ceil(args.gb * 1e9 / block_bytes)))
    total_bytes = nblocks * block_bytes
    free = shutil.disk_usage(args.directory).free
    if total_bytes > 0.9 * free:
        sys.exit(f'{total_bytes / 1e9:.1f} GB requested, but only '
                 f'{free / 1e9:.1f} GB is free in {args.directory}')
    ext = '.zarr' if args.format == 'zarr' else '.raw'
    path = os.path.join(args.directory, f'disk_bench_{timestamp()}{ext}')
    one, two = needed_rates()

    print(f'=== disk benchmark: {path}')
    print(f'format {args.format}'
          + (f' ({args.compressor})' if args.format == 'zarr' else '')
          + f', {ntones} tones, {block_bytes / MiB:.0f} MiB blocks, '
          f'{nblocks} blocks = {total_bytes / 1e9:.1f} GB')
    print(f'needed: {one:.1f} MiB/s (1 module x dec 0 x 128 ch), '
          f'{two:.1f} MiB/s (2 modules x dec 0)')
    print('Early windows can read high because Linux buffers writes in RAM. '
          'Later windows and the total (which includes a final sync) are '
          'the real disk speed.\n')

    # A few distinct noise blocks, so compression and caching are realistic
    rng = np.random.default_rng(0)
    blocks = [(1_000_000 + rng.normal(0, 3e4, (2, ntones, chunk_N)))
              .astype(np.int32) for _ in range(3)]

    report_every = max(1, int(args.report_gb * 1e9 // block_bytes))
    f = z = None
    try:
        if args.format == 'zarr':
            z = create_z(path, ntones, nblocks * chunk_N, chunk_N,
                         args.compressor)
        else:
            f = open(path, 'wb')
        t_start = time.perf_counter()
        c_start = time.process_time()
        t_win, k_win, slowest = t_start, 0, 0.
        for k in range(nblocks):
            t = time.perf_counter()
            blk = blocks[k % len(blocks)]
            if z is not None:
                z[:, :, k * chunk_N:(k + 1) * chunk_N] = blk
            else:
                f.write(blk.data)
            slowest = max(slowest, time.perf_counter() - t)
            if (k + 1) % report_every == 0 or k + 1 == nblocks:
                now = time.perf_counter()
                win_rate = (k + 1 - k_win) * block_bytes / MiB / (now - t_win)
                cum_rate = (k + 1) * block_bytes / MiB / (now - t_start)
                print(f'{(k + 1) * block_bytes / 1e9:8.1f} GB  '
                      f'window {win_rate:7.1f} MiB/s  '
                      f'cumulative {cum_rate:7.1f} MiB/s  '
                      f'slowest block {slowest:6.2f} s '
                      f'({block_bytes / MiB / slowest:.0f} MiB/s)')
                t_win, k_win, slowest = now, k + 1, 0.
        t_sync = time.perf_counter()
        if f is not None:
            f.flush()
            os.fsync(f.fileno())
        os.sync()
        t_end = time.perf_counter()
        cpu = time.process_time() - c_start

        rate = total_bytes / MiB / (t_end - t_start)
        print(f'\ntotal {total_bytes / 1e9:.1f} GB in {t_end - t_start:.1f} s '
              f'(final sync {t_end - t_sync:.1f} s): {rate:.1f} MiB/s')
        print(f'CPU use: {cpu / (t_end - t_start):.2f} cores')
        print(f'margin: {rate / one:.1f}x for 1 module x dec 0, '
              f'{rate / two:.1f}x for 2 modules x dec 0')
    finally:
        if f is not None:
            f.close()
        if not args.keep:
            remove(path)


################################################################################
#################################### stream ####################################
################################################################################
COUNTER_NOTES = {
    'nic rx_missed_errors': 'NIC ring overflow (hardware/driver)',
    'nic rx_fifo_errors': 'NIC FIFO overflow (hardware/driver)',
    'nic rx_over_errors': 'NIC overrun (hardware/driver)',
    'nic rx_dropped': 'dropped by driver/stack (can include unrelated '
                      'frames, e.g. LLDP)',
    'softnet dropped': 'kernel backlog full (net.core.netdev_max_backlog)',
    'softnet time_squeeze': 'kernel ran out of budget emptying the NIC ring',
    'udp RcvbufErrors': 'socket buffer full (receive loop too slow/paused)',
    'udp InErrors': 'UDP receive errors (includes RcvbufErrors)',
}


def read_counters(interface):
    """Return system-wide NIC and kernel receive counters."""
    c = {}
    stats_dir = f'/sys/class/net/{interface}/statistics'
    for name in ('rx_packets', 'rx_errors', 'rx_dropped', 'rx_missed_errors',
                 'rx_fifo_errors', 'rx_over_errors'):
        with open(os.path.join(stats_dir, name)) as f:
            c['nic ' + name] = int(f.read())
    with open('/proc/net/softnet_stat') as f:
        rows = [line.split() for line in f]
    c['softnet dropped'] = sum(int(r[1], 16) for r in rows)
    c['softnet time_squeeze'] = sum(int(r[2], 16) for r in rows)
    with open('/proc/net/snmp') as f:
        udp = [line.split()[1:] for line in f if line.startswith('Udp:')]
    udp = dict(zip(udp[0], map(int, udp[1])))
    c['udp RcvbufErrors'] = udp['RcvbufErrors']
    c['udp InErrors'] = udp['InErrors']
    return c



def verdict(deltas, interface, queue_drops, stall_drops, gaps):
    """Return messages explaining where packets were lost."""
    messages = []
    if sum(deltas[k] for k in ('nic rx_missed_errors', 'nic rx_fifo_errors',
                               'nic rx_over_errors')):
        messages.append(
            'NIC overflow: hardware/driver limit, which faster code will not '
            f'fix. Try `sudo ethtool -G {interface} rx 4096` (check '
            '`ethtool -g` for the maximum) or a better NIC.')
    if deltas['softnet dropped']:
        messages.append('kernel backlog overflow: raise '
                        'net.core.netdev_max_backlog (e.g. 10000).')
    if deltas['udp RcvbufErrors']:
        messages.append(
            'socket buffer overflow: the receive loop could not keep up or '
            'paused. A faster (compiled) receive loop or a larger socket '
            'buffer would help. Check that no other UDP receiver was running.')
    if queue_drops:
        messages.append('rfmux queue overflow: increase --queue-size.')
    if stall_drops:
        messages.append('writer stalls: the writer could not keep up. Try '
                        '--writer process, more --slots, or --compressor none.')
    if gaps and not messages:
        messages.append(
            'sequence gaps with no local drops counted: packets were probably '
            'lost before reaching this computer (CRS, switch, cable).')
    if not messages and not gaps:
        messages.append('no packets lost.')
    return messages


def save_json(path, result):
    import json
    with open(path, 'w') as f:
        json.dump(result, f, indent = 1)



def parse_modules(spec):
    """Parse a comma-separated module list, e.g. '1,2'."""
    modules = [int(x) for x in spec.split(',')]
    if any(mi not in range(1, 9) for mi in modules) or \
       len(set((mi - 1) % 4 for mi in modules)) != len(modules):
        sys.exit('--modules must be distinct modules in 1-8, with at most one '
                 'per position across the analog banks')
    return modules


def maybe_set_decimation(args, modules):
    """Set the board decimation if --set-decimation was passed."""
    if not args.set_decimation:
        return
    short = not args.long
    if short and args.channels > 128:
        sys.exit('--channels > 128 needs --long')
    configure_crs(args.serial, args.dec, short, modules, args.host)


def configure_crs(serial, dec, short, modules, host = None):
    """Set the board decimation with rfmux. host (a hostname or IP) skips the
    mDNS lookup of rfmux<serial>.local."""
    import asyncio
    import rfmux

    async def run():
        entry = f'serial: "{serial:04d}"'
        if host:
            entry += f', hostname: "{host}"'
        s = rfmux.load_session(f'!HardwareMap [ !CRS {{ {entry} }} ]')
        d = s.query(rfmux.CRS).one()
        await d.resolve()
        await d.set_decimation(dec, short = short, module = modules)
        return await d.get_decimation()

    stage = asyncio.run(run())
    print(f'Decimation set: stage {stage}, short = {short}, '
          f'modules = {modules}')
    time.sleep(0.5)


def writer_main(cfg, buffers, work_q, free_qs, result_q):
    """
    Assemble per-module blocks into (2, ntones, N) shards and write them.

    Receives (module_pos, slot, k, rows) items on work_q, where module_pos is
    the module's position in the module list and k is the module's block
    number. Once every module has delivered block k, the blocks are
    transposed into one shard, written to zarr (if cfg['path']), and their
    slots returned on free_qs. None on work_q ends the writer.

    In process mode buffers is None and the shared memory is attached here.
    """
    shm = None
    try:
        nmod, nslots = cfg['nmod'], cfg['nslots']
        chunk_N, nt = cfg['chunk_N'], cfg['nt']
        if buffers is None:
            # Spawned children share the parent's resource tracker, so
            # attaching here does not unlink the segment when this exits
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name = cfg['shm_name'])
            buffers = np.ndarray((nmod, nslots, chunk_N, nt, 2), np.int32,
                                 buffer = shm.buf)
        z = None
        if cfg['path']:
            z = create_z(cfg['path'], nmod * nt, cfg['nframes'], chunk_N,
                         cfg['compressor'])
        shard = np.empty((2, nmod * nt, chunk_N), np.int32)
        pending = {}
        stats = dict(blocks = 0, bytes = 0, busy_s = 0., max_block_s = 0.,
                     max_pending = 0, incomplete = 0)
        result_q.put('ready')

        while (item := work_q.get()) is not None:
            m, slot, k, rows = item
            pending.setdefault(k, {})[m] = (slot, rows)
            stats['max_pending'] = max(stats['max_pending'], len(pending))
            if len(pending[k]) < nmod:
                continue
            t = time.perf_counter()
            parts = pending.pop(k)
            N = min(rows for _, rows in parts.values())
            for m, (slot, _) in parts.items():
                blk = buffers[m, slot, :N]
                shard[0, m * nt:(m + 1) * nt, :N] = blk[:, :, 0].T
                shard[1, m * nt:(m + 1) * nt, :N] = blk[:, :, 1].T
                free_qs[m].put(slot)
            blk = None
            if z is not None:
                z[:, :, k * chunk_N:k * chunk_N + N] = shard[:, :, :N]
            dt = time.perf_counter() - t
            stats['busy_s'] += dt
            stats['max_block_s'] = max(stats['max_block_s'], dt)
            stats['blocks'] += 1
            stats['bytes'] += 2 * nmod * nt * N * 4
        stats['incomplete'] = len(pending)
        result_q.put(stats)
    except Exception:
        result_q.put({'error': traceback.format_exc()})
    finally:
        buffers = None
        if shm is not None:
            try:
                shm.close()
            except BufferError:
                pass


def cmd_stream(args):
    from rfmux import streamer
    from rfmux.tools.parser import resolve_interface

    ### Inputs
    modules = parse_modules(args.modules)
    nmod, nt = len(modules), args.channels
    fs = sample_freq(args.dec)
    nframes = int(fs * args.duration)
    ntotal = nmod * nt
    chunk_N = min(chunk_length(ntotal, args.chunk_mb,
                               args.min_chunk_samples), nframes)
    nslots = 1 if args.writer == 'none' else args.slots
    rate = fs * ntotal * 8 / MiB
    path = None
    if args.out is not None:
        if args.writer == 'none':
            sys.exit('--out needs --writer thread or process')
        need = nframes * ntotal * 8
        free = shutil.disk_usage(args.out).free
        if need > 0.9 * free:
            sys.exit(f'{need / 1e9:.1f} GB needed, but only '
                     f'{free / 1e9:.1f} GB is free in {args.out}')
        path = os.path.join(args.out, f'stream_bench_{timestamp()}.zarr')

    print(f'=== stream benchmark: modules {modules}, dec {args.dec} '
          f'({fs:,.0f} Hz), {nt} channels, {args.duration:g} s')
    print(f'expected {fs * nmod:,.0f} packets/s, {rate:.1f} MiB/s, '
          f'{nframes * ntotal * 8 / 1e9:.2f} GB total')
    print(f'writer {args.writer}'
          + (f', zarr ({args.compressor}) to {path}' if path else
             ', shards assembled but not saved' if args.writer != 'none'
             else '')
          + f', {chunk_N * ntotal * 8 / MiB:.0f} MiB shards '
          f'({chunk_N / fs:.2f} s), {nslots} slots/module')

    maybe_set_decimation(args, modules)

    ### Block buffers and writer
    shape = (nmod, nslots, chunk_N, nt, 2)
    shm = writer = None
    if args.writer == 'process':
        import multiprocessing as mp
        from multiprocessing import shared_memory
        ctx = mp.get_context('spawn')
        shm = shared_memory.SharedMemory(
            create = True, size = int(np.prod(shape)) * 4)
        buffers = np.ndarray(shape, np.int32, buffer = shm.buf)
        work_q, result_q = ctx.Queue(), ctx.Queue()
        free_qs = [ctx.Queue() for _ in modules]
        writer_buffers = None
    else:
        buffers = np.empty(shape, np.int32)
        work_q, result_q = queue.Queue(), queue.Queue()
        free_qs = [queue.Queue() for _ in modules]
        writer_buffers = buffers
    buffers.fill(0)  # touch pages now, not in the receive loop
    # Per-module, per-slot flat views (chunk_N, 2 * nt) for the receive loop
    views = [[buffers[m, s].reshape(chunk_N, 2 * nt) for s in range(nslots)]
             for m in range(nmod)]

    try:
        if args.writer != 'none':
            cfg = dict(nmod = nmod, nslots = nslots, chunk_N = chunk_N,
                       nt = nt, nframes = nframes, path = path,
                       compressor = args.compressor,
                       shm_name = shm.name if shm else None)
            wargs = (cfg, writer_buffers, work_q, free_qs, result_q)
            if args.writer == 'process':
                writer = ctx.Process(target = writer_main, args = wargs,
                                     daemon = True)
            else:
                writer = threading.Thread(target = writer_main, args = wargs,
                                          daemon = True)
            writer.start()
            msg = result_q.get(timeout = 120)
            if msg != 'ready':
                sys.exit(f'writer failed to start:\n{msg["error"]}')

        ### Receive
        local_free = [list(range(nslots)) for _ in modules]
        slot = [None] * nmod
        rows, counts, block_num = [0] * nmod, [0] * nmod, [0] * nmod
        gaps, stall_drops = [0] * nmod, [0] * nmod
        last_seq = [None] * nmod
        first_t, last_t = [None] * nmod, [None] * nmod
        queues = {}
        pmap = {(mi - 1) % 4: m for m, mi in enumerate(modules)}
        nt2 = 2 * nt

        def acquire(m):
            if local_free[m]:
                return local_free[m].pop()
            try:
                return free_qs[m].get_nowait()
            except queue.Empty:
                return None

        def send(m):
            if args.writer == 'none':
                local_free[m].append(slot[m])
            else:
                work_q.put((m, slot[m], block_num[m], rows[m]))
            slot[m] = None
            rows[m] = 0
            block_num[m] += 1

        wait_s = proc_s = max_iter = 0.
        slow_iters = max_depth = 0
        stop_reason = 'complete'
        interface_ip = resolve_interface(args.interface)
        with streamer.get_multicast_socket(
            crs_hostname = None, port = streamer.STREAMER_PORT,
            interface = interface_ip, buffer_size = args.rcvbuf,
        ) as sock:
            rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) // 2
            receiver = streamer.ReadoutPacketReceiver(
                sock, reorder_window = args.reorder_window,
                queue_max_size = args.queue_size)
            if args.gc_off:
                gc.disable()
            counters0 = read_counters(args.interface)
            t_start = time.perf_counter()
            c_start = time.thread_time()
            deadline = t_start + 1.5 * args.duration + 10
            last_packet = [t_start] * nmod
            try:
                while min(counts) < nframes:
                    t0 = time.perf_counter()
                    receiver.receive_batch(args.batch, timeout_ms = 1000)
                    t1 = time.perf_counter()
                    wait_s += t1 - t0
                    for serial, module, q in receiver.get_all_queues():
                        m = pmap.get(module) if serial == args.serial else None
                        if m is None or counts[m] >= nframes:
                            q.clear()
                            continue
                        queues[m] = q
                        depth = q.size()
                        if depth:
                            max_depth = max(max_depth, depth)
                            last_packet[m] = last_t[m] = t1
                            if first_t[m] is None:
                                first_t[m] = t1
                        while (packet := q.try_pop()):
                            pkt = packet.to_python()
                            seq = pkt.seq
                            if last_seq[m] is not None:
                                gaps[m] += (seq - last_seq[m] - 1) & 0xFFFFFFFF
                            last_seq[m] = seq
                            if slot[m] is None:
                                slot[m] = acquire(m)
                                if slot[m] is None:
                                    stall_drops[m] += 1
                                    continue
                            raw = pkt.raw_samples
                            if raw.shape[0] < nt2:
                                raise RuntimeError(
                                    f'module {modules[m]} packets have '
                                    f'{raw.shape[0] // 2} channels, but '
                                    f'--channels is {nt}')
                            views[m][slot[m]][rows[m]] = raw[:nt2]
                            rows[m] += 1
                            counts[m] += 1
                            if rows[m] == chunk_N or counts[m] == nframes:
                                send(m)
                            if counts[m] >= nframes:
                                break
                    t2 = time.perf_counter()
                    dt = t2 - t1
                    proc_s += dt
                    max_iter = max(max_iter, dt)
                    slow_iters += dt > 0.01
                    stalled = [modules[m] for m in range(nmod)
                               if counts[m] < nframes and
                               t2 - last_packet[m] > args.stall_timeout]
                    if stalled:
                        stop_reason = (f'no packets from modules {stalled} '
                                       f'for {args.stall_timeout:g} s')
                        break
                    if t2 > deadline:
                        stop_reason = 'deadline (1.5 x duration + 10 s) reached'
                        break
            except KeyboardInterrupt:
                stop_reason = 'interrupted'
            t_end = time.perf_counter()
            cpu = time.thread_time() - c_start
            counters1 = read_counters(args.interface)
            if args.gc_off:
                gc.enable()
            queue_drops = {m: q.get_stats().packets_dropped
                           for m, q in queues.items()}
            rstats = receiver.get_stats()

        ### Flush partial blocks and stop the writer
        for m in range(nmod):
            if slot[m] is not None and rows[m]:
                send(m)
        wstats = None
        if writer is not None:
            work_q.put(None)
            print('waiting for writer to finish...')
            wstats = result_q.get()
            writer.join()

        ### Report
        wall = t_end - t_start
        print(f'\nstopped: {stop_reason}, {wall:.1f} s')
        print(f'socket buffer {rcvbuf / MiB:.0f} MiB, rfmux receiver: '
              f'{rstats.total_packets_received:,} packets, '
              f'{rstats.invalid_packets} invalid, '
              f'{rstats.wrong_magic} wrong magic')
        print(f'\n{"module":>6} {"saved":>11} {"rate_Hz":>9} {"seq_gaps":>9} '
              f'{"rfmux_q":>8} {"stalls":>7}')
        for m, mi in enumerate(modules):
            span = (last_t[m] - first_t[m]) if first_t[m] is not None else 0
            r = counts[m] / span if span > 0 else 0
            print(f'{mi:>6} {counts[m]:>11,} {r:>9,.0f} {gaps[m]:>9,} '
                  f'{queue_drops.get(m, 0):>8,} {stall_drops[m]:>7,}')

        print('\nNIC/kernel counters (system-wide, change during run):')
        deltas = {k: counters1[k] - counters0[k] for k in counters0}
        for k, v in deltas.items():
            note = COUNTER_NOTES.get(k, '')
            print(f'  {k:<22} {v:>11,}  {note}')

        print(f'\nreceive loop: busy {100 * proc_s / wall:.0f}% of wall '
              f'time, waiting {100 * wait_s / wall:.0f}%, main-thread CPU '
              f'{100 * cpu / wall:.0f}%')
        print(f'  slowest iteration {1e3 * max_iter:.1f} ms, '
              f'{slow_iters} iterations > 10 ms, max rfmux queue depth '
              f'{max_depth} (limit {args.queue_size})')
        if wstats is not None:
            if 'error' in wstats:
                print(f'\nwriter failed:\n{wstats["error"]}')
            else:
                busy = wstats['busy_s']
                wrate = wstats['bytes'] / MiB / busy if busy else 0
                print(f'writer: {wstats["blocks"]} shards, '
                      f'{wstats["bytes"] / 1e9:.2f} GB, busy '
                      f'{100 * busy / wall:.0f}% of wall time, '
                      f'{wrate:.0f} MiB/s while busy '
                      f'({wrate / rate if rate else 0:.1f}x the stream rate)')
                print(f'  slowest shard {wstats["max_block_s"]:.2f} s, '
                      f'max shards waiting for all modules '
                      f'{wstats["max_pending"]}, incomplete at end '
                      f'{wstats["incomplete"]}')

        ### Interpretation
        lost = sum(gaps) + sum(stall_drops)
        complete = all(c >= nframes for c in counts)
        messages = verdict(deltas, args.interface, sum(queue_drops.values()),
                           sum(stall_drops), sum(gaps))
        print('\nverdict:')
        for msg in messages:
            print(f'  - {msg}')
        if args.json:
            save_json(args.json, dict(
                method = 'zarr' if path else f'lean/{args.writer}',
                modules = modules, dec = args.dec, channels = nt,
                duration = args.duration, rate_mib_s = rate,
                packets_per_s = fs * nmod, ok = complete and not lost,
                stop_reason = stop_reason, wall_s = wall,
                saved = counts, expected = nframes, seq_gaps = gaps,
                rfmux_queue_drops = [queue_drops.get(m, 0)
                                     for m in range(nmod)],
                stall_drops = stall_drops, counters = deltas,
                loop_busy = proc_s / wall, loop_cpu = cpu / wall,
                max_iter_ms = 1e3 * max_iter, slow_iters = slow_iters,
                max_queue_depth = max_depth, writer = wstats,
                verdict = messages))
    finally:
        views = buffers = None
        if shm is not None:
            try:
                shm.close()
            except BufferError:
                pass
            shm.unlink()
        if path and not args.keep:
            remove(path)


def cmd_parser(args):
    """
    Run the rfmux parser the way citkid's 'parser' method does (with
    --drop-stats added), and report its drops.
    """
    import contextlib
    import re
    import tempfile
    from rfmux.tools import parser

    modules = parse_modules(args.modules)
    nmod, nt = len(modules), args.channels
    fs = sample_freq(args.dec)
    # citkid adds 0.1 s, because the parser stops when any module is done
    nframes = int(fs * (args.duration + 0.1))
    rate = fs * nmod * nt * 8 / MiB
    need = nframes * nmod * nt * 8
    free = shutil.disk_usage(args.out).free
    if need > 0.9 * free:
        sys.exit(f'{need / 1e9:.1f} GB needed, but only {free / 1e9:.1f} GB '
                 f'is free in {args.out}')
    path = os.path.join(args.out, f'parser_bench_{timestamp()}')
    print(f'=== parser benchmark: modules {modules}, dec {args.dec} '
          f'({fs:,.0f} Hz), {nt} channels, {args.duration:g} s')
    print(f'expected {fs * nmod:,.0f} packets/s, {rate:.1f} MiB/s, dirfile '
          f'{path}')
    maybe_set_decimation(args, modules)

    pargs = ['-i', args.interface, '-d', path, '-c', f'1-{nt}',
             '-s', f'{args.serial:04d}', '-n', f'{nframes:d}', '--drop-stats']
    try:
        with tempfile.TemporaryFile('w+') as out, \
             tempfile.TemporaryFile('w+') as err:
            counters0 = read_counters(args.interface)
            t_start = time.perf_counter()
            try:
                with contextlib.redirect_stdout(out), \
                     contextlib.redirect_stderr(err):
                    parser.main(*pargs)
            except SystemExit:
                pass  # the parser exits when it is done
            except KeyboardInterrupt:
                print('interrupted')
            wall = time.perf_counter() - t_start
            counters1 = read_counters(args.interface)
            out.seek(0)
            err.seek(0)
            drop_lines = sum(line.startswith('Dropped packet') for line in out)
            summary = err.read()
        size = sum(os.path.getsize(os.path.join(root, f))
                   for root, _, files in os.walk(path) for f in files) \
            if os.path.isdir(path) else 0
    finally:
        if not args.keep:
            remove(path)

    # Summary lines are 'Serial S module M: N packets seen (D dropped)', with
    # M numbered 1-4 within the analog bank
    seen = {int(m): (int(n), int(d)) for m, n, d in re.findall(
        r'module (\d+): (\d+) packets seen \((\d+) dropped\)', summary)}
    deltas = {k: counters1[k] - counters0[k] for k in counters0}
    print(f'\n{wall:.1f} s, dirfile {size / 1e9:.2f} GB')
    print(f'{"module":>6} {"seen":>11} {"dropped":>9}')
    saved, gaps = [], []
    for mi in modules:
        n, d = seen.get((mi - 1) % 4 + 1, (0, 0))
        saved.append(n)
        gaps.append(d)
        print(f'{mi:>6} {n:>11,} {d:>9,}')
    print(f'({drop_lines} drop events)')
    print('\nNIC/kernel counters (system-wide, change during run):')
    for k, v in deltas.items():
        print(f'  {k:<22} {v:>11,}  {COUNTER_NOTES.get(k, "")}')
    messages = verdict(deltas, args.interface, 0, 0, sum(gaps))
    print('\nverdict:')
    for msg in messages:
        print(f'  - {msg}')
    if not seen:
        print(f'parser output:\n{summary}')
    if args.json:
        # The parser stops when its fastest module reaches nframes, so slower
        # modules can end slightly short without having dropped packets
        save_json(args.json, dict(
            method = 'parser', modules = modules, dec = args.dec,
            channels = nt, duration = args.duration, rate_mib_s = rate,
            packets_per_s = fs * nmod,
            ok = bool(seen) and not sum(gaps) and
                 all(n >= 0.99 * nframes for n in saved),
            wall_s = wall, saved = saved, expected = nframes,
            seq_gaps = gaps, counters = deltas, dirfile_bytes = size,
            verdict = messages))


def cmd_memory(args):
    """Run citkid's capture_to_memory (the 'memory' method) and report
    drops. Imports citkid."""
    from citkid.crs.util import capture_to_memory

    modules = parse_modules(args.modules)
    nmod, nt = len(modules), args.channels
    fs = sample_freq(args.dec)
    nframes = int(fs * args.duration)
    rate = fs * nmod * nt * 8 / MiB
    need = nframes * nmod * nt * 8
    print(f'=== memory benchmark: modules {modules}, dec {args.dec} '
          f'({fs:,.0f} Hz), {nt} channels, {args.duration:g} s, '
          f'{need / 1e9:.2f} GB in RAM')
    maybe_set_decimation(args, modules)

    error = None
    counters0 = read_counters(args.interface)
    t_start = time.perf_counter()
    try:
        buffers, dropped = capture_to_memory(
            args.interface, args.serial, modules, nt, nframes)
        del buffers
    except (RuntimeError, KeyboardInterrupt) as e:
        error = repr(e)
        dropped = {}
    wall = time.perf_counter() - t_start
    counters1 = read_counters(args.interface)
    deltas = {k: counters1[k] - counters0[k] for k in counters0}
    gaps = [dropped.get(mi, 0) for mi in modules]

    print(f'\n{wall:.1f} s' + (f', error: {error}' if error else ''))
    for mi, d in zip(modules, gaps):
        print(f'  module {mi}: {d:,} dropped')
    print('\nNIC/kernel counters (system-wide, change during run):')
    for k, v in deltas.items():
        print(f'  {k:<22} {v:>11,}  {COUNTER_NOTES.get(k, "")}')
    messages = verdict(deltas, args.interface, 0, 0, sum(gaps))
    if error:
        messages.insert(0, f'capture failed: {error}')
    print('\nverdict:')
    for msg in messages:
        print(f'  - {msg}')
    if args.json:
        save_json(args.json, dict(
            method = 'memory', modules = modules, dec = args.dec,
            channels = nt, duration = args.duration, rate_mib_s = rate,
            packets_per_s = fs * nmod, ok = error is None and not sum(gaps),
            wall_s = wall, expected = nframes, seq_gaps = gaps,
            counters = deltas, error = error, verdict = messages))


################################################################################
##################################### main #####################################
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest = 'cmd', required = True)

    d = sub.add_parser('disk', help = 'zarr/raw write throughput, no CRS')
    d.add_argument('directory', help = 'directory to write the test data in')
    d.add_argument('--gb', type = float, default = 20,
                   help = 'amount of data to write, in GB (default 20)')
    d.add_argument('--ntones', type = int, default = 128,
                   help = 'tones in the array (default 128)')
    d.add_argument('--min-chunk-samples', type = int, default = 16384,
                   help = 'minimum time samples per chunk (default 16384, '
                          'citkid MIN_CHUNK_SAMPLES)')
    d.add_argument('--chunk-mb', type = float, default = 128,
                   help = 'minimum shard size in MiB (default 128, citkid '
                          'MIN_SHARD_BYTES)')
    d.add_argument('--format', choices = ('zarr', 'raw'), default = 'zarr',
                   help = 'zarr (citkid layout) or one raw file, like a '
                          'dirfile field (default zarr)')
    d.add_argument('--compressor', choices = ('zstd', 'none'),
                   default = 'zstd', help = 'zarr compressor (default zstd, '
                                            'the zarr default)')
    d.add_argument('--report-gb', type = float, default = 1,
                   help = 'print progress every this many GB (default 1)')
    d.add_argument('--keep', action = 'store_true',
                   help = 'keep the test data (deleted by default)')

    def crs_args(sp, out_help = None):
        sp.add_argument('-i', '--interface', required = True,
                        help = 'network interface, e.g. enp5s0')
        sp.add_argument('-s', '--serial', type = int, required = True,
                        help = 'CRS serial number')
        sp.add_argument('-m', '--modules', required = True,
                        help = 'comma-separated modules (1-8), e.g. 1,2')
        sp.add_argument('--dec', type = int, required = True,
                        choices = range(7),
                        help = 'decimation stage the board is (or will be) at')
        sp.add_argument('-t', '--duration', type = float, default = 30,
                        help = 'capture length in s (default 30)')
        sp.add_argument('-c', '--channels', type = int, default = 128,
                        help = 'channels kept per module (default 128)')
        sp.add_argument('--set-decimation', action = 'store_true',
                        help = 'set the board to --dec for --modules first. '
                               'Affects every session using the board.')
        sp.add_argument('--long', action = 'store_true',
                        help = 'with --set-decimation, use long '
                               '(1024-channel) packets')
        sp.add_argument('--host', default = None,
                        help = 'CRS hostname or IP for --set-decimation '
                               '(default rfmux<serial>.local, via mDNS)')
        sp.add_argument('--json', default = None,
                        help = 'also save the results to this JSON file')
        if out_help:
            sp.add_argument('--out', required = True, help = out_help)
            sp.add_argument('--keep', action = 'store_true',
                            help = 'keep the data (deleted by default)')

    p = sub.add_parser('parser', help = "rfmux parser, as citkid's 'parser' "
                                        "method runs it")
    crs_args(p, out_help = 'directory to write the dirfile in (citkid\'s '
                           'tmp_directory)')

    mem = sub.add_parser('memory', help = "citkid's capture_to_memory (the "
                                          "'memory' method)")
    crs_args(mem)

    s = sub.add_parser('stream', help = 'lean receive loop, optionally with '
                                        'a zarr writer')
    crs_args(s)
    s.add_argument('--writer', choices = ('none', 'thread', 'process'),
                   default = 'none',
                   help = 'none: receive only. thread/process: assemble '
                          'shards in a writer thread/process (default none)')
    s.add_argument('--out', default = None,
                   help = 'directory to write zarr data in. Without it, the '
                          'writer assembles shards but does not save them')
    s.add_argument('--compressor', choices = ('zstd', 'none'),
                   default = 'zstd', help = 'zarr compressor (default zstd)')
    s.add_argument('--min-chunk-samples', type = int, default = 16384,
                   help = 'minimum time samples per chunk (default 16384, '
                          'citkid MIN_CHUNK_SAMPLES)')
    s.add_argument('--chunk-mb', type = float, default = 128,
                   help = 'minimum shard size in MiB (default 128)')
    s.add_argument('--slots', type = int, default = 3,
                   help = 'block buffers per module for the writer '
                          '(default 3)')
    s.add_argument('--keep', action = 'store_true',
                   help = 'keep the zarr data (deleted by default)')
    s.add_argument('--batch', type = int, default = 256,
                   help = 'receive_batch size (default 256, as the parser)')
    s.add_argument('--reorder-window', type = int, default = 256,
                   help = 'rfmux reorder window (default 256)')
    s.add_argument('--queue-size', type = int, default = 4096,
                   help = 'rfmux per-module queue size (default 4096)')
    s.add_argument('--rcvbuf', type = int, default = 67108864,
                   help = 'requested socket buffer in bytes (default 64 MiB, '
                          'capped by net.core.rmem_max)')
    s.add_argument('--stall-timeout', type = float, default = 10,
                   help = 'stop if a module sends nothing for this long, in '
                          's (default 10)')
    s.add_argument('--gc-off', action = 'store_true',
                   help = 'disable Python garbage collection while receiving')

    args = parser.parse_args()
    {'disk': cmd_disk, 'parser': cmd_parser, 'memory': cmd_memory,
     'stream': cmd_stream}[args.cmd](args)


if __name__ == '__main__':
    main()
