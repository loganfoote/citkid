#!/usr/bin/env python3
"""
Benchmark CRS streaming over a ladder of data rates.

Configures the board with citkid (configure_system, NCOs on modules 1-4, no
tones written), then for each configuration (number of modules, decimation
stage, packet length) runs each method in its own process with
crs_stream_benchmark.py:
    lean     lean receive loop, no writer (best case for a Python receiver)
    memory   citkid's capture_to_memory ('memory' method)
    parser   rfmux parser writing a dirfile to --out ('parser' method)
    zarr     lean loop with a zarr writer process writing to --out

Takes over the whole board and leaves it at decimation stage 6 with long
packets on modules 1-4; reconfigure it afterwards. Results and logs are saved
to results/rate_sweep_<time>/.

Configurations are written as <modules>x<dec>[L][:<tones>], e.g. 2x0
(modules 1-2 at dec 0, short packets) or 4x4L (modules 1-4 at dec 4, long
packets). The network load is set by the packets (128 or 1024 channels). By
default every channel is written to disk; :<tones> writes only that many
tones per module, as when fewer tones are set, e.g. 4x5L:500.

Example:
    python crs_rate_sweep.py -s 45 -i enp5s0 --out /media/elijah/LF2
    python crs_rate_sweep.py -s 45 -i enp5s0 --out /media/elijah/LF2 \\
        --configs 1x0,2x0 --methods lean,zarr -t 60
"""
import argparse
import asyncio
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'crs_stream_benchmark.py')
MiB = 1024 ** 2

DEFAULT_CONFIGS = (
    # Short packets (128 channels)
    '1x0,2x0,3x0,4x0,2x1,3x1,4x1,4x2,'
    # Long packets (1024 channels), where the firmware allows them
    '1x3L,2x3L,1x4L,2x4L,3x4L,4x4L,4x5L'
    )
METHODS = ('lean', 'memory', 'parser', 'zarr')


def parse_config(spec):
    """Parse '<modules>x<dec>[L][:<tones>]' into (nmod, dec, long, ntones),
    where ntones is the number of tones written per module."""
    spec, _, ntones = spec.partition(':')
    long = spec.endswith('L')
    nmod, dec = spec.rstrip('L').split('x')
    ntones = int(ntones) if ntones else (1024 if long else 128)
    return int(nmod), int(dec), long, ntones


def config_label(cfg):
    """Return the '<modules>x<dec>[L][:<tones>]' label of a configuration."""
    nmod, dec, long, ntones = cfg
    label = f'{nmod}x{dec}{"L" if long else ""}'
    if ntones != (1024 if long else 128):
        label += f':{ntones}'
    return label


def rates(cfg):
    """Return (network MiB/s, packets/s, written MiB/s) of a configuration."""
    nmod, dec, long, ntones = cfg
    fs = sample_freq(dec)
    nch = 1024 if long else 128
    return fs * nmod * nch * 8 / MiB, fs * nmod, fs * nmod * ntones * 8 / MiB


def sample_freq(dec_stage):
    return 625e6 / (256 * 64 * 2 ** dec_stage)


def loss_summary(r):
    """Return a short table cell for one result."""
    if r is None:
        return '-'
    if 'error' in r and r.get('ok') is None:
        return r['error']
    if r['ok']:
        return 'ok'
    nmod = len(r['modules'])
    lost = sum(r.get('seq_gaps', [])) + sum(r.get('stall_drops', []) or [])
    frac = lost / (r['expected'] * nmod) if r.get('expected') else 0
    c = r.get('counters', {})
    codes = ''
    if c.get('nic rx_missed_errors', 0) + c.get('nic rx_fifo_errors', 0) + \
       c.get('nic rx_over_errors', 0):
        codes += 'N'
    if c.get('softnet dropped', 0):
        codes += 'K'
    if c.get('udp RcvbufErrors', 0):
        codes += 'S'
    if sum(r.get('rfmux_queue_drops', []) or []):
        codes += 'Q'
    if sum(r.get('stall_drops', []) or []):
        codes += 'W'
    if r.get('error'):
        codes += 'E'
    return f'LOST {100 * frac:.2g}% {codes}'.strip()


def run_method(method, cfg, args, run_dir):
    """Run one method for one configuration in a subprocess. Returns the
    result dict, or a dict with 'error'."""
    nmod, dec, long, ntones = cfg
    name = f'{config_label(cfg).replace(":", "-")}_{method}'
    json_path = os.path.join(run_dir, name + '.json')
    log_path = os.path.join(run_dir, name + '.log')
    sub = {'lean': 'stream', 'zarr': 'stream'}.get(method, method)
    cmd = [sys.executable, BENCH, sub, '-i', args.interface,
           '-s', str(args.serial),
           '-m', ','.join(str(m) for m in range(1, nmod + 1)),
           '--dec', str(dec), '-t', str(args.duration),
           '-c', str(ntones), '--set-decimation',
           '--json', json_path]
    if long:
        cmd.append('--long')
    if args.host:
        cmd += ['--host', args.host]
    if method == 'zarr':
        cmd += ['--writer', 'process', '--out', args.out]
    elif method == 'parser':
        cmd += ['--out', args.out]
    timeout = 3 * args.duration + 180
    with open(log_path, 'w') as log:
        try:
            proc = subprocess.run(cmd, stdout = log, stderr = subprocess.STDOUT,
                                  timeout = timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    with open(log_path) as f:
        log_text = f.read()
    if 'packets have 128 channels' in log_text:
        return {'error': 'no long packets', 'unavailable': True}
    return {'error': f'failed ({code})', 'log': log_text[-2000:]}


async def main_async(args):
    from citkid.crs.instrument import CRS

    configs = [parse_config(c) for c in args.configs.split(',')]
    methods = args.methods.split(',')
    if any(m not in METHODS for m in methods):
        sys.exit(f'--methods must be from {METHODS}')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(HERE, 'results', f'rate_sweep_{stamp}')
    os.makedirs(run_dir)
    print(f'results in {run_dir}')

    ### Configure the board
    crs = CRS(serial_number = args.serial, interface = args.interface)
    if args.host:
        # citkid always looks the board up by mDNS; connect by host instead
        import rfmux
        session = rfmux.load_session(
            f'!HardwareMap [ !CRS {{ serial: "{args.serial:04d}", '
            f'hostname: "{args.host}" }} ]')
        crs.d = session.query(rfmux.CRS).one()
    await crs.configure_system(clock_source = 'VCXO', full_scale_dbm = 7,
                               analog_bank_high = False)
    await crs.set_nco({1: 0.5e9, 2: 1.0e9, 3: 1.5e9, 4: 2.0e9})

    ### Optional disk benchmark
    if args.disk_gb:
        print(f'\ndisk benchmark, {args.disk_gb:g} GB to {args.out}')
        with open(os.path.join(run_dir, 'disk.log'), 'w') as log:
            subprocess.run([sys.executable, BENCH, 'disk', args.out,
                            '--gb', str(args.disk_gb)],
                           stdout = log, stderr = subprocess.STDOUT)

    ### Sweep
    results = {}
    try:
        for cfg in configs:
            nmod, dec, long, ntones = cfg
            label = config_label(cfg)
            nch = 1024 if long else 128
            rate, _, written = rates(cfg)
            print(f'\n{label}: {nmod} module(s), dec {dec}, {nch} ch '
                  f'({ntones} written, {written:.1f} MiB/s), '
                  f'{rate:.1f} MiB/s, {sample_freq(dec) * nmod:,.0f} '
                  'packets/s')
            for method in methods:
                t = time.perf_counter()
                r = run_method(method, cfg, args, run_dir)
                results[(label, method)] = r
                print(f'  {method:<7} {loss_summary(r):<24} '
                      f'({time.perf_counter() - t:.0f} s)')
                if r.get('unavailable'):
                    break
                time.sleep(2)
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        ### Leave the board at the sweep stage
        await crs.set_decimation(6, short = False, module_idxs = [1, 2, 3, 4])

    ### Summary
    labels = list(dict.fromkeys(label for label, _ in results))
    header = f'{"config":<11} {"net MiB/s":>9} {"pkt/s":>8} {"disk MiB/s":>10}  ' \
        + ''.join(f'{m:<24}' for m in methods)
    lines = [header]
    for label in labels:
        net, pkts, written = rates(parse_config(label))
        line = f'{label:<11} {net:>9.1f} {pkts:>8,.0f} {written:>10.1f}  '
        line += ''.join(f'{loss_summary(results.get((label, m))):<24}'
                        for m in methods)
        lines.append(line)
    lines.append('codes: N NIC, K kernel backlog, S socket buffer, '
                 'Q rfmux queue, W writer stall, E error')
    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump({f'{k[0]}_{k[1]}': v for k, v in results.items()}, f,
                  indent = 1)


def main():
    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--serial', type = int, required = True,
                        help = 'CRS serial number')
    parser.add_argument('-i', '--interface', required = True,
                        help = 'network interface, e.g. enp5s0')
    parser.add_argument('--out', required = True,
                        help = 'directory for parser/zarr data (deleted '
                               'after each run)')
    parser.add_argument('--host', default = None,
                        help = 'CRS hostname or IP used by the benchmark '
                               'runs (default rfmux<serial>.local, via mDNS)')
    parser.add_argument('-t', '--duration', type = float, default = 20,
                        help = 'seconds per run (default 20)')
    parser.add_argument('--configs', default = DEFAULT_CONFIGS,
                        help = f'comma-separated configurations (default '
                               f'{DEFAULT_CONFIGS})')
    parser.add_argument('--methods', default = ','.join(METHODS),
                        help = f'comma-separated methods (default '
                               f'{",".join(METHODS)})')
    parser.add_argument('--disk-gb', type = float, default = 0,
                        help = 'run the disk benchmark with this many GB '
                               'first (default 0, skip)')
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
