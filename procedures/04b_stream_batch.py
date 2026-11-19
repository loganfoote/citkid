"""
04b: calibrate a tone set and take the timestreams in RUNS without a
window, each writing the tone chunks its stage needs
(tune{k}/cal{j}/write{w}/ts{n}). Needs crs_init.py first. To take
several sets or settings in a row, loop over ``take`` (or
``citkid.crs.procedures.stream_batch``) as ``main`` does.

Usage:
    python 04b_stream_batch.py --device deviceA --modules 1
    python 04b_stream_batch.py --device deviceA --modules 1 --tune 0
"""
import argparse
import asyncio
import os
from citkid import daq_layout as dl
from citkid.crs import board_lock
from citkid.crs.instrument import CRS
from citkid.crs.procedures import stream_batch
from citkid.crs.sim import SIM_SERIAL, simulated_crs

################################################################################
################################### Settings ###################################
################################################################################
from settings import (SERIAL_NUMBER, INTERFACE, MODULES, DATA_DIR, COOLDOWN,
                      DEVICE, SUFFIX)

# Calibration
TONE_SET = 0
CALIBRATION = None
FRES_UPDATE = None
NPOINTS_FINE = 1000
NPOINTS_GAIN = 50
GAIN_SPAN_FACTOR = 10
NSAMPS = 200

# Timestreams
TONE_CHUNKS = None
RUNS = [
    dict(name = 'dark_dec6', dec_stage = 6, duration_s = 600.,
         notes = 'dark, long'),
    dict(name = 'dark_dec0', dec_stage = 0, duration_s = 60.,
         notes = 'dark, fast'),
]

NOTES = """
"""

################################################################################
################################### Script #####################################
################################################################################
def parse_args(argv = None):
    """
    Parse the command-line options (defaults from the settings above).

    Parameters:
    argv (list of str or None): arguments. None (default): sys.argv.

    Returns:
    args (argparse.Namespace): the options.
    """
    p = argparse.ArgumentParser(description = 'Take timestreams without a '
                                'window.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--tune', type = int, default = None,
                   help = 'tuning (default: the latest with tuned tones)')
    p.add_argument('--simulate', action = 'store_true',
                   help = "stream the store's simulated device")
    p.add_argument('--notes', default = NOTES,
                   help = 'notes saved with the calibration')
    return p.parse_args(argv)

def store_path(args):
    """
    Return the device store's path.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str): ``--store``, or
        ``<data dir>/<cooldown>/<device>[_<suffix>].zarr``.
    """
    if args.store:
        return os.path.expanduser(args.store)
    return dl.store_path(args.data_dir, args.cooldown, args.device,
                         args.suffix)

def find_tuning(root, k = None):
    """
    Return the tuning to stream.

    Parameters:
    root (zarr.Group): the store root.
    k (int or None): tuning index. None: the latest with ``tones``.

    Returns:
    tune (zarr.Group): the tuning.

    Raises:
    KeyError: if there is no such tuning, or it has no tones.
    """
    if k is None:
        done = [j for j in dl.list_indices(root, 'tune')
                if 'tones' in root[dl.set_name(j, 'tune')]]
        if not done:
            raise KeyError('no tuning has tones yet: run 03_tune.py')
        k = done[-1]
    tune = root[dl.set_name(k, 'tune')]
    if 'tones' not in tune:
        raise KeyError(f'{tune.basename} has no tones yet: run 03_tune.py')
    return tune

async def connect(args, root, path):
    """
    Attach to this session's modules (or the simulated board) with global
    control and check the CRS against the store.

    Parameters:
    args (argparse.Namespace): options.
    root (zarr.Group): the store root.
    path (str): store path (for ``--simulate``).

    Returns:
    crs (CRS or SimulatedCRS): the attached instrument.
    """
    if args.simulate:
        crs = simulated_crs(path, args.modules, seed = 200,
                            global_control = True)
    else:
        crs = CRS(serial_number = args.serial, interface = args.iface,
                  module_idxs = args.modules, global_control = True)
    # Attaching sets the other module settings to their defaults
    await crs.attach(full_scale_dbm = 7)
    dl.check_crs_identity(root, crs)
    return crs

async def take(crs, tune, notes = ''):
    """
    Calibrate TONE_SET (or use CALIBRATION) and take RUNS, each writing
    its stage's tone chunks (or TONE_CHUNKS). Change the settings and
    call it again for the next set.

    Parameters:
    crs (CRS or SimulatedCRS): the attached instrument.
    tune (zarr.Group): the tuning.
    notes (str): notes saved with a calibration taken here. Default ''.

    Returns:
    result (dict): ``calibration``, ``runs`` (``stream_batch``).
    """
    result = await stream_batch(
        crs, tune, TONE_SET, RUNS, calibration = CALIBRATION,
        fres_update = FRES_UPDATE, tone_chunks = TONE_CHUNKS,
        notes = notes, status = print, npoints_fine = NPOINTS_FINE,
        npoints_gain = NPOINTS_GAIN, npoints_rough = None, nsamps = NSAMPS,
        gain_span_factor = GAIN_SPAN_FACTOR)
    for r, paths in zip(RUNS, result['runs']):
        print(f"{r['name']}: {paths}")
    return result

def main(args):
    """
    Connect, hold the board, and take the runs.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    result (dict): ``take``'s result.
    """
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    tune = find_tuning(root, args.tune)
    serial = SIM_SERIAL if args.simulate else args.serial
    owner = f'04b_stream_batch.py ({os.path.basename(path)}, {tune.basename})'
    with board_lock.exclusive(serial, owner = owner):
        async def run():
            """
            Connect and take the runs on one event loop.

            Returns:
            result (dict): ``take``'s result.
            """
            crs = await connect(args, root, path)
            return await take(crs, tune, (args.notes or '').strip())

        return asyncio.run(run())

if __name__ == '__main__':
    main(parse_args())
