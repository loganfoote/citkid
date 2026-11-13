"""
Browse a device store (not a numbered step: use it any time, e.g. to
remind yourself which timestreams were taken and what their notes say
before setting up 05_calibrate_streams.py or 06_find_photons.py).

Opens a window (``citkid.daq_browser``) on the device store
``<DATA_DIR>/<COOLDOWN>/<DEVICE>[_<SUFFIX>].zarr``, or on a store chosen
in a file dialog (``PICK`` / ``--pick``, or when that store does not
exist; Open (Ctrl+O) in the window chooses another at any time): a tree
of its groups
and arrays (VNA sweeps, resonator lists, tunings with their calibrations,
tone writes and streams, stream calibrations, photon results), each with
a one-line summary, and the attributes of whatever is selected (names,
notes, settings). The Timestream runs tab lists every run with its name,
notes, stage, length, tone set, and calibration; type to filter them, and
double-click one to find its stream in the tree. The store is opened
read-only, so it can be browsed while another script writes to it
(F5 rereads it). Nothing is saved, and no CRS is needed.

With --runs, it prints the timestream runs instead of opening the window
(e.g. over ssh without a display).

Usage:
    python store_browser.py --device deviceA
    python store_browser.py --pick
    python store_browser.py --store ~/data/sim/simdevice.zarr
    python store_browser.py --device deviceA --runs

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.daq_browser import run_store_browser

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in
                            # DATA_DIR/COOLDOWN/DEVICE[_SUFFIX].zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
SUFFIX = ''                 # store name suffix: the store is
                            # DEVICE_SUFFIX.zarr ('': DEVICE.zarr)
PICK = False                # choose the store in a file dialog instead

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
    p = argparse.ArgumentParser(description = 'Browse a device store.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--pick', action = argparse.BooleanOptionalAction,
                   default = PICK,
                   help = 'choose the store in a file dialog')
    p.add_argument('--runs', action = 'store_true',
                   help = 'print the timestream runs instead of opening '
                   'the window')
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
    name = args.device + (f'_{args.suffix}' if args.suffix else '')
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{name}.zarr')

def main(args):
    """
    Open the browser on the store, or print its timestream runs.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str or None): the store's path (the one chosen in the dialog),
        or None if none was opened.
    """
    path = store_path(args)
    if args.runs:
        root = dl.open_store(path)
        dl.check_store(root)
        print(f'{path}: {dl.tree(root, max_depth = 1).count(chr(10)) + 1} '
              'top-level groups')
        print(dl.describe_stream_runs(root))
        return path
    if args.pick:
        path = None
    elif not os.path.exists(path):
        print(f'{path} does not exist: choose a store in the dialog')
        path = None
    path = run_store_browser(path)
    print(f'Browsed {path}' if path else 'No store opened')
    return path

if __name__ == '__main__':
    main(parse_args())
