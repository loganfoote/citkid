"""
Show a saved VNA sweep (vna{k}, the latest by default) in a window.
No CRS needed.

Usage:
    python vna_view.py --device deviceA
    python vna_view.py --device deviceA --vna 1
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.vna.view import run_vna_view

################################################################################
################################### Settings ###################################
################################################################################
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX

VNA = None

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
    p = argparse.ArgumentParser(description = 'View a saved VNA sweep.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--vna', type = int, default = VNA,
                   help = 'VNA sweep index (default: the latest)')
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

def main(args):
    """
    Open the window on the sweep and wait until it is closed.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    k (int): index of the sweep shown.
    """
    path = store_path(args)
    k = run_vna_view(path, args.vna)
    print(f'Viewed vna{k} of {path}')
    return k

if __name__ == '__main__':
    main(parse_args())
