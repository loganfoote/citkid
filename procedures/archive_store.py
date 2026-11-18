"""
Zip a finished device store into one file (uncompressed, opens read-only
without unpacking). The store is left in place.

Usage:
    python archive_store.py --device deviceA
    python archive_store.py --store ~/data/cd1/deviceA.zarr --zip /mnt/a.zip
"""
import argparse
import os
from citkid import daq_layout as dl

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'
COOLDOWN = 'cooldown'
DEVICE = 'device'
SUFFIX = ''

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
    p = argparse.ArgumentParser(description = 'Zip a device store.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--zip', default = None,
                   help = 'zip path (default: next to the store, .zarr.zip)')
    p.add_argument('--no-verify', action = 'store_true',
                   help = "don't check the zip against the store")
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
    Zip the store and print what it holds.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    zip_path (str): the zip's path.
    """
    path = store_path(args)
    dl.check_store(dl.open_store(path))
    zip_path = dl.archive_store(
        path, None if args.zip is None else os.path.expanduser(args.zip),
        verify = not args.no_verify)
    size = os.path.getsize(zip_path) / 1024 ** 2
    print(f'Archived {path}\n      to {zip_path} ({size:,.1f} MiB)')
    print(dl.tree(dl.open_store(zip_path), max_depth = 2))
    return zip_path

if __name__ == '__main__':
    main(parse_args())
