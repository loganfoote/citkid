"""
Zip a finished device store into one file (not a numbered step: run it
when a device's measurements are done).

The zip is uncompressed (zarr already compressed the data), opens
read-only without unpacking (``daq_layout.open_store``), and can be ported
from (``daq_layout.port_tuning``). It is checked against the store after
writing. An existing zip is never overwritten, and the store itself is
left in place: delete it yourself once the zip is safely stored.

Usage:
    python archive_store.py --device deviceA
    python archive_store.py --store ~/data/cd1/deviceA.zarr --zip /mnt/a.zip

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import os
from citkid import daq_layout as dl

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in
                            # DATA_DIR/COOLDOWN/DEVICE[_SUFFIX].zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
SUFFIX = ''                 # store name suffix: the store is
                            # DEVICE_SUFFIX.zarr ('': DEVICE.zarr)

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
    name = args.device + (f'_{args.suffix}' if args.suffix else '')
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{name}.zarr')

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
