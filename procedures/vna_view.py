"""
View a saved VNA sweep (not a numbered step: use it after 01_vna_sweep.py
and before 02_find_resonances.py, or any time).

Opens a window (``citkid.vna.view``) with |S21| in dB and the phase of
S21 versus frequency for ``vna{k}`` of the device store
``<DATA_DIR>/<COOLDOWN>/<DEVICE>[_<SUFFIX>].zarr`` (the latest sweep by
default), with the band edges marked and the sweep's settings in the header. The
plots share the frequency axis: drag to pan, scroll to zoom, and press R
to fit the data again. Nothing is saved, and no CRS is needed.
01_vna_sweep.py can open this window itself when its sweep finishes
(``VIEW`` / ``--view``).

Usage:
    python vna_view.py --device deviceA
    python vna_view.py --device deviceA --vna 1
    python vna_view.py --store ~/data/sim/simdevice.zarr

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import os
from citkid.vna.view import run_vna_view

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in
                            # DATA_DIR/COOLDOWN/DEVICE[_SUFFIX].zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
SUFFIX = ''                 # store name suffix: the store is
                            # DEVICE_SUFFIX.zarr ('': DEVICE.zarr)
VNA = None                  # sweep index (vna{k}); None: the latest

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
    name = args.device + (f'_{args.suffix}' if args.suffix else '')
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{name}.zarr')

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
