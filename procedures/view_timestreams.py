"""
View saved timestreams: the plots of 4a_stream.py (timestream, loop,
PSD) on runs already in the store, with no board. Choose the tuning,
the run, the chunks and the tone in the window.

Usage:
    python view_timestreams.py --device deviceA
    python view_timestreams.py --device deviceA --tuning BBT80K
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.crs.stream_viewer import run_timestream_viewer
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX, LIGHT_MODE

TUNING = None

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
    p = argparse.ArgumentParser(description = 'View saved timestreams.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--tuning', default = TUNING,
                   help = 'tuning shown first, by name or index (default: '
                   'the latest with runs)')
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
    Open the viewer on the store.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    win (TimestreamViewer): the window, closed.
    """
    set_theme(LIGHT_MODE)
    path = store_path(args)
    root = dl.open_store(path)
    dl.check_store(root)
    return run_timestream_viewer(root, path = path, tuning = args.tuning,
                                 title = "Timestreams: "
                                 + os.path.basename(path))

if __name__ == '__main__':
    main(parse_args())
