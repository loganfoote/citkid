"""
Write a VNA sweep's results as plain files (plot, text summary, CSV of
the resonators) to OUT_DIR. Run after 02_find_resonances.py.

Usage:
    python vna_report.py --device deviceA --out ~/reports
    python vna_report.py --pick --out ~/reports
"""
import argparse
import os
from citkid.vna.report import export_vna_report

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'
COOLDOWN = 'cooldown'
DEVICE = 'device'
SUFFIX = ''
OUT_DIR = '~/data/reports'

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
    p = argparse.ArgumentParser(
        description = 'Write the results of a VNA sweep to plain files.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--pick', action = 'store_true',
                   help = 'choose the store in a file dialog')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device;'
                   ' may be a zip)')
    p.add_argument('--sweep', type = int, default = None,
                   help = 'VNA sweep index (default: the latest fitted)')
    p.add_argument('--out', default = OUT_DIR, help = 'output directory')
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
    Write the report files.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    paths (dict or None): ``plot``, ``info``, ``csv``: the files written
        (None if the dialog was cancelled).
    """
    import matplotlib
    matplotlib.use('Agg')
    store = store_path(args)
    if args.pick:
        from citkid.daq_browser import choose_store
        store = choose_store()
        if store is None:
            print('No store chosen.')
            return None
    paths = export_vna_report(store, os.path.expanduser(args.out),
                              k = args.sweep)
    for name, path in paths.items():
        print(f'{name}: {path}')
    return paths

if __name__ == '__main__':
    main(parse_args())
