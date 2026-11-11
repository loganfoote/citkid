"""
Export the results of a VNA screening (after 01_vna_sweep.py and
02_find_resonances.py) as plain files, for people who don't use zarr or
Python (not a numbered step: run it whenever a report is useful).

Writes to ``OUT_DIR`` (``citkid.vna.report.export_vna_report``):

- ``<device>_vna<k>.png``: the gain-removed sweep with the resonances
  marked, and Qi, Qc, and phi vs frequency where the fits are reliable;
  the title has the device name;
- ``<device>_vna<k>_info.txt``: the sweep's settings (power, averaging,
  resolution, range, modules, Nyquist zones, bands), the finding and
  fitting steps, the number of resonances, the frequency schedule (range,
  gaps, banks), typical quality factors, data-quality notes, and notes;
- ``<device>_vna<k>_resonators.csv``: one row per resonator (``res_idx``
  as in the store's ``resonators{k}``): fr, Qi, Qc, Qr, amp, phi with
  errors (NaN where unreliable), flags, and fit status.

Usage:
    python vna_report.py --device deviceA --out ~/reports
    python vna_report.py --store ~/data/cd1/deviceA.zarr --sweep 0

Edit the settings below; the command-line options override them for one
run.
"""
import argparse
import os
from citkid.vna.report import export_vna_report

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
OUT_DIR = '~/data/reports'  # where the files go

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
    path (str): ``--store``, or ``<data dir>/<cooldown>/<device>.zarr``.
    """
    if args.store:
        return os.path.expanduser(args.store)
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{args.device}.zarr')

def main(args):
    """
    Write the report files.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    paths (dict): ``plot``, ``info``, ``csv``: the files written.
    """
    import matplotlib
    matplotlib.use('Agg')
    paths = export_vna_report(store_path(args), os.path.expanduser(args.out),
                              k = args.sweep)
    for name, path in paths.items():
        print(f'{name}: {path}')
    return paths

if __name__ == '__main__':
    main(parse_args())
