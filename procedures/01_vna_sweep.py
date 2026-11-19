"""
01: take a full VNA sweep of one device and save it as the next vna{k}
of the device store. Needs crs_init.py first.

Usage:
    python 01_vna_sweep.py --device deviceA --modules 1 --nyquist both
    python 01_vna_sweep.py --device deviceA --modules 1 --view
"""
import argparse
import asyncio
import os
from citkid import daq_layout as dl
from citkid.crs.instrument import CRS
from citkid.crs.procedures import vna_range, vna_step, vna_sweep_to_store
from citkid.vna.view import run_vna_view

################################################################################
################################### Settings ###################################
################################################################################
from settings import (SERIAL_NUMBER, INTERFACE, MODULES, DATA_DIR, COOLDOWN,
                      DEVICE, SUFFIX)

SETUP = dict(fridge_ch = None, dac_attenuation = None, dac_filters = None,
             adc_filters = None)
NYQUIST = '1'
AMPLITUDE = -55.
QR_MEDIAN = 2e4
PTS_PER_LW = 25.
NSAMPS = 100
VIEW = False

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
    p = argparse.ArgumentParser(description = 'Take a full VNA sweep.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--nyquist', default = NYQUIST,
                   choices = ['1', '2', 'both'])
    p.add_argument('--fmin', type = float, default = None,
                   help = "override the zone's lower edge, Hz")
    p.add_argument('--fmax', type = float, default = None,
                   help = "override the zone's upper edge, Hz")
    p.add_argument('--amplitude', type = float, default = AMPLITUDE)
    p.add_argument('--qr-median', type = float, default = QR_MEDIAN)
    p.add_argument('--pts-per-lw', type = float, default = PTS_PER_LW)
    p.add_argument('--npoints', type = int, default = None,
                   help = 'points per channel (default: fastest)')
    p.add_argument('--nsamps', type = int, default = NSAMPS)
    p.add_argument('--view', action = argparse.BooleanOptionalAction,
                   default = VIEW,
                   help = 'open the sweep in a window when done')
    p.add_argument('--notes', default = NOTES,
                   help = 'notes saved with the data (quote multi-line text)')
    return p.parse_args(argv)

def save_notes(grp, notes):
    """
    Save notes with a group the script made (attribute 'notes'), if any.

    Parameters:
    grp (zarr.Group): the group.
    notes (str): the notes (``--notes`` or ``NOTES``); blank is not saved.
    """
    if notes and notes.strip():
        grp.attrs['notes'] = notes.strip()

def store_path(args):
    """
    Return the device store's path.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str): ``<data dir>/<cooldown>/<device>[_<suffix>].zarr``.
    """
    return dl.store_path(args.data_dir, args.cooldown, args.device,
                         args.suffix)

async def main(args):
    """
    Attach to the modules, sweep, save, and show the sweep if asked to.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    k (int): index of the saved sweep (``vna{k}``).
    """
    path = store_path(args)
    root = dl.create_store(path, args.device, args.cooldown, setup = SETUP)
    crs = CRS(serial_number = args.serial, interface = args.iface,
              module_idxs = args.modules, global_control = False)
    # Attaching sets the other module settings to their defaults
    await crs.attach(full_scale_dbm = 7)
    fmin, fmax = vna_range(args.nyquist, args.fmin, args.fmax)
    k, f, _ = await vna_sweep_to_store(
        crs, root, fmin, fmax, args.amplitude, args.nsamps,
        step = vna_step(args.qr_median, args.pts_per_lw),
        npoints_per_ch = args.npoints)
    save_notes(root[f'vna{k}'], args.notes)
    print(f'Saved {len(f)} points as vna{k} in {path}')
    print(dl.tree(root, max_depth = 2))
    if args.view:
        run_vna_view(root, k)
    return k

if __name__ == '__main__':
    asyncio.run(main(parse_args()))
