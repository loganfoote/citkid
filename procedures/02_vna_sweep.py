"""
02: full VNA sweep of one device, saved to its store.

Sweeps the first Nyquist zone (100-2500 MHz), the second (2500-5000
MHz), or both with this session's modules, leaving a 10 MHz guard band on
each side of 2500 MHz (the point nearest it glitches), in several passes
if the range is wider than the modules cover at once, on a geometric grid
(constant df/f, from the expected median Qr: 25 points per linewidth
there by default; every channel sweeps downward), and saves the stitched
sweep as the next ``vna/sweep{k}`` of the device store
``<DATA_DIR>/<COOLDOWN>/<DEVICE>.zarr`` (created if needed; see
``documents/daq_layout.md``). The next step finds the resonances on it.

Needs the board configured by 00_crs_init.py. This session attaches to
its modules only, so several of these can run at once, one per device.
Before sweeping, it checks that the CRS (serial number, firmware, rfmux)
is the one the store was measured with; the other CRS settings may differ
between acquisitions and are saved with the sweep.

Usage:
    python 02_vna_sweep.py --device deviceA --modules 1
    python 02_vna_sweep.py --device deviceA --modules 1 2 3 4 --nyquist both

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import asyncio
import os
from citkid import daq_layout as dl
from citkid.crs.instrument import CRS
from citkid.crs.procedures import vna_range, vna_step, vna_sweep_to_store

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46          # CRS serial number
INTERFACE = 'enp5s0'        # Ethernet interface connected to the CRS
DATA_DIR = '~/data'         # stores go in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
# Free-form description saved with a new store (fridge channel,
# attenuators, filters, ...)
SETUP = dict(fridge_ch = None, dac_attenuation = None, dac_filters = None,
             adc_filters = None)
MODULES = [1]               # modules connected to this device
FULL_SCALE_DBM = 7          # DAC full scale of these modules, 0-7 dBm
NYQUIST = '1'               # '1' (100-2500 MHz), '2' (2500-5000), 'both'
AMPLITUDE = -55.            # tone power, dBm
QR_MEDIAN = 2e4             # expected median Qr
PTS_PER_LW = 25.            # points per linewidth at QR_MEDIAN (df/f 2e-6)
NSAMPS = 100                # samples averaged per point
# > 10 PTS_PER_LW ideal for fitting

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
    p = argparse.ArgumentParser(description = 'Full VNA sweep to a store.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--full-scale', type = float, default = FULL_SCALE_DBM)
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
    return p.parse_args(argv)

def store_path(args):
    """
    Return the device store's path.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str): ``<data dir>/<cooldown>/<device>.zarr``.
    """
    return os.path.join(os.path.expanduser(args.data_dir), args.cooldown,
                        f'{args.device}.zarr')

async def main(args):
    """
    Attach to the modules, sweep, and save.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    k (int): index of the saved sweep (``vna/sweep{k}``).
    """
    path = store_path(args)
    root = dl.create_store(path, args.device, args.cooldown, setup = SETUP)
    crs = CRS(serial_number = args.serial, interface = args.iface,
              module_idxs = args.modules, global_control = False)
    # Attaching sets the other module settings to their defaults
    await crs.attach(full_scale_dbm = args.full_scale)
    fmin, fmax = vna_range(args.nyquist, args.fmin, args.fmax)
    k, f, _ = await vna_sweep_to_store(
        crs, root, fmin, fmax, args.amplitude, args.nsamps,
        step = vna_step(args.qr_median, args.pts_per_lw),
        npoints_per_ch = args.npoints)
    print(f'Saved {len(f)} points as vna/sweep{k} in {path}')
    print(dl.tree(root, max_depth = 2))
    return k

if __name__ == '__main__':
    asyncio.run(main(parse_args()))
