"""
Quick VNA sweeps of one device in a window (wiring, power, resonances),
saved only as a CSV if asked. Needs crs_init.py first.

Usage:
    python vna_quick_check.py --modules 1
    python vna_quick_check.py --modules 1 --no-gui --csv quick.csv
"""
import argparse
import asyncio
from citkid.crs.instrument import CRS
from citkid.crs.procedures import vna_range, vna_step, vna_sweep
from citkid.crs.quick_check import run_quick_check, save_sweep_csv

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46
INTERFACE = 'enp5s0'
MODULES = [1]
NYQUIST = '1'
AMPLITUDE = -50.
QR_MEDIAN = 2e4
PTS_PER_LW = 5.
NSAMPS = 10

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
        description = 'Take quick VNA sweeps of a device.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
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
    p.add_argument('--nsamps', type = int, default = NSAMPS)
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'sweep once without the window')
    p.add_argument('--csv', default = None,
                   help = 'with --no-gui: save the sweep (f, i, q) here')
    return p.parse_args(argv)

async def connect(args, modules = None):
    """
    Connect to modules of the configured board.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.
    modules (list of int or None): modules. None (default): ``--modules``.

    Returns:
    crs (CRS): the attached instrument.
    """
    crs = CRS(serial_number = args.serial, interface = args.iface,
              module_idxs = args.modules if modules is None else modules,
              global_control = False)
    # Attaching sets the other module settings to their defaults
    await crs.attach(full_scale_dbm = 7)
    return crs

async def sweep_once(args):
    """
    Connect, sweep, and save the sweep (no window).

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    f, z (np.array): the sweep.
    """
    crs = await connect(args)
    fmin, fmax = vna_range(args.nyquist, args.fmin, args.fmax)
    f, z, _ = await vna_sweep(
        crs, fmin, fmax, args.amplitude, args.nsamps,
        step = vna_step(args.qr_median, args.pts_per_lw))
    print(f'{len(f):,} points')
    if args.csv:
        save_sweep_csv(args.csv, f, z)
        print(f'Saved {args.csv}')
    return f, z

def main(args):
    """
    Open the quick-check window, or sweep once with --no-gui.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    if --no-gui:
        f, z: see ``sweep_once``.
    else:
        result (tuple or None): the window's last sweep (``f``, ``z``,
            ``bands``), or None.
    """
    if args.no_gui:
        return asyncio.run(sweep_once(args))
    return run_quick_check(
        lambda modules: connect(args, modules), args.modules,
        nyquist = args.nyquist,
        fmin = args.fmin, fmax = args.fmax, amplitude = args.amplitude,
        qr_median = args.qr_median, pts_per_lw = args.pts_per_lw,
        nsamps = args.nsamps)

if __name__ == '__main__':
    main(parse_args())
