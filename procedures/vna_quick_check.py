"""
Quick VNA check of one device (not a numbered step: use it whenever
a quick look helps).

Opens a window (``citkid.crs.quick_check``) with the settings below,
connects to this session's modules, and waits for the Sweep button
(Ctrl+R). A sweep covers the first Nyquist zone (100-2500 MHz), the second
(2500-5000 MHz), or both, on a geometric grid (constant df/f), leaving a
10 MHz guard band on each side of 2500 MHz (the point nearest it
glitches), with a progress bar, and plots |S21| and phase. A running
sweep can be cancelled (Cancel, Esc); it stops at the next point and
clears the board's tones. The settings can be changed in the window and
the sweep run again; the modules too (type them, e.g. "1 2" for a device
split over two modules, and press Connect, or just Sweep: it connects to
the typed modules first). Nothing is saved to zarr; the sweep can be
saved as a CSV (columns f in Hz, i, q). Use it to check the wiring,
power levels, and that resonances are there before the full sweep
(01_vna_sweep.py).

With --no-gui (e.g. over ssh without a display), it sweeps once with the
settings below and saves the sweep with --csv.

Needs the board configured by crs_init.py. This session attaches to
its modules only, so several of these (or other module scripts) can run at
once on different modules.

Usage:
    python vna_quick_check.py --modules 1
    python vna_quick_check.py --modules 1 2 --nyquist both
    python vna_quick_check.py --modules 1 --no-gui --csv quick.csv

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import asyncio
from citkid.crs.instrument import CRS
from citkid.crs.procedures import vna_range, vna_step, vna_sweep
from citkid.crs.quick_check import run_quick_check, save_sweep_csv

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46          # CRS serial number
INTERFACE = 'enp5s0'        # Ethernet interface connected to the CRS
MODULES = [1]               # modules connected to this device
NYQUIST = '1'               # '1' (100-2500 MHz), '2' (2500-5000), 'both'
AMPLITUDE = -50.            # tone power, dBm
QR_MEDIAN = 2e4             # expected median Qr
PTS_PER_LW = 5.             # points per linewidth at QR_MEDIAN (df/f = 1e-5)
NSAMPS = 10                 # samples averaged per point

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
