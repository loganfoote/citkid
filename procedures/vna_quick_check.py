"""
Quick VNA check of one device (not a numbered step: use it whenever
a quick look helps).

Opens a window (``citkid.crs.quick_check``) with the settings below,
connects to this session's modules, and waits for the Sweep button
(Ctrl+R). A sweep covers the first Nyquist zone (100-2500 MHz), the second
(2500-5000 MHz), or both, on a geometric grid (constant df/f), leaving a
10 MHz guard band on each side of 2500 MHz (the point nearest it
glitches), with a progress bar, and plots |S21| and phase with a rough
count of the
resonances (matched filter). The settings can be changed in the window
and the sweep run again. Nothing is saved to zarr; the plot can be saved
as a PNG. Use it to check the wiring, power levels, and that resonances
are there before the full sweep (01_vna_sweep.py).

With --no-gui (e.g. over ssh without a display), it sweeps once with the
settings below, prints the count, and saves the plot with --png.

Needs the board configured by crs_init.py. This session attaches to
its modules only, so several of these (or other module scripts) can run at
once on different modules.

Usage:
    python vna_quick_check.py --modules 1
    python vna_quick_check.py --modules 1 2 --nyquist both
    python vna_quick_check.py --modules 1 --no-gui --png quick.png

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from citkid.crs.instrument import CRS
from citkid.crs.procedures import vna_range, vna_step, vna_sweep
from citkid.crs.quick_check import rough_resonances, run_quick_check

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46          # CRS serial number
INTERFACE = 'enp5s0'        # Ethernet interface connected to the CRS
MODULES = [1]               # modules connected to this device
FULL_SCALE_DBM = 7          # DAC full scale of these modules, 0-7 dBm
NYQUIST = '1'               # '1' (100-2500 MHz), '2' (2500-5000), 'both'
AMPLITUDE = -50.            # tone power, dBm
QR_MEDIAN = 2e4             # expected median Qr
PTS_PER_LW = 5.             # points per linewidth at QR_MEDIAN (df/f 1e-5)
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
    p = argparse.ArgumentParser(description = 'Quick VNA check.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
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
    p.add_argument('--nsamps', type = int, default = NSAMPS)
    p.add_argument('--no-count', action = 'store_true',
                   help = "don't count resonances")
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'sweep once without the window')
    p.add_argument('--png', default = None,
                   help = 'with --no-gui: save the plot here')
    return p.parse_args(argv)

async def connect(args):
    """
    Connect to this session's modules of the configured board.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    crs (CRS): the attached instrument.
    """
    crs = CRS(serial_number = args.serial, interface = args.iface,
              module_idxs = args.modules, global_control = False)
    # Attaching sets the other module settings to their defaults
    await crs.attach(full_scale_dbm = args.full_scale)
    return crs

def plot_sweep(f, z, bands, title = ''):
    """
    Plot |S21| and the unwrapped phase, with the band edges marked.

    Parameters:
    f (np.array): frequencies in Hz.
    z (np.array): complex S21.
    bands (dict): ``vna_sweep`` band arrays.
    title (str): figure title. Default ''.

    Returns:
    fig (matplotlib.figure.Figure): the figure.
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex = True, figsize = (10, 6))
    ax0.plot(f / 1e6, 20 * np.log10(np.abs(z)), lw = 0.5)
    ax1.plot(f / 1e6, np.unwrap(np.angle(z)), lw = 0.5)
    for lo in bands['chunk_edges'][1:, 0]:
        for ax in (ax0, ax1):
            ax.axvline(lo / 1e6, color = 'gray', ls = ':', lw = 0.8)
    ax0.set_ylabel('|S21| (dB)')
    ax1.set_ylabel('Phase (rad)')
    ax1.set_xlabel('Frequency (MHz)')
    ax0.set_title(title)
    fig.tight_layout()
    return fig

async def sweep_once(args):
    """
    Connect, sweep, count resonances, and save the plot (no window).

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    f, z (np.array): the sweep.
    n_found (int or None): rough resonance count (None with --no-count).
    """
    crs = await connect(args)
    fmin, fmax = vna_range(args.nyquist, args.fmin, args.fmax)
    f, z, bands = await vna_sweep(
        crs, fmin, fmax, args.amplitude, args.nsamps,
        step = vna_step(args.qr_median, args.pts_per_lw))
    n_found = None
    title = (f'Modules {args.modules}, {args.amplitude:g} dBm, '
             f'{len(f)} points')
    if not args.no_count:
        n_found = len(rough_resonances(f, z))
        print(f'About {n_found} resonances (matched filter; a rough count)')
        title += f', ~{n_found} resonances'
    if args.png:
        fig = plot_sweep(f, z, bands, title)
        fig.savefig(args.png, dpi = 150)
        plt.close(fig)
        print(f'Saved {args.png}')
    return f, z, n_found

def main(args):
    """
    Open the quick-check window, or sweep once with --no-gui.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    if --no-gui:
        f, z, n_found: see ``sweep_once``.
    else:
        result (tuple or None): the window's last sweep (``f``, ``z``,
            ``bands``, ``fr``), or None.
    """
    if args.no_gui:
        return asyncio.run(sweep_once(args))
    return run_quick_check(
        lambda: connect(args), args.modules, nyquist = args.nyquist,
        fmin = args.fmin, fmax = args.fmax, amplitude = args.amplitude,
        qr_median = args.qr_median, pts_per_lw = args.pts_per_lw,
        nsamps = args.nsamps, count = not args.no_count)

if __name__ == '__main__':
    main(parse_args())
