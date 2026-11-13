"""
Check whether setting a module's NCO to the frequency it already has
changes the board's phase frame (not a numbered step: a diagnostic).

A calibration sweep is only usable while the board measures in the same
phase frame it was taken in, and citkid assumes that every NCO setting
starts a new frame. If a setting to the same frequency turns out to keep
the phase, a restarted session could set the NCOs again without going
stale. This script measures it:

1. writes a few tones around the NCO (``OFFSETS_HZ`` from ``NCO_HZ``,
   through the device: off-resonance tones are fine) and reads their
   complex S21 twice without touching the NCO (the control: the
   measurement's own phase noise and drift);
2. sets the NCO to the same frequency again (forced) and reads them again;
3. sets the NCO away by ``AWAY_HZ`` and back, and reads them again;

and repeats this ``REPEATS`` times. For each step it reports the phase
change of every tone against the reading before it, in degrees, and the
spread of that change across the tones: a change common to all tones is a
rotation of the frame, one growing with the offset from the NCO is a
delay. If the changes after the NCO settings look like the control, the
frame is kept.

The session attaches to one module (it needs the board configured by
crs_init.py), writes tones to it, and leaves them cleared. Nothing is
saved. Close 04_stream.py and other scripts using the module first: the
NCO settings here would put their calibrations out of frame.

Usage:
    python nco_phase_check.py
    python nco_phase_check.py --module 2 --nco 6.5e8 --repeats 5

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import asyncio
import numpy as np
from citkid.crs import board_lock
from citkid.crs.instrument import CRS

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 1           # CRS serial number
INTERFACE = 'enp2s0'        # Ethernet interface connected to the CRS
MODULE = 1                  # module to test
NCO_HZ = 6.5e8              # NCO frequency, Hz (e.g. a tone set's)
# Tone frequencies relative to the NCO, Hz
OFFSETS_HZ = [-150e6, -50e6, -10e6, -1e6, 1e6, 10e6, 50e6, 150e6]
ARES = -50.                 # tone power, dBm
NSAMPS = 200                # samples averaged per reading
REPEATS = 3                 # rounds of the four readings
AWAY_HZ = 1e6               # the away-and-back setting moves the NCO by

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
        description = 'Does setting an NCO to its own frequency change the '
        'phase frame?')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--module', type = int, default = MODULE)
    p.add_argument('--nco', type = float, default = NCO_HZ,
                   help = 'NCO frequency, Hz')
    p.add_argument('--nsamps', type = int, default = NSAMPS)
    p.add_argument('--repeats', type = int, default = REPEATS)
    return p.parse_args(argv)

def phase_changes(z0, z1):
    """
    Return the phase change of each tone between two readings.

    Parameters:
    z0, z1 (np.array, complex): the readings (one value per tone).

    Returns:
    dphi (np.array): phase change in degrees, in (-180, 180].
    ratio (np.array): amplitude ratio |z1| / |z0|.
    """
    z0, z1 = np.asarray(z0).ravel(), np.asarray(z1).ravel()
    return np.degrees(np.angle(z1 / z0)), np.abs(z1) / np.abs(z0)

def describe(label, offsets, z0, z1):
    """
    Describe one step's phase changes.

    Parameters:
    label (str): the step.
    offsets (array-like): tone offsets from the NCO, Hz.
    z0, z1 (np.array, complex): readings before and after the step.

    Returns:
    text (str): one line per tone, then the common change and the
        spread across tones.
    """
    dphi, ratio = phase_changes(z0, z1)
    lines = [f'{label}:']
    for off, d, r in zip(np.asarray(offsets).ravel(), dphi, ratio):
        lines.append(f'  {off / 1e6:+8.1f} MHz: {d:+8.2f} deg, '
                     f'|z| x {r:.4f}')
    # A common rotation: the circular mean; the spread around it
    mean = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(dphi)))))
    spread = np.degrees(np.angle(np.exp(1j * np.radians(dphi - mean))))
    lines.append(f'  common change {mean:+.2f} deg, spread across tones '
                 f'{np.std(spread):.2f} deg (rms), largest '
                 f'{np.max(np.abs(spread)):.2f} deg')
    return '\n'.join(lines)

async def run(args):
    """
    Attach to the module, take the readings, and print the comparison.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    readings (list of dict): per round, the readings 'a', 'b' (control),
        'c' (after the same NCO again), 'd' (after away and back), as
        complex arrays (one value per tone).
    """
    offsets = np.asarray(OFFSETS_HZ, dtype = float)
    freqs = (args.nco + offsets)[:, None]
    ares = np.full(len(offsets), ARES)
    crs = CRS(serial_number = args.serial, interface = args.iface,
              module_idxs = [args.module], global_control = False)
    await crs.attach(full_scale_dbm = 7, verbose = False)
    m = args.module

    async def read():
        """
        Read the tones once.

        Returns:
        z (np.array, complex): one value per tone.
        """
        _, z = await crs.sweep(freqs, ares, args.nsamps, verbose = False)
        return z[:, 0]

    await crs.set_nco({m: args.nco}, verbose = False, force = True)
    out = []
    for i in range(args.repeats):
        print(f'\nRound {i + 1} of {args.repeats}')
        a = await read()
        b = await read()
        print(describe('Control (no NCO setting)', offsets, a, b))
        await crs.set_nco({m: args.nco}, verbose = False, force = True)
        c = await read()
        print(describe('Same NCO set again', offsets, b, c))
        await crs.set_nco({m: args.nco + AWAY_HZ}, verbose = False,
                          force = True)
        await crs.set_nco({m: args.nco}, verbose = False, force = True)
        d = await read()
        print(describe(f'NCO moved {AWAY_HZ / 1e6:g} MHz away and back',
                       offsets, c, d))
        out.append(dict(a = a, b = b, c = c, d = d))
    await crs.clear_tones()
    print('\nIf the changes after the NCO settings are no larger than the '
          "control's, the phase frame survives a setting to the same "
          'frequency.')
    return out

def main(args):
    """
    Run the check while marking the board in use.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    readings (list of dict): see ``run``.
    """
    with board_lock.in_use(args.serial, owner = 'nco_phase_check.py'):
        return asyncio.run(run(args))

if __name__ == '__main__':
    main(parse_args())
