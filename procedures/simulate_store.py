"""
Make a device store with a VNA sweep of a simulated device (not a
numbered step: for trying the analysis steps and for example notebooks).

The sweep is taken by ``crs.procedures.vna_sweep_to_store`` with a
simulated CRS (``crs.sim``), so it is saved exactly like a real one
(``vna0``). The device's true parameters are saved in the store's
``simulation`` group; ``crs.sim.load_truth`` loads them and
``vna.simulate.compare_to_truth`` compares a fit with them.

Usage:
    python simulate_store.py --store ~/data/sim/simdevice.zarr
    python 02_find_resonances.py --store ~/data/sim/simdevice.zarr

Edit the settings below; the command-line options override them for one
run.
"""
import argparse
import asyncio
import os
from citkid import daq_layout as dl
from citkid.crs.procedures import vna_step
from citkid.crs.sim import simulate_vna_store

################################################################################
################################### Settings ###################################
################################################################################
STORE = '~/data/sim/simdevice.zarr'   # new store path
SEED = 0                    # device, gain, and noise seed
NYQUIST = '1'               # '1' (100-2500 MHz), '2' (2500-5000), 'both'
AMPLITUDE = -55.            # tone power, dBm
QR_MEDIAN = 2e4             # expected median Qr (sets the step)
PTS_PER_LW = 25.            # points per linewidth at QR_MEDIAN
NSAMPS = 100                # samples averaged per point
SNR_DB = 40.                # signal-to-noise ratio per quadrature at
                            # -55 dBm with 100 samples averaged

# Notes saved with the data this script makes (attribute 'notes'), e.g.
# the filters and attenuators on the DAC and ADC lines. Multi-line is fine.
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
    p = argparse.ArgumentParser(
        description = 'Make a store with a simulated device.')
    p.add_argument('--store', default = STORE)
    p.add_argument('--seed', type = int, default = SEED)
    p.add_argument('--nyquist', default = NYQUIST,
                   choices = ['1', '2', 'both'])
    p.add_argument('--amplitude', type = float, default = AMPLITUDE)
    p.add_argument('--qr-median', type = float, default = QR_MEDIAN)
    p.add_argument('--pts-per-lw', type = float, default = PTS_PER_LW)
    p.add_argument('--nsamps', type = int, default = NSAMPS)
    p.add_argument('--snr-db', type = float, default = SNR_DB)
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

async def main(args):
    """
    Simulate the device and save the store.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    truth (pd.DataFrame): the device's true parameters.
    """
    path = os.path.expanduser(args.store)
    root, truth = await simulate_vna_store(
        path, seed = args.seed, nyquist = args.nyquist,
        amplitude = args.amplitude, nsamps = args.nsamps,
        step = vna_step(args.qr_median, args.pts_per_lw),
        snr_db = args.snr_db)
    save_notes(root['vna0'], args.notes)
    print(f'Simulated {len(truth)} resonators; saved {path}')
    print(dl.tree(root, max_depth = 1))
    return truth

if __name__ == '__main__':
    asyncio.run(main(parse_args()))
