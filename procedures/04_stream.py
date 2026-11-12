"""
04: take timestreams of the tuned tones in a window.

Opens ``crs.stream_gui`` on a tuning (``tune{k}`` with ``tones``, from
03_tune.py; the latest by default):

- Calibration: calibrate a tone set once (a target sweep with the
  settings below, which the window lets you change), saved as the
  tuning's ``cal{j}``; pick which saved calibration to use when there are
  several. A calibration takes long (minutes per set at ``NSAMPS``
  samples per point), so it is never repeated by a run.
- Preview: capture a few seconds (``PREVIEW_S``) of the chosen
  calibration's tones at a decimation stage, and look at one tone's
  timestream as x (projected onto the IQ loop's tangent at the tone),
  along/across the loop, or I and Q. Nothing is saved.
- Run: name a timestream (with notes), choose its stage and length (or a
  preset from ``PRESETS``), and run it on the selected tone set: the
  stream is saved under the calibration it belongs to and the tone write
  that put its tones on the board (``tune{k}/cal{j}/write{w}/ts{n}``),
  in chunks at fast stages where the tones don't all stream at once, and
  the run (``ts{n}``) is listed in the tuning's run list
  (``tune{k}/streams``). Repeat for the next timestream. Tones stay
  written on the board between captures and runs, so streams of the same
  tones don't write them again; Write tones writes the set again on
  purpose. For another tone set, select and calibrate it: setting its
  NCO starts a new phase frame, so the sets calibrated before need a new
  calibration when you come back to them.

Streaming needs the board's whole network bandwidth, so this script
takes the board for itself (``citkid.crs.board_lock``) for as long as its
window is open: it won't start while another script is sweeping or
streaming, and while it is open every other script fails to connect,
sweep, or stream (with a message saying who has the board). It also has
global control, to set the decimation. Run crs_init.py first. ``--simulate``
streams the simulated device of a store made by simulate_store.py.

Usage:
    python 04_stream.py --device deviceA --modules 1
    python 04_stream.py --device deviceA --modules 1 --tune 0

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.crs import board_lock
from citkid.crs.instrument import CRS
from citkid.crs.sim import SIM_SERIAL, simulated_crs
from citkid.crs.stream_gui import run_stream_gui

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46          # CRS serial number
INTERFACE = 'enp5s0'        # Ethernet interface connected to the CRS
DATA_DIR = '~/data'         # stores are in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
MODULES = [1]               # modules connected to this device
# Preview
PREVIEW_DEC = 6             # preview decimation stage
PREVIEW_S = 5.              # preview duration, s
# Runs offered as presets (name, decimation stage, duration in s, notes)
PRESETS = [
    dict(name = 'dark_dec6', dec_stage = 6, duration_s = 600.,
         notes = 'dark, long'),
    dict(name = 'dark_dec0', dec_stage = 0, duration_s = 60.,
         notes = 'dark, fast'),
]
# Calibration sweeps (as in 03_tune.py)
NPOINTS_FINE = 1000         # fine sweep points per tone
NPOINTS_GAIN = 50           # gain sweep points per tone
GAIN_SPAN_FACTOR = 10       # gain sweep span = 10 x fine sweep span
NSAMPS = 200                # samples averaged per point

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
    p = argparse.ArgumentParser(description = 'Take timestreams.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--tune', type = int, default = None,
                   help = 'tuning (default: the latest with tuned tones)')
    p.add_argument('--simulate', action = 'store_true',
                   help = "stream the store's simulated device")
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

def find_tuning(root, k = None):
    """
    Return the tuning to stream.

    Parameters:
    root (zarr.Group): the store root.
    k (int or None): tuning index. None: the latest with ``tones``.

    Returns:
    tune (zarr.Group): the tuning.

    Raises:
    KeyError: if there is no such tuning, or it has no tones.
    """
    if k is None:
        done = [j for j in dl.list_indices(root, 'tune')
                if 'tones' in root[dl.set_name(j, 'tune')]]
        if not done:
            raise KeyError('no tuning has tones yet: run 03_tune.py')
        k = done[-1]
    tune = root[dl.set_name(k, 'tune')]
    if 'tones' not in tune:
        raise KeyError(f'{tune.basename} has no tones yet: run 03_tune.py')
    return tune

def main(args):
    """
    Open the timestream window.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    runs (list of list of str): stream paths of each run.
    """
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    tune = find_tuning(root, args.tune)

    async def connect():
        """
        Attach to this session's modules (or the simulated board) and
        check the CRS against the store.

        Returns:
        crs (CRS or SimulatedCRS): the attached instrument.
        """
        if args.simulate:
            crs = simulated_crs(path, args.modules, seed = 200,
                                global_control = True)
        else:
            crs = CRS(serial_number = args.serial, interface = args.iface,
                      module_idxs = args.modules,
                      global_control = True)
        # Attaching sets the other module settings to their defaults
        await crs.attach(full_scale_dbm = 7)
        dl.check_crs_identity(root, crs)
        return crs

    serial = SIM_SERIAL if args.simulate else args.serial
    owner = f'04_stream.py ({os.path.basename(path)}, {tune.basename})'
    # The board is this script's while the window is open
    with board_lock.exclusive(serial, owner = owner):
        return run_stream_gui(
            connect, tune, preview_dec = PREVIEW_DEC, preview_s = PREVIEW_S,
            presets = PRESETS, sweep_kwargs = dict(
                npoints_fine = NPOINTS_FINE, npoints_gain = NPOINTS_GAIN,
                npoints_rough = None, nsamps = NSAMPS,
                gain_span_factor = GAIN_SPAN_FACTOR))

if __name__ == '__main__':
    main(parse_args())
