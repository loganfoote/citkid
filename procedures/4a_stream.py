"""
04a: take timestreams of a tuning's tones in a window: calibrate a tone
set, write its tones, preview, and run named timestreams
(tune{k}/cal{j}/write{w}/ts{n}). 4b_stream_batch.py does it without
a window.
Takes the board for itself while open. Needs crs_init.py first.

Usage:
    python 4a_stream.py --device deviceA --modules 1
    python 4a_stream.py --device deviceA --modules 1 --tuning 0
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.crs import board_lock
from citkid.crs.instrument import CRS
from citkid.crs.sim import SIM_SERIAL, simulated_crs
from citkid.crs.stream_gui import run_stream_gui
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import (SERIAL_NUMBER, INTERFACE, MODULES, DATA_DIR, COOLDOWN,
                      DEVICE, SUFFIX, LIGHT_MODE, SETUP)

# Calibration sweep
NPOINTS_FINE = 1000
NPOINTS_GAIN = 50
GAIN_SPAN_FACTOR = 10
NSAMPS = 200
FRES_UPDATE = None

# Preview and runs
PREVIEW_DEC = 6
PREVIEW_S = 5.
PRESETS = [
    dict(name = 'dark_dec6', dec_stage = 6, duration_s = 600.,
         notes = 'dark, long'),
    dict(name = 'dark_dec0', dec_stage = 0, duration_s = 60.,
         notes = 'dark, fast'),
]
EXTRA_FEATURES = False


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
    p.add_argument('--suffix', default = SUFFIX,
                   help = 'store name suffix: <device>_<suffix>.zarr '
                   '(blank: <device>.zarr)')
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--modules', type = int, nargs = '+', default = MODULES)
    p.add_argument('--tuning', default = None,
                   help = 'tuning, by name or index (default: the '
                   'latest with tuned tones)')
    p.add_argument('--simulate', action = 'store_true',
                   help = "stream the store's simulated device")
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
    return dl.store_path(args.data_dir, args.cooldown, args.device,
                         args.suffix)

def find_tuning(root, k = None):
    """
    Return the tuning to stream.

    Parameters:
    root (zarr.Group): the store root.
    k (int, str or None): tuning index or name (``daq_layout.find_tuning``).
        None: the latest with ``tones``.

    Returns:
    tune (zarr.Group): the tuning.

    Raises:
    KeyError: if there is no such tuning, or it has no tones.
    """
    if k is None:
        done = [j for j in dl.list_indices(root, 'tune')
                if 'tones' in root[dl.set_name(j, 'tune')]]
        if not done:
            raise KeyError('no tuning has tones yet: run 3_tune.py')
        k = done[-1]
    tune = dl.find_tuning(root, k)
    if tune is None:
        raise KeyError(f'no tuning {k!r}')
    if 'tones' not in tune:
        raise KeyError(f'{tune.basename} has no tones yet: run 3_tune.py')
    return tune

def main(args):
    """
    Open the timestream window.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    runs (list of list of str): stream paths of each run.
    """
    set_theme(LIGHT_MODE)
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    tune = find_tuning(root, args.tuning)

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
    owner = f'4a_stream.py ({os.path.basename(path)}, {tune.basename})'
    # The board is this script's while the window is open
    with board_lock.exclusive(serial, owner = owner):
        return run_stream_gui(
            connect, tune, preview_dec = PREVIEW_DEC, preview_s = PREVIEW_S,
            presets = PRESETS, sweep_kwargs = dict(
                npoints_fine = NPOINTS_FINE, npoints_gain = NPOINTS_GAIN,
                nsamps = NSAMPS,
                gain_span_factor = GAIN_SPAN_FACTOR),
            fres_update = FRES_UPDATE, setup = SETUP,
            extra_features = EXTRA_FEATURES)

if __name__ == '__main__':
    main(parse_args())
