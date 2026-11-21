"""
04c: take a sweep of optical power: at each point, set the power
(``set_optical_power``, edited for the setup and settling there),
calibrate the tone set (or reuse a calibration) and take the
timestreams in RUNS (``stream_batch``, as 04b). The point's name is
saved with its calibration sweep and its runs, for 09_nep.py, where
the names are mapped to powers. Needs crs_init.py first.

Usage:
    python 04c_power_sweep.py --device deviceA --modules 1
    python 04c_power_sweep.py --device deviceA --modules 1 --from BBT80K
"""
import argparse
import asyncio
import os
from citkid import daq_layout as dl
from citkid.crs import board_lock
from citkid.crs.instrument import CRS
from citkid.crs.procedures import stream_batch
from citkid.crs.sim import SIM_SERIAL, simulated_crs

################################################################################
################################### Settings ###################################
################################################################################
from settings import (SERIAL_NUMBER, INTERFACE, MODULES, DATA_DIR, COOLDOWN,
                      DEVICE, SUFFIX)

# Tuning
TONE_SET = 0

# Points
POINTS = [
    dict(name = 'BBT40K'),
    dict(name = 'BBT80K'),
    dict(name = 'BBT80K_again', cal = 'BBT80K'),
]

# Timestreams at each point
RUNS = [
    dict(dec_stage = 0, duration_s = 200.),
]
TONE_CHUNKS = None

# Calibration sweep
FRES_UPDATE = dict(method = 'spacing', max_shift = 0.1, npoints_update = None)
NPOINTS_FINE = 1000
NPOINTS_GAIN = 50
GAIN_SPAN_FACTOR = 10
NSAMPS = 200

NOTES = """
"""

def set_optical_power(name):
    """
    Set the optical power of a point and wait for it to settle: edit
    this for the setup (the blackbody temperature, an attenuator, ...).

    Parameters:
    name (str): the point's name, e.g. 'BBT40K'.
    """
    print(f'Set the optical power for {name}')

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
    p = argparse.ArgumentParser(description = 'Take a sweep of optical '
                                'power.')
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
    p.add_argument('--from', dest = 'start', default = None,
                   help = 'start at this point (its name), e.g. to go on '
                   'after a stop')
    p.add_argument('--simulate', action = 'store_true',
                   help = "stream the store's simulated device")
    p.add_argument('--notes', default = NOTES,
                   help = 'notes saved with each calibration')
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
            raise KeyError('no tuning has tones yet: run 03_tune.py')
        k = done[-1]
    tune = dl.find_tuning(root, k)
    if tune is None:
        raise KeyError(f'no tuning {k!r}')
    if 'tones' not in tune:
        raise KeyError(f'{tune.basename} has no tones yet: run 03_tune.py')
    return tune

async def connect(args, root, path):
    """
    Attach to this session's modules (or the simulated board) with global
    control and check the CRS against the store.

    Parameters:
    args (argparse.Namespace): options.
    root (zarr.Group): the store root.
    path (str): store path (for ``--simulate``).

    Returns:
    crs (CRS or SimulatedCRS): the attached instrument.
    """
    if args.simulate:
        crs = simulated_crs(path, args.modules, seed = 200,
                            global_control = True)
    else:
        crs = CRS(serial_number = args.serial, interface = args.iface,
                  module_idxs = args.modules, global_control = True)
    await crs.attach(full_scale_dbm = 7)
    dl.check_crs_identity(root, crs)
    return crs

def point_runs(point, taken = ()):
    """
    Return the runs of a point: ``RUNS`` named after it.

    Parameters:
    point (dict): the point (``name``).
    taken (iterable of str): run names the tuning has already (a name
        is used once per tuning). Default none.

    Returns:
    runs (list of dict): ``stream_batch`` runs, named
        '<point>_dec<stage>', then '_2', '_3', ... when the name is
        taken or a stage repeats, with the point's name in their notes.
    """
    runs, used = [], set(taken)
    for r in RUNS:
        base = f"{point['name']}_dec{int(r['dec_stage'])}"
        name, k = base, 1
        while name in used:
            k += 1
            name = f'{base}_{k}'
        used.add(name)
        runs.append(dict(r, name = name, notes = point['name']
                         + (f"; {r['notes']}" if r.get('notes') else '')))
    return runs

def record_point(root, tune, result, point):
    """
    Save a point's name with its runs (attribute ``point``), and with
    its calibration sweep when the sweep was taken for it.

    Parameters:
    root (zarr.Group): the store root.
    tune (zarr.Group): the tuning.
    result (dict): ``stream_batch``'s result (``calibration``, ``runs``).
    point (dict): the point (``name``, optionally ``cal``).
    """
    if not point.get('cal'):
        root[result['calibration']].attrs['point'] = point['name']
    paths = {p for run in result['runs'] for p in run}
    for rec in dl.stream_runs(root, tune.basename):
        if paths & set(rec.get('paths') or []):
            dl.update_stream_run(tune, int(rec['run']),
                                 point = point['name'])

def calibration_of(point, results):
    """
    Return the calibration a point reuses: ``cal`` is the name of an
    earlier point of this sweep (its sweep) or a sweep's path.

    Parameters:
    point (dict): the point.
    results (dict): point name: ``stream_batch``'s result, so far.

    Returns:
    cal (str or None): the sweep's path, or None to take one.

    Raises:
    KeyError: if ``cal`` names a point not taken before this one.
    """
    cal = point.get('cal')
    if cal is None:
        return None
    if cal in results:
        return results[cal]['calibration']
    if '/' in str(cal):
        return str(cal)
    raise KeyError(f"point {point['name']!r} reuses the calibration of "
                   f'{cal!r}, which is not an earlier point of this sweep '
                   "(give a sweep's path, e.g. 'tune0/cal2', to reuse an "
                   'older one)')

async def take(crs, tune, args):
    """
    Take the sweep: every point from ``--from`` on. A point with ``cal``
    reuses that calibration (one NCO window only: with several, the
    sets are retuned at each point, so every point takes its own).

    Parameters:
    crs (CRS or SimulatedCRS): the attached instrument.
    tune (zarr.Group): the tuning.
    args (argparse.Namespace): options.

    Returns:
    results (dict): point name: ``stream_batch``'s result.

    Raises:
    KeyError: if ``--from`` names no point.
    """
    names = [p['name'] for p in POINTS]
    if args.start is not None and args.start not in names:
        raise KeyError(f'no point {args.start!r}; the points are '
                       f"{', '.join(names)}")
    first = 0 if args.start is None else names.index(args.start)
    root = dl.open_store(store_path(args), mode = 'a')
    results = {}
    for point in POINTS[first:]:
        print(f"Point {point['name']}")
        set_optical_power(point['name'])
        notes = (args.notes or '').strip()
        taken = [r['name'] for r in dl.stream_runs(root, tune.basename)]
        result = await stream_batch(
            crs, tune, TONE_SET, point_runs(point, taken),
            calibration = calibration_of(point, results),
            fres_update = FRES_UPDATE,
            tone_chunks = TONE_CHUNKS, status = print,
            notes = point['name'] + (f'; {notes}' if notes else ''),
            npoints_fine = NPOINTS_FINE, npoints_gain = NPOINTS_GAIN,
            nsamps = NSAMPS, gain_span_factor = GAIN_SPAN_FACTOR)
        record_point(root, tune, result, point)
        results[point['name']] = result
        print(f"{point['name']}: {result['calibration']}, "
              f"{sum(len(r) for r in result['runs'])} streams")
    return results

def main(args):
    """
    Connect, hold the board, and take the sweep.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    results (dict): ``take``'s result.
    """
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    tune = find_tuning(root, args.tuning)
    serial = SIM_SERIAL if args.simulate else args.serial
    owner = f'04c_power_sweep.py ({os.path.basename(path)}, {tune.basename})'
    with board_lock.exclusive(serial, owner = owner):
        async def run():
            """
            Connect and take the sweep on one event loop.

            Returns:
            results (dict): ``take``'s result.
            """
            crs = await connect(args, root, path)
            return await take(crs, tune, args)

        return asyncio.run(run())

if __name__ == '__main__':
    main(parse_args())
