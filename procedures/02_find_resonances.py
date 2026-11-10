"""
02: find and fit the resonances of a VNA sweep, and save the resonator
list for tuning.

Steps, on ``vna{k}`` of the device store (the latest sweep by default):

1. find: the CNN finder (default; ``vna.res_finder_cnn``) or the
   interactive auto finder (``--finder auto``) writes ``fres_auto``;
2. manual: check and edit the list in the manual finder
   (``vna.res_finder_manual``), which writes ``fres_manual``;
3. fit: fit every listed resonance (``vna.full_sweep``), print a
   description of the sweep, and save ``full_sweep_fit`` with each
   resonance's flags and which parameters are reliable (``valid_fr``,
   ``valid_Qi``, ``valid_Qc``, ``valid_phi``; Qr and amp are as reliable
   as Qc);
4. resonators: save the list for tuning as ``resonators{k}``
   (``vna.full_sweep.resonator_list``): fitted fr, Qr where reliable, and
   span factors Qr / 5.

Steps already done are skipped, so the script picks up where it stopped;
``--redo`` repeats a step and the ones after it. A resonator list that a
tuning uses is never replaced.

No CRS is needed: this runs on the saved sweep.

Usage:
    python 02_find_resonances.py --device deviceA
    python 02_find_resonances.py --device deviceA --sweep 1 --finder auto
    python 02_find_resonances.py --device deviceA --redo fit

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.vna.full_sweep import resonator_list, run_full_sweep_fit

################################################################################
################################### Settings ###################################
################################################################################
DATA_DIR = '~/data'         # stores are in DATA_DIR/COOLDOWN/DEVICE.zarr
COOLDOWN = 'cooldown'       # cooldown name
DEVICE = 'device'           # device name
FINDER = 'cnn'              # 'cnn' (automatic) or 'auto' (interactive)
SPAN_LW = 5.                # target sweep spans, in linewidths (qres = Qr/5)

STEPS = ('find', 'manual', 'fit', 'resonators')

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
    p = argparse.ArgumentParser(description = 'Find and fit resonances.')
    p.add_argument('--data-dir', default = DATA_DIR)
    p.add_argument('--cooldown', default = COOLDOWN)
    p.add_argument('--device', default = DEVICE)
    p.add_argument('--store', default = None,
                   help = 'store path (instead of data dir/cooldown/device)')
    p.add_argument('--sweep', type = int, default = None,
                   help = 'VNA sweep index (default: the latest)')
    p.add_argument('--finder', default = FINDER, choices = ['cnn', 'auto'])
    p.add_argument('--redo', default = None, choices = STEPS,
                   help = 'repeat this step and the ones after it')
    p.add_argument('--no-manual', action = 'store_true',
                   help = 'skip the manual check (use the finder list '
                   'as it is; for simulations and quick looks)')
    p.add_argument('--span-lw', type = float, default = SPAN_LW)
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

def todo(step, done, args):
    """
    Decide whether to run a step.

    Parameters:
    step (str): one of ``STEPS``.
    done (bool): whether its output exists.
    args (argparse.Namespace): options (``redo``).

    Returns:
    run (bool): True if the step must run.
    """
    redo = args.redo is not None and \
        STEPS.index(step) >= STEPS.index(args.redo)
    if done and not redo:
        print(f'{step}: done already (--redo {step} repeats it)')
    return redo or not done

def find(f, z, grp, args):
    """
    Find candidate resonances (``fres_auto``).

    Parameters:
    f, z (np.array): the sweep.
    grp (zarr.Group): the ``vna{k}`` group.
    args (argparse.Namespace): options.
    """
    if args.finder == 'cnn':
        from citkid.vna.res_finder_cnn import run_res_finder_cnn
        run_res_finder_cnn(f, z, grp, overwrite = True)
    else:
        from citkid.vna.res_finder_auto import run_res_finder_auto
        run_res_finder_auto(f, z, grp)
    if 'res_finder_auto' in grp:
        grp['res_finder_auto'].attrs['kind'] = 'decision'

def manual(f, z, grp, args):
    """
    Check and edit the list (``fres_manual``).

    Parameters:
    f, z (np.array): the sweep.
    grp (zarr.Group): the ``vna{k}`` group.
    args (argparse.Namespace): options (``no_manual``).

    Returns:
    ok (bool): False if the user cancelled.
    """
    if args.no_manual:
        if 'fres_manual' in grp:
            del grp['fres_manual']
        dl.write_array(grp, 'fres_manual', grp['fres_auto'][:])
        grp.attrs['fres_manual_checked'] = False
        print(f"manual: skipped; fres_manual = fres_auto "
              f"({grp['fres_manual'].shape[0]} resonances)")
        return True
    from citkid.vna.res_finder_manual import run_res_finder_manual
    fres = run_res_finder_manual(f, z, grp, grp)
    if fres is None:
        print('manual: cancelled')
        return False
    grp.attrs['fres_manual_checked'] = True
    return True

def fit(f, z, grp, args):
    """
    Fit every listed resonance (``full_sweep_fit``).

    Parameters:
    f, z (np.array): the sweep.
    grp (zarr.Group): the ``vna{k}`` group.
    args (argparse.Namespace): options.
    """
    res, _ = run_full_sweep_fit(f, z, grp, downward = bool(
        grp.attrs.get('downward', True)), overwrite = True)
    out = grp['full_sweep_fit']
    out.attrs['kind'] = 'analysis'
    n = len(res)
    counts = {name: int(np.sum(out[name][:])) for name in
              ('valid_fr', 'valid_Qi', 'valid_Qc', 'valid_phi')}
    print(f'fit: {n} resonances; reliable: ' + ', '.join(
        f'{k[6:]} {v}' for k, v in counts.items()) +
        ' (Qr and amp as Qc)')

def resonators(root, grp, k, args):
    """
    Save the resonator list for tuning (``resonators{k}``).

    Parameters:
    root (zarr.Group): the store root.
    grp (zarr.Group): the ``vna{k}`` group.
    k (int): the sweep index.
    args (argparse.Namespace): options.

    Returns:
    ok (bool): False if the list is used by a tuning and can't change.
    """
    fres, qres, Qr = resonator_list(grp['full_sweep_fit'],
                                    span_lw = args.span_lw)
    name = f'resonators{k}'
    users = dl.tunings_using(root, name)
    if users:
        print(f'resonators: {name} is used by {users} and is kept; '
              'take a new VNA sweep for a new list')
        return False
    dl.save_resonators(root, k, fres, qres, f'vna{k}/full_sweep_fit',
                       Qr = Qr, overwrite = True)
    root[name].attrs['span_lw'] = float(args.span_lw)
    print(f'resonators: saved {len(fres)} as {name} '
          f'({int(np.isfinite(Qr).sum())} with a reliable Qr)')
    return True

def main(args):
    """
    Run the steps that aren't done yet.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    k (int or None): the sweep index, or None if stopped early.
    """
    path = store_path(args)
    root = dl.open_store(path, mode = 'a')
    dl.check_store(root)
    k = dl.latest_index(root, 'vna') if args.sweep is None else args.sweep
    grp = root[f'vna{k}']
    f, z = dl.load_vna_sweep(root, k)
    print(f'{path}: vna{k}, {len(f):,} points')
    if todo('find', 'fres_auto' in grp, args):
        find(f, z, grp, args)
    if todo('manual', 'fres_manual' in grp, args):
        if not manual(f, z, grp, args):
            return None
    if todo('fit', 'full_sweep_fit' in grp, args):
        fit(f, z, grp, args)
    if todo('resonators', f'resonators{k}' in root, args):
        if not resonators(root, grp, k, args):
            return None
    return k

if __name__ == '__main__':
    main(parse_args())
