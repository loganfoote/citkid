"""
Write a screening store's VNA sweep as an s21 h5 file (datasets f, i, q,
the format of Z:/s21_full/formatted, plus fres), choosing the store and
the output in file dialogs and, with several sweeps that have fitted
resonators, which sweep to use.

Usage:
    python export_s21_h5.py
    python export_s21_h5.py --store ~/data/cd1/deviceA.zarr --vna 0
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl

################################################################################
################################### Settings ###################################
################################################################################
STORE = None
OUT_DIR = None
VNA = None

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
        description = 'Write a VNA sweep of a store as an s21 h5 file.')
    p.add_argument('--store', default = STORE,
                   help = 'store path (default: a file dialog)')
    p.add_argument('--out', default = OUT_DIR,
                   help = 'output file or folder (default: a file dialog)')
    p.add_argument('--vna', type = int, default = VNA,
                   help = 'sweep index (default: asked when several have '
                   'fitted resonators)')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'no dialogs: --store and --out are needed')
    return p.parse_args(argv)

def sweeps_with_resonators(root):
    """
    List the VNA sweeps that have a fitted resonator list.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    vnas (list of int): sweep indices k with a ``resonators{k}``.
    """
    return [k for k in dl.list_indices(root, 'vna')
            if dl.set_name(k, dl.RESONATORS_PREFIX) in root]

def pick_output(default):
    """
    Choose the output file in a file dialog.

    Parameters:
    default (str): the file name offered.

    Returns:
    path (str or None): the file, or None if cancelled.
    """
    from citkid.qt_compat import get_qapp
    from pyqtgraph.Qt import QtWidgets
    get_qapp()
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        None, 'Save the s21 h5 file', default, 'HDF5 (*.h5)')
    return path or None

def pick_vna(root, vnas, gui):
    """
    Choose the sweep among those with fitted resonators.

    Parameters:
    root (zarr.Group): the store root.
    vnas (list of int): the candidates (several).
    gui (bool): If True, ask in a dialog, else in the terminal.

    Returns:
    k (int or None): the sweep index, or None if cancelled.
    """
    labels = []
    for k in vnas:
        a = root[dl.set_name(k, 'vna')].attrs
        n = root[dl.set_name(k, dl.RESONATORS_PREFIX)]['fres_all'].shape[0]
        labels.append(f"vna{k}: {a.get('created')}, {n} resonators"
                      + (f", {a['notes']}" if a.get('notes') else ''))
    if gui:
        from citkid.qt_compat import get_qapp
        from pyqtgraph.Qt import QtWidgets
        get_qapp()
        choice, ok = QtWidgets.QInputDialog.getItem(
            None, 'Which sweep?', 'Sweeps with fitted resonators:', labels,
            len(labels) - 1, False)
        return vnas[labels.index(choice)] if ok else None
    print('\n'.join(labels))
    answer = input('Sweep index: ').strip()
    return int(answer) if answer else None

def export(root, k, path):
    """
    Write one sweep and its resonators to an h5 file.

    Parameters:
    root (zarr.Group): the store root.
    k (int): the sweep index.
    path (str): the output file.

    Returns:
    n (int): resonators written.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError('export_s21_h5.py needs h5py: add it to the '
                          'environment (uv add h5py)') from exc
    f, z = dl.load_vna_sweep(root, k)
    res = dl.load_resonators(root, k)
    fres = np.sort(res['fres_all'])
    with h5py.File(path, 'w') as fh:
        fh.create_dataset('f', data = np.asarray(f, dtype = float))
        fh.create_dataset('i', data = np.asarray(z.real, dtype = float))
        fh.create_dataset('q', data = np.asarray(z.imag, dtype = float))
        fh.create_dataset('fres', data = fres)
        fh.attrs.update(dict(device = str(root.attrs.get('device', '')),
                             cooldown = str(root.attrs.get('cooldown', '')),
                             store = str(root.store), vna = int(k),
                             resonators = dl.set_name(k,
                                                      dl.RESONATORS_PREFIX)))
    return len(fres)

def main(args):
    """
    Choose the store, the sweep and the output, and write the file.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    path (str or None): the file written, or None if cancelled.
    """
    gui = not args.no_gui
    if args.store:
        store = args.store
    elif gui:
        from citkid.daq_browser import choose_store
        store = choose_store()
    else:
        store = None
    if not store:
        print('No store chosen.')
        return None
    root = dl.open_store(os.path.expanduser(store))
    vnas = sweeps_with_resonators(root)
    if not vnas:
        raise KeyError(f'{store} has no VNA sweep with fitted resonators: '
                       'run 02_find_resonances.py first')
    k = args.vna
    if k is None:
        k = vnas[0] if len(vnas) == 1 else pick_vna(root, vnas, gui)
        if k is None:
            print('No sweep chosen.')
            return None
    if k not in vnas:
        raise KeyError(f'vna{k} has no fitted resonators (sweeps with them: '
                       f'{vnas})')
    name = f"s21_{root.attrs.get('device', 'device')}.h5"
    out = args.out
    if out is None:
        out = pick_output(name) if gui else None
    elif os.path.isdir(os.path.expanduser(out)):
        out = os.path.join(out, name)
    if not out:
        print('No output chosen.')
        return None
    out = os.path.expanduser(out)
    n = export(root, k, out)
    print(f'Wrote {out}: vna{k}, {n} resonators')
    return out

if __name__ == '__main__':
    main(parse_args())
