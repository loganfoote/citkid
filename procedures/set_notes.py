"""
Add or change the notes of a store or of one of its groups (a run's
tuning, a photon result, a PSD set, ...); the report site shows them.
With no text, lists the notes the store has.

Usage:
    python set_notes.py --store Z:/cd/deviceA_screen.zarr
    python set_notes.py --store Z:/cd/deviceA_screen.zarr --path \
        photons/dark "laser warmed up during this run"
"""
import argparse
import os
from citkid import daq_layout as dl

################################################################################
################################### Settings ###################################
################################################################################
STORE = ''

################################################################################
################################### Script #####################################
################################################################################
def parse_args(argv = None):
    """
    Parse the command-line options.

    Parameters:
    argv (list of str or None): arguments. None (default): sys.argv.

    Returns:
    args (argparse.Namespace): the options.
    """
    p = argparse.ArgumentParser(description = 'Notes on a store or group.')
    p.add_argument('--store', default = STORE, help = 'store path')
    p.add_argument('--path', default = '',
                   help = "group, e.g. photons/dark (default: the store)")
    p.add_argument('text', nargs = '?', default = None,
                   help = 'the notes (blank text removes them)')
    return p.parse_args(argv)

def main(args):
    """
    Set or list the notes.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    notes (list of tuple): (path, text), the store's notes afterwards.
    """
    if not args.store:
        raise ValueError('give the store: --store path, or STORE')
    path = os.path.expanduser(args.store)
    if args.text is not None:
        root = dl.open_store(path, mode = 'a')
        dl.set_notes(root, args.path, args.text)
    root = dl.open_store(path)
    notes = dl.list_notes(root)
    for group, text in notes:
        print(f"{group or '(store)'}: {text}")
    if not notes:
        print('No notes')
    return notes

if __name__ == '__main__':
    main(parse_args())
