"""
09: build one device's pages of the report site in OUT_DIR, with the
report figures and tables, and the top page listing every device built
so far. The store is STORE, --store, or chosen in a dialog. The store
browser opens first on its Notes tab (edit notes, attach files to the
report; close it to build). A store that has not changed since its last
build is left alone.

Usage:
    python 09_report_site.py
    python 09_report_site.py --store Z:/cd/deviceA_screen.zarr --no-gui
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.pipeline import site

################################################################################
################################### Settings ###################################
################################################################################
STORE = None
OUT_DIR = '~/reports'
RES_PAGES = True
INTERACTIVE = True

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
    p = argparse.ArgumentParser(description = 'Build the report site.')
    p.add_argument('--out', default = OUT_DIR, help = 'site folder')
    p.add_argument('--store', default = STORE,
                   help = 'the store (default: STORE, or a dialog)')
    p.add_argument('--force', action = 'store_true',
                   help = 'rebuild devices that have not changed')
    p.add_argument('--no-res-pages', action = 'store_true',
                   help = 'skip the resonator pages')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'skip the store browser')
    p.add_argument('--overwrite', action = 'store_true',
                   help = 'replace an existing report of the store, '
                   'attachments included, without asking')
    p.add_argument('--update', action = 'store_true',
                   help = 'rebuild an existing report keeping its '
                   'attachments and notes.md, without asking')
    p.add_argument('--index-only', action = 'store_true',
                   help = 'only rebuild the top page (e.g. after copying '
                   'a device folder in)')
    return p.parse_args(argv)

def existing_report(store, args):
    """
    Decide what to do when the store's report exists: replace it (its
    attachments too), rebuild it keeping the attachments and notes.md,
    or stop. ``--overwrite`` and ``--update`` decide; otherwise a popup
    asks, or without the window the run stops.

    Parameters:
    store (str): the store.
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    choice (str or None): 'overwrite', 'update', 'new' (no report yet),
        or None to stop.
    """
    root = dl.open_store(os.path.expanduser(store))
    out_dir = site.device_dir(args.out, root)
    if not os.path.exists(os.path.join(out_dir, site.MANIFEST)):
        return 'new'
    if args.overwrite:
        return 'overwrite'
    if args.update:
        return 'update'
    if INTERACTIVE and not args.no_gui:
        from citkid.daq_browser import ask_existing_report
        return ask_existing_report(out_dir)
    print(f'{out_dir} exists: run with --update to rebuild it keeping its '
          'attachments, or --overwrite to replace it')
    return None

def main(args):
    """
    Build the store's pages and the top page.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    built (list of str or None): the device folder, if it was built
        (empty when it was up to date or with ``--index-only``); None if
        no store was chosen.
    """
    if args.index_only:
        print(f'Site: {site.build_index(args.out)}')
        return []
    store = args.store
    if store is None:
        from citkid.daq_browser import choose_store
        store = choose_store()
        if store is None:
            print('No store chosen.')
            return None
    choice = existing_report(store, args)
    if choice is None:
        print('Cancelled: the report is left as it is.')
        return None
    if choice == 'overwrite':
        import shutil
        shutil.rmtree(site.device_dir(args.out, dl.open_store(
            os.path.expanduser(store))))
        args.force = True
    if INTERACTIVE and not args.no_gui:
        # Notes and attachments first; the store shown last is built
        from citkid.daq_browser import run_store_browser
        shown = run_store_browser(os.path.expanduser(store),
                                  title = 'Notes for the report',
                                  site_dir = os.path.expanduser(args.out),
                                  tab = 'Notes')
        store = shown or store
    out = site.build_device(os.path.expanduser(store), args.out,
                            force = args.force,
                            res_pages = RES_PAGES and not args.no_res_pages,
                            progress = print)
    index = site.build_index(args.out)
    print(f'Site: {index}')
    return [out] if out is not None else []

if __name__ == '__main__':
    main(parse_args())
