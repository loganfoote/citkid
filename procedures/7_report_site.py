"""
7: build one device's pages of the report site in REPORT_DIR, with the
report figures and tables. The store is STORE, --store, or chosen in a
dialog. The site's top page is index.php, which the web server fills
from the device folders present (--index-only writes it). A dialog
asks who did the work (saved in the store, shown on the pages), then
the store browser opens on its Notes tab (edit notes, attach files to
the report; close it to build). A store that has not changed since its
last build is left alone.

Usage:
    python 7_report_site.py
    python 7_report_site.py --store Z:/cd/deviceA_screen.zarr --no-gui
    python 7_report_site.py --store <store> --by "LF DAQ + analysis" --fill
"""
import argparse
import os
from citkid import daq_layout as dl
from citkid.pipeline import site
from citkid.qt_compat import set_theme

################################################################################
################################### Settings ###################################
################################################################################
from settings import LIGHT_MODE

REPORT_DIR = '~/reports'
STORE = None
WORK_BY = None
RES_PAGES = True
SXX_RUN = None
INTERACTIVE = True
INDEX = False

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
    p.add_argument('--out', default = REPORT_DIR, help = 'site folder')
    p.add_argument('--store', default = STORE,
                   help = 'the store (default: STORE, or a dialog)')
    p.add_argument('--force', action = 'store_true',
                   help = 'rebuild a store that has not changed (an '
                   'existing report is rebuilt anyway)')
    p.add_argument('--no-res-pages', action = 'store_true',
                   help = 'skip the resonator pages')
    p.add_argument('--sxx-run', default = SXX_RUN,
                   help = 'the timestream run the Sxx values come from '
                   '(default: SXX_RUN, or the latest spectra)')
    p.add_argument('--no-gui', action = 'store_true',
                   help = 'skip the store browser')
    p.add_argument('--overwrite', action = 'store_true',
                   help = 'replace an existing report of the store, '
                   'attachments included, without asking')
    p.add_argument('--update', action = 'store_true',
                   help = 'rebuild an existing report keeping its '
                   'attachments and notes.md, without asking')
    p.add_argument('--by', default = WORK_BY,
                   help = "who did the work, e.g. 'LF DAQ + analysis' "
                   '(default: WORK_BY, the value saved in the store, '
                   'or a dialog)')
    p.add_argument('--fill', action = 'store_true',
                   help = 'only write --by into the store and its '
                   'existing report, without rebuilding the pages')
    p.add_argument('--index-only', action = 'store_true',
                   help = 'only write the top page (index.php) and '
                   'refresh the row each device folder carries')
    return p.parse_args(argv)

def existing_report(store, args):
    """
    Decide what to do when the store's report exists: replace it (its
    attachments too; the comments added on the served site stay),
    rebuild it keeping the attachments and notes.md,
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

def work_by(store, args):
    """
    Settle who did the work: ``--by`` (or ``WORK_BY``), else the value
    the store holds, offered in a dialog with the window; without the
    window the store's value is used as it is. The value is saved in
    the store when it changed.

    Parameters:
    store (str): the store.
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    text (str or None): who did the work; None to stop (Cancel, or
        without the window when the store has no value and ``--by``
        was not given).
    """
    root = dl.open_store(os.path.expanduser(store), mode = 'a')
    saved = str(root.attrs.get(dl.WORK_BY) or '')
    text = args.by
    if text is None:
        if INTERACTIVE and not args.no_gui:
            from citkid.daq_browser import ask_work_by
            text = ask_work_by(saved)
            if text is None:
                return None
        elif saved:
            text = saved
        else:
            print('Who did the work? Run with --by, e.g. '
                  '--by "LF DAQ + analysis"')
            return None
    text = text.strip()
    if text != saved:
        dl.set_work_by(root, text)
    return text

def main(args):
    """
    Build the store's pages, and the top page if ``INDEX``.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    built (list of str or None): the device folder, if it was built
        (empty when it was up to date or with ``--index-only``); None if
        no store was chosen.
    """
    set_theme(LIGHT_MODE)
    if args.index_only:
        print(f'Site: {site.build_index(args.out, refresh = True)}')
        return []
    store = args.store
    if store is None:
        from citkid.daq_browser import choose_store
        store = choose_store()
        if store is None:
            print('No store chosen.')
            return None
    if work_by(store, args) is None:
        print('Cancelled: nothing built.')
        return None
    if args.fill:
        root = dl.open_store(os.path.expanduser(store))
        out_dir = site.device_dir(args.out, root)
        try:
            out = site.fill_work_by(out_dir, root.attrs.get(dl.WORK_BY, ''))
        except FileNotFoundError:
            print(f'{out_dir} holds no report yet: saved in the store only')
            return []
        if INDEX:
            print(f'Site: {site.build_index(args.out)}')
        return [out]
    choice = existing_report(store, args)
    if choice is None:
        print('Cancelled: the report is left as it is.')
        return None
    if choice == 'overwrite':
        # Everything the build made goes; the site's own comments stay
        site.clear_device_dir(site.device_dir(args.out, dl.open_store(
            os.path.expanduser(store))))
    if choice in ('overwrite', 'update'):
        # The user asked for the report again: rebuild it even if the
        # store has not changed
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
                            progress = print, sxx_run = args.sxx_run)
    if INDEX:
        print(f'Site: {site.build_index(args.out)}')
    return [out] if out is not None else []

if __name__ == '__main__':
    main(parse_args())
