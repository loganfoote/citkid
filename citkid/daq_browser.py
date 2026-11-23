"""
Interactive browser of a device store.

``run_store_browser`` opens a window on a store (``citkid.daq_layout``).
The left side lists what the procedures made, one tab per kind (VNA
sweeps, resonator lists, tunings, calibration sweeps, timestream runs,
stream calibrations, photon and HEE results, spectra, NEP sweeps;
``daq_layout.store_inventory``), each a table filtered by typing, and a
last tab with the store's own tree of groups and arrays for when the
layout matters. The right side is about the selected item, whichever
tab it was chosen in: a summary (what it is, when, its notes, what it
was made from and what depends on it, as links), its attributes (or an
array's shape and first values), the tone marks, and the notes editor
with Rename and Delete. Double-click, or Open in store tree, shows the
item's group in the tree; Reveal folder opens its directory in the file
browser; a right-click gives the same actions. Open (Ctrl+O) picks
another store. The store is opened read-only for browsing (Refresh, F5,
rereads it); notes, marks, names and deletions are written through a
writable handle. ``procedures/store_browser.py`` is the script.
"""
import html
import json
import os
import subprocess
import sys
import numpy as np
import zarr
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from . import daq_layout as dl
from .qt_compat import add_help
from .qt_compat import (
    Qt, delete_on_close, fit_window_to_screen, get_qapp, hint_label,
    widget_font_stylesheet
)

# Attributes shown first, in this order; the rest follow alphabetically
FIRST_ATTRS = ('name', 'notes', 'kind', 'created', 'tone_set', 'tone_sets',
               'calibration', 'calibrations', 'dec_stage', 'duration_s',
               'complete')
# Longest attribute value shown in full (longer ones are cut, e.g. the
# pipeline YAML saved with a stream calibration)
MAX_VALUE_CHARS = 4000
# Values shown of a selected array
PREVIEW_VALUES = 8
# Columns of the runs table: (title, key of ``daq_layout.stream_runs``)
RUN_COLUMNS = (('Run', 'group'), ('Name', 'name'), ('Notes', 'notes'),
               ('Stage', 'dec_stage'), ('Length (s)', 'duration_s'),
               ('Tone set', 'tone_set'), ('Calibration', 'calibration'),
               ('Tuning', 'tuning'), ('Chunks', 'n_chunks'),
               ('Created', 'created'))

# Columns of the kind tables (all kinds but the runs)
KIND_COLUMNS = ('Item', 'Name', 'Created', 'Made from', 'Notes')

################################################################################
################################### Helpers ####################################
################################################################################
# The tone-marks tab's columns
MARK_COLUMNS = ('res', 'created', 'kind', 'scope', 'stage', 'tuning',
                'text')

def first_line(text, limit = 80):
    """
    Return the first line of a text, shortened.

    Parameters:
    text: the text (anything; None and '' give '').
    limit (int): longest result. Default 80.

    Returns:
    line (str): the first non-blank line, with '...' if cut.
    """
    if text is None:
        return ''
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ''
    line = lines[0] if len(lines) == 1 else lines[0] + ' ...'
    return line if len(line) <= limit else line[:limit - 3] + '...'

def summarize(item):
    """
    Describe a group or array in one line for the tree.

    Parameters:
    item (zarr.Group or zarr.Array): the item.

    Returns:
    text (str): for an array, its shape and dtype; for a group, its
        name and notes if it has them, else what its attributes say
        (tone set, calibration, stage and length, number of resonators
        or tones, ...).
    """
    if isinstance(item, zarr.Array):
        return f'{tuple(item.shape)} {item.dtype}'
    a = item.attrs
    parts = []
    if a.get('name') and str(a['name']) != item.basename:
        parts.append(str(a['name']))
    if 'tone_set' in a:
        parts.append(f"set {a['tone_set']}")
    elif a.get('tone_sets') is not None:
        parts.append(f"sets {list(a['tone_sets'])}")
    if 'dec_stage' in a:
        parts.append(f"stage {a['dec_stage']}")
    if a.get('duration_s') is not None:
        parts.append(f"{a['duration_s']:g} s")
    if 'amplitude' in a:
        parts.append(f"{a['amplitude']:g} dBm")
    for key, what in (('res_idxs_all', 'resonators'), ('fres', 'tones'),
                      ('tones/fres', 'tones')):
        try:
            n = item[key].shape[0]
        except (KeyError, AttributeError):
            continue
        parts.append(f'{n} {what}')
        break
    if a.get('resonators'):
        parts.append(str(a['resonators']))
    if a.get('selection') is not None:
        parts.append(f"{len(a['selection'])} selected")
    if a.get('complete') is False:
        parts.append('incomplete')
    if a.get('notes'):
        parts.append(first_line(a['notes'], 60))
    return ', '.join(parts)

def format_value(value):
    """
    Format an attribute value for the attributes panel.

    Parameters:
    value: the value (JSON-like).

    Returns:
    text (str): multi-line strings as they are, dicts and lists as
        indented JSON, the rest as JSON; cut at ``MAX_VALUE_CHARS``.
    """
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent = 2, default = str)
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS] + \
            f'\n... ({len(text) - MAX_VALUE_CHARS:,} more characters)'
    return text

def describe_item(item):
    """
    Describe a group or array for the attributes panel.

    Parameters:
    item (zarr.Group or zarr.Array): the item.

    Returns:
    text (str): its path; for an array, its shape, dtype, chunks, and
        first values; then its attributes, ``FIRST_ATTRS`` first.
    """
    lines = [item.path or '/', '']
    if isinstance(item, zarr.Array):
        lines += [f'shape: {tuple(item.shape)}', f'dtype: {item.dtype}',
                  f'chunks: {tuple(item.chunks)}']
        if item.size and item.size <= 10 ** 7:
            flat = np.asarray(item[...]).ravel()[:PREVIEW_VALUES]
            more = ' ...' if item.size > PREVIEW_VALUES else ''
            lines.append(f'values: {flat.tolist()}{more}')
        lines.append('')
    attrs = dict(item.attrs)
    keys = [k for k in FIRST_ATTRS if k in attrs] + \
        sorted(k for k in attrs if k not in FIRST_ATTRS)
    if not keys:
        lines.append('(no attributes)')
    for k in keys:
        text = format_value(attrs[k])
        if '\n' in text:
            lines.append(f'{k}:')
            lines += ['    ' + ln for ln in text.splitlines()]
        else:
            lines.append(f'{k}: {text}')
    return '\n'.join(lines)

def store_label(root, path = None):
    """
    Describe a store in one line.

    Parameters:
    root (zarr.Group): the store root.
    path (str or None): its path, if known. Default None.

    Returns:
    text (str): path, device, cooldown, creation time, citkid version.
    """
    a = root.attrs
    parts = [str(path) if path else a.get('device', 'store')]
    for key in ('device', 'cooldown', 'created'):
        if a.get(key):
            parts.append(f"{key} {a[key]}")
    if a.get('citkid_version'):
        parts.append(f"citkid {a['citkid_version']}")
    return '; '.join(parts)

def open_folder(folder):
    """
    Open a directory in the system's file browser.

    Parameters:
    folder (str): the directory.
    """
    if sys.platform == 'win32':
        os.startfile(folder)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', folder])
    else:
        subprocess.Popen(['xdg-open', folder])

################################################################################
#################################### Window ####################################
################################################################################
class StoreBrowserWindow(QtWidgets.QMainWindow):
    """
    Window browsing a store: the items by kind (and the store's tree)
    on the left, the selected item (summary, attributes, tone marks,
    notes) on the right.

    Attributes:
    root (zarr.Group or None): the store root (read-only); None while no
        store is open.
    path (str or None): the store's path, if known.
    kinds (QTabWidget): the left tabs: one table per kind
        (``kind_tables``), the runs table, and the store tree.
    kind_tables (dict): kind title: QTableWidget.
    items (dict): ``daq_layout.store_inventory`` of the store shown.
    tree (QTreeWidget): the store's groups and arrays (children are read
        when a group is expanded).
    summary (QTextBrowser): the selected item's summary, with links to
        the items it was made from and those that depend on it.
    attrs (QPlainTextEdit): the selected item's description.
    runs_table (QTableWidget): the timestream runs.
    runs (list of dict): the runs shown (``daq_layout.stream_runs``).
    filter_edit (QLineEdit): text the shown runs must contain.
    """
    # The path of the store shown (or '') when the window is closed
    closed = QtCore.Signal(str)

    def __init__(self, root = None, path = None, title = 'Store browser',
                 ui_scale = 1.0, site_dir = None):
        """
        Build the window, on a store if given.

        Parameters:
        root (zarr.Group, str, os.PathLike, or None): store root or path
            (opened read-only). Default None: no store until one is
            opened (``open_path``, or the Open button).
        path (str or None): the store's path, for the header. Default
            None: from ``root`` if it is a path.
        title (str): window title. Default 'Store browser'.
        ui_scale (float): font size multiplier. Default 1.0.
        site_dir (str or None): the report site's root folder: the Notes
            tab can then attach files to an item (copied into the site's
            device folder, never into the store). Default None: no
            attaching.
        """
        super().__init__()
        self.site_dir = site_dir
        self.root, self.path = None, None
        self.runs, self._shown_runs = [], []
        self.items = {}
        self._selected = None       # path of the item shown on the right
        self._selecting = False
        self._title = title
        self.setWindowTitle(title)
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.info)
        split = QtWidgets.QSplitter()
        split.addWidget(self._build_kinds())
        split.addWidget(self._build_tabs())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton('Open... (Ctrl+O)')
        self.open_btn.setToolTip('Choose a store (.zarr directory) in a '
                                 'file dialog')
        self.open_btn.clicked.connect(self.pick_store)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self,
                            self.pick_store)
        row.addWidget(self.open_btn)
        self.refresh_btn = QtWidgets.QPushButton('Refresh (F5)')
        self.refresh_btn.clicked.connect(self.refresh)
        QtWidgets.QShortcut(QtGui.QKeySequence('F5'), self, self.refresh)
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(hint_label(
            'Pick a kind on the left and click an item: the right side '
            'shows it. Double-click shows its group in the store tree; '
            'right-click for Rename, Delete and Reveal folder. F5 rereads '
            'the store, Ctrl+O opens another.'))
        self.setCentralWidget(central)
        add_help(self, 'Store browser', [
            ('Buttons', [
                ('Open... (Ctrl+O)', 'Choose a store'),
                ('Refresh (F5)', 'Reread the store (another script may be '
                 'writing it)')]),
            ('Left tabs', [
                ('One per kind', 'What the procedures made: VNA sweeps, '
                 'resonator lists, tunings, calibration sweeps, runs, '
                 'stream calibrations, photons, HEEs, spectra, NEP; type '
                 'to filter; double-click to show the item in the tree'),
                ('Store', "The store's groups and arrays; expand a group "
                 'to read its contents')]),
            ('Right tabs', [
                ('Summary', 'What the selected item is, its notes, what '
                 'it was made from and what depends on it (links); Open '
                 'in store tree and Reveal folder'),
                ('Attributes', "The selected item's attributes, as saved"),
                ('Tone marks', 'The bad marks and notes on tones: clear one, '
                 'or '
                 'make it global (later stages too) or local (its stage only)'),
                ('Notes', 'Edit the notes of the store, a step, a result or a '
                 'run (saved by themselves); Rename changes the name of a '
                 'tuning, run, PSD group or result; Delete removes an item and '
                 'what depends on it, after asking; for a report, the Report '
                 'column leaves unchecked items out of it, and Attach file '
                 'copies a file into the report next to the item')]),
            ('Keyboard', [('H', 'Show / hide this help')]),
        ])
        if root is not None:
            self.open_path(root, path)
        else:
            self.refresh()

    def _build_kinds(self):
        """
        Build the left tabs: a table per kind of item, the runs table,
        and the store tree.

        Returns:
        kinds (QTabWidget): the tabs.
        """
        self.kinds = QtWidgets.QTabWidget()
        self.kinds.setUsesScrollButtons(True)
        self.kind_tables, self.kind_filters, self.kind_labels = {}, {}, {}
        self._kind_rows = {}        # kind: entries shown, by row
        for title, _ in dl.ITEM_KINDS:
            if title == 'Runs':
                self.kinds.addTab(self._build_runs(), title)
                continue
            page = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(page)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(f'Filter the {title.lower()}')
            edit.setClearButtonEnabled(True)
            edit.textChanged.connect(
                lambda _t, title = title: self._fill_kind(title))
            vbox.addWidget(edit)
            table = QtWidgets.QTableWidget()
            table.setColumnCount(len(KIND_COLUMNS))
            table.setHorizontalHeaderLabels(list(KIND_COLUMNS))
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(
                QtWidgets.QAbstractItemView.SelectRows)
            table.setSelectionMode(
                QtWidgets.QAbstractItemView.SingleSelection)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setStretchLastSection(True)
            table.currentCellChanged.connect(
                lambda row, _c, _pr, _pc, title = title:
                self._on_kind_row(title, row))
            table.cellDoubleClicked.connect(
                lambda row, _c, title = title: self._open_in_tree())
            table.setContextMenuPolicy(Qt.CustomContextMenu)
            table.customContextMenuRequested.connect(
                lambda pos, table = table: self._context_menu(table, pos))
            vbox.addWidget(table, stretch = 1)
            label = QtWidgets.QLabel()
            vbox.addWidget(label)
            self.kind_tables[title] = table
            self.kind_filters[title] = edit
            self.kind_labels[title] = label
            self.kinds.addTab(page, title)
        self.kinds.addTab(self._build_tree(), 'Store')
        self.kinds.currentChanged.connect(lambda _i: self.flush_notes())
        return self.kinds

    def _build_runs(self):
        """
        Build the runs tab: the table of every timestream run, filtered
        by typing.

        Returns:
        page (QWidget): the tab.
        """
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText(
            'Filter runs by name, notes, calibration, ...')
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._fill_runs)
        vbox.addWidget(self.filter_edit)
        self.runs_table = QtWidgets.QTableWidget()
        self.runs_table.setColumnCount(len(RUN_COLUMNS))
        self.runs_table.setHorizontalHeaderLabels(
            [title for title, _ in RUN_COLUMNS])
        self.runs_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.runs_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.runs_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.cellDoubleClicked.connect(self._on_run_activated)
        self.runs_table.currentCellChanged.connect(self._on_run_row)
        self.runs_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.runs_table.customContextMenuRequested.connect(
            lambda pos: self._context_menu(self.runs_table, pos))
        vbox.addWidget(self.runs_table, stretch = 1)
        self.runs_label = QtWidgets.QLabel()
        vbox.addWidget(self.runs_label)
        return page

    def _build_tree(self):
        """
        Build the tree of groups and arrays.

        Returns:
        tree (QTreeWidget): the tree.
        """
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['Name', 'Kind', 'Summary'])
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 70)
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(
            lambda pos: self._context_menu(self.tree, pos))
        return self.tree

    def _build_summary(self):
        """
        Build the summary tab: the selected item's description with
        links, and the buttons that act on it.

        Returns:
        page (QWidget): the tab.
        """
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        self.summary = QtWidgets.QTextBrowser()
        self.summary.setOpenLinks(False)
        self.summary.anchorClicked.connect(self._on_anchor)
        vbox.addWidget(self.summary, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.tree_btn = QtWidgets.QPushButton('Open in store tree')
        self.tree_btn.setToolTip("Show the item's group in the Store tab")
        self.tree_btn.clicked.connect(self._open_in_tree)
        self.reveal_btn = QtWidgets.QPushButton('Reveal folder')
        self.reveal_btn.setToolTip("Open the item's directory in the "
                                   'file browser')
        self.reveal_btn.clicked.connect(self.reveal_selected)
        self.summary_rename_btn = QtWidgets.QPushButton('Rename...')
        self.summary_rename_btn.clicked.connect(self._on_rename)
        self.summary_delete_btn = QtWidgets.QPushButton('Delete...')
        self.summary_delete_btn.clicked.connect(
            lambda: self.delete_selected())
        for b in (self.tree_btn, self.reveal_btn, self.summary_rename_btn,
                  self.summary_delete_btn):
            b.setEnabled(False)
            row.addWidget(b)
        row.addStretch(1)
        vbox.addLayout(row)
        return page

    def _build_tabs(self):
        """
        Build the attributes panel and the runs table.

        Returns:
        tabs (QTabWidget): the tabs.
        """
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_summary(), 'Summary')
        self.attrs = QtWidgets.QPlainTextEdit()
        self.attrs.setReadOnly(True)
        self.attrs.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.attrs.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont))
        self.tabs.addTab(self.attrs, 'Attributes')
        self.tabs.addTab(self._build_marks(), 'Tone marks')
        self.tabs.addTab(self._build_notes(), 'Notes')
        return self.tabs

    def _build_marks(self):
        """
        Build the tone-marks panel: every entry of the registry, and
        buttons to clear the selected bad mark or change its scope
        (local: that stage's results only; global: every later step of
        the tuning too).

        Returns:
        page (QWidget): the panel.
        """
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        self.marks_table = QtWidgets.QTableWidget()
        self.marks_table.setColumnCount(len(MARK_COLUMNS))
        self.marks_table.setHorizontalHeaderLabels(MARK_COLUMNS)
        self.marks_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.marks_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.marks_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.marks_table.verticalHeader().setVisible(False)
        self.marks_table.horizontalHeader().setStretchLastSection(True)
        self.marks_table.currentCellChanged.connect(
            lambda *_a: self._update_mark_buttons())
        vbox.addWidget(self.marks_table, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.marks_label = QtWidgets.QLabel()
        row.addWidget(self.marks_label, stretch = 1)
        self.clear_mark_btn = QtWidgets.QPushButton('Clear mark')
        self.clear_mark_btn.setToolTip('Clear the selected bad mark (a '
                                       "'clear' entry is added; the "
                                       'history stays)')
        self.clear_mark_btn.clicked.connect(self._on_clear_mark)
        row.addWidget(self.clear_mark_btn)
        self.global_mark_btn = QtWidgets.QPushButton('Make global')
        self.global_mark_btn.setToolTip('The tone also leaves every '
                                        'later step of its tuning')
        self.global_mark_btn.clicked.connect(
            lambda: self._on_rescope('global'))
        row.addWidget(self.global_mark_btn)
        self.local_mark_btn = QtWidgets.QPushButton('Make local')
        self.local_mark_btn.setToolTip("The tone leaves only its stage's "
                                       'results')
        self.local_mark_btn.clicked.connect(
            lambda: self._on_rescope('local'))
        row.addWidget(self.local_mark_btn)
        vbox.addLayout(row)
        self._mark_rows = []       # (res_idx, entry, active) per row
        self._update_mark_buttons()
        return page

    def _fill_marks(self):
        """
        List every registry entry, newest last, saying which bad marks
        are active (not cleared since), keeping the selected row.
        """
        current = self.marks_table.currentRow()
        self._mark_rows = []
        if self.root is not None:
            for r, entries in sorted(dl.tone_marks(self.root).items()):
                active = {e['created'] + e.get('stage', '') for e in
                          dl.active_bad_marks(self.root, r)}
                for e in entries:
                    self._mark_rows.append(
                        (r, e, e['kind'] == 'bad' and
                         e['created'] + e.get('stage', '') in active))
        self.marks_table.setRowCount(len(self._mark_rows))
        n_bad = 0
        for i, (r, e, active) in enumerate(self._mark_rows):
            n_bad += active
            scope = '' if e['kind'] == 'note' else dl.mark_scope(e)
            cells = (str(r), e.get('created', ''), e['kind'] + (
                ' (active)' if active else ''), scope, e.get('stage', ''),
                e.get('tuning') or '', first_line(e.get('text', ''), 120))
            for j, text in enumerate(cells):
                self.marks_table.setItem(i, j, QtWidgets.QTableWidgetItem(
                    text))
        self.marks_table.resizeColumnsToContents()
        self.marks_label.setText(
            f'{n_bad} active bad marks, {len(self._mark_rows)} entries')
        if 0 <= current < len(self._mark_rows):
            self.marks_table.setCurrentCell(current, 0)
        self._update_mark_buttons()

    def _selected_mark(self):
        """
        Return the selected active bad mark, if any.

        Returns:
        res_idx (int or None): the resonator.
        entry (dict or None): the entry.
        """
        i = self.marks_table.currentRow()
        if 0 <= i < len(self._mark_rows) and self._mark_rows[i][2]:
            return self._mark_rows[i][0], self._mark_rows[i][1]
        return None, None

    def _update_mark_buttons(self):
        """
        Enable the buttons that apply to the selected mark.
        """
        r, e = self._selected_mark()
        can = e is not None and bool(self.path)
        scope = dl.mark_scope(e) if e is not None else ''
        self.clear_mark_btn.setEnabled(can)
        self.global_mark_btn.setEnabled(can and scope == 'local')
        self.local_mark_btn.setEnabled(can and scope == 'global')

    def _writable_root(self):
        """
        Open the store for writing a mark.

        Returns:
        root (zarr.Group): the store root, writable.
        """
        return dl.open_store(self.path, mode = 'a')

    def _on_clear_mark(self):
        """
        Clear the selected bad mark (the button).
        """
        r, e = self._selected_mark()
        if e is None:
            return
        dl.clear_tone_mark(self._writable_root(), r, 'cleared in the store '
                           'browser', e.get('stage', ''), dl.mark_scope(e))
        self.refresh()

    def _on_rescope(self, scope):
        """
        Change the selected bad mark's scope (the buttons).

        Parameters:
        scope (str): 'global' or 'local'.
        """
        r, e = self._selected_mark()
        if e is None:
            return
        dl.rescope_tone_mark(self._writable_root(), r, e['created'], scope)
        self.refresh()

    def _build_notes(self):
        """
        Build the notes panel: everything notes can be kept on (the
        store, every step and result, every run) with an editor for the
        selected row. Edits are saved by themselves: on a row change, a
        tab change, a refresh, and when the window closes.

        Returns:
        page (QWidget): the panel.
        """
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        self.notes_table = QtWidgets.QTableWidget()
        # For a report (site_dir), a first column of boxes chooses what
        # the report shows in full; the rest is one line each
        self._report_col = 0 if self.site_dir is not None else None
        self._col0 = 1 if self.site_dir is not None else 0
        headers = (['Report'] if self.site_dir is not None else []) + \
            ['Item', 'Kind', 'Notes']
        self.notes_table.setColumnCount(len(headers))
        self.notes_table.setHorizontalHeaderLabels(headers)
        self.notes_table.itemChanged.connect(self._on_report_toggled)
        self.notes_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.notes_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.notes_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.notes_table.verticalHeader().setVisible(False)
        self.notes_table.horizontalHeader().setStretchLastSection(True)
        self.notes_table.currentCellChanged.connect(self._on_notes_row)
        vbox.addWidget(self.notes_table, stretch = 1)
        row = QtWidgets.QHBoxLayout()
        self.notes_label = QtWidgets.QLabel()
        row.addWidget(self.notes_label, stretch = 1)
        self.rename_btn = QtWidgets.QPushButton('Rename...')
        self.rename_btn.setToolTip('Change the name of the selected '
                                   'tuning, run, PSD group or result')
        self.rename_btn.clicked.connect(self._on_rename)
        self.rename_btn.setEnabled(False)
        row.addWidget(self.rename_btn)
        self.delete_btn = QtWidgets.QPushButton('Delete...')
        self.delete_btn.setToolTip('Delete the selected item from the '
                                   'store, with what depends on it, '
                                   'after asking')
        self.delete_btn.clicked.connect(lambda: self.delete_selected())
        self.delete_btn.setEnabled(False)
        row.addWidget(self.delete_btn)
        self.attach_btn = QtWidgets.QPushButton('Attach file...')
        self.attach_btn.setToolTip('Copy a file into the report site next '
                                   "to this item's notes (not into the "
                                   'store)')
        self.attach_btn.clicked.connect(self._on_attach)
        self.attach_btn.setEnabled(False)
        self.attach_btn.setVisible(self.site_dir is not None)
        row.addWidget(self.attach_btn)
        self.detach_btn = QtWidgets.QPushButton('Remove file...')
        self.detach_btn.setToolTip("Remove one of this item's attached "
                                   'files from the report site')
        self.detach_btn.clicked.connect(self._on_detach)
        self.detach_btn.setEnabled(False)
        self.detach_btn.setVisible(self.site_dir is not None)
        row.addWidget(self.detach_btn)
        vbox.addLayout(row)
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setFixedHeight(110)
        self.notes_edit.setPlaceholderText('Notes of the selected item')
        self.notes_edit.textChanged.connect(self._on_notes_edited)
        vbox.addWidget(self.notes_edit)
        self._notes = []            # dl.notes_entries of the store shown
        self._notes_row = None      # the row the editor holds
        self._notes_dirty = {}      # row: edited text, not saved yet
        self._notes_loading = False
        self.tabs.currentChanged.connect(lambda _i: self.flush_notes())
        return page

    def _fill_notes(self):
        """
        List everything notes can be kept on, keeping the selected row.
        """
        self._notes_loading = True
        current = self._notes_row
        self._notes = dl.notes_entries(self.root) if self.root is not None \
            else []
        self.notes_table.setRowCount(len(self._notes))
        for i, e in enumerate(self._notes):
            if self._report_col is not None:
                box = QtWidgets.QTableWidgetItem()
                box.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled |
                             Qt.ItemIsSelectable)
                box.setCheckState(Qt.Checked if e['in_report']
                                  else Qt.Unchecked)
                self.notes_table.setItem(i, self._report_col, box)
            for j, text in enumerate((e['label'], e['kind'],
                                      first_line(e['notes'], 120))):
                self.notes_table.setItem(
                    i, j + self._col0, QtWidgets.QTableWidgetItem(text))
        for j in range(self._col0 + 2):
            self.notes_table.resizeColumnToContents(j)
        self._notes_dirty = {}
        self._notes_row = None
        writable = bool(self.path) and not str(self.path).endswith('.zip')
        self.notes_edit.setReadOnly(not writable)
        if current is not None and current < len(self._notes):
            self.notes_table.setCurrentCell(current, 0)
        self._notes_loading = False
        self._on_notes_row(self.notes_table.currentRow(), 0, -1, -1)

    def _on_notes_row(self, row, _column = 0, _prev_row = -1, _prev_col = -1):
        """
        Put the selected item's notes in the editor, saving an edit of
        the row before.

        Parameters:
        row (int): the table row (-1: none).
        _column, _prev_row, _prev_col (int): unused.
        """
        if self._notes_loading:
            return
        self.flush_notes(refresh = False)
        self._notes_loading = True
        if 0 <= row < len(self._notes):
            e = self._notes[row]
            self._notes_row = row
            self.notes_edit.setPlainText(e['notes'])
            what = 'the store' if e['kind'] == 'store' else e['label']
            text = f'Notes of {what}' + (
                '' if not self.notes_edit.isReadOnly()
                else ' (read-only store)')
            files = self.attachments(e['path'])
            if files:
                text += '; attached: ' + ', '.join(files)
            self.notes_label.setText(text)
            if not self._selecting and e['kind'] != 'store':
                self._select_item(e['path'], notes_row = False)
        else:
            self._notes_row = None
            self.notes_edit.setPlainText('')
            self.notes_label.setText('Select an item to see or edit its '
                                     'notes')
        self.attach_btn.setEnabled(self.site_dir is not None and
                                   self._notes_row is not None and
                                   self.root is not None)
        self._update_detach()
        self._update_item_buttons()
        self._notes_loading = False

    def _update_item_buttons(self):
        """
        Enable Rename for a selected item that has a name and Delete for
        any selected item but the store (a writable store).
        """
        e = self._notes[self._notes_row] if self._notes_row is not None \
            else None
        writable = e is not None and not self.notes_edit.isReadOnly()
        self.rename_btn.setEnabled(writable and e.get('name') is not None)
        self.delete_btn.setEnabled(writable and e['kind'] != 'store')
        self.summary_rename_btn.setEnabled(self.rename_btn.isEnabled())
        self.summary_delete_btn.setEnabled(self.delete_btn.isEnabled())
        here = e is not None and e['kind'] != 'store'
        self.tree_btn.setEnabled(here)
        self.reveal_btn.setEnabled(here and bool(self.path) and
                                   not str(self.path).endswith('.zip'))

    def _on_report_toggled(self, item):
        """
        Save a change of an item's Report box (the report shows an
        unchecked item as one line).

        Parameters:
        item (QTableWidgetItem): the box.
        """
        if self._notes_loading or self._report_col is None or \
                item.column() != self._report_col or not self.path:
            return
        e = self._notes[item.row()]
        flag = item.checkState() == Qt.Checked
        if flag == e['in_report']:
            return
        dl.set_in_report(self._writable_root(), e['path'], flag,
                         run = e['run'])
        e['in_report'] = flag

    def _on_rename(self):
        """
        Ask a new name for the selected item (the Rename button).
        """
        e = self._notes[self._notes_row] if self._notes_row is not None \
            else None
        if e is None or e.get('name') is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename', f"New name of {e['label']}:",
            QtWidgets.QLineEdit.Normal, e['name'])
        if ok and name.strip() and name.strip() != e['name']:
            self.rename_selected(name.strip())

    def rename_selected(self, name):
        """
        Rename the selected item (``daq_layout.rename_item``) and reread
        the store. An error is shown in a box.

        Parameters:
        name (str): the new name.

        Returns:
        path (str or None): the item's path afterwards; None on an error.
        """
        e = self._notes[self._notes_row] if self._notes_row is not None \
            else None
        if e is None or not self.path:
            return None
        self.flush_notes(refresh = False)
        try:
            path = dl.rename_item(self._writable_root(), e['path'], name,
                                  run = e['run'])
        except (ValueError, OSError, KeyError) as err:
            QtWidgets.QMessageBox.warning(self, 'Rename', str(err))
            return None
        self.refresh()
        return path

    def delete_selected(self, ask = True):
        """
        Delete the selected item from the store, with what depends on
        it (``daq_layout.delete_item``), after asking once, and again
        when something depends on it.

        Parameters:
        ask (bool): If True (default), ask first.

        Returns:
        deleted (list of str): the paths deleted (none if refused).
        """
        e = self._notes[self._notes_row] if self._notes_row is not None \
            else None
        if e is None or e['kind'] == 'store' or not self.path:
            return []
        root = self._writable_root()
        if e['run'] is not None:
            rec = root[e['run'][0]][dl.STREAMS_GROUP].attrs['runs'][
                int(e['run'][1])]
            paths = [p for p in rec.get('paths') or [] if p in root]
        else:
            paths = [e['path']]
        deps = dl.dependents(root, paths)
        if ask:
            box = QtWidgets.QMessageBox
            text = f"Delete {e['label']} from the store? This cannot be " \
                'undone.'
            if deps:
                text += ('\n\nThese depend on it and will be deleted too:'
                         '\n' + '\n'.join(deps))
            if box.question(self, 'Delete', text, box.Yes | box.No,
                            box.No) != box.Yes:
                return []
            if deps and box.question(
                    self, 'Delete', f"Really delete {e['label']} and the "
                    f'{len(deps)} item(s) that depend on it?',
                    box.Yes | box.No, box.No) != box.Yes:
                return []
        self.flush_notes(refresh = False)
        deleted = dl.delete_item(root, e['path'], run = e['run'])
        self.refresh()
        return deleted

    def _update_detach(self):
        """
        Enable the Remove file button when the selected item has
        attached files.
        """
        self.detach_btn.setEnabled(
            self.attach_btn.isEnabled() and
            bool(self.attachments(self._notes[self._notes_row]['path'])))

    def attachments(self, item):
        """
        Return the files attached to an item in the report site.

        Parameters:
        item (str): the item's path ('' for the store).

        Returns:
        names (list of str): file names (none without a site folder).
        """
        if self.site_dir is None or self.root is None:
            return []
        from .pipeline import site
        out_dir = site.device_dir(self.site_dir, self.root)
        return [n for n, _ in site.attachment_links(
            site.load_attachments(out_dir), item)]

    def attach_file(self, source):
        """
        Attach a file to the selected item: copied into the report
        site's device folder and linked from the pages next to the
        item's notes.

        Parameters:
        source (str): the file.

        Returns:
        rel (str or None): the copy's path relative to the device folder,
            or None if nothing is selected or there is no site folder.
        """
        if self.site_dir is None or self._notes_row is None or \
                self.root is None:
            return None
        from .pipeline import site
        e = self._notes[self._notes_row]
        out_dir = site.device_dir(self.site_dir, self.root)
        rel = site.add_attachment(out_dir, e['path'], source)
        self._show_attachments(e)
        return rel

    def remove_attachment(self, name):
        """
        Remove an attached file from the selected item: the record goes
        and the copy is deleted from the report site (the store is not
        touched).

        Parameters:
        name (str): the file name, as ``attachments`` lists it.

        Returns:
        removed (bool): True if the item had the file.
        """
        if self.site_dir is None or self._notes_row is None or \
                self.root is None:
            return False
        from .pipeline import site
        e = self._notes[self._notes_row]
        out_dir = site.device_dir(self.site_dir, self.root)
        links = dict(site.attachment_links(site.load_attachments(out_dir),
                                           e['path']))
        if name not in links:
            return False
        removed = site.remove_attachment(out_dir, e['path'], links[name])
        self._show_attachments(e)
        return removed

    def _show_attachments(self, e):
        """
        Refresh the label and the Remove file button after an item's
        attachments changed.

        Parameters:
        e (dict): the item's notes entry.
        """
        files = self.attachments(e['path'])
        what = 'the store' if e['kind'] == 'store' else e['label']
        text = f'Notes of {what}'
        if files:
            text += '; attached: ' + ', '.join(files)
        self.notes_label.setText(text)
        self._update_detach()

    def _on_detach(self):
        """
        Choose one of the selected item's attached files to remove (the
        button).
        """
        if self._notes_row is None:
            return
        files = self.attachments(self._notes[self._notes_row]['path'])
        if not files:
            return
        name, ok = QtWidgets.QInputDialog.getItem(
            self, 'Remove an attached file', 'File to remove from the '
            'report:', files, 0, False)
        if ok and name:
            self.remove_attachment(name)

    def _on_attach(self):
        """
        Choose a file to attach to the selected item (the button).
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Attach a file to the report', os.path.expanduser('~'))
        if path:
            self.attach_file(path)

    def _on_notes_edited(self):
        """
        Remember an edit of the shown item's notes, to save later.
        """
        if self._notes_loading or self._notes_row is None or \
                self.notes_edit.isReadOnly():
            return
        self._notes_dirty[self._notes_row] = self.notes_edit.toPlainText()

    def flush_notes(self, refresh = True):
        """
        Save the edited notes to the store (the Notes tab saves by itself:
        on a row or tab change, a refresh, and when the window closes).

        Parameters:
        refresh (bool): If True (default), reread the store afterwards
            when something was saved.

        Returns:
        saved (list of str): the labels of the items written.
        """
        if not self._notes_dirty or not self.path:
            return []
        root = dl.open_store(self.path, mode = 'a')
        saved = []
        for row, text in list(self._notes_dirty.items()):
            e = self._notes[row]
            if e['run'] is not None:
                dl.set_run_notes(root, e['run'][0], e['run'][1], text)
            else:
                dl.set_notes(root, e['path'], text)
            e['notes'] = text.strip()
            self.notes_table.item(row, self._col0 + 2).setText(
                first_line(e['notes'], 120))
            saved.append(e['label'])
        self._notes_dirty = {}
        if refresh:
            self.refresh()
        return saved

    ############################################################################
    def open_path(self, root, path = None):
        """
        Open a store (read-only) and show it.

        Parameters:
        root (zarr.Group, str, or os.PathLike): store root or path
            (``.zarr`` directory or ``.zip`` archive).
        path (str or None): the store's path, for the header. Default
            None: from ``root`` if it is a path.

        Raises:
        ValueError: if it is not a layout store (the store shown before,
            if any, stays).
        """
        if not isinstance(root, zarr.Group):
            path = os.path.expanduser(str(root)) if path is None else path
        self.flush_notes(refresh = False)
        new_root = dl.open_store(root)
        self.root, self.path = new_root, path
        self._selected = None
        name = os.path.basename(os.path.normpath(path)) if path \
            else self.root.attrs.get('device', 'store')
        self.setWindowTitle(f'{self._title}: {name}')
        self.tree.setCurrentItem(None)
        self.refresh()

    def pick_store(self):
        """
        Choose a store in a file dialog (the Open button, Ctrl+O) and
        open it. A choice that is not a store is reported in the header
        and the store shown before stays.

        Returns:
        path (str or None): the store opened, or None if none was.
        """
        start = os.path.dirname(os.path.normpath(self.path)) if self.path \
            else os.path.expanduser('~')
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Open a store (.zarr directory)', start,
            QtWidgets.QFileDialog.ShowDirsOnly)
        if not path:
            return None
        try:
            self.open_path(path)
        except (ValueError, OSError, KeyError) as e:
            self.info.setText(f'{path} is not a device store: {e}')
            return None
        return path

    def refresh(self):
        """
        Reread the store: rebuild the tree (keeping the selection where
        it still exists) and the runs table.
        """
        self.flush_notes(refresh = False)
        current = self.current_path()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.blockSignals(False)
        if self.root is None:
            self.info.setText('No store open: press Open (Ctrl+O) and '
                              'choose a .zarr directory.')
            self.attrs.setPlainText('')
            self.summary.setHtml('')
            self._fill_marks()
            self.runs = []
            self._fill_runs()
            self.items = {}
            self._fill_kinds()
            self._fill_notes()
            return
        self.root = dl.open_store(self.path if self.path else self.root)
        self.info.setText(store_label(self.root, self.path))
        self.tree.blockSignals(True)
        self._add_children(self.tree.invisibleRootItem(), self.root)
        self.tree.blockSignals(False)
        self.runs = dl.stream_runs(self.root)
        self._fill_runs()
        self.items = dl.store_inventory(self.root)
        self._fill_kinds()
        self._fill_marks()
        self._fill_notes()
        selected = self._selected
        if not (current and self.select_path(current)):
            self.attrs.setPlainText(describe_item(self.root))
        if selected and selected in self.root:
            if not self.select_item_path(selected):
                self._select_item(selected)
        else:
            self._selected = None
            self.summary.setHtml(self._store_html())

    def _add_children(self, parent, grp):
        """
        Add a group's groups and arrays under a tree item.

        Parameters:
        parent (QTreeWidgetItem): the item.
        grp (zarr.Group): the group.
        """
        for name in sorted(grp.group_keys(), key = dl._natural):
            child = grp[name]
            item = QtWidgets.QTreeWidgetItem(
                parent, [name, str(child.attrs.get('kind', '')),
                         summarize(child)])
            item.setData(0, Qt.UserRole, child.path)
            # A placeholder makes the item expandable; the real children
            # are read when it is expanded
            item.setChildIndicatorPolicy(
                QtWidgets.QTreeWidgetItem.ShowIndicator)
        for name in sorted(grp.array_keys(), key = dl._natural):
            arr = grp[name]
            item = QtWidgets.QTreeWidgetItem(
                parent, [name, 'array', summarize(arr)])
            item.setData(0, Qt.UserRole, arr.path)
            item.setChildIndicatorPolicy(
                QtWidgets.QTreeWidgetItem.DontShowIndicator)

    def _on_expanded(self, item):
        """
        Read a group's children the first time it is expanded.

        Parameters:
        item (QTreeWidgetItem): the group's item.
        """
        if item.childCount() or item.data(0, Qt.UserRole + 1):
            return
        item.setData(0, Qt.UserRole + 1, True)
        grp = self.root[item.data(0, Qt.UserRole)]
        if not isinstance(grp, zarr.Group):
            return
        self._add_children(item, grp)
        if not item.childCount():
            item.setChildIndicatorPolicy(
                QtWidgets.QTreeWidgetItem.DontShowIndicator)

    def _on_current(self, item, _previous = None):
        """
        Show the selected item's attributes.

        Parameters:
        item (QTreeWidgetItem or None): the selected item.
        _previous (QTreeWidgetItem or None): the one before (unused).
        """
        if item is None:
            self.attrs.setPlainText(describe_item(self.root) if self.root
                                    is not None else '')
            return
        path = item.data(0, Qt.UserRole)
        try:
            self.attrs.setPlainText(describe_item(self.root[path]))
        except (KeyError, OSError, ValueError) as e:
            self.attrs.setPlainText(f'{path}\n\nCould not read it: {e!r}\n'
                                    'Refresh (F5) if the store changed.')
            return
        if not self._selecting:
            self._select_item(path, attrs = False)

    def current_path(self):
        """
        Return the selected item's path in the store.

        Returns:
        path (str or None): the path, or None if nothing is selected.
        """
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def select_path(self, path):
        """
        Expand the tree down to a path and select it.

        Parameters:
        path (str): path in the store, e.g. 'tune0/cal0/write0/ts0'.

        Returns:
        found (bool): True if the path is in the tree.
        """
        parent = self.tree.invisibleRootItem()
        item = None
        for name in str(path).strip('/').split('/'):
            if item is not None:
                item.setExpanded(True)
                self._on_expanded(item)
                parent = item
            item = next((parent.child(i) for i in range(parent.childCount())
                         if parent.child(i).text(0) == name), None)
            if item is None:
                return False
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        return True

    ############################################################################
    def _fill_runs(self):
        """
        Fill the runs table with the runs containing the filter text.
        """
        text = self.filter_edit.text().strip().lower()
        shown = [r for r in self.runs if not text or any(
            text in str(r.get(key, '')).lower() for _, key in RUN_COLUMNS)]
        self.runs_table.setRowCount(len(shown))
        for i, r in enumerate(shown):
            for j, (_, key) in enumerate(RUN_COLUMNS):
                value = r.get(key)
                if key == 'tone_set' and value is None:
                    value = r.get('tone_sets')
                if key == 'calibration' and value is None:
                    value = r.get('calibrations')
                if key == 'notes':
                    value = first_line(value, 200)
                cell = QtWidgets.QTableWidgetItem(
                    '' if value is None else str(value))
                cell.setData(Qt.UserRole, i)
                self.runs_table.setItem(i, j, cell)
        self._shown_runs = shown
        self.runs_table.resizeColumnsToContents()
        self.runs_label.setText(
            f'{len(shown)} of {len(self.runs)} runs' if text
            else f'{len(self.runs)} runs')

    def _on_run_activated(self, row, _col = None):
        """
        Select a run's first stream in the tree.

        Parameters:
        row (int): the table row.
        _col (int or None): the column (unused).
        """
        r = self._shown_runs[row]
        paths = r.get('paths') or []
        if paths and self.select_path(paths[0]):
            self.kinds.setCurrentWidget(self.tree)
            self.tabs.setCurrentWidget(self.attrs)

    def _on_run_row(self, row, _col = 0, _prev_row = -1, _prev_col = -1):
        """
        Show the run selected in the runs table.

        Parameters:
        row (int): the table row (-1: none).
        _col, _prev_row, _prev_col (int): unused.
        """
        if self._selecting or not (0 <= row < len(self._shown_runs)):
            return
        paths = self._shown_runs[row].get('paths') or []
        if paths:
            self._select_item(paths[0])

    ############################################################################
    def _fill_kinds(self):
        """
        Fill every kind table from the inventory, and show the counts.
        """
        for title in self.kind_tables:
            self._fill_kind(title)
        for i, (title, _) in enumerate(dl.ITEM_KINDS):
            n = len(self.runs) if title == 'Runs' else \
                len(self.items.get(title, []))
            self.kinds.setTabText(i, f'{title} ({n})' if n else title)

    def _fill_kind(self, title):
        """
        Fill one kind's table with its items containing the filter text,
        keeping the selection.

        Parameters:
        title (str): the kind.
        """
        table = self.kind_tables[title]
        text = self.kind_filters[title].text().strip().lower()
        entries = self.items.get(title, [])
        shown = [e for e in entries if not text or any(
            text in str(v).lower() for v in (
                e['label'], e['name'], e['created'], ' '.join(e['made']),
                e['notes']))]
        self._selecting = True
        table.setRowCount(len(shown))
        for i, e in enumerate(shown):
            values = (e['label'], e['name'] or '', e['created'],
                      ', '.join(e['made']), first_line(e['notes'], 120))
            for j, v in enumerate(values):
                table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))
        self._kind_rows[title] = shown
        for j in range(len(KIND_COLUMNS) - 1):
            table.resizeColumnToContents(j)
        self._selecting = False
        self.kind_labels[title].setText(
            f'{len(shown)} of {len(entries)}' if text else f'{len(entries)}')
        if self._selected is not None:
            for i, e in enumerate(shown):
                if e['path'] == self._selected:
                    self._selecting = True
                    table.setCurrentCell(i, 0)
                    self._selecting = False

    def _on_kind_row(self, title, row):
        """
        Show the item selected in a kind table.

        Parameters:
        title (str): the kind.
        row (int): the table row (-1: none).
        """
        shown = self._kind_rows.get(title, [])
        if self._selecting or not (0 <= row < len(shown)):
            return
        self._select_item(shown[row]['path'])

    def entry_of(self, path):
        """
        Return the inventory entry of an item.

        Parameters:
        path (str): the item's path (a run's first stream).

        Returns:
        entry (dict or None): the entry (``daq_layout.store_inventory``),
            or None if the path is not an item.
        """
        for entries in self.items.values():
            for e in entries:
                if e['path'] == path:
                    return e
        return None

    def _select_item(self, path, notes_row = True, attrs = True):
        """
        Show an item on the right: its summary, attributes, and its row
        of the notes table (which enables Rename and Delete).

        Parameters:
        path (str): the item's path (any group or array of the store).
        notes_row (bool): If True (default), select its notes row.
        attrs (bool): If True (default), describe it in Attributes.
        """
        if self.root is None:
            return
        self._selected = path
        self._selecting = True
        try:
            if attrs:
                try:
                    self.attrs.setPlainText(describe_item(self.root[path]))
                except (KeyError, OSError, ValueError) as e:
                    self.attrs.setPlainText(f'{path}\n\nCould not read '
                                            f'it: {e!r}')
            self.summary.setHtml(self._summary_html(path))
            if notes_row:
                row = next((i for i, e in enumerate(self._notes)
                            if e['path'] == path), None)
                if row is not None and row != self.notes_table.currentRow():
                    self._selecting = False
                    self.notes_table.setCurrentCell(row, 0)
                    self._selecting = True
                elif row is None:
                    for b in (self.tree_btn, self.reveal_btn):
                        b.setEnabled(bool(self.path))
                    self.reveal_btn.setEnabled(
                        bool(self.path) and not str(self.path).endswith('.zip'))
        finally:
            self._selecting = False

    def select_item_path(self, path):
        """
        Select an item in its kind table (switching to that tab) and show
        it on the right; a path that is no item is shown in the tree.

        Parameters:
        path (str): the item's path.

        Returns:
        found (bool): True if the path is an item or in the tree.
        """
        for title, entries in self.items.items():
            for e in entries:
                if e['path'] != path:
                    continue
                tab = [t for t, _ in dl.ITEM_KINDS].index(title)
                self.kinds.setCurrentIndex(tab)
                if title == 'Runs':
                    row = next((i for i, r in enumerate(self._shown_runs)
                                if (r.get('paths') or [None])[0] == path),
                               None)
                    table = self.runs_table
                else:
                    row = next((i for i, x in enumerate(
                        self._kind_rows.get(title, [])) if x is e), None)
                    table = self.kind_tables[title]
                if row is None:
                    self._select_item(path)
                elif row == table.currentRow():
                    self._select_item(path)
                else:
                    table.setCurrentCell(row, 0)
                return True
        if self.select_path(path):
            self.kinds.setCurrentWidget(self.tree)
            return True
        return False

    def _store_html(self):
        """
        Describe the store for the summary when nothing is selected.

        Returns:
        text (str): HTML.
        """
        parts = [f'<h3>{html.escape(store_label(self.root, self.path))}'
                 '</h3><p>']
        for title, _ in dl.ITEM_KINDS:
            n = len(self.runs) if title == 'Runs' else \
                len(self.items.get(title, []))
            if n:
                parts.append(f'{title}: {n}<br>')
        notes = self.root.attrs.get('notes')
        if notes:
            parts.append('</p><p>' + html.escape(str(notes)).replace(
                '\n', '<br>'))
        return ''.join(parts) + '</p>'

    def _summary_html(self, path):
        """
        Describe an item for the summary: what it is, when, its notes,
        what it was made from and what depends on it, as links.

        Parameters:
        path (str): the item's path.

        Returns:
        text (str): HTML; links are ``item:<path>``.
        """
        e = self.entry_of(path)
        link = lambda p: (f'<a href="item:{html.escape(p)}">'
                          f'{html.escape(p)}</a>')
        try:
            item = self.root[path]
        except KeyError:
            return f'<p>{html.escape(path)} is not in the store.</p>'
        if e is None:
            kind = 'array' if isinstance(item, zarr.Array) else \
                str(item.attrs.get('kind', 'group'))
            return (f'<h3>{html.escape(path)}</h3><p>{html.escape(kind)}: '
                    f'{html.escape(summarize(item))}</p>')
        head = html.escape(e['label'])
        if e['name'] and e['name'] not in e['label']:
            head += f" ({html.escape(e['name'])})"
        parts = [f'<h3>{head}</h3><p>{html.escape(e["kind_title"])}']
        if e['created']:
            parts.append(f", {html.escape(e['created'])}")
        parts.append('<br>' + html.escape(summarize(item)) if not e['run']
                     else '')
        if e['run'] is not None and e['record']:
            r = e['record']
            parts.append(f"<br>stage {r.get('dec_stage')}, "
                         f"{r.get('duration_s')} s, tone set "
                         f"{r.get('tone_set', r.get('tone_sets'))}, "
                         f"{r.get('n_chunks') or 0} chunks")
            if r.get('point') is not None:
                parts.append(f"<br>point {html.escape(str(r['point']))}")
            if r.get('power_w') is not None:
                parts.append(f"<br>power {r['power_w']:.4g} W")
        if not e['in_report']:
            parts.append('<br>left out of the report')
        parts.append('</p>')
        if e['notes']:
            parts.append('<p>' + html.escape(e['notes']).replace(
                '\n', '<br>') + '</p>')
        if e['made']:
            parts.append('<p>Made from: ' + ', '.join(
                link(p) for p in e['made']) + '</p>')
        paths = [path]
        if e['run'] is not None and e['record']:
            paths = [p for p in e['record'].get('paths') or [] if p in
                     self.root] or [path]
        try:
            deps = dl.dependents(self.root, paths)
        except Exception:
            deps = []
        if deps:
            parts.append('<p>Depends on it (deleted with it): ' + ', '.join(
                link(p) for p in deps) + '</p>')
        return ''.join(parts)

    def _on_anchor(self, url):
        """
        Select the item a summary link names.

        Parameters:
        url (QUrl): the link (``item:<path>``).
        """
        text = url.toString()
        if text.startswith('item:'):
            self.select_item_path(text[5:])

    def _open_in_tree(self):
        """
        Show the selected item's group in the Store tab (the button, or a
        double-click).
        """
        if self._selected and self.select_path(self._selected):
            self.kinds.setCurrentWidget(self.tree)
            self.tabs.setCurrentWidget(self.attrs)

    def reveal_selected(self):
        """
        Open the selected item's directory in the file browser (the
        button): a local .zarr store only.

        Returns:
        folder (str or None): the directory opened, or None.
        """
        if not self._selected or not self.path or \
                str(self.path).endswith('.zip'):
            return None
        folder = os.path.join(os.path.normpath(self.path),
                              *self._selected.split('/'))
        if not os.path.isdir(folder):
            self.info.setText(f'{folder} is not a directory')
            return None
        open_folder(folder)
        return folder

    def _context_menu(self, widget, pos):
        """
        Show the actions on the selected item (a right-click in a table
        or the tree).

        Parameters:
        widget (QWidget): the table or tree.
        pos (QPoint): where, in the widget.
        """
        if self._selected is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction('Open in store tree', self._open_in_tree)
        act = menu.addAction('Reveal folder', self.reveal_selected)
        act.setEnabled(self.reveal_btn.isEnabled())
        act = menu.addAction('Rename...', self._on_rename)
        act.setEnabled(self.rename_btn.isEnabled())
        act = menu.addAction('Delete...', lambda: self.delete_selected())
        act.setEnabled(self.delete_btn.isEnabled())
        menu.exec_(widget.mapToGlobal(pos))

    def closeEvent(self, event):
        """
        Save edited notes and report the store shown when the window
        closes.

        Parameters:
        event (QCloseEvent): the close event.
        """
        self.flush_notes(refresh = False)
        self.closed.emit(self.path or '')
        super().closeEvent(event)

################################################################################
################################### Launcher ###################################
################################################################################
def choose_store(start = None):
    """
    Choose a device store in a directory dialog.

    Parameters:
    start (str or None): the folder the dialog opens in. Default None:
        the home folder.

    Returns:
    path (str or None): the store chosen, or None if cancelled.
    """
    get_qapp()
    path = QtWidgets.QFileDialog.getExistingDirectory(
        None, 'Choose a device store (.zarr directory)',
        os.path.expanduser(start or '~'), QtWidgets.QFileDialog.ShowDirsOnly)
    return path or None

def ask_work_by(current = ''):
    """
    Ask who did the work, for the report (a line at the top of the
    device page and a column of the site's top page).

    Parameters:
    current (str): the value the store holds, offered as the default.
        Default ''.

    Returns:
    text (str or None): the text typed (blank removes the value), or
        None on Cancel.
    """
    get_qapp()
    text, ok = QtWidgets.QInputDialog.getText(
        None, 'Who did the work',
        'Who did the work, for the report (initials: DAQ, analysis),\n'
        "e.g. 'LF + BE DAQ, LF analysis':", text = current or '')
    return text.strip() if ok else None

def ask_existing_report(out_dir):
    """
    Ask what to do with a report that exists: replace it, rebuild it
    keeping its attached files and notes.md, or stop.

    Parameters:
    out_dir (str): the device's folder in the report site.

    Returns:
    choice (str or None): 'overwrite', 'update', or None (Cancel, or
        closing the box).
    """
    get_qapp()
    box = QtWidgets.QMessageBox()
    box.setWindowTitle('The report exists')
    box.setIcon(QtWidgets.QMessageBox.Icon.Question)
    box.setText(f'{out_dir} already holds a report of this store.')
    box.setInformativeText(
        'Keep attachments: rebuild the pages, keeping the attached files '
        'and notes.md. Overwrite: replace everything, attached files '
        'included.')
    keep = box.addButton('Keep attachments',
                         QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    over = box.addButton('Overwrite',
                         QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
    box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(keep)
    box.exec() if hasattr(box, 'exec') else box.exec_()
    clicked = box.clickedButton()
    if clicked is keep:
        return 'update'
    if clicked is over:
        return 'overwrite'
    return None

def run_store_browser(path = None, title = 'Store browser',
                      ui_scale = 1.0, site_dir = None, tab = None):
    """
    Open the browser and wait until the window is closed.

    Parameters:
    path (str, os.PathLike, or None): the store's path. None (default):
        start with the file dialog to choose one.
    title (str): window title (the store's name is added). Default
        'Store browser'.
    ui_scale (float): font size multiplier. Default 1.0.
    site_dir (str or None): the report site's folder, to attach files
        from the Notes tab. Default None.
    tab (str or None): the tab to open on, e.g. 'Notes'. Default None:
        the first.

    Returns:
    path (str or None): the path of the store shown last, or None if
        none was opened.
    """
    app = get_qapp(title)
    win = StoreBrowserWindow(path, title = title, ui_scale = ui_scale,
                             site_dir = site_dir)
    if tab is not None:
        for i in range(win.tabs.count()):
            if win.tabs.tabText(i) == tab:
                win.tabs.setCurrentIndex(i)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    if path is None:
        QtCore.QTimer.singleShot(0, win.pick_store)
    opened = []
    win.closed.connect(lambda p: opened.append(p or None))
    # exec() is the modern name (PyQt6+); exec_() is kept for PyQt5.
    (getattr(app, 'exec', None) or app.exec_)()
    return opened[-1] if opened else (
        os.path.expanduser(str(path)) if path is not None else None)
