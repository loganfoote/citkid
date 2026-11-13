"""
Interactive browser of a device store.

``run_store_browser`` opens a window on a store (``citkid.daq_layout``):
a tree of its groups and arrays on the left (sweeps, resonator lists,
tunings with their calibrations, tone writes and streams, stream
calibrations, photon results), each with its kind and a one-line summary
(a timestream's name and notes, a calibration's tone set, an array's
shape), and on the right the attributes of the selected item (notes and
settings, as saved) or, for an array, its shape, dtype, chunks, and the
first values. A second tab lists the timestream runs of every tuning
(name, notes, stage, length, tone set, calibration), filtered by typing;
double-clicking a run selects its stream in the tree. Open (Ctrl+O)
picks another store in a file dialog. Nothing is written; the store is
opened read-only, so it can be browsed while a script is writing to it
(Refresh, F5, rereads it). ``procedures/store_browser.py`` is the
script.
"""
import json
import os
import numpy as np
import zarr
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from . import daq_layout as dl
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

################################################################################
################################### Helpers ####################################
################################################################################
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
    if a.get('name'):
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

################################################################################
#################################### Window ####################################
################################################################################
class StoreBrowserWindow(QtWidgets.QMainWindow):
    """
    Window browsing a store: the tree of its groups and arrays on the
    left, the selected item's attributes or the list of timestream runs
    on the right.

    Attributes:
    root (zarr.Group or None): the store root (read-only); None while no
        store is open.
    path (str or None): the store's path, if known.
    tree (QTreeWidget): the store's groups and arrays (children are read
        when a group is expanded).
    attrs (QPlainTextEdit): the selected item's description.
    runs_table (QTableWidget): the timestream runs.
    runs (list of dict): the runs shown (``daq_layout.stream_runs``).
    filter_edit (QLineEdit): text the shown runs must contain.
    """
    # The path of the store shown (or '') when the window is closed
    closed = QtCore.Signal(str)

    def __init__(self, root = None, path = None, title = 'Store browser',
                 ui_scale = 1.0):
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
        """
        super().__init__()
        self.root, self.path = None, None
        self.runs, self._shown_runs = [], []
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
        split.addWidget(self._build_tree())
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
            'Expand a group to read its contents; click an item for its '
            'attributes. Double-click a run in the Timestream runs tab to '
            'find its stream. F5 rereads the store, Ctrl+O opens another.'))
        self.setCentralWidget(central)
        if root is not None:
            self.open_path(root, path)
        else:
            self.refresh()

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
        return self.tree

    def _build_tabs(self):
        """
        Build the attributes panel and the runs table.

        Returns:
        tabs (QTabWidget): the tabs.
        """
        self.tabs = QtWidgets.QTabWidget()
        self.attrs = QtWidgets.QPlainTextEdit()
        self.attrs.setReadOnly(True)
        self.attrs.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.attrs.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont))
        self.tabs.addTab(self.attrs, 'Attributes')
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
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.cellDoubleClicked.connect(self._on_run_activated)
        vbox.addWidget(self.runs_table, stretch = 1)
        self.runs_label = QtWidgets.QLabel()
        vbox.addWidget(self.runs_label)
        self.tabs.addTab(page, 'Timestream runs')
        return self.tabs

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
        new_root = dl.open_store(root)
        self.root, self.path = new_root, path
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
        current = self.current_path()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.blockSignals(False)
        if self.root is None:
            self.info.setText('No store open: press Open (Ctrl+O) and '
                              'choose a .zarr directory.')
            self.attrs.setPlainText('')
            self.runs = []
            self._fill_runs()
            return
        self.root = dl.open_store(self.path if self.path else self.root)
        self.info.setText(store_label(self.root, self.path))
        self.tree.blockSignals(True)
        self._add_children(self.tree.invisibleRootItem(), self.root)
        self.tree.blockSignals(False)
        self.runs = dl.stream_runs(self.root)
        self._fill_runs()
        if not (current and self.select_path(current)):
            self.attrs.setPlainText(describe_item(self.root))

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
            self.tabs.setCurrentWidget(self.attrs)

    def closeEvent(self, event):
        """
        Report the store shown when the window closes.

        Parameters:
        event (QCloseEvent): the close event.
        """
        self.closed.emit(self.path or '')
        super().closeEvent(event)

################################################################################
################################### Launcher ###################################
################################################################################
def run_store_browser(path = None, title = 'Store browser',
                      ui_scale = 1.0):
    """
    Open the browser and wait until the window is closed.

    Parameters:
    path (str, os.PathLike, or None): the store's path. None (default):
        start with the file dialog to choose one.
    title (str): window title (the store's name is added). Default
        'Store browser'.
    ui_scale (float): font size multiplier. Default 1.0.

    Returns:
    path (str or None): the path of the store shown last, or None if
        none was opened.
    """
    app = get_qapp(title)
    win = StoreBrowserWindow(path, title = title, ui_scale = ui_scale)
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
