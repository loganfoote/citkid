"""
Tests for the store browser window (``citkid.daq_browser``), on a small
store in the current layout.
"""
import numpy as np
import os
import pytest
from pyqtgraph.Qt import QtWidgets
from citkid import daq_layout as dl
from citkid.daq_browser import (
    StoreBrowserWindow, describe_item, first_line, format_value,
    run_store_browser, store_label, summarize
)


@pytest.fixture(scope = 'module')
def qapp():
    """
    Return the QApplication (offscreen).

    Returns:
    app (QApplication): the application.
    """
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope = 'module')
def store(tmp_path_factory):
    """
    Make a store with a resonator list, a tuning with two calibrations,
    a tone write and stream under each, and the run list.

    Returns:
    path (pathlib.Path): the store.
    """
    path = tmp_path_factory.mktemp('browser') / 'devA_cold.zarr'
    root = dl.create_store(path, 'devA', 'cd1')
    dl.save_resonators(root, 0, [1e9, 1.1e9, 1.2e9], 2e4, 'x')
    tune, _ = dl.new_tuning(root, 0)
    dl.save_tones(dl.create_group(tune, 'tones', 'decision'),
                  [1e9, 1.1e9, 1.2e9], 3e4, -50., [0, 1, 2],
                  tone_set = [0, 0, 1])
    for j, (s, notes) in enumerate(((0, 'after warm-up'), (1, ''))):
        cal = dl.create_group(tune, f'cal{j}', 'raw', attrs = dict(
            tone_set = s, complete = True, notes = notes, nsamps = 100))
        n = 2 if s == 0 else 1
        cal.create_array('fres', data = np.full(n, 1e9))
        cal.create_array('res_idxs', data = np.arange(n))
        w = dl.create_group(cal, 'write0', 'raw', attrs = dict(
            tones_reused = False))
        w.create_array('sweep_rows', data = np.arange(n))
        st = dl.create_group(w, f'ts{j}', 'raw', attrs = dict(
            name = f'dark{j}', notes = f'lights off\nrun {j}', dec_stage = 6,
            duration_s = 2., run = j, timestamp = 't'))
        st.create_array('z', data = np.zeros((2, n, 8), dtype = np.int32),
                        chunks = (1, 1, 4))
        st.create_array('counts_to_s21', data = np.ones(n))
        st.create_array('dt', data = np.array(0.25))
        dl.register_stream_run(tune, dict(
            name = f'dark{j}', notes = f'lights off\nrun {j}', dec_stage = 6,
            duration_s = 2., tuning = 'tune0', tones = 'tune0/tones',
            resonators = 'resonators0', complete = True, tone_set = s,
            calibration = f'tune0/cal{j}', paths = [st.path]))
    return path


def test_helpers(store):
    root = dl.open_store(store)
    assert first_line(None) == '' and first_line(' \n\n') == ''
    assert first_line('a\nb') == 'a ...' and first_line('x' * 90, 10) == \
        'x' * 7 + '...'
    assert summarize(root['resonators0']) == '3 resonators'
    assert summarize(root['tune0']) == '3 tones, resonators0'
    assert summarize(root['tune0/cal0']) == 'set 0, 2 tones, after warm-up'
    assert summarize(root['tune0/cal1']) == 'set 1, 1 tones'
    assert summarize(root['tune0/cal0/write0/ts0']) == \
        'dark0, stage 6, 2 s, lights off ...'
    assert summarize(root['tune0/cal0/write0/ts0/z']) == '(2, 2, 8) int32'
    assert format_value('a\nb') == 'a\nb' and format_value([1, 2]) == \
        '[\n  1,\n  2\n]'
    cut = format_value('y' * 5000)
    assert cut.startswith('y' * 4000)
    assert cut.endswith('(1,000 more characters)')
    text = describe_item(root['tune0/cal0/write0/ts0'])
    lines = text.splitlines()
    assert lines[0] == 'tune0/cal0/write0/ts0'
    # name and notes first, multi-line notes indented, the rest sorted
    assert lines[2] == 'name: dark0' and lines[3] == 'notes:'
    assert lines[4:6] == ['    lights off', '    run 0']
    assert lines[6] == 'kind: raw'
    assert 'run: 0' in lines and 'timestamp: t' in lines
    text = describe_item(root['tune0/cal0/write0/ts0/counts_to_s21'])
    assert 'shape: (2,)' in text and 'values: [1.0, 1.0]' in text
    assert '(no attributes)' in text
    assert describe_item(root).startswith('/\n')
    label = store_label(root, '/p/devA_cold.zarr')
    assert label.startswith('/p/devA_cold.zarr; device devA; cooldown cd1')


def test_window(qapp, store):
    win = StoreBrowserWindow(store, title = 't')
    win.show()
    qapp.processEvents()
    assert win.windowTitle() == 't: devA_cold.zarr'
    assert 'device devA' in win.info.text()
    top = win.tree.invisibleRootItem()
    names = [top.child(i).text(0) for i in range(top.childCount())]
    assert names == ['resonators0', 'tune0']
    kinds = [top.child(i).text(1) for i in range(top.childCount())]
    assert kinds == ['decision', 'decision']
    # Children are read on expansion: the tuning's groups, then arrays
    tune_item = top.child(1)
    assert tune_item.childCount() == 0
    tune_item.setExpanded(True)
    children = [tune_item.child(i).text(0)
                for i in range(tune_item.childCount())]
    assert children == ['cal0', 'cal1', 'streams', 'tones']
    assert tune_item.child(0).text(2) == 'set 0, 2 tones, after warm-up'
    # Selecting an item shows its attributes; arrays show their shape
    assert win.select_path('tune0/cal0/write0/ts0/z')
    assert win.current_path() == 'tune0/cal0/write0/ts0/z'
    assert win.attrs.toPlainText().startswith(
        'tune0/cal0/write0/ts0/z\n\nshape: (2, 2, 8)\ndtype: int32\n'
        'chunks: (1, 1, 4)')
    assert not win.select_path('tune0/nothing')
    # The runs table lists both runs with their notes; the filter keeps
    # the matching ones; double-clicking one selects its stream
    assert win.runs_table.rowCount() == 2
    assert win.runs_table.item(1, 1).text() == 'dark1'
    assert win.runs_table.item(1, 2).text() == 'lights off ...'
    assert win.runs_table.item(0, 6).text() == 'tune0/cal0'
    assert win.runs_label.text() == '2 runs'
    win.filter_edit.setText('dark1')
    assert win.runs_table.rowCount() == 1 and \
        win.runs_label.text() == '1 of 2 runs'
    win._on_run_activated(0, 0)
    assert win.current_path() == 'tune0/cal1/write0/ts1'
    assert win.tabs.currentWidget() is win.attrs
    win.filter_edit.setText('')
    # Refresh rereads the store and keeps the selection
    root = dl.open_store(store, mode = 'a')
    dl.create_group(root, 'ts_cal0', 'analysis', attrs = dict(
        selection = [dict(name = 'dark0')]))
    win.refresh()
    assert win.current_path() == 'tune0/cal1/write0/ts1'
    names = [top.child(i).text(0) for i in range(top.childCount())]
    assert names == ['resonators0', 'ts_cal0', 'tune0']
    assert top.child(1).text(2) == '1 selected'
    # The notes tab lists the store, every step and result, and every
    # run; the editor holds the selected row's notes and saves them by
    # itself on a row change, a tab change, or closing
    labels = [win.notes_table.item(i, 0).text()
              for i in range(win.notes_table.rowCount())]
    assert labels == ['(store)', 'ts_cal0', 'tune0', 'tune0/cal0',
                      'ts0 dark0', 'tune0/cal1', 'ts1 dark1']
    kinds = [win.notes_table.item(i, 1).text()
             for i in range(win.notes_table.rowCount())]
    assert kinds[0] == 'store' and kinds[4] == 'timestream run'
    assert win.notes_table.item(3, 2).text() == 'after warm-up'
    assert win.notes_table.item(6, 2).text() == 'lights off ...'
    win.tabs.setCurrentIndex(3)
    win.notes_table.setCurrentCell(2, 0)
    assert 'Notes of tune0' in win.notes_label.text()
    win.notes_edit.setPlainText('retuned after warm-up')
    win.notes_table.setCurrentCell(0, 0)
    assert dl.open_store(store)['tune0'].attrs['notes'] == \
        'retuned after warm-up'
    assert win.notes_table.item(2, 2).text() == 'retuned after warm-up'
    win.notes_edit.setPlainText('cooled on Monday')
    win.tabs.setCurrentIndex(0)
    assert dl.open_store(store).attrs['notes'] == 'cooled on Monday'
    assert dl.list_notes(dl.open_store(store))[0] == ('', 'cooled on Monday')
    # A run's notes go to its record and its stream group; saved on close
    win.notes_table.setCurrentCell(6, 0)
    win.notes_edit.setPlainText('lights on')
    win.close()
    root = dl.open_store(store)
    assert dl.stream_runs(root)[1]['notes'] == 'lights on'
    assert root['tune0/cal1/write0/ts1'].attrs['notes'] == 'lights on'
    assert ('ts1 dark1', 'lights on') in dl.list_notes(root)


def test_attachments(qapp, store, tmp_path):
    """
    With a site folder, a file can be attached to the selected item: it
    is copied into the site's device folder and listed, never into the
    store.
    """
    from citkid.pipeline import site
    site_dir = tmp_path / 'site'
    win = StoreBrowserWindow(store, title = 't', site_dir = str(site_dir))
    win.show()
    qapp.processEvents()
    assert not win.attach_btn.isEnabled()
    win.tabs.setCurrentIndex(3)
    win.notes_table.setCurrentCell(2, 0)         # tune0
    assert win.attach_btn.isEnabled()
    f = tmp_path / 'wiring.txt'
    f.write_text('photo of the wiring')
    rel = win.attach_file(str(f))
    dev = site_dir / 'cd1' / 'devA'
    assert rel == 'attachments/tune0/wiring.txt'
    assert (dev / rel).read_text() == 'photo of the wiring'
    assert site.load_attachments(str(dev)) == {'tune0': [rel]}
    assert 'attached: wiring.txt' in win.notes_label.text()
    # Again: not listed twice; another name beside it; the store itself
    assert win.attach_file(str(f)) == rel
    assert site.load_attachments(str(dev)) == {'tune0': [rel]}
    g = tmp_path / 'other.txt'
    g.write_text('x')
    assert win.attach_file(str(g)) == 'attachments/tune0/other.txt'
    win.notes_table.setCurrentCell(0, 0)
    assert win.attach_file(str(g)) == 'attachments/store/other.txt'
    assert win.attachments('') == ['other.txt']
    assert 'attachments' not in os.listdir(store)
    win.close()


def test_open_and_pick(qapp, store, tmp_path, monkeypatch):
    """
    The window starts empty without a store; Open picks a store in a
    file dialog, and a choice that is not a store leaves the one shown.
    """
    win = StoreBrowserWindow(title = 't')
    win.show()
    qapp.processEvents()
    assert win.root is None and 'No store open' in win.info.text()
    assert win.tree.topLevelItemCount() == 0 and win.runs_label.text() == \
        '0 runs'
    assert win.windowTitle() == 't'
    choice = [str(store)]
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: choice[0]))
    assert win.pick_store() == str(store)
    assert win.path == str(store) and win.windowTitle() == 't: devA_cold.zarr'
    assert win.tree.topLevelItemCount() == 3 and 'device devA' in \
        win.info.text()
    assert win.attrs.toPlainText().startswith('/\n')
    # Not a store: reported, the store shown stays
    choice[0] = str(tmp_path)
    assert win.pick_store() is None
    assert 'is not a device store' in win.info.text()
    assert win.path == str(store) and win.tree.topLevelItemCount() == 3
    # Cancelled: nothing changes
    choice[0] = ''
    assert win.pick_store() is None
    assert win.path == str(store)
    closed = []
    win.closed.connect(closed.append)
    win.close()
    assert closed == [str(store)]


def test_run_store_browser(qapp, store, monkeypatch):
    from pyqtgraph.Qt import QtCore
    from citkid.daq_browser import StoreBrowserWindow as W

    def close_window():
        """Close the browser window (ends the event loop)."""
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, W):
                w.close()

    QtCore.QTimer.singleShot(0, close_window)
    assert run_store_browser(store) == str(store)
    # Without a path, the dialog chooses the store
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: str(store)))
    QtCore.QTimer.singleShot(50, close_window)
    assert run_store_browser() == str(store)
    # ... and nothing is opened if it is cancelled
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: ''))
    QtCore.QTimer.singleShot(50, close_window)
    assert run_store_browser() is None
