"""
Viewer of saved timestreams (``procedures/view_timestreams.py``): the
plots of the 4a window on runs already in a store, with no board.

Left: the tuning, its runs, the chunks to read (from which chunk, how
many), the tone, and the Plots box (x, parallel and perpendicular, or I
and Q; Show PSD; Subtract offset). Right: the timestream, the
calibration loop with the timestream's points, and the PSD. View run
(or a double-click) reads the chunks; R rescales the plots.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from .. import daq_layout as dl
from ..qt_compat import Qt, add_help
from ..qt_compat import (
    delete_on_close, fit_window_to_screen, freeze_heights, get_qapp,
    run_responsive, widget_font_stylesheet)
from .stream_gui import MODES, load_run_view, plot_view


class TimestreamViewer(QtWidgets.QMainWindow):
    """
    Window showing saved timestream runs of a store.

    Attributes:
    root (zarr.Group): the store root (read-only).
    tune (zarr.Group or None): the tuning shown.
    tones (dict): its tone list (``daq_layout.load_tones``).
    runs (list of dict): its runs (``daq_layout.stream_runs``).
    view (dict or None): the run shown (``load_run_view``).
    """

    def __init__(self, root, path = None, tuning = None,
                 title = 'Timestreams', ui_scale = 1.0):
        """
        Build the window on a store.

        Parameters:
        root (zarr.Group, str or os.PathLike): the store root or path.
        path (str or None): the store's path, for the title. Default
            None.
        tuning (int, str or None): the tuning to show first
            (``daq_layout.find_tuning``). Default None: the latest with
            runs.
        title (str): window title. Default 'Timestreams'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self._ui_scale = float(ui_scale) if ui_scale else 1.0
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.root = root if hasattr(root, 'attrs') else dl.open_store(root)
        self.path = path if path is not None else (
            None if hasattr(root, 'attrs') else str(root))
        self.setWindowTitle(title if self.path is None
                            else f'{title}: {self.path}')
        self.tune, self.tones, self.runs, self.view = None, None, [], None
        self._build_panel()
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._scroll)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.plots, stretch = 2)
        right.addWidget(self.psd_plot, stretch = 1)
        lay.addLayout(right, stretch = 1)
        self.setCentralWidget(central)
        add_help(self, 'Timestreams', [
            ('Buttons', [
                ('View run', "Read the chosen chunks of the selected run "
                 '(or double-click it)'),
                ('Show PSD', 'The PSD of what the timestream plot shows'),
                ('Subtract offset', "Each series minus its mean")]),
            ('Keyboard', [('R', 'Rescale the plots'),
                          ('H', 'Show / hide this help')]),
            ('Mouse', [('Scroll / drag', 'Zoom / pan a plot')]),
        ])
        QtWidgets.QShortcut(QtCore.Qt.Key_R, self, self.rescale)
        self._fill_tune_box(tuning)

    ############################################################################
    def _build_panel(self):
        """
        Build the left panel: tuning, runs, chunks, tone, plots.
        """
        self._panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self._panel)
        form = QtWidgets.QFormLayout()
        self.tune_box = QtWidgets.QComboBox()
        self.tune_box.currentIndexChanged.connect(self._on_tune_box)
        form.addRow('Tuning', self.tune_box)
        lay.addLayout(form)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        lay.addWidget(self.header)
        lay.addWidget(QtWidgets.QLabel('Runs (double-click to view)'))
        self.runs_list = QtWidgets.QListWidget()
        self.runs_list.currentRowChanged.connect(self._on_run_selected)
        self.runs_list.itemDoubleClicked.connect(lambda _: self.view_run())
        lay.addWidget(self.runs_list, stretch = 1)
        form = QtWidgets.QFormLayout()
        self.start_box = QtWidgets.QSpinBox()
        self.start_box.setRange(0, 0)
        self.start_box.setToolTip('The first zarr chunk (along time) read')
        self.count_box = QtWidgets.QSpinBox()
        self.count_box.setRange(1, 1)
        self.count_box.setToolTip('zarr chunks along time read')
        self.chunk_label = QtWidgets.QLabel()
        form.addRow('From chunk', self.start_box)
        form.addRow('Chunks', self.count_box)
        form.addRow('', self.chunk_label)
        self.tone_box = QtWidgets.QComboBox()
        self.tone_box.currentIndexChanged.connect(self._plot)
        form.addRow('Tone', self.tone_box)
        lay.addLayout(form)
        self.view_btn = QtWidgets.QPushButton('View run')
        self.view_btn.clicked.connect(self.view_run)
        self.view_btn.setEnabled(False)
        lay.addWidget(self.view_btn)
        box = QtWidgets.QGroupBox('Plots')
        form = QtWidgets.QFormLayout(box)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.currentIndexChanged.connect(self._plot)
        form.addRow('Show', self.mode_box)
        self.psd_box = QtWidgets.QCheckBox('Show PSD')
        self.psd_box.toggled.connect(self._on_psd_box)
        self.offset_box = QtWidgets.QCheckBox('Subtract offset')
        self.offset_box.setToolTip('Subtract the mean of each series '
                                   'shown (I and Q separately)')
        self.offset_box.toggled.connect(self._plot)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.psd_box)
        row.addWidget(self.offset_box)
        form.addRow('', row)
        lay.addWidget(box)
        self.rescale_btn = QtWidgets.QPushButton('Rescale plots (R)')
        self.rescale_btn.clicked.connect(self.rescale)
        lay.addWidget(self.rescale_btn)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setFixedHeight(int(90 * self._ui_scale))
        self.status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.status)
        freeze_heights(lay, skip = (self.runs_list,))
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidget(self._panel)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        width = self._panel.sizeHint().width() + int(40 * self._ui_scale)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            width = min(width, screen.availableGeometry().width() // 2)
        self._scroll.setFixedWidth(max(width, int(360 * self._ui_scale)))
        self._scroll.setMinimumHeight(300)

    def _build_plots(self):
        """
        Build the timestream, IQ and PSD plots (as the 4a window's).
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.ts_plot = self.plots.addPlot(row = 0, col = 0)
        self.ts_plot.setLabel('bottom', 'Time (s)')
        self.ts_plot.addLegend()
        self.iq_plot = self.plots.addPlot(row = 1, col = 0)
        self.iq_plot.setAspectLocked(True)
        self.iq_plot.setLabel('bottom', 'I')
        self.iq_plot.setLabel('left', 'Q')
        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setLogMode(x = True, y = True)
        self.psd_plot.setLabel('bottom', 'Frequency (Hz)')
        self.psd_plot.addLegend()
        self.psd_plot.setVisible(False)

    ############################################################################
    def _fill_tune_box(self, which = None):
        """
        List the store's tunings with runs and select one.

        Parameters:
        which (int, str or None): the tuning to select
            (``daq_layout.find_tuning``). None: the latest with runs.
        """
        self.tune_box.blockSignals(True)
        self.tune_box.clear()
        with_runs = []
        for k in dl.list_indices(self.root, 'tune'):
            tune = self.root[dl.set_name(k, 'tune')]
            if 'tones' in tune and dl.stream_runs(self.root, tune.basename):
                with_runs.append(tune)
                self.tune_box.addItem(
                    f'{dl.tuning_name(tune)} ({tune.basename})', tune.path)
        self.tune_box.blockSignals(False)
        if not with_runs:
            self.header.setText('No tuning has timestream runs.')
            return
        chosen = with_runs[-1]
        if which is not None:
            found = dl.find_tuning(self.root, which)
            if found is not None and found.path in [t.path for t in with_runs]:
                chosen = found
        self.tune_box.setCurrentIndex(self.tune_box.findData(chosen.path))
        self._on_tune_box()

    def _on_tune_box(self):
        """
        Show the tuning chosen in the box.
        """
        path = self.tune_box.currentData()
        if path is None:
            return
        self.set_tuning(self.root[path])

    def set_tuning(self, tune):
        """
        Show a tuning: its runs and tones.

        Parameters:
        tune (zarr.Group): the tuning, with ``tones``.
        """
        self.tune = tune
        self.tones = dl.load_tones(tune['tones'])
        self.view = None
        self.runs = dl.stream_runs(self.root, tune.basename)
        sets = sorted(set(self.tones['tone_set'].tolist()))
        self.header.setText(
            f'{dl.tuning_name(tune)} ({tune.basename}): '
            f"{len(self.tones['fres'])} tones in {len(sets)} tone set(s), "
            f'{len(self.runs)} run(s)')
        self.runs_list.blockSignals(True)
        self.runs_list.clear()
        for r in self.runs:
            self.runs_list.addItem(self._run_label(r))
        self.runs_list.blockSignals(False)
        self._fill_tones(np.arange(len(self.tones['fres'])))
        self._plot()
        if self.runs:
            self.runs_list.setCurrentRow(len(self.runs) - 1)

    @staticmethod
    def _run_label(run):
        """
        Describe a run for the list.

        Parameters:
        run (dict): a run (``daq_layout.stream_runs``).

        Returns:
        label (str): one line.
        """
        return (f"{run['group']}: {run.get('name')}, stage "
                f"{run.get('dec_stage')}, {run.get('duration_s', 0):g} s, "
                f"set {run.get('tone_set', run.get('tone_sets'))}, "
                f"{run.get('calibration', '')}, "
                f"{run.get('n_chunks') or 0} chunk(s), "
                f"{run.get('created', '')}"
                + (f", point {run['point']}" if run.get('point') else ''))

    def _fill_tones(self, rows):
        """
        List tones in the Tone box, keeping the selected one if listed.

        Parameters:
        rows (array-like int): indices into the tuning's tones.
        """
        current = self.tone_box.currentData()
        self.tone_box.blockSignals(True)
        self.tone_box.clear()
        for r in rows:
            self.tone_box.addItem(
                f"res {self.tones['res_idxs'][r]}: "
                f"{self.tones['fres'][r] / 1e6:.4f} MHz", int(r))
        if current is not None and self.tone_box.findData(current) >= 0:
            self.tone_box.setCurrentIndex(self.tone_box.findData(current))
        self.tone_box.blockSignals(False)

    def _on_run_selected(self, row):
        """
        Offer the selected run's chunks and tones.

        Parameters:
        row (int): the row in the runs list.
        """
        run = self.runs[row] if 0 <= row < len(self.runs) else None
        n = int(run.get('n_chunks', 0)) if run else 0
        self.start_box.setRange(0, max(n - 1, 0))
        self.start_box.setValue(0)
        self.count_box.setRange(1, max(n, 1))
        self.count_box.setValue(1)
        self.chunk_label.setText(
            f"of {n} ({run['chunk_s']:.3g} s each)" if n else
            ('' if run is None else 'no streams'))
        self.view_btn.setEnabled(bool(n))
        if run is not None:
            # One set's tones, or every tone of a run of several sets
            s = run.get('tone_set')
            self._fill_tones(np.arange(len(self.tones['fres'])) if s is None
                             else np.flatnonzero(self.tones['tone_set'] == s))

    def view_run(self):
        """
        Read the selected run's chunks and show them (the button, or a
        double-click).

        Returns:
        view (dict or None): ``load_run_view``, or None if nothing was
            read.
        """
        row = self.runs_list.currentRow()
        if not 0 <= row < len(self.runs):
            return None
        run = self.runs[row]
        if not run.get('n_chunks'):
            self.status.setText(f"{run['group']} has no streams to show.")
            return None
        start, k = int(self.start_box.value()), int(self.count_box.value())
        tone = self.tone_box.currentData()
        # A run of several tone sets: the set of the chosen tone
        s = run.get('tone_set')
        if s is None and tone is not None:
            s = int(self.tones['tone_set'][int(tone)])
        run = self.set_run(run, s)
        if run is None:
            self.status.setText(f'No streams of tone set {s} in this run.')
            return None
        set_rows = np.flatnonzero(self.tones['tone_set'] == s)
        self.status.setText(f"Reading {k} chunk(s) of {run['group']} from "
                            f'chunk {start}...')
        QtWidgets.QApplication.processEvents()
        try:
            view = run_responsive(load_run_view, self.root, run, set_rows, k,
                                  tone, start)
        except Exception as exc:
            self.status.setText(f'Could not read the run: {exc!r}')
            return None
        self.view = view
        if self.tone_box.currentData() not in view['rows']:
            self._fill_tones(view['rows'])
            self.tone_box.setCurrentIndex(
                self.tone_box.findData(int(view['rows'][0])))
        self.status.setText(
            f"Showing {view['label']}: chunks {view['start']} to "
            f"{view['start'] + view['chunks'][0] - 1} of {view['chunks'][1]}, "
            f"{view['zt'].shape[1]:,} samples "
            f"({view['zt'].shape[1] * view['dt']:.3g} s from "
            f"{view['t0']:.3g} s) of {len(view['rows'])} tones from "
            f"{view['source']}, on {view['calibration']}.")
        self._plot()
        return view

    def set_run(self, run, s):
        """
        Return a run's streams of one tone set as a run of that set
        (a run may stream several sets, each on its own sweep).

        Parameters:
        run (dict): the run (``daq_layout.stream_runs``).
        s (int or None): the tone set. None: the run as it is.

        Returns:
        run (dict or None): the run with ``paths``, ``calibration`` and
            ``tone_set`` of that set; None if it has no streams of it.
        """
        if s is None or run.get('tone_set') == s:
            return run
        try:
            sources, _ = dl.select_streams(self.root, run = run['run'],
                                           tuning = self.tune.basename,
                                           tone_sets = [int(s)])
        except (KeyError, ValueError):
            return None
        if not sources:
            return None
        src = sources[0]
        paths = [f"{src['group']}/{w}/{src.get('stream', dl.STREAM_NAME)}"
                 for w in src['writes']]
        return dict(run, paths = paths, calibration = src['group'],
                    tone_set = int(s))

    ############################################################################
    def _plot(self):
        """
        Plot the selected tone of the viewed run (``plot_view``).
        """
        r = self.tone_box.currentData()
        plot_view(self.ts_plot, self.iq_plot, self.psd_plot, self.view,
                  self.view, r, mode = self.mode_box.currentIndex(),
                  offset = self.offset_box.isChecked(),
                  show_psd = self.psd_box.isChecked(),
                  label = self.view['label'] if self.view else '')

    def _on_psd_box(self, show):
        """
        Show or hide the PSD plot and draw it (the box).

        Parameters:
        show (bool): the box.
        """
        self.psd_plot.setVisible(bool(show))
        self._plot()

    def rescale(self):
        """
        Fit the plots to their data again (the button, or R).
        """
        for plot in (self.ts_plot, self.iq_plot, self.psd_plot):
            plot.enableAutoRange()


def run_timestream_viewer(root, path = None, tuning = None,
                          title = 'Timestreams', ui_scale = 1.0):
    """
    Open the timestream viewer and wait until it is closed.

    Parameters:
    root, path, tuning, title, ui_scale: see ``TimestreamViewer``.

    Returns:
    win (TimestreamViewer): the window, closed.
    """
    app = get_qapp(title)
    win = TimestreamViewer(root, path = path, tuning = tuning, title = title,
                           ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win
