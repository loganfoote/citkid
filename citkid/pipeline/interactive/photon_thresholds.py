"""
Window for choosing the photon detection thresholds of one timestream.

The left panel lists the tones (resonator index, pulses in the height
window, rate) and holds the thresholds. The plots show, for the selected
tone or for all tones together:

- the first-pass peak heights, in units of their median, with the upper
  cut ``init_hmax``;
- the heights of the optimal filter peaks (the coincidence candidates, so
  heights below hmin are shown too), in units of the mean photon, with the
  expected noise peaks (the inverted filter output's peaks, scaled to the
  whole record) and the height window [hmin, hmax];
- the tone's mean pulse in the height window.

The threshold lines can be dragged. "Rerun" applies the thresholds either
to the selected tone (an override for that resonator) or to all tones (the
global values, used by every tone without an override), and runs the
detection again for the tones they change, through the ``rerun`` callable
(``pipeline.photon_finder.detect_timestream``). Close the window (or press
Done) to keep the thresholds and results.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
from ...qt_compat import (
    delete_on_close, fit_window_to_screen, get_qapp, run_responsive,
    widget_font_stylesheet
)

class PhotonThresholdWindow(QtWidgets.QMainWindow):
    """
    Photon threshold window (see the module description).

    Attributes:
    results (dict): row: detection result (dict) or error message (str).
    thresholds (Thresholds): the current thresholds.
    rows (list of int): the timestream's DataSet rows.
    res_idxs (list of int): their resonator indices.
    """

    def __init__(
        self, ts, results, thresholds, rerun, title = 'Photon thresholds',
        ui_scale = 1.0
    ):
        """
        Build the window.

        Parameters:
        ts (dict): the timestream (``photon_finder.list_timestreams``).
        results (dict): its ``detect_timestream`` results.
        thresholds (Thresholds): starting thresholds (changed in place).
        rerun (callable): function of (rows, thresholds) returning the new
            results of those rows (dict row: result), e.g. a wrapper of
            ``detect_timestream``.
        title (str): window title. Default 'Photon thresholds'.
        ui_scale (float): font size multiplier. Default 1.0.
        """
        super().__init__()
        self.setWindowTitle(title)
        if ui_scale and ui_scale != 1.0:
            self.setStyleSheet(widget_font_stylesheet(ui_scale))
        self.results = dict(results)
        self.thresholds = thresholds
        self.rows = [int(r) for r in ts['rows']]
        self.res_idxs = [int(r) for r in ts['res_idxs']]
        self.dt = float(ts['dt'])
        self._rerun = rerun
        self._updating = False
        self._build_controls()
        self._build_plots()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._panel)
        lay.addWidget(self.plots, stretch = 1)
        self.setCentralWidget(central)
        self._fill_list()
        self.tone_list.setCurrentRow(0)
        self.select(0)

    ############################################################################
    def _build_controls(self):
        """
        Build the left panel: tone list, thresholds, and buttons.
        """
        self._panel = QtWidgets.QWidget()
        self._panel.setMaximumWidth(330)
        lay = QtWidgets.QVBoxLayout(self._panel)
        self.tone_list = QtWidgets.QListWidget()
        self.tone_list.currentRowChanged.connect(self.select)
        lay.addWidget(QtWidgets.QLabel('Tones (* : own thresholds)'))
        lay.addWidget(self.tone_list, stretch = 1)
        self.all_tones = QtWidgets.QCheckBox('Plot all tones together')
        self.all_tones.toggled.connect(self.update_plots)
        lay.addWidget(self.all_tones)
        form = QtWidgets.QFormLayout()

        def spin(lo, hi, step):
            """
            Make a threshold spin box.

            Parameters:
            lo, hi (float): range.
            step (float): step.

            Returns:
            box (QDoubleSpinBox): the spin box.
            """
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(2)
            box.setSingleStep(step)
            box.valueChanged.connect(self._on_spin)
            return box

        self.hmin_box = spin(0.01, 100., 0.05)
        self.hmax_box = spin(0.01, 1000., 0.05)
        self.init_hmax_box = spin(0.1, 1000., 0.5)
        form.addRow('hmin (photons)', self.hmin_box)
        form.addRow('hmax (photons)', self.hmax_box)
        form.addRow('init_hmax (x median)', self.init_hmax_box)
        lay.addLayout(form)
        self.scope_tone = QtWidgets.QRadioButton('Apply to this tone')
        self.scope_all = QtWidgets.QRadioButton('Apply to all tones')
        self.scope_all.setChecked(True)
        lay.addWidget(self.scope_tone)
        lay.addWidget(self.scope_all)
        self.rerun_btn = QtWidgets.QPushButton('Rerun (Ctrl+R)')
        self.rerun_btn.clicked.connect(self.rerun)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, self.rerun)
        self.reset_btn = QtWidgets.QPushButton('Use the global thresholds')
        self.reset_btn.clicked.connect(self.reset_tone)
        self.done_btn = QtWidgets.QPushButton('Done')
        self.done_btn.clicked.connect(self.close)
        for b in (self.rerun_btn, self.reset_btn, self.done_btn):
            lay.addWidget(b)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def _build_plots(self):
        """
        Build the three plots and the draggable threshold lines.
        """
        self.plots = pg.GraphicsLayoutWidget()
        self.init_plot = self.plots.addPlot(row = 0, col = 0)
        self.init_plot.setLabel('bottom', 'First-pass height / median')
        self.init_plot.setLabel('left', 'Peaks')
        self.init_plot.setLogMode(y = True)
        self.h_plot = self.plots.addPlot(row = 1, col = 0)
        self.h_plot.setLabel('bottom', 'Filter height (photons)')
        self.h_plot.setLabel('left', 'Peaks')
        self.h_plot.setLogMode(y = True)
        self.h_plot.addLegend()
        self.pulse_plot = self.plots.addPlot(row = 2, col = 0)
        self.pulse_plot.setLabel('bottom', 'Time (ms)')
        self.pulse_plot.setLabel('left', 'Mean pulse (photons)')
        self.init_curve = self.init_plot.plot(stepMode = 'center')
        self.h_curve = self.h_plot.plot(stepMode = 'center', name = 'peaks')
        self.noise_curve = self.h_plot.plot(
            stepMode = 'center', pen = pg.mkPen('r'), name = 'noise')
        self.pulse_curve = self.pulse_plot.plot()

        def line(plot, color):
            """
            Add a draggable vertical line.

            Parameters:
            plot (PlotItem): the plot.
            color (str): pen color.

            Returns:
            line (InfiniteLine): the line.
            """
            ln = pg.InfiniteLine(0, movable = True,
                                 pen = pg.mkPen(color, width = 2))
            plot.addItem(ln)
            ln.sigPositionChanged.connect(self._on_line)
            return ln

        self.init_line = line(self.init_plot, 'y')
        self.hmin_line = line(self.h_plot, 'g')
        self.hmax_line = line(self.h_plot, 'g')

    ############################################################################
    def _fill_list(self):
        """
        Fill the tone list from the results.
        """
        current = self.tone_list.currentRow()
        self.tone_list.blockSignals(True)
        self.tone_list.clear()
        for row, r_idx in zip(self.rows, self.res_idxs):
            res = self.results.get(row)
            mark = '*' if r_idx in self.thresholds.overrides else ' '
            if isinstance(res, dict):
                text = (f"{mark} res {r_idx}: {len(res['events'])} pulses, "
                        f"{res['rate']:.3g}/s")
            else:
                text = f'{mark} res {r_idx}: failed'
            self.tone_list.addItem(text)
        self.tone_list.blockSignals(False)
        if current >= 0:
            self.tone_list.setCurrentRow(current)

    def current(self):
        """
        Return the selected tone.

        Returns:
        i (int): its index in ``rows``.
        """
        return max(self.tone_list.currentRow(), 0)

    def select(self, i):
        """
        Show tone i: its thresholds and plots.

        Parameters:
        i (int): index in ``rows``.
        """
        if i < 0:
            return
        th = self.thresholds.for_tone(self.res_idxs[i])
        self._set_values(th['hmin'], th['hmax'], th['init_hmax'])
        res = self.results.get(self.rows[i])
        if isinstance(res, dict):
            self.status.setText('\n'.join(res['log']))
        else:
            self.status.setText(f'Detection failed: {res}')
        self.update_plots()

    def _set_values(self, hmin, hmax, init_hmax):
        """
        Set the spin boxes and lines without triggering updates.

        Parameters:
        hmin, hmax (float): height window.
        init_hmax (float or None): first-pass cut.
        """
        self._updating = True
        self.hmin_box.setValue(hmin)
        self.hmax_box.setValue(hmax)
        self.init_hmax_box.setValue(1000. if init_hmax is None else
                                    init_hmax)
        self.hmin_line.setValue(hmin)
        self.hmax_line.setValue(hmax)
        self.init_line.setValue(self.init_hmax_box.value())
        self._updating = False

    def values(self):
        """
        Return the thresholds in the spin boxes.

        Returns:
        values (dict): 'hmin', 'hmax', 'init_hmax'.
        """
        return dict(hmin = self.hmin_box.value(),
                    hmax = self.hmax_box.value(),
                    init_hmax = self.init_hmax_box.value())

    def _on_spin(self):
        """
        Move the lines to the spin boxes' values.
        """
        if self._updating:
            return
        self._updating = True
        self.hmin_line.setValue(self.hmin_box.value())
        self.hmax_line.setValue(self.hmax_box.value())
        self.init_line.setValue(self.init_hmax_box.value())
        self._updating = False

    def _on_line(self):
        """
        Copy dragged lines to the spin boxes.
        """
        if self._updating:
            return
        self._updating = True
        self.hmin_box.setValue(self.hmin_line.value())
        self.hmax_box.setValue(self.hmax_line.value())
        self.init_hmax_box.setValue(self.init_line.value())
        self._updating = False

    ############################################################################
    def update_plots(self):
        """
        Draw the histograms and the mean pulse of the selected tone (or the
        histograms of all tones).
        """
        if self.all_tones.isChecked():
            chosen = [self.results.get(r) for r in self.rows]
        else:
            chosen = [self.results.get(self.rows[self.current()])]
        chosen = [r for r in chosen if isinstance(r, dict)]
        init, h, noise_h, noise_w = [], [], [], []
        for res in chosen:
            if len(res['init_h']):
                init.append(res['init_h'] / np.median(res['init_h']))
            h.append(res['coinc_h'])
            noise_h.append(res['neg_h'])
            noise_w.append(np.full(len(res['neg_h']),
                                   1 / max(res['clean_frac'], 1e-6)))
        self._hist(self.init_curve, init, np.linspace(0, 10, 201))
        top = max([self.hmax_box.value() * 1.5] +
                  [float(np.percentile(x, 99)) for x in h if len(x)])
        bins = np.linspace(0, top, 151)
        self._hist(self.h_curve, h, bins)
        self._hist(self.noise_curve, noise_h, bins, noise_w)
        if len(chosen) == 1 and len(chosen[0]['template']):
            res = chosen[0]
            t = (np.arange(len(res['template'])) - res['p']) * self.dt
            self.pulse_curve.setData(t * 1e3,
                                     res['template'] / res['amp_ph'])
        else:
            self.pulse_curve.setData([], [])

    @staticmethod
    def _hist(curve, parts, bins, weights = None):
        """
        Draw a step histogram (counts below 1 are drawn at 0.5 for the log
        axis).

        Parameters:
        curve (PlotDataItem): the curve.
        parts (list of np.array): values to histogram together.
        bins (np.array): bin edges.
        weights (list of np.array or None): weights. Default None.
        """
        if not parts:
            curve.setData([], [])
            return
        x = np.concatenate(parts)
        w = None if weights is None else np.concatenate(weights)
        counts, _ = np.histogram(x, bins = bins, weights = w)
        curve.setData(bins, np.maximum(counts, 0.5))

    ############################################################################
    def rerun(self):
        """
        Apply the thresholds (to the selected tone or to all tones) and run
        the detection again where they changed.
        """
        vals = self.values()
        th = self.thresholds
        if self.scope_tone.isChecked():
            r_idx = self.res_idxs[self.current()]
            th.overrides[r_idx] = vals
            rows = [self.rows[self.current()]]
        else:
            th.hmin, th.hmax = vals['hmin'], vals['hmax']
            th.init_hmax = vals['init_hmax']
            rows = [r for r, ri in zip(self.rows, self.res_idxs)
                    if ri not in th.overrides]
        self.status.setText(f'Running {len(rows)} tones...')
        new = run_responsive(self._rerun, rows, th)
        self.results.update(new)
        self._fill_list()
        self.select(self.current())

    def reset_tone(self):
        """
        Remove the selected tone's own thresholds and run it again with the
        global ones.
        """
        r_idx = self.res_idxs[self.current()]
        if self.thresholds.overrides.pop(r_idx, None) is None:
            return
        row = self.rows[self.current()]
        self.results.update(run_responsive(self._rerun, [row],
                                           self.thresholds))
        self._fill_list()
        self.select(self.current())

def run_photon_thresholds(
    ts, results, thresholds, rerun, title = 'Photon thresholds',
    ui_scale = 1.0
):
    """
    Open the photon threshold window and wait until it is closed.

    Parameters:
    ts, results, thresholds, rerun, title, ui_scale: see
        ``PhotonThresholdWindow``.

    Returns:
    thresholds (Thresholds): the chosen thresholds.
    results (dict): the detection results with them.
    """
    app = get_qapp(title)
    win = PhotonThresholdWindow(ts, results, thresholds, rerun, title = title,
                                ui_scale = ui_scale)
    delete_on_close(win)
    fit_window_to_screen(win)
    win.show()
    (getattr(app, 'exec', None) or app.exec_)()
    return win.thresholds, win.results
