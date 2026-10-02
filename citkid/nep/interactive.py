"""
Interactive photon-noise NEP fitting: step through sets of NEP vs incident
power, choose the minimum power of the photon-noise dominated regime for each,
and fit the optical efficiency.

The fit itself is ``fitter.fit_nep_photon``; this module only holds UI state.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..pipeline_v2.interactive.core import scale_plot_fonts, widget_font_stylesheet
from ..qt_compat import TITLE_BAR_MARGIN, delete_on_close, fit_window_to_screen, get_qapp
from .batch import check_sets, per_set_values
from .fitter import fit_nep_photon
from .funcs import nep_photon
from .store import NEPFitStore


def _confirm_overwrite_nep_fits(message):
    """
    Ask whether to overwrite saved NEP fits of different data.

    Parameters:
    message (str): description of the saved fits, shown in the popup.

    Returns:
    overwrite (bool): True if the user chose Overwrite, False if they chose
        Cancel or closed the popup.
    """
    box = QtWidgets.QMessageBox()
    box.setWindowTitle('Existing NEP Fits Found')
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setText(message + '\n\nOverwrite them with the new fits, or cancel?')
    overwrite_btn = box.addButton('Overwrite', QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = box.addButton('Cancel', QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel_btn)
    (getattr(box, 'exec', None) or box.exec_)()
    return box.clickedButton() is overwrite_btn


class NEPFitWindow(QtWidgets.QMainWindow):
    """
    Window for fitting the photon-noise NEP of several sets, one at a time.

    Each set is shown as NEP vs incident power on log axes: fitted points in
    blue, excluded points in grey, the fitted photon-noise NEP in red, and the
    minimum fitted power ``p_min`` as a draggable dashed line (also set with
    the spin box). The fit is redone whenever ``p_min`` changes.

    Parameters:
    powers (list of array-like): incident powers of each set (W).
    neps (list of array-like): NEPs of each set, referred to incident power
        (W / Hz^0.5). NaN points are ignored; a set with no usable points has
        NaN fit values.
    nu (float): photon frequency (Hz).
    p_min (float, array-like, or None): initial minimum fitted power, one for
        every set or one per set. None (default) or 0 fits every point. With
        a single value, a ``p_min`` chosen on one set carries over to sets not
        yet visited.
    nep_errs (list of array-like or None): NEP uncertainties of each set, or
        None (default) to weight points equally (see ``fit_nep_photon``).
    names (list of str or None): set names shown in the toolbar. None
        (default) uses the set indices.
    group (zarr.Group or None): group to save each set's ``p_min``, bad flag,
        viewed flag and fit results to, so a later session resumes where this
        one stopped. None (default) keeps them in memory only.
    start_idx (int or None): set to start at. None (default) starts at the
        first set not yet viewed (or 0).
    title (str): window title. Default 'NEP Fit'.
    ui_scale (float): font and widget size multiplier. Default 1.0.
    parent (QWidget or None): parent widget.

    Raises:
    ValueError: if the inputs don't have one entry per set, or there are no
        sets.
    RuntimeError: if ``group`` holds fits of different data and the user
        cancels the overwrite popup.

    Notes:
    Shortcuts: ←/A and →/D previous/next set, B mark the set bad (or unmark
    it), Shift+A apply this set's ``p_min`` to every set, R rescale. A set
    counts as viewed once you leave it, or if it is open when the window
    closes.
    """

    def __init__(self, powers, neps, nu, p_min=None, nep_errs=None, names=None, group=None,
                 start_idx=None, title='NEP Fit', ui_scale=1.0, parent=None):
        """
        Build the window. See the class docstring for parameters.
        """
        check_sets(powers, neps, nep_errs)
        n_sets = len(neps)
        if n_sets == 0:
            raise ValueError('There must be at least one set')
        super().__init__(parent)
        delete_on_close(self)
        self._n_sets = n_sets
        self._nu = float(nu)
        self._powers = [np.asarray(p, dtype=float).ravel() for p in powers]
        self._neps = [np.asarray(n, dtype=float).ravel() for n in neps]
        self._nep_errs = (None if nep_errs is None
                          else [np.asarray(e, dtype=float).ravel() for e in nep_errs])
        self._names = ([str(i) for i in range(n_sets)] if names is None
                       else [str(n) for n in per_set_values(names, n_sets, 'names')])
        self._carry_p_min = np.ndim(p_min) == 0
        self._p_min = np.array([0.0 if v is None else float(v)
                                for v in per_set_values(p_min, n_sets, 'p_min')])
        self._bad = np.zeros(n_sets, dtype=bool)
        self._viewed = np.zeros(n_sets, dtype=bool)
        # Sets whose p_min the user has chosen, so it is saved and not carried over.
        self._touched = np.zeros(n_sets, dtype=bool)
        self._eta = np.full(n_sets, np.nan)
        self._eta_err = np.full(n_sets, np.nan)
        self._n_fit = np.zeros(n_sets, dtype=int)
        self._masks = [np.zeros(len(n), dtype=bool) for n in self._neps]

        self._store = None
        if group is not None:
            store = NEPFitStore(group, n_sets, self._nu, self._names)
            if not store.matches_definition():
                if not _confirm_overwrite_nep_fits(store.describe_existing()):
                    raise RuntimeError('User cancelled operation')
                store.clear()
            self._store = store
            self._load_state()

        for i in range(n_sets):
            self._refit(i)
        if start_idx is None:
            unviewed = np.flatnonzero(~self._viewed)
            start_idx = int(unviewed[0]) if len(unviewed) else 0
        self._idx = max(0, min(int(start_idx), n_sets - 1))
        self._updating = False  # set while widgets are updated programmatically

        self._ui_scale = ui_scale
        self._build_ui()
        self.setWindowTitle(title)
        self._show_set()
        self.ensurePolished()
        self.layout().activate()
        fit_window_to_screen(self, frac=0.9, size=(round(900 * ui_scale), round(650 * ui_scale)),
                             height_margin=TITLE_BAR_MARGIN)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _load_state(self):
        """
        Restore p_min, bad and viewed flags of the sets saved in the group.
        """
        fields = self._store.load()
        if fields is None:
            return
        saved = np.asarray(fields['row_exists'], dtype=bool)
        self._p_min[saved] = np.asarray(fields['p_min'], dtype=float)[saved]
        self._bad[saved] = np.asarray(fields['bad'], dtype=bool)[saved]
        self._viewed[saved] = np.asarray(fields['viewed'], dtype=bool)[saved]
        self._touched[saved] = True

    def _save(self):
        """
        Save every set's state and fit results to the group, if there is one.
        """
        if self._store is None:
            return
        self._store.save({
            'p_min': self._p_min, 'eta': self._eta, 'eta_err': self._eta_err,
            'n_fit': self._n_fit, 'bad': self._bad, 'viewed': self._viewed,
            'row_exists': self._touched | self._viewed | self._bad,
        })

    def _refit(self, i):
        """
        Fit one set at its current p_min (NaN results if it is marked bad).

        Parameters:
        i (int): set index.
        """
        if self._bad[i]:
            self._eta[i], self._eta_err[i], self._n_fit[i] = np.nan, np.nan, 0
            self._masks[i] = np.zeros(len(self._neps[i]), dtype=bool)
            return
        err = None if self._nep_errs is None else self._nep_errs[i]
        eta, eta_err, mask, n_fit = fit_nep_photon(
            self._powers[i], self._neps[i], self._nu, self._p_min[i], err)
        self._eta[i], self._eta_err[i], self._n_fit[i] = eta, eta_err, n_fit
        self._masks[i] = mask

    @property
    def results(self):
        """
        Fit results of every set (copies).

        Returns:
        p_min (np.array): minimum fitted power of each set (W); 0 means every
            point was fit.
        eta (np.array): optical efficiency of each set, NaN for bad sets and
            sets without usable points.
        eta_err (np.array): uncertainty in eta of each set.
        n_fit (np.array): number of points fit in each set.
        bad (np.array): True for sets marked bad.
        """
        return (self._p_min.copy(), self._eta.copy(), self._eta_err.copy(),
                self._n_fit.copy(), self._bad.copy())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        """
        Build the toolbar, plot and shortcuts.
        """
        central = QtWidgets.QWidget()
        central.setStyleSheet(widget_font_stylesheet(self._ui_scale))
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        self._prev_btn = QtWidgets.QPushButton('◀')
        self._prev_btn.clicked.connect(lambda: self._go(-1))
        self._idx_spin = QtWidgets.QSpinBox()
        self._idx_spin.setRange(0, self._n_sets - 1)
        self._idx_spin.valueChanged.connect(self._on_idx_spin)
        self._next_btn = QtWidgets.QPushButton('▶')
        self._next_btn.clicked.connect(lambda: self._go(1))
        self._name_label = QtWidgets.QLabel()
        self._p_min_spin = pg.SpinBox(value=0.0, dec=True, step=0.1, minStep=1e-30,
                                      bounds=(0, None), siPrefix=True, suffix='W')
        self._p_min_spin.setMinimumWidth(round(110 * self._ui_scale))
        self._p_min_spin.setToolTip('Minimum fitted power (0 fits every point)')
        self._p_min_spin.sigValueChanged.connect(self._on_p_min_spin)
        self._bad_btn = QtWidgets.QPushButton('Mark bad')
        self._bad_btn.setCheckable(True)
        self._bad_btn.clicked.connect(self._toggle_bad)
        self._all_btn = QtWidgets.QPushButton('p_min to all')
        self._all_btn.setToolTip("Apply this set's p_min to every set")
        self._all_btn.clicked.connect(self._apply_p_min_to_all)
        for widget in (self._prev_btn, QtWidgets.QLabel('set:'), self._idx_spin,
                       self._next_btn, self._name_label):
            bar.addWidget(widget)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel('p_min:'))
        bar.addWidget(self._p_min_spin)
        bar.addWidget(self._bad_btn)
        bar.addWidget(self._all_btn)
        bar.addSpacing(12)
        hints = QtWidgets.QLabel('[←/A  →/D] navigate    [B] bad    '
                                 '[⇧A] p_min to all    [R] rescale    drag line: p_min')
        hints.setWordWrap(True)
        hints.setStyleSheet('color: palette(mid); font-style: italic;')
        bar.addWidget(hints, 1)
        layout.addLayout(bar)

        self._plot_widget = pg.PlotWidget()
        self._plot = self._plot_widget.getPlotItem()
        self._plot.setLogMode(x=True, y=True)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel('bottom', 'Incident power (W)')
        self._plot.setLabel('left', 'NEP (W/√Hz)')
        scale_plot_fonts(self._ui_scale, self._plot)
        # PlotDataItems (not ScatterPlotItems) follow the plot's log mode.
        self._excluded = self._plot.plot(pen=None, symbol='o', symbolSize=8, symbolPen=None,
                                         symbolBrush=pg.mkBrush(150, 150, 150, 200))
        self._fitted = self._plot.plot(pen=None, symbol='o', symbolSize=8, symbolPen=None,
                                       symbolBrush=pg.mkBrush(100, 180, 255, 230))
        self._curve = self._plot.plot(pen=pg.mkPen((255, 80, 80), width=2))
        # The line lives in the view's log10 coordinates.
        self._line = pg.InfiniteLine(angle=90, movable=True,
                                     pen=pg.mkPen('w', width=1.5, style=QtCore.Qt.DashLine))
        self._line.sigPositionChanged.connect(self._on_line_moved)
        self._plot.addItem(self._line, ignoreBounds=True)
        layout.addWidget(self._plot_widget, 1)

        for keys, slot in ((('Left', 'A'), lambda: self._go(-1)),
                           (('Right', 'D'), lambda: self._go(1)),
                           (('B',), self._toggle_bad),
                           (('Shift+A',), self._apply_p_min_to_all),
                           (('R',), self._plot.autoRange)):
            for key in keys:
                QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)

    def _show_set(self, rescale=True):
        """
        Show the current set: toolbar values, line, plot and fit.

        Parameters:
        rescale (bool): If True (default), auto-range the plot.
        """
        i = self._idx
        self._updating = True
        try:
            self._idx_spin.setValue(i)
            self._name_label.setText(self._names[i])
            self._p_min_spin.setValue(self._p_min[i])
            self._bad_btn.setChecked(bool(self._bad[i]))
        finally:
            self._updating = False
        self._redraw()
        if rescale:
            self._plot.autoRange()

    def _redraw(self):
        """
        Redraw the current set's points, fit curve, p_min line and title.
        """
        i = self._idx
        power, nep, mask = self._powers[i], self._neps[i], self._masks[i]
        with np.errstate(invalid='ignore'):
            drawable = np.isfinite(power) & np.isfinite(nep) & (power > 0) & (nep > 0)
        self._fitted.setData(power[mask], nep[mask])
        self._excluded.setData(power[drawable & ~mask], nep[drawable & ~mask])
        positive = power[drawable]
        if np.isfinite(self._eta[i]) and len(positive) > 1:
            ps = np.geomspace(positive.min(), positive.max(), 200)
            self._curve.setData(ps, nep_photon(ps, self._eta[i], self._nu))
        else:
            self._curve.setData([], [])
        line_power = self._p_min[i] if self._p_min[i] > 0 else (
            positive.min() if len(positive) else np.nan)
        self._updating = True
        try:
            if np.isfinite(line_power):
                self._line.setValue(np.log10(line_power))
                self._line.show()
            else:
                self._line.hide()
        finally:
            self._updating = False
        self._plot.setTitle(self._title_text(i))

    def _title_text(self, i):
        """
        Describe one set's fit for the plot title.

        Parameters:
        i (int): set index.

        Returns:
        text (str): set position, name, and eta or why there is none.
        """
        head = f'set {i + 1}/{self._n_sets} ({self._names[i]})'
        if self._bad[i]:
            return f'{head}: marked bad'
        if not np.isfinite(self._eta[i]):
            return f'{head}: no usable points'
        err = '' if not np.isfinite(self._eta_err[i]) else f' ± {self._eta_err[i]:.2g}'
        return f'{head}: η = {self._eta[i]:.4g}{err}  (n = {self._n_fit[i]})'

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _set_p_min(self, value):
        """
        Set the current set's p_min and refit it.

        Parameters:
        value (float): minimum fitted power (W); 0 fits every point.
        """
        i = self._idx
        self._p_min[i] = max(float(value), 0.0)
        self._touched[i] = True
        self._refit(i)
        self._redraw()

    def _on_p_min_spin(self, spin):
        """
        Refit after the p_min spin box changes.

        Parameters:
        spin (pg.SpinBox): the spin box.
        """
        if not self._updating:
            self._set_p_min(spin.value())

    def _on_line_moved(self, line):
        """
        Refit after the p_min line is dragged, and update the spin box.

        Parameters:
        line (pg.InfiniteLine): the line (at log10 of the power).
        """
        if self._updating:
            return
        value = 10.0 ** line.value()
        self._updating = True
        try:
            self._p_min_spin.setValue(value)
        finally:
            self._updating = False
        self._set_p_min(value)

    def _on_idx_spin(self, value):
        """
        Go to the set chosen in the set spin box.

        Parameters:
        value (int): set index.
        """
        if not self._updating:
            self._go_to(int(value))

    def _go(self, delta):
        """
        Move to a neighbouring set.

        Parameters:
        delta (int): number of sets to move (negative moves back).
        """
        self._go_to(max(0, min(self._n_sets - 1, self._idx + delta)))

    def _go_to(self, new_idx):
        """
        Leave the current set (marking it viewed and saving) and show another.

        With a single initial p_min, a set not yet visited takes the p_min of
        the set being left.

        Parameters:
        new_idx (int): set index.
        """
        if new_idx == self._idx:
            return
        old = self._idx
        self._viewed[old] = True
        if self._carry_p_min and not (self._touched[new_idx] or self._viewed[new_idx]):
            self._p_min[new_idx] = self._p_min[old]
            self._refit(new_idx)
        self._save()
        self._idx = new_idx
        self._show_set()

    def _toggle_bad(self):
        """
        Mark the current set bad (NaN results), or unmark it.
        """
        i = self._idx
        self._bad[i] = not self._bad[i]
        self._touched[i] = True
        self._refit(i)
        self._show_set(rescale=False)
        self._save()

    def _apply_p_min_to_all(self):
        """
        Apply the current set's p_min to every set and refit them.
        """
        value = self._p_min[self._idx]
        self._p_min[:] = value
        self._touched[:] = True
        for i in range(self._n_sets):
            self._refit(i)
        self._show_set(rescale=False)
        self._save()

    def closeEvent(self, event):
        """
        Mark the open set viewed and save before closing.

        Parameters:
        event (QCloseEvent): close event.
        """
        self._viewed[self._idx] = True
        self._save()
        super().closeEvent(event)


def run_nep_fit(powers, neps, nu, p_min=None, nep_errs=None, names=None, group=None,
                start_idx=None, title='NEP Fit', ui_scale=1.0):
    """
    Open the NEP fit window, wait until it is closed, and return the results.

    Parameters:
    powers (list of array-like): incident powers of each set (W).
    neps (list of array-like): NEPs of each set, referred to incident power
        (W / Hz^0.5). NaN points are ignored.
    nu (float): photon frequency (Hz).
    p_min (float, array-like, or None): initial minimum fitted power, one for
        every set or one per set. None (default) or 0 fits every point.
    nep_errs (list of array-like or None): NEP uncertainties of each set, or
        None (default) to weight points equally.
    names (list of str or None): set names. None (default) uses the indices.
    group (zarr.Group or None): group to save the fits and session state to
        (see ``NEPFitWindow``). None (default) keeps them in memory only.
    start_idx (int or None): set to start at. None (default) resumes at the
        first set not yet viewed.
    title (str): window title. Default 'NEP Fit'.
    ui_scale (float): font and widget size multiplier. Default 1.0.

    Returns:
    p_min (np.array): minimum fitted power of each set (W); 0 means every
        point was fit.
    eta (np.array): optical efficiency of each set, NaN for bad sets and sets
        without usable points.
    eta_err (np.array): uncertainty in eta of each set.
    n_fit (np.array): number of points fit in each set.
    bad (np.array): True for sets marked bad.

    Raises:
    ValueError: if the inputs don't have one entry per set.
    RuntimeError: if ``group`` holds fits of different data and the user
        cancels the overwrite popup.
    """
    app = get_qapp(title)
    win = NEPFitWindow(powers, neps, nu, p_min=p_min, nep_errs=nep_errs, names=names,
                       group=group, start_idx=start_idx, title=title, ui_scale=ui_scale)
    win.show()
    app.exec()
    return win.results
