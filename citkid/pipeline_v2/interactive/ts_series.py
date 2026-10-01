"""
Interactive TS analysis across a parameter series.

Like the IQ series window, but each series point runs the standard ``'ts'``
analysis (gain fit, IQ circle fit, x calibration) instead of IQ fitting, and
the series plot can show any scalar output of that analysis on x and y,
chosen from two dropdowns (e.g. the reduced PSD ``sxx_10`` vs power
``ares``).

For each resonator (data_idx), displays:
  - Left:  the selected y vs x for every series index, and a ``|S21|``
           waterfall.
  - Right: the TS panels (GainFitPanel, CircleFitPanel, XCalPanel) for the
           selected series index.

Usage
-----
::

    import zarr
    from citkid.pipeline_v2.interactive import run_ts_series

    def make_custom_steps(series_idx):
        # Return list[plStep] that load data for this series index.
        ...

    run_ts_series(
        make_custom_steps=make_custom_steps,
        root=zarr.open_group('ts_series.zarr', 'a'),
        n_series=7,
        x='ares',
        y='sxx_10',
    )

All keyboard shortcuts and buttons of the IQ series window apply (see
``iq_series``), except Mark Bad Above (Ctrl+Shift+B), which is for IQ
series only.
"""

from ...xcal import reduced_params
from ...qt_compat import get_qapp
from . import gain  # noqa: F401 — registers GainFitPanel
from . import circ  # noqa: F401 — registers CircleFitPanel
from . import xcal  # noqa: F401 — registers XCalPanel
from .iq_series import (
    QtWidgets,
    IQSeriesWindow,
    SeriesValues,
    _check_series_values,
    dataset_quantity,
    series_runners,
)

# Step grouping that matches ts_analysis.yaml
_TS_PANELS = [
    ('fit_gain',),
    ('fit_iq_circle', 'get_idx_t', 'get_theta_phase_offset'),
    ('get_xcal_mask', 'fit_x_theta'),
]

# Scalar per-resonator quantities produced by the 'ts' calibration and
# analysis, offered in the x and y dropdowns by default.
DEFAULT_TS_QUANTITIES = (
    ['ares', 'fres', 'qres', 'ft', 'circ_radius', 'theta_phase_offset', 'idx_t']
    + [f'sxx_{freq}'.replace('.', 'p') for freq in reduced_params._freqs]
    + [f'sfactor_{freq}'.replace('.', 'p') for freq in reduced_params._freqs]
)


def normalize_quantities(quantities=None):
    """
    Turn a quantity specification into getters for the series plot.

    Parameters:
    quantities (list, dict, or None): Quantities to offer. A list of
        DataSet parameter names (scalar per resonator, or global), or a dict
        mapping a label to a parameter name, a callable
        ``f(AR, data_idx) -> float or None`` (for derived quantities), or a
        ``SeriesValues`` (one fixed value per series point). None (default)
        uses ``DEFAULT_TS_QUANTITIES``.

    Returns:
    getters (dict): Label -> ``getter(AR, data_idx)`` or ``SeriesValues``,
        in the given order.

    Raises:
    ValueError: If no quantities are given.
    TypeError: If a dict value is not a string, callable, or SeriesValues.
    """
    if quantities is None:
        quantities = DEFAULT_TS_QUANTITIES
    if not isinstance(quantities, dict):
        quantities = {str(name): str(name) for name in quantities}
    getters = {}
    for label, source in quantities.items():
        if isinstance(source, SeriesValues):
            getters[str(label)] = source
        elif isinstance(source, str):
            getters[str(label)] = dataset_quantity(source)
        elif callable(source):
            getters[str(label)] = _float_or_none(source)
        else:
            raise TypeError(
                f"quantity '{label}' must be a parameter name, a callable, or SeriesValues; "
                f"got {source!r}")
    if not getters:
        raise ValueError('quantities must contain at least one quantity')
    return getters


def _float_or_none(func):
    """
    Wrap a user quantity function so it returns a float or None.

    Parameters:
    func (callable): ``func(AR, data_idx)``.

    Returns:
    getter (callable): Same call, returning None if ``func`` raises or
        returns something that isn't a finite number.
    """
    def getter(AR, data_idx):
        """
        Call ``func`` and convert its result.
        """
        try:
            value = func(AR, data_idx)
            if value is None:
                return None
            value = float(value)
        except Exception:
            return None
        return value if value == value and abs(value) != float('inf') else None

    getter.__name__ = getattr(func, '__name__', 'quantity')
    return getter


class TSSeriesWindow(IQSeriesWindow):
    """
    Series window for the 'ts' analysis, with selectable x and y.

    The background worker pre-fits every resonator for every series index and
    then computes the selected x and y, since some (e.g. reduced PSDs) are
    slow. Changing a dropdown keeps already-computed values, and the worker
    then computes the new quantity for every resonator.

    Parameters:
    ARs (list of AnalysisRunner): One per series index, using the 'ts'
        calibration and analysis.
    x (str or None): Label of the initial x quantity. None (default) uses
        the ``x_values`` quantity if given, else 'ares'.
    y (str): Label of the initial y quantity. Default 'sxx_10'.
    quantities (list, dict, or None): Quantities offered on x and y. See
        ``normalize_quantities``. None (default) uses
        ``DEFAULT_TS_QUANTITIES``.
    x_values (array-like, SeriesValues, or None): One fixed x value per
        series point (the same for every resonator), e.g. the drive power
        of each DataSet when it isn't stored in it. Added to the dropdowns
        as ``x_name``. None (default) adds nothing.
    x_name (str or None): Dropdown label of ``x_values``. None (default)
        uses 'x'.
    **kwargs: Other ``IQSeriesWindow`` arguments (``start_series_idx``,
        ``start_idx``, ``data_idxs``, ``title``, ``ui_scale``,
        ``plot_scale``, ``parent``, ``state_group``, ``xy_fit``,
        ``background_fitting``).

    Raises:
    ValueError: If ``x`` or ``y`` is not one of the quantities, or series
        values don't have one entry per series point.
    """

    # Mark Bad Above is for IQ series only.
    _MARK_BAD_ABOVE = False

    def __init__(self, ARs, x=None, y='sxx_10', quantities=None, x_values=None, x_name=None,
                 **kwargs):
        """
        Build the window. See the class docstring for parameters.
        """
        # Set before the base class builds the toolbar, which adds the
        # dropdowns through _add_toolbar_controls.
        self._quantities = normalize_quantities(quantities)
        if x_values is not None:
            label = 'x' if x_name is None else x_name
            self._quantities[label] = (
                x_values if isinstance(x_values, SeriesValues) else SeriesValues(x_values))
            if x is None:
                x = label
        if x is None:
            x = 'ares'
        for getter in self._quantities.values():
            _check_series_values(getter, len(ARs))
        for label in (x, y):
            if label not in self._quantities:
                raise ValueError(
                    f"'{label}' is not one of the quantities {list(self._quantities)}")
        self._initial_x, self._initial_y = x, y
        kwargs.setdefault('title', 'TS Series')
        super().__init__(
            ARs,
            x_param_name=x,
            x_name=x,
            y_func=self._quantities[y],
            y_name=y,
            panels=_TS_PANELS,
            precompute_values=True,
            **kwargs,
        )
        self._x_getter = self._quantities[x]
        self._x_key, self._y_key = x, y

    def _add_toolbar_controls(self, layout):
        """
        Add the x and y quantity dropdowns to the first toolbar row.

        Parameters:
        layout (QHBoxLayout): First toolbar row.
        """
        labels = list(self._quantities)
        self._x_combo = QtWidgets.QComboBox()
        self._y_combo = QtWidgets.QComboBox()
        for combo, initial, axis in ((self._x_combo, self._initial_x, 'x'),
                                     (self._y_combo, self._initial_y, 'y')):
            combo.addItems(labels)
            combo.setCurrentText(initial)
            combo.setToolTip(f'Quantity on the series plot {axis} axis')
            combo.currentTextChanged.connect(self._on_quantity_changed)
            layout.addSpacing(8)
            layout.addWidget(QtWidgets.QLabel(f'{axis}:'))
            layout.addWidget(combo)

    def _on_quantity_changed(self, _text=None):
        """
        Show the quantities selected in the dropdowns.

        Parameters:
        _text (str or None): New dropdown text (unused; both are read).
        """
        x = self._x_combo.currentText()
        y = self._y_combo.currentText()
        if (x, y) == (self._x_key, self._y_key):
            return
        self.set_quantities(x, self._quantities[x], x, y, self._quantities[y], y)

    def select_quantities(self, x=None, y=None):
        """
        Select the plotted quantities from code (same as the dropdowns).

        Parameters:
        x (str or None): Label of the x quantity, or None to keep it.
        y (str or None): Label of the y quantity, or None to keep it.

        Raises:
        ValueError: If a label is not one of the quantities.
        """
        for label in (x, y):
            if label is not None and label not in self._quantities:
                raise ValueError(
                    f"'{label}' is not one of the quantities {list(self._quantities)}")
        if x is not None:
            self._x_combo.setCurrentText(x)
        if y is not None:
            self._y_combo.setCurrentText(y)


def run_ts_series(
    make_custom_steps=None,
    root=None,
    n_series=None,
    x=None,
    y='sxx_10',
    quantities=None,
    cal_yaml_path='ts',
    analysis_yaml_path='ts',
    start_series_idx=0,
    start_idx=None,
    data_idxs=None,
    title='TS Series',
    ui_scale=1.0,
    plot_scale=1.0,
    xy_fit=None,
    datasets=None,
    x_values=None,
    x_name=None,
):
    """
    Build one AnalysisRunner per series index for the 'ts' analysis, then
    launch the TSSeriesWindow.

    Pass either ``make_custom_steps`` (with ``root`` and ``n_series``), to
    create one DataSet per series point, or ``datasets``, to use DataSets you
    already have.

    Parameters:
    make_custom_steps (callable or None): ``make_custom_steps(series_idx) ->
        list[plStep]`` calibration steps that load one series point's data
        (as for ``custom_steps_template.py``: global data, global-res data,
        ``ft``, ``zt``, and the fine and gain sweeps). Each point's DataSet is
        stored in ``root/series_{i:03d}``. None if ``datasets`` is given.
    root (zarr.Group or None): Parent zarr group for the series points (with
        ``make_custom_steps``) and for the session state
        (``root.attrs['series_state']``) and xy fits. With ``datasets`` it is
        optional: None keeps the session state in memory only.
    n_series (int or None): Number of series indices. Required with
        ``make_custom_steps``; with ``datasets``, defaults to (and must equal)
        ``len(datasets)``.
    x (str or None): Label of the initial x quantity. None (default) uses
        the ``x_values`` quantity if given, else 'ares'.
    y (str): Label of the initial y quantity. Default 'sxx_10'.
    quantities (list, dict, or None): Quantities offered in the x and y
        dropdowns: parameter names, or a dict of label -> parameter name,
        ``f(AR, data_idx)``, or ``SeriesValues``. None (default) uses
        ``DEFAULT_TS_QUANTITIES``.
    x_values (array-like, SeriesValues, or None): One fixed x value per
        series point (the same for every resonator), e.g. the drive power
        of each DataSet when it isn't stored in it. Added to the dropdowns
        as ``x_name``. None (default) adds nothing.
    x_name (str or None): Dropdown label of ``x_values``. None (default)
        uses 'x'.
    cal_yaml_path (str): Calibration YAML path or alias, for
        ``make_custom_steps``. Default 'ts'.
    analysis_yaml_path (str or None): Analysis YAML path or alias. Default
        'ts'. With ``datasets``, None uses each DataSet's embedded analysis
        definition.
    start_series_idx (int): Initial series index. Default 0.
    start_idx (int or None): Position in ``data_idxs`` to start at. None
        (default) resumes at the first entry not viewed in an earlier
        session.
    data_idxs (list of int or None): Ordered data indices to review. None
        (default) uses every row.
    title (str): Window title. Default 'TS Series'.
    ui_scale (float): Font and widget size multiplier. Default 1.0.
    plot_scale (float): Plot area height multiplier. Default 1.0.
    xy_fit (SeriesXYFit or None): Optional fit of the plotted y vs x, one per
        resonator. It always fits the currently selected quantities. None
        (default) disables fitting.
    datasets (list of DataSet or None): One existing 'ts' DataSet per series
        point, in order, all with the same number of rows. The background
        worker opens its own copies (``DataSet.copy``). None if
        ``make_custom_steps`` is given.

    Returns:
    win (TSSeriesWindow): The window (after it is closed).

    Raises:
    ValueError: If ``x`` or ``y`` is not one of the quantities, or the series
        inputs are inconsistent (see ``iq_series.series_runners``).
    TypeError: If ``datasets`` contains something other than DataSets.
    """
    ARs = series_runners(make_custom_steps, datasets, cal_yaml_path, analysis_yaml_path,
                         root, n_series)
    app = get_qapp(title)
    win = TSSeriesWindow(
        ARs,
        x=x,
        y=y,
        quantities=quantities,
        x_values=x_values,
        x_name=x_name,
        start_series_idx=start_series_idx,
        start_idx=start_idx,
        data_idxs=data_idxs,
        title=title,
        ui_scale=ui_scale,
        plot_scale=plot_scale,
        state_group=root,
        xy_fit=xy_fit,
        background_fitting=True,
    )
    win.show()
    app.exec()
    return win
