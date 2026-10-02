"""
Quick matplotlib plots of one row of a pipeline_v2 DataSet, for fast checks.

``plot`` draws one plot type and ``plot_full_cal`` draws every available plot
type into one figure. Both are also available as ``DataSet.plot`` and
``DataSet.plot_full_cal``.

Speed: each plot type reads only its own parameters for one row, timestreams
are thinned to ``max_points`` points (5000 by default, keeping sparse tails,
see ``signal.iq.density_subsample_idx``), and ``plot_full_cal`` draws into a
single figure rather than rendering and stitching separate ones.
"""

from collections import namedtuple
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from ..xcal.plot import (
    plot_circfit,
    plot_gain_fit,
    plot_nonlinear_iq_fit,
    plot_s21,
    plot_sparper,
    plot_xcal,
)

# Timestream points drawn per plot by default.
DEFAULT_MAX_POINTS = 5000

PlotType = namedtuple('PlotType', ['title', 'func', 'param_names', 'optional', 'defaults',
                                   'n_axes', 'axes_arg'])
PlotType.__doc__ = """
One quick-plot type.

Attributes:
title (str): default figure (or panel) title.
func (callable): plot function from ``citkid.xcal.plot``.
param_names (list of str): DataSet parameters passed positionally to ``func``.
optional (frozenset of str): parameters passed as None when unavailable
    (e.g. no timestream in an IQ-only dataset).
defaults (dict): default keyword arguments of ``func`` that callers may
    override through ``**kwargs``.
n_axes (int): number of axes the plot draws into.
axes_arg (str): name of ``func``'s axes argument.
"""

PLOT_TYPES = {
    'raw_data': PlotType(
        'Raw Data', plot_s21, ['ff', 'zf', 'zt', 'fg', 'zg'],
        frozenset({'zt', 'fg', 'zg'}), {'max_points': DEFAULT_MAX_POINTS}, 2, 'axs'),
    'gain_fit': PlotType(
        'Gain Fit', plot_gain_fit, ['fg', 'zg', 'gain_mask', 'p_amp', 'p_phase'],
        frozenset(), {}, 2, 'axs'),
    's21_rmv': PlotType(
        'Gain Removed Fine Sweep', plot_s21, ['ff', 'zf_rmv', 'zt_rmv'],
        frozenset({'zt_rmv'}), {'max_points': DEFAULT_MAX_POINTS}, 2, 'axs'),
    'circfit': PlotType(
        'Circle Fit', plot_circfit,
        ['zf_rmv', 'circ_origin', 'circ_radius', 'zt_rmv', 'circ_mask'],
        frozenset({'zt_rmv', 'circ_mask'}), {'max_points': DEFAULT_MAX_POINTS}, 1, 'ax'),
    'sparper': PlotType(
        'Par/Per Noise PSDs', plot_sparper, ['f_sparper', 'spar', 'sper'],
        frozenset(), {'nbins': 100, 'fmin': 0.1}, 1, 'ax'),
    'xcal': PlotType(
        r'$x$ Calibration', plot_xcal,
        ['thetaf', 'xf', 'zf_cent', 'xcal_mask', 'poly_x', 'thetat', 'zt_cent',
         'xcal_std_cutoff'],
        frozenset({'thetat', 'zt_cent', 'xcal_std_cutoff'}),
        {'max_points': DEFAULT_MAX_POINTS}, 2, 'axs'),
    'iq_fit': PlotType(
        'Nonlinear IQ Fit', plot_nonlinear_iq_fit, ['ff', 'zf_rmv', 'iq_popt', 'iq_mask'],
        frozenset({'iq_mask'}), {}, 2, 'axs'),
}

# Order of the panels in plot_full_cal.
FULL_CAL_TYPES = ['raw_data', 'gain_fit', 's21_rmv', 'circfit', 'sparper', 'xcal', 'iq_fit']

# Fixed panel geometry (inches). Axes are placed directly, without a layout
# engine, which is most of matplotlib's time for these small plots.
PANEL_W, PANEL_H = 8.0, 3.2
_LEFT, _RIGHT, _BOTTOM, _TOP, _GAP = 0.8, 1.95, 0.55, 0.45, 0.85
# Legends sit outside the axes, in the right margin.
_STYLE = {'legend.fontsize': 'small'}
DPI = 72


class MissingPlotData(ValueError):
    """
    A parameter a plot type needs is not available for the requested row.
    """


def _row_value(DS, name, data_idx):
    """
    Return one row of a DataSet parameter (or the value of a global one).

    Parameters:
    DS (DataSet): dataset.
    name (str): parameter name.
    data_idx (int): row.

    Returns:
    value: the parameter's value for the row.
    """
    value = getattr(DS, name)
    meta = DS._param_meta.get(name)
    if meta is not None and meta['global']:
        return value
    return value[data_idx]


def _plot_args(DS, data_idx, plot_type):
    """
    Collect the positional arguments of a plot type for one row.

    Parameters:
    DS (DataSet): dataset.
    data_idx (int): row.
    plot_type (str): key of ``PLOT_TYPES``.

    Returns:
    args (list): values in ``param_names`` order; None for unavailable
        optional parameters.

    Raises:
    MissingPlotData: if a required parameter is unavailable.
    """
    spec = PLOT_TYPES[plot_type]
    args = []
    for name in spec.param_names:
        try:
            args.append(_row_value(DS, name, data_idx))
        except Exception as exc:
            if name in spec.optional:
                args.append(None)
                continue
            raise MissingPlotData(
                f"plot type '{plot_type}' needs '{name}', which is not available for "
                f"data_idx {data_idx}: {exc}") from exc
    return args


def _panel_axes(fig, n_axes, row, col, nrows, title):
    """
    Add one panel's axes at a fixed position, with room for labels and the
    legend to the right of the last axes.

    Parameters:
    fig (matplotlib.figure.Figure): figure of ``PANEL_W`` x ``PANEL_H``
        panels.
    n_axes (int): number of axes in the panel.
    row, col (int): panel position (row 0 at the top).
    nrows (int): number of panel rows in the figure.
    title (str): panel title.

    Returns:
    axs (matplotlib.axes.Axes or np.ndarray of Axes): one Axes if
        ``n_axes`` is 1, else an array.
    """
    fig_w, fig_h = fig.get_size_inches()
    x0, y0 = col * PANEL_W, (nrows - 1 - row) * PANEL_H
    width = (PANEL_W - _LEFT - _RIGHT - _GAP * (n_axes - 1)) / n_axes
    height = PANEL_H - _BOTTOM - _TOP
    axs = [
        fig.add_axes([(x0 + _LEFT + i * (width + _GAP)) / fig_w, (y0 + _BOTTOM) / fig_h,
                      width / fig_w, height / fig_h])
        for i in range(n_axes)
    ]
    fig.text((x0 + PANEL_W / 2) / fig_w, (y0 + PANEL_H - 0.08) / fig_h, title,
             ha='center', va='top', fontsize='large')
    return axs[0] if n_axes == 1 else np.array(axs, dtype=object)


def _panels_figure(n_panels, ncols):
    """
    Create a figure for a grid of panels.

    Parameters:
    n_panels (int): number of panels.
    ncols (int): panels per row.

    Returns:
    fig (matplotlib.figure.Figure): the figure.
    nrows (int): number of panel rows.
    """
    nrows = -(-n_panels // ncols)
    fig = plt.figure(figsize=(PANEL_W * ncols, PANEL_H * nrows + 0.3 * (n_panels > 1)),
                     dpi=DPI)
    return fig, nrows


def _check_plot_type(plot_type):
    """
    Raise if a plot type is unknown.

    Parameters:
    plot_type (str): plot type name.

    Raises:
    ValueError: if ``plot_type`` is not a key of ``PLOT_TYPES``.
    """
    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"plot_type '{plot_type}' not recognized. Available types: {list(PLOT_TYPES)}")


def _draw(spec, args, kwargs, axes):
    """
    Call a plot function with its defaults overridden by the matching kwargs.

    Parameters:
    spec (PlotType): plot type.
    args (list): positional arguments.
    kwargs (dict): caller's keyword arguments; those not used by this plot
        type are ignored.
    axes (Axes, array of Axes, or None): axes to draw into, or None for a new
        figure.

    Returns:
    fig, axs: what the plot function returns.
    """
    options = {key: kwargs.get(key, value) for key, value in spec.defaults.items()}
    with plt.rc_context(_STYLE):
        return spec.func(*args, **options, **{spec.axes_arg: axes})


def plot(DS, data_idx, plot_type, title=None, **kwargs):
    """
    Plot one plot type for one row of a DataSet.

    Parameters:
    DS (DataSet): dataset.
    data_idx (int): row to plot.
    plot_type (str): one of 'raw_data', 'gain_fit', 's21_rmv', 'circfit',
        'sparper', 'xcal', 'iq_fit' (keys of ``PLOT_TYPES``).
    title (str or None): figure title. None (default) uses the plot type's
        title.
    **kwargs: options of the plot function, e.g. ``max_points`` (timestream
        points drawn; default 5000, None draws all) or ``nbins`` and ``fmin``
        for 'sparper'. Options a plot type doesn't use are ignored.

    Returns:
    fig (matplotlib.figure.Figure): the figure.
    axs (matplotlib.axes.Axes or np.ndarray of Axes): its axes.

    Raises:
    ValueError: if ``plot_type`` is unknown, or a parameter it needs is not
        available for ``data_idx`` (``MissingPlotData``).
    """
    _check_plot_type(plot_type)
    spec = PLOT_TYPES[plot_type]
    args = _plot_args(DS, data_idx, plot_type)
    fig, _ = _panels_figure(1, 1)
    axs = _panel_axes(fig, spec.n_axes, 0, 0, 1, spec.title if title is None else title)
    _draw(spec, args, kwargs, axs)
    return fig, axs


def plot_full_cal(DS, data_idx, plot_types=None, as_figure=False, **kwargs):
    """
    Plot every available plot type for one row of a DataSet in one figure.

    Panels are laid out three per row. Plot types whose data the dataset
    doesn't have (e.g. the circle fit of an IQ-only dataset) are left out.
    A plot type that fails (e.g. a row marked bad) shows the error in its
    panel instead of stopping the figure.

    Parameters:
    DS (DataSet): dataset.
    data_idx (int): row to plot.
    plot_types (list of str or None): plot types to try, in order. None
        (default) tries ``FULL_CAL_TYPES``.
    as_figure (bool): If True, return the matplotlib figure (e.g. to view or
        zoom it in a notebook). If False (default), return a PNG image, as
        the legacy pipeline did, and close the figure.
    **kwargs: options of the plot functions (see ``plot``), passed to every
        plot type that uses them.

    Returns:
    if as_figure:
        fig (matplotlib.figure.Figure): the combined figure.
    else:
        png (io.BytesIO): the figure as a PNG image, e.g. write
            ``png.getvalue()`` to a file or show it with
            ``IPython.display.Image(png.getvalue())``.

    Raises:
    ValueError: if a plot type is unknown, or none of them has its data.
    """
    plot_types = list(FULL_CAL_TYPES if plot_types is None else plot_types)
    for plot_type in plot_types:
        _check_plot_type(plot_type)
    panels, skipped = [], []
    for plot_type in plot_types:
        try:
            panels.append((plot_type, _plot_args(DS, data_idx, plot_type)))
        except MissingPlotData:
            skipped.append(plot_type)
    if not panels:
        raise ValueError(f'None of the plot types {plot_types} has data for data_idx '
                         f'{data_idx}')

    ncols = min(3, len(panels))
    fig, nrows = _panels_figure(len(panels), ncols)
    for i, (plot_type, args) in enumerate(panels):
        spec = PLOT_TYPES[plot_type]
        axs = _panel_axes(fig, spec.n_axes, i // ncols, i % ncols, nrows, spec.title)
        try:
            _draw(spec, args, kwargs, axs)
        except Exception as exc:
            for ax in np.atleast_1d(axs):
                ax.clear()
                ax.set_axis_off()
            first = np.atleast_1d(axs)[0]
            first.text(0.02, 0.5, f'{plot_type} failed:\n{type(exc).__name__}: {exc}',
                       transform=first.transAxes, va='center', wrap=True, color='C3')
    title = f'data_idx {data_idx}'
    if skipped:
        title += f'  (no data for: {", ".join(skipped)})'
    fig.text(0.5, 1, title, ha='center', va='top', fontsize='x-large')
    if as_figure:
        return fig
    png = BytesIO()
    fig.savefig(png, format='png')  # fixed layout: no tight bbox (a second draw)
    plt.close(fig)
    png.seek(0)
    return png
