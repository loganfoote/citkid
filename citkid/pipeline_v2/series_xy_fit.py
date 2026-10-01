"""
Fit the series windows' y vs x data for each resonator, and store the fits.

The series windows plot one y value per series index against an x value (e.g.
the nonlinearity ``a`` against power) for each resonator (``data_idx``).
``SeriesXYFit`` describes an optional fit of that curve, and
``SeriesXYFitStore`` saves one fit per resonator to a small zarr group.

This is a deliberately small store rather than a ``DataSet``: a fit turns all
series points of one row into a few numbers, so it doesn't need the calibration and
invalidation machinery of the pipeline.
"""

import numpy as np

from .dataset import _io_lock_for, _retry_io


class SeriesXYFit:
    """
    Description of a fit of the series plot's y vs x data, one per resonator.

    Attributes:
    fit (callable): ``fit(x, y)`` returning the fit outputs, in
        ``output_names`` order (a tuple, or a single value if there is one
        output). ``x`` and ``y`` are exactly what the series plot shows, with
        unavailable points removed and sorted by x.
    output_names (list of str): Names of the fit outputs, used for saving
        and for the plot title.
    model (callable): ``model(xs, *outputs)`` returning the fitted y values at
        the x values ``xs``, for plotting the fit.
    name (str): Name of the fit, shown in the plot title and stored with the
        saved fits. Default 'xy_fit'.
    group (zarr.Group or None): Group to save the fits to. None (default)
        uses ``root/xy_fit`` in ``run_iq_series`` (or keeps the fits in
        memory if the series window has no state group).
    n_samples (int): Number of x samples used to draw the fitted curve.
        Default 200.
    min_points (int or None): Fewest usable points to attempt a fit. With
        fewer, the outputs are NaN and no curve is drawn. None (default) uses
        ``len(output_names)``.
    """

    def __init__(self, fit, output_names, model, name='xy_fit', group=None,
                 n_samples=200, min_points=None):
        """
        Store the fit definition.

        Parameters:
        fit (callable): See the class docstring.
        output_names (list of str): See the class docstring.
        model (callable): See the class docstring.
        name (str): See the class docstring.
        group (zarr.Group or None): See the class docstring.
        n_samples (int): See the class docstring.
        min_points (int or None): See the class docstring.

        Raises:
        ValueError: If ``output_names`` is empty or has duplicates.
        """
        output_names = [str(n) for n in output_names]
        if not output_names:
            raise ValueError('output_names must contain at least one name')
        if len(set(output_names)) != len(output_names):
            raise ValueError(f'output_names must be unique; got {output_names}')
        self.fit = fit
        self.output_names = output_names
        self.model = model
        self.name = str(name)
        self.group = group
        self.n_samples = int(n_samples)
        self.min_points = len(output_names) if min_points is None else int(min_points)

    def run(self, x, y):
        """
        Fit the usable points of one resonator's series.

        Parameters:
        x (np.ndarray): x value of each series index (NaN if unavailable).
        y (np.ndarray): y value of each series index (NaN if unavailable, e.g.
            marked bad).

        Returns:
        outputs (list of np.ndarray): Fit outputs in ``output_names`` order,
            or None if there are fewer than ``min_points`` usable points.

        Raises:
        ValueError: If ``fit`` returns the wrong number of outputs.
        Exception: Anything ``fit`` raises.
        """
        xs, ys = usable_points(x, y)
        if len(xs) < self.min_points:
            return None
        result = self.fit(xs, ys)
        if len(self.output_names) == 1:
            result = (result,)
        result = tuple(result)
        if len(result) != len(self.output_names):
            raise ValueError(
                f"fit returned {len(result)} outputs, but output_names has "
                f"{len(self.output_names)}: {self.output_names}"
            )
        return [np.asarray(value, dtype=float) for value in result]

    def curve(self, x, outputs):
        """
        Evaluate the model across the range of the usable x values.

        Parameters:
        x (np.ndarray): x value of each series index (NaN if unavailable).
        outputs (list of np.ndarray or None): Fit outputs, or None.

        Returns:
        xs, ys (np.ndarray or None): ``n_samples`` x values spanning the
            usable x range and the model at them, or None, None if there are
            no outputs, any output is NaN, or fewer than 2 usable x values.
        """
        if outputs is None or any(np.any(~np.isfinite(o)) for o in outputs):
            return None, None
        finite = np.asarray(x, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < 2 or finite.min() == finite.max():
            return None, None
        xs = np.linspace(finite.min(), finite.max(), self.n_samples)
        return xs, np.asarray(self.model(xs, *outputs), dtype=float)

    def describe(self, outputs):
        """
        Format the scalar fit outputs for display.

        Parameters:
        outputs (list of np.ndarray or None): Fit outputs, or None.

        Returns:
        text (str): e.g. ``'slope = 0.031, intercept = 0.12'``. Non-scalar
            outputs are skipped. Empty if there are no outputs.
        """
        if outputs is None:
            return ''
        parts = [
            f'{name} = {float(value):.4g}'
            for name, value in zip(self.output_names, outputs)
            if np.ndim(value) == 0
        ]
        return ', '.join(parts)


def usable_points(x, y):
    """
    Return the points that have both x and y, sorted by x.

    Parameters:
    x (array-like): x values (NaN if unavailable).
    y (array-like): y values (NaN if unavailable).

    Returns:
    xs, ys (np.ndarray): Finite pairs, sorted by x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[keep], kind='stable')
    return x[keep][order], y[keep][order]


class SeriesXYFitStore:
    """
    Zarr storage for one ``SeriesXYFit`` result per resonator.

    The group holds one array per fit output (shape ``(nrows, ...)``),
    ``fit_x`` and ``fit_y`` (shape ``(nrows, n_series)``, the exact inputs of
    each saved fit, so a fit can be redone only when its inputs change), and
    ``row_exists`` (rows that have been fitted, including failed fits, whose
    outputs are NaN). Each array is a single small chunk that is rewritten on
    update, so the group has only a few files. Access is guarded by the same
    lock as the DataSets on the store, so the UI thread and the background
    worker can both use it.

    Attributes:
    group (zarr.Group): The fit group.
    xy_fit (SeriesXYFit): The fit definition.
    nrows (int): Number of resonators.
    n_series (int): Number of series indices.
    """

    _ATTR = 'series_xy_fit'

    def __init__(self, group, xy_fit, nrows, n_series):
        """
        Attach to a fit group. Call ``matches_definition`` before saving.

        Parameters:
        group (zarr.Group): Group to store the fits in.
        xy_fit (SeriesXYFit): The fit definition.
        nrows (int): Number of resonators.
        n_series (int): Number of series indices.
        """
        self.group = group
        self.xy_fit = xy_fit
        self.nrows = int(nrows)
        self.n_series = int(n_series)
        self._lock = _io_lock_for(group)

    def _definition(self):
        """
        Return the definition stored with the fits.

        Returns:
        definition (dict): Fit name, output names, nrows and n_series.
        """
        return {
            'name': self.xy_fit.name,
            'output_names': list(self.xy_fit.output_names),
            'nrows': self.nrows,
            'n_series': self.n_series,
        }

    def matches_definition(self):
        """
        Check whether existing saved fits use this fit definition.

        Returns:
        matches (bool): True if the group is empty or its stored definition
            matches, False if it holds fits from a different definition.
        """
        with self._lock:
            stored = self.group.attrs.get(self._ATTR)
            has_arrays = any(True for _ in self.group.array_keys())
        if stored is None:
            return not has_arrays
        return dict(stored) == self._definition()

    def describe_existing(self):
        """
        Describe the fits already saved in the group, for a popup.

        Returns:
        text (str): Description of the stored definition.
        """
        with self._lock:
            stored = self.group.attrs.get(self._ATTR)
        if stored is None:
            return 'The xy fit group contains data from an unknown source.'
        stored = dict(stored)
        return (f"The xy fit group already contains fits from '{stored.get('name')}' "
                f"with outputs {stored.get('output_names')}.")

    def clear(self):
        """
        Delete all saved fits and record this fit's definition.
        """
        def clear():
            """
            Delete every array and write the definition.
            """
            for key in list(self.group.array_keys()):
                del self.group[key]
            self.group.attrs[self._ATTR] = self._definition()

        with self._lock:
            _retry_io(clear)

    def has_fit(self, data_idx):
        """
        Check whether a resonator has a saved fit (possibly failed).

        Parameters:
        data_idx (int): Resonator index.

        Returns:
        exists (bool): True if a fit was saved for ``data_idx``.
        """
        with self._lock:
            if 'row_exists' not in self.group:
                return False
            return bool(_retry_io(lambda: self.group['row_exists'][int(data_idx)]))

    def fitted_rows(self):
        """
        Return which resonators have a saved fit.

        Returns:
        mask (np.ndarray): Boolean array of length ``nrows``.
        """
        with self._lock:
            if 'row_exists' not in self.group:
                return np.zeros(self.nrows, dtype=bool)
            return np.asarray(_retry_io(lambda: self.group['row_exists'][...]), dtype=bool)

    def load(self, data_idx):
        """
        Load a resonator's saved fit.

        Parameters:
        data_idx (int): Resonator index.

        Returns:
        fit_x, fit_y (np.ndarray or None): Inputs of the saved fit, or None,
            None if there is no saved fit.
        outputs (list of np.ndarray or None): Saved outputs, or None if there
            is no saved fit or it failed.
        """
        di = int(data_idx)

        def read():
            """
            Read one row of every array.
            """
            if 'row_exists' not in self.group or not self.group['row_exists'][di]:
                return None, None, None
            fit_x = np.asarray(self.group['fit_x'][di])
            fit_y = np.asarray(self.group['fit_y'][di])
            names = self.xy_fit.output_names
            if not all(name in self.group for name in names):
                return fit_x, fit_y, None
            outputs = [np.asarray(self.group[name][di], dtype=float) for name in names]
            if any(np.all(~np.isfinite(o)) for o in outputs):
                outputs = None
            return fit_x, fit_y, outputs

        with self._lock:
            return _retry_io(read)

    def is_current(self, data_idx, x, y):
        """
        Check whether a resonator's saved fit used exactly these inputs.

        Parameters:
        data_idx (int): Resonator index.
        x (np.ndarray): Current x values, one per series index.
        y (np.ndarray): Current y values, one per series index.

        Returns:
        current (bool): True if a saved fit exists with the same x and y
            (NaN equal to NaN).
        """
        fit_x, fit_y, _ = self.load(data_idx)
        if fit_x is None:
            return False
        return (np.array_equal(fit_x, np.asarray(x, dtype=float), equal_nan=True)
                and np.array_equal(fit_y, np.asarray(y, dtype=float), equal_nan=True))

    def save(self, data_idx, x, y, outputs):
        """
        Save a resonator's fit and the inputs it used.

        Parameters:
        data_idx (int): Resonator index.
        x (np.ndarray): x values used, one per series index.
        y (np.ndarray): y values used, one per series index.
        outputs (list of np.ndarray or None): Fit outputs in ``output_names``
            order, or None if the fit failed or wasn't attempted (saved as
            NaN once the output arrays exist).
        """
        di = int(data_idx)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        def require(name, row_shape, dtype, fill):
            """
            Return a single-chunk array, creating it filled with ``fill``.
            """
            if name in self.group:
                return self.group[name]
            shape = (self.nrows, *row_shape)
            return self.group.create_array(
                name, data=np.full(shape, fill, dtype=dtype), chunks=shape,
            )

        def write():
            """
            Write the inputs, outputs and row flag.
            """
            if self._ATTR not in self.group.attrs:
                self.group.attrs[self._ATTR] = self._definition()
            require('fit_x', (self.n_series,), float, np.nan)[di] = x
            require('fit_y', (self.n_series,), float, np.nan)[di] = y
            for i, name in enumerate(self.xy_fit.output_names):
                if outputs is not None:
                    value = outputs[i]
                    require(name, value.shape, float, np.nan)[di] = value
                elif name in self.group:
                    self.group[name][di] = np.nan
            require('row_exists', (), bool, False)[di] = True

        with self._lock:
            _retry_io(write)
