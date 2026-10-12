"""
Fit the series windows' y vs x data for each resonator, and store the fits.

The series windows plot one y value per series index against an x value (e.g.
the nonlinearity ``a`` against power) for each resonator (``data_idx``).
``SeriesXYFit`` describes an optional fit of that curve with M parameters, and
``SeriesXYFitStore`` saves the fitted parameters of every resonator as one
(nrows, M) array (``popt``), with the guesses used (``p0``) if the fit takes
guesses.

This is a deliberately small store rather than a ``DataSet``: a fit turns all
series points of one row into a few numbers, so it doesn't need the calibration and
invalidation machinery of the pipeline.
"""

from collections import namedtuple

import numpy as np

from .dataset import _io_lock_for, _retry_io


class SeriesXYFit:
    """
    Description of a fit of the series plot's y vs x data, one per resonator.

    Attributes:
    fit (callable): the fit, called with the plotted points of one resonator
        (``x`` and ``y``, with unavailable points removed and sorted by x):
        if not ``uses_guess``:
            ``fit(x, y) -> popt``, the M fitted parameters.
        else:
            ``fit(x, y, p0) -> (popt, p0)``. ``p0`` is None unless the user
            typed a guess in the window; on None, the function makes its own
            guess. It returns the fitted parameters and the guess it used
            (both M values).
    param_names (list of str): names of the M parameters, for the plot title
        and the guess inputs. The fits are saved as one array, not by name.
    model (callable): ``model(xs, *popt)`` returning the fitted y values at
        the x values ``xs`` (a float64 array), for plotting the fit. The
        parameters are passed as Python floats, so numba functions with
        float64 signatures work. It must return y in the units of the series
        plot's y axis.
    name (str): Name of the fit, shown in the plot title and stored with the
        saved fits. Default 'xy_fit'.
    group (zarr.Group or None): Group to save the fits to. None (default)
        uses ``root/xy_fit`` in ``run_iq_series`` (or keeps the fits in
        memory if the series window has no state group).
    n_samples (int): Number of x samples used to draw the fitted curve.
        Default 200.
    min_points (int or None): Fewest usable points to attempt a fit. With
        fewer, the parameters are NaN and no curve is drawn. None (default)
        uses ``len(param_names)``.
    uses_guess (bool): If True, ``fit`` takes and returns a guess (see
        ``fit``), the guesses are saved, and the series window shows them in
        inputs the user can edit. Default False.
    """

    def __init__(
        self, fit, param_names, model, name='xy_fit', group=None, n_samples=200,
        min_points=None, uses_guess=False
    ):
        """
        Store the fit definition. See the class docstring for parameters.

        Raises:
        ValueError: If ``param_names`` is empty or has duplicates.
        """
        param_names = [str(n) for n in param_names]
        if not param_names:
            raise ValueError('param_names must contain at least one name')
        if len(set(param_names)) != len(param_names):
            raise ValueError(f'param_names must be unique; got {param_names}')
        self.fit = fit
        self.param_names = param_names
        self.model = model
        self.name = str(name)
        self.group = group
        self.n_samples = int(n_samples)
        self.min_points = len(param_names) if min_points is None else int(min_points)
        self.uses_guess = bool(uses_guess)

    @property
    def n_params(self):
        """
        Number of fit parameters (M).

        Returns:
        n (int): ``len(param_names)``.
        """
        return len(self.param_names)

    def _as_params(self, values, what):
        """
        Convert fit parameters (or a guess) to a float array of length M.

        Parameters:
        values (array-like): the values.
        what (str): what they are, for the error message.

        Returns:
        values (np.ndarray): float array of shape (M,).

        Raises:
        ValueError: If there aren't M scalar values.
        """
        values = np.asarray(values, dtype=float).ravel()
        if len(values) != self.n_params:
            raise ValueError(
                f"fit returned {len(values)} {what} values, but param_names has "
                f"{self.n_params}: {self.param_names}")
        return values

    def run(self, x, y, p0=None):
        """
        Fit the usable points of one resonator's series.

        Parameters:
        x (np.ndarray): x value of each series index (NaN if unavailable).
        y (np.ndarray): y value of each series index (NaN if unavailable, e.g.
            marked bad).
        p0 (array-like or None): guess of the M parameters, for a fit that
            ``uses_guess``; None lets the fit make its own guess. Must be
            None if the fit doesn't use guesses.

        Returns:
        popt (np.ndarray or None): fitted parameters, shape (M,), or None if
            there are fewer than ``min_points`` usable points.
        p0 (np.ndarray or None): guess the fit used, shape (M,), or None if
            the fit doesn't use guesses or wasn't attempted.

        Raises:
        ValueError: If the fit returns the wrong number of values, or ``p0``
            is given to a fit that doesn't use guesses.
        Exception: Anything ``fit`` raises.
        """
        if p0 is not None and not self.uses_guess:
            raise ValueError(f"xy fit '{self.name}' doesn't use guesses (uses_guess=False)")
        xs, ys = usable_points(x, y)
        if len(xs) < self.min_points:
            return None, None
        if not self.uses_guess:
            return self._as_params(self.fit(xs, ys), 'parameter'), None
        p0_in = None if p0 is None else self._as_params(p0, 'guess')
        result = self.fit(xs, ys, p0_in)
        # Two scalars with several parameters means the guess was forgotten.
        forgot_guess = (isinstance(result, tuple) and len(result) == 2
                        and self.n_params > 1 and np.ndim(result[0]) == 0)
        if forgot_guess or not (isinstance(result, tuple) and len(result) == 2):
            raise ValueError(
                f"xy fit '{self.name}' uses guesses, so fit must return (popt, p0); "
                f"got {type(result).__name__}")
        return self._as_params(result[0], 'parameter'), self._as_params(result[1], 'guess')

    def curve(self, x, popt, log_x=False):
        """
        Evaluate the model across the range of the usable x values.

        Parameters:
        x (np.ndarray): x value of each series index (NaN if unavailable).
        popt (np.ndarray or None): fitted parameters, or None.
        log_x (bool): If True (for a log x axis), space the samples
            geometrically and use only x > 0. If False (default), space them
            linearly.

        Returns:
        xs, ys (np.ndarray or None): ``n_samples`` x values spanning the
            usable x range and the model at them, or None, None if there are
            no parameters, any is NaN, or fewer than 2 distinct usable x
            values.
        """
        if popt is None or not np.all(np.isfinite(popt)):
            return None, None
        finite = np.asarray(x, dtype=float)
        finite = finite[np.isfinite(finite)]
        if log_x:
            finite = finite[finite > 0]
        if len(finite) < 2 or finite.min() == finite.max():
            return None, None
        spacing = np.geomspace if log_x else np.linspace
        xs = spacing(finite.min(), finite.max(), self.n_samples)
        return xs, np.asarray(self.model(xs, *[float(p) for p in popt]), dtype=float)

    def describe(self, popt):
        """
        Format the fitted parameters for display.

        Parameters:
        popt (np.ndarray or None): fitted parameters, or None.

        Returns:
        text (str): e.g. ``'slope = 0.031, intercept = 0.12'``. Empty if
            there are no parameters.
        """
        if popt is None:
            return ''
        return ', '.join(f'{name} = {float(value):.4g}'
                         for name, value in zip(self.param_names, popt))


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


SavedXYFit = namedtuple('SavedXYFit', ['fit_x', 'fit_y', 'popt', 'p0', 'p0_user'])
SavedXYFit.__doc__ = """
One resonator's saved xy fit.

Attributes:
fit_x, fit_y (np.ndarray): x and y the fit used, one per series index.
popt (np.ndarray or None): fitted parameters, or None if the fit failed or
    wasn't attempted.
p0 (np.ndarray or None): guess the fit used, or None (no guess, or the fit
    doesn't use guesses).
p0_user (bool): True if the guess was entered by the user (it is reused
    when the resonator is refitted).
"""


class SeriesXYFitStore:
    """
    Zarr storage for one ``SeriesXYFit`` result per resonator.

    The group holds ``popt`` (shape ``(nrows, M)``: the fitted parameters,
    NaN for failed fits), with ``p0`` (``(nrows, M)``: the guesses used) and
    ``p0_user`` (``(nrows,)``: guesses entered by the user) if the fit uses
    guesses; ``fit_x`` and ``fit_y`` (shape ``(nrows, n_series)``, the exact
    inputs of each saved fit, so a fit can be redone only when its inputs
    change); and ``row_exists`` (rows that have been fitted, including failed
    fits). Each array is a single small chunk that is rewritten on update, so
    the group has only a few files. Access is guarded by the same lock as the
    DataSets on the store, so the UI thread and the background worker can
    both use it.

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
        definition (dict): Fit name, parameter names, whether it uses
            guesses, nrows and n_series.
        """
        return {
            'name': self.xy_fit.name,
            'param_names': list(self.xy_fit.param_names),
            'uses_guess': self.xy_fit.uses_guess,
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
        names = stored.get('param_names', stored.get('output_names'))
        return (f"The xy fit group already contains fits from '{stored.get('name')}' "
                f"with parameters {names}.")

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

    def popt(self):
        """
        Return the fitted parameters of every resonator.

        Returns:
        popt (np.ndarray): shape (nrows, M); NaN for resonators without a
            (successful) fit.
        """
        with self._lock:
            if 'popt' not in self.group:
                return np.full((self.nrows, self.xy_fit.n_params), np.nan)
            return np.asarray(_retry_io(lambda: self.group['popt'][...]), dtype=float)

    def load(self, data_idx):
        """
        Load a resonator's saved fit.

        Parameters:
        data_idx (int): Resonator index.

        Returns:
        saved (SavedXYFit or None): the saved fit, or None if there is none.
        """
        di = int(data_idx)

        def optional(name):
            """
            Read one row of an array, or None if the array doesn't exist.
            """
            return np.asarray(self.group[name][di], dtype=float) if name in self.group else None

        def read():
            """
            Read one row of every array.
            """
            if 'row_exists' not in self.group or not self.group['row_exists'][di]:
                return None
            popt = optional('popt')
            if popt is not None and not np.all(np.isfinite(popt)):
                popt = None
            p0 = optional('p0')
            if p0 is not None and not np.all(np.isfinite(p0)):
                p0 = None
            p0_user = bool(self.group['p0_user'][di]) if 'p0_user' in self.group else False
            return SavedXYFit(np.asarray(self.group['fit_x'][di]),
                              np.asarray(self.group['fit_y'][di]), popt, p0, p0_user)

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
        saved = self.load(data_idx)
        if saved is None:
            return False
        return (np.array_equal(saved.fit_x, np.asarray(x, dtype=float), equal_nan=True)
                and np.array_equal(saved.fit_y, np.asarray(y, dtype=float), equal_nan=True))

    def save(self, data_idx, x, y, popt, p0=None, p0_user=False):
        """
        Save a resonator's fit and the inputs it used.

        Parameters:
        data_idx (int): Resonator index.
        x (np.ndarray): x values used, one per series index.
        y (np.ndarray): y values used, one per series index.
        popt (np.ndarray or None): fitted parameters, or None if the fit
            failed or wasn't attempted (saved as NaN).
        p0 (np.ndarray or None): guess used, or None (saved as NaN). Only
            saved if the fit uses guesses.
        p0_user (bool): True if the guess was entered by the user. Default
            False.
        """
        di = int(data_idx)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = self.xy_fit.n_params

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
            Write the inputs, parameters, guess and row flag.
            """
            if self._ATTR not in self.group.attrs:
                self.group.attrs[self._ATTR] = self._definition()
            require('fit_x', (self.n_series,), float, np.nan)[di] = x
            require('fit_y', (self.n_series,), float, np.nan)[di] = y
            require('popt', (m,), float, np.nan)[di] = (
                np.full(m, np.nan) if popt is None else np.asarray(popt, dtype=float))
            if self.xy_fit.uses_guess:
                require('p0', (m,), float, np.nan)[di] = (
                    np.full(m, np.nan) if p0 is None else np.asarray(p0, dtype=float))
                require('p0_user', (), bool, False)[di] = bool(p0_user)
            require('row_exists', (), bool, False)[di] = True

        with self._lock:
            _retry_io(write)
