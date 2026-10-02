import numpy as np


class NEPFitStore:
    """
    Zarr storage for the interactive NEP fits, one entry per set.

    The group holds one small array per field (``p_min``, ``eta``,
    ``eta_err``, ``n_fit``, ``bad``, ``viewed``, ``row_exists``), each a
    single chunk rewritten on save, so the group has only a few files. The
    definition (number of sets, ``nu``, set names) is stored in the group
    attributes, so fits of different data are not mixed.

    Attributes:
    group (zarr.Group): the fit group.
    n_sets (int): number of sets.
    nu (float): photon frequency (Hz).
    names (list of str): set names.
    """

    _ATTR = 'nep_fit'
    FLOAT_FIELDS = ('p_min', 'eta', 'eta_err')
    INT_FIELDS = ('n_fit',)
    BOOL_FIELDS = ('bad', 'viewed', 'row_exists')

    def __init__(self, group, n_sets, nu, names):
        """
        Attach to a fit group. Call ``matches_definition`` before saving.

        Parameters:
        group (zarr.Group): group to store the fits in.
        n_sets (int): number of sets.
        nu (float): photon frequency (Hz).
        names (list of str): set names.
        """
        self.group = group
        self.n_sets = int(n_sets)
        self.nu = float(nu)
        self.names = [str(n) for n in names]

    def _definition(self):
        """
        Return the definition stored with the fits.

        Returns:
        definition (dict): number of sets, nu, and set names.
        """
        return {'n_sets': self.n_sets, 'nu': self.nu, 'names': self.names}

    def matches_definition(self):
        """
        Check whether existing saved fits use this definition.

        Returns:
        matches (bool): True if the group is empty or its stored definition
            matches, False if it holds fits of different data.
        """
        stored = self.group.attrs.get(self._ATTR)
        if stored is None:
            return not any(True for _ in self.group.array_keys())
        return dict(stored) == self._definition()

    def describe_existing(self):
        """
        Describe the fits already saved in the group, for a popup.

        Returns:
        text (str): description of the stored definition.
        """
        stored = self.group.attrs.get(self._ATTR)
        if stored is None:
            return 'The NEP fit group contains data from an unknown source.'
        stored = dict(stored)
        return (f"The NEP fit group already contains fits of {stored.get('n_sets')} "
                f"sets at nu = {stored.get('nu'):.4g} Hz.")

    def clear(self):
        """
        Delete all saved fits and record this definition.
        """
        for key in list(self.group.array_keys()):
            del self.group[key]
        self.group.attrs[self._ATTR] = self._definition()

    def load(self):
        """
        Load every saved field.

        Returns:
        fields (dict): field name -> np.array of length ``n_sets``, or None
            if nothing is saved. Sets never saved have ``row_exists`` False.
        """
        if 'row_exists' not in self.group:
            return None
        return {key: np.asarray(self.group[key][...]) for key in
                self.FLOAT_FIELDS + self.INT_FIELDS + self.BOOL_FIELDS
                if key in self.group}

    def save(self, fields):
        """
        Write every field (all sets at once; the arrays are small).

        Parameters:
        fields (dict): field name -> array-like of length ``n_sets``, for
            every field in ``FLOAT_FIELDS``, ``INT_FIELDS`` and
            ``BOOL_FIELDS``.
        """
        self.group.attrs[self._ATTR] = self._definition()
        dtypes = ([(k, np.float64) for k in self.FLOAT_FIELDS]
                  + [(k, np.int64) for k in self.INT_FIELDS]
                  + [(k, np.bool_) for k in self.BOOL_FIELDS])
        for key, dtype in dtypes:
            values = np.asarray(fields[key], dtype=dtype)
            if key in self.group:
                self.group[key][...] = values
            else:
                self.group.create_array(key, data=values)
