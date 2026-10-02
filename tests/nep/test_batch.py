import numpy as np
import pytest
import zarr

from citkid.nep.batch import fit_nep_photon_sets, per_set_values
from citkid.nep.funcs import nep_photon
from citkid.nep.store import NEPFitStore

NU = 150e9


def _sets():
    """
    Build three sets: two good, one all NaN.

    Returns:
    powers, neps (list of np.ndarray): per-set data.
    """
    power = np.geomspace(1e-13, 1e-10, 8)
    return ([power, power, np.full(8, np.nan)],
            [nep_photon(power, 0.4, NU), nep_photon(power, 0.6, NU), np.full(8, np.nan)])


def test_fit_sets_returns_arrays_with_nan_for_empty_sets():
    powers, neps = _sets()

    eta, eta_err, n_fit = fit_nep_photon_sets(powers, neps, NU, p_min=[0, 1e-12, 0],
                                              progress=False)

    assert all(isinstance(a, np.ndarray) for a in (eta, eta_err, n_fit))
    np.testing.assert_allclose(eta[:2], [0.4, 0.6])
    assert np.isnan(eta[2]) and np.isnan(eta_err[2])
    np.testing.assert_array_equal(n_fit, [8, int(np.sum(powers[1] >= 1e-12)), 0])


@pytest.mark.parametrize('kwargs, match', [
    ({'p_min': [1e-12]}, 'p_min'),
    ({'nep_errs': [np.ones(8)]}, 'nep_errs'),
])
def test_fit_sets_checks_per_set_inputs(kwargs, match):
    powers, neps = _sets()
    with pytest.raises(ValueError, match=match):
        fit_nep_photon_sets(powers, neps, NU, progress=False, **kwargs)


def test_fit_sets_checks_shapes():
    with pytest.raises(ValueError, match='same shape'):
        fit_nep_photon_sets([np.ones(3)], [np.ones(4)], NU, progress=False)
    with pytest.raises(ValueError, match='same number of sets'):
        fit_nep_photon_sets([np.ones(3)], [], NU, progress=False)


def test_per_set_values():
    assert per_set_values(None, 2, 'x') == [None, None]
    assert per_set_values(3.0, 2, 'x') == [3.0, 3.0]
    assert per_set_values([1, 2], 2, 'x') == [1, 2]


def test_store_round_trip_and_definition_check():
    group = zarr.open_group(zarr.storage.MemoryStore(), mode='w')
    store = NEPFitStore(group, 2, NU, ['a', 'b'])
    assert store.matches_definition() and store.load() is None
    fields = {'p_min': [1e-12, 0.0], 'eta': [0.5, np.nan], 'eta_err': [0.01, np.nan],
              'n_fit': [4, 0], 'bad': [False, True], 'viewed': [True, False],
              'row_exists': [True, True]}

    store.save(fields)
    loaded = NEPFitStore(group, 2, NU, ['a', 'b']).load()

    np.testing.assert_allclose(loaded['p_min'], [1e-12, 0.0])
    np.testing.assert_array_equal(loaded['bad'], [False, True])
    assert not NEPFitStore(group, 3, NU, ['a', 'b', 'c']).matches_definition()
    other = NEPFitStore(group, 2, 2 * NU, ['a', 'b'])
    assert not other.matches_definition() and 'nu' in other.describe_existing()
    other.clear()
    assert other.matches_definition() and other.load() is None
