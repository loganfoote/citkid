"""
Tests for citkid.multitone.fres.

Synthetic resonator sweeps are constructed so that the correct resonance
frequency is known exactly, allowing strict equality checks on the index
(and therefore frequency) returned by each method.

Resonator model
---------------
For each helper function, we build a simple IQ sweep whose "resonance feature"
is placed at a known index ``k`` in an N-point array:

  update_fr_minS21  : |S21| has a dip at index k (linear baseline + Gaussian dip).
  update_fr_spacing : IQ speed peaks at index k (sinusoidal IQ loop, speed ∝ |dz/df|).
  update_fr_distance: furthest-from-off-resonance point is at index k (circle in IQ plane).
"""

import warnings
import numpy as np
import pytest
from citkid.multitone.fres import (
    update_fres,
    update_fr_minS21,
    update_fr_spacing,
    update_fr_distance,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic sweeps
# ---------------------------------------------------------------------------

def _make_f(N=200, f0=5e9, span=10e6):
    """Uniformly spaced frequency array centred on f0."""
    return np.linspace(f0 - span / 2, f0 + span / 2, N)


def _minS21_sweep(f, k):
    """
    |S21| with a Gaussian dip of depth 10 dB at index k over a linear baseline.
    Returns complex z with |z| = 10^(dB/20) and constant phase.
    """
    N = len(f)
    baseline = np.linspace(0, 2, N)            # 2 dB slope
    sigma = N / 20
    dip = -10.0 * np.exp(-0.5 * ((np.arange(N) - k) / sigma) ** 2)
    dB = baseline + dip
    mag = 10 ** (dB / 20)
    return mag.astype(complex)


def _spacing_sweep(f, k):
    """
    IQ data with a single isolated spike at index k.

    All values are 0+0j except z[k]=1+0j.  Then:
      d[k-1] = |z[k]-z[k-1]| = 1,  d[k] = |z[k+1]-z[k]| = 1
      score[k] = d[k] + d[k-1] = 2 > score[k±1] = 1 > 0 elsewhere.
    The argmax is unambiguously at k.
    """
    z = np.zeros(len(f), dtype=complex)
    z[k] = 1.0
    return z


def _distance_sweep(f, k):
    """
    IQ data where z[k] is far from the off-resonance reference.

    All values are 0.5+0j so the edge-based offres = 0.5+0j.  z[k]
    is set 10j away, making it unambiguously the furthest point.
    k must not fall within the edge window (indices 0–9 or last 10).
    """
    z = np.full(len(f), 0.5 + 0j, dtype=complex)
    z[k] = 0.5 + 10j
    return z


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 200
K = 80       # resonance index for all single-resonator tests
F0 = 5e9
SPAN = 10e6

@pytest.fixture
def f1d():
    return _make_f(N, F0, SPAN)


@pytest.fixture
def z_mins21(f1d):
    return _minS21_sweep(f1d, K)


@pytest.fixture
def z_spacing(f1d):
    return _spacing_sweep(f1d, K)


@pytest.fixture
def z_distance(f1d):
    return _distance_sweep(f1d, K)


# ---------------------------------------------------------------------------
# update_fr_minS21
# ---------------------------------------------------------------------------

class TestUpdateFrMinS21:

    def test_single_returns_float(self, f1d, z_mins21):
        fr = update_fr_minS21(f1d, z_mins21)
        assert isinstance(fr, float)

    def test_single_correct_index(self, f1d, z_mins21):
        fr = update_fr_minS21(f1d, z_mins21)
        assert fr == pytest.approx(f1d[K])

    def test_batch_returns_array(self, f1d, z_mins21):
        f2d = np.stack([f1d, f1d])
        z2d = np.stack([z_mins21, z_mins21])
        fr = update_fr_minS21(f2d, z2d)
        assert isinstance(fr, np.ndarray)
        assert fr.shape == (2,)

    def test_batch_correct_values(self, f1d, z_mins21):
        K2 = 120
        f2d = np.stack([f1d, f1d])
        z2d = np.stack([z_mins21, _minS21_sweep(f1d, K2)])
        fr = update_fr_minS21(f2d, z2d)
        assert fr[0] == pytest.approx(f1d[K])
        assert fr[1] == pytest.approx(f1d[K2])

    def test_flat_baseline(self, f1d):
        """With no baseline slope, the dip index is still recovered."""
        N = len(f1d)
        k = 60
        sigma = N / 20
        dip = -10.0 * np.exp(-0.5 * ((np.arange(N) - k) / sigma) ** 2)
        z = (10 ** (dip / 20)).astype(complex)
        fr = update_fr_minS21(f1d, z)
        assert fr == pytest.approx(f1d[k])

    def test_dip_at_first_index(self, f1d):
        z = _minS21_sweep(f1d, 0)
        fr = update_fr_minS21(f1d, z)
        assert fr == pytest.approx(f1d[0])

    def test_dip_at_last_index(self, f1d):
        z = _minS21_sweep(f1d, N - 1)
        fr = update_fr_minS21(f1d, z)
        assert fr == pytest.approx(f1d[N - 1])

    def test_accepts_array_like(self, f1d, z_mins21):
        """Accepts plain Python lists."""
        fr = update_fr_minS21(f1d.tolist(), z_mins21.tolist())
        assert isinstance(fr, float)


# ---------------------------------------------------------------------------
# update_fr_spacing
# ---------------------------------------------------------------------------

class TestUpdateFrSpacing:

    def test_single_returns_float(self, f1d, z_spacing):
        fr = update_fr_spacing(f1d, z_spacing)
        assert isinstance(fr, float)

    def test_single_correct_index(self, f1d, z_spacing):
        fr = update_fr_spacing(f1d, z_spacing)
        assert fr == pytest.approx(f1d[K])

    def test_batch_returns_array(self, f1d, z_spacing):
        f2d = np.stack([f1d, f1d])
        z2d = np.stack([z_spacing, z_spacing])
        fr = update_fr_spacing(f2d, z2d)
        assert isinstance(fr, np.ndarray)
        assert fr.shape == (2,)

    def test_batch_correct_values(self, f1d, z_spacing):
        K2 = 150
        f2d = np.stack([f1d, f1d])
        z2d = np.stack([z_spacing, _spacing_sweep(f1d, K2)])
        fr = update_fr_spacing(f2d, z2d)
        assert fr[0] == pytest.approx(f1d[K])
        assert fr[1] == pytest.approx(f1d[K2])

    def test_edge_bins_are_zero_score(self, f1d):
        """Bins 0 and N-1 should never win (padded with 0 score)."""
        # Uniform spacing → all interior scores equal; edges are 0
        z = np.exp(1j * np.linspace(0, 2 * np.pi, N, endpoint=False))
        fr = update_fr_spacing(f1d, z)
        assert fr != f1d[0]
        assert fr != f1d[-1]

    def test_accepts_array_like(self, f1d, z_spacing):
        fr = update_fr_spacing(f1d.tolist(), z_spacing.tolist())
        assert isinstance(fr, float)


# ---------------------------------------------------------------------------
# update_fr_distance
# ---------------------------------------------------------------------------

class TestUpdateFrDistance:

    def test_single_returns_float(self, f1d, z_distance):
        fr = update_fr_distance(f1d, z_distance)
        assert isinstance(fr, float)

    def test_single_correct_index(self, f1d, z_distance):
        fr = update_fr_distance(f1d, z_distance)
        assert fr == pytest.approx(f1d[K])

    def test_batch_returns_array(self, f1d, z_distance):
        f2d = np.stack([f1d, f1d])
        z2d = np.stack([z_distance, z_distance])
        fr = update_fr_distance(f2d, z2d)
        assert isinstance(fr, np.ndarray)
        assert fr.shape == (2,)

    def test_batch_correct_values(self, f1d, z_distance):
        K2 = 150
        f2d = np.stack([f1d, f1d])
        z2d = np.stack([z_distance, _distance_sweep(f1d, K2)])
        fr = update_fr_distance(f2d, z2d)
        assert fr[0] == pytest.approx(f1d[K])
        assert fr[1] == pytest.approx(f1d[K2])

    def test_offres_from_edges(self, f1d):
        """Off-resonance reference must come from edges, not the centre."""
        # All values at 0 except at index K which is far away
        z = np.zeros(N, dtype=complex)
        z[K] = 1 + 1j   # clearly furthest from 0
        fr = update_fr_distance(f1d, z)
        assert fr == pytest.approx(f1d[K])

    def test_short_array_n_edge_clamp(self):
        """n_edge clamps to 1 for very short arrays without raising."""
        f = np.linspace(4e9, 6e9, 4)
        z = np.array([0.0+0j, 0.0+0j, 10.0+0j, 0.0+0j])
        fr = update_fr_distance(f, z)
        assert fr == pytest.approx(f[2])

    def test_accepts_array_like(self, f1d, z_distance):
        fr = update_fr_distance(f1d.tolist(), z_distance.tolist())
        assert isinstance(fr, float)


# ---------------------------------------------------------------------------
# update_fres
# ---------------------------------------------------------------------------

class TestUpdateFres:

    def _make_batch(self, f1d, method):
        """Return (fs, zs) for M=3 resonators with known answers."""
        self.ks = [40, 80, 140]
        builders = {
            'mins21':   _minS21_sweep,
            'spacing':  _spacing_sweep,
            'distance': _distance_sweep,
        }
        build = builders[method]
        fs = np.stack([f1d] * 3)
        zs = np.stack([build(f1d, k) for k in self.ks])
        return fs, zs

    def test_method_none_returns_fres_copy(self, f1d):
        fres = np.array([4.9e9, 5.0e9, 5.1e9])
        qres = np.array([1e4, 1e4, 1e4])
        res_idxs = np.array([0, 1, 2])
        fs = np.stack([f1d] * 3)
        zs = np.ones_like(fs, dtype=complex)
        result = update_fres(fs, zs, fres, qres, res_idxs, method='none')
        np.testing.assert_array_equal(result, fres)
        assert result is not fres            # returns a copy

    def test_method_none_does_not_modify_input(self, f1d):
        fres = np.array([4.9e9, 5.0e9, 5.1e9])
        original = fres.copy()
        fs = np.stack([f1d] * 3)
        zs = np.ones_like(fs, dtype=complex)
        update_fres(fs, zs, fres, fres, np.array([0, 1, 2]), method='none')
        np.testing.assert_array_equal(fres, original)

    def test_invalid_method_raises(self, f1d):
        fs = np.stack([f1d])
        zs = np.ones_like(fs, dtype=complex)
        fres = np.array([5e9])
        with pytest.raises(ValueError, match="method must be"):
            update_fres(fs, zs, fres, fres, np.array([0]), method='bad')

    @pytest.mark.parametrize("method", ["mins21", "spacing", "distance"])
    def test_all_resonators_updated(self, f1d, method):
        fs, zs = self._make_batch(f1d, method)
        fres = np.array([F0] * 3)
        qres = np.array([1e4] * 3)
        res_idxs = np.array([0, 1, 2])   # all resonators
        result = update_fres(fs, zs, fres, qres, res_idxs, method=method)
        for i, k in enumerate(self.ks):
            assert result[i] == pytest.approx(f1d[k]), \
                f"method={method}, resonator {i}: expected f[{k}]={f1d[k]:.0f}, got {result[i]:.0f}"

    @pytest.mark.parametrize("method", ["mins21", "spacing", "distance"])
    def test_cal_tones_not_updated(self, f1d, method):
        """Calibration tones (res_idxs < 0) must keep their input fres."""
        fs, zs = self._make_batch(f1d, method)
        fres = np.array([4.99e9, 5.00e9, 5.01e9])
        qres = np.array([1e4] * 3)
        res_idxs = np.array([-1, 1, -2])  # indices 0 and 2 are cal tones
        result = update_fres(fs, zs, fres, qres, res_idxs, method=method)
        # Cal tones unchanged
        assert result[0] == pytest.approx(fres[0])
        assert result[2] == pytest.approx(fres[2])
        # Resonator updated
        assert result[1] == pytest.approx(f1d[self.ks[1]])

    @pytest.mark.parametrize("method", ["mins21", "spacing", "distance"])
    def test_all_cal_tones(self, f1d, method):
        """If every entry is a calibration tone, fres is returned unchanged."""
        fs = np.stack([f1d] * 3)
        zs = np.ones_like(fs, dtype=complex)
        fres = np.array([4.9e9, 5.0e9, 5.1e9])
        qres = np.array([1e4] * 3)
        res_idxs = np.array([-1, -2, -3])
        result = update_fres(fs, zs, fres, qres, res_idxs, method=method)
        np.testing.assert_array_equal(result, fres)

    def test_returns_ndarray(self, f1d):
        fs, zs = self._make_batch(f1d, 'distance')
        fres = np.array([F0] * 3)
        res_idxs = np.array([0, 1, 2])
        result = update_fres(fs, zs, fres, fres, res_idxs, method='distance')
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_input_fres_not_mutated(self, f1d):
        """update_fres must not modify the input fres array."""
        fs, zs = self._make_batch(f1d, 'distance')
        fres = np.array([F0] * 3)
        original = fres.copy()
        res_idxs = np.array([0, 1, 2])
        update_fres(fs, zs, fres, fres.copy(), res_idxs, method='distance')
        np.testing.assert_array_equal(fres, original)
