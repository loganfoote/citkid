import numpy as np
import pytest
from citkid.res.mask import nearby_tone_mask

F = np.linspace(990., 1010., 201)  # 0.1 Hz spacing, tone at 1000


def test_no_neighbors_keeps_everything():
    assert nearby_tone_mask(F, 1000., []).all()
    # NaN and the tone itself are ignored; tones outside the sweep don't cut
    assert nearby_tone_mask(F, 1000., [np.nan, 1000., 2000., 0.]).all()


def test_cut_at_the_midpoint_on_each_side():
    mask = nearby_tone_mask(F, 1000., [1012.])
    np.testing.assert_array_equal(mask, F <= 1006.)
    mask = nearby_tone_mask(F, 1000., [988., 995., 1030.])
    np.testing.assert_array_equal(mask, F >= 997.5)
    # Both sides; only the nearest tone on each side matters
    mask = nearby_tone_mask(F, 1000., [980., 992., 1016., 1020.])
    np.testing.assert_array_equal(mask, (F >= 996.) & (F <= 1008.))


@pytest.mark.parametrize('others, min_frac', [
    ([999.4, 1000.6], 0.25),     # close tones on both sides
    ([999.9, 1004.], 0.25),      # very close on one side: widened equally
    ([999.4, 1000.6], 0.5),
])
def test_never_keeps_less_than_min_frac(others, min_frac):
    mask = nearby_tone_mask(F, 1000., others, min_frac = min_frac)
    n_min = int(np.ceil(min_frac * len(F)))
    assert mask.sum() == n_min
    kept = F[mask]
    # One contiguous window, containing the tone and the original window
    assert np.all(np.diff(np.flatnonzero(mask)) == 1)
    assert kept.min() <= 1000. <= kept.max()
    lo, hi = (1000. + others[0]) / 2, (1000. + others[1]) / 2
    assert np.all(mask[(F >= lo) & (F <= hi)])
    # Both edges moved outward by the same amount (to the sample spacing)
    assert abs((lo - kept.min()) - (kept.max() - hi)) <= 0.1 + 1e-9


def test_tiny_sweeps_keep_at_least_three_points():
    f = np.linspace(999., 1001., 5)
    assert nearby_tone_mask(f, 1000., [999.9, 1000.1]).sum() == 3
    assert nearby_tone_mask(f[:2], 1000., [999.9, 1000.1]).all()
    assert nearby_tone_mask([], 1000., [999.]).shape == (0,)
