import numpy as np
import pytest
from citkid.responsivity.funcs import (
    responsivity,
    responsivity_int,
    responsivity_int_x0,
)

P0 = 1e-12


@pytest.mark.parametrize("R0", [-1e8, 1e8])
def test_responsivity_is_derivative_of_integrated_forms(R0):
    P = np.linspace(0., 10 * P0, 2001)
    expected = responsivity(P, R0, P0)
    for x in [responsivity_int_x0(P, R0, P0, 1e-5),
              responsivity_int(P, R0, P0, 1.)]:
        np.testing.assert_allclose(np.gradient(x, P)[1:-1], expected[1:-1],
                                   rtol = 1e-4)


def test_responsivity_negative_R0_means_x_falls_with_power():
    P = np.linspace(0., 10 * P0, 11)
    assert np.all(responsivity(P, -1e8, P0) < 0)
    assert np.all(np.diff(responsivity_int_x0(P, -1e8, P0, 0.)) < 0)
