import pytest 
import numpy as np 
from citkid.res import funcs 

@pytest.mark.parametrize("y0,a", [
    ([0.0], 0.0),
    ([1], 0.0),
    ([-1], 0.0),
    ([0.0], 0.5),
    ([1], 0.5),
    ([-1], 0.5),
    ([0.0], 1.0),
    ([1], 1.0),
    ([-1], 1.0),
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.5),
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.77), 
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.8),
    ([-10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10], 0.01),
])
def test_get_y(y0, a):
    raise NotImplementedError("Need to test cardan first")
    y0 = np.array(y0, dtype = np.float64)
    # compare to numy root finding
    for largest in [True, False]:
        y_exp = []
        for y0i in y0:
            coeffs = [4, -4 * y0i, 1, -(y0i + a)]
            roots = np.roots(coeffs)
            real_mask = np.isclose(roots.imag, 0, atol=1e-12)
            real_roots = roots[real_mask].real
            y_exp.append(np.max(real_roots) if largest else np.min(real_roots))
        y_exp = np.array(y_exp, dtype = np.float64)
        y = funcs.get_y(y0, a, largest = largest)
        assert np.allclose(y, y_exp)