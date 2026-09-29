import numpy as np

def guess_p0_responsivity_int(power, x, guess_nfit = 3):
    """
    Get an initial guess for responsivity_int.
    This function works best if there are at least three data points at
    P >> P_0. Otherwise, an alternative initial guess may be required.

    Parameters:
    power (array-like): array of blackbody powers in W
    x (array-like): array of fractional frequency shifts in Hz / Hz. This must
        be scaled close to x(P = 0) = 0 for the initial guess to work well
    guess_nfit (int): number of high-power (P >> P_0) points in the data

    Returns:
    p0 (list): initial guess parameters [R0, P0, c]
    bounds (list): bounds for scipy.optimize.curve_fit
    """
    if len(power) < 4:
        e = 'Data must be at least length 4 for the initial guess.'
        e += ' Try a custom initial guess.'
        raise ValueError(e)
    poly = np.polyfit(np.sqrt(power[-guess_nfit:]), x[-guess_nfit:], 1)
    d0 = poly[1]
    if d0 < 0:
        d0 = - np.median(x)
        # It would be good to further explore this behavior. It seems to work,
        # but I don't understand why
    P0 = (d0 / poly[0]) ** 2
    R0 = - d0 / (2 * P0)
    c0 = (d0 + 1) / (1 - 2 * R0 * P0)
    p0 = [R0 / 1e9, P0 * 1e16, c0]
    bounds = get_bounds_responsivity_int(p0)
    return p0, bounds

def get_bounds_responsivity_int(p0):
    """
    Get the bounds for responsivity_int given the initial guess

    Parameters:
    p0 (list): initial guess

    Returns:
    bounds (list): [lower_bounds, upper_bounds] corresponding to p0
    """
    bounds = [[p0[0] * 10, p0[1] / 10, p0[2] * 0.9],
              [p0[0] / 10, p0[1] * 10, p0[2] * 1.1]]
    # Flip bounds if they are reversed
    for i, pi in enumerate(p0):
        if bounds[0][i] > bounds[1][i]:
            bounds[0][i], bounds[1][i] = bounds[1][i], bounds[0][i]
    return bounds
