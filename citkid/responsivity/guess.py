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

def guess_p0_responsivity_int_x0(power, x, guess_nfit = 3):
    """
    Get an initial guess for responsivity_int_x0.
    This function works best if there are at least three data points at
    P >> P_0 and the lowest-power point is at P << P_0. Otherwise, an
    alternative initial guess may be required.

    Parameters:
    power (array-like): array of blackbody powers in W, sorted in ascending
        order.
    x (array-like): array of fractional frequency shifts in Hz / Hz,
        corresponding to power.
    guess_nfit (int): number of high-power (P >> P_0) points in the data

    Returns:
    p0 (list): initial guess parameters [R0, P0, x0], scaled for
        responsivity_int_x0_for_fitter
    bounds (list): bounds for scipy.optimize.curve_fit
    """
    if len(power) < 4:
        e = 'Data must be at least length 4 for the initial guess.'
        e += ' Try a custom initial guess.'
        raise ValueError(e)
    power, x = np.asarray(power), np.asarray(x)
    # x at the lowest power approximates x0
    x0 = x[0]
    # At P >> P0, x ~ 2 R0 sqrt(P0 P) + x0 - 2 R0 P0
    poly = np.polyfit(np.sqrt(power[-guess_nfit:]), x[-guess_nfit:], 1)
    d0 = poly[1] - x0 # = -2 R0 P0
    if d0 < 0:
        d0 = x0 - np.median(x)
    P0 = (d0 / poly[0]) ** 2
    R0 = - d0 / (2 * P0)
    p0 = [R0 / 1e9, P0 * 1e16, x0 * 1e6]
    bounds = get_bounds_responsivity_int_x0(p0, np.ptp(x) * 1e6)
    return p0, bounds

def get_bounds_responsivity_int_x0(p0, x0_span):
    """
    Get the bounds for responsivity_int_x0 given the initial guess

    Parameters:
    p0 (list): initial guess [R0, P0, x0], scaled for
        responsivity_int_x0_for_fitter
    x0_span (float): x0 is bounded to p0[2] +/- x0_span, in the scaled units
        of p0[2] (1e6 * x)

    Returns:
    bounds (list): [lower_bounds, upper_bounds] corresponding to p0
    """
    bounds = [[p0[0] * 10, p0[1] / 10, p0[2] - x0_span],
              [p0[0] / 10, p0[1] * 10, p0[2] + x0_span]]
    # Flip bounds if they are reversed
    for i, pi in enumerate(p0):
        if bounds[0][i] > bounds[1][i]:
            bounds[0][i], bounds[1][i] = bounds[1][i], bounds[0][i]
    return bounds

def guess_p0_responsivity_int_x0_fixed_R0(power, x, R0, guess_nfit = 3):
    """
    Get an initial guess for responsivity_int_x0 with R0 fixed.
    This function works best if there are at least three data points at
    P >> P_0 and the lowest-power point is at P << P_0. Otherwise, an
    alternative initial guess may be required.

    Parameters:
    power (array-like): array of blackbody powers in W, sorted in ascending
        order.
    x (array-like): array of fractional frequency shifts in Hz / Hz,
        corresponding to power.
    R0 (float): fixed responsivity at P = 0 (1 / W).
    guess_nfit (int): number of high-power (P >> P_0) points in the data

    Returns:
    p0 (list): initial guess parameters [P0, x0], scaled for
        responsivity_int_x0_for_fitter
    bounds (list): bounds for scipy.optimize.curve_fit
    """
    if len(power) < 3:
        e = 'Data must be at least length 3 for the initial guess.'
        e += ' Try a custom initial guess.'
        raise ValueError(e)
    power, x = np.asarray(power), np.asarray(x)
    # x at the lowest power approximates x0
    x0 = x[0]
    # At P >> P0, x ~ 2 R0 sqrt(P0 P) + x0 - 2 R0 P0
    poly = np.polyfit(np.sqrt(power[-guess_nfit:]), x[-guess_nfit:], 1)
    P0 = (poly[0] / (2 * R0)) ** 2
    if not np.isfinite(P0) or P0 <= 0:
        P0 = np.median(power)
    p0 = [P0 * 1e16, x0 * 1e6]
    bounds = get_bounds_responsivity_int_x0_fixed_R0(p0, np.ptp(x) * 1e6)
    return p0, bounds

def get_bounds_responsivity_int_x0_fixed_R0(p0, x0_span):
    """
    Get the bounds for responsivity_int_x0 with R0 fixed given the initial
    guess

    Parameters:
    p0 (list): initial guess [P0, x0], scaled for
        responsivity_int_x0_for_fitter
    x0_span (float): x0 is bounded to p0[1] +/- x0_span, in the scaled units
        of p0[1] (1e6 * x)

    Returns:
    bounds (list): [lower_bounds, upper_bounds] corresponding to p0
    """
    return [[p0[0] / 10, p0[1] - x0_span],
            [p0[0] * 10, p0[1] + x0_span]]
