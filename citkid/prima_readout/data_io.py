import numpy as np
import os
from ..util import save_fig, fix_path

def load_x_cal(path):
    """
    Loads IQ to x calibration data

    Parameters:
    path (str): path to the data to load

    Returns:
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the IQ circle
    v (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    p_x (array-like): polynomial fit parameters to x versus theta
    """
    data = np.load(fix_path(path))
    p_amp, p_phase = data['p_amp'], data['p_phase']
    origin, v = data['origin'], data['v']
    p_x = data['p_x']
    return p_amp, p_phase, origin, v, p_x

def save_x_cal(data_directory, filename, p_amp, p_phase, origin, v, p_x, figs,
               plot_directory = '', make_directories = False):
    """
    Saves IQ to x calibration data and plots

    Parameters:
    data_directory (str): directory to save the calibration data
    filename (str): filename for the output files, without file extension
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the IQ circle
    v (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    p_x (array-like): polynomial fit parameters to x versus theta
    figs:
        fig_gain (pyplot.figure or None): gain sweep calibration plot if plotq,
            else None
        fig_fine (pyplot.figure or None): fine sweep calibration plot if plotq,
            else None
        fig_x (pyplot.figure or None): theta to x calibration plot if plotq,
            else None
    plot_directory (str): directory to save the plots, if they exist
    make_directories (bool): if True, makes the data and plot directories if
        they do not exist
    """
    # set up output directories
    data_directory = fix_path(data_directory)
    plot_directory = fix_path(plot_directory)
    plotq = any([f is not None for f in figs])
    if make_directories:
        os.makedirs(data_directory, exist_ok = True)
    if make_directories and plotq:
        os.makedirs(plot_directory, exist_ok = True)
    if not os.path.exists(data_directory):
        e = f"The directory '{data_directory}' does not exist."
        raise FileNotFoundError(e)
    if not os.path.exists(plot_directory) and plotq:
        e = f"The directory '{plot_directory}' does not exist."
        raise FileNotFoundError(e)
    # save data
    path = data_directory + filename + '.npz'
    np.savez(path, p_amp = p_amp, p_phase = p_phase, origin = origin, v = v,
             p_x = p_x)
    # save plots
    if plotq:
        for fig, suffix in zip(figs, ['gain', 'fine', 'x']):
            plot_filename = filename + '_' + suffix
            save_fig(fig, plot_filename, plot_directory, ftype = 'png')
