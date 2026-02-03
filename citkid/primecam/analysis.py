import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from ..res.fitter import fit_nonlinear_iq_with_gain
from ..res.gain import fit_and_remove_gain_phase
from ..res.data_io import make_fit_row, separate_fit_row
from ..util import save_fig
from .update_ares import get_dbm
import matplotlib.pyplot as plt
from ..res.gain import remove_gain
from ..noise.analysis import compute_psd
from ..noise.data_io import save_psd
from .data_io import import_iq_noise

import warnings 

warnings.warn(
    "citkid.primecam.analysis is deprecated, and will be removed in a future"
    "version. Please use pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)

def fit_iq(directory, out_directory, file_suffix, power_number, in_atten,
           constant_atten, temperature_index, temperature,
           resonator_indices = None, extra_fitdata_values = {}, downward = True,
           plotq = False, plot_factor = 1, overwrite = False, verbose = True,
           catch_exceptions = False):
    """
    Fits all IQ loops in a target scan

    Parameters:
    directory (str): directory containing the data for logging
    out_directory (str or None): directory to save the plots and data, or
        None to bypass saving data
    file_suffix (str): suffix of saved files
    power_number (int): power index for logging
    in_atten (np.array): variable input attenuations for logging
    constant_atten (np.array): constant input attenuations for logging. The
        total attenuation between the RFSoC and the device is
        in_atten + constant_atten. Any amplification on the input to the
        cryostat should be taken into account here
    temperature_index (int): temperature index for logging
    temperature (float): temperature in K for logging
    resonator_indices (np.array or None): If np.array, list of resonator
        indices corresponding to each resonator in the target sweep. If
        None, resonator indices are assigned by their index into fres
    extra_fitdata_values (dict): keys (str) are data column names and values
        (single value or np.array with same length as number of targets) are set
        to that data column
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.
    plotq (bool): If True, plots IQ fits and saves them
    plot_factor (int): for plotting a subset of resonators. Plots every
        plot_factor resonators
    overwrite (bool): if not True, raises an exception if the output data file
        already exists
    verbose (bool): If True, displays a progress bar as data is taken
    catch_exceptions (bool): If True, catches any exceptions that occur while
        fitting and proceeds

    Returns:
    data (pd.DataFrame): DataFrame of fit data
    """
    directory = os.path.normpath(os.path.expanduser(directory))
    # Import data
    fres_initial, fres, ares, Qres, fcal_indices, frough, zrough,\
           fgain, zgain, ffine, zfine, znoise, noise_dt =\
    import_iq_noise(directory, file_suffix, import_noiseq = False)
    # Set up output files
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    if out_directory is not None:
        out_directory = os.path.normpath(os.path.expanduser(out_directory))
        os.makedirs(out_directory, exist_ok = True)
        fit_plot_directory = os.path.join(out_directory, 'plots_iq')
        if not os.path.exists(fit_plot_directory) and plotq:
            os.makedirs(fit_plot_directory)
        out_path = os.path.join(out_directory, f'fitdata{file_suffix}.csv')
        if not overwrite and os.path.exists(out_path):
            raise Exception(f'{out_path} already exists!!!')
    # Split data
    fgains, zgains = split_sweep(fgain, zgain, len(fgain) // len(fres))
    ffines, zfines = split_sweep(ffine, zfine, len(ffine) // len(fres))
    Qres = np.array(Qres, dtype = float)
    Qres[fcal_indices] = np.inf
    # Assign resonator indices if they are not given
    data = pd.DataFrame([])
    if resonator_indices is None:
        resonator_indices = list(range(len(fres)))
    # Iterate through resonators and fit
    pbar = resonator_indices
    if verbose:
        pbar = tqdm(pbar, leave = False)
        pbar.set_description('Fitting IQ Loops')
    for pbar_index, resonator_index in enumerate(pbar):
        plotq_single = ((pbar_index % plot_factor) == 0) and plotq
        ffine, zfine = ffines[pbar_index], zfines[pbar_index]
        fgain, zgain = fgains[pbar_index], zgains[pbar_index]
        fr, Qr = fres[pbar_index], Qres[pbar_index]
        # Cut adjacent resonators from data before fitting
        for index in range(len(fres)):
            fr, Qr = fres[index], Qres[index]
            if len(ffine):
                span = fr / (2 * Qr)
                if abs((fr - np.mean(ffine)) > 1e3):
                    ix0 = ffine < fr - span
                    ix1 = ffine > fr + span
                    ix_fine = ix0|ix1
                    ffine, zfine = ffine[ix_fine], zfine[ix_fine]

        file_prefix = f'Tn{temperature_index}Fn{resonator_index}'
        file_prefix += f'Pn{power_number}{file_suffix}'
        if plotq_single:
            plot_path = os.path.join(
                fit_plot_directory,
                f'{file_prefix}_fit.png',
            )
        else:
            plot_path = ''
        try:
            if pbar_index not in fcal_indices:
                # For on-resonance, fit IQ loops
                fitrow, fig = \
                    fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine, fres,
                                               Qres, plotq = plotq_single,
                                               downward = downward,
                                               return_dataframe = True)
                fitrow['plotpath'] = plot_path
            else:
                # for off-resonance, just fit gain
                p_amp, p_phase, z_rmvd, (fig, axs) = \
                    fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, fres,
                                              Qres, plotq = plotq_single)
                p = [np.nan] * 7
                res = np.nan
                fitrow = make_fit_row(p_amp, p_phase, p, p, p, res,
                                      downward = downward,
                                      plot_path = plot_path)
            if not fig is None:
                save_fig(fig, file_prefix + '_fit', fit_plot_directory)
                plt.close(fig)
        except Exception as e:
            if not catch_exceptions:
                raise e
            fitrow = pd.Series([], dtype = 'object')
        fitrow['resonatorIndex'] = resonator_index
        fitrow['dataIndex'] = pbar_index
        fitrow['f0'] = np.mean(ffine) # Mean of ffine is the noise frequency
        fitdf = pd.DataFrame(fitrow).T
        data = pd.DataFrame(pd.concat([data, fitdf]))

    data['dataDirectory'] = directory
    data['temperature'] = temperature
    data['temperatureIndex'] = temperature_index

    data['powerNumber'] = power_number
    data['rfsocPower'] = ares
    data['inAtten'] = in_atten
    data['constantAtten'] = constant_atten
    data['outputPower'] = get_dbm(ares, fres)
    data['power'] = data.outputPower - data.inAtten - data.constantAtten
    for key in extra_fitdata_values:
        data[key] = extra_fitdata_values[key]
    data = data.reset_index(drop = True)
    if out_directory is not None:
        data.to_csv(out_path, index = False)
    return data

def analyze_noise(main_out_directory, file_suffix, noise_index, tstart = 0,
                  plot_calq = False, plot_psdq = False,
                  plot_timestreamq = False, plot_factor = 1,
                  deglitch_nstd = 10, cr_nstd = 5, cr_width = 100e-6,
                  cr_peak_spacing = 100e-6, cr_removal_time = 1e-3,
                  overwrite = False, verbose = False, catch_exceptions = False):
    """
    Analyze noise data to produce timestreams and PSDs

    Parameters:
    main_out_directory (str): directory where the IQ loop fit data csv file is
        saved
    file_suffix (str): file suffix of the data
    noise_index (int): noise index to analyze
    tstart (float): number of seconds from the beginning of the timestream to
        cut out of the data.
    plot_calq (bool): If True, plots the calibrations
    plot_psdq (bool): If True, plots the PSDs
    plot_timestreamq (bool): If True, plots the timestreams
    plot_factor (int): for plotting a subset of data. Plots every plot_factor
        datasets
    deglitch_nstd (float or None): threshold for removing glitched data points
        from the timestream, or None to bypass deglitching. Points more than
        deglitch_nstd times the standard deviations of the theta timestream are
        removed from the data.
    cr_nstd (float): number of standard deviations above the mean for find_peaks
    cr_width (int): width of cosmic rays in seconds
    cr_peak_spacing (float): number of seconds spacing between cosmic rays
    cr_removal_time (float): number of seconds to remove around each peak
    overwrite (bool): if False, raises an error instead of overwriting files
    verbose (bool): if True, displays a progress bar while analyzing noise
    catch_exceptions (bool): If True, catches any exceptions that occur while
        analyzing noise data and proceeds

    Returns:
    data_new (pd.DataFrame): output data with the noise analysis parameters
        inserted. This DataFrame is also saved as a csv file in
        main_out_directory with name
        f'fitdata_noise{file_suffix}_{noise_index:02d}.csv'
    """
    main_out_directory = os.path.normpath(
        os.path.expanduser(main_out_directory)
    )
    out_directory = os.path.join(main_out_directory, 'noise_data')
    plot_directory = os.path.join(main_out_directory, 'noise_plots')
    os.makedirs(out_directory, exist_ok = True)
    if any([plot_calq, plot_psdq, plot_timestreamq]):
        os.makedirs(plot_directory, exist_ok = True)
    file_suffix0 = file_suffix
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    data = pd.read_csv(
        os.path.join(main_out_directory, f'fitdata{file_suffix}.csv')
    )
    outpath = os.path.join(
        main_out_directory,
        f'fitdata_noise{file_suffix}_{noise_index:02d}.csv',
    )
    if os.path.exists(outpath) and not overwrite:
        raise Exception(f'{outpath} already exists!!!')
    # Import data
    directory = data.iloc[0].dataDirectory
    fres_initial, fres, ares, Qres, fcal_indices, frough, zrough,\
           fgain, zgain, ffine, zfine, znoise, noise_dt =\
    import_iq_noise(directory, file_suffix0, import_noiseq = False)
    inoise, qnoise = np.load(
        os.path.join(directory, f'noise{file_suffix}_{noise_index:02d}.npy')
    )
    dt = float(
        np.load(
            os.path.join(
                directory,
                f'noise{file_suffix}_{noise_index:02d}_tsample.npy',
            )
        )
    )
    fgains, zgains = split_sweep(fgain, zgain, len(fgain) // len(fres))
    ffines, zfines = split_sweep(ffine, zfine, len(ffine) // len(fres))

    pbar = list(range(len(ffines)))
    if verbose:
        pbar = tqdm(pbar, leave = False)
        pbar.set_description('noise index')
    data_new = pd.DataFrame([])
    for index in pbar:
        plot_calq_single = ((index % plot_factor) == 0) and plot_calq
        plot_psdq_single = ((index % plot_factor) == 0) and plot_psdq
        plot_timestreamq_single = ((index % plot_factor) == 0) and plot_timestreamq
        prefix = f'Fn{index:02d}_NI{noise_index}'
        iq_fit_row = data[data.dataIndex == index].iloc[0]
        ffine, zfine = ffines[index], zfines[index]

        i, q = inoise[index], qnoise[index]
        fnoise = fres[index]
        znoise = i + 1j * q
        znoise = znoise[int(tstart / dt):]

        p_amp, p_phase, p0, popt, perr, res, downward, plot_path =\
            separate_fit_row(iq_fit_row)

        zfine = remove_gain(ffine, zfine, p_amp, p_phase)
        znoise = remove_gain(fnoise, znoise, p_amp, p_phase)
        try:
            if index in fcal_indices:
                psd_onres, psd_offres, timestream_onres, timestream_offres,\
                    cr_indices, theta_range, poly, xcal_data, figs =\
                compute_psd(ffine, zfine, None, None, None,
                            fnoise_offres = fnoise, znoise_offres = znoise,
                            dt_offres = dt, flag_crs = False,
                            deglitch_nstd = deglitch_nstd,
                            plot_calq = plot_calq_single,
                            plot_psdq = plot_psdq_single,
                            plot_timestreamq = plot_timestreamq_single)
                row =\
                save_psd(psd_onres, psd_offres, timestream_onres, timestream_offres,
                 cr_indices, theta_range, poly, xcal_data, figs, None, dt,
                 out_directory, plot_directory, prefix = prefix, iq_fit_row = iq_fit_row)
            else:
                psd_onres, psd_offres, timestream_onres, timestream_offres,\
                    cr_indices, theta_range, poly, xcal_data, figs =\
                compute_psd(ffine, zfine, fnoise, znoise, dt,
                            fnoise_offres = None, znoise_offres = None,
                            dt_offres = None, flag_crs = True,
                            deglitch_nstd = deglitch_nstd,
                            plot_calq = plot_calq_single,
                            plot_psdq = plot_psdq_single,
                            plot_timestreamq = plot_timestreamq_single,
                            cr_nstd = cr_nstd, cr_width = cr_width,
                            cr_peak_spacing = cr_peak_spacing,
                            cr_removal_time = cr_removal_time)
                row =\
                save_psd(psd_onres, psd_offres, timestream_onres, timestream_offres,
                 cr_indices, theta_range, poly, xcal_data, figs, dt, None,
                 out_directory, plot_directory, prefix = prefix, iq_fit_row = iq_fit_row)
        except Exception as e:
            if not catch_exceptions:
                raise e
            row = iq_fit_row
            plt.close('all')
        row['noiseFrequency'] = fnoise
        data_new = pd.concat([data_new, pd.DataFrame(row).T])
    data_new = data_new.reset_index(drop = True)
    data_new.to_csv(outpath, index = False)
    return data_new

################################################################################
######################### Utility functions ####################################
################################################################################
def split_sweep(f, z, npoints):
    """
    Splits an S21 sweep into subarrays corresponding to each tone

    Parameters:
    f, z (np.array): multi-tone sweep frequency and complex S21 data
    npoints (int): number of points

    Returns:
    fs, zs (list): values (np.array) are single-tone sweep frequency and complex
        S21 data for each tone
    """
    fs = [f[npoints * i: npoints * (i + 1)] for i in range(len(f) // npoints)]
    zs = [z[npoints * i: npoints * (i + 1)] for i in range(len(f) // npoints)]
    return fs, zs
