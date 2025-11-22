import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from ..res.fitter import fit_nonlinear_iq_with_gain
from ..res.funcs import nonlinear_iq
from ..res.gain import fit_and_remove_gain_phase
from ..res.data_io import make_fit_row, separate_fit_row
from ..util import fix_path, save_fig
import matplotlib.pyplot as plt
from ..res.gain import remove_gain
from ..noise.analysis import compute_psd
from ..noise.data_io import save_psd
from .data_io import import_iq_noise
from .fres import cut_fine_sweep
from rfmux.core.transferfunctions import compensate_psd_for_cics
import matplotlib
matplotlib.use('Agg')

def fit_iq(directory, out_directory, file_suffix, power_number, in_atten,
           constant_atten, temperature_index, temperature, rejected_points = [],
           extra_fitdata_values = {}, plotq = False, plot_factor = 1,
           downward = True, cut_to_qres = False, overwrite = False,
           verbose = True, catch_exceptions = False):
    """
    Fits all IQ loops in a target sweep

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
    rejected_points (array-like): indices to discard from fine sweep data before
        fitting
    extra_fitdata_values (dict): keys (str) are data column names and values
        (single value or np.array with same length as number of targets) are set
        to that data column
    plotq (bool): If True, plots IQ fits and saves them
    plot_factor (int): for plotting a subset of resonators. Plots every
        plot_factor resonators
    downward (bool): If True, solves the equation for a downward sweep. If
        False, solves for an upward sweep.
    cut_to_qres (bool): If True, cuts the fine sweep data to the span of fres / qres
    overwrite (bool): if not True, raises an exception if the output data file
        already exists
    verbose (bool): If True, displays a progress bar as data is taken
    catch_exceptions (bool): If True, catches any exceptions that occur while
        fitting and proceeds

    Returns:
    data (pd.DataFrame): DataFrame of fit data
    """
    directory = fix_path(directory)
    # Import data
    fres_initial, fres, ares, qres, fcal_indices, fres_all, qres_all, frough, zrough,\
           fgains, zgains, ffines, zfines, znoises, noise_dt, res_indices, fres_noise =\
    import_iq_noise(directory, file_suffix, import_noiseq = False)
    rejected_points = list(rejected_points)
    # Set up output files
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    if out_directory is not None:
        out_directory = fix_path(out_directory)
        os.makedirs(out_directory, exist_ok = True)
        fit_plot_directory = out_directory + 'plots_iq/'
        if not os.path.exists(fit_plot_directory) and plotq:
            os.makedirs(fit_plot_directory)
        out_path = out_directory + f'fitdata{file_suffix}.csv'
        if not overwrite and os.path.exists(out_path):
            raise Exception(f'{out_path} already exists!!!')
    # Split data
    qres = np.array(qres, dtype = float)
    qres[fcal_indices] = np.inf
    qres_all = np.array(qres_all, dtype = float)
    # Assign resonator indices if they are not given
    data = pd.DataFrame([])
    # Iterate through resonators and fit
    pbar = res_indices
    if verbose:
        pbar = tqdm(pbar, leave = False)
        pbar.set_description('Fitting IQ Loops')
    for pbar_index, resonator_index in enumerate(pbar):
        plotq_single = ((pbar_index % plot_factor) == 0) and plotq
        ffine, zfine = ffines[pbar_index], zfines[pbar_index]
        fgain, zgain = fgains[pbar_index], zgains[pbar_index]
        if len(rejected_points):
            ffine, zfine = np.delete(ffine, rejected_points), np.delete(zfine, rejected_points)
        fr, Qr = fres[pbar_index], qres[pbar_index]
        # Cut adjacent resonators from data before fitting
        if pbar_index not in fcal_indices:
            ffine, zfine = cut_fine_sweep(ffine, zfine, fres, fres / qres)
        if cut_to_qres:
            ix = np.abs(ffine - fr) < fr / Qr
            ffine, zfine = ffine[ix], zfine[ix]

        file_prefix = f'Tn{temperature_index}Fn{resonator_index}'
        file_prefix += f'Pn{power_number}{file_suffix}'
        if plotq_single:
            plot_path = fit_plot_directory + file_prefix + '_fit.png'
        else:
            plot_path = ''
        try:
            if pbar_index not in fcal_indices:
                # For on-resonance, fit IQ loops
                fitrow, fig = \
                    fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine,
                                               fres_all, qres_all,
                                               plotq = plotq_single,
                                               downward = downward,
                                               return_dataframe = True)
                fitrow['plotpath'] = plot_path
                fitrow['fcal'] = 0
            else:
                # for off-resonance, just fit gain
                p_amp, p_phase, z_rmvd, (fig, axs) = \
                    fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, fres_all,
                                              qres_all, plotq = plotq_single)
                p = [np.nan] * 7
                res = np.nan
                fitrow = make_fit_row(p_amp, p_phase, p, p, p, res,
                                      downward = downward,
                                      plot_path = plot_path)
                fitrow['fcal'] = 1
            if not fig is None:
                save_fig(fig, file_prefix + '_fit', fit_plot_directory)
                fig.clear()
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
    if callable(constant_atten):
        data['constantAtten'] = constant_atten(data.f0)
    else:
        data['constantAtten'] = constant_atten
    data['outputPower'] = ares
    data['power'] = data.outputPower - data.inAtten - data.constantAtten
    for key in extra_fitdata_values:
        data[key] = extra_fitdata_values[key]
    data = data.reset_index(drop = True)
    if out_directory is not None:
        data.to_csv(out_path, index = False)
    return data

def analyze_noise(main_out_directory, file_suffix, noise_index, tstart = 0,
                  plot_calq = False, plot_psdq = False,
                  plot_timestreamq = False, plot_factor = 1, min_cal_points = 5,
                  deglitch_nstd = 10, cr_nstd = 5, cr_width = 100e-6,
                  cr_peak_spacing = 100e-6, cr_removal_time = 1e-3,
                  circfit_npoints = None, correct_cic = False,
                  overwrite = False, catch_exceptions = False,
                  res_whitelist = None, xcal_weight_sigma = None,
                  xcal_weight_theta0 = 0.0, circfit_mode = 'sequential',
                  verbose = False):
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
    min_cal_points (int): minimum number of points for the x calibration fit
    deglitch_nstd (float or None): threshold for removing glitched data points
        from the timestream, or None to bypass deglitching. Points more than
        deglitch_nstd times the standard deviations of the theta timestream are
        removed from the data.
    cr_nstd (float): number of standard deviations above the mean for find_peaks
    cr_width (int): width of cosmic rays in seconds
    cr_peak_spacing (float): number of seconds spacing between cosmic rays
    cr_removal_time (float): number of seconds to remove around each peak
    circfit_npoints (int): if not None, limits the number of points in the
        circle fit to circfit_npoints around the noise ball
    correct_cic (bool): If True, corrects the PSDs for the CIC rolloff
        NOT IMPLEMENTED YET: WAITING FOR RFMUX UPDATE
    overwrite (bool): if False, raises an error instead of overwriting files
    catch_exceptions (bool): If True, catches any exceptions that occur while
        analyzing noise data and proceeds
    res_whitelist (list or None): if not None, only process the resonator
		indices listed; otherwise process all
    xcal_weight_sigma (float): stdev of gaussian weight function for
        fitter in radians. Defaults to None for no weighting.
    xcal_weight_theta0 (float): center point in radians for gaussian
        weight function
    circfit_mode (str): method used to select the points used in fitting
		the IQ circle ('sequential' uses adjacent indices, 'nearest_z'
		uses distance in the z plane)
    verbose (bool): if True, displays a progress bar while analyzing noise

    Returns:
    data_new (pd.DataFrame): output data with the noise analysis parameters
        inserted. This DataFrame is also saved as a csv file in
        main_out_directory with name
        f'fitdata_noise{file_suffix}_{noise_index:02d}.csv'
    """
    out_directory = main_out_directory + 'noise_data/'
    plot_directory = main_out_directory + 'noise_plots/'
    os.makedirs(out_directory, exist_ok = True)
    if any([plot_calq, plot_psdq, plot_timestreamq]):
        os.makedirs(plot_directory, exist_ok = True)
    file_suffix0 = file_suffix
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    data = pd.read_csv(main_out_directory + f'fitdata{file_suffix}.csv')
    outpath = main_out_directory + f'fitdata_noise{file_suffix}_{noise_index:02d}.csv'
    if os.path.exists(outpath) and not overwrite:
        raise Exception(f'{outpath} already exists!!!')
    # Import data
    directory = data.iloc[0].dataDirectory
    fres_initial, fres, ares, qres, fcal_indices, fres_all, qres_all, frough, zrough,\
           fgains, zgains, ffines, zfines, znoises, noise_dt, res_indices, fres_noise =\
    import_iq_noise(directory, file_suffix0, import_noiseq = True)
    inoise, qnoise = np.load(directory + f'noise{file_suffix}_{noise_index:02d}.npy')
    dt = float(np.load(directory + f'noise{file_suffix}_tsample_{noise_index:02d}.npy'))

    pbar = res_indices
    if verbose:
        pbar = tqdm(pbar, leave = False)
        pbar.set_description('noise index')
    data_new = pd.DataFrame([])
    for data_index, res_index in enumerate(pbar):
        if (res_whitelist is not None) and (res_index not in res_whitelist):
            continue
        plot_calq_single = ((data_index % plot_factor) == 0) and plot_calq
        plot_psdq_single = ((data_index % plot_factor) == 0) and plot_psdq
        plot_timestreamq_single = ((data_index % plot_factor) == 0) and plot_timestreamq
        prefix = f'Fn{res_index:02d}_NI{noise_index}{file_suffix}'
        iq_fit_row = data[data.dataIndex == data_index].iloc[0]
        ffine, zfine = ffines[data_index], zfines[data_index]

        i, q = inoise[data_index], qnoise[data_index]
        fnoise = fres_noise[data_index]
        znoise = i + 1j * q
        znoise = znoise[int(tstart / dt):]

        p_amp, p_phase, p0, popt, perr, res, downward, plot_path =\
            separate_fit_row(iq_fit_row)

        zfine = remove_gain(ffine, zfine, p_amp, p_phase)
        znoise = remove_gain(fnoise, znoise, p_amp, p_phase)

        if circfit_mode == 'sequential':
            # Sequential steps near noise mean (previous default behavior)
            if circfit_npoints is not None:
                ix_mid = np.argmin(np.abs(np.mean(znoise) - zfine))
                ix0, ix1 = ix_mid - circfit_npoints // 2, ix_mid + (circfit_npoints - circfit_npoints // 2)
                ix0 = max(ix0, 0)
                ix1 = min(ix1, len(ffine))
                ffine, zfine = ffine[ix0:ix1], zfine[ix0:ix1]
        elif circfit_mode == 'nearest_z':
            # Take the n points closest to the median of the noise by
            # checking the 2D distance in z.  Helps exclude large
            # IQ sweep jumps next to noise balls.
            zfine_dist = np.abs(zfine - np.nanmedian(znoise))
            i_nearest = np.argsort(zfine_dist)[:circfit_npoints]
            ffine, zfine = ffine[i_nearest], zfine[i_nearest]
        else:
            raise ValueError("Invalid value for circfit_mode: " + str(circfit_mode))

        assert len(zfine), "No fine sweep points remain after trim"

        try:
            if data_index in fcal_indices:
                psd_onres, psd_offres, timestream_onres, timestream_offres,\
                    cr_indices, theta_range, poly, xcal_data, figs =\
                compute_psd(ffine, zfine, None, None, None,
                            fnoise_offres = fnoise, znoise_offres = znoise,
                            dt_offres = dt, flag_crs = False,
                            deglitch_nstd = deglitch_nstd,
                            plot_calq = plot_calq_single,
                            plot_psdq = plot_psdq_single, min_cal_points = min_cal_points,
                            plot_timestreamq = plot_timestreamq_single,
                            xcal_weight_theta0 = xcal_weight_theta0,
                            xcal_weight_sigma = xcal_weight_sigma)
                if correct_cic:
                    for i in range(1, 3):
                        ftrim_off, s = compensate_psd_for_cics(psd_offres[0], 10 ** (psd_offres[i] / 10),
                                                dec_stage=fir_stage, spectrum_cutoff=0.9)
                        psd_offres[i] = 10 * np.log10(s)
                    psd_offres[0] = ftrim_off
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
                            plot_psdq = plot_psdq_single, min_cal_points = min_cal_points,
                            plot_timestreamq = plot_timestreamq_single,
                            cr_nstd = cr_nstd, cr_width = cr_width,
                            cr_peak_spacing = cr_peak_spacing,
                            cr_removal_time = cr_removal_time,
                            xcal_weight_theta0 = xcal_weight_theta0,
                            xcal_weight_sigma = xcal_weight_sigma)

                if correct_cic:
                    for i in range(1, 4):
                        ftrim_off, s = compensate_psd_for_cics(psd_offres[0], 10 ** (psd_offres[i] / 10),
                                                dec_stage=fir_stage, spectrum_cutoff=0.9)
                        psd_onres[i] = 10 * np.log10(s)
                    ftrim_on, psd_onres[3] = apply_cic2_comp_psd(psd_onres[0], psd_onres[3], 1 / dt, trim=0.15)
                    psd_onres[0] = ftrim_on
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


def plot_fits_batch(directory, file_suffix, plot_directory):
    """

    Parameters:
    directory (str): directory containing the data to fit
    file_suffix (str): file suffix of the data
    plot_directory (str): directory to save plots
    """
    warnings.warn('Someone needs to document this', UserWarning)
    data = fit_iq(directory, None, file_suffix, 0, 0, 0, 0, 0, rejected_points = [],
                      plotq = False, verbose = True, catch_exceptions = True)

    fres_initial, fres, ares, qres, fcal_indices, fres_all, qres_all, frough, zrough,\
           fgain, zgain, ffine, zfine, znoises, noise_dt, res_indices0 =\
    import_iq_noise(directory, file_suffix, import_noiseq = False)

    fs, zs, popts, ress, res_indices = [], [], [], [], []
    for index in [d for d in range(len(fres)) if d not in fcal_indices]:
        row = data[data.dataIndex == index].iloc[0]
        p_amp, p_phase, p0, popt, perr, res, downward, plot_path = separate_fit_row(row, prefix = 'iq')

        ff, zf = ffine[index], zfine[index]
        zs.append(remove_gain(ff, zf, p_amp, p_phase))
        fs.append(ff)
        popts.append(popt)
        ress.append(res)
        res_indices.append(res_indices0[index])
        # fres.append(fres[index])

    plot_directory = fix_path(plot_directory)
    os.makedirs(plot_directory, exist_ok = True)


    num_plots = len(fs)

    plots_per_fig = 100
    max_n_cols = 4
    num_figs = (num_plots - 1) // plots_per_fig + 1

    for fig_index in range(num_figs):
        ix0, ix1 = fig_index * plots_per_fig, (fig_index + 1) * plots_per_fig
        f0, z0 = fs[ix0:ix1], zs[ix0:ix1]
        rs0 = res_indices[ix0:ix1]
        popt0, res0 = popts[ix0:ix1], ress[ix0:ix1]
        res_ix0 = res_indices[ix0:ix1]
        fres0 = fres[ix0:ix1]

        data_indices = np.arange(ix0, ix1, 1)

        naxs = len(f0)
        nrows = naxs // max_n_cols
        len_last_row = naxs % max_n_cols
        if len_last_row > max_n_cols or len_last_row == 0:
            ncols = max_n_cols
        else:
            ncols = len_last_row
        fig, axs = plt.subplots(nrows, ncols, figsize = [3 * ncols, 2.5 * nrows], layout = 'tight')
        axs = axs.flatten()
        for index, ax in enumerate(axs):
            ax.set(ylabel = 'Q', xlabel = 'I')

            f, z = f0[index], z0[index]
            popt, res = popt0[index], res0[index]
            ri = res_ix0[index]
            fr = fres0[index]
            ax.set(title = f'Fn {ri}')

            color = plt.cm.viridis(0.)

            ax.plot(np.real(z), np.imag(z), '.', color = color)
            fsamp = np.linspace(min(f), max(f), 200)
            zsamp = nonlinear_iq(fsamp, *popt)
            ax.plot(np.real(zsamp), np.imag(zsamp), '--k')

        save_fig(fig, f'fres_update_{fig_index}', plot_directory, ftype = 'png',
                            tight_layout = False, close_fig = True)

################################################################################
######################### Utility functions ####################################
################################################################################

def make_cal_tones(fres, ares, qres, max_n_tones = 1000,
                   res_indices = None, fcal_power = -55):
    '''
    Adds calibration tones to the given resonator list. Fills in largest spaces
    between resonators, up to max_n_tones.

    Parameters:
    fres (np.array): frequency array in Hz
    ares (np.array): amplitude array in Hz
    qres (np.array): span factor array
    max_n_tones (int): maximum number of tones after adding cal tones
    res_indices (np.array or None): resonator indices corresponding to
        fres
    fcal_power (float): calibration tone power, in the same units as ares

    Returns:
    fres, ares, qres (np.array): frequency, amplitude, and span factor arrays
        with calibration tones added
    fcal_indices (np.array): calibration tone indices into fres, ares, qres
    new_res_indices (np.array): new resonator index list with the new
        calibration tones. Calibration tone resonator indices are negative
    '''
    fres = np.asarray(fres, dtype = float)
    ares = np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    ix = np.argsort(fres)
    fres, ares, qres = fres[ix], ares[ix], qres[ix]
    if res_indices is not None:
        res_indices = np.asarray(res_indices)
        res_indices = res_indices[ix]
    if res_indices is None:
        res_indices = np.asarray(range(len(fres)))
    new_res_indices = np.asarray(res_indices, dtype = int)

    ix = np.flip(np.argsort(np.diff(fres) / fres[:1]))[:max_n_tones - len(fres)]
    fcal = np.sort([np.mean(fres[i:i+2]) for i in ix])
    fcal_indices = np.searchsorted(fres, fcal)
    fcal_indices += np.asarray(range(len(fcal_indices)), dtype = int)
    for fcal_index, fres_index in enumerate(fcal_indices):
        fres = np.insert(fres, fres_index, fcal[fcal_index])
        ares = np.insert(ares, fres_index, fcal_power)
        qres = np.insert(qres, fres_index, np.inf)
        new_index = -fcal_index - 1
        new_res_indices = np.insert(new_res_indices, fres_index,
                                          new_index)
    return fres, ares, qres, fcal_indices, new_res_indices
