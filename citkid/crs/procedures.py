import os
import numpy as np
from tqdm.auto import tqdm
from ..multitone.ares import update_ares_pscale, update_ares_addonly
from ..multitone.fres import update_fres
from ..multitone.analysis import fit_iq
from ..multitone.plot import plot_ares_opt
from ..util import save_fig
from ..res.gain import fit_and_remove_gain_phase, remove_gain
from ..multitone.data_io import import_iq_noise
from ..noise.analysis import compute_psd_simple
import warnings

async def take_iq_noise(inst, fres, ares, qres, fcal_indices, res_indices,
                        out_directory, file_suffix, noise_time = 200,
                        take_noise = False, gain_span_factor = 10,
                        npoints_noisefreq_update = None, npoints_fine = 600,
                        npoints_gain = 100, npoints_rough = 300, nsamps = 10,
                        take_rough_sweep = False, update_fres_from_fine = False,
                        fres_update_method = 'distance', fir_stage = 6,
                        fres_all = None, qres_all = None, cable_delay = 0,
                        batch_size = 1000,
                        parser_loc = '/home/daq1/github/rfmux/firmware/r1.5.6/parser',
                        wait_for_noise = False, verbose = True):
    """
    Takes multitone IQ sweeps and noise.

    Parameters:
    inst (multitone instrument): initialized multitone instrument class, with
        'sweep', 'write_tones', and 'capture_noise' methods.
    fres (array-like): array of resonance frequencies in Hz.
    ares (array-like): array of amplitudes in RFSoC units.
    qres (array-like): array of span factors for cutting out of adjacent
        datasets. Resonances should span fres / qres.
    fcal_indices (array-like): indices (into fres, ares, qres) of calibration
        tones.
    res_indices (array-like): Array of resonators indices (int).
    out_directory (str): directory to save the data.
    file_suffix (str): suffix for file names.
    noise_time (float or None): noise timestream length in seconds.
    take_noise (bool): If True, takes noise data.
    gain_span_factor (float): gain span is (gain_span_factor * fine_span).
    npoints_noisefreq_update (int): Number of points around the center of the
        fine sweep that are used to update frequencies before taking noise, or
        None to bypass updating.
    npoints_fine (int): number of points per resonator in the fine sweep.
    npoints_gain (int): number of points per resonator in the gain sweep.
    npoints_rough (int): number of points per resonator in the rough sweep.
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies.
    nsamps (int): number of samples to average over per point in the sweeps.
    take_rough_sweep (bool): If True, takes a rough sweep and updates
        frequencies before taking noise.
    update_fres_from_fine (bool): if True, updates frequencies from the fine
        sweep before taking noise.
    fres_update_method (str): method for updating the tone frequencies, if
        take_rough_sweep is True. See .fres.update_fres for methods.
    fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz
            5 -> 1,192.09 Hz
            4 -> 2,384.19 Hz, might crash
            3 -> will definetely crash.
    fres_all (array-like): list of all frequencies for analysis, if fres is
        incomplete.
    qres_all (array-like): array of span factors corresponding to fres_all.
    cable_delay (float): Cable delay estimate to improve frequency update.
    batch_size (int): noise processing batch size in MB.
    parser_loc (str): path to the parser file.
    wait_for_noise (bool): If True, after taking IQ loops, waits for user input
        before proceeding with noise measurements.
    verbose (bool): If True, displays progress bars while taking data.
    """
    # Set up temporary data path
    data_path = 'tmp/parser_data_00/'
    if os.path.exists(data_path) and take_noise:
        raise FileExistsError(f'{data_path} already exists')
    os.makedirs(out_directory, exist_ok = True)
    # Format input arrays
    fres = np.asarray(fres, dtype = float)
    ares= np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    fcal_indices = np.asarray(fcal_indices, dtype = int)
    res_indices = np.asarray(res_indices, dtype = int)
    spans = fres / qres
    # Save input arrays
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    if take_rough_sweep or update_fres_from_fine:
        np.save(out_directory + f'fres_initial{file_suffix}.npy', fres)
    np.save(out_directory + f'ares{file_suffix}.npy', ares)
    np.save(out_directory + f'qres{file_suffix}.npy', qres)
    np.save(out_directory + f'fcal_indices{file_suffix}.npy',
            fcal_indices)
    np.save(out_directory + f'res_indices{file_suffix}.npy', res_indices)
    if fres_all is not None:
        np.save(out_directory + f'fres_all{file_suffix}.npy', fres)
        np.save(out_directory + f'qres_all{file_suffix}.npy', qres)
    # Make qres for sweeps that works with cal tones
    msg = "Adjusting cal-tone qres: Logan doesn't remember writing this, where did it come from?"
    warnings.warn(msg, UserWarning)
    qres0 = qres.copy()
    qres0[fcal_indices] = np.median(qres)
    # rough sweep
    if take_rough_sweep:
        filename = f's21_rough{file_suffix}.npy'
        f, z = await inst.sweep_qres(fres, ares, qres0, npoints = npoints_rough,
                                     nsamps = nsamps, verbose = verbose,
                                     pbar_description = 'Rough sweep')
        np.save(out_directory + filename, [f, np.real(z), np.imag(z)])
        if npoints_noisefreq_update:
            ix0 = npoints_rough // 2 - npoints_noisefreq_update // 2
            ix1 = npoints_rough // 2 + npoints_noisefreq_update // 2
            ix1 += npoints_noisefreq_update % 2
            f0 = [fi[ix0: ix1] for fi in f]
            z0 = [zi[ix0: ix1] for zi in z]
        else:
            f0, z0 = f, z
        fres = update_fres(f0, z0, fres, spans, fcal_indices,
                           method = fres_update_method,
                           cable_delay = cable_delay)
        await inst.write_tones(fres, ares)
        np.save(out_directory + f'fres{file_suffix}.npy', fres)

    # Gain sweep
    filename = f's21_gain{file_suffix}.npy'
    f, z = await inst.sweep_qres(fres, ares, qres0 / gain_span_factor,
                                 npoints = npoints_gain, nsamps = nsamps,
                                 verbose = verbose,
                                 pbar_description = 'Gain sweep')
    np.save(out_directory + filename, [f, np.real(z), np.imag(z)])

    # Fine sweep
    filename = f's21_fine{file_suffix}.npy'
    f, z = await inst.sweep_qres(fres, ares, qres0, npoints = npoints_fine,
                                 nsamps = nsamps, verbose = verbose,
                                 pbar_description = 'Fine sweep')
    np.save(out_directory + filename, [f, np.real(z), np.imag(z)])
    if update_fres_from_fine:
        if (npoints_noisefreq_update is not None):
            ix0 = npoints_fine // 2 - npoints_noisefreq_update // 2
            ix1 = npoints_fine // 2 + npoints_noisefreq_update // 2
            ix1 += npoints_noisefreq_update % 2
            f0 = [fi[ix0: ix1] for fi in f]
            z0 = [zi[ix0: ix1] for zi in z]
        else:
            f0, z0 = f, z
        fres = update_fres(f0, z0, fres, spans, fcal_indices,
                           method = 'spacing', cable_delay = cable_delay)
    np.save(out_directory + f'fres{file_suffix}.npy', fres)

    # Noise
    if take_noise:
        while True and wait_for_noise:
            msg = "Type 'y' to proceed with noise measurement"
            proceed = input(msg).strip().lower()
            if proceed == 'y':
                break
        filename = f'noise{file_suffix}.npy'
        z = await inst.capture_noise(fres, ares, noise_time,
                                     fir_stage = fir_stage,
                                     delete_parser_data = True,
                                     return_dbc = True, batch_process = True,
                                     outpath = out_directory + filename,
                                     batch_size = batch_size,
                                     parser_loc = parser_loc, verbose = verbose)

async def take_rough_sweep(inst, fres, ares, qres, fcal_indices, res_indices,
                           out_directory, file_suffix, npoints = 600,
                           nsamps = 10, plot_directory = '', fres_all = None,
                           qres_all = None, plotq = False,
                           fres_update_method = 'distance', cable_delay = 0):
    """
    Takes a single rough IQ sweep for updating frequencies

    Parameters:
    inst (multitone instrument): initialized multitone instrument class, with
        'sweep', 'write_tones', and 'capture_noise' methods
    fres (np.array): array of resonance frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of span factors for cutting out of adjacent datasets.
        Resonances should span fres / qres
    fcal_indices (np.array): indices (into fres, ares, qres) of calibration tones
    out_directory (str): directory to save the data
    file_suffix (str): suffix for file names
    noise_time (float or None): noise timestream length in seconds
    if_bw (float): IF bandwidth. 1 / if_bw is the averaging time per data point
        in the IQ loops
    fine_bw (float): fine sweep bandwidth in Hz. Gain bandwidth is 10 X fine
        bandwidth
    rough_bw (float): rough sweep bandwidth in Hz
    npoints_fine (int): number of points per resonator in the fine sweep
    npoints_gain (int): number of points per resonator in the gain sweep
    npoints_rough (int): number of points per resonator in the rough sweep
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies
    fres_update_method (str): method for updating the tone frequencies, if
        take_rough_sweep is True. See .fres.update_fres for methods
    nnoise_timestreams (int): number of noise timestreams to take sequentially.
        Set to 0 to bypass noise acquisition
    fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz
            5 -> 1,192.09 Hz
            4 -> 2,384.19 Hz, might crash
            3 -> will definetely crash
    """
    for d in (plot_directory, out_directory):
        if d != '':
            os.makedirs(d, exist_ok = True)
    fres = np.asarray(fres, dtype = float)
    ares= np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    fcal_indices = np.asarray(fcal_indices, dtype = int)
    spans = fres / qres
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    np.save(out_directory + f'fres_initial{file_suffix}.npy', fres)
    np.save(out_directory + f'ares{file_suffix}.npy', ares)
    np.save(out_directory + f'qres{file_suffix}.npy', qres)
    np.save(out_directory + f'fcal_indices{file_suffix}.npy',
            fcal_indices)
    np.save(out_directory + f'res_indices{file_suffix}.npy', res_indices)
    if fres_all is not None:
        np.save(out_directory + f'fres_all{file_suffix}.npy', ares)
        np.save(out_directory + f'qres_all{file_suffix}.npy', ares)
    # Make qres for sweeps that works with cal tones
    qres0 = qres.copy()
    qres0[fcal_indices] = np.median(qres)
    # # write initial target comb
    # await inst.write_tones(fres, ares)
    # rough sweep
    filename = f's21_rough{file_suffix}.npy'
    f, z = await inst.sweep_qres(fres, ares, qres0, npoints = npoints,
                                 nsamps = nsamps, verbose = True,
                                 pbar_description = 'Rough sweep')
    np.save(out_directory + filename, [f, np.real(z), np.imag(z)])
    fres = update_fres(f, z, fres, spans, fcal_indices,
                        method = fres_update_method, plotq = plotq, res_indices = res_indices,
                        plot_directory = plot_directory, cable_delay = cable_delay)
    np.save(out_directory + f'fres_interim{file_suffix}.npy', fres)


async def optimize_ares(inst, out_directory, fres, ares, qres, fcal_indices, res_indices,
                        dbm_max = -50, a_target = 0.5, n_iterations = 10, n_addonly = 3,
                        fres_update_method = 'distance', skip_first = False, start_index = 0,
                        npoints_gain = 50, npoints_fine = 400, plot_directory = None, bypass_indices = [],
                        verbose = False, nsamps = 10, dbm_change_pscale = 2, fres_all = None, qres_all = None,
                        dbm_change_addonly = 1, addonly_threshold = 0.2, res_threshold = 2e-3):
    """
    Optimize tone powers using by iteratively fitting IQ loops and using a_nl
    of each fit to scale each tone power

    Parameters:
    inst (citkid.primecam.instrument.RFSOC): RFSOC instance
    fres (np.array): array of center frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of resonators Qs for cutting data. Resonances should
        span fres / qres
    fcal_indices (np.array): calibration tone indices
    max_dbm (float): maximum power per tone in dBm
    a_target (float): target value for a_nl. Must be in range (0, 0.77]
    n_iterations (int): total number of iterations
    n_addonly (int): number of iterations at the end to optimize using
        update_ares_addonly. Iterations before these use update_ares_pscale
    fine_bw (float): fine sweep bandwidth in MHz. See take_iq_noise
    fres_update_method (str): method for updating frequencies. See update_fres
    skip_first (bool): If True, skips taking data on the first iteration, and
        instead starts from fitting (assumes the data already exists)
    start_index (int): file index to start from.
    npoints_gain (int): number of points in the gain sweep
    npoints_fine (int): number of points in the fine sweep
    plot_directory (str or None): path to save histograms as the optimization is
        running. If None, doesn't save plots
    bypass_indices (array-like): resonator indices to bypass optimization
    verbose (bool): if True, displays a progress bar of the iteration number
    N_accums (int): number of accumulations for the target sweeps
    threshold (float): optimization will occur within (1-threshold) and
        (1+threshold) of the target during the addonly phase of power optimization
    """
    if plot_directory is not None:
        os.makedirs(plot_directory, exist_ok = True)
    fres, ares, qres = np.asarray(fres), np.asarray(ares), np.asarray(qres)
    bypass_indices = np.asarray(bypass_indices)
    pbar0 = list(range(start_index, n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices and res_indices[i] not in bypass_indices]
    a_nls = []
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('Sweeping')
        file_suffix = f'{idx0:02d}'
        if not skip_first or idx0 != start_index:
            await take_iq_noise(inst, fres, ares, qres, fcal_indices, res_indices, out_directory, file_suffix,
                                take_noise = False, take_rough_sweep = False, npoints_gain = npoints_gain,
                                fres_all = fres_all, qres_all = qres_all,
                                npoints_fine = npoints_fine, nsamps = nsamps)

        # Fit IQ loops
        if verbose:
            pbar0.set_description('fitting')
        data = fit_iq(out_directory, None, file_suffix, 0, 0, 0, 0, 0, rejected_points = [],
                      plotq = False, verbose = verbose, catch_exceptions = True) # Turn off catch_exceptions
        a_nl = np.array(data.sort_values('dataIndex').iq_a, dtype = float)
        res = np.array(data.sort_values('dataIndex').iq_a, dtype = float) - a_target
        a_nls.append(a_nl)
        np.save(out_directory + f'a_nl_{file_suffix}.npy', a_nl)
        if plot_directory is not None:
            fig_hist, fig_opt = plot_ares_opt(a_nls, fcal_indices)
            save_fig(fig_hist, 'ares_hist', plot_directory)
            save_fig(fig_opt, 'ares_opt', plot_directory)
        # Update ares
        fit_idx1 = [i for i in fit_idx if res[i] <= res_threshold and np.isfinite(a_nl[i])] # Don't update if bad fit
        if n_iterations - idx0 > n_addonly:
            ares[fit_idx1] = update_ares_pscale(fres[fit_idx1], ares[fit_idx1],
                                           a_nl[fit_idx1], a_target = a_target,
                                           dbm_max = dbm_max, dbm_change_high = dbm_change_pscale,
                                           dbm_change_low = dbm_change_pscale)
        else:
            ares[fit_idx1] = update_ares_addonly(fres[fit_idx1], ares[fit_idx1],
                                                a_nl[fit_idx1],
                                                a_target = a_target,
                                                dbm_max = dbm_max,
                                                dbm_change_high = dbm_change_addonly,
                                                dbm_change_low = dbm_change_addonly,
                                                threshold = addonly_threshold)

        # update fres
        f, i, q = np.load(out_directory + f's21_fine_{file_suffix}.npy')
        fres = update_fres(f, i + 1j * q, fres, qres,
                           fcal_indices = fcal_indices, method = fres_update_method)
        # for the last iteration, save the updated ares list
        if idx0 == len(fres) - 1:
            np.save(out_directory + f'ares_{idx0 + 1:02d}', ares)


async def optimize_ares_noise(inst, out_directory, fres, ares, qres, fcal_indices, res_indices,
                              dbm_max = -50, sfact_target = 2, n_iterations = 10, sfact_freq = 30,
                              fres_update_method = 'distance', skip_first = False, start_index = 0,
                              npoints_gain = 50, npoints_fine = 400, plot_directory = None, bypass_indices = [],
                              verbose = False, nsamps = 10, dbm_change = 2, fres_all = None, qres_all = None):
    """
    Optimize tone powers by iteratively taking noise data and comparing parallel to perpendicular noise

    Parameters:
    inst (citkid.primecam.instrument.RFSOC): RFSOC instance
    out_directory (str): directory to save data
    fres (array-like): array of center frequencies in Hz
    ares (array-like): array of amplitudes in RFSoC units
    qres (array-like): array of resonators Qs for cutting data. Resonances should
        span fres / qres
    fcal_indices (array-like): calibration tone indices
    res_indices (array-like): resonator indices
    max_dbm (float): maximum power per tone in dBm
    sfact_target (float): target value to exceed for Spar / Sper
    n_iterations (int): total number of iterations
    sfact_freq (float): frequency at which Spar and Sper are averaged (in a 10% bin) to
        determine sfactor = Spar / Sper
    skip_first (bool): If True, skips taking data on the first iteration, and
        instead starts from fitting (assumes the data already exists)
    start_index (int): file index to start from.
    npoints_gain (int): number of points in the gain sweep
    npoints_fine (int): number of points in the fine sweep
    plot_directory (str or None): plots are not implemented yet
    bypass_indices (array-like): resonator indices to bypass optimization
    verbose (bool): if True, displays a progress bar of the iteration number
    nsamps (int): number of samples per frequency in the sweeps for averaging
    dbm_change (float): amount added to each power that is under sfact_target
    fres_all (array-like): full list of resonance frequencies for gain sweep fitting
    qres_all (array-like): full list of resonance q-factors for gain sweep fitting
    """
    if plot_directory is not None:
        os.makedirs(plot_directory, exist_ok = True)
    fres, ares, qres = np.array(fres), np.array(ares), np.array(qres)
    bypass_indices = np.asarray(bypass_indices)
    pbar0 = list(range(start_index, n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices and res_indices[i] not in bypass_indices]
    a_nls = []
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('Sweeping')
        file_suffix = f'{idx0:02d}'
        if not skip_first or idx0 != start_index:
            await take_iq_noise(inst, fres, ares, qres, fcal_indices, res_indices, out_directory, file_suffix,
                                take_noise = True, noise_time = int(sfact_freq / 5), take_rough_sweep = False,
                                npoints_gain = npoints_gain, fres_all = fres_all, qres_all = qres_all,
                                npoints_fine = npoints_fine, nsamps = nsamps)

        # Calibrate noise
        fres_initial, fres, ares, qres, fcal_indices, fres_all, qres_all, frough, zrough,\
           fgains, zgains, ffines, zfines, znoises, noise_dt, res_indices =\
            import_iq_noise(out_directory, file_suffix, import_noiseq = True)
        sfactors = np.empty(len(fres))
        for di in range(len(fres)):
            if di in fcal_indices:
                sfactors[di] = np.nan
            else:
                ff, zf, fg, zg, zn = ffines[di], zfines[di], fgains[di], zgains[di], znoises[di]
                fn = fres[di]
                p_amp, p_phase, zf_rmv, _ =\
                fit_and_remove_gain_phase(fg, zg, ff, zf, frs = fres_all, Qrs = qres_all, plotq=False)
                zn_rmv = remove_gain(fn, zn, p_amp, p_phase)
                f_psd, spar, sper = compute_psd_simple(ff, zf_rmv, fn, zn_rmv, noise_dt, deglitch_nstd = 5)
                ix = np.abs(f_psd - sfact_freq) < (sfact_freq / 10)
                sfactors[di] = np.mean(spar[ix]) / np.mean(sper[ix])
        np.save(out_directory + f'sfact_{idx0:02d}.npy', sfactors)
        a_increase_idx = [di for di in fit_idx if sfactors[di] < sfact_target]
        ares[a_increase_idx] += dbm_change
        ares[ares > dbm_max] = dbm_max
        # update fres
        f, i, q = np.load(out_directory + f's21_fine_{file_suffix}.npy')
        fres = update_fres(f, i + 1j * q, fres, qres,
                           fcal_indices = fcal_indices, method = fres_update_method)
        # for the last iteration, save the updated ares list
        if idx0 == len(fres) - 1:
            np.save(out_directory + f'ares_{idx0 + 1:02d}.npy', ares)
            np.save(out_directory + f'fres_{idx0 + 1:02d}.npy', fres)

################################################################################
############################# Sequential functions #############################
################################################################################
async def take_iq_noise_sequential(inst, module_index, ncos, fres, ares, qres, fcal_indices, res_indices,
                                   out_directory, file_suffix,
                  noise_time = 200, take_noise = False, gain_span_factor = 10, npoints_noisefreq_update = None,
                  npoints_fine = 600, npoints_gain = 100, npoints_rough = 300, nsamps = 10,
                  take_rough_sweep = False, fres_update_method = 'distance', fir_stage = 6,
                  fres_all = None, qres_all = None, verbose = True, cable_delay = 0,
                  take_fast_noise = False, fast_noise_time = 10, n_fast_noise = 1, nco_indices_to_skip = []):
    """
    Takes multitone IQ sweeps and noise.

    Parameters:
    inst (multitone instrument): initialized multitone instrument class, with
        'sweep', 'write_tones', and 'capture_noise' methods
    ncos (array-like): nco frequencies in Hz
    fres (np.array): array of resonance frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of span factors for cutting out of adjacent datasets.
        Resonances should span fres / qres
    fcal_indices (np.array): indices (into fres, ares, qres) of calibration tones
    out_directory (str): directory to save the data
    file_suffix (str): suffix for file names
    noise_time (float or None): noise timestream length in seconds
    if_bw (float): IF bandwidth. 1 / if_bw is the averaging time per data point
        in the IQ loops
    fine_bw (float): fine sweep bandwidth in Hz. Gain bandwidth is 10 X fine
        bandwidth
    npoints_noisefreq_update (int): Number of points around the center of the fine sweep that
        are used to update frequencies before taking noise, or None to bypass updating
    rough_bw (float): rough sweep bandwidth in Hz
    npoints_fine (int): number of points per resonator in the fine sweep
    npoints_gain (int): number of points per resonator in the gain sweep
    npoints_rough (int): number of points per resonator in the rough sweep
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies
    fres_update_method (str): method for updating the tone frequencies, if
        take_rough_sweep is True. See .fres.update_fres for methods
    nnoise_timestreams (int): number of noise timestreams to take sequentially.
        Set to 0 to bypass noise acquisition
    fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz
            5 -> 1,192.09 Hz
            4 -> 2,384.19 Hz, might crash
            3 -> will definetely crash
    fres_all (array-like): list of all frequencies for analysis, if fres is
        incomplete
    qres_all (array-like): array of span factors corresponding to fres_all
    nco_indices_to_skip (list): list of NCO indices to skip data acquisition. Use this when
        the code failed partway through. take_iq_noise must have finished running for the
        indices that are skipped. The function will import the data instead of taking and
        overwriting it
    """
    if take_fast_noise:
        raise Exception('Fast noise not yet implemented for sequential')
    ix = np.argsort(fres)
    fres, ares, qres = fres[ix], ares[ix], qres[ix]
    res_indices = res_indices[ix]
    fcal_indices = ix[fcal_indices]
    ncos = np.asarray(ncos)
    ixs = {index: [] for index in range(len(ncos))}
    for di, fr in enumerate(fres):
        nco_index = np.argmin(abs(ncos - fr))
        ixs[nco_index].append(di)

    fres_initial_out, fres_out = np.empty(fres.size, dtype = float), np.empty(fres.size, dtype = float)
    ares_out, qres_out = np.empty(fres.size, dtype = float), np.empty(fres.size, dtype = float)
    fcal_indices_out = fcal_indices.copy()
    res_indices_out = np.empty(fres.size, dtype = int)
    if take_rough_sweep:
        frough_out, zrough_out = np.empty((len(fres), npoints_rough), dtype = float), np.empty((len(fres), npoints_rough), dtype = complex)
    else:
        frough_out, zrough_out = None, None
    ffine_out, zfine_out = np.empty((len(fres), npoints_fine), dtype = float), np.empty((len(fres), npoints_fine), dtype = complex)
    fgain_out, zgain_out = np.empty((len(fres), npoints_gain), dtype = float), np.empty((len(fres), npoints_gain), dtype = complex)
    nco_indices_out = np.empty(len(fres), dtype = int)
    fcal_indices_all = []
    if take_noise:
        fres_noise_out = np.empty(len(fres), dtype = float)
        znoise_out = None
    pbar = ncos
    if verbose:
        pbar = tqdm(pbar, leave = False)
    for nco_index, nco in enumerate(pbar):
        if verbose:
            pbar.set_description(f'NCO: {round(nco * 1e-6, 4)} MHz')
        file_suffix0 = file_suffix + f'_NCO{nco_index}'
        ix = ixs[nco_index]
        for di in ix:
            nco_indices_out[di] = nco_index
        fres0 = fres[ix]
        qres0 = qres[ix]
        ares0 = ares[ix]
        fcal_indices0 = [fc - min(ix) for fc in fcal_indices if fc in ix]
        res_indices0 = res_indices[ix]
        await inst.set_nco({module_index: nco})
        if nco_index not in nco_indices_to_skip:
            await take_iq_noise(inst, fres0, ares0, qres0, fcal_indices0, res_indices0, out_directory, file_suffix0,
                    noise_time = noise_time, take_noise = take_noise, gain_span_factor = gain_span_factor,
                    npoints_noisefreq_update = npoints_noisefreq_update, cable_delay = cable_delay,
                    npoints_fine = npoints_fine, npoints_gain = npoints_gain, npoints_rough = npoints_rough,
                    nsamps = nsamps, take_rough_sweep = take_rough_sweep, fres_update_method = fres_update_method,
                    fir_stage = fir_stage, fres_all = fres_all, qres_all = qres_all, verbose = verbose,
                    take_fast_noise = take_fast_noise, fast_noise_time = fast_noise_time, n_fast_noise = n_fast_noise)

        fres_initial0, fres_out[ix], ares_out[ix], qres_out[ix], fcal_indices0, fres_all0, qres_all0, \
            frough0, zrough0, fgain_out[ix], zgain_out[ix], ffine_out[ix], \
            zfine_out[ix], znoise0, noise_dt, res_indices_out[ix], fres_noise0 =\
        import_iq_noise(out_directory, file_suffix0, import_noiseq = take_noise)
        fcal_indices_all.append([fc + min(ix) for fc in fcal_indices0])
        if take_rough_sweep:
            fres_initial_out[ix] = fres_initial0
            frough_out[ix] = frough0
            zrough_out[ix] = zrough0
        if take_noise:
            fres_noise_out[ix] = fres_noise0
            if znoise_out is None:
                znoise_out = np.empty((len(fres), znoise0.shape[1]), dtype = complex)
            elif znoise_out.shape[1] < znoise0.shape[1]:
                znoise0 = znoise0[:, :znoise_out.shape[1]]
            elif znoise_out.shape[1] > znoise0.shape[1]:
                znoise_out = znoise_out[:, :znoise0.shape[1]]
            znoise_out[ix] = znoise0
    if take_rough_sweep:
        np.save(out_directory + f's21_rough_{file_suffix}.npy',
                [frough_out, np.real(zrough_out), np.imag(zrough_out)])
        np.save(out_directory + f'fres_initial_{file_suffix}.npy', fres_initial_out)
    np.save(out_directory + f'NCO_indices_{file_suffix}.npy', nco_indices_out)
    np.save(out_directory + f's21_fine_{file_suffix}.npy',
                [ffine_out, np.real(zfine_out), np.imag(zfine_out)])
    np.save(out_directory + f's21_gain_{file_suffix}.npy',
                [fgain_out, np.real(zgain_out), np.imag(zgain_out)])

    if take_rough_sweep:
        np.save(out_directory + f's21_rough_{file_suffix}.npy',
                [frough_out, np.real(zrough_out), np.imag(zrough_out)])
        np.save(out_directory + f'fres_initial_{file_suffix}.npy', fres_initial_out)

    np.save(out_directory + f's21_fine_{file_suffix}.npy',
                [ffine_out, np.real(zfine_out), np.imag(zfine_out)])
    np.save(out_directory + f's21_gain_{file_suffix}.npy',
                [fgain_out, np.real(zgain_out), np.imag(zgain_out)])

    np.save(out_directory + f'fres_{file_suffix}.npy', fres_out)
    np.save(out_directory + f'qres_{file_suffix}.npy', qres_out)
    np.save(out_directory + f'ares_{file_suffix}.npy', ares_out)
    np.save(out_directory + f'fres_all_{file_suffix}.npy', fres_all)
    np.save(out_directory + f'qres_all_{file_suffix}.npy', qres_all)
    np.save(out_directory + f'fcal_indices_{file_suffix}.npy', np.array(fcal_indices_out, dtype = int))
    np.save(out_directory + f'res_indices_{file_suffix}.npy' , np.array(res_indices_out, dtype = int))

    if take_noise:
        np.save(out_directory + f'noise_{file_suffix}_00.npy', [np.real(znoise_out), np.imag(znoise_out)])
        np.save(out_directory + f'noise_{file_suffix}_tsample_00.npy', noise_dt)
        np.save(out_directory + f'fres_noise_{file_suffix}.npy', fres_noise_out)

    for nco_index in range(len(ncos)):
        prefixes = ['fres', 'qres', 'ares', 'fres_all', 'qres_all', 'fcal_indices', 'res_indices',
                    's21_fine', 's21_gain']
        if take_rough_sweep:
            prefixes += ['s21_rough', 'fres_initial']
        paths = [out_directory + prefix + f'_{file_suffix}_NCO{nco_index}.npy' for prefix in prefixes]
        if take_noise:
            paths.append(out_directory + f'fres_noise_{file_suffix}_NCO{nco_index}.npy')
            paths.append(out_directory + f'noise_{file_suffix}_NCO{nco_index}_00.npy')
            paths.append(out_directory + f'noise_{file_suffix}_NCO{nco_index}_tsample_00.npy')
        for path in paths:
            os.remove(path)
