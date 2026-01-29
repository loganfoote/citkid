import numpy as np
import warnings
import zarr

from tests import crs 

async def target_sweep(
    crs,
    fres,
    ares,
    qres,
    res_idxs,
    grp,
    gain_span_factor = 10,
    npoints_fine = 500,
    npoints_gain = 50,
    npoints_rough = None,
    nsamps = 100,
    fres_update_method = 'spacing',
    fres_all = None,
    qres_all = None,
    cable_delay = 0.0,
    verbose = True,
):
    """
    Take multitone IQ sweeps and noise.

    Parameters:
    crs (CRS): initialized CRS instrument class, with
        'sweep', 'write_tones', and 'capture_noise' methods.
    fres (array-like): array of resonant frequencies in Hz.
    ares (array-like): array of amplitudes in RFSoC units.
    qres (array-like): array of span factors for cutting out of adjacent
        datasets. Resonances should span fres / qres.
    res_idxs (array-like): Array of resonator indices.
    grp (str): Zarr group to which data is saved.
    gain_span_factor (float): gain span is (gain_span_factor * fine_span).
    npoints_fine (int or None): number of points per resonator in the 
        fine sweep.
    npoints_gain (int or None): number of points per resonator in the 
        gain sweep.
    npoints_rough (int or None): number of points per resonator in the 
        rough sweep.
    nsamps (int): number of samples to average over per sweep point. 
    fres_update_method (str): method for updating the tone frequencies, if
        take_rough_sweep is True. See .fres.update_fres for methods.
    fres_all (array-like): list of all frequencies for analysis, if fres is
        incomplete.
    qres_all (array-like): array of span factors corresponding to fres_all.
    cable_delay (float): Cable delay estimate to improve frequency update.
    verbose (bool): If True, displays progress bars while taking data.

    Returns:
    None
    """
    # Input validation 
    _validate_target_sweep_inputs(
        crs, fres, ares, qres, res_idxs, grp, gain_span_factor, npoints_fine, 
        npoints_gain, npoints_rough, nsamps, fres_update_method, fres_all, 
        qres_all, cable_delay, verbose
    )

    f, z = await crs.sweep_qres(
            fres,
            ares,
            qres,
            npoints = npoints_rough,
            nsamps = nsamps,
            verbose = verbose,
            pbar_description = 'Rough sweep',
        )
    
    raise NotImplementedError("target_sweep is a work in progress")
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
    msg = (
        "Adjusting cal-tone qres: Logan doesn't remember writing this, "
        "where did it come from?"
    )
    warnings.warn(msg, UserWarning)
    qres0 = qres.copy()
    qres0[fcal_indices] = np.median(qres)
    # rough sweep
    if take_rough_sweep:
        filename = f's21_rough{file_suffix}.npy'
        f, z = await crs.sweep_qres(
            fres,
            ares,
            qres0,
            npoints = npoints_rough,
            nsamps = nsamps,
            verbose = verbose,
            pbar_description = 'Rough sweep',
        )
        np.save(out_directory + filename, [f, np.real(z), np.imag(z)])
        if npoints_noisefreq_update:
            ix0 = npoints_rough // 2 - npoints_noisefreq_update // 2
            ix1 = npoints_rough // 2 + npoints_noisefreq_update // 2
            ix1 += npoints_noisefreq_update % 2
            f0 = [fi[ix0: ix1] for fi in f]
            z0 = [zi[ix0: ix1] for zi in z]
        else:
            f0, z0 = f, z
        fres = update_fres(
            f0,
            z0,
            fres,
            spans,
            fcal_indices,
            method = fres_update_method,
            cable_delay = cable_delay,
        )
        await crs.write_tones(fres, ares)
        np.save(out_directory + f'fres{file_suffix}.npy', fres)

    # Gain sweep
    filename = f's21_gain{file_suffix}.npy'
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres0 / gain_span_factor,
        npoints = npoints_gain,
        nsamps = nsamps,
        verbose = verbose,
        pbar_description = 'Gain sweep',
    )
    np.save(out_directory + filename, [f, np.real(z), np.imag(z)])

    # Fine sweep
    filename = f's21_fine{file_suffix}.npy'
    f, z = await crs.sweep_qres(
        fres,
        ares,
        qres0,
        npoints = npoints_fine,
        nsamps = nsamps,
        verbose = verbose,
        pbar_description = 'Fine sweep',
    )
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
        fres = update_fres(
            f0,
            z0,
            fres,
            spans,
            fcal_indices,
            method = 'spacing',
            cable_delay = cable_delay,
        )
    np.save(out_directory + f'fres{file_suffix}.npy', fres)

    # Noise
    if take_noise:
        while True and wait_for_noise:
            msg = "Type 'y' to proceed with noise measurement"
            proceed = input(msg).strip().lower()
            if proceed == 'y':
                break
        filename = f'noise{file_suffix}.npy'
        z = await crs.capture_noise(
            fres,
            ares,
            noise_time,
            dec_stage = dec_stage,
            delete_parser_data = True,
            return_dbc = True,
            batch_process = True,
            outpath = out_directory + filename,
            batch_size = batch_size,
            parser_loc = parser_loc,
            verbose = verbose,
        )


def _validate_target_sweep_inputs(
    crs,
    fres,
    ares,
    qres, 
    res_idxs,
    grp,
    gain_span_factor,
    npoints_fine,
    npoints_gain,
    npoints_rough,
    nsamps,
    fres_update_method,
    fres_all,
    qres_all,
    cable_delay,
    verbose,
):
    if not isinstance(crs, crs.CRS):
        raise TypeError("crs must be an instance of CRS class.") 
    fres = np.asarray(fres, dtype = np.float64)
    ares= np.asarray(ares, dtype = np.float64)
    qres = np.asarray(qres, dtype = np.float64)
    res_idxs = np.asarray(res_idxs, dtype = np.int32)
    if fres.ndim != 1:
        raise ValueError("fres must be a 1D array.")
    if not (fres.shape == ares.shape == qres.shape == res_idxs.shape):
        raise ValueError(
            "fres, ares, qres, and res_idxs must have the same shape."
        )
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.")
    # Need to check that none of the values we want to write already exist 
    if not isinstance(gain_span_factor, (int, float, np.integer, np.floating)) \
        or gain_span_factor <= 1:
        raise ValueError("gain_span_factor must be a number > 1.") 
    if npoints_fine is not None and \
        (not isinstance(npoints_fine, (int, np.integer)) or npoints_fine <= 0):
        raise ValueError("npoints_fine must be a positive integer.") 
    if npoints_gain is not None and \
        (not isinstance(npoints_gain, (int, np.integer)) or npoints_gain <= 0):
        raise ValueError("npoints_gain must be a positive integer.") 
    if npoints_rough is not None and \
        (not isinstance(npoints_rough, (int, np.integer)) or npoints_rough <=0):
        raise ValueError("npoints_rough must be a positive integer.") 
    if not isinstance(nsamps, (int, np.integer)) or nsamps <= 0:
        raise ValueError("nsamps must be a positive integer.") 
    if fres_update_method not in ['distance', 'spacing', 'minS21']:
        msg = "fres_update_method must be 'distance', 'spacing', or 'minS21'."
        raise ValueError(msg)
    if fres_all is not None:
        fres_all = np.asarray(fres_all, dtype = np.float64)
        if fres_all.ndim != 1:
            raise ValueError("fres_all must be a 1D array.")
    if qres_all is not None:
        qres_all = np.asarray(qres_all, dtype = np.float64)
        if qres_all.ndim != 1:
            raise ValueError("qres_all must be a 1D array.") 
    if (fres_all is not None) or (qres_all is not None):
        if not (fres_all.shape == qres_all.shape):
            raise ValueError("fres_all and qres_all must have the same shape.")
    if not isinstance(cable_delay, (float, np.np.floating)) or \
        cable_delay < 0:
        raise ValueError("cable_delay must be a positive number.") 
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")