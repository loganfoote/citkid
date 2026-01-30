import numpy as np
import zarr
from citkid.multitone.fres import update_fres
from . import util

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
    cable_delay = 0.0,
    verbose = True,
):
    """
    Execute a target sweep procedure on the CRS instrument, consisting of
    a rough sweep, gain sweep, and fine sweep, saving data to the provided Zarr
    group. The tone frequencies are updated after the rough sweep according to 
    the specified method. Any of the sweeps can be disabled by setting the 
    corresponding npoints parameter to None.

    Parameters:
    crs (CRS): initialized CRS instrument class.
    fres (array-like float64): array of resonant frequencies in Hz.
    ares (array-like float64): array of amplitudes in RFSoC units.
    qres (array-like float64): array of span factors for cutting out of adjacent
        datasets. Resonances should span fres / qres.
    res_idxs (array-like int32): Array of resonator indices.
    grp (zarr.Group): Zarr group to which data is saved.
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
    cable_delay (float): Cable delay estimate to improve frequency update.
    verbose (bool): If True, displays progress bars while taking data.

    Returns:
    None
    """
    # Input validation 
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        crs, fres, ares, qres, res_idxs, grp, gain_span_factor, npoints_fine, 
        npoints_gain, npoints_rough, nsamps, fres_update_method, 
        cable_delay, verbose
    )

    # Save input data 
    grp.create_array(name = f'ares', data = ares)
    grp.create_array(name = f'qres', data = qres)
    grp.create_array(name = f'res_idxs', data = res_idxs)
    grp.create_array(name = f'nsamps', data = nsamps, dtype = np.int32)
    
    if npoints_rough is not None:
        # Rough sweep 
        fres_rough = fres.copy()
        f_rough, z_rough = await crs.sweep_qres(
                fres_rough,
                ares,
                qres,
                npoints = npoints_rough,
                nsamps = nsamps,
                verbose = verbose,
                pbar_description = 'Rough sweep',
            )
        
        # Save sweep data and fres_rough
        _save_sweep_data(grp, 's21_rough', f_rough, z_rough)
        grp.create_array(name = f'fres_rough', data = fres_rough)
        # Save fres update method and cable delay
        grp.create_array(name = f'fres_update_method', 
                         data = fres_update_method)
        grp.create_array(name = f'cable_delay', data = cable_delay)

        # Assign ch_map 
        ch_map = crs.ch_map

        # Update fres based on rough sweep
        fres = update_fres(
            f_rough, z_rough, 
            fres, qres, 
            fcal_indices = np.where(res_idxs < 0)[0], 
            method = fres_update_method,
            cable_delay = cable_delay,
            plotq = False
            )
        # Some assertions until update_fres is well-tested
        assert isinstance(fres, np.ndarray)
        assert fres.shape == len(res_idxs)
        assert fres.dtype == np.float64
    else:
        ch_map = None
        
    # save fres here (after potential update) 
    grp.create_array(name = f'fres', data = fres)

    # Gain sweep 
    f_gain, z_gain = await crs.sweep_qres(
            fres,
            ares,
            qres / gain_span_factor,
            npoints = npoints_gain,
            nsamps = nsamps,
            ch_map = ch_map,
            verbose = verbose,
            pbar_description = 'Gain sweep',
        )
    _save_sweep_data(grp, 's21_gain', f_gain, z_gain)
    ch_map = crs.ch_map
    for module_idx, chs in ch_map.items():
        chs = np.asarray(chs, dtype = np.int32)
        grp.create_array(name = f'ch_map_mod{module_idx:d}', data = chs)

    # Fine sweep
    f_fine, z_fine = await crs.sweep_qres(
            fres,
            ares,
            qres,
            npoints = npoints_fine,
            nsamps = nsamps,
            ch_map = ch_map,
            verbose = verbose,
            pbar_description = 'Fine sweep',
        )
    _save_sweep_data(grp, 's21_fine', f_fine, z_fine)

################################################################################
# CRS config saving helper 
################################################################################
def write_crs_config_to_zarr(crs, grp):
    """
    Write CRS configuration parameters to a Zarr group. 

    Parameters:
    crs (CRS): initialized CRS instrument class.
    grp (zarr.Group): Zarr group to which configuration data is saved.

    Returns:
    None
    """
    # Input validation 
    if not isinstance(crs, crs.CRS):
        raise TypeError("crs must be an instance of CRS class.")
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.") 
    for name in ['ch_map', 'nco_freqs', 'firmware_release',
                 'analog_bank_high', 'bw', 'clock_source',
                 'dec_module_idxs', 'dec_short', 'dec_stage',
                 'extended_bw', 'sample_freq', 'serial_number',
                 'rfmux_version', 'citkid_version']:
        if not hasattr(crs, name):
            raise ValueError(f"crs is missing attribute '{name}'.")
        if name in grp.keys():
            raise ValueError(f"Zarr group already contains dataset '{name}'.")
    if not hasattr(crs.firmware_release, 'version') or \
        not isinstance(crs.firmware_release.version, str):
        raise ValueError("crs.firmware_release.version must be a string.")
    util._validate_ch_map(crs.ch_map)
    
    # ch_map
    for module_idx, chs in crs.ch_map.items():
        grp.create_array(
            name = f'chs_module{module_idx:d}',
            data = np.asarray(
                chs, 
                dtype = np.int32
            )
        ) 
    # nco_freqs
    for module_idx, nco in crs.nco_freqs.items():
        name = f'nco_module{module_idx:d}'
        grp.create_array(
            name = name,
            data = np.asarray(
                nco, 
                dtype = np.float64
            )
        )
    # CRS firmware version
    name = 'firmware_version'
    grp.create_array(
        name = name,
        data = np.array(
            crs.firmware_release.version,
            dtype = None
        )
    )
    # Other single-value parameters
    for name, dtype in [
        ('analog_bank_high', np.bool_), 
        ('bw', np.float64), 
        ('clock_source', None),
        ('dec_module_idxs', np.uint8),
        ('dec_short', np.bool_),
        ('dec_stage', np.uint8),
        ('extended_bw', np.bool_),
        ('sample_freq', np.float64),
        ('serial_number', np.uint16),
        ('rfmux_version', None),
        ('citkid_version', None)
    ]:
        grp.create_array(
            name = name,
            data = np.asarray(
                getattr(crs, name),
                dtype = dtype
            )
        )

################################################################################
########################### Zarr saving helper #################################
################################################################################
def _save_sweep_data(grp, prefix, f, z):
    """
    Save sweep data to zarr group.
    
    Parameters:
    grp (zarr.Group): zarr group to save data to.
    prefix (str): prefix for dataset names.
    f (np.ndarray, float64): frequency data.
    z (np.ndarray, complex128): s21 data.
    
    Returns:
    None
    """
    # Input validation
    f = np.asarray(f, dtype = np.float64) 
    z = np.asarray(z, dtype = np.complex128)
    if f.shape != z.shape:
        raise ValueError("f and z must have the same shape.")
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.") 
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string.") 
    
    # Save to group
    grp.create_array(name = f'{prefix}_f', 
                    data = f, 
                    chunks = (1, f.shape[1]), 
                    dtype = np.float64
                    )
    grp.create_array(name = f'{prefix}_z', 
                    data = z, 
                    chunks = (1, z.shape[1]), 
                    dtype = np.complex128
                    )
    
################################################################################
########################### Input validation helpers ###########################
################################################################################
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
    cable_delay,
    verbose,
):
    """
    Validate inputs to target_sweep function.
    
    Parameters:
    see target_sweep function.

    Returns:
    fres, ares, qres, res_idxs: validated and converted inputs.
    """
    if not isinstance(crs, crs.CRS):
        raise TypeError("crs must be an instance of CRS class.") 
    # Validate array-like inputs
    fres = np.asarray(fres, dtype = np.float64)
    ares = np.asarray(ares, dtype = np.float64)
    qres = np.asarray(qres, dtype = np.float64)
    res_idxs = np.asarray(res_idxs, dtype = np.int32)
    if fres.ndim != 1:
        raise ValueError("fres must be a 1D array.")
    if not (fres.shape == ares.shape == qres.shape == res_idxs.shape):
        raise ValueError(
            "fres, ares, qres, and res_idxs must have the same shape."
        )
    # gain_span_factor
    if not isinstance(gain_span_factor, (int, float, np.integer, np.floating)) \
        or gain_span_factor <= 1:
        raise ValueError("gain_span_factor must be a number > 1.") 
    # npoints_fine, npoints_gain, npoints_rough
    if npoints_fine is not None and \
        (not isinstance(npoints_fine, (int, np.integer)) or npoints_fine <= 0):
        raise ValueError("npoints_fine must be a positive integer.") 
    if npoints_gain is not None and \
        (not isinstance(npoints_gain, (int, np.integer)) or npoints_gain <= 0):
        raise ValueError("npoints_gain must be a positive integer.") 
    if npoints_rough is not None and \
        (not isinstance(npoints_rough, (int, np.integer)) or npoints_rough <=0):
        raise ValueError("npoints_rough must be a positive integer.") 
    # nsamps
    if not isinstance(nsamps, (int, np.integer)) or nsamps <= 0:
        raise ValueError("nsamps must be a positive integer.") 
    # fres_update_method
    if fres_update_method not in ['distance', 'spacing', 'minS21']:
        msg = "fres_update_method must be 'distance', 'spacing', or 'minS21'."
        raise ValueError(msg)
    # cable_delay
    if not isinstance(cable_delay, (float, np.floating)) or \
        cable_delay < 0:
        raise ValueError("cable_delay must be a positive number.") 
    # verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")
    
    # Validate zarr group and ensure that datasets do not already exist
    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError("grp must be a zarr Group instance.")
    def zarr_key_exists_check(grp, name):
        if name in grp:
            raise ValueError(f"Zarr group already contains dataset '{name}'.")
    if npoints_rough is not None:
        for name in ['s21_rough_f', 's21_rough_z', 'fres_rough']:
            zarr_key_exists_check(grp, name)
    if npoints_gain is not None:
        for name in ['s21_gain_f', 's21_gain_z']:
            zarr_key_exists_check(grp, name)
    if npoints_fine is not None:
        for name in ['s21_fine_f', 's21_fine_z']:
            zarr_key_exists_check(grp, name) 
    zarr_key_exists_check(grp, 'fres')       
            
    # Return validated inputs
    return fres, ares, qres, res_idxs