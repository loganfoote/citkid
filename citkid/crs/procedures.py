import numpy as np
import zarr
from typing import TYPE_CHECKING
from citkid.multitone.fres import update_fres
from . import util

if TYPE_CHECKING:
    from . import instrument as inst

async def target_sweep(
    crs,
    fres,
    ares,
    qres,
    res_idxs,
    grp,
    ch_map = None, 
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
    ch_map (dict or None): Channel mapping dictionary. If None, crs.ch_map is 
        generated during the first sweep (rough or gain). 
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
    fres (np.ndarray, float64): updated resonant frequencies after rough sweep.
    ch_map (dict): channel mapping dictionary after sweeps.
    """
    # Input validation 
    fres, ares, qres, res_idxs = _validate_target_sweep_inputs(
        crs, fres, ares, qres, res_idxs, grp, gain_span_factor, npoints_fine, 
        npoints_gain, npoints_rough, nsamps, fres_update_method, 
        cable_delay, verbose
    )

    # Save input data 
    grp.create_array(name = 'ares', data = ares)
    grp.create_array(name = 'qres', data = qres)
    grp.create_array(name = 'res_idxs', data = res_idxs)
    grp.attrs['nsamps'] = nsamps
    util.write_system_cfg_to_zarr(crs, grp)
    
    if npoints_rough is not None:
        # Rough sweep 
        grpr = grp.require_group('rough_sweep')
        fres_rough = fres.copy()
        f_rough, z_rough = await crs.sweep_qres(
                fres_rough,
                ares,
                qres,
                npoints = npoints_rough,
                nsamps = nsamps,
                ch_map = ch_map,
                dec_grp = grpr,
                verbose = verbose,
                pbar_description = 'Rough sweep',
            )
        
        # Save sweep data and fres_rough
        _save_sweep_data(grpr, '', f_rough, z_rough)
        grpr.create_array(name = 'fres', data = fres_rough)

        # Save fres update method and cable delay
        grpr.attrs['fres_update_method'] = fres_update_method 
        grpr.attrs['cable_delay'] = cable_delay

        # Assign ch_map 
        ch_map = crs.ch_map 
        # Is ch_map persistent after sweep? - it should be, but check

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
        assert fres.shape == res_idxs.shape
        assert fres.dtype == np.float64
        
    # save fres (after potential update) 
    grp.create_array(name = f'fres', data = fres)

    # Gain sweep 
    if npoints_gain is not None:
        grpg = grp.require_group('gain_sweep')
        f_gain, z_gain = await crs.sweep_qres(
                fres,
                ares,
                qres / gain_span_factor,
                npoints = npoints_gain,
                nsamps = nsamps,
                ch_map = ch_map,
                dec_grp = grpg,
                verbose = verbose,
                pbar_description = 'Gain sweep',
            )
        _save_sweep_data(grpg, '', f_gain, z_gain)
        # If ch_map was determined within the gain sweep, 
        # # save here so fine sweep is consistent
        ch_map = crs.ch_map 

    # Fine sweep
    if npoints_fine is not None:
        grpf = grp.require_group('fine_sweep')
        f_fine, z_fine = await crs.sweep_qres(
                fres,
                ares,
                qres,
                npoints = npoints_fine,
                nsamps = nsamps,
                ch_map = ch_map,
                dec_grp = grpf,
                verbose = verbose,
                pbar_description = 'Fine sweep',
            )
        _save_sweep_data(grpf, '', f_fine, z_fine)

    # return updated fres and ch_map 
    return fres, ch_map

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
    if prefix:
        prefix += '_'
    grp.create_array(name = f'{prefix}f', 
                     data = f, 
                     chunks = (1, f.shape[1])
                    )
    grp.create_array(name = f'{prefix}z', 
                     data = z, 
                     chunks = (1, z.shape[1])
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
    # Check if it's a CRS instance by verifying class name
    if type(crs).__name__ != 'CRS' and type(crs).__name__ != 'DummyCRS':
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