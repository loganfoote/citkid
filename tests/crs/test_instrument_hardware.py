import pytest
import rfmux
import numpy as np 
import zarr 
import os 
import shutil
import re
from citkid.crs.instrument import CRS
from citkid.crs import util
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

################################################################################
# Fixtures
################################################################################
@pytest.fixture(scope="module", autouse=True)
def require_hardware(pytestconfig):
    if not pytestconfig.getoption("--crs"):
        pytest.skip("Hardware tests disabled (use --crs)")

    sn = pytestconfig.getoption("--crs_sn")
    if sn is None:
        pytest.skip("No CRS serial number specified (--crs_sn)")

    iface = pytestconfig.getoption("--crs_iface")
    if iface is None:
        pytest.skip("No CRS interface specified (--crs_iface)")

################################################################################
# CRS intitialization function 
################################################################################
def initialize_crs(pytestconfig):
    sn = int(pytestconfig.getoption("--crs_sn"))
    iface = pytestconfig.getoption("--crs_iface")

    crs = CRS(sn, iface)
    return crs

################################################################################
# General hardware interaction tests
################################################################################
@pytest.mark.asyncio
async def test_crs_config(pytestconfig, capsys):
    """ 
    Tests CRS __init__, set_analog_bank, set_clock_source, 
    set_extended_module_bandwidth, and configure_system methods.
    """
    crs = initialize_crs(pytestconfig)
    assert isinstance(crs.d, rfmux.core.schema.CRS)

    ### Test set_analog_bank
    await crs.configure_system(verbose = False)
    for analog_bank_high in [True, False]:
        for full_scale_dbm in [1, 3, 7]:
            await crs.set_analog_bank(analog_bank_high,
                                    full_scale_dbm)
            
            # Check dac scale
            assert crs.d.full_scale_dbm == full_scale_dbm

    ### Test clock source 
    # should reset to VCXO and raise warning
    with pytest.warns(
        UserWarning,
        match = ("Requested clock source SMA unavailable. "
                "Using VCXO instead.")
    ):
        await crs.set_clock_source("SMA")

    ### set_extended_module_bandwidth
    with pytest.warns(
        UserWarning,
        match = "Extended module bandwidth set"
    ):
        await crs.set_extended_bw(True)

    await crs.set_extended_bw(False)

    ### test configure_system verbose 
    captured = capsys.readouterr() 
    await crs.configure_system(
        clock_source = "VCXO", full_scale_dbm = 7,
        analog_bank_high = False, verbose = False
    )
    captured = capsys.readouterr() 
    assert captured.out == ""
    assert captured.err == ""

    await crs.configure_system(
        clock_source = "VCXO", full_scale_dbm = 7,
        analog_bank_high = False, verbose = True
    )
    captured = capsys.readouterr() 
    msg = "Set decimation: dec_stage = 6, short = False, modules = []\n"
    msg += "System configured\nClocking source is VCXO\n" 
    assert captured.out == msg
    assert captured.err == ""

@pytest.mark.asyncio
async def test_nco_tones(pytestconfig, capsys):
    """ 
    Tests CRS set_nco, write_tones, _clear_channels, and disable_module methods.
    """
    ### Initialize board 
    crs = initialize_crs(pytestconfig)
    await crs.configure_system(
        clock_source = "VCXO", full_scale_dbm = 7,
        analog_bank_high = False, verbose = False
    )

    ### set_nco
    # Check first initialization
    nco_freqs = {
        1: 0.5e9,
        2: 0.99e9,
        4: 1.48e9
    }
    nco_freqs0 = nco_freqs.copy()
    await crs.set_nco(nco_freqs)
    # no modification to input
    assert nco_freqs0 == nco_freqs 
    # crs.nco_freqs is measured - should be slightly different from input
    assert nco_freqs != crs.nco_freqs
    # Should have the same keys and values 
    for key, val in nco_freqs.items():
        assert np.isclose(val, crs.nco_freqs[key], 1)
    assert set(nco_freqs.keys()) == set(crs.nco_freqs.keys())
    captured = capsys.readouterr() 
    assert captured.out == (
    "Module 1 NCO is 500.0 MHz\n"
    "Module 2 NCO is 990.0 MHz\n"
    "Module 4 NCO is 1480.0 MHz\n" 
    )
    assert captured.err == ""

    # Check when one is modified 
    nco_freqs = {1: 1e9}
    new_dict = {1: 0.5e9, 2: 0.99e9, 4: 1.48e9}
    await crs.set_nco(nco_freqs, verbose = False)
    assert set(new_dict.keys()) == set(crs.nco_freqs.keys())
    for key, val in new_dict.items():
        assert np.isclose(val, crs.nco_freqs[key], 1)

    # Check when new module is set
    nco_freqs = {3: 1e9}
    new_dict = {3: 1e9, 1: 0.5e9, 2: 0.99e9, 4: 1.48e9}
    await crs.set_nco(nco_freqs, verbose = False)
    assert set(new_dict.keys()) == set(crs.nco_freqs.keys())
    for key, val in new_dict.items():
        assert np.isclose(val, crs.nco_freqs[key], 1)
    captured = capsys.readouterr() 
    assert captured.out == ""
    assert captured.err == ""

    ### write_tones, _clear_channels, disable_module 
    await crs.disable_modules([1, 2, 3, 4])

    # Try to set without NCOs set

    # Needs to be slightly narrower than 250 MHz because 
    # NCO will be slightly off from what is set
    fres = np.linspace(751e6, 1249e6, 900)
    ares = np.ones(fres.shape, dtype = np.float64) * -55
    with pytest.raises(RuntimeError):
        await crs.write_tones(fres, ares)

    # set on one module with NCOs set 
    await crs.set_nco({1: 1e9}, verbose = False) 
    await crs.write_tones(fres, ares)
    await validate_ch_maps(crs)

    # Confirm validate_ch_maps fails with wrong fres_map
    crs.fres_map = {1: np.flip(fres)} 
    with pytest.raises(AssertionError):
        await validate_ch_maps(crs)

    # Two modules
    fres = np.linspace(751e6, 1249e6, 900)
    ares = np.random.choice([-55., -50., -60.], 
                    fres.shape)
    await crs.set_nco({2: 1.49e9}, verbose = False) 
    await crs.write_tones(fres, ares)
    await validate_ch_maps(crs) 
    # clear module 1 
    await crs._clear_channels([1])
    await validate_ch_maps(crs) 
    assert 1 not in crs.fres_map.keys()
    # re-write tones 
    await crs.write_tones(fres, ares) 
    assert len(crs.fres_map[1]) > 0 
    # reset module 1 NCO: should clear tones 
    await crs.set_nco({1: 1e9}, verbose = False)
    await validate_ch_maps(crs) 
    assert 1 not in crs.fres_map.keys()
    assert len(crs.fres_map[2]) > 0
    # disable module 2 
    await crs.disable_modules([2]) 
    await validate_ch_maps(crs) 
    assert list(crs.nco_freqs.keys()) == [1]
    assert 2 not in crs.fres_map.keys()

    # Check custom ch map 
    await crs.set_nco({1: 1e9, 2: 1e9}, verbose = False)
    fres = np.array([1e9, 1.2e9]) 
    ares = np.array([-55, -50]) 
    await crs.write_tones(fres, ares)
    await validate_ch_maps(crs) 
    assert len(crs.fres_map[1]) == 2 
    assert len(crs.fres_map[2]) == 0

    ch_map = {1: [1], 2: [0]}
    await crs.write_tones(fres, ares, ch_map = ch_map)
    await validate_ch_maps(crs) 
    assert np.allclose(crs.fres_map[1], [1.2e9], 
                    atol = 1e-3)
    assert np.allclose(crs.fres_map[2], [1e9], 
                    atol = 1e-3)
    assert np.allclose(crs.ares_map[1], [-50.], 
                    atol = 1e-3)
    assert np.allclose(crs.ares_map[2], [-55], 
                    atol = 1e-3)

    # allow_missing 
    await crs.disable_modules([1, 2, 3, 4])
    await crs.set_nco({1: 1e9}, verbose = False)
    fres = np.array([1e9, 2e9]) 
    ares = np.array([-55, -50]) 

    msg = "Tones must be within 250 MHz of an NCO frequency."
    msg_w = msg + " Ignoring 1 tone(s)."
    with pytest.raises(ValueError, match = re.escape(msg)):
        await crs.write_tones(fres, ares)
    with pytest.warns(UserWarning, match = re.escape(msg_w)):
        await crs.write_tones(fres, ares, 
                            allow_missing = True)
    await validate_ch_maps(crs) 
    assert len(crs.fres_map[1]) == 1 

@pytest.mark.asyncio 
async def test_decimation_placeholder():
    """ Placeholder for future decimation tests. """
    pass

@pytest.mark.asyncio
async def test_sweep(pytestconfig, monkeypatch):
    """ tests CRS sweep method """
    ### Test verbose and basic functionality
    crs = initialize_crs(pytestconfig) 
    await crs.configure_system(
        clock_source = "VCXO", full_scale_dbm = 7,
        analog_bank_high = False, verbose = False
    )

    # Initialize NCOs: two modules
    nco_freqs = {
        1: 0.5e9,
        2: 0.9e9,
    }
    await crs.set_nco(nco_freqs, verbose = False) 
    
    # Patch the tqdm symbol imported in citkid.crs.instrument
    spy = TqdmSpy() 
    monkeypatch.setattr("citkid.crs.instrument.tqdm", spy)
    fres = np.linspace(260e6, 1140e6, 500)
    ares = np.ones(fres.shape, dtype = np.float64) * -55
    
    nsamps = 10
    npoints = 10
    span = 100e3
    frequencies = np.linspace(fres + span / 2, fres - span / 2, npoints).T
    await validate_ch_maps(crs)
    f, z = await crs.sweep(
        frequencies, 
        ares, 
        nsamps = nsamps,
        allow_missing = False,
        verbose = True,
        pbar_description = "test desc"
        )
    await validate_ch_maps(crs)
    # Check that tqdm was called
    assert len(spy.calls) == 2 # one for each module
    _, kwargs1, stub1 = spy.calls[0]
    _, kwargs2, stub2 = spy.calls[1]
    assert kwargs1["total"] == npoints
    assert kwargs2["total"] == npoints
    assert kwargs1["leave"] is False
    assert kwargs2["leave"] is False
    assert stub1.desc == "test desc"
    assert stub2.desc == "test desc"
    # Check that output shapes and values are correct 
    assert f.dtype == np.float64 
    assert np.allclose(f, frequencies, atol = 1) 
    assert z.dtype == np.complex128
    assert z.shape == (len(fres), npoints)

    ### Custom channel map 
    # Initialize NCOs: two modules
    nco_freqs = {
        3: 1e9,
        4: 1e9,
    }
    await crs.set_nco(nco_freqs, verbose = False) 

    fres = np.array([1.1e9, 0.9e9])
    ares = np.ones(fres.shape, dtype = np.float64) * -55
    ch_map = {3: [0], 4: [1]}
    nsamps = 10
    npoints = 10
    span = 100e3
    frequencies = np.linspace(fres + span / 2, fres - span / 2, npoints).T
    await validate_ch_maps(crs)
    f, z = await crs.sweep(
        frequencies, 
        ares, 
        nsamps = nsamps,
        ch_map = ch_map,
        allow_missing = False,
        verbose = False
        )
    await validate_ch_maps(crs)
    assert len(crs.fres_map[3]) == 0 
    assert len(crs.fres_map[4]) == 0
    assert crs.fres_map.get(1) is None
    assert crs.fres_map.get(2) is None
    assert f.dtype == np.float64
    assert np.allclose(f, frequencies, atol = 1)
    assert z.dtype == np.complex128
    assert z.shape == (len(fres), npoints) 

    ### missing channels 
    nco_freqs = {
        1: 1e9,
        2: 1e9,
        3: 1e9, 
        4: 1e9
    }
    await crs.set_nco(nco_freqs, verbose = False)

    fres = np.array([1e9, 2e9])
    ares = np.ones(fres.shape, dtype = np.float64) * -55
     
    nsamps = 10
    npoints = 10
    span = 100e3
    frequencies = np.linspace(fres + span / 2, fres - span / 2, npoints).T

    msg = "Tones must be within 250 MHz of an NCO frequency."
    msg_w = msg + " Proceeding with 1 missing channel(s)."
    await validate_ch_maps(crs)
    with pytest.raises(ValueError, match = re.escape(msg)):
        f, z = await crs.sweep(
            frequencies, 
            ares, 
            nsamps = nsamps,
            allow_missing = False,
        verbose = False
        )
    await validate_ch_maps(crs)
    with pytest.warns(UserWarning, match = re.escape(msg_w)):
        f, z = await crs.sweep(
            frequencies, 
            ares, 
            nsamps = nsamps,
            allow_missing = True,
            verbose = False
        )
    await validate_ch_maps(crs)

    assert f.dtype == np.float64
    assert np.allclose(f[0], frequencies[0], atol = 1)
    assert np.all(np.isnan(f[1]))
    assert z.dtype == np.complex128
    assert np.all(np.isfinite(z[0])) 
    assert np.all(np.isnan(z[1]))

    ### ch_map with missing channels 
    nco_freqs = {
        1: 1e9,
        2: 1e9,
        3: 1e9,
        4: 1e9,
    }
    await validate_ch_maps(crs)
    await crs.set_nco(nco_freqs, verbose = False) 
    await validate_ch_maps(crs)
    fres = np.array([1.1e9, 0.9e9])
    ares = np.ones(fres.shape, dtype = np.float64) * -55
    ch_map = {3: [1]}  # missing channel 0
    nsamps = 10
    npoints = 10
    span = 100e3
    frequencies = np.linspace(fres + span / 2, fres - span / 2, npoints).T
    await validate_ch_maps(crs)
    with pytest.warns(UserWarning, match = re.escape(msg_w)):
        f, z = await crs.sweep(
            frequencies, 
            ares, 
            nsamps = nsamps,
            ch_map = ch_map,
            allow_missing = True,
            verbose = False
            )
    await validate_ch_maps(crs)
    assert len(crs.fres_map[3]) == 0
    assert crs.fres_map.get(1) is None
    assert crs.fres_map.get(2) is None
    assert crs.fres_map.get(4) is None
    assert f.dtype == np.float64
    assert np.allclose(f[1], frequencies[1], atol = 1)
    assert np.all(np.isnan(f[0]))
    assert z.dtype == np.complex128
    assert z.shape == (len(fres), npoints) 
    assert np.all(np.isfinite(z[1]))
    assert np.all(np.isnan(z[0]))

@pytest.mark.asyncio
async def test_stream(pytestconfig, monkeypatch, tmp_path):
    """ 
    Tests whether streaming executes properly, and if the amplitue and phase 
    matches sweep (just for a couple modules).
    """
    ### Initialize board 
    crs = initialize_crs(pytestconfig)
    await crs.configure_system(
        clock_source = "VCXO", full_scale_dbm = 7,
        analog_bank_high = False, verbose = False
    )

    ### set_nco
    await crs.set_nco({1: 1e9, 2: 2e9})

    fres = np.array([0.9e9, 1e9, 1.1e9, 1.9e9, 2.0e9, 2.1e9])
    ares = np.array([-55, -55, -50, -50, -55, -50], dtype = np.float64)

    root = zarr.open(os.path.join(tmp_path, 'data.zarr'), mode = 'a')
    grp = root.require_group('sample_0')

    spy = TqdmSpy()
    monkeypatch.setattr("citkid.util.tqdm", spy)
    await crs.capture_ts(
        fres = fres,
        ares = ares,
        ts_duration_s = 4,
        dec_stage = 4,
        grp = grp,
        tmp_directory = os.path.join(tmp_path, 'tmp/'),
        batch_size_mb = 1,
        delete_parser_data = True,
        verbose = True
    )
    _, zsweep = await crs.sweep_linear(
        fres, ares, 1e3, 1, 100, 
        allow_missing = False, center_fres = True, downward = True,
        verbose = False
    )
    assert len(spy.calls) == 1 # one for each module
    _, kwargs1, stub1 = spy.calls[0]
    assert kwargs1["leave"] is False
    assert stub1.desc == "Streaming"

    # Check output 
    counts_to_dbc = grp['counts_to_dbc']
    dt = float(grp['dt'][...])
    z = grp['z']

    assert counts_to_dbc.shape == (len(fres),)
    assert np.isclose(dt, 1 / util.get_sample_freq(4))
    assert z.shape[0] == 2
    assert z.shape[1] == len(fres)
    assert np.isclose(z.shape[2] * dt, 4.0, atol = 0.2)

    # Check that sweep matches parser 
    z = z[0, :, :] + 1j * z[1, :, :]
    z *= np.array(counts_to_dbc)[:, np.newaxis]
    z_avg = np.mean(z, axis = 1) 
    zsweep = zsweep[:, 0]

    abs_tol = 1e-2
    abs_diff = np.abs(zsweep - z_avg)
    flat_idx = np.nanargmax(abs_diff)
    max_idx = np.unravel_index(flat_idx, abs_diff.shape)
    max_abs = abs_diff.ravel()[flat_idx]
    if max_abs > abs_tol:
        zs_val = zsweep.ravel()[flat_idx]
        za_val = z_avg.ravel()[flat_idx]
        raise AssertionError(
            "Max absolute diff exceeds tolerance: "
            f"idx={max_idx}, zsweep={zs_val}, z_avg={za_val}, "
            f"abs_diff={max_abs} (tol={abs_tol})"
        )

def test_parser_to_zarr():
    raise Exception("Known bug: parser_to_zarr cannot handle missing tones.")

################################################################################
# Loopback tests 
################################################################################
@pytest.mark.asyncio 
async def test_loopback(pytestconfig, tmp_path):
    """ Placeholder for future loopback tests. 
    Tests all CRS modules in loopback mode across the first and second Nyquist 
    zones. Ensures that loopback profile is within 5 dB of specifications. 
    Ensures that streaming matches loopback values.
    """
    raise NotImplementedError("Need to add phase to loopback test plots.")
    ### Initialize board 
    crs = initialize_crs(pytestconfig)
    full_scale_dbm = 7
    await crs.configure_system(
        clock_source = "VCXO", full_scale_dbm = full_scale_dbm,
        analog_bank_high = False, verbose = False
    )

    # Setup sweep parameters
    ncos = [0.25, 0.75, 1.25, 1.75, 2.25, 
        2.75, 3.25, 3.75, 4.25, 4.75] 
    ncos = np.array(ncos, dtype = np.float64) * 1e9 
    amplitude = -50 
    npoints = 1 
    nsamps = 10

    data = {} 
    data_ts = {}
    tmp_directory = 'tmp/'
    zarr_path = os.path.join(tmp_directory, 'tmp_zarr.zarr')
    root = zarr.open(zarr_path, mode = 'a')
    run_idx = 0
    try:
        for analog_bank_high in [False, True]:
            await crs.set_analog_bank(analog_bank_high,
                                        full_scale_dbm)
            for nco in ncos:
                if analog_bank_high:
                    all_modules = range(5, 9)  
                else:
                    all_modules = range(1, 5)
                await crs.set_nco(
                    {idx: nco for idx in all_modules},
                    verbose = False
                )
                
                # Create fres and ares arrays -> same as sweep_full, 
                # but with custom channel map since all NCOs are the 
                # same
                tone_bw = crs.bw / 1024 + 200
                # spacing = tone_bw / npoints
                fres = np.concatenate(
                    [np.linspace(nco - crs.bw / 2 + 10 + tone_bw,
                                nco + crs.bw / 2 - 10 - tone_bw,
                                1024) 
                    for nco in crs.nco_freqs.values()]
                )
                ares = amplitude * np.ones(len(fres))
                
                ch_map = {}
                offset = min(all_modules)
                for idx in all_modules:
                    ch_map[idx] = list(
                        range((idx - offset) * 1024, 
                            (idx - offset) * 1024 + 1024)
                    )
                # Sweep
                f, z = await crs.sweep_linear(
                    fres, ares, tone_bw, npoints, nsamps, 
                    ch_map = ch_map, allow_missing = False, 
                    center_fres = True, downward = True, 
                    verbose = False
                    )
                subdata = {k: [f[v].flatten(), z[v].flatten()] 
                        for k, v in ch_map.items()
                        }
                # Append to data
                for idx in subdata:
                    if idx in data:
                        for i in range(2):
                            data[idx][i] = np.concatenate(
                                [data[idx][i], subdata[idx][i]]
                            )
                    else:
                        data[idx] = subdata[idx]
                # Take short ts 
                ts_duration_s = 0.2 
                dec_stage = 5 
                grp = root.require_group(f'run_{run_idx:d}')
                run_idx += 1
                ts_path = os.path.join(tmp_directory, 'parser/') 
                os.makedirs(ts_path, exist_ok = True)
                await crs.capture_ts(
                    fres,
                    ares, 
                    ts_duration_s,
                    dec_stage,
                    grp,
                    ch_map = ch_map,
                    allow_missing = False,
                    tmp_directory = ts_path,
                    batch_size_mb = 1000,
                    chunk_size_mb = 128,
                    delete_parser_data = True,
                    verbose = False
                )

                # Import ts and append to ts_data
                z = np.array(grp['z'], dtype = np.float64)
                s = np.array(grp['counts_to_dbc'])
                z *= s[np.newaxis, :, np.newaxis]
                z = z[0] + 1j * z[1]
                for k, chs in ch_map.items():
                    if k in data_ts:
                        data_ts[k][0] = np.concatenate(
                            [data_ts[k][0], fres[chs]]
                        )
                        data_ts[k][1].extend(z[chs])
                    else:
                        data_ts[k] = [fres[chs], 
                                    [zi for zi in z[chs]]]

        # Plot loopback profiles
        fig, axs = plt.subplots(
            2, 4, figsize = [16, 8], 
            sharey = True, sharex = True, layout = 'tight'
        ) 
        for ax in axs[-1, :]:
            ax.set(xlabel = 'Frequency (GHz)')
        for ax in axs[:, 0]:
            ax.set(ylabel = r'$|S_{21}|$ (dBc)')
            
        for (module_idx, (fi, zi)), ax in zip(
                data.items(), axs.flatten()
            ):
            
            mask = (fi > 50e6)
            c = plt.cm.cividis(0.)
            fi, zi = fi[mask], zi[mask] 
            mask = fi < 2.3e9
            ax.plot(fi[mask] / 1e9, 20 * np.log10(np.abs(zi[mask])), 
                    color = c, aa = False)
            mask = fi > 2.6e9
            ax.plot(fi[mask] / 1e9, 20 * np.log10(np.abs(zi[mask])), 
                    color = c, aa = False)
            ax.set(title = f'Module: {module_idx:d}')

        # Save figure for reference regardless of test outcome
        artifact_dir = pytestconfig.rootpath / 'crs_test_data'
        artifact_dir.mkdir(parents = True, exist_ok = True)
        fig_paths = [
            tmp_path / 'loopback_profile.png',
            artifact_dir / 'loopback_profile.png'
        ]
        for fig_path in fig_paths:
            fig.savefig(fig_path, dpi = 150)

        status_paths = [
            tmp_path / 'loopback_profile_status.txt',
            artifact_dir / 'loopback_profile_status.txt'
        ]
        try:
            # Confirms that streamed noise matches IQ to within 0.01
            for module_idx, (_, zsweep) in data.items():
                zt = data_ts[module_idx][1] 
                z_avg = np.array([np.mean(zi) for zi in zt])
                abs_tol = 1e-2
                abs_diff = np.abs(zsweep - z_avg)
                flat_idx = np.nanargmax(abs_diff)
                max_idx = np.unravel_index(flat_idx, abs_diff.shape)
                max_abs = abs_diff.ravel()[flat_idx]
                if max_abs > abs_tol:
                    zs_val = zsweep.ravel()[flat_idx]
                    za_val = z_avg.ravel()[flat_idx]
                    raise AssertionError(
                        "Max absolute diff exceeds tolerance: "
                        f"idx={max_idx}, zsweep={zs_val}, z_avg={za_val}, "
                        f"abs_diff={max_abs} (tol={abs_tol})"
                    )
            for status_path in status_paths:
                status_path.write_text("PASS\n")
        except AssertionError as exc:
            for status_path in status_paths:
                status_path.write_text(f"FAIL: {exc}\n")
            raise
    finally:
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path, ignore_errors = True)
        if os.path.exists(ts_path):
            shutil.rmtree(ts_path, ignore_errors = True)
        

################################################################################
# Helper functions
################################################################################
async def validate_ch_maps(crs):
    """
    Validates that channel frequencies and amplitudes on the 
    board correspond to the values stored in fres_map, ares_map 
    and nco_freqs.

    Parameters:
    crs (citkid.crs.instrument.CRS): CRS object. 
    """
    for module_idx, nco in crs.nco_freqs.items():
        if module_idx in crs.fres_map:
            fres = crs.fres_map[module_idx]
            ares = crs.ares_map[module_idx]
        else:
            assert module_idx not in crs.ares_map 
            fres = np.array([], dtype = np.float64)
            ares = np.array([], dtype = np.float64)
        N = len(fres) 
        async with crs.d.tuber_context() as ctx:
            for ch in range(1, 1025):
                ctx.get_frequency(
                    channel = ch,
                    module = module_idx
                )
                ctx.get_amplitude(
                    channel = ch,
                    module = module_idx
                )
            results = await ctx()
        fres_meas = np.array(results[::2], 
                             dtype = np.float64) 
        assert all(fres_meas[N:] == 0.0)
        fres_meas += nco
        ares_meas = np.array(results[1::2], 
                             dtype = np.float64)
        assert all(ares_meas[N:] == 0.0)
        
        assert np.allclose(
            fres, 
            fres_meas[:N], 
            atol = 1e-3
        )
        
        ares_amp = 10 ** ((ares - crs.d.full_scale_dbm) / 20)
        assert np.allclose(
            ares_amp, 
            ares_meas[:N], 
            atol = 1e-7
        )

class TqdmSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        stub = _TqdmStub(args[0] if args else None, kwargs)
        self.calls.append((args, kwargs, stub))
        return stub


class _TqdmStub:
    def __init__(self, iterable, kwargs=None):
        self.iterable = iterable
        self.desc = None if kwargs is None else kwargs.get("desc")
        self.n = 0

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else [])

    def set_description(self, desc):
        self.desc = desc

    def refresh(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
