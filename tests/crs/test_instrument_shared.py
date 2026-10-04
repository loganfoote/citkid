"""
Tests for sharing one CRS between several sessions, per-module settings, and
related bookkeeping.

Covers module ownership (``module_idxs``), ``global_control``, ``attach``,
``configure_modules``, shared decimation, and the tone/sweep bookkeeping that
depends on them.
"""

import os
import pytest
import numpy as np
import zarr
import warnings
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from citkid.crs.instrument import CRS, _sweep
from citkid.crs.procedures import _validate_target_sweep_inputs
from .conftest import mock_module_settings, default_module_cfg


def make_crs(module_idxs = None, global_control = True):
    """
    Create a CRS with mocked rfmux. Must be called inside the
    ``mock_rfmux_base`` fixture.

    Parameters:
    module_idxs (list or None): modules owned by the session.
    global_control (bool): whether the session has global control.

    Returns:
    CRS: CRS instance.
    """
    return CRS(serial_number = 1, interface = 'eth0',
               module_idxs = module_idxs, global_control = global_control)


def mock_attach_device(device, analog_bank_high = False, dec_stage = 6):
    """
    Mock the device calls used by ``CRS.attach``.

    Parameters:
    device (MagicMock): mock device.
    analog_bank_high (bool): value returned by get_analog_bank.
    dec_stage (int or None): value returned by get_decimation.

    Returns:
    None
    """
    device.resolve = AsyncMock()
    firmware_release = MagicMock()
    firmware_release.version = '1.6.0'
    device.get_firmware_release = AsyncMock(return_value = firmware_release)
    device.get_clock_source = AsyncMock(return_value = 'VCXO')
    device.get_analog_bank = AsyncMock(return_value = analog_bank_high)
    device.get_extended_module_bandwidth = AsyncMock(return_value = False)
    device.get_decimation = AsyncMock(return_value = dec_stage)
    mock_module_settings(device)


################################################################################
################################## __init__ ####################################
################################################################################

def test_init_defaults_to_all_modules(mock_rfmux_base):
    """Test that a default session owns all modules and is not shared."""
    crs = make_crs()
    assert crs.module_idxs == list(range(1, 9))
    assert crs.global_control is True
    assert crs.shared is False
    assert crs.ntones == 0
    assert crs.analog_bank_high is None
    assert crs.module_cfg == {}


@pytest.mark.parametrize("module_idxs, expected", [
    ([3, 1, 1], [1, 3]),
    (np.array([2, 4]), [2, 4]),
    ((5,), [5]),
])
def test_init_normalizes_module_idxs(mock_rfmux_base, module_idxs, expected):
    """Test module_idxs is converted to a sorted list of unique ints."""
    crs = make_crs(module_idxs = module_idxs)
    assert crs.module_idxs == expected
    assert all(type(mi) is int for mi in crs.module_idxs)
    assert crs.shared is True


def test_init_all_modules_explicit_not_shared(mock_rfmux_base):
    """Test that explicitly owning all modules is not a shared session."""
    crs = make_crs(module_idxs = list(range(1, 9)))
    assert crs.shared is False


def test_init_no_global_control_is_shared(mock_rfmux_base):
    """Test that a session without global control is shared."""
    crs = make_crs(global_control = False)
    assert crs.shared is True


@pytest.mark.parametrize("module_idxs, error", [
    ([0], ValueError),
    ([9], ValueError),
    ([], ValueError),
    ([1.0], TypeError),
    ('1', TypeError),
    (1, TypeError),
    ([True], TypeError),
])
def test_init_invalid_module_idxs(mock_rfmux_base, module_idxs, error):
    """Test invalid module_idxs raise errors."""
    with pytest.raises(error):
        make_crs(module_idxs = module_idxs)


def test_init_invalid_global_control(mock_rfmux_base):
    """Test non-boolean global_control raises TypeError."""
    with pytest.raises(TypeError, match = 'global_control'):
        make_crs(global_control = 1)


################################################################################
############################# global control ###################################
################################################################################

@pytest.mark.asyncio
@pytest.mark.parametrize("method, args", [
    ('configure_system', ()),
    ('set_clock_source', ('VCXO',)),
    ('set_analog_bank', (False, 7)),
    ('set_extended_bw', (False,)),
])
async def test_global_methods_require_global_control(
    mock_rfmux_base, method, args
):
    """Test board-wide setters raise without global control."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [3, 4], global_control = False)
    device.set_clock_source = AsyncMock()
    device.set_analog_bank = AsyncMock()
    device.set_extended_module_bandwidth = AsyncMock()

    with pytest.raises(RuntimeError, match = 'does not have global control'):
        await getattr(crs, method)(*args)
    device.set_clock_source.assert_not_called()
    device.set_analog_bank.assert_not_called()
    device.set_extended_module_bandwidth.assert_not_called()


@pytest.mark.asyncio
async def test_set_analog_bank_no_owned_modules_in_bank(mock_rfmux_base):
    """Test set_analog_bank raises before writing if no modules are owned."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [1, 2])
    device.set_analog_bank = AsyncMock()

    with pytest.raises(ValueError, match = 'requested analog bank'):
        await crs.set_analog_bank(True, 7)
    device.set_analog_bank.assert_not_called()


@pytest.mark.asyncio
async def test_set_analog_bank_configures_only_owned_modules(mock_rfmux_base):
    """Test set_analog_bank configures owned modules in the bank only."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [1, 2])
    device.set_analog_bank = AsyncMock()
    device.get_analog_bank = AsyncMock(return_value = False)
    mock_module_settings(device, full_scale_dbm = 5)

    await crs.set_analog_bank(False, 5)

    assert sorted(crs.module_cfg) == [1, 2]
    assert device.set_dac_scale.call_count == 2
    device.set_dac_scale.assert_any_call(5, 'DBM', 1)
    device.set_dac_scale.assert_any_call(5, 'DBM', 2)


################################################################################
################################### attach #####################################
################################################################################

@pytest.mark.asyncio
async def test_attach_reads_board_and_configures_own_modules(
    mock_rfmux_base, capsys
):
    """Test attach reads board-wide settings without writing them."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [3, 4], global_control = False)
    mock_attach_device(device, analog_bank_high = False, dec_stage = 6)
    device.set_clock_source = AsyncMock()
    device.set_analog_bank = AsyncMock()
    device.set_decimation = AsyncMock()
    device.set_timestamp_port = AsyncMock()

    await crs.attach(full_scale_dbm = 7, verbose = True)

    # Board-wide settings are read, not written
    device.resolve.assert_called_once()
    device.set_clock_source.assert_not_called()
    device.set_analog_bank.assert_not_called()
    device.set_decimation.assert_not_called()
    device.set_timestamp_port.assert_not_called()
    assert crs.clock_source == 'VCXO'
    assert crs.analog_bank_high is False
    assert crs.extended_bw is False
    assert crs.bw == 500e6
    assert crs.dec_stage == 6
    assert crs.sample_freq == pytest.approx(625e6 / (256 * 64 * 64))
    assert crs.dec_short is None
    assert crs.dec_module_idxs is None
    assert crs.firmware_release.version == '1.6.0'

    # Only this session's modules are configured
    assert sorted(crs.module_cfg) == [3, 4]
    assert device.set_dac_scale.call_count == 2
    assert crs.ntones == 0
    assert 'Attached' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_attach_high_bank_forgets_low_modules(mock_rfmux_base):
    """Test attach drops state for modules outside the active bank."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [2, 6], global_control = False)
    mock_attach_device(device, analog_bank_high = True)
    crs.nco_freqs = {2: 4.0e9}
    crs.module_cfg = {2: default_module_cfg()}

    await crs.attach(verbose = False)

    assert crs.active_module_idxs == [6]
    assert crs.nco_freqs == {}
    assert sorted(crs.module_cfg) == [6]


@pytest.mark.asyncio
async def test_attach_no_modules_in_bank_raises(mock_rfmux_base):
    """Test attach raises if none of the session's modules are active."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [1, 2], global_control = False)
    mock_attach_device(device, analog_bank_high = True)

    with pytest.raises(RuntimeError, match = 'active analog bank'):
        await crs.attach(verbose = False)
    device.set_dac_scale.assert_not_called()


@pytest.mark.asyncio
async def test_attach_wrong_firmware_raises(mock_rfmux_base):
    """Test attach validates the firmware version."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [1], global_control = False)
    mock_attach_device(device)
    device.get_firmware_release.return_value.version = '1.5.0'

    with pytest.raises(RuntimeError, match = 'firmware must be version'):
        await crs.attach(verbose = False)


@pytest.mark.asyncio
async def test_attach_no_streaming(mock_rfmux_base):
    """Test attach handles a board that is not streaming."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [1], global_control = False)
    mock_attach_device(device, dec_stage = None)

    await crs.attach(verbose = False)

    assert crs.dec_stage is None
    assert crs.sample_freq is None


################################################################################
############################## configure_modules ###############################
################################################################################

@pytest.fixture
def configured_crs(mock_rfmux_base):
    """CRS owning modules 1 and 2, with the low bank active."""
    _, device = mock_rfmux_base
    crs = make_crs(module_idxs = [1, 2])
    crs.analog_bank_high = False
    mock_module_settings(device)
    return crs


@pytest.mark.asyncio
async def test_configure_modules_defaults(configured_crs):
    """Test configure_modules writes rfmux defaults to active modules."""
    crs = configured_crs
    d = crs.d

    await crs.configure_modules()

    for mi in [1, 2]:
        d.set_dac_scale.assert_any_call(7, 'DBM', mi)
        d.set_adc_attenuator.assert_any_call(0, module = mi)
        d.set_nyquist_zone.assert_any_call(1, module = mi)
        d.set_cable_length.assert_any_call(0.0, module = mi)
        d.set_adc_calibration_mode.assert_any_call('AUTO', module = mi)
        d.set_adc_autocal.assert_any_call(True, module = mi)
    assert d.set_dac_scale.call_count == 2
    assert crs.module_cfg == {1: default_module_cfg(),
                              2: default_module_cfg()}


@pytest.mark.asyncio
async def test_configure_modules_custom_values(configured_crs):
    """Test configure_modules writes and stores custom values."""
    crs = configured_crs
    d = crs.d
    d.get_dac_scale.return_value = 3
    d.get_adc_attenuator.return_value = 10
    d.get_nyquist_zone.return_value = 2
    d.get_cable_length.return_value = 1.5
    d.get_adc_calibration_mode.return_value = 'MODE1'
    d.get_adc_autocal.return_value = False

    await crs.configure_modules(
        full_scale_dbm = 3, adc_attenuation_db = 10, nyquist_zone = 2,
        cable_length_m = 1.5, adc_autocal = False,
        adc_calibration_mode = 'MODE1', module_idxs = [2]
        )

    d.set_adc_attenuator.assert_called_once_with(10, module = 2)
    assert list(crs.module_cfg) == [2]
    assert crs.module_cfg[2] == {
        'full_scale_dbm': 3.0, 'adc_attenuation_db': 10.0,
        'nyquist_zone': 2, 'cable_length_m': 1.5,
        'adc_calibration_mode': 'MODE1', 'adc_autocal': False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("getter, value, name", [
    ('get_dac_scale', 5, 'DAC full scale'),
    ('get_adc_attenuator', 3, 'ADC attenuation'),
    ('get_nyquist_zone', 2, 'Nyquist zone'),
    ('get_cable_length', 1.0, 'cable length'),
    ('get_adc_calibration_mode', 'MODE2', 'ADC calibration mode'),
    ('get_adc_autocal', False, 'ADC autocal'),
])
async def test_configure_modules_readback_mismatch(
    configured_crs, getter, value, name
):
    """Test configure_modules raises if a setting does not read back."""
    crs = configured_crs
    getattr(crs.d, getter).return_value = value

    with pytest.raises(RuntimeError, match = f'Failed to set {name}'):
        await crs.configure_modules()
    assert crs.module_cfg == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs, error", [
    ({'full_scale_dbm': 8}, ValueError),
    ({'adc_attenuation_db': -1}, ValueError),
    ({'adc_attenuation_db': 1.5}, ValueError),
    ({'adc_attenuation_db': True}, ValueError),
    ({'nyquist_zone': 3}, ValueError),
    ({'cable_length_m': -1.0}, ValueError),
    ({'cable_length_m': 'a'}, ValueError),
    ({'adc_autocal': 1}, TypeError),
    ({'adc_calibration_mode': 'auto'}, ValueError),
    ({'module_idxs': [3]}, ValueError),
    ({'module_idxs': [5]}, ValueError),
])
async def test_configure_modules_invalid_inputs(configured_crs, kwargs, error):
    """Test configure_modules validates inputs before writing."""
    crs = configured_crs
    if kwargs.get('module_idxs') == [5]:
        crs.module_idxs = [1, 2, 5]

    with pytest.raises(error):
        await crs.configure_modules(**kwargs)
    crs.d.set_dac_scale.assert_not_called()


@pytest.mark.asyncio
async def test_configure_modules_requires_configuration(mock_rfmux_base):
    """Test configure_modules raises before configure_system/attach."""
    crs = make_crs()
    with pytest.raises(RuntimeError, match = 'not configured'):
        await crs.configure_modules()


################################################################################
################################ set_decimation ################################
################################################################################

@pytest.fixture
def dec_crs(mock_rfmux_base):
    """Factory for CRS objects with mocked decimation calls."""
    _, device = mock_rfmux_base
    device.set_decimation = AsyncMock()
    device.get_decimation = AsyncMock(return_value = 6)

    def factory(module_idxs = None, global_control = True):
        crs = make_crs(module_idxs = module_idxs,
                       global_control = global_control)
        crs.analog_bank_high = False
        return crs
    with patch('citkid.crs.instrument.time.sleep'):
        yield factory, device


@pytest.mark.asyncio
async def test_set_decimation_no_global_control_checks_stage(dec_crs):
    """Test set_decimation only checks the stage without global control."""
    factory, device = dec_crs
    crs = factory(module_idxs = [3], global_control = False)

    await crs.set_decimation(6, short = True, module_idxs = [3])

    device.set_decimation.assert_not_called()
    device.get_decimation.assert_called_once()
    assert crs.dec_stage == 6
    assert crs.dec_short is None
    assert crs.dec_module_idxs is None
    assert crs.sample_freq == pytest.approx(625e6 / (256 * 64 * 64))


@pytest.mark.asyncio
@pytest.mark.parametrize("board_stage", [3, None])
async def test_set_decimation_no_global_control_mismatch(dec_crs, board_stage):
    """Test set_decimation raises if the board stage differs."""
    factory, device = dec_crs
    crs = factory(module_idxs = [3], global_control = False)
    device.get_decimation.return_value = board_stage

    with pytest.raises(RuntimeError, match = 'does not have global control'):
        await crs.set_decimation(6)
    device.set_decimation.assert_not_called()


@pytest.mark.asyncio
async def test_set_decimation_shared_stage_6_streams_bank(dec_crs):
    """Test a shared session streams all bank modules at stage 6."""
    factory, device = dec_crs
    crs = factory(module_idxs = [1, 2])
    crs.fres_map = {1: np.ones(10)}

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await crs.set_decimation(6)

    device.set_decimation.assert_called_once_with(
        6, short = False, module = [1, 2, 3, 4])


@pytest.mark.asyncio
async def test_set_decimation_shared_stage_6_high_bank(dec_crs):
    """Test a shared session streams the high bank at stage 6."""
    factory, device = dec_crs
    crs = factory(module_idxs = [5])
    crs.analog_bank_high = True

    await crs.set_decimation(6)

    device.set_decimation.assert_called_once_with(
        6, short = False, module = [5, 6, 7, 8])


@pytest.mark.asyncio
async def test_set_decimation_shared_other_stage_warns(dec_crs):
    """Test a shared session warns and streams its own modules below 6."""
    factory, device = dec_crs
    crs = factory(module_idxs = [1, 2])
    crs.fres_map = {1: np.ones(10), 2: np.array([])}

    with pytest.warns(UserWarning, match = 'Other sessions cannot sweep'):
        await crs.set_decimation(3)

    device.set_decimation.assert_called_once_with(3, short = True,
                                                  module = [1])


@pytest.mark.asyncio
async def test_set_decimation_single_session_uses_fres_map(dec_crs):
    """Test a single session streams modules with tones at stage 6."""
    factory, device = dec_crs
    crs = factory()
    crs.fres_map = {2: np.ones(200), 1: np.ones(10)}

    await crs.set_decimation(6)

    device.set_decimation.assert_called_once_with(6, short = False,
                                                  module = [1, 2])


@pytest.mark.asyncio
async def test_set_decimation_numpy_module_idxs(dec_crs):
    """Test repeated calls with numpy module_idxs don't raise."""
    factory, device = dec_crs
    crs = factory()

    await crs.set_decimation(6, short = False,
                             module_idxs = np.array([2, 1]))
    await crs.set_decimation(6, short = False,
                             module_idxs = np.array([1, 2]))

    assert crs.dec_module_idxs == [1, 2]
    assert device.set_decimation.call_count == 2
    device.set_decimation.assert_called_with(6, short = False,
                                             module = [1, 2])


@pytest.mark.asyncio
async def test_set_decimation_unchanged_does_not_sleep(mock_rfmux_base):
    """Test the settling sleep only happens when settings change."""
    _, device = mock_rfmux_base
    device.set_decimation = AsyncMock()
    crs = make_crs()
    crs.analog_bank_high = False

    with patch('citkid.crs.instrument.time.sleep') as mock_sleep:
        await crs.set_decimation(6, short = False, module_idxs = [1, 2])
        await crs.set_decimation(6, short = False, module_idxs = [2, 1])
    mock_sleep.assert_called_once()


################################################################################
############################### module ownership ###############################
################################################################################

@pytest.mark.asyncio
async def test_set_nco_not_owned_raises(configured_crs):
    """Test set_nco refuses modules owned by another session."""
    crs = configured_crs
    crs.d.set_nco_frequency = AsyncMock()

    with pytest.raises(ValueError, match = r'Modules \[3\] are not owned'):
        await crs.set_nco({1: 4.0e9, 3: 4.5e9}, verbose = False)
    crs.d.set_nco_frequency.assert_not_called()
    assert crs.nco_freqs == {}


@pytest.mark.asyncio
async def test_set_nco_requires_configuration(mock_rfmux_base):
    """Test set_nco raises before configure_system/attach."""
    crs = make_crs()
    with pytest.raises(RuntimeError, match = 'not configured'):
        await crs.set_nco({1: 4.0e9}, verbose = False)


@pytest.mark.asyncio
async def test_capture_ts_clears_only_owned_modules(configured_crs, tmp_path):
    """Test capture_ts clears this session's modules in the bank only."""
    crs = configured_crs
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock(return_value = np.array([4.0e9]))
    crs.stream = AsyncMock()
    grp = zarr.open_group(tmp_path / 'data.zarr', mode = 'w')

    with patch('citkid.crs.instrument.util.write_system_cfg_to_zarr'), \
         patch('citkid.crs.instrument.time.sleep'):
        await crs.capture_ts([4.0e9], [-50.], 1.0, 6, grp, verbose = False,
                             tmp_directory = str(tmp_path / 'tmp'))

    assert [c.args[0] for c in crs._clear_channels.call_args_list] == \
        [[1, 2], [1, 2]]


################################################################################
################################ write_tones ###################################
################################################################################

@pytest.fixture
def tones_crs(base_crs):
    """CRS with NCOs on modules 1 and 2 and mocked hardware calls."""
    crs = base_crs
    crs.analog_bank_high = False
    crs.nco_freqs = {1: 4.0e9, 2: 5.0e9}
    crs.d.clear_channels = AsyncMock()
    mock_modules = MagicMock()
    mock_modules._write_tones = AsyncMock()
    mock_modules._sweep = AsyncMock()
    with patch('citkid.crs.instrument.util.get_modules',
               return_value = mock_modules):
        yield crs, mock_modules


@pytest.mark.asyncio
async def test_write_tones_returns_dithered_frequencies(tones_crs):
    """Test write_tones returns the dithered frequencies in input order."""
    crs, mock_modules = tones_crs
    fres = np.array([5.1e9, 3.9e9, 6.0e9, 4.1e9, 4.05e9])
    ares = np.full(5, -50.)

    with pytest.warns(UserWarning, match = 'Ignoring 1 tone'):
        fres_written = await crs.write_tones(fres, ares, allow_missing = True)

    assert fres_written.shape == fres.shape
    assert np.isnan(fres_written[2])
    ok = ~np.isnan(fres_written)
    assert np.all(np.abs(fres_written[ok] - fres[ok]) <= 50)
    for mi, chs in crs.ch_map.items():
        np.testing.assert_array_equal(fres_written[chs], crs.fres_map[mi])
    # Full scale map is passed to the macro
    assert mock_modules._write_tones.call_args.args[3] == {
        mi: 7.0 for mi in range(1, 9)}
    assert crs.ntones == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("ch_map, match", [
    ({3: [0]}, 'no NCO frequency set'),
    ({1: [0, 2]}, r'must be in \[0, 2\)'),
    ({1: [-1]}, r'must be in \[0, 2\)'),
    ({1: [0], 2: [0, 1]}, 'must not be duplicated'),
])
async def test_write_tones_validates_ch_map(tones_crs, ch_map, match):
    """Test write_tones validates user ch_map before clearing channels."""
    crs, mock_modules = tones_crs
    crs._clear_channels = AsyncMock()

    with pytest.raises(ValueError, match = match):
        await crs.write_tones([4.0e9, 4.1e9], [-50., -50.], ch_map = ch_map)
    crs._clear_channels.assert_not_called()
    mock_modules._write_tones.assert_not_called()


@pytest.mark.asyncio
async def test_write_tones_too_many_channels_per_module(tones_crs):
    """Test write_tones refuses more than 1024 channels on one module."""
    crs, _ = tones_crs
    fres = np.linspace(3.9e9, 4.1e9, 1025)

    with pytest.raises(ValueError, match = 'maximum is 1024'):
        await crs.write_tones(fres, np.full(1025, -60.))


@pytest.mark.asyncio
async def test_write_tones_unconfigured_module_raises(tones_crs):
    """Test write_tones raises if a module has no module settings."""
    crs, _ = tones_crs
    del crs.module_cfg[1]

    with pytest.raises(RuntimeError, match = 'has not been configured'):
        await crs.write_tones([4.0e9], [-50.])


################################################################################
#################################### sweep #####################################
################################################################################

def populate_sweep_outputs(
    nco_freqs, fres_map, ares_map, sweep_f, sweep_z, **kwargs
):
    """
    Fill sweep outputs as the _sweep macro would.

    Parameters:
    nco_freqs, fres_map, ares_map, sweep_f, sweep_z, kwargs: see _sweep.

    Returns:
    None
    """
    for mi, freqs in fres_map.items():
        if len(freqs):
            sweep_f[mi] = freqs
            sweep_z[mi] = np.ones(freqs.shape, dtype = complex)


@pytest.mark.asyncio
async def test_sweep_missing_message_with_user_ch_map(tones_crs):
    """Test the missing-channel message when ch_map is provided."""
    crs, _ = tones_crs
    with pytest.raises(ValueError, match = 'Missing channels detected'):
        await crs.sweep([[4.0e9, 4.1e9], [4.0e9, 4.1e9]], [-50., -50.], 10,
                        ch_map = {1: [0]}, verbose = False)


@pytest.mark.asyncio
async def test_sweep_missing_message_without_ch_map(tones_crs):
    """Test the missing-channel message when ch_map is generated."""
    crs, _ = tones_crs
    with pytest.raises(ValueError, match = 'Tones must be within'):
        await crs.sweep([[4.0e9, 4.1e9], [7.0e9, 7.1e9]], [-50., -50.], 10,
                        verbose = False)


@pytest.mark.asyncio
@pytest.mark.parametrize("global_control, expected", [(True, None),
                                                      (False, 6)])
async def test_sweep_passes_dec_stage_check(
    tones_crs, global_control, expected
):
    """Test sessions without global control check the stage while sweeping."""
    crs, mock_modules = tones_crs
    crs.global_control = global_control
    crs.set_decimation = AsyncMock()
    mock_modules._sweep.side_effect = populate_sweep_outputs

    f, z = await crs.sweep([[4.0e9, 4.1e9]], [-50.], 10, verbose = False)

    kwargs = mock_modules._sweep.call_args.kwargs
    assert kwargs['check_dec_stage'] == expected
    assert kwargs['full_scale_dbm'] == {mi: 7.0 for mi in range(1, 9)}
    assert crs.ntones == 1
    assert f.shape == z.shape == (1, 2)


################################################################################
################################# _sweep macro #################################
################################################################################

def make_sweep_module(dec_stage = 6, nchannels = 1024):
    """
    Create a mock ReadoutModule for the _sweep macro.

    Parameters:
    dec_stage (int): value returned by ctx.get_decimation.
    nchannels (int): number of channels returned by get_samples.

    Returns:
    module (MagicMock): mock readout module.
    ctx (Mock): mock tuber context.
    """
    import rfmux
    module = MagicMock(spec = rfmux.ReadoutModule)
    module.module = 1
    module._write_tones = AsyncMock()
    d = Mock()
    module.crs = d
    d.clear_channels = AsyncMock()

    ctx = Mock()
    future = Mock()
    future.result.return_value = dec_stage
    ctx.get_decimation.return_value = future
    ctx.side_effect = None

    async def call():
        return []
    ctx.return_value = None
    ctx_call = AsyncMock(side_effect = call)

    class Context:
        async def __aenter__(self):
            return ctx_obj

        async def __aexit__(self, *args):
            return None

    class CtxObj:
        set_frequency = ctx.set_frequency
        get_decimation = ctx.get_decimation

        def __call__(self):
            return ctx_call()
    ctx_obj = CtxObj()
    d.tuber_context = Mock(side_effect = lambda: Context())

    samples = Mock()
    samples.mean.i = np.ones(nchannels)
    samples.mean.q = np.zeros(nchannels)
    d.get_samples = AsyncMock(return_value = samples)
    return module, ctx


@pytest.mark.asyncio
async def test_sweep_macro_checks_dec_stage():
    """Test the _sweep macro raises if the stage changes mid-sweep."""
    module, ctx = make_sweep_module(dec_stage = 3)
    sweep_f, sweep_z = {}, {}

    with pytest.raises(RuntimeError, match = 'decimation stage changed'):
        await _sweep(module, {1: 4.0e9}, {1: np.array([[4.0e9, 4.1e9]])},
                     {1: np.array([-50.])}, sweep_f, sweep_z, nsamps = 5,
                     verbose = False, full_scale_dbm = {1: 7.0},
                     check_dec_stage = 6)
    module.crs.clear_channels.assert_called()
    module.crs.get_samples.assert_not_called()
    assert sweep_f == {}


@pytest.mark.asyncio
async def test_sweep_macro_no_dec_check_by_default():
    """Test the _sweep macro does not read the stage by default."""
    module, ctx = make_sweep_module(dec_stage = 3)
    sweep_f, sweep_z = {}, {}

    await _sweep(module, {1: 4.0e9}, {1: np.array([[4.0e9, 4.1e9]])},
                 {1: np.array([-50.])}, sweep_f, sweep_z, nsamps = 5,
                 verbose = False, full_scale_dbm = {1: 7.0})

    ctx.get_decimation.assert_not_called()
    assert sweep_z[1].shape == (1, 2)
    module._write_tones.assert_called_once()
    assert module._write_tones.call_args.args[3] == {1: 7.0}


@pytest.mark.asyncio
async def test_sweep_macro_short_packets_raises():
    """Test the _sweep macro raises if get_samples returns too few channels."""
    module, _ = make_sweep_module(nchannels = 128)
    freqs = np.linspace(3.9e9, 4.1e9, 200)[:, np.newaxis] * np.ones((1, 2))

    with pytest.raises(RuntimeError, match = 'short packets'):
        await _sweep(module, {1: 4.0e9}, {1: freqs},
                     {1: np.full(200, -60.)}, {}, {}, nsamps = 5,
                     verbose = False, full_scale_dbm = {1: 7.0})
    module.crs.clear_channels.assert_called()


@pytest.mark.asyncio
async def test_sweep_macro_requires_full_scale():
    """Test the _sweep macro requires the full scale map."""
    module, _ = make_sweep_module()
    with pytest.raises(TypeError, match = 'full_scale_dbm'):
        await _sweep(module, {1: 4.0e9}, {1: np.array([[4.0e9]])},
                     {1: np.array([-50.])}, {}, {}, nsamps = 5,
                     verbose = False)


################################################################################
################################# capture_ts ###################################
################################################################################

@pytest.mark.asyncio
async def test_capture_ts_saves_written_and_requested_fres(
    configured_crs, tmp_path
):
    """Test capture_ts saves the dithered and requested frequencies."""
    crs = configured_crs
    fres = np.array([4.0e9, 4.1e9, 9.0e9])
    written = np.array([4.0e9 + 12, 4.1e9 - 7, np.nan])
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock(return_value = written)
    crs.stream = AsyncMock()
    grp = zarr.open_group(tmp_path / 'data.zarr', mode = 'w')

    with patch('citkid.crs.instrument.util.write_system_cfg_to_zarr'), \
         patch('citkid.crs.instrument.time.sleep'):
        await crs.capture_ts(fres, [-50.] * 3, 1.0, 6, grp, verbose = False,
                             allow_missing = True,
                             tmp_directory = str(tmp_path / 'tmp'))

    np.testing.assert_array_equal(grp['fres'][:], written)
    np.testing.assert_array_equal(grp['fres_requested'][:], fres)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ['fres', 'fres_requested', 'ares',
                                  'crs_config'])
async def test_capture_ts_conflicts_checked_before_writing(
    configured_crs, tmp_path, name
):
    """Test capture_ts checks grp conflicts before touching the board."""
    crs = configured_crs
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    grp = zarr.open_group(tmp_path / 'data.zarr', mode = 'w')
    if name == 'crs_config':
        grp.require_group(name)
    else:
        grp.create_array(name = name, data = np.array([1.0]))

    with pytest.raises(ValueError, match = 'already contains required names'):
        await crs.capture_ts([4.0e9], [-50.], 1.0, 6, grp, verbose = False,
                             tmp_directory = str(tmp_path / 'tmp'))
    crs._clear_channels.assert_not_called()
    crs.write_tones.assert_not_called()


@pytest.mark.asyncio
async def test_capture_ts_clears_if_config_write_fails(
    configured_crs, tmp_path
):
    """Test capture_ts clears tones if saving the configuration fails."""
    crs = configured_crs
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock(return_value = np.array([4.0e9]))
    crs.stream = AsyncMock()
    grp = zarr.open_group(tmp_path / 'data.zarr', mode = 'w')

    with patch('citkid.crs.instrument.util.write_system_cfg_to_zarr',
               side_effect = ValueError('cfg failed')), \
         patch('citkid.crs.instrument.time.sleep'):
        with pytest.raises(ValueError, match = 'cfg failed'):
            await crs.capture_ts([4.0e9], [-50.], 1.0, 6, grp,
                                 verbose = False,
                                 tmp_directory = str(tmp_path / 'tmp'))
    assert crs._clear_channels.call_count == 2
    crs.stream.assert_not_called()


################################################################################
################################### stream #####################################
################################################################################

@pytest.mark.asyncio
@pytest.mark.parametrize("module_idxs, dec_stage, expected", [
    ([1, 2], 3, [3, 6]),
    ([1, 2], 6, [6]),
    (None, 3, [3]),
])
async def test_stream_restores_shared_decimation(
    base_crs, tmp_path, module_idxs, dec_stage, expected
):
    """Test shared sessions restore stage 6 after streaming."""
    crs = base_crs
    if module_idxs is not None:
        crs.module_idxs = module_idxs
        crs.shared = True
    crs.analog_bank_high = False
    crs.fres_map = {1: np.array([4.0e9])}
    crs.ares_map = {1: np.array([-50.])}
    crs.ch_map = {1: np.array([0])}
    crs.ntones = 1

    async def set_dec(stage, verbose = True):
        crs.sample_freq = 625e6 / (256 * 64 * 2 ** stage)
    crs.set_decimation = AsyncMock(side_effect = set_dec)
    grp = zarr.open_group(tmp_path / 'data.zarr', mode = 'w')

    with patch('citkid.crs.instrument.util.write_acq_cfg_to_zarr'), \
         patch('rfmux.tools.parser', create = True) as mock_parser, \
         patch('citkid.crs.instrument.util.parser_to_zarr') as mock_p2z, \
         patch('citkid.crs.instrument.shutil.rmtree'):
        mock_parser.main = MagicMock(side_effect = SystemExit(0))
        await crs.stream(1.0, dec_stage, grp, verbose = False,
                         tmp_directory = str(tmp_path / 'tmp'))

    assert [c.args[0] for c in crs.set_decimation.call_args_list] == expected
    # dt is from the streaming stage, not the restored stage
    dt = mock_p2z.call_args.args[7]
    assert dt == pytest.approx(256 * 64 * 2 ** dec_stage / 625e6)


################################################################################
############################### target_sweep ###################################
################################################################################

@pytest.mark.parametrize("existing", ['ares', 'qres', 'res_idxs', 'nsamps'])
def test_target_sweep_validates_other_conflicts(existing):
    """Test target_sweep checks every name it writes."""
    crs = Mock()
    crs.__class__.__name__ = 'DummyCRS'
    grp = zarr.group()
    if existing == 'nsamps':
        grp.attrs['nsamps'] = 1
    else:
        grp.create_array(name = existing, data = np.array([1.0]))

    with pytest.raises(ValueError, match = existing):
        _validate_target_sweep_inputs(
            crs, np.array([4e9]), np.array([-50.]), np.array([1e4]),
            np.array([0]), grp, 10, 500, 50, None, 100, 'spacing', True
            )
