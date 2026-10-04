import os
import asyncio
import shutil
import warnings
import time
import zarr
import numpy as np
from tqdm.auto import tqdm

# citkid imports
from ..util import run_with_time_bar
from . import util
from .. import __version__ as citkid_version

# rfmux imports
import rfmux
from rfmux.algorithms.measurement import take_netanal

ALL_MODULE_IDXS = tuple(range(1, 9))
ADC_CALIBRATION_MODES = ('AUTO', 'MODE1', 'MODE2')
STREAM_METHODS = ('parser', 'memory')

class CRS:
    def __init__(
        self, serial_number, interface, module_idxs = None,
        global_control = True
    ):
        """
        Initialize the CRS object.

        Note that the system must be configured using ``CRS.configure_system``
        (or attached to using ``CRS.attach``) before measurements.

        Several python sessions may share one CRS, each using different
        modules. In that case, one session has global control and calls
        ``configure_system``. The other sessions use
        ``global_control = False`` and call ``attach``. Each session is given
        the modules it owns with ``module_idxs``, and the module lists must
        not overlap. See ``CRS.set_decimation`` for how streaming is shared.

        Parameters:
        serial_number (int): CRS serial number e.g., 27.
        interface (str): Ethernet interface identifier e.g., 'enp2s0'.
        module_idxs (array-like int or None): modules that this session may
            write NCO frequencies, tones, and module settings to, and clear
            channels on. If None (default), all modules (1-8) are allowed,
            which is appropriate when a single session uses the CRS.
        global_control (bool): If True (default), this session may change
            board-wide settings (clock source, timestamp port, analog bank,
            extended bandwidth, and decimation). If False, this session only
            reads them and raises an error when they do not match what it
            needs.

        Returns:
        None
        """
        # Input validation
        if rfmux.__version__ != '1.4.1':
            raise RuntimeError('rfmux version 1.4.1 is required')
        if not isinstance(serial_number, int):
            raise TypeError('serial_number must be an integer')
        if not util.interface_exists(interface):
            raise ValueError(f'interface {interface} does not exist')
        if not isinstance(global_control, bool):
            raise TypeError('global_control must be a boolean value')
        module_idxs = _normalize_module_idxs(
            ALL_MODULE_IDXS if module_idxs is None else module_idxs,
            'module_idxs'
            )
        if not len(module_idxs):
            raise ValueError('module_idxs must contain at least one module')

        # Store inputs
        self.serial_number = serial_number
        self.interface = interface
        self.module_idxs = module_idxs
        self.global_control = global_control
        # A session is shared if other sessions may use the remaining modules
        self.shared = (module_idxs != list(ALL_MODULE_IDXS)) or \
                      (not global_control)
        self.rfmux_version = '1.4.1'
        self.citkid_version = citkid_version
        self.nco_freqs = {}

        # Initialize CRS object
        session_str = (
            '!HardwareMap [ !CRS { '
            + f'serial: "{serial_number:04d}"'
        )
        session_str += ' } ]'
        s = rfmux.load_session(session_str)
        self.d = s.query(rfmux.CRS).one()

        # System configuration. Set by configure_system or attach
        self.firmware_release = None
        self.clock_source = None
        self.analog_bank_high = None
        self.extended_bw = None
        # Set module bw, in case self.extended_bw is not called
        self.bw = 500e6
        # Per-module settings, set by configure_modules
        self.module_cfg = {}

        # Attributes to keep track of tone frequency, amplitude,
        # and channel mappings
        self.fres_map = {}
        self.ares_map = {}
        self.ch_map = {}
        # Number of channels indexed by self.ch_map, including missing ones
        self.ntones = 0

        # Attributes that keep track of decimation
        self.dec_stage = None
        self.dec_short = None
        self.dec_module_idxs = None
        # Suppresses set_decimation printouts. Updated by configure_system
        self.suppress_dec_printout = True

    @property
    def bank_module_idxs(self):
        """
        Return the module indices of the active analog bank.

        Returns:
        list of int: [5, 6, 7, 8] if the high bank is active, else
            [1, 2, 3, 4].
        """
        return list(range(5, 9)) if self.analog_bank_high else list(range(1, 5))

    @property
    def active_module_idxs(self):
        """
        Return the modules this session owns in the active analog bank.

        Returns:
        list of int: modules in both ``self.module_idxs`` and
            ``self.bank_module_idxs``.
        """
        return [mi for mi in self.module_idxs if mi in self.bank_module_idxs]

    async def configure_system(
        self, clock_source = "VCXO", full_scale_dbm = 7,
        analog_bank_high = False, verbose = True, suppress_dec_printout = True
    ):
        """
        Resolve the system, validate the CRS firmware version, set the timestamp
        port, clock source, extended bandwidth, analog bank, module settings,
        and decimation. Requires ``global_control``.

        Module settings other than the DAC full scale are reset to their rfmux
        defaults on this session's modules (see ``CRS.configure_modules``).
        Decimation is set to stage 6 with long packets, streaming every module
        in the analog bank.

        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal
            voltage controlled crystal oscillator or 'SMA' for the external 10
            MHz reference (reference should be 5 Vpp). Note that the clock
            source will default to 'VCXO' if the specified source is
            unavailable, and a warning will be raised.
        full_scale_dbm (int): full scale power in dBm. Range is [0, 7].
        analog_bank_high (bool): if True, uses modules 5-8 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4). Can be changed later using
            self.set_analog_bank.
        verbose (bool): If True, gets and prints the clocking source.
        suppress_dec_printout (bool): If True (default), suppress the
            decimation printout of every subsequent call to
            self.set_decimation, even if verbose is True. Stored as
            ``self.suppress_dec_printout``, which defaults to True before
            configure_system is called.

        Returns:
        None
        """
        # Input validation
        self._require_global_control('configure_system')
        _validate_configure_system_params(clock_source, full_scale_dbm,
                                          analog_bank_high, verbose,
                                          suppress_dec_printout)

        self.suppress_dec_printout = suppress_dec_printout

        # Resolve the system and validate firmware version
        await self.d.resolve()
        await self._check_firmware()

        # Set the timestamp port. Bypass if already set
        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted:
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)

        # Set the clock source
        await self.set_clock_source(clock_source, verbose = verbose)

        # Default extended bandwidth to False
        await self.set_extended_bw(False)

        # Set the analog bank and module settings
        await self.set_analog_bank(analog_bank_high, full_scale_dbm)

        # Set the decimation, streaming every module in the bank
        await self.set_decimation(6, short = False,
                                  module_idxs = self.bank_module_idxs,
                                  verbose = verbose)

        # Store number of tones
        self.ntones = 0

        # Print configuration if verbose.
        if verbose:
            print('System configured')

    async def attach(self, full_scale_dbm = 7, verbose = True):
        """
        Connect to a CRS that was configured by another session, without
        changing board-wide settings. Resolve the system, validate the
        firmware version, read the clock source, analog bank, extended
        bandwidth, and decimation stage, and configure this session's modules
        (see ``CRS.configure_modules``).

        Parameters:
        full_scale_dbm (int): full scale power in dBm for this session's
            modules. Range is [0, 7].
        verbose (bool): If True, prints the board configuration.

        Returns:
        None
        """
        # Input validation
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be a boolean value')

        # Resolve the system and validate firmware version
        await self.d.resolve()
        await self._check_firmware()

        # Read board-wide settings
        self.clock_source = await self.d.get_clock_source()
        self.analog_bank_high = bool(await self.d.get_analog_bank())
        self.extended_bw = bool(await self.d.get_extended_module_bandwidth())
        self.bw = 600e6 if self.extended_bw else 500e6
        dec_stage = await self.d.get_decimation()
        self.dec_stage = dec_stage
        self.sample_freq = None if dec_stage is None else \
            util.get_sample_freq(int(dec_stage))
        # Short mode and the streamed modules can't be read from the board
        self.dec_short = None
        self.dec_module_idxs = None

        # Forget modules outside of the active bank
        other_bank = [mi for mi in ALL_MODULE_IDXS
                      if mi not in self.bank_module_idxs]
        self._forget_modules(other_bank, forget_ncos = True)
        for mi in other_bank:
            self.module_cfg.pop(mi, None)
        if not self.active_module_idxs:
            raise RuntimeError(
                f'None of this session\'s modules {self.module_idxs} are in '
                f'the active analog bank {self.bank_module_idxs}'
                )

        # Configure this session's modules
        await self.configure_modules(full_scale_dbm = full_scale_dbm)
        self.ntones = 0

        if verbose:
            print(f'Attached: clock source = {self.clock_source}, '
                  f'analog bank high = {self.analog_bank_high}, '
                  f'extended bw = {self.extended_bw}, '
                  f'decimation stage = {dec_stage}, '
                  f'modules = {self.active_module_idxs}')

    async def _check_firmware(self):
        """
        Read the firmware release into ``self.firmware_release`` and raise an
        error if it is not the required version.

        Returns:
        None

        Raises:
        RuntimeError: if the firmware version is not the required version.
        """
        self.firmware_release = await self.d.get_firmware_release()
        req_version = '1.6.0'
        if self.firmware_release.version != req_version:
            raise RuntimeError(f"CRS firmware must be version {req_version}")

    async def set_clock_source(self, clock_source, verbose = True):
        """
        Set the clock source to 'VCXO' or 'SMA'. Requires ``global_control``.

        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal
            voltage controlled crystal oscillator or 'SMA' for the external 10
            MHz reference (reference should be 5 Vpp).
        verbose (bool): If True, prints the clock source after confirming.

        Returns:
        None
        """
        # Input validation
        self._require_global_control('set_clock_source')
        _validate_configure_system_params(clock_source, 1, True, True)

        # Set and check clock source
        await self.d.set_clock_source(clock_source)
        self.clock_source = await self.d.get_clock_source()

        # Raise warning if requested clock source is unavailable
        if self.clock_source != clock_source:
            warnings.warn(
                f"Requested clock source {clock_source} unavailable. "
                + f"Using {self.clock_source} instead.",
                UserWarning
            )

        # Print clock source if verbose
        if verbose:
            print(f'Clock source set: {self.clock_source}')

    async def set_analog_bank(self, analog_bank_high, full_scale_dbm):
        """
        Set the analog bank to high (modules 5-8) or low (modules 1-4), and
        configure this session's modules in the new bank with
        ``CRS.configure_modules``, using rfmux defaults for every module
        setting except the DAC full scale. Requires ``global_control``.

        Parameters:
        analog_bank_high (bool): if True, uses modules 5-8 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4).
        full_scale_dbm (int): full scale power in dBm. Range is [0, 7].

        Returns:
        None
        """
        # Input validation
        self._require_global_control('set_analog_bank')
        _validate_configure_system_params("SMA", full_scale_dbm,
                                         analog_bank_high, True)
        bank = range(5, 9) if analog_bank_high else range(1, 5)
        if not any(mi in bank for mi in self.module_idxs):
            raise ValueError(
                f'None of this session\'s modules {self.module_idxs} are in '
                f'the requested analog bank {list(bank)}'
                )

        # Set analog bank
        await self.d.set_analog_bank(high = analog_bank_high)
        abh = await self.d.get_analog_bank()
        if abh != analog_bank_high:
            raise RuntimeError("Failed to set analog bank")

        # Remove other modules from nco_freqs, maps, and module settings
        other_bank = [mi for mi in ALL_MODULE_IDXS if mi not in bank]
        self._forget_modules(other_bank, forget_ncos = True)
        for mi in other_bank:
            self.module_cfg.pop(mi, None)

        # Store analog bank
        self.analog_bank_high = analog_bank_high

        # Configure this session's modules in the bank
        await self.configure_modules(full_scale_dbm = full_scale_dbm)

    async def configure_modules(
        self, full_scale_dbm = 7, adc_attenuation_db = 0, nyquist_zone = 1,
        cable_length_m = 0.0, adc_autocal = True, adc_calibration_mode = 'AUTO',
        module_idxs = None
    ):
        """
        Set, read back, and store the per-module analog settings. Every
        setting is written on every call, so settings that are not passed
        are reset to their defaults. Defaults are the rfmux defaults, except
        ``full_scale_dbm``. The read-back values are stored in
        ``self.module_cfg`` and saved with the system configuration.

        Parameters:
        full_scale_dbm (int or float): DAC full scale power in dBm. Range is
            [0, 7]. Default is 7.
        adc_attenuation_db (int): ADC attenuation in dB. Default 0.
        nyquist_zone (int): DAC Nyquist zone (mix mode). 1 (default, NRZ) or
            2 (RC, boosts output in the second Nyquist zone).
        cable_length_m (float): cable length in m used by the CRS to correct
            the phase for cable delay. Default 0.0 (no correction).
        adc_autocal (bool): If True (default), ADC calibration runs
            continuously.
        adc_calibration_mode (str): ADC calibration mode. Can be 'AUTO'
            (default), 'MODE1', or 'MODE2'.
        module_idxs (array-like int or None): modules to configure. If None
            (default), configures this session's modules in the active analog
            bank (``self.active_module_idxs``).

        Returns:
        None
        """
        # Input validation
        self._require_configured()
        _validate_configure_system_params("SMA", full_scale_dbm, True, True)
        if isinstance(adc_attenuation_db, bool) or \
           not isinstance(adc_attenuation_db, (int, np.integer)) or \
           adc_attenuation_db < 0:
            raise ValueError('adc_attenuation_db must be a non-negative int')
        if isinstance(nyquist_zone, bool) or nyquist_zone not in (1, 2):
            raise ValueError('nyquist_zone must be 1 or 2')
        if isinstance(cable_length_m, bool) or \
           not isinstance(cable_length_m, (int, float, np.integer, np.floating))\
           or cable_length_m < 0:
            raise ValueError('cable_length_m must be a non-negative number')
        if not isinstance(adc_autocal, bool):
            raise TypeError('adc_autocal must be a boolean value')
        if adc_calibration_mode not in ADC_CALIBRATION_MODES:
            raise ValueError(
                f'adc_calibration_mode must be one of {ADC_CALIBRATION_MODES}'
                )
        if module_idxs is None:
            module_idxs = self.active_module_idxs
        module_idxs = _normalize_module_idxs(module_idxs, 'module_idxs')
        self._check_owned_modules(module_idxs)
        not_in_bank = [mi for mi in module_idxs
                       if mi not in self.bank_module_idxs]
        if not_in_bank:
            raise ValueError(f'Modules {not_in_bank} are not in the active '
                             f'analog bank {self.bank_module_idxs}')
        adc_attenuation_db = int(adc_attenuation_db)
        cable_length_m = float(cable_length_m)

        # Write settings
        d = self.d
        coros = []
        for mi in module_idxs:
            coros += [
                d.set_dac_scale(full_scale_dbm, d.UNITS.DBM, mi),
                d.set_adc_attenuator(adc_attenuation_db, module = mi),
                d.set_nyquist_zone(nyquist_zone, module = mi),
                d.set_cable_length(cable_length_m, module = mi),
                d.set_adc_calibration_mode(adc_calibration_mode, module = mi),
                d.set_adc_autocal(adc_autocal, module = mi),
            ]
        await asyncio.gather(*coros)

        # Read settings back
        coros = []
        for mi in module_idxs:
            coros += [
                d.get_dac_scale(d.UNITS.DBM, mi),
                d.get_adc_attenuator(module = mi),
                d.get_nyquist_zone(module = mi),
                d.get_cable_length(module = mi),
                d.get_adc_calibration_mode(module = mi),
                d.get_adc_autocal(module = mi),
            ]
        results = await asyncio.gather(*coros)

        # Check and store settings
        for idx, mi in enumerate(module_idxs):
            dac, att, nz, cl, mode, autocal = results[6 * idx: 6 * idx + 6]
            if not np.isclose(dac, full_scale_dbm, atol = 0.1):
                raise RuntimeError(
                    f"Failed to set DAC full scale on module {mi}")
            if not np.isclose(att, adc_attenuation_db, atol = 0.5):
                raise RuntimeError(
                    f"Failed to set ADC attenuation on module {mi}")
            if nz != nyquist_zone:
                raise RuntimeError(
                    f"Failed to set Nyquist zone on module {mi}")
            if not np.isclose(cl, cable_length_m, atol = 1e-3):
                raise RuntimeError(
                    f"Failed to set cable length on module {mi}")
            if mode != adc_calibration_mode:
                raise RuntimeError(
                    f"Failed to set ADC calibration mode on module {mi}")
            if bool(autocal) != adc_autocal:
                raise RuntimeError(
                    f"Failed to set ADC autocal on module {mi}")
            self.module_cfg[mi] = {
                'full_scale_dbm': float(dac),
                'adc_attenuation_db': float(att),
                'nyquist_zone': int(nz),
                'cable_length_m': float(cl),
                'adc_calibration_mode': str(adc_calibration_mode),
                'adc_autocal': bool(autocal),
            }

    async def set_extended_bw(self, extended):
        """
        Choose between the standard (500 MHz) and extended (600 MHz) bandwidth.
        Requires ``global_control``.

        Only extend the bandwidth if you know what you are doing. See
        `crs.d.set_extended_module_bandwidth` for details.

        Parameters:
        extended (bool): If True, extends the bandwidth to 600 MHz. Else
            sets the bandwidth to 500 MHz.

        Returns:
        None
        """
        # Input validation
        self._require_global_control('set_extended_bw')
        if not isinstance(extended, bool):
            raise TypeError('extended must be a boolean value')

        # Set and check extended bandwidth
        await self.d.set_extended_module_bandwidth(extended)
        ex = await self.d.get_extended_module_bandwidth()
        if ex != extended:
            raise RuntimeError("Failed to set extended module bandwidth")

        # Store extended_bw and bw
        self.extended_bw = extended
        self.bw = 600e6 if extended else 500e6

        # Raise warning if extended bandwidth is set
        if extended:
            warnings.warn(f"Extended module bandwidth set", UserWarning)

    async def set_decimation(
        self, dec_stage, short = None, module_idxs = None, verbose = True
    ):
        """
        Set the decimation stage, with optional short mode and module indices.
        If short and/or module_idxs are not provided, they are determined
        automatically (see below). Raises an error if the configuration will
        drop packets.

        Decimation is a board-wide setting. How it is handled depends on the
        session:
            Single session (default): short and module_idxs are determined
                from self.fres_map.
            Shared session with global control: at stage 6 (the sweep
                stage), the defaults are long packets and every module in the
                analog bank, so that other sessions can keep sweeping. At
                other stages, the defaults are determined from self.fres_map
                and a warning is raised, because other sessions can't sweep
                until stage 6 is restored.
            Session without global control: nothing is written. Raises an
                error if the board's decimation stage is not dec_stage. short
                and module_idxs are ignored, because the board can't report
                them.

        Parameters:
        dec_stage (int): decimation stage (0-6). Approximate
            values are:
            6 ->    596 Hz
            5 ->  1,192 Hz
            4 ->  2,384 Hz
            3 ->  4,768 Hz
            2 ->  9,537 Hz
            1 -> 19,073 Hz
            0 -> 38,147 Hz
        short (bool or None): If True, enables short mode (max 128 tones per
            module). If False, uses the full number of tones. If None, short is
            determined automatically.
        module_idxs (array-like int or None): module indices to stream. If None
            or empty, module_idxs is determined automatically.
        verbose (bool): If True, prints the decimation settings after
            confirming, unless ``self.suppress_dec_printout`` is True.

        Returns:
        None
        """
        # Input validation
        if isinstance(dec_stage, bool) or \
           not isinstance(dec_stage, (int, np.integer)) or \
           dec_stage < 0 or dec_stage > 6:
            raise ValueError('dec_stage must be an integer between 0 and 6')
        dec_stage = int(dec_stage)
        if short is not None and not isinstance(short, bool):
            raise TypeError('short must be a boolean or None')
        if module_idxs is not None:
            if not isinstance(module_idxs, (list, np.ndarray)):
                raise TypeError('module_idxs must be a list of integers.')
            module_idxs = _normalize_module_idxs(module_idxs, 'module_idxs')

        # Without global control, check the board's decimation stage
        if not self.global_control:
            board_stage = await self.d.get_decimation()
            if board_stage != dec_stage:
                raise RuntimeError(
                    f'The CRS decimation stage is {board_stage}, but this '
                    f'session needs stage {dec_stage}. This session does not '
                    'have global control. Set the decimation from the session '
                    'with global control.'
                    )
            self.sample_freq = util.get_sample_freq(dec_stage)
            self.dec_stage = dec_stage
            self.dec_short = None
            self.dec_module_idxs = None
            if verbose and not self.suppress_dec_printout:
                print(f'Decimation checked: stage = {dec_stage}')
            return

        # Determine short and module_idxs if not provided
        shared_sweep_stage = self.shared and dec_stage == 6
        if self.shared and not shared_sweep_stage:
            warnings.warn(
                f'Decimation stage {dec_stage} set in a shared session. Other '
                'sessions cannot sweep until stage 6 is restored.',
                UserWarning
                )
        if short is None:
            if shared_sweep_stage:
                short = False
            elif self.fres_map:
                max_ntones = max([len(f) for f in self.fres_map.values()])
                short = max_ntones <= 128
            else:
                short = False
        if module_idxs is None or len(module_idxs) == 0:
            if shared_sweep_stage:
                module_idxs = self.bank_module_idxs
            else:
                module_idxs = sorted(
                    int(k) for k, v in self.fres_map.items() if len(v)
                    )

        # Set decimation
        await self.d.set_decimation(
            dec_stage, short = short, module = module_idxs
            ) # This will do nothing if the parameters have not changed
        self.sample_freq = util.get_sample_freq(dec_stage)
        changed = (self.dec_stage != dec_stage) or \
                  (self.dec_short != short) or \
                  (self.dec_module_idxs != module_idxs)
        if changed:
            time.sleep(0.1)
        self.dec_stage = dec_stage
        self.dec_short = short
        self.dec_module_idxs = module_idxs

        # Print decimation info
        if verbose and not self.suppress_dec_printout:
            msg = f'Decimation set: stage = {dec_stage}, short = {short}, '
            msg += f'modules = {module_idxs}'
            print(msg)

    async def set_nco(self, nco_freqs, verbose = True):
        """
        Set the NCO frequency.

        Parameters:
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz. Modules must be owned by this session
            and in the active analog bank.
        verbose (bool): If True, prints the NCO frequencies after confirming.

        Returns:
        None
        """
        # Input validation
        self._require_configured()
        _validate_nco_freqs(nco_freqs, self.analog_bank_high)
        self._check_owned_modules(list(nco_freqs.keys()))

        # Set NCO frequencies
        nco_freqs = nco_freqs.copy()
        modules = util.get_modules(self.d, list(nco_freqs.keys()))
        await modules._set_nco(nco_freqs)
        self.nco_freqs.update(nco_freqs)

        # Clear channels on modules with newly set NCO frequencies
        await self._clear_channels(list(nco_freqs.keys()))

        # Print NCO frequencies if verbose
        for module_idx, nco in nco_freqs.items():
            if verbose:
                nco_str = f'{round(nco * 1e-6, 6)}'
                print(f'Module {module_idx} NCO is {nco_str} MHz')

    async def disable_modules(self, module_idxs):
        """
        Disable modules by clearing channels and removing NCO frequencies.

        Parameters:
        module_idxs (list of int): module indices to disable. Must be owned by
            this session.

        Returns:
        None
        """
        # Input validation
        if not all(isinstance(mi, (int, np.integer)) for mi in module_idxs):
            raise TypeError('module_idxs must be a list of integers')
        self._check_owned_modules(module_idxs)

        # Clear channels
        await self._clear_channels(module_idxs)

        # Update self.nco_freqs
        self._forget_modules(module_idxs, forget_ncos = True)

    async def _clear_channels(self, module_idxs):
        """
        Clear all channels on the specified modules.

        Parameters:
        module_idxs (array-like int): module indices to clear channels on.
            Must be owned by this session.

        Returns:
        None
        """
        # Input validation
        if not all(isinstance(mi, (int, np.integer)) for mi in module_idxs):
            raise TypeError('module_idxs must be a list of integers')
        self._check_owned_modules(module_idxs)

        # Clear channels and update fres_map, ares_map, and ch_map
        for module_idx in module_idxs:
            await self.d.clear_channels(module = module_idx)
        self._forget_modules(module_idxs)

    def _forget_modules(self, module_idxs, forget_ncos = False):
        """
        Remove modules from ``self.fres_map``, ``self.ares_map``, and
        ``self.ch_map``, and optionally from ``self.nco_freqs``. Reset
        ``self.ntones`` to 0 if no channels remain in ``self.ch_map``. Does
        not communicate with the board.

        Parameters:
        module_idxs (array-like int): module indices to remove.
        forget_ncos (bool): If True, also removes the modules from
            ``self.nco_freqs``.

        Returns:
        None
        """
        for module_idx in module_idxs:
            self.fres_map.pop(module_idx, None)
            self.ares_map.pop(module_idx, None)
            self.ch_map.pop(module_idx, None)
            if forget_ncos:
                self.nco_freqs.pop(module_idx, None)
        if not any(len(chs) for chs in self.ch_map.values()):
            self.ntones = 0

    def _require_global_control(self, action):
        """
        Raise an error if this session does not have global control.

        Parameters:
        action (str): name of the action, used in the error message.

        Returns:
        None

        Raises:
        RuntimeError: if ``self.global_control`` is False.
        """
        if not self.global_control:
            raise RuntimeError(
                f'{action} changes board-wide settings, but this session does '
                'not have global control (global_control = False).'
                )

    def _require_configured(self):
        """
        Raise an error if neither ``configure_system`` nor ``attach`` has set
        the analog bank.

        Returns:
        None

        Raises:
        RuntimeError: if ``self.analog_bank_high`` is None.
        """
        if self.analog_bank_high is None:
            raise RuntimeError(
                'The CRS is not configured: call configure_system (or attach '
                'for a session without global control) first.'
                )

    def _check_owned_modules(self, module_idxs):
        """
        Raise an error if any module is not owned by this session.

        Parameters:
        module_idxs (array-like int): module indices to check.

        Returns:
        None

        Raises:
        ValueError: if any module is not in ``self.module_idxs``.
        """
        not_owned = [int(mi) for mi in module_idxs
                     if mi not in self.module_idxs]
        if not_owned:
            raise ValueError(
                f'Modules {not_owned} are not owned by this session. Allowed '
                f'modules are {self.module_idxs}.'
                )

    def _check_ch_map(self, ch_map, nchannels):
        """
        Validate a channel map against the set NCOs, configured modules, and
        number of channels.

        Parameters:
        ch_map (dict): keys (int) are module indices and values (np.array
            int32) are channel indices.
        nchannels (int): number of channels being written. Channel indices
            must be in [0, nchannels).

        Returns:
        None

        Raises:
        ValueError: if a module has no NCO frequency set, a module has more
            than 1024 channels, or channel indices are out of range or
            duplicated.
        RuntimeError: if a module with channels has not been configured.
        """
        for module_idx, chs in ch_map.items():
            if module_idx not in self.nco_freqs:
                raise ValueError(
                    f'ch_map includes module {module_idx}, which has no NCO '
                    'frequency set')
            if len(chs) and module_idx not in self.module_cfg:
                raise RuntimeError(
                    f'Module {module_idx} has not been configured: call '
                    'configure_system, attach, or configure_modules first.')
            if len(chs) > 1024:
                raise ValueError(
                    f'Module {module_idx} has {len(chs)} channels; the '
                    'maximum is 1024')
        all_chs = np.concatenate(
            [np.asarray(v, dtype = np.int64) for v in ch_map.values()]
            ) if ch_map else np.array([], dtype = np.int64)
        if np.any(all_chs < 0) or np.any(all_chs >= nchannels):
            raise ValueError(
                f'ch_map channel indices must be in [0, {nchannels})')
        if len(np.unique(all_chs)) != len(all_chs):
            raise ValueError('ch_map channel indices must not be duplicated')

    def _full_scale_map(self):
        """
        Return the DAC full scale of each configured module.

        Returns:
        dict: keys (int) are module indices and values (float) are DAC full
            scale powers in dBm.
        """
        return {mi: cfg['full_scale_dbm'] for mi, cfg in self.module_cfg.items()}

    async def write_tones(
        self, fres, ares, ch_map = None, allow_missing = False
    ):
        """
        Write tones for the provided frequencies and amplitudes.

        Tone frequencies are dithered by rfmux by up to about 50 Hz before
        they are written. The written frequencies are returned and stored in
        ``self.fres_map``.

        Parameters:
        fres (array-like): tone frequencies in Hz.
        ares (array-like): tone powers in dBm.
        ch_map (dict): keys (int) are module indices and values (array-like int)
            are channel indices to write tones to. If None, automatically maps
            tones to NCOs based on frequency.
        allow_missing (bool): If True, ignores tones that are outside the
            bandwidth of all NCOs. If False, raises an error if any tones are
            outside the bandwidth of all NCOs.

        Returns:
        fres_written (np.array): written (dithered) tone frequencies in Hz, in
            the order of fres. Missing tones are NaN.
        """
        # Input validation
        if not len(self.nco_freqs):
            raise RuntimeError("NCO frequencies are not set")
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)
        if fres.shape != ares.shape:
            raise ValueError('fres and ares must be the same shape')
        if len(fres) == 0:
            return np.array([], dtype = np.float64)
        ch_map = util._validate_ch_map(ch_map)

        # Store ntones for later
        ntones = len(fres)

        # Get ch_map
        if ch_map is None:
            ch_map, missing_chs = util.create_ch_map(
                self.nco_freqs, fres, self.bw
                )
        else:
            missing_chs = []
            for ch in range(len(fres)):
                if not any(ch in ch_list for ch_list in ch_map.values()):
                    missing_chs.append(ch)
        self._check_ch_map(ch_map, ntones)

        # Handle missing channels
        if len(missing_chs):
            msg = (f"Tones must be within {self.bw / 2e6:.0f} "
                   "MHz of an NCO frequency.")
            if allow_missing:
                msg += f" Ignoring {len(missing_chs)} tone(s)."
                warnings.warn(msg, UserWarning)
            else:
                raise ValueError(msg)

        # Clear existing channels first
        await self._clear_channels(list(self.fres_map.keys()))

        # Split fres and ares into dictionaries
        self.fres_map.update({key: fres[val] for key, val in ch_map.items()})
        self.ares_map.update({key: ares[val] for key, val in ch_map.items()})
        self.ch_map.update(ch_map)

        # Dither frequencies
        for key, fres_module in self.fres_map.items():
            # Dither separately for each NCO
            if not len(fres_module):
                continue
            self.fres_map[key] = take_netanal._safe_concatenate_frequencies(
                fres_module, self.nco_freqs[key]
                )

        # Write tones
        modules = util.get_modules(self.d, list(self.fres_map.keys()))
        await modules._write_tones(self.nco_freqs, self.fres_map,
                                   self.ares_map, self._full_scale_map())
        self.ntones = ntones

        # Return written frequencies in the order of fres
        fres_written = np.full(ntones, np.nan, dtype = np.float64)
        for key, chs in self.ch_map.items():
            fres_written[chs] = self.fres_map[key]
        return fres_written

    async def sweep(
        self, frequencies, ares, nsamps, ch_map = None, allow_missing = False,
        dec_grp = None, verbose = True, pbar_description = 'Sweeping'
    ):
        """
        Perform a frequency sweep and return complex S21 at each frequency.

        Parameters:
        frequencies (M X N array-like float): the first index M is the channel
            index (max len 1024) and the second index N is the frequency in Hz
            for a single point in the sweep.
        ares (M array-like float): amplitudes in dBm for each channel.
        nsamps (int): number of samples to average per point.
        ch_map (dict): keys (int) are module indices and values (array-like int)
            are channel indices to write tones to. If None, automatically maps
            tones to NCOs based on frequency.
        allow_missing (bool): If True, ignores tones that are outside the
            allowed frequency range and inserts NaNs in the output. If False,
            raises an error if any tones are outside the allowed frequency
            range.
        dec_grp (zarr.Group or None): if provided, writes decimation settings
            to this zarr group.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        f (M X N np.array): swept (dithered) frequencies in Hz. Missing
            channels are NaN.
        z (M X N np.array): complex S21 data in dBc corresponding to f.
            Missing channels are NaN.
        """
        # Input validation
        frequencies = np.asarray(frequencies, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)
        if frequencies.shape[0] != ares.shape[0]:
            raise ValueError('frequencies and ares must have the same length '
                             'along axis 0')
        if frequencies.ndim != 2:
            raise ValueError('frequencies must be a 2D array-like object')
        if not isinstance(nsamps, int) or nsamps <= 0:
            raise ValueError('nsamps must be a positive integer')
        if not len(self.nco_freqs):
            raise RuntimeError("NCO frequencies are not set")
        ch_map = util._validate_ch_map(ch_map)
        if not isinstance(pbar_description, str):
            raise TypeError('pbar_description must be a string')
        if dec_grp is not None and not isinstance(
            dec_grp, zarr.core.group.Group
            ):
            raise TypeError('dec_grp must be a zarr.Group or None')

        ### Map frequencies to modules/channels
        # Get ch_map
        user_ch_map = ch_map is not None
        if not user_ch_map:
            ch_map, missing_chs = util.create_ch_map(
                self.nco_freqs, frequencies, self.bw
                )
        else:
            missing_chs = []
            for ch in range(len(frequencies)):
                if not any(ch in ch_list for ch_list in ch_map.values()):
                    missing_chs.append(ch)
        self._check_ch_map(ch_map, len(frequencies))

        # Handle missing channels
        if len(missing_chs):
            if user_ch_map:
                msg = "Missing channels detected in ch_map."
            else:
                msg = (f"Tones must be within {self.bw / 2e6:.0f} "
                    "MHz of an NCO frequency.")
            if allow_missing:
                msg += f" Proceeding with {len(missing_chs)} "
                msg += "missing channel(s)."
                warnings.warn(msg, UserWarning)
            else:
                raise ValueError(msg)

        # Clear existing channels first
        await self._clear_channels(list(self.fres_map.keys()))

        # Update fres_map, ares_map, ch_map
        self.fres_map.update({key: frequencies[val]
                              for key, val in ch_map.items()})
        self.ares_map.update({key: ares[val]
                              for key, val in ch_map.items()})
        self.ch_map.update(ch_map)
        self.ntones = len(frequencies)

        # Dither frequencies
        for key, freqs in self.fres_map.items():
            if not len(freqs):
                continue
            # Each point is the sweep is dithered across the NCO
            for idx, freq in enumerate(freqs.T):
                freq_dithered = take_netanal._safe_concatenate_frequencies(
                    freq, self.nco_freqs[key]
                    )
                self.fres_map[key][:, idx] = freq_dithered

        ### Set dec_stage to 6 for sweeping, save if dec_grp is provided
        await self.set_decimation(6, verbose = verbose)
        if dec_grp is not None:
            util.write_acq_cfg_to_zarr(self, dec_grp)

        ### Sweep
        # Without global control, check that the decimation stage stays at 6
        check_dec_stage = None if self.global_control else 6
        sweep_f, sweep_z = {}, {}
        modules = util.get_modules(self.d, list(self.fres_map.keys()))
        await modules._sweep(self.nco_freqs, self.fres_map,
                            self.ares_map, sweep_f, sweep_z, nsamps = nsamps,
                            verbose = verbose,
                            pbar_description = pbar_description,
                            full_scale_dbm = self._full_scale_map(),
                            check_dec_stage = check_dec_stage)

        # Clear ares_map after sweeping, since d.sweep clears channels
        for module_idx in ch_map.keys():
            self.fres_map[module_idx] = np.array([], dtype = np.float64)
            self.ares_map[module_idx] = np.array([], dtype = np.float64)

        ### Create f, z to fill with sweep results
        f = np.full(frequencies.shape, np.nan, dtype = float)
        z = np.full(frequencies.shape, np.nan + 1j * np.nan, dtype = complex)

        for module_idx, chs in ch_map.items():
            # Note: self.ch_map may modules that are not used here
            # ch_map only has modules used in this sweep
            chs = np.asarray(chs, dtype = np.int32)
            if chs.size == 0:
                continue
            f[chs, :] = sweep_f[module_idx]
            z[chs, :] = sweep_z[module_idx]

        # Convert to dBc and return
        z /= 10 ** (ares[:, np.newaxis] / 20)
        return f, z

    async def sweep_span(
        self, fres, ares, span, npoints, nsamps, ch_map = None,
        allow_missing = False, center_fres = True, downward = True, log = False,
        dec_grp = None, verbose = True, pbar_description = 'Sweeping'
    ):
        """
        Perform a frequency sweep where each channel is swept over the same
        frequency span, with either linear or logarithmic spacing.

        Parameters:
        fres (array-like): center frequencies in Hz.
        ares (array-like): amplitudes in dBm.
        span (float): span around each frequency to sweep in Hz.
        npoints (int): number of sweep points per channel.
        nsamps (int): number of samples to average per point.
        ch_map (dict): keys (int) are module indices and values (array-like int)
            are channel indices to write tones to. If None, automatically maps
            channels to modules.
        allow_missing (bool): If True, ignores tones that are outside the
            frequency range. If False, raises an error if any tones are
            outside the frequency range.
        center_fres (bool): If True, fres is the center of each band. Else,
            fres is the starting frequency.
        downward (bool): if True, sweeps from high to low frequency. Else,
            sweeps from low to high frequency.
        log (bool): If True, uses logarithmic spacing between points. Else,
            uses linear spacing.
        dec_grp (zarr.Group or None): if provided, writes decimation settings
            to this zarr group.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.


        Returns:
        f (M X N np.array): array of frequencies where M is the channel index
            and N is the index of each point in the sweep
        z (M X N np.array): array of complex S21 data corresponding to f
        """
        # Input validation
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)
        if not isinstance(span, (float, np.floating)) or span <= 0:
            raise ValueError('span must be a positive float')
        if not isinstance(npoints, int) or npoints <= 0:
            raise ValueError('npoints must be a positive integer')
        if dec_grp is not None and not isinstance(
            dec_grp, zarr.core.group.Group
            ):
            raise TypeError('dec_grp must be a zarr.Group or None')
        # other validation is performed in self.sweep

        # Create freqs array
        func = np.geomspace if log else np.linspace
        if center_fres:
            if downward:
                freqs = func(fres + span / 2, fres - span / 2, npoints).T
            else:
                freqs = func(fres - span / 2, fres + span / 2, npoints).T
        else:
            if downward:
                freqs = func(fres + span, fres, npoints).T
            else:
                freqs = func(fres, fres + span, npoints).T

        # Sweep and return
        f, z = await self.sweep(
            freqs, ares, nsamps, ch_map = ch_map, allow_missing = allow_missing,
            dec_grp = dec_grp, verbose = verbose,
            pbar_description = pbar_description
            )
        return f, z

    async def sweep_qres(
        self, fres, ares, qres, npoints, nsamps, ch_map = None,
        allow_missing = False, dec_grp = None, verbose = True,
        pbar_description = 'Sweeping'
    ):
        """
        Perform a downward frequency sweep where the span around each frequency
        is set equal to fres / qres.

        Parameters:
        fres (array-like): center frequencies in Hz.
        ares (array-like): amplitudes in dBm.
        qres (array): sweep spans in Q-like form. Spans of each sweep are
            fres / qres.
        npoints (int): number of sweep points per channel.
        nsamps (int): number of samples to average per point.
        ch_map (dict): keys (int) are module indices and values (array-like int)
            are channel indices to write tones to. If None, automatically maps
            channels to modules.
        allow_missing (bool): If True, ignores tones that are outside the
            frequency range. If False, raises an error if any tones are
            outside the frequency range.
        dec_grp (zarr.Group or None): if provided, writes decimation settings
            to this zarr group.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index
            and N is the index of each point in the sweep.
        z (M X N np.array): array of complex S21 data corresponding to f.
        """
        # Input validation
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)
        qres = np.asarray(qres, dtype = np.float64)
        if not isinstance(npoints, int) or npoints <= 0:
            raise ValueError('npoints must be a positive integer')
        if dec_grp is not None and not isinstance(
            dec_grp, zarr.core.group.Group
            ):
            raise TypeError('dec_grp must be a zarr.Group or None')
        # other validation is performed in self.sweep

        # Create freqs array
        spans = fres / qres
        freqs = np.linspace(fres + spans / 2, fres - spans / 2, npoints).T

        # Sweep and return
        f, z = await self.sweep(
            freqs, ares, nsamps, ch_map = ch_map, allow_missing = allow_missing,
            dec_grp = dec_grp, verbose = verbose,
            pbar_description = pbar_description
            )
        return f, z

    async def sweep_full(
        self, amplitude, npoints_per_tone, nsamps, log = False, downward = True,
        dec_grp = None, verbose = True, pbar_description = 'Sweeping'
    ):
        """
        Perform a frequency sweep over the full bandwidth around the NCO
        frequency.

        Parameters:
        amplitude (float): amplitude in dBm.
        npoints_per_tone (int): number of sweep points per tone.
        nsamps (int): number of samples to average per point.
        log (bool): If True, uses logarithmic spacing between points. Else,
            uses linear spacing.
        downward (bool): If True, sweeps frequencies in descending order.
        dec_grp (zarr.Group or None): if provided, writes decimation settings
            to this zarr group.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        f (np.array): array of frequencies in Hz.
        z (np.array): array of complex S21 data corresponding to f.
        """
        # Input validation
        amplitude = float(amplitude)
        if not isinstance(npoints_per_tone, int) or npoints_per_tone <= 0:
            raise ValueError('npoints must be a positive integer')
        if dec_grp is not None and not isinstance(
            dec_grp, zarr.core.group.Group
            ):
            raise TypeError('dec_grp must be a zarr.Group or None')
        # other validation is performed in self.sweep_linear

        # Create fres and ares arrays
        ncos = self.nco_freqs.values()
        # spacing = tone_bw / npoints
        func = np.geomspace if log else np.linspace
        # Add small margin to avoid numerical errors pushing freqs outside bw
        margin = self.bw * 1e-9  # 0.000001% margin
        frequencies = np.concatenate([
            func(
                nco - self.bw / 2 + margin,
                nco + self.bw / 2 - margin,
                1024 * npoints_per_tone
            )
            for nco in ncos])
        frequencies = np.array(np.split(frequencies, 1024 * len(ncos)))
        if downward:
            frequencies = np.flip(frequencies, axis = 1)
        ch_map = {idx: range(i * 1024, (i + 1) * 1024)
                  for i, idx in enumerate(self.nco_freqs.keys())}
        ares = amplitude * np.ones(len(frequencies))

        # Sweep and return
        f, z = await self.sweep(
            frequencies=frequencies, ares=ares, nsamps=nsamps, ch_map=ch_map,
            allow_missing = False, dec_grp = dec_grp, verbose = verbose,
            pbar_description = pbar_description
        )

        # Flatten and sort
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_ts(
        self, fres, ares, ts_duration_s, dec_stage, grp, ch_map = None,
        allow_missing = False, tmp_directory = 'tmp/', batch_size_mb = 1000,
        chunk_size_mb = 128, delete_parser_data = True, verbose = True,
        method = 'parser'
    ):
        """
        Clear all tones on this session's modules, write tones using fres and
        ares, capture a timestream of length ts_time using the parser or
        directly into memory, and then clear all tones on this session's
        modules.

        Parameters:
        fres (array-like): tone frequencies in Hz.
        ares (array-like): tone amplitudes in dBm.
        ts_duration_s (float): timestream length in seconds.
        dec_stage (int): dec_stage frequency downsampling factor.
            See self.set_decimation for details.
        grp (zarr.Group): zarr group to save the batch data. The written
            (dithered) tone frequencies are saved as 'fres' (NaN for missing
            tones), and the requested frequencies as 'fres_requested'.
        ch_map (dict): keys (int) are module indices and values (array-like int)
            are channel indices to write tones to. If None, automatically maps
            tones to NCOs based on frequency.
        allow_missing (bool): If True, ignores tones that are outside the
            bandwidth of all NCOs. If False, raises an error if any tones are
            outside the bandwidth of all NCOs.
        tmp_directory (str): directory to save temporary parser data before
            converting. Data is streamed to disk, so the drive must have
            fast enough I/O performance with sufficient free space.
        batch_size_mb (float): batch size, in MB. Approximate size of each chunk
            that is loaded into memory when converting parser data to zarr.
        chunk_size_mb (float): size of each zarr chunk along the time axis,
            in MB.
        delete_parser_data (bool): If True, deletes the parser data files
            after importing the data.
        verbose (bool): If True, displays a progress bar while taking data.
        method (str): Can be 'parser' (default) or 'memory'. See
            ``CRS.stream``. For 'memory', the available memory is checked
            before tones are written, assuming the worst case of up to 1024
            tones on every module with an NCO set.

        Returns:
        None

        Raises:
        MemoryError: if method is 'memory' and the data would not fit in
            memory with headroom.
        """
        # Validate stream inputs early to fail fast before write_tones
        ts_duration_s, dec_stage, ch_map, allow_missing, tmp_directory, \
           _, batch_size_mb, chunk_size_mb, delete_parser_data, \
           verbose = _validate_stream_input(
            ts_duration_s, dec_stage, grp, ch_map, allow_missing, tmp_directory,
            batch_size_mb, chunk_size_mb,
            delete_parser_data, verbose, method
        )
        conflicts = {'fres', 'fres_requested', 'ares', 'crs_config'} & \
                    set(grp.keys())
        if conflicts:
            raise ValueError(
                f'grp already contains required names: {sorted(conflicts)}')
        self._require_configured()
        if method == 'memory':
            # Worst case before the tones are mapped to modules
            ntones = int(np.size(fres))
            util.check_memory_for_stream(
                max(len(self.nco_freqs), 1), min(ntones, 1024), ntones,
                int(util.get_sample_freq(dec_stage) * ts_duration_s),
                chunk_size_mb
                )
        # fres and ares validation is performed in self.write_tones

        # Clear all channels - ensures that unused modules are cleared
        idx_to_clear = self.active_module_idxs
        await self._clear_channels(idx_to_clear)

        try:
            # Write tones
            fres_written = await self.write_tones(
                fres, ares, ch_map = ch_map, allow_missing = allow_missing
                )

            # Write configuration to zarr
            cfg_grp = grp.require_group('crs_config')
            util.write_system_cfg_to_zarr(self, cfg_grp)

            # Save tones
            grp.create_array(name = 'fres', data = fres_written)
            grp.create_array(name = 'fres_requested',
                             data = np.asarray(fres, dtype = np.float64))
            grp.create_array(name = 'ares',
                             data = np.asarray(ares, dtype = np.float64))

            # Sleep briefly to ensure tones are settled
            time.sleep(0.5)

            # Stream
            await self.stream(
                ts_duration_s = ts_duration_s,
                dec_stage = dec_stage,
                grp = grp,
                tmp_directory = tmp_directory,
                batch_size_mb = batch_size_mb,
                chunk_size_mb = chunk_size_mb,
                delete_parser_data = delete_parser_data,
                verbose = verbose,
                method = method
            )
        finally:
            # Clear all channels after streaming (success or failure)
            await self._clear_channels(idx_to_clear)

    async def stream(
        self, ts_duration_s, dec_stage, grp, tmp_directory = 'tmp/',
        batch_size_mb = 1000, chunk_size_mb = 128, delete_parser_data = True,
        verbose = True, method = 'parser'
    ):
        """
        Capture a timestream using the parser or directly into memory. Does
        not change written tones - assumes fres_map and ares_map match the
        currently written tones.

        Both methods receive the same streamed packets and save the same
        format: 'counts_to_s21', 'dt', and the int32 array 'z' of shape
        (2, ntones, nsamples). The parser method stops when any module reaches
        ts_duration_s + 0.1 s of data. The memory method stops when every
        module has exactly ts_duration_s of data, and warns if packets were
        dropped.

        Note on streaming capabilities: If all modules have less than 129 tones,
        the system will stream 128 tones per module. Otherwise, it will stream
        1024 tones per module. Streaming will only be performed on modules with
        tones set (see self.fres_map). Setting the decimation stage will fail if
        the requested number of modules/tones will cause packet loss. For low
        decimation stages, aim for less than 129 tones per module, and use fewer
        modules.

        In a shared session with global control, the decimation is restored
        to stage 6 after streaming at another stage, so that other sessions
        can sweep.

        Parameters:
        ts_duration_s (float): timestream length in seconds.
        dec_stage (int): dec_stage frequency downsampling factor.
            See self.set_decimation for details.
        grp (zarr.Group): zarr group to save the batch data.
        tmp_directory (str): directory to save temporary parser data before
            converting. Data is streamed to disk, so the drive must have
            fast enough I/O performance with sufficient free space. Only used
            by the parser method.
        batch_size_mb (float): batch size, in MB. Approximate size of each chunk
            that is loaded into memory when converting parser data to zarr.
            Only used by the parser method.
        chunk_size_mb (float): size of each zarr chunk along the time axis,
            in MB.
        delete_parser_data (bool): If True, deletes the parser data files
            after importing the data. Only used by the parser method.
        verbose (bool): If True, displays a progress bar while taking data.
        method (str): Can be 'parser' (default, the rfmux parser writes the
            data to tmp_directory, which is then converted to zarr in
            batches) or 'memory' (the data is received into memory and then
            written to zarr, without temporary files). Before streaming, the
            memory method checks that the data fits in the available memory
            with headroom (see ``util.check_memory_for_stream``).

        Returns:
        None

        Raises:
        MemoryError: if method is 'memory' and the data would not fit in
            memory with headroom.
        """
        # Validate inputs
        ts_duration_s, dec_stage, self.ch_map, _, \
        tmp_directory, data_directory, batch_size_mb, chunk_size_mb, \
        delete_parser_data, verbose = \
        _validate_stream_input(
            ts_duration_s, dec_stage, grp, self.ch_map, False,
            tmp_directory, batch_size_mb, chunk_size_mb, delete_parser_data,
            verbose, method
        )
        max_ntones = max(
            [len(f) for f in self.fres_map.values()]
            ) if len(self.fres_map) else 0
        if self.ntones == 0 or max_ntones == 0:
            raise RuntimeError('No tones are written')
        stream_module_idxs = sorted(
            int(mi) for mi, f in self.fres_map.items() if len(f))
        if method == 'memory':
            nframes = int(util.get_sample_freq(dec_stage) * ts_duration_s)
            util.check_memory_for_stream(
                len(stream_module_idxs), max_ntones, self.ntones, nframes,
                chunk_size_mb
                )

        ### Start CRS config group
        cfg_grp = grp.require_group('crs_config')

        ### Set decimation stage
        await self.set_decimation(dec_stage, verbose = verbose)
        dt = 1 / self.sample_freq
        try:
            util.write_acq_cfg_to_zarr(self, cfg_grp)

            # Save timestamp
            from datetime import datetime
            grp.attrs['timestamp'] = datetime.now().strftime('%Y%m%d,%H:%M:%S')

            if method == 'parser':
                ### Run parser
                # Prepare parser arguments
                chs = '1-' + f'{max_ntones}'
                T = ts_duration_s + 0.1
                nframes = int(self.sample_freq * T)
                # As of 20260127, parser ends when any module reaches nframes,
                # so 0.1 s is added to ensure all modules reach desired
                # ts_duration_s. This will likely be fixed in future parser
                # versions.
                args = [
                    '-i', self.interface,
                    '-d', data_directory,
                    '-c', chs,
                    '-s', f'{self.serial_number:04d}',
                    '-n', f'{nframes:d}'
                    ]

                # Run parser
                from rfmux.tools import parser
                try:
                    if verbose:
                        run_with_time_bar(
                            parser.main,
                            T,
                            'Streaming',
                            *args
                            )
                    else:
                        parser.main(*args)
                except SystemExit as e:
                    # parser.main raises SystemExit when it is done
                    pass
            else:
                ### Receive packets into memory
                capture_args = (self.interface, self.serial_number,
                                stream_module_idxs, max_ntones, nframes)
                if verbose:
                    buffers, dropped = run_with_time_bar(
                        util.capture_to_memory, ts_duration_s, 'Streaming',
                        *capture_args
                        )
                else:
                    buffers, dropped = util.capture_to_memory(*capture_args)
        finally:
            # Restore the shared sweep stage so other sessions can sweep
            if self.shared and self.global_control and dec_stage != 6:
                await self.set_decimation(6, verbose = verbose)

        ### Process data
        if method == 'parser':
            util.parser_to_zarr(
                data_directory,
                grp,
                self.serial_number,
                self.ntones,
                max_ntones,
                self.ch_map,
                self.ares_map,
                dt,
                batch_size_mb = batch_size_mb,
                chunk_size_mb = chunk_size_mb
            )

            ### Delete parser data
            if delete_parser_data:
                shutil.rmtree(data_directory)
        else:
            lost = {mi: n for mi, n in dropped.items() if n}
            if lost:
                warnings.warn(
                    f'Packets were dropped while streaming (module: number '
                    f'dropped): {lost}. The saved samples are not evenly '
                    'spaced in time.', UserWarning
                    )
            util.memory_to_zarr(
                buffers,
                grp,
                self.ntones,
                max_ntones,
                self.ch_map,
                self.ares_map,
                dt,
                chunk_size_mb = chunk_size_mb
            )

################################################################################
################## Methods registered to rfmux.ReadoutModule ###################
################################################################################
@rfmux.macro(rfmux.ReadoutModule, register=True)
async def _set_nco(module, nco_freqs):
        """
        Set the NCO frequency

        Parameters:
        module (rfmux.ReadoutModule): readout module object.
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz. This should not be a round number.

        Returns:
        nco_meas (float): Measured NCO frequency in Hz.
        """
        # Input validation
        if not isinstance(nco_freqs, dict):
            raise TypeError('nco_freqs must be a dictionary')
        d = module.crs
        module_idx = module.module
        if module_idx not in nco_freqs:
            raise ValueError(f'NCO frequency for module {module_idx} not '
                             'provided in nco_freqs')
        nco_freq = nco_freqs[module_idx]
        if not isinstance(nco_freq, (float, np.floating)):
            raise TypeError('NCO frequency must be a float')

        # Set NCO frequency on the module
        await d.set_nco_frequency(nco_freq, module = module_idx)

        # Confirm set NCO frequency is close to measured frequency
        nco_meas = await d.get_nco_frequency(module = module_idx)
        if not np.isclose(nco_meas, nco_freq, atol = 1, rtol = 0):
            err = f'Failed to set NCO frequency to {nco_freq} Hz. '
            err += f'Set to {nco_meas} Hz instead.'
            raise RuntimeError(err)

        # Update nco_freqs with measured value
        nco_freqs[module_idx] = nco_meas

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def _write_tones(module, nco_freqs, fres_map, ares_map, full_scale_dbm):
        """
        Write an array of tones given frequencies and amplitudes.

        Parameters:
        module (rfmux.ReadoutModule): readout module object.
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz.
        fres_map (dict): keys (int) are module indices and values (array-like)
            are frequencies in Hz.
        ares_map (dict): keys (int) are module indices and values (array-like)
            are powers in dBm.
        full_scale_dbm (dict): keys (int) are module indices and values
            (float) are DAC full scale powers in dBm.

        Returns:
        None
        """
        # Prepare fres and ares
        d = module.crs
        module_idx = module.module
        fres, ares = fres_map[module_idx], ares_map[module_idx]
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)

        # Check NCO and input parameters
        try:
            nco = nco_freqs[module_idx]
        except:
            raise Exception('NCO frequency has not been set')
        try:
            fs_dbm = full_scale_dbm[module_idx]
        except KeyError:
            raise RuntimeError(f'Module {module_idx} has not been configured')

        # ares validation
        if any(ares > fs_dbm):
            err = f'ares must not exceed {fs_dbm} dBm: raise '
            err += 'full_scale_dbm or lower powers'
            raise ValueError(err)
        if any(ares < -60) and len(ares) < 100:
            err = f"values in ares are < -60 dBm: digitization noise may occur"
            warnings.warn(err, UserWarning)
        if len(ares):
            total_dbm = 10 * np.log10(np.sum(10 ** (ares / 10)))
            if total_dbm > fs_dbm:
                warnings.warn(
                    f'Total tone power on module {module_idx} is '
                    f'{total_dbm:.1f} dBm, above the DAC full scale of '
                    f'{fs_dbm} dBm: the DAC output will clip', UserWarning
                    )
        ares_amplitude = 10 ** ((ares - fs_dbm) / 20)

        # Clear channels
        await d.clear_channels(module = module_idx)

        # Write frequencies and amplitudes
        async with d.tuber_context() as ctx:
            for ch, (fr, ar) in enumerate(zip(fres, ares_amplitude)):
                ctx.set_frequency(fr - nco, channel = ch + 1,
                                  module = module_idx)
                ctx.set_amplitude(ar, channel = ch+1, module = module_idx)
            await ctx()

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def _sweep(
    module, nco_freqs, frequencies_map, ares_map, sweep_f, sweep_z, nsamps = 10,
    verbose = True, pbar_description = 'Sweeping', full_scale_dbm = None,
    check_dec_stage = None
):
        """
        Perform a frequency sweep and return the complex S21 value at each
        frequency. Performs sweeps over axis 0 of frequencies simultaneously.

        Parameters:
        module (rfmux.ReadoutModule): readout module object.
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz.
        frequencies_map (dict): keys (int) are module indices and values
            (M X N array-like float) are arrays where the first index M is the
            channel index (max len 1024) and the second index N is the frequency
            in Hz for a single point in the sweep.
        ares_map (dict): keys (int) are module indices and values
            (M array-like float) are amplitudes in dBm for each channel.
        sweep_f (dict): output parameter. keys (int) are module indices and
            values (M X N array-like float) are arrays where the first index M
            is the channel index and the second index N is the frequency in Hz
            for a single point in the sweep.
        sweep_z (dict): output parameter. keys (int) are module indices and
            values (M X N array-like complex) are arrays where the first index M
            is the channel index and the second index N is the complex S21 data
            in V for each frequency in f.
        nsamps (int): number of samples to average per point.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.
        full_scale_dbm (dict): keys (int) are module indices and values
            (float) are DAC full scale powers in dBm. Required.
        check_dec_stage (int or None): If not None, the board's decimation
            stage is read with every frequency update, and an error is raised
            if it is not check_dec_stage. Used by sessions without global
            control to detect changes made by another session.

        Returns:
        None
        """
        d = module.crs
        module_idx = module.module

        # Validate inputs
        _validate_sweep_input(module_idx, nco_freqs, frequencies_map, ares_map,
                              sweep_f, sweep_z, nsamps, verbose,
                              pbar_description)
        if not isinstance(full_scale_dbm, dict):
            raise TypeError('full_scale_dbm must be a dictionary')

        frequencies = np.asarray(frequencies_map[module_idx],
                                 dtype = np.float64)
        ares = np.asarray(ares_map[module_idx], dtype = np.float64)
        nco_freq = nco_freqs[module_idx]

        if not len(frequencies):
            return

        n_chs, n_points = frequencies.shape

        # Call write_tones to clear channels and initialize amplitudes with
        # first frequency of sweep
        fres_map = {module_idx: [fi[0] for fi in frequencies]}
        await module._write_tones(nco_freqs, fres_map, ares_map,
                                  full_scale_dbm)

        # Initialize z array
        z = np.empty((n_chs, n_points), dtype = complex)

        pbar = range(n_points)
        if verbose:
            pbar = tqdm(pbar, total = n_points, leave = False)
            pbar.set_description(pbar_description)

        for sweep_idx in pbar:
            # Write frequencies
            dec_future = None
            async with d.tuber_context() as ctx:
                for ch in range(n_chs):
                    f = frequencies[ch, sweep_idx]
                    ctx.set_frequency(f - nco_freq, channel = ch + 1,
                                      module = module_idx)
                if check_dec_stage is not None:
                    dec_future = ctx.get_decimation()
                await ctx()
            if dec_future is not None and \
               dec_future.result() != check_dec_stage:
                await d.clear_channels(module = module_idx)
                raise RuntimeError(
                    f'The CRS decimation stage changed to '
                    f'{dec_future.result()} during the sweep on module '
                    f'{module_idx} (expected {check_dec_stage}). Another '
                    'session changed the decimation.'
                    )
            samples = await d.get_samples(
                nsamps, module = module_idx, average = True
                )
            # format and average data
            zi = np.asarray(samples.mean.i) + 1j * np.asarray(samples.mean.q)
            if len(zi) < n_chs:
                await d.clear_channels(module = module_idx)
                raise RuntimeError(
                    f'get_samples returned {len(zi)} channels on module '
                    f'{module_idx}, but {n_chs} tones are written. The CRS is '
                    'streaming short packets (max 128 channels).'
                    )
            zi = (
                zi[:n_chs]
                * rfmux.core.transferfunctions.VOLTS_PER_ROC
                / np.sqrt(2)
            )
            z[:, sweep_idx] = zi

        # Turn off channels
        await d.clear_channels(module = module_idx)
        sweep_f[module_idx] = frequencies
        sweep_z[module_idx] = z

################################################################################
########################### Input validation helpers ###########################
################################################################################
def _normalize_module_idxs(module_idxs, name):
    """
    Convert module indices to a sorted list of unique python ints and check
    that they are in [1, 8].

    Parameters:
    module_idxs (array-like int): module indices.
    name (str): parameter name, used in error messages.

    Returns:
    list of int: sorted, unique module indices.

    Raises:
    TypeError: if module_idxs is not an iterable of integers.
    ValueError: if any module index is outside of [1, 8].
    """
    if isinstance(module_idxs, (str, bytes)) or \
       not hasattr(module_idxs, '__iter__'):
        raise TypeError(f'{name} must be a list of integers')
    module_idxs = list(module_idxs)
    if not all(isinstance(mi, (int, np.integer)) and
               not isinstance(mi, bool) for mi in module_idxs):
        raise TypeError(f'{name} must be a list of integers')
    module_idxs = sorted(set(int(mi) for mi in module_idxs))
    if any(mi not in ALL_MODULE_IDXS for mi in module_idxs):
        raise ValueError(f'{name} values must be in [1, 8]')
    return module_idxs

def _validate_configure_system_params(
    clock_source, full_scale_dbm, analog_bank_high, verbose,
    suppress_dec_printout = True
):
    """
    Validate the input parameters for configure_system.

    Parameters:
    see docstring of configure_system for parameter description.

    Returns:
    None
    """
    if clock_source not in ['VCXO', 'SMA']:
        raise ValueError("clock_source must be 'VCXO' or 'SMA'")
    if not isinstance(full_scale_dbm, (int, float)):
        raise TypeError('full_scale_dbm must be a number')
    if full_scale_dbm < 0. or full_scale_dbm > 7:
        raise ValueError('full_scale_dbm must be in [0, 7]')
    if not isinstance(analog_bank_high, bool):
        raise TypeError('analog_bank_high must be a boolean value')
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean value')
    if not isinstance(suppress_dec_printout, bool):
        raise TypeError('suppress_dec_printout must be a boolean value')

def _validate_nco_freqs(nco_freqs, analog_bank_high):
    """
    Validate the nco_freqs dictionary format.

    Parameters:
    See docstring of write_tones for parameter description.

    Returns:
    None
    """
    if not isinstance(nco_freqs, dict):
        raise TypeError('nco_freqs must be a dictionary')
    for mi, nco in nco_freqs.items():
        if not isinstance(nco, (float, np.floating)):
            msg = 'nco_freqs values must be float NCO frequencies in Hz.'
            raise TypeError(msg)
        if nco <= 0 or nco >= 5e9:
            msg = f'NCO frequency {nco} Hz is out of range [0, 5] GHz.'
            raise ValueError(msg)
        if not isinstance(mi, (int, np.integer)):
            raise TypeError('nco_freqs keys must be integer module indices')
        if analog_bank_high and not (5 <= mi <= 8):
            raise ValueError(
                f'Module index {mi} is out of range [5, 8] for high '
                'analog bank.'
                )
        if not analog_bank_high and not (1 <= mi <= 4):
            raise ValueError(
                f'Module index {mi} is out of range [1, 4] for low '
                'analog bank.'
                )

def _validate_stream_input(
    ts_duration_s, dec_stage, grp, ch_map, allow_missing, tmp_directory,
    batch_size_mb, chunk_size_mb, delete_parser_data, verbose, method = 'parser'
):
    """
    Validate inputs for the stream method.

    Parameters:
    ts_duration_s (float): timestream length in seconds.
    dec_stage (int): dec_stage frequency downsampling factor.
    grp (zarr.Group): zarr group to save the batch data.
    ch_map (dict): keys (int) are module indices and values (array-like int)
        are channel indices to write tones to. If None, automatically maps
        tones to NCOs based on frequency.
    allow_missing (bool): whether to allow missing channels in ch_map.
    tmp_directory (str): directory to save temporary parser data. Only
        created and checked if method is 'parser'.
    batch_size_mb (float): batch size in MB.
    chunk_size_mb (float): chunk size in MB.
    delete_parser_data (bool): whether to delete parser data after import.
    verbose (bool): whether to display progress bars.
    method (str): 'parser' (default) or 'memory'. See ``CRS.stream``.

    Returns:
    ts_duration_s (float): timestream length in seconds.
    dec_stage (int): decimation stage.
    ch_map (dict or None): validated ch_map.
    allow_missing (bool): whether to allow missing channels.
    tmp_directory (str): normalized temporary directory.
    data_directory (str or None): parser data directory, or None if method
        is 'memory'.
    batch_size_mb (float): batch size in MB.
    chunk_size_mb (float): chunk size in MB.
    delete_parser_data (bool): whether to delete parser data.
    verbose (bool): whether to display progress bars.

    Raises:
    ValueError: if parameters have invalid values
    TypeError: if parameters have invalid types
    FileExistsError: if data_directory already exists
    """
    if method not in STREAM_METHODS:
        raise ValueError(f'method must be one of {STREAM_METHODS}')

    ts_duration_s = float(ts_duration_s)
    if ts_duration_s <= 0:
        raise ValueError('ts_duration_s must be a positive float')

    dec_stage = int(dec_stage)
    if dec_stage < 0 or dec_stage > 6:
        raise ValueError('dec_stage must be an integer between 0 and 6')

    if not isinstance(grp, zarr.core.group.Group):
        raise TypeError('grp must be a zarr.Group object')
    # Check that required names don't already exist in the group
    existing_names = set(grp.keys())
    required_names = {'counts_to_s21', 'dt', 'z'}
    conflicts = existing_names & required_names
    if conflicts:
        msg = f'grp already contains required names: {sorted(conflicts)}'
        raise ValueError(msg)

    ch_map = util._validate_ch_map(ch_map)

    allow_missing = bool(allow_missing)

    tmp_directory = os.path.normpath(os.path.expanduser(tmp_directory))
    if method == 'parser':
        os.makedirs(tmp_directory, exist_ok = True)
        data_directory = os.path.join(tmp_directory, 'parser_data_00')
        if os.path.exists(data_directory):
            raise FileExistsError(f'{data_directory} already exists')
    else:
        data_directory = None

    batch_size_mb, chunk_size_mb = util._validate_batch_chunk_sizes(
        batch_size_mb, chunk_size_mb
    )

    delete_parser_data = bool(delete_parser_data)
    verbose = bool(verbose)
    return ts_duration_s, dec_stage, ch_map, allow_missing, tmp_directory, \
           data_directory, batch_size_mb, chunk_size_mb, delete_parser_data, \
           verbose

def _validate_sweep_input(
    module_idx, nco_freqs, frequencies_map, ares_map, sweep_f, sweep_z, nsamps,
    verbose, pbar_description
):
    """
    Validate inputs for the _sweep macro function.

    Parameters:
    See docstring of _sweep for parameter descriptions.

    Returns:
    None

    Raises:
    TypeError: if inputs are not the correct type.
    ValueError: if inputs have invalid values.
    """
    # Validate nco_freqs
    if not isinstance(nco_freqs, dict):
        raise TypeError('nco_freqs must be a dictionary')
    if module_idx not in nco_freqs:
        raise ValueError(f'nco_freqs does not contain module index {module_idx}')
    if not isinstance(nco_freqs[module_idx], (float, np.floating)):
        raise TypeError(
            f'nco_freqs[{module_idx}] must be a float NCO frequency in Hz'
        )

    # Validate frequencies_map
    if not isinstance(frequencies_map, dict):
        raise TypeError('frequencies_map must be a dictionary')
    if module_idx not in frequencies_map:
        raise ValueError(f'frequencies_map does not contain module index {module_idx}')
    # Check that value is 2D array-like (list of lists or 2D array)
    try:
        arr = np.asarray(frequencies_map[module_idx], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f'frequencies_map[{module_idx}] must be a 2D array '
                f'(M channels X N frequencies), got shape {arr.shape}'
            )
    except (ValueError, TypeError) as e:
        raise TypeError(
            f'frequencies_map[{module_idx}] must be 2D array-like of floats'
        ) from e

    # Validate ares_map
    if not isinstance(ares_map, dict):
        raise TypeError('ares_map must be a dictionary')
    if module_idx not in ares_map:
        raise ValueError(f'ares_map does not contain module index {module_idx}')
    try:
        arr = np.asarray(ares_map[module_idx], dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f'ares_map[{module_idx}] must be a 1D array, got shape {arr.shape}'
            )
    except (ValueError, TypeError) as e:
        raise TypeError(
            f'ares_map[{module_idx}] must be 1D array-like of floats'
        ) from e

    # Validate sweep_f and sweep_z (output dicts should not contain module_idx)
    if not isinstance(sweep_f, dict):
        raise TypeError('sweep_f must be a dictionary')
    if module_idx in sweep_f:
        raise ValueError(
            f'sweep_f already contains module index {module_idx}. '
            'Output dictionaries should not contain the current module index.'
        )

    if not isinstance(sweep_z, dict):
        raise TypeError('sweep_z must be a dictionary')
    if module_idx in sweep_z:
        raise ValueError(
            f'sweep_z already contains module index {module_idx}. '
            'Output dictionaries should not contain the current module index.'
        )

    # Validate nsamps
    if not isinstance(nsamps, (int, np.integer)):
        raise TypeError('nsamps must be an integer')
    if nsamps <= 0:
        raise ValueError('nsamps must be greater than 0')

    # Validate verbose
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')

    # Validate pbar_description
    if not isinstance(pbar_description, str):
        raise TypeError('pbar_description must be a string')
