import os
import asyncio
import shutil
import warnings
import time
import numpy as np
from tqdm.auto import tqdm

# citkid imports
from ..util import run_with_time_bar
from . import util

# rfmux imports
import rfmux
from rfmux.algorithms.measurement import take_netanal


class CRS:
    def __init__(self, serial_number = 27, interface = 'enp2s0'):
        """
        Initialize the CRS object.

        Note that the system must be configured using
        `CRS.configure_system` before measurements.

        Parameters:
        serial_number (int): CRS serial number.
        interface (str): Ethernet interface identifier.

        Returns:
        None
        """
        # Input validation 
        if rfmux.__version__ != '1.3.2':
            raise RuntimeError('rfmux version 1.3.2 is required') 
        if not isinstance(serial_number, int):
            raise TypeError('serial_number must be an integer') 
        if not util.interface_exists(interface):
            raise ValueError(f'interface {interface} does not exist') 
        
        # Store inputs
        self.serial_number = serial_number
        self.interface = interface
        self.nco_freqs = {}

        # Initialize CRS object
        session_str = (
            '!HardwareMap [ !CRS { '
            + f'serial: "{serial_number:04d}"'
        )
        session_str += ' } ]'
        s = rfmux.load_session(session_str)
        self.d = s.query(rfmux.CRS).one()

        # Set module bw, in case self.extended_bw is not called 
        self.bw = 500e6

        # Attributes to keep track of tone frequency, amplitude, 
        # and channel mappings
        self.fres_map = {}
        self.ares_map = {}
        self.ch_map = {}

        # Attributes that keep track of decimation
        self.dec_stage = None
        self.dec_short = None 
        self.dec_module_idxs = None

    async def configure_system(self, clock_source = "VCXO", full_scale_dbm = 7,
                               analog_bank_high = False, verbose = True):
        """
        Resolve the system, validate the CRS firmware version, set the timestamp 
        port, clock source, extended bandwidth, analog bank, and DAC full scale.

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

        Returns:
        None
        """
        # Input validation
        _validate_configure_system_params(clock_source, full_scale_dbm, 
                                          analog_bank_high, verbose)
        
        # Resolve the system
        await self.d.resolve()

        # Validate firmware version
        self.firmware_release = await self.d.get_firmware_release() 
        if self.firmware_release.version != '1.6.0rc3':
            raise RuntimeError("CRS firmware must be version 1.6.0rc3")

        # Set the timestamp port. Bypass if already set
        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted:
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)

        # Set the clock source
        await self.set_clock_source(clock_source)

        # Default extended bandwidth to False
        await self.set_extended_bw(False)

        # Set the analog bank and DAC full scales
        await self.set_analog_bank(analog_bank_high, full_scale_dbm)

        # Set the decimation 
        await self.set_decimation(6, short = False, module_idxs = None,
                                 verbose = verbose) 
        
        # Store number of tones
        self.ntones = 0

        # Print configuration if verbose.
        if verbose:
            print('System configured')
            print("Clocking source is", self.clock_source)
        
    async def set_clock_source(self, clock_source):
        """
        Set the clock source to 'VCXO' or 'SMA'.
        
        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal
            voltage controlled crystal oscillator or 'SMA' for the external 10
            MHz reference (reference should be 5 Vpp).
            
        Returns:
        None
        """
        # Input validation
        _validate_configure_system_params(clock_source, 1, True, True)
        
        # Set and check clock source
        await self.d.set_clock_source(clock_source)
        self.clock_source = await self.d.get_clock_source() 
        if self.clock_source != clock_source:
            warnings.warn(
                f"Requested clock source {clock_source} unavailable. "
                + f"Using {self.clock_source} instead.",
                UserWarning
            )

    async def set_analog_bank(self, analog_bank_high, full_scale_dbm):
        """
        Set the analog bank to high (modules 5-8) or low (modules 1-4).

        Parameters:
        analog_bank_high (bool): if True, uses modules 5-8 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4).
        full_scale_dbm (int): full scale power in dBm. Range is [0, 7].

        Returns:
        None
        """
        # Input validation
        _validate_configure_system_params("SMA", full_scale_dbm, 
                                         analog_bank_high, True)
        
        # Set analog bank
        await self.d.set_analog_bank(high = analog_bank_high)
        abh = await self.d.get_analog_bank()
        if abh != analog_bank_high:
            raise RuntimeError("Failed to set analog bank")
    
        # Remove other modules from nco_freqs and maps
        remove_module_idxs = range(1, 5) if analog_bank_high else range(5, 9)
        for module_idx in remove_module_idxs:
            if module_idx in self.nco_freqs:
                del self.nco_freqs[module_idx]
            if module_idx in self.fres_map:
                del self.fres_map[module_idx]
            if module_idx in self.ares_map:
                del self.ares_map[module_idx]
            if module_idx in self.ch_map:
                del self.ch_map[module_idx]
        
        # Store analog bank
        self.analog_bank_high = analog_bank_high

        # Set DAC scale to full_scale_dbm
        module_idxs = range(5, 9) if analog_bank_high else range(1, 5)
        coros = [
            self.d.set_dac_scale(full_scale_dbm, self.d.UNITS.DBM,
                                 module_idx)
            for module_idx in module_idxs
        ]
        await asyncio.gather(*coros) 

        # Confirm DAC scale was set correctly
        coros = [
            self.d.get_dac_scale(self.d.UNITS.DBM, module_idx)
            for module_idx in module_idxs
                ]
        results = await asyncio.gather(*coros) 
        if not np.allclose(results, full_scale_dbm, atol = 0.1):
            raise RuntimeError("Failed to set DAC full scale")
        
        # Store full scale in self.d
        self.d.full_scale_dbm = full_scale_dbm

    async def set_extended_bw(self, extended):
        """
        Choose between the standard (500 MHz) and extended (600 MHz) bandwidth.

        Only extend the bandwidth if you know what you are doing. See
        `crs.d.set_extended_module_bandwidth` for details.

        Parameters:
        extended (bool): If True, extends the bandwidth to 600 MHz. Else
            sets the bandwidth to 500 MHz.

        Returns:
        None
        """
        # Input validation
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

    async def set_decimation(self, dec_stage, short = None, module_idxs = None,
                             verbose = True):
        """
        Sets the decimation stage, with optional short mode and module indices. 
        If short and/or module_idxs are not provided, they are determined using 
        self.fres_map. Raises an error if the configuration will drop packets. 

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
            set based on the maximum number of tones in self.fres_map.
        module_idxs (array-like int or None): module indices to stream. If None,
            module_idxs is set to all modules with tones in self.fres_map. 
        verbose (bool): If True, prints the decimation settings after 
            confirming.

        Returns:
        None
        """
        # Input validation 
        if not isinstance(dec_stage, int) or dec_stage < 0 or dec_stage > 6:
            raise ValueError('dec_stage must be an integer between 0 and 6') 
        if short is not None and not isinstance(short, bool):
            raise TypeError('short must be a boolean or None') 
        if module_idxs is not None:
            if (not isinstance(module_idxs, (list, np.ndarray))) or \
               (not all(isinstance(mi, (int, np.integer)) 
                        for mi in module_idxs)):
                raise TypeError('module_idxs must be a list of integers.') 
            
        # Determine short and module_idxs if not provided
        if short is None:
            if self.fres_map:
                max_ntones = max([len(f) for f in self.fres_map.values()])
                short = max_ntones <= 128
            else:
                short = False
        if module_idxs is None or len(module_idxs) == 0:
            module_idxs = [k for k, v in self.fres_map.items() if len(v)]

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
        if verbose:
            msg = f'Set decimation: dec_stage = {dec_stage}, short = {short}, '
            msg += f'modules = {module_idxs}'
            print(msg)

    async def set_nco(self, nco_freqs, verbose = True):
        """
        Set the NCO frequency.

        Parameters:
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz.
        verbose (bool): If True, prints the NCO frequencies after confirming.

        Returns:
        None
        """
        # Input validation 
        _validate_nco_freqs(nco_freqs, self.analog_bank_high)
            
        # Set NCO frequencies
        nco_freqs = nco_freqs.copy()
        modules = util.get_modules(self.d, list(nco_freqs.keys()))
        await modules._set_nco(nco_freqs)
        self.nco_freqs.update(nco_freqs)

        # Clear channels on modules with newly set NCO frequencies 
        await self._clear_channels(list(nco_freqs.keys()))
        for module_idx, nco in nco_freqs.items():
            if verbose:
                nco_str = f'{round(nco * 1e-6, 6)}'
                print(f'Module {module_idx} NCO is {nco_str} MHz')

    async def disable_modules(self, module_idxs):
        """
        Disable modules by clearing channels and removing NCO frequencies. 

        Parameters:
        module_idxs (int): module indices to disable.

        Returns:
        None
        """
        # Input validation
        if not all(isinstance(mi, (int, np.integer)) for mi in module_idxs):
            raise TypeError('module_idxs must be a list of integers')

        # Clear channels
        await self._clear_channels(module_idxs)
        
        # Update self.nco_freqs
        for module_idx in module_idxs:
            if module_idx in self.nco_freqs:
                del self.nco_freqs[module_idx]

    async def _clear_channels(self, module_idxs):
        """
        Clear all channels on the specified modules. 
        
        Parameters:
        module_idxs (array-like int): module indices to clear channels on.

        Returns:
        None
        """
        # Input validation
        if not all(isinstance(mi, (int, np.integer)) for mi in module_idxs):
            raise TypeError('module_idxs must be a list of integers')
        
        # Clear channels and update fres_map and ares_map
        for module_idx in module_idxs:
            await self.d.clear_channels(module = module_idx)
            self.ntones -= len(self.ch_map.get(module_idx, []))
            if module_idx in self.fres_map:
                del self.fres_map[module_idx]
            if module_idx in self.ares_map:
                del self.ares_map[module_idx]
            if module_idx in self.ch_map:
                del self.ch_map[module_idx]

    async def write_tones(self, fres, ares, ch_map = None, 
                          allow_missing = False):
        """
        Write tones for the provided frequencies and amplitudes.

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
        None
        """
        # Input validation 
        if not len(self.nco_freqs):
            raise RuntimeError("NCO frequencies are not set")
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)
        if fres.shape != ares.shape:
            raise ValueError('fres and ares must be the same shape')
        if len(fres) == 0:
            return 0 
        _validate_ch_map(ch_map)

        # Store ntones for later
        ntones = len(fres)

        # Get ch_map
        if ch_map is None:
            ch_map, missing_chs = util.create_ch_map(
                self.nco_freqs, fres, self.bw
                )
        else:
            ch_map = ch_map.copy()
            missing_chs = []
            for ch in range(len(fres)):
                if not any(ch in ch_list for ch_list in ch_map.values()):
                    missing_chs.append(ch)

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
        for key, fres in self.fres_map.items():
            # Dither separately for each NCO
            if not len(fres):
                continue 
            self.fres_map[key] = take_netanal._safe_concatenate_frequencies(
                fres, self.nco_freqs[key]
                )

        # Write tones
        modules = util.get_modules(self.d, list(self.fres_map.keys()))
        await modules._write_tones(self.nco_freqs, self.fres_map,
                                  self.ares_map)
        self.ntones = ntones

    async def sweep(
            self, frequencies, ares, nsamps, ch_map = None, 
            allow_missing = False, verbose = True, pbar_description = 'Sweeping'
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
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        np.ndarray: complex S21 values for each frequency.
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
        _validate_ch_map(ch_map) 
        if not isinstance(pbar_description, str):
            raise TypeError('pbar_description must be a string')
        
        ### Map frequencies to modules/channels
        # Get ch_map
        if ch_map is None:
            ch_map, missing_chs = util.create_ch_map(
                self.nco_freqs, frequencies, self.bw
                )
        else:
            ch_map = ch_map.copy()
            missing_chs = []
            for ch in range(len(frequencies)):
                if not any(ch in ch_list for ch_list in ch_map.values()):
                    missing_chs.append(ch)
            
        # Handle missing channels
        if len(missing_chs):
            if ch_map is not None:
                msg = (f"Tones must be within {self.bw / 2e6:.0f} "
                    "MHz of an NCO frequency.")
            else: 
                msg = "Missing channels detected in ch_map."
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

        ### Set dec_stage to 6 for sweeping
        await self.set_decimation(6, verbose = verbose)

        ### Sweep
        sweep_f, sweep_z = {}, {}
        modules = util.get_modules(self.d, list(self.fres_map.keys()))
        await modules._sweep(self.nco_freqs, self.fres_map,
                            self.ares_map, sweep_f, sweep_z, nsamps = nsamps,
                            verbose = verbose,
                            pbar_description = pbar_description)
        
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

    async def sweep_linear(
            self, fres, ares, span, npoints, nsamps, ch_map = None, 
            allow_missing = False, center_fres = True, downward = True, 
            verbose = True, pbar_description = 'Sweeping'
            ):
        """
        Performs a frequency sweep where each channel is swept over the same
        frequency span.

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
        # other validation is performed in self.sweep

        # Create freqs array
        if center_fres:
            if downward:
                freqs = np.linspace(fres + span / 2, fres - span / 2, npoints).T
            else:
                freqs = np.linspace(fres - span / 2, fres + span / 2, npoints).T
        else:
            if downward:
                freqs = np.linspace(fres + span, fres, npoints).T
            else:
                freqs = np.linspace(fres, fres + span, npoints).T

        # Sweep and return 
        f, z = await self.sweep(
            freqs, ares, nsamps, ch_map = ch_map, allow_missing = allow_missing, 
            verbose = verbose, pbar_description = pbar_description
            )
        return f, z

    async def sweep_qres(
            self, fres, ares, qres, npoints, nsamps, ch_map = None,
            allow_missing = False, verbose = True, pbar_description = 'Sweeping'
            ):
        """
        Performs a frequency sweep where the span around each frequency is set
        equal to fres / qres.

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
        # other validation is performed in self.sweep
        
        # Create freqs array
        spans = fres / qres
        freqs = np.linspace(fres + spans / 2, fres - spans / 2, npoints).T

        # Sweep and return
        f, z = await self.sweep(
            freqs, ares, nsamps, ch_map = ch_map, allow_missing = allow_missing, 
            verbose = verbose, pbar_description = pbar_description
            )
        return f, z

    async def sweep_full(
            self, amplitude, npoints, nsamps,
            verbose = True, pbar_description = 'Sweeping'
            ):
        """
        Performs a frequency sweep over the full bandwidth around the NCO
        frequency.

        Parameters:
        amplitude (float): amplitude in dBm.
        npoints (int): number of sweep points per channel.
        nsamps (int): number of samples to average per point.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        f (np.array): array of frequencies in Hz.
        z (np.array): array of complex S21 data corresponding to f.
        """
        # Input validation 
        amplitude = float(amplitude)
        if not isinstance(npoints, int) or npoints <= 0:
            raise ValueError('npoints must be a positive integer') 
        # other validation is performed in self.sweep_linear

        # Create fres and ares arrays
        ncos = self.nco_freqs.values()
        tone_bw = self.bw / 1024 + 200
        # spacing = tone_bw / npoints
        fres = np.concatenate([np.linspace(nco - self.bw / 2 + 10 + tone_bw,
                                           nco + self.bw / 2 - 10 - tone_bw,
                                           1024) for nco in ncos])
        ares = amplitude * np.ones(len(fres))
        
        # Sweep
        f, z = await self.sweep_linear(
            fres, ares, tone_bw, npoints, nsamps, ch_map = None, 
            allow_missing = False, center_fres = True, downward = True, 
            verbose = verbose, pbar_description = pbar_description
            )
        
        # Flatten and sort
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_ts(
            self,
            fres,
            ares, 
            ts_duration_s,
            dec_stage,
            grp,
            ch_map = None,
            allow_missing = False,
            tmp_directory = 'tmp/',
            batch_size_mb = 1000,
            chunk_size_mb = 128,
            delete_parser_data = True,
            verbose = True
    ): 
        """
        Clears all tones, writes tones using fres and ares, captures a
        timestream of length ts_time using the parser, and then clears all 
        tones.

        Parameters:
        fres (array-like): tone frequencies in Hz.
        ares (array-like): tone amplitudes in dBm.
        ts_duration_s (float): timestream length in seconds.
        dec_stage (int): dec_stage frequency downsampling factor. 
            See self.set_decimation for details.
        grp (zarr.Group): zarr group to save the batch data.
        ch_map (dict): keys (int) are module indices and values (array-like int)
            are channel indices to write tones to. If None, automatically maps
            tones to NCOs based on frequency.
        allow_missing (bool): If True, ignores tones that are outside the
            bandwidth of all NCOs. If False, raises an error if any tones are
            outside the bandwidth of all NCOs.
        tmp_directory (str): directory to save temporary parser data before
            converting. Data is streamed to disk, so the drive must have
            fast enough I/O performance with sufficient free space.
        batch_size_mb (int): batch size, in MB. Approximate size of each chunk 
            that is loaded into memory when converting parser data to zarr.
        chunk_size_mb (int): size of each zarr chunk along the time axis, in MB.
        delete_parser_data (bool): If True, deletes the parser data files
            after importing the data.
        verbose (bool): If True, displays a progress bar while taking data.

        Returns:
        None
        """
        # fres and ares validation perfomed in self.write_tones 
        # Remaining input validation performed in self.stream

        # Clear all channels - ensures that unused modules are cleared 
        idx_to_clear = range(5, 9) if self.analog_bank_high else range(1, 5)
        await self._clear_channels(idx_to_clear)

        # Write tones
        await self.write_tones(
            fres, ares, ch_map = ch_map, allow_missing = allow_missing
            )
        time.sleep(0.5) # frequency change has transient
        
        # Stream  
        try:
            await self.stream(
                ts_duration_s = ts_duration_s,
                dec_stage = dec_stage,
                grp = grp,
                tmp_directory = tmp_directory,
                batch_size_mb = batch_size_mb,
                chunk_size_mb = chunk_size_mb,
                delete_parser_data = delete_parser_data,
                verbose = verbose
            )
        except Exception as e:
            # On failure to stream, clear all channels 
            await self._clear_channels(idx_to_clear) 
            raise e 
        
        # Clear all channels after streaming
        await self._clear_channels(idx_to_clear)

    
    async def stream(
        self,
        ts_duration_s,
        dec_stage,
        grp,
        tmp_directory = 'tmp/',
        batch_size_mb = 1000,
        chunk_size_mb = 128,
        delete_parser_data = True,
        verbose = True
    ):
        """
        Captures a timestream using the parser. Does not change written tones - 
        assumes fres_map and ares_map match the currently written tones.

        Note on streaming capabilities: If all modules have less than 129 tones,
        the system will stream 128 tones per module. Otherwise, it will stream 
        1024 tones per module. Streaming will only be performed on modules with 
        tones set (see self.fres_map). Setting the decimation stage will fail if 
        the requested number of modules/tones will cause packet loss. For low 
        decimation stages, aim for less than 129 tones per module, and use fewer 
        modules. 

        Parameters:
        ts_duration_s (float): timestream length in seconds.
        dec_stage (int): dec_stage frequency downsampling factor. 
            See self.set_decimation for details.
        grp (zarr.Group): zarr group to save the batch data.
        tmp_directory (str): directory to save temporary parser data before
            converting. Data is streamed to disk, so the drive must have
            fast enough I/O performance with sufficient free space.
        batch_size_mb (int): batch size, in MB. Approximate size of each chunk 
            that is loaded into memory when converting parser data to zarr.
        chunk_size_mb (int): size of each zarr chunk along the time axis, in MB.
        delete_parser_data (bool): If True, deletes the parser data files
            after importing the data.
        verbose (bool): If True, displays a progress bar while taking data.

        Returns:
        None
        """
        ### Input validation -> move to util to use in take_ts?
        tmp_directory = os.path.normpath(os.path.expanduser(tmp_directory))
        os.makedirs(tmp_directory, exist_ok = True)
        data_directory = os.path.join(tmp_directory, 'parser_data_00')
        if os.path.exists(data_directory):
            raise FileExistsError(f'{data_directory} already exists')
        
        ### Set decimation stage
        await self.set_decimation(dec_stage, verbose = verbose)

        ### Run parser
        # Prepare parser arguments
        max_ntones = max(
            [len(f) for f in self.fres_map.values()]
            ) if len(self.fres_map) else 0
        chs = '1-' + f'{max_ntones}'
        T = ts_duration_s + 0.1
        nframes = int(self.sample_freq * T)
        # As of 20260127, parser ends when any module reaches nframes, so 0.1 s 
        # is added to ensure all modules reach desired ts_duration_s. This will 
        # likely be fixed in future parser versions.
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
                await run_with_time_bar(
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
        
        ### Process data 
        util.parser_to_zarr(
            data_directory, 
            grp,
            self.serial_number,
            self.ntones,
            max_ntones,
            self.ch_map,
            self.ares_map,
            1 / self.sample_freq,
            batch_size_mb = batch_size_mb,
            chunk_size_mb = chunk_size_mb
        )

        
        ### Delete parser data
        if delete_parser_data:
            shutil.rmtree(data_directory)

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
        d = module.crs
        module_idx = module.module
        nco_freq = nco_freqs[module_idx]
        await d.set_nco_frequency(nco_freq, module = module_idx)

        nco_meas = await d.get_nco_frequency(module = module_idx)
        if not np.isclose(nco_meas, nco_freq, atol = 1):
            err = f'Failed to set NCO frequency to {nco_freq} Hz. '
            err += f'Set to {nco_meas} Hz instead.'
            raise RuntimeError(err)
        
        nco_freqs[module_idx] = nco_meas

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def _write_tones(module, nco_freqs, fres_map, ares_map):
        """
        Writes an array of tones given frequencies and amplitudes.

        Parameters:
        module (rfmux.ReadoutModule): readout module object.
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz.
        fres_map (dict): keys (int) are module indices and values (array-like)
            are frequencies in Hz.
        ares_map (dict): keys (int) are module indices and values (array-like)
            are powers in dBm.
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

        # ares validation 
        if any(ares > d.full_scale_dbm):
            err = f'ares must not exceed {d.full_scale_dbm} dBm: raise '
            err += 'full_scale_dbm or lower powers'
            raise ValueError(err)
        if any(ares < -60) and len(ares) < 100:
            err = f"values in ares are < 60 dBm: digitization noise may occur"
            warnings.warn(err, UserWarning)
        ares_amplitude = 10 ** ((ares - d.full_scale_dbm) / 20)

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
async def _sweep(module, nco_freqs, frequencies_map, ares_map, sweep_f,
                sweep_z, nsamps = 10, verbose = True,
                pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep and returns the complex S21 value at each
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
        nsamps (int): number of samples to average per point.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        z (M X N array-like complex): complex S21 data in V for each frequency
            in f.
        """
        d = module.crs
        module_idx = module.module
        frequencies = np.asarray(frequencies_map[module_idx], 
                                 dtype = np.float64)
        ares = np.asarray(ares_map[module_idx], dtype = np.float64)
        nco_freq = nco_freqs[module_idx]

        if not len(frequencies):
            return np.array([], dtype = float), np.array([], dtype = complex)
        
        n_chs, n_points = frequencies.shape
        if len(ares) != n_chs:
            raise ValueError('ares and frequencies are not the same length')

        # Call write_tones to clear channels and initialize amplitudes with 
        # first frequency of sweep
        fres_map = {module_idx: [fi[0] for fi in frequencies]}
        await module._write_tones(nco_freqs, fres_map, ares_map)

        # Initialize z array
        z = np.empty((n_chs, n_points), dtype = complex)

        pbar = range(n_points)
        if verbose:
            pbar = tqdm(pbar, total = n_points, leave = False)
            pbar.set_description(pbar_description)

        for sweep_idx in pbar:
            # Write frequencies
            async with d.tuber_context() as ctx:
                for ch in range(n_chs):
                    f = frequencies[ch, sweep_idx]
                    ctx.set_frequency(f - nco_freq, channel = ch + 1,
                                      module = module_idx)
                await ctx()
            samples = await d.get_samples(
                nsamps, module = module_idx, average = True
                ) 
            # format and average data
            zi = np.asarray(samples.mean.i) + 1j * np.asarray(samples.mean.q)
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
def _validate_configure_system_params(clock_source, full_scale_dbm, 
                                     analog_bank_high, verbose):
    """
    Validate the input parameters for configure_system.
    
    Parameters:
    see full_scale_dbm docstring.
    
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
    
def _validate_nco_freqs(nco_freqs, analog_bank_high):
    """
    Validate the nco_freqs dictionary format.
    
    Parameters:
    nco_freqs (dict): keys (int) are module indices and values (float)
        are NCO frequencies in Hz.
    analog_bank_high (bool): if True, uses modules 5-8. Else uses modules 1-4.

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
        
def _validate_ch_map(ch_map):
    """
    Validate the ch_map dictionary format.

    Parameters:
    ch_map (dict): keys (int) are module indices and values (array-like int)
        are channel indices to write tones to.

    Returns:
    None
    """
    if ch_map is not None:
        if not isinstance(ch_map, dict):
            raise TypeError('ch_map must be a dictionary') 
        for key, val in ch_map.items():
            if not isinstance(key, (int, np.integer)):
                raise TypeError('ch_map keys must be integers')
            if not all(isinstance(ch, (int, np.integer)) for ch in val):
                raise TypeError('ch_map values must be lists of integers') 
