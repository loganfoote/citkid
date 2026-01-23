import os
import asyncio
import shutil
import warnings
import numpy as np
from time import sleep
from tqdm.auto import tqdm
from . import util
import rfmux
from rfmux.algorithms.measurement import take_netanal
from rfmux.tools import parser
import warnings

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
        full_scale_dbm (int): full scale power in dBm. Range is [-18, 7].
        analog_bank_high (bool): if True, uses modules 1-4 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4). Can be changed later using 
            self.set_analog_bank.
        verbose (bool): If True, gets and prints the clocking source.

        Returns:
        None
        """
        # Input validation
        validate_configure_system_params(clock_source, full_scale_dbm, 
                                         analog_bank_high, verbose)
        
        # Resolve the system
        await self.d.resolve()

        # Validate firmware version
        self.firmware_release = await self.d.get_firmware_release() 
        if self.firmware_release.version != '1.6.0rc2':
            raise RuntimeError("CRS firmware must be version 1.6.0rc2")

        # Set the timestamp port. Bypass if already set
        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted:
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)

        # Set the clock source
        await self.set_clock_source(clock_source)

        # Default extended bandwidth to False
        await self.set_extended_module_bw(False)

        # Set the analog bank and DAC full scales
        await self.set_analog_bank(analog_bank_high, full_scale_dbm)

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
        validate_configure_system_params(clock_source, 1, True, True)
        
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
        analog_bank_high (bool): if True, uses modules 1-4 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4).
        full_scale_dbm (int): full scale power in dBm. Range is [-18, 7].

        Returns:
        None
        """
        # Input validation
        validate_configure_system_params("SMA", full_scale_dbm, 
                                         analog_bank_high, True)
        
        # Set analog bank
        await self.d.set_analog_bank(high = analog_bank_high)
        abh = await self.d.get_analog_bank()
        if abh != analog_bank_high:
            raise RuntimeError("Failed to set analog bank")
        
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

    async def set_extended_module_bw(self, extended):
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
        
        # Store extended bandwidth
        self.extended_bw = extended

        # Raise warning if extended bandwidth is set
        if extended:
            warnings.warn(f"Extended module bandwidth set", UserWarning)

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
        nco_freqs = nco_freqs.copy()
        modules = util.get_modules(self.d, list(nco_freqs.keys()))
        await modules.set_nco(nco_freqs)
        self.nco_freqs.update(nco_freqs)
        for module_idx, nco in nco_freqs.items():
            if verbose:
                nco_str = f'{round(nco * 1e-6, 6)}'
                print(f'Module {module_idx} NCO is {nco_str} MHz')

    async def write_tones(self, fres, ares):
        """
        Write tones for the provided frequencies and amplitudes.

        Splits the tones into the appropriate modules using the NCO frequencies.
        This could lead to behavior where one tone jumps between NCOs during a
        run. Consider setting tones per NCO or shifting NCOs with resonances.
        tones into the appropriate modules using the NCO frequencies.
        Note: this could lead to behavior where one tone jumps between NCOs
        during a measurement run. To mitigate this, we could consider setting
        the tones for each NCO manually, or shifting the NCOs with the
        resonances.

        Parameters:
        fres (array-like): tone frequencies in Hz.
        ares (array-like): tone powers in dBm.

        Returns:
        max_ntones (int): Maximum tones per module.
        """
        # Split fres and ares into dictionaries
        if not len(self.nco_freqs):
            raise Exception("NCO frequencies are not set")
        channel_idxs = list(range(len(fres)))
        self.fres_dict = {key: [] for key in self.nco_freqs.keys()}
        self.ares_dict = {key: [] for key in self.nco_freqs.keys()}
        self.ch_ix_dict = {key: [] for key in self.nco_freqs.keys()}
        for ch_ix, fr, ar in zip(channel_idxs, fres, ares):
            module_idx = min(self.nco_freqs,
                             key = lambda k: np.abs(self.nco_freqs[k] - fr))
            self.fres_dict[module_idx].append(fr)
            self.ares_dict[module_idx].append(ar)
            self.ch_ix_dict[module_idx].append(ch_ix)
        self.fres_dict = {
            key: np.array(value) for key, value in self.fres_dict.items()
        }
        self.ares_dict = {
            key: np.array(value) for key, value in self.ares_dict.items()
        }
        self.ch_ix_dict = {
            key: np.array(value) for key, value in self.ch_ix_dict.items()
        }
        # Confirm that the tones are within the NCO bandwidths
        for module_idx in self.nco_freqs.keys():
            bw_half = 312.5e6 if self.extended_bw else 250e6
            diffs = (
                self.fres_dict[module_idx]
                - self.nco_freqs[module_idx]
            )
            if any(np.abs(diffs) > bw_half):
                err = f'All of fres must be within {round(bw_half / 1e6, 1)} '
                err += 'MHz of an NCO frequency'
                raise ValueError(err)
        # Write tones
        modules = util.get_modules(self.d, list(self.fres_dict.keys()))
        await modules.write_tones(self.nco_freqs, self.fres_dict,
                                  self.ares_dict)
        
        max_ntones = max([len(f) for f in self.fres_dict.values()])
        return max_ntones

    async def sweep(self, frequencies, ares, nsamps = 10, return_dbc = True,
                    verbose = True, pbar_description = 'Sweeping'):
        """
        Perform a frequency sweep and return complex S21 at each frequency.

        Parameters:
        frequencies (M X N array-like float): the first index M is the channel
            index (max len 1024) and the second index N is the frequency in Hz
            for a single point in the sweep
        ares (M array-like float): amplitudes in dBm for each channel
        nsamps (int): number of samples to average per point
        return_dbc (bool): If True, divides the output by the tone power
        verbose (bool): If True, displays a progress bar while sweeping
        pbar_description (str): description for the progress bar

        Returns:
        np.ndarray: complex S21 values for each frequency.
        """
        frequencies, ares = np.asarray(frequencies), np.asarray(ares)
        if not len(self.nco_freqs):
            raise Exception("NCO frequencies are not set")
        # Split frequencies and ares into dictionaries
        channel_idxs = list(range(len(frequencies)))
        self.frequencies_dict = {key: [] for key in self.nco_freqs.keys()}
        self.ares_dict = {key: [] for key in self.nco_freqs.keys()}
        self.ch_ix_dict = {key: [] for key in self.nco_freqs.keys()}
        def select_nco(k):
            ncos = [
                np.abs(self.nco_freqs[k] - fr)
                for fr in [max(freqs), min(freqs)]
            ]
            return max(ncos)
        for ch_ix, freqs, ar in zip(channel_idxs, frequencies, ares):
            module_idx = min(self.nco_freqs, key = select_nco)
            self.frequencies_dict[module_idx].append(freqs)
            self.ares_dict[module_idx].append(ar)
            self.ch_ix_dict[module_idx].append(ch_ix)
        self.frequencies_dict = {key: np.array(value)
                                 for key, value in self.frequencies_dict.items()
                                 }
        self.ares_dict = {key: np.array(value)
                          for key, value in self.ares_dict.items()
                          }
        self.ch_ix_dict = {key: np.array(value)
                           for key, value in self.ch_ix_dict.items()
                           }
        # Confirm that frequencies are in each NCO bandwidth
        for module_idx in self.nco_freqs.keys():
            bw_half = 312.5e6 if self.extended_bw else 250e6
            diffs = self.frequencies_dict[module_idx].copy()
            diffs -= self.nco_freqs[module_idx]
            if any(np.abs(diffs).flatten() > bw_half):
                err = 'All of frequencies must be within '
                err += f'{round(bw_half / 1e6, 1)} MHz of an NCO frequency'
                raise ValueError(err)

        # Set dec_stage
        dec_stage = 6
        await self.d.set_decimation(dec_stage)
        # sweep
        sweep_f, sweep_z = {}, {}
        modules = util.get_modules(self.d, list(self.frequencies_dict.keys()))
        await modules.sweep(self.nco_freqs, self.frequencies_dict,
                            self.ares_dict, sweep_f, sweep_z, nsamps = nsamps,
                            verbose = verbose,
                            pbar_description = pbar_description)
        # Create f, z from sweep results
        nres = frequencies.shape[0]
        f = np.empty(frequencies.shape, dtype = float)
        z = np.empty(frequencies.shape, dtype = complex)
        for res_idx in range(nres):
            module_idx, ch_idx = util.find_key_and_idx(self.ch_ix_dict,
                                                        res_idx)
            f[res_idx] = sweep_f[module_idx][ch_idx]
            z[res_idx] = sweep_z[module_idx][ch_idx]
        if return_dbc:
            z /= 10 ** (ares[:, np.newaxis] / 20)
        return f, z

    async def sweep_linear(self, fres, ares, bw = 20e3, npoints = 10,
                           nsamps = 10, return_dbc = True, center_fres = True,
                           downward = True, verbose = True,
                           pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where each channel is swept over the same
        frequency span.

        Parameters:
        fres (array-like): center frequencies in Hz.
        ares (array-like): amplitudes in dBm.
        bw (float): span around each frequency to sweep in Hz.
        npoints (int): number of sweep points per channel.
        nsamps (int): number of samples to average per point.
        return_dbc (bool): If True, divides the output by the tone power.
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
        fres, ares = np.asarray(fres), np.asarray(ares)
        if center_fres:
            if downward:
                f = np.linspace(fres + bw / 2, fres - bw / 2, npoints).T
            else:
                f = np.linspace(fres - bw / 2, fres + bw / 2, npoints).T
        else:
            if downward:
                f = np.linspace(fres + bw, fres, npoints).T
            else:
                f = np.linspace(fres, fres + bw, npoints).T
        f, z = await self.sweep(f, ares, nsamps = nsamps,
                                return_dbc = return_dbc, verbose = verbose,
                                pbar_description = pbar_description)
        return f, z

    async def sweep_qres(self, fres, ares, qres, npoints = 10, nsamps = 10,
                         return_dbc = True, verbose = True,
                         pbar_description = 'Sweeping'):
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
        return_dbc (bool): If True, divides the output by the tone power.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index
            and N is the index of each point in the sweep.
        z (M X N np.array): array of complex S21 data corresponding to f.
        """
        fres, ares, qres = np.asarray(fres), np.asarray(ares), np.asarray(qres)
        spans = fres / qres
        f = np.linspace(fres + spans / 2, fres - spans / 2, npoints).T
        f, z = await self.sweep(f, ares, nsamps = nsamps,
                                return_dbc = return_dbc, verbose = verbose,
                                pbar_description = pbar_description)
        return f, z

    async def sweep_full(self, amplitude, npoints = 10, nsamps = 10,
                         return_dbc = True, verbose = True,
                         pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep over the full bandwidth around the NCO
        frequency.

        Parameters:
        amplitude (float): amplitude in dBm.
        npoints (int): number of sweep points per channel.
        nsamps (int): number of samples to average per point.
        return_dbc (bool): If True, divides the output by the tone power.
        verbose (bool): If True, displays a progress bar while sweeping.
        pbar_description (str): description for the progress bar.

        Returns:
        f (np.array): array of frequencies in Hz.
        z (np.array): array of complex S21 data corresponding to f.
        """
        ncos = list(self.nco_freqs.values())
        bw_total = 600e6 if self.extended_bw else 500e6
        bw = bw_total / 1024 + 200
        spacing = bw / npoints
        fres = np.concatenate([np.linspace(nco - bw_total / 2 + 10 + bw,
                                           nco + bw_total / 2 - 10 - bw,
                                           1024) for nco in ncos])
        ares = amplitude * np.ones(len(fres))
        # Left off here
        f, z = await self.sweep_linear(fres, ares, bw = bw - spacing,
                                       npoints = npoints, nsamps = nsamps,
                                       return_dbc = return_dbc,
                                       verbose = verbose,
                                       pbar_description = pbar_description)
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_noise(
        self,
        fres,
        ares,
        noise_time,
        dec_stage = 6,
        fast_modules = [1],
        tmp_directory = 'tmp/',
        delete_parser_data = True,
        return_dbc = True,
        batch_process = False,
        outpath = '',
        batch_size = 1000,
        verbose = True,
    ):
        """
        Captures a noise timestream using the parser.

        Parameters:
        fres (array-like): tone frequencies in Hz.
        ares (array-like): tone amplitudes in dBm.
        noise_time (float): timestream length in seconds.
        dec_stage (int): dec_stage frequency downsampling factor.
            6 ->   596.05 Hz
            5 -> 1,192.09 Hz
            4 -> 2,384.19 Hz
            ...
            1 -> 19 kHz
            0 -> 38 kHz, can only be used with 1 module at a time. Make sure
                 active module is module 1.
        fast_modules (array-like): up to 2 modules that you want to run at 38
            or 19 kHz.
        tmp_directory (str): directory to save temporary parser data before
            converting to .npy. Data is streamed to disk, so the drive must be
            fast enough with sufficient free space.
        delete_parser_data (bool): If True, deletes the parser data files
            after importing the data.
        return_dbc (bool): If true, divides the output by the tone power.
        batch_process (bool): If True, processes the noise data in batches.
        outpath (str): path to save the batch data. Data will be saved in
            multiple files with suffices appended to outpath.
        batch_size (int): batch size, in MB.
        verbose (bool): If True, displays a progress bar while taking data.

        Returns:
        z (M X N np.array): first index is channel index and second index is
            complex S21 data point in the timestream.
        """
        # Type checks
        tmp_directory = os.path.normpath(os.path.expanduser(tmp_directory))
        os.makedirs(tmp_directory, exist_ok = True)
        data_directory = os.path.join(tmp_directory, 'parser_data_00')
        if os.path.exists(data_directory):
            raise FileExistsError(f'{data_directory} already exists')
        
        if batch_process and not outpath.endswith('.npy'):
            raise ValueError('outpath must end with .npy')
        if dec_stage > 2:
            module_idxs = list(self.nco_freqs.keys())
        else:
            module_idxs = fast_modules

        fres, ares = np.asarray(fres), np.asarray(ares)
        
        # set dec stage
        if dec_stage ==0:
            # as of 1.5.6, can only use 2 module or else packets drop
            await self.d.set_decimation(0, short=True, module=fast_modules)
        elif dec_stage == 1:
            # don't know if this restriction is necessary for stage 1
            await self.d.set_decimation(1, short=True, module=fast_modules)
        else:
            await self.d.set_decimation(dec_stage)
        self.sample_frequency = util.get_sample_frequency(dec_stage)
        if verbose:
            print(f'dec stage is {await self.d.get_decimation()}')

        # set the tones
        max_ntones = await self.write_tones(fres, ares)
        sleep(1)
        # Collect the data
        channels = '1-' + f'{max_ntones}'
        nframes = int(self.sample_frequency * (noise_time + 1))
        # Need to fine-tune time offset to get exact timestream length 
        args = [
            '-i', self.interface,
            '-d', data_directory,
            '-c', channels, 
            '-s', f'{self.serial_number:04d}',
            '-n', f'{nframes:d}'
            ]
        try:
            parser.main(*args)
        except SystemExit as e:
            raise e 
            code = e.code # parser exists when done
        # Set dec stage back
        await self.d.set_decimation(6)
        # read the data and convert to z
        if not batch_process: 
            # Legacy method 
            # -> larger data files, without batch processing, and slower
            z = [[]] * len(fres)
            for module_idx in module_idxs:
                ntones = len(self.ch_ix_dict[module_idx])
                zi = util.convert_parser_to_z(data_directory, self.serial_number,
                                         module_idx, ntones = ntones,
                                         max_ntones = max_ntones)
                # fres0 = self.fres_dict[module_idx].copy()
                for idx, ch_idx in enumerate(self.ch_ix_dict[module_idx]):
                    z[ch_idx] = zi[idx]
            # Sometimes the number of points is not exact
            data_len = min([len(zi) for zi in z])
            z = np.array([zi[:data_len] for zi in z])
            if delete_parser_data:
                shutil.rmtree(data_directory)
            if return_dbc:
                z /= 10 ** (ares[:, np.newaxis] / 20)
            return z
        
        # batch processing
        scale_factor = (
            rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
        )
        scale_factor = np.array(scale_factor, dtype = np.float64)
        if return_dbc:
            p_scale = 1 / 10 ** (ares[:, np.newaxis].astype(np.float64) / 20)
            scale_factor = scale_factor * p_scale
        np.save(
            outpath.replace('.npy', f'_batch_scale_factor.npy'),
            scale_factor,
        )
        np.save(
            outpath.replace('.npy', f'_batch_tsample.npy'),
            1 / self.sample_frequency,
        )
        util.convert_parser_to_z_batch(data_directory, outpath, self.serial_number,
                      module_idxs, ntones = len(fres),
                      max_ntones = max_ntones,
                      return_dbc = return_dbc, ares = ares,
                      ch_ix_dict = self.ch_ix_dict,
                      batch_size = batch_size)
        if delete_parser_data:
            shutil.rmtree(data_directory)

################################################################################
################## Methods registered to rfmux.ReadoutModule ###################
################################################################################

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def set_nco(module, nco_freqs):
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
async def write_tones(module, nco_freqs, fres_dict, ares_dict):
        """
        Writes an array of tones given frequencies and amplitudes.

        Parameters:
        module (rfmux.ReadoutModule): readout module object.
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz.
        fres_dict (dict): keys (int) are module indices and values (array-like)
            are frequencies in Hz.
        ares_dict (dict): keys (int) are module indices and values (array-like)
            are powers in dBm.
        """
        # Prepare fres and ares
        d = module.crs
        module_idx = module.module
        fres, ares = fres_dict[module_idx], ares_dict[module_idx]
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)

        # Check NCO and input parameters
        try:
            nco = nco_freqs[module_idx]
        except:
            raise Exception('NCO frequency has not been set')
        
        # Dither frequencies 
        fres = take_netanal._safe_concatenate_frequencies(fres, nco)

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
async def sweep(module, nco_freqs, frequencies_dict, ares_dict, sweep_f,
                sweep_z, nsamps = 10, verbose = True,
                pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep and returns the complex S21 value at each
        frequency. Performs sweeps over axis 0 of frequencies simultaneously.

        Parameters:
        module (rfmux.ReadoutModule): readout module object.
        nco_freqs (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz.
        frequencies_dict (dict): keys (int) are module indices and values
            (M X N array-like float) are arrays where the first index M is the
            channel index (max len 1024) and the second index N is the frequency
            in Hz for a single point in the sweep.
        ares_dict (dict): keys (int) are module indices and values
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
        frequencies = np.asarray(frequencies_dict[module_idx])
        ares = np.asarray(ares_dict[module_idx])

        if not len(frequencies):
            return np.array([], dtype = float), np.array([], dtype = complex)
        
        # Dither frequencies per sweep idx
        nco_freq = nco_freqs[module_idx]
        for ch, fres in enumerate(frequencies):
            frequencies[ch] = take_netanal._safe_concatenate_frequencies(fres, 
                                                                       nco_freq)
        
        n_channels, n_points = frequencies.shape
        if len(ares) != n_channels:
            raise ValueError('ares and frequencies are not the same length')

        # Write amplitudes
        fres_dict = {module_idx: [fi[0] for fi in frequencies]}
        await module.write_tones(nco_freqs, fres_dict, ares_dict)

        # Initialize z array
        z = np.empty((n_channels, n_points), dtype = complex)

        pbar = range(n_points)
        if verbose:
            pbar = tqdm(pbar, total = n_points, leave = False)
            pbar.set_description(pbar_description)

        for sweep_idx in pbar:
            # Write frequencies
            async with d.tuber_context() as ctx:
                for ch in range(n_channels):
                    f = frequencies[ch, sweep_idx]
                    ctx.set_frequency(f - nco_freq, channel = ch + 1,
                                      module = module_idx)
                await ctx()
            samples = await d.get_samples(nsamps,
                                          module = module_idx,
                                          average = True) # channel = ??? instead of cutting channels later
            # format and average data
            zi = np.asarray(samples.mean.i) + 1j * np.asarray(samples.mean.q)
            zi = (
                zi[:n_channels]
                * rfmux.core.transferfunctions.VOLTS_PER_ROC
                / np.sqrt(2)
            )
            z[:, sweep_idx] = zi

        # Turn off channels
        await d.clear_channels(module = module_idx)
        sweep_f[module_idx] = frequencies
        sweep_z[module_idx] = z

def validate_configure_system_params(clock_source, full_scale_dbm, 
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
    assert full_scale_dbm
    if not isinstance(analog_bank_high, bool):
        raise TypeError('analog_bank_high must be a boolean value')
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean value')
