import os
import shutil
import warnings
import subprocess
import numpy as np
from time import sleep
from tqdm.auto import tqdm
import rfmux
from .util import find_key_and_index, convert_parser_to_z, get_modules

class CRS:
    def __init__(self, serial_number = 27):
        """
        Initializes the crs object d. Not that the system must be
        configured using CRS.configure_system before measurements

        Parameters:
        serial_number (int): CRS serial number
        """
        self.serial_number = serial_number
        session_str = '!HardwareMap [ !CRS { ' + f'serial: "{serial_number:04d}"'
        session_str += ' } ]'
        s = rfmux.load_session(session_str)
        self.d = s.query(rfmux.CRS).one()
        self.nco_freq_dict = {}

    async def configure_system(self, clock_source="SMA", full_scale_dbm = 7,
                               analog_bank_high = False, verbose = True):
        """
        Resolves the system, sets the timestamp port, sets the clock source, and
        sets the DAC scale

        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal
            voltage controlled crystal oscillator or 'SMA' for the external 10
            MHz reference (reference should be 5 Vpp)
        full_scale_dbm (int): full scale power in dBm. Range is [-18, 7]
        analog_bank_high (bool): if True, uses modules 1-4 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4)
        verbose (bool): If True, gets and prints the clocking source
        """
        # Resolve the system
        await self.d.resolve()
        # Set the timestamp port. bypass if already set
        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted:
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)
        # Set the clock source
        await self.d.set_clock_source(clock_source)
        # Default extended bandwidth to False
        await self.set_extended_module_bandwidth(False)
        # Set the analog bank
        await self.set_analog_bank(analog_bank_high)
        # Set the DAC scale and routing. Routing may be obsolete but I'm not
        # absolutely sure. Might be worth testing and removing
        module_indices = range(5, 9) if analog_bank_high else range(1, 5)
        for module_index in module_indices:
            await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module_index)
            await self.d.set_dac_scale(full_scale_dbm, self.d.UNITS.DBM,
                                       module_index)
        self.full_scale_dbm = full_scale_dbm
        self.d.full_scale_dbm = full_scale_dbm
        # Print configuration if verbose. Sleep to be safe, mostly for routing
        sleep(1)
        if verbose:
            print('System configured')
            print("Clocking source is", await self.d.get_clock_source())

    async def set_analog_bank(self, analog_bank_high):
        """
        Sets the analog bank to either high (modules 5-8) or low (modules 1-4)

        Parameters:
        analog_bank_high (bool): if True, uses modules 1-4 (DAC/ADC 5-8). Else
            uses modules 1-4 (DAC/ADC 1-4)
        """
        await self.d.set_analog_bank(high = analog_bank_high)
        self.analog_bank_high = analog_bank_high

    async def set_extended_module_bandwidth(self, extended):
        """
        Choose between the standard module bandwidth of 500 MHz and the extended
        module bandwidth of 600 MHz. Only extend the bandwidth if you know what
        you are doing. See docstring for crs.d.set_extended_module_bandwidth for
        details

        Parameters:
        extended (bool): If True, extends the bandwidth to 600 MHz. Else
            sets the bandwidth to 500 MHz
        """
        await self.d.set_extended_module_bandwidth(extended)
        self.extended_bw = extended
        if extended:
            warnings.warn(f"Extended module bandwidth set", UserWarning)

    async def set_nco(self, nco_freq_dict, verbose = True):
        """
        Set the NCO frequency

        Parameters:
        nco_freq_dict (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz
        verbose (bool): If True, prints the NCO frequencies after confirming
        """
        modules = get_modules(self.d, list(nco_freq_dict.keys()))
        await modules.set_nco(nco_freq_dict)
        for module_index, nco_set in nco_freq_dict.items():
            nco_meas = await self.d.get_nco_frequency(module = module_index)
            self.nco_freq_dict[module_index] = nco_meas
            if verbose:
                nco_str = f'{round(nco_meas * 1e-6, 6)}'
                print(f'Module {module_index} NCO is {nco_str} MHz')

    async def write_tones(self, fres, ares, return_max_ntones = False):
        """
        Writes an array of tones given frequencies and amplitudes. Splits the
        tones into the appropriate modules using the NCO frequencies.
        Note: this could lead to behavior where one tone jumps between NCOs
        during a measurement run. To mitigate this, we could consider setting
        the tones for each NCO manually, or shifting the NCOs with the
        resonances

        Parameters:
        fres (array-like): tone frequencies in Hz
        ares (array-like): tone powers in dBm
        return_max_ntones (bool): If True, returns 
            max_ntones (int): maximum number of tones on any given module
        """
        # Split fres and ares into dictionaries
        if not len(self.nco_freq_dict):
            raise Exception("NCO frequencies are not set")
        channel_indices = list(range(len(fres)))
        self.fres_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ares_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ch_ix_dict = {key: [] for key in self.nco_freq_dict.keys()}
        for ch_ix, fr, ar in zip(channel_indices, fres, ares):
            module_index = min(self.nco_freq_dict,
                             key = lambda k: np.abs(self.nco_freq_dict[k] - fr))
            self.fres_dict[module_index].append(fr)
            self.ares_dict[module_index].append(ar)
            self.ch_ix_dict[module_index].append(ch_ix)
        self.fres_dict = {key: np.array(value) for key, value in self.fres_dict.items()}
        self.ares_dict = {key: np.array(value) for key, value in self.ares_dict.items()}
        self.ch_ix_dict = {key: np.array(value) for key, value in self.ch_ix_dict.items()}
        # Confirm that the tones are within the NCO bandwidths
        for module_index in self.nco_freq_dict.keys():
            bw_half = 312.5e6 if self.extended_bw else 250e6
            diffs = self.fres_dict[module_index] - self.nco_freq_dict[module_index]
            if any(np.abs(diffs) > bw_half):
                err = f'All of fres must be within {round(bw_half / 1e6, 1)} '
                err += 'MHz of an NCO frequency'
                raise ValueError(err)
        # Write tones
        modules = get_modules(self.d, list(self.fres_dict.keys()))
        await modules.write_tones(self.nco_freq_dict, self.fres_dict,
                                  self.ares_dict)
        if return_max_ntones:
            max_ntones = max([len(f) for f in self.fres_dict.values()])
            return max_ntones 

    async def sweep(self, frequencies, ares, nsamps = 10, verbose = True,
                    pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep and returns the complex S21 value at each
        frequency. Performs sweeps over axis 0 of frequencies simultaneously

        Parameters:
        frequencies (M X N array-like float): the first index M is the channel
            index (max len 1024) and the second index N is the frequency in Hz
            for a single point in the sweep
        ares (M array-like float): amplitudes in dBm for each channel
        nsamps (int): number of samples to average per point
        verbose (bool): If True, displays a progress bar while sweeping
        pbar_description (str): description for the progress bar
        """
        frequencies, ares = np.asarray(frequencies), np.asarray(ares)
        if not len(self.nco_freq_dict):
            raise Exception("NCO frequencies are not set")
        # Split frequencies and ares into dictionaries
        channel_indices = list(range(len(frequencies)))
        self.frequencies_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ares_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ch_ix_dict = {key: [] for key in self.nco_freq_dict.keys()}
        select_nco = lambda k: max([np.abs(self.nco_freq_dict[k] - fr) for fr in [max(freqs), min(freqs)]])
        for ch_ix, freqs, ar in zip(channel_indices, frequencies, ares):
            module_index = min(self.nco_freq_dict, key = select_nco)
            self.frequencies_dict[module_index].append(freqs)
            self.ares_dict[module_index].append(ar)
            self.ch_ix_dict[module_index].append(ch_ix)
        self.frequencies_dict = {key: np.array(value) for key, value in self.frequencies_dict.items()}
        self.ares_dict = {key: np.array(value) for key, value in self.ares_dict.items()}
        self.ch_ix_dict = {key: np.array(value) for key, value in self.ch_ix_dict.items()}
        # Confirm that frequencies are in each NCO bandwidth
        for module_index in self.nco_freq_dict.keys():
            bw_half = 312.5e6 if self.extended_bw else 250e6
            diffs = self.frequencies_dict[module_index] - self.nco_freq_dict[module_index]
            if any(np.abs(diffs).flatten() > bw_half):
                err = 'All of frequencies must be within '
                err += f'{round(bw_half / 1e6, 1)} MHz of an NCO frequency'
                raise ValueError(err)

        # Set fir_stage
        fir_stage = 6
        await self.d.set_fir_stage(fir_stage)
        # Sweep
        sweep_f, sweep_z = {}, {}
        modules = get_modules(self.d, list(self.frequencies_dict.keys()))
        await modules.sweep(self.nco_freq_dict, self.frequencies_dict,
                            self.ares_dict, sweep_f, sweep_z, nsamps = nsamps,
                            verbose = verbose,
                            pbar_description = pbar_description)
        # Create f, z from sweep results
        nres = frequencies.shape[0]
        f = np.empty(frequencies.shape, dtype = float)
        z = np.empty(frequencies.shape, dtype = complex)
        for res_index in range(nres):
            module_index, ch_index = find_key_and_index(self.ch_ix_dict,
                                                        res_index)
            f[res_index] = sweep_f[module_index][ch_index]
            z[res_index] = sweep_z[module_index][ch_index]
        z /= 10 ** (ares[:, np.newaxis] / 20)
        return f, z

    async def sweep_linear(self, fres, ares, bw = 20e3, npoints = 10,
                           nsamps = 10, verbose = True,
                           pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where each channel is swept over the same
        frequency span

        Parameters:
        fres (array-like): center frequencies in Hz
        ares (array-like): amplitudes in dBm
        bw (float): span around each frequency to sweep in Hz
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point
        verbose (bool): If True, displays a progress bar while sweeping
        pbar_description (str): description for the progress bar

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index
            and N is the index of each point in the sweep
        z (M X N np.array): array of complex S21 data corresponding to f
        """
        fres, ares = np.asarray(fres), np.asarray(ares)
        f = np.linspace(fres + bw / 2, fres - bw / 2, npoints).T
        f, z = await self.sweep(f, ares, nsamps = nsamps, verbose = verbose,
                                pbar_description = pbar_description)
        return f, z

    async def sweep_qres(self, fres, ares, qres, npoints = 10, nsamps = 10,
                         verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where the span around each frequency is set
        equal to fres / qres

        Parameters:
        fres (array-like): center frequencies in Hz
        ares (array-like): amplitudes in dBm
        qres (arrz_noise_dict(int): number of sweep points per channel
        nsamps (int): number of samples to average per point
        verbose (bool): If True, displays a progress bar while sweeping
        pbar_description (str): description for the progress bar

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index
            and N is the index of each point in the sweep
        z (M X N np.array): array of complex S21 data corresponding to f
        """
        fres, ares, qres = np.asarray(fres), np.asarray(ares), np.asarray(qres)
        spans = fres / qres
        f = np.linspace(fres + spans / 2, fres - spans / 2, npoints).T
        f, z = await self.sweep(f, ares, nsamps = nsamps, verbose = verbose,
                                pbar_description = pbar_description)
        return f, z

    async def sweep_full(self, amplitude, npoints = 10, nsamps = 10,
                         verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep over the full bandwidth around the NCO
        frequency

        Parameters:
        amplitude (float): amplitude in dBm
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point
        verbose (bool): If True, displays a progress bar while sweeping
        pbar_description (str): description for the progress bar

        Returns:
        f (np.array): array of frequencies in Hz
        z (np.array): array of complex S21 data corresponding to f
        """
        ncos = list(self.nco_freq_dict.values())
        bw_total = 625e6 if self.extended_bw else 500e6
        bw = bw_total / 1024 + 200
        spacing = bw / npoints
        fres = np.concatenate([np.linspace(nco - bw_total / 2 + 10 + bw, 
                                           nco + bw_total / 2 - 10 - bw, 
                                           1024) for nco in ncos])
        ares = amplitude * np.ones(len(fres))
        # Left off here
        f, z = await self.sweep_linear(fres, ares, bw = bw - spacing,
                                       npoints = npoints, nsamps = nsamps,
                                       verbose = verbose,
                                       pbar_description = pbar_description)
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_noise(self, fres, ares, noise_time, fir_stage = 6,
                            parser_loc='/home/daq1/github/rfmux/firmware/r1.5.5/parser',
                            interface='enp2s0', delete_parser_data = True,
                            verbose = True):
        """
        Captures a noise timestream using the parser.

        Parameters:
        fres (array-like): tone frequencies in Hz
        ares (array-like): tone amplitudes in dBm
        noise_time (float): timestream length in seconds
        fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz
            5 -> 1,192.09 Hz
            4 -> 2,384.19 Hz, will drop some packets
        parser_loc (str): path to the parser file
        interface (str): Ethernet interface identifier
        delete_parser_data (bool): If True, deletes the parser data files
            after importing the data
        verbose (bool): If True, displays a progress bar while taking data

        Returns:
        z (M X N np.array): first index is channel index and second index is
            complex S21 data point in the timestream
        """
        module_indices = list(self.nco_freq_dict.keys())
        if fir_stage <= 4:
            warnings.warn(f"packets will drop if fir_stage < 5", UserWarning)
        fres, ares = np.asarray(fres), np.asarray(ares)
        os.makedirs('tmp/', exist_ok = True)
        data_path = 'tmp/parser_data_00/'
        if os.path.exists(data_path):
            raise FileExistsError(f'{data_path} already exists')
        # set fir stage
        await self.d.set_fir_stage(fir_stage)
        self.sample_frequency = 625e6 / (256 * 64 * 2 ** fir_stage)
        if verbose:
            print(f'fir stage is {await self.d.get_fir_stage()}')

        # set the tones
        max_ntones = await self.write_tones(fres, ares, return_max_ntones = True)
        sleep(1)
        # Collect the data
        num_samps = int(self.sample_frequency*(noise_time + 10))
        channels = '1-' + f'{max_ntones}'
        parser = subprocess.Popen([parser_loc, '-c', channels, '-d', data_path,
                                   '-i', interface, '-s', f'{self.serial_number:04d}',
                                   '-n', str(num_samps)], shell=False)
        pbar = list(range(int(noise_time) + 20))
        if verbose:
            pbar = tqdm(pbar, leave = False)
        for i in pbar:
            sleep(1)
        # Set fir stage back
        await self.d.set_fir_stage(6)
        # read the data and convert to z
        z = [[]] * len(fres)
        for module_index in module_indices:
            zi = convert_parser_to_z(data_path, self.serial_number, module_index,
                                     ntones = len(self.ch_ix_dict[module_index]))
            fres0 = self.fres_dict[module_index].copy()
            for index, ch_index in enumerate(self.ch_ix_dict[module_index]):
                z[ch_index] = zi[index]
        # Sometimes the number of points is not exact
        data_len = min([len(zi) for zi in z])
        z = np.array([zi[:data_len] for zi in z])
        if delete_parser_data:
            shutil.rmtree('tmp/')
        z /= 10 ** (ares[:, np.newaxis] / 20)
        return z

    async def capture_fast_noise(self, frequency, amplitude, time = 1,
                                 nsegments = 10, verbose = False):
        """
        Captures noise with a 2.44 MHz sample rate on a single channel. Turns on
        only a single channel to avoid noise spikes from neighboring channels.
        Note that the output will have to be corrected for the nonlinear PFB bin
        after taking a PSD. Temporarily changes the NCO frequency to center the
        tone on a PFB bin

        Parameters:
        frequency (float): tone frequency in Hz
        amplitude (float): tone amplitude in dBm
        time (float): timestream length in s. Max is 4 s
        nsegments (int): number of sequential timestreams to capture and average
            linearly over
        verbose (bool): if True, prints NCO frequency settings
        """
        warnings.warn('This function is a work in progress, and likely will not output what you expect', UserWarning)
        # Set up parameters for noise capture
        select_nco = key = lambda k: np.abs(self.nco_freq_dict[k] - frequency)
        module_index = min(self.nco_freq_dict, select_nco)
        bw_half = 312.5e6 if self.extended_bw else 250e6
        if np.abs(frequency - self.nco_freq_dict[module_index] > bw_half):
            err = f'Frequency must be within {round(bw_half / 1e6, 1)} MHz of '
            err += 'an NCO frequency'
            raise ValueError(err)
        fsample = 625e6 / 256
        nsamps = int(time * fsample)
        if nsamps > 1e7:
            raise ValueError('Time must be less than 4 s')
        # Capture samples
        samples = await self.d.get_pfb_samples(int(nsamps), channel = 1,
                                                   module = module_index,
                                                   binlim = 1e6, trim = True,
                                                   nsegments = nsegments,
                                                   reference = 'relative', # dBc / Hz
                                                   reset_NCO = True, # shifts NCO to bin center
                                                   )
        f = np.asarray(samples.spectrum.freq_iq)
        z = np.asarray(samples.spectrum.psd_i + 1j * samples.spectrum.psd_q)
        z = np.array(samples.i + 1j * samples.q)
        
        # This might be already applied with reference = True
        # z *= rfmux.core.utils.transferfunctions.VOLTS_PER_ROC / np.sqrt(2)
        # z /= 10 ** (ares[:, np.newaxis] / 20)
        return None, None
        return f, z
    
################################################################################
################## Methods registered to rfmux.ReadoutModule ###################
################################################################################

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def set_nco(module, nco_freq_dict):
        """Set the NCO frequency

        Parameters:
        module (rfmux.ReadoutModule): readout module object
        nco_freq_dict (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz. This should not be a round number
        """
        d = module.crs
        module_index = module.module
        nco_freq = nco_freq_dict[module_index]
        await d.set_nco_frequency(nco_freq, module = module_index)

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def write_tones(module, nco_freq_dict, fres_dict, ares_dict):
        """
        Writes an array of tones given frequencies and amplitudes

        Parameters:
        module (rfmux.ReadoutModule): readout module object
        nco_freq_dict (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz
        fres_dict (dict): keys (int) are module indices and values (array-like)
            are frequencies in Hz
        ares_dict (dict): keys (int) are module indices and values (array-like)
            are powers in dBm
        """
        # Prepare fres and ares
        d = module.crs
        module_index = module.module
        fres, ares = fres_dict[module_index], ares_dict[module_index]
        fres = np.asarray(fres)
        ares = np.asarray(ares)
        # Randomize frequencies a little. This might be unneccesary but I kept it in to be safe
        ix = [i for i in range(len(fres)) if i not in [np.argmin(fres), np.argmax(fres)]]
        # don't randomize lowest and highest frequency to avoid exceeding bandwidth
        fres[ix] += np.random.uniform(-50, 50, len(fres) - 2)
        comb_sampling_freq =rfmux.core.utils.transferfunctions.COMB_SAMPLING_FREQ
        threshold = 101.
        fres[fres%(comb_sampling_freq/512) < threshold] += threshold
        # Check NCO and input parameters
        try:
            nco = nco_freq_dict[module_index]
        except:
            raise Exception('NCO frequency has not been set')
        if any(ares > d.full_scale_dbm):
            err = f'ares must not exceed {d.full_scale_dbm} dBm: raise '
            err += 'full_scale_dbm or lower powers'
            raise ValueError(err)
        if any(ares < -60) and len(ares) < 100:
            err = f"values in ares are < 60 dBm: digitization noise may occur"
            warnings.warn(err, UserWarning)
        ares_amplitude = 10 ** ((ares - d.full_scale_dbm) / 20)

        await d.clear_channels(module = module_index)

        async with d.tuber_context() as ctx:
            for ch, (fr, ar) in enumerate(zip(fres, ares_amplitude)):
                ctx.set_frequency(fr - nco, channel = ch + 1,
                                  module = module_index)
                ctx.set_amplitude(ar, channel = ch+1, module = module_index)
            await ctx()

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def sweep(module, nco_freq_dict, frequencies_dict, ares_dict, sweep_f,
                sweep_z, nsamps = 10, verbose = True,
                pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep and returns the complex S21 value at each
        frequency. Performs sweeps over axis 0 of frequencies simultaneously

        Parameters:
        module (rfmux.ReadoutModule): readout module object
        nco_freq_dict (dict): keys (int) are module indices and values (float)
            are NCO frequencies in Hz
        frequencies_dict (dict): keys (int) are module indices and values
            (M X N array-like float) are arrays where the first index M is the
            channel index (max len 1024) and the second index N is the frequency
            in Hz for a single point in the sweep
        ares_dict (dict): keys (int) are module indices and values
            (M array-like float) are amplitudes in dBm for each channel
        nsamps (int): number of samples to average per point
        verbose (bool): If True, displays a progress bar while sweeping
        pbar_description (str): description for the progress bar

        Returns:
        z (M X N array-like complex): complex S21 data in V for each frequency
            in f
        """
        d = module.crs
        module_index = module.module
        frequencies = np.asarray(frequencies_dict[module_index])
        ares = np.asarray(ares_dict[module_index])

        if not len(frequencies):
            return np.array([], dtype = float), np.array([], dtype = complex)
        nco_freq = nco_freq_dict[module_index]
        n_channels, n_points = frequencies.shape
        if len(ares) != n_channels:
            raise ValueError('ares and frequencies are not the same length')

        # Write amplitudes
        fres_dict = {module_index: [fi[0] for fi in frequencies]}
        await module.write_tones(nco_freq_dict, fres_dict, ares_dict)
        # Initialize z array
        z = np.empty((n_channels, n_points), dtype = complex)

        pbar = range(n_points)
        if verbose:
            pbar = tqdm(pbar, total = n_points, leave = False)
            pbar.set_description(pbar_description)

        for sweep_index in pbar:
            # Write frequencies
            async with d.tuber_context() as ctx:
                for ch in range(n_channels):
                    f = frequencies[ch, sweep_index]
                    ctx.set_frequency(f - nco_freq, channel = ch + 1,
                                      module = module_index)
                await ctx()
            nsamples_discard = 0 # 15
            samples = await d.py_get_samples(nsamps + nsamples_discard,
                                             module = module_index,
                                             average = True)
            # format and average data
            zi = np.asarray(samples.mean.i) + 1j * np.asarray(samples.mean.q)
            zi = zi[:n_channels] * rfmux.core.utils.transferfunctions.VOLTS_PER_ROC / np.sqrt(2)
            z[:, sweep_index] = zi

        # Turn off channels
        await d.clear_channels(module = module_index)
        sweep_f[module_index] = frequencies
        sweep_z[module_index] = z
