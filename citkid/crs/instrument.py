# import sys
# sys.path.append('/home/daq1/github/rfmux/')

import os
import rfmux
import shutil
import warnings
import subprocess
import numpy as np
from time import sleep
from tqdm.auto import tqdm
from .util import volts_to_dbm, remove_internal_phaseshift, convert_parser_to_z
from .util import volts_per_roc
from .util import find_key_and_index
from hidfmux.core.utils.transferfunctions import apply_pfb_correction
from hidfmux.core.utils import transferfunctions
from hidfmux.analysis.noise_processing import separate_iq_fft_to_i_and_q
 
class CRS:
    def __init__(self, crs_sn = 27, splitter = True): 
        """ 
        Initializes the crs object d. Not that the system must be 
        configured using CRS.configure_system before measurements 

        Parameters:
        crs_sn (int): CRS serial number 
        splitter (bool): If True, the splitter is attached. This is used to calculate
            power
        """ 
        self.crs_sn = crs_sn
        self.splitter = splitter
        s = rfmux.load_session('!HardwareMap [ !CRS { ' + f'serial: "{crs_sn:04d}"' + ' } ]')
        self.d = s.query(rfmux.CRS).one()
        self.volts_per_roc = volts_per_roc
        self.d.volts_per_roc = volts_per_roc
        self.nco_freq_dict = {}
        
    async def configure_system(self, clock_source="SMA", full_scale_dbm = 1, analog_bank_high = False,
                               verbose = True):
        """
        Resolves the system, sets the timestamp port, sets the clock source, and sets the full scale
        in dBm 
        
        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal voltage controlled
            crystal oscillator or 'SMA' for the external 10 MHz reference (5 Vpp I think) 
        full_scale_dbm (int): full scale power in dBm. range is [???, 7?]
        analog_bank_high (bool): if True, uses modules 1-4 = DAC/ADC 5-8. Else uses modules 1-4 = DAC/ADC 1-4
        verbose (bool): If True, gets and prints the clocking source 
        """
        await self.d.resolve()

        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted: 
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)  
        
        await self.d.set_clock_source(clock_source)

        await self.d.set_analog_bank(high = analog_bank_high)
        self.analog_bank_high = analog_bank_high
        for module_index in range(1, 5):
            await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module_index)
            m = module_index
            if self.analog_bank_high:
                m += 4
            await self.d.set_dac_scale(full_scale_dbm, self.d.UNITS.DBM, m)
        self.full_scale_dbm = full_scale_dbm
        self.d.full_scale_dbm = full_scale_dbm
                
        sleep(1)
        if verbose:
            print('System configured')
            print("Clocking source is", await self.d.get_clock_source())

        
    async def set_nco(self, nco_freq_dict, verbose = True):
        """Set the NCO frequency
        
        Parameters:
        module (int): module index
        nco_freq_dict (dict): keys (int) are module indices and values (float) are 
            NCO frequencies in Hz. These should not be round numbers
        verbose (bool): If True, gets and prints the NCO frequencies
        """
        modules = list(nco_freq_dict.keys())
        if any([m not in [1,2,3,4] for m in modules]):
            raise ValueError(f'modules must be in range [1, 4]')
        if any([nco > 2.5e9 - 325e6 for nco in nco_freq_dict.values()]):
            raise ValueError('NCOs must be less than 2.175 GHz to avoid Nyquist reflections')
        modules = list(nco_freq_dict.keys())
        await self.d.modules.filter(rfmux.ReadoutModule.module.in_(modules)).set_nco(nco_freq_dict, 
                self.analog_bank_high)
        for key, value in nco_freq_dict.items(): 
            if self.analog_bank_high:
                i = 4 
            else:
                i = 0
            self.nco_freq_dict[key] = await self.d.get_nco_frequency(self.d.UNITS.HZ, module = key + i)
            if verbose:
                print(f'Module {key} NCO is {round(value * 1e-6, 6)} MHz') 
        
    async def write_tones(self, fres, ares):
        """
        Writes an array of tones given frequencies and amplitudes. Splits the 
        tones into the appropriate modules using the NCO frequencies 

        Parameters:
        fres (array-like): tone frequencies in Hz 
        ares (array-like): tone powers in dBm (+- 2 dBm precision for now) 
        """ 
        # Split fres and ares into dictionaries 
        if not len(self.nco_freq_dict):
            raise Exception("NCO frequencies are not set") 
        channel_indices = list(range(len(fres))) 
        self.fres_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ares_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ch_ix_dict = {key: [] for key in self.nco_freq_dict.keys()}
        for ch_ix, fr, ar in zip(channel_indices, fres, ares):
            module_index = min(self.nco_freq_dict, key = lambda k: np.abs(self.nco_freq_dict[k] - fr)) 
            self.fres_dict[module_index].append(fr) 
            self.ares_dict[module_index].append(ar) 
            self.ch_ix_dict[module_index].append(ch_ix)
        self.fres_dict = {key: np.array(value) for key, value in self.fres_dict.items()}
        self.ares_dict = {key: np.array(value) for key, value in self.ares_dict.items()}
        self.ch_ix_dict = {key: np.array(value) for key, value in self.ch_ix_dict.items()}
        for module_index in self.nco_freq_dict.keys():
            if any(np.abs(self.fres_dict[module_index] - self.nco_freq_dict[module_index]) > 325e6):
                raise ValueError('All of fres must be within 325 MHz of an NCO frequency') 
        # Write tones 
        modules = list(self.fres_dict.keys())
        await self.d.modules.filter(rfmux.ReadoutModule.module.in_(modules)).write_tones(self.nco_freq_dict, self.fres_dict, 
                                                                                         self.ares_dict)

    async def sweep(self, frequencies, ares, nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep and returns the complex S21 value at each frequency. Performs sweeps over 
        axis 0 of frequencies simultaneously 

        Parameters:
        frequencies (M X N array-like float): the first index M is the channel index (max len 1024) and the second index 
            N is the frequency in Hz for a single point in the sweep
        ares (M array-like float): amplitudes in dBm for each channel 
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 
        """ 
        frequencies, ares = np.asarray(frequencies), np.asarray(ares)
        # Split frequencies and ares into dictionaries 
        if not len(self.nco_freq_dict):
            raise Exception("NCO frequencies are not set")  

        channel_indices = list(range(len(frequencies))) 
        self.frequencies_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ares_dict = {key: [] for key in self.nco_freq_dict.keys()}
        self.ch_ix_dict = {key: [] for key in self.nco_freq_dict.keys()}
        for ch_ix, freqs, ar in zip(channel_indices, frequencies, ares):
            module_index = min(self.nco_freq_dict, key = lambda k: max([np.abs(self.nco_freq_dict[k] - fr) for fr in [max(freqs), min(freqs)]])) 
            self.frequencies_dict[module_index].append(freqs) 
            self.ares_dict[module_index].append(ar) 
            self.ch_ix_dict[module_index].append(ch_ix)
        self.frequencies_dict = {key: np.array(value) for key, value in self.frequencies_dict.items()}
        self.ares_dict = {key: np.array(value) for key, value in self.ares_dict.items()}
        self.ch_ix_dict = {key: np.array(value) for key, value in self.ch_ix_dict.items()}
        for module_index in self.nco_freq_dict.keys():
            if any(np.abs(self.frequencies_dict[module_index] - self.nco_freq_dict[module_index]).flatten() > 325e6):
                raise ValueError('All of frequencies must be within 325 MHz of an NCO frequency') 
                
        # Set fir_stage
        fir_stage = 6 
        await self.d.set_fir_stage(fir_stage) 
        # Sweep 
        modules = list(self.frequencies_dict.keys())
        sweep_f, sweep_z = {}, {}
        await self.d.modules.filter(rfmux.ReadoutModule.module.in_(modules)).sweep(self.nco_freq_dict, self.frequencies_dict, self.ares_dict, 
                                                                                   sweep_f, sweep_z,
                                                                                   nsamps = nsamps, verbose = verbose, 
                                                                                   pbar_description = pbar_description)
        # Create f, z from sweep results 
        nres = frequencies.shape[0]
        f = np.empty(frequencies.shape, dtype = float)
        z = np.empty(frequencies.shape, dtype = complex)
        for res_index in range(nres):
            module_index, ch_index = find_key_and_index(self.ch_ix_dict, res_index) 
            f[res_index] = sweep_f[module_index][ch_index]
            z[res_index] = sweep_z[module_index][ch_index]
        z /= 10 ** (ares[:, np.newaxis] / 20)
        if self.splitter:
            z /= 10 ** (-10.5 * 2 / 20) # 10.5 dB loss in either direction 
        return f, z

    async def sweep_linear(self, fres, ares, bw = 20e3, npoints = 10, 
                           nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where each channel is swept over the same range in 
        frequencies

        Parameters:
        fres (array-like): center frequencies in Hz 
        ares (array-like): amplitudes in dBm
        bw (float): span around each frequency to sweep in Hz 
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index and 
            N is the index of each point in the sweep 
        z (M X N np.array): array of complex S21 data corresponding to f 
        """ 
        fres, ares = np.asarray(fres), np.asarray(ares)
        f = np.linspace(fres + bw / 2, fres - bw / 2, npoints).T
        f, z = await self.sweep(f, ares, nsamps = nsamps, 
                            verbose = verbose, pbar_description = pbar_description)
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
        f (M X N np.array): array of frequencies where M is the channel index and 
            N is the index of each point in the sweep 
        z (M X N np.array): array of complex S21 data corresponding to f 
        """ 
        fres, ares, qres = np.asarray(fres), np.asarray(ares), np.asarray(qres)
        spans = fres / qres 
        f = np.linspace(fres + spans / 2, fres - spans / 2, npoints).T
        f, z = await self.sweep(f, ares, nsamps = nsamps, 
                            verbose = verbose, pbar_description = pbar_description)
        return f, z

    async def sweep_full(self, amplitude, npoints = 10, 
                         nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep over the full 600 MHz bandwidth around the NCO 
        frequency

        Parameters:
        amplitude (float): amplitude in dBm
        npoints (int): number of sweep points per channel
        nsamps (iz_noise_dict
        Returns:
        f (np.array): array of frequencies in Hz 
        z (np.array): array of complex S21 data corresponding to f 
        """ 
        ncos = list(self.nco_freq_dict.values())
        bw = 600e6 / 1024 + 200
        spacing = bw / npoints 
        fres = np.concatenate([np.linspace(nco - 300e6 + 10 + bw, nco + 300e6 - 10 - bw, 1024) for nco in ncos])
        ares = amplitude * np.ones(len(fres))
        f, z = await self.sweep_linear(fres, ares, bw = bw - spacing, npoints = npoints,
                                      nsamps = nsamps, verbose = verbose, 
                                      pbar_description = pbar_description)
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_fast_noise(self, frequency, amplitude, time, verbose = False):
        """ 
        Captures noise with a 2.44 MHz sample rate on a single channel. Turns on only 
        a single channel to avoid noise spikes from neighboring channels. Note that 
        the output will have to be corrected for the nonlinear PFB bin after taking a 
        PSD. It iz_noise_dicts harder to correct the timestream so single-photon events and 
        cosmic rays will not have the correct shape. 
        Temporarily changes the NCO frequency to center the tone on a PFB bin 
        
        Parameters:
        frequency (float): tone frequency in Hz 
        amplitude (float): tone amplitude in dBm 
        nsamps (int): number of samples. Max is 1e6
        verbose (bool): if True, prints NCO frequency settings 
        """
        module_index = min(self.nco_freq_dict, key = lambda k: np.abs(self.nco_freq_dict[k] - frequency)) 
        if np.abs(frequency - self.nco_freq_dict[module_index] > 300e6):
            raise ValueError('Frequency must be within 300 MHz of an NCO frequency') 
        fsample = 625e6 / 256
        nsamps = int(time * fsample) 
        # Adjust NCO to center the frequency on a PFB bin to minimize nonlinearity
        comb_sampling_freq = 625e6
        bin_centers = np.arange(-256, 256, 1) * comb_sampling_freq / 512
        nco_freq0 = self.nco_freq_dict[module_index] 
        frd = frequency - nco_freq0
        ix = np.argmin(np.abs(frd - bin_centers)) 
        center_offset_freq = frd - bin_centers[ix] 
        nco_freq_dict_temp = {module_index: nco_freq0 + center_offset_freq}
        await self.sz_noise_dict.set_dmfd_routing(self.d.ROUTING.ADC, module_index)
        sleep(1)
        pfb_samples = await self.d.get_pfb_samples(int(nsamps), 'RAW', 1, module_index) # May have 20% exact difference 
        pfb_samples = np.array([complex(*sample) for sample in pfb_samples])
        await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, module_index)
        
        fraw, fft_corr_raw, builtin_gain_factor, pfb_sample_len =\
            apply_pfb_correction(pfb_samples, self.nco_freq_dict[module_index], frequency, binlim = 1.1e6, trim=True)
        cal_samples = await self.d.get_pfb_samples(2100, 'RAW', 1, module_index)
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module_index)
        cal_samples = np.array([complex(*sample) for sample in cal_samples][100:])
        fcal, fft_corr_cal, builtin_gain_factor_cal, pfb_sample_len_cal =\
            apply_pfb_correction(cal_samples, self.nco_freq_dict[module_index], frequency, binlim = 1.1e6, trim=True)

        ifft_raw, qfft_raw = [np.fft.fftshift(x) for x in separate_iq_fft_to_i_and_q(np.fft.fftshift(fft_corr_raw))]
        zraw = ifft_raw + 1j * qfft_raw
        ifft_cal, qfft_cal = [np.fft.fftshift(x) for x in separate_iq_fft_to_i_and_q(np.fft.fftshift(fft_corr_cal))]
        zcal = ifft_cal + 1j * qfft_cal
        zcal = np.mean(zcal)

        # Max of nsamps is 1e5 
        z = remove_internal_phaseshift(frequency, zraw, zcal) * self.volts_per_roc
        # Adjust NCO back to its original value 
        nco_freq_dict_temp = {module_index: nco_freq0}
        self.set_nco(nco_freq_dict_temp, verbose = verbose)
        z /= 10 ** (ares[:, np.newaxis] / 20)
        return fraw, z

    async def capture_noise(self, fres, ares, noise_time, fir_stage = 6,
                            parser_loc='/home/daq1/github/citkid/citkid/crs/parser',
                            interface='enp2s0', delete_parser_data = False,
                            verbose = True, return_raw = False):
        """
        Captures a noise timestream using the parser.
        
        Parameters:
        fres (array-like): tone frequencies in Hz 
        ares (array-like): tone amplitudes in dBm 
        fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz 
            5 -> 1,192.09 Hz 
            4 -> 2,384.19 Hz, will drop some packets
        parser_loc (str): path to the parser file 
        data_path (str): path to the data output file for the parser 
        interface (str): Ethernet interface identifier 
        delete_parser_data (bool): If True, deletes the parser data files 
            after importing the data 
        return_raw (bool): if True, also returns raw z data and calibration data
        
        Returns:
        z (M X N np.array): first index is channel index and second index is complex S21 data 
            point in the timestream 
        zcal (M X N np.array): calibration data 
        zraw (M X N np.array): raw data before applying the calibration 
        """
        module_indices = list(self.nco_freq_dict.keys()) 
        modules = [self.d.modules[module_index - 1] for module_index in module_indices] 
        if fir_stage <= 4:
            warnings.warn(f"packets will drop if fir_stage < 5", UserWarning)
        fres, ares = np.asarray(fres), np.asarray(ares) 
        os.makedirs('tmp/', exist_ok = True)
        data_path = 'tmp/parser_data_00/'
        if os.path.exists(data_path):
            raise FileExistsError(f'{data_path} already exists')
        # set fir stage
        await self.d.set_fir_stage(fir_stage) # Probably will drop packets after 4
        # py_get_samples will error if fir_stage is too low, but parser will not error
        self.sample_frequency = 625e6 / (256 * 64 * 2 ** fir_stage) 
        if verbose:
            print(f'fir stage is {await self.d.get_fir_stage()}')
    
        # set the tones
        await self.write_tones(fres, ares)
        sleep(1)
        # Get the calibration data 
        noise_zcal_dict = {}
        modules = list(self.ch_ix_dict.keys())
        await self.d.modules.filter(rfmux.ReadoutModule.module.in_(modules)).get_noise_cal(self.fres_dict, noise_zcal_dict)
        sleep(0.1)
        # sleep(10)
        # raise Exception('Make this sleep statement longer')
        # Collect the data 
        num_samps = int(self.sample_frequency*(noise_time + 10))
        parser = subprocess.Popen([parser_loc, '-d', data_path, '-i', interface, '-s', 
                                   f'{self.crs_sn:04d}', '-n', str(num_samps)], 
                                   shell=False)
        pbar = list(range(int(noise_time) + 20))
        if verbose:
            pbar = tqdm(pbar, leave = False)
        for i in pbar:
            sleep(1) 
        # Set fir stage back
        await self.d.set_fir_stage(6)
        # read the data and convert to z 
        nres = len(fres)
        z = [[]] * nres
        zraw = [[]] * nres 
        zcal = [[]] * nres 
        for module_index in module_indices:
            zrawi = convert_parser_to_z(data_path, self.crs_sn, module_index, ntones = len(self.ch_ix_dict[module_index])) 
            zi = remove_internal_phaseshift(self.fres_dict[module_index][:, np.newaxis], zrawi, noise_zcal_dict[module_index][:, np.newaxis])
            for index, ch_index in enumerate(self.ch_ix_dict[module_index]):
                z[ch_index] = zi[index]
                zraw[ch_index] = zrawi[index]
                zcal[ch_index] = noise_zcal_dict[module_index][index]
        # Sometimes the number of points is not exact
        data_len = min([len(zi) for zi in z]) 
        z = np.array([zi[:data_len] for zi in z])
        zraw = np.array([zi[:data_len] for zi in zraw])
        if delete_parser_data:
            shutil.rmtree('tmp/')

        z /= 10 ** (ares[:, np.newaxis] / 20)
        if self.splitter:
            z /= 10 ** (-10.5 * 2 / 20) 
        if return_raw:
            return z, zcal, zraw 
        return z
    
######################################################################################################
################################ Methods added to rfmux.ReadoutModule ################################
######################################################################################################

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def set_nco(module, nco_freq_dict, analog_bank_high):
        """Set the NCO frequency
        
        Parameters:
        module (rfmux.ReadoutModule): readout module object
        nco_freq_dict (dict): keys (int) are module indices and values (float) are 
            NCO frequencies in Hz. This should not be a round number
        """
        d = module.crs
        module_index = module.module
        nco_freq = nco_freq_dict[module_index] 
        if analog_bank_high:
            module_index += 4
        await d.set_nco_frequency(nco_freq, d.UNITS.HZ, module = module_index)

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def write_tones(module, nco_freq_dict, fres_dict, ares_dict):
        """
        Writes an array of tones given frequencies and amplitudes 

        Parameters:
        module (rfmux.ReadoutModule): readout module object
        nco_freq_dict (dict): keys (int) are module indices and values (float) are 
            NCO frequencies in Hz
        fres_dict (dict): keys (int) are module indices and values (array-like) are 
            frequencies in Hz 
        ares_dict (dict): keys (int) are module indices and values (array-like) are 
            powers in dBm (+- 2 dBm precision for now)
        """
        # Prepare fres and ares
        d = module.crs
        module_index = module.module 
        fres, ares = fres_dict[module_index], ares_dict[module_index]
        fres = np.asarray(fres) 
        ares = np.asarray(ares)
        # Randomize frequencies a little
        fres += np.random.uniform(-50, 50, fres.shape)
        comb_sampling_freq = transferfunctions.get_comb_sampling_freq()
        threshold = 101.
        fres[fres%(comb_sampling_freq/512) < threshold] += threshold
        # Check NCO and input parameters
        try:
            nco = nco_freq_dict[module_index]
        except:
            raise Exception('NCO frequency has not been set')
        if any(ares > d.full_scale_dbm):
            raise ValueError(f'ares must not exceed {d.full_scale_dbm} dBm: raise full_scale_dbm or lower powers')
        if any(ares < -60) and len(ares) < 100:
            warnings.warn(f"values in ares are < 60 dBm: digitization noise may occur", UserWarning)
        ares_amplitude = 10 ** ((ares - d.full_scale_dbm) / 20)
        
        await d.clear_channels(module = module_index)
        
        async with d.tuber_context() as ctx:
            for ch, (fr, ar) in enumerate(zip(fres, ares_amplitude)):
                ## To be wrapped in context_manager 
                ctx.set_frequency(fr - nco, d.UNITS.HZ, ch + 1, module=module_index)
                ctx.set_amplitude(ar, d.UNITS.NORMALIZED, target = d.TARGET.DAC, channel=ch+1, 
                                  module=module_index)
            await ctx()

@rfmux.macro(rfmux.ReadoutModule, register=True)
async def sweep(module, nco_freq_dict, frequencies_dict, ares_dict, sweep_f, sweep_z, nsamps = 10, 
                verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep and returns the complex S21 value at each frequency. Performs sweeps over 
        axis 0 of frequencies simultaneously 

        Parameters:
        module (rfmux.ReadoutModule): readout module object
        nco_freq_dict (dict): keys (int) are module indices and values (float) are 
            NCO frequencies in Hz
        frequencies_dict (dict): keys (int) are module indices and values (M X N array-like float) are araays where
            the first index M is the channel index (max len 1024) and the second index N is the frequency in Hz 
            for a single point in the sweep
        ares_dict (dict): keys (int) are module indices and values (M array-like float) are amplitudes in dBm for each channel 
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        z (M X N array-like complex): complex S21 data in V for each frequency in f 
        """
        d = module.crs
        module_index = module.module  
        frequencies, ares = np.asarray(frequencies_dict[module_index]), np.asarray(ares_dict[module_index])
        # Randomize frequencies a little
        frequencies += np.random.uniform(-50, 50, frequencies.shape)
        comb_sampling_freq = transferfunctions.get_comb_sampling_freq()
        threshold = 101.
        frequencies[frequencies%(comb_sampling_freq/512) < threshold] += threshold

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
        zcal = np.empty((n_channels, n_points), dtype = complex) 
        zraw = np.empty((n_channels, n_points), dtype = complex)

        pbar = range(n_points) 
        if verbose:
            pbar = tqdm(pbar, total = n_points, leave = False)
            pbar.set_description(pbar_description)
        
        for sweep_index in pbar:
            # Write frequencies 
            async with d.tuber_context() as ctx:
                for ch in range(n_channels):
                    f = frequencies[ch, sweep_index]
                    ctx.set_frequency(f - nco_freq, d.UNITS.HZ, ch + 1, module = module_index)
                await ctx()
            nsamples_discard = 0 # 15
            # take data and loopback calibration data
            await d.set_dmfd_routing(d.ROUTING.CARRIER, module_index)
            sleep(0.5)
            samples_cal = await d.py_get_samples(20 + nsamples_discard, module = module_index )
            await d.set_dmfd_routing(d.ROUTING.ADC, module_index)
            sleep(0.5)
            samples = await d.py_get_samples(nsamps + nsamples_discard,module = module_index)
            # format and average data 
            zi = np.asarray(samples.i) + 1j * np.asarray(samples.q)
            zi = np.mean(zi[:n_channels, nsamples_discard:] , axis = 1)
            zical = np.asarray(samples_cal.i) + 1j * np.asarray(samples_cal.q) 
            zical = np.mean(zical[:n_channels, nsamples_discard:], axis = 1)
            # adjust for loopback calibration
            zcal[:, sweep_index] = zical 
            zraw[:, sweep_index] = zi * d.volts_per_roc 
            zi = remove_internal_phaseshift(frequencies[:, sweep_index], zi, zical) 
            z[:, sweep_index] = zi * d.volts_per_roc 
        # Turn off channels 
        await d.clear_channels(module = module_index)
        sweep_f[module_index] = frequencies 
        sweep_z[module_index] = z

@rfmux.macro(rfmux.ReadoutModule, register=True) 
async def get_noise_cal(module, fres_dict, noise_zcal_dict):
    """
    Acquires noise calibration data 

    Parameters:
    module (rfmux.ReadoutModule): readout module object
    fres_dict (dict): keys (int) are module indices and values (array-like) are 
        frequencies in Hz 
    """
    d = module.crs
    module_index = module.module 
    fres = np.asarray(fres_dict[module_index])
    # Get calibration data
    nsamples_discard = 0
    await d.set_dmfd_routing(d.ROUTING.CARRIER, module_index) 
    samples_cal = await d.py_get_samples(20 + nsamples_discard, module = module_index)
    zcal = np.asarray(samples_cal.i) + 1j * np.asarray(samples_cal.q) 
    zcal = np.mean(zcal[:len(fres), nsamples_discard:], axis = 1)
    await d.set_dmfd_routing(d.ROUTING.ADC, module_index)
    np.save(f'tmp/zcal_{module_index}.npy', [np.real(zcal), np.imag(zcal)]) # Save in case it crashes
    noise_zcal_dict[module_index] = zcal 
