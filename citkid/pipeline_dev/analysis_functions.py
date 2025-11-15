
import numpy as np
import pandas as pd
import zarr

from importlib.metadata import version

from citkid.res.fitter import fit_nonlinear_iq_with_gain, fit_iq_circle
from citkid.res.gain import fit_gain, remove_gain
from citkid.res.data_io import make_fit_row
from citkid.noise.analysis import calculate_theta_A, calibrate_x
from citkid.noise.psd import get_psd


def load_data(data_path, ixs_to_load):
	"""
	Example function to load IQ scan and noise data from .npy files.
	"""
	fnoises = np.load(data_path+'fres.npy', mmap_mode='r')
	fnoises = fnoises[ixs_to_load]
	I, Q = np.load(data_path+'noise.npy', mmap_mode='r')
	znoises = I[ixs_to_load] + 1j*Q[ixs_to_load]
	f, I, Q = np.load(data_path+'s21_fine.npy', mmap_mode='r')
	ffines = f[ixs_to_load]
	zfines = I[ixs_to_load] + 1j*Q[ixs_to_load]
	f, I, Q = np.load(data_path+'s21_gain.npy', mmap_mode='r')
	fgains = f[ixs_to_load]
	zgains = I[ixs_to_load] + 1j*Q[ixs_to_load]
	res_indices = np.load(data_path+'res_indices.npy', mmap_mode='r')
	res_indices = res_indices[ixs_to_load]
	fcal_indices = np.where(res_indices<0)[0]
	
	return ffines, zfines, fgains, zgains, fnoises, znoises, fcal_indices


def load_pipeline_history(zarr_path):
	"""
	Function to load pipeline history stored in a .zarr structure
	and return it in a Pandas DataFrame.

	Parameters:
		zarr_path (str): Path to a .zarr file with analysis results
	Returns:
		history_df: Pandas DataFrame with the pipeline history
	"""
	root = zarr.open_group(zarr_path, mode='r')
	function_calls = root['history/columns'][:]
	res_ixs = root['history/res_ix'][:]
	versions = root['history/versions'][:,:]

	data = {'res_ix': res_ixs}
	for ii in range(len(function_calls)):
		data[function_calls[ii]] =  versions[ii]

	history_df = pd.DataFrame(data=data)
	return history_df



def fit_iq_pipeline(ffines, zfines, fgains, zgains, frs, Qrs, fcal_indices, downward):
	"""
	Fits IQ loops.

	Parameters:
		ffines: Arrays of fine scan frequencies
		zfines: Arrays of fine scan complex S21
		fnoises: Array of readout frequencies where noise data was taken
		znoises: Arrays of complex S21 noise data
		frs: Array of center frequencies of resonances to cut out
			during gain fitting
		Qrs: Spans to cut out for gain fitting, relative to the frequencies frs
		fcal_indices: Indices of off-resonance calibration tones
		downward (bool): True if the scan went from high to low freq, 
			False if from low to high.
	"""
	
	data = pd.DataFrame()
	fr_spans = np.array([frs, frs/Qrs]).T
	
	for res_ix in range(len(ffines)):
		
		ffine, zfine = ffines[res_ix], zfines[res_ix]
		fgain, zgain = fgains[res_ix], zgains[res_ix]
		
		if res_ix in fcal_indices:
			p_amp, p_phase, (fig, axs) = \
				fit_gain(fgain, zgain, fr_spans, plotq = False)
				
			p = [np.nan] * 7
			res = np.nan
			row = make_fit_row(p_amp, p_phase, p, p, p, res,
								  downward = True,
								  plot_path = '', floats_only=True)
			row['fcal'] = 1
			row['res_ix'] = res_ix
		
		else:
			row, fig = \
				fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine, frs, Qrs,
										   downward = True, plotq = False,
										   return_dataframe = True, floats_only=True)
			row['fcal'] = 0
			row['res_ix'] = res_ix
			
		fitdf = pd.DataFrame(row).T
		data = pd.DataFrame(pd.concat([data, fitdf]))
		
	return data


def remove_gain_pipeline(ffines, zfines, fnoises, znoises, p_amps, p_phases):
		"""
		Removes gain and cable delay from fine scans and noise.

		Parameters:
			ffines: Arrays of fine scan frequencies
			zfines: Arrays of fine scan complex S21
			fnoises: Array of readout frequencies where noise data was taken
			znoises: Arrays of complex S21 noise data
			p_amps: Arrays of gain amplitude polynomial fit parameters.
				Shape = (# KIDs, 3)
			p_phases: Arrays of cable delay polynomial fit parameters.
				Shape = (# KIDs, 2)
		Returns:
			zfines_rmvd: Arrays of fine scan complex S21 with gain and cable delay removed
			znoises_rmvd: Arrays of noise complex S21 with gain and cable delay removed
		"""
		zfines_rmvd = np.full(zfines.shape, np.nan, dtype='complex128')
		znoises_rmvd = np.full(znoises.shape, np.nan, dtype='complex128')

		for ii in range(len(ffines)):
			try:
				p_amp = p_amps[ii]
				p_phase = p_phases[ii]
				zfines_rmvd[ii] = remove_gain(ffines[ii], zfines[ii], p_amp, p_phase)
				znoises_rmvd[ii] = remove_gain(fnoises[ii], znoises[ii], p_amp, p_phase)
			except:
				pass
			
		return zfines_rmvd, znoises_rmvd


def deglitch_noise_pipeline(noise_data, nstd):
	"""
	Removes glitches / spikes from noise timestreams.
	This function is agnostic to the units of data passed in - 
	it can be complex S21 data, phase, x, etc.

	Parameters:
		noise_data: Array of noise timestreams, shape=(#KIDs, #data points)
		nstd: Number of standard deviations from the median 
			above which to remove glitches by setting them
			to the median value.
	Returns:
		noise_deglitched: Deglitched array of noise timestreams
	"""
	noise_medians = np.nanmedian(noise_data, axis=1)
	noise_stds = np.nanstd(noise_data, axis=1)
	noise_deglitched = np.copy(noise_data)
	for ii in range(len(noise_data)):
		arr = noise_deglitched[ii]
		mask = (abs(arr - noise_medians[ii])) > nstd*noise_stds[ii]
		arr[mask] = noise_medians[ii]
		noise_deglitched[ii] = arr

	return noise_deglitched


def fit_iq_circle_pipeline(zfines):
	"""
	Fits circles to complex S21 fine scan data.

	Parameters:
		zfines: Array of complex S21 fine sweeps
	Returns:
		circle_fits: Array of circle fits (A,B,R). Shape=(#KIDs, 3)
	"""
	circle_fits = np.full((len(zfines), 3), np.nan)
	for ii in range(len(circle_fits)):
		popt, _ = fit_iq_circle(zfines[ii], x0=None, plotq = False)
		circle_fits[ii] = popt

	return circle_fits


def calculate_theta_A_pipeline(zfines, znoises, circle_fits):
	"""
	Converts complex S21 noise data to phase and amplitude units.

	Parameters:
		zfines: Array of complex S21 fine sweeps
		znoises: Array of complex S21 noise data
		circle_fits: Array of circle fits (A,B,R). Shape=(#KIDs, 3)
	Returns:
		theta_fines (np.array): values of theta corresponding to the fine scans
		theta_noises (np.array): theta timestreams corresponding the the noise data
		A_noises (np.array): amplitude timestreams corresponding to the noise data
	"""
	theta_fines = np.full(zfines.shape, np.nan)
	theta_noises = np.full(znoises.shape, np.nan)
	A_noises = np.full(znoises.shape, np.nan)
	for ii in range(len(zfines)):
		origin = circle_fits[ii][0] + 1j*circle_fits[ii][1]
		theta_fine, theta_noise, A_noise = \
			calculate_theta_A(zfines[ii], znoises[ii], origin)
		theta_fines[ii] = theta_fine
		theta_noises[ii] = theta_noise
		A_noises[ii] = A_noise

	return theta_fines, theta_noises, A_noises


def calibrate_x_pipeline(ffines, theta_fines, theta_noises,
						poly_deg, min_cal_points):
	"""
	Calibrates phase noise data to fractional frequency
	shift data x.

	Parameters:
		ffines: Arrays of fine scan frequencies
		theta_fines (np.array): values of theta corresponding to the fine scans
		theta_noises (np.array): theta timestreams corresponding the the noise data
		poly_deg: Polynomial degree for frequency vs. phase fit
		min_cal_points: Minimum number of points to use in polynomial fit
	Returns:
		polys: Array of polynomial fits.
		theta_ranges: Ranges of theta used for the polynomial fits.
	"""
	polys = np.full((len(ffines), poly_deg+1), np.nan)
	theta_ranges = np.full((len(ffines), 2), np.nan)

	for ii in range(len(ffines)):
		poly, theta_range, (ix0, ix1) = \
			calibrate_x(ffines[ii], theta_fines[ii], theta_noises[ii], 
			poly_deg, min_cal_points)
		polys[ii] = poly
		theta_ranges[ii] = theta_range

	return polys, theta_ranges


def calculate_x_pipeline(polys, theta_noises):
	"""
	Converts phase noise data to fractional frequency shifts.

	Parameters:
		polys: Array of polynomial fits for theta vs. frequency
		theta_noises (np.array): theta timestreams corresponding the the noise data
	Returns:
		x_noises: fractional frequency shift timestreams corresponding the the noise data
	"""
	x_noises = np.full(theta_noises.shape, np.nan)
	for ii in range(len(polys)):
		f0_arr = np.polyval(polys[ii], theta_noises[ii])
		x = 1-f0_arr/np.nanmedian(f0_arr)
		x -= np.nanmedian(x)
		x_noises[ii] = x

	return x_noises


def calculate_psd_pipeline(x_noises, fsamp):
	"""
	Calculates PSDs from frequency shift noise data.

	Parameters:
		x_noises: fractional frequency shift timestreams corresponding the the noise data
		fsamp: Sample frequency at which noise data was taken
	Returns:
		f_fft: FFT frequencies
		sxxs: Arrays of Sxx
	"""
	f_fft = np.fft.rfftfreq(x_noises.shape[1], d = 1/fsamp)
	sxxs = np.full((len(x_noises), len(f_fft)), np.nan)
	for ii in range(len(x_noises)):
		sxxs[ii] = get_psd(x_noises[ii], 1/fsamp, get_frequencies = False)

	return f_fft, sxxs