
import numpy as np
import zarr
import yaml
import importlib.util
from importlib.metadata import version
import sys
import os
import shutil

from .util import printc
from .analysis_functions import *

class MultitoneDataset():
	"""
	Multi-tone data set class.
	A data set refers to one multi-tone measurement comprising:
		1) fine sweeps
		2) gain sweeps
		3) noise timestreams
		...
		
	Parameters:
		data_path: Data directory. 
			Contains raw data: fine and gain sweeps, noise, 
			indices of off-res/calibration tones, sample rate...
		results_path: Results directory.
			Contains derived data products: calibration of S21 -> theta...
		config_file: Path to configuration file ending in .yaml
	"""
	def __init__(self, data_path, results_path, config_file):
		
		printc('Loading configuration file', 'info')
		try:
			with open(config_file) as file:
				config_params = yaml.safe_load(file)
				
			printc('Configuration file loaded', 'ok')
		
		except Exception as e:
			printc('Fail loading configuration file. '+str(e), 'fail')
			return

		self.data_path = data_path
		self.results_path = results_path
		self.ixs_to_analyze = eval(str(config_params['DETECTORS_TO_ANALYZE']))
		self.ixs_to_analyze = np.array(self.ixs_to_analyze, dtype=int)
		self.gain_fit_Q = np.array(config_params['GAIN_FIT_Q'])
		if not self.gain_fit_Q.shape:
			self.gain_fit_Q = np.full(len(self.ixs_to_analyze), self.gain_fit_Q)
		self.noise_sample_rate = config_params['NOISE_SAMPLE_RATE']
		self.overwrite = config_params['OVERWRITE']
		self.previous_result_path = None
		self.unsaved_tasks = []

	def __getattr__(self, name):
		"""
		Reconstructs an attribute and returns it if it does not already exist.

		Parameters:
			name (str): name of attribute to get
		Returns:
			attr: Attribute which was loaded
		"""
		if name in ['circle_fits', 'iq_fits']:
			self.previous_result_path = self.results_path+name+'.zarr'
			root = zarr.open_group(self.previous_result_path, mode='r')
			columns = root['result/columns'][:]
			fitdata = root['result/fitdata'][:]
			data = {columns[ii]: fitdata[:, ii] for ii in range(len(columns))}
			attr = pd.DataFrame(data=data)
		elif name == 'polys':
			self.previous_result_path = self.results_path+'x_calibrations.zarr'
			root = zarr.open_group(self.previous_result_path, mode='r')
			attr = root['result/polys'][:,:]
		elif name == 'x_noises':
			attr = calculate_x_pipeline(self.polys, self.theta_noises)
		return attr


	def add_pipeline_history(self, function_name, root):
		"""
		Adds a history of function calls to the .zarr result for the
		current function call.

		Parameters:
			function_name (str): Name of the current function call
			root: Root group for the current .zarr result
		"""
		history = root.create_group(name='history')
		citkid_version = version("citkid")
		function_names = np.append(self.unsaved_tasks, function_name)

		if self.previous_result_path is not None:
			previous_root = zarr.open_group(self.previous_result_path, mode='r')
			res_ixs_data = previous_root['history/res_ix'][:]

			ixs_in_arr = np.searchsorted(res_ixs_data, self.ixs_to_analyze)
			version_data = previous_root['history/versions'][:,:]
			new_version_data = np.full((len(function_names), len(res_ixs_data)), '', dtype=object)
			new_version_data[:, ixs_in_arr] = citkid_version
			version_data = np.append(version_data, new_version_data, axis=0)

			column_data = previous_root['history/columns'][:]
			column_data = np.append(column_data, function_names)

		else:
			res_ixs_data = self.ixs_to_analyze

			version_data = np.full((len(function_names), len(res_ixs_data)), '', dtype=object)
			version_data[:,:] = citkid_version

			column_data = function_names
		
		res_ixs = history.create_array(name='res_ix', data=res_ixs_data)

		versions = history.create_array(name='versions', shape=version_data.shape,
										dtype=zarr.dtype.VariableLengthUTF8())
		versions[:,:] = version_data

		columns = history.create_array(name='columns', shape=column_data.shape,
										dtype=zarr.dtype.VariableLengthUTF8())
		columns[:] = column_data

		self.unsaved_tasks = []


	def save_outputs(self, data, task, save, overwrite):
		"""
		Function to either save the outputs of the current pipeline task,
		or add the task to the list of unsaved outputs.

		Parameters:
			data: The pipeline outputs to be saved
			task (str): The name of the pipeline task
			save (bool): Whether to save the outputs
			overwrite (bool): Whether to overwrite the existing outputs
		"""

		name_table = {
			'fit_gain': 'iq_fits',
			'fit_nonlinear_iq_with_gain': 'iq_fits',
			'fit_iq_circle': 'circle_fits',
			'calibrate_x': 'x_calibrations'
		}

		if save:
			filename = name_table[task]
			store_path = self.results_path+filename+'.zarr'

			if os.path.exists(store_path):
				if overwrite:
					shutil.rmtree(store_path, ignore_errors=True)
				else:
					printc(f'Output file already exists!', 'fail')
					raise Exception('Use a new output path or set OVERWRITE to True in the config file.')

			root = zarr.open_group(store_path, mode='w')
			result = root.create_group(name='result')

			if task == 'fit_gain' or task == 'fit_nonlinear_iq_with_gain':
				fitdata = result.create_array(name='fitdata', data=data.to_numpy())
				columns = np.array(data.columns)
				fit_columns = result.create_array(name='columns', shape=len(columns),
												dtype=zarr.dtype.VariableLengthUTF8())
				fit_columns[:] = columns
			elif task == 'fit_iq_circle':
				fitdata = result.create_array(name='fitdata', data=data)
				columns = np.array(['A', 'B', 'R'])
				fit_columns = result.create_array(name='columns', shape=len(columns),
												dtype=zarr.dtype.VariableLengthUTF8())
				fit_columns[:] = columns
			elif task == 'calibrate_x':
				polys = result.create_array(name='polys', data=data)

			self.add_pipeline_history(task, root)
			self.previous_result_path = store_path
		
		else:
			self.unsaved_tasks = np.append(self.unsaved_tasks, task)

	# ------------------------- #
	# R U N   R E D U C T I O N #
	# ------------------------- #
	def run_reduction(self, reduction_file, overwrite=False):
		"""
		Run data reduction steps.
		
		Can start from the raw data, or at any intermediate analysis step
		if the necessary analysis products are given.
		
		Parameters:

		"""
		# load steps file
		printc('Loading reduction file', 'info')

		try:
			with open(reduction_file) as file:
				reduction_steps_file = yaml.safe_load(file)
	
				reduction_steps = reduction_steps_file['RED_STEPS']
				printc('Reduction file loaded', 'ok')

		except Exception as e:
			
			printc(f'Fail loading instructions file.\n{e}', 'fail')
			return

		for step in np.sort(list(reduction_steps.keys())):
			
			# task name
			task = reduction_steps[step]['task']
			
			name = f'{step}.{reduction_steps[step]['task']}'
			printc(name, 'info')

			# parameters
			keys_dict = reduction_steps[step]['params']

			# step name
			if task == 'load_data':
				path_to_file = keys_dict['path_to_file']
				spec = importlib.util.spec_from_file_location("load_data", path_to_file)
				load_data = importlib.util.module_from_spec(spec)
				sys.modules["load_data"] = load_data
				spec.loader.exec_module(load_data)
				
				ffines, zfines, fgains, zgains, fnoises, znoises, fcal_indices = \
					load_data.load_data(self.data_path, self.ixs_to_analyze)
				
				self.ffines = ffines
				self.zfines = zfines
				self.fgains = fgains
				self.zgains = zgains
				self.fnoises = fnoises
				self.znoises = znoises
				self.fcal_indices = fcal_indices
				
			elif task == 'fit_gain':
				data = fit_gain_pipeline(self.ixs_to_analyze, self.fgains, self.zgains, 
							  self.fnoises, self.gain_fit_Q, self.fcal_indices, keys_dict['downward'])

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)
			
			elif task == 'fit_nonlinear_iq_with_gain':
				data = fit_iq_pipeline(self.ixs_to_analyze, self.ffines, self.zfines, self.fgains, self.zgains, 
							  self.fnoises, self.gain_fit_Q, self.fcal_indices, keys_dict['downward'])

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)

			elif task == 'remove_gain':
				iq_fits = self.iq_fits
				columns = iq_fits.columns
				res_ixs = np.array(iq_fits.res_ix, dtype=int)
				ixs_in_arr = np.searchsorted(res_ixs, self.ixs_to_analyze)

				p_amps = np.array([iq_fits.iq_pamp_00, 
								iq_fits.iq_pamp_01, 
								iq_fits.iq_pamp_02]).T[ixs_in_arr]
				p_phases = np.array([iq_fits.iq_pphase_00, 
								iq_fits.iq_pphase_01]).T[ixs_in_arr]

				self.zfines, self.znoises = remove_gain_pipeline(self.ffines, self.zfines, 
														self.fnoises, self.znoises, 
														p_amps, p_phases)

				data = [self.zfines, self.znoises]

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)

			elif task == 'deglitch_noise':
				self.znoises = deglitch_noise_pipeline(self.znoises, keys_dict['nstd'])

				data = self.znoises

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)

			elif task == 'fit_iq_circle':
				data = fit_iq_circle_pipeline(self.zfines)

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)

			elif task == 'calculate_theta_A':
				self.theta_fines, self.theta_noises, self.A_noises = \
					calculate_theta_A_pipeline(self.zfines, self.znoises, self.circle_fits)

				data = [self.theta_fines, self.theta_noises, self.A_noises]

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)

			elif task == 'calibrate_x':
				poly_deg = keys_dict['poly_deg']
				min_cal_points = keys_dict['min_cal_points']

				polys, self.theta_fines = \
					calibrate_x_pipeline(self.ffines, self.theta_fines, self.theta_noises,
										poly_deg, min_cal_points)

				data = polys

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)


			elif task == 'calculate_psd':
				self.f_fft, self.sxxs = calculate_psd_pipeline(self.x_noises, self.noise_sample_rate)

				data = [self.f_fft, self.sxxs]

				self.save_outputs(data, task, keys_dict['save'], self.overwrite)

