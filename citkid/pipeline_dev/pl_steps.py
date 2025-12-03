import numpy as np
from citkid.xcal import gain, corr, circle, xcal

class plStep:
    def __init__(self, name, func, param_names, return_names, save = False,
                 func_vectorized = False):
        """
        Class to represent a step in the analysis or calibration pipeline.

        Parameters:
        name (str): Name of the pipeline step.
        func (callable): Function to execute for this step.
        param_names (list of str): Names of the parameters to pass to the 
            function.
        return_names (list of str): Names of the attributes to store the 
            function's return values.
        save (bool): Whether to save the results after running the step.
        func_vectorized (bool): Whether the function can handle vectorized 
            inputs (multiple resonators at once).
        """
        assert type(name) == str 
        assert callable(func)
        assert type(param_names) == list
        assert type(return_names) == list
        assert type(param_names[0]) == str
        assert type(return_names[0]) == str
        self.name = name
        self.func = func
        self.param_names = param_names
        self.return_names = return_names
        self.save = save
        self.func_vectorized = func_vectorized

    def run(self, PL, data_idx = None):
        """
        Executes the pipeline step.

        Parameters:
        PL (pipeline): the pipeline object to execute the step on.
        data_idx (int, optional): data index, or None to pass in all 
            resonators. 
        """
        params = [PL._get_attr_data_idx(name, data_idx) for name in self.param_names]
        results = self.func(*params)
        if not isinstance(results, tuple):
            results = (results,)
        for name, value in zip(self.return_names, results):
            PL._set_attr_data_idx(name, value, data_idx) 
        if self.save:
            PL.save(self.return_names, data_idx = data_idx)

        # if isinstance(data_idx, int):
        # elif isinstance(data_idx, list):
        #     if self.func_vectorized:
        #         results = self.func(*params) 
        #     else:
        #         results = [] 
        #         for di in data_idx:
                    # ...

    def __repr__(self):
        s = f"Pipeline Step: {self.name}"
        s += f", Function: {self.func.__module__}.{self.func.__name__}"
        return s
    
    def __str__(self):
        s = f"Pipeline Step: {self.name}"
        s += f"\n\tFunction: {self.func.__module__}.{self.func.__name__}"
        s += f"\n\tInput Parameters: {self.param_names}"
        s += f"\n\tOutput Parameters: {self.return_names}"
        s += f"\n\tSave Results: {self.save}"
        return s

# Problem -> how to deal with the fact that offres cm removal needs to be run on 
# bins of frequencies 
# How to deal with parameters that need to be None on the first run, but maybe 
# should be enforced later

# name, function, input parameter names, output parameter names

# Calibration steps
cal_steps =\
(# calibration steps
 ('rmv_gain_f', gain.remove_gain, 
  ['ff', 'zf', 'p_amp', 'p_phase'], ['zf_rmv'], False),
 ('rmv_gain_t', gain.remove_gain, 
  ['ft', 'zt', 'p_amp', 'p_phase'], ['zt_rmv'], False),
 ('center_f', circle.cent_rot_s21, 
  ['zf_rmv', 'circ_origin', 'theta_phase_offset'], ['zf_cent'], False),
 ('center_t', circle.cent_rot_s21, 
  ['zt_rmv', 'circ_origin', 'theta_phase_offset'], ['zt_cent'], False),
 ('get_theta_f', circle.convert_to_theta, 
  ['zf_cent', 'unwrap_theta_f'], ['theta_f'], False),
 ('get_theta_t', circle.convert_to_theta, 
  ['zt_cent', 'unwrap_theta_t'], ['theta_t'], False),
 ('get_x_f', lambda ff, ft: 1 - ff / ft, 
  ['ff', 'ft'], ['x_f'], False),
 ('get_x_t', np.polyval, 
  ['theta_t', 'poly_x'], ['x_t'], True),
# analysis steps
 ('make_fr_spans', gain.make_fr_spans, ['fres_all', 'qres_all', 'fg'], ['fr_spans'], True),
 ('fit_gain', gain.fit_gain, 
  ['fg', 'zg', 'fr_spans'], ['pamp', 'pphase', 'gain_mask'], True),
 ('fit_circ', circle.fit_iq_circle, 
  ['zf_rmv', 'idx_circfit'], ['circ_origin', 'circ_radius'], True),
 ('get_theta_phase_offset', np.median, 
  ['zt_rmv'], ['theta_phase_offset'], True),
 ('get_xcal_idx', xcal.get_xcal_idx,
  ['ff', 'theta_f', 'theta_t', 'xcal_idx0_offset', 'xcal_idx1_offset', 
   'xcal_std_cutoff'], ['xcal_idx'], False),
 ('cut_xf', lambda x, t, idx: (x[idx], t[idx]), 
  ['x_f', 'theta_f', 'xcal_idx'], ['x_f_cut', 'theta_f_cut'], False),
 ('fit_x_theta', np.polyfit, 
  ['x_f_cut', 'theta_f_cut', 'poly_x_deg'], ['poly_x'], True),
# extra steps
 ('get_A_t', circle.convert_to_A, 
  ['zt_cent'], ['A_t'], False),
 ('get_sparper', circle.get_spar_sper, 
  ['theta_t', 'A_t', 'circ_radius', 'dt', 'sparper_get_freqs'], ['spar', 
                                                                 'sper'], False)
)

cal_steps = [plStep(*cs) for cs in cal_steps]

# ('rmv_cm_offres', corr.remove_cm_complex, 
# ['zt_rmv', 'aI', 'aQ', 'AI', 'AQ', 'cm_offres_idx', 'theta_cm_offres'],
# ['zt_cm_rmv']),

# ('calc_cm_offres', corr.calc_cm_complex, 
# ['zt_rmv', 'theta_cm_offres', 'N_comp_offres', 'N_iter_offres', 'dt', 
# 'lowpass_params_offres', 'highpass_params_offres', 'verbose'],
# ['aI', 'aQ', 'AI', 'AQ', 'sigI_iter', 'sigQ_iter', 'aI_full', 'aQ_full', 
# 'theta_cm_offres']),