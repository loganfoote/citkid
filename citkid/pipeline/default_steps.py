import numpy as np

from .framework import plStep
from ..xcal import gain, circle, xcal 
from ..res import fitter as res_fitter
from ..signal import psd
from ..xcal import reduced_params
 
############################## Default cal steps ###############################
# name, function, input parameter names, output parameter names, 
# save, func_vectorized
default_cal_steps =\
(
 ('rmv_gain_f', gain.remove_gain, 
  ['ff', 'zf', 'p_amp', 'p_phase'], ['zf_rmv'], 'per-row'),

 ('rmv_gain_t', gain.remove_gain, 
  ['ft', 'zt', 'p_amp', 'p_phase'], ['zt_rmv'], 'per-row'),

 ('center_f', circle.cent_rot_s21, 
  ['zf_rmv', 'circ_origin', 'theta_phase_offset'], ['zf_cent'], 'per-row'),

 ('center_t', circle.cent_rot_s21, 
  ['zt_rmv', 'circ_origin', 'theta_phase_offset'], ['zt_cent'], 'per-row'),

 ('rotate_offres_t', lambda z, t: circle.cent_rot_s21(z, 0 + 0j, t), 
  ['zt_rmv', 'theta_phase_offset'], ['zt_cent'], 'per-row'),

 ('rotate_offres_f', lambda z, t: circle.cent_rot_s21(z, 0 + 0j, t), 
  ['zf_rmv', 'theta_phase_offset'], ['zf_cent'], 'per-row'),

 ('get_thetaf', lambda z, idx_t: circle.convert_to_theta(
     z, unwrap = True, idx_t = idx_t), 
  ['zf_cent', 'idx_t'], ['thetaf'], 'per-row'),

 ('get_thetat', lambda z: circle.convert_to_theta(z, unwrap = False), 
  ['zt_cent'], ['thetat'], 'per-row'),

 ('get_xf', lambda ff, ft: 1 - ff / ft, 
  ['ff', 'ft'], ['xf'], 'per-row'),

 ('get_xt', np.polyval, 
  [ 'poly_x', 'thetat'], ['xt'], 'per-row'),

 # extra steps
 ('get_At', circle.convert_to_A, 
  ['zt_cent'], ['At'], 'per-row'),

 ('get_sparper', 
  lambda a, b, c, d: circle.get_spar_sper(a, b, c, d, get_freqs = True), 
  ['thetat', 'At', 'circ_radius', 'dt'], 
  ['f_sparper', 'spar', 'sper'], 'per-row'),

  ('get_sxx', lambda x, dt: psd.get_psd(x, dt, get_frequencies = True),
   ['xt', 'dt'],
   ['f_sxx', 'sxx'], 'per-row'),

   ('get_sxx_reduced', reduced_params.get_sxx_reduced_default_freqs,
    ['f_sxx', 'sxx'], 
    [f'sxx_{_freq}'.replace('.', 'p') for _freq in reduced_params._freqs], 
    'per-row'),

   ('get_sfactor_reduced', reduced_params.get_sfactor_reduced_default_freqs,
    ['f_sparper', 'spar', 'sper'], 
    [f'sfactor_{_freq}'.replace('.', 'p') for _freq in reduced_params._freqs], 
    'per-row')
  
)

############################ Default analysis steps ############################
default_analysis_steps =\
(
 ('make_fr_spans', gain.make_fr_spans, 
  ['fres_all', 'qres_all'], ['fr_spans'], 'global'),

 ('fit_gain', gain.fit_gain, 
  ['fg', 'zg', 'fr_spans', 'span_mult'], ['p_amp', 'p_phase', 'gain_mask'], 
  'per-row'),

 ('fit_iq_circle', circle.fit_iq_circle, 
  ['zf_rmv', 'circ_mask'], ['circ_origin', 'circ_radius'], 'per-row'),

 ('get_idx_t', lambda ff, ft: np.argmin(np.abs(ff - ft)), 
  ['ff', 'ft'], ['idx_t'], 'per-row'),

 ('get_theta_phase_offset', circle.get_theta_phase_offset, 
  ['zt_rmv', 'circ_origin'], ['theta_phase_offset'], 'per-row'),

 ('get_theta_phase_offset_offres', 
   lambda z: circle.get_theta_phase_offset(z, 0 + 0j), 
  ['zt_rmv'], ['theta_phase_offset'], 'per-row'),

 ('get_xcal_mask', xcal.get_xcal_mask,
  ['ff', 'thetaf', 'thetat', 'xcal_idx0_offset', 'xcal_idx1_offset', 
   'xcal_std_cutoff'], ['xcal_mask'], 'per-row'),

 ('fit_x_theta', 
  lambda thetaf, xf, xcal_mask: xcal.fit_x_theta(
                                    thetaf, xf, xcal_mask, poly_x_deg = 3), 
  ['thetaf', 'xf', 'xcal_mask'], ['poly_x'], 'per-row'),

 ('fit_iq', res_fitter.fit_nonlinear_iq_pl, 
  ['ff', 'zf_rmv', 'iq_mask'], ['iq_p0', 'iq_popt', 'iq_nrmse'], 'per-row'),
)

############################## Convert to plSteps ##############################
default_cal_steps = [plStep(*cs) for cs in default_cal_steps]
default_analysis_steps = [plStep(*ans) for ans in default_analysis_steps]