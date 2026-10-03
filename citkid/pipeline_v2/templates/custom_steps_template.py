import numpy as np
import os
import zarr
from citkid.pipeline_v2.framework import plStep

# main_dir must be the only hard-coded path. It can be replaced with
# DataSet(..., custom_main_dir_overwrite=...) after the data is moved.
main_dir = '/path/to/data/'
data_path = os.path.join(main_dir, 'example.zarr')
root = zarr.open(data_path, mode = 'r')

# To check these steps: create the DataSet, then run DS.check_cal(0). It runs
# every calibration step for data_idx 0 and prints each step's output shapes
# and dtypes, structure problems (e.g. ff and zf of different lengths), and
# for a failing step the error traced into its function.

def load_global_data():
    dt = root['ts_00/dt'][...]
    nrows = root['fres'].shape[0]

    fres_all = root['fres_all'][...]
    qres_all = np.ones_like(fres_all) * 6000    
    return fres_all, qres_all, dt, nrows

def load_global_res_data():
    fres = root['fres'][...]
    ares = root['ares'][...]
    qres = root['qres'][...]
    res_idxs = root['res_idxs'][...]
    return fres, qres, ares, res_idxs 

def load_ft(data_idx):
    ft = root['ts_00/fres'][data_idx] 
    return ft

def load_zt(data_idx):
    zt = root['z'][:, data_idx, :10_000] 
    zt = zt[0] + 1j * zt[1] 
    zt *= np.array(root['counts_to_s21'][data_idx])[:, np.newaxis]
    return zt

def load_data_f(data_idx):
    ff = np.array(root['f'][data_idx, :])
    zf = np.array(root['z'][data_idx, :])
    idx = np.argsort(ff)
    return ff[idx], zf[idx]

def load_data_g(data_idx):
    fg = np.array(root['f'][data_idx, :])
    zg = np.array(root['z'][data_idx, :])
    idx = np.argsort(fg)
    return fg[idx], zg[idx]

custom_steps =\
[('load_global_data', load_global_data, 
  [], ['fres_all', 'qres_all', 'dt', 'nrows'], 
  'global'),
 ('load_global_res_data', load_global_res_data,
  [], ['fres', 'qres', 'ares', 'res_idxs'], 
  'global-res'),
 ('load_ft', load_ft,
  ['data_idx'], ['ft'], 
  'vectorized'),
  ('load_zt', load_zt,
  ['data_idx'], ['zt'], 
  'vectorized'),
 ('load_data_f', load_data_f,
  ['data_idx'], ['ff', 'zf'], 
  'per-row'),
 ('load_data_g', load_data_g,
  ['data_idx'], ['fg', 'zg'], 
  'per-row')
]

custom_cal_steps = [plStep(*cs) for cs in custom_steps]
custom_analysis_steps = []
