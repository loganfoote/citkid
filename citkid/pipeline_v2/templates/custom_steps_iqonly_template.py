import numpy as np
import os
import zarr
from citkid.pipeline_v2.framework import plStep

main_directory = '/path/to/data/'
data_path = os.path.join(main_directory, 'example.zarr') 
root = zarr.open(data_path, mode = 'r')

def load_global_data():
    fres_all = np.load(os.path.join(main_directory, 'fres_init/fres.npy'))
    fres_all = np.sort(fres_all)
    qres_all = np.ones_like(fres_all) * 8000    
    nrows = root['fres'].shape[0]
    return fres_all, qres_all, nrows

def load_global_res_data():
    fres = np.array(root['fres'])
    ares = np.array(root['ares'])
    qres = np.array(root['qres']) 
    res_idxs = np.array(root['res_idxs']) 
    return fres, qres, ares, res_idxs 

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
  [], ['fres_all', 'qres_all', 'nrows'], 
  'global'),
 ('load_global_res_data', load_global_res_data,
  [], ['fres', 'qres', 'ares', 'res_idxs'], 
  'global-res'),
 ('load_data_f', load_data_f,
  ['data_idx'], ['ff', 'zf'], 
  'per-row'),
 ('load_data_g', load_data_g,
  ['data_idx'], ['fg', 'zg'], 
  'per-row')
]

custom_cal_steps = [plStep(*cs) for cs in custom_steps]
custom_analysis_steps = []
