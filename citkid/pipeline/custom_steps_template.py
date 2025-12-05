import numpy as np 
import os
from citkid.pipeline_dev.pl_steps import plStep

def load_global_data(directory):
    d = os.path.join(directory, 'raw/')
    def get_path(name):
        return os.path.join(d, f'{name}_3K.npy')
    fres_all = np.load(get_path('fres_all'))
    qres_all = np.load(get_path('qres_all'))
    dt = np.load(os.path.join(d, 'noise_3K_batch_tsample.npy'))
    return fres_all, qres_all, dt

def load_global_res_data(directory):
    d = os.path.join(directory, 'raw/')
    def get_path(name):
        return os.path.join(d, f'{name}_3K.npy')
    ares = np.load(get_path('ares'))
    fres = np.load(get_path('fres')) 
    qres = np.load(get_path('qres'))
    res_idxs = np.load(get_path('res_indices')).astype(int)
    return fres, qres, ares, res_idxs

def load_data_t(directory, data_idx, batches):
    """
    data_idx can be single or list of res indices
    """
    d = os.path.join(directory, 'raw/')
    zt = None
    for batch in batches:
        zi = np.load(os.path.join(d, f'noise_3K_batch{batch:02d}.npy'), 
                    mmap_mode = 'r')
        if zt is None:
            zt = zi[data_idx]
        else:
            zt = np.concatenate((zt, zi[data_idx]))
    return zt

def load_data_f(directory, data_idx):
    """
    data_idx can be single or list of res indices
    """ 
    d = os.path.join(directory, 'raw/')
    ff, i, q = np.load(os.path.join(d, 's21_fine_3K.npy'), mmap_mode = 'r')
    zf = i[data_idx] + 1j * q[data_idx]
    return ff[data_idx], zf

custom_steps =\
[('load_global_data', load_global_data, ['directory'], 
  ['fres_all', 'qres_all', 'dt'], 
   'global'),
 ('load_global_res_data', load_global_res_data, 
  ['directory'], ['fres', 'qres', 'ares', 'res_idxs'], 'global-res'),
 ('load_data_t', load_data_t, 
  ['directory', 'data_idx', 'batches'], ['zt'], 'vectorized'),
 ('load_data_f', load_data_f,
  ['directory', 'data_idx'], ['ff', 'zf'], 'vectorized')
]

custom_steps = [plStep(*cs) for cs in custom_steps]