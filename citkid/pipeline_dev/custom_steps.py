import numpy as np 
import os
from citkid.pipeline_dev.pl_steps import plStep

def load_metadata(directory):
    d = os.path.join(directory, 'raw/')
    def get_path(name):
        return os.path.join(d, f'{name}_3K.npy')
    ares = np.load(get_path('ares'))
    fres = np.load(get_path('fres')) 
    fres_all = np.load(get_path('fres_all'))
    qres = np.load(get_path('qres'))
    qres_all = np.load(get_path('qres_all'))
    res_idxs = np.load(get_path('res_idx')).astype(int)
    dt = np.load(os.path.join(d, 'noise_3K_batch_tsample.npy'))
    return fres, qres, ares, res_idxs, fres_all, qres_all, dt

def load_data_t(directory, res_idx, batches):
    """
    res_idx can be single or list of res indices
    """
    d = os.path.join(directory, 'raw/')
    zt = None
    for batch in batches:
        zi = np.load(os.path.join(d, f'noise_3K_batch{batch:02d}.npy'), 
                    mmap_mode = 'r')
        if zt is None:
            zt = zi[res_idx]
        else:
            zt = np.concatenate((zt, zi[res_idx]))
    return zt

def load_data_f(directory, res_idx):
    """
    res_idx can be single or list of res indices
    """ 
    d = os.path.join(directory, 'raw/')
    zf = np.load(os.path.join(d, 's21_fine_3K.npy'), mmap_mode = 'r')
    return zf[res_idx]

custom_steps =\
[('load_metadata', load_metadata, ['directory'], 
  ['fres', 'qres', 'ares', 'res_idxs', 'fres_all', 'qres_all', 'dt'], 
  False, True),
 ('load_data_t', load_data_t, 
  ['directory', 'res_idx', 'batches'], ['zt'], False, True),
 ('load_data_f', load_data_f,
  ['directory', 'res_idx'], ['zf'], False, True)
]

custom_steps = [plStep(*cs) for cs in custom_steps]