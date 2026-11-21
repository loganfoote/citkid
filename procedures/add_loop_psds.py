"""
Temporary: add the loop-component spectra (spar_binned, sper_binned,
sfactor_reduced) to PSD groups made before 6c_psds.py computed them,
in place, with each group's own settings (with the CRS decimation droop
undone and the spectra ending at 0.8 of the Nyquist frequency, as 6c
does now; the group's Sxx stays as it was). Groups that have them are
skipped.

Usage:
    python add_loop_psds.py --store Z:/cd/deviceA_screen.zarr
    python add_loop_psds.py --store Z:/cd/deviceA_screen.zarr --psd 0
"""
import argparse
import os
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline import photon_finder as pf
from citkid.pipeline import psds
from citkid.pipeline.stream_sources import apply_stream_calibration

################################################################################
################################### Settings ###################################
################################################################################
from settings import MEMORY_GB

################################################################################
################################### Script #####################################
################################################################################
def parse_args(argv = None):
    """
    Parse the command-line options.

    Parameters:
    argv (list of str or None): arguments. None (default): sys.argv.

    Returns:
    args (argparse.Namespace): the options.
    """
    p = argparse.ArgumentParser(
        description = 'Add the loop-component spectra to saved PSD groups.')
    p.add_argument('--store', required = True, help = 'store path')
    p.add_argument('--psd', type = int, default = None,
                   help = 'one group index (default: every group)')
    p.add_argument('--redo', action = 'store_true',
                   help = 'compute them again where they exist')
    return p.parse_args(argv)

def add_loop_psds(root, grp, timestreams, datasets, memory_gb = MEMORY_GB):
    """
    Compute and save one PSD group's loop-component spectra.

    Parameters:
    root (zarr.Group): the store root (writable).
    grp (zarr.Group): the ``ts_psd{p}`` group.
    timestreams (list of dict): ``psds.list_timestreams`` output.
    datasets (dict): its DataSets.
    memory_gb (float): timestreams kept in memory per write, GB.

    Returns:
    done (list of str): the timestream groups written.
    """
    f_reduced = np.asarray(grp['f_reduced'][:], dtype = float)
    done = []
    names = sorted(n for n in grp.group_keys() if n.startswith('timestream'))
    for name in names:
        t = grp[name]
        a = t.attrs
        ts = next((x for x in timestreams
                   if x['run'] == a['run'] and x['tone_set'] == a['tone_set']
                   and x['cal'] == a['calibration']), None)
        if ts is None:
            print(f'{t.path}: its timestream is not listed, skipped')
            continue
        DS = datasets[ts['cal']][ts['dataset']]['DS']
        n = t['res_idxs'].shape[0]
        rows_saved = np.asarray(t['rows'][:], dtype = int)
        pos = {int(r): i for i, r in enumerate(rows_saved)}
        status = np.asarray(t['status'][:], dtype = str)
        valid = np.asarray(t['valid_reduced'][:], dtype = bool)
        bins = dict(nbins = int(a['nbins']), fmin = float(a['fmin']))
        dt, T = float(a['dt']), int(a['stop'] - a['start'])
        out = dict(spar_binned = None, sper_binned = None,
                   sfactor_reduced = np.full((n, len(f_reduced)), np.nan))
        for w, wname, rows in pf.tone_writes(root, ts):
            rows = np.asarray([r for r in rows if int(r) in pos],
                              dtype = np.int64)
            if not len(rows):
                continue
            idx = np.array([pos[int(r)] for r in rows], dtype = np.int64)
            need = 3 * len(rows) * T * 8
            if need > memory_gb * 1e9:
                raise MemoryError(f'{t.path} {wname}: {need / 1e9:.1f} GB '
                                  'needed; raise MEMORY_GB')
            print(f'{t.path}, {wname}: {len(rows)} tones')
            cal = apply_stream_calibration(DS, root, ts['source'],
                                           rows = rows,
                                           outputs = ('thetat', 'At'))
            radius = np.asarray(DS.circ_radius[rows.tolist()], dtype = float)
            par = cal['thetat']
            par *= radius[:, None]
            perp = cal['At']
            del cal
            bad = np.array([s.startswith('marked bad') for s in status[idx]])
            par[bad] = perp[bad] = np.nan
            fb, sb, sr_par = psds.psd_summary(par, dt, f_reduced, valid,
                                              **bins)
            psds._fill(out, 'spar_binned', idx, fb, sb, n)
            fb, sb, sr_per = psds.psd_summary(perp, dt, f_reduced, valid,
                                              **bins)
            psds._fill(out, 'sper_binned', idx, fb, sb, n)
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                out['sfactor_reduced'][idx] = 10 * np.log10(sr_par / sr_per)
            del par, perp
        for key in ('spar_binned', 'sper_binned'):
            if out[key] is None:
                out[key] = np.zeros((n, 0))
        for key, value in out.items():
            if key in t:
                del t[key]
            dl.write_array(t, key, value)
        done.append(t.path)
    return done

def main(args):
    """
    Add the loop-component spectra to the chosen groups.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    done (list of str): the timestream groups written.
    """
    root = dl.open_store(os.path.expanduser(args.store), mode = 'a')
    timestreams, datasets = psds.list_timestreams(root)
    ps = dl.list_indices(root, psds.PREFIX) if args.psd is None \
        else [int(args.psd)]
    done = []
    for p in ps:
        grp = root[dl.set_name(p, psds.PREFIX)]
        names = [n for n in grp.group_keys() if n.startswith('timestream')]
        if not args.redo and all('spar_binned' in grp[n] for n in names):
            print(f'{grp.path}: has the loop spectra already, skipped')
            continue
        done += add_loop_psds(root, grp, timestreams, datasets, MEMORY_GB)
    print(f'Done: {len(done)} timestreams')
    return done

if __name__ == '__main__':
    main(parse_args())
