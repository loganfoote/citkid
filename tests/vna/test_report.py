"""
Tests of the VNA report export (citkid.vna.report): the figure with the
gain panel, the info text with the cable delay, and the CSVs.
"""
import asyncio
import os
import numpy as np
import pandas as pd
from citkid import daq_layout as dl
from citkid.vna import full_sweep as fs
from citkid.vna.report import export_vna_report


def test_export_vna_report(tmp_path):
    """
    Export a fitted simulated sweep: the plot, the info text with the
    cable delay, the resonator CSV and the gain CSV (frequency, gain in
    dB, phase and its line).
    """
    from citkid.crs.sim import simulate_vna_store
    path = tmp_path / 'sim.zarr'
    root, truth = asyncio.run(simulate_vna_store(
        path, device_kwargs = dict(f_lo = 1.0e9, f_hi = 1.1e9, n_banks = 1,
                                   n_per_bank = 10, pair_frac = 0.),
        verbose = False))
    grp = root['vna0']
    f, z = dl.load_vna_sweep(root, 0)
    grp.create_array('fres_manual', data = truth.fr.values)
    fs.run_full_sweep_fit(f, z, grp, progress = False)
    paths = export_vna_report(path, tmp_path / 'out')
    assert set(paths) == {'plot', 'info', 'csv', 'gain_csv'}
    for p in paths.values():
        assert os.path.exists(p)
    info = open(paths['info'], encoding = 'utf-8').read()
    assert 'Cable delay:' in info and 'ns' in info
    gain = pd.read_csv(paths['gain_csv'])
    assert list(gain.columns) == ['f', 'gain_db', 'phase', 'phase_fit']
    assert np.isfinite(gain['phase_fit']).all()
    assert np.nanmax(np.abs(gain['phase'] - gain['phase_fit'])) < 1.
    table = pd.read_csv(paths['csv'])
    assert len(table) == len(truth)
