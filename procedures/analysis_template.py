"""
Template for an analysis of your own: open a store, apply a stream
calibration to a timestream in memory, and read the tone parameters.
Copy what you need; there is no main script.
"""
import numpy as np
from citkid import daq_layout as dl
from citkid.pipeline.stream_sources import calibrated_timestream
from settings import DATA_DIR, COOLDOWN, DEVICE, SUFFIX

################################################################################
################################### Settings ###################################
################################################################################
CALIBRATION_NAME = None
TIMESTREAM = 'dark_dec6'
TONE_SET = None

################################################################################
################################### Template ###################################
################################################################################
root = dl.open_store(dl.store_path(DATA_DIR, COOLDOWN, DEVICE, SUFFIX))

# The calibration (the latest ts_cal, or CALIBRATION_NAME: 'ts_cal2' or 2)
# applied
# to the timestream (a run name, or an index in the calibration's list):
# xt is the fractional frequency shift, thetat the loop angle and At the
# amplitude, one row per calibrated tone of the tone set, one column per
# sample
DS, ts, out = calibrated_timestream(
    root, TIMESTREAM, cal = CALIBRATION_NAME, tone_set = TONE_SET,
    outputs = ('xt', 'thetat', 'At'))
xt, thetat, At = out['xt'], out['thetat'], out['At']
t = np.arange(xt.shape[1]) * ts['dt']
rows, res_idxs = ts['rows'], ts['res_idxs']
print(f"{ts['run']} {ts['name']}: tone set {ts['tone_set']}, "
      f"{len(rows)} tones, {xt.shape[1]} samples at {1 / ts['dt']:.0f} Hz")

# Tone parameters of the DataSet, one value per row of the tone set (index
# them with rows to match xt), and per-row arrays: ff and zf are the
# calibration sweep of a row, xf and thetaf its calibrated sweep
fres, qres, ares = DS.fres[rows], DS.qres[rows], DS.ares[rows]
ff, zf = DS.ff[rows[0]], DS.zf[rows[0]]
xf, thetaf = DS.xf[rows[0]], DS.thetaf[rows[0]]

# The run's record, its streams and the raw stream (counts)
run = dl.stream_runs(root)[ts['run_index']]
stream = root[run['paths'][0]]
z_counts, counts_to_s21 = stream['z'], stream['counts_to_s21'][:]

# Reduce and save your own results next to the pipeline's: a group at the
# root with kind 'analysis', one array per quantity, and the inputs in its
# attributes
grp = dl.create_group(root, 'my_analysis', 'analysis', attrs = dict(
    timestream = ts['name'], run = ts['run'],
    calibration = str(CALIBRATION_NAME)))
dl.write_array(grp, 'res_idxs', res_idxs)
dl.write_array(grp, 'x_rms', np.nanstd(xt, axis = 1))
