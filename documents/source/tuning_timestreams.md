---
title: "Tuning and timestreams"
subtitle: "3_tune, 4a_stream, 4b_stream_batch, 4c_ts_series, view_timestreams, 5_calibrate_streams"
---

# Overview

After VNA screening: tune the tones, take timestreams and calibrate
them. The analyses of the calibrated timestreams (photons, HEEs,
noise spectra) are in "Timestream analysis", the NEP of a blackbody
series in "NEP", and the pages in "Report". Configure the board with
`crs_init.py` first (see "Getting started").

| Script | What it does | Saves |
|---|---|---|
| `3_tune.py` | Plans the tones, sweeps the power, chooses each tone's power and frequency (two windows) | `tune{k}` |
| `4a_stream.py` | Window for calibrating, writing tones, previewing and taking named timestream runs | `ts{n}` |
| `4b_stream_batch.py` | The same without a window: calibrate, write, take the runs in the settings | `ts{n}` |
| `4c_ts_series.py` | A series of timestreams over an external parameter (blackbody power, temperature, ...): at each named point, a calibration and the runs | `ts{n}` |
| `view_timestreams.py` | Any time: look at saved runs with the plots of 4a (no board) | nothing |
| `5_calibrate_streams.py` | Calibrates timestreams, in a window and then in batch | `ts_cal{c}` |

```text
<device>.zarr/
  tune{k}/              tuning (3)
    cal{j}/             calibration target sweep of one tone set (4)
      write{w}/         tones written for streaming
        ts{n}/          stream of run n with these tones
    streams/            run list
  ts_cal{c}/            stream calibrations (5)
  photons/<name>/       photon results (6a), one per timestream
  hee/<name>/        HEE tails (6b), one per timestream
  ts_psd{p}/            noise spectra (6c, 6d)
  nep/<name>/           NEP of a power sweep (6d)
  tone_marks/           bad marks and notes on tones, from every stage
```

Calibrated timestreams are never saved: the analysis scripts apply a
calibration in memory and save only their results.

## Tone sets

A module reads out one NCO window (500 MHz, or 600 MHz with extended
bandwidth) with at most 1024 tones. Tones that don't fit the modules'
windows simultaneously are divided into **tone sets**, written and
measured one after another. The sets are planned once, with the tuning:

- window edges go in the widest gaps between resonances, `EDGE_MARGIN`
  from every tone;
- each window gets at most the tones that stream simultaneously at
  `STREAM_DEC` (128 at stages 0-1, 1024 at stages 2-6), so the NCO
  frequencies never change while streaming;
- `WINDOWS` (`--windows`) measures only some windows (`--list-windows`
  prints them); `FMIN`, `FMAX` keep only a frequency range.

## Tone marks

At any stage a tone can be marked bad with a reason, or given a note
(the tone is kept). Both go to the store's tone marks, one history per
resonator index, with the stage, the tuning and the time.

A bad mark has a scope, chosen in the reason dialog:

- **local**: the tone is left out of that stage's results only (the
  default in 6a, 6b and 6c);
- **global**: also out of every later stage of the same tuning (the
  default in 3 and 5; forced in 5 and by a panel's Bad Data, where
  the outputs become NaN). 6a, 6b and 6c count as one stage, so a
  global mark in any of them leaves the tone out of the other two and
  of the reports, in whatever order they run. A global mark never
  reaches an earlier stage or another tuning.

Clearing a mark (B again, un-rejecting, or the store browser) clears
the mark of that stage and scope only. The reports apply the marks as
they are when the report is built. `store_browser.py` (see Getting
started) lists every
mark (Tone marks tab) and can clear one or change its scope.

In every window B marks the tone bad, with the reason box (type or
pick a reason, Enter), and B again clears the mark; N adds a note.
M and T do the same, for habit. The reason box opens on the tone's
current reason when it is marked already.

| Where | Mark bad | Default scope | Note |
|---|---|---|---|
| `3_tune.py`, IQ series window | Mark Tone Bad (B); X marks the selected fit point bad, Shift+X every point of the tone (asks keep as a note or remove as bad), Ctrl+Shift+X the points above. The toolbar shows the tone's marks | global | Add Note (N) |
| `3_tune.py`, fq_finder window | Rejecting a tone (B, with the chosen reason; B again restores it); un-rejecting clears it | global | N |
| `5_calibrate_streams.py` window | Mark tone bad (B), or a panel's Bad Data | global (forced); local (forced) for the IQ fit panel, whose fit no later stage uses | N |
| `6a`, `6b` windows | B (again to clear) | local | N |
| `6c_psds.py` overview window | B (again to clear); marked tones and `BAD_TONES` are left out of the spectra | local | N |

Resonances the plan drops as too close get a note automatically.

# 3_tune.py: make a tuning

Creates a tuning, `tune{k}`, in these steps. `TUNING_NAME` names it: the
name (or index) of an existing tuning continues that one, a new name
starts a new one, and `None` continues the latest. Tunings made
before names are named `tune{k}`. Done steps are skipped; `--redo
power_fits`, `fq_finder` or `tones` repeats one and those after it;
sweeps are never retaken.

| Step | What it does | Saves |
|---|---|---|
| plan | From a resonator list, drops tones within `MIN_SEP_LW` linewidths of another (`PAIRS`: both, or the shallower), plans the tone sets, adds `N_CAL_TONES` off-resonance tones in all, spread over the NCO windows | `tones_init`, `crs_config` |
| power_sweep | Target sweep at each power in `POWERS` | `power_sweep/set{i}` |
| power_fits | IQ series window: IQ loop fits at every power, $a_\mathrm{nl}$ vs power per resonator (the $a_\mathrm{nl}$ axis stays on 0 to 1); the power where $a_\mathrm{nl}$ reaches `ANL` is marked (Shift+click or drag to choose another); no fit: the median of the chosen powers of the `FALLBACK_NEAREST` tones nearest in frequency; clipped to `ARES_LIMITS` | `power_fits`, `power_choice` |
| fq_sweep | Target sweep at the chosen powers, without the tones marked bad globally so far (in the power fits window, or any global mark of this tuning); they are then out of the fq_finder and the tones | `fq_sweep` |
| fq_finder | Window: set each tone's frequency and span, or reject it (the tone's power is shown); resumes where it stopped. Closed early, a box asks whether to make the tones from the resonators set so far or to stop and continue later | `fres_opt`, `qres_opt`, `reject_reason` |
| tones | The tuned tones, without rejected and bad-marked ones | `tones` |

## Calibration tones

Off-resonance tones swept and streamed with the resonators, for the
gain and the common modes. `N_CAL_TONES` is their number for the
whole tuning (`0`: none); the plan step places them like this:

- The count is split as evenly as possible over the tuning's NCO
  windows (one per tone set and module), the first windows getting the
  remainder.
- In each window, a tone goes to the middle of each of the widest gaps
  between neighbouring resonances (the window's edges count as gaps)
  that is at least `CAL_MIN_SEP_LW` linewidths from every resonance.
- A window with fewer such gaps than its share, or no room under its
  tone limit, passes the rest to the other windows.
- When every window is short, fewer tones are placed and the plan step
  prints how many fit. The tuning's `n_cal_tones` attribute holds the
  number placed.

Calibration tones have negative `res_idxs` in the tone list, and the
report and the analysis scripts tell them apart from resonators by
that.

`citkid.daq_layout.port_tuning` copies `tones/`, `crs_config/` and the
resonator list to another store (a new cooldown) without the sweeps.

To tune again at the powers of an earlier tuning, give `TUNING_NAME` a new
name and set `PORT_TUNING_NAME` to the earlier tuning's name or index (and
`PORT_STORE` to its store's path when it is in another store): the
plan, power sweep and power fits are skipped (the same settings
continue the ported tuning later),
`tones_init` is the earlier tuning's tones at the frequencies, spans
and powers `PREDICT` returns (`predict_tones` in the script, from the
earlier `fres`, `qres` and `ares`; `None` keeps them), and only the fq
sweep, fq_finder and tones steps run.

## Settings

{{settings:3_tune.py}}

| Setting | Meaning |
|---|---|
| `TUNING_NAME` | The tuning's name (or index): continued if it exists, else made (`None`: the latest, or a first one) |
| `RESONATORS` | Resonator list index (`None`: the latest) |
| `MIN_SEP_LW` | Drop tones closer than this to another resonance, linewidths |
| `PAIRS` | Close pairs: `'drop_both'`, or `'keep_stronger'` (the deeper dip) |
| `EDGE_MARGIN` | Least distance of a tone from an NCO window edge, Hz |
| `STREAM_DEC` | Fastest decimation stage that will be streamed |
| `N_CAL_TONES`, `CAL_MIN_SEP_LW` | Off-resonance calibration tones in the whole tuning (spread over its NCO windows, in the widest gaps; fewer when the gaps or the windows' room run out, which the plan step says), and their least distance from a resonance, linewidths |
| `WINDOWS` | NCO windows to use (`None`: all) |
| `FMIN`, `FMAX` | Tone frequency range, Hz (`None`: no limit) |
| `POWERS` | Power sweep, dBm |
| `SHUFFLE`, `SEED` | Randomize the order of each tone's powers |
| `NPOINTS_FINE`, `NPOINTS_GAIN` | Sweep points per tone |
| `GAIN_SPAN_FACTOR` | Gain sweep span, in fine sweep spans |
| `NSAMPS` | Samples averaged per point |
| `F_AVG` | Fine sweep points averaged in the fits |
| `A_FIT_RANGE` | $a_\mathrm{nl}$ range of the $a_\mathrm{nl}$ vs power fits |
| `ANL` | Target $a_\mathrm{nl}$: each tone's power is the one where its $a_\mathrm{nl}$ vs power fit reaches this |
| `ARES_LIMITS` | Chosen powers are clipped to this, dBm |
| `FALLBACK_NEAREST` | A tone whose power was not chosen (no fit) gets the median power of this many tones nearest to it in frequency |
| `FQ_WINDOW` | `False` (or `--no-fq-window`): no fq_finder window; every tone takes the automatic frequency update and keeps its span, nothing is rejected |
| `FRES_UPDATE_METHOD`, `NPOINTS_UPDATE` | fq_finder's starting frequencies: `'none'`, `'spacing'` (the point of fastest motion around the loop), `'spacing_fit'` (a Lorentzian fit to that motion, the point when the fit fails), `'minS21'`, `'distance'`; only the points within `NPOINTS_UPDATE` of the swept frequency are considered (`None`: all) |
| `RMV_GAIN_SIMPLE` | fq_finder: simple gain correction of the display |
| `PORT_TUNING_NAME`, `PORT_STORE` | A new tuning keeps the powers of this tuning (name or index; `None`: a full tuning) of this store (`None`) or the store at this path |
| `PREDICT` | `None`, or a function of the earlier `fres`, `qres`, `ares` returning `fres_pred`, `qres_pred`, `ares_pred` (`predict_tones` in the script) |
| `STEPS`, `REDOABLE` | The steps of a tuning in order, and those `--redo` may run again; leave them |

## Command line

`--modules`, `--tuning name`, `--redo <step>`, `--windows`,
`--list-windows`, `--port-tune name`, `--no-fq-window`, `--simulate`.

# 4a_stream.py: take timestreams

A window on one tuning (the latest with tones, or `--tuning name`; the
Tuning box at the top switches to another):

| Part | What it does |
|---|---|
| Calibration | Calibrate the selected tone set (a target sweep, saved as `cal{j}`), once per set; pick which saved calibration to use |
| Tones on the board | Write tones writes the chosen chunk of the set from its calibration (the chunks are planned for the stage set there: a fast stage takes fewer tones, so a big set is written and streamed one chunk at a time); Clear tones clears the board. Nothing else writes tones |
| Preview | Capture `PREVIEW_S` seconds of the tones on the board (not saved) |
| Plots | The capture or the viewed run, whichever was last (the plot title says which): one tone as x, parallel/perpendicular to the loop, or I and Q, its PSD with Show PSD, and without the mean of each series with Subtract offset |
| Run | Name, notes, decimation stage and duration (or a `PRESETS` entry); the tones on the board are streamed and saved as `cal{j}/write{w}/ts{n}` and listed in the run list; a name is used once per tuning |
| View run | Shows a run's first chunks like a capture |
| Adjust tones (`EXTRA_FEATURES`) | Saves the tuning's tones as a new, named tuning with every power offset by the dB given (or set to one power) and every frequency offset by a fraction of the total span (highest minus lowest tone), then switches to it; it needs its own calibrations |
| Grey buttons | A grey button says beside it what it needs |
| Cancel | Stops the running calibration, capture or run at its next sweep point or packet batch; what it saved is removed (the calibration group, or the run's streams and record). No shortcut, so it is not hit by accident |

A calibration is valid only while the NCO frequencies are unchanged, so
work one tone set at a time: calibrate, write the tones, preview and
run, then the next set. Restarting the script keeps the calibrations
usable unless something else set the NCOs in between (a power cycle,
`crs_init.py`). Tones stay written between captures and runs; a
calibration sweep clears them. The script holds the board while its
window is open. `--simulate` streams the store's simulated device.

## Settings

{{settings:4a_stream.py}}

| Setting | Meaning |
|---|---|
| `PREVIEW_DEC`, `PREVIEW_S` | Preview decimation stage and duration, s |
| `PRESETS` | Runs offered in the window: name, stage, duration, notes |
| `EXTRA_FEATURES` | `True`: show the Adjust tones box |
| `NPOINTS_FINE`, `NPOINTS_GAIN`, `GAIN_SPAN_FACTOR`, `NSAMPS` | Calibration sweeps, as in `3_tune.py` |
| `FRES_UPDATE` | `None`, or `dict(method = 'spacing', max_shift = 0.1, npoints_update = None)`: after each calibration sweep, move every tone to the resonance its fine sweep shows (`method`: `'spacing'`, `'spacing_fit'`, `'distance'` or `'minS21'`, as in `3_tune.py`; only the points within `npoints_update` of the swept frequency are considered, `None`: all), if the move is at most `max_shift` of the sweep span; a tone whose update is farther keeps its frequency. The frequencies swept are kept as `fres_sweep` |

## Command line

`--modules`, `--tuning name`, `--simulate`.

# 4b_stream_batch.py: timestreams without a window

Takes the tone sets in `TONE_SET` (one, a list, or every set of the
tuning) one after another: a set is calibrated (or its
`CALIBRATION_NAME` sweep reused), then every run in `RUNS` is taken
on it, each writing and streaming the tone chunks its stage needs
(or `TONE_CHUNKS`); then the next set. A calibration holds for one
set at a time, since calibrating moves the NCOs. With several sets
each run is saved once per set, named `<name>_set<s>`. The tones are
cleared at the end. Progress bars show the tone sets (with several),
each calibration sweep and each stream. Everything is set in the
script; nothing is asked. To take the same sets again with other
settings, loop over
the script's `take` (or `citkid.crs.procedures.stream_batch`) from
your own script, changing the settings between calls.

## Settings

{{settings:4b_stream_batch.py}}

| Setting | Meaning |
|---|---|
| `TONE_SET` | The tone set to calibrate and stream: an index, a list of indices in the order streamed, or `None` for every set of the tuning |
| `CALIBRATION_NAME` | `None`: take a calibration sweep per set; a name (`'dark_cal'`): reuse the tuning's calibration sweep of that name if it has one (it must be in frame), else take one and name it (with several sets, `dark_cal_set0`, `dark_cal_set1`, ...); or a saved sweep's path (`'tune0/cal2'`) to reuse, one set only |
| `FRES_UPDATE` | As in `4a_stream.py` |
| `TONE_CHUNKS` | Which of the chunks planned for a run's stage to stream, e.g. `[0, 1]` (`None`: all) |
| `RUNS` | The timestreams taken on each set, in order: `name` (with several sets, `<name>_set<s>` per set), `dec_stage`, `duration_s`, `notes` |
| `NPOINTS_FINE`, `NPOINTS_GAIN`, `GAIN_SPAN_FACTOR`, `NSAMPS` | Calibration sweep, as in `3_tune.py` |

## Command line

`--modules`, `--tuning name`, `--simulate`, `--notes`.

# 4c_ts_series.py: a series of timestreams

A loop over the points of an external parameter: the blackbody
power, the stage temperature, anything set between timestreams. At
each point of `POINTS` (a name, e.g. the blackbody temperature), the
script calls `set_point` (edit it for the setup; it also waits for
the parameter to settle), calibrates `TONE_SET` and takes `RUNS` as
`4b_stream_batch.py` does, naming each run `<point>_dec<stage>`
(`_2`, `_3`, ... when the name is taken). The point's name is saved
with its runs and its calibration sweep. A point with `cal` reuses
the calibration of an earlier point (its name) or a saved sweep (its
path) instead of taking one; this suits a single NCO window only,
since with several the sets are retuned at each point. A blackbody
series is analyzed for NEP by `6d_nep.py` ("NEP"), which maps the
point names to powers; a series taken by hand works the same way
there: name the runs, and give `POINTS` in 6d with the run names.

## Settings

{{settings:4c_ts_series.py}}

| Setting | Meaning |
|---|---|
| `TONE_SET` | Tone set calibrated and streamed at every point |
| `POINTS` | The points in order: `dict(name = ...)`, with `cal = 'name'` (an earlier point) or `cal = 'tune0/cal2'` to reuse a calibration |
| `RUNS` | Timestreams at each point: `dec_stage`, `duration_s` (and `notes`); the names come from the point |
| `TONE_CHUNKS`, `FRES_UPDATE`, `NPOINTS_FINE`, `NPOINTS_GAIN`, `GAIN_SPAN_FACTOR`, `NSAMPS` | As in `4b_stream_batch.py` |

## Command line

`--from name` (start at that point, e.g. after a stop), `--tuning
name`, `--simulate`, `--notes`.


# view_timestreams.py: look at saved timestreams

The plots of `4a_stream.py` on runs already in the store, with no
board: choose the tuning, a run, which zarr chunks to read (from
which chunk, how many) and the tone, then **View run** (or
double-click the run). A run of several tone sets lists every tone,
and the chosen tone picks the set. The timestream (x, parallel and
perpendicular to the loop, or I and Q, with Subtract offset), the
calibration loop with the timestream's points, and the PSD (Show
PSD); R rescales, H shows the help. `--tuning name` opens on a
tuning.

# 5_calibrate_streams.py: calibrate timestreams

`CALIBRATE` lists the runs to calibrate: by name or `run = n`, with the
zarr chunks to load (a few are enough) and optionally the tone sets.
Empty, the script lists the runs. The chosen parts of the runs taken
on one calibration sweep are joined into one DataSet (`set{s}`, or
`set{s}_cal{j}` when several sweeps of a tone set are chosen; one row
per tone) and calibrated (gain fit, IQ circle fit, x calibration),
first in the window and then in batch. A DataSet belongs to its
sweep: the analyses apply it only to that sweep's timestreams. The
result is `ts_cal{c}`, named `CALIBRATION_NAME`. The same `CALIBRATE`
continues in the latest one; `CALIBRATION_NAME` continues in the one with
that name, or starts a new one with it when none has it; another
choice or `--new` starts a new one.

In the window, Mark tone bad (B) or a panel's Bad Data asks a reason,
sets the tone's outputs to NaN and records the mark; Add note (T) keeps
the tone. Bad Data on the IQ fit panel (`FIT_IQ`) marks the fit bad
only: the mark is local, so the timestream stays good for the later
scripts, and the report's fit table shows the row as bad.

## Settings

{{settings:5_calibrate_streams.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATE` | Runs to calibrate, one `dict(...)` per run with the keys `name` (the latest run with it) or `run` (its index n in `ts{n}`); `chunks` (left out: all, `n`: the first n, `(first, stop)`); `tone_sets` (left out: all). Empty: list the runs |
| `CALIBRATION_NAME` | Name of the calibration: continued when it exists (e.g. to add the IQ fits to it), made otherwise; an index (`2`, `'ts_cal2'`) continues that one. `None`: the latest made with `CALIBRATE`, else a new one named `ts_cal{c}` |
| `F_AVG` | Fine sweep points averaged together |
| `INTERACTIVE` | Review in the window before the batch |
| `FIT_IQ` | Also fit each resonance (nonlinear IQ fit): an IQ fit panel in the window, the rest in batch; the fits go to the report |

## Command line

`--cal name`, `--new`, `--fit-iq`, `--no-gui`.
