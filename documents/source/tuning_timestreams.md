---
title: "Tuning and timestreams"
subtitle: "03_tune, 04_stream, 05_calibrate_streams, 06_find_photons"
---

# Overview

After VNA screening (see the VNA screening document), the tuning procedure
finds the optimal tone power and frequency for each resonator.
Timestreams are then taken with the tuned tones and calibrated, and the
photon pulses in them are found.

| Script | What it does | Saves |
|---|---|---|
| `03_tune.py` | Plans the tones, sweeps the tone power, and finds each tone's optimal power and frequency (with two interactive windows) | `tune{k}` |
| `04_stream.py` | Opens a window for previewing timestreams and taking named timestream runs | `ts{n}` |
| `05_calibrate_streams.py` | Calibrates selected timestreams, first in an interactive window, then in batch | `ts_cal{c}` |
| `06_find_photons.py` | Finds the photon pulses and cosmic rays in calibrated timestreams, with a window for choosing the thresholds | `ts_photons{p}` |

Before running the measurement scripts, make sure the board is
configured with `crs_init.py`. The "Getting started" document describes
it, along with the environment, how the scripts are run (settings,
command-line options, notes, `--simulate`), the device store, and the CRS
terms used here (modules, NCOs, decimation stages, ...).

A tuning group holds the tuning measurements, the tuned tones, and
everything streamed with them. Before streaming a tone set, `04_stream.py`
takes a calibration target sweep of it (`cal{j}`). The tones written to
the board for streaming are saved under that sweep (`write{w}`), and each
stream of those tones is saved under the write (`ts{n}`). A timestream
run, `ts{n}`, is one named timestream, usually of one tone set; the
tuning's run list (`streams`) records each run, with its streams.

```text
<device>.zarr/
  tune{k}/              tuning (03)
    cal{j}/             calibration target sweep of one tone set (04)
      write{w}/         tones written for streaming (the whole tone set,
                        or part of it at fast decimation stages)
        ts{n}/          stream of run n with these tones
    streams/            run list (attribute runs): per run, its name,
                        notes, decimation stage, duration, tone set,
                        calibration, and streams
  ts_cal{c}/            stream calibrations (05)
  ts_photons{p}/        photon results (06)
```

Calibrated timestreams are never saved. `05_calibrate_streams.py` saves
calibrations, and the analysis scripts (such as `06_find_photons.py`)
apply them to the raw streams they analyze, in memory, and save only
their results. Stores made by earlier versions, with the runs at the
store root (`ts{n}/set{s}/write{w}/stream`), can be converted with
`citkid.daq_layout.migrate_root_runs`.

## Tone sets

Each module reads out a 500 MHz band centered on its NCO frequency (600
MHz with extended bandwidth), with at most 1024 tones. We call this band
an NCO window. If the tones don't all fit in the available modules at
once, they are divided into **tone sets**. A tone set is the group of
tones written to the modules together, with one NCO frequency per module.
A sweep of all the tones sweeps one tone set after another and combines
(stitches) them into a single sweep. Each stream contains the tones of
one tone set.

The tone sets are assigned once, when the tuning is planned:

- The tones are covered with as few NCO windows as possible. Window edges
  are placed in the widest gaps between resonances, and every tone is at
  least `EDGE_MARGIN` from an edge.
- Windows are assigned to the modules in order, and each pass through the
  modules makes one tone set.
- If extended bandwidth is enabled on the board, a window can be 600 MHz
  wide, with its tones in the central 575 MHz.
- `STREAM_DEC` limits each window to the number of tones that can be
  streamed at once at that decimation stage (currently 128 at stages 0-1
  and 1024 at stages 2-6). This way the NCO frequencies never need to
  change during streaming.
- `WINDOWS` (`--windows`) selects a subset of the windows, to measure only
  part of the array. `--list-windows` prints the windows with their NCO
  frequencies, tone ranges, and numbers of tones.
- `FMIN` and `FMAX` (`--fmin`, `--fmax`, in Hz) keep only the resonators
  in a frequency range as tones, e.g. to leave out a few stragglers found
  far above the main band that you want fitted by 02 but not measured.
  Resonances outside the range still count as neighbours for `MIN_SEP_LW`.
  The tone sets are planned on the tones that remain.

# 03_tune.py: make a tuning

The script creates a new tuning, `tune{k}`, and runs these steps:

1. **plan**: Start from a resonator list (`resonators{j}`, the latest by
   default). Remove tones that are too close to another resonance, closer
   than `MIN_SEP_LW` times the larger of the two linewidths. `PAIRS` sets
   whether both tones of a close pair are removed, or only the tone of the
   shallower resonance (by the dip depths that `02_find_resonances.py`
   measures on the VNA sweep). Divide the remaining tones into tone
   sets. Optionally, add `N_CAL_TONES` off-resonance
   calibration tones per window (`res_idxs` -1, -2, ...). The initial
   tones are saved in `tones_init`, and the CRS settings, including each
   tone set's NCO frequencies, in `crs_config`.
2. **power_sweep**: Take a target sweep at each power in `POWERS`,
   stitched over the tone sets (`power_sweep/set{i}`). If the sweep is
   interrupted, the next run continues from the next power.
3. **power_fits**: The IQ series window fits the IQ loops at every power,
   then fits each resonator's nonlinearity parameter $a_\mathrm{nl}$ as a
   function of power (using $a_\mathrm{nl}$ values in `A_FIT_RANGE`). The
   power where $a_\mathrm{nl}$ reaches `ANL` is marked; Shift+click or
   drag the line to choose a different power. Resonators whose fit never
   reaches `ANL` get the median of the other chosen powers. The powers are
   then clipped to `ARES_LIMITS` and saved in `power_choice`.
4. **fq_sweep**: Take a stitched target sweep at the chosen powers.
5. **fq_finder**: In the fq_finder window, set each tone's frequency and
   sweep span, or reject the tone. If the window is closed before every
   tone is done, the next run continues where it left off.
6. **tones**: Save the tuned tones in `tones` (`fres`, `qres`, `ares`,
   `res_idxs`, `tone_set`, `tone_module`), without the rejected tones.

Steps that are already done are skipped, so the script picks up where it
stopped. `--redo power_fits`, `--redo fq_finder`, or `--redo tones`
repeats that step and the steps after it (`power_fits` only until the fq
sweep has been taken). Sweeps are never retaken in an existing tuning; use
`--new` to start a new tuning instead.

## Contents of a tuning

```text
tune{k}/               attributes: resonators (the resonator list that
                       res_idxs refer to)
  crs_config/          modules, analog bank, bandwidth, module settings,
                       and the NCO frequencies of each tone set
  tones_init/          initial tones and the settings used to plan them
  power_sweep/set{i}   stitched target sweep at each power (raw)
  power_fits/          IQ loop fits (analysis)
  power_choice/        chosen powers
  fq_sweep/            stitched target sweep at the chosen powers, with
                       the fq_finder results fres_opt, qres_opt,
                       reject_reason
  tones/               tuned tones
```

`tones/`, `crs_config/`, and the resonator list contain everything needed
to measure with the tuning. `citkid.daq_layout.port_tuning` copies only
these to another store (for example, for a new cooldown), without the
sweeps.

## Settings

{{settings:03_tune.py}}

## Command line

{{help:03_tune.py}}

# 04_stream.py: take timestreams

The script opens a window for one tuning (by default, the latest tuning
that has tuned tones). The window has three parts:

- **Calibration** (saved): Choose a tone set and the sweep settings (fine
  and gain sweep points, samples per point, gain sweep span; the defaults
  are the settings below) and press Calibrate. This takes a target sweep
  of the tone set and saves it as the tuning's next `cal{j}`. A
  calibration takes minutes, so each tone set is calibrated once, and
  every preview and run of the set after that uses the calibration. The
  list shows the set's saved calibrations; the latest usable one is
  chosen, and you can choose another.

  A calibration is only valid while the NCO frequencies stay as they were
  when it was taken. Calibrating or streaming another tone set sets other
  NCO frequencies, after which the earlier calibrations are marked as
  stale and can't be used. So work one tone set at a time: calibrate it,
  then preview and run as often as needed, then go on to the next set.
- **Preview** (not saved): Capture writes the calibration's tones and
  streams for a few seconds (`PREVIEW_S`) at the chosen decimation stage.
  Select a tone to plot its timestream, either as the fractional
  frequency shift x (the projection onto the tangent of the IQ loop at
  the tone frequency), as the components along and across the loop, or as
  I and Q. The calibration loop and the timestream are also plotted in
  the IQ plane.
- **Run** (saved): Enter a name, notes, a decimation stage, and a
  duration, or pick a preset from `PRESETS`, and run. The selected tone
  set is streamed with its chosen calibration. The stream is saved under
  the calibration and the tone write that put its tones on the board
  (`cal{j}/write{w}/ts{n}`), and the run is added to the tuning's run
  list. At fast decimation stages, a tone set with too many tones to
  stream at once is streamed in several writes, with the same NCO
  frequencies. The window lists the runs taken with this tuning.

Tones stay written on the board between captures and runs. A capture or
run of the same tones with the same calibration streams them without
writing them again (the window shows "Tones on the board"). Write tones
writes the selected set again on purpose. A different tone set, a
different part of a set, or a new calibration sweep (which clears the
board) means a new write.

Streaming uses the board's full network bandwidth, so this script needs
exclusive use of the board for as long as its window is open. It takes
global control of the board (to change the decimation stage) and creates
a lock file on the DAQ computer (`~/.citkid/locks/crs<serial>.lock`). It
won't start while another script is sweeping or streaming on the board.
While its window is open, any other script that tries to use the board
(`vna_quick_check.py`, `01_vna_sweep.py`, `03_tune.py`, another
`04_stream.py`, ...) stops immediately with an error that names the
running `04_stream.py` session. Close the window to release the board. If
a session crashes, its lock is ignored.

With `--simulate`, the script streams the simulated device of a store made
by `simulate_store.py`. With `SIM_PULSES = True` (or `--sim-pulses`), the
simulated streams also contain photon pulses on each tone and cosmic rays
on all tones at once, set by `SIM_PULSE_SETTINGS`, so that
`06_find_photons.py` can be tried on them. While pulses are on, the
simulated board uses the signal-to-noise ratio `SIM_SNR_DB` (80 dB by
default), because the stores' default noise level hides photon-sized
pulses at decimation stages 0 and 1.

## Settings

{{settings:04_stream.py}}

## Command line

{{help:04_stream.py}}

# 05_calibrate_streams.py: calibrate timestreams

List the timestreams to calibrate in `CALIBRATE`. Each entry selects:

- a run of `04_stream.py`, by name (the latest run with that name) or by
  index (`run = n` for `ts{n}`);
- the part of the run to load, in zarr chunks along time (`chunks`: all
  chunks, the first n, or `(first, stop)`). A few chunks are usually
  enough, and loading only a few keeps the calibration fast;
- optionally, a subset of the tone sets.

If `CALIBRATE` is empty, the script lists the available runs with their
names, tunings, decimation stages, durations, tone sets, numbers and
lengths of chunks, and notes.

The selected timestreams are loaded into one DataSet per tone set (and
resonator list), named `set{s}`. Each row is one tone. Its timestream is
the selected parts of every chosen run of that tone set, joined end to
end, so the calibration is fit once on all of them, even if the runs
have different decimation stages. `write_idx` records which tones were
streamed together (for common-mode removal). The rows are calibrated
(gain fit, IQ circle fit, and x calibration), first in the interactive
window and then in batch for the remaining rows. Calibration tones get
their own DataSet, calibrated with the 'ts_offres' path.

The calibration is not applied to the runs here, and no calibrated
timestream is saved. Later scripts apply the calibration to whichever raw
timestreams they analyze, in memory (see `06_find_photons.py`).

The results are saved in a `ts_cal{c}` group at the store root. Running
the script again with the same `CALIBRATE` continues in the same group; a
different `CALIBRATE`, or `--new`, starts a new group.

## Settings

{{settings:05_calibrate_streams.py}}

## Command line

{{help:05_calibrate_streams.py}}

# 06_find_photons.py: find photons

Choose a stream calibration made by `05_calibrate_streams.py`
(`CALIBRATION`, or `--cal c`; by default the latest `ts_cal{c}`) and the
timestreams to search (`TIMESTREAMS`). In this script, a timestream is
one tone set of one run, over the whole run. Any run can be searched if
the calibration covers its tone set; it doesn't have to be one of the
runs the calibration was fit on. The calibration is applied to the raw
streams in memory; the calibrated timestreams are not saved.

If `TIMESTREAMS` is empty, the script lists the timestreams it can search
(run, name, tone set, duration, number of tones) and the previous results
of the script, and stops. Otherwise, list the timestreams to search in
`TIMESTREAMS`, by index (as listed) or by run name. A run name selects
every tone set of the latest run with that name.

The script can be run as often as needed, with different settings or
timestreams. Each run either searches from scratch, with the threshold
window, or reuses a previous result (see "Reusing a previous result").

The detection timestreams are kept in memory up to `MEMORY_GB`, so the
threshold window can rerun tones quickly. Beyond that, they are computed
again from the raw streams when needed.

## Photon detection

Each tone is searched with an optimal filter that is built iteratively:

1. **First pass**: the peaks of the timestream (smoothed over 3 samples)
   above `INIT_NSIGMA` times the noise are found. Peaks higher than
   `INIT_HMAX` times the median peak height are left out. This cut is
   loose on purpose: it only keeps large pulses, such as cosmic rays, out
   of the first template, so that as many photons as possible are kept.
2. **Optimal filter**: the selected pulses are averaged into a template,
   the noise spectrum is measured away from the pulses, and the optimal
   filter is built from the two. The filtered timestream gives each pulse
   a height in units of the mean photon pulse, so the thresholds don't
   depend on the tone's responsivity. The pulses with heights between
   `HMIN` and `HMAX` are used for the next template. This is repeated
   `OF_ITERATIONS` times. Optionally, `HMIN` is chosen automatically
   (`HMIN_FALSE_FRAC`) as the lowest height at which few of the accepted
   pulses are expected to be noise.

## Choosing the thresholds

When searching from scratch, a window opens after the detection of each
timestream (skip it with `INTERACTIVE = False` or `--no-gui`). The left side lists the tones
with their number of pulses and their rate. The plots show, for the
selected tone or for all tones together:

- the first-pass peak heights, in units of their median, with the
  `INIT_HMAX` cut;
- the heights of the filtered pulses in units of the mean photon, with the
  expected number of noise peaks overlaid (measured from the filter output
  with the sign flipped, which contains noise but no pulses), and the
  `HMIN` and `HMAX` lines;
- the mean pulse of the selected tone.

Drag the lines or type the values, then press Rerun (Ctrl+R). With "Apply
to all tones", the values become the global thresholds, and every tone
without its own thresholds is run again. With "Apply to this tone", the
selected tone gets its own thresholds (marked with `*` in the list). "Use
the global thresholds" removes them. Close the window (or press Done) to
continue.

## Reusing a previous result

Set `REFERENCE` (or `--reference p`) to a previous result of the script,
`ts_photons{p}`, to reuse its work instead of searching from scratch.
For example, search one timestream from scratch and check its thresholds
in the window, then search other timestreams with that result as the
reference. Each timestream uses the reference's result for the same tone
set (or, if it has none, its first one). The window doesn't open.
`REUSE` (or `--reuse`) sets what is reused:

- `'settings'` (default): the reference's thresholds, including the ones
  set for single tones. The full detection is run on each timestream.
- `'filter'`: the reference's optimal filter for each tone (matched by
  resonator index), built from its mean pulse and noise spectrum, and its
  thresholds. The filter is applied without iterating, and the pulse
  heights are in units of the reference's mean photon. Tones that the
  reference has no filter for are detected from scratch. The timestreams
  must have the reference's decimation stage.

## Coincidences and cosmic rays

After the detection, the tones of each tone write (the tones streamed
together) are compared. A pulse is coincident if another tone in the same
write has a pulse within `COINC_WIN` samples before or after it (2 samples
by default: 52 µs at decimation stage 0). For this comparison, pulses down
to `COINC_NSIGMA` times the filter noise count, so that small pulses of a
cosmic ray on other tones are also found.

- **Photons**: each tone's mean photon pulse averages the pulses with
  heights between `HMIN` and `HMAX` that are not coincident with any
  other tone. Its decay time is fit.
- **Cosmic rays**: the pulses of all tones in a write are grouped into
  events. An event starts at the earliest pulse not yet in an event and
  includes every pulse up to 2 × `COINC_WIN` samples later. Events seen by
  at least `COSMIC_MIN_MULT` tones (3 by default) are saved as cosmic
  rays, with the time, the number of tones, and for each tone its pulse
  height (from the filter, and the raw peak of the timestream, since the
  filter height is biased for pulses longer than a photon pulse).

## What is saved

Each run of the script makes a new `ts_photons{p}` group at the store
root, with the calibration it used (attribute `source`) and one
`timestream{j}` group per listed timestream:

```text
ts_photons{p}/          attributes: source (ts_cal{c}), timestreams,
                        reference, reuse
  timestream{j}/        attributes: run, tone set, calibration sweep and
                        writes, samples, sample period, how it was
                        detected, thresholds, settings
                        per tone: res_idxs, status (empty if detected),
                        hmin, hmax, sigma_y, amp_ph (the height unit,
                        in x), amp_ph_own (this timestream's mean photon
                        height), rate, n_used, tau, template,
                        avg_pulse (mean photon pulse), psd, filter
    pulses/             every pulse found: tone, sample, height, raw peak,
                        candidate (at least hmin), in_window, mult
                        (number of tones), used (in the mean photon pulse)
    cosmic/events/      t0 (sample), time_s, mult, tone write
    cosmic/hits/        one row per tone of each event: tone, sample,
                        height (photons and x), raw peak
```

## Settings

{{settings:06_find_photons.py}}

## Command line

{{help:06_find_photons.py}}
