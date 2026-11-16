---
title: "Tuning and timestreams"
subtitle: "03_tune, 04_stream, store_browser, 05_calibrate_streams, 06a_find_photons, 06b_cosmic_rays, 07_psds, 08_report"
---

# Overview

After VNA screening: tune the tones, take timestreams, calibrate them,
find photons or cosmic rays, and compute noise spectra. Configure the
board with `crs_init.py` first (see "Getting started").

| Script | What it does | Saves |
|---|---|---|
| `03_tune.py` | Plans the tones, sweeps the power, chooses each tone's power and frequency (two windows) | `tune{k}` |
| `04_stream.py` | Window for calibrating, previewing and taking named timestream runs | `ts{n}` |
| `store_browser.py` | Any time: browse a store (groups, attributes, runs, tone marks) | Nothing |
| `05_calibrate_streams.py` | Calibrates timestreams, in a window and then in batch | `ts_cal{c}` |
| `06a_find_photons.py` | Finds photons and cosmic rays, with a threshold window | `photons/<name>` |
| `06b_cosmic_rays.py` | Fits cosmic-ray tails pulse by pulse, with a threshold window | `cosmic/<name>` |
| `07_psds.py` | Noise spectra before and after common-mode removal | `ts_psd{p}` |
| `08_report.py` | Lifetimes and noise spectra as figures, tables and text | Files in a folder |

```text
<device>.zarr/
  tune{k}/              tuning (03)
    cal{j}/             calibration target sweep of one tone set (04)
      write{w}/         tones written for streaming
        ts{n}/          stream of run n with these tones
    streams/            run list
  ts_cal{c}/            stream calibrations (05)
  photons/<name>/       photon results (06a), one per timestream
  cosmic/<name>/        cosmic-ray tails (06b), one per timestream
  ts_psd{p}/            noise spectra (07)
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
resonator index, with the stage and the time. A bad tone is dropped by
the stages after (the tuned set, averages and fits) until the mark is
cleared. `store_browser.py` lists the marks (Tone marks tab).

| Where | Mark bad | Note |
|---|---|---|
| `03_tune.py`, IQ series window | Mark Tone Bad (Ctrl+M); Mark All Series Bad asks keep (note) or remove (bad) | Add Note (Ctrl+T) |
| `03_tune.py`, fq_finder window | Rejecting a tone; un-rejecting clears it | N |
| `05_calibrate_streams.py` window | Mark tone bad (M), or a panel's Bad Data | T |
| `06a`, `06b` windows | B (again to clear) | T |
| `07_psds.py` | No window: marked tones are left out, plus `BAD_TONES` | |

Resonances the plan drops as too close get a note automatically.

# 03_tune.py: make a tuning

Creates `tune{k}` in these steps. Done steps are skipped; `--redo
power_fits`, `fq_finder` or `tones` repeats one and those after it;
sweeps are never retaken (`--new` starts a new tuning).

| Step | What it does | Saves |
|---|---|---|
| plan | From a resonator list, drops tones within `MIN_SEP_LW` linewidths of another (`PAIRS`: both, or the shallower), plans the tone sets, adds `N_CAL_TONES` off-resonance tones per window | `tones_init`, `crs_config` |
| power_sweep | Target sweep at each power in `POWERS` | `power_sweep/set{i}` |
| power_fits | IQ series window: IQ loop fits at every power, $a_\mathrm{nl}$ vs power per resonator; the power where $a_\mathrm{nl}$ reaches `ANL` is marked (Shift+click or drag to choose another); no fit: the median of the others; clipped to `ARES_LIMITS` | `power_fits`, `power_choice` |
| fq_sweep | Target sweep at the chosen powers | `fq_sweep` |
| fq_finder | Window: set each tone's frequency and span, or reject it; resumes where it stopped | `fres_opt`, `qres_opt`, `reject_reason` |
| tones | The tuned tones, without rejected and bad-marked ones | `tones` |

`citkid.daq_layout.port_tuning` copies `tones/`, `crs_config/` and the
resonator list to another store (a new cooldown) without the sweeps.

## Settings

{{settings:03_tune.py}}

| Setting | Meaning |
|---|---|
| `MODULES` | Modules connected to the device |
| `RESONATORS` | Resonator list index (`None`: the latest) |
| `MIN_SEP_LW` | Drop tones closer than this to another resonance, linewidths |
| `PAIRS` | Close pairs: `'drop_both'`, or `'keep_stronger'` (the deeper dip) |
| `EDGE_MARGIN` | Least distance of a tone from an NCO window edge, Hz |
| `STREAM_DEC` | Fastest decimation stage that will be streamed |
| `N_CAL_TONES`, `CAL_MIN_SEP_LW` | Off-resonance calibration tones per window, and their least distance from a resonance, linewidths |
| `WINDOWS` | NCO windows to use (`None`: all) |
| `FMIN`, `FMAX` | Tone frequency range, Hz (`None`: no limit) |
| `POWERS` | Power sweep, dBm |
| `SHUFFLE`, `SEED` | Randomize the order of each tone's powers |
| `NPOINTS_FINE`, `NPOINTS_GAIN`, `NPOINTS_ROUGH` | Sweep points per tone (rough: `None` for none) |
| `GAIN_SPAN_FACTOR` | Gain sweep span, in fine sweep spans |
| `NSAMPS` | Samples averaged per point |
| `F_AVG` | Fine sweep points averaged in the fits |
| `A_FIT_RANGE` | $a_\mathrm{nl}$ range of the $a_\mathrm{nl}$ vs power fits |
| `ANL` | Target $a_\mathrm{nl}$ |
| `ARES_LIMITS` | Chosen powers are clipped to this, dBm |
| `FRES_UPDATE_METHOD` | fq_finder: `'none'`, `'spacing'`, `'minS21'`, `'distance'` |
| `RMV_GAIN_SIMPLE` | fq_finder: simple gain correction of the display |
| `STEPS`, `REDOABLE` | The steps; leave them |

## Command line

`--modules`, `--new`, `--redo <step>`, `--windows`, `--list-windows`,
`--simulate`.

# 04_stream.py: take timestreams

A window on one tuning (the latest with tones, or `--tune k`):

| Part | What it does |
|---|---|
| Calibration | Calibrate the selected tone set (a target sweep, saved as `cal{j}`), once per set; pick which saved calibration to use |
| Preview | Capture `PREVIEW_S` seconds and plot one tone as x, along/across the loop, or I and Q (not saved) |
| Run | Name, notes, decimation stage and duration (or a `PRESETS` entry); the stream is saved as `cal{j}/write{w}/ts{n}` and listed in the run list |
| View run | Shows a run's first chunks like a capture |

A calibration is valid only while the NCO frequencies are unchanged, so
work one tone set at a time: calibrate, preview and run, then the next
set. Restarting the script keeps the calibrations usable unless something
else set the NCOs in between (a power cycle, `crs_init.py`). Tones stay
written between captures and runs. The script holds the board while its
window is open. `--simulate` streams a simulated device; `--sim-pulses`
adds photons and cosmic rays to it.

## Settings

{{settings:04_stream.py}}

| Setting | Meaning |
|---|---|
| `MODULES` | Modules connected to the device |
| `PREVIEW_DEC`, `PREVIEW_S` | Preview decimation stage and duration, s |
| `PRESETS` | Runs offered in the window: name, stage, duration, notes |
| `NPOINTS_FINE`, `NPOINTS_GAIN`, `GAIN_SPAN_FACTOR`, `NSAMPS` | Calibration sweeps, as in `03_tune.py` |
| `SIM_PULSES`, `SIM_PULSE_SETTINGS` | `--simulate`: photon pulses and cosmic rays in the streams |
| `SIM_SNR_DB` | Simulated S/N while pulses are on, dB (`None`: the store's) |

## Command line

`--modules`, `--tune k`, `--simulate`, `--sim-pulses`.

# store_browser.py: look around a store

Opens a store read-only (another script can keep writing; **F5**
rereads): a tree of its groups with one-line summaries, the selected
item's attributes, the timestream runs (filter by typing, double-click
to find the stream) and the tone marks. **Open** (Ctrl+O) chooses
another store; `--pick` starts with the dialog; `--runs` prints the runs
instead of opening the window.

## Settings

{{settings:store_browser.py}}

| Setting | Meaning |
|---|---|
| `PICK` | Choose the store in a file dialog |

# 05_calibrate_streams.py: calibrate timestreams

`CALIBRATE` lists the runs to calibrate: by name or `run = n`, with the
zarr chunks to load (a few are enough) and optionally the tone sets.
Empty, the script lists the runs. The chosen parts of every run of a
tone set are joined into one DataSet (`set{s}`, one row per tone) and
calibrated (gain fit, IQ circle fit, x calibration), first in the window
and then in batch. The result is `ts_cal{c}`; the same `CALIBRATE`
continues in it, another or `--new` starts a new one.

In the window, Mark tone bad (M) or a panel's Bad Data asks a reason,
sets the tone's outputs to NaN and records the mark; Add note (T) keeps
the tone.

## Settings

{{settings:05_calibrate_streams.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATE` | Runs to calibrate: `name` (the latest run with it) or `run = n`; `chunks` (`None`: all, `n`: the first n, `(first, stop)`); `tone_sets` (`None`: all) |
| `F_AVG` | Fine sweep points averaged together |
| `INTERACTIVE` | Review in the window before the batch |

## Command line

`--new`, `--no-gui`.

# 06a_find_photons.py: find photons

Choose a stream calibration (`CALIBRATION`, the latest by default) and
the timestreams (`TIMESTREAMS`, by index as listed or run name; empty
lists them). A timestream is one tone set of one run. Tones whose
calibration failed are left out. Each result is `photons/<name>` (the
run's name, `_set<s>` with several tone sets, or `NAME`), one per
timestream: when it exists, a popup offers to continue from it
(default), overwrite it (asks again) or skip; with `--no-gui`,
`--continue` or `--overwrite` decide.

## Detection

| Step | What it does |
|---|---|
| First pass | Peaks above `INIT_NSIGMA` noise sigma, below `INIT_HMAX` times the median peak, seed the template |
| Optimal filter | Template from the selected pulses, noise spectrum away from them; heights in kHz/GHz (x × 1e6); pulses in [`HMIN`, `HMAX`] make the next template, `OF_ITERATIONS` times; `HMIN_FALSE_FRAC` chooses `HMIN` automatically |
| Isolation | Only isolated pulses enter a template: none within the template length, none within the filter length of a pulse above `HMAX` |
| Coincidences | Per tone write, a pulse within `COINC_WIN` samples of another tone's pulse is coincident; pulses down to `COINC_NSIGMA` filter sigma count |
| Photons | The mean photon pulse averages the uncoincident pulses in the height window; its decay time is fit from `TAU_SKIP` samples after the maximum over `TAU_FIT_S` |
| Cosmic rays | Events seen by at least `COSMIC_MIN_MULT` tones, with each tone's pulse height |

`HMAX` keeps large pulses out of the template and mean pulse only; every
pulse above `HMIN` is saved.

## The window

Opens after each timestream's detection (`--no-gui` skips it). Left: the
tone list (`*` reviewed, `x` bad), the thresholds, the decay fit, and
the buttons. Right, for the selected tone (or all tones together):

| Plot | Shows |
|---|---|
| First-pass heights | In units of their median, with the `INIT_HMAX` line |
| Filter heights | In kHz/GHz with the expected noise peaks (red) and the `HMIN`, `HMAX` lines |
| Mean pulse | With the decay fit and the end of the filter's template |
| Snippet | The detection timestream (grey) and the filter output (blue), the height window as lines, filter peaks marked: green in the window, red below `HMIN`, orange above `HMAX`; slider and Prev/Next move along the record |

| Action | How |
|---|---|
| Set a threshold | Drag a line, type in a box, click a histogram (moves the nearer line), or drag on the snippet (sets `HMIN` and `HMAX` from the drag's bottom and top) |
| Apply | Rerun (Ctrl+R): to this tone (its own values) or to all tones (the global values; the scope goes back to this tone). A tone shows `*` once reviewed: given its own values, or looked at and left; apply to all does not count |
| Decay fit | Skip and length boxes, Apply fit (Ctrl+F), this tone or all; no detection, so immediate |
| Move | A / D, Prev / Next (pending thresholds are applied first) |
| Units | Heights in kHz/GHz or photons (each tone's mean photon) |
| Revert | Ctrl+D puts the script's thresholds back on the tone |
| Mark bad / note | B (reason; again to clear), T (note) |
| Overview | O: histograms over all tones of photon height, decay time and rate, with draggable review limits; click a bar or a listed tone to open it; "show only tones outside" filters the list |
| Rescale | R |

## Saved results, continuing and reusing

- Continue (the popup, or `--continue`) opens the saved result in the
  window with no detection; only changed tones rerun.
- `--refit-tau` refits every saved decay time with `TAU_SKIP` and
  `TAU_FIT_S`, no detection.
- `REFERENCE = '<name>'` searches other timestreams with a saved result,
  without the window: `REUSE = 'settings'` (its thresholds) or
  `'filter'` (its optimal filters, same decimation stage).

```text
photons/<name>/         attrs: run, tone set, calibration, mode,
                        thresholds, settings
                        per tone: res_idxs, status, hmin, hmax, sigma_y,
                        amp_ph, amp_ph_own, rate, n_used, tau, tau_skip,
                        tau_fit_s, template, avg_pulse, psd, filter,
                        bad, bad_reason
    pulses/             every pulse: tone, sample, height, raw peak,
                        candidate, in_window, mult, used
    cosmic/events/      t0, time_s, mult, write_idx
    cosmic/hits/        per tone and event: sample, height, raw peak
    detail/             the detection detail, for continuing
```

Results of earlier versions (`ts_photons{p}`) are moved into this layout
by `citkid.pipeline.photon_finder.migrate_results(root)`.

## Settings

{{settings:06a_find_photons.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATION` | `ts_cal` index (`None`: the latest) |
| `TIMESTREAMS` | Timestreams, by index as listed or run name; empty: list them |
| `NAME` | Result name for a single timestream (`''`: the run's name) |
| `REFERENCE`, `REUSE` | A saved result to reuse (its name), and `'settings'` or `'filter'` |
| `INTERACTIVE` | Check the thresholds in the window |
| `INIT_NSIGMA`, `INIT_HMAX` | First pass: peak threshold (noise sigma) and upper cut (times the median peak) |
| `OF_ITERATIONS` | Optimal-filter iterations |
| `HMIN`, `HMAX` | Height window of single photons, kHz/GHz |
| `HMIN_FALSE_FRAC`, `HMIN_AUTO_MAX` | Choose `HMIN` automatically (`None`: fixed), up to this height |
| `TEMPLATE_PRE_S`, `TEMPLATE_POST_S` | Template before and after the maximum, s |
| `FILTER_POST_S` | The filter uses this much of the template after the maximum, s |
| `TAU_SKIP`, `TAU_FIT_S` | Decay fit: samples skipped after the maximum, fit length, s |
| `HIGHPASS_HZ` | Drift removal, Hz |
| `COINC_NSIGMA`, `COINC_WIN` | Coincidences: pulses counted (filter sigma), window (± samples) |
| `COSMIC_MIN_MULT` | Fewest tones of a cosmic ray |
| `MEMORY_GB` | Timestreams kept in memory, GB |

## Command line

`--cal c`, `--name`, `--continue`, `--overwrite`, `--reference name
--reuse filter`, `--refit-tau`, `--no-gui`.

# 06b_cosmic_rays.py: cosmic-ray lifetimes

Cosmic rays hit many tones simultaneously, so a device without photons
still gives each tone's quasiparticle lifetime. Same timestream choice,
naming and existing-result popup as `06a_find_photons.py`; results are
`cosmic/<name>`.

| Step | What it does |
|---|---|
| Events | `EVENTS = 'photons'`: the cosmic rays `06a` saved (or those of `PHOTONS`); `'triggers'`: peaks above `TRIG_NSIGMA` on each tone, clustered within `COINC_WIN` samples, kept when `COSMIC_MIN_MULT` tones see them; `'auto'`: photons if saved, else triggers |
| Pulses | A snippet `PRE_S` before to `POST_S` after each event, baseline subtracted; the peak within `SEARCH_S` |
| Threshold | Only pulses with peaks in [`HMIN`, `HMAX`] are averaged |
| Alignment | The pulses differ in height and near the peak but decay alike, so each is shifted to where it falls through `LEVEL` (10 kHz/GHz) and they are averaged from there |
| Tail fit | An exponential over `FIT_S` from the alignment point of the mean |
| Result | Per tone, the mean's decay time and the pulses averaged |

## The window

Tone list, boxes for the threshold (`HMIN`), the level and the fit
length, Apply (Ctrl+R) to this tone or all. Plots: the whole timestream
with the pulses found (green averaged, red left out) and the threshold
line; the aligned pulses (blue) with their mean (white), the fit
(yellow) and the level line. Drag a line or edit a box, then Apply.
The plots zoom with the mouse (scroll, drag, right-drag for one axis);
R puts them back: the timestream's heights up to a little above the
99th percentile of the pulses, and the mean pulse over the fit range up
to its own peak. B marks bad, T adds a note, O opens the overview
(median height, decay time, pulse rate).

```text
cosmic/<name>/          attrs: run, tone set, calibration, events,
                        n_events, thresholds, settings
                        per tone: res_idxs, status, hmin, hmax, level,
                        fit_s, n_pulses, n_used, rate, tau, n_fit,
                        avg_pulse (the aligned mean; attr anchor), bad,
                        bad_reason
    events/             t0, time_s, mult, write_idx
    pulses/             per tone and event: t0, h, p, in_window, start,
                        used
```

## Settings

{{settings:06b_cosmic_rays.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATION`, `TIMESTREAMS`, `NAME`, `INTERACTIVE` | As in `06a_find_photons.py` |
| `EVENTS` | `'auto'`, `'photons'`, or `'triggers'` |
| `PHOTONS` | Photon result to take the events from (`None`: this timestream's) |
| `TRIG_NSIGMA`, `COINC_WIN`, `COSMIC_MIN_MULT` | Triggers: threshold (noise sigma), largest gap within an event (samples), fewest tones |
| `HMIN`, `HMAX` | Peak height window of the pulses fitted, kHz/GHz |
| `LEVEL` | A tail begins where the pulse falls below this, kHz/GHz |
| `FIT_S` | Fit length, s |
| `PRE_S`, `POST_S` | Snippet before and after an event, s |
| `SEARCH_S` | Peak search after the event, s |
| `HIGHPASS_HZ` | Drift removal, Hz |
| `MEMORY_GB` | Timestreams kept in memory, GB |

## Command line

`--cal c`, `--name`, `--events triggers`, `--photons name`,
`--continue`, `--overwrite`, `--no-gui`.

# 07_psds.py: noise spectra

`TIMESTREAMS` chooses timestreams (index, run name, or a dict with the
tone sets and calibration; empty lists them). The timestreams of one run
of the script are saved together in `ts_psd{p}` with one grid of reduced
frequencies, named by `NAME`. Tones marked bad, and `BAD_TONES`, are
left out.

| Step | What it does |
|---|---|
| Deglitch (optional) | `'peaks'`: each tone's peaks above `GLITCH_NSTD` sigma; `'photons'`: a photon result's filters (`PHOTONS`) and its cosmic rays (`GLITCH_PULSES = 'all'`: every pulse); `GLITCH_PRE_S` to `GLITCH_POST_S` around each glitch is replaced with noise |
| Spectra | $S_{xx}$ per tone: reduced values (the mean within 20% of 0.1, 0.3, 1, 3, 10, ... Hz, from `FMIN_MULT` / duration to Nyquist / 2 / `FMAX_DIV`) and `NBINS` logarithmic bins; also the spectra of the loop components, along the loop ($r\theta$) and across it ($A$), and their ratio in dB at the reduced frequencies, the s-factor |
| Common modes | Per tone write, `CN_N_COMP` modes by `citkid.xcal.corr.calc_cm` (`CN_N_ITER` iterations; highpass `CN_HIGHPASS_HZ` while finding the weights, lowpass `CN_LOWPASS_HZ` on the modes while iterating; the removed modes are unfiltered), then the spectra again on the cleaned tones, and the modes' spectra |
| Review | The overview window on the raw spectra, before the common modes (skip it with `INTERACTIVE = False` or `--no-gui`); tones marked bad there stay out of the common-mode fit |

## The window

Three histograms over all tones of the raw $S_{xx}$ (1/Hz, log axis)
at frequencies picked from the reduced grid, each with
draggable review limits; the histograms do not zoom. The list shows the
tones outside the limits (or all), with their values; click a tone, or
near its bar, to see its binned spectrum. B marks the tone bad with a
reason (saved with the result and in the store's tone marks; again to
clear), T adds a note. The limits are saved with the result. The common
modes are then found without the bad tones.

```text
ts_psd{p}/              attrs: name, timestreams, settings, reference
                        f_reduced
  timestream{j}/        attrs: run, tone set, calibration, duration, bins
                        f_binned, valid_reduced
                        per tone: res_idxs, write_idx, status, bad,
                        bad_reason, n_glitches, glitch_frac,
                        sxx_reduced, sxx_binned, spar_binned,
                        sper_binned, sfactor_reduced, sxx_reduced_cn,
                        sxx_binned_cn, cn_weights
    common_modes/write{k}/   rows, f_binned, sxx_binned, sigma_iter
```

`citkid.pipeline.psds.load_group(root, p)` reads a result back, by index
or name.

## Settings

{{settings:07_psds.py}}

| Setting | Meaning |
|---|---|
| `TIMESTREAMS` | Index as listed, run name, or `dict(name = ..., or run = n, tone_sets = [...], cal = c)`; empty: list them |
| `NAME` | Name of this set of spectra |
| `INTERACTIVE` | Review each timestream in the overview window |
| `NBINS` | Logarithmic bins |
| `FMIN_MULT` | Lowest frequency, times 1 / duration |
| `FMAX_DIV` | Highest reduced frequency: Nyquist / 2 / this |
| `CN_REMOVE`, `CN_N_COMP`, `CN_N_ITER` | Common-mode removal, components, iterations |
| `CN_LOWPASS_HZ` | On the modes while iterating (`None`: Nyquist / 10) |
| `CN_HIGHPASS_HZ` | While finding the weights (`None`: `FMIN_MULT` / duration; `0`: off) |
| `CN_FILTER_ORDER` | Order of both filters |
| `BAD_TONES` | Resonators left out, besides the tone marks |
| `DEGLITCH` | `None`, `'peaks'`, or `'photons'` |
| `GLITCH_PRE_S`, `GLITCH_POST_S` | Replaced around each glitch, s |
| `GLITCH_SIGN`, `GLITCH_NSTD`, `GLITCH_BASELINE_S`, `GLITCH_DISTANCE_S`, `GLITCH_ROUNDS` | `'peaks'`: sign of a glitch in x, threshold (sigma), baseline bin (s), least spacing (s), rounds |
| `PHOTONS`, `GLITCH_PULSES` | `'photons'`: the photon result, and `'cosmic'` or `'all'` pulses |
| `COINC_WIN`, `COSMIC_MIN_MULT` | Cosmic rays: window (± samples), fewest tones |
| `MEMORY_GB` | Timestreams kept in memory per write, GB |

## Command line

`--name`, `--no-gui`, `--no-cn`, `--deglitch photons --photons name`.

# 08_report.py: results in plain files

Writes every timestream result of a store to a folder, after
`06a_find_photons.py`, `06b_cosmic_rays.py` and `07_psds.py`:

| File | Contents |
|---|---|
| `<device>_lifetime_<kind>_<name>.png` | One per photon and cosmic-ray result: pulse height (photons: the mean pulse's optimal-filter height) and decay time against resonant frequency, with histograms at the side; bad tones hollow |
| `<device>_lifetime_<kind>_<name>.csv` | One row per tone: res_idx, fres, height, tau, n_used, status, bad, bad_reason |
| `<device>_psd<p>.png` | One per PSD group, over the tones of all its timestreams: the Sxx mean with the 1st to 99th percentile shaded, before and after common-mode removal (off-resonance tones dashed, if any); Sxx against resonant frequency at 0.1, 1, 10 and 100 Hz (those the group has) with histograms at the side; the same for the loop components and the s-factor |
| `<device>_psd<p>.csv` | One row per tone: timestream, run, res_idx, fres, status, bad, and Sxx (raw and common modes removed) and s-factor at every reduced frequency |
| `<device>_psd<p>_spectra.csv` | The mean and range of every spectrum kind at every binned frequency |
| `<device>_timestreams.txt` | What each result holds: tones, bad tones with reasons, medians and ranges |

## Settings

{{settings:08_report.py}}

| Setting | Meaning |
|---|---|
| `OUT_DIR` | Folder the files go in |
| `PHOTONS`, `COSMIC`, `PSDS` | Results to report, by name (or PSD group index); `None`: all |

## Command line

`--out folder`, `--pick` (choose the store in a dialog).
