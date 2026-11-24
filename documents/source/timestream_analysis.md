---
title: "Timestream analysis"
subtitle: "6a_find_photons, 6b_hee, 6c_psds, analysis_template"
---

# Overview

The analyses of calibrated timestreams (`5_calibrate_streams.py`,
in "Tuning and timestreams"). They are independent: run any of
them, in any order, on the same calibration. Tones marked bad in one
window stay out of the others (the tone marks, in "Tuning and
timestreams"). For a power sweep, `6d_nep.py` (in "NEP") does the
calibration and the spectra itself.

| Script | What it does | Saves |
|---|---|---|
| `6a_find_photons.py` | Finds photons and HEEs, with a threshold window | `photons/<name>` |
| `6b_hee.py` | Fits HEE tails pulse by pulse, with a threshold window | `hee/<name>` |
| `6c_psds.py` | Noise spectra before and after common-mode removal | `ts_psd{p}` |
| `analysis_template.py` | Lines to copy into your own analysis | your group |

# analysis_template.py: your own analysis

`analysis_template.py` has no main script: copy its lines into your own
code. It opens the store, applies a stream calibration to one
timestream in memory (`calibrated_timestream`: the calibration and the
run by name or index; `xt`, `thetat`, `At` per tone and sample), reads
the tone parameters from the DataSet (`fres`, `qres`, `ares`,
`res_idxs`; `ff`, `zf`, `xf`, `thetaf` per row), reaches the raw stream,
and saves reduced results in a group of the store.


# 6a_find_photons.py: find photons

Each timestream is calibrated with the stream calibration of its own
sweep (there is normally one; `CALIBRATION_NAME` chooses when a sweep has
several, and a calibration of another sweep is refused). Choose
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
| Photons | The mean photon pulse averages the uncoincident pulses in the height window; its lifetime is fit from `TAU_SKIP` samples after the maximum over `TAU_FIT_S` |
| High-energy events | Events seen by at least `HEE_MIN_MULT` tones, with each tone's pulse height |

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
| Overview | O: histograms over all tones of photon height, lifetime and rate, with draggable review limits; click a bar or a listed tone to open it |
| Tone list | Two boxes above it: show only the flagged tones (outside the overview limits), and include the tones marked bad. A tone "not run" (`TONES_BEFORE_WINDOW`) runs when selected |
| Rescale | R |

## Saved results, continuing and reusing

- Continue (the popup, or `--continue`) opens the saved result in the
  window with no detection; only changed tones rerun.
- `--refit-tau` refits every saved lifetime with `TAU_SKIP` and
  `TAU_FIT_S`, no detection.
- `REFERENCE_NAME = '<name>'` searches other timestreams with a saved result,
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
    hee/events/      t0, time_s, mult, write_idx
    hee/hits/        per tone and event: sample, height, raw peak
    detail/             the detection detail, for continuing
```

Results of earlier versions (`ts_photons{p}`) are moved into this layout
by `citkid.pipeline.photon_finder.migrate_results(root)`.

## Settings

{{settings:6a_find_photons.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATION_NAME` | Calibration to use when a sweep has several, by name or `ts_cal` index (`None`: each timestream's own) |
| `TIMESTREAMS` | Timestreams, by index as listed or run name; empty: list them |
| `NAME` | Result name for a single timestream (`''`: the run's name) |
| `REFERENCE_NAME`, `REUSE` | A saved result to reuse (its name), and `'settings'` or `'filter'` |
| `INTERACTIVE` | Check the thresholds in the window |
| `TONES_BEFORE_WINDOW` | Tones detected before the window opens (`None`: all). The others are listed "not run" and run when selected, when thresholds are applied to all tones, or with the window's thresholds when it closes |
| `WINDOW_FIRST_ONLY` | Open the window on the first chosen timestream only; the others take its result as the reference (`REUSE`) |
| `INIT_NSIGMA`, `INIT_HMAX` | First pass: peak threshold (noise sigma) and upper cut (times the median peak) |
| `OF_ITERATIONS` | Rounds of optimal filtering: each round builds the template and filter from the pulses the last one found in the height window |
| `HMIN`, `HMAX` | Height window of single photons, kHz/GHz |
| `HMIN_FALSE_FRAC`, `HMIN_AUTO_MAX` | Choose `HMIN` automatically (`None`: fixed), up to this height |
| `TEMPLATE_PRE_S`, `TEMPLATE_POST_S` | Template before and after the maximum, s |
| `FILTER_POST_S` | The filter uses this much of the template after the maximum, s |
| `TAU_SKIP`, `TAU_FIT_S` | Decay fit: samples skipped after the maximum, fit length, s |
| `HIGHPASS_HZ` | Cutoff of the highpass (a running median) applied to x before the detection, to remove drifts, Hz |
| `COINC_NSIGMA`, `COINC_WIN` | Coincidences: pulses counted (filter sigma), window (± samples) |
| `HEE_MIN_MULT` | Fewest tones that must see an event for it to count as high-energy |

## Command line

`--cal name`, `--name`, `--continue`, `--overwrite`, `--reference name
--reuse filter`, `--refit-tau`, `--no-gui`.

# 6b_hee.py: HEE lifetimes

High-energy events hit many tones simultaneously, so a device without photons
still gives each tone's quasiparticle lifetime. Same timestream choice,
naming and existing-result popup as `6a_find_photons.py`; results are
`hee/<name>`.

| Step | What it does |
|---|---|
| Events | `EVENTS = 'photons'`: the HEEs `6a` saved (or those of `PHOTONS_NAME`); `'triggers'`: peaks above `TRIG_NSIGMA` on each tone, clustered within `COINC_WIN` samples, kept when `HEE_MIN_MULT` tones see them; `'auto'`: photons if saved, else triggers |
| Pulses | A snippet `PRE_S` before to `POST_S` after each event, baseline subtracted; the peak within `SEARCH_S` |
| Threshold | Only pulses with peaks in [`HMIN`, `HMAX`] are averaged |
| Alignment | The pulses differ in height and near the peak but decay alike: each is first placed where it falls through `LEVEL` (10 kHz/GHz), then moved by the few samples that best overlap the mean tail over the fit range, and they are averaged over that range only |
| Fit start | The fit begins after the latest peak of the mean within the peak search window (`SEARCH_S`), where averaging peaks and dips leaves bumps; `FIT_START_S`, or the yellow line in the window, sets it by hand |
| Tail fit | An exponential over `FIT_S` from the alignment point of the mean |
| Result | Per tone, the mean's lifetime and the pulses averaged |

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
(median height, lifetime, pulse rate). Two boxes above the tone list
show only the flagged tones (outside the overview limits) and
include the tones marked bad. A tone "not run"
(`TONES_BEFORE_WINDOW`) runs when selected.

```text
hee/<name>/          attrs: run, tone set, calibration, events,
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

{{settings:6b_hee.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATION_NAME`, `TIMESTREAMS`, `NAME`, `INTERACTIVE`, `TONES_BEFORE_WINDOW` | As in `6a_find_photons.py` |
| `WINDOW_FIRST_ONLY` | Open the window on the first chosen timestream only; the others take its values (global and per tone) |
| `EVENTS` | `'auto'`, `'photons'`, or `'triggers'` |
| `PHOTONS_NAME` | Photon result to take the events from (`None`: this timestream's) |
| `TRIG_NSIGMA`, `COINC_WIN`, `HEE_MIN_MULT` | Triggers: threshold (noise sigma), largest gap within an event (samples), fewest tones |
| `HMIN`, `HMAX` | Peak height window of the pulses fitted, kHz/GHz |
| `LEVEL` | A tail begins where the pulse falls below this, kHz/GHz |
| `FIT_S` | Length of the exponential fit of the mean tail, from its alignment point, s |
| `FIT_START_S` | Where the fit of the mean tail begins, s after the alignment point (`None`: after the latest peak within `SEARCH_S`) |
| `PRE_S`, `POST_S` | Snippet before and after an event, s |
| `SEARCH_S` | Each tone's pulse peak is looked for within this time after the event, s |
| `HIGHPASS_HZ` | Cutoff of the highpass applied to x before the fits, to remove drifts, Hz |

## Command line

`--cal name`, `--name`, `--events triggers`, `--photons name`,
`--continue`, `--overwrite`, `--no-gui`.

# 6c_psds.py: noise spectra

`TIMESTREAMS` chooses timestreams (index, run name, or a dict with the
tone sets and calibration; empty lists them). The timestreams of one run
of the script are saved together in `ts_psd{p}` with one grid of reduced
frequencies, named by `NAME`. Tones marked bad, and `BAD_TONES`, are
left out. When a set with that name exists (without a name: one with the
same `TIMESTREAMS`), a popup offers a new set (default; the window opens
with the saved limits), computing it again with its saved settings, bad
marks and limits without the window, or overwriting it (asks again);
with `--no-gui`, `--new`, `--load` or `--overwrite` decide.
A timestream whose spectra were saved with the same settings, on the
same calibration and with the same tones marked bad is not computed
again: the saved spectra are kept (or copied into the new set), and
the window opens on them. `--overwrite` always computes, e.g. after
calibrating again.

| Step | What it does |
|---|---|
| Deglitch (optional) | Pulses are cut out of each timestream before the spectra, by peak finding or with a photon result's filters (see Deglitching below) |
| Spectra | $S_{xx}$ per tone: reduced values (the mean within 20% of 0.1, 0.3, 1, 3, 10, ... Hz, from `FMIN_MULT` / duration to Nyquist / 2 / `FMAX_DIV`) and `NBINS` logarithmic bins; also the spectra of the loop components, parallel to the loop ($r\theta$, dissipation) and perpendicular to it ($A$, amplitude), and their ratio in dB at the reduced frequencies, the s-factor. The droop of the CRS decimation filters is undone (the rfmux transfer function), and the spectra end at 0.8 of the Nyquist frequency |
| Common modes | Per tone write, `CN_N_COMP` modes by `citkid.xcal.corr.calc_cm` (`CN_N_ITER` iterations; highpass `CN_HIGHPASS_HZ` while finding the weights, lowpass `CN_LOWPASS_HZ` on the modes while iterating; the removed modes are unfiltered), then the spectra again on the cleaned tones, and the modes' spectra |
| Review | The overview window on the raw spectra, before the common modes (skip it with `INTERACTIVE = False` or `--no-gui`); tones marked bad there stay out of the common-mode fit. The timestreams stay in memory, so after the window only the common modes are computed |

## Deglitching

A pulse (a photon or a high-energy event) adds power to the spectrum
at every frequency, so for the noise alone the pulses are found
first and the samples around each one are replaced with Gaussian
noise of the tone's own level. `DEGLITCH` picks how they are found:

| `DEGLITCH` | How the pulses are found |
|---|---|
| `None` | Not at all: the spectra include the pulses |
| `'peaks'` | Peak finding on each tone alone: no photon result needed. Slow drifts are removed first by binning the timestream to `GLITCH_BASELINE_S` bins and subtracting that baseline, so the threshold applies to the fast noise. `GLITCH_SIGN` says which way a pulse goes in x (`-1`: a pulse lowers the frequency, the usual case). A peak counts when it exceeds `GLITCH_NSTD` standard deviations of the baseline-subtracted timestream; peaks closer than `GLITCH_DISTANCE_S` are one pulse. `GLITCH_ROUNDS` repeats the search: the pulses found in a round are removed and the standard deviation is taken again without them, so the next round catches the smaller pulses that the large ones hid by inflating the threshold. Two rounds is usually enough; one is fine for a quiet timestream |
| `'photons'` | The optimal filters of a photon result (`PHOTONS_NAME`, from `6a_find_photons.py`) find the pulses, as the photon search did. `GLITCH_PULSES = 'hee'` cuts only the high-energy events (`COINC_WIN`, `HEE_MIN_MULT`, as in 6a) and leaves the photons in; `'all'` cuts every pulse |

Either way, `GLITCH_PRE_S` before and `GLITCH_POST_S` after each
pulse are replaced; `GLITCH_POST_S` should cover the pulse tail, a few
lifetimes. Each tone's result records the pulses cut and the fraction
of samples replaced (`n_glitches`, `glitch_frac`).

## The window

Three histograms over all tones of the raw $S_{xx}$ (1/Hz, log axis)
at frequencies picked from the reduced grid, each with
draggable review limits; the histograms do not zoom. The list shows the
flagged tones (outside the limits) or all, with or without the tones
marked bad (the boxes above it), with their values; click a tone, or
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

{{settings:6c_psds.py}}

| Setting | Meaning |
|---|---|
| `CALIBRATION_NAME` | Calibration to use when a sweep has several (`None`: each timestream's own) |
| `TIMESTREAMS` | Index as listed, run name, or `dict(name = ..., or run = n, tone_sets = [...], cal = c)`; empty: list them |
| `NAME` | Name of this set of spectra |
| `INTERACTIVE` | Review each timestream in the overview window |
| `NBINS` | Number of logarithmic frequency bins of the binned spectra |
| `FMIN_MULT` | Lowest frequency, times 1 / duration |
| `FMAX_DIV` | Highest reduced frequency: Nyquist / 2 / this |
| `CN_REMOVE`, `CN_N_COMP`, `CN_N_ITER` | Remove the common modes (`True`); how many modes to remove; iterations of the fit of the weights |
| `CN_LOWPASS_HZ` | On the modes while iterating (`None`: Nyquist / 10) |
| `CN_HIGHPASS_HZ` | While finding the weights (`None`: `FMIN_MULT` / duration; `0`: off) |
| `CN_FILTER_ORDER` | Order of the highpass and lowpass filters above |
| `BAD_TONES` | Resonators left out, besides the tone marks |
| `DEGLITCH` | `None` (keep the pulses), `'peaks'`, or `'photons'`: see Deglitching |
| `GLITCH_PRE_S`, `GLITCH_POST_S` | Time replaced before and after each pulse, s |
| `GLITCH_SIGN` | `'peaks'`: direction of a pulse in x (`-1`: downward) |
| `GLITCH_NSTD` | `'peaks'`: peak threshold, in standard deviations of the baseline-subtracted timestream |
| `GLITCH_BASELINE_S` | `'peaks'`: bin length of the slow baseline subtracted first, s |
| `GLITCH_DISTANCE_S` | `'peaks'`: peaks closer than this are one pulse, s |
| `GLITCH_ROUNDS` | `'peaks'`: searches, each with the threshold taken without the pulses found so far |
| `PHOTONS_NAME` | `'photons'`: name of the photon result whose filters find the pulses |
| `GLITCH_PULSES` | `'photons'`: `'hee'` cuts the high-energy events only, `'all'` every pulse |
| `COINC_WIN`, `HEE_MIN_MULT` | `'photons'`: an event spans pulses within ± this many samples, and needs this many tones to count as high-energy |

## Command line

`--name`, `--new`, `--load`, `--overwrite`, `--no-gui`, `--no-cn`,
`--deglitch photons --photons name`.
