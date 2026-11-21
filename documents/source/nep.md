---
title: "NEP from a power sweep"
subtitle: "4c_power_sweep, 6d_nep"
---

# Overview

A power sweep: timestreams at several optical powers
(`4c_power_sweep.py`, after a tuning from "Tuning and timestreams"),
then the NEP of every tone (`6d_nep.py`, which calibrates the
timestreams and computes their spectra itself, in place of
`5_calibrate_streams.py` and `6c_psds.py`).

| Script | What it does | Saves |
|---|---|---|
| `4c_power_sweep.py` | At each point of a power sweep: set the power, calibrate (or reuse), take the runs; the point's name is saved with them | `ts{n}` |
| `6d_nep.py` | NEP from a power sweep: calibration, IQ fits, spectra, responsivity and photon-noise fits, with a review window | `nep/<name>` |

# 4c_power_sweep.py: a sweep of optical power

At each point of `POINTS` (a name, e.g. the blackbody temperature), the
script calls `set_optical_power` (edit it for the setup; it also
waits for the power to settle), calibrates `TONE_SET` and takes
`RUNS` as `4b_stream_batch.py` does, naming each run
`<point>_dec<stage>` (`_2`, `_3`, ... when the name is taken). The
point's name is saved with its runs and its calibration sweep, and
`6d_nep.py` maps the names to powers. A point with `cal` reuses the
calibration of an earlier point of the sweep (its name) or a saved
sweep (its path) instead of taking one; this suits a single NCO
window only, since with several the sets are retuned at each point.
A sweep taken by hand works the same way: name the runs, and give
`POINTS` in 6d with the run names.

## Settings

{{settings:4c_power_sweep.py}}

| Setting | Meaning |
|---|---|
| `TONE_SET` | Tone set calibrated and streamed at every point |
| `POINTS` | The points in order: `dict(name = ...)`, with `cal = 'name'` (an earlier point) or `cal = 'tune0/cal2'` to reuse a calibration |
| `RUNS` | Timestreams at each point: `dec_stage`, `duration_s` (and `notes`); the names come from the point |
| `TONE_CHUNKS`, `FRES_UPDATE`, `NPOINTS_FINE`, `NPOINTS_GAIN`, `GAIN_SPAN_FACTOR`, `NSAMPS` | As in `4b_stream_batch.py` |

## Command line

`--from name` (start at that point, e.g. after a stop), `--tuning
name`, `--simulate`, `--notes`.

# 6d_nep.py: NEP from a power sweep

Takes the points of a sweep (`POINTS`: the point names of 4c, or run
names, with their incident powers), calibrates each point's timestream
and fits its resonances
(the steps of `5_calibrate_streams.py`, one DataSet per calibration sweep), computes the spectra
with peak removal and common-mode removal (as 6c, saved as a PSD set),
and per tone:

| Step | What it does |
|---|---|
| x(P) | The IQ fits' $f_r$ at each power relative to the lowest power; a point whose fit failed is NaN (never the calibration's frequency) |
| Responsivity | `citkid.responsivity` fit of x(P) for $R_0$, $P_0$, $x_0$ (at least 4 points); $R_0 \geq 0$ (x rising with power) is not a KID response and gives no NEP |
| NEP(P) | $\sqrt{S_{xx}(P)} / |R(P)|$ at `FREQ_HZ`, with $R(P) = R_0 (1 + P / P_0)^{-1/2}$ |
| Photon noise | `citkid.nep` fit of NEP(P) from `p_min` up for $\eta$; NEPinc is the NEP at the lowest power, NEPabs $= \eta \,$ NEPinc |

The result is `nep/<NAME>`; running again resumes its review
(`--overwrite` computes it again). The photon frequency `NU` is one
value, or a dict of resonator index to frequency with a `'default'`.

## The window

The tone list (`*` reviewed, `x` bad; boxes to show only the tones
outside the overview limits, with or without the bad ones), the point
to edit and **Calibration...** (C): the calibration and IQ fit panels of
this tone at that point. When they close, the tone's $f_r$ and
$S_{xx}$ there are recomputed from its new calibration without searching
again: the glitch positions found in the batch are cut again, and the
common modes found in the batch are taken out with weights fitted anew.
`p_min` (the box, in units of the highest power, or the dashed line)
refits the photon noise; **Apply p_min to all** gives it to every tone.
B marks the tone bad (local to this result), T adds a note, O opens the
overview ($R_0$, $P_0$, $x_0$, $\eta$, NEPinc, NEPabs with draggable
limits), R rescales.

**Done** runs the final pass: the points whose calibrations changed get
their spectra computed again in full (peak search and common modes),
every tone is fitted again, and the window opens again on the final
values with the overview. Done again, with nothing changed, finishes.

```text
nep/<name>/             attrs: points, calibration, psd, freq_hz, nu,
                        settings, limits, changed
                        res_idxs, nu, power_w, rows (tone, point)
                        per tone and point: fr, x, sxx, nep
                        per tone: R0, P0, x0 (and _err), f1, eta,
                        eta_err, n_fit, p_min, nep_inc, nep_abs,
                        status, bad, bad_reason
    review/point{j}/    glitch positions and decimated common modes
```

## Settings

{{settings:6d_nep.py}}

| Setting | Meaning |
|---|---|
| `NAME` | Result name (`nep/<NAME>`; the calibration and PSD set are named so too) |
| `POINTS` | `{'BBT40K': 1e-15, ...}`: the point names (4c) or run names and their incident powers in W; a point taken twice is given by its run names |
| `TUNING` | Only this tuning's runs (`None`: every tuning) |
| `CALIBRATION` | Stream calibration to continue, by name or index (`None`: one named `NAME`) |
| `NU` | Photon frequency, Hz: one value, or `dict(res_idx = Hz, ..., default = Hz)` |
| `FREQ_HZ` | Frequency of $S_{xx}$ for the NEP (on the reduced grid 0.1, 0.3, 1, 3, 10, ... Hz) |
| `F_AVG` | Fine sweep points averaged in the fits |
| `INTERACTIVE` | Open the review window |
| `BAD_TONES` | Resonators left out, besides the tone marks |
| `NBINS`, `FMIN_MULT`, `FMAX_DIV`, `CN_*` | Spectra and common modes, as in `6c_psds.py` |
| `MODES_RATE_HZ` | Sample rate the common modes are kept at for the review, Hz (well above `FREQ_HZ`) |
| `DEGLITCH`, `GLITCH_*` | Peak removal, as in `6c_psds.py` (`GLITCH_NSTD` 12: only the largest spikes) |

## Command line

`--name`, `--tuning name`, `--cal name`, `--overwrite`, `--no-gui`,
`--notes`.
