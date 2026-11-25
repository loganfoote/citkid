---
title: "NEP from a blackbody series"
subtitle: "6d_nep"
---

# Overview

A blackbody series: timestreams at several optical powers, taken
with `4c_ts_series.py` ("Tuning and timestreams") with the points
named by blackbody temperature, then the NEP of every tone
(`6d_nep.py`, which calibrates the timestreams and computes their
spectra itself, in place of `5_calibrate_streams.py` and
`6c_psds.py`).

| Script | What it does | Saves |
|---|---|---|
| `6d_nep.py` | NEP from a blackbody series: calibration, IQ fits, spectra, responsivity and photon-noise fits, with a review window | `nep/<name>` |

# 6d_nep.py: NEP from a blackbody series

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
B marks the tone bad (local to this result), N adds a note, O opens the
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
| `TUNING_NAME` | Only this tuning's runs (`None`: every tuning) |
| `CALIBRATION_NAME` | Stream calibration to continue, by name or index (`None`: one named `NAME`) |
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
