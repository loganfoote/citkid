---
title: "VNA screening"
subtitle: "vna_quick_check, 01_vna_sweep, 02_find_resonances, vna_report"
---

# Overview

The VNA screening scripts in the procedures folder take full VNA sweeps of
a device with the CRS, then find and fit its resonances.

Before running them, make sure the board is configured with
`crs_init.py`. The "Getting started" document describes it, along with
the environment, how the scripts are run, the device store, and the CRS
terms used here.

| Script | When to run it | What it does | Saves |
|---|---|---|---|
| `vna_quick_check.py` | Any time | Takes quick VNA sweeps of one device in an interactive window | Optionally, a CSV of f, I, Q |
| `01_vna_sweep.py` | Once per device, or after a change to the setup | Takes a full VNA sweep | `vna{k}` |
| `02_find_resonances.py` | After each VNA sweep | Finds, checks, and fits the resonances | `full_sweep_fit`, `resonators{k}` |
| `vna_report.py` | After `02_find_resonances.py` | Writes a plot, a text summary, and a table of the results | Files in a folder |

These scripts save these groups in the device store:

```text
<device>.zarr/
  vna{k}/            full VNA sweep (raw), with the resonance finding and
                     fitting results
    crs_config/        CRS settings used for the sweep
    fres_auto, res_finder_auto/, fres_manual   resonance frequencies found
    full_sweep_fit/    fits of every resonance (analysis)
  resonators{k}/     resonator list from vna{k}, used for tuning
```

# vna_quick_check.py: quick look at a device

This script opens a window for checking the wiring and the tone power,
and for checking that the resonances are there before the full sweep.
Nothing is saved to the store. The last sweep can be saved as a CSV file
with columns `f` (Hz), `i`, and `q` (Save data).

The script does not configure the board (see `crs_init.py` in the
"Getting started" document). It connects to the board given by the
serial number and network interface settings, and attaches to its modules without changing the board-wide
settings, so it can run alongside the other scripts. The modules must be
in the analog bank chosen by `crs_init.py` (1-4, or 5-8 with
`--bank-high`). Modules in the other bank are listed in the status line
and not used.

The window has these controls:

- **Modules**: the modules connected to the device. Use several if a
  splitter feeds the device to more than one module. Type them (for
  example `1 2` or `5 6`) and press Connect to change them.
- **Nyquist zone**: zone 1 (100-2500 MHz), zone 2 (2500-5000 MHz), or
  both. Sweeps skip 10 MHz on each side of 2500 MHz, because the point
  nearest the Nyquist frequency is unreliable. The range can also be set
  by hand (From, To).
- **Resolution**: the frequency step is set by the number of points per
  linewidth at the expected median Qr, so
  df/f = 1 / (median Qr x points per linewidth). The points are spaced
  geometrically, so every resonance with the same Qr gets the same number
  of points per linewidth.
- **Plan**: shows the number of points, bands, and passes, the points per
  channel, and the time the sweep will take.
- **Sweep** (Ctrl+R): takes the sweep with a progress bar, and plots
  |S21| and phase with the band edges marked. If "Count resonances after
  sweeping" is checked, the resonances are counted with a matched filter
  (a rough count) and marked on the plot. **Count now** counts the
  resonances of the last sweep without taking new data.

With `--no-gui` (for example, over ssh without a display), the script
takes one sweep with the settings, prints the resonance count, and saves
the sweep as a CSV if `--csv` is given.

## Settings

{{settings:vna_quick_check.py}}

## Command line

{{help:vna_quick_check.py}}

# 01_vna_sweep.py: full VNA sweep

This script takes a full VNA sweep of one device and saves it as the next
`vna{k}` in the device store. If the store doesn't exist yet, it is
created, with the `SETUP` setting saved as a description of the setup
(fridge channel, attenuators, filters, ...).

- The sweep covers the chosen Nyquist zone or zones (`NYQUIST`). The
  points are spaced geometrically, with df/f = 1 / (`QR_MEDIAN` x
  `PTS_PER_LW`). The default is 25 points per linewidth at Qr = 2e4
  (df/f = 2e-6), which works best for the resonance finders and the
  fitter: 10-80 points per linewidth for a typical resonance, and at
  least 5 for the narrowest.
- Every channel sweeps downward in frequency. The range is divided into
  bands, each no wider than the module bandwidth and with at most 1024
  channels. If there are more bands than modules, the bands are measured
  in several passes. The number of points per channel is chosen to make
  the sweep as fast as possible (`--npoints` overrides it).
- With several modules (through a splitter), the bands are divided among
  the modules and measured in parallel.
- The script saves `f` and `z` (combined into one sweep in ascending
  frequency), the band arrays (edges, NCO frequencies, modules, passes,
  Nyquist zones, channels, and start times), the sweep settings, and the
  CRS settings (`crs_config`).

## Settings

{{settings:01_vna_sweep.py}}

## Command line

{{help:01_vna_sweep.py}}

# 02_find_resonances.py: find and fit the resonances

This script analyzes a saved VNA sweep (by default the latest; choose
another with `--sweep k`). It does not need the CRS. It runs these steps:

1. **find**: The CNN finder (default) or the interactive automatic finder
   (`--finder auto`) finds the resonances and saves their frequencies in
   `fres_auto`, with the finder's settings and results in
   `res_finder_auto/`.
2. **manual**: Check and edit the list of resonances in the manual finder
   window, which saves `fres_manual`. With `--no-manual`, the finder's
   list is used as it is (useful for simulated data).
3. **fit**: Every resonance is fit (overlapping resonances are fit
   together), the gain of the sweep is fit, and a summary of the sweep is
   printed. The fits are saved in `full_sweep_fit`, along with each
   resonance's flags and which of its parameters are reliable.
4. **resonators**: Save the resonator list used for tuning,
   `resonators{k}`, in ascending frequency. It holds the fitted resonance
   frequencies (or the finder's frequencies where the fit failed), Qr
   where it is reliable (used for linewidths), the target sweep span
   factors Qr / `SPAN_LW` (so that target sweeps span `SPAN_LW`
   linewidths), and the dip depths, which `03_tune.py` uses to choose
   between close resonators. The depths are measured directly on the
   gain-removed sweep, as 1 - min |S21| within half a linewidth of each
   resonance (and never past halfway to its nearest neighbor), so they
   don't depend on the fit: the fitted Qr/Qc is unreliable for exactly the
   overlapping resonances being compared.

Steps that are already done are skipped, so the script picks up where it
stopped. `--redo <step>` repeats that step and the steps after it. A
resonator list that is used by a tuning is never replaced; take a new VNA
sweep to make a new list.

## Which fitted parameters are reliable

The independent fit parameters are fr, Qi, Qc, and phi. Qr and amp are
computed from Qi and Qc and are as reliable as Qc.

| Fit result | fr | Qi | Qc, phi (and Qr, amp) |
|---|---|---|---|
| Good fit | yes | yes | yes |
| Overlap: a neighboring resonance affects the fitted pole but not the zero | yes | yes | no |
| Poor fit (large residuals), distorted, or beyond bifurcation | yes | no | no |
| Failed fit | no | no | no |

These results are saved as `valid_fr`, `valid_Qi`, `valid_Qc`, and
`valid_phi` in `full_sweep_fit`. Where the fit failed, the resonance
frequency in `resonators{k}` and in the report is the frequency that the
resonance finder passed to the fitter (`fres_input`), not a fitted value,
and `valid_fr` is False.

## Settings

{{settings:02_find_resonances.py}}

## Command line

{{help:02_find_resonances.py}}

# vna_report.py: results in plain files

This script writes the reduced results of a VNA sweep in human-readable
form: a plot, a text summary, and a table. Run it after
`02_find_resonances.py`. It writes these files to a folder:

- `<device>_vna<k>.png`: the sweep with the gain removed and the
  resonances marked (open circles where the fit is unreliable), and plots
  of Qi, Qc, and phi vs frequency for the reliable fits.
- `<device>_vna<k>_info.txt`: the sweep settings (power, averaging,
  resolution, frequency range, modules, Nyquist zones, and bands), how the
  resonances were found and fit, the number of resonances, how they are
  distributed in frequency (range, gaps, and banks), typical quality
  factors, data-quality notes, and the user's notes.
- `<device>_vna<k>_resonators.csv`: one row per resonator, with `res_idx`
  matching `resonators{k}`. The columns are fr, Qi, Qc, Qr, amp, and phi
  with their errors (NaN where unreliable), the flags, the fit status,
  and the `valid_*` columns. `fr` is never NaN: where the fit failed, it
  is the finder's frequency, with `valid_fr` False and `fr_err` NaN.

## Settings

{{settings:vna_report.py}}

## Command line

{{help:vna_report.py}}

# Trying it on simulated data

`simulate_store.py` makes a store with a VNA sweep of a simulated device
(banks of resonators, close pairs, nonlinearity, a realistic gain, and
noise). The sweep is taken with the same procedure as a real sweep, and
the device's true parameters are saved with it. `02_find_resonances.py`
and `vna_report.py` then run on it the same way as on real data, and
`citkid.vna.simulate.compare_to_truth` compares the fits with the true
parameters.

## Settings

{{settings:simulate_store.py}}

## Command line

{{help:simulate_store.py}}
