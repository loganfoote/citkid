---
title: "VNA screening"
subtitle: "vna_quick_check, 01_vna_sweep, vna_view, 02_find_resonances"
---

# Overview

Full VNA sweeps of a device with the CRS, then the resonances found and
fitted. Configure the board with `crs_init.py` first (see "Getting
started").

| Script | When | What it does | Saves |
|---|---|---|---|
| `vna_quick_check.py` | Any time | Quick sweeps in a window | A CSV, if asked |
| `01_vna_sweep.py` | Once per device or setup | Full VNA sweep | `vna{k}` |
| `vna_view.py` | After a sweep | Shows a saved sweep | Nothing |
| `02_find_resonances.py` | After each sweep | Finds, checks and fits the resonances | `full_sweep_fit`, `resonators{k}` |
| `export_s21_h5.py` | After `02_find_resonances.py` | The sweep as an s21 h5 file (`f`, `i`, `q`, `fres`), for the resonance-finder tools | One `.h5` file |

```text
<device>.zarr/
  vna{k}/            full VNA sweep (raw); crs_config/, fres_auto,
                     fres_manual, res_finder_auto/, full_sweep_fit/
  resonators{k}/     resonator list from vna{k}, used for tuning
```

# vna_quick_check.py: quick look at a device

A window for checking the wiring, the power and the resonances before
the full sweep. Nothing is saved to the store; Save data writes the last
sweep as a CSV (`f`, `i`, `q`).

| Control | Effect |
|---|---|
| Modules | The device's modules (`1 2` for a splitter); Sweep connects to them |
| Nyquist zone | 1, 2 or both, or a range by hand (From, To) |
| Resolution | Points per linewidth at the expected median Qr: df/f = 1 / (Qr × points) |
| Plan | Points, bands, passes and sweep time |
| Sweep (Ctrl+R) | Takes the sweep and plots \|S21\| and phase |
| Cancel (Esc) | Stops the sweep and clears the tones |
| R | Rescales the plots (drag to pan, scroll to zoom) |

Don't check a module another script is sweeping or streaming. `--no-gui`
takes one sweep and saves it with `--csv`.

## Settings

{{settings:vna_quick_check.py}}

| Setting | Meaning |
|---|---|
| `NYQUIST` | `'1'` (100-2500 MHz), `'2'` (2500-5000 MHz), `'both'` |
| `AMPLITUDE` | Tone power, dBm |
| `QR_MEDIAN`, `PTS_PER_LW` | Expected median Qr and points per linewidth there |
| `NSAMPS` | Samples averaged per point |

## Command line

`--modules`, `--nyquist`, `--no-gui --csv file.csv`.

# 01_vna_sweep.py: full VNA sweep

Takes a full sweep of one device and saves it as the next `vna{k}`
(creating the store with `SETUP` if needed).

- Geometric spacing, df/f = 1 / (`QR_MEDIAN` × `PTS_PER_LW`); 25 points
  per linewidth at Qr = 2e4 suits the finders and the fitter.
- The range is divided into bands of at most the module bandwidth and
  1024 channels, measured in passes, in parallel over the modules.
- `VIEW` (`--view`) opens the sweep in the view window when done.

## Settings

{{settings:01_vna_sweep.py}}

| Setting | Meaning |
|---|---|
| `SETUP` | Description saved with a new store (fridge channel, attenuators, filters) |
| `NYQUIST` | `'1'` (100-2500 MHz), `'2'` (2500-5000 MHz), `'both'` |
| `AMPLITUDE` | Tone power, dBm |
| `QR_MEDIAN`, `PTS_PER_LW` | Expected median Qr and points per linewidth there |
| `NSAMPS` | Samples averaged per point |
| `VIEW` | Open the sweep in a window when done |

## Command line

`--modules`, `--nyquist`, `--view` / `--no-view`.

# vna_view.py: look at a saved sweep

Shows `vna{k}` (the latest, or `--vna k`): \|S21\| in dB and the phase
against frequency, with the band edges and the sweep's settings. Drag to
pan, scroll to zoom, **R** rescales, **Unwrap phase** toggles the phase
ramp. No CRS needed.

## Settings

{{settings:vna_view.py}}

| Setting | Meaning |
|---|---|
| `VNA` | Sweep index (`None`: the latest) |

## Command line

`--vna k`.

# 02_find_resonances.py: find and fit the resonances

Runs on the latest sweep (or `--sweep k`), without the CRS:

| Step | What it does | Saves |
|---|---|---|
| find | The CNN finder (default; needs PyTorch) or the interactive automatic finder (`--finder auto`) | `fres_auto`, `res_finder_auto/` |
| manual | Check and edit the list in the manual finder window (`--no-manual` skips it) | `fres_manual` |
| fit | Fits every resonance (double resonances together) and the gain, and prints a summary | `full_sweep_fit` |
| resonators | The list for tuning, in ascending frequency: fitted fr, Qr where reliable, span factors Qr / `SPAN_LW`, dip depths | `resonators{k}` |

Steps already done are skipped; `--redo <step>` repeats it and those
after it. A list a tuning uses is never replaced: take a new sweep.

## Which fitted parameters are reliable

| Fit result | fr | Qi | Qc, phi (and Qr, amp) |
|---|---|---|---|
| Good fit | yes | yes | yes |
| Overlap with a neighbor | yes | yes | no |
| Poor fit, distorted, or beyond bifurcation | yes | no | no |
| Failed fit | no | no | no |

Saved as `valid_fr`, `valid_Qi`, `valid_Qc`, `valid_phi`. Where the fit
failed, fr is the finder's frequency.

## Settings

{{settings:02_find_resonances.py}}

| Setting | Meaning |
|---|---|
| `FINDER` | `'cnn'` (needs PyTorch) or `'auto'` (interactive) |
| `SPAN_LW` | Target sweep span, linewidths (`qres = Qr / SPAN_LW`) |

## Command line

`--sweep k`, `--finder auto`, `--no-manual`, `--redo <step>`.

# Trying it on simulated data

`simulation/simulate_store.py` makes a store with a VNA sweep of a
simulated device and its true parameters. `02_find_resonances.py
--store ...` then runs on it as on real data, and `03_tune.py`,
`04a_stream.py` and `04b_stream_batch.py` take `--simulate`.
`citkid.vna.simulate.compare_to_truth` compares the fits with the
truth. The simulation settings live only here, not in the scripts.

## Settings

{{settings:simulation/simulate_store.py}}

| Setting | Meaning |
|---|---|
| `STORE` | Path of the new store |
| `SEED` | Device, gain and noise seed |
| `NYQUIST`, `AMPLITUDE`, `QR_MEDIAN`, `PTS_PER_LW`, `NSAMPS` | As in `01_vna_sweep.py` |
| `SNR_DB` | S/N per quadrature at -55 dBm with 100 samples |
| `PULSES` | `None`, or a dict of photon pulses and HEEs in the device's streams: `rate` (photons per tone, Hz), `x` (shift), `tau` (s), `hee_rate` (Hz), `hee_x`, `hee_spread`; keys left out take the defaults |
