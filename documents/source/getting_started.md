---
title: "Getting started with the procedures"
subtitle: "The environment, CRS basics, and crs_init"
---

# Overview

The scripts in `procedures/` take a device from a configured CRS board
to fitted resonances, tuned tones, calibrated timestreams, photons and
noise spectra. Run them in this order:

| Step | Script | Document |
|---|---|---|
| 1 | `crs_init.py`: configure the board (after a power cycle) | this one |
| 2 | `vna_quick_check.py` (optional): check the wiring | VNA screening |
| 3 | `1_vna_sweep.py`: full VNA sweep | VNA screening |
| 4 | `vna_view.py` (optional): look at the sweep | VNA screening |
| 5 | `2_find_resonances.py`: find and fit the resonances | VNA screening |
| 6 | `3_tune.py`: tone powers and frequencies | Tuning and timestreams |
| 7 | `4a_stream.py`: take timestreams in a window | Tuning and timestreams |
| 8 | `4b_stream_batch.py` (instead of 4a): take timestreams without a window | Tuning and timestreams |
| 9 | `4c_power_sweep.py` (for a power sweep): timestreams at each optical power | NEP |
| 10 | `store_browser.py` (optional, any time): look around the store | this one |
| 10b | `view_timestreams.py` (optional, any time): look at saved timestreams | Tuning and timestreams |
| 11 | `5_calibrate_streams.py`: calibrate the timestreams | Tuning and timestreams |
| 12 | `6a_find_photons.py`: photons and high-energy events (HEEs) | Timestream analysis |
| 13 | `6b_hee.py`: lifetimes from HEEs | Timestream analysis |
| 14 | `6c_psds.py`: noise spectra | Timestream analysis |
| 15 | `6d_nep.py` (for a power sweep, instead of 5 and 6c): NEP from a power sweep | NEP |
| 16 | `7_report_site.py`: browsable pages of every device | Report |

# The environment

Before running, activate the uv environment (see the citkid README for
installation and `uv sync`):

```text
source .venv/bin/activate      # macOS and Linux
.venv\Scripts\activate         # Windows
```

The CNN resonance finder needs `uv sync --extra cnn`.

# Running the scripts

Copy `settings.py` and the scripts into a folder for the run. Set
`settings.py` once: it holds what every script shares, and each script
imports it from its own folder. Set the rest of a script's settings
block at its top (a comment line names each group of settings), and
run it. Command-line options override a setting for one run; `--help`
lists them.

{{settings:settings.py}}

| Setting | Meaning |
|---|---|
| `SERIAL_NUMBER`, `INTERFACE` | The CRS serial number and the Ethernet interface to it |
| `MODULES` | Modules connected to the device |
| `DATA_DIR`, `COOLDOWN`, `DEVICE` | The device store: `<DATA_DIR>/<COOLDOWN>/<DEVICE>.zarr` (`--store` gives a path instead) |
| `SUFFIX` | Names the store `<DEVICE>_<SUFFIX>.zarr` (`''`: no suffix) |
| `MEMORY_GB` | Timestreams kept in memory by the analysis scripts, GB |
| `LIGHT_MODE` | `True`: white plots with black text in every window, for screens where the dark plots lack contrast |

Every script also has `NOTES`, text saved with the data (`--notes`).
Every window has the citkid icon; replace `citkid/icon.png` to change
it.

The store holds everything measured and analyzed. Raw data is never
overwritten: a repeated measurement gets the next index (`vna0`, `vna1`,
...). Calibrated timestreams are never saved. A different board or
firmware needs a new store.

- **Several devices**: `crs_init.py` configures the whole board; the
  other scripts attach to their own modules, so they can run
  simultaneously on different devices. `4a_stream.py` and
  `4b_stream_batch.py` are the exception: they need the whole board
  while they run, and other scripts stop with an error naming them.
- **Simulated devices**: `simulation/simulate_store.py` makes a store
  with a simulated device (its settings are the only simulation
  settings); `3_tune.py`, `4a_stream.py` and `4b_stream_batch.py`
  take `--simulate`.

# CRS basics

| Term | Meaning |
|---|---|
| Module | One readout chain (DAC out, ADC in). 8 modules in two analog banks (1-4, 5-8); one bank is active, chosen by `crs_init.py` |
| Tone | A frequency `fres` and power `ares` (dBm) written to a module; up to 1024 per module |
| DAC full scale | The DAC's largest output power, always 7 dBm |
| NCO | A module's center frequency; the module reads out a band around it |
| Module bandwidth | 500 MHz (600 MHz with `--extended-bw`, more loss at the edges); a band at one NCO frequency is an NCO window |
| Tone set | The tones written together when they don't all fit the modules' windows; sets are measured one after another |
| Nyquist zones | 100-2500 MHz (zone 1) and 2500-5000 MHz (zone 2); sweeps skip 10 MHz on each side of 2500 MHz |
| Global control | Needed for board-wide settings; `crs_init.py`, `4a_stream.py` and `4b_stream_batch.py` take it |

## Decimation stages

Streams are averaged down in decimation stages 0 (fastest) to 6
(slowest); sweeps use stage 6. Sample frequency: 625 MHz / (256 × 64 ×
2^stage).

| Stage | Sample frequency | Sample period | Tones that stream simultaneously |
|----|----------|---------|------------------------------|
| 0 | 38.15 kHz | 26.2 µs | 128, one module |
| 1 | 19.07 kHz | 52.4 µs | 128 per module, up to 2 modules |
| 2 | 9.54 kHz | 105 µs | 1024 on one module, or 128 per module on up to 4 |
| 3 | 4.77 kHz | 210 µs | 1024 per module on up to 2, or 128 per module on up to 4 |
| 4-6 | 2.38 kHz, 1.19 kHz, 596 Hz | 419 µs, 839 µs, 1.68 ms | 1024 per module on up to 4 modules |

Tones beyond the limit stream in sequential groups with the same NCO
frequencies.

# crs_init.py: configure the board

Sets the clock source, the analog bank, the DAC full scale (7 dBm), the
module bandwidth and the decimation stage (6), then exits; the board
keeps them. Run it before the other scripts and after a power cycle, not
while another script is measuring.

## Settings

{{settings:crs_init.py}}

| Setting | Meaning |
|---|---|
| `CLOCK_SOURCE` | `'VCXO'` (internal) or `'SMA'` (external 10 MHz) |
| `ANALOG_BANK_HIGH` | `False`: modules 1-4; `True`: modules 5-8 |
| `EXTENDED_BW` | `True`: 600 MHz module bandwidth |

## Command line

`--clock SMA`, `--bank-high`, `--extended-bw`.

# store_browser.py: look around a store

Opens a store read-only (another script can keep writing; **F5**
rereads). The left side lists what the procedures made, one tab per
kind, each a table filtered by typing; the last tab, **Store**, is the
store's own tree of groups and arrays for when the layout matters. The
right side is about the item selected in any of them:

| Right tab | Shows |
|---|---|
| Summary | What the item is and when it was made, its notes, what it was made from and what depends on it (links that select those items); Open in store tree, Reveal folder, Rename..., Delete... |
| Attributes | Its attributes as saved; for an array, its shape, dtype, chunks and first values |
| Tone marks | Every bad mark and note on a tone; Clear mark, Make global and Make local act on the selected bad mark |
| Notes | The notes of the store and of every item; select a row and edit below (saved when you move on); Rename..., Delete... |

| Left tab | Lists |
|---|---|
| VNA sweeps, Resonators, Tunings, Calibration sweeps | What the VNA and tuning scripts made |
| Runs | Every timestream run (name, notes, stage, length, tone set, calibration, tuning, chunks) |
| Stream calibrations, Photons, HEEs, Spectra, NEP | The analyses |
| Store | The tree; expand a group to read its contents |

**Rename...** changes the name of a tuning, run, PSD group, NEP sweep
or photon or HEE result (references follow). **Delete...** removes the
item from the store with everything that depends on it (a resonator
list with its tunings, a run with its results, ...), after asking, and
asking again when something depends on it; the Summary lists what would
go. Double-click an item, or **Open in store tree**, to see its group;
**Reveal folder** opens its directory in the file browser; a right-click
offers the same. **H** shows the help. **Open** (Ctrl+O) chooses another
store; `--pick` starts with the dialog; `--runs` prints the runs instead
of opening the window.

## Settings

{{settings:store_browser.py}}

| Setting | Meaning |
|---|---|
| `PICK` | Choose the store in a file dialog |
