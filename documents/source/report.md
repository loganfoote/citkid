---
title: "Report"
subtitle: "7_report_site, set_notes"
---

# Overview

The pages of every device: what was taken, the figures and the
results of the analyses, one page per resonator. Notes written
with `set_notes.py` or the store browser appear on the pages.

| Script | What it does | Saves |
|---|---|---|
| `7_report_site.py` | Browsable pages of every device: what was taken, the figures, one page per resonator | Files in a folder |
| `set_notes.py` | Any time: add notes to the store or a group, shown on the pages | `notes` attributes |

# 7_report_site.py: browsable pages

Builds a static site (plain HTML, opens from the drive): one page per
device store, from the results in the store, and a top page of every
device built so far with the store's notes and its reduced numbers:
tuned tones of the resonances found (yield), their frequency range,
and the 1st to 99th percentile of the photon and HEE lifetimes and
of Sxx at 10 Hz. Each run builds one store (`STORE`, `--store`, or
a dialog). The store browser opens first on its Notes tab: edit the
notes of any item, uncheck its **Report** box to leave it out of the
report (the page then shows one line saying what it is, with its
notes; the choice is kept in the store), or **Attach file...** to
copy a file (a photo, a plot, a PDF) into the site next to that
item, linked from the page and never put in the store (**Remove
file...** takes one away again); close the browser to build. A report that exists is not replaced
without asking: a popup offers to keep the attached files and
`notes.md` (rebuilding the pages), to overwrite everything, or to stop;
without the window, `--update` or `--overwrite` decide. Only reduced
results are copied; timestreams and pulses stay in the store, which
every item names by path.

| Page | Contents |
|---|---|
| `index.html` | Every cooldown and device (wafer, die, store suffix, counts); type under a heading to filter, click one to sort. No file paths, so the site can move |
| `<cooldown>/<device>/index.html` | What was taken (sweeps, tunings, calibration sweeps, runs with their settings, calibrations, results), with notes and store paths; the VNA, tuning (chosen powers), stream calibration (the IQ fits, with `FIT_IQ`), photon, HEE and noise figures (every version in a dropdown, latest first, with its notes); a sortable table of the resonators; the tone marks |
| `<cooldown>/<device>/res/res<r>.html` | One per resonator: its IQ loop at the chosen power, calibration sweep, mean photon and HEE pulses, and spectra; its numbers from every step and result |
| `<cooldown>/<device>/figures/`, `data/` | The figures (PNG) and tables (CSV) |
| `<cooldown>/<device>/notes.md` | Written by hand; shown at the top of the device page and never overwritten |
| `<cooldown>/<device>/attachments/` | Files attached in the store browser, by item, linked from the page |

Each section of the device page folds (click its heading); the
resonator and tone-mark tables start folded, with a filter box. A
device folder is self-contained: copy it into another site's
`<cooldown>` folder and run `7_report_site.py --index-only` there to
list it.

Notes come from the store: the `notes` of the store itself, the VNA
sweep, tuning, runs and results, set when they are taken or later in
the store browser's Notes tab or with `set_notes.py`
(`python set_notes.py --store <store> --path photons/dark "text"`;
without text it lists the notes).

## Settings

{{settings:7_report_site.py}}

| Setting | Meaning |
|---|---|
| `REPORT_DIR` | The report site folder; the device's pages go in `<REPORT_DIR>/<cooldown>/<device>` (`--out` gives another folder for one run) |
| `STORE` | The store; `None`: a dialog |
| `RES_PAGES` | One page per resonator (about 100 kB each) |
| `INTERACTIVE` | Open the store browser on its Notes tab first |
| `INDEX` | Also rebuild the top page (`index.html`) after the device; `--index-only` rebuilds it alone |

## Command line

`--store path`, `--update`, `--overwrite`, `--no-res-pages`,
`--no-gui`, `--index-only`.
