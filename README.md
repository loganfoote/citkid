# Kinetic Inductance Detector data acquisition and analysis
Collection of Kinetic Inductance Detector (KID) data acquisition and analysis code. Packages:
<ul>
 <li> <b> citkid.res </b>: nonlinear resonance model fitting, S21 generation, and resonance frequency updates from target sweeps </li>
 <li> <b> citkid.xcal </b>: gain removal, circle fitting, and fractional frequency (x) calibration </li>
 <li> <b> citkid.signal </b>: PSDs, filtering, and optimal filtering of timestreams </li>
 <li> <b> citkid.nep </b>: photon-noise NEP fitting </li>
 <li> <b> citkid.responsivity </b>: Mattis-Bardeen responsivity model fitting </li>
 <li> <b> citkid.res_vs_temp </b>: Mattis-Bardeen model fitting of resonance frequencies and quality factors versus temperature </li>
 <li> <b> citkid.vna </b>: interactive resonance finding, matching, and target sweep frequency/span selection </li>
 <li> <b> citkid.crs </b>: t0.technology CRS readout interface software and measurement procedures </li>
 <li> <b> citkid.pipeline </b>: zarr-backed calibration and analysis pipeline with interactive review windows </li>
 <li> <b> citkid.primecam </b>: PrimeCam readout interface software and measurement procedures (legacy) </li>
</ul>

Example notebooks are in `notebooks/` (the newer ones run on synthetic
data). Measurement procedure scripts are in `procedures/`, with guides in
`documents/procedures/` (`vna_screening.pdf`: VNA sweeps and resonance
finding; `tuning_timestreams.pdf`: tuning and timestreams). Other documents in
`documents/`: `pipeline_guide.pdf`, `xcal_derivations.pdf`,
`crs_streaming_format.pdf`, and the derivations for resonator fitting and
optimal filtering, `res_fitting.pdf` and `optimal_filtering.pdf`. The PDFs
are built from the Markdown sources in `documents/source/` with
`uv run python documents/build_pdfs.py` (needs pandoc and xelatex).

## Installation with uv (recommended)
The development environment is managed with [uv](https://docs.astral.sh/uv/)
and pinned in `uv.lock`.
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository: `git clone https://github.com/loganfoote/citkid`
3. Navigate to the repository directory
4. Create the environment in `.venv/`, with citkid installed in editable
   mode plus the test tools and JupyterLab:
```bash
uv sync
```
5. Activate the environment so that `python`, `jupyter`, and the
   procedure scripts can import citkid:
```bash
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```
   Activation lasts for the current shell session; run the command again in
   each new terminal, or select `.venv/bin/python` as the interpreter or
   Jupyter kernel in your editor. Run `deactivate` to leave the environment.

Alternatively, run single commands inside the environment without activating
it, for example
```bash
uv run jupyter lab
```
`uv run` finds the environment by searching for `pyproject.toml` in the
current directory and its parents, so it only works from inside this
repository. To use it from another directory (for example a notebooks
repository that imports citkid), pass the path to this repository:
```bash
uv run --project /path/to/citkid script.py
```
or set `UV_PROJECT=/path/to/citkid` in your shell. Otherwise, activate the
environment as in step 5.

To also install PyTorch for the CNN resonance finder, run
`uv sync --extra cnn`.

## Installation with pip
1. Clone this repository: `git clone https://github.com/loganfoote/citkid`
2. Navigate to the repository directory
3. Install the package using pip:
```bash
python -m pip install .
```
Or, to install in editable mode with the test tools and JupyterLab, run
```bash
python -m pip install --editable ".[dev,notebooks]"
```

## Installation with conda (legacy)
`environment.yml` and `environment-cnn.yml` are kept for conda users:
```bash
conda env create -f environment.yml
```
They are no longer the reference environment. On Windows the conda
environment installs a second copy of Qt 6 (through conda-forge's
matplotlib), which prevents `rfmux`, and so `citkid.crs`, from importing.
Use the uv environment for CRS work.

## Citation

If you use this software, please cite it as below.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19683541.svg)](https://doi.org/10.5281/zenodo.19683541)

```bibtex
@software{citkid,
  author       = {Foote, Logan},
  title        = {citkid: Kinetic Inductance Detector data acquisition and analysis},
  version      = {1.0.0},
  date         = {2026-04-21},
  doi          = {10.5281/zenodo.19683541},
  url          = {https://github.com/loganfoote/citkid},
  license      = {Apache-2.0},
}
```

