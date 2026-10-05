# Kinetic Inductance Detector data acquisition and analysis
Collection of Kinetic Inductance Detector (KID) data acquisition and analysis code. Packages:
<ul>
 <li> <b> citkid.res </b>: nonlinear resonance model fitting and S21 generation </li>
 <li> <b> citkid.xcal </b>: gain removal, circle fitting, and fractional frequency (x) calibration </li>
 <li> <b> citkid.signal </b>: PSDs, filtering, and optimal filtering of timestreams </li>
 <li> <b> citkid.nep </b>: photon-noise NEP fitting </li>
 <li> <b> citkid.responsivity </b>: Mattis-Bardeen responsivity model fitting </li>
 <li> <b> citkid.res_vs_temp </b>: Mattis-Bardeen model fitting of resonance frequencies and quality factors versus temperature </li>
 <li> <b> citkid.vna </b>: interactive resonance finding, matching, and target sweep frequency/span selection </li>
 <li> <b> citkid.crs </b>: t0.technology CRS readout interface software and measurement procedures </li>
 <li> <b> citkid.pipeline_v2 </b>: zarr-backed calibration and analysis pipeline with interactive review windows. To be renamed to "pipeline" </li>
 <li> <b> citkid.multitone </b>: resonance frequency updates for multitone readouts (other modules are legacy) </li>
 <li> <b> citkid.primecam </b>: PrimeCam readout interface software and measurement procedures (legacy) </li>
</ul>

Example notebooks are in `notebooks/`, and derivations are in `documents/`.

## Installation with conda environment setup
1. Clone this repository: `git clone https://github.com/loganfoote/citkid`
2. Navigate to the repository directory
3. Create the conda environment and install the repository in editable mode with
```bash
conda env create -f environment.yml
```
To activate the environment, run 
```bash
conda activate citkid
```
## Installation without conda environment setup
1. Clone this repository: `git clone https://github.com/loganfoote/citkid`
2. Navigate to the repository directory
3. Install the package using pip:
```bash
python -m pip install .
```
 Or, to install in editable mode run  
 ```bash
 python -m pip install --editable .
```

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

