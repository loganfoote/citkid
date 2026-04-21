# Kinetic Inductance Detector data acquisition and analysis
Collection of Kinetic Inductance Detector (KID) data acquisition and analysis code. Currently implemented analyses are:
<ul>
 <li> <b> citkid.res </b>: nonlinear resonance model fitting  </li>
 <li> <b> citkid.noise </b>: noise analysis </li>
 <li> <b> citkid.responsivity </b>: optical responsivity fitting  </li>
 <li> <b> citkid.res_vs_temp </b>: Mattis-Bardeen model fitting of resonance frequencies and quality factors versus temperature </li>
 <li> <b> citkid.multitone </b>: general multitone data acquisition and analysis procedures </li>
 <li> <b> citkid.primecam </b>: PrimeCam readout interface software and measurement procedures </li>
 <li> <b> citkid.crs </b>: t0.technology CRS readout interface software and measurement procedures </li>
</ul>

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

