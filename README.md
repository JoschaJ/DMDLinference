# DMDLinference

Codes accompanying the paper [Jahns-Schindler & Spitler (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv250814434J) used for inferring cosmological parameters using Fast Radio Burst (FRB) dispersion measures and gravitational wave luminosity distances.

## Overview

This package implements Bayesian inference to constrain the Hubble constant (H₀) and the baryon density in the IGM by combining:
- FRB dispersion measure (DM) observations
- Gravitational wave luminosity distance measurements
- Existing DM-z constraints

## Main Modules

- `mcmc.py` - Core MCMC inference with fixed host DM parameters
- `simulate_data.py` - Functions and runnable code to simulate an FRB population and use it for the inference (creates data for figures 2 and 3)
- `mcmc_free_host.py` - Extended inference allowing free host DM parameters (creates data for fig. 4)
- `corner_plots.py` - Generate corner plots from the mcmc chains. Needs to be used by outcommenting and uncommenting the correct lines.

## Installation via conda

```bash
conda create -c conda-forge -n gwmcmc python=3.11 numpy scipy matplotlib astropy emcee corner h5py tqdm seaborn
conda activate gwmcmc
```

## Dependencies

- numpy
- scipy
- emcee
- astropy
- h5py
- multiprocessing
- corner

## Usage

Key parameters include:
- H₀: Hubble constant
- Ωᵦh²f: Baryon density parameter
- μₕₒₛₜ, σₖₒₛₜ: Host DM distribution parameters

Run inference with:
```bash
python simulate_data.py  # Fixed host parameters
python mcmc_free_host.py  # Free host parameters
```

Results are saved as HDF5 files for further analysis and plotting.