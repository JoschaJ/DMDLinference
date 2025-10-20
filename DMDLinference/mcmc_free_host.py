#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the inference leaving mu_host and sigma_host free."""
import os
import numpy as np
import emcee

import mcmc
import config

from multiprocessing import Pool

from mcmc import p_DMcosmic, lum_dist, log_prior
from mcmc import IntegrationAssistant, z_of_DL, log10_normal
from simulate_data import simulate_FRBs, log_p_H0, p_DL

F = 0.32

def p_DM(DL, H0, ObfH0, mu_host, sigma_host, ia):
    """Probability measuring a DM given the distance and parameters."""
    z = z_of_DL(DL, H0)
    if z.shape:
        z = z[:, np.newaxis]

    p_DMhost = (1+z)*log10_normal(ia.DMcosmic[..., ::-1]*(1+z), mu=mu_host, sigma=sigma_host)
    integral = np.sum(p_DMhost*p_DMcosmic(ia.DMcosmic, z, F, ObfH0, Om=0.3), axis=-1)*ia.dDM

    return integral


def log_probability(theta, ia):
    """Sum up all log probabilities"""
    # Extract parameters
    H0, Obhsqf, mu_host, sigma_host = theta
    ObfH0 = Obhsqf * 10000 / H0

    lp = log_prior(H0, Obhsqf, mu_host, sigma_host) + ia.log_p_H0(H0, Obhsqf)

    # Evaluate the likelihood, only if the prior is non-infinite.
    if np.isfinite(lp):
        # Integrate over DL.
        log_prob = lp + np.sum(np.log(
            np.sum(ia.p_DL_sampled*ia.p_DM(ia.DL_samples, H0, ObfH0, mu_host, sigma_host, ia)*ia.dDL, axis=-1)))
    else:
        log_prob = -np.inf
    return log_prob


def log_probability_without_FRBs(theta, ia):
    """Sum up all log probabilities"""
    # Extract parameters
    H0, Obhsqf, mu_host, sigma_host = theta
    ObfH0 = Obhsqf * 10000 / H0

    lp = log_prior(H0, Obhsqf, mu_host, sigma_host)

    # Evaluate the likelihood, only if the prior is non-infinite.
    if np.isfinite(lp):
        log_prob = lp + np.sum(np.log(
                np.sum(ia.p_DL_sampled*p_DM(ia.DL_samples, H0, ObfH0, mu_host, sigma_host, ia)*ia.dDL, axis=-1)))
    else:
        log_prob = -np.inf
    return log_prob


def log_prior(H0, Obhsqf, mu_host, sigma_host):
    """Define the combination of all prior probabilities."""
    if 20 < H0 < 150. and 0.0 < Obhsqf < 0.1 and 1 < mu_host < 4 and 0.01 < sigma_host < 2 :
        return 0
    else:
        return -np.inf


if __name__ == '__main__':
    # Simulate FRBs.
    c = 299792.458
    Obhsqf = 0.02242*0.844
    H0 = 73
    Om = 0.3
    mu_host = 2
    DM0 = 10**mu_host
    sigma_host = .57  #0.57
    F = 0.32
    # mcmc.mu_host = mu_host
    # mcmc.sigma_host = sigma_host
    mcmc.F = F

    n_FRBs = 100
    z_mean = 0.2
    z_sigma = 0.2

    DL_mean = c/H0*lum_dist(z_mean, Om=0.3)
    eDL = 0.2
    sigma_DL = eDL*DL_mean
    DL_meas, DMexc, DM_host = simulate_FRBs(n_FRBs, z_mean=z_mean, z_sigma=z_sigma, sigma_DL=sigma_DL, Obhsqf=Obhsqf,
                                            H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host, seed=42)

    # Add 10 low z, low eD_L FRBs.
    n_FRBs2 = 20
    z_mean2 = 0.1
    z_sigma2 = 0.1
    DL_mean2 = c/H0*lum_dist(z_mean, Om=0.3)
    eDL2 = 0.1
    sigma_DL2 = eDL2*DL_mean2
    DL_meas2, DMexc2, DM_host2 = simulate_FRBs(n_FRBs2, z_mean=z_mean2, z_sigma=z_sigma2, sigma_DL=sigma_DL2, Obhsqf=Obhsqf,
                                            H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host, seed=121102)
    n_FRBs = n_FRBs + n_FRBs2

    # Concatenate the two simulated sets.
    DL_meas, DMexc, DM_host = np.concatenate((DL_meas, DL_meas2)), np.concatenate((DMexc, DMexc2)), np.concatenate((DM_host, DM_host2)),

    # Initialize the grid over which to integrate D_L and DM_cosmic
    n_rect_DL = 100
    n_rect_DM = 120
    DL_min = min(DL_mean-5*sigma_DL, DL_mean2-5*sigma_DL2)
    DL_max = max(DL_mean+5*sigma_DL, DL_mean2+5*sigma_DL2)

    ia = IntegrationAssistant(DMexc=DMexc, DL_min=DL_min, DL_max=DL_max,
                           n_rect_DM=n_rect_DM, n_rect_DL=n_rect_DL, p_DL=p_DL,
                           p_DL_kwargs={'DL_measured' : DL_meas, 'sigma_DL' : sigma_DL})

    # Redefine prior for our D_L distributions.
    ia.log_p_H0 = log_p_H0
    ia.p_DM = p_DM

    # Do inference for the simulated FRBs. Initialize the walkers.
    nwalkers = 8
    rng = np.random.default_rng()
    H0_init = rng.normal(70, 10, size=(nwalkers, 1))
    Obhsqf_init = rng.normal(Obhsqf, 0.0025, size=(nwalkers, 1))
    mu_host_init = rng.normal(mu_host, 0.1, size=(nwalkers, 1))
    sigma_host_init = rng.normal(sigma_host, 0.05, size=(nwalkers, 1))
    initial = np.concatenate((H0_init, Obhsqf_init, mu_host_init, sigma_host_init), axis=1)

    ndim = 4
    nsteps = 5000

    # # Set up a backend to save the chains to.
    # filename = os.path.join(config.DATA_DIR,
    #                         f"simulated_{n_FRBs}FRBs_free_host_vary_z{z_mean}_{z_mean2}_eDL{eDL}_{eDL2}_{nwalkers}x{nsteps}steps.h5")
    # if os.path.isfile(filename):
    #     print("Warning: File exists and will be appended to.")
    # backend = emcee.backends.HDFBackend(filename)
    # # backend.reset(nwalkers, ndim)

    # with Pool() as pool:
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[ia],
    #                                     backend=backend, pool=pool)
    #     sampler.run_mcmc(initial, nsteps, progress=True, progress_kwargs={'mininterval':5})

    # Sample the GW-FRB posterior without the FRB-z prior.
    filename = os.path.join(config.DATA_DIR,
                            f"simulated_noz_{n_FRBs}FRBs_free_host_vary_z{z_mean}_{z_mean2}_eDL{eDL}_{eDL2}_{nwalkers}x{nsteps}steps.h5")
    backend = emcee.backends.HDFBackend(filename)

    with Pool() as pool:
        sampler_noz = emcee.EnsembleSampler(nwalkers, ndim, log_probability_without_FRBs, args=[ia],
                                            backend=backend, pool=pool)
        sampler_noz.run_mcmc(initial, nsteps, progress=True, progress_kwargs={'mininterval':5})

