#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:08:54 2023

@author: jjahns
"""
import os
import numpy as np
import emcee
import corner

import mcmc
import config

from multiprocessing import Pool
from scipy.stats import norm

from mcmc import p_DMcosmic, lum_dist, log_probability, log_p_H0_with_prior, log_prior
from mcmc import log_probability_without_FRBs, initialize_integration

def draw_DM(frb_zs, Obhsqf=0.017, H0=70, F=0.32, Om=0.3, DM0=100, sigma_host=1, rng=None):
    """Draw a DM for each given reshift.

    Given the parameter values from this function simulates the DM.
    Obh70 can not be given at the moment, would have to give it to
    averag_DM.

    Args:
        frb_zs (array-like): Redshifts of FRBs.
        F (float): Fraction of baryonic DM in the intergalactic medium.
        mu (float): Mean of the lognormal distribution for DM_host.
        lognorm_s (float): Standard deviation of the lognormal
            distribution for DM_host.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        numpy.ndarray: Total dispersion measures (DM) for the FRBs.

    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw a Delta from it's PDF. Multiply by <DM_cosmic> to get a DM.
    dm_cosmic = [float(draw_DM_cosmic(z, Obhsqf=Obhsqf, H0=H0, F=F, Om=Om, n_samples=1, rng=rng))
                 for z in frb_zs]

    # Draw a DM_host.
    dm_host = rng.lognormal(mean=np.log(DM0), sigma=np.log(10)*sigma_host, size=len(frb_zs))

    return dm_host/(1+frb_zs) + dm_cosmic, dm_host


def draw_DM_cosmic(z, Obhsqf=0.017, H0=70, F=0.32, Om=0.3, n_samples=1, rng=None):
    """Draw DM from p_cosmic.

    Following Macquart et al. 2020 the PDF can be described by their
    equation (4). Here Delta = DM_cosmic / <DM_cosmic>, that means to
    get DM_cosmic the returned number has to be multiplied by the
    average of DM_cosmic. This is because frb.dm.igm.average_DM() is
    very slow and should only be used once (with cumul=True) and then be
    interpolated.

    Args:
        z (float): Redshift.
        f (float): Strength of baryon feedback F.
        n_samples (int): Number to draw.
        rng (numpy.random.Generator, optional): Random number generator.

    Returns:
        numpy.ndarray: Delta for given z, defined as
            DM_cosmic / <DM_cosmic>.
    """
    # Create 20000 values of the PDF to create the inverse from.
    DM_values = np.linspace(1/1000., 10000., 20000)

    ObfH0 = Obhsqf * 10000 / H0
    pdf = p_DMcosmic(DM_values, z, F, ObfH0, Om=Om)

    # Invert the CDF.
    cum_values = pdf.cumsum()/pdf.sum()

    if rng is None:
        rng = np.random.default_rng()
    r = rng.random(n_samples)

    return np.interp(r, cum_values, DM_values)


def positive_normal(loc, scale, size=None, rng=None):
    """Draw only positive values from a normal distribution.

    Only takes arrays. Is a bit inefficient because I feel lazy and don't want
    to handle cases efficiently, where e.g. loc is an array.
    """
    if rng is None:
        rng = np.random.default_rng()
    var = rng.normal(loc=loc, scale=scale, size=size)
    neg = var < 0
    while np.any(neg):
        var[neg] = rng.normal(loc=loc, scale=scale, size=size)[neg]
        neg = var < 0

    return var


def simulate_FRBs(n_draw, z_mean, z_sigma, sigma_DL, Obhsqf=0.017, H0=70, F=0.32, Om=0.3, DM0=100,
                  sigma_host=1, seed=None):
    c = 299792.458

    rng = np.random.default_rng(seed=seed)
    zs = positive_normal(loc=z_mean, scale=z_sigma, size=n_draw, rng=rng)

    DMs, DM_host = draw_DM(zs, Obhsqf=Obhsqf, H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host, rng=rng)

    DL_mean = c/H0*lum_dist(zs, Om)  # in Mpc, 20% uncertainty
    DL = positive_normal(loc=DL_mean, scale=sigma_DL, rng=rng)

    return DL, DMs, DM_host


# def log_prior(DL, H0, Obhsqf, prior_args):
#     """Redefine prior probabilities for our D_L."""
#     DL_meas = prior_args[0]  #np.asarray(prior_args[:len(DL)//2])
#     sigma_DL = prior_args[1]  #np.asarray(prior_args[len(DL)//2:])
#     if np.all(0 < DL) and 10 < H0 < 150. and 0.0 < Obhsqf < 0.1 :
#         return np.sum(- np.log(sigma_DL*np.sqrt(2*np.pi)) - (DL-DL_meas)**2/(2*sigma_DL**2))
#     else:
#         return -np.inf


def p_DL(DL_samps, **kwargs):
    DL_measured = kwargs['DL_measured']
    DL_measured = np.expand_dims(DL_measured, axis=-1)
    sigma_DL = kwargs['sigma_DL']
    return norm.pdf(DL_samps, loc=DL_measured, scale=sigma_DL)


# Redefine the FRB prior to a different one from James et al. Adapt the following code from mcmc.
def log_p_H0(H0, Obhsqf):
    """Probability of H0 given Omega_b*h^2*f_d.

    To not have to change everything I leave the dependency as is although it
    should only depend on ObfH0.
    """
    # Define the constant that James2022 assumed for O_b*H_0^2*f_d (ignoring z dependence of f_d)
    james_const = 0.02242*0.844
    translated_H0 = H0*james_const/Obhsqf
    log_p_H0 = norm.logpdf(translated_H0, loc=73, scale=0.8) + np.log(james_const/Obhsqf)

    return log_p_H0


if __name__ == '__main__':
    # Simulate FRBs.
    c = 299792.458
    Obhsqf = 0.02242*0.844
    H0 = 73
    Om = 0.3
    mu_host = 2.23
    DM0 = 10**mu_host
    sigma_host = .57  #0.57
    F = 0.32
    mcmc.mu_host = mu_host
    mcmc.sigma_host = sigma_host
    mcmc.F = F
    mcmc.log_p_H0 = log_p_H0

    n_FRBs = 100
    z_mean = 0.2

    DL_mean = c/H0*lum_dist(z_mean, Om=0.3)
    eDL = 0.2
    sigma_DL = eDL*DL_mean
    DL_meas, DMexc, DM_host = simulate_FRBs(n_FRBs, z_mean=z_mean, z_sigma=0, sigma_DL=sigma_DL, Obhsqf=Obhsqf,
                                            H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host, seed=45)

    # Add 10 low z, low eD_L FRBs.
    n_FRBs2 = 20
    z_mean2 = 0.1
    DL_mean2 = c/H0*lum_dist(z_mean, Om=0.3)
    eDL2 = 0.1
    sigma_DL2 = eDL2*DL_mean2
    DL_meas2, DMexc2, DM_host2 = simulate_FRBs(n_FRBs2, z_mean=z_mean2, z_sigma=0, sigma_DL=sigma_DL2, Obhsqf=Obhsqf,
                                            H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host, seed=121105)
    n_FRBs = n_FRBs + n_FRBs2

    # Concatenate the two simulated sets.
    DL_meas, DMexc, DM_host = np.concatenate((DL_meas, DL_meas2)), np.concatenate((DMexc, DMexc2)), np.concatenate((DM_host, DM_host2)),

    # Redefine prior for our D_L distributions.
    mcmc.p_DL = p_DL

    # Initialize the grid over which to integrate D_L and DM_cosmic
    n_rect_DL = 100
    n_rect_DM = 120
    DL_min = min(DL_mean-5*sigma_DL, DL_mean2-5*sigma_DL2)
    DL_max = max(DL_mean+5*sigma_DL, DL_mean2+5*sigma_DL2)
    initialize_integration(DMexc=DMexc, DL_min=DL_min, DL_max=DL_max,
                           n_rect_DM=n_rect_DM, n_rect_DL=n_rect_DL,
                           p_DL_kwargs={'DL_measured' : DL_meas, 'sigma_DL' : sigma_DL})

    # Do inference for the simulated FRBs. Initialize the walkers.
    nwalkers = 23
    rng = np.random.default_rng()
    H0_init = rng.normal(70, 10, size=(nwalkers, 1))
    Obhsqf_init = rng.normal(Obhsqf, 0.0025, size=(nwalkers, 1))
    initial = np.concatenate((H0_init, Obhsqf_init), axis=1)

    ndim = 2
    nsteps = 5000

    # Set up a backend to save the chains to.
    filename = os.path.join(config.DATA_DIR,
                            f"simulated_{n_FRBs}FRBs_tight_prior_z{z_mean}_{z_mean2}_eDL{eDL}_{eDL2}_{nwalkers}x{nsteps}steps_d.h5")
    if os.path.isfile(filename):
        print("Warning: File exists and will be appended to.")
    backend = emcee.backends.HDFBackend(filename)
    # backend.reset(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        backend=backend, pool=pool)
        sampler.run_mcmc(initial, nsteps, progress=True, progress_kwargs={'mininterval':5})

    # Sample the James prior.
    # ndim_J = 2
    # nsteps_J = 30_000

    # filename = os.path.join(config.DATA_DIR, f"1kFRB_prior_{nwalkers}x{nsteps_J}steps.h5")
    # backend = emcee.backends.HDFBackend(filename)

    # initial_J = np.concatenate((H0_init, Obhsqf_init), axis=1)

    # # Test log_p James
    # with Pool() as pool:
    #     sampler_J2 = emcee.EnsembleSampler(nwalkers, ndim_J, log_p_H0_with_prior, backend=backend,
    #                                         pool=pool)
    #     sampler_J2.run_mcmc(initial_J, nsteps_J, progress=True, progress_kwargs={'mininterval':5})

    # Sample the GW-FRB posterior without the FRB-z prior.
    filename = os.path.join(config.DATA_DIR,
                            f"simulated_noz_{n_FRBs}FRBs_tight_prior_z{z_mean}_{z_mean2}_eDL{eDL}_{eDL2}_{nwalkers}x{nsteps}steps_d.h5")
    backend = emcee.backends.HDFBackend(filename)

    with Pool() as pool:
        sampler_noz = emcee.EnsembleSampler(nwalkers, ndim, log_probability_without_FRBs,
                                            backend=backend, pool=pool)
        sampler_noz.run_mcmc(initial, nsteps, progress=True, progress_kwargs={'mininterval':5})
