#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:08:54 2023

@author: jjahns
"""
import numpy as np
import emcee
import corner

import mcmc

from mcmc import p_DMcosmic, lum_dist, log_probability, log_p_H0_with_prior
from mcmc import log_probability_without_FRBs

def draw_DM(frb_zs, Obf=0.035, H0=70, F=0.32, Om=0.3, DM0=100, sigma_host=1, rng=None):
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
    dm_cosmic = [float(draw_DM_cosmic(z, Obf=Obf, H0=H0, F=F, Om=Om, n_samples=1, rng=rng))
                 for z in frb_zs]

    # Draw a DM_host.
    dm_host = rng.lognormal(mean=np.log(DM0), sigma=np.log(10)*sigma_host, size=len(frb_zs))

    return dm_host/(1+frb_zs) + dm_cosmic, dm_host


def draw_DM_cosmic(z, Obf=0.035, H0=70, F=0.32, Om=0.3, n_samples=1, rng=None):
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

    pdf = p_DMcosmic(DM_values, z, F, H0, Obf=Obf, Om=Om)

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


def simulate_FRBs(n_draw, z_mean, z_sigma, sigma_DL, Obf=0.035, H0=70, F=0.32, Om=0.3, DM0=100,
                  sigma_host=1):
    c = 299792.458

    rng = np.random.default_rng()
    zs = positive_normal(loc=z_mean, scale=z_sigma, size=n_draw, rng=rng)

    DMs, DM_host = draw_DM(zs, Obf=Obf, H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host, rng=rng)

    DL_mean = c/H0*lum_dist(zs, Om)  # in Mpc, 20% uncertainty
    DL = positive_normal(loc=DL_mean, scale=sigma_DL, rng=rng)

    return DL, DMs, DM_host


def log_prior(DL, H0, Obf, prior_args):
    """Redefine prior probabilities for our D_L."""
    DL_meas = prior_args[0]  #np.asarray(prior_args[:len(DL)//2])
    sigma_DL = prior_args[1]  #np.asarray(prior_args[len(DL)//2:])
    if np.all(0 < DL) and 10 < H0 < 150. and 0.0 < Obf < 0.2 :
        return np.sum(- np.log(sigma_DL*np.sqrt(2*np.pi)) - (DL-DL_meas)**2/(2*sigma_DL**2))
    else:
        return -np.inf


if __name__ == '__main__':
    # Simulate FRBs.
    n_FRBs = 10

    c = 299792.458
    Obf = 0.035
    H0 = 70
    Om = 0.3
    z_mean=0.03
    mu_host = 2.23
    DM0 = 10**mu_host
    sigma_host = 0.57
    F = 0.32
    sigma_DL = 0.1*c/H0*lum_dist(z_mean, Om=0.3)
    DL_meas, DMexc, DM_host = simulate_FRBs(n_FRBs, z_mean=z_mean, z_sigma=0, sigma_DL=sigma_DL, Obf=Obf,
                                   H0=H0, F=F, Om=Om, DM0=DM0, sigma_host=sigma_host)

    # Redefine prior for our D_L distributions.
    mcmc.log_prior = log_prior

    # Do inference for the simulated FRBs. Initialize the walkers.
    nwalkers = 100
    rng = np.random.default_rng()
    H0_init = rng.normal(70, 10, size=(nwalkers, 1))
    Obf_init = rng.normal(0.035, 0.005, size=(nwalkers, 1))
    DL_init = rng.normal(DL_meas, 10, size=(nwalkers, n_FRBs))
    DMhost_init = rng.uniform(0, DMexc, size=(nwalkers, n_FRBs))
    initial = np.concatenate((H0_init, Obf_init, DL_init, DMhost_init), axis=1)

    ndim = 2 + 2*n_FRBs
    nsteps = 50000

    # Set up a backend to save the chains to.
    filename = f"../Data/simulated_{n_FRBs}FRBs_{nwalkers}x{nsteps}steps.h5"
    backend = emcee.backends.HDFBackend(filename)
    # backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(DMexc, DL_meas, sigma_DL), backend=backend)
    sampler.run_mcmc(initial, nsteps, progress=True)

    # Sample the James prior.
    ndim_J = 2
    nsteps_J = 50000

    filename = f"../Data/James_prior_{nwalkers}x{nsteps}steps.h5"
    backend = emcee.backends.HDFBackend(filename)

    initial_J = np.concatenate((H0_init, Obf_init), axis=1)
    # sampler_J = emcee.EnsembleSampler(nwalkers, ndim_J, log_p_Obf_with_prior, backend=backend)
    # sampler_J.run_mcmc(initial_J, nsteps_J, progress=True,)

    # Test log_p James
    sampler_J2 = emcee.EnsembleSampler(nwalkers, ndim_J, log_p_H0_with_prior, backend=backend)
    sampler_J2.run_mcmc(initial_J, nsteps_J, progress=True,)

    # Sample the GW-FRB posterior without the FRB-z prior.
    filename = f"../Data/simulated_noz_{n_FRBs}FRBs_{nwalkers}x{nsteps}steps.h5"
    backend = emcee.backends.HDFBackend(filename)

    sampler_noz = emcee.EnsembleSampler(nwalkers, ndim, log_probability_without_FRBs,
                                    args=(DMexc, DL_meas, sigma_DL), backend=backend)
    sampler_noz.run_mcmc(initial, nsteps, progress=True)



