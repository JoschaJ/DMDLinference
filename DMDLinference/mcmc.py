#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:32:29 2023

@author: jjahns
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import emcee
import corner
import astropy.constants as const
import astropy.units as u

from multiprocessing import Pool
from scipy.interpolate import CubicSpline  # interpolate James' data points
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from scipy.special import hyp2f1, hyp1f1, gamma
from scipy.optimize import minimize_scalar

import config

# Define a prior probability function on O_m h^2 f_d from FRBs by James.
james_file = os.path.join(config.DATA_IN_DIR, 'James2022_H0_posterior.csv')
james_data = np.loadtxt(james_file, delimiter=';')

H0_points = james_data[:, 0]
p_H0 = james_data[:, 1]

p_H0_unnorm = CubicSpline(H0_points, p_H0, extrapolate=False)
norm_p_H0 = quad(p_H0_unnorm, H0_points.min(), H0_points.max())[0]

# Define the constant that James2022 assumed for O_b*H_0^2*f_d (ignoring z dependence of f_d)
james_const = 0.02242*0.844


def log_p_H0(H0, Obhsqf):
    """Probability of H0 given Omega_b*h^2*f_d."""
    translated_H0 = H0*james_const/Obhsqf
    log_p_H0 = np.where((H0_points.min() < translated_H0) & (translated_H0 < H0_points.max()),
                     np.log(p_H0_unnorm(translated_H0)/norm_p_H0), -np.inf)

    return log_p_H0


def rectangle_integration(f, a, b, n=100):
    """Integrate f from a to b using the rectangle rule

    n : Number of rectangles.
    """
    dx = (b-a)/n
    x = np.linspace(a+dx/2, b-dx/2, n)
    return np.sum(f(x))*dx

# Get the measured posterior on the luminosity distance of GW190425.
fn = os.path.join(config.DATA_IN_DIR, "IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_nocosmo.h5")
data = h5py.File(fn,'r')

posterior_samples = data['C01:IMRPhenomPv2_NRTidal:LowSpin']['posterior_samples']
DLum = posterior_samples['luminosity_distance']
p_DL_tmp = gaussian_kde(DLum)
p_DL = lambda DL, *args: p_DL_tmp(DL)


def lum_dist(z, Om=0.3):
    """The luminosity distance for a flat LCDM universe in units of the hubble distance"""
    Ode = 1-Om
    int_efunc = lambda z: ((z+1) * hyp2f1(1/3, 1/2, 4/3, -Om*(z+1)**3/Ode)
                           / np.sqrt(Ode))  # One factor z+1 less would be the comoving distance
    return (z+1) * (int_efunc(z) - int_efunc(0))

# Get z(d_L) in units
z = np.linspace(0, 10, 1000)
z_of_DL_hubble_unit = CubicSpline(lum_dist(z), z, extrapolate=False)

def z_of_DL(DL, H0):
    """Redshift given the luminosity distance in Mpc"""
    c = 299792.458  # in km/s
    return z_of_DL_hubble_unit(H0/c * DL)

log10n_norm = np.log10(np.e)/np.sqrt(2*np.pi)

def log10_normal(x, mu, sigma):
    """Probability density to measure a given host DM in the host frame"""
    return log10n_norm / (x*sigma) * np.exp(-(np.log10(x)-mu)**2/(2*sigma**2))


def average_DM(z, H0, Obhsqf, Om=0.3):
    free_elec = 0.875

    coeff_top = 3 * const.c
    coeff_bot = 8 * np.pi * const.G * const.m_p
    coeff = coeff_top / coeff_bot
    coeff = coeff.to("pc Mpc s cm**-3 km**-1").value  # Unit of DM/H_0

    # Use the (flat universe) analytic solution of the integral over (1+z)/E(z)
    Ode = 1-Om
    int_1pz_efunc = lambda z: (1/2 * (z+1) * (z+1) * hyp2f1(1/2, 2/3, 5/3, -Om*(z+1)**3/Ode)
                           / np.sqrt(Ode))
    integrand = int_1pz_efunc(z) - int_1pz_efunc(0)

    return coeff * Obhsqf * 10000 / H0 * free_elec * integrand


gamma133 = gamma(1/3)*3
gamma562 = 2**(1/2)*gamma(5/6)


def average_delta_minus_1(C0, sigma):
    # Compute x that is used in every hyp1f1
    hyp_x = -C0**2/18/sigma**2

    normalization = 3*(12*sigma)**(1/3) / (gamma133*sigma*hyp1f1(1/6, 1/2, hyp_x)
                                           + gamma562*C0*hyp1f1(2/3, 3/2, hyp_x))
    #normalization = 3*np.cbrt(12*sigma)/(gamma(1/3)*3*sigma*hyp1f1(1/6, 1/2, hyp_x)
    #                                     + np.sqrt(2)*C0*gamma(5/6)*hyp1f1(2/3, 3/2, hyp_x))
    # if not np.isfinite(normalization) or not np.isfinite(hyp1f1(1/6, 1/2, hyp_x)):
    #     print(hyp_x, C0, sigma)

    avrg_DM = normalization/3 * (gamma(1/6)*hyp1f1(1/3, 1/2, hyp_x)/(2**5/9/sigma**2)**(1/6)
                                    + C0*gamma(2/3)*hyp1f1(5/6, 3/2, hyp_x)/(18*sigma**2)**(1/3))
    return np.abs(avrg_DM-1)


def p_DMcosmic(DMcosmic, z, F, H0, Obhsqf, Om):
    """Stolen from frb.dm.cosmic
    PDF(Delta) following the McQuinn formalism describing the DM_cosmic PDF

    See Macquart+2020 for details

    Args:
        Delta (float or np.ndarray):
            DM / averageDM values
        C0 (float):
            parameter
        F (float):
        A (float, optional):
        alpha (float, optional):
        beta (float, optional):

    Returns:
        float or np.ndarray:
    """
    # Calculate Delta from the mean DM.
    avrg_DM = average_DM(z, H0=H0, Obhsqf=Obhsqf, Om=Om)
    delta = DMcosmic/avrg_DM

    # Calculate C_0 for the given z.
    sigma = F/np.sqrt(z)
    if isinstance(sigma, float):
        C0 = minimize_scalar(average_delta_minus_1, args=(sigma)).x
    else:
        C0 = [minimize_scalar(average_delta_minus_1, args=(sig)).x for sig in sigma]
        C0 = np.array(C0).reshape(sigma.shape)

    # Calculate the normalization. This might not be needed.
    hyp_x = -C0**2/18/sigma**2
    A = 3*(12*sigma)**(1/3) / (gamma133*sigma*hyp1f1(1/6, 1/2, hyp_x)
                              + gamma562*C0*hyp1f1(2/3, 3/2, hyp_x))

    # Put everything together into the PDF. Correct the normalization for
    # the transition Delt -> DM by a factor avrg_DM.
    alpha = -3
    beta = -3
    pdf = A / avrg_DM * delta**beta * np.exp(-((delta**alpha-C0)**2 / (2*alpha**2*sigma**2)))
    return pdf


def p_DM(DL, H0, Obhsqf):
    """Probability measureing a DM given the distance and parameters."""
    z = z_of_DL(DL, H0)
    if z.shape:
        z = z[:, np.newaxis]

    p_DMhost = (1+z)*log10_normal(DMcosmic[..., ::-1]*(1+z), mu=mu_host, sigma=sigma_host)
    integral = np.sum(p_DMhost*p_DMcosmic(DMcosmic, z, F, H0, Obhsqf, Om=0.3), axis=-1)*dDM

    return integral


def initialize_integration(DMexc, DL_min, DL_max, n_rect_DM, n_rect_DL, p_DL_kwargs={}):
    """Define global variables for integration that stay the same during one run"""
    # Integral step and DMcosmos samples.
    a, b = 0, DMexc
    global dDM, DMcosmic
    dDM = (b-a)/n_rect_DM
    DMcosmic = np.linspace(a+dDM/2, b-dDM/2, n_rect_DM, axis=-1)
    dDM = np.expand_dims(dDM, axis=-1)
    DMcosmic = np.expand_dims(DMcosmic, axis=-2)

    # Integral step and DL samples.
    global dDL, DL_samples, p_DL_sampled
    DL_min = np.where(DL_min > 0, DL_min, 0)
    dDL = (DL_max-DL_min)/n_rect_DL
    DL_samples = np.linspace(DL_min+dDL/2, DL_max-dDL/2, n_rect_DL)
    p_DL_sampled = p_DL(DL_samples, **p_DL_kwargs)


def log_prior(H0, Obhsqf):
    """Define the combination of all prior probabilities."""
    if 10 < H0 < 150. and 0.0 < Obhsqf < 0.1 :
        return 0
    else:
        return -np.inf


def log_prior_DMhost(DMhost, DMexc):
    """Put a flat prior on DM_host below the measured DM"""

    if np.all(0 <= DMhost) and np.all(DMhost <= DMexc):
        log_prob = 0
    else:
        log_prob = -np.inf
    return log_prob


def log_probability(theta):
    """Sum up all log probabilities"""
    # Extract parameters
    H0, Obhsqf = theta

    lp = log_prior(H0, Obhsqf) + log_p_H0(H0, Obhsqf)

    # Evaluate the likelihood, only if the prior is non-infinite.
    if np.isfinite(lp):
        # Integrate over DL.
        log_prob = lp + np.sum(np.log(np.sum(p_DL_sampled*p_DM(DL_samples, H0, Obhsqf)*dDL, axis=-1)))
    else:
        log_prob = -np.inf
    return log_prob


def log_probability_without_FRBs(theta):
    """Sum up all log probabilities"""
    # Extract parameters
    H0, Obhsqf = theta

    lp = log_prior(H0, Obhsqf)

    # Evaluate the likelihood, only if the prior is non-infinite.
    if np.isfinite(lp):
        log_prob = lp + np.sum(np.log(np.sum(p_DL_sampled*p_DM(DL_samples, H0, Obhsqf)*dDL, axis=-1)))
    else:
        log_prob = -np.inf
    return log_prob


def log_p_H0_with_prior(theta):
    """p(Obf,H_0) with a prior to plot contours from only the FRB-z"""
    H0, Obhsqf = theta
    if .1 < H0 < 150. and 0.0 < Obhsqf < .5:
        return log_p_H0(H0, Obhsqf)
    else:
        return -np.inf


if __name__ == '__main__':
    DMexc_ne2001 = 79.4

    # Fixing values to the ones reported by James et al. 2022
    # (i.e. using delta function priors instead of marginalizing)
    mu_host = 2.23
    sigma_host = 0.57
    F = 0.32
    #f_igm = 0.83

    nwalkers = 24
    rng = np.random.default_rng()
    # DL_init = rng.normal(DLum.mean(), 10, size=nwalkers)
    n_rect_DL = 100
    n_rect_DM = 120
    initialize_integration(DMexc=DMexc_ne2001, DL_min=DLum.min(), DL_max=DLum.max(),
                           n_rect_DM=n_rect_DM, n_rect_DL=n_rect_DL)

    H0_init = rng.normal(70, 5, size=nwalkers)
    h_init = 0.7
    Obhsqf_init = rng.normal(0.035*h_init**2, 0.005*h_init**2, size=nwalkers)
    initial = np.stack((H0_init, Obhsqf_init,), axis=1)
    # initial = [DL_init, H0_init, Obhsqf_init, DMhost_init]
    nwalkers, ndim = initial.shape
    nsteps = 20000

    # Set up a backend to save the chains to.
    filename = os.path.join(config.DATA_DIR, f"real_FRB_{nwalkers}x{nsteps}steps.h5")
    backend = emcee.backends.HDFBackend(filename)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_without_FRBs,
                                        backend=backend, pool=pool)
        sampler.run_mcmc(initial, nsteps, progress=True, progress_kwargs={'mininterval':5})

    tau = sampler.get_autocorr_time()
    print(tau)

    # Sample the James prior.
    # ndim = 2
    # nsteps_J = 5000

    # initial_J = np.stack((H0_init, Obhsqf_init), axis=1)
    # sampler_J = emcee.EnsembleSampler(nwalkers, ndim, log_p_Obf_with_prior)
    # sampler_J.run_mcmc(initial_J, nsteps_J, progress=True,)

    # labels=['$H_0$', r'$\Omega_b f_d$', '$D_L$'] + [r"DM$_\mathrm{host}$"+f"{str(frb)}"
    #                                                for frb in range(ndim-3)]
    # labels=(['$H_0$', r'$\Omega_b f_d$'])
    # fig = corner.corner(sampler_J, labels=labels, smooth=0.5,
    #                   levels=(0.68, 0.95),
    #                   color='r',
    #                   hist_kwargs={'density' : True},
    #                   plot_density=False,
    #                   plot_datapoints=False,
    #                   fill_contours=True,
    #                   contour_kwargs={'linewidths' : 1.,},
    #                   contourf_kwargs={'colors' : ['w',(1,0,0,.3), (1,0,0,.6)],},
    #                   )
    # fig = corner.corner(sampler.chain[:,:,:2], labels=labels, smooth=0.5,
    #                     levels=(0.68, 0.95),
    #                     color='b',
    #                     fig=fig,
    #                     hist_kwargs={'density' : True},
    #                     plot_density=False,
    #                     plot_datapoints=False,
    #                     fill_contours=True,
    #                     contour_kwargs={'linewidths' : 1.,},
    #                     contourf_kwargs={'colors' : [(0,0,0,0),(0,0,1,.3), (0,0,1,.6)],},
    #                     )
    # fig.savefig(f"Figures/real_FRB_{nwalkers}x{nsteps}steps.png", dpi=500, bbox_inches='tight')
