#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:32:29 2023

@author: jjahns
"""
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

# Define a prior probability function on O_mH_0f_d from FRBs by James.
james_data = np.loadtxt('James2022_H0_posterior.csv', delimiter=';')

H0_points = james_data[:, 0]
p_H0 = james_data[:, 1]

# Define the constant that James2022 assumed for O_b*H_0^2*f_d (ignoring z dependence of f_d)
james_const = 0.02242*10000*0.844


def ObHf(H0):
    """Omega_b*H_0*f_d"""
    return james_const/H0


p_ObHf_points = p_H0*james_const/ObHf(H0_points)**2
ObHf_points = ObHf(H0_points)[::-1]
p_ObHf_unnorm = CubicSpline(ObHf_points, p_ObHf_points[::-1], extrapolate=False)

normalization = quad(p_ObHf_unnorm, ObHf_points.min(), ObHf_points.max())[0]
# def p_OHf(ObHf):
#     return p_OHf_unnorm(ObHf)/normalization

def log_p_Obf(Obf, H0):
    """Probability of Omega_b*f_d given H0"""
    log_p_Obf = np.where((ObHf_points.min()/H0 < Obf) & (Obf < ObHf_points.max()/H0),
                     np.log(p_ObHf_unnorm(Obf*H0)*H0/normalization), -np.inf)

    return log_p_Obf


p_H0_unnorm = CubicSpline(H0_points, p_H0, extrapolate=False)
norm_p_H0 = quad(p_H0_unnorm, H0_points.min(), H0_points.max())[0]


def log_p_H0(H0, Obf):
    """Probability of Omega_b*f_d given H0"""
    log_p_H0 = np.where((H0_points.min() < ObHf(H0)/Obf) & (ObHf(H0)/Obf < H0_points.max()),
                     np.log(p_H0_unnorm(ObHf(H0)/Obf)/norm_p_H0), -np.inf)

    return log_p_H0


# x = np.linspace(ObHf_points.min()/70, ObHf_points.max()/70)
# plt.plot(x, p_Obf(x, H0=70))

# Get the measured posterior on the luminosity distance of GW190425.
fn = "IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_nocosmo.h5"
data = h5py.File(fn,'r')

posterior_samples = data['C01:IMRPhenomPv2_NRTidal:LowSpin']['posterior_samples']
DLum = posterior_samples['luminosity_distance']
p_DL = gaussian_kde(DLum)
# p_DL = lambda DL, *prior_args: p_DL(DL)

# Fixing values to the ones reported by James et al. 2022
# (i.e. using delta function priors instead of marginalizing)
mu_host = 2.23
sigma_host = 0.57
F = 0.32
#f_igm = 0.83

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

# z2 = np.linspace(0, 10, 1000)
# DL = np.linspace(0, 120, 1000)
# plt.plot(DL, z_of_DL(DL, 70))
# c = 299792.458
# plt.plot(lum_dist(z2)*c/70/1000, z2)
logn_norm = np.log(np.log10(np.e))-np.log(np.sqrt(2*np.pi))

def log_log10_normal(x, mu, sigma):
    """Probability density to measure a given host DM in the observers frame"""
    return logn_norm - np.log(x*sigma) - (np.log10(x)-mu)**2/(2*sigma**2)


def average_DM(z, H0, Obf, Om=0.3):
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

    return coeff * H0 * Obf * free_elec * integrand


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


def log_p_DMcosmic(DMcosmic, z, F, H0, Obf, Om):
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
    avrg_DM = average_DM(z, H0=H0, Obf=Obf, Om=Om)
    delta = DMcosmic/avrg_DM

    # Calculate C_0 for the given z.
    sigma = F/np.sqrt(z)
    if isinstance(sigma, float):
        C0 = minimize_scalar(average_delta_minus_1, args=(sigma)).x
    else:
        C0 = [minimize_scalar(average_delta_minus_1, args=(sig)).x for sig in sigma]
        C0 = np.array(C0)

    # Calculate the normalization. This might not be needed.
    hyp_x = -C0**2/18/sigma**2
    A = 3*(12*sigma)**(1/3) / (gamma133*sigma*hyp1f1(1/6, 1/2, hyp_x)
                              + gamma562*C0*hyp1f1(2/3, 3/2, hyp_x))

    # Put everything together into the PDF. Correct the normalization for
    # the transition Delt -> DM by a factor avrg_DM.
    alpha = -3
    beta = -3
    log_pdf = np.log(A / avrg_DM * delta**beta) - ((delta**alpha-C0)**2 / (2*alpha**2*sigma**2))
    return log_pdf


def p_DMcosmic(DMcosmic, z, F, H0, Obf, Om):
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
    avrg_DM = average_DM(z, H0=H0, Obf=Obf, Om=Om)
    delta = DMcosmic/avrg_DM

    # Calculate C_0 for the given z.
    sigma = F/np.sqrt(z)
    if isinstance(sigma, float):
        C0 = minimize_scalar(average_delta_minus_1, args=(sigma)).x
    else:
        C0 = [minimize_scalar(average_delta_minus_1, args=(sig)).x for sig in sigma]
        C0 = np.array(C0)

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


def log_p_DM(DMexc, DMhost, DL, H0, Obf):
    """Probability measureing a DM given the distance and parameters."""
    z = z_of_DL(DL, H0)
    log_p_DMhost = log_log10_normal(DMhost*(1+z), mu=mu_host, sigma=sigma_host)
    log_p = log_p_DMhost + log_p_DMcosmic(DMexc-DMhost, z, F, H0, Obf, Om=0.3)
    # integ = quad(p_product, 0.001, DMexc-0.001)
    return np.sum(log_p)


def log_prior(DL, H0, Obf, prior_args):
    """Define the combination of all prior probabilities."""
    if np.all(0 < DL) and 10 < H0 < 500. and 0.0 < Obf < 1.0 :
        return np.sum(np.log(p_DL(DL)))
    else:
        return -np.inf


def log_prior_DMhost(DMhost, DMexc):
    """Put a flat prior on DM_host below the measured DM"""

    if np.all(0 <= DMhost) and np.all(DMhost <= DMexc):
        log_prob = 0
    else:
        log_prob = -np.inf
    return log_prob


def log_probability(theta, *args):
    """Sum up all log probabilities"""
    # Extract parameters
    H0, Obf, *DL_and_DMhost = theta
    n_FRB = len(DL_and_DMhost)//2
    DL = np.asarray(DL_and_DMhost[:n_FRB])
    DMhost = np.asarray(DL_and_DMhost[n_FRB:])

    # Extract aruments.
    DMexc = args[0]
    prior_args = args[1:]

    lp = log_prior(DL, H0, Obf, prior_args)
    lp += log_prior_DMhost(DMhost, DMexc)

    # Evaluate the likelihood, only if the prior is non-infinite.
    if np.isfinite(lp):
        log_prob = lp + log_p_H0(H0, Obf) + log_p_DM(DMexc, DMhost, DL, H0, Obf)
    else:
        log_prob = -np.inf
    return log_prob


def log_probability_without_FRBs(theta, *args):
    """Sum up all log probabilities"""
    # Extract parameters
    H0, Obf, *DL_and_DMhost = theta
    n_FRB = len(DL_and_DMhost)//2
    DL = np.asarray(DL_and_DMhost[:n_FRB])
    DMhost = np.asarray(DL_and_DMhost[n_FRB:])

    # Extract aruments.
    DMexc = args[0]
    prior_args = args[1:]

    lp = log_prior(DL, H0, Obf, prior_args)
    lp += log_prior_DMhost(DMhost, DMexc)

    # Evaluate the likelihood, only if the prior is non-infinite.
    if np.isfinite(lp):
        log_prob = lp + log_p_DM(DMexc, DMhost, DL, H0, Obf)
    else:
        log_prob = -np.inf
    return log_prob


def log_p_Obf_with_prior(theta):
    """p(Obf,H_0) with a prior to plot contours from only the FRB-z"""
    H0, Obf = theta
    if .1 < H0 < 150. and 0.0 < Obf < 1.0:
        return log_p_Obf(Obf, H0)
    else:
        return -np.inf


def log_p_H0_with_prior(theta):
    """p(Obf,H_0) with a prior to plot contours from only the FRB-z"""
    H0, Obf = theta
    if .1 < H0 < 150. and 0.0 < Obf < 1.0:
        return log_p_H0(H0, Obf)
    else:
        return -np.inf


if __name__ == '__main__':
    DMexc_ne2001 = 79.4
    nwalkers = 100
    rng = np.random.default_rng()
    DL_init = rng.normal(DLum.mean(), 10, size=nwalkers)
    H0_init = rng.normal(70, 5, size=nwalkers)
    Obf_init = rng.normal(0.035, 0.005, size=nwalkers)
    DMhost_init = rng.uniform(0, DMexc_ne2001, size=nwalkers)
    initial = np.stack((H0_init, Obf_init, DL_init, DMhost_init), axis=1)
    # initial = [DL_init, H0_init, Obf_init, DMhost_init]
    nwalkers, ndim = initial.shape
    nsteps = 100000

    # Set up a backend to save the chains to.
    filename = f"../Data/real_FRB_{nwalkers}x{nsteps}steps.h5"
    backend = emcee.backends.HDFBackend(filename)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(DMexc_ne2001,),
                                        backend=backend, pool=pool)
        sampler.run_mcmc(initial, nsteps, progress=True)

    tau = sampler.get_autocorr_time()
    print(tau)

    # Sample the James prior.
    # ndim = 2
    # nsteps_J = 5000

    # initial_J = np.stack((H0_init, Obf_init), axis=1)
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
