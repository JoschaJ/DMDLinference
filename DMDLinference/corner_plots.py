#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:58:41 2023

@author: jjahns
"""
import os
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import arviz as az

from scipy.stats import mode

import config

# Load results from inference on simulated FRB-GW events.
# Define filenames.
mcmc_set = "free_host" #"free_host"  # "10fixed_host"
if mcmc_set == "10fixed_host":
    filename = os.path.join(config.DATA_DIR, "simulated_10FRBs_z0.2_0.1_eDL0.2_0.4_8x5000steps_h.h5")
    filename_prior = os.path.join(config.DATA_DIR, "James_prior_8x10000steps.h5")
    filename_noz = os.path.join(config.DATA_DIR, "simulated_noz_10FRBs_z0.2_0.1_eDL0.2_0.4_8x5000steps_h.h5")
if mcmc_set == "120fixed_host":
    filename = os.path.join(config.DATA_DIR, "simulated_120FRBs_tight_prior_z0.2_0.1_eDL0.2_0.1_23x5000steps_d.h5")
    filename_prior = os.path.join(config.DATA_DIR, "1kFRB_prior_8x10000steps.h5")
    filename_noz = os.path.join(config.DATA_DIR, "simulated_noz_120FRBs_tight_prior_z0.2_0.1_eDL0.2_0.1_8x5000steps_d.h5")
if mcmc_set == "free_host":
    filename = os.path.join(config.DATA_DIR, "simulated_120FRBs_free_host_z0.2_0.1_eDL0.2_0.1_8x5000steps_d.h5")
    filename_prior = os.path.join(config.DATA_DIR, "1kFRB_prior_8x10000steps.h5")
    filename_noz = os.path.join(config.DATA_DIR, "simulated_noz_120FRBs_free_host_z0.2_0.1_eDL0.2_0.1_8x5000steps_d.h5")

    # filename = os.path.join(config.DATA_DIR, #"simulated_120FRBs_free_host_vary_z0.2_0.1_eDL0.2_0.1_8x5000steps.h5")
    #                         # "simulated_120FRBs_free_host_z0.2_0.1_eDL0.2_0.1_8x5000steps_d.h5")
    #                         "simulated_120FRBs_tight_prior_z0.2_0.1_eDL0.2_0.1_23x5000steps_d.h5")
    #                         # "simulated_10FRBs_z0.1_eDL0.4_24x5000steps.h5")
    #                         # #"simulated_110FRBs_z0.2_0.1_eDL0.2_0.1_23x5000steps.h5") #"real_FRB_24x5000steps.h5")  #
    # filename_prior = os.path.join(config.DATA_DIR, "1kFRB_prior_23x30000steps.h5")  # "James_prior_24x5000steps.h5")  #

    # # Results without taking James et al. or other FRB-z prior into account.
    # filename_noz = os.path.join(config.DATA_DIR,
    #                             "simulated_noz_120FRBs_tight_prior_z0.2_0.1_eDL0.2_0.1_8x5000steps_d.h5")
    #                             # "simulated_noz_120FRBs_free_host_vary_z0.2_0.1_eDL0.2_0.1_8x5000steps.h5")
    #                             #"simulated_noz_10FRBs_z0.1_eDL0.4_24x5000steps.h5")
    #                             #"simulated_noz_110FRBs_z0.2_0.1_eDL0.2_0.1_23x5000steps.h5") #"real_FRB_noz_24x5000steps.h5")  #

assert(os.path.isfile(filename)), "Combined file " + filename + " not exist."
assert(os.path.isfile(filename_prior)), "Prior file " + filename_prior + " not exist."
assert(os.path.isfile(filename_noz)), "No-z file " + filename_noz + " not exist."

sampler = emcee.backends.HDFBackend(filename)

#tau = sampler.get_autocorr_time()
#burnin = int(2 * np.max(tau))
samples = sampler.get_chain()
n_vars = samples.shape[2]
fig, axes = plt.subplots(n_vars, figsize=(10, 7), sharex=True)
labels = ['$H_0$ (km/s/Mpc)', r'$\Omega_b h^2 f_d$', '$\mu_\mathrm{host}$', '$\sigma_\mathrm{host}$']
for i in range(n_vars):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
fig.savefig(os.path.join(config.DATA_DIR, "chains.png"))

samples = sampler.get_chain(discard=500)

# Results from only James
sampler_J = emcee.backends.HDFBackend(filename_prior)

# tau = sampler_J.get_autocorr_time()
# burnin = int(2 * np.max(tau))
samples_J = sampler_J.get_chain(discard=1000)

# Load results without taking James et al. or other FRB-z prior into account.
sampler_noz = emcee.backends.HDFBackend(filename_noz)

#tau = sampler_noz.get_autocorr_time()
#burnin = int(2 * np.max(tau))
samples_noz = sampler_noz.get_chain(discard=500)

# Plot corner plot with all DMs and D_Ls
# labels=(['$H_0$', r'$\Omega_b f_d$']
#         + [r"$D_L$"+f"{str(frb)}" for frb in range(n_FRBs)]
#         + [r"DM$_\mathrm{host}$"+f"{str(frb)}" for frb in range(n_FRBs)]
#         )
# fig = corner.corner(sampler, labels=labels, truths=[H0, Obf, *DL_meas, *DM_host])

labels=(['$H_0$ (km/s/Mpc)', r'$\Omega_\mathrm{b} h^2 f_\mathrm{d}$', '$\mu_\mathrm{host}$', '$\sigma_\mathrm{host}$'])

cm2 = sns.color_palette('deep') #plt.get_cmap('tab10')
cm = sns.color_palette('pastel')

alpha = 0.5

color_set = [[(0,0,0,0), list(cm[2]) + [alpha], list(cm2[2]) + [alpha]],
             [(0,0,0,0), list(cm[1]) + [alpha], list(cm2[1]) + [alpha]],
             [(0,0,0,0), list(cm[0]) + [alpha], list(cm2[0]) + [alpha]]]

if n_vars == 2:
    if mcmc_set == "10fixed_host":
        range_vals = [(20, 150), (0, 0.07)]
        smooth = 1.
    else:
        range_vals = [(40, 100), (0, 0.03)]  #0.045
        smooth = 0.5
    truth_vals = [73, 0.02242*0.844]
    labelpad = 0
elif n_vars == 4:
    range_vals = [(40, 100), (0, 0.03), (1.5, 2.5), (0.3, 1)]
    truth_vals = [73, 0.02242*0.844, 2, 0.57]
    smooth = 1.
    labelpad = 0.1

plot_kwargs = dict(labels=labels,
                   bins=50,
                   hist_bin_factor=0.5,
                   #smooth=.1, #.5,
                   levels=(0.68, 0.95),
                   plot_density=False,
                   plot_datapoints=False,
                   fill_contours=True,
                   max_n_ticks=3,
                   labelpad=labelpad,
                   range=range_vals,
                   )
fig = plt.figure(figsize=(5, 5))
if n_vars == 2:
    fig = corner.corner(samples_J.swapaxes(0,1),
                        color=cm2[2],
                        hist_kwargs={'density' : True, 'lw' : 1.5, 'label' : "FRB-$z$ constraints", 'linestyle':':'},  #
                        contour_kwargs={'linewidths' : 1., 'colors' : [cm2[2]], 'algorithm' : 'threaded', 'linestyles':':'},
                        contourf_kwargs={'colors' : color_set[0],},
                        smooth=0.5,
                        **plot_kwargs,
                        )
fig = corner.corner(samples_noz,  #.swapaxes(0,1)
                    color=cm2[1],
                    fig=fig,
                    hist_kwargs={'density' : True, 'lw' : 1.5, 'label' : "FRB-GW constraints", 'linestyle':'--'},  #
                    contour_kwargs={'linewidths' : 1., 'colors' : [cm2[1]], 'linestyles':'--'},
                    contourf_kwargs={'colors' : color_set[1]},
                    smooth=smooth,
                    **plot_kwargs,
                    )
fig = corner.corner(samples, #swapaxes(0,1)
                    color=cm2[0],
                    fig=fig,
                    truths=truth_vals,
                    hist_kwargs={'density' : True, 'lw' : 1.5, 'label' : "Combined constraints"},  #
                    contour_kwargs={'linewidths' : 1., 'colors' : [cm2[0]]},
                    contourf_kwargs={'colors' : color_set[2],},
                    smooth=smooth,
                    **plot_kwargs
                    )
axs = fig.get_axes()

if n_vars == 2:
    plt.legend(bbox_to_anchor=(1.05, 1.5), loc="upper right")
elif n_vars == 4:
    axs[-1].set_xticks([0.4, 0.6, 0.8])
    axs[-4].set_yticks([0.4, 0.6, 0.8])
    plt.legend(bbox_to_anchor=(1.05, 2.7), loc="upper right")
# sample_labels = ["Combined constraints",
#                  "FRB-GW constraints only",
#                  "FRB-z priors"
#                  ]
# plt.legend(
#         handles=[
#             mlines.Line2D([], [], color=cm2[i], label=sample_labels[i])
#             for i in range(len(sample_labels))
#         ],
#         # fontsize=20, frameon=False,
#         bbox_to_anchor=(1.15, 2),
#         loc="upper right"
#     )

fig_path = os.path.splitext(filename)[0] + ".png"
fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
print("Some test statistics:")
samples = samples.swapaxes(0,1)
print("BFMI (values smaller than 0.3 indicate poor sampling):")
print(f"{az.bfmi(samples)}")
print("rank normalized splitR-hat (Values greater than one indicate that one or more chains have not yet converged):")
print(az.rhat(az.convert_to_dataset(samples)))

hdi_H0 = az.hdi(samples[:,:,0].flatten(), .68) #np.percentile(samples[:,:,0], [15.9,50,84.1])  # 68 percent
H0 = np.median(samples[:,:,0])
obhf = np.median(samples[:,:,1])
hdi_obhf = az.hdi(samples[:,:,1].flatten(), .68)
# H0_mode = mode(samples[:,:,0].flatten())
# obhf_mode = mode(samples[:,:,1].flatten())
print(fr"The uncertainty is ~+-{(hdi_H0[1]-hdi_H0[0])/2:0.2f}")
print(fr"H_0 is {H0}^+{hdi_H0[1]-H0:0.2f}_{hdi_H0[0]-H0:0.2f}. Relative: {(hdi_H0[1]-hdi_H0[0])/2/H0}")
print(fr"Obh2f*100 is {obhf*100}^+{(hdi_obhf[1]-obhf)*100:0.3f}_{(hdi_obhf[0]-obhf)*100:0.3f}. Relative: {(hdi_obhf[1]-hdi_obhf[0])/2/obhf}")
