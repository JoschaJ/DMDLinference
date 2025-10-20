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
filename = os.path.join(config.DATA_DIR, "simulated_120FRBs_free_host_vary_z0.2_0.1_eDL0.2_0.1_8x5000steps.h5")
                        #"simulated_120FRBs_free_host_z0.2_0.1_eDL0.2_0.1_8x5000steps_d.h5")
#"simulated_120FRBs_tight_prior_z0.2_0.1_eDL0.2_0.1_23x5000steps_d.h5")  # "simulated_10FRBs_z0.1_eDL0.4_24x5000steps.h5")  #"simulated_110FRBs_z0.2_0.1_eDL0.2_0.1_23x5000steps.h5") #"real_FRB_24x5000steps.h5")  #
sampler = emcee.backends.HDFBackend(filename)

#tau = sampler.get_autocorr_time()
#burnin = int(2 * np.max(tau))

fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ['$H_0$ (km/s/Mpc)', r'$\Omega_b h^2 f_d$', '$\mu_\mathrm{host}$', '$\sigma_\mathrm{host}$']
for i in range(4):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
fig.savefig(os.path.join(config.DATA_DIR, "chains.png"))

samples = sampler.get_chain(discard=500)

# # Results from only James
# filename1 = os.path.join(config.DATA_DIR, "1kFRB_prior_23x30000steps.h5")  # "James_prior_24x5000steps.h5")  #
# sampler_J = emcee.backends.HDFBackend(filename1)

# # tau = sampler_J.get_autocorr_time()
# # burnin = int(2 * np.max(tau))

# samples_J = sampler_J.get_chain(discard=1000)

# Results without taking James et al. into account.
filename2 = os.path.join(config.DATA_DIR, "simulated_noz_120FRBs_free_host_vary_z0.2_0.1_eDL0.2_0.1_8x5000steps.h5") #"simulated_noz_10FRBs_z0.1_eDL0.4_24x5000steps.h5")  #"simulated_noz_110FRBs_z0.2_0.1_eDL0.2_0.1_23x5000steps.h5") #"real_FRB_noz_24x5000steps.h5")  #
sampler_noz = emcee.backends.HDFBackend(filename2)

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

# color_set = [[(0,0,0,0), cm(2, alpha=0.3), cm(2, alpha=0.6)],
#              [(0,0,0,0), cm(1, alpha=0.3), cm(1, alpha=0.6)],
#              [(0,0,0,0), cm(0, alpha=0.3), cm(0, alpha=0.6)]]
alpha = 0.5

color_set = [[(0,0,0,0), list(cm[2]) + [alpha], list(cm2[2]) + [alpha]],
             [(0,0,0,0), list(cm[1]) + [alpha], list(cm2[1]) + [alpha]],
             [(0,0,0,0), list(cm[0]) + [alpha], list(cm2[0]) + [alpha]]]

plot_kwargs = dict(labels=labels,
                   bins=100,
                   hist_bin_factor=0.5,
                   #smooth=.1, #.5,
                   levels=(0.68, 0.95),
                   plot_density=False,
                   plot_datapoints=False,
                   fill_contours=True,
                   max_n_ticks=3,
                   labelpad=0.1,
                   range=[(40, 100), (0, 0.03), (1.5, 2.5), (0.3, 1)], #0.045
                   )
# fig = corner.corner(samples_J.swapaxes(0,1),
#                     color=cm2[2],
#                     hist_kwargs={'density' : True, 'lw' : 2., 'label' : "FRB-z constraints"},  #
#                     contour_kwargs={'linewidths' : .5, 'colors' : [cm2[2]], 'algorithm' : 'threaded'},
#                     contourf_kwargs={'colors' : color_set[0],},
#                     **plot_kwargs,
#                     )
fig = plt.figure(figsize=(5, 5))
fig = corner.corner(samples_noz,  #.swapaxes(0,1)
                    color=cm2[1],
                    fig=fig,
                    hist_kwargs={'density' : True, 'lw' : 2., 'label' : "FRB-GW constraints"},  #
                    contour_kwargs={'linewidths' : .5, 'colors' : [cm2[1]]},
                    contourf_kwargs={'colors' : color_set[1],},
                    smooth=2,
                    **plot_kwargs,
                    )
fig = corner.corner(samples, #swapaxes(0,1)
                    color=cm2[0],
                    fig=fig,
                    truths=[73, 0.02242*0.844, 2, 0.57],
                    hist_kwargs={'density' : True, 'lw' : 2., 'label' : "Combined constraints"},  #
                    contour_kwargs={'linewidths' : .5, 'colors' : [cm2[0]]},
                    contourf_kwargs={'colors' : color_set[2],},
                    smooth=1,
                    **plot_kwargs
                    )
axs = fig.get_axes()
axs[-1].set_xticks([0.4, 0.6, 0.8])
axs[-4].set_yticks([0.4, 0.6, 0.8])
# sample_labels = ["Combined constraints",
#                  "FRB-GW constraints only",
#                  "FRB-z priors"
#                  ]
plt.legend(bbox_to_anchor=(1.05, 4), loc="upper right")
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
