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

# Load results from inference on simulated FRB-GW events.
filename = "../Data/simulated_10FRBs_z1_24x500steps.h5"
sampler = emcee.backends.HDFBackend(filename)

tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))

samples = sampler.get_chain(discard=burnin)

# Results from only James
filename1 = "../Data/James_prior_100x50000steps.h5"
sampler_J = emcee.backends.HDFBackend(filename1)

tau = sampler_J.get_autocorr_time()
burnin = int(2 * np.max(tau))

samples_J = sampler_J.get_chain(discard=burnin)

# Results without taking James et al. into account.
filename2 = "../Data/simulated_noz_10FRBs_z1_24x500steps.h5"
sampler_noz = emcee.backends.HDFBackend(filename2)

tau = sampler_noz.get_autocorr_time()
burnin = int(2 * np.max(tau))

samples_noz = sampler_noz.get_chain(discard=burnin)

# Plot corner plot with all DMs and D_Ls
# labels=(['$H_0$', r'$\Omega_b f_d$']
#         + [r"$D_L$"+f"{str(frb)}" for frb in range(n_FRBs)]
#         + [r"DM$_\mathrm{host}$"+f"{str(frb)}" for frb in range(n_FRBs)]
#         )
# fig = corner.corner(sampler, labels=labels, truths=[H0, Obf, *DL_meas, *DM_host])

labels=(['$H_0$', r'$\Omega_b f_d$'])

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
                   smooth=1.,
                   levels=(0.68, 0.95),
                   plot_density=False,
                   plot_datapoints=False,
                   fill_contours=True,
                   range=[(10, 150), (0, 0.2)],
                   )
fig = corner.corner(samples_J.swapaxes(0,1),
                    color=cm2[2],
                    hist_kwargs={'density' : True, 'lw' : 2., 'label' : "FRB-z priors"},
                    contour_kwargs={'linewidths' : .5, 'colors' : [cm2[2]]},
                    contourf_kwargs={'colors' : color_set[0],},
                    **plot_kwargs,
                    )
fig = corner.corner(samples_noz[:,:,:2].swapaxes(0,1),
                    color=cm2[1],
                    fig=fig,
                    hist_kwargs={'density' : True, 'lw' : 2., 'label' : "FRB-GW constraints only"},
                    contour_kwargs={'linewidths' : .5, 'colors' : [cm2[1]]},
                    contourf_kwargs={'colors' : color_set[1],},
                    **plot_kwargs,
                    )
fig = corner.corner(samples[:,:,:2].swapaxes(0,1),
                    color=cm2[0],
                    fig=fig,
                    hist_kwargs={'density' : True, 'lw' : 2., 'label' : "Combined constraints"},
                    contour_kwargs={'linewidths' : .5, 'colors' : [cm2[0]]},
                    contourf_kwargs={'colors' : color_set[2],},
                    **plot_kwargs
                    )
# sample_labels = ["Combined constraints",
#                  "FRB-GW constraints only",
#                  "FRB-z priors"
#                  ]
plt.legend(bbox_to_anchor=(1.05, 2), loc="upper right")
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
