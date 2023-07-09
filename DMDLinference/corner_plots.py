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

import mcmc

# Load results from inference on simulated FRB-GW events.
filename = "../Data/simulated_10FRBs_100x50000steps.h5"
sampler = emcee.backends.HDFBackend(filename)

# tau = sampler.get_autocorr_time()
# burnin = int(2 * np.max(tau))

samples = sampler.get_chain(discard=10000)

# Results from only James
filename1 = "../Data/James_prior_100x50000steps.h5"
sampler_J = emcee.backends.HDFBackend(filename1)

tau = sampler_J.get_autocorr_time()
burnin = int(2 * np.max(tau))

samples_J = sampler_J.get_chain(discard=burnin)

# Results without taking James et al. into account.
filename2 = "../Data/simulated_noz_10FRBs_100x50000steps.h5"
sampler_noz = emcee.backends.HDFBackend(filename2)

# tau = sampler_noz.get_autocorr_time()
# burnin = int(2 * np.max(tau))

samples_noz = sampler_noz.get_chain(discard=10000)

# Plot corner plot with all DMs and D_Ls
# labels=(['$H_0$', r'$\Omega_b f_d$']
#         + [r"$D_L$"+f"{str(frb)}" for frb in range(n_FRBs)]
#         + [r"DM$_\mathrm{host}$"+f"{str(frb)}" for frb in range(n_FRBs)]
#         )
# fig = corner.corner(sampler, labels=labels, truths=[H0, Obf, *DL_meas, *DM_host])
labels=(['$H_0$', r'$\Omega_b f_d$'])

plot_kwargs = dict(labels=labels,
                   smooth=1.,
                   levels=(0.68, 0.95),
                   plot_density=False,
                   plot_datapoints=False,
                   fill_contours=True,
                   range=[(10, 150), (0, 0.2)]
                   )
fig = corner.corner(samples_J.swapaxes(0,1),
                    color='r',
                    hist_kwargs={'density' : True},
                    contour_kwargs={'linewidths' : 1.,},
                    contourf_kwargs={'colors' : [(0,0,0,0), (1,0,0,.3), (1,0,0,.6)],},
                    **plot_kwargs,
                    )
fig = corner.corner(samples_noz[:,:,:2].swapaxes(0,1),
                    color='g',
                    fig=fig,
                    hist_kwargs={'density' : True},
                    contour_kwargs={'linewidths' : 1.,},
                    contourf_kwargs={'colors' : [(0,0,0,0), (0,1,0,.3), (0,1,0,.6)],},
                    **plot_kwargs,
                    )
fig = corner.corner(samples[:,:,:2].swapaxes(0,1),
                    color='b',
                    fig=fig,
                    hist_kwargs={'density' : True},
                    contour_kwargs={'linewidths' : 1.,},
                    contourf_kwargs={'colors' : [(0,0,0,0), (0,0,1,.3), (0,0,1,.6)],},
                    **plot_kwargs
                    )

fig_path = os.path.splitext(filename)[0] + ".png"
fig.savefig(fig_path, dpi=500, bbox_inches='tight')
