#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:58:41 2023

@author: jjahns
"""
import numpy as np
import emcee
import corner

import mcmc


tau = sampler.get_autocorr_time()
print(tau)

# Plot corner plot with all DMs and D_Ls
labels=(['$H_0$', r'$\Omega_b f_d$']
        + [r"$D_L$"+f"{str(frb)}" for frb in range(n_FRBs)]
        + [r"DM$_\mathrm{host}$"+f"{str(frb)}" for frb in range(n_FRBs)]
        )
# fig = corner.corner(sampler, labels=labels, truths=[H0, Obf, *DL_meas, *DM_host])
labels=(['$H_0$', r'$\Omega_b f_d$'])
fig = corner.corner(sampler_J, labels=labels, smooth=0.5,
                  levels=(0.68, 0.95),
                  color='r',
                  hist_kwargs={'density' : True},
                  plot_density=False,
                  plot_datapoints=False,
                  fill_contours=True,
                  contour_kwargs={'linewidths' : 1.,},
                  contourf_kwargs={'colors' : ['w',(1,0,0,.3), (1,0,0,.6)],},
                  )
fig = corner.corner(sampler_noz.chain[:,:,:2], labels=labels, smooth=0.5,
                    levels=(0.68, 0.95),
                    color='g',
                    fig=fig,
                    hist_kwargs={'density' : True},
                    plot_density=False,
                    plot_datapoints=False,
                    fill_contours=True,
                    contour_kwargs={'linewidths' : 1.,},
                    contourf_kwargs={'colors' : [(0,0,0,0),(0,1,0,.3), (0,1,0,.6)],},
                    )
fig = corner.corner(sampler.chain[:,:,:2], labels=labels, smooth=0.5,
                    levels=(0.68, 0.95),
                    color='b',
                    fig=fig,
                    hist_kwargs={'density' : True},
                    plot_density=False,
                    plot_datapoints=False,
                    fill_contours=True,
                    contour_kwargs={'linewidths' : 1.,},
                    contourf_kwargs={'colors' : [(0,0,0,0),(0,0,1,.3), (0,0,1,.6)],},
                    )
fig.savefig(f"Figures/{n_FRBs}FRBs_{nwalkers}x{nsteps}steps.png", dpi=500, bbox_inches='tight')