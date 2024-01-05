#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 18:00:57 2023

@author: jjahns
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
from mcmc import average_DM, lum_dist, z_of_DL

c = 299792.458
Obhsqf_fid = .049*0.844
H0_fid = 70

def Obhsqf_lin(H0):
    return Obhsqf_fid*H0/H0_fid

def Obhsqf_no_dep(H0):
    return Obhsqf_fid*np.ones_like(H0)


def Obf_full(H0, DL):
    """Omega_b*f_d when not using the approximation but the full solution."""
    z = z_of_DL(DL, H0)
    z_fid = z_of_DL(DL, H0_fid)
    return Obhsqf_fid * average_DM(z_fid, H0_fid, Obhsqf_fid) / average_DM(z, H0, Obhsqf_fid)


H0 = np.linspace(10, 150)
plt.plot(H0, Obhsqf_lin(H0), label=r"$\propto H_0$", color=sns.color_palette()[1])

zs = [.1, 1, 2, 3]

cmap = sns.color_palette("crest", n_colors=len(zs))
plt.plot(H0, Obhsqf_no_dep(H0), color=cmap[0], label=r"$\propto 1$")

for z, color in zip(zs, cmap):
    DL = c/H0_fid * lum_dist(z)
    plt.plot(H0, Obf_full(H0, DL), color=color, label=f"Full solution for $z={z}$")  # $\langle DM \rangle(\Omega_\mathrm b f_\mathrm d , H_0)$

# DL = c/H0_fid * lum_dist(1)
# plt.plot(H0, Obf_full(H0, DL), label=f"$z=1$")
# DL = c/H0_fid * lum_dist(2)
# plt.plot(H0, Obf_full(H0, DL), label=f"$z=2$")
# DL = c/H0_fid * lum_dist(3)
# plt.plot(H0, Obf_full(H0, DL), label=f"$z=3$")

plt.xlim(60, 80)
plt.ylim(.02, .06)

plt.xlabel("$H_0$")
plt.ylabel(r"$\Omega_\mathrm{b}h^2f_\mathrm{d}$")
plt.legend()

plt.savefig(os.path.join(config.DATA_DIR, "Obhsqf_vs_H0.pdf"), bbox_inches='tight', pad_inches=0.01, dpi=300)