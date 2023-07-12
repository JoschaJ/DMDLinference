#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 18:00:57 2023

@author: jjahns
"""
import numpy as np
import matplotlib.pyplot as plt

from mcmc import average_DM, lum_dist, z_of_DL

c = 299792.458
Obf_fid = .049*0.844
H0_fid = 70

def Obf_lin(H0):
    return Obf_fid*H0_fid/H0

def Obf_sq(H0):
    return Obf_fid*(H0_fid/H0)**2


def Obf_full(H0, DL):
    """Omega_b*f_d when not using the approximation but the full solution."""
    z = z_of_DL(DL, H0)
    z_fid = z_of_DL(DL, H0_fid)
    return Obf_fid * average_DM(z_fid, H0_fid, Obf_fid) / average_DM(z, H0, Obf_fid)

DL = c/H0_fid * lum_dist(0.1)

H0 = np.linspace(10, 150)
plt.plot(H0, Obf_lin(H0), label="Degeneracy")
plt.plot(H0, Obf_sq(H0))
plt.plot(H0, Obf_full(H0, DL))

DL = c/H0_fid * lum_dist(1)
plt.plot(H0, Obf_full(H0, DL))
DL = c/H0_fid * lum_dist(2)
plt.plot(H0, Obf_full(H0, DL))
DL = c/H0_fid * lum_dist(3)
plt.plot(H0, Obf_full(H0, DL))

plt.xlim(60, 80)

plt.xlabel("$H_0$")
plt.ylabel(r"$\Omega_\mathrm{b}f_\mathrm{d}$")