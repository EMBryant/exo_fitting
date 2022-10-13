#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:27:22 2022

@author: eb
"""

import numpy as np
import matplotlib.pyplot as plt
import radvel
import pandas as pd
import sys
from math import pi
import batman as bm

def load_rv_data(file_name, file_format='txt', unit='m/s'):
    if file_format=='txt':
        dat = np.loadtxt(file_name)
    elif file_format=='csv':
        dat = np.loadtxt(file_name, delimiter=',')
    else:
        print('I can only handle txt or csv files atm')
        print('Sorry.')
        sys.exit(1)
    
    t = dat[:, 0]
    rv = dat[:, 1]
    erv = dat[:, 2]
    if unit == 'km/s':
        rv *= 1000.
        erv *= 1000.
    
    return t, rv, erv
    

def initialise_batman_lc_model(x, tc, per, rprs, ars, inc, 
                               ecc=0., omega=90., q1=0.25, q2=0.35,
                               os = None, exptime = None):
    pm = bm.TransitParams()
    
    pm.t0 = tc
    pm.per = per
    pm.rp = rprs
    pm.a = ars
    pm.inc = inc
    
    pm.ecc = ecc
    pm.w = omega
    
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)

    pm.u = [u1, u2]
    pm.limb_dark='quadratic'
    
    pm.fp = 0.
    pm.t_secondary = pm.t0 + pm.per/2.
    
    if os:
        m = bm.TransitModel(pm, np.array(x, dtype=float),
                            supersample_factor=os, exp_time=exptime)
    else:
        m = bm.TransitModel(pm, np.array(x, dtype=float))
    
    return m, pm

def get_ld_coeffs_kipping13(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    
    return u1, u2

def calculate_batman_lc(bm_mod, bm_params,
                        tc, per, rprs, ars, inc, u1=0.35, u2=0.15,
                        ecc = None, omega=None):
    bm_params.t0  = tc
    bm_params.per = per
    bm_params.rp  = rprs
    bm_params.a   = ars
    bm_params.inc = inc
    
    bm_params.u = [u1, u2]
    
    if not ecc is None:
        bm_params.ecc = ecc
        bm_params.w = omega
    
    return bm_mod.light_curve(bm_params)
    
def calculate_radvel_keplerian(x, tc, per, k, ecc=0., omega=pi/2.):
    tperi = radvel.orbit.timetrans_to_timeperi(tc, per, ecc, omega)
    
    return radvel.kepler.rv_drive(x, np.array([per, tperi, ecc, omega, k]))


def lnprior(theta, fit_ecc=False):
    tc, per = theta[0], theta[1]
    rprs, ars, inc = theta[2], theta[3], theta[4]
    q1, q2 = theta[5], theta[6]
    
    tcmin, tcmax = tc - 0.05 * per, tc + 0.05 * per
    permin, permax = per * 0.95, per * 1.05
    
    if not tcmin <= tc <= tcmax:
        return -np.inf
    if not permin <= per <= permax:
        return -np.inf
    if not 0. <= rprs <= 1.:
        return -np.inf
    if not 1.1 <= ars:
        return -np.inf
    if not 0. <= inc <= 90.:
        return -np.inf
    if not 0. <= q1 <= 1.:
        return -np.inf
    if not 0. <= q2 <= 1.:
        return -np.inf
    
    if fit_ecc:
        ecc = theta[7]
        omega = theta[8]
        if not 0. <= ecc <= 0.9:
            return -np.inf
        if not -180. <= omega <= 180.:
            return -np.inf
    return 0.

def lnlike(theta, bm_model, bm_lcparams,
           bjd_lc, flux_obs, e_flux_obs,
           bjd_rv, rv_obs, e_rv_obs,
           fit_ecc=False):
    tc, per = theta[0], theta[1]
    rprs, ars, inc = theta[2], theta[3], theta[4]
    q1, q2 = theta[5], theta[6]
    f0 = theta[7]
    
    k, gamma = theta[8], theta[9]
    
    u1, u2 = get_ld_coeffs_kipping13(q1, q2)
    
    if fit_ecc:
        ecc, omega = theta[10], theta[11]
        lcmod = calculate_batman_lc(bm_model, bm_lcparams,
                                    tc, per, rprs, ars, inc, u1=u1, u2=u2,
                                    ecc = ecc, omega=omega) + f0
        
        rvmod = calculate_radvel_keplerian(bjd_rv, tc, per, k, 
                                            ecc=ecc, omega=omega * pi / 180.) + gamma
    
    else:
        lcmod = calculate_batman_lc(bm_model, bm_lcparams,
                                    tc, per, rprs, ars, inc, u1=u1, u2=u2) + f0
        
        rvmod = calculate_radvel_keplerian(bjd_rv, tc, per, k) + gamma
        
    lnl_trans = -0.5 * np.sum((flux_obs - lcmod)**2 / e_flux_obs**2 
                               + np.log10(2.0 * pi * e_flux_obs**2)
                             )
    lnl_rv    = -0.5 * np.sum((rv_obs - rvmod)**2 / e_rv_obs**2 
                                + np.log10(2.0 * pi * e_rv_obs**2)
                             )
    
    return lnl_trans + lnl_rv
    

def lnprob(theta, bm_model, bm_lcparams,
           bjd_lc, flux_obs, e_flux_obs,
           bjd_rv, rv_obs, e_rv_obs,
           fit_ecc=False):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    lnl = lnlike(theta, bm_model, bm_lcparams,
           bjd_lc, flux_obs, e_flux_obs,
           bjd_rv, rv_obs, e_rv_obs,
           fit_ecc=False)
    
    return lp + lnl
        

       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    