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
    




