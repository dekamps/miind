#!/usr/bin/env python3

import numpy as np
import miind.grid_generate as grid_generate

def cond(y,t):
    param_dict={

    'tau_m':   20e-3,
    'E_r': -65e-3,
    'E_e':  0e-3,
    'tau_s':  5e-3,
    'g_max': 0.8,
    'V_min':-66.0e-3,
    'V_max':  -55.0e-3,
    'V_th': -55.0e-3, #'V_max', # sometimes used in other scripts
    'N_V': 200,
    'w_min':0.0,
    'w_max':  10.0,
    'N_w': 20,

    }

    v = y[0];
    w = y[1];

    v_prime = (-(v-param_dict['E_r'])  - w*(v-param_dict['E_e']))/param_dict['tau_m']
    w_prime = -w/param_dict['tau_s']

    return [v_prime, w_prime]


grid_generate.generate(cond, 1e-05, 1, 1e-12, 'cond', -55.0e-3, -65e-3, 0.0, -67.0e-3, -54.0e-3, -0.2, 1.0, 100, 100, efficacy_orientation='w')
