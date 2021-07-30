#!/usr/bin/env python3

import numpy as np
import miind.grid_generate as grid_generate

def fn(y,t):
    param_dict={

    'tau_m':   20e-3,
    'E_r': -65e-3,
    'E_e':  0e-3,
    'tau_s':  5e-3,

    'V_min':-66.0e-3,
    'V_max':  -55.0e-3,
    'V_th': -55.0e-3, #'V_max', # sometimes used in other scripts
    'N_V': 200,
    'w_min':0.0,
    'w_max':  0.8,
    'N_w': 20,

    'I': 0, #0.5 to match FN mesh

    }

    v = y[0];
    w = y[1];

    v_prime = v - v**3/3 - w + param_dict['I']
    w_prime = .08*(v + .7 - .8*w)

    return [v_prime, w_prime]

grid_generate.generate(fn, 0.1, 1e-03, 1e-12, 'fn', 2.5, -2.5, 0.0, -2.5, 2.5, -1.0, 1.2, 500, 500)
