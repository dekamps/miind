#!/usr/bin/env python3

import numpy as np
import miind.grid_generate as grid_generate

def adEx(y,t):
    C = 281
    g_l = 30
    E_l = -70.6
    v_t = -50.4
    tau = 2.0
    alpha = 4.0
    tau_w = 144.0

    v = y[0];
    w = y[1];

    v_prime = (-g_l*(v - E_l) + g_l*tau*np.exp((v - v_t)/tau) - w) / C
    w_prime = (alpha*(v - E_l) - w) / tau_w

    return [v_prime, w_prime]

grid_generate.generate(adEx, 0.001, 0.001, 1e-12, 'adex', -50.4, -70.6, 80.5, -75, -45, -50, 400, 100, 100)
