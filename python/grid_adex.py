#!/usr/bin/env python3

import numpy as np
import grid_generate

def adEx(y,t):
    C = 200
    g_l = 10
    E_l = -70.0
    v_t = -50.0
    tau = 2.0
    alpha = 2.0
    tau_w = 30.0

    v = y[0];
    w = y[1];

    v_prime = (-g_l*(v - E_l) + g_l*tau*np.exp((v - v_t)/tau) - w) / C
    w_prime = (alpha*(v-E_l) - w) / tau_w

    return [v_prime, w_prime]

grid_generate.generate(adEx, 1, 0.001, 1e-12, 'adex', -10, -58, 0.0, -90, -40, -20, 60, 300, 100)

