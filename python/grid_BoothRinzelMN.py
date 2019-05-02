#!/usr/bin/env python3

import numpy as np
import grid_generate

def BoothRinzelMNSoma(y, t):
    V_na = 1;
    V_k = -0.7;

    g_na = 1.0;
    g_kdr = 2.0;
    g_l = 0.5;
    V_l = -0.5;

    v_1 = -0.01;
    v_2 = 0.15;
    v_3 = -0.04;
    v_4 = 0.1;

    phi_s = 0.2;

    tau_v = 0.001;
    tau_w = 0.001;

    v = y[0];
    w = y[1];

    I_l = -g_l * (v - V_l);
    I_na = -g_na * (v - V_na) * ((1/2)*(1 + (np.tanh((v - v_1)/v_2))));
    I_k = -g_kdr * w * (v - V_k);

    w_inf = ((1/2)*(1 + (np.tanh((v - v_3)/v_4))));
    tau_s = (1 / np.cosh((v - v_3)/(2*v_4)));

    v_prime = (I_l + I_na + I_k)/tau_v;
    w_prime = (phi_s * (w_inf - w) / tau_s)/tau_w;

    return [v_prime, w_prime]

def BoothRinzelMNDendrite(y, t):
    V_ca = 1;
    V_k = -0.7;

    g_ca = 1.5;
    g_k = 0.5;
    g_l = 0.5;
    V_l = -0.5;

    v_1 = 0.05;
    v_2 = 0.15;
    v_3 = 0;
    v_4 = 0.1;

    phi_d = 0.2;

    v = y[0];
    w = y[1];

    tau_v = 0.001;
    tau_w = 0.001;

    I_l = -g_l * (v - V_l);
    I_ca = -g_ca * (v - V_ca) * ((1/2)*(1 + (np.tanh((v - v_1)/v_2))));
    I_k = -g_k * w * (v - V_k);

    w_inf = ((1/2)*(1 + (np.tanh((v - v_3)/v_4))));
    tau_s = (1 / np.cosh((v - v_3)/(2*v_4)));

    v_prime = (I_l + I_ca + I_k)/tau_v;
    w_prime = (phi_d * (w_inf - w) / tau_s)/tau_w;

    return [v_prime, w_prime]


grid_generate.generate(BoothRinzelMNSoma, 0.0001, 1, 1e-8, 'soma', 0.98, -1.48, 0.0, -1.5, 1.0, -0.2, 1.2, 300, 300)
grid_generate.generate(BoothRinzelMNDendrite, 0.0001, 1, 1e-8, 'dendrite', 0.98, -1.48, 0.0, -1.5, 1.0, -0.2, 1.2, 300, 300)
