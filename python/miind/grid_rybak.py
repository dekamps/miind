#!/usr/bin/env python3

import numpy as np
import miind.grid_generate as grid_generate

def rybak(y, t):
    g_nap = 0.25; #mS
    g_na = 30;
    g_k = 1;
    theta_m = -47.1; #mV
    sig_m = -3.1; #mV
    theta_h = -59; #mV
    sig_h = 8; #mV
    tau_h = 1200; #ms
    E_na = 55; #mV
    E_k = -80;
    C = 1; #uF
    g_l = 0.1; #mS
    E_l = -64.0; #mV
    I = 0.0; #
    I_h = 0; #

    v = y[0];
    h = y[1];

    I_nap = -g_nap * h * (v - E_na) * ((1 + np.exp((-v-47.1)/3.1))**-1);
    I_l = -g_l*(v - E_l);
    I_na = -g_na * (v - E_na) * (((1 + np.exp((v+35)/-7.8))**-1)**3);
    I_k = -g_k * (v - E_k) * (((1 + np.exp((v+28)/-15))**-1)**4);

    v_prime = ((I_nap + I_l + I_na + I_k) / C)+I;
    h_prime = (((1 + (np.exp((v - theta_h)/sig_h)))**(-1)) - h ) / (tau_h/np.cosh((v - theta_h)/(2*sig_h))) + I_h;

    return [v_prime, h_prime]

def rybakInterneuron(y, t):
    E_na = 55; #mV
    E_k = -80;
    C = 1; #uF

    g_na = 120; #mS
    g_k = 100;
    g_l = 0.51; #mS
    E_l = -64.0; #mV

    theta_h = -55; #mV
    sig_h = 7; #mV

    k = 0.79

    I = 0.0; #
    I_h = 0; #

    v = y[0];
    h = y[1];

    I_l = -g_l*(v - E_l);
    I_na = -g_na * (v - E_na) * h * (((1 + np.exp((v+35)/-7.8))**-1)**3);
    I_k = -g_k * (v - E_k) * ((k - (1.06*h))**4);

    v_prime = ((I_l + I_na + I_k) / C)+I;
    h_prime = (((1 + (np.exp((v - theta_h)/sig_h)))**(-1)) - h ) / (30/(np.exp((v+50)/15) + np.exp(-(v+50)/16))) + I_h;

    return [v_prime, h_prime]


#grid_generate.generate(rybakInterneuron, 0.001, 0.001, 1e-3, 'rybak_interneuron', 60.0, -64.0, 0.0, -90, 60, -0.4, 1.0,300, 300)
grid_generate.generate(rybak, 0.1, 0.001, 1e-3, 'rybak_burster', -30, -56, -0.004, -120, -20, -0.4, 1.0 ,400, 300)
