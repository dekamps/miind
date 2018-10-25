#!/usr/bin/env python

import argparse
import codegen
import sys
import os
import os.path as op
import directories
import jobs
import miind_api as api
import matplotlib.pyplot as plt
import directories
import numpy as np
from scipy.integrate import odeint
import math

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
    I_na = -g_na * 0.7243 * (v - E_na) * (((1 + np.exp((v+35)/-7.8))**-1)**3);
    I_k = -g_k * (v - E_k) * (((1 + np.exp((v+28)/-15))**-1)**4);

    v_prime = ((I_nap + I_l + I_na + I_k) / C)+I;
    h_prime = (((1 + (np.exp((v - theta_h)/sig_h)))**(-1)) - h ) / (tau_h/np.cosh((v - theta_h)/(2*sig_h))) + I_h;

    return [v_prime, h_prime]

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

def generate(func, timestep, basename, threshold_v, reset_v, reset_shift_h, grid_v_min, grid_v_max, grid_h_min, grid_h_max, grid_v_res, grid_h_res,efficacy_orientation='v'):

    with open(basename + '.rev', 'w') as rev_file:
        rev_file.write('<Mapping Type="Reversal">\n')
        rev_file.write('</Mapping>\n')

    with open(basename + '.stat', 'w') as stat_file:
        stat_file.write('<Stationary>\n')
        stat_file.write('</Stationary>\n')

    with open(basename + '.mesh', 'w') as mesh_file:
        mesh_file.write('ignore\n')
        mesh_file.write('{}\n'.format(timestep/1000.0))

        for i in np.linspace(0,1.0,grid_v_res):
            svs_1 = [];
            sus_1 = [];
            svs_2 = [];
            sus_2 = [];

            for j in np.linspace(0,1.0,grid_h_res):
                x1 = (i*(grid_v_max-grid_v_min))+grid_v_min
                y1 = (j*(grid_h_max-grid_h_min))+grid_h_min

                if (efficacy_orientation == 'v'):
                    svs_1.append(x1)
                    sus_1.append(y1)
                    svs_2.append(x1 + ((1.0/grid_v_res)*(grid_v_max-grid_v_min)))
                    sus_2.append(y1)
                else:
                    sus_1.append(x1)
                    svs_1.append(y1)
                    sus_2.append(x1 + ((1.0/grid_v_res)*(grid_v_max-grid_v_min)))
                    svs_2.append(y1)

            for s in svs_1:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            for s in sus_1:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            for s in svs_2:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            for s in sus_2:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            mesh_file.write('closed\n')
        mesh_file.write('end\n')

    with open(basename + '_transform.rev', 'w') as rev_file:
        rev_file.write('<Mapping Type="Reversal">\n')
        rev_file.write('</Mapping>\n')

    with open(basename + '_transform.stat', 'w') as stat_file:
        stat_file.write('<Stationary>\n')
        stat_file.write('</Stationary>\n')

    with open(basename + '_transform.mesh', 'w') as mesh_file:
        mesh_file.write('ignore\n')
        mesh_file.write('{}\n'.format(timestep/1000.0))

        progress = 0
        count = 0
        ten_percent = (int)(grid_v_res / 10)

        for i in np.linspace(0,1.0,grid_v_res):
            svs_1 = [];
            sus_1 = [];
            svs_2 = [];
            sus_2 = [];

            count = count + 1
            if (count % ten_percent == 0):
                print('{} % complete.'.format(progress))
                progress += 10

            for j in np.linspace(0,1.0,grid_h_res):
                x1 = (i*(grid_v_max-grid_v_min))+grid_v_min
                y1 = (j*(grid_h_max-grid_h_min))+grid_h_min

                tspan = np.linspace(0, timestep,2)

                t_1 = odeint(func, [x1,y1], tspan, atol=1e-3, rtol=1e-3)
                t_2 = odeint(func, [x1 + ((1.0/grid_v_res)*(grid_v_max-grid_v_min)),y1], tspan, atol=1e-3, rtol=1e-3)

                t_x1 = t_1[1][0]
                t_y1 = t_1[1][1]
                t_x2 = t_2[1][0]
                t_y2 = t_2[1][1]

                if (math.isnan(t_x1) or math.isnan(t_y1)):
                    t_x1 = x1 + (grid_v_max-grid_v_min)
                    t_y1 = y1

                if (math.isnan(t_x2) or math.isnan(t_y2)):
                    t_x2 = x1 + (grid_v_max-grid_v_min) + 1
                    t_y2 = y1

                if (abs(t_x1 - t_x2) < 0.00001):
                    t_x2 = t_x1 + 0.00001

                if (abs(t_y1 - t_y2) < 0.00001):
                    t_y2 - t_y1 + 0.00001

                if (efficacy_orientation == 'v'):
                    svs_1.append(t_x1)
                    sus_1.append(t_y1)
                    svs_2.append(t_x2)
                    sus_2.append(t_y2)
                else:
                    sus_1.append(t_x1)
                    svs_1.append(t_y1)
                    sus_2.append(t_x2)
                    svs_2.append(t_y2)

            for s in svs_1:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            for s in sus_1:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            for s in svs_2:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            for s in sus_2:
                mesh_file.write('{}\t'.format(s))
            mesh_file.write('\n')
            mesh_file.write('closed\n')
        mesh_file.write('end\n')

    api.MeshTools.buildModelFileFromMesh(basename, reset_v, threshold_v)
    api.MeshTools.buildModelFileFromMesh(basename + '_transform', reset_v, threshold_v)

    mod_file = open(basename + '.model', 'r')
    lines = mod_file.readlines()
    mod_file.close()

    mod_file = open(basename + '.model', 'w')
    for l in range(len(lines)):
        if l != 3:
            mod_file.write(lines[l])
    mod_file.close()

    mod_file = open(basename + '_transform.model', 'r')
    lines = mod_file.readlines()
    mod_file.close()

    mod_file = open(basename + '_transform.model', 'w')
    for l in range(len(lines)):
        if l != 3:
            mod_file.write(lines[l])
    mod_file.close()

    api.MeshTools.buildTransformFileFromModel(basename, 100000)
    api.MeshTools.buildTransformFileFromModel(basename, reset_shift_w=reset_shift_h, mode='resettransform')

    # filename = basename + '.mesh'
    # if os.path.exists(filename):
    #     os.remove(filename)

    filename = basename + '.rev'
    if os.path.exists(filename):
        os.remove(filename)

    filename = basename + '.stat'
    if os.path.exists(filename):
        os.remove(filename)

    filename = basename + '.res'
    if os.path.exists(filename):
        os.remove(filename)

    # filename = basename + '_transform.mesh'
    # if os.path.exists(filename):
    #     os.remove(filename)

    filename = basename + '_transform.rev'
    if os.path.exists(filename):
        os.remove(filename)

    filename = basename + '_transform.stat'
    if os.path.exists(filename):
        os.remove(filename)

    filename = basename + '_transform.model'
    if os.path.exists(filename):
        os.remove(filename)

generate(rybak, 1, 'grid', -10, -56, -0.004, -80, -40, -0.4, 1.0, 300, 200)
#generate(adEx, 1, 'adex', -10, -58, 0.0, -90, -40, -20, 60, 300, 100)
