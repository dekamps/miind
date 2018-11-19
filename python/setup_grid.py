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

    'I': 0.5,

    }

    v = y[0];
    w = y[1];

    v_prime = v - v**3/3 - w + param_dict['I']
    w_prime = .08*(v + .7 - .8*w)

    return [v_prime, w_prime]

def generate(func, timestep, timestep_multiplier, tolerance, basename, threshold_v, reset_v, reset_shift_h, grid_v_min, grid_v_max, grid_h_min, grid_h_max, grid_v_res, grid_h_res,efficacy_orientation='v', threshold_capture_v=0):

    grid_d1_res = grid_v_res;
    grid_d1_min = grid_v_min;
    grid_d1_max = grid_v_max;

    grid_d2_res = grid_h_res;
    grid_d2_min = grid_h_min;
    grid_d2_max = grid_h_max;

    if (efficacy_orientation == 'v'):
        grid_d1_res = grid_h_res;
        grid_d1_min = grid_h_min;
        grid_d1_max = grid_h_max;

        grid_d2_res = grid_v_res;
        grid_d2_min = grid_v_min;
        grid_d2_max = grid_v_max;

    with open(basename + '.rev', 'w') as rev_file:
        rev_file.write('<Mapping Type="Reversal">\n')
        rev_file.write('</Mapping>\n')

    with open(basename + '.stat', 'w') as stat_file:
        stat_file.write('<Stationary>\n')
        stat_file.write('</Stationary>\n')

    with open(basename + '.mesh', 'w') as mesh_file:
        mesh_file.write('ignore\n')
        mesh_file.write('{}\n'.format(timestep*timestep_multiplier))

        for i in (np.array(range(grid_d1_res))) * (1.0/(grid_d1_res)):
            svs_1 = [];
            sus_1 = [];
            svs_2 = [];
            sus_2 = [];

            for j in (np.array(range(grid_d2_res))) * (1.0/(grid_d2_res)):
                if (efficacy_orientation != 'v'):
                    x1 = (i*(grid_d1_max-grid_d1_min))+grid_d1_min
                    y1 = (j*(grid_d2_max-grid_d2_min))+grid_d2_min

                    svs_1.append(x1)
                    sus_1.append(y1)
                    svs_2.append(x1 + ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)))
                    sus_2.append(y1)

                else:
                    y1 = (i*(grid_d1_max-grid_d1_min))+grid_d1_min
                    x1 = (j*(grid_d2_max-grid_d2_min))+grid_d2_min

                    svs_1.append(x1)
                    sus_1.append(y1)
                    svs_2.append(x1)
                    sus_2.append(y1+ ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)))

            if (threshold_capture_v > 0):
                if (efficacy_orientation != 'v'):
                    svs_1.append(x1)
                    sus_1.append(threshold_capture_v)
                    svs_2.append(x1 + ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)))
                    sus_2.append(threshold_capture_v)
                else:
                    svs_1.append(threshold_capture_v)
                    sus_1.append(y1)
                    svs_2.append(threshold_capture_v)
                    sus_2.append(y1+ ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)))
            else:
                if (efficacy_orientation != 'v'):
                    x1 = (i*(grid_d1_max-grid_d1_min))+grid_d1_min
                    y1 = ((j+(1.0/(grid_d2_res)))*(grid_d2_max-grid_d2_min))+grid_d2_min
                    svs_1.append(x1)
                    sus_1.append(y1)
                    svs_2.append(x1 + ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)))
                    sus_2.append(y1)
                else:
                    y1 = (i*(grid_d1_max-grid_d1_min))+grid_d1_min
                    x1 = ((j+(1.0/(grid_d2_res)))*(grid_d2_max-grid_d2_min))+grid_d2_min
                    svs_1.append(x1)
                    sus_1.append(y1)
                    svs_2.append(x1)
                    sus_2.append(y1+ ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)))


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
        mesh_file.write('{}\n'.format(timestep*timestep_multiplier))

        progress = 0
        count = 0
        ten_percent = (int)((grid_d1_res) / 10)

        for i in (np.array(range(grid_d1_res))) * (1.0/(grid_d1_res)):
            svs_1 = [];((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min))
            sus_1 = [];
            svs_2 = [];
            sus_2 = [];

            count = count + 1
            if (count % ten_percent == 0):
                print('{} % complete.'.format(progress))
                progress += 10

            for j in (np.array(range(grid_d2_res+1))) * (1.0/(grid_d2_res)):

                if (efficacy_orientation != 'v'):

                    x1 = (i*(grid_d1_max-grid_d1_min))+grid_d1_min
                    y1 = (j*(grid_d2_max-grid_d2_min))+grid_d2_min

                    tspan = np.linspace(0, timestep, 11)

                    t_1 = odeint(func, [x1,y1], tspan, atol=tolerance, rtol=tolerance)
                    t_2 = odeint(func, [x1 + ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min)),y1], tspan, atol=tolerance, rtol=tolerance)

                else:
                    y1 = (i*(grid_d1_max-grid_d1_min))+grid_d1_min
                    x1 = (j*(grid_d2_max-grid_d2_min))+grid_d2_min

                    tspan = np.linspace(0, timestep,11)

                    t_1 = odeint(func, [x1,y1], tspan, atol=tolerance, rtol=tolerance)
                    t_2 = odeint(func, [x1,y1+ ((1.0/grid_d1_res)*(grid_d1_max-grid_d1_min))], tspan, atol=tolerance, rtol=tolerance)

                t_x1 = t_1[-1][0]
                t_y1 = t_1[-1][1]
                t_x2 = t_2[-1][0]
                t_y2 = t_2[-1][1]

                svs_1.append(t_x1)
                sus_1.append(t_y1)
                svs_2.append(t_x2)
                sus_2.append(t_y2)

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

    api.MeshTools.buildTransformFileFromModel(basename, 1000000000)
    api.MeshTools.buildTransformFileFromModel(basename, reset_shift_w=reset_shift_h, mode='resettransform')

    # filename = basename + '.mesh'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '.rev'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '.stat'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '.res'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '_transform.mesh'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '_transform.rev'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '_transform.stat'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '_transform.model'
    # if os.path.exists(filename):
    #     os.remove(filename)

#generate(rybak, 1, 0.001, 1e-3, 'grid', -10, -56, -0.004, -110, 0, -0.4, 1.0,300, 300)
# generate(adEx, 1, 0.001, 1e-12, 'adex', -10, -58, 0.0, -90, -40, -20, 60, 300, 100)
#generate(cond, 1e-05, 1, 1e-12, 'cond', -55.0e-3, -65e-3, 0.0, -67.0e-3, -54.0e-3, -0.2, 1.0, 100, 100, efficacy_orientation='w')
generate(fn, 1e-03, 1, 1e-12, 'fn', 3.0, -3.0, 0.0, -3.0, 3.0, -1.0, 3.0, 500, 500)
