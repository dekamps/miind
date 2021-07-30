import argparse
import miind.codegen3 as codegen
import sys
import os
import os.path as op
import miind.miind_api as api
import matplotlib.pyplot as plt
import miind.directories3 as directories
import numpy as np
from scipy.integrate import odeint
import math

def generate(func, timestep, timescale, tolerance, basename, threshold_v, reset_v, reset_shift_h, grid_v_min, grid_v_max, grid_h_min, grid_h_max, grid_v_res, grid_h_res,efficacy_orientation='v', threshold_capture_v=0):

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
        mesh_file.write('{}\n'.format(timestep*timescale))

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
        mesh_file.write('{}\n'.format(timestep*timescale))

        progress = 0
        count = 0
        ten_percent = (int)((grid_d1_res) / 10)
        if ten_percent == 0:
            ten_percent = 1

        for i in (np.array(range(grid_d1_res))) * (1.0/(grid_d1_res)):
            svs_1 = [];
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

    # filename = basename + '.rev'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #
    # filename = basename + '.stat'
    # if os.path.exists(filename):
    #     os.remove(filename)

    #filename = basename + '.res'
    #if os.path.exists(filename):
    #    os.remove(filename)

    #filename = basename + '_transform.mesh'
    #if os.path.exists(filename):
    #    os.remove(filename)

    #filename = basename + '_transform.rev'
    #if os.path.exists(filename):
    #    os.remove(filename)

    #filename = basename + '_transform.stat'
    #if os.path.exists(filename):
    #    os.remove(filename)

    #filename = basename + '_transform.model'
    #if os.path.exists(filename):
    #    os.remove(filename)
