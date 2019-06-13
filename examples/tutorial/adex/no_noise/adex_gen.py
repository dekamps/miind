#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from numpy.linalg import norm

def adEx(y,t):
    # Defaults from Brette & Gerstner
    C = 281
    g_l = 30
    E_l = -70.6
    v_t = -50.4
    tau = 2.0
    alpha = 4.0
    tau_w = 144.0

    v = y[0];
    w = y[1];

    v_prime = (-g_l*(v - E_l) + g_l*tau*np.exp((v - v_t)/tau) - w) / (C)
    w_prime = (alpha*(v - E_l) - w) / (tau_w)

    return [v_prime, w_prime]

def adExBack(y,t):
    [v,w] = adEx(y,t)
    return [-v, -w]

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def isPolyConvex(ps):
    return np.abs(np.arccos(np.dot((ps[1]-ps[0])/norm(ps[1]-ps[0]),(ps[2]-ps[0])/norm(ps[2]-ps[0])))) < np.abs(np.arccos(np.dot((ps[1]-ps[0])/norm(ps[1]-ps[0]),(ps[3]-ps[0])/norm(ps[3]-ps[0])))) and np.abs(np.arccos(np.dot((ps[2]-ps[1])/norm(ps[2]-ps[1]),(ps[3]-ps[1])/norm(ps[3]-ps[1])))) < np.abs(np.arccos(np.dot((ps[2]-ps[1])/norm(ps[2]-ps[1]),(ps[0]-ps[1])/norm(ps[0]-ps[1]))))

def writeStrip(meshfile, strip_v_1, strip_w_1, strip_v_2, strip_w_2):
    for v in strip_v_1:
        meshfile.write(str(v) + '\t')
    meshfile.write('\n')
    for w in strip_w_1:
        meshfile.write(str(w) + '\t')
    meshfile.write('\n')
    for v in strip_v_2:
        meshfile.write(str(v) + '\t')
    meshfile.write('\n')
    for w in strip_w_2:
        meshfile.write(str(w) + '\t')
    meshfile.write('\n')
    meshfile.write('closed\n')

def generateStat(v_min, v_max, w_min, w_max):
    # This method should be generalised for multiple stationary cells
    with open('adex.stat','w') as statfile:
        statfile.write('<Stationary>\n')
        format = "%.9f"
        # Additional stationary cell for placing mass at a desired location
        statfile.write('<Quadrilateral>\n')
        statfile.write('<vline>' +  str(-50.1) + ' ' + str(-50.1) + ' ' +  str(-50.0) + ' ' + str(-50.0) + '</vline>\n')
        statfile.write('<wline>' +  str(100.0) + ' ' + str(100.1) + ' ' +  str(100.1) + ' ' + str(100.0) + '</wline>\n')
        statfile.write('</Quadrilateral>\n')
        # Actual stationary cell covering the stationary point
        statfile.write('<Quadrilateral>\n')
        statfile.write('<vline>' +  str(v_min) + ' ' + str(v_min) + ' ' +  str(v_max) + ' ' + str(v_max) + '</vline>\n')
        statfile.write('<wline>' +  str(w_min) + ' ' + str(w_max) + ' ' +  str(w_max) + ' ' + str(w_min) + '</wline>\n')
        statfile.write('</Quadrilateral>\n')
        statfile.write('</Stationary>')

def generateRev(revfile, stat_cell, strip_num, cell_num=0, fraction=1.0):
    revfile.write(str(strip_num) + ',' + str(cell_num))
    revfile.write('\t')
    revfile.write(str(stat_cell[0]) + ',' + str(stat_cell[1]))
    revfile.write('\t')
    revfile.write(str(fraction) + '\n')

def buildBlock(meshfile, revfile, strip_num, start_points, tspan, func, max_v, min_v, max_w, min_w, ax=None, reverse=False, include_reversal=True):
    lines = []
    for strip in range(len(start_points)-1):
        strip_v_1 = []
        strip_w_1 = []
        strip_v_2 = []
        strip_w_2 = []

        t_1 = odeint(func, start_points[strip], tspan)
        t_2 = odeint(func, start_points[strip+1], tspan)

        t_1 = np.array([t for t in t_1[:] if t[0] < max_v and t[1] < max_w and t[0] > min_v and t[1] > min_w])
        t_2 = np.array([t for t in t_2[:] if t[0] < max_v and t[1] < max_w and t[0] > min_v and t[1] > min_w])

        num_cells = 0

        for cell in range(min(len(t_1[0:]), len(t_2[0:]))-1):
            p1 = [t_1[cell][0],t_1[cell][1]]
            p2 = [t_1[cell+1][0],t_1[cell+1][1]]
            p3 = [t_2[cell][0],t_2[cell][1]]
            p4 = [t_2[cell+1][0],t_2[cell+1][1]]

            if PolyArea([p1[0],p2[0],p3[0],p4[0]], [p1[1],p2[1],p3[1],p4[1]]) < 0.000001:
                break

            if not isPolyConvex(np.array([p1,p2,p4,p3])):
                break

            strip_v_1 = strip_v_1 + [p1[0]]
            strip_w_1 = strip_w_1 + [p1[1]]
            strip_v_2 = strip_v_2 + [p3[0]]
            strip_w_2 = strip_w_2 + [p3[1]]

            lines = lines + [[p1,p3],[p4,p2]]
            num_cells = num_cells + 1

        # perhaps this strip had cells that were so small or abnormal, we
        # excluded all (most) cells - ignore this strip.
        if num_cells <= 2:
            continue;

        if reverse:
            strip_v_1.reverse()
            strip_w_1.reverse()
            strip_v_2.reverse()
            strip_w_2.reverse()

        writeStrip(meshfile, strip_v_1, strip_w_1, strip_v_2, strip_w_2)

        if include_reversal:
            generateRev(revfile, [0,1], strip_num)
        strip_num = strip_num + 1

        if ax:
            ax.plot(t_1[:,0], t_1[:,1], color='k')
            ax.plot(t_2[:,0], t_2[:,1], color='k')

    if ax:
        lc = mc.LineCollection(lines, linewidths=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)

    return strip_num

def generateMesh():

    # Adex model expects threshold = -43mV
    # Recommended reset = -60mV
    # in miindio.py
    # generate-model adex -60 -43
    # generate-empty-fid adex
    # generate-matrix adex 1 1000 0.0 100.0 True

    generateStat(-70.7, -70.5, -4.1 ,-3.9)

    with open('adex.mesh','w') as meshfile:
        with open('adex.rev','w') as revfile:
            strip_num = 1
            revfile.write('<Mapping type=\"Reversal\">\n')

            meshfile.write('ignore\n')
            timestep = 0.2 # ms
            meshfile.write('{}\n'.format(timestep/1000))

            fig, ax = plt.subplots()

            # Left
            start_v = -80
            start_range_w = range(-300,500,10)
            start_points = [[start_v,w] for w in start_range_w]
            tspan = np.linspace(0, 80, 80/timestep)

            strip_num = buildBlock(meshfile, revfile, strip_num, start_points, tspan, adEx, -30.0, -80.01, 300, -800, ax)
            start_left_strip = strip_num

            ############################

            # Left Sub-threshold

            start_v = -50
            start_range_w = range(-300,500,10)
            start_points = [[start_v,w] for w in start_range_w]
            tspan = np.linspace(0, 50, 50/timestep)

            strip_num = buildBlock(meshfile, revfile, strip_num, start_points, tspan, adEx, -30.0, -80.01, 300, -800, ax)
            left_strip_num = strip_num
            ####################################################

            # Right Sub-threshold
            start_v = -50
            start_range_w = range(-300,500,10)
            start_points = [[start_v,w] for w in start_range_w]
            tspan = np.linspace(0, 10, 10/timestep)

            strip_num = buildBlock(meshfile, revfile, strip_num, start_points, tspan, adExBack, -30.0, -80.01, 300, -800, ax, True,False)

            right_strip_num = strip_num

            for r in range(right_strip_num - left_strip_num):
                revfile.write(str(left_strip_num+r) + ',' + str(0))
                revfile.write('\t')
                revfile.write(str(start_left_strip+r) + ',' + str(1))
                revfile.write('\t')
                revfile.write(str(1.0) + '\n')

            ####################################################

            # Super-threshold

            start_v = -40
            start_range_w = range(-300,500,10)
            start_points = [[start_v,w] for w in start_range_w]
            tspan = np.linspace(0, 5, 5/timestep)

            strip_num = buildBlock(meshfile, revfile, strip_num, start_points, tspan, adExBack, -30.0, -80.01, 300, -800, ax, True, False)

            ################################################################

            meshfile.write('end')
            revfile.write('</Mapping>')

            plt.show()

generateMesh()
