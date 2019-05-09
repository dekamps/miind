import cond_MC
import os
import utilities as ut
import matplotlib.pyplot as plt
import numpy as np
import subprocess as  sp

RATE_RANGE = np.arange(0,3100,100)


event_dir = 'mc_events'
T_SIM = 2.0

def generate_events(h=0.05, g_max = 10.0):
    if not os.path.exists(event_dir):
        os.makedirs(event_dir)
    sp.call(['cp','cond_MC.py',event_dir])
    with ut.cd(event_dir):
        for rate in RATE_RANGE:
            cond_MC.simulation(rate, T_SIM, g_max, h)

def gain_curve(h, g_max):
    n_avg = 10  # ms
    ins   = []
    outs  = []
    with ut.cd(event_dir):
        for rate in RATE_RANGE:
            fn = str(rate) + '_' + str(g_max) + '_' + str(h) + '.rates'
            f = open(fn)
            lines = f.readlines()
            fs = [ float(x) for x in lines[1].split() ]
            ins.append(rate)
            sum = 0.
            for x in fs[-n_avg:]:
                sum += x
            freq = sum/float(n_avg)
            outs.append(freq)
    return np.array(ins), np.array(outs)

#generate_events(h=0.05,g_max=0.8)
def write_out_main_curve():
    for h in [0.05, 0.1]:
        for g_max in [ 0.8, 10.0]:
            ins, outs = gain_curve(h, g_max)
            fn = 'gain_curve_' + str(h) + '_' + str(g_max) + '.dat'
            with open(fn,'w') as f:
                f.write(str(h) +'\n')
                f.write(str(g_max) + '\n')
                for x in ins:
                    f.write(str(x) +' \t')
                f.write('\n')
                for y in outs:
                    f.write(str(y) + '\t')
                f.write('\n')

if __name__ == "__main__":
    write_out_main_curve()
