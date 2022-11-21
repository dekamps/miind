#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import random
import csv
from scipy.stats import poisson

def cond3d(current, dt, I=0):
    E_l = -70.6
    V_thres = -50.4 
    E_e = 0.0
    E_i = -75
    C = 281
    g_l = 0.03
    tau_e = 2.728
    tau_i = 10.49
    
    v = current[0]
    w = current[1] 
    u = current[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -w / tau_e
    u_prime = -u / tau_i

    return [v + dt*v_prime, w + dt*w_prime, u + dt*u_prime]

def simulate_cond3d_pop(I=0.0, num_neurons = 500, sim_time = 0.5, input_rate=500):
    timestep = 1 # ms
    v0 = -65 # mV
    w0 = 0.0
    u0 = 0.0
    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = np.array([[v0,w0,u0] for a in range(num_neurons)])
    t_1 = np.empty((0,num_neurons,3))
    r_1 = []
    n = 0
    for t in tspan:
        n += 1
        print(n)
        
        spikes = 0
        for nn in range(num_neurons):
            current[nn] = cond3d(current[nn], timestep, I)
            current[nn][1] = current[nn][1] + (poisson.rvs(input_rate*0.001)*1.5)
            current[nn][1] = current[nn][1] + (poisson.rvs(50*0.001)*1.5)
            current[nn][2] = current[nn][2] + (poisson.rvs(50*0.001)*1.5)
            
            if (current[nn][0] > -50.4):
                spikes += 1
                current[nn][0] = -70.6
        
        r_1 = r_1 + [(spikes / num_neurons) / 0.001]
        t_1 = np.concatenate((t_1,np.reshape(current, (1,num_neurons,3))))
    
    return tspan, np.array(t_1), np.array(r_1)

# cond3D population

def cond_run(_N=1000, _input_rates=[5,10,15,20,25,30,35,40,45,50]):
    for rate in _input_rates:
        times, data, firing_rates = simulate_cond3d_pop(I=0.0, num_neurons=_N, sim_time=2000, input_rate=rate)
        
        with open("direct_small_times" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow([times[t]])

        with open("direct_small_v" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,0])
                
        with open("direct_small_w" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,1])
                
        with open("direct_small_u" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,2]) 

        with open("direct_small_rate" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow([firing_rates[t]])   
                
        with open("direct_small_steady" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n in range(_N):
                filewriter.writerow([np.mean(data[int(len(data[:,n,0])*0.75):,n,0])]) 


# Straight Comparison of Activity
input_rate = 2
N = 10000
input_rates=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,100]

cond_run(_N=N, _input_rates=input_rates)

read_times = []
with open("direct_small_times" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_data_v = np.empty((0,N))
with open("direct_small_v" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_v = np.concatenate((read_data_v,reshaped), axis=0)
    read_data_v = np.array(read_data_v)
    
read_data_w = np.empty((0,N))
with open("direct_small_w" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_w = np.concatenate((read_data_w,reshaped), axis=0)
    read_data_w = np.array(read_data_w)

read_data_u = np.empty((0,N))
with open("direct_small_u" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_u = np.concatenate((read_data_u,reshaped), axis=0)
    read_data_u = np.array(read_data_u)

fig, ax1 = plt.subplots()

color = '#0000FF'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Average Membrane Potential, v (mV)', color=color)
ax1.plot(read_times, np.mean(read_data_v, axis=1), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color1 = '#990000'
color2 = '#009900'
color3 = '#FF9900'
ax2.set_ylabel('Conductance Variables, w,u', color=color1)  # we already handled the x-label with ax1
ax2.plot(read_times, np.mean(read_data_w, axis=1), color=color1, ls='--')
ax2.plot(read_times, np.mean(read_data_u, axis=1), color=color2, ls=':')
ax2.tick_params(axis='y', labelcolor=color1)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
