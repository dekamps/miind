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

def simulate_cond3d(I=0.0):
    sim_time = 2000 # ms
    timestep = 1 # ms
    v0 = -65 # mV
    w0 = 0.0
    u0 = 0.0
    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = [v0,w0,u0]
    t_1 = []
    n = 0
    for t in tspan:
        n += 1
        print(n)
        current = cond3d(current, timestep, I)
        #current[1] = current[1] + (poisson.rvs(I*0.001)*0.1)
        current[2] = current[2] + (poisson.rvs(I*0.001)*0.1)
        if (current[0] > -50.4):
                current[0] = -70.6
        t_1 = t_1 + [current]

    return tspan, np.array(t_1)

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

# network parameters taken from "Neuronal circuits overcome imbalance in excitation and inhibition by adjusting connection numbers"
def simulate_cond3d_EI_avg(I=0.0, e_num_neurons = 2000, i_num_neurons = 400, sim_time = 0.5, input_rate=500):
    timestep = 1 # ms
    v0 = -65 # mV
    w0 = 0.0
    u0 = 0.0
    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = np.array([[[v0,w0,u0] for a in range(e_num_neurons)],[[v0,w0,u0] for a in range(i_num_neurons)]])
    refracts = np.array([[0 for a in range(e_num_neurons)],[0 for a in range(i_num_neurons)]])
    spikes = np.array([[-999 for a in range(e_num_neurons)],[-999 for a in range(i_num_neurons)]])
    t_e = np.empty((0,e_num_neurons,3))
    t_i = np.empty((0,i_num_neurons,3))
    r_e = []
    r_i = []
    n = 0
    
    
    last_e_rate = 0.0
    last_i_rate = 0.0
    
    for t in tspan:
        n += 1
        print(n)
        
        e_rate = 0.0
        i_rate = 0.0
        
        for nn in range(e_num_neurons):
            if (refracts[0][nn] > 0):
                refracts[0][nn] -= timestep
                continue
            
            current[0][nn] = cond3d(current[0][nn], timestep, I)
            current[0][nn][1] = current[0][nn][1] + (poisson.rvs(input_rate*0.001)*1.5)
            current[0][nn][1] = current[0][nn][1] + (poisson.rvs(last_e_rate*80)*2)
            current[0][nn][2] = current[0][nn][2] + (poisson.rvs(last_i_rate*20)*8)
            
            if (spikes[0][nn] > -999):
                if (spikes[0][nn] < 0):
                    e_rate += 1
                    spikes[0][nn] = -999
                else:
                    spikes[0][nn] -= timestep
            
            if (current[0][nn][0] > -50.4):
                current[0][nn][0] = -70.6
                spikes[0][nn] = 3
                refracts[0][nn] = 2
        
        e_rate = (e_rate / e_num_neurons)
        
        for nn in range(i_num_neurons):
            if (refracts[1][nn] > 0):
                refracts[1][nn] -= timestep
                continue
            
            current[1][nn] = cond3d(current[1][nn], timestep, I)
            current[1][nn][1] = current[1][nn][1] + (poisson.rvs(input_rate*0.001)*1.5)
            current[1][nn][1] = current[1][nn][1] + (poisson.rvs(last_e_rate*80)*2)
            current[1][nn][2] = current[1][nn][2] + (poisson.rvs(last_i_rate*20)*8)
                
            if (spikes[1][nn] > -999):
                if (spikes[1][nn] < 0):
                    i_rate += 1
                    spikes[1][nn] = -999
                else:
                    spikes[1][nn] -= timestep
                    
            if (current[1][nn][0] > -50.4):
                current[1][nn][0] = -70.6
                spikes[1][nn] = 3
                refracts[1][nn] = 2
        i_rate = (i_rate / i_num_neurons)
        
        last_e_rate = e_rate
        last_i_rate = i_rate
        
        r_e = r_e + [last_e_rate/0.001]
        r_i = r_i + [last_i_rate/0.001]
        
        reshaped = np.reshape(current[0], (1,e_num_neurons,3))
        t_e = np.concatenate((t_e,reshaped), axis=0)
        
        reshaped = np.reshape(current[1], (1,i_num_neurons,3))
        t_i = np.concatenate((t_i,reshaped), axis=0)
    
    return tspan, t_e, t_i, np.array(r_e), np.array(r_i)

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
input_rates=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,100,150,200,250,300,350,400,450,500]

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
