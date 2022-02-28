#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import random
import csv
from scipy.stats import poisson

def hh(current, dt, I=0):
    V_k = -90
    V_na = 50
    V_l = -65
    g_k = 30
    g_na = 100
    g_l = 0.5
    C = 1.0
    V_t = -63
    
    v = current[0]
    m = current[1] 
    n = current[2]
    h = current[3]

    alpha_m = (0.32 * (13 - v + V_t)) / (np.exp((13 - v + V_t)/4) - 1) 
    alpha_n = (0.032 * (15 - v + V_t)) / (np.exp((15 - v + V_t)/5) - 1)
    alpha_h = 0.128 * np.exp((17 - v + V_t)/18)

    beta_m = (0.28 * (v - V_t - 40)) / (np.exp((v - V_t - 40)/5) - 1)
    beta_n = 0.5 * np.exp((10 - v + V_t)/40)
    beta_h = 4 / (1 + np.exp((40 - v + V_t)/5))
    

    v_prime = (-(g_k*(n**4)*(v - V_k)) - (g_na*(m**3)*h*(v - V_na)) - (g_l*(v - V_l)) + I) / C
    m_prime = (alpha_m * (1 - m)) - (beta_m*m)
    n_prime = (alpha_n * (1 - n)) - (beta_n*n)
    h_prime = (alpha_h * (1 - h)) - (beta_h*h)

    return [v + dt*v_prime, m + dt*m_prime, n + dt*n_prime, h + dt*h_prime]

def simulate_hh_pop(I=0.0, num_neurons = 500, sim_time = 0.5, input_rate=200):
    timestep = 0.01 # ms
    v0 = -70 # mV
    m0 = 0.05
    n0 = 0.3
    h0 = 0.6
    tolerance = 1e-4
    tspan = np.linspace(0, sim_time, int(sim_time/timestep))
    current = np.array([[v0,m0,n0,h0] for a in range(num_neurons)])
    t_1 = np.empty((0,num_neurons,4))
    n = 0
    for t in tspan:
        n += 1
        print(n)
        
        for nn in range(num_neurons):
            current[nn] = hh(current[nn], timestep, I)
            current[nn][0] = current[nn][0] + (poisson.rvs(input_rate*100*0.00001)*3)
        
        t_1 = np.concatenate((t_1,np.reshape(current, (1,num_neurons,4))))
    
    return tspan, np.array(t_1)

def hh_run(_N=1000, _input_rates=[5,10,15,20,25,30,35,40,45,50]):
    for rate in _input_rates:
        times, data = simulate_hh_pop(I=0.0, num_neurons=_N, sim_time=40, input_rate=rate)
        
        with open("direct_hh_times" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow([times[t]])

        with open("direct_hh_v" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,0])
                
        with open("direct_hh_m" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,1])
                
        with open("direct_hh_n" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,2]) 
                
        with open("direct_hh_h" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in range(len(times)):
                filewriter.writerow(data[t,:,3]) 

        with open("direct_hh_steady" + str(rate) + ".csv", "w", newline='') as outfile:
            filewriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n in range(_N):
                filewriter.writerow([np.mean(data[int(len(data[:,n,0])*0.75):,n,0])]) 


# Straight Comparison of Activity
input_rate = 200
N = 1000
input_rates=[1,2,3,4,5,6,7,8,9,20,30,40]

hh_run(_N=N, _input_rates=input_rates)

read_times = []
with open("direct_hh_times" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_data_v = np.empty((0,N))
with open("direct_hh_v" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_v = np.concatenate((read_data_v,reshaped), axis=0)
    read_data_v = np.array(read_data_v)
    
read_data_m = np.empty((0,N))
with open("direct_hh_m" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_m = np.concatenate((read_data_m,reshaped), axis=0)
    read_data_m = np.array(read_data_m)

read_data_n = np.empty((0,N))
with open("direct_hh_n" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_n = np.concatenate((read_data_n,reshaped), axis=0)
    read_data_n = np.array(read_data_n)
    
read_data_h = np.empty((0,N))
with open("direct_hh_h" + str(input_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,N))
        read_data_h = np.concatenate((read_data_h,reshaped), axis=0)
    read_data_h = np.array(read_data_h)

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
ax2.plot(read_times, np.mean(read_data_m, axis=1), color=color1, ls='--')
ax2.plot(read_times, np.mean(read_data_n, axis=1), color=color2, ls=':')
ax2.plot(read_times, np.mean(read_data_h, axis=1), color=color3, ls='-.')
ax2.tick_params(axis='y', labelcolor=color1)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

