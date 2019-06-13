#import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import time
from scipy.stats import poisson
import argparse

def adEx(y):
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

def simulation(rate, t_end = 1.0, h = 1.0, reset_jump = 100):

    start=time.time()

    dt=0.0001 #time step
    nu=rate #input rate
    N_pop=100000 #number of neurons
    T_max = t_end

    out_step = 0.001
    factor=int(np.around(out_step/dt))
    N_steps=int(np.around(T_max/dt)) #number of time-steps)

    v_save=np.zeros((N_pop,int(N_steps/factor)))
    w_save=np.zeros((N_pop,int(N_steps/factor)))

    v=(-70.6)*np.ones(N_pop)
    w=(4.0)*np.ones(N_pop)

    #initial conditions
    t=0

    out=np.zeros(N_steps)
    for i in range(N_steps):

        [dv,dw]=adEx([v,w])
        spikes=poisson.rvs(nu*dt,size=N_pop) #generating spikes

        v=v+(dt*1000)*dv+h*spikes
        w=w+(dt*1000)*dw

        if i % factor == 0:
            v_save[:,int(i/factor)]=v
            w_save[:,int(i/factor)]=w

        out[i]=np.sum(v>(-40.0))   #calculate number of firing neurons
        r = [True if a > (-40.0) else False for a in v]
        v[v>(-40.0)]=(-70.0)  #firing neurons reset
        w[r]+=reset_jump

        t=t+dt

    # plt.scatter(v_save[:,80], w_save[:,80])
    # plt.show()

    #binning firing neurons to get a smooth firing rate curve
    #factor=50 #number of time-steps in each bin
    rate_smooth=np.zeros(int(N_steps/factor))
    for j in range(int(N_steps/factor)):
        rate_smooth[j]=np.sum(out[factor*j:factor*(j+1)-1])/dt/factor/N_pop

    t_axis=dt*factor*np.arange(int(N_steps/factor))

    end=time.time()
    print('Runtime for simulation is {}s'.format(end-start))

    plt.plot(t_axis,rate_smooth)
    plt.show()

simulation(1500)
