import pylab
import numpy as np
import matplotlib.pyplot as plt
import miind.miindsimv as miind
from miind.miind_api.MiindSimulation import MiindSimulation
import os
import glob
import shutil
import csv

models = ['cond3d_small_50x50x50', 'cond3d_small_100x100x100', 'cond3d_small_150x150x150', 'cond3d_small_50x50x300']

rates = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]

model_files = [m + '.model' for m in models]
tmat_files = [m + '.tmat' for m in models]

time_end = "2.0"

dataE = [[[] for m in range(len(models))] for r in range(len(rates))]

directory_exists = [False for m in range(len(models))]

for i in range(len(models)):
    for r in range(len(rates)):
        output_dir = 'cond3D_TIME_END_' + str(time_end) + '_MODEL_FILE_' + model_files[i] + '_TMAT_FILE_' + tmat_files[i] + '_RATE_' + str(rates[r]) + '_'
        try:
            os.mkdir(output_dir)
            
            # Copy the sim file to the output directory
            shutil.copy2("cond3D.xml", output_dir)

            # Copy all support files to output directory
            for file in glob.glob(model_files[i]):
                shutil.copy2(file, output_dir)
                
            for file in glob.glob(tmat_files[i]):
                shutil.copy2(file, output_dir)
                
            os.chdir(output_dir)
                    
            miind.init("cond3D.xml", TIME_END=time_end, MODEL_FILE=model_files[i], TMAT_FILE=tmat_files[i], RATE=str(rates[r]))
            ts = miind.getTimeStep()
            ln = miind.getSimulationLength()
            miind.startSimulation()

            for j in range(int(ln / ts)):
                miind.evolveSingleStep([rates[r]])

            miind.endSimulation()

            # delete the sim file
            os.remove("cond3D.xml")

            # delete all support files in output directory
            for file in glob.glob(model_files[i]):
                os.remove(file)
                
            for file in glob.glob(tmat_files[i]):
                os.remove(file)

            # back to base
            os.chdir('..')
            
            directory_exists[i] = True
            
            with open(output_dir + "/avg_0") as avg_file:
                reader = csv.reader(avg_file, delimiter='\t')
                for row in reader:
                    if len(row) > 1:
                        dataE[r][i].append([float(a) for a in row[:-1]])
            
        except:
            # We've already run the simulations so just read the output
            directory_exists[i] = True
            
            with open(output_dir + "/avg_0") as avg_file:
                reader = csv.reader(avg_file, delimiter='\t')
                for row in reader:
                    if len(row) > 1:
                        dataE[r][i].append([float(a) for a in row[:-1]])

# Compare current-rate curves

fig, ax1 = plt.subplots()
avg_rates = []

for rate in rates:
    read_rates = []
    with open("direct_small_rate" + str(rate) + ".csv", newline='') as infile:
        filereadear = csv.reader(infile, delimiter=',', quotechar='|')
        for row in filereadear:
            row = [float(r) for r in row]
            read_rates = read_rates + [float(row[0])]
            
    avg_rates = avg_rates + [np.mean(read_rates[int(len(read_rates)*0.75):])]
    
colorE = '#0000FF'
ax1.set_xlabel('Input Poisson Rate (Hz)', size=14)
ax1.set_ylabel('Average Steady Firing Rate (Hz)', size=14)
line1 = ax1.scatter(rates, avg_rates, color=colorE, s=25, marker="x", linewidths=2)
line1.set_label('Monte Carlo Simulation')
ax1.tick_params(axis='y')

colours = ['#770000','#990000','#CC0000','#00AA00']
labels = ['50x50x50', '100x100x100', '150x150x150', '50x50x300']
styles = ['--','-.','-','-']
for d in range(len(directory_exists)):
    avg_rates = []
    for rate in rates:
        if not directory_exists[d]:
            continue

        sim = MiindSimulation("cond3D.xml", TIME_END=time_end, MODEL_FILE=model_files[d], TMAT_FILE=tmat_files[d], RATE=str(rate))
        avg_rates = avg_rates + [np.mean(sim.rates[0][int(len(sim.rates[0])*0.75):])]
   
    ln, = ax1.plot(rates, avg_rates, color=colours[d], ls=styles[d])
    ln.set_label(labels[d])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#ax1.legend()
plt.show()

# Compare current-voltage curves

fig, ax1 = plt.subplots()
avg_volts = []

for rate in rates:
    read_data_v = np.empty((0,10000))
    with open("direct_small_u" + str(rate) + ".csv", newline='') as infile:
        filereadear = csv.reader(infile, delimiter=',', quotechar='|')
        for row in filereadear:
            row = [float(r) for r in row]
            reshaped = np.reshape(row, (1,10000))
            read_data_v = np.concatenate((read_data_v,reshaped), axis=0)
        read_data_v = np.array(read_data_v)
            
    avg_volts = avg_volts + [np.mean(np.mean(read_data_v, axis=1)[int(len(np.mean(read_data_v, axis=1))*0.75):])]
    
colorE = '#0000FF'
ax1.set_xlabel('Input Poisson Rate (Hz)', size=14)
ax1.set_ylabel('Average Steady Inhibitory Conductance, u (nS)', size=14)
line1 = ax1.scatter(rates, avg_volts, color=colorE, s=25, marker="x", linewidths=2)
line1.set_label('Monte Carlo Simulation')
ax1.tick_params(axis='y')

colours = ['#770000','#990000','#CC0000','#00AA00']
labels = ['50x50x50', '100x100x100', '150x150x150', '50x50x300']
styles = ['--','-.','-','-']
for d in range(len(directory_exists)):
    avg_volts = []
    if (d == 0):
        continue
    for r in range(len(rates)):
        if not directory_exists[d]:
            continue

        dataE[r][d]  = np.array(dataE[r][d])
        avg_volts = avg_volts + [a for a in [np.mean(dataE[r][d][:,1][int(len(dataE[r][d][:,1])*0.75):])]]
        
    ln, = ax1.plot(rates, avg_volts, color=colours[d], ls=styles[d])
    ln.set_label(labels[d])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#ax1.legend()
plt.show()
