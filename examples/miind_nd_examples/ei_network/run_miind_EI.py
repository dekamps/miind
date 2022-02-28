import pylab
import numpy as np
import matplotlib.pyplot as plt
import miind.miindsimv as miind
from miind.miind_api.MiindSimulation import MiindSimulation
import os
import glob
import shutil
import csv

models = ['cond3d_100x10x40', 'cond3d_100x25x40', 'cond3d_100x25x40','cond3d_100x25x40']
Inhib_percent = [50, 20, 10, 90]

#models = ['cond3d_100x25x40']
#Inhib_percent = [20]

model_files = [m + '.model' for m in models]
tmat_files = [m + '.tmat' for m in models]

time_end = "1.0"

dataE = [[] for j in range(len(Inhib_percent))]
dataI = [[] for j in range(len(Inhib_percent))]

directory_exists = [False for m in range(len(models))]

for j in range(len(Inhib_percent)):
    output_dir = 'cond3D_EI_TE_' + str(time_end) + '_MF_' + model_files[j] + '_TMF_' + tmat_files[j] + '_N_EE_' + str(100-Inhib_percent[j]) + '_N_EI_' + str(100-Inhib_percent[j]) + '_N_IE_' + str(Inhib_percent[j]) + '_N_II_' + str(Inhib_percent[j]) + '_'
    
    
    try:
        os.mkdir(output_dir)
        
        # Copy the sim file to the output directory
        shutil.copy2("cond3D_EI.xml", output_dir)

        # Copy all support files to output directory
        for file in glob.glob(model_files[j]):
            shutil.copy2(file, output_dir)
            
        for file in glob.glob(tmat_files[j]):
            shutil.copy2(file, output_dir)
            
        os.chdir(output_dir)
                
        miind.init("cond3D_EI.xml", TE=time_end, MF=model_files[j], TMF=tmat_files[j], N_EE=str((100-Inhib_percent[j])), N_EI = str((100-Inhib_percent[j])), N_IE=str(Inhib_percent[j]), N_II=str(Inhib_percent[j]))
        ts = miind.getTimeStep()
        ln = miind.getSimulationLength()
        miind.startSimulation()
        
        ere = 0.0
        eri = 500.0

        for k in range(int(ln / ts)):
            if (k > 2):
                ere = 500.0
            miind.evolveSingleStep([ere,eri])

        miind.endSimulation()

        # delete the sim file
        os.remove("cond3D_EI.xml")

        # delete all support files in output directory
        for file in glob.glob(model_files[j]):
            os.remove(file)
            
        for file in glob.glob(tmat_files[j]):
            os.remove(file)

        # back to base
        os.chdir('..')
        
    except:
        # We've already run the simulations so just read the output
        directory_exists[j] = True
        
        with open(output_dir + "/avg_0") as avg_file:
            reader = csv.reader(avg_file, delimiter='\t')
            for row in reader:
                if len(row) > 1:
                    dataE[j].append([float(a) for a in row[:-1]])
                    
        with open(output_dir + "/avg_1") as avg_file:
            reader = csv.reader(avg_file, delimiter='\t')
            for row in reader:
                if len(row) > 1:
                    dataI[j].append([float(a) for a in row[:-1]])

fig, ax1 = plt.subplots()
check_inhib_index = 0

read_times = []
with open("direct_ei_times_inhib_" + str(Inhib_percent[check_inhib_index]*10) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_data_v = np.empty((0,1000-(Inhib_percent[check_inhib_index]*10)))
with open("direct_e_v_inhib_" + str(Inhib_percent[check_inhib_index]*10) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,1000-(Inhib_percent[check_inhib_index]*10)))
        read_data_v = np.concatenate((read_data_v,reshaped), axis=0)
    read_data_v = np.array(read_data_v)

colorE = '#0000FF'
ax1.set_xlim([0, 0.5])
ax1.set_xlabel('Time (s)', size=14)
ax1.set_ylabel('Average Membrane Potential, v (mV)', size=14)
ax1.plot([t*0.001 for t in read_times], np.mean(read_data_v, axis=1)[:int(len(np.mean(read_data_v, axis=1)))], color=colorE)
line1 = ax1.scatter([t*0.001 for t in read_times][::5], np.mean(read_data_v, axis=1)[:int(len(np.mean(read_data_v, axis=1))):5], color=colorE, s=25, marker="x", linewidths=2)
line1.set_label('Monte Carlo Excitatory Population')
ax1.tick_params(axis='y')
    
colours = ['#990000']
styles = ['-']
dataE[check_inhib_index] = np.array(dataE[check_inhib_index])   
ln, = ax1.plot(dataE[check_inhib_index][:,0], dataE[check_inhib_index][:,3], color=colours[0], ls=styles[0])
ln.set_label('MIIND Simulation Excitatory Population')
ax1.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# Compare average firing rate to different resolutions

fig, ax1 = plt.subplots()

read_times = []
with open("direct_ei_times_inhib_" + str(Inhib_percent[check_inhib_index]*10) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_rates = []
with open("direct_e_rate_inhib_" + str(Inhib_percent[check_inhib_index]*10) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        read_rates = read_rates + [float(row[0])]

colorE = '#0000FF'
ax1.set_xlabel('Time (s)', size=14)
ax1.set_ylabel('Average Firing Rate (Hz)', size=14)
ax1.plot([f * 0.001 for f in read_times[:int(len(read_times)):5]], [np.sum(read_rates[r:r+5])/5 for r in [i for i in range(len(read_rates))][:int(len(read_times)):5]], color=colorE)
ax1.scatter([f * 0.001 for f in read_times[:int(len(read_times)):5]], [np.sum(read_rates[r:r+5])/5 for r in [i for i in range(len(read_rates))][:int(len(read_times)):5]], color=colorE, s=15, marker="x", linewidths=2)
ax1.tick_params(axis='y')
    
colours = ['#990000','#00CC00','#009900','#007700','#005500','#003300','#001100']
styles = ['-','-.',':','--','-.',':','--']

sim = MiindSimulation("cond3D_EI.xml", TE=time_end, MF=model_files[check_inhib_index], TMF=tmat_files[check_inhib_index], N_EE=str((100-Inhib_percent[check_inhib_index])), N_EI = str((100-Inhib_percent[check_inhib_index])), N_IE=str(Inhib_percent[check_inhib_index]), N_II=str(Inhib_percent[check_inhib_index]))
ax1.plot(sim.rates['times'][:1000], sim.rates[0], color=colours[0], ls=styles[0])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
