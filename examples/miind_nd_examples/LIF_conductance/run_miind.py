import pylab
import numpy as np
import matplotlib.pyplot as plt
import miind.miindsimv as miind
from miind.miind_api.MiindSimulation import MiindSimulation
import os
import glob
import shutil
import csv

models = ['cond3d_small_50x50x50', 'cond3d_small_100x100x100', 'cond3d_small_150x150x150', 'cond3d_small_200x200x100', 'cond3d_small_100x100x200', 'cond3d_small_50x50x300']

model_files = [m + '.model' for m in models]
tmat_files = [m + '.tmat' for m in models]

time_end = "1.0"

dataE = [[] for m in range(len(models))]
dataI = [[] for m in range(len(models))]

check_rate = 100

directory_exists = [False for m in range(len(models))]

for i in range(len(models)):
    output_dir = 'cond3D_TIME_END_' + str(time_end) + '_MODEL_FILE_' + model_files[i] + '_TMAT_FILE_' + tmat_files[i] + '_'
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
                
        miind.init("cond3D.xml", TIME_END=time_end, MODEL_FILE=model_files[i], TMAT_FILE=tmat_files[i])
        ts = miind.getTimeStep()
        ln = miind.getSimulationLength()
        miind.startSimulation()

        for j in range(int(ln / ts)):
            miind.evolveSingleStep([check_rate])

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
                    dataE[i].append([float(a) for a in row[:-1]])
        
    except:
        # We've already run the simulations so just read the output
        directory_exists[i] = True
        
        with open(output_dir + "/avg_0") as avg_file:
            reader = csv.reader(avg_file, delimiter='\t')
            for row in reader:
                if len(row) > 1:
                    dataE[i].append([float(a) for a in row[:-1]])

# Compare average membrane potential to different resolutions

fig, ax1 = plt.subplots()

read_times = []
with open("direct_small_times" + str(100) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_data_v = np.empty((0,10000))
with open("direct_small_v" + str(100) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,10000))
        read_data_v = np.concatenate((read_data_v,reshaped), axis=0)
    read_data_v = np.array(read_data_v)

colorE = '#0000FF'
ax1.set_xlabel('Time (s)', size=14)
ax1.set_ylabel('Squared Error from Monte Carlo Mean Membrane Potential', size=14)
#line1 = ax1.scatter([t*0.001 for t in read_times][:int(len(read_times)/2):20], np.mean(read_data_v, axis=1)[:int(len(np.mean(read_data_v, axis=1))/2):20], color=colorE, s=25, marker="x", linewidths=2)
#line1.set_label('Monte Carlo Simulation')
ax1.tick_params(axis='y')
    
colours = ['#770000','#990000','#CC0000','#007700','#009900','#00AA00']
labels = ['50x50x50', '100x100x100', '150x150x150', '200x200x100', '100x100x200', '50x50x300']
styles = ['--','-.','-','-.','--','-']
for d in range(len(colours)):
    if not directory_exists[d]:
        continue

    dataE[d]  = np.array(dataE[d])   
    ln, = ax1.plot(dataE[d][:,0], [((dataE[d][:,3][a]) - np.mean(read_data_v, axis=1)[:int(len(np.mean(read_data_v, axis=1))/2)][a])*((dataE[d][:,3][a]]) - np.mean(read_data_v, axis=1)[:int(len(np.mean(read_data_v, axis=1))/2)][a]) for a in range(len(dataE[d][:,3]))], color=colours[d], ls=styles[d])
    ln.set_label(labels[d])

#ax1.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#fig.savefig('cond3d_avg_membrane_pot_res_compare_large.pdf', format='pdf')
plt.show()

# Compare average firing rate to different resolutions

fig, ax1 = plt.subplots()

read_times = []
with open("direct_small_times" + str(100) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_rates = []
with open("direct_small_rate" + str(100) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        read_rates = read_rates + [float(row[0])]

colorE = '#0000FF'
ax1.set_xlabel('Time (s)', size=14)
ax1.set_ylabel('Average Firing Rate (Hz)', size=14)
print(np.array([np.sum(read_rates[r:r+20])/20 for r in [i for i in range(len(read_rates))][::20]][:int(len(read_rates)/2):20]).shape, np.array([f * 0.001 for f in read_times[:int(len(read_times)/2):20]]).shape )
line1 = ax1.scatter([f * 0.001 for f in read_times[:int(len(read_times)/2):20]], [np.sum(read_rates[r:r+20])/20 for r in [i for i in range(len(read_rates))][:int(len(read_times)/2):20]], color=colorE, s=25, marker="x", linewidths=2)
line1.set_label('Monte Carlo Simulation')
ax1.tick_params(axis='y')
    
colours = ['#770000','#990000','#CC0000','#007700','#009900','#00AA00']
labels = ['50x50x50', '100x100x100', '150x150x150', '200x200x100', '100x100x200', '50x50x300']
styles = ['--','-.','-','-.','--','-']
for d in range(len(directory_exists)):
    if not directory_exists[d]:
        continue

    sim = MiindSimulation("cond3D.xml", TIME_END=time_end, MODEL_FILE=model_files[d], TMAT_FILE=tmat_files[d])
    ln, = ax1.plot(sim.rates['times'], sim.rates[0], color=colours[d], ls=styles[d])
    ln.set_label(labels[d])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#ax1.legend()
plt.show()
