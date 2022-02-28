import pylab
import numpy as np
import matplotlib.pyplot as plt
import miind.miindsimv as miind
from miind.miind_api.MiindSimulation import MiindSimulation
import os
import glob
import shutil
import csv

models = ['hh_50x50x50x50']

rates = [1,2,3,4,5,6,7,8,9,20,30,40]

model_files = [m + '.model' for m in models]
tmat_files = [m + '.tmat' for m in models]

time_end = "0.2"

dataE = [[[] for m in range(len(models))] for r in rates]
dataI = [[[] for m in range(len(models))] for r in rates]

check_rate_id = 1
check_rate = rates[check_rate_id]


directory_exists = [False for m in range(len(models))]

for i in range(len(models)):
    for r in range(len(rates)):
        output_dir = 'hh_TIME_END_' + str(time_end) + '_MODEL_FILE_' + model_files[i] + '_TMAT_FILE_' + tmat_files[i] + '_RATE_' + str(rates[r])
        try:
            os.mkdir(output_dir)
            
            # Copy the sim file to the output directory
            shutil.copy2("hh.xml", output_dir)

            # Copy all support files to output directory
            for file in glob.glob(model_files[i]):
                shutil.copy2(file, output_dir)
                
            for file in glob.glob(tmat_files[i]):
                shutil.copy2(file, output_dir)
                
            os.chdir(output_dir)
                    
            miind.init("hh.xml", TIME_END=time_end, MODEL_FILE=model_files[i], TMAT_FILE=tmat_files[i], RATE=str(rates[r]))
            ts = miind.getTimeStep()
            ln = miind.getSimulationLength()
            miind.startSimulation()

            for j in range(int(ln / ts)):
                miind.evolveSingleStep([rates[r]])

            miind.endSimulation()

            # delete the sim file
            os.remove("hh.xml")

            # delete all support files in output directory
            for file in glob.glob(model_files[i]):
                os.remove(file)
                
            for file in glob.glob(tmat_files[i]):
                os.remove(file)

            # back to base
            os.chdir('..')
            
        except:
            # We've already run the simulations so just read the output
            directory_exists[i] = True
            
            with open(output_dir + "/avg_0") as avg_file:
                reader = csv.reader(avg_file, delimiter='\t')
                for row in reader:
                    if len(row) > 1:
                        dataE[r][i].append([float(a) for a in row[:-1]])

# Compare average membrane potential to different resolutions

fig, ax1 = plt.subplots()

read_times = []
with open("direct_hh_times" + str(check_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        read_times = read_times + [float(row[0])]

read_data_v = np.empty((0,1000))
with open("direct_hh_v" + str(check_rate) + ".csv", newline='') as infile:
    filereadear = csv.reader(infile, delimiter=',', quotechar='|')
    for row in filereadear:
        row = [float(r) for r in row]
        reshaped = np.reshape(row, (1,1000))
        read_data_v = np.concatenate((read_data_v,reshaped), axis=0)
    read_data_v = np.array(read_data_v)

colorE = '#0000FF'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Average Membrane Potential, v (mV)', color=colorE)
ax1.scatter([t*0.001 for t in read_times][:int(len(read_times)/2)], np.mean(read_data_v, axis=1)[:int(len(np.mean(read_data_v, axis=1))/2)], color=colorE, s=5, marker="x")
ax1.tick_params(axis='y', labelcolor=colorE)
    
colours = ['#00FF00','#00CC00']
styles = ['--','-.']
for d in range(len(directory_exists)):
    if not directory_exists[d]:
        continue

    dataE[check_rate_id][d]  = np.array(dataE[check_rate_id][d])   
    ax1.plot(dataE[check_rate_id][d][:,0], [a for a in dataE[check_rate_id][d][:,4]], color=colours[d], ls=styles[d])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
