import pylab
import numpy
import matplotlib.pyplot as plt
import miind.miindsim as miind
from miind.miind_api.MiindSimulation import MiindSimulation
import os
import glob
import shutil

efficacies = [0.01 * (n+1) for n in range(10)]
rates = [1200 / (n+1) for n in range(10)]
firing_rates = [0 for n in range(10)]
directories_exist = False

for i in range(len(efficacies)):
    output_dir = 'lif_efficacy_' + str(efficacies[i]) + '_rate_' + str(rates[i]) + '_'
    
    try:
        os.mkdir(output_dir)
        
        # Copy the sim file to the output directory
        shutil.copy2("lif.xml", output_dir)

        # Copy all potential support files to output directory
        for file in glob.glob('*.model'):
            shutil.copy2(file, output_dir)
            
        for file in glob.glob('*.mat'):
            shutil.copy2(file, output_dir)
            
        for file in glob.glob('*.tmat'):
            shutil.copy2(file, output_dir)
            
        os.chdir(output_dir)
                
        miind.init("lif.xml", efficacy=str(efficacies[i]), rate=str(rates[i]))
        ts = miind.getTimeStep()
        ln = miind.getSimulationLength()
        miind.startSimulation()

        for j in range(int(ln / ts)):
            miind.evolveSingleStep([0.0])

        miind.endSimulation()

        # delete the sim file
        os.remove("lif.xml")

        # delete all support files in output directory
        for file in glob.glob('*.model'):
            os.remove(file)
            
        for file in glob.glob('*.mat'):
            os.remove(file)
            
        for file in glob.glob('*.tmat'):
            os.remove(file)

        # back to base
        os.chdir('..')
        
    except:
        # We've already run the simulations so just read the output
        sim = MiindSimulation("lif.xml", efficacy=str(efficacies[i]), rate=str(rates[i]))
        firing_rates[i] = sim.rates[1][-1]
        directories_exist = True

if directories_exist:
    plt.figure()
    plt.plot(firing_rates)
    plt.title("Firing Rate.")
    plt.show()
