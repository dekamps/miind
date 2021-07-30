#!/usr/bin/env python3

import sys
import pylab
import numpy
import matplotlib.pyplot as plt
import shutil
import glob
import os

# For this run command only, open up the xml file and check to see
# if there's a MeshAlgorithmGroup or GridAlgorithmGroup inside so we
# know whether to import miindsim or miindsimv
# user Python scripts should just import the one they know they need.

from miind.miind_api import MiindSimulation
sim = MiindSimulation(sys.argv[1])
if sim.requires_cuda:
    print("Group algorithm detected. Importing Cuda MIIND (miindsimv).")
    import miind.miindsimv as miind
else:
    import miind.miindsim as miind

if len(sys.argv) < 2:
    print("run expects a simulation file name as parameter.")
else:
    num_nodes = 1
    miind_vars = {}
    if len(sys.argv) > 2:
        if sys.argv[2].isdigit():
            try:
                num_nodes = int(sys.argv[2])
                if num_nodes < 1:
                    print('If parameter 2 is the number of nodes, it must be greater than 0.')
            except ValueError:
                print('If parameter 2 is the number of nodes, it must be a integer.')
        else:
            for a in sys.argv[2:]:
                key, val = a.strip().split('=')
                miind_vars[key] = val

    try:
        # Create the output directory and copy all required files into it
        sim_basename = os.path.basename(os.path.splitext(sys.argv[1])[0])
        output_dir = sim_basename + '_'
        for k,v in miind_vars.items():
            output_dir = output_dir + k + '_' + v + '_'
        
        try:
            os.mkdir(output_dir)
        except:
            # delete output dir and rebuild it
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
        
        # Copy the sim file to the output directory
        shutil.copy2(sys.argv[1], output_dir)
        
        # Copy all potential support files to output directory
        for file in glob.glob('*.model'):
            shutil.copy2(file, output_dir)
            
        for file in glob.glob('*.mat'):
            shutil.copy2(file, output_dir)
            
        for file in glob.glob('*.tmat'):
            shutil.copy2(file, output_dir)
            
        os.chdir(output_dir)
        
        miind.init(num_nodes, sys.argv[1], **miind_vars)

        timestep = miind.getTimeStep()
        simulation_length = miind.getSimulationLength()
        print('Timestep from XML : {}'.format(timestep))
        print('Sim time from XML : {}'.format(simulation_length))

        miind.startSimulation()

        for i in range(int(simulation_length/timestep)):
            miind.evolveSingleStep([])

        miind.endSimulation()
        
        # delete the sim file
        os.remove(sys.argv[1])
        
        # delete all support files in output directory
        for file in glob.glob('*.model'):
            os.remove(file)
            
        for file in glob.glob('*.mat'):
            os.remove(file)
            
        for file in glob.glob('*.tmat'):
            os.remove(file)
        
        # back to base
        os.chdir('..')
        
    except Exception as inst:
        print(inst)

