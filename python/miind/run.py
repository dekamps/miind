#!/usr/bin/env python3

import sys
import pylab
import numpy
import matplotlib.pyplot as plt

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
    miind.init(sys.argv[1])

    timestep = miind.getTimeStep()
    simulation_length = miind.getSimulationLength()
    print('Timestep from XML : {}'.format(timestep))
    print('Sim time from XML : {}'.format(simulation_length))

    miind.startSimulation()

    for i in range(int(simulation_length/timestep)):
        miind.evolveSingleStep([])

    miind.endSimulation()

