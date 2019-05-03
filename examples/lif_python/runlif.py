# Once the Python Shared Library has been built in MIIND,
# copy this file to the results directory (where the .cpp and .so files were
# created).

import pylab
import numpy
import matplotlib.pyplot as plt
import liblif as miind

# Comment out MPI, comm and rank lines below if not using
# MPI
#######################
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#######################

number_of_nodes = 1
miind.init(number_of_nodes)

timestep = miind.getTimeStep()
simulation_length = miind.getSimulationLength()
print('Timestep from XML : {}'.format(timestep))
print('Sim time from XML : {}'.format(simulation_length))

miind.startSimulation()

constant_input = [2500]
activities = []
for i in range(int(simulation_length/timestep)): #0.001 is the time step defined in the xml
    activities.append(miind.evolveSingleStep(constant_input)[0])

miind.endSimulation()

plt.figure()
plt.plot(activities)
plt.title("Firing Rate.")

plt.show()

