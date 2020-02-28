#!/usr/bin/env python

def simulation_run(efficacy, rate):
	import sys
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
	miind.init(number_of_nodes, efficacy)

	timestep = miind.getTimeStep()
	simulation_length = miind.getSimulationLength()
	print('Timestep from XML : {}'.format(timestep))
	print('Sim time from XML : {}'.format(simulation_length))

	miind.startSimulation()

	constant_input = [rate]
	activities = []
	for i in range(int(simulation_length/timestep)):
	    activities.append(miind.evolveSingleStep(constant_input)[0])

	miind.endSimulation()

	return activities[-1]

if __name__ == '__main__':
    args = sys.argv # should be ['runlif.py',efficacy, rate]
    simulation_run(float(args[1]), float(args[2]))
