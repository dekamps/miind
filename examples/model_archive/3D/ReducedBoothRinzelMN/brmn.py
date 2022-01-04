import matplotlib.pyplot as plt
import miind.miindsimv as miind
import numpy as np

miind.init(1, "brmn.xml", TIME_END="0.55")

timestep = miind.getTimeStep()
simulation_length = miind.getSimulationLength()
print('Timestep from XML : {}'.format(timestep))
print('Sim time from XML : {}'.format(simulation_length))

miind.startSimulation()

constant_input = [600, 200]

activities = []
for i in range(int(simulation_length/timestep)):
    if i > 0.04 / timestep:
        constant_input = [1500, 200]
    if i > 0.06 / timestep:
        constant_input = [600, 200]
    if i > 0.1 / timestep:
        constant_input = [0, 2500]
    if i > 0.12 / timestep:
        constant_input = [600, 200]
    activities.append(miind.evolveSingleStep(constant_input))

miind.endSimulation()

activities = np.array(activities).transpose()

plt.figure()
plt.plot(activities[0,:])
plt.title("Firing Rate.")

plt.show()
