import matplotlib.pyplot as plt
import miind.miindsimv as miind
import numpy as np

miind.init(1, "syn.xml")

timestep = miind.getTimeStep()
simulation_length = miind.getSimulationLength()
print('Timestep from XML : {}'.format(timestep))
print('Sim time from XML : {}'.format(simulation_length))

miind.startSimulation()

constant_input = [100, 100]
constant_input_1 = [100, 100]
constant_input_2 = [500, 100]

activities = []
for i in range(int(simulation_length/timestep)):
    if i > 0.1 / timestep:
        constant_input = constant_input_2
    if i > 0.2 / timestep:
        constant_input = constant_input_1
    if i > 0.3 / timestep:
        constant_input = constant_input_2
    if i > 0.4 / timestep:
        constant_input = constant_input_1
    if i > 0.5 / timestep:
        constant_input = constant_input_2
    if i > 0.6 / timestep:
        constant_input = constant_input_1
    if i > 0.7 / timestep:
        constant_input = constant_input_2
    if i > 0.8 / timestep:
        constant_input = constant_input_1
    if i > 0.9 / timestep:
        constant_input = constant_input_2
    if i > 1.0 / timestep:
        constant_input = constant_input_1
    if i > 1.1 / timestep:
        constant_input = constant_input_2
    if i > 1.2 / timestep:
        constant_input = constant_input_1
    if i > 1.3 / timestep:
        constant_input = constant_input_2
    if i > 1.4 / timestep:
        constant_input = constant_input_1
    if i > 1.5 / timestep:
        constant_input = constant_input_2
    if i > 1.6 / timestep:
        constant_input = constant_input_1
    if i > 1.7 / timestep:
        constant_input = constant_input_2
    if i > 1.8 / timestep:
        constant_input = constant_input_1
    if i > 1.9 / timestep:
        constant_input = constant_input_2
    activities.append(miind.evolveSingleStep(constant_input))

miind.endSimulation()

activities = np.array(activities).transpose()

plt.figure()
plt.plot(activities[0,:])
plt.plot(activities[1,:])
plt.title("Firing Rate.")

plt.show()
