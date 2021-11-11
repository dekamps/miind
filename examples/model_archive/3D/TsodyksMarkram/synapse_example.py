import matplotlib.pyplot as plt
import miind.miindsimv as miind

miind.init(1, "syn.xml")

timestep = miind.getTimeStep()
simulation_length = miind.getSimulationLength()
print('Timestep from XML : {}'.format(timestep))
print('Sim time from XML : {}'.format(simulation_length))

miind.startSimulation()

constant_input = [10]
constant_input_2 = [40]

activities = []
for i in range(int(simulation_length/timestep)):
    if i > 0.5 / timestep:
        constant_input = constant_input_2
    activities.append(miind.evolveSingleStep(constant_input)[0])

miind.endSimulation()

plt.figure()
plt.plot(activities)
plt.title("Firing Rate.")

plt.show()
