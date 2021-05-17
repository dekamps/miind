import pylab
import numpy
import matplotlib.pyplot as plt
import miind.miindsim as miind

efficacies = [0.01 * (n+1) for n in range(10)]
rates = [1200 / (n+1) for n in range(10)]
firing_rates = [0 for n in range(10)]

for i in range(len(efficacies)):
    miind.init("lif.xml", efficacy=str(efficacies[i]))
    ts = miind.getTimeStep()
    ln = miind.getSimulationLength()
    miind.startSimulation()
    
    rs = []
    for j in range(int(ln / ts)):
        rs.append(miind.evolveSingleStep([rates[i]])[0])

    miind.endSimulation()
    
    firing_rates[i] = rs[-1]

plt.figure()
plt.plot(firing_rates)
plt.title("Firing Rate.")
plt.show()
