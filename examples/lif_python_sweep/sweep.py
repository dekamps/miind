import pylab
import numpy
import matplotlib.pyplot as plt

efficacies = [0.01 * (n+1) for n in range(10)]
rates = [1200 / (n+1) for n in range(10)]
firing_rates = [0 for n in range(10)]

for n in range(10):
	import runlif
	firing_rates[n] = runlif.simulation_run(efficacies[n], rates[n])

plt.figure()
plt.plot(firing_rates)
plt.title("Firing Rate.")
plt.show()
