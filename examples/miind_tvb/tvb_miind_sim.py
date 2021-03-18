import pylab
import numpy
import matplotlib.pyplot as plt

from tvb.simulator.lab import *

import miind.TvbMiindModel as tvbmiind

model = tvbmiind.Miind("izh.xml", 76, duplicate_connections=True)

white_matter = connectivity.Connectivity.from_file()
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=numpy.array([0.0154]))

# dt of the integrator should be defined to match MIIND
integrator = integrators.Identity(dt = model.miind_time_step)
#integrator = integrators.EulerDeterministic()

mon_raw = monitors.Raw()
mon_tavg = monitors.TemporalAverage(period=2**-2)
what_to_watch = (mon_raw, mon_tavg)

sim = simulator.Simulator(model = model, connectivity = white_matter,
                          coupling = white_matter_coupling,
                          integrator = integrator, monitors = what_to_watch)

sim.configure()

raw_data = []
raw_time = []
tavg_data = []
tavg_time = []

for raw, tavg in sim(simulation_length=model.simulation_length):
    if not raw is None:
    	raw_time.append(raw[0])
    	raw_data.append(raw[1])
    if not tavg is None:
    	tavg_time.append(tavg[0])
    	tavg_data.append(tavg[1])

RAW = numpy.array(raw_data)
TAVG = numpy.array(tavg_data)

plt.figure(1)
plt.plot(raw_time, RAW[:, 0, :, 0])
plt.title("Default TVB whole-brain connectivity using populations \n of Izhikevich simple neurons simulated in MIIND.")
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (s)")
plt.show()
