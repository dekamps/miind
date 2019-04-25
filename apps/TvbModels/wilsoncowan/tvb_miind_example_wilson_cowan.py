import pylab
import numpy
import matplotlib.pyplot as plt

from tvb.simulator.lab import *

white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])

# The Miind_WilsonCowan model allows us to set the initial values of each node
# at t=0. The rest of the simulator history is defined as normal.
# In general, Miind simulations set their own t=0 initial values and ignore any
# initial state set by TVB - this is because Miind defines the intial state in
# terms of density, not firing rate or any other variable.
#
# Currently, the Connectivity.number_of_regions is unavailable when loaded from
# a source file (as with the default) so we manually set 76 here
# Alternatively, len(Connectivity.region_labels) can be used.
numpy.random.seed(0)
E_init = numpy.zeros(76) #numpy.random.rand(76)
I_init = numpy.zeros(76) #numpy.random.rand(76)
inits = numpy.row_stack((E_init, I_init))
initial_conditions = numpy.expand_dims(numpy.expand_dims(inits, axis=0), axis=3)


# The Miind_WilsonCowan model class constructor takes :
#
# The number of nodes in the connectivity
# The simulation length, understood to be in ms
# The simulation time step understood to be in ms
# the initial values as a row stacked 1 dimensional array
model = models.Miind_WilsonCowan(76, 0.2, 0.01, inits)
#model = models.WilsonCowan()

white_matter_coupling = coupling.Linear(a=0.0154)

# dt of the integrator should be defined to match MIIND
integrator = integrators.Identity(dt = 0.01)
#integrator = integrators.EulerDeterministic()

mon_raw = monitors.Raw()
mon_tavg = monitors.TemporalAverage(period=2**-2)
what_to_watch = (mon_raw, mon_tavg)

sim = simulator.Simulator(model = model, connectivity = white_matter,
                          coupling = white_matter_coupling,
                          integrator = integrator, monitors = what_to_watch,
                          initial_conditions = initial_conditions)

sim.configure()

raw_data = []
raw_time = []
tavg_data = []
tavg_time = []

for raw, tavg in sim(simulation_length=10**2):
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
plt.title("Wilson-Cowan Excitatory Population Mean Firing Rate \nUsing MIIND's Wilson Cowan model in TVB")
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (ms)")
plt.show()
