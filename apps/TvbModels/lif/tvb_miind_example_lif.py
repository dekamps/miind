import pylab
import numpy
import matplotlib.pyplot as plt

from tvb.simulator.lab import *
reload(models)
reload(simulator)

time_step = 0.001
simulation_length = 1

# Only the master MPI process (rank=0) will implement the TVB simulation. All
# other processes need only instantiate the Miind model and call configure()
if comm.Get_rank() == 0 :
    # Using the default connectivity as an example : This contains 76 nodes which
    # must be communicated to the Miind model
    white_matter = connectivity.Connectivity(load_default=True)
    white_matter.speed = numpy.array([4.0])

# Currently, the Connectivity.number_of_regions is unavailable when loaded from
# a source file (as with the default) so we manually set 76 here
# Alternatively, len(Connectivity.region_labels) can be used.
#
# The Miind model class constructor takes :
#
# The filename of a shared library in the cwd which implements the MIIND simulation
# The number of nodes in the connectivity
# The simulation length, understood to be in ms
model = models.Miind('libmiindlif.so',76, simulation_length)

if comm.Get_rank() == 0 :
    white_matter_coupling = coupling.Linear(a=0.0154)

    # Many Miind models have a fixed time step defined in the mesh. TVB must match this
    # value to avoid an incorrect time scale.
    # Also note that the Miind model expects the Identity integrator similar to
    # the Wong-Wang model.
    integrator = integrators.Identity(dt = time_step)

    mon_raw = monitors.Raw()
    mon_tavg = monitors.TemporalAverage(period=time_step*20)
    what_to_watch = (mon_raw, mon_tavg)

    sim = simulator.Simulator(model = model, connectivity = white_matter,
                          coupling = white_matter_coupling,
                          integrator = integrator, monitors = what_to_watch)

    sim.configure()

    raw_data = []
    raw_time = []
    tavg_data = []
    tavg_time = []

    for raw, tavg in sim(simulation_length=simulation_length):
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
    plt.title("Simple LIF Neuron Population Mean Firing Rate \nUsing the MIIND Adapter model in TVB")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Time (s)")

    plt.figure(2)
    plt.plot(tavg_time, TAVG[:, 0, :, 0])
    plt.title("Simple LIF Neuron Population Mean Firing Rate \nUsing the MIIND Adapter model in TVB (Time Averaged)")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Time (s)")

    plt.show()
else :
    # If this process is a child process (rank>0) calling configure here is all
    # that's required.
    model.configure()
