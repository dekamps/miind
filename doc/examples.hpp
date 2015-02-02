/**
\page page_examples Example

\section example_intro Introduction
We will first show an example using leaky-integrate-and-fire neurons. Quadratic-integrate-and-fire neurons can be handled
in a similar way.

This example comes from Omurtag A., Knight B. W., Sirovich L,   On the simulation of large populations of neurons   <i>J. of Comp. Neurosc.</i> (2000) ; 8(1):51-63

Neural circuits can be simulated in terms of models of individual neurons, but such simulations quickly become intractable. Usually
some concessions are made: populations are considered to be homogeneous and relatively large, i.e. at least several hundreds of neurons.
Modern simulators like NEST and BRIAN (and the interface PyNN) offer considerable support to create populations, but such simulations tend to get
large and unwieldy, not least because after simulation a substantial amount of post-processing needs to be done.

It is possible to use to so-called rate-based or neural mass models to simulate a neural circuit. This is a valuable approach, but sacrifices
the direct connection between the spiking neurons in the populations and population-level aggregates. It is not always clear that this is 
possibe; we know of population behaviour of spiking neurons that can not be represented well by neural mass models.

Population density techniques can render the behaviour of a population of spiking model neurons accurately, if the population is
large enough. It is worth pointing out that much of our under standing of populations of spiking neurons and circuits comprised of them
comes from population density analyses.  Reecent progress means that many neural (point) model neurons can be studied, not just leaky-integrate-and-fire 
(LIF) neurons. The case for these techniques, and an overview of their potential and  their limitations is made elsewhere. Here, we present a few
simple examples, and the emphasis in understanding how to reproduce them using MIIND.


Consider a population of leaky-integrate-and fire neurons that is at rest initially, i.e. their neurons are at equilibrium potential.
At $t=0$ all neurons start to receive a Poisson spike train. In NEST such a simulation can be easily created, and the results can be
represented as a spike raster:

\image html spikeraster.png

It can be seen that the population initially does not respond, then fires a volley, followed by a period of relative rest. The population activity
was correlated due to the common onset of its input but decorrelates over time. It settles in a state where individual neurons fire irregularly,
but the population as a whole has a well defined steady state output.

Population density techniques can calculate all quantities that are defined at the population level, e.g. the population firing rate. When
the spike raster is converted to a population firing rate, a transient response is visible, indictade by the red markers in the Figure below.

\image html rate.png

The response can also be calculated using population density techniques, which fits the firing rate calculated from the NEST simulation very well.
Population density techniques essentially calculate a histogram over the neuron's state space. All other population-level quantities can be derived
from this. The figure below shows a histogram of the neuron membrane potential for the entire population. Again, the markers indicate a NEST
simulation, the black line indicates a population density calculation.

\image html dense.png

The result demonstrates a fundamental property of population density techniques: when the identity of indivdual spikes doesn't matter (i.e.
the precise neuron from which they originated and the precise destination are irrelevant) and only population-level quantities are of interest,
population density techniques are equivalent to simulation of populations of spiking neurons.

There are two versions of this example. You are strongly recommended to consider the XML example first, as it is nearly self documenting.
-  \subpage sub_page_xml_example
-  \subpage sub_page_cpp_example

*/
