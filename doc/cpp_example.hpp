/**
 * \page sub_page_cpp_example A C++ Example
 *
 * \section sec_cpp_example_introduction Introduction
 * We will replicate the experiment discussed in section \ref page_examples. The original paper considered a large population of leaky-integrate-and-fire
 * (LIF) neurons. Membrane parameters are rescaled and dimensionless: reset and reversal potential are 0, threshold is one. 
 * Each neuron in this population receives a Poisson distributed spike train of 800 Hz. The synapses are delta-synapses with
 * a synaptic efficacy of $h=0.03$, i.e. each inciming spike raises the membrane potential of the postsynaptic neuron by 3 percent
 * of the threshold potential. All neurons are at rest initially. Input is switched on at $t=0$. <i>Each neuron sees spike trains that
 * are individually different</i>. Such a simulation can easily be set up in NEST or BRIAN and the results can be inspected in the spike
 * raster of section \ref page_examples. The set up in MIIND is not very different. The code for this example will be compiled upon installation
 * of MIIND. The code can be found in the apps/BasicDemos directory resides in file population-example.cpp; the resulting executable is called
 * populationDemo.
 *
 * \include population-example.cpp
 */
