// Copyright (c) 2005 - 2011 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_POPULISTLIB_POPULISTLIB_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_POPULISTLIB_INCLUDE_GUARD

#include "AdaptationParameter.h"
#include "BasicDefinitions.h"
#include "CharacteristicAlgorithmCode.h"
#include "PopulationAlgorithm.h"
#include "IntegralRateComputation.h"
#include "InterpolationRebinner.h"
#include "NonCirculantSolver.h"
#include "OneDMAlgorithm.h"
#include "OneDMParameter.h"
#include "ParseResponseCurveMetaFile.h"
#include "PopulationAlgorithmCode.h"
#include "ProbabilityQueue.h"
#include "QIFAlgorithmCode.h"
#include "Response.h"
#include "RefractiveCirculantSolver.h"
#include "WorkFlowAnalyzer.h"
#include "WorkFlowTest.h"

/*! \page PopulistLib PopulistLib
 * \section Introduction
 *
 * A general overview of population density methods is given by Omurtag et al. (2001) http://www.springerlink.com/content/j8327pq70t1355l5/.
 * The numerical implementation of the algorithms is due to de Kamps (2003,2006,2008).
 * In most neural simulators individual neurons are simulated. Often, these individual neurons are placed in groups called populations and in many cases
 * the behaviour of populations and the network as a whole is more important than that of individual neurons. Population level simulations can be
 * carried out with neural simulators such as NEURON, GENESIS or NEST. Populations in these simulators are created by simulating identical versions
 * of individual neurons in groups and providing them with input that has the same statistics. Population behaviour is than calculated by aggregating
 * population measures of individual neuron statistics. For example, the population firing rate is calculated in terms of the firing rate of individual
 * neurons in the group by counting the number neurons that spike during a brief time interval, and dividing them by the total number of neurons in the
 * population and the duration of the yime interval. This gives the so-called population rate. To calculate population quantities, then, requires the
 * simulation of a large number of individual neural instances, which is often expensive in terms of computational resources, and aggregating information
 * on individual neurons to obtain population measures. In complex networks of populations this information is often hard to obtain from the
 * neural simulator. Fortunately, it sometimes is possible to model population measures directly, without resorting to the simulation of individual neurons.
 * Wilson-Cowan dynamics is one example of this and discussed elsewhere. Importantly, Wilson-Cowan dynamics is defined in terms of a single population
 * measure, say its firing rate. This makes it hard to simulate some dynamical aspects of population behaviour accurately. It does have its place
 * in cognitive modelling however, as it elucidates network behaviour in simple terms, and may be biologically plausible if rapid transients are not
 * important. Wilson-Cowan dynamics is discussed in the section \ref wilson_cowan. If Wilson-Cowan dynamics is not adequate, for example if rapid
 * transients are important, population density methods may be more appropriate. 
 *
 * As the name suggests, population density methods characterise a entire population by a single density function \f$ \rho(\vec{v} ) \f$, where \f$ vec{V} \f$  is
 * the state space of an individual neuron. For leaky-integrate-and-fire neurons this is simply the neuron's membrane potential. The evolution of the density function
 * is governed by a partial differential equation. Solving this equation is often computationally more efficient than simulating thousands of neurons.
 * Calculating population measures from the density function is cheap and simple. Under certain assumptions about the input, the partial differential
 * equation is equivalent to that of a diffusion process. The use of Fokker Planck equations for leaky-integrate-and-fire neurons is probably the best
 * known example of population density techniques. The unique aspect of MIIND's population density techniques are that the methods work outside the
 * diffusion regime, i.e. for low firing rates and large synaptic efficacies, and that they are not necessarily restricted to leaky-integrate-and-fire
 * neurons. For example, we provide an implementation for the Izhikevich neuron.

 * In the MIIND documentation we discuss three aspects of the methods:
 * <ol> <li>\subpage pop_background. This is a high level overview of population density technques. It explains the basic method and assumptions that go into the technique. It explains
 * different use scenarios: \ref pop_balanced, which is related to use of Fokker-Planck equations, \ref pop_single and \ref pop_mixed.</li>
 * <li>\subpage pop_model_programs This discusses demonstration programs for the different use scenarios. </li>
 * <li>\subpage pop_code  This discusses the design and implementation of the code.</li>
 * </ol>
 *
 * \page pop_background A brief introduction to population density methods
 * \section A mathematical description of the PopulationAlgorithm
 *
 * In its simplest form the  PopulationAlgorithm models the population density function of a group of leaky-integrate-and-fire neurons (LIF).
 * LIF neurons are determined by single state variable: membrane potential \f$ V \f$. It is assumed that \f$ V_{min} \le V \le V_{th} \f$, where
 *  \f$ V_{th}\f$ is
 * the threshold potential of the neuron is a proper property of the neuron. The minimum potential that neurons
 * can achieve, \f$ V_{min}\f$ is determined by practical considerations: the LIF dynamics tends to drive
 * the membrane potential towards equilibrium value \f$ V_{rev} \f$, and this makes it hard for neurons to reach extreme negative values. A sensible
 * setting for \f$ V_{min} \f$ can easily be found in practice.
 *
 * A population of LIF neurons is determined completely by a density function \f$ \rho(V) \f$. The basis for most work on the algorithm is the following
 * assumptions, which in themselves are unrealistic (we will show later how the algorithm can be used in realistic settings):
 * <ol>
 * <li> Populations are homogeneous and see, statistically, the same input.</li>
 * <li> Each population receives spike trains which are Poisson distributed.</li>
 * <li> Poisson distributions are inhomogeneous.</li>
 * <li> Each spike that the population receives causes the postsynatiptic potential of a neuron in the population to rise by a fixed amount \f$ h \f$ </li>
 * </ol> 
 *
 * Under these assumptions the evolution of the density \f$ \rho(v) \f$ is given by a partial differential equation:
 *
 * The output firing rate, \f$ f \f$, is now completely determined by the density function \f$ \rho(v) \f$ and the frequency of the population's input spike train:
 *
 * Although complex, these equations specify completely how PopulationAlgorithm should behave. The algorithm receives input from other nodes
 * 
 * \section sec_pop_background_parameters The parameters required for the PopulationAlgorithm
 *
 * \section pop_background_intro Introduction
 * In this section we will discuss the following examples
 * <ul>
 *     <li> A single population of leaky-integrate-and-fire neurons. The input rate = 800 Hz and each input spike causes an instantaneous jump
 *           in the postsynaptic potential of 3% of the threshold potential (more accurately \f$ h = 0.03(V_{th} - V_{rev}) \f$, see below for definitions).
 *			Although unrealistic - at the very least one would expected a distribution of postsynaptic potential jumps- this provides a good introduction
 *			in the concepts of population density modelling as well at the code. The later examples are progressively more realistic.</li>
 *		<li>Balanced excitation-inhibition. A population received high rate input from excitatory and inhibitory populations, so that the input effectively
 *		approaches Gaussian white noise. The program demonstrates how to set this up. The results can be used for validation against analytic results.</li>
 * </ul>
 *
 * \page pop_code Design and implementation of the C++ code for popuation density techniques
 *
 * \section pop_code_intro Introduction
 *
 * Population density techniques are run within the DynamicLib framework. So, implementating a population density algorithm requires a class which
 * derives from AbstractAlgorithm, which is called PopulationAlgorithm and needs a number of parameters, which are collectively in class
 * PopulistParameter. The code in \ref pop_example_single illustrates the point. To create a network, first one needs to create nodes,
 * each node needs to be configured with an algorithm. In the example one node is created with a RateAlgorithm, which makes the node's activity
 * behave according to a predefined function, and therefore can be used as input to the network. The second node is created with a Populationalgorithm.
 * The nodes are added to the network, and connections between the nodes are defined in the network. This conforms closely to the simpler simulations
 * shown in DynamicLib. There are three points different here, and you can see them reflected in the example program, shown in section \ref
 *  pop_model_programs_single.
 *
 * <ol>
 * <li> The connections between nodes are no longer represented by a double, but by a struct containing two numbers: PopulationConnection. The reason
 * is that connections between two neuron populations are usually not characterised by a single double, but by two or even three numbers.</li>
 * <li> Extra parameters for the PopulationAlgorithm are required.</li>
 * <li> Visualisation of the population density during running, i.e. showing the state as well as the firing rate of a populations.</li>
 * </ol>
 * This determines the position that PopulationAlgorithm takes in the DynamicNetwork hierarchy: it is a normal AbstractAlgorithm, but one that
 * requires a different type for connections than the often used double. In the description of DynamicNetwork you see that this class has a template
 * argument, which specifies this type. Before we discuss the implications, we must now discuss the mathematical form of the algorithm, and how
 * it can be used in different scenarios.

 * \section sec_pop_pro_type The type of  the network capable of PopulationAlgorithm.
 *
 * Remember that a DynamicNetwork has a template argument. Often this is a double, and a DynamicNetwork<double> is typedefed to D_Network.
 * This template argument determines the type of the connections between nodes. 
 *
 * \page pop_model_programs Example Programs
 * \section pop_model_programs_single A population receiving a Poisson distributed input spike train with fixed post synaptic efficacies
 * \dontinclude PopulistExample.cpp
 * \skip main
 * \until return 
 * \line }
 *
 * 
 * \section Balanced excitation-inhibition
 *
 * The piece of code describes balanced excitatory-inhibitory input. Code for networks for with balanced
 * excitation-inhibition is well developed, but may have to be adapted to your requirements, and the documentation for 
 * this library is still patchy.
 * To give you a flavour of what can be done with this library, we show two pieces of example code that should work
 * straight out of the box (the only thing you wll have to do is to include the relevant headers), and which 
 * are part of MIIND's test suite. Python examples are shown on the
 * \ref page_python_population "Populist Python page".
 *
 * The first example, taken straight from the test suite shows how a Gaussian white noise is fed into a leaky-integrate-and-fire
 * population. The population is effectively described by an Ornstein-Uhlenbeck process.
 *
 *
 *! \page page_python_population The Populist Python page
 *
 * \section section_py
 * Below is the Python script which gives identical results to the C++ program for balanced excitatory-inhibitory input
 *
 * \code
 *
 * import Populist
 * from Populist import *
 * # Set the neuron parameters of the output population
 * #
 * par=OrnsteinUhlenbeckParameter()
 * par._theta      = 20e-3
 * par._tau        = 10e-3
 * par._V_reversal = 0.0
 * par._V_reset    = 0.0
 * #
 * # Set mu and sigma
 * mu    = 17e-3       # (V)
 * sigma = 2e-3        # (V)
 * #
 * # In order to approximate a diffusion process set a small value for input
 * # weights (small relative to theta).
 * #
 * J= 0.01* par._theta
 * #
 * # Now convert mu and sigma to input rates of an excitatory and inhibitory
 * # population.
 * #
 * nu_e = (J*mu + sigma*sigma)/(2*J*J*par._tau)
 * nu_i = (sigma*sigma - J*mu)/(2*J*J*par._tau)
 * #
 * # some parameters specific to the algorithm
 * #
 * V_min  = -10e-3
 * n_bins = 10000
 * n_add  = 1
 * f_exp  = 1.1
 * rebinner = InterpolationRebinner()
 * ratealg  = IntegralRateComputation()
 * density = InitialDensityParameter(par._V_reversal,0)
 * par_spec = PopulistSpecificParameter(V_min,n_bins,n_add,density,f_exp,rebinner,ratealg,SINGLE_POPULATION_MODE)
 *
 * #
 * # Now create the network
 * #
 * net = Pop_Net()
 * par_pop = PopulistParameter(par,par_spec)
 * alg_pop = PopulationAlgorithm(par_pop)
 * id_pop  = net.AddNode(alg_pop,EXCITATORY)
 * #
 * # Create input populations and add them to the network
 * #
 * alg_rate_exc=OURateAlgorithm(nu_e)
 * id_e = net.AddNode(alg_rate_exc,EXCITATORY)
 * con_e = OrnsteinUhlenbeckConnection(1,J)
 * alg_rate_inh=OURateAlgorithm(nu_i)
 * id_i = net.AddNode(alg_rate_inh,INHIBITORY)
 * con_i = OrnsteinUhlenbeckConnection(1,-J)
 * 
 * net.MakeFirstInputOfSecond(id_e,id_pop,con_e)
 * net.MakeFirstInputOfSecond(id_i,id_pop,con_i)
 * #
 * handler=RootReportHandler("data.root",1,1)
 * handler.SetFrequencyRange(0,20)
 * handler.SetDensityRange(-0.01,10)
 * handler.SetTimeRange(0,0.3)
 * handler.SetPotentialRange(-0.001,0.020)
 * handler.AddNodeToCanvas(id_pop)
 * #
 * # Configure the simulation
 * #
 * par_run = SimulationRunParameter(handler,100000,0.,0.3,1e-3,1e-3,1e-3,"simulation.log")
 *
 * net.ConfigureSimulation(par_run)
 * net.Evolve()
 *
 * \endcode
 */
#endif // include guard
