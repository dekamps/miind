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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_DYNAMICLIB_DYNAMICLIB_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DYNAMICLIB_INCLUDE_GUARD


#include "AsciiReportHandler.h"
#include "AlgorithmBuilder.h"
#include "BasicDefinitions.h"
#include "DelayAlgorithmCode.h"
#include "DynamicLibException.h"
#include "DynamicNetworkCode.h"
#include "DynamicNetworkImplementationCode.h"
#include "GraphKey.h"
#include "NodePosition.h"
#include "RateFunctorCode.h"
#include "ReportValue.h"
#include "RootFileInterpreter.h"
#include "RootReportHandler.h"
#include "RootHighThroughputHandler.h"
#include "WeightedLink.h"
#include "WilsonCowanAlgorithm.h"

/*! \page DynamicLib DynamicLib 
 *
 * \section intro_sec Introduction
 * Python users: please read this page first, up to the section on Wilson-Cowan equations. Then consult \ref page_python "the python page for DynamicLib"
 * This section gives an introduction to the usage of DynamicLib and presents two examples: a fairly straightforward
 * one, which shows in general how systems of coupled equations can be solved, and a more complex one with
 * a specific application in neuroscience. This example is particularly relevant in the light  of the work done
 * by La Camera et al. (2005).

 * DynamicNodes derive from AbstractSparseNode. They represent a significant extension: not only can
 * they query other nodes for their activities, they also contain a reference to an AbstractAlgorithm, 
 * which in turn maintains a NodeState. The key idea is that DynamicNodes can evolve their NodeState 
 * by requesting their Algorithms to do so:
 * DynamicNode has an Evolve() method, which calls its Algorithms Evolve() method. The NodeState takes 
 * over the role that the activation value had in SparseNode. The NodeState describes the state of the 
 * node at a certain time $t$ and the Algorithm's Evolve() method evolves that node's state over a time \f$\Delta t\f$, 
 * which is usually small. DynamicNodes, like AbstractSparseNodes
 * maintain a list of nodes that connect to them with a weight for every connection. At every time \f$t\f$, they are able to evaluate the
 * instantaneous contribution of other nodes to itself and that input is passed to the node's Algorithm as a parameter. 
 * 
 * \image html SimulationLoop.png
 *
 * A single DynamicNode is derived from a SparseNode. It has an AbstractAlgorithm (oval), which operates on a NodeState 
 * (rectangle). 
 * When prompted by the simulation loop, the AbstractReportHandler sends the current NodeState to a central file. DynamicNodes are almost 
 * autonomous. The central simulation loop determines which Node is in line to evolve its NodeState over a short time step, but the Nodes 
 * themselves collect input from other Nodes and deliver this to their own Algorithms which evolve the Node's NodeState. This setup is easy 
 * to parallelise.
 *	
 * At the highest level DynamicNetwork's Evolve() method initiates a loop over all nodes, the simulation loop, 
 * in which it requests that every DynamicNode evolve itself over a short period of time. 
 * The DynamicNetwork does this repeatedly and in such a way a simulation of the network dynamics emerges. 
 * A DynamicNode is also configured with a ReportHandler. At fixed times, the simulation loop queries the
 * DynamicNodes for a Report. The ReportHandler of the DynamicNode delivers the
 * Report and the Reports are written to disk, so that a record of the simulation is produced.
 * Also, the simulation loop maintains a log file to indicate how far the simulation has progressed and to keep a record
 * of exceptional conditions that occured during simulation.  In the Figure above we show a graphical representation
 * of the classes involved in DynamicNetwork.
 *
 * Possibly, this sounds somewhat abstract, so we give an example first.
 *
 * \section  example_sec Example: Modelling Wilson-Cowan equations 
 *
 *
 * Consider a network which consists of two populations, one of which is  described by the Equation below:
 *
 * \f$
 *    \tau \frac{d E}{dt}  =  -E + f(\alpha E  + \epsilon \nu)
 * \f$
 * and one of which simply maintains a fixed output rate and serves as an external population to the network.
 *
 * \dontinclude DynamicLibTest.cpp
 * \skip DynamicLibTest::WilsonCowanTest
 * \until ending
 *
 * In the example above we show how a network is set up. 
 * First, the WilsonCowanAlgorithms is defined and configured with the appropriate WilsonCowanParameter
 * parameter, which defines the sigmoid parameters. 
 * Also, a network needs input: therefore a RateAlgorithm is created, an algorithm whose only action is to
 * set the  NodeState of the DynamicNode to which it belongs  to a fixed rate (the NodeState consist of a 
 * single floating point value in this case).
 * The nodes are then created in the DynamicNetwork, with their own copy of the WilsonCowanAlgorithm 
 * (or RateAlgorithm).
 * A user receives a NodeId as a reference to DynamicNode that just has been created in the DynamicNetwork.
 * These NodeIds can then be used
 * to define Connections in the network. After the definition of the DynamicNetwork, one only has to 
 * Configure it and to Evolve it.
 *
 * In this code a standard sigmoid is used: a function of the form:
 * \f$
 * f(x) = \frac{f_{max}}{1 + e^{-\beta x}},
 * \f$
 * where \f$ f_{max} \f$ is the maximum response of the node and \f$\beta\f$ is a noise parameter. In the code above
 * one can see how these parameters are set. There is no need to call or define
 * numerical integrators from the user's point of view. 
 * Setting up large networks is a trivial exercise. It just comes down to using AddNode and MakeFirstInputOfSecond
 * repeatedly. A very large network which
 * was to model a neuronal architecture for compositional representations \cite{vdvelde2006} is shown in the Figure below.
 * \image html BBS.png
 * \section sec_example2 Modeling steady states of spiking neurons: dyadic connections
 * Wilson-Cowan equations are perhaps the most widely used modeling technique in large-scale network modeling. It has been shown that 
 * Wilson-Cowan equations adequately describe the trend in the activity of a group of neurons reasonably well \cite{gerstner1995}, but not 
 * the transient dynamics\footnote{This is due to a procedure called time coarse graining, which was used to reduce integral equations to the 
 * kind of differential equations that are nowadays commonly referred to as Wilson-Cowan equations}. Also, Wilson-Cowan techniques contain a 
 * sigmoid for which the original motivation \cite{wilson1972} is not considered to be valid anymore.
 * \citeA{amit1997a} introduced a modeling technique which describes networks of spiking (LIF) neurons in terms of their steady state. 
 * It is possible to derive these equations from first principles, under some reasonably plausible assumptions about connectivity and firing 
 * rates in cortex. So states in these networks actually describe steady-state activity of networks of spiking LIF neurons as accurately as 
 * direct simulations would.
 * The response of a neuronal population is not only determined by the average input,
 * as in ANNs and in Wilson-Cowan equations of the kind discussed above, but also by the {\it variability} of the input. This reflects the
 * fact the input is assumed to consists of stochastic spike trains which are assumed to be Poisson distributed. 
 * Clearly, this is a step up in neuronal realism.  This is one reason for demonstrating the solution of these equations with
 * DynamicLib. Another important reason is that it demonstrates the flexibility of DynamicLib in
 * handling connections of any kind: as we will see connections for these kind of networks are dyadic, 
 * a single connection is determined by two floating point numbers.
 * The first number is the average efficacy from a neuron in one population to another and the second is the effective number of 
 * connections between the two populations. DynamicLib is able to handle connections of any type due to C++'s template mechanism. 
 * We will now present the Equations used by \citeA{amit1997a}:
 * \f$
 *  \nu_i = \phi_i( \mu_i , \sigma_i),
 * \f$
 * where:
 * \f$
 *   \phi_i( \mu_i, \sigma_i) \equiv \left\{ \tau_{ref,i} +\sqrt{\pi} \tau_i \int^{ \frac{ \theta_i - \mu_i}{\sigma_i} }_{ \frac{V_{reset,i} - \mu_i}{\sigma_i} }
 *    du \left[ 1 + \mbox{erf}(u) \right] e^{u^2} \right\}^{-1} 
 * \f$ and \newline
 * \f$
 * \mu_i     =  \tau_i \sum_j J_{ij}N_{ij}\nu_j, 
 * \f$
 * \newline
 * \f$
 * \sigma_i  =  \sqrt{ \sum_j \tau_i J^2_{ij}N_{ij} }.
 * \f$
 *
 * \f$ \tau_i \f$, \f$\tau_{ref,i}\f$ are the membrane time constant  and   the absolute refractory period, respectively, 
 * in s, \f$\theta_i\f$ and \f$V_{reset,i}\f$ the threshold potential and the reset potential, respectively,  in V, all for neurons 
 * in population \f$i\f$. 
 * \f$N_{ij}\f$ is the effective number of neurons from population \f$j\f$ seen by  a neuron in population \f$i\f$ and $J_{ij}$ the average efficacy 
 * from a spike in population \f$j\f$ on a neuron 
 * in population \f$i\f$ in V. These equations form a closed system which can be solved in $\nu_{i}$.
 *
 * In practice, one does this by introducing a 
 * pseudo-dynamics:
 * \f$\tau_i  \frac{d \nu_i}{dt} = -\nu_i + \phi(\mu_i, \sigma_i) \f$
 * and selecting initial values $f\nu_i(0)$\f. The script below shows how this is done. The main difference is the use
 * of dyadic connections: a connection is determined by a tuple \f$(N, J)\f$, as shown above. Therefore the type
 * of the connections used in this example is slightly different. A fragment shows the differences:
 *
 * \code
 * // Note we now need an OU_Network instead of a D_Network
 *	OU_Network net;
 *
 *	Potential sigma = 2e-3;
 *	Potential mu    = 20e-3;
 *
 *	Time tau = PARAMETER_NEURON._tau; 
 *	Rate nu = mu*mu/(sigma*sigma*tau);
 *	Rate J = sigma*sigma/mu;
 *
 *	OU_Connection 
 *		con
 *		(
 *			1,
 *			J
 *		);
 *
 *	// Define a node with a fixed output rate
 *	OU_RateAlgorithm rate_alg(nu);
 *	NodeId id_rate = net.AddNode(rate_alg,EXCITATORY);
 *
 *	// Define the receiving node 
 *	OU_Algorithm algorithm_exc(PARAMETER_NEURON);
 *	NodeId id = net.AddNode(algorithm_exc,EXCITATORY);
 *
 *	// connect the two nodes
 *	net.MakeFirstInputOfSecond(id_rate,id,con);
 *
 *	// define a handler to store the simulation results
 *	RootReportHandler 
 *		handler
 *		(
 *			"test/ouresponse.root",	// simulation results 
 *			false,		// do not display on screen
 *			true		// write into file
 *		);
 *
 *	SimulationRunParameter
 *		par_run
 *		(
 * 			handler,	// the handler object
 *			1000000,	// maximum number of iterations
 *			0,		// start time of simulation
 *			0.1,		// end time of simulation
 *			1e-4,		// report time
 *			1e-4,		// update time
 *			1e-5,		// network step time
 *			"test/ouresponse.log"   // log file name
 *		);
 *
 *	bool b_configure = net.ConfigureSimulation(par_run);
 *
 *	bool b_evolve = net.Evolve();
 *	// ending
 *  \endcode 
 */




#endif // include guard

