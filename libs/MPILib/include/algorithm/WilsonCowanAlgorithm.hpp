/*
 * WilsonCowanAlgorithm.hpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMS_WILSONCOWANALGORITHM_HPP_
#define MPILIB_ALGORITHMS_WILSONCOWANALGORITHM_HPP_

#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/algorithm/WilsonCowanParameter.hpp>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>

namespace MPILib {
namespace algorithm{

/*! \page wilson_cowan The Wilson Cowan Algorithm
 * This page contains the following sections:
 * <ol>
 * <li>\ref wilson_cowan_introduction</li>
 * <li>\ref wilson_cowan_mathematical_structure</li>
 * <li>\ref wilson_cowan_example</li>
 * <li>\ref wilson_cowan_complete</li>
 * </ol>
 *  \section wilson_cowan_introduction Introduction
 *  In 1972 Jack Cowan and Hugh Wilson introduced a very influential method for simulating neural populations. It is used
 * to model a population's firing rate, or population rate. The definition for a population rate is that once observes
 * during a short time interval \f$\Delta t\f$ one observes what fraction of the neurons have spiked during that time interval.
 * This fraction, divided by \f$ \Delta t\f$ is the population rate. Clearly, there is a trade off between the width of the time
 * interval and the accuracy by which the firing rate can be determined. The time window needs to be short in order to capture
 * changes quickly, but the number of neurons that have spiked need to be large enough for an accurate estitmate, so that for a larger
 * population a shorter time interval can be chosen. In modelling infinitely large populations are assumed and the existence
 * of an instantaneous population firing rate can be assumed. In the interpretation of real neuronal data firng rates must be
 * studied with statistical techniques. We refer to the appropriate literature; the algorithm described here can only be used for
 * modelling purposes.
 *
 * In the application of this algorithm it is assumed that a population can be adequately described by itspopulation  firing rate. This
 * is not always true. In section \ref pop_introduction reasons are given for why more sophisticated techniques are sometimes required.
 * A full discussion of when these assumptions are appropriate and when not is outside the scope of this document, but a few rules
 * of the thumb are:
 * <ul>
 * <li>Rapid transients of the order of millisconds can be ignored, the long term (tens of milliseconds) trend of the signal is more important</li>
 * <li>The variability of inputs and outputs can be ignored; information is effectively carried by mean firing rates</li>
 * </ul>
 * In the original paper, Wilson and Cowan made a number of assumptions to arrive at an integro-differential equations. Via a procedure
 * which they called 'time-coarsed' graining (effectively a moving averaging of the signal) they arrived at a set of differential
 * equations. The original assumptions made by Wilson and Cowan are now believed to be incorrect, but equations very similar
 * to theirs emerge from more recent analyses (Gerstner, Einevoll). In modern usage these
 * equations are called ..., so that equations of this kind remain relevant as a biologically plausible way of modelling neuronal
 * dynamics under suitable assumptions. This is fortunate, since there is a vast body of cognitive neuroscience that has used these equations.
 *
 * \section wilson_cowan_mathematical_structure The mathematical structure of Wilson-Cowan equations
 * As an example, consider a single circuit of an excitatory population, an inhibitory population, driven by an external input. This circuit
 * is modelled by the following set of equations:
 *  \f{eqnarray*}{
 *	\tau_E \frac{dE}{dt} & = & -E + f( \alpha E - \beta I + V(t) ) \\
 *	\tau_I \frac{dI}{dt} & = &-I + f( \gamma E - \delta I + V(t) )
 *  \f}
 *  Here \f$E(t)\f$ (\f$I(t)\f$) is the population rate of the excitatory (inhibitory) population in spikes per second. $V(t)$ is the firing
 *  rate of the external input in spikes per second. \f$\alpha\f$, \f$\beta\f$, \f$\gamma\f$ and \f$\delta\f$ are postive constants, to be defined
 *  by the modeller and function \f$ f(x) \f$ is a so-called sigmoidal function. By default this is set by:
 * \f[
 * f(x) = F_{max}\frac{1}{1 + e^{- \beta x}}.
 * \f]
 * As a consequence of this sigmoidal function, the firing rates saturate of a population saturate to a maximal firing rate \f$ F_{max} \f$,
 * no matter how large the input they received. They are also bounded from below by zero, no matter how strong the inhibition they receive.
 * The non-linearity can be controlled by parameter \f$\beta\f$, where a high value makes the sigmoid resemble a step function. The modeller has to specify these constants before the start of the
 * simulation. If the external input \f$ V(t) \f$ is constant throught the simulation, it can be set directly as a parameter, \f$ V \f$
 * of the algorithm. Otherwise, RateFunctor must be used to define a node in the network which provides this input (see
 * section \ref time_varyinginput_to_networks. Each population is also determined by its time constant \f$ \tau \f$. The four parameters together:
 * \f$ F_{max} \tau, \beta, V \f$ are defined in class WilsonCowanParameter and used in the configuration of WilsonCowanAlgorithm.
 *
 * For many purposes the shape of the sigmoid is not too critical, but when a really well motivated sigmoid is required that can be defined
 * in terms of the gain function of a population of leaky-integrate-and-fire neurons (LIF), another version of the Wilson-Cowan can be used:
 * the OUAlgorithm. It is a very useful algorithm to determine steady state firing rates of circuits consisting of LIF neurons, as modelled
 * by NEST simulations, or population density algorithms and is highly useful in determing parameter seetings of such simulations.
 * Please consult the documentation there to find out more about its usage.
 *
 * So why use algorithms with the simple sigmoid described above at all, if more 'realistic' ones exist (such as used by OUAlgorithm)? Two answers:
 * simplicity and a close correspondence to Artificial Neural Networks (ANNs). This correspondence makes it possible to convert ANNs into
 * networks of neural populations with Wilson-Cowan dynamics. The whole of ClamLib is devoted to such conversions, so please
 * consult the documentation there to find out more about it. Wilson-Cowan algorithms are the workhorse of ClamLib. But they also
 * are useful for comparisons to other modelling studies in the literature of (cognitive) neuroscience. All in all, there is sufficient
 * rationale for there existence here.
 *
 * \section wilson_cowan_example A simple example of a simulation with Wilson-Cowan dynamics
 * So, let's conver the set of equations described above into a working example. WilsoCowanAlgorithm must be used in the context
 * of a DynamicNetwork. Since the connection parameters of the network \f$ \alpha, ... \delta \f$ are real numbers, we can represent them
 * by a double. The type of the connection of a network determines the type of the network. The intricacies are described in the
 * the documentation of DynamicNetwork. Here it is sufficient to  note that D_DynamicNetwork can be used to model a network where
 * the connections between populations can be described by a single number, which is represented by a double. (This is not always the case,
 * consult the documentation of OUAlgorithm for an example of connections that are determined by two numbers).
 *
 * First, we will describe a code snippet showing how to simulate a single populations, below we will give a full program for a simple circuit.
 * \dontinclude DynamicLibTest.cpp
 * \skip network_wctest
 * \until ending
 * In this snippet a Wilson-Cowan population is simulated without external input. It is clear that the steady state will quickly converge to
 * \f$ E = f(0) \f$. For our choice of sigmoid \f$ f(0) = \frac{1}{2}F_{max} \f$, hence 50 Hz. The time evolution is shown here:
 *
 * \image html dynamiclib_test_wilsonresponse.png
 *
 * \section wilson_cowan_complete Example: a complete program.
 * The
 * program below described the set up of a fully functional program that can be run to perform a simulation. Much of is self-explanatory,
 * below, we will go through the program and show the output of the program. First, a function is defined which creates a D_DynamicNetwork, representing the connection
 * structure of the equations above:
 * \dontinclude WilsonCowanExample.cpp
 * \skip using
 * \until }
 *  \skipline}
 *
 * As you can see the code is dull and repetetive, but follows the pattern set by the Wilson-Cowan equations closely.
 * Now the simulation must be run. The code to do this is identical to other simulations and very straightforward. Because it is
 * so repetetive, its actually possible to dispense with programming altogether, and to run this simulation from an XML file. (see \ref running_from_XML).
 *
 * In order to run this simulation find the executable
 * WilsonCowanExample in your miind installation. Its code looks as follows:
 * \dontinclude ExampleWilsonCowan.cpp
 * \skip <iostream>
 * \until }
 * \skipline }
 * The results you obtain should be identical to the figures shown here:
 * The simulation
 * results are displayed immediately on a canvas that pops up automatically.
 * The results are shown in this graph:
 * \image html dynamiclib_test_wilsoncowandouble.png
 * Note that this visualisation is very useful in inspecting
 * preliminary simulation results, like here -where the aim is to illustrate an example- and in algorithm development, but is not recommended for bulk processing, as it slows down
 * simulations. It is then recommended to first simulate, and analyze the results later as shown in section \ref \inspecting_simulation_results.
 */

/**
 * @brief The usage of this algorithm is descibed on page \ref wilson_cowan. An example of a fully functional programming
 * using this algorithm is also presented there. Here we present the documentation required by C++
 * clients of this algorithm.
 *
 * This algorithm is defined for usage in MPINetwork. This describes network of nodes connected by link which are
 * described by a single number which internally are represented by a double.
 *
 */

class WilsonCowanAlgorithm: public AlgorithmInterface<double> {
public:
	WilsonCowanAlgorithm();

	WilsonCowanAlgorithm(const WilsonCowanParameter&);

	virtual ~WilsonCowanAlgorithm();

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual WilsonCowanAlgorithm* clone() const;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector, Time time);

	/**
	 * The current timepoint
	 * @return The current time point
	 */
	virtual Time getCurrentTime() const;

	/**
	 * The calculated rate of the node
	 * @return The rate of the node
	 */
	virtual Rate getCurrentRate() const;

	virtual AlgorithmGrid getGrid() const;

private:

	double innerProduct(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector);

	vector<double> getInitialState() const;

	WilsonCowanParameter _parameter;

	NumtoolsLib::DVIntegrator<WilsonCowanParameter> _integrator;

};

} /* namespace algorithm */
} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_WILSONCOWANALGORITHM_HPP_ */
