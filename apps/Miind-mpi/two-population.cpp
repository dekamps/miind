/*
 * two-population.cpp
 *
 *  Created on: 28.06.2012
 *      Author: david
 */

#include <MPILib/include/MPINodeCode.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/populist/RateFunctor.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/PopulistParameter.hpp>
#include <MPILib/include/populist/InitialDensityParameter.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/populist/PopulationAlgorithm.hpp>

namespace mpi = boost::mpi;
using namespace MPILib;

const Rate RATE_TWOPOPULATION_EXCITATORY_BACKGROUND = 2.0; // Hz
const double TWOPOPULATION_FRACTION = 0.5;

const double TWOPOPULATION_C_E = 20000;
const double TWOPOPULATION_C_I = 2000;

const Efficacy TWOPOPULATION_J_EE = 20e-3 / 170.0;
const Efficacy TWOPOPULATION_J_IE = 20e-3 / 70.15;

const double g = 3.0;

const Efficacy TWOPOPULATION_J_EI = g * TWOPOPULATION_J_EE;
const Efficacy TWOPOPULATION_J_II = g * TWOPOPULATION_J_IE;

const Time TWOPOPULATION_TIME_BEGIN = 0; // 0 sec
const Time TWOPOPULATION_TIME_END = 0.05; // 1 sec
const Time TWOPOPULATION_TIME_REPORT = 1e-3; // 10 ms
const Time TWOPOPULATION_TIME_UPDATE = 1e-2; // 100 ms
const Time TWOPOPULATION_TIME_NETWORK = 1e-6; // 0.1 ms

const populist::OrnsteinUhlenbeckParameter TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER(
		20e-3, // V_threshold: 20 mV
		0,     // V_reset: 0 mV
		0,     // V_reversal
		2e-3,  // tau refractive
		10e-3  // tau membrane; 10 ms
		);

const populist::OrnsteinUhlenbeckParameter TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER(
		20e-3,  // V_threshold; 20 mV
		0,      // V_reset: 0 mV
		0,      // V_reversal
		2e-3,   // tau refractive
		3e-3    // tau membrane 3 ms
		);

const populist::InitialDensityParameter TWOPOP_INITIAL_DENSITY(0.0, 0.0);

const Number TWOPOP_NUMBER_OF_INITIAL_BINS = 550;
const Number TWOPOP_NUMBER_OF_BINS_TO_ADD = 1;
const Number TWOPOP_MAXIMUM_NUMBER_OF_ITERATIONS = 1000000;

const double TWOPOP_EXPANSION_FACTOR = 1.1;

const double SIGMA = 2.0e-3F;
//! ResponseParameterBrunel
//! parameter as in Amit & Brunel (1997)
struct ResponseParameterBrunel {

	double mu;
	double sigma;
	double theta;
	double V_reset;
	double V_reversal;
	double tau_refractive;
	double tau;
};

const ResponseParameterBrunel RESPONSE_CURVE_PARAMETER = { 0,		// mu
		SIGMA,	// sigma
		20e-3F,	// theta
		10e-3F,	// V_reset
		0,		// V_reversal
		0.004F,	// tau ref
		0.020F	// tau exc
		};

const Potential TWOPOP_V_MIN = -1.0 * RESPONSE_CURVE_PARAMETER.theta;

const populist::PopulistSpecificParameter TWOPOP_SPECIFIC(TWOPOP_V_MIN,
		TWOPOP_NUMBER_OF_INITIAL_BINS, TWOPOP_NUMBER_OF_BINS_TO_ADD,
		TWOPOP_INITIAL_DENSITY, TWOPOP_EXPANSION_FACTOR,
		"NumericalZeroLeakEquations");

const populist::PopulistParameter TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP(
		TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER, TWOPOP_SPECIFIC);

const populist::PopulistParameter TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP(
		TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER, TWOPOP_SPECIFIC);

inline Rate CorticalBackground(Time t) {
	return RATE_TWOPOPULATION_EXCITATORY_BACKGROUND;
}

template<class Algorithm, class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution> CreateTwoPopulationNetwork //Edited by Johannes: namespacequalifier removed
(
		NodeId* p_id_cortical_background, //
		NodeId* p_id_excitatory_main, //
		NodeId* p_id_inhibitory_main, //
		NodeId* p_id_rate, //
		const typename Algorithm::Parameter& par_exc,
		const typename Algorithm::Parameter& par_inh //
		) {
	MPINetwork<WeightValue, NodeDistribution> network;

	// Create cortical background, and add to network
	populist::RateFunctor<WeightValue> cortical_background(CorticalBackground);
	*p_id_cortical_background = network.AddNode(cortical_background,
			EXCITATORY);

	// Create excitatory main population
	Algorithm algorithm_excitatory(par_exc);
	*p_id_excitatory_main = network.AddNode(algorithm_excitatory, EXCITATORY);

	// Create inhibitory main population
	Algorithm algorithm_inhibitory(par_inh);
	*p_id_inhibitory_main = network.AddNode(algorithm_inhibitory, INHIBITORY);

	// Background and excitatory connection only differ in x, 1- x
	WeightValue connection_J_EE_BG(
			TWOPOPULATION_C_E * (1 - TWOPOPULATION_FRACTION),
			TWOPOPULATION_J_EE);

	network.MakeFirstInputOfSecond(*p_id_cortical_background,
			*p_id_excitatory_main, connection_J_EE_BG);

	// Excitatory connection to itself

	WeightValue connection_J_EE(TWOPOPULATION_C_E * TWOPOPULATION_FRACTION,
			TWOPOPULATION_J_EE);

	network.MakeFirstInputOfSecond(*p_id_excitatory_main, *p_id_excitatory_main,
			connection_J_EE);

	// Background connection to I

	WeightValue connection_J_IE_BG(
			static_cast<Number>(TWOPOPULATION_C_E * (1 - TWOPOPULATION_FRACTION)),
			TWOPOPULATION_J_IE);

	network.MakeFirstInputOfSecond(*p_id_cortical_background,
			*p_id_inhibitory_main, connection_J_IE_BG);

	// E to I
	WeightValue connection_J_IE(
			static_cast<Number>(TWOPOPULATION_C_E * TWOPOPULATION_FRACTION),
			TWOPOPULATION_J_IE);

	network.MakeFirstInputOfSecond(*p_id_excitatory_main, *p_id_inhibitory_main,
			connection_J_IE);

	// I to E
	WeightValue connection_J_EI(TWOPOPULATION_C_I, -TWOPOPULATION_J_EI);

	network.MakeFirstInputOfSecond(*p_id_inhibitory_main, *p_id_excitatory_main,
			connection_J_EI);

	// I to I
	WeightValue connection_J_II(TWOPOPULATION_C_I, -TWOPOPULATION_J_II);

	network.MakeFirstInputOfSecond(*p_id_inhibitory_main, *p_id_inhibitory_main,
			connection_J_II);

	return network;

}

const report::handler::RootReportHandler TWOPOP_HANDLER("test/twopoptest", true	// in file
		);

const Time TWOPOP_T_BEGIN = 0;
const Time TWOPOP_T_END = 0.05;
const Time TWOPOP_T_REPORT = 1e-5;
const Time TWOPOP_T_NETWORK = 1e-5;

const SimulationRunParameter TWOPOP_PARAMETER(TWOPOP_HANDLER,
		TWOPOP_MAXIMUM_NUMBER_OF_ITERATIONS, TWOPOP_T_BEGIN, TWOPOP_T_END,
		TWOPOP_T_REPORT, TWOPOP_T_REPORT, TWOPOP_T_NETWORK, "test/twopoptest");

int main(int argc, char* argv[]) {

	try {

		NodeId id_cortical_background;
		NodeId id_excitatory_main;
		NodeId id_inhibitory_main;
		NodeId id_rate;
		MPINetwork<OrnsteinUhlenbeckConnection, utilities::CircularDistribution> network =
				CreateTwoPopulationNetwork<
						populist::PopulationAlgorithm_<
						populist::OrnsteinUhlenbeckConnection>,
						populist::OrnsteinUhlenbeckConnection,
						utilities::CircularDistribution>(
						&id_cortical_background, &id_excitatory_main,
						&id_inhibitory_main, &id_rate,
						TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP,
						TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP);



		try {
			network.configureSimulation(TWOPOP_PARAMETER);

			network.evolve();
		} catch (std::exception&e) {
			std::cout << e.what() << std::endl;
		}

		boost::timer::auto_cpu_timer te;
		te.start();

		//timed calculation
		world.barrier();
		te.stop();

		if (world.rank() == 0) {

			std::cout << "Time of Envolve methode of processor 0: \n";
			te.report();
		}

	} catch (std::exception & e) {
		std::cout << e.what();
		env.abort(1);
		return 1;
	}

	return 0;
}
}
