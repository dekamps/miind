/*
 * main.cpp
 *
 *  Created on: May 16, 2012
 *      Author: david
 */

#include <MPILib/config.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#endif
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>
#include <boost/serialization/string.hpp>
#include <boost/serialization/base_object.hpp>

#include <exception>
#include <MPILib/include/MPINodeCode.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/algorithm/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/algorithm/WilsonCowanParameter.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>
#include <MPILib/include/report/handler/RootHighThroughputHandler.hpp>

#include <MPILib/include/algorithm/RateAlgorithmCode.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>


using namespace MPILib;

const report::handler::RootHighThroughputHandler WILSONCOWAN_HIGH_HANDLER(
		"test/wilsonresponseHigh", // file where the simulation results are written
		true // generate graphs
		);

const report::handler::RootReportHandler WILSONCOWAN_HANDLER(
		"test/wilsonresponse", // file where the simulation results are written
		false // only rate diagrams
		);

const SimulationRunParameter PAR_WILSONCOWAN(WILSONCOWAN_HIGH_HANDLER, // the handler object
		1000000, // maximum number of iterations
		0, // start time of simulation
		0.5, // end time of simulation
		1e-4, // report time
		1e-5, // network step time
		"test/wilsonresponse" // log file name without extension
		);


int main(int argc, char* argv[]) {
#ifdef ENABLE_MPI
	// initialise the mpi environment this cannot be forwarded to a class
	boost::mpi::environment env(argc, argv);
#endif
	utilities::MPIProxy mpiProxy;
	try {
		MPINetwork<double, utilities::CircularDistribution> network;

		Time tau = 10e-3; //10 ms
		Rate rate_max = 100.0;
		double noise = 1.0;

		// define some efficacy
		Efficacy epsilon = 1.0;

		// define some input rate
		Rate nu = 0;

		// Define a node with a fixed output rate
		algorithm::RateAlgorithm<double> rate_alg(nu);
		int id_rate = network.addNode(rate_alg, EXCITATORY);

		// Define the receiving node
		algorithm::WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

		algorithm::WilsonCowanAlgorithm algorithm_exc(par_sigmoid);
		int id = network.addNode(algorithm_exc, EXCITATORY);

		// connect the two nodes
		network.makeFirstInputOfSecond(id_rate, id, epsilon);

		network.configureSimulation(PAR_WILSONCOWAN);

		boost::timer::auto_cpu_timer te;
		te.start();

		network.evolve();
		mpiProxy.barrier();
		te.stop();
		if (mpiProxy.getRank() == 0) {
			std::cout << "Time of Envolve methode of processor 0: \n";
			te.report();
		}

	} catch (std::exception & e) {
		std::cout << e.what();
#ifdef ENABLE_MPI
		//Abort the MPI environment in the correct way :)
		env.abort(1);
#endif
		return 1;
	}

	return 0;
}
