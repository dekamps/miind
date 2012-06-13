/*
 * main.cpp
 *
 *  Created on: May 16, 2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <boost/serialization/string.hpp>
#include <boost/serialization/base_object.hpp>

#include <exception>
#include <MPILib/include/MPINodeCode.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/algorithm/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/algorithm/WilsonCowanParameter.hpp>
#include <MPILib/include/reportHandler/RootReportHandler.hpp>

#include <MPILib/include/algorithm/RateAlgorithm.hpp>
#include <MPILib/include/utilities/walltime.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>

namespace mpi = boost::mpi;

const RootReportHandler WILSONCOWAN_HANDLER(
		"test/wilsonresponse.root", // file where the simulation results are written
		false // only rate diagrams
		);

const SimulationRunParameter PAR_WILSONCOWAN(WILSONCOWAN_HANDLER, // the handler object
		1000000, // maximum number of iterations
		0, // start time of simulation
		0.5, // end time of simulation
		1e-4, // report time
		1e-4, // update time
		1e-5, // network step time
		"test/wilsonresponse.log" // log file name
		);

const RootReportHandler WILSONCOWAN_HANDLER1(
		"test/wilsonresponse1.root", // file where the simulation results are written
		false // only rate diagrams
		);

const SimulationRunParameter PAR_WILSONCOWAN1(WILSONCOWAN_HANDLER1, // the handler object
		1000000, // maximum number of iterations
		0, // start time of simulation
		0.5, // end time of simulation
		1e-4, // report time
		1e-4, // update time
		1e-5, // network step time
		"test/wilsonresponse1.log" // log file name
		);

using namespace MPILib;

int main(int argc, char* argv[]) {
	// initialize mpi

	mpi::environment env(argc, argv);
	mpi::communicator world;
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
		algorithm::RateAlgorithm rate_alg(nu);
		int id_rate = network.addNode(rate_alg, 1);

		// Define the receiving node
		algorithm::WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

		algorithm::WilsonCowanAlgorithm algorithm_exc(par_sigmoid);
		int id = network.addNode(algorithm_exc, 1);

		// connect the two nodes
		network.makeFirstInputOfSecond(id_rate, id, epsilon);

		if (world.rank() == 0) {
			network.configureSimulation(PAR_WILSONCOWAN);

		} else {
			network.configureSimulation(PAR_WILSONCOWAN1);
		}

		double time, time_start = 0.0;

		time = walltime(&time_start);

		network.evolve();

		time = walltime(&time);

		if (world.rank() == 0) {
			double maxtime;
			mpi::reduce(world, time, maxtime, mpi::maximum<double>(), 0);
			std::cout << "The max time is " << maxtime << " sec" << std::endl;
		} else {
			reduce(world, time, mpi::maximum<double>(), 0);
		}

	} catch (std::exception & e) {
		std::cout << e.what();
		env.abort(1);
		return 1;
	}

	return 0;
}
