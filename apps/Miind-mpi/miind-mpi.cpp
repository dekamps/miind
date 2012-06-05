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
#include <MPILib/include/Sleep10secAlgorithm.hpp>
#include <MPILib/include/utilities/walltime.hpp>

namespace mpi = boost::mpi;
using namespace MPILib;

int main(int argc, char* argv[]) {
	// initialize mpi

	mpi::environment env(argc, argv);
	mpi::communicator world;
	try {
		MPINetwork<double> network;
		Sleep10secAlgorithm alg;

		int node0 = network.AddNode(alg, 1);
		int node1 = network.AddNode(alg, 1);
		int node2 = network.AddNode(alg, 1);

		double weight = 3.1;
		network.MakeFirstInputOfSecond(node0, node1, weight);
		weight = 1.2;
		network.MakeFirstInputOfSecond(node0, node2, weight);

		weight = 6.4;

		network.MakeFirstInputOfSecond(node1, node0, weight);
		weight = 6.1;
		network.MakeFirstInputOfSecond(node2, node1, weight);

		double time, time_start = 0.0;

		time = walltime(&time_start);

		network.Evolve();

		time = walltime(&time);


		if (world.rank() == 0) {
			double maxtime;
			mpi::reduce(world, time, maxtime, mpi::maximum<double>(), 0);
			std::cout << "The max time is " << maxtime << " sec"
					<< std::endl;
		} else {
			reduce(world, time, mpi::maximum<double>(), 0);
		}

	} catch (std::exception & e) {
		std::cout << e.what();
	};

	return 0;
}
